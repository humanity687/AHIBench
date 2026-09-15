"""realengine.py — 真实文本版引擎：LLM 蒸馏摘要 + 混合检索（向量 + 关键词双池）。"""

import json
import os
import re
import threading
import time
import concurrent.futures as cf
from concurrent.futures import as_completed
from concurrent.futures import TimeoutError as FutureTimeoutError

from core import Atom, Engine, ATOM_WEIGHT_CHAR

SUMMARIZE_PROMPT_NEUTRAL = (
    "你是记忆蒸馏器。把以下内容概括成中性摘要卡（只输出一行）：\n"
    "【事实】中性概括：必须保留专有名词（人名/地名/物品名）、数字、因果链；"
    "禁止编造、禁止评价、禁止改变事实的归属；禁止第一人称视角（不用\"我\"）；"
    "涉及来源时写明是谁说的。\n"
    "时序分流规则（重要）：源内容带〔早期〕标记的句子是历史状态——"
    "同一实体的先后状态矛盾时，以最新状态为当前事实，旧状态保留为历史"
    "（如\"曾学 Rust，现学 Go\"）；禁止把旧状态当当前事实复述。\n"
    "【内容】\n{content}"
)

SUMMARIZE_PROMPT_STYLED = (
    "你是记忆蒸馏器。把以下内容概括成一张摘要卡：\n"
    "【事实】中性概括：必须保留专有名词（人名/地名/物品名）、数字、因果链；"
    "禁止编造、禁止评价、禁止改变事实的归属；涉及来源时写明是谁说的。\n"
    "时序分流规则（重要）：源内容带〔早期〕标记的句子是历史状态——"
    "同一实体的先后状态矛盾时，以最新状态为当前事实，旧状态保留为历史"
    "（如\"曾学 Rust，现学 Go\"）；禁止把旧状态当当前事实复述。\n"
    "如需保留一句主观评注（可选，不加也不违规），在【事实】下另起一行写"
    "【注】（40 字内）：用你自己的认知习惯与口吻写，但不引入新事实、不篡改事实归属。\n"
    "{style}"
    "【内容】\n{content}"
)

SUMMARIZE_CHAR_CAP = 3000

VEC_MODEL_ORDER = ["BAAI/bge-small-zh-v1.5", "sentence-transformers/all-MiniLM-L12-v2",
                   "sentence-transformers/all-MiniLM-L6-v2"]

# 三通道权重（先拍后调）：摘要·向量 / 摘要·关键词 / 原子·粗筛+向量精排
W_VEC_NODE = 0.4
W_BM_NODE = 0.3
W_BM_ATOM = 0.3
ATOM_MIN_LEN = 8
ATOM_RERANK_CAP = 60
ATOM_VEC_MIN = 0.2

# 区间全原子回调（[span] 工具）上限（先拍后调）
SPAN_ATOM_CAP = 40      # 单次回调最多原子数
SPAN_CHAR_CAP = 1500    # 单次回调最多字符数

# 叶子展开（命中概括卡 → 展开全部后代原子为影子）上限（先拍后调）
EXPAND_ATOM_CAP = 80    # 单轮展开最多原子数
EXPAND_CHAR_CAP = 3000  # 单轮展开最多字符数

# 自动下钻披露（auto_expand，先拍后调）：预算富余时把概括卡下的直接原子子卡
# 激活为影子——"只活一轮，没用就消失"（每轮重算：新鲜写入或本轮被召回命中者才披露）
EXPAND_RATIO = 0.85     # 组装预算使用率低于此 → 富余，尝试下钻
EXPAND_FRESH_WINDOW = 3 # created_round 距今 ≤ 此轮数 = "新鲜"

# 笔记本（轨道B）注入上限（先拍后调）
NB_CONSTRAINT_CHARS = 400   # 常驻约束总字符上限
NB_STATE_CHARS = 600        # 状态条目总字符上限
NB_WORKLOG_CHARS = 400      # 工作记录总字符上限


# 蒸馏批量超时（秒）：SDK read_timeout 之上加总时长兜底——httpx read 是字节
# 间隔超时，服务器 trickle（持续心跳字节）可绕过；future 超时 → 丢弃标记重试
DISTILL_BATCH_TIMEOUT = 150.0


class _DaemonPool:
    """极简 daemon 线程池（自实现，2026-08-28）：
    - submit 永不阻塞：内部队列 + 固定 max_workers 个 daemon worker（挂起的 LLM
      调用占槽 → 队列积压，主循环不卡、并发不膨胀）；
    - 返回标准 concurrent.futures.Future（as_completed(timeout) 兼容）；
    - 线程 daemon：挂起 worker 不阻塞进程退出（Python 线程无法强杀）。"""

    def __init__(self, max_workers):
        import queue
        self._q = queue.Queue()
        self._workers = [threading.Thread(target=self._drain, daemon=True,
                                          name=f"distill-w{i}")
                         for i in range(max(1, int(max_workers)))]
        for w in self._workers:
            w.start()

    def _drain(self):
        while True:
            fut, fn, args = self._q.get()
            if fut is None:
                return
            try:
                fut.set_result(fn(*args))
            except BaseException as e:
                fut.set_exception(e)

    def submit(self, fn, *args):
        fut = cf.Future()
        self._q.put((fut, fn, args))
        return fut


def bigrams(text):
    t = re.sub(r"\s+", "", text or "")
    if len(t) < 2:
        return set(t) if t else set()
    return {t[i:i + 2] for i in range(len(t) - 1)}


def _load_encoder():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    # 强制离线：只用本地缓存，禁止每次加载都去 HF Hub 检查更新（网络问题会挂死）
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    for name in VEC_MODEL_ORDER:
        try:
            return SentenceTransformer(name, local_files_only=True)
        except Exception:
            continue
    return None


class RealEngine(Engine):
    def __init__(self, llm, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm
        self.distill = {"calls": 0, "prompt_chars": 0, "completion_tokens": 0, "seconds": 0.0}
        self._encoder = kwargs.get("encoder", None)
        self._vec_cache = {}
        self.expand_atom_cap = kwargs.get("expand_atom_cap", EXPAND_ATOM_CAP)
        self.expand_char_cap = kwargs.get("expand_char_cap", EXPAND_CHAR_CAP)
        self.expand_ratio = kwargs.get("expand_ratio", EXPAND_RATIO)
        self.expand_fresh_window = kwargs.get("expand_fresh_window", EXPAND_FRESH_WINDOW)
        self._recent_boost = set()  # 本轮被召回命中（HP 提升）的原子/节点
        # 笔记本（轨道B 最小实现）：kind ∈ constraint / state / schedule
        # entry = {"kind": str, "content": str, "status": str, "updated_round": int}
        self.notebook = {}
        # 预蒸馏（决策 ③）：预测下轮 merge 候选 → 后台先做 LLM 摘要 → merge 命中复用
        # key = 组内卡 id 有序列表 tuple；编辑/墓碑必改卡 id → key 天然失效（无脏复用）
        self._pre_distill = {}
        self._pre_distill_lock = threading.Lock()
        self.pre_distill_used = 0    # 指标：命中复用次数
        self.pre_distill_wasted = 0  # 指标：启动但未用次数（token 浪费）
        # 并行蒸馏（迭代：一轮 merge 的多低 HP 区间并行 LLM 摘要，替代串行）
        self.distill_parallel = kwargs.get("distill_parallel", 4)  # 并发上限
        self.distill_batch_timeout = kwargs.get("distill_batch_timeout",
                                                DISTILL_BATCH_TIMEOUT)
        # 共享 daemon 线程池：并发被 max_workers 限死（挂起 worker 不膨胀、不挡退出）
        self._distill_pool = _DaemonPool(max_workers=max(1, int(self.distill_parallel)))
        self._last_merge_wall = 0.0  # 指标：最近一次 merge_pass 墙钟时间
        # 人格染色（迭代）：style_provider 由 agent 层注入（快照+自我样本），
        # 未注入（neutral）→ 纯中性蒸馏
        self.style_provider = None   # callable -> str | None
        self.style_alpha = kwargs.get("style_alpha", 1.0)   # 染色强度：0/0.5/1（染色 v3）
        self.fact_drift = 0          # 指标：事实校验失败的蒸馏次数（只留日志不重蒸馏）
        self.first_person_violations = 0  # 指标：人称-来源绑定违规次数（只留日志）
        self.distill_snap_path = kwargs.get("distill_snap_path", None)  # 蒸馏卡快照 JSONL 路径
        self._src_cache = {}         # 渲染用：节点子树来源聚合缓存

    # ── 并行蒸馏（迭代：多低 HP 区间并行摘要 → 提高单轮蒸馏吞吐）──

    def _merge_runs(self, force_out: bool = False) -> None:
        """覆写 core._merge_runs：合并前把本轮全部候选 chunk **并行**蒸馏
        （上限 distill_parallel 并发），完成后由父类走原有合并路径
        （_weighted_avg 命中缓存 → 跳过现场串行蒸馏）。
        正确性：不同 run/chunk 区间互不重叠（父区间 = 子区间并集不变量），
        摘要彼此独立，并行安全；失败 chunk 标 failed → 合并时现场串行兜底。"""
        import time as _t
        t0 = _t.time()
        runs = self._find_merge_runs(force_out)
        # 收集全部候选 chunk（与父类 _merge_run→_chunk_run 同口径）
        todo = []
        with self._pre_distill_lock:
            for run in runs:
                for chunk in self._chunk_run(run):
                    key = tuple(chunk)
                    cached = self._pre_distill.get(key)
                    if cached is None:
                        parts = self._group_parts(chunk)
                        if not parts:
                            continue
                        self._pre_distill[key] = {"text": None, "ready": False,
                                                  "failed": False}
                        todo.append((key, parts))
        # 分批并行蒸馏（批内并发 ≤ distill_parallel；future 超时 → 丢弃标记，
        # 现场串行兜底——join 无超时是 2026-08-28 主实验 neutral 卡死 8 分钟的根因）
        limit = max(1, int(self.distill_parallel))
        for i in range(0, len(todo), limit):
            batch = todo[i:i + limit]
            futs = {self._distill_pool.submit(self._pre_distill_worker, key, parts):
                    (key, parts) for key, parts in batch}
            done = set()
            try:
                for fut in as_completed(futs, timeout=self.distill_batch_timeout):
                    done.add(fut)
            except FutureTimeoutError:
                pass
            # 超时未完成的 → 标 failed（merge 现场 _weighted_avg 串行蒸馏兜底）；
            # 线程让它自生自灭（daemon + 池槽有限，并发不膨胀）
            if len(done) < len(futs):
                with self._pre_distill_lock:
                    for fut, (key, _p) in futs.items():
                        if fut not in done:
                            cached = self._pre_distill.get(key)
                            if cached is not None and not cached.get("ready"):
                                cached["failed"] = True
        super()._merge_runs(force_out)
        self._last_merge_wall = _t.time() - t0

    # ── 预蒸馏（决策 ③：低 HP 驱动 + 后台提前蒸馏）──────────

    def pre_distill_launch(self, max_tasks=2, force_out=False):
        """预测下轮 merge 候选组（复用 core._find_merge_runs 同款分组），
        对尚无缓存且不在运行中的候选启动后台蒸馏。每轮最多 max_tasks 个新任务。
        源内容 = 组内卡文本（原子取原文、节点取摘要），截断 SUMMARIZE_CHAR_CAP。"""
        if max_tasks <= 0:
            return
        runs = self._find_merge_runs(force_out)
        launched = 0
        with self._pre_distill_lock:
            pending_keys = {k for k, v in self._pre_distill.items()
                            if not v.get("ready") and not v.get("failed")}
            candidates = [r for r in runs if tuple(r) not in self._pre_distill
                          and tuple(r) not in pending_keys]
        for run in candidates:
            if launched >= max_tasks:
                break
            key = tuple(run)
            parts = self._group_parts(run)
            if not parts:
                continue
            self._pre_distill[key] = {"text": None, "ready": False, "failed": False}
            launched += 1
            self._distill_pool.submit(self._pre_distill_worker, key, parts)
        return launched

    def _atom_part(self, c) -> str:
        """原子 → 蒸馏源文本（D：时序分流引导，2026-08-30 拍定）。
        HP 低于合并阈值 = 已出上下文生命周期 → 加〔早期〕前缀，
        引导蒸馏以"历史态"叙述而非"当前态"复述。
        （上下文管理定位：旧值不删——生命周期把它扔到后面，
        蒸馏负责把它正确分流为记忆中的历史。）"""
        v = self.atoms[c].value or ""
        if self.atoms[c].hp < self.hp_merge_threshold:
            return f"〔早期〕{v}"
        return v

    def _group_parts(self, run):
        """组内卡 → 源文本列表（原子取原文、节点取摘要；跳过不可用）。
        节点文本取【事实】段（防复合：注/索引行不进上层蒸馏）。"""
        parts = []
        for c in run:
            if self._is_atom(c):
                if self.atoms[c].alive:
                    parts.append(self._atom_part(c))
            else:
                nd = self.nodes.get(c)
                if nd and nd.alive and nd.value is not None:
                    parts.append(self._fact_part(nd.value))
        return parts

    def _pre_distill_worker(self, key, parts):
        """后台蒸馏：LLM 摘要 → 写缓存（锁）。失败标 failed（merge 时现场蒸馏兜底）。
        key = 组内卡 id 列表，作为 children 传入（蒸馏快照溯源需要）。"""
        try:
            text = self._summarize(parts, children=list(key))
            with self._pre_distill_lock:
                cached = self._pre_distill.get(key)
                if cached is not None:
                    cached["text"] = text
                    cached["ready"] = True
        except Exception:
            with self._pre_distill_lock:
                cached = self._pre_distill.get(key)
                if cached is not None:
                    cached["failed"] = True

    def pre_distill_take(self, children):
        """merge/rebuild 构建摘要时命中缓存 → 取走复用。key = 组内卡 id 列表。"""
        key = tuple(children)
        with self._pre_distill_lock:
            cached = self._pre_distill.pop(key, None)
        if cached and cached.get("ready") and cached.get("text"):
            self.pre_distill_used += 1
            return cached["text"]
        if cached is not None:
            self.pre_distill_wasted += 1
        return None

    # ── 笔记本（轨道B）───────────────────────────────

    def nb_set(self, key, kind, content, status="current"):
        key = key.strip()
        if not key:
            return None
        if kind == "worklog":
            key = "current"   # 固定 key 覆盖 = "当前状态"语义（旧工作记录被去重）
        self.notebook[key] = {
            "key": key, "kind": kind, "content": content.strip(),
            "status": status, "updated_round": self.round,
        }
        self.events.append(("notebook", key, kind))
        return key

    def nb_get(self, key):
        return self.notebook.get(key)

    def nb_render(self, goal=""):
        """组装注入（去重发送，聊天拍定 2026-08-14）：
        - 去重：存储层同 key 覆盖，每个 key 只发最新条目；旧版本不注入；
        - 不滞留：每轮组装时从 notebook 现场独立生成，注入段不进任何持久结构；
        - 位置：调用方把本段拼接在卡片流**后面**（组装顺序微调，见 AGENTS.md §10）；
        - 选取：constraint 恒在（每 key 最新）；state 按 goal 关键词精确选取；
          schedule 仅注入 planted/advancing 活动项；worklog 按 updated_round 排序注入
          （固定 key 覆盖 = "当前状态"语义，旧工作记录自然被去重）。
        返回 [(kind, key, content), ...]。"""
        out = []
        constraints = [e for e in self.notebook.values() if e["kind"] == "constraint"]
        for e in constraints:
            out.append(("constraint", None, e["content"]))
        goal_bigrams = bigrams(goal) if goal else set()
        used = 0
        states = sorted(
            [e for e in self.notebook.values() if e["kind"] == "state"],
            key=lambda e: -e["updated_round"],
        )
        if goal_bigrams:
            matched = []
            for e in states:
                kb = bigrams(e["content"][:60]) | bigrams(e["key"]) if e["key"] else bigrams(e["content"][:60])
                if kb & goal_bigrams:
                    matched.append(e)
            # goal 过滤是次要优化：无命中时回退全量（去重发送 + cap 兜底），不静默清空
            if matched:
                states = matched
        for e in states:
            used += len(e["content"])
            if used > NB_STATE_CHARS:
                break
            out.append(("state", e["key"], e["content"]))
        for e in self.notebook.values():
            if e["kind"] == "schedule" and e.get("status") in ("planted", "advancing"):
                out.append(("schedule", e["key"], e["content"]))
        used = 0
        worklogs = sorted(
            [e for e in self.notebook.values() if e["kind"] == "worklog"],
            key=lambda e: -e["updated_round"],
        )
        for e in worklogs:
            used += len(e["content"])
            if used > NB_WORKLOG_CHARS:
                break
            out.append(("worklog", e["key"], e["content"]))
        return out

    def _new_atom(self, value, weight=None, metadata=None):
        if weight is None:
            weight = 2 if len(value or "") > ATOM_WEIGHT_CHAR else 1
        a = Atom(id=self._nid(), value=value, hp=self.initial_hp, created_round=self.round,
                 weight=weight, metadata=dict(metadata or {}))
        self.atoms[a.id] = a
        return a

    def insert_units(self, units, left_id=None, right_id=None):
        """插入带元数据的原子单元列表（聊天适配：media 切分产物）。
        units = [(text, meta), ...]；meta 写入 Atom.metadata。
        语义与 core.insert 一致：left_id=None 且 right_id=None 时插入链头；
        追加到链尾请显式传 left_id=self.tail。"""
        new = []
        for text, meta in units:
            new.append(self._new_atom(text, metadata=meta))
        return self.insert(left_id, right_id, atoms=new)

    def replace_units(self, old_ids, units):
        """卡级替换（编辑路径）：墓碑旧原子 + 插入带元数据的新卡。
        旧卡定位用锚点（prev/next），区间失效/置脏/stale/commits 全走编辑路径。"""
        alive_old = [aid for aid in old_ids if aid in self.atoms and self.atoms[aid].alive]
        if not alive_old:
            return []
        left = self.atoms[alive_old[0]].prev
        right = self.atoms[alive_old[-1]].next
        self.delete(alive_old)
        new = []
        for text, meta in units:
            new.append(self._new_atom(text, metadata=meta))
        return self.insert(left, right, atoms=new)

    def _atom_text(self, aid):
        """检索/渲染用文本：代码原子优先 docstring（P0-3），其余为原文。"""
        a = self.atoms[aid]
        doc = (a.metadata or {}).get("doc")
        return doc if doc else (a.value or "")

    # ── 向量编码（懒加载，按节点 version 缓存失效）────────

    def _encoder_model(self):
        if self._encoder is None:
            self._encoder = _load_encoder()
        return self._encoder

    def _node_vec(self, nid):
        nd = self.nodes[nid]
        if nd.value is None:
            return None
        cached = self._vec_cache.get(nid)
        if cached is not None and cached[0] == nd.version:
            return cached[1]
        enc = self._encoder_model()
        if enc is None:
            return None
        emb = enc.encode(self._fact_part(nd.value), normalize_embeddings=True)
        self._vec_cache[nid] = (nd.version, emb)
        return emb

    def _query_vec(self, q):
        enc = self._encoder_model()
        if enc is None:
            return None
        return enc.encode(q, normalize_embeddings=True)

    # ── 内容生成（LLM 蒸馏）────────────────────────────

    def _weighted_avg(self, children):
        """merge 建父节点/脏重建的摘要入口：先查预蒸馏缓存（决策 ③），
        命中直接用（跳过现场 LLM）；未命中 → 现场蒸馏（降级）。"""
        cached = self.pre_distill_take(children)
        if cached is not None:
            return cached
        parts = []
        for c in children:
            if self._is_atom(c):
                parts.append(self._atom_part(c))
            else:
                v = self.nodes[c].value
                if v is not None:
                    # 防复合（染色 v3）：子卡入上层摘要只用【事实】段——
                    # 【注】与〔另有X条〕索引行不参与上层蒸馏
                    parts.append(self._fact_part(v))
        if not parts:
            return None
        return self._summarize(parts, children=children)

    def _summarize(self, parts, children=None):
        content = "\n".join(parts)
        if len(content) > SUMMARIZE_CHAR_CAP:
            content = content[:SUMMARIZE_CHAR_CAP] + "……（截断）"
        style = ""
        if self.style_alpha > 0 and self.style_provider is not None:
            try:
                style = (self.style_provider() or "").strip()
            except Exception:
                style = ""
        if self.style_alpha > 0:
            prompt = SUMMARIZE_PROMPT_STYLED.format(
                content=content, style=style + "\n" if style else "")
        else:
            prompt = SUMMARIZE_PROMPT_NEUTRAL.format(content=content)
        t0 = time.time()
        text, usage, _lat = self.llm.chat(
            "你是一个可靠的中文记忆蒸馏器。", prompt
        )
        self.distill["calls"] += 1
        self.distill["prompt_chars"] += len(prompt)
        self.distill["completion_tokens"] += usage.get("completion_tokens", 0)
        self.distill["seconds"] += time.time() - t0
        text = (text or "").strip()
        # 双字段解析（容错：无标记 → 全当事实，安全方向）+ 【注】程序硬截断
        fact, note = self._parse_summary(text)
        if self.style_alpha <= 0:
            note = ""   # α=0 中性基线：程序强制无【注】（prompt 不听话也拦）
        if not fact:
            fact = "【摘要生成失败】"
        # 事实校验：数字保留率低于阈值 → 只记日志/指标（不重蒸馏）
        self._fact_check(content, fact)
        # 人称-来源强制绑定（染色 v3）：第一人称句缺来源主语 → 只记日志
        self._first_person_check(fact)
        # 差集索引（染色 v3 指针密度）：源 parts 中未被摘要覆盖的 → 程序化索引行
        uncovered = self._uncovered_count(parts, fact)
        if uncovered > 0:
            fact = f"{fact}〔另有{uncovered}条未展开，可@span下钻〕"
        # 蒸馏卡快照落盘（观测层 P0：展开比指纹的测量地基）
        self._distill_snapshot(parts, children, fact, note)
        return fact if not note else f"【事实】{fact}\n【注】{note}"

    def _distill_snapshot(self, parts, children, fact, note):
        """每次蒸馏 append 一行 JSONL（含源 parts/摘要/索引/α），供指纹分析。零行为影响。"""
        if not self.distill_snap_path:
            return
        row = {
            "ts": round(time.time(), 2),
            "round": self.round,
            "children": children if children is not None else None,
            "parts": [p[:400] for p in parts],
            "fact": fact[:300],
            "note": note,
            "style_alpha": self.style_alpha,
            "style_on": bool(self.style_provider is not None),
        }
        try:
            with open(self.distill_snap_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _uncovered_count(self, parts, fact):
        """差集覆盖率检查（程序化，零 LLM）：每个源 part 提取特征 token
        （数字/拉丁词/长 CJK 词），全部未出现在摘要中 → 判为未展开。
        宁可多计不少计（多计只是索引保守，少计 = 事实静默丢失）。"""
        count = 0
        fact = fact or ""
        for p in parts:
            p = p or ""
            toks = set(re.findall(r"\d+(?:[\.\-/年]\d+)*|[A-Za-z][A-Za-z0-9_-]{2,}", p))
            if not toks:
                continue
            if not any(t in fact for t in toks):
                count += 1
        return count

    def _first_person_check(self, fact):
        """人称-来源强制绑定（染色 v3，只记日志不干预）：第一人称句必须含来源主语
        （人名/用户/管理员/agent 名）。违规计数进 events，与 fact_drift 同待遇。"""
        src_toks = ("probe", "0xbf5d36", "BF5D36", "管理员", "用户", "澜", "林深",
                    "neutral", "lan", "lin-shen", "agent", "你")
        n_bad = 0
        for sent in re.split(r"[。；！？\n]", fact or ""):
            if re.search(r"我(们|的|注意到|觉得|看到|记录|听见)", sent):
                if not any(t in sent for t in src_toks):
                    n_bad += 1
        if n_bad:
            self.first_person_violations += n_bad
            self.events.append(("person_source_violation", n_bad))

    def dump_tree(self, path):
        """树快照落盘（观测层）：全树序列化为一行 JSONL。
        原子含原文/HP/owner/weight/source/replied；节点含内容/children/parent/
        span/dirty/placeholder/version/激活计数；外加 tiling 与 round。零行为影响。"""
        def atom_dict(a):
            d = {"id": a.id, "text": (a.value or "")[:500], "hp": round(a.hp, 1),
                 "alive": a.alive, "owner": a.owner, "weight": a.weight,
                 "created_round": a.created_round}
            md = a.metadata or {}
            if md.get("source"):
                d["source"] = md["source"]
            if md.get("replied"):
                d["replied"] = True
            if md.get("pending_dismiss"):
                d["pending_dismiss"] = True
            return d

        def node_dict(n):
            return {"id": n.id, "parent": n.parent, "children": list(n.children),
                    "content": (n.value or "")[:600], "hp": round(n.hp, 1),
                    "dirty": n.dirty, "placeholder": n.placeholder,
                    "version": n.version, "created_round": n.created_round,
                    "activation_count": n.activation_count,
                    "span": [self._first_atom(n.id), self._last_atom(n.id)]}

        snap = {
            "ts": round(time.time(), 2),
            "round": self.round,
            "atoms": {aid: atom_dict(a) for aid, a in self.atoms.items()},
            "nodes": {nid: node_dict(n) for nid, n in self.nodes.items() if n.alive},
            "tiling": list(self.tiling),
            "stat": self.stat(),
        }
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(snap, ensure_ascii=False) + "\n")
        except Exception:
            pass

    @staticmethod
    def _parse_summary(text):
        """双字段容错解析。返回 (fact, note)。无标记 → 全文本为 fact（中性兜底）。"""
        m = re.match(r"\s*【事实】\s*(.*?)\s*(?:【注】\s*(.*))?\s*$", text or "", re.S)
        if m:
            fact = m.group(1).strip()[:200]
            note = (m.group(2) or "").strip()[:80]   # 程序硬截断（不靠 prompt 约束）
            return fact, note
        return (text or "").strip()[:200], ""

    def _fact_check(self, source_text, fact_text):
        """事实校验（只留日志）：源内容数字保留率 < 60% → 记 fact_drift。
        效率约束：纯正则零 LLM，不重蒸馏。"""
        nums = set(re.findall(r"\d+(?:[\.\-/年]\d+)*", source_text or ""))
        if not nums:
            return
        keep = sum(1 for n in nums if n in (fact_text or ""))
        ratio = keep / len(nums)
        if ratio < 0.6:
            self.fact_drift += 1
            self.events.append(("fact_drift", f"数字保留率 {ratio:.0%}（{keep}/{len(nums)}）"))

    def _fact_part(self, text):
        """检索用：只取【事实】段（染色【注】与〔另有X条〕索引行不参与打分，
        防风格词与索引噪声干扰检索）。"""
        m = re.match(r"\s*【事实】\s*(.*?)\s*(?:【注】|$)", text or "", re.S)
        fact = m.group(1).strip() if m else (text or "")
        fact = re.sub(r"〔另有[^〕]*〕", "", fact)
        return fact.strip()

    def _node_sources(self, nid):
        """节点子树来源聚合（程序维护，不依赖蒸馏）：渲染摘要卡时附〔源〕标注。
        缓存一轮组装（assemble 开头清空，重建后自动失效）。"""
        cached = self._src_cache.get(nid)
        if cached is not None:
            return cached
        srcs = set()
        stack = [nid]
        while stack:
            c = stack.pop()
            if self._is_atom(c):
                if self.atoms[c].alive:
                    srcs.add((self.atoms[c].metadata or {}).get("source") or "?")
            else:
                nd = self.nodes.get(c)
                if nd and nd.alive:
                    stack.extend(nd.children)
        self._src_cache[nid] = srcs
        return srcs

    # ── 混合检索池（三通道：摘要·向量 / 摘要·关键词 / 原子·粗筛+向量精排）────

    def _bm_ratio(self, text, q):
        qb = bigrams(q)
        if not qb:
            return 0.0
        tb = bigrams(text)
        if not tb:
            return 0.0
        return len(tb & qb) / len(qb)

    def _atom_vec(self, aid):
        cached = self._vec_cache.get(aid)
        if cached is not None:
            return cached
        enc = self._encoder_model()
        if enc is None:
            return None
        emb = enc.encode(self._atom_text(aid), normalize_embeddings=True)
        self._vec_cache[aid] = emb
        return emb

    def _retrieval_pool(self, q):
        pool = {}
        qv = self._query_vec(q)
        qb = bigrams(q)
        for nid, nd in self.nodes.items():
            if not (nd.alive and not nd.dirty and not nd.placeholder and nd.value is not None):
                continue
            bm = self._bm_ratio(self._fact_part(nd.value), q)
            vec = 0.0
            if qv is not None:
                nv = self._node_vec(nid)
                if nv is not None:
                    vec = float(nv @ qv)
            pool[nid] = W_VEC_NODE * vec + W_BM_NODE * bm
        if qb:
            # 原子通道：代码原子只用 docstring 打分（P0-3）；文本不可变 → 向量缓存永久有效
            cands = [aid for aid, a in self.atoms.items()
                     if a.alive and len(self._atom_text(aid)) >= ATOM_MIN_LEN
                     and bigrams(self._atom_text(aid)) & qb]
            if qv is not None:
                for aid in cands[:ATOM_RERANK_CAP]:
                    av = self._atom_vec(aid)
                    if av is None:
                        continue
                    vec = float(av @ qv)
                    if vec >= ATOM_VEC_MIN:
                        pool[aid] = W_BM_ATOM * self._bm_ratio(self._atom_text(aid), q) \
                                    + W_VEC_NODE * vec
            else:
                for aid in cands:
                    pool[aid] = W_BM_ATOM * self._bm_ratio(self._atom_text(aid), q)
        return pool

    def _verbatim_match(self, a, q):
        q = (q or "").strip()
        return len(q) >= 2 and q in a.value

    # ── 重复输入检测（聊天适配 P2-8）──────────────────

    def find_similar(self, text, threshold=0.8, topk=3, exclude_ids=None):
        """新文本与**存活卡**的 bigram 重叠率（Jaccard，分母取 max）≥ 阈值 → 命中。
        命中卡并入本轮 shadows（活一轮影子），让 AI 知道"之前也听过/写过"。
        只搜存活卡：墓碑 = 已被编辑替换，AI 编辑前刚见过，塞回墓碑违背墓碑语义。
        exclude_ids：排除本次刚写入的新卡（防自己匹配自己）。纯计算（零 LLM）。"""
        qb = bigrams(text)
        if not qb:
            return []
        ex = set(exclude_ids or ())
        scored = []
        for aid, a in self.atoms.items():
            if not a.alive or aid in ex:
                continue
            tb = bigrams(self._atom_text(aid))
            if not tb:
                continue
            j = len(tb & qb) / max(len(tb), len(qb))
            if j >= threshold:
                scored.append((j, aid))
        scored.sort(key=lambda x: -x[0])
        return [aid for _, aid in scored[:topk]]

    # ── DISMISS（聊天适配：事实冲突的延迟裁决）────────────────

    def dismiss_pending(self):
        """全部挂起中的懒标记（notebook kind=schedule 且 status=planted）。"""
        return [e for e in self.notebook.values()
                if e["kind"] == "schedule" and e.get("status") == "planted"]

    def dismiss_plant(self, text, budget=None, exclude_ids=None):
        """挂懒标记（零破坏）：立即检索冲突描述 → 命中项展开为后代原子并标
        pending_dismiss 元数据；懒标记存轨道B（schedule 类，planted），每轮由调用方
        持续预检索追踪。返回 (key, 待失效原子列表，供注入影子)。
        排除集 = exclude_ids ∪ 本轮及之后写入的新卡（防新信息原子命中自己）。"""
        text = (text or "").strip()
        if not text:
            return None, []
        seq = sum(1 for e in self.notebook.values() if e["kind"] == "schedule")
        key = f"dismiss_{seq + 1}"
        self.nb_set(key, "schedule", text, status="planted")
        hits = self.retrieve(text, budget=budget if budget is not None else self.retrieval_budget)
        atoms = self.expand_leaves(hits)
        ex = set(exclude_ids or ())
        ex |= {aid for aid in self.atoms if self.atoms[aid].created_round >= self.round}
        atoms = [aid for aid in atoms if aid not in ex]
        for aid in atoms:
            self.atoms[aid].metadata["pending_dismiss"] = True
        return key, atoms

    def dismiss_track(self, budget=None):
        """每轮持续追踪：对全部 planted 懒标记预检索，命中项展开为后代原子并标
        pending_dismiss（注入影子）。**只标挂起之前已存在的旧卡**（created_round <
        挂起轮次）——挂起后的新信息/讨论不标，防标注扩散误导裁决。纯计算。"""
        out = []
        for e in self.dismiss_pending():
            hits = self.retrieve(e["content"],
                                 budget=budget if budget is not None else self.retrieval_budget)
            atoms = self.expand_leaves(hits)
            planted_round = e.get("updated_round", 0)
            for aid in atoms:
                if self.atoms[aid].created_round < planted_round:
                    self.atoms[aid].metadata["pending_dismiss"] = True
                    out.append(aid)
        return out

    def dismiss_confirm(self, atom_ids):
        """裁决成立：墓碑指定原子（编辑路径，可回滚），懒标记全部 payback 结案，
        清除残留的 pending_dismiss 标注（存活卡的标注随结案失效）。"""
        dead = [aid for aid in atom_ids if aid in self.atoms and self.atoms[aid].alive]
        if not dead:
            return []
        result = self.delete(dead)
        for e in self.dismiss_pending():
            e["status"] = "payback"
        for a in self.atoms.values():
            a.metadata.pop("pending_dismiss", None)
        return result

    def dismiss_cancel(self):
        """裁决不成立：关闭懒标记 + 清除 pending_dismiss 标注。"""
        for e in self.dismiss_pending():
            e["status"] = "closed"
        for a in self.atoms.values():
            a.metadata.pop("pending_dismiss", None)

    # ── 区间全原子回调（[span] 工具）───────────────────

    def span_atoms(self, nid):
        """回调摘要卡 nid 覆盖区间内的全部存活原子（按位置序、原文全分辨率）。
        取 [span_first_atom, span_last_atom] 沿顺序链收集，受 SPAN_ATOM_CAP/SPAN_CHAR_CAP 限制。
        失败返回 None（不可用于组装）；成功返回原子 id 列表（可为空）。"""
        if nid not in self.nodes or not self.nodes[nid].alive:
            return None
        lo = self._first_atom(nid)
        hi = self._last_atom(nid)
        if lo is None or hi is None:
            return []
        chain = self.chain_ids()
        try:
            i1, i2 = chain.index(lo), chain.index(hi)
        except ValueError:
            return []
        if i1 > i2:
            i1, i2 = i2, i1
        out, chars = [], 0
        for aid in chain[i1:i2 + 1]:
            if len(out) >= SPAN_ATOM_CAP or chars >= SPAN_CHAR_CAP:
                break
            if self.atoms[aid].alive:
                out.append(aid)
                chars += len(self.atoms[aid].value or "")
        return out

    # ── 叶子展开（C：命中概括卡 → 展开全部后代原子为影子）──────────

    def expand_leaves(self, items, budget=None):
        """把检索命中的节点展开为其**全部后代存活原子**（按位置序、原文全分辨率），
        原子命中保持原样。用于影子注入：概括卡被激活 → 其下原子段整体可见（活一轮）。
        受 EXPAND_ATOM_CAP/EXPAND_CHAR_CAP 限制（先拍后调）。"""
        cap_atoms = self.expand_atom_cap
        cap_chars = self.expand_char_cap
        if budget is not None:
            cap_chars = min(cap_chars, budget)
        out, seen = [], set()
        chars = 0
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            if self._is_atom(item):
                if self.atoms[item].alive:
                    out.append(item)
                    chars += len(self.atoms[item].value or "")
                continue
            if not (self.nodes[item].alive and not self.nodes[item].dirty):
                continue
            for aid in self._leaf_atoms(item):
                if aid in seen:
                    continue
                seen.add(aid)
                if self.atoms[aid].alive:
                    out.append(aid)
                    chars += len(self.atoms[aid].value or "")
                    if len(out) >= cap_atoms or chars >= cap_chars:
                        return out
        return out

    # ── 组装（按字符预算，真实文本）────────────────────

    def item_chars(self, item):
        if self._is_atom(item):
            return len(self.atoms[item].value or "")
        return len(self.nodes[item].value or "")

    def emitted_chars(self, emitted):
        total = 0
        for kind, item, _value in emitted:
            total += self.item_chars(item) + 14
        return total

    def assemble(self, shadows=None, budget=None):
        self._src_cache = {}   # 来源聚合缓存每轮组装重算（节点重建后自动失效）
        b = budget if budget is not None else self.assembly_budget
        emitted = self._emit()
        if shadows:
            emitted = self._inject_shadows(emitted, shadows)
        if self.emitted_chars(emitted) > b:
            self.merge_pass()
            emitted = self._emit()
            if shadows:
                emitted = self._inject_shadows(emitted, shadows)
        if self.emitted_chars(emitted) > b:
            emitted = self._trim(emitted, b)
        return emitted

    def _trim(self, emitted, budget):
        node_positions = [(i, e) for i, e in enumerate(emitted) if e[0] in ("node", "shadow")]
        node_positions.sort(key=lambda ip: self._item_hp(ip[1][1]))
        out = list(emitted)
        trimmed = []
        for i, _ in node_positions:
            if self.emitted_chars(out) <= budget:
                break
            trimmed.append(out[i])
            out = [e for j, e in enumerate(out) if j != i]
        if self.emitted_chars(out) > budget:
            # 兜底：仍超 → 尾部硬截（记录被截项明细：kind/id/HP/字符）
            over = self.emitted_chars(out) - budget
            dropped = []
            while out and self.emitted_chars(out) > budget:
                e = out.pop()
                dropped.append((e[0], e[1], self._item_hp(e[1]),
                                len(self.atoms[e[1]].value or "") if self._is_atom(e[1])
                                else len(self.nodes[e[1]].value or "")))
            self.events.append(("hard_trim", over, dropped))
        return out

    # ── 组装（覆写：budget = 字符数；core 桩保持数值替身语义）────────

    def assemble(self, shadows=None, budget=None) -> list:
        if budget is None:
            budget = self.assembly_budget
        emitted = self._emit()
        if shadows:
            emitted = self._inject_shadows(emitted, shadows)
        if self._emission_chars(emitted) > budget:
            self.merge_pass()
            emitted = self._emit()
            if shadows:
                emitted = self._inject_shadows(emitted, shadows)
        if self._emission_chars(emitted) > budget:
            emitted = self._trim_chars(emitted, budget)
        return emitted

    def _emission_chars(self, emitted) -> int:
        return len(self.render_emission(emitted))

    def _trim_chars(self, emitted, budget) -> list:
        """字符预算裁剪：低 HP 节点/影子先裁（原子卡永不裁）；仍超 → 硬截断记日志。"""
        lens = [len(self.render_emission([e])) + 1 for e in emitted]
        total = sum(lens)
        idxs = [i for i, e in enumerate(emitted) if e[0] in ("node", "shadow")]
        idxs.sort(key=lambda i: self._item_hp(emitted[i][1]))
        drop = set()
        for i in idxs:
            if total <= budget:
                break
            drop.add(i)
            total -= lens[i]
        out = [e for i, e in enumerate(emitted) if i not in drop]
        if total > budget:
            kept, t = [], 0
            for e in out:
                ln = len(self.render_emission([e])) + 1
                if kept and t + ln > budget:
                    break
                kept.append(e)
                t += ln
            if len(kept) < len(out):
                self.events.append(("hard_trim", len(out) - len(kept)))
            out = kept
        return out

    # ── 渲染（给 LLM 看的记忆流，含从属标注）────────────

    def render_emission(self, emitted, fold_files=False, unfold_ids=None):
        """fold_files=True（代码场景拍定 2026-08-30）：文件原子折叠为"首行索引"
        （AST 骨架语义：只显示 class/def 签名行，正文必须 read 获取——View B 唯一
        内容真源，消除"槽位第二抄本"的锚点抄写污染）。
        unfold_ids：召回命中的文件原子（在 tiling 中、被影子注入跳过）→ 展开逐字，
        召回语义 = 把原文拉回视野。"""
        unfold = set(unfold_ids or ())
        lines = []
        for kind, item, _value in emitted:
            if kind == "atom":
                a = self.atoms[item]
                src = (a.metadata or {}).get("source") or ""
                if fold_files and src.startswith("file:") and item not in unfold:
                    first_line = (a.value or "").splitlines()[0] if (a.value or "").splitlines() else ""
                    if not first_line.strip():
                        first_line = "…"
                    elif len(first_line) > 90:
                        first_line = first_line[:90] + "…"
                    fname = src.split("file:", 1)[1].rsplit("/", 1)[-1]
                    lines.append(f"[#{item}·文件索引〔{fname}〕] {first_line}")
                    continue
                parent = a.owner
                depth = len(self._owner_chain(item))
                mark = f" ⊂N{parent}" if parent is not None else ""
                src_tag = f"〔{src}〕" if src else ""
                pend = "〔待失效〕" if (a.metadata or {}).get("pending_dismiss") else ""
                rep = "〔已回复〕" if (a.metadata or {}).get("replied") else ""
                lines.append(f"{'  ' * depth}[#{item}·原文{mark}{src_tag}{pend}{rep}] {a.value}")
            elif kind == "shadow" and self._is_atom(item):
                a = self.atoms[item]
                parent = a.owner
                depth = len(self._owner_chain(item))
                mark = f" ⊂N{parent}" if parent is not None else ""
                src = (a.metadata or {}).get("source")
                src_tag = f"〔{src}〕" if src else ""
                pend = "〔待失效〕" if (a.metadata or {}).get("pending_dismiss") else ""
                rep = "〔已回复〕" if (a.metadata or {}).get("replied") else ""
                lines.append(f"{'  ' * depth}[#{item}·召回原文{mark}{src_tag}{pend}{rep}] {a.value}")
            else:
                nd = self.nodes[item]
                parent = nd.parent
                depth = self._depth(item)
                tag = "召回影子" if kind == "shadow" else "摘要"
                mark = f" ⊂N{parent}" if parent is not None else ""
                # 程序维护的区间来源标注（不依赖蒸馏；防"来源漂移"在展示层可见）
                srcs = self._node_sources(item)
                src_tag = ""
                if srcs:
                    clean = sorted(s for s in srcs if s and s != "?")
                    if clean:
                        src_tag = f"〔源:{'/'.join(clean)}〕"
                if nd.value is None:
                    content = "【占位：近期修改，概要暂缺】"
                else:
                    content = nd.value
                lines.append(f"{'  ' * depth}[N{item}·{tag}{mark}{src_tag}] {content}")
        return "\n".join(lines)

    def story_full_text(self):
        parts = []
        cur = self.head
        while cur is not None:
            parts.append(self.atoms[cur].value)
            cur = self.atoms[cur].next
        return "\n".join(parts)
