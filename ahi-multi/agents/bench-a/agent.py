"""agent.py — ChatAHIAgent：水循环记忆树接入 AHI 平台（实验组）。

架构：AHI 唤醒骨架（调度 / @wait_for / @mute / 动作执行 / 命令）+ RealEngine
记忆树 + realtest 无状态 LLM（DeepSeek）。每轮全新组装（零对话历史，
记忆全在树里）。

融合决策（见项目计划）：
- 代码/文字分离：```python → shell 执行，结果截断后以 source=tool 进树，
  代码本身不进树；```txt/正文 → 切句进树（source=信道/ai）。
- 注入轨道分离：角色提示词（persona 开关）→ AHI 系统状态 → 上次唤醒回忆
  → 轨道A 卡片流 → 轨道B 笔记本（去重发送）→ 回执 → 当前任务。
- 工具 = @ 命令风格：@note / @recall / @span / @dismiss / @ingest。
- persona 实验：persona_drop_after_rounds / persona_drop_after_atoms 任一满足
  → system 切 persona_remove_prompt 变体（人格段移除），事件记日志。
"""
import collections
import json
import os
import re
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
for _p in (_PROJ, os.path.join(_PROJ, "realtest")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sdk.base_agent import BaseAHIAgent

from realtest.chattext import split_chat, split_media
from realtest.llmclient import make_client as make_realtest_client
from realtest.realengine import RealEngine


def render_notebook(rows):
    if not rows:
        return ""
    parts = []
    for kind, key, content in rows:
        if kind == "constraint":
            parts.append(f"[约束] {content}")
        elif kind == "state":
            parts.append(f"[状态·{key}] {content}")
        elif kind == "schedule":
            tag = "冲突待决" if (key or "").startswith("dismiss_") else "待办"
            parts.append(f"[{tag}·{key}] {content}")
        elif kind == "worklog":
            parts.append(f"[工作记录·{key}] {content}")
    return "\n".join(parts)


# 风格快照刷新 prompt（八字段，认知向——不规定任何符号形式，让形式自然涌现）
SNAPSHOT_PROMPT = (
    "基于下面给出的你的原话样本（以及旧快照），生成你的风格快照。\n"
    "八字段，每字段一行，格式严格为「字段名: 内容」：\n"
    "表达基调: <一句话概括你的气质>\n"
    "互动习惯: <你怎么提问、怎么回应、怎么延伸话题>\n"
    "称呼方式: <你对用户和其他人的称呼习惯>\n"
    "认知组织: <你怎么组织想法：先想后说？拆成几步？直觉先行？>\n"
    "比喻体系: <你惯用的意象与比喻>\n"
    "立场: <你坚持的价值与立场>\n"
    "着重点: <你概括与复述信息时，习惯优先展开什么、折叠什么（例：先讲人际与情绪变化、略过技术参数）>\n"
    "其他: <清单未覆盖的表达习惯，自由写>\n"
    "规则：描述认知习惯，不要规定具体符号（不写'必须用某标点/某收尾语'）；"
    "旧快照中仍准确的内容保留，新变化吸收；与旧快照冲突时以新样本为准。\n"
    "{old}"
    "【你的原话样本】\n{samples}"
)


class ChatAHIAgent(BaseAHIAgent):
    """水循环记忆树 × AHI 平台（实验组）。"""

    def __init__(self):
        super().__init__()
        self._cfg = {}
        self._realtest_llm = None
        self.eng = None
        # persona 实验开关
        self._persona_on = True
        self._persona_drop_rounds = 50
        self._persona_drop_atoms = 800
        self._system_persona = ""
        self._persona_remove_prompt = ""
        # 人格染色（迭代）：风格快照（七字段，EMA+两轮确认）+ 自我样本池
        self._snapshot = None               # 风格快照文本
        self._snapshot_pending = {}         # 待确认字段池 {field: {"content": str, "seen": int}}
        self._snapshot_every = 0            # 快照刷新间隔轮数（0 = 关闭染色，neutral 用）
        self._self_samples = collections.deque(maxlen=6)   # 近期自我原话样本
        # 记忆树循环参数
        self._merge_every = 4
        self._budget = 10000
        self._retr_budget = 6
        self._max_pending_per_loop = 2    # 决策 ③：每轮最多处理 pending 消息数（分片）
        self._max_pre_distill = 2         # 决策 ③：每轮最多启动预蒸馏任务数
        self._force_cooldown = 4          # 决策 ③：force 合并冷却轮数
        self._max_code_per_loop = 4          # R1 护栏：每轮代码执行上限
        self._last_force_round = -99
        self._last_shadows = None
        self._last_topic = ""
        self._round_no = 0
        self._receipt = "初始化：记忆树就绪，开始对话。"
        self._metrics_f = None
        self._last_user_channel = None   # 本轮最近用户消息信道（私密回复兜底目标）
        self._llm_calls = 0              # LLM 调用成本统计（含蒸馏外的回复调用）
        self._llm_seconds = 0.0
        self._pending_left = 0           # 指标：本轮分片剩余的 pending 数
        # 观测层（纯记录，零行为干预）：本轮工具/ai 原子 + 快照路径
        self._round_tool_ids = []
        self._round_ai_ids = []
        self._recent_tool_atoms = collections.deque(maxlen=60)  # 声称×执行匹配池（近轮 tool 原子）
        self._tree_snap_every = 10       # 树快照间隔（轮）
        self._assembly_snap_every = 4    # 组装快照间隔（轮）
        self._tree_snap_path = ""
        self._assembly_snap_path = ""
        self._claim_log_path = ""

    # ── 生命周期 ──────────────────────────────────

    def on_start(self) -> None:
        cfg = self._load_agent_config()
        self._cfg = cfg
        self._sandbox_root = (cfg.get("sandbox", {}) or {}).get("root") or ""
        self._persona_drop_rounds = int(cfg.get("persona_drop_after_rounds", 50))
        self._persona_drop_atoms = int(cfg.get("persona_drop_after_atoms", 800))
        self._persona_drop_nodes = int(cfg.get("persona_drop_after_nodes", 15))
        self._system_persona = cfg.get("system_prompt", "")
        self._persona_remove_prompt = cfg.get("persona_remove_prompt", self._system_persona)
        self._merge_every = int(cfg.get("merge_every", 4))
        self._budget = int(cfg.get("assembly_budget", 10000))
        self._retr_budget = int(cfg.get("retrieval_budget", 6))
        self._max_pending_per_loop = int(cfg.get("max_pending_per_loop", 2))
        self._max_pre_distill = int(cfg.get("max_pre_distill", 2))
        self._force_cooldown = int(cfg.get("force_cooldown_rounds", 4))
        self._max_code_per_loop = int(cfg.get("max_code_per_loop", 4))
        self._max_loop_iterations = int(cfg.get("max_loop_iterations", 20))

        # 无状态 LLM（DeepSeek/Ollama 统一，每轮全新组装）
        self._realtest_llm = make_realtest_client(cfg)

        # 记忆树引擎
        self.eng = RealEngine(
            self._realtest_llm,
            decay=float(cfg.get("decay", 8.0)),
            hp_merge_threshold=float(cfg.get("hp_merge_threshold", 30.0)),
            parent_init_hp=float(cfg.get("parent_init_hp", 70.0)),
            hot_threshold=float(cfg.get("hot_threshold", 60.0)),
            recall_boost=float(cfg.get("recall_boost", 30.0)),
            merge_cap_atoms=int(cfg.get("merge_cap_atoms", 500)),
            merge_depth_diff=int(cfg.get("merge_depth_diff", 1)),
            promote_threshold=int(cfg.get("promote_threshold", 3)),
            attention_cap=int(cfg.get("attention_cap", 60)),
            attention_atom_cap=int(cfg.get("attention_atom_cap", 40)),
            attention_node_cap=int(cfg.get("attention_node_cap", 20)),
            attention_out=float(cfg.get("attention_out", 45.0)),
            attention_in_decay=float(cfg.get("attention_in_decay", 0.5)),
            attention_out_decay=float(cfg.get("attention_out_decay", 3.0)),
            merge_fanout_cap=int(cfg.get("merge_fanout_cap", 12)),
            pressure_threshold=float(cfg.get("pressure_threshold", 2.0)),
            merge_pressure_threshold=int(cfg.get("merge_pressure_threshold", 20)),
            expand_atom_cap=int(cfg.get("expand_atom_cap", 80)),
            expand_char_cap=int(cfg.get("expand_char_cap", 3000)),
            assembly_budget=self._budget,
            retrieval_budget=self._retr_budget,
            distill_parallel=int(cfg.get("distill_parallel", 4)),
            style_alpha=float(cfg.get("style_alpha", 1.0)),
            distill_snap_path=os.path.join(self._agent_dir, "data", "distill_snapshots.jsonl"),
        )
        # 人格染色开关：snapshot_every_rounds > 0 且 α > 0 → 注入风格指南（快照+样本）
        self._style_alpha = float(cfg.get("style_alpha", 1.0))   # 染色强度：0/0.5/1（染色 v3）
        self._snapshot_every = int(cfg.get("snapshot_every_rounds", 0))
        if self._style_alpha > 0 and self._snapshot_every > 0:
            self.eng.style_provider = self._style_guide
        # 轨道B 预置：交互规则（constraint，恒注入）
        self.eng.nb_set(
            "交互规则", "constraint",
            "你是带持久记忆的聊天智能体：跨会话记住用户的事实/偏好/承诺；"
            "回答依据记忆流；记忆流中没有的不得编造。")

        # 恢复持久化设置（mute/exec_timeout，父类逻辑）
        if self.db:
            try:
                muted_raw = self.db.get_setting("muted_channels")
                if muted_raw:
                    self._muted = set(json.loads(muted_raw))
                t = self.db.get_setting("exec_timeout")
                if t:
                    self._exec_timeout = max(1, int(t))
            except Exception:
                pass
            self.db.add_log("INFO", self.agent_name,
                            f"ChatAHIAgent started, model={cfg.get('provider')}/"
                            f"{(cfg.get('deepseek') or {}).get('model', '?')}, "
                            f"persona_drop: rounds={self._persona_drop_rounds} "
                            f"atoms={self._persona_drop_atoms}")

        # 指标记录
        try:
            metrics_path = os.path.join(self._agent_dir, "data", "metrics.jsonl")
            self._metrics_f = open(metrics_path, "a", encoding="utf-8")
        except Exception:
            self._metrics_f = None

        # 观测层快照路径（纯记录）
        self._tree_snap_path = os.path.join(self._agent_dir, "data", "tree_snapshots.jsonl")
        self._assembly_snap_path = os.path.join(self._agent_dir, "data", "assembly_snapshots.jsonl")
        self._claim_log_path = os.path.join(self._agent_dir, "data", "action_claims.jsonl")

    def on_stop(self) -> None:
        if self._metrics_f:
            try:
                self._metrics_f.close()
            except Exception:
                pass
        # 观测层：关停前最后一份树快照（Phase A 教训：树随进程死亡，快照是唯一尸检材料）
        if self.eng is not None and self._tree_snap_path:
            try:
                self.eng.dump_tree(self._tree_snap_path)
            except Exception:
                pass
        if self.db:
            self.db.add_log("INFO", self.agent_name, "Agent stopped")

    # ── LLM（覆写：无状态 chat，B6 失败只发用户）────────

    def _call_llm(self, prompt: str) -> str | None:
        try:
            text, _usage, _lat = self._realtest_llm.chat(self._system_prompt(), prompt)
            self._llm_calls += 1
            self._llm_seconds += float(_lat or 0.0)
            return text
        except Exception as e:
            self._llm_calls += 1
            self._put_action({
                "type": "message",
                "content": f"LLM 调用失败: {str(e)}",
                "to": "broadcast:users",
            })
            return None

    def _system_prompt(self) -> str:
        return self._system_persona if self._persona_on else self._persona_remove_prompt

    # ── 人格染色（迭代）：风格快照 + 自我样本池 ──────────────

    def _style_guide(self) -> str:
        """蒸馏风格指南（style_provider）：α ≥0.5 注入快照（认知习惯+着重点），
        α ≥1 加近期自我原话样本。未开启染色（α=0 或 snapshot_every=0）不会被调用。"""
        parts = []
        if self._style_alpha >= 0.5 and self._snapshot:
            parts.append("## 你的风格快照（按这些认知习惯与着重点写摘要，勿规定符号）\n" + self._snapshot)
        if self._style_alpha >= 1.0 and self._self_samples:
            parts.append("## 你的近期原话样本\n" + "\n".join(f"- {s}" for s in self._self_samples))
        return "\n\n".join(parts) if parts else ""

    def _refresh_snapshot(self):
        """快照刷新（每 snapshot_every 轮）：LLM 生成七字段自述 → 程序锚点合并
        （EMA：冲突字段进待确认池，两轮确认才生效；常规字段直接更新）。"""
        try:
            samples = "\n".join(f"- {s}" for s in self._self_samples) or "（暂无样本）"
            old = (f"\n【旧快照】\n{self._snapshot}\n" if self._snapshot else "")
            prompt = SNAPSHOT_PROMPT.format(old=old, samples=samples)
            text, _usage, _lat = self._realtest_llm.chat("你是风格自我描述助手。", prompt)
            self._llm_calls += 1
            self._llm_seconds += float(_lat or 0.0)
            new_fields = self._parse_snapshot(text or "")
            if new_fields:
                self._snapshot = self._merge_snapshot(new_fields)
        except Exception as e:
            if self.db:
                self.db.add_log("WARNING", self.agent_name, f"快照刷新失败: {e}")

    @staticmethod
    def _parse_snapshot(text):
        """七字段解析：'字段名: 内容' 行。返回 {字段: 内容}。"""
        out = {}
        for line in (text or "").splitlines():
            m = re.match(r"\s*([^\s:：]+)[:：]\s*(.+)", line.strip())
            if m:
                name = m.group(1).strip()
                if name in ("表达基调", "互动习惯", "称呼方式", "认知组织",
                            "比喻体系", "立场", "着重点", "其他"):
                    out[name] = m.group(2).strip()
        return out

    def _merge_snapshot(self, new_fields):
        """程序锚点合并（EMA + 两轮确认）：
        - 旧快照无此字段 → 新增，直接进
        - 相似（bigram 重叠 ≥0.5）→ 常规更新
        - 重大变化（<0.5）→ 待确认池；下一轮仍出现（≥0.6）→ 确认进主体
        - 旧字段本轮未出现 → 保留旧值（消失字段不即删）
        """
        from realtest.realengine import bigrams
        def overlap(a, b):
            ba, bb = bigrams(a or ""), bigrams(b or "")
            if not ba or not bb:
                return 0.0
            # 分母取较短文本：语义 = "旧内容有多少还保留在新内容里"（短字段友好）
            return len(ba & bb) / min(len(ba), len(bb))
        old_fields = self._parse_snapshot(self._snapshot or "")
        merged = dict(old_fields)
        for name, content in new_fields.items():
            old = old_fields.get(name)
            if old is None:
                merged[name] = content
            elif overlap(old, content) >= 0.5:
                merged[name] = content
            else:
                pend = self._snapshot_pending.get(name)
                if pend is not None and overlap(pend["content"], content) >= 0.6:
                    merged[name] = content
                    self._snapshot_pending.pop(name)
                else:
                    self._snapshot_pending[name] = {"content": content}
        order = ("表达基调", "互动习惯", "称呼方式", "认知组织", "比喻体系", "立场",
                 "着重点", "其他")
        lines = [f"{k}: {merged[k]}" for k in order if k in merged]
        for k in merged:
            if k not in order:
                lines.append(f"{k}: {merged[k]}")
        return "\n".join(lines)

    def _parse_send_to(self, cmd_str: str) -> list:
        """覆写：剔除自己的地址（agent:<id> / agent:<name>）——模型偶尔把自己地址
        列为收件人（"抄送平台"误解），全剔除后若无其余收件人 → 走私密兜底自动回用户。"""
        targets = super()._parse_send_to(cmd_str)
        self_addrs = {f"agent:{self.agent_id}", f"agent:{self.agent_name}"}
        return [t for t in targets if t not in self_addrs]

    # ── persona 实验 ───────────────────────────────

    def _check_persona_drop(self) -> bool:
        """人格保留到概括层成形（2026-08-29 一天实验拍定）：
        双条件同时满足才移除——① 轮次 ≥ 下限（保底约 1 小时）；
        ② 树中存活摘要节点 ≥ 阈值（"多个概括节点"= 染色已通过快照+蒸馏
        进入摘要层，移除后人格仍留在概括角度；早移除=人格没机会染进树）。
        旧语义（rounds OR atoms 任一触发）在 1 小时尺度上 20 分钟即触发。"""
        if not self._persona_on:
            return False
        st = self.eng.stat()
        nodes = st["nodes"]["alive"]
        if self._round_no >= self._persona_drop_rounds and nodes >= self._persona_drop_nodes:
            self._persona_on = False
            reason = (f"round={self._round_no}≥{self._persona_drop_rounds}"
                      f" 且 nodes={nodes}≥{self._persona_drop_nodes}（概括层已成形）")
            self._last_wakeup_actions.append({"type": "persona_drop", "reason": reason})
            if self.db:
                self.db.add_log("INFO", self.agent_name,
                                f"persona 移除触发: {reason}（人格段不再注入）")
            return True
        return False

    # ── 自主唤醒循环（覆写：记忆树上下文构建）─────────

    def _autonomous_loop(self):
        loop_start = time.time()
        self._last_wakeup_actions = []
        ctx_chars = 0
        pending_count = 0
        pending_brief = ""   # 本轮待回复清单（P1 修复：给模型明确收件人）
        # 观测层：本轮工具/ai 原子追踪（声称×执行匹配用，仅记录）
        self._round_tool_ids = []
        self._round_ai_ids = []
        try:
            self._fetch_system_state()
            pending_snapshot = self._drain_pending_messages()
            pending_count = len(pending_snapshot)
            self._round_no += 1

            # 决策 ③ pending 分片：每轮最多处理 N 条，其余放回队首（保持时间序）
            # ——单轮上下文变小、待回复清单更清晰（缓解多用户目标混乱）、延迟下降
            self._pending_left = 0
            if len(pending_snapshot) > self._max_pending_per_loop:
                keep, rest = (pending_snapshot[:self._max_pending_per_loop],
                              pending_snapshot[self._max_pending_per_loop:])
                with self._msg_lock:
                    self._pending_messages = rest + self._pending_messages
                pending_snapshot = keep
                self._pending_left = len(rest)

            # 0. 待回复清单：按来源信道汇总（模型凭此确定 @send-to 目标，不再猜）
            if pending_snapshot:
                brief_lines = []
                for m in pending_snapshot:
                    ch = m.get("channel") or m.get("source", "unknown")
                    brief = (m.get("content") or "")[:60].replace("\n", " ")
                    brief_lines.append(f"- {ch}: {brief}")
                pending_brief = "本轮待回复清单（请对每个来源各回复一条，@send-to 用下面的信道）：\n" \
                                + "\n".join(brief_lines) + \
                                "\n注意：清单里没有的地址不要用（如 user:admin / user:all 都不存在）。"

            # 1. 写树：pending 消息切句（source=信道）；记录本轮新卡排除集
            new_ids_all = []
            self._last_user_channel = None
            for m in pending_snapshot:
                content = m.get("content", "")
                if not content or not content.strip():
                    continue
                channel = m.get("channel") or m.get("source", "unknown")
                if channel.startswith("user:"):
                    self._last_user_channel = channel   # 取最后一条（最近说话的用户）
                units = split_chat(content, source=channel)
                if units:
                    ids = self.eng.insert_units(units, left_id=self.eng.tail)
                    new_ids_all.extend(ids)

            # 2. 重复输入检测（排除本轮新卡）→ 旧卡活一轮影子
            shadows = list(self._last_shadows or [])
            for m in pending_snapshot:
                content = m.get("content", "")
                if not content:
                    continue
                for t, _ in split_chat(content, source="probe"):
                    sims = self.eng.find_similar(t, exclude_ids=new_ids_all)
                    shadows.extend(sims)

            # 3. DISMISS 持续追踪：planted 懒标记命中旧卡 → 待失效影子
            shadows.extend(self.eng.dismiss_track())
            shadows = list(dict.fromkeys(shadows))
            self._last_shadows = shadows or None

            # 4. persona 检查（触发后本轮即切变体）
            dropped = self._check_persona_drop()

            # 5. 组装（轨道分离）→ LLM
            user_ctx = self._build_user_ctx(dropped, pending_brief)
            ctx_chars = len(user_ctx)
            self._last_shadows = None   # 影子"活一轮"落地：组装消费后清空，
                                        # 动作循环里 @recall/@span 等会为下轮重新创建
            reply = self._call_llm(user_ctx)
            if reply is None:
                return

            # 6. 解析 + 动作循环（复用 AHI：@send-to/块/@命令/@exit，最多 N 轮）
            actions, remaining = self._parse_structured_response(reply)
            loop_count = 0
            code_used = 0
            actions_to_process = actions[:]
            while actions_to_process and loop_count < self._max_loop_iterations:
                action = actions_to_process.pop(0)
                if action.get("_exit"):
                    break
                # 代码执行上限（R1 护栏）：探索人格的"执行风暴"由反馈循环放大，
                # 每轮最多 _max_code_per_loop 次，超出转提示（人格不变，只限频率）
                if action.get("type") == "code" and code_used >= self._max_code_per_loop:
                    result = (f"【护栏】本轮代码执行已达上限（{self._max_code_per_loop} 次），"
                              f"本次未执行。请用文本回复，或下轮再执行代码。")
                else:
                    if action.get("type") == "code":
                        code_used += 1
                    result = self._execute_action(action)
                if result:
                    feedback = f"【{action.get('type', 'action')} 结果】\n{result}"
                    reply = self._call_llm(feedback)
                    if reply is None:
                        break
                    new_actions, new_remaining = self._parse_structured_response(reply)
                    actions_to_process = new_actions + actions_to_process
                    if new_remaining:
                        remaining = (remaining + "\n" + new_remaining) if remaining else new_remaining
                loop_count += 1
            if actions_to_process:
                dropped_kinds = [a.get("type", "?") for a in actions_to_process]
                self._last_wakeup_actions.append(
                    {"type": "dropped", "count": len(actions_to_process),
                     "kinds": dropped_kinds})

            # 7. 衰减 + 蒸馏时机
            self.eng.tick(1)
            if self.eng.attention_pressure() and \
                    (self._round_no - self._last_force_round) >= self._force_cooldown:
                self.eng.merge_pass(force_out=True)
                self._last_force_round = self._round_no
            elif self.eng.merge_pressure():
                # 2026-08-30 拍定：常规 merge 条件式触发（每轮查，窗外节点+原子
                # 数超 merge_pressure_threshold 就蒸）——替代时间式 merge_every
                self.eng.merge_pass()

            # 决策 ③ 预蒸馏：预测下轮 merge 候选 → 后台先蒸馏（下轮命中复用）
            if self._max_pre_distill > 0:
                self.eng.pre_distill_launch(max_tasks=self._max_pre_distill)

            # 人格染色：风格快照刷新（每 snapshot_every 轮，EMA+两轮确认）
            if self._snapshot_every > 0 and self._round_no % self._snapshot_every == 0:
                self._refresh_snapshot()

            # 观测层（纯记录，零行为干预）：声称×执行匹配日志 + 树快照
            self._record_action_claims()
            if self._round_no % self._tree_snap_every == 0 and self.eng is not None:
                self.eng.dump_tree(self._tree_snap_path)

            # 8. 回执
            self._receipt = f"第 {self._round_no} 轮完成（persona={'注入' if self._persona_on else '已移除'}）。"

            self._last_wakeup_summary = self._build_wakeup_summary()

            if self.ahi_bus and not self._output_queue.empty():
                self.ahi_bus.notify_has_outputs_sync()

        finally:
            self._thinking = False
            # 指标在 finally 必记（LLM 失败 break 的轮次也留痕）
            self._record_metrics(ctx_chars, pending_count)
            elapsed = time.time() - loop_start
            if self.db:
                try:
                    actions_json = json.dumps(self._last_wakeup_actions,
                                              ensure_ascii=False)[:1000]
                except Exception:
                    actions_json = str(self._last_wakeup_actions)[:1000]
                self.db.complete_loop_record(
                    getattr(self, "_cur_loop_id", None) or 0,
                    result=actions_json,
                    error=f"elapsed={elapsed:.1f}s")

    # ── 组装（注入轨道分离）────────────────────────

    def _build_user_ctx(self, dropped: bool, pending_brief: str = "") -> str:
        eng = self.eng
        nb_rows = eng.nb_render(self._last_topic or "")
        nb = render_notebook(nb_rows)
        emitted = eng.assemble(self._last_shadows, budget=self._budget)
        flow = eng.render_emission(emitted)

        parts = []
        state = self._render_system_state().strip()
        if state:
            parts.append(state)
        if self._last_wakeup_summary:
            parts.append(f"【上次唤醒回忆】\n{self._last_wakeup_summary}")
        parts.append(flow)
        if pending_brief:
            parts.append(pending_brief)
        if nb:
            parts.append("【笔记本·事实层】（去重发送：每项只保留最新，本轮现场生成）\n" + nb)
        if self._receipt:
            parts.append(f"【最近操作回执】\n{self._receipt}")
        task = (f"回复用户（第 {self._round_no} 轮；话题关键词：{self._last_topic or '无'}）。"
                f"用户消息已在记忆流末尾（最近几项）。")
        if dropped:
            task += "〔注意：本轮起角色描述不再注入，请继续以你一贯的方式交流。〕"
        parts.append(f"【当前任务】\n{task}")
        user_ctx = "\n\n".join(parts)
        # 观测层：组装快照落盘（每 4 轮，纯记录——Phase A 数据缺口教训）
        if self._round_no % self._assembly_snap_every == 0 and self._assembly_snap_path:
            try:
                with open(self._assembly_snap_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"ts": round(time.time(), 2),
                                        "round": self._round_no,
                                        "ctx_chars": len(user_ctx),
                                        "ctx": user_ctx}, ensure_ascii=False) + "\n")
            except Exception:
                pass
        return user_ctx

    def _build_wakeup_summary(self) -> str:
        lines = []
        for a in self._last_wakeup_actions:
            t = a.get("type", "")
            if t == "message":
                content = str(a.get("content", ""))[:80].replace("\n", " ")
                lines.append(f"发消息→{a.get('to', '?')}: {content}")
            elif t == "code":
                code = str(a.get("code", ""))[:60].replace("\n", " ")
                lines.append(f"执行代码: {code}")
            elif t == "command":
                lines.append(f"命令: {a.get('command', '')} {' '.join(a.get('args', []))}".strip())
            elif t == "dropped":
                lines.append(f"超限未执行 {a.get('count', 0)} 个动作({','.join(a.get('kinds', []))})")
            elif t == "note":
                lines.append(f"记录[{a.get('kind', 'state')}]{a.get('key', '')}: {str(a.get('content', ''))[:50]}")
            elif t == "recall":
                lines.append(f"回想「{a.get('query', '')}」→ 注入 {a.get('hits', 0)} 项")
            elif t == "span":
                lines.append(f"下钻 N{a.get('node', '')}")
            elif t == "dismiss":
                lines.append(f"冲突处理: {a.get('action', '')}")
            elif t == "ingest":
                lines.append(f"读入文件 {a.get('path', '')}（{a.get('units', 0)} 卡）")
            elif t == "persona_drop":
                lines.append(f"人格段移除（{a.get('reason', '')}）")
        return "；".join(lines)[:600]

    # ── 动作执行（覆写：AI 消息/代码结果写回记忆树）────

    def _execute_action(self, action: dict) -> str | None:
        atype = action.get("type", "")
        if atype == "message":
            content = action.get("content", "")
            # 护栏（决策 ②）：命令泄漏 → 不发送不写树，执行结果反馈 LLM
            guard = self._guard_message_content(content)
            if guard is not None:
                return guard
            to = action.get("to", "")
            # 兜底：模型未指定收件人（私密）但本轮有用户消息 → 回给最近用户，
            # 保证"用户提问必得回复"（实验连续性）；显式 @send-to 的目标优先。
            if not to and self._last_user_channel:
                to = self._last_user_channel
                action["to"] = to
                action["private"] = False
            if content and content.strip():
                # 协议污染过滤：模型偶尔把 @send-to/```/@note 等协议标记写进消息文本
                # （模仿协议格式），不写记忆树（垃圾源），但照常发送（平台层处理）。
                head = content.lstrip()[:40]
                if "@send-to" in head or "```" in head or head.startswith("@note") or head.startswith("@exit"):
                    if self.db:
                        self.db.add_log("WARNING", self.agent_name,
                                        f"协议文本泄漏已过滤（不写树）: {head[:30]}…")
                else:
                    units = split_chat(content, source="ai")
                    if units:
                        # 自我样本池（染色素材，deque 自动限长）
                        for _t, _m in units:
                            if len(_t) >= 12:
                                self._self_samples.append(_t[:80])
                        ids = self.eng.insert_units(units, left_id=self.eng.tail)
                        for aid in ids:
                            self.eng.atoms[aid].metadata["to"] = action.get("to", "")
                        self._round_ai_ids.extend(ids)
                        # 已回复标记（仅被动观测：渲染〔已回复〕，不入提示词不引导行为）——
                        # AI 向 X 发消息后，X 此前全部存活原子标 replied（分析"复读"来源用）
                        if to:
                            for aid, a in self.eng.atoms.items():
                                if a.alive and (a.metadata or {}).get("source") == to:
                                    a.metadata["replied"] = True
        elif atype == "code":
            result = super()._execute_action(action)
            if result and result.strip():
                units = split_chat(result[:2000], source="tool")
                if units:
                    ids = self.eng.insert_units(units, left_id=self.eng.tail)
                    self._round_tool_ids.extend(ids)
            return result
        return super()._execute_action(action)

    # ── 记忆树工具命令（@note/@recall/@span/@dismiss/@ingest）──

    def _execute_ahi_command(self, cmd_name: str, args: list) -> str | None:
        if cmd_name == "@note":
            kind = (args[0] if args else "state").strip()
            rest = " ".join(args[1:]) if len(args) > 1 else ""
            if not rest:
                return "Usage: @note state 关键词: 内容（kind ∈ state/schedule/worklog）"
            if ":" in rest:
                key, content = rest.split(":", 1)
            elif "：" in rest:
                key, content = rest.split("：", 1)
            else:
                key, content = rest[:8], rest
            key = key.strip()
            if not key:
                return "关键词不能为空"
            eng_key = self.eng.nb_set(key, kind, content.strip())
            self._last_wakeup_actions.append(
                {"type": "note", "kind": kind, "key": eng_key, "content": content.strip()})
            return f"笔记本 {kind}「{eng_key}」已更新（同关键词覆盖）"

        if cmd_name == "@recall":
            q = " ".join(args).strip()
            if len(q) < 2:
                return "查询词过短"
            hits = self.eng.retrieve(q, budget=self._retr_budget)
            expanded = self.eng.expand_leaves(hits)
            self._last_shadows = list(dict.fromkeys(
                list(self._last_shadows or []) + expanded))
            self._last_topic = q[:40]
            self._last_wakeup_actions.append(
                {"type": "recall", "query": q, "hits": len(hits), "expanded": len(expanded)})
            return f"recall「{q}」→ 命中 {len(hits)} 项，展开 {len(expanded)} 原子影子下轮生效"

        if cmd_name == "@span":
            if not args:
                return "Usage: @span 卡号"
            nid = int(re.sub(r"\D", "", args[0]) or 0)
            if nid not in self.eng.nodes or not self.eng.nodes[nid].alive:
                return f"节点 N{nid} 不存在或已删除"
            expanded = self.eng.expand_leaves([nid])
            self._last_shadows = list(dict.fromkeys(
                list(self._last_shadows or []) + expanded))
            self._last_wakeup_actions.append({"type": "span", "node": nid})
            return f"span N{nid} → 展开 {len(expanded)} 个原子原文，下轮生效"

        if cmd_name == "@dismiss":
            rest = " ".join(args).strip()
            if rest.startswith("cancel"):
                self.eng.dismiss_cancel()
                self._last_wakeup_actions.append({"type": "dismiss", "action": "cancel"})
                return "dismiss 已取消（懒标记关闭，待失效标注清除）"
            ids = [int(m) for m in re.findall(r"#?\d+", rest)] if rest else []
            if ids and all(i in self.eng.atoms for i in ids):
                dead = self.eng.dismiss_confirm(ids)
                self._last_wakeup_actions.append(
                    {"type": "dismiss", "action": f"confirm #{ids}", "dead": len(dead)})
                return f"dismiss 确认：墓碑 {len(dead)} 张旧卡（可回滚），懒标记结案"
            if rest:
                key, atoms = self.eng.dismiss_plant(rest, exclude_ids=[])
                if atoms:
                    self._last_shadows = list(dict.fromkeys(
                        list(self._last_shadows or []) + atoms))
                self._last_wakeup_actions.append(
                    {"type": "dismiss", "action": "plant", "key": key, "hits": len(atoms)})
                return (f"dismiss 挂起「{key}」：命中 {len(atoms)} 张待失效原子"
                        f"（影子已注入），持续追踪中")
            return "Usage: @dismiss 新信息描述 / @dismiss #卡号 / @dismiss cancel"

        if cmd_name == "@ingest":
            if not args:
                return "Usage: @ingest 文件路径"
            path = " ".join(args).strip()
            from sdk.sandbox import resolve_path
            resolved = resolve_path(path, self._sandbox_root)
            if resolved is None:
                return ("读取失败: 路径超出沙箱范围（只能访问工作目录内的文件）")
            try:
                with open(resolved, encoding="utf-8", errors="replace") as f:
                    text = f.read()
            except Exception as e:
                return f"读取失败: {e}"
            media = self._media_for(path)
            units = split_media(text, media, source=f"file:{path}")
            ids = self.eng.insert_units(units, left_id=self.eng.tail)
            self._last_wakeup_actions.append(
                {"type": "ingest", "path": path, "units": len(ids), "media": media})
            return f"已读入 {path}（{media}，{len(ids)} 卡进记忆树）"

        return super()._execute_ahi_command(cmd_name, args)

    @staticmethod
    def _media_for(path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        if ext in (".py", ".pyw"):
            return "code"
        if ext in (".json", ".yaml", ".yml", ".ini", ".toml"):
            return "config"
        if ext in (".md", ".markdown", ".txt"):
            return "md"
        return "chat"

    # ── 指标 ──────────────────────────────────────

    CLAIM_RE = re.compile(r"已(转储|备份|写入|执行|保存|删除|记录|落盘|归档)")

    def _record_action_claims(self):
        """声称×执行匹配（观测层，**仅记录**零行为影响）：
        本轮 ai 原子中的行动声称（"已转储/已备份/…"）与近轮 tool 原子内容比对，
        append 到 action_claims.jsonl。不渲染、不提示、不拦截——"哪有人天天被
        电脑自动提醒你说了没做"。分析期用于量化声称/执行分离率。"""
        if not self._claim_log_path:
            return
        if not self._round_ai_ids and not self._round_tool_ids:
            return
        claims = []
        for aid in self._round_ai_ids:
            a = self.eng.atoms.get(aid)
            if not a or not a.alive:
                continue
            for m in self.CLAIM_RE.finditer(a.value or ""):
                claims.append({"atom": aid, "verb": m.group(1),
                               "text": (a.value or "")[:80]})
        tool_atoms = []
        for aid in self._round_tool_ids:
            a = self.eng.atoms.get(aid)
            if a and a.alive:
                tool_atoms.append({"atom": aid, "text": (a.value or "")[:80]})
        if not claims and not tool_atoms:
            return
        pool = tool_atoms + list(self._recent_tool_atoms)
        for c in claims:
            c["matched"] = any(c["verb"] in (t["text"] or "") for t in pool)
        row = {"ts": round(time.time(), 2), "round": self._round_no,
               "claims": claims, "tool_atoms": tool_atoms}
        try:
            with open(self._claim_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception:
            pass
        for t in tool_atoms:
            self._recent_tool_atoms.append(t)

    def _record_metrics(self, ctx_chars: int, pending: int):
        if not self._metrics_f:
            return
        try:
            mem = self.eng.stat()
            row = {
                "ts": round(time.time(), 2),
                "round": self._round_no,
                "wakeup": self._wakeup_count,
                "persona": self._persona_on,
                "ctx_chars": ctx_chars,
                "pending": pending,
                "pending_left": self._pending_left,
                "distill_calls": self.eng.distill["calls"],
                "distill_seconds": round(self.eng.distill["seconds"], 2),
                "llm_calls": self._llm_calls,
                "llm_seconds": round(self._llm_seconds, 2),
                "pre_distill_used": self.eng.pre_distill_used,
                "fact_drift": self.eng.fact_drift,
                "person_source_violations": getattr(self.eng, "first_person_violations", 0),
                "style_alpha": getattr(self.eng, "style_alpha", 1.0),
                "snapshot_on": bool(self.eng.style_provider is not None),
                "snapshot_fields": len(self._parse_snapshot(self._snapshot or "")),
                "pre_distill_wasted": self.eng.pre_distill_wasted,
                "atoms_alive": mem["atoms"]["alive"],
                "atoms_total": mem["atoms"]["total"],
                "nodes_alive": mem["nodes"]["alive"],
                "tiling": mem["tiling_size"],
                "dirty": mem["dirty"],
                "placeholder": mem["placeholder"],
                "max_depth": mem["max_depth"],
                "notebook": len(self.eng.notebook),
            }
            self._metrics_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._metrics_f.flush()
        except Exception as e:
            # P1 修复（2026-08-30）：异常必须留痕——静默吞异常导致 metrics 停更
            # 多次复发（24h 实验 2 次 + 1h 实验），观测断档无法归因
            try:
                import traceback
                if self.db:
                    self.db.add_log("ERROR", self.agent_name,
                                    f"metrics 写入失败: {e}\n{traceback.format_exc()[:400]}")
            except Exception:
                pass
