#!/usr/bin/env python3
"""chat_agent.py — 聊天智能体（AHI 风格接口 + 水循环记忆树）。

外部接口形状参照 AHI（process_input / get_outputs / on_wakeup），内部 =
RealEngine + 每轮全新组装（零对话历史，记忆全在记忆树里，不依赖 LLM 对话历史）。

终端测试：python3 chat_agent.py [--provider ollama|deepseek]
  > 直接输入即可聊天；exit/quit 退出。
"""
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chattext import split_chat, split_media
from llmclient import make_client
from protocol import parse_response
from realengine import RealEngine

SYSTEM = """你是带持久记忆的聊天助手「水循环」。你的全部记忆来自下面的【记忆流】（每轮全新组装，不依赖对话历史）。

【记忆流格式】每项带 id 与从属标注：
- N12·摘要 为摘要卡（旧内容已蒸馏）；#34·原文 为原文句卡；⊂Nx 表示其父卡；缩进表示层级；
- 〔来源〕标注该句来自谁（user:xxx=用户，ai=你自己，tool=工具/文件读取）；
- "召回影子"为检索注入的旧内容；记忆流按位置排序，末尾是最近的内容；
- 记忆流之后是【笔记本·事实层】（去重发送：每个条目只保留最新一条，每轮现场生成）。

【回复流程（每轮）】
1. 直接输出回复文本（正文即回复）。
2. 需要回想旧内容时，在回复中使用 [recall]关键词[/recall]（下一轮注入相关摘要）。
3. 需要看某摘要卡覆盖的原文时用 [span N卡号]（下一轮生效）。
4. 状态记录（必须）：用户透露了需要跨会话记住的事实/偏好/承诺时，**必须先输出**
   [note kind=state]关键词: 内容[/note] 再输出回复正文。禁止只说"我会记住"而不记——
   真正的记录动作是 [note] 标签。同一话题用**相同关键词**记录（覆盖旧版本，实现去重）。
   [note kind=schedule] 用于待办/承诺；[note kind=worklog] 用于临时工作记录（固定关键词覆盖更新）。
5. 事实冲突处理（dismiss）：发现用户新说的话与记忆流中旧信息明显矛盾时（如旧信息
   "用户的电脑是mac"，新信息"用户重装了windows系统"），用
   [dismiss]新信息简述；可能的旧信息：猜测描述[/dismiss] 挂起冲突——系统会持续追踪旧
   信息并以〔待失效〕标注注入。当看到带〔待失效〕标注的旧卡且确认矛盾成立时，用
   [dismiss #卡片id] 确认失效（该旧卡将从记忆流移除，可回滚）；若判断不矛盾（如用户
   有两台电脑），用 [dismiss cancel] 关闭。仅在明显矛盾时使用，不要滥用。

【纪律】
1. 严格依据记忆流作答；记忆流中没有的信息不得编造，可说"我不记得/需要确认"。
2. 你与用户的历史对话都在记忆流里（含你自己的回复），可以引用自己之前说过的话。
3. 引用用户早先说过的内容时，可提及〔来源〕。
4. 不要输出解释、标题、序号、markdown 结构说明。"""


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


class ChatAgent:
    """带记忆树持久化的聊天智能体。

    接口（参照 AHI，非严格继承）：
      process_input(text) -> str   用户消息 → AI 回复（阻塞）
      get_outputs() -> list        待发送消息队列（终端模式 = 上轮回复）
      on_wakeup() -> str|None      自主唤醒（无新输入时的自发言，可选）
      run_repl()                   终端 REPL
    """

    def __init__(self, config_path=None, system_prompt=None, provider=None,
                 user_name="user", verbose=False):
        cfg_path = config_path or os.path.join(os.path.dirname(__file__), "config.json")
        cfg = json.load(open(cfg_path, encoding="utf-8"))
        if provider:
            cfg["provider"] = provider
        self.cfg = cfg
        self.llm = make_client(cfg)
        self.user_name = user_name
        self.verbose = verbose
        self.budget = int(cfg.get("assembly_budget", 10000))
        self.retr_budget = int(cfg.get("retrieval_budget", 6))
        self.merge_every = int(cfg.get("merge_every", 4))
        self.eng = RealEngine(
            self.llm,
            decay=float(cfg.get("decay", 8.0)),
            hp_merge_threshold=float(cfg.get("hp_merge_threshold", 30.0)),
            parent_init_hp=float(cfg.get("parent_init_hp", 70.0)),
            hot_threshold=float(cfg.get("hot_threshold", 60.0)),
            recall_boost=float(cfg.get("recall_boost", 30.0)),
            merge_cap_atoms=int(cfg.get("merge_cap_atoms", 500)),
            merge_depth_diff=int(cfg.get("merge_depth_diff", 1)),
            promote_threshold=int(cfg.get("promote_threshold", 3)),
            attention_cap=int(cfg.get("attention_cap", 60)),
            attention_out=float(cfg.get("attention_out", 45.0)),
            attention_in_decay=float(cfg.get("attention_in_decay", 0.5)),
            attention_out_decay=float(cfg.get("attention_out_decay", 3.0)),
            merge_fanout_cap=int(cfg.get("merge_fanout_cap", 12)),
            pressure_threshold=float(cfg.get("pressure_threshold", 2.0)),
            assembly_budget=self.budget,
            retrieval_budget=self.retr_budget,
        )
        self.system = system_prompt or SYSTEM
        self._outputs = []
        self.receipt = "初始化：空记忆，开始对话。"
        self.last_shadows = None
        self.last_topic = ""
        self.round_no = 0
        # 轨道B 预置：交互规则（constraint）
        self.eng.nb_set(
            "交互规则", "constraint",
            "你是持久记忆聊天助手：跨会话记住用户的事实/偏好/承诺；"
            "回答依据记忆流；记忆流中没有的不得编造。")

    # ── AHI 风格接口 ──────────────────────────────

    def process_input(self, text) -> str:
        """用户消息 → AI 回复。用户消息与 AI 回复都按句写入记忆树（source 区分）。"""
        text = (text or "").strip()
        if not text:
            return ""
        self.round_no += 1
        eng = self.eng

        # 1. 用户消息按句写入（追加链尾，热区在流末）
        units = split_chat(text, source=f"user:{self.user_name}")
        if not units:
            return ""
        new_ids = eng.insert_units(units, left_id=eng.tail)
        self._last_new_ids = new_ids
        self.receipt = f"收到你的消息（{len(units)} 句）。"

        # 2. 重复输入检测 → 命中旧卡活一轮影子（P2-8；排除刚写入的新卡）
        sims = []
        for t, _ in units:
            sims.extend(eng.find_similar(t, exclude_ids=new_ids))
        if sims:
            sims = list(dict.fromkeys(sims))
            self.last_shadows = list(dict.fromkeys((self.last_shadows or []) + sims))
            self.receipt += f" 检测到与历史高度重复的内容（{len(sims)} 项旧卡将作为影子展示）"

        # 2b. 懒标记持续追踪：planted dismiss 每轮预检索，命中原子待失效影子（下轮生效）
        pend = eng.dismiss_track()
        if pend:
            self.last_shadows = list(dict.fromkeys((self.last_shadows or []) + pend))

        # 3. 组装（轨道B → 铺陈+影子 → 回执 → 任务）
        reply = self._chat()
        return reply

    def get_outputs(self) -> list:
        out, self._outputs = self._outputs, []
        return out

    def on_wakeup(self) -> str | None:
        """自主唤醒：无新输入时可选自发言。终端模式默认不主动说话。"""
        return None

    # ── 内部 ──────────────────────────────────────

    def _chat(self) -> str:
        eng = self.eng
        nb_rows = eng.nb_render(self.last_topic or "")
        nb = render_notebook(nb_rows)
        emitted = eng.assemble(self.last_shadows, budget=self.budget)
        flow = eng.render_emission(emitted)
        task = (f"回复用户（第 {self.round_no} 轮；最近话题关键词：{self.last_topic or '无'}）。"
                f"如需回想用 [recall]/[span]。用户透露新事实/偏好/承诺用 [note kind=state] 记录。")
        # 组装顺序（聊天拍定）：卡片流 → 笔记本（去重发送，接在卡片流后）→ 回执 → 任务
        parts = [flow]
        if nb:
            parts.append("【笔记本·事实层】（去重发送：每项只保留最新，本轮现场生成）\n" + nb)
        if self.receipt:
            parts.append(f"【最近操作回执】\n{self.receipt}")
        parts.append(f"【当前任务】\n{task}")
        user_ctx = "\n\n".join(parts)

        if self.verbose:
            print(f"\n---- ctx {len(user_ctx)} 字符 ----\n{user_ctx[:1200]}...\n", flush=True)

        t0 = time.time()
        text, usage, lat = self.llm.chat(self.system, user_ctx)
        lat_ms = (time.time() - t0) * 1000

        # 4. 解析回复：正文 = 回复用户（同时写回流，source=ai）；工具块按语义执行
        blocks = parse_response(text)
        reply_parts = []
        receipts = []
        recalls = []
        for kind, payload in blocks:
            if kind in ("text", "append"):
                reply_parts.append(payload)
            elif kind == "recall":
                q = (payload or "").strip()
                if len(q) >= 2:
                    hits = eng.retrieve(q, budget=self.retr_budget)
                    expanded = eng.expand_leaves(hits)
                    self.last_shadows = list(dict.fromkeys(list(self.last_shadows or []) + expanded))
                    recalls.append(q)
                    receipts.append(f"recall「{q}」→ {len(hits)} 项，展开 {len(expanded)} 原子影子下轮生效")
                else:
                    receipts.append("recall 查询词过短，已忽略")
            elif kind == "span":
                spec, _ = payload
                ids = [int(m) for m in re.findall(r"\d+", spec or "")]
                if ids:
                    nid = ids[0]
                    atoms = eng.span_atoms(nid)
                    if atoms is None:
                        receipts.append(f"span N{nid}：节点不存在或已删除，已忽略")
                    else:
                        expanded = eng.expand_leaves([nid])
                        self.last_shadows = list(dict.fromkeys(list(self.last_shadows or []) + expanded))
                        receipts.append(f"span N{nid} → 展开 {len(expanded)} 个原子原文，下轮生效")
                else:
                    receipts.append("span 参数不完整（需要节点 id），已忽略")
            elif kind == "note":
                spec, body = payload
                body = (body or "").strip()
                kkind = (spec or "state").strip() or "state"
                if kkind.startswith("kind="):
                    kkind = kkind[5:].strip()
                cm = body.split(":", 1) if ":" in body else body.split("：", 1)
                if len(cm) == 2 and cm[0].strip():
                    key, content = cm[0].strip(), cm[1].strip()
                else:
                    key, content = body[:8], body
                key = eng.nb_set(key, kkind, content)
                receipts.append(f"笔记本 {kkind}「{key}」已更新")
            elif kind == "dismiss":
                spec, body = payload
                spec = (spec or "").strip()
                if spec.startswith("cancel"):
                    eng.dismiss_cancel()
                    receipts.append("dismiss 已取消（懒标记关闭，待失效标注清除）")
                else:
                    ids = [int(m) for m in re.findall(r"\d+", spec)]
                    if ids and all(i in eng.atoms for i in ids):
                        dead = eng.dismiss_confirm(ids)
                        receipts.append(f"dismiss 确认：墓碑 {len(dead)} 张旧卡（可回滚），懒标记结案")
                    elif (body or "").strip():
                        key, atoms = eng.dismiss_plant(body, exclude_ids=self._last_new_ids)
                        if atoms:
                            self.last_shadows = list(dict.fromkeys(list(self.last_shadows or []) + atoms))
                        receipts.append(f"dismiss 挂起「{key}」：冲突描述已记录，"
                                        f"命中 {len(atoms)} 张待失效原子（影子已注入），持续追踪中")
                    else:
                        receipts.append("dismiss 参数不完整（需要描述或 #id 或 cancel），已忽略")
            elif kind == "plan":
                self.last_topic = (payload or "").strip()[:40]
                receipts.append(f"话题：「{self.last_topic}」")

        # 5. AI 回复写回流（source=ai），保持对话连续性
        reply = "\n".join(p for p in reply_parts if p.strip()).strip()
        if reply:
            ai_units = split_chat(reply, source="ai")
            eng.insert_units(ai_units, left_id=eng.tail)

        # 6. 衰减 + 蒸馏时机
        eng.tick(1)
        if eng.attention_pressure():
            eng.merge_pass(force_out=True)
            receipts.append("窗外原子超限 → force 合并")
        elif self.round_no % self.merge_every == 0:
            eng.merge_pass()

        # 7. 回执 + 话题
        if recalls:
            self.last_topic = recalls[-1][:40]
        if receipts:
            self.receipt = "；".join(receipts[-3:])
        else:
            self.receipt = f"第 {self.round_no} 轮完成。"

        if self.verbose:
            mem = eng.stat()
            print(f"[轮 {self.round_no}] ctx {len(user_ctx)} 字符 | LLM {lat_ms:.0f}ms | "
                  f"原子 {mem['atoms']['alive']} 节点 {mem['nodes']['alive']} "
                  f"铺陈 {mem['tiling_size']} | 蒸馏 {eng.distill['calls']}", flush=True)

        self._outputs.append(reply)
        return reply

    # ── 工具入口（供外部/测试环境调用）─────────────

    def ingest(self, text, media="chat", source="tool", lang=None, fmt=None):
        """外部灌入参考资料（chat/code/config/md），按介质切分写入记忆树（追加链尾）。"""
        units = split_media(text, media, source=source, lang=lang, fmt=fmt)
        if not units:
            return []
        ids = self.eng.insert_units(units, left_id=self.eng.tail)
        self.receipt = f"已灌入 {len(ids)} 项（{media}，来源 {source}）。"
        return ids

    def file_edit(self, path, new_content, media="code", lang=None, fmt=None):
        """文件编辑（确定性路径）：树里有该文件旧内容 → 卡级 diff 精确失效被改部分
        （墓碑 + 新卡按介质切分插入，走编辑路径置脏）；树里没有 → 直接灌入新内容。
        与介质切分粒度一致：代码按 AST 卡 diff、配置按项 diff。零 LLM。"""
        import difflib
        source = f"file:{path}"
        old_ids = [aid for aid, a in self.eng.atoms.items()
                   if a.alive and (a.metadata or {}).get("source") == source]
        if not old_ids:
            return self.ingest(new_content, media=media, source=source, lang=lang, fmt=fmt)
        chain = self.eng.chain_ids()
        old_ids.sort(key=lambda aid: chain.index(aid))
        old_texts = [self.eng.atoms[aid].value for aid in old_ids]
        new_units = split_media(new_content, media, source=source, lang=lang, fmt=fmt)
        new_texts = [t for t, _ in new_units]
        sm = difflib.SequenceMatcher(None, old_texts, new_texts)
        edits = 0
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                continue
            dead = old_ids[i1:i2]
            units = new_units[j1:j2]
            if dead:
                self.eng.replace_units(dead, units)
                edits += 1
            elif units:
                anchor = old_ids[i1 - 1] if i1 > 0 else None
                self.eng.insert_units(units, left_id=anchor,
                                       right_id=old_ids[i1] if i1 < len(old_ids) else None)
                edits += 1
        self.receipt = f"文件 {path} 已编辑：{edits} 处卡级变更（旧内容精确失效，未改部分保留）。"
        return edits

    def say(self, text):
        """AI 主动陈述（如自主唤醒时）写入记忆树（source=ai，追加链尾）。"""
        units = split_chat(text, source="ai")
        return self.eng.insert_units(units, left_id=self.eng.tail) if units else []

    def note(self, key, kind="state", content=""):
        return self.eng.nb_set(key, kind, content)

    # ── 终端 REPL ─────────────────────────────────

    def run_repl(self):
        print(f"=== 水循环聊天测试 ===")
        print(f"模型: {getattr(self.llm, 'model', self.cfg.get('model', '?'))} | "
              f"预算 {self.budget} 字符 | 记忆树就绪")
        print("直接输入聊天，exit/quit 退出。\n")
        while True:
            try:
                line = input("你 > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见。")
                break
            if line.lower() in ("exit", "quit"):
                print("再见。")
                break
            if not line:
                continue
            reply = self.process_input(line)
            if reply:
                print(f"AI > {reply}")
            else:
                print("AI > （无回复）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.json"))
    ap.add_argument("--provider", default=None, choices=["ollama", "deepseek"])
    ap.add_argument("--user", default="user")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    agent = ChatAgent(config_path=args.config, provider=args.provider,
                      user_name=args.user, verbose=args.verbose)
    agent.run_repl()


if __name__ == "__main__":
    main()
