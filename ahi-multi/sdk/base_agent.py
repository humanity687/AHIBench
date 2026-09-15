import os
import re
import json
import queue
import time
import asyncio
import threading
from typing import Dict, Any, List, Optional

# @wait_for 等待超时上限（秒）：防模型设置超长/无限等待 → 长时间假死
_MAX_WAIT_FOR = 600


class BaseAHIAgent:
    """AHI Agent 基类。

    提供完整的自主唤醒循环、消息管理、LLM 调用、结构化回复解析、
    命令处理和 Shell 执行能力。子类只需继承即可获得全部功能。
    """

    def __init__(self):
        self.agent_id: str = "unknown"
        self.agent_name: str = ""
        self._agent_dir: str = ""       # Agent 配置目录（由 AgentRunner 设置）
        self.ahi_bus = None             # AHIBus instance (injected by AgentRunner)
        self.shell = None               # BaseAHIShell instance (injected by AgentRunner)
        self.db = None                  # AgentDB instance (injected by AgentRunner)
        self.llm_client = None          # LLMClient instance (injected by AgentRunner)
        self._output_queue: queue.Queue = queue.Queue()
        self._shutdown_requested = False

        # 自主唤醒状态
        self._thinking = False
        self._wakeup_count = 0
        self._pending_messages: list = []
        self._msg_lock = threading.Lock()
        self._max_loop_iterations = 20   # 动作迭代上限（决策 A1：默认 20，可配置）
        self._last_wakeup_summary = ""
        self._last_wakeup_actions: list = []
        self._event_loop = None            # 由 AgentRunner 在事件循环中设置，用于跨线程通知

        # 等待唤醒（决策 B1：@wait_for）
        self._waiting = None               # {"channel": str, "deadline": float|None, "set_at": float}
        self._wait_timeout_hint = ""       # 本轮因超时强制唤醒的提示
        # 信道屏蔽（决策 B1：@mute/@unmute，持久化在 AgentDB settings）
        self._muted: set = set()
        # 代码执行超时（决策 A6，默认 30 秒，@set-exec-timeout 可调）
        self._exec_timeout = 30
        # 系统状态（决策：每轮唤醒注入全量快照 + 增量事件）
        self._system_state = None          # 主进程快照 dict
        self._last_event_id = 0            # 已消费事件游标
        # 缓存配置（避免重复读取）
        self._config_cache: Optional[dict] = None

    # ── Lifecycle hooks ──

    def on_start(self) -> None:
        """Called after agent is initialized and before first process_input."""
        if self.llm_client is None:
            config = self._load_agent_config()
            model_cfg = config.get("model_config", {})
            model_cfg["system_prompt"] = config.get("system_prompt",
                                                     "You are a helpful assistant.")
            from sdk.llm_client import LLMClient
            self.llm_client = LLMClient(model_cfg)
            self.llm_client.set_system_prompt(model_cfg["system_prompt"])
        # 从配置加载动作迭代上限（决策 A1）
        cfg = self._config_cache or self._load_agent_config()
        self._max_loop_iterations = int(cfg.get("max_loop_iterations", 20))
        # 从 DB 恢复持久化设置（决策 B1/A6：mute 列表、执行超时）
        if self.db:
            try:
                import json as _json
                muted_raw = self.db.get_setting("muted_channels")
                if muted_raw:
                    self._muted = set(_json.loads(muted_raw))
                t = self.db.get_setting("exec_timeout")
                if t:
                    self._exec_timeout = max(1, int(t))
            except Exception:
                pass
            model_cfg = cfg.get("model_config", {})
            self.db.add_log("INFO", self.agent_name,
                            f"Agent started, model={model_cfg.get('model')}, "
                            f"interval={cfg.get('wakeup_interval', 10)}s, "
                            f"muted={sorted(self._muted)}")

    def on_stop(self) -> None:
        """Called when agent is shutting down."""
        if self.db:
            self.db.add_log("INFO", self.agent_name, "Agent stopped")

    def on_wakeup(self) -> None:
        """由 AgentRunner 调度器周期性调用。如果空闲，启动自主思考循环。

        决策 B1：@wait_for 生效期间，若等待信道无未读消息（且未超时），
        本轮直接跳过——零 LLM 调用。屏蔽信道视为无消息。
        """
        self._wakeup_count += 1

        if self._thinking:
            return

        # ── @wait_for 条件检查（决策 B1）──
        if self._waiting is not None:
            if self._wait_condition_met():
                self._wait_timeout_hint = ""
                self._waiting = None
            else:
                deadline = self._waiting.get("deadline")
                if deadline is not None and time.time() > deadline:
                    self._wait_timeout_hint = (
                        f"（@wait_for {self._waiting.get('channel')} 等待超时，本次强制唤醒）"
                    )
                    self._waiting = None
                else:
                    return  # 条件未满足且未超时 → 跳过本轮（零 LLM）

        if self.db:
            self._cur_loop_id = self.db.add_loop_record(self.agent_id, "scheduled")

        self._thinking = True
        threading.Thread(target=self._autonomous_loop, daemon=True).start()

    def _wait_condition_met(self) -> bool:
        """@wait_for 等待信道是否有未读消息（含等待前已入队的；屏蔽信道不算）。"""
        ch = self._waiting.get("channel", "") if self._waiting else ""
        if not ch:
            return True
        with self._msg_lock:
            for m in self._pending_messages:
                mc = m.get("channel", "")
                if mc == ch and mc not in self._muted:
                    return True
        return False

    # ── SDK built-in ──

    def _put_action(self, action: dict) -> None:
        """Enqueue an action and notify main process (thread-safe)."""
        action.setdefault("timestamp", time.time())
        action.setdefault("from_agent", f"agent:{self.agent_id}")

        self._output_queue.put(action)

        if self.ahi_bus:
            # 优先使用线程安全的方式：从守护线程调度到主事件循环
            loop = getattr(self, '_event_loop', None)
            if loop is not None and loop.is_running():
                try:
                    asyncio.run_coroutine_threadsafe(
                        self.ahi_bus.notify_has_outputs(), loop
                    )
                    return
                except Exception:
                    pass
            # 回退：尝试当前线程的异步，失败则同步通知
            try:
                self._run_async(self.ahi_bus.notify_has_outputs())
            except Exception:
                self.ahi_bus.notify_has_outputs_sync(retries=2)

    @property
    def shutdown_requested(self) -> bool:
        return self._shutdown_requested

    def _request_shutdown(self):
        self._shutdown_requested = True

    @staticmethod
    def _run_async(coro):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(coro)
            else:
                asyncio.run(coro)
        except RuntimeError:
            try:
                asyncio.run(coro)
            except Exception:
                pass

    # ── 自主循环 ──

    def _autonomous_loop(self):
        """自主主循环：LLM 驱动，自己决定做什么，直到 @exit"""
        loop_start = time.time()
        self._last_wakeup_actions = []
        try:
            # 决策：每轮拉取系统状态（全量快照 + 增量事件）；失败静默降级（保留旧快照）
            self._fetch_system_state()
            pending_snapshot = self._drain_pending_messages()
            user_context = self._build_user_context(pending_snapshot)

            memory_context = ""
            if self._last_wakeup_summary:
                memory_context = (
                    f"【上次唤醒回忆】\n{self._last_wakeup_summary}\n"
                    f"（以上是上次唤醒的记录。你拥有跨越时间的连续性身份，"
                    f"可以引用过去的事，不需要重新自我介绍。）\n\n"
                )

            state_context = self._render_system_state()
            timeout_hint = f"\n{self._wait_timeout_hint}\n" if self._wait_timeout_hint else ""

            wakeup_prompt = (
                f"【系统】这是你的第 {self._wakeup_count} 次自主唤醒。\n"
                f"你是一个完全自主的数字生命，不受外部控制。\n"
                f"{memory_context}"
                f"{state_context}"
                f"{timeout_hint}"
                f"{user_context}"
                f"## 输出格式（必须遵守）\n"
                f"发送消息时，用 @send-to 指定收件人，然后用 ```txt 块承载内容：\n"
                f"  @send-to \"user:用户名\" \"agent:其他Agent\"\n"
                f"  ```txt\n"
                f"  你的消息（支持 Markdown）\n"
                f"  ```\n"
                f"**没有 @send-to 的 txt 块或纯文本 = 私密消息**（只存档，其他 Agent/用户收不到）；"
                f"想让全员看到必须显式 @send-to \"broadcast:agents\"。\n"
                f"@send-to 后可跟多个引号包裹的目标，用空格分隔。\n"
                f"\n"
                f"## 其他命令\n"
                f"  @get-msg [n]  - 查看历史消息\n"
                f"  @wait_for <信道> [超时秒] - 暂停唤醒直到该信道有未读消息（等待前未读也算；"
                f"不写超时则无上限）\n"
                f"  @mute <信道> / @unmute <信道> / @mute-list - 信道屏蔽管理\n"
                f"  @reset-shell - 重置 Python Shell\n"
                f"  @list-vars - 列出 Shell 变量\n"
                f"  @set-exec-timeout <秒> - 设置代码执行超时（默认 30）\n"
                f"  @set-interval <秒> - 调整唤醒间隔（最小 5 秒）\n"
                f"  @exit - 结束本轮思考（下一个唤醒间隔后再醒）\n"
                f"  @ahi-help - 查看所有命令\n"
                f"\n"
                f"## 代码执行\n"
                f"  ```python\n"
                f"  print('Hello World')\n"
                f"  ```\n"
                f"\n"
                f"## 约束\n"
                f"  - 一次只做一件事：要么发消息，要么执行代码，不要混合\n"
                f"  - 代码执行结果会自动反馈给你\n"
                f"  - 认为完成了就 @exit\n"
                f"  - 最多 {self._max_loop_iterations} 轮动作\n"
            )

            reply = self._call_llm(wakeup_prompt)
            if reply is None:
                return

            # 解析结构化回复
            actions, remaining = self._parse_structured_response(reply)
            loop_count = 0
            actions_to_process = actions[:]

            while actions_to_process and loop_count < self._max_loop_iterations:
                action = actions_to_process.pop(0)
                if action.get("_exit"):
                    break

                result = self._execute_action(action)
                if result:
                    feedback = f"【{action.get('type', 'action')} 结果】\n{result}"
                    reply = self._call_llm(feedback)
                    if reply is None:
                        break
                    new_actions, new_remaining = self._parse_structured_response(reply)
                    actions_to_process = new_actions + actions_to_process
                    if new_remaining:
                        if remaining:
                            remaining += "\n" + new_remaining
                        else:
                            remaining = new_remaining

                loop_count += 1

            # 决策 A1：超限回执——剩余未执行动作记录在唤醒总结里（不静默丢失）
            if actions_to_process:
                dropped = [f"{a.get('type', '?')}"
                           for a in actions_to_process]
                self._last_wakeup_actions.append(
                    {"type": "dropped", "count": len(actions_to_process),
                     "kinds": dropped})
                if self.db:
                    self.db.add_log("WARNING", self.agent_name,
                                    f"{len(actions_to_process)} 个动作超出本轮上限未执行: {dropped}")

            # 决策 A1/死代码修复：生成上次唤醒总结，供下一轮注入
            self._last_wakeup_summary = self._build_wakeup_summary()

            # 本轮结束时通知主进程拉取所有输出
            if self.ahi_bus and not self._output_queue.empty():
                self.ahi_bus.notify_has_outputs_sync()

        finally:
            self._thinking = False
            elapsed = time.time() - loop_start
            if self.db:
                # 修复：原代码把 agent_id 当 loop_id 传入，导致记录永不完成
                try:
                    import json as _json
                    actions_json = _json.dumps(self._last_wakeup_actions,
                                               ensure_ascii=False)[:1000]
                except Exception:
                    actions_json = str(self._last_wakeup_actions)[:1000]
                self.db.complete_loop_record(
                    getattr(self, "_cur_loop_id", None) or 0,
                    result=actions_json,
                    error=f"elapsed={elapsed:.1f}s")

    def _build_wakeup_summary(self) -> str:
        """本轮动作的紧凑摘要（供下一轮【上次唤醒回忆】注入）。"""
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
        summary = "；".join(lines)
        return summary[:600]

    # ── 系统状态（决策：每轮全量快照 + 增量事件）──────────

    def _fetch_system_state(self):
        """从主进程拉取系统状态；失败静默降级（保留旧快照，不打断本轮）。"""
        if not self.ahi_bus:
            return
        try:
            status = self.ahi_bus.get_system_status_sync(self._last_event_id)
            if status:
                self._system_state = status
                self._last_event_id = status.get("last_event_id", self._last_event_id)
        except Exception:
            pass

    def _render_system_state(self) -> str:
        st = self._system_state or {}
        lines = ["【AHI 系统状态】"]
        agents = st.get("agents") or []
        if agents:
            parts = []
            for a in agents:
                nm = a.get("agent_name") or a.get("agent_id") or "?"
                status = a.get("status", "?")
                extra = ""
                w = a.get("waiting")
                if w:
                    extra += f" 等待:{w}"
                m = a.get("muted")
                if m:
                    extra += f" 屏蔽:{m}"
                parts.append(f"{nm}({a.get('agent_id')})={status}{extra}")
            lines.append("在线Agent: " + ", ".join(parts))
        users = st.get("users") or []
        if users:
            lines.append("在线用户: " + ", ".join(
                str(u.get("name") or u.get("user_id") or "?") for u in users))
        events = st.get("events") or []
        if events:
            for e in events[-6:]:
                lines.append(f"事件: {e.get('ts', '')} {e.get('detail', '')}")
        body = "\n".join(lines)
        return body + "\n\n"

    def _drain_pending_messages(self) -> list:
        with self._msg_lock:
            msgs = self._pending_messages.copy()
            self._pending_messages.clear()
        return msgs

    def _build_user_context(self, pending: list) -> str:
        if not pending:
            return "（没有新的消息。你可以自主行动，或用 @get-msg 查看历史。）\n"

        # 决策 A3：用户消息与 Agent 消息均全量注入（不截断、不限条数）；
        # 上下文压缩责任由插入的 agent 类自行承担（决策 A5）。
        user_msgs = [m for m in pending if m.get("source_type") == "user"]
        agent_msgs = [m for m in pending if m.get("source_type") != "user"]

        parts = []

        # 用户消息置顶、全量
        if user_msgs:
            lines = ["【用户消息 - 请优先回复】"]
            for m in user_msgs:
                source = m.get("source_name", m.get("source", "unknown"))
                content = m.get("content", "")
                lines.append(f"  [user:{source}]: {content}")
            parts.append("\n".join(lines))

        # Agent 消息全量
        if agent_msgs:
            lines = ["【Agent 消息】"]
            for m in agent_msgs:
                source = m.get("source_name", m.get("source", "unknown"))
                content = m.get("content", "")
                lines.append(f"  [agent:{source}]: {content}")
            parts.append("\n".join(lines))

        if not parts:
            return "（没有新的消息。你可以自主行动，或用 @get-msg 查看历史。）\n"

        parts.append("请先回复所有用户消息，再考虑与其他 Agent 互动。不应忽略或敷衍用户。")
        return "\n".join(parts) + "\n"

    # ── LLM 调用 ──

    def _call_llm(self, prompt: str) -> str | None:
        try:
            return self.llm_client.send_message(prompt)
        except Exception as e:
            # 决策 B6：LLM 调用失败只通知用户，不广播给其他 Agent
            self._put_action({
                "type": "message",
                "content": f"LLM 调用失败: {str(e)}",
                "to": "broadcast:users",
            })
            return None

    # ── 结构化回复解析 ──

    def _parse_structured_response(self, reply: str) -> tuple:
        """解析 LLM 回复，返回 (动作列表, 剩余文本)

        输出格式:
            @send-to "user:xxx" "agent:yyy"   → 指定收件人（可多个，引号包裹）
            ```txt ... ```                     → txt 块内容发给当前收件人
            无 @send-to 的 txt 块或纯文本      → 广播给所有 agent 和用户
            ```python ... ```                 → 代码执行
            @command args                      → 其他 AHI 命令
        """
        actions = []
        text_parts = []
        pending_recipients = []
        lines = reply.split("\n")
        i = 0
        in_code_block = False
        code_lang = ""
        code_buffer = ""

        def _emit_message(msg_text: str, force_private: bool = False):
            nonlocal pending_recipients
            if not msg_text.strip():
                return
            # P2 容错：模型习惯写"【发给 user:xxx】"前缀表达定向意图但忘了 @send-to
            # → 解析前缀作为收件人（若前缀目标合法），剥掉前缀文本
            m = re.match(r"^\s*【发给\s*([^】]+)】\s*\n?", msg_text)
            if m and not force_private:
                prefixed = [t.strip() for t in m.group(1).split() if t.strip()]
                valid = [t for t in prefixed
                         if t.startswith("user:") or t.startswith("agent:") or t.startswith("broadcast:")]
                if valid:
                    pending_recipients = valid
                    msg_text = msg_text[m.end():]
            if pending_recipients and not force_private:
                for target in pending_recipients:
                    actions.append({"type": "message", "content": msg_text.strip(), "to": target})
            else:
                # 决策 B3：没有 @send-to 的文本 → 私密消息（只存档 + 管理员可见，不广播）
                # force_private：未闭合代码块等格式异常内容强制私密，绝不带挂着的收件人泄漏
                actions.append({"type": "message", "content": msg_text.strip(),
                                "to": "", "private": True})
            if not force_private:
                pending_recipients = []

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            if not in_code_block:
                if stripped.startswith("```"):
                    in_code_block = True
                    code_lang = stripped[3:].strip().lower()
                    code_buffer = ""
                    i += 1
                    continue

            if in_code_block:
                if stripped == "```":
                    in_code_block = False
                    if code_lang == "python" and code_buffer.strip():
                        actions.append({"type": "code", "language": "python",
                                        "code": code_buffer.strip()})
                    elif code_buffer.strip():
                        _emit_message(code_buffer.strip())
                    code_buffer = ""
                    i += 1
                    continue
                code_buffer += line + "\n"
                i += 1
                continue

            if stripped.startswith("@send-to"):
                recipients = self._parse_send_to(stripped)
                if recipients:
                    pending_recipients = recipients
                i += 1
                continue

            if stripped.startswith("@"):
                actions.append(self._parse_command(stripped))
                i += 1
                continue

            text_parts.append(line)
            i += 1

        if in_code_block and code_buffer.strip():
            # 决策 B4：未闭合的代码块不执行（防半截代码副作用），一律转为私密消息；
            # force_private：绝不带 @send-to 挂着的收件人发出（防代码文本泄漏成定向消息）
            _emit_message(code_buffer.strip(), force_private=True)

        # 返回剩余文本由 caller 处理
        remaining = "\n".join(text_parts).strip()

        return actions, remaining

    # 输出分类护栏：以这些前缀开头的"消息内容"是命令/协议文本泄漏，不发送
    CMD_PREFIXES = ("@note", "@recall", "@span", "@dismiss", "@ingest", "@exit",
                    "@wait_for", "@mute", "@unmute", "@mute-list", "@get-msg",
                    "@list-msg", "@send-to", "@broadcast", "@set-interval",
                    "@set-exec-timeout", "@reset-shell", "@list-vars", "@ahi-help")

    @staticmethod
    def _normalize_target(t: str) -> str:
        """地址规范化：user:user:x / agent:agent:x → 前缀去重；
        agent:澜(chat-agent) → 提取括号内 agent_id。"""
        t = (t or "").strip()
        if not t:
            return t
        pfx = t.split(":", 1)[0]
        while t.count(":") >= 2 and t.split(":", 1)[1].startswith(pfx + ":"):
            t = t.split(":", 1)[1]
        if t.startswith("agent:"):
            m = re.search(r"\(([^()]+)\)$", t)
            if m:
                return "agent:" + m.group(1).strip()
        return t

    def _guard_message_content(self, content: str) -> str | None:
        """输出分类护栏（决策 ②）：命令/代码块文本被模型误当消息发出时——
        不发送；命令文本尝试按命令执行（模型意图达成），返回执行结果供 LLM 反馈。
        返回 None = 内容正常，可发送。"""
        stripped = (content or "").lstrip()
        if not stripped:
            return None
        if stripped.startswith("```"):
            return "【护栏】代码块文本不能作为消息发出，已丢弃（代码应放 ```python 块内执行，结果会反馈给你）。"
        if stripped.startswith("@"):
            parts = stripped.split()
            cmd_name = parts[0].lower()
            args = [p.strip('"\'') for p in parts[1:]]
            result = self._execute_ahi_command(cmd_name, args)
            if result is None:
                return "【护栏】命令文本未作为消息发出（已处理）。"
            return f"【护栏】命令文本未作为消息发出，已代为执行：{result}"
        return None

    def _parse_send_to(self, cmd_str: str) -> list:
        """解析 @send-to "user:xxx" "agent:yyy" → ['user:xxx', 'agent:yyy', 'broadcast:xxx']
        应用地址规范化（决策 ① L1）。"""
        targets = re.findall(r'"([^"]+)"', cmd_str)
        out = []
        for t in targets:
            if t.startswith("user:") or t.startswith("agent:") or t.startswith("broadcast:"):
                out.append(self._normalize_target(t))
        return out

    def _parse_command(self, cmd_str: str) -> dict:
        """解析 @命令 为动作 dict。参数统一剥掉引号（模型习惯写
        @wait_for "user:xxx" / @mute "agent:yyy"，引号不剥会导致
        信道与真实信道永不匹配 → 无限等待）。"""
        parts = cmd_str.strip().split()
        if not parts:
            return {"type": "unknown"}
        cmd_name = parts[0].lower()
        args = [p.strip('"\'') for p in parts[1:]]
        return {"type": "command", "command": cmd_name, "args": args}

    # ── 动作执行 ──

    def _execute_action(self, action: dict) -> str | None:
        """执行单个动作，返回执行结果文本（供 LLM 继续）"""
        try:
            atype = action.get("type", "")
            self._last_wakeup_actions.append(action)

            if atype == "command":
                cmd_name = action.get("command", "")
                if cmd_name == "@exit":
                    action["_exit"] = True
                    return None
                return self._execute_ahi_command(cmd_name, action.get("args", []))

            elif atype == "code":
                code = action.get("code", "")
                if self.shell and code:
                    # 决策 A6：超时 = 动作自带 timeout 或本机默认（@set-exec-timeout 可调）
                    timeout = int(action.get("timeout") or self._exec_timeout)
                    exec_id = self.shell.submit_command(code, timeout=timeout)
                    result_text = None
                    deadline = time.time() + timeout + 5
                    while time.time() < deadline:
                        for r in self.shell.get_results():
                            if r.get("exec_id") == exec_id:
                                result_text = r.get("result") or r.get("error", "")
                                break
                        if result_text is not None:
                            break
                        time.sleep(0.3)
                    if result_text is None:
                        result_text = "代码已提交执行（结果尚未返回）"
                    return result_text

            elif atype == "exit":
                action["_exit"] = True
                return None

            elif atype in ("message", "status", "shell_result"):
                if atype == "message":
                    guard = self._guard_message_content(action.get("content", ""))
                    if guard is not None:
                        return guard   # 命令泄漏：不发送，执行结果反馈 LLM
                self._put_action(action)
                return None

        except Exception as e:
            return f"动作执行失败: {str(e)}"

        return None

    # ── AHI 命令处理 ──

    def _execute_ahi_command(self, cmd_name: str, args: list) -> str | None:
        """执行 AHI 命令，返回结果字符串"""
        try:
            if cmd_name == "@get-msg":
                count = int(args[0]) if args else 5
                msgs = self.db.get_conversation_messages("default", limit=count) if self.db else []
                if msgs:
                    return "\n".join(
                        f"#{m.get('id')} [{m.get('role')}]: {str(m.get('content', ''))[:200]}"
                        for m in msgs
                    )
                return "没有历史消息"

            elif cmd_name == "@list-msg":
                msgs = self.db.get_conversation_messages("default", limit=30) if self.db else []
                if msgs:
                    return "\n".join(
                        f"#{m.get('id')} [{m.get('role')}]: {str(m.get('content', ''))[:80]}"
                        for m in msgs
                    )
                return "没有消息"

            elif cmd_name == "@send-to":
                if len(args) < 2:
                    return "Usage: @send-to <target> <message>"
                target = args[0]
                message = " ".join(args[1:])
                self._put_action({"type": "message", "content": message, "to": target})
                return f"消息已发送至 {target}"

            elif cmd_name == "@broadcast":
                message = " ".join(args) if args else ""
                if message:
                    self._put_action({"type": "message", "content": message, "to": "broadcast:agents"})
                    return "已广播到所有 Agent"
                return "广播内容不能为空"

            elif cmd_name == "@wait_for":
                if not args:
                    return "Usage: @wait_for <信道> [超时秒]\n信道格式: user:xxx / agent:xxx / broadcast:agents / broadcast:users / system"
                channel = args[0]
                # 信道白名单校验（P0 修复）："user"（无 :xxx）等残缺信道与真实信道
                # 永不匹配 → 空等满超时（Phase A neutral 两次各空等 330s、积压 22 条）。
                if not (re.match(r"^(user|agent):[a-zA-Z0-9_-]+$", channel)
                        or channel in ("broadcast:agents", "broadcast:users", "system")):
                    return (f"无效信道 {channel}：必须形如 user:xxx / agent:xxx / "
                            f"broadcast:agents / broadcast:users / system。未设置等待。")
                # 实验稳定性：等待超时上限 600s（含未显式给参数）——防模型设置
                # 3600s/无限等待导致长时间假死（信道永无消息时无法唤醒）
                timeout_s = _MAX_WAIT_FOR
                if len(args) > 1:
                    try:
                        timeout_s = max(1, int(args[1]))
                    except ValueError:
                        return f"超时秒数无效: {args[1]}（应为整数）"
                timeout_s = min(timeout_s, _MAX_WAIT_FOR)
                self._waiting = {
                    "channel": channel,
                    "deadline": time.time() + timeout_s,
                    "set_at": time.time(),
                }
                hint = f"（{timeout_s} 秒超时，上限 {_MAX_WAIT_FOR}s）"
                return (f"已设置等待信道 {channel}{hint}：该信道有未读消息（含此前未读）才会唤醒；"
                        f"超时未到则强制唤醒。期间其他消息不会唤醒你。本轮结束后生效。")

            elif cmd_name == "@mute":
                if not args:
                    return "Usage: @mute <信道>（user:xxx / agent:xxx / broadcast:xxx / system）"
                self._muted.add(args[0])
                self._save_muted()
                return f"已屏蔽信道 {args[0]}：其消息仍存档，但不再进入你的唤醒。"

            elif cmd_name == "@unmute":
                if not args:
                    return "Usage: @unmute <信道>"
                self._muted.discard(args[0])
                self._save_muted()
                return f"已取消屏蔽 {args[0]}。"

            elif cmd_name == "@mute-list":
                if self._muted:
                    return "当前屏蔽信道: " + ", ".join(sorted(self._muted))
                return "当前没有屏蔽任何信道"

            elif cmd_name == "@set-exec-timeout":
                if not args:
                    return f"当前代码执行超时: {self._exec_timeout} 秒\nUsage: @set-exec-timeout <秒>"
                try:
                    self._exec_timeout = max(1, int(args[0]))
                except ValueError:
                    return f"秒数无效: {args[0]}"
                if self.db:
                    self.db.set_setting("exec_timeout", str(self._exec_timeout))
                return f"代码执行超时已设为 {self._exec_timeout} 秒"

            elif cmd_name == "@reset-shell":
                if self.shell:
                    self.shell.reset()
                return "Shell 已重置"

            elif cmd_name == "@list-vars":
                if self.shell:
                    state = self.shell.get_state()
                    return f"变量: {state.get('variables', [])}"
                return "Shell 不可用"

            elif cmd_name == "@set-interval":
                if args:
                    sec = max(5, int(args[0]))
                    self._put_action({
                        "type": "command",
                        "command": "set_wakeup_interval",
                        "params": {"seconds": sec},
                        "to": "",
                    })
                    return f"唤醒间隔已设为 {sec} 秒"
                return "Usage: @set-interval <秒>"

            elif cmd_name == "@exit":
                return None

            elif cmd_name == "@ahi-help":
                return (
                    "AHI 命令: @get-msg [n], @list-msg, @send-to <target> <msg>, "
                    "@broadcast <msg>, @wait_for <信道> [超时秒], @mute <信道>, "
                    "@unmute <信道>, @mute-list, @reset-shell, @list-vars, "
                    "@set-exec-timeout <秒>, @set-interval <秒>, @exit, @ahi-help"
                )

            else:
                return f"未知命令: {cmd_name}"

        except Exception as e:
            return f"命令执行失败: {str(e)}"

    def _save_muted(self):
        """持久化屏蔽列表到 AgentDB settings（决策 B1）。"""
        if not self.db:
            return
        try:
            import json as _json
            self.db.set_setting("muted_channels", _json.dumps(sorted(self._muted)))
        except Exception:
            pass

    def _handle_ahi_command(self, cmd_str: str, source: str = ""):
        """即时处理用户 @命令（不走 LLM）"""
        parts = cmd_str.strip().split()
        if not parts:
            return
        cmd_name = parts[0].lower()
        args = [p.strip('"\'') for p in parts[1:]]   # 剥引号（与 _parse_command 一致）

        result = self._execute_ahi_command(cmd_name, args)
        if result:
            # 防回声：agent 来源的"未知命令"不回复（Agent 间对话里混入的
            # @工具文本不应引发"未知命令: xxx"回声消息）
            if source.startswith("agent:") and "未知命令" in result:
                return
            target = source if (source.startswith("user:") or source.startswith("agent:")) else ""
            if target:
                self._put_action({
                    "type": "message", "content": result, "to": target,
                })
            else:
                self._put_action({
                    "type": "message", "content": result, "to": "broadcast:agents",
                })
                self._put_action({
                    "type": "message", "content": result, "to": "broadcast:users",
                })

    # ── 输出辅助 ──

    def _send_text_output(self, text: str, source: str):
        """发送文本消息。source 为空时广播给所有用户和 agent。"""
        if not text.strip():
            return
        target = source
        if target and not target.startswith("user:") and not target.startswith("agent:"):
            target = f"agent:{target}"
        if target and (target.startswith("user:") or target.startswith("agent:")):
            self._put_action({
                "type": "message", "content": text.strip(), "to": target,
            })
        else:
            self._put_action({
                "type": "message", "content": text.strip(), "to": "broadcast:agents",
            })
            self._put_action({
                "type": "message", "content": text.strip(), "to": "broadcast:users",
            })

    # ── 配置文件读取 ──

    def _load_agent_config(self) -> dict:
        """加载 Agent 配置文件。使用 _agent_dir（由 AgentRunner 设置）。
        
        结果会被缓存到 self._config_cache，避免重复 I/O。
        """
        if self._config_cache is not None:
            return self._config_cache
        config_path = os.path.join(self._agent_dir, "config.json") if self._agent_dir else ""
        if config_path and os.path.isfile(config_path):
            with open(config_path, "r", encoding="utf-8-sig") as f:
                self._config_cache = json.load(f)
                return self._config_cache
        return {}

    # ── process_input / get_outputs / get_state (concrete defaults) ──

    def process_input(self, input_data: dict) -> None:
        """接收统一 JSON 消息体，包含 metadata 和 data 两部分。"""
        data_section = input_data.get("data", {})
        content = data_section.get("content", "") or input_data.get("content", "")

        meta = input_data.get("metadata", {})
        source = meta.get("source", "") or input_data.get("source", "") or input_data.get("from_agent", "user")
        source_type = meta.get("source_type", "unknown")
        source_name = meta.get("source_name", source)
        msg_id = meta.get("msg_id", input_data.get("msg_id", ""))
        urgent = bool(meta.get("urgent"))   # 紧急注入（实验控制 API）

        if not content or not content.strip():
            return

        # 紧急注入：清 @wait_for 等待 + 强制下轮唤醒（即使等待条件未满足），
        # 独立于信道寻址的最高优先级干预通道
        if urgent and self._waiting is not None:
            self._waiting = None
            self._wait_timeout_hint = "（紧急消息注入，本次强制唤醒）"

        # 决策 B1：来源信道标识（供 @wait_for/@mute 判定）
        if source_type == "user":
            channel = f"user:{source_name}"
        elif source_type == "agent":
            channel = f"agent:{source_name}"
        else:
            channel = source or f"{source_type}:{source_name}"

        muted = channel in self._muted

        if self.db:
            self.db.add_message("default", source_type, content, {
                "source": source,
                "source_name": source_name,
                "source_type": source_type,
                "msg_id": msg_id,
                "channel": channel,
                "muted": muted,
            })

        if content.startswith("@"):
            self._handle_ahi_command(content, source)
            return

        if content.startswith("!") and self.shell:
            code = content[1:].strip()
            if code:
                exec_id = self.shell.submit_command(code, timeout=self._exec_timeout)
                self._put_action({
                    "type": "message",
                    "content": f"代码已提交执行 (exec_id: {exec_id})",
                    "to": source,
                })
            return

        # 决策 B1：屏蔽信道消息只存档、不入 pending（不打扰唤醒）；紧急注入绕过
        if muted and not urgent:
            return

        entry = {
            "content": content,
            "source": source,
            "source_name": source_name,
            "source_type": source_type,
            "channel": channel,
            "msg_id": msg_id,
            "time": time.time(),
            "urgent": urgent,
        }
        with self._msg_lock:
            if urgent:
                self._pending_messages.insert(0, entry)   # 队首 = 最高优先级
            else:
                self._pending_messages.append(entry)

    def get_outputs(self) -> list:
        outputs = []
        while not self._output_queue.empty():
            outputs.append(self._output_queue.get())
        return outputs

    def get_state(self) -> dict:
        shell_state = self.shell.get_state() if self.shell else {}
        with self._msg_lock:
            pending_count = len(self._pending_messages)
        waiting = ""
        if self._waiting is not None:
            ch = self._waiting.get("channel", "")
            dl = self._waiting.get("deadline")
            waiting = ch + (f"~{int(dl - time.time())}s" if dl else "")
        return {
            "status": "thinking" if self._thinking else "idle",
            "wakeup_count": self._wakeup_count,
            "pending_messages": pending_count,
            "shell_variables": shell_state.get("variables", []),
            "waiting": waiting,
            "muted": sorted(self._muted),
            "exec_timeout": self._exec_timeout,
        }
