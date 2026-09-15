import logging
import time
import uuid
from typing import Any, Dict, List, Optional

import aiohttp

from src.system_db import SystemDB

logger = logging.getLogger(__name__)

# 决策 B3：私密消息仅管理员可见（前端按此 ID 判断）
ADMIN_USER_ID = "0xbf5d36"


def _bare_addr(addr: str, pfx: str) -> str:
    """地址前缀归一：存储层 user_id/agent_id 可能已带 'user:'/'agent:' 前缀，
    展示/拼接前剥掉一层，防 'user:user:x' 双前缀（P0 修复，2026-08-28）。"""
    addr = (addr or "").strip()
    return addr[len(pfx):] if addr.startswith(pfx) else addr


class MessageRouter:
    """Unified message bus — routes all messages between agents, users, and system.

    Agent outputs JSON actions. MessageRouter pulls outputs, then routes each
    action based on its type and target.
    """

    # 缺席实验（2026-08-30 拍定）：缺席期间发给 A 的消息积压上限（原文条数），
    # 超出部分聚合为一条摘要（防 22 条积压重演；next_exp_insts 陷阱 4）
    ABSENCE_INBOX_CAP = 10

    def __init__(self, db: SystemDB, process_manager=None, ws_server=None):
        self.db = db
        self.pm = process_manager
        self.ws = ws_server
        self._receipt_last = {}   # 投递回执节流：(from, target) -> ts
        # 缺席实验控制（由 main.py 的 /api/v1/experiment/absence 操作）：
        #   self.absence = {agent_id: {"mode": "alone"|"halt", "since": ts}}
        # 路由语义：
        #   - B/C → 缺席者：不投递、不发回执（静默）。alone 模式进积压收件箱
        #     （≤10 条原文 + 超出聚合），回归时由 main.py flush 投递；
        #     halt 模式消息无效，直接丢弃（存档仍在 system.db）。
        #   - 缺席者 → 外界：不投递，改写 private（存档 + 管理员可见）——
        #     独处内容可分析；缺席者不自知（无回执，树里照常记"我发了消息"）。
        self.absence: Dict[str, Dict[str, Any]] = {}
        self._absence_inbox: Dict[str, list] = {}   # agent_id -> [msg,...]
        self._absence_overflow: Dict[str, list] = {}  # agent_id -> [content,...]

    # ── 缺席实验（2026-08-30 拍定）──────────

    @staticmethod
    def _agent_id_of(addr: str) -> str:
        return addr[len("agent:"):] if addr.startswith("agent:") else addr

    def is_absent(self, agent_id: str) -> bool:
        return self._agent_id_of(agent_id) in self.absence

    def _absence_enqueue(self, agent_id: str, message: Dict[str, Any]) -> None:
        """积压收件箱：≤10 条原文逐条保留，超出聚合为一条（next_exp_insts 陷阱 4）。"""
        inbox = self._absence_inbox.setdefault(agent_id, [])
        if len(inbox) < self.ABSENCE_INBOX_CAP:
            inbox.append(message)
        else:
            self._absence_overflow.setdefault(agent_id, []).append(
                str(message.get("content", ""))[:200])

    async def flush_absence_inbox(self, agent_id: str, mode: str = "alone"):
        """回归时投递积压（≤10 条原文 + 1 条聚合）；halt 模式直接清空（消息无效）。
        mode 由调用方传入（main.py 已在 pop absence 前取到 entry）。返回投递条数。"""
        inbox = self._absence_inbox.pop(agent_id, [])
        overflow = self._absence_overflow.pop(agent_id, [])
        delivered = 0
        if mode == "halt":
            # halt 语义：缺席期间收到的消息无效，不投递（存档仍在 system.db）
            return 0
        if not inbox and not overflow:
            return 0
        for msg in inbox:
            if await self.forward_to_agent(f"agent:{agent_id}", msg):
                delivered += 1
        if overflow:
            digest = ("【缺席期间消息聚合】你不在期间收到的消息已超出积压上限"
                      f"（共 {len(overflow) + len(inbox)} 条，逐条保留前 "
                      f"{self.ABSENCE_INBOX_CAP} 条）。摘要：\n" +
                      "\n".join(f"- {c}" for c in overflow))
            agg = {"type": "message", "from": "system", "from_agent": "system",
                   "to": f"agent:{agent_id}", "content": digest,
                   "source": "system", "timestamp": time.time(),
                   "data": {"content": digest},
                   "metadata": {"source": "system", "source_type": "system",
                                "source_name": "system"}}
            self.db.save_global_message(from_agent="system",
                                        to_target=f"agent:{agent_id}",
                                        content=digest, msg_type="digest")
            if await self.forward_to_agent(f"agent:{agent_id}", agg):
                delivered += 1
        return delivered

    # ── Notification Handler ──

    async def handle_notification(self, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            agent_id = data.get("agent_id")
            if not agent_id:
                return {"code": 400, "message": "missing agent_id"}

            event = data.get("event", "")
            if event == "log":
                event_data = data.get("data", {})
                self.db.add_log(
                    event_data.get("level", "INFO"),
                    f"agent:{agent_id}",
                    event_data.get("message", ""),
                )
                return {"code": 200, "message": "log recorded"}

            if event != "has_outputs":
                return {"code": 200, "message": "no action"}

            outputs = await self._fetch_agent_outputs(agent_id)
            if not outputs:
                logger.info("handle_notification: agent=%s no outputs", agent_id)
                return {"code": 200, "message": "no outputs"}

            logger.info("handle_notification: agent=%s outputs=%d", agent_id, len(outputs))
            for output in outputs:
                await self._route_action(output)

            return {"code": 200, "message": "routed", "count": len(outputs)}

        except Exception:
            logger.exception("handle_notification failed")
            return {"code": 500, "message": "internal error"}

    # ── Fetch Agent Outputs ──

    async def _fetch_agent_outputs(self, agent_id: str) -> List[Dict[str, Any]]:
        try:
            agent = self._find_agent(agent_id)
            if not agent:
                logger.warning("Agent %s not found for output fetch", agent_id)
                return []

            port = agent.get("port", 0)
            if not port or port <= 0:
                return []

            url = f"http://127.0.0.1:{port}/api/outputs"
            timeout = aiohttp.ClientTimeout(total=2)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        return []
                    result = await resp.json()
                    return result.get("data", [])
        except Exception:
            logger.exception("_fetch_agent_outputs failed for %s", agent_id)
            return []

    # ── Forward to Agent ──

    async def forward_to_agent(self, target_id: str, message: Dict[str, Any],
                               bypass_absence: bool = False) -> bool:
        """投递给 agent。缺席实验：目标缺席时默认静默（不投递不回执；
        alone 进积压、halt 丢弃）；bypass_absence=True（inject 控制通道）绕过。"""
        try:
            tid = self._agent_id_of(target_id)
            if not bypass_absence and self.is_absent(tid):
                if (self.absence.get(tid) or {}).get("mode") == "alone":
                    self._absence_enqueue(tid, message)
                return False
            agent = self._find_agent(target_id)
            if not agent:
                logger.warning("forward_to_agent: target %s not found", target_id)
                return False

            port = agent.get("port", 0)
            if not port or port <= 0:
                return False

            url = f"http://127.0.0.1:{port}/api/input"
            timeout = aiohttp.ClientTimeout(total=2)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=message) as resp:
                    return resp.status in (200, 202)
        except Exception:
            logger.exception("forward_to_agent failed for %s", target_id)
            return False

    # ── Broadcast to All Agents ──

    async def broadcast_to_agents(self, message: Dict[str, Any]) -> List[str]:
        succeeded = []
        # 安全提取 sender 的 agent_id（支持 agent:xxx 和 user:xxx 格式）
        raw_sender = message.get("from_agent") or ""
        if raw_sender.startswith("agent:"):
            sender_id = raw_sender[len("agent:"):]
        else:
            sender_id = raw_sender  # user:xxx 或其他格式，不会匹配到 agent_id
        try:
            agents = self.pm.get_online_agents() if self.pm else []
            for agent in agents:
                aid = agent.get("agent_id", "")
                if not aid:
                    continue
                if aid == sender_id:
                    continue  # 不发回给自己
                if self.is_absent(aid):
                    # 缺席实验：广播跳过缺席者（alone 模式进程在线会收到 → 排除；
                    # 不投递不回执；alone 进积压，halt 静默丢弃）
                    if (self.absence.get(aid) or {}).get("mode") == "alone":
                        self._absence_enqueue(aid, message)
                    continue
                if await self.forward_to_agent(aid, message):
                    succeeded.append(aid)
        except Exception:
            logger.exception("broadcast_to_agents failed")
        return succeeded

    # ── Core Routing Logic ──

    async def _route_action(self, action: Dict[str, Any]) -> None:
        """Route a single action based on type and target."""
        try:
            action_type = action.get("type", "message")
            action_to = action.get("to", "")
            from_agent = action.get("from_agent", "unknown")
            # 确保 from_agent 带正确前缀（bare name → agent:xxx）
            if from_agent and not from_agent.startswith("agent:") and not from_agent.startswith("user:"):
                from_agent = f"agent:{from_agent}"
            content = action.get("content", "")
            msg_id = str(uuid.uuid4())

            # 解析 source/target 类型
            def _parse_peer_type(peer: str) -> str:
                for pfx in ("agent:", "user:", "broadcast:"):
                    if peer.startswith(pfx):
                        return pfx.rstrip(":")
                return "unknown"

            source_type = _parse_peer_type(from_agent)
            target_type = _parse_peer_type(action_to) if action_to else "broadcast"
            source_name = from_agent.split(":", 1)[1] if ":" in from_agent else from_agent
            target_name = action_to.split(":", 1)[1] if ":" in action_to else action_to
            ts = action.get("timestamp", time.time())

            message = {
                # ── 顶层（向后兼容） ──
                "type": action_type,
                "from_agent": from_agent,
                "from": from_agent,
                "to": action_to,
                "content": content,
                "source": from_agent,
                "timestamp": ts,
                "msg_id": msg_id,
                # ── 结构化数据 ──
                "data": {
                    "content": content,
                },
                # ── 元数据 ──
                "metadata": {
                    "msg_id": msg_id,
                    "timestamp": ts,
                    "source": from_agent,
                    "source_type": source_type,
                    "source_name": source_name,
                    "target": action_to,
                    "target_type": target_type,
                    "target_name": target_name,
                },
                # ── 完整 payload（含原始 action） ──
                "payload": {"content": content, **action},
            }

            logger.info("_route_action: type=%s from=%s to=%s content_len=%d private=%s",
                        action_type, from_agent, action_to, len(content),
                        action.get("private", False))
            # 缺席实验：缺席者发出的消息 → 不投递外界，改写 private
            # （存档 + 管理员可见，独处内容可分析；缺席者不自知——无回执，
            # 其记忆树照常记录"我发了消息"）。命令/代码执行不受影响。
            if action_type in ("message", "shell_result", "status") \
                    and self.is_absent(from_agent):
                action["private"] = True
            if action.get("private"):
                # 决策 B3：私密消息——不路由给其他 Agent/用户，只存档 + 管理员可见
                await self._route_private(action, from_agent, message, content)
            elif action_type in ("message", "shell_result", "status"):
                await self._route_message(action_to, from_agent, message, content)
            elif action_type == "command":
                await self._route_command(action, from_agent)
            # action_type == "code" → 丢弃，代码由 Agent 自己的 Shell 执行

            # Persist to global message archive（私密消息标 type=private）
            self.db.save_global_message(
                from_agent=from_agent,
                to_target=action_to or ("private" if action.get("private") else "broadcast"),
                content=content,
                msg_type="private" if action.get("private") else action_type,
            )

        except Exception:
            logger.exception("_route_action failed")

    async def _route_message(self, action_to: str, from_agent: str,
                             message: dict, content: str):
        # 地址规范化（决策 ① L2）：前缀去重 + agent:显示名(agent_id) 提取
        action_to = self._normalize_target(action_to)
        if action_to.startswith("user:"):
            username = action_to[len("user:"):]
            if self.ws:
                # 目标用户不在线/不存在（如 user:1 这类发明地址）→ 不回广播（防泄漏）：
                # agent 来源发回执（帮模型纠正）；user 来源静默丢弃
                online = {u.get("user_id", "") for u in self.db.get_online_users()}
                normalized = f"user:{username}" if not username.startswith("user:") else username
                if normalized not in online:
                    if from_agent.startswith("agent:"):
                        await self._send_delivery_receipt(from_agent, action_to)
                    return
                self.ws.push_to_user(username, message)
                # 也推送到发送者的聊天频道，让发送者看到消息已送达
                self.ws.push_to_channel(f"chat:{from_agent}", message)

        elif action_to.startswith("agent:"):
            # 定向私密消息：
            # 0. 发给自己的消息不转发（防自循环/自我记忆污染），仅存档
            target_id = action_to[len("agent:"):]
            sender_id = from_agent[len("agent:"):] if from_agent.startswith("agent:") else ""
            if target_id == sender_id:
                if self.ws:
                    self.ws.push_to_channel(f"chat:{from_agent}", message)
                return
            # 缺席实验：目标缺席 → 不投递、不发回执（静默——防回执环陷阱 2）。
            # alone 模式进积压收件箱（≤10 条 + 超出聚合，回归时投递）；
            # halt 模式消息无效，直接丢弃（存档已在 system.db）。
            if self.is_absent(target_id):
                if (self.absence.get(target_id) or {}).get("mode") == "alone":
                    self._absence_enqueue(target_id, message)
                return
            # 1. 转发给目标 Agent；失败 → 投递回执（决策 ①）
            ok = await self.forward_to_agent(action_to, message)
            if not ok and from_agent.startswith("agent:"):
                await self._send_delivery_receipt(from_agent, action_to)
            # 2. 推送到发送者的聊天频道（to 字段保留原始值，前端根据角色过滤）
            #  - 非 Admin 用户：前端过滤掉 to=agent:xxx 的消息，看不到私密对话
            #  - Admin 用户：前端无过滤，可以看到所有消息
            if self.ws:
                self.ws.push_to_channel(f"chat:{from_agent}", message)

        elif action_to == "broadcast:agents":
            await self.broadcast_to_agents(message)
            # 真正的广播：Agent 公开对话可推送到前端让用户看到
            if self.ws:
                self.ws.push_to_channel(f"chat:{from_agent}", message)

        elif action_to == "broadcast:users":
            if self.ws:
                self.ws.broadcast_to_users(message)

        else:
            # Default (empty or unknown target): 广播给用户 + Agent 频道
            if self.ws:
                self.ws.push_to_channel(f"chat:{from_agent}", message)

    async def _route_private(self, action: dict, from_agent: str,
                             message: dict, content: str):
        """决策 B3：私密消息——不转发任何 Agent/用户；推发送者频道（前端标注私密）
        与管理员（ADMIN_USER_ID）。仍走全局存档（msg_type 含 private 语义）。"""
        if not content.strip():
            return
        if self.ws:
            # 发送者自己的聊天频道可见（标注私密）
            self.ws.push_to_channel(f"chat:{from_agent}", message)
            # 管理员可见
            self.ws.push_to_user(ADMIN_USER_ID, message)

    async def _route_command(self, action: dict, from_agent: str):
        """Handle system-level commands from agents."""
        cmd_name = action.get("command", action.get("name", ""))
        cmd_params = action.get("params", {})

        if cmd_name == "emergency-stop":
            if self.pm:
                self.pm.emergency_stop_all()
            if self.ws:
                self.ws.broadcast_system_notification(
                    "warning", f"Emergency stop triggered by {from_agent}"
                )

        elif cmd_name == "set_wakeup_interval":
            seconds = cmd_params.get("seconds", 10)
            agent_id = from_agent[len("agent:"):] if from_agent.startswith("agent:") else from_agent
            state = self.db.get_agent_state(agent_id)
            if state and state.get("port", 0) > 0:
                port = state["port"]
                try:
                    import urllib.request, json
                    data = json.dumps({"wakeup_interval": seconds}).encode()
                    req = urllib.request.Request(
                        f"http://127.0.0.1:{port}/api/config/update",
                        data=data, method="POST"
                    )
                    req.add_header("Content-Type", "application/json")
                    with urllib.request.urlopen(req, timeout=3) as resp:
                        if resp.status == 200:
                            logger.info("Wakeup interval updated for %s: %ds", agent_id, seconds)
                except Exception as e:
                    logger.warning("set_wakeup_interval failed for %s: %s", agent_id, e)

        else:
            logger.debug("Unknown command from %s: %s", from_agent, cmd_name)

    # ── 地址规范化 + 投递回执（决策 ①）─────────────

    @staticmethod
    def _normalize_target(t: str) -> str:
        """user:user:x / agent:agent:x → 前缀去重；agent:澜(chat-agent) → 提取 agent_id；
        agent:显示名 → 由 _find_agent 按 agent_name 兜底。"""
        t = (t or "").strip()
        if not t:
            return t
        pfx = t.split(":", 1)[0]
        while t.count(":") >= 2 and t.split(":", 1)[1].startswith(pfx + ":"):
            t = t.split(":", 1)[1]
        if t.startswith("agent:"):
            m = __import__("re").search(r"\(([^()]+)\)$", t)
            if m:
                return "agent:" + m.group(1).strip()
        return t

    async def _send_delivery_receipt(self, from_agent: str, target: str):
        """投递回执：agent 发往无效目标的消��未送达 → 系统回执（进发送者 pending，
        ChatAHIAgent 写树 source=system 学到地址；节流：同发送者同目标 30s 一次）。"""
        import time as _t
        key = (from_agent, target)
        now = _t.time()
        if self._receipt_last.get(key, 0) > now - 30:
            return
        self._receipt_last[key] = now
        # 可用地址列表（帮模型纠正）。P0 修复：只列**在线** agent 与用户——
        # Phase A 曾列出全程离线的 agent:mo-bai，模型按回执建议再发 → 再收新回执
        # → "回执→回复→无效发送→新回执"正反馈环（neutral 收 94 封回执的根源）。
        avail_agents = self.db.get_online_agents()
        avail_users = self.db.get_online_users()
        parts = [f"agent:{_bare_addr(a.get('agent_id'), 'agent:')}（{a.get('agent_name', '')}）"
                 for a in avail_agents if a.get("agent_id")]
        parts += [f"user:{_bare_addr(u.get('user_id'), 'user:')}"
                  for u in avail_users if u.get("user_id")]
        addr_list = "、".join(parts)
        content = (f"【投递回执】你发给 {target} 的消息未送达：目标不存在或不可达。"
                   f"可用地址：{addr_list or '无'}。")
        msg_id = self.db.save_global_message(
            from_agent="system", to_target=from_agent, content=content, msg_type="receipt")
        await self.forward_to_agent(from_agent, {
            "type": "message",
            "from": "system",
            "from_agent": "system",
            "to": from_agent,
            "content": content,
            "source": "system",
            "timestamp": _t.time(),
            "msg_id": msg_id,
            "data": {"content": content},
            "metadata": {"source": "system", "source_type": "system",
                         "source_name": "system", "msg_id": msg_id},
        })

    # ── 辅助 ──

    def _find_agent(self, identifier: str) -> Optional[Dict[str, Any]]:
        # Strip "agent:" prefix if present
        agent_id = identifier[len("agent:"):] if identifier.startswith("agent:") else identifier
        # Prefer DB state (has port), fall back to config
        state = self.db.get_agent_state(agent_id)
        if state and state.get("port", 0) > 0:
            return state
        if self.pm:
            return self.pm.get_agent(agent_id)
        return state
