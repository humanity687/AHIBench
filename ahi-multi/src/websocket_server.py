import json
import time
import uuid
import logging
import threading
import asyncio
from typing import Dict, Any, Optional, Callable, List

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class WebSocketServer:
    def __init__(self):
        self._connections: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        # 外部注入的聊天消息处理器（由 main.py 设置）
        self._on_incoming_chat: Optional[Callable] = None
        # 用户上下线事件回调（由 main.py 设置，决策：系统状态事件流）
        self._on_user_online: Optional[Callable] = None
        self._on_user_offline: Optional[Callable] = None
        self._user_conn_count: Dict[str, int] = {}

    # ── 消息构造 ──

    @staticmethod
    def _make_message(channel: str, msg_type: str, payload: Any,
                      msg_id: Optional[str] = None) -> dict:
        return {
            "channel": channel,
            "type": msg_type,
            "payload": payload,
            "timestamp": time.time(),
            "msg_id": msg_id or uuid.uuid4().hex,
        }

    # ── Channel 订阅管理 ──

    def _subscribe(self, conn_id: str, channel: str):
        with self._lock:
            if conn_id in self._connections:
                conn = self._connections[conn_id]
                if channel not in conn["channels"]:
                    conn["channels"].append(channel)

    def _unsubscribe(self, conn_id: str, channel: str):
        with self._lock:
            if conn_id in self._connections:
                conn = self._connections[conn_id]
                if channel in conn["channels"]:
                    conn["channels"].remove(channel)

    # ── 推送方法 ──

    @staticmethod
    def _enrich_payload(message: dict) -> dict:
        """将消息顶层元数据合并到 payload 中，确保前端能读取 from_agent/source/to"""
        payload = message.get("payload", message)
        if isinstance(payload, dict):
            for key in ("from_agent", "source", "to"):
                if key in message and key not in payload:
                    payload[key] = message[key]
        return payload

    def push_to_channel(self, channel: str, message: dict):
        formatted = self._make_message(
            channel,
            message.get("type", "message"),
            self._enrich_payload(message),
            message.get("msg_id"),
        )
        payload = json.dumps(formatted, ensure_ascii=False)
        targets = []
        with self._lock:
            for conn in self._connections.values():
                if channel in conn.get("channels", []):
                    targets.append(conn["websocket"])
        logger.info("push_to_channel: channel=%s targets=%d msg_type=%s", channel, len(targets), message.get("type",""))
        for ws in targets:
            self._safe_send(ws, payload)

    def push_to_user(self, user_id: str, message: dict):
        # 统一 user_id 格式：确保带 "user:" 前缀（前端 auth 时发送的是 "user:xxx"）
        normalized_uid = f"user:{user_id}" if not user_id.startswith("user:") else user_id
        channel = message.get("channel", f"chat:{message.get('from_agent', '')}")
        formatted = self._make_message(
            channel,
            message.get("type", "message"),
            self._enrich_payload(message),
            message.get("msg_id"),
        )
        payload = json.dumps(formatted, ensure_ascii=False)
        targets = []
        with self._lock:
            for conn in self._connections.values():
                if conn.get("user_id") == normalized_uid:
                    targets.append(conn["websocket"])
        logger.info("push_to_user: user_id=%s normalized=%s targets=%d", user_id, normalized_uid, len(targets))
        for ws in targets:
            self._safe_send(ws, payload)

    def broadcast_to_users(self, message: dict):
        with self._lock:
            user_ids = list(set(
                c.get("user_id") for c in self._connections.values() if c.get("user_id")
            ))
        logger.info("broadcast_to_users: targets=%s msg_type=%s", user_ids, message.get("type",""))
        for uid in user_ids:
            self.push_to_user(uid, message)

    def get_connected_users(self) -> List[str]:
        with self._lock:
            return list(set(
                c.get("user_id", "") for c in self._connections.values()
                if c.get("user_id")
            ))

    # ── Agent 上下线广播 ──

    def broadcast_agent_online(self, agent_id: str):
        self.push_to_channel("system", {
            "type": "agent_online",
            "payload": {"agent_id": agent_id},
        })

    def broadcast_agent_offline(self, agent_id: str):
        self.push_to_channel("system", {
            "type": "agent_offline",
            "payload": {"agent_id": agent_id},
        })

    def broadcast_system_notification(self, level: str, content: str):
        self.push_to_channel("system", {
            "type": "system_notification",
            "payload": {"level": level, "content": content},
        })

    # ── WebSocket 连接处理 ──

    async def _handle_ws(self, websocket: WebSocket):
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        await websocket.accept()
        conn_id = uuid.uuid4().hex
        with self._lock:
            self._connections[conn_id] = {
                "websocket": websocket,
                "user_id": None,
                "channels": [],
            }

        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                msg_type = data.get("type", "")
                payload = data.get("payload", {})

                if msg_type == "auth":
                    user_id = payload.get("user_id", "")
                    if user_id:
                        with self._lock:
                            first = self._user_conn_count.get(user_id, 0) == 0
                            self._user_conn_count[user_id] = self._user_conn_count.get(user_id, 0) + 1
                            self._connections[conn_id]["user_id"] = user_id
                        if first and self._on_user_online:
                            try:
                                self._on_user_online(user_id, payload.get("name", ""))
                            except Exception:
                                pass
                        self._subscribe(conn_id, "system")
                        await self._safe_send_ws(websocket, self._make_message(
                            "system", "auth_ok", {"user_id": user_id}
                        ))

                elif msg_type == "subscribe":
                    channel = payload.get("channel", "")
                    if channel:
                        self._subscribe(conn_id, channel)
                        await self._safe_send_ws(websocket, self._make_message(
                            "system", "subscribed", {"channel": channel}
                        ))

                elif msg_type == "unsubscribe":
                    channel = payload.get("channel", "")
                    if channel:
                        self._unsubscribe(conn_id, channel)
                        await self._safe_send_ws(websocket, self._make_message(
                            "system", "unsubscribed", {"channel": channel}
                        ))

                elif msg_type == "chat_message":
                    # 前端通过 WS 发来的聊天消息，返回给调用方处理
                    target = data.get("to", "")
                    content = payload.get("content", "")
                    with self._lock:
                        user_id = self._connections[conn_id].get("user_id", "anonymous")
                    source = payload.get("source") or user_id
                    await self._handle_incoming_chat(conn_id, source, target, content)

        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            with self._lock:
                conn_info = self._connections.pop(conn_id, None)
                user_id = (conn_info or {}).get("user_id")
            # 连接断开：用户可能下线
            if user_id:
                with self._lock:
                    self._user_conn_count[user_id] = self._user_conn_count.get(user_id, 1) - 1
                    offline = self._user_conn_count[user_id] <= 0
                    if offline:
                        self._user_conn_count.pop(user_id, None)
                if offline and self._on_user_offline:
                    try:
                        self._on_user_offline(user_id)
                    except Exception:
                        pass

    async def _handle_incoming_chat(self, conn_id: str,
                                    source: str, target: str, content: str):
        """由外部注入的 chat 处理器接管（通过 self._on_incoming_chat 显式属性注入）。"""
        if self._on_incoming_chat is not None:
            await self._on_incoming_chat(source, target, content)
        else:
            logger.warning("_on_incoming_chat handler not set, chat message dropped")

    @staticmethod
    async def _safe_send_ws(websocket: WebSocket, msg: dict):
        try:
            await websocket.send_text(json.dumps(msg, ensure_ascii=False))
        except Exception:
            pass

    def _safe_send(self, websocket: WebSocket, payload: str):
        """线程安全的 WebSocket 推送。
        
        从异步上下文调用时：直接使用主事件循环。
        从非异步线程调用时：使用 _loop（主事件循环引用），回退到自建临时循环。
        """
        # 如果已有主事件循环引用，用它调度
        if self._loop is not None:
            try:
                asyncio.run_coroutine_threadsafe(
                    websocket.send_text(payload), self._loop
                )
                return
            except Exception as e:
                logger.warning("_safe_send via main loop failed: %s", e)

        # 尝试获取当前运行的事件循环
        try:
            loop = asyncio.get_running_loop()
            asyncio.run_coroutine_threadsafe(websocket.send_text(payload), loop)
            return
        except RuntimeError:
            pass

        # 最后的回退：创建一个新的事件循环并运行（仅在没有 WS 连接时触发）
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(websocket.send_text(payload))
            loop.close()
        except Exception as e:
            logger.warning("_safe_send all fallbacks failed: %s", e)
    
    @staticmethod
    def _get_loop_status(self) -> str:
        """返回当前事件循环状态（用于诊断）"""
        if self._loop is None:
            return "loop=None"
        return f"loop={id(self._loop)} running={self._loop.is_running()}"

    def get_ws_handler(self) -> Callable:
        async def ws_endpoint(websocket: WebSocket):
            await self._handle_ws(websocket)
        return ws_endpoint
