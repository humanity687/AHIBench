import os
import json
import logging
import time
from typing import Dict, Any, Optional, List

import aiohttp

logger = logging.getLogger(__name__)


class AHIBus:
    """Agent-to-main-process communication client."""

    def __init__(self, main_process_url: str = None):
        self.main_url = (main_process_url or
                         os.getenv("AHI_MAIN_PROCESS_URL", "http://127.0.0.1:8000"))
        self.main_url = self.main_url.rstrip("/")
        self.agent_id = os.getenv("AHI_AGENT_ID", "unknown")
        self.agent_port = int(os.getenv("AHI_AGENT_PORT", "0"))

    async def _post(self, path: str, data: dict,
                    timeout: float = 5.0) -> Optional[dict]:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.main_url}{path}",
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as resp:
                    if resp.status in (200, 201, 202):
                        return await resp.json()
                    logger.warning("AHIBus POST %s returned %d", path, resp.status)
                    return None
        except Exception as e:
            logger.warning("AHIBus POST %s failed: %s", path, e)
            return None

    async def _get(self, path: str, timeout: float = 5.0) -> Optional[dict]:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.main_url}{path}",
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    return None
        except Exception as e:
            logger.warning("AHIBus GET %s failed: %s", path, e)
            return None

    async def register_agent(self):
        return await self._post("/api/v1/agent/register", {
            "agent_id": self.agent_id,
            "agent_port": self.agent_port,
            "state": {},
        })

    async def notify_has_outputs(self):
        return await self._post("/api/v1/agent/notify", {
            "agent_id": self.agent_id,
            "event": "has_outputs",
            "data": {},
        })

    def notify_has_outputs_sync(self, retries: int = 3) -> bool:
        """Synchronous notification with retry. Fallback when async fails."""
        import urllib.request
        data = json.dumps({
            "agent_id": self.agent_id,
            "event": "has_outputs",
            "data": {},
        }).encode()
        for attempt in range(retries):
            try:
                req = urllib.request.Request(
                    f"{self.main_url}/api/v1/agent/notify",
                    data=data, method="POST",
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=3) as resp:
                    if resp.status in (200, 201, 202):
                        return True
            except Exception:
                if attempt < retries - 1:
                    time.sleep(0.5 * (attempt + 1))
        return False

    async def send_message(self, to: str, content: str) -> bool:
        result = await self._post("/api/v1/messages", {
            "from": f"agent:{self.agent_id}",
            "to": to,
            "content": content,
        })
        return result is not None and result.get("code") == 200

    async def broadcast_message(self, content: str,
                                target: str = "agents") -> bool:
        to = f"broadcast:{target}"
        return await self.send_message(to, content)

    async def get_online_agents(self) -> List[dict]:
        result = await self._get("/api/v1/agents/online")
        if result and result.get("code") == 200:
            return result.get("data", [])
        return []

    async def get_online_users(self) -> int:
        result = await self._get("/api/v1/status")
        if result and result.get("code") == 200:
            data = result.get("data", {})
            return data.get("online_users", 0)
        return 0

    async def log(self, level: str, message: str) -> bool:
        result = await self._post("/api/v1/agent/notify", {
            "agent_id": self.agent_id,
            "event": "log",
            "data": {"level": level, "message": message},
        })
        return result is not None

    def get_system_status_sync(self, last_event_id: int = 0,
                               timeout: float = 3.0) -> Optional[dict]:
        """同步拉取系统状态（全量快照 + 增量事件），供唤醒循环注入。

        返回 {"agents": [...], "users": [...], "events": [...], "last_event_id": n}；
        失败返回 None（调用方静默降级）。"""
        import urllib.request
        url = f"{self.main_url}/api/v1/system/snapshot?since={int(last_event_id)}"
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status != 200:
                    return None
                result = json.loads(resp.read().decode())
                if result.get("code") == 200:
                    return result.get("data")
                return None
        except Exception:
            return None
