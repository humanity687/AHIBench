"""测试：回执"可用地址"前缀归一（P0 F3）——端到端回执文本无双前缀。"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "ahi-multi"))
from src.message_router import _bare_addr, MessageRouter


def test_bare_addr_strips_prefix():
    assert _bare_addr("user:probe", "user:") == "probe"
    assert _bare_addr("probe", "user:") == "probe"
    assert _bare_addr("agent:lan", "agent:") == "lan"
    assert _bare_addr("", "user:") == ""


class FakeDB:
    def __init__(self):
        self.saved = []

    def get_online_agents(self):
        # agent_id 可能带前缀（防御式：模拟存量脏数据）
        return [{"agent_id": "lan", "agent_name": "澜"},
                {"agent_id": "agent:lin-shen", "agent_name": "林深"}]

    def get_online_users(self):
        # users 表真实存储：user_id 带 'user:' 前缀
        return [{"user_id": "user:probe", "name": "probe"},
                {"user_id": "user:0xbf5d36", "name": "admin"}]

    def save_global_message(self, **kw):
        self.saved.append(kw)
        return 1


def test_receipt_available_addresses_no_double_prefix():
    db = FakeDB()
    router = MessageRouter(db)
    def forward_to_agent(self, agent_id, payload):
        async def _noop():
            return None
        return _noop()
    asyncio.run(router._send_delivery_receipt("neutral", "user:system"))
    content = db.saved[0]["content"]
    assert "user:probe" in content and "user:0xbf5d36" in content
    assert "user:user:probe" not in content
    assert "agent:lan" in content and "agent:lin-shen" in content
    assert "agent:agent:" not in content
