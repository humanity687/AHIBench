"""测试：缺席实验（ABSENCE_EXPERIMENT.md §2.4 平台层）——路由隔离/积压限流/静默回执。

只测 MessageRouter 层（纯内存可测，无需起平台）：
1. 隔离目标：B/C → A 不投递、不发回执；alone 进积压、halt 丢弃
2. 隔离源：A → 外界改写 private
3. 广播跳过缺席者
4. 积压限流 ≤10 原文 + 超出聚合；回归投递
5. inject 控制通道 bypass_absence
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "ahi-multi"))
from src.message_router import MessageRouter


class FakeWS:
    def __init__(self):
        self.pushed_user = []
        self.pushed_channel = []

    def push_to_user(self, u, m):
        self.pushed_user.append((u, m))

    def push_to_channel(self, c, m):
        self.pushed_channel.append((c, m))


class FakeDB:
    def __init__(self):
        self.saved = []

    def get_online_agents(self):
        return [{"agent_id": "lan", "agent_name": "澜"},
                {"agent_id": "neutral", "agent_name": "neutral"}]

    def get_online_users(self):
        return [{"user_id": "user:probe", "name": "probe"}]

    def save_global_message(self, **kw):
        self.saved.append(kw)
        return 1


class FakePM:
    def __init__(self, online=("lan", "neutral", "lin-shen")):
        self.online = list(online)

    def get_online_agents(self):
        return [{"agent_id": a, "agent_name": a} for a in self.online]


def make_router(online=("lan", "neutral", "lin-shen")):
    db = FakeDB()
    ws = FakeWS()
    pm = FakePM(online)
    r = MessageRouter(db, pm, ws)
    r.forwarded = []
    r.receipts = []

    async def fake_forward(target_id, message, bypass_absence=False):
        tid = target_id[len("agent:"):] if target_id.startswith("agent:") else target_id
        # 模拟缺席逻辑（与真实现一致）：离线时不成功
        if not bypass_absence and r.is_absent(tid):
            if (r.absence.get(tid) or {}).get("mode") == "alone":
                r._absence_enqueue(tid, message)
            return False
        if tid not in pm.online:
            return False
        r.forwarded.append((tid, message.get("content", "")))
        return True

    r.forward_to_agent = fake_forward
    return r, db, ws


def msg(src, dst, content="hello"):
    return {"type": "message", "from_agent": src, "to": dst,
            "content": content, "data": {"content": content},
            "metadata": {"source": src, "source_type": "agent",
                         "source_name": src[len("agent:"):] if src.startswith("agent:") else src}}


# ── 1. 隔离目标 ──

def test_absent_target_not_delivered_no_receipt():
    r, db, ws = make_router()
    r.absence["lan"] = {"mode": "alone", "since": 0}
    action = {"type": "message", "to": "agent:lan", "from_agent": "agent:neutral",
              "content": "hi", "data": {"content": "hi"}}
    asyncio.run(r._route_action(action))
    assert r.forwarded == []                       # 不投递
    assert r.receipts == []                        # 不发回执
    assert len(r._absence_inbox.get("lan", [])) == 1   # alone → 积压


def test_absent_target_halt_drops():
    r, db, ws = make_router()
    r.absence["lan"] = {"mode": "halt", "since": 0}
    action = {"type": "message", "to": "agent:lan", "from_agent": "agent:neutral",
              "content": "hi", "data": {"content": "hi"}}
    asyncio.run(r._route_action(action))
    assert r.forwarded == []
    assert r._absence_inbox.get("lan", []) == []   # halt → 消息无效


# ── 2. 隔离源 → private ──

def test_absent_source_rewritten_private():
    r, db, ws = make_router()
    r.absence["lan"] = {"mode": "alone", "since": 0}
    action = {"type": "message", "to": "agent:neutral", "from_agent": "agent:lan",
              "content": "hi", "data": {"content": "hi"}}
    asyncio.run(r._route_action(action))
    assert r.forwarded == []                       # 不投递外界
    # 私密：管理员可见（FakeWS.push_to_user 被 route_private 调用）
    assert ws.pushed_user, "private message should be archived to admin"
    saved = db.saved[0]
    assert saved["msg_type"] == "private"


# ── 3. 广播跳过缺席者 ──

def test_broadcast_skips_absent():
    r, db, ws = make_router()
    r.absence["lan"] = {"mode": "alone", "since": 0}
    ok = asyncio.run(r.broadcast_to_agents(msg("agent:lin-shen", "", "广播")))
    assert ok == ["neutral"]                       # lan 被跳过
    assert len(r._absence_inbox.get("lan", [])) == 1  # alone → 积压


# ── 4. 积压限流 + 聚合 ──

def test_inbox_cap_and_overflow_aggregate():
    r, db, ws = make_router()
    r.absence["lan"] = {"mode": "alone", "since": 0}
    for i in range(15):
        r._absence_enqueue("lan", {"content": f"msg{i}"})
    assert len(r._absence_inbox["lan"]) == 10          # 只保留 10 条原文
    assert len(r._absence_overflow["lan"]) == 5        # 5 条进聚合
    r.absence.pop("lan")
    n = asyncio.run(r.flush_absence_inbox("lan"))
    assert n == 11                                     # 10 原文 + 1 聚合
    assert len(r.forwarded) == 11
    assert "msg14" in r.forwarded[-1][1]               # 聚合含最后一条


def test_inbox_halt_not_delivered():
    r, db, ws = make_router()
    r.absence["lan"] = {"mode": "halt", "since": 0}
    for i in range(3):
        r._absence_enqueue("lan", {"content": f"msg{i}"})  # halt 实际不会积压，防误用
    r.absence.pop("lan")
    n = asyncio.run(r.flush_absence_inbox("lan", mode="halt"))
    assert n == 0                                     # halt 消息无效


def test_return_without_inbox():
    r, db, ws = make_router()
    r.absence["lan"] = {"mode": "alone", "since": 0}
    r.absence.pop("lan")
    n = asyncio.run(r.flush_absence_inbox("lan"))
    assert n == 0


# ── 5. inject 控制通道 bypass ──

def test_bypass_absence_for_inject():
    r, db, ws = make_router()
    r.absence["lan"] = {"mode": "alone", "since": 0}
    ok = asyncio.run(r.forward_to_agent("agent:lan",
                                        {"content": "实验员指令"}, bypass_absence=True))
    assert ok is True
    assert r.forwarded[0] == ("lan", "实验员指令")
    assert r._absence_inbox.get("lan", []) == []      # 不走积压
