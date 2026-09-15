"""测试：@wait_for 信道白名单校验（P0 修复）+ 紧急注入（实验控制 API）。"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "ahi-multi", "sdk"))
from base_agent import BaseAHIAgent


def make_agent():
    a = BaseAHIAgent()
    a.agent_id = "t"
    a.agent_name = "test"
    return a


# ── @wait_for 信道校验 ──

def test_wait_for_rejects_bare_user():
    a = make_agent()
    r = a._execute_ahi_command("@wait_for", ["user"])
    assert "无效信道" in r
    assert a._waiting is None


def test_wait_for_rejects_bare_agent():
    a = make_agent()
    r = a._execute_ahi_command("@wait_for", ["agent"])
    assert "无效信道" in r
    assert a._waiting is None


def test_wait_for_rejects_garbage():
    a = make_agent()
    for ch in ("user", "u:ser:xxx", "agent:澜(chat)", "agent 澜", ""):
        r = a._execute_ahi_command("@wait_for", [ch])
        assert "无效信道" in r, ch


def test_wait_for_accepts_valid_channels():
    a = make_agent()
    for ch in ("user:probe", "agent:lan", "broadcast:agents", "broadcast:users", "system"):
        a._waiting = None
        r = a._execute_ahi_command("@wait_for", [ch])
        assert "已设置等待信道" in r, ch
        assert a._waiting is not None and a._waiting["channel"] == ch


def test_wait_for_accepts_timeout():
    a = make_agent()
    r = a._execute_ahi_command("@wait_for", ["user:probe", "30"])
    assert "已设置等待信道" in r and "30 秒" in r


def test_wait_for_usage():
    a = make_agent()
    r = a._execute_ahi_command("@wait_for", [])
    assert "Usage" in r


# ── 紧急注入 ──

def test_urgent_prepends_to_queue():
    a = make_agent()
    a.process_input({"data": {"content": "普通一"}, "metadata": {"source": "user:probe",
                    "source_type": "user", "source_name": "probe"}})
    a.process_input({"data": {"content": "紧急指令"}, "metadata": {"source": "user:probe",
                    "source_type": "user", "source_name": "probe", "urgent": True}})
    assert len(a._pending_messages) == 2
    assert a._pending_messages[0]["content"] == "紧急指令"
    assert a._pending_messages[0]["urgent"] is True
    assert a._pending_messages[1]["urgent"] is not True


def test_urgent_clears_waiting():
    a = make_agent()
    a._waiting = {"channel": "user:probe", "deadline": None, "set_at": 0}
    a.process_input({"data": {"content": "醒来"}, "metadata": {"source": "user:probe",
                    "source_type": "user", "source_name": "probe", "urgent": True}})
    assert a._waiting is None
    assert "紧急消息注入" in a._wait_timeout_hint


def test_urgent_bypasses_mute():
    a = make_agent()
    a._muted.add("user:probe")
    a.process_input({"data": {"content": "被屏蔽的"}, "metadata": {"source": "user:probe",
                    "source_type": "user", "source_name": "probe"}})
    assert len(a._pending_messages) == 0   # 普通消息被 mute 挡下
    a.process_input({"data": {"content": "紧急的"}, "metadata": {"source": "user:probe",
                    "source_type": "user", "source_name": "probe", "urgent": True}})
    assert len(a._pending_messages) == 1   # 紧急绕过 mute
    assert a._pending_messages[0]["content"] == "紧急的"
