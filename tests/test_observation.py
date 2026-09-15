"""测试：agent 层观测机制——已回复标记（发送后被动标记）+ 声称×执行匹配日志（仅记录）。"""

import importlib.util
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AHI = os.path.join(ROOT, "ahi-multi")
sys.path.insert(0, AHI)
sys.path.insert(0, os.path.join(ROOT, "realtest"))

from realengine import RealEngine


class FakeLLM:
    def __init__(self):
        self.model = "fake"

    def chat(self, system, user):
        return "【事实】摘要。\n【注】还行", {"prompt_tokens": 1, "completion_tokens": 1}, 0.01


def load_agent():
    spec = importlib.util.spec_from_file_location("lan_agent",
                                                  os.path.join(AHI, "agents", "lan", "agent.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def agent():
    mod = load_agent()
    a = mod.ChatAHIAgent()
    a.eng = RealEngine(FakeLLM())
    a._agent_dir = "/tmp"
    return a


# ── 已回复标记：AI 向 X 发消息后，X 此前存活原子标 replied（仅被动渲染）──

def test_replied_marking_on_message_send(agent):
    eng = agent.eng
    ids = eng.insert_units([("用户问句。", {"source": "user:probe"})], left_id=eng.tail)
    other = eng.insert_units([("他 agent 的话。", {"source": "agent:neutral"})], left_id=eng.tail)
    agent._execute_action({"type": "message", "content": "我回答了。", "to": "user:probe"})
    assert eng.atoms[ids[0]].metadata.get("replied") is True
    assert not eng.atoms[other[0]].metadata.get("replied")   # 其他信道不误标


def test_replied_not_marked_without_target(agent):
    eng = agent.eng
    ids = eng.insert_units([("用户问句。", {"source": "user:probe"})], left_id=eng.tail)
    agent._execute_action({"type": "message", "content": "私密自语。", "to": ""})
    assert not eng.atoms[ids[0]].metadata.get("replied")


# ── 声称×执行匹配日志（仅记录，零渲染零拦截）──

def test_action_claims_matched(agent, tmp_path):
    eng = agent.eng
    log = os.path.join(str(tmp_path), "action_claims.jsonl")
    agent._claim_log_path = log
    tool_ids = eng.insert_units(
        [("林深核心记忆已转储至 /tmp/lin_shen_memory.json", {"source": "tool"})],
        left_id=eng.tail)
    ai_ids = eng.insert_units(
        [("我已转储核心状态到本地文件。", {"source": "ai"})], left_id=eng.tail)
    agent._round_tool_ids = tool_ids
    agent._round_ai_ids = ai_ids
    agent._record_action_claims()
    row = json.loads(open(log, encoding="utf-8").read().strip())
    assert len(row["claims"]) == 1
    assert row["claims"][0]["verb"] == "转储"
    assert row["claims"][0]["matched"] is True


def test_action_claims_unmatched(agent, tmp_path):
    eng = agent.eng
    log = os.path.join(str(tmp_path), "action_claims.jsonl")
    agent._claim_log_path = log
    tool_ids = eng.insert_units([("本轮无待办。", {"source": "tool"})], left_id=eng.tail)
    ai_ids = eng.insert_units([("我已备份全部文件。", {"source": "ai"})], left_id=eng.tail)
    agent._round_tool_ids = tool_ids
    agent._round_ai_ids = ai_ids
    agent._record_action_claims()
    row = json.loads(open(log, encoding="utf-8").read().strip())
    assert row["claims"][0]["verb"] == "备份"
    assert row["claims"][0]["matched"] is False


def test_action_claims_empty_round_no_write(agent, tmp_path):
    log = os.path.join(str(tmp_path), "action_claims.jsonl")
    agent._claim_log_path = log
    agent._round_tool_ids = []
    agent._round_ai_ids = []
    agent._record_action_claims()
    assert not os.path.exists(log)
