"""test_dismiss.py — DISMISS（懒标记延迟裁决）+ file_edit（卡级 diff 精确失效）。"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "realtest"))

from realengine import RealEngine


class FakeLLM:
    model = "fake"

    def chat(self, system, user):
        return "摘要占位", {"prompt_tokens": 0, "completion_tokens": 0}, 0.0


def make_eng():
    eng = RealEngine(FakeLLM())
    eng.insert_units([("我有一台mac电脑，用起来很顺手。", {"source": "user:u"}),
                      ("我用它写代码。", {"source": "user:u"})], left_id=eng.tail)
    eng.tick(1)  # 模拟真实场景：旧卡在上一轮写入（created_round < 当前 round）
    return eng


def test_dismiss_plant_and_track():
    eng = make_eng()
    key, atoms = eng.dismiss_plant("用户说重装了windows系统；旧信息可能是：用户的电脑是mac")
    assert key and key.startswith("dismiss_")
    # 懒标记存轨道B schedule（planted）
    entry = eng.notebook[key]
    assert entry["kind"] == "schedule" and entry["status"] == "planted"
    # 立即检索命中（"mac"/"电脑"关键词）→ 待失效标注
    assert any(eng.atoms[a].metadata.get("pending_dismiss") for a in atoms)
    # 持续追踪
    tracked = eng.dismiss_track()
    assert tracked


def test_dismiss_confirm_tombstones_and_pays_back():
    eng = make_eng()
    key, atoms = eng.dismiss_plant("重装windows；旧信息可能是mac电脑")
    dead = eng.dismiss_confirm(atoms)
    assert dead
    for aid in dead:
        assert not eng.atoms[aid].alive
    # 懒标记结案
    assert eng.notebook[key]["status"] == "payback"
    # 组装不再出现被墓碑的旧句
    emitted = eng.assemble()
    values = [v for _, _, v in emitted]
    assert "我有一台mac电脑，用起来很顺手。" not in values
    # 可回滚：逆向一次编辑
    eng.rollback()
    assert eng.atoms[dead[0]].alive


def test_dismiss_cancel_clears_marks():
    eng = make_eng()
    key, atoms = eng.dismiss_plant("冲突描述")
    eng.dismiss_cancel()
    assert eng.notebook[key]["status"] == "closed"
    assert not any((a.metadata or {}).get("pending_dismiss") for a in eng.atoms.values())


def test_dismiss_confirm_without_plant_is_noop():
    eng = make_eng()
    assert eng.dismiss_confirm([99999]) == []


def test_file_edit_partial_diff():
    eng = RealEngine(FakeLLM())
    code1 = '''def greet(name):
    """问候函数。"""
    return f"hello {name}"

def add(a, b):
    return a + b

VALUE = 1
'''
    code2 = '''def greet(name):
    """问候函数。"""
    return f"hi {name}"

def add(a, b, c):
    return a + b + c

VALUE = 1
'''
    # 改为真实切分路径（与 file_edit 一致）
    from chattext import split_media
    units = split_media(code1, "code", source="file:test.py")
    # 重建引擎，用介质切分灌入
    eng2 = RealEngine(FakeLLM())
    eng2.insert_units(units, left_id=eng2.tail)
    # file_edit 由 chat_agent 提供；完整验证卡级精确失效
    from chat_agent import ChatAgent
    agent = ChatAgent.__new__(ChatAgent)
    agent.eng = eng2
    agent.llm = FakeLLM()
    edits = agent.file_edit("test.py", code2, media="code")
    assert edits >= 1
    alive_texts = [a.value for a in agent.eng.atoms.values() if a.alive]
    joined = "\n".join(alive_texts)
    assert "hi {name}" in joined        # 新内容生效
    assert "hello {name}" not in joined  # 被改部分失效
    assert "def add(a, b, c):" in joined
    assert "VALUE = 1" in joined         # 未改部分保留


def test_file_edit_no_old_content_ingests():
    from chat_agent import ChatAgent
    agent = ChatAgent.__new__(ChatAgent)
    agent.eng = RealEngine(FakeLLM())
    agent.llm = FakeLLM()
    result = agent.file_edit("new.py", "def f():\n    return 1\n", media="code")
    # 无旧内容 → 走 ingest（返回新卡 id 列表）
    assert result and all(isinstance(x, int) for x in result)
    assert any(a.alive and "def f" in a.value for a in agent.eng.atoms.values())
