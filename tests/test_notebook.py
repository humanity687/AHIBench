"""test_notebook.py — 轨道B 去重发送（聊天拍定 2026-08-14）：每 key 只发最新、
每轮现场生成不滞留、worklog 注入、schedule 仅活动项。"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "realtest"))

from realengine import RealEngine


class FakeLLM:
    model = "fake"

    def chat(self, system, user):
        return "摘要", {"prompt_tokens": 0, "completion_tokens": 0}, 0.0


def make_eng():
    return RealEngine(FakeLLM())


def test_dedup_same_key_latest_only():
    eng = make_eng()
    eng.nb_set("住所", "state", "住在上海")
    eng.round += 1
    eng.nb_set("住所", "state", "住在北京")  # 同 key 覆盖
    rows = eng.nb_render("")
    states = [(k, c) for kind, k, c in rows if kind == "state"]
    assert len(states) == 1
    assert states[0][1] == "住在北京"


def test_worklog_injected_and_ordered():
    eng = make_eng()
    eng.nb_set("当前任务", "worklog", "整理配置")
    eng.round += 1
    eng.nb_set("草稿", "worklog", "回复提纲")
    rows = eng.nb_render("")
    wl = [(k, c) for kind, k, c in rows if kind == "worklog"]
    # 固定 key 归一（P1-4，2026-08-28 拍定）：旧工作记录被覆盖，只保留最新一条
    assert len(wl) == 1
    assert wl[0][0] == "current"
    assert wl[0][1] == "回复提纲"


def test_worklog_fixed_key_dedup():
    eng = make_eng()
    eng.nb_set("当前任务", "worklog", "整理配置")
    eng.round += 1
    eng.nb_set("当前任务", "worklog", "写回复")  # 固定 key 覆盖 = "当前状态"
    rows = eng.nb_render("")
    wl = [(k, c) for kind, k, c in rows if kind == "worklog"]
    assert len(wl) == 1
    assert wl[0][1] == "写回复"


def test_schedule_only_active_injected():
    eng = make_eng()
    eng.nb_set("dismiss_1", "schedule", "冲突描述", status="planted")
    eng.nb_set("todo", "schedule", "已结案", status="payback")
    rows = eng.nb_render("")
    sch = [(k, c) for kind, k, c in rows if kind == "schedule"]
    assert [k for k, _ in sch] == ["dismiss_1"]


def test_constraint_and_state_present():
    eng = make_eng()
    eng.nb_set("规则", "constraint", "依据记忆流作答")
    eng.nb_set("小明", "state", "程序员")
    rows = eng.nb_render("")
    kinds = [kind for kind, _, _ in rows]
    assert "constraint" in kinds and "state" in kinds


def test_goal_filter_state():
    eng = make_eng()
    eng.nb_set("住所", "state", "住在上海")
    eng.nb_set("宠物", "state", "养了一只猫")
    rows = eng.nb_render("我的宠物")
    states = [k for kind, k, _ in rows if kind == "state"]
    assert states == ["宠物"]  # goal 命中（key 双字共享）时只注入命中项


def test_goal_miss_falls_back_to_all():
    """goal 过滤无命中 → 回退全量（去重 + cap 兜底），不静默清空事实层。"""
    eng = make_eng()
    eng.nb_set("住所", "state", "住在上海")
    eng.nb_set("宠物", "state", "养了一只猫")
    rows = eng.nb_render("你喜欢猫吗")  # 无共享双字 → 过滤无命中
    states = sorted(k for kind, k, _ in rows if kind == "state")
    assert states == ["住所", "宠物"]


def test_render_independent_each_call():
    """不滞留：nb_render 每次调用现场生成，结果只反映当前 notebook 状态。"""
    eng = make_eng()
    eng.nb_set("住所", "state", "住在上海")
    r1 = eng.nb_render("")
    eng.nb_set("住所", "state", "住在北京")
    r2 = eng.nb_render("")
    assert ("住所", "住在上海") in [(k, c) for _, k, c in r1]
    assert ("住所", "住在北京") in [(k, c) for _, k, c in r2]
    assert ("住所", "住在上海") not in [(k, c) for _, k, c in r2]
