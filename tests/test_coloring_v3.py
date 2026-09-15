"""测试：染色 v3（指针密度）——差集索引 / α 门控 / 人称-来源绑定 /
防复合（_fact_part）/ 蒸馏快照落盘 / 树快照落盘 / 〔已回复〕被动渲染。"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "realtest"))
from realengine import RealEngine


class ParamLLM:
    """可配置摘要输出的假 LLM。"""
    def __init__(self, fact="【事实】这是摘要", note="【注】还行"):
        self.fact = fact
        self.note = note
        self.calls = 0

    def chat(self, system, user):
        self.calls += 1
        text = self.fact if self.note is None else f"{self.fact}\n{self.note}"
        return text, {"prompt_tokens": 1, "completion_tokens": 1}, 0.01


def make_engine(llm, **kw):
    return RealEngine(llm, distill_parallel=1, **kw)


# ── 差集索引 ──

def test_uncovered_count_detects_missing_tokens():
    eng = make_engine(ParamLLM())
    parts = ["probe 说生日是 3 月 14 日。", "会议定了新协议 v0_5 双人在场。"]
    fact = "生日是 3 月 14 日。"          # 只覆盖第一个 part
    assert eng._uncovered_count(parts, fact) == 1
    assert eng._uncovered_count(parts, "3 月 14 日 v0_5 协议") == 0   # 全覆盖
    assert eng._uncovered_count(["纯文本无特征。", "另一个纯文本。"], fact) == 0  # 无 token 跳过


def test_summarize_appends_index_line():
    llm = ParamLLM(fact="【事实】生日是 3 月 14 日。", note=None)
    eng = make_engine(llm)
    parts = ["probe 说生日是 3 月 14 日。", "会议定了新协议 v0_5 双人在场。"]
    out = eng._summarize(parts)
    assert "〔另有1条未展开，可@span下钻〕" in out


def test_summarize_no_index_when_covered():
    llm = ParamLLM(fact="【事实】3 月 14 日 v0_5 都讲到了。", note=None)
    eng = make_engine(llm)
    parts = ["probe 说生日是 3 月 14 日。", "会议定了新协议 v0_5 双人在场。"]
    out = eng._summarize(parts)
    assert "另有" not in out


def test_fact_part_strips_index_and_note():
    eng = make_engine(ParamLLM())
    v = "【事实】核心内容。〔另有2条未展开，可@span下钻〕\n【注】风格评注"
    assert eng._fact_part(v) == "核心内容。"


def test_group_parts_uses_fact_part_only():
    eng = make_engine(ParamLLM())
    a1 = eng._new_atom("原子原文一", metadata={"source": "user:probe"})
    eng.insert(eng.tail, None, atoms=[a1])
    nid = eng._nid()
    import core
    nd = core.Node(id=nid, value="【事实】节点摘要。〔另有3条未展开，可@span下钻〕\n【注】不应进上层")
    nd.children = [a1.id]
    eng.nodes[nid] = nd
    parts = eng._group_parts([nid])
    assert parts == ["节点摘要。"]


# ── α 门控 ──

def test_alpha_zero_suppresses_style_provider():
    called = []
    llm = ParamLLM()
    eng = make_engine(llm, style_alpha=0.0)
    eng.style_provider = lambda: called.append(1) or "风格指南"
    eng._summarize(["一段", "两段"])
    assert called == []


def test_alpha_one_calls_style_provider():
    called = []
    llm = ParamLLM()
    eng = make_engine(llm, style_alpha=1.0)
    eng.style_provider = lambda: called.append(1) or "风格指南"
    eng._summarize(["一段", "两段"])
    assert called == [1]


# ── 人称-来源绑定（只记日志）──

def test_first_person_check_counts_violations():
    eng = make_engine(ParamLLM())
    eng._first_person_check("我注意到这个改动。")                 # 无来源主语 → 违规
    assert eng.first_person_violations >= 1
    before = eng.first_person_violations
    eng._first_person_check("我注意到 probe 说要出差。")          # 有来源 → 不违规
    assert eng.first_person_violations == before
    eng._first_person_check("灯亮着，河还在流。")                 # 非第一人称 → 不违规
    assert eng.first_person_violations == before


def test_pre_distill_worker_passes_children(tmp_path):
    """并行蒸馏路径的蒸馏快照必须带 children（Phase B 指纹溯源依赖）。"""
    p = os.path.join(str(tmp_path), "ds.jsonl")
    eng = make_engine(ParamLLM(), distill_snap_path=p)
    key = (11, 22, 33)
    parts = ["a 3 月 14 日。", "b。", "c。"]
    eng._pre_distill_worker(key, parts)
    row = json.loads(open(p, encoding="utf-8").read().strip())
    assert row["children"] == [11, 22, 33]


# ── 快照落盘 ──

def test_distill_snapshot_written(tmp_path):
    p = os.path.join(str(tmp_path), "distill_snapshots.jsonl")
    llm = ParamLLM()
    eng = make_engine(llm, distill_snap_path=p)
    eng._summarize(["源内容一。", "源内容二。"])
    lines = open(p, encoding="utf-8").read().strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["round"] == eng.round
    assert row["parts"] and row["fact"]
    assert row["style_alpha"] == 1.0


def test_dump_tree_written(tmp_path):
    p = os.path.join(str(tmp_path), "tree_snapshots.jsonl")
    eng = make_engine(ParamLLM())
    eng.insert_units([("你好。", {"source": "user:probe"}), ("再见。", {"source": "ai"})],
                     left_id=eng.tail)
    eng.dump_tree(p)
    row = json.loads(open(p, encoding="utf-8").read().strip())
    assert row["round"] == eng.round
    assert len(row["atoms"]) == 2
    assert row["tiling"] == eng.tiling
    assert any(a.get("source") == "user:probe" for a in row["atoms"].values())


# ── 〔已回复〕被动渲染 ──

def test_render_replied_label():
    eng = make_engine(ParamLLM())
    ids = eng.insert_units([("问个问题。", {"source": "user:probe"})], left_id=eng.tail)
    eng.atoms[ids[0]].metadata["replied"] = True
    emitted = eng.assemble()
    rendered = eng.render_emission(emitted)
    assert "〔已回复〕" in rendered
    assert "〔user:probe〕" in rendered
