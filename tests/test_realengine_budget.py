"""测试：RealEngine 字符预算组装（F2 修复）+ α 门控【注】（F1）+ worklog 归一（P1-4）。

core 桩引擎的 assemble 保持条目数替身语义（见 test_assembly.py）；本文件测生产路径。
"""

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "realtest"))
from realengine import RealEngine


class FakeLLM:
    def __init__(self, text="【事实】中性摘要。"):
        self.text = text
        self.calls = 0
        self.last_prompt = ""

    def chat(self, system, user):
        self.calls += 1
        self.last_prompt = str(user)
        return self.text, {"prompt_tokens": 1, "completion_tokens": 1}, 0.01


def mk(**kw):
    kw.setdefault("decay", 5.0)
    kw.setdefault("attention_in_decay", 1.0)   # 中性衰减：隔离注意力窗口效应
    kw.setdefault("attention_out_decay", 1.0)
    kw.setdefault("distill_parallel", 1)
    return RealEngine(FakeLLM(), encoder=None, **kw)


# ── F2：字符预算 ──

def test_assembly_char_budget_trims_low_hp_nodes():
    eng = mk(assembly_budget=100000)  # 先放宽合并
    eng.write([f"第{i}句内容。" for i in range(20)])
    eng.tick(15)
    eng.merge_pass()          # 20 原子 → 1 节点
    eng.write(["新原子一。", "新原子二。"])
    node = [nid for nid, nd in eng.nodes.items() if nd.alive][0]
    emitted = eng._emit()
    assert emitted[0][0] == "node" and emitted[0][1] == node
    full_chars = eng._emission_chars(emitted)
    assert full_chars > 0
    # 预算 = 一半字符 → 裁剪后应小于预算
    out = eng.assemble(budget=full_chars // 2)
    assert eng._emission_chars(out) <= full_chars // 2


def test_assembly_char_budget_never_drops_atoms():
    eng = mk(assembly_budget=100000)
    eng.write(["一。", "二。", "三。", "四。", "五。"])
    out = eng.assemble(budget=1)   # 预算 1 字符 → 原子永不裁 → 硬截断
    kinds = [k for k, _, _ in out]
    assert all(k == "atom" for k in kinds)
    assert any(e[0] == "hard_trim" for e in eng.events)
    assert len(out) >= 1


def test_assembly_char_budget_over_flow_triggers_merge():
    eng = mk(assembly_budget=100000)
    eng.write([f"句子{i}号。" for i in range(30)])
    eng.tick(15)
    before = eng.distill["calls"]
    eng.assemble(budget=100)   # 30 原子渲染远超 100 字符 → 触发 merge_pass
    assert eng.distill["calls"] > before


# ── F1：α 门控【注】──

def test_alpha_zero_forces_note_empty():
    llm = FakeLLM(text="【事实】生日是 3 月 14 日。\n【注】这日子有点意思")
    eng = RealEngine(llm, encoder=None, distill_parallel=1, style_alpha=0.0)
    out = eng._summarize(["probe 说生日是 3 月 14 日。"])
    assert "【注】" not in out
    assert "生日" in out


def test_alpha_one_note_optional():
    llm = FakeLLM(text="【事实】生日是 3 月 14 日。")
    eng = RealEngine(llm, encoder=None, distill_parallel=1, style_alpha=1.0)
    out = eng._summarize(["probe 说生日是 3 月 14 日。"])
    assert "【注】" not in out          # LLM 不加【注】也不违规（可加可不加）
    assert "生日" in out


def test_alpha_one_note_kept_when_present():
    llm = FakeLLM(text="【事实】生日是 3 月 14 日。\n【注】π 日同款")
    eng = RealEngine(llm, encoder=None, distill_parallel=1, style_alpha=1.0)
    out = eng._summarize(["probe 说生日是 3 月 14 日。"])
    assert "【注】π 日同款" in out


# ── P1-4：worklog 固定 key 覆盖 ──

def test_worklog_key_normalized_to_current():
    eng = mk()
    eng.nb_set("第1轮", "worklog", "第一条记录")
    eng.nb_set("第2轮", "worklog", "第二条记录")
    worklogs = [e for e in eng.notebook.values() if e["kind"] == "worklog"]
    assert len(worklogs) == 1
    assert worklogs[0]["content"] == "第二条记录"
    assert worklogs[0]["key"] == "current"


# ── join 无超时修复（2026-08-28 主实验 neutral 卡死 8 分钟根因）──

class HangingLLM:
    """模拟 DeepSeek 挂起：不返回也不抛异常。"""
    def __init__(self, hang=5.0, text="【事实】中性摘要。"):
        self.hang = hang
        self.text = text
        self.calls = 0

    def chat(self, system, user):
        self.calls += 1
        import time
        time.sleep(self.hang)
        return self.text, {"prompt_tokens": 1, "completion_tokens": 1}, 0.01


def test_merge_runs_hanging_worker_timeout_no_deadlock():
    """蒸馏 worker 挂起 → batch 超时标 failed → merge 现场兜底完成，不死锁。
    （无超时 join 会无限等待=2026-08-28 neutral 卡死根因；修复后：
     批等待 ≤0.3s，现场兜底每 chunk ≤ hang 时长，整体有界）"""
    eng = RealEngine(HangingLLM(hang=5.0), encoder=None, distill_parallel=2,
                     decay=5.0, attention_in_decay=1.0, attention_out_decay=1.0,
                     distill_batch_timeout=0.3)
    eng.write([f"第{i}句内容。" for i in range(30)])
    eng.tick(15)
    t0 = time.time()
    eng.merge_pass()
    wall = time.time() - t0
    assert wall < 30.0, f"merge_pass 卡死: wall={wall:.1f}s"
    # 超时未完成的缓存被 pre_distill_take 消费（pop+计数 wasted）→ 现场兜底蒸馏
    assert eng.pre_distill_wasted > 0, "应存在被超时标记并消费的 failed 缓存"


def test_daemon_pool_caps_concurrency_submit_never_blocks():
    """池并发被 max_workers 限死；submit 永不阻塞（队列缓冲）。"""
    import threading
    eng = mk(distill_parallel=2)
    pool = eng._distill_pool
    started = []
    ev = threading.Event()

    def slow(n):
        started.append(n)
        ev.wait()   # 全部任务挂起：验证同时执行数 ≤ 并发上限
        return n

    f0 = pool.submit(slow, 0)
    f1 = pool.submit(slow, 1)
    f2 = pool.submit(slow, 2)   # 应排队，不阻塞调用
    time.sleep(0.15)
    assert len(started) == 2, f"并发应被限死为 2，实际 {len(started)}"
    ev.set()
    assert f0.result(timeout=2) == 0
    assert f2.result(timeout=2) == 2


# ── B2：使用度沿树上继承（2026-08-29 拍定，属性测试断言）──

def test_merge_parent_hp_inherits_group_usage():
    """高 HP 成员合并 → 父 HP 高；老重组 → 父保底 70。
    注意：merge 条件 = HP < 30，故组均 HP ∈ [0,30)，父 HP ∈ [70,100)。
    继承幅度 = 组均（最大 ~30 = recall_boost 量级）。"""
    eng = mk()
    ids = eng.write([f"句{i}。" for i in range(4)])
    for aid in ids:
        eng.atoms[aid].hp = 25.0   # 低但非零（近期合并的组）
    eng.merge_pass()
    hot_parent = [nd for nd in eng.nodes.values() if nd.alive]
    assert hot_parent, "应有父节点"
    assert hot_parent[0].hp == pytest.approx(95.0)  # 70 + 25
    # 老重组：成员 HP=0 → 父 = 70 保底
    eng2 = mk()
    ids2 = eng2.write([f"句{i}。" for i in range(4)])
    for aid in ids2:
        eng2.atoms[aid].hp = 0.0
    eng2.merge_pass()
    cold_parent = [nd for nd in eng2.nodes.values() if nd.alive]
    assert cold_parent[0].hp == pytest.approx(70.0), "老重组父 HP = 保底 70"


def test_merge_parent_hp_never_exceeds_max():
    """父 HP 恒 ≤ max_hp（合并路径组均 <30，父 ≤100；属性断言）。"""
    eng = mk()
    ids = eng.write([f"句{i}。" for i in range(4)])
    for aid in ids:
        eng.atoms[aid].hp = 29.9
    eng.merge_pass()
    p = [nd for nd in eng.nodes.values() if nd.alive][0]
    assert p.hp <= 150.0


def test_node_window_slow_decay_and_no_cliff():
    """节点窗内慢衰减（×in_decay）；出窗 = decay（无 3x 悬崖）。"""
    eng = mk()
    ids = eng.write([f"句{i}。" for i in range(4)])
    for aid in ids:
        eng.atoms[aid].hp = 0.0
    eng.merge_pass()
    p = [nd for nd in eng.nodes.values() if nd.alive][0]
    p.hp = 70.0
    eng.tick(1)
    # 窗内（HP 70 ≥ 45 且在 node_cap 前 20）：-5（decay5 × 0.5）
    assert p.hp == pytest.approx(65.0)
    p.hp = 30.0
    eng.tick(1)
    # 出窗：-5（decay），不是 -15（悬崖）
    assert p.hp == pytest.approx(25.0)


# ── B+D：时序分流（2026-08-30 拍定：上下文管理定位——旧值生命周期过后
#    以〔早期〕标注流向蒸馏，以历史态叙述而非当前态复述）──

def test_atom_part_marks_low_hp_as_early():
    eng = mk()
    ids = eng.write(["我在学 Rust。", "我改学 Go 了。"])
    old, new = ids[0], ids[1]
    eng.atoms[old].hp = 0.0          # 旧值：出生命周期
    eng.atoms[new].hp = 80.0         # 新值：当前
    assert eng._atom_part(old) == "〔早期〕我在学 Rust。"
    assert eng._atom_part(new) == "我改学 Go 了。"


def test_group_parts_uses_early_markers():
    eng = mk()
    ids = eng.write(["我在学 Rust。", "我改学 Go 了。"])
    eng.atoms[ids[0]].hp = 0.0
    eng.atoms[ids[1]].hp = 80.0
    parts = eng._group_parts(ids)
    assert parts == ["〔早期〕我在学 Rust。", "我改学 Go 了。"]


def test_weighted_avg_path_marks_early_atoms():
    """现场蒸馏路径（无缓存）也用 _atom_part。"""
    eng = mk()
    ids = eng.write(["我在学 Rust。", "我改学 Go 了。"])
    eng.atoms[ids[0]].hp = 0.0
    eng.atoms[ids[1]].hp = 80.0
    eng._pre_distill = {}   # 清缓存强制现场蒸馏
    out = eng._weighted_avg(ids)
    # FakeLLM 返回固定文本；验证不崩且 LLM 收到的 prompt 含〔早期〕
    assert out is not None
    assert eng.llm.last_prompt and "〔早期〕" in eng.llm.last_prompt


def test_prompts_contain_time_ordering_rule():
    from realtest.realengine import SUMMARIZE_PROMPT_NEUTRAL, SUMMARIZE_PROMPT_STYLED
    for p in (SUMMARIZE_PROMPT_NEUTRAL, SUMMARIZE_PROMPT_STYLED):
        assert "时序分流规则" in p and "以最新状态为当前事实" in p
