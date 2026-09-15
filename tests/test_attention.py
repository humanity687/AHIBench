"""注意力窗口（§4.3b）：不对称衰减 + cap 排名截断。

核心语义：窗口内 = 存活原子按 HP 降序排名 ≤ attention_cap 且 HP ≥ attention_out；
窗口内慢衰减，窗口外快衰减。只影响 HP 演化速率，不改变任何组合顺序。
"""
import pytest

from core import Engine


def make_engine(**kw):
    defaults = dict(decay=10.0, attention_cap=3, attention_out=40.0,
                    attention_in_decay=0.5, attention_out_decay=3.0)
    defaults.update(kw)
    e = Engine(**defaults)
    e.write([1.0, 2.0, 3.0, 4.0, 5.0])
    return e


def test_new_atoms_are_in_window():
    """新原子 HP 满 → 自动在窗口内（慢衰减），cap 内全部慢衰减。"""
    e = make_engine(attention_cap=10)
    hp0 = [e.atoms[a].hp for a in e.chain_ids()]
    e.tick(1)
    # 窗口内 ×0.5：10 → 5
    assert [e.atoms[a].hp for a in e.chain_ids()] == [h - 5.0 for h in hp0]


def test_out_window_decays_faster():
    """出窗（HP < attention_out）→ 快衰减 ×3；窗口内慢衰减。"""
    e = make_engine(attention_cap=10)
    last = e.chain_ids()[-1]
    e.atoms[last].hp = 30.0          # 低于阈值 40 → 窗口外
    in_ = e.chain_ids()[0]
    e.atoms[in_].hp = 90.0           # 在窗口内
    e.tick(1)
    assert e.atoms[last].hp == pytest.approx(0.0)   # 30 - 10×3
    assert e.atoms[in_].hp == pytest.approx(85.0)   # 90 - 10×0.5


def test_cap_truncation_accelerates_tail():
    """窗口内原子数超过 atom_cap → 排名超出的按窗口外衰减（即使 HP ≥ 阈值）。"""
    e = make_engine(attention_atom_cap=2)
    hps = [100.0, 95.0, 90.0, 85.0, 80.0]
    for a, h in zip(e.chain_ids(), hps):
        e.atoms[a].hp = h
    e.tick(1)
    got = [e.atoms[a].hp for a in e.chain_ids()]
    # 排名前 2（100/95）：慢衰减 -5；排名 3-5（90/85/80）：快衰减 -30
    assert got == pytest.approx([95.0, 90.0, 60.0, 55.0, 50.0])


def test_recall_boost_returns_to_window():
    """窗口外卡回血到 HP ≥ attention_out → 自动回窗（对称、无状态）。"""
    e = make_engine(attention_cap=10)
    aid = e.chain_ids()[-1]
    e.atoms[aid].hp = 30.0
    e.tick(1)  # 出窗快衰减 → 0
    assert e.atoms[aid].hp == pytest.approx(0.0)
    # 回血（recall 命中语义）到 60（≥ 40 阈值）→ 恢复慢衰减
    e.atoms[aid].hp = 60.0
    e.tick(1)
    assert e.atoms[aid].hp == pytest.approx(55.0)  # 60 - 10×0.5


def test_attention_nodes_window_slow_decay():
    """2026-08-29：节点入窗——节点窗内慢衰减（×in_decay），出窗维持 decay（无悬崖）。"""
    e = make_engine()
    e.tick(15)  # 降 HP
    e.merge_pass()
    node = [n for n, nd in e.nodes.items() if nd.alive][0]
    nd = e.nodes[node]
    nd.hp = 70.0
    e.tick(1)
    # 窗口内（HP 70 ≥ 45 且在 node_cap 内）：慢衰减 -5
    assert nd.hp == pytest.approx(65.0)
    # 出窗（HP 低）：衰减 = decay（10），不是快衰减
    nd.hp = 30.0
    e.tick(1)
    assert nd.hp == pytest.approx(20.0)


def test_tree_internal_node_can_enter_window():
    """2026-08-30 拍定：树内（非铺陈）节点与铺陈节点平等参与节点窗口——
    被召回/影子的树内节点 HP ≥ attention_out 且排名入 cap → 窗内慢衰减。"""
    e = make_engine(attention_node_cap=2)
    # 写入 30 个原子 → 低HP合并（fanout 12 分块 → 3 节点）→ 再合并成父
    e.write([float(i) for i in range(30)])
    for a in e.chain_ids():
        e.atoms[a].hp = 20.0
    e.merge_pass()
    top = [n for n, nd in e.nodes.items() if nd.alive and nd.parent is None]
    assert len(top) >= 2, f"第一次合并应产生 ≥2 顶层节点，实际 {len(top)}"
    for n in top:
        e.nodes[n].hp = 20.0
    e.merge_pass()
    internal = [n for n, nd in e.nodes.items()
                if nd.alive and n not in set(e.tiling)]
    assert internal, "应存在树内节点（父在铺陈、子在树内）"
    nd = e.nodes[internal[0]]
    nd.hp = 70.0      # HP ≥ 45
    e.tick(1)
    # 树内节点入窗（node_cap=2 内）→ 慢衰减 -5（decay=10 × in_decay=0.5）
    assert nd.hp == pytest.approx(65.0)
    # 出窗：衰减 = decay（10），无悬崖
    nd.hp = 30.0
    e.tick(1)
    assert nd.hp == pytest.approx(20.0)


def test_merge_pressure_counts_nodes_and_atoms():
    """2026-08-30：常规 merge 触发信号 = 铺陈窗外节点数 + 窗外原子数
    （weight 计）> merge_pressure_threshold。纯计算零 LLM。"""
    e = make_engine(attention_atom_cap=3, attention_node_cap=2,
                    merge_pressure_threshold=5)
    # 5 个原子全部压到窗外（<40）
    for a in e.chain_ids():
        e.atoms[a].hp = 30.0
    # 无节点：窗外 = 5 原子（weight 1）→ 5 > 5 False
    assert e.merge_pressure() is False
    # 再压 1 个到 <45 但 ≥40？直接用 weight 构造：一个 weight=2 的原子
    e.atoms[e.chain_ids()[0]].weight = 2
    # 现在窗外 = 4×1 + 2 = 6 > 5 → True
    assert e.merge_pressure() is True
    # 全部回窗 → False
    for a in e.chain_ids():
        e.atoms[a].hp = 100.0
        e.atoms[a].weight = 1
    assert e.merge_pressure() is False


def test_merge_pressure_counts_out_nodes():
    """窗外节点也计入 merge 触发信号。"""
    e = make_engine(merge_pressure_threshold=3)
    for a in e.chain_ids():
        e.atoms[a].hp = 100.0   # 原子全部窗内
    e.tick(15)
    e.merge_pass()
    node = [n for n, nd in e.nodes.items() if nd.alive][0]
    e.nodes[node].hp = 30.0     # 节点出窗（<45）
    # 窗外 = 0 原子 + 1 节点 → 1 > 3 False
    assert e.merge_pressure() is False
    # 阈值降为 0 → True
    e.merge_pressure_threshold = 0
    assert e.merge_pressure() is True


def test_order_structures_unchanged():
    """注意力窗口不改任何顺序：链序、铺陈、merge 分组保持位置序。"""
    e = make_engine()
    e.tick(15)
    chain_before = e.chain_ids()
    tiling_before = list(e.tiling)
    e.merge_pass()
    e.tick(10)
    e.merge_pass()
    # 链序依然按位置（原子 ID 顺序 = 写入顺序）
    assert e.chain_ids() == chain_before
    # 铺陈无重叠无缝隙：从头到尾遍历
    seen = set()
    for item in e.tiling:
        assert item not in seen
        seen.add(item)
    # 铺陈项要么是原子要么是节点，都在链上有序位置
    chain = e.chain_ids()
    order_pos = {a: i for i, a in enumerate(chain)}
    seq = []
    for item in e.tiling:
        if e._is_atom(item):
            seq.append(order_pos[item])
    assert seq == sorted(seq), "铺陈中原子仍按位置序"


def test_merge_after_attention_still_requires_low_hp_group():
    """窗口外 ≠ 立即合并：常规 merge 仍要求连续低 HP 组（缓冲语义）。"""
    e = make_engine()
    ids = e.chain_ids()
    # 只把第 2、4 个原子压低（不连续），第 1、3、5 保持高 HP
    for i, a in enumerate(ids):
        e.atoms[a].hp = 25.0 if i in (1, 3) else 90.0
    e.merge_pass()
    # 低 HP 原子不连续 → 不合成组 → 仍是裸原子（无节点产生）
    assert sum(1 for nd in e.nodes.values() if nd.alive) == 0


def test_force_out_merges_outside_window_atoms():
    """force_out 模式：窗外（HP < attention_out）原子立即参与合并，即使未到合并线。"""
    e = make_engine()  # attention_out=40, hp_merge_threshold=30, decay=10
    ids = e.chain_ids()
    # 全部原子压到 45：常规 merge 不合（45 > 30），force_out 合（45 < 40? 不，45 >= 40）
    # 压到 35：窗外（< 40）但未到合并线（>= 30）——force_out 才合
    for a in ids:
        e.atoms[a].hp = 35.0
    e.merge_pass()
    assert sum(1 for nd in e.nodes.values() if nd.alive) == 0, "常规模式不应合并窗外未到线原子"
    e.merge_pass(force_out=True)
    assert sum(1 for nd in e.nodes.values() if nd.alive) >= 1, "force_out 应合并窗外原子"


def test_fanout_cap_limits_children_per_parent():
    """扇出上限：单父子卡数 ≤ merge_fanout_cap。"""
    e = make_engine(attention_cap=10, merge_fanout_cap=3)
    ids = e.chain_ids()
    for a in ids:
        e.atoms[a].hp = 20.0  # 全部低 HP
    e.merge_pass()
    for nid, nd in e.nodes.items():
        if nd.alive:
            assert len(nd.children) <= 3, f"节点 N{nid} 子卡数 {len(nd.children)} > 3"


def test_attention_pressure_flag():
    """窗外原子数超 attention_atom_cap × 阈值 → attention_pressure() 为真（按铺陈口径）。"""
    e = make_engine(attention_atom_cap=3, pressure_threshold=1.5)
    for a in e.chain_ids():
        e.atoms[a].hp = 30.0  # 5 个窗外原子 > 3×1.5=4.5
    assert e.attention_pressure() is True
    for a in e.chain_ids():
        e.atoms[a].hp = 100.0
    assert e.attention_pressure() is False
    # 3 个窗外原子 ≤ 4.5 → False
    for i, a in enumerate(e.chain_ids()):
        e.atoms[a].hp = 30.0 if i < 3 else 100.0
    assert e.attention_pressure() is False


def test_pressure_counts_only_tiling_atoms():
    """pressure 按铺陈口径：已合并进父节点的窗外原子不计入。"""
    e = make_engine(attention_cap=3, pressure_threshold=1.0)
    for a in e.chain_ids():
        e.atoms[a].hp = 20.0
    e.merge_pass(force_out=True)  # 全部合并进节点
    # 全库存活窗外原子 = 5，但铺陈中 = 0（tiling 只剩节点）
    assert e.attention_pressure() is False


def test_force_out_skips_nodes():
    """force 模式只合并纯原子段：节点不参与（节点=概括卡，只走常规低 HP 合并）。"""
    e = make_engine()  # attention_out=40, hp_merge_threshold=30
    ids = e.chain_ids()
    for a in ids:
        e.atoms[a].hp = 35.0
    e.merge_pass(force_out=True)  # 合并成一个节点（5 个原子）
    node = [nid for nid, nd in e.nodes.items() if nd.alive][0]
    # 再模拟：节点与窗外原子相邻，force 合并不应把节点收编
    e.atoms[ids[0]].hp = 25.0
    e.merge_pass(force_out=True)
    for nid, nd in e.nodes.items():
        if nd.alive and nd.parent is None:
            # force 合并产生的节点：子卡必须全是原子
            assert all(e._is_atom(c) for c in nd.children), \
                f"force 合并不应包含节点子卡: N{nid}"


def test_expand_leaves_full_descendants():
    """叶子展开：命中节点 → 全部后代存活原子（按位置序）。"""
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "realtest"))
    from realengine import RealEngine
    class FakeLLM:
        def chat(self, a, b): return "摘要。", {"prompt_tokens": 1, "completion_tokens": 1}, 0.01
    eng = RealEngine(FakeLLM(), encoder=None, assembly_budget=10000,
                     attention_in_decay=1.0, attention_out_decay=1.0)
    eng.write(["第一句。", "第二句。", "第三句。", "第四句。"])
    eng.tick(15)
    eng.merge_pass()  # 4 原子 → 1 节点
    node = [nid for nid, nd in eng.nodes.items() if nd.alive][0]
    leaves = eng.expand_leaves([node])
    assert len(leaves) == 4
    assert leaves == eng.chain_ids()  # 全部后代原子、位置序
    # 原子命中保持原样
    assert eng.expand_leaves([leaves[0]]) == [leaves[0]]
    # 脏节点不展开
    eng.nodes[node].dirty = True
    assert eng.expand_leaves([node]) == []
