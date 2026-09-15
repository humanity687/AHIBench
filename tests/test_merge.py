"""测试 D：merge pass / 脏重建 / 占位。"""

import pytest

from core import Engine
from util import mk_engine, assert_dirty_exact, dirty_set, single_node_book


def test_low_run_merges_into_single_parent():
    e, ids, n = single_node_book(10)
    nd = e.nodes[n]
    assert len(nd.children) == 10
    assert abs(nd.value - 4.5) < 1e-9
    # B2（2026-08-29）：父 HP = 保底 70 + 组内成员加权均 HP（原子 tick15 后 HP=25）
    assert nd.hp == pytest.approx(95.0)
    assert all(e.atoms[aid].owner == n for aid in ids)
    assert e.tiling == [n]


def test_high_hp_breaks_run():
    e = mk_engine(decay=5.0)
    a = e.write([float(i) for i in range(5)])
    e.tick(15)
    e.merge_pass()
    n1 = e.tiling[0]
    b = e.write([100.0 + i for i in range(5)])
    e.tick(8)
    e.merge_pass()
    assert e.tiling == [n1, *b]


def test_single_member_run_not_merged():
    e = mk_engine(decay=5.0)
    e.write([1.0, 2.0, 3.0])
    e.tick(15)
    e.merge_pass()
    n1 = e.tiling[0]
    b = e.write([50.0])
    e.tick(9)
    e.merge_pass()
    assert e.tiling == [n1, b[0]]
    e.tick(5)
    e.merge_pass()
    assert e.tiling == [n1, b[0]]
    e.tick(1)
    e.merge_pass()
    assert len(e.tiling) == 1


def test_cap_chunking():
    e = mk_engine(decay=5.0, merge_cap_atoms=4)
    e.write([float(i) for i in range(10)])
    e.tick(15)
    e.merge_pass()
    assert len(e.tiling) == 3
    for nid in e.tiling:
        assert e._subtree_atom_count(nid) <= 4
    vs = sorted(e.nodes[nid].value for nid in e.tiling)
    assert abs(vs[0] - 1.5) < 1e-9
    assert abs(vs[1] - 5.5) < 1e-9
    assert abs(vs[2] - 8.5) < 1e-9


def test_chunking_never_creates_single_child_parent():
    e = mk_engine(decay=5.0, merge_cap_atoms=3)
    e.write([float(i) for i in range(10)])
    e.tick(15)
    e.merge_pass()
    for nid in e.tiling:
        assert len(e.nodes[nid].children) >= 2, f"node {nid} has {len(e.nodes[nid].children)} children"


def test_merge_run_with_oversized_member_still_multi_child():
    e = mk_engine(decay=5.0, merge_cap_atoms=3)
    e.write([float(i) for i in range(10)])
    e.tick(15)
    e.merge_pass()
    assert len(e.tiling) == 3
    b = e.write([100.0 + i for i in range(3)])
    e.tick(15)
    e.merge_pass()
    for nid in e.tiling:
        assert len(e.nodes[nid].children) >= 2, f"node {nid} has {len(e.nodes[nid].children)} children"


def test_balance_rule_prevents_comb():
    e = mk_engine(decay=5.0, merge_depth_diff=1)
    for i in range(8):
        e.write([float(i * 10 + j) for j in range(4)])
        e.tick(15)
        e.merge_pass()
    heights = e._heights()
    max_height = max(heights.values(), default=0)
    assert max_height <= 5, f"comb: height {max_height} for 8 sequential batches"
    for nid in e.tiling:
        assert e.nodes[nid].alive
    assert e.emission_atoms(e._emit()) == e.chain_ids()


def test_balance_rule_allows_close_height_merge():
    e = mk_engine(decay=5.0, merge_depth_diff=1)
    e.write([float(i) for i in range(4)])
    e.tick(15)
    e.merge_pass()
    assert len(e.tiling) == 1
    e.write([100.0 + i for i in range(4)])
    e.tick(15)
    e.merge_pass()
    assert len(e.tiling) == 1
    assert len(e.nodes[e.tiling[0]].children) == 5


def test_merge_retired_members_not_in_tiling():
    e, ids, n = single_node_book(10)
    assert all(aid not in e.tiling for aid in ids)
    assert all(aid in e.nodes[n].children for aid in ids)


def test_rebuild_drops_dead_children():
    e, ids, n1 = single_node_book(10)
    e.delete([ids[2], ids[3]])
    e.merge_pass()
    assert ids[2] not in e.nodes[n1].children
    assert ids[3] not in e.nodes[n1].children
    expected = (sum(range(10)) - 2 - 3) / 8
    assert abs(e.nodes[n1].value - expected) < 1e-9
    assert not e.nodes[n1].dirty


def test_placeholder_with_hot_children():
    e, ids, n1 = single_node_book(10)
    new = e.insert(ids[4], ids[5], [50.0, 60.0])
    e.merge_pass()
    nd = e.nodes[n1]
    assert nd.placeholder
    assert nd.value is None
    assert all(a in nd.children for a in new)


def test_placeholder_resolves_after_cooldown():
    e, ids, n1 = single_node_book(10)
    new = e.insert(ids[4], ids[5], [50.0, 60.0])
    e.merge_pass()
    assert e.nodes[n1].placeholder
    e.tick(9)
    e.merge_pass()
    nd = e.nodes[n1]
    assert not nd.placeholder
    assert abs(nd.value - (45.0 + 110.0) / 12.0) < 1e-9


def test_node_fully_deleted_dropped():
    e, ids, n1 = single_node_book(10)
    e.delete(ids)
    e.merge_pass()
    assert not e.nodes[n1].alive
    assert n1 not in e.tiling
    assert e.chain_ids() == []


def test_dirty_rebuilt_bottom_up_nested():
    e = mk_engine(decay=5.0)
    a = e.write([float(i) for i in range(10)])
    e.tick(15)
    e.merge_pass()
    b = e.write([float(i) for i in range(10, 20)])
    e.tick(10)
    e.merge_pass()
    e.tick(5)
    e.merge_pass()
    p = e.tiling[0]
    e.replace([a[2]], [1000.0])
    assert e.nodes[p].dirty
    e.merge_pass()
    assert not e.nodes[p].dirty
    for nid, nd in e.nodes.items():
        if nd.alive and not nd.dirty:
            if nd.placeholder:
                assert nd.value is None
            else:
                assert abs(nd.value - e.oracle_value(nid)) < 1e-9


def test_cross_depth_remerge_self_heal():
    e, ids, n1 = single_node_book(10)
    e.insert(ids[4], ids[5], [99.0])
    e.tick(9)
    e.merge_pass()
    assert len(e.tiling) == 1
    assert not e.nodes[n1].placeholder
    b = e.write([200.0 + i for i in range(5)])
    e.tick(15)
    e.merge_pass()
    assert len(e.tiling) == 1
    p = e.tiling[0]
    assert e.nodes[p].children[0] == n1
    assert len(e.nodes[p].children) == 6
    assert abs(e.nodes[p].value - e.oracle_value(p)) < 1e-9


def test_rebuild_preserves_span_order():
    e, ids, n1 = single_node_book(10)
    new = e.insert(ids[2], ids[3], [7.0])
    e.delete([ids[5]])
    e.merge_pass()
    nd = e.nodes[n1]
    assert nd.children == [ids[0], ids[1], ids[2], *new, ids[3], ids[4], *ids[6:]]
