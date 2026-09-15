"""测试 C：组装（clean tiling）/ 预算裁剪 / 盲区。"""

from core import Engine
from util import mk_engine, single_node_book


def test_emission_chain_order():
    e, ids, n1 = single_node_book(10)
    emitted = e._emit()
    assert emitted == [("node", n1, e.nodes[n1].value)]
    assert e.emission_atoms(emitted) == e.chain_ids()


def test_emission_with_hot_zone():
    e, ids, n1 = single_node_book(10)
    b = e.write([50.0, 51.0])
    emitted = e._emit()
    assert emitted == [("node", n1, e.nodes[n1].value),
                       ("atom", b[0], 50.0), ("atom", b[1], 51.0)]
    assert e.emission_atoms(emitted) == e.chain_ids()


def test_dirty_node_expands_to_children_in_order():
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
    e.delete([a[2]])
    emitted = e._emit()
    assert e.emission_atoms(emitted) == e.chain_ids()
    assert all(k == "atom" for k, _, _ in emitted)
    e.merge_pass()
    emitted = e._emit()
    assert emitted == [("node", p, e.nodes[p].value)]
    assert e.emission_atoms(emitted) == e.chain_ids()


def test_edit_region_peeled_view():
    e, ids, n1 = single_node_book(10)
    new = e.insert(ids[4], ids[5], [99.0, 98.0])
    emitted = e._emit()
    assert e.emission_atoms(emitted) == e.chain_ids()
    assert all(k == "atom" for k, _, _ in emitted)
    e.merge_pass()
    emitted = e._emit()
    assert emitted == [("node", n1, e.nodes[n1].value)]


def test_budget_overflow_triggers_merge():
    e = mk_engine(decay=5.0, merge_fanout_cap=50)
    e.write([float(i) for i in range(20)])
    e.tick(15)
    out = e.assemble(budget=10)
    assert len(out) == 1
    assert out[0][0] == "node"
    assert e.emission_atoms(out) == e.chain_ids()


def test_trim_drops_nodes_keeps_atoms_then_hard_trim():
    e = mk_engine(decay=5.0)
    e.write([float(i) for i in range(5)])
    e.tick(15)
    e.merge_pass()
    e.write([100.0 + i for i in range(5)])
    out = e.assemble(budget=4)
    assert len(out) == 4
    assert all(k == "atom" for k, _, _ in out)
    assert any(k[0] == "hard_trim" for k in e.events)


def test_trim_prefers_low_hp_nodes():
    e = mk_engine(decay=5.0)
    e.write([1.0, 2.0, 3.0])
    e.tick(15)
    e.merge_pass()
    b = e.write([50.0, 51.0])
    e.tick(9)
    e.merge_pass()
    e.tick(6)
    e.merge_pass()
    n2 = e.tiling[0]
    e.write([100.0, 101.0])
    out = e.assemble(budget=3)
    assert len(out) == 3
    kinds = [k for k, _, _ in out]
    assert "node" in kinds
    for k, item, _ in out:
        if k == "node":
            assert not e.nodes[item].dirty


def test_assembly_never_emits_dirty_node():
    e, ids, n1 = single_node_book(10)
    e.delete([ids[0]])
    for budget in (100, 3):
        out = e.assemble(budget=budget)
        for k, item, _ in out:
            if k == "node":
                assert not e.nodes[item].dirty


def test_placeholder_emitted_as_marker():
    e, ids, n1 = single_node_book(10)
    e.insert(ids[4], ids[5], [50.0])
    e.merge_pass()
    emitted = e._emit()
    assert emitted == [("node", n1, None)]
    assert e.nodes[n1].placeholder
