"""测试 B：编辑标记完备性（收养/LCA/置脏精确性）。"""

from core import Engine
from util import mk_engine, assert_dirty_exact, dirty_set, sibling_book, single_node_book


def test_insert_inside_node_adopts_and_marks():
    e, ids, n1 = single_node_book(10)
    new = e.insert(ids[3], ids[4], [99.0, 98.0])
    assert all(e.atoms[a].owner == n1 for a in new)
    assert e.nodes[n1].children[4:6] == new
    assert_dirty_exact(e, new)
    assert e.atoms[ids[3]].next == new[0]
    assert e.atoms[new[-1]].next == ids[4]


def test_insert_inside_node_no_overmark():
    e, ids, n1 = single_node_book(10)
    e.insert(ids[3], ids[4], [99.0])
    assert dirty_set(e) == {n1}
    assert not any(e.nodes[n].dirty for n in e.nodes if e.nodes[n].alive and n != n1)


def test_append_no_marking():
    e, ids, n1 = single_node_book(10)
    new = e.write([50.0])
    assert dirty_set(e) == set()
    assert e.atoms[new[0]].owner is None
    assert new[0] in e.tiling


def test_insert_book_start_no_adoption():
    e, ids, n1 = single_node_book(10)
    new = e.insert(None, ids[0], [7.0])
    assert dirty_set(e) == set()
    assert e.atoms[new[0]].owner is None
    assert e.tiling[0] == new[0]


def test_insert_between_node_and_hot_zone_no_adoption():
    e = mk_engine(decay=5.0)
    a = e.write([float(i) for i in range(5)])
    e.tick(15)
    e.merge_pass()
    n1 = e.tiling[0]
    b = e.write([100.0, 101.0])
    new = e.insert(a[-1], b[0], [50.0])
    assert e.atoms[new[0]].owner is None
    assert dirty_set(e) == set()
    assert e.tiling == [n1, *new, *b]


def test_insert_between_siblings_adopts_into_parent():
    e, a1, a2, n1, n2, p = sibling_book()
    new = e.insert(a1[-1], a2[0], [5.0])
    assert e.atoms[new[0]].owner == p.id
    assert_dirty_exact(e, new)
    assert dirty_set(e) == {p.id}
    assert e.nodes[p.id].children.index(n1.id) + 1 == e.nodes[p.id].children.index(new[0])
    assert not e.nodes[n1.id].dirty and not e.nodes[n2.id].dirty


def test_delete_inside_node_marks_chain_only():
    e, ids, n1 = single_node_book(10)
    e.delete([ids[2], ids[3]])
    assert_dirty_exact(e, [ids[2], ids[3]])
    assert dirty_set(e) == {n1}
    assert not e.atoms[ids[2]].alive
    assert e.atoms[ids[1]].next == ids[4]


def test_delete_tiling_atom_no_marking():
    e = mk_engine(decay=5.0)
    a = e.write([float(i) for i in range(5)])
    e.tick(15)
    e.merge_pass()
    b = e.write([50.0])
    e.delete([b[0]])
    assert dirty_set(e) == set()
    assert b[0] not in e.tiling


def test_replace_spans_two_nodes():
    e = mk_engine(decay=5.0)
    a = e.write([float(i) for i in range(10)])
    e.tick(15)
    e.merge_pass()
    n1 = e.tiling[0]
    b = e.write([float(i) for i in range(10, 20)])
    e.tick(10)
    e.merge_pass()
    e.tick(5)
    e.merge_pass()
    p = e.tiling[0]
    assert e.nodes[p].children[0] == n1
    assert e.nodes[n1].parent == p
    affected_old = [a[0], b[0]]
    new = e.replace(affected_old, [3.0])
    expected = set(e._owner_chain(a[0])) | set(e._owner_chain(b[0]))
    assert dirty_set(e) == expected
    assert e.nodes[n1].dirty and e.nodes[p].dirty
    assert e.atoms[new[0]].owner is None


def test_replace_partial_node_marks_whole_chain():
    e, ids, n1 = single_node_book(10)
    new = e.replace([ids[4]], [42.0])
    assert_dirty_exact(e, [ids[4]])
    assert dirty_set(e) == {n1}
    assert e.atoms[ids[3]].next == new[0]
    assert e.atoms[new[0]].next == ids[5]


def test_delete_last_atom_of_node_marks_node():
    e, ids, n1 = single_node_book(10)
    e.delete([ids[9]])
    assert_dirty_exact(e, [ids[9]])
    assert dirty_set(e) == {n1}


def test_insert_region_containing_prior_mid_insert():
    e, ids, n1 = single_node_book(10)
    mid = e.insert(ids[3], ids[4], [99.0, 98.0])
    e.merge_pass()
    assert dirty_set(e) == set()
    new = e.replace([ids[3], *mid, ids[4]], [5.0, 6.0])
    assert_dirty_exact(e, [ids[3], *mid, ids[4]])
    assert e.atoms[new[0]].owner == n1
    assert e.chain_ids() == [*ids[:3], *new, *ids[5:]]


def test_edit_then_immediate_assemble_no_dirty_node_emitted():
    e, ids, n1 = single_node_book(10)
    e.delete([ids[2], ids[3]])
    emitted = e._emit()
    kinds = {k for k, _, _ in emitted}
    assert kinds == {"atom"}
    assert e.emission_atoms(emitted) == e.chain_ids()
    new = e.insert(ids[1], ids[4], [7.0, 8.0])
    emitted = e._emit()
    assert e.emission_atoms(emitted) == e.chain_ids()
    assert e.emission_atoms(emitted) == [ids[0], ids[1], *new, ids[4], *ids[5:]]


def test_insert_between_atoms_of_child_nodes_adopts_deepest():
    e, ids, n1 = single_node_book(10)
    new = e.insert(ids[2], ids[7], [1.0])
    assert e.atoms[new[0]].owner == n1


def test_clean_nodes_values_valid_after_ops():
    e, ids, n1 = single_node_book(10)
    e.insert(ids[1], ids[2], [50.0])
    e.merge_pass()
    assert dirty_set(e) == set()
    for nid, nd in e.nodes.items():
        if nd.alive and not nd.dirty:
            if nd.placeholder:
                assert nd.value is None
            else:
                assert abs(nd.value - e.oracle_value(nid)) < 1e-9
