"""测试 G：回滚对称性（内容级）。"""

from core import Engine
from util import mk_engine, single_node_book


def snapshot(e):
    return (list(e.chain_ids()),
            [e.atoms[a].value for a in e.chain_ids()],
            list(e.tiling))


def test_rollback_insert():
    e, ids, n1 = single_node_book(10)
    new = e.insert(ids[2], ids[3], [7.0])
    e.rollback()
    assert e.chain_ids() == ids
    assert e.tiling == [n1]
    assert not e.atoms[new[0]].alive


def test_rollback_delete():
    e, ids, n1 = single_node_book(10)
    e.delete([ids[2]])
    e.rollback()
    assert e.chain_ids() == ids
    assert e.atoms[ids[2]].alive


def test_rollback_replace():
    e, ids, n1 = single_node_book(10)
    new = e.replace([ids[2], ids[3]], [7.0])
    e.rollback(2)
    assert e.chain_ids() == ids
    assert not e.atoms[new[0]].alive
    assert e.atoms[ids[2]].alive and e.atoms[ids[3]].alive


def test_rollback_undoes_write():
    e, ids, n1 = single_node_book(10)
    e.rollback()
    assert e.chain_ids() == []
    assert e.tiling == [n1]
    e.merge_pass()
    assert e.tiling == []


def test_rollback_multi_reverse_order():
    e, ids, n1 = single_node_book(10)
    b = e.write([50.0, 51.0])
    e.delete([ids[0]])
    e.rollback(2)
    assert e.chain_ids() == ids
    assert not e.atoms[b[0]].alive
    assert e.atoms[ids[0]].alive


def test_rollback_content_equality_after_pass():
    e, ids, n1 = single_node_book(10)
    pre = snapshot(e)
    e.insert(ids[1], ids[2], [7.0, 8.0])
    e.delete([ids[5]])
    e.rollback(2)
    assert e.chain_ids() == pre[0]
    assert [e.atoms[a].value for a in e.chain_ids()] == pre[1]
    assert e.tiling == pre[2]
    e.merge_pass()
    assert e.chain_ids() == pre[0]
    for nid, nd in e.nodes.items():
        if nd.alive and not nd.dirty:
            assert abs(nd.value - e.oracle_value(nid)) < 1e-9


def test_rollback_restores_tiling_atom():
    e = mk_engine(decay=5.0)
    a = e.write([1.0, 2.0])
    e.tick(15)
    e.merge_pass()
    b = e.write([50.0])
    e.delete([b[0]])
    assert b[0] not in e.tiling
    e.rollback()
    assert e.atoms[b[0]].alive
    assert b[0] in e.tiling
    assert e.chain_ids() == [a[0], a[1], b[0]]


def test_rollback_marks_dirty():
    e, ids, n1 = single_node_book(10)
    e.insert(ids[1], ids[2], [7.0])
    assert e.nodes[n1].dirty
    e.rollback()
    assert e.nodes[n1].dirty
    e.merge_pass()
    assert not e.nodes[n1].dirty
    assert abs(e.nodes[n1].value - 4.5) < 1e-9


def test_rollback_empty_ledger():
    e = mk_engine()
    e.rollback(5)
    assert e.chain_ids() == []
