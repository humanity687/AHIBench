"""测试 A：原子层 + 顺序链表基础行为。"""

import pytest

from core import Engine
from util import mk_engine


def test_empty_book_write():
    e = mk_engine()
    ids = e.write([1.0, 2.0, 3.0])
    assert e.chain_ids() == ids
    assert e.head == ids[0]
    assert e.tail == ids[-1]
    assert e.tiling == ids


def test_append_order():
    e = mk_engine()
    a = e.write([1.0, 2.0])
    b = e.write([3.0, 4.0])
    assert e.chain_ids() == a + b
    assert e.tiling == a + b


def test_insert_middle_order():
    e = mk_engine()
    a = e.write([1.0, 2.0, 3.0])
    mid = e.insert(a[1], a[2], [9.0, 8.0])
    assert e.chain_ids() == [a[0], a[1], *mid, a[2]]
    assert e.tiling == [a[0], a[1], *mid, a[2]]


def test_insert_head_order():
    e = mk_engine()
    a = e.write([1.0, 2.0])
    mid = e.insert(None, a[0], [9.0])
    assert e.chain_ids() == [*mid, *a]
    assert e.head == mid[0]


def test_insert_empty_book():
    e = mk_engine()
    ids = e.insert(None, None, [5.0])
    assert e.chain_ids() == ids
    assert e.head == ids[0] and e.tail == ids[0]


def test_delete_unlinks():
    e = mk_engine()
    a = e.write([1.0, 2.0, 3.0, 4.0])
    e.delete([a[1], a[2]])
    assert e.chain_ids() == [a[0], a[3]]
    assert e.head == a[0] and e.tail == a[3]


def test_delete_keeps_tombstone_row():
    e = mk_engine()
    a = e.write([1.0, 2.0])
    e.delete([a[0]])
    assert not e.atoms[a[0]].alive
    assert a[0] in e.atoms
    assert e.chain_ids() == [a[1]]


def test_delete_head_and_tail():
    e = mk_engine()
    a = e.write([1.0, 2.0, 3.0])
    e.delete([a[0]])
    assert e.head == a[1]
    e.delete([a[2]])
    assert e.tail == a[1]
    assert e.chain_ids() == [a[1]]


def test_delete_all():
    e = mk_engine()
    a = e.write([1.0, 2.0])
    e.delete(a)
    assert e.chain_ids() == []
    assert e.head is None and e.tail is None


def test_ids_never_reused():
    e = mk_engine()
    a = e.write([1.0])
    e.delete(a)
    b = e.write([2.0])
    assert b[0] != a[0]
    assert e._idc == 2


def test_write_after_delete_chain_continues():
    e = mk_engine()
    a = e.write([1.0, 2.0])
    e.delete([a[0]])
    b = e.write([3.0])
    assert e.chain_ids() == [a[1], b[0]]


def test_replace_basic():
    e = mk_engine()
    a = e.write([1.0, 2.0, 3.0, 4.0])
    new = e.replace([a[1], a[2]], [9.0])
    assert e.chain_ids() == [a[0], new[0], a[3]]
    assert not e.atoms[a[1]].alive
    assert not e.atoms[a[2]].alive


def test_replace_entire_book():
    e = mk_engine()
    a = e.write([1.0, 2.0])
    new = e.replace(a, [5.0, 6.0, 7.0])
    assert e.chain_ids() == new
    assert e.tiling == new


def test_replace_requires_alive():
    e = mk_engine()
    a = e.write([1.0, 2.0])
    e.delete(a)
    with pytest.raises(ValueError):
        e.replace(a, [5.0])


def test_position_map():
    e = mk_engine()
    a = e.write([1.0, 2.0, 3.0])
    e.delete([a[1]])
    pos = e._position_map()
    assert pos[a[0]] == 0
    assert pos[a[2]] == 1


def test_mixed_ops_order():
    e = mk_engine()
    a = e.write([1.0, 2.0, 3.0, 4.0])
    e.delete([a[1]])
    b = e.insert(a[2], a[3], [7.0])
    e.delete([a[0]])
    c = e.write([8.0])
    assert e.chain_ids() == [a[2], *b, a[3], *c]
