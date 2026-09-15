"""测试 F：检索（数值替身）+ 逐字通道。"""

from core import Engine
from util import mk_engine, sibling_book


def test_retrieve_top_and_path():
    e, a1, a2, n1, n2, p = sibling_book()
    res = e.retrieve(2.0, budget=2)
    assert set(res) == {n1.id, p.id}
    assert all(not e.nodes[nid].dirty for nid in res)


def test_retrieve_positional_order():
    e, a1, a2, n1, n2, p = sibling_book()
    res = e.retrieve(11.0, budget=10)
    assert res == [p.id, n1.id, n2.id]


def test_retrieve_budget_cap_path_atomic():
    e, a1, a2, n1, n2, p = sibling_book()
    res = e.retrieve(2.0, budget=1)
    assert set(res) == {n1.id, p.id}
    res = e.retrieve(2.0, budget=1)
    assert len(res) <= 3


def test_retrieve_result_dedup():
    e, a1, a2, n1, n2, p = sibling_book()
    res = e.retrieve(2.0, budget=10)
    assert len(res) == len(set(res))


def test_retrieve_excludes_dirty():
    e, a1, a2, n1, n2, p = sibling_book()
    e.delete([a1[0]])
    res = e.retrieve(2.0, budget=10)
    assert all(not e.nodes[nid].dirty for nid in res)
    assert n1.id not in res


def test_retrieve_excludes_placeholder():
    e, a1, a2, n1, n2, p = sibling_book()
    e.nodes[n1.id].placeholder = True
    e.nodes[n1.id].value = None
    res = e.retrieve(2.0, budget=10)
    assert n1.id not in res


def test_retrieve_hp_boost():
    e, a1, a2, n1, n2, p = sibling_book()
    before = e.nodes[p.id].hp
    e.retrieve(6.5, budget=10)
    assert e.nodes[p.id].hp > before


def test_retrieve_no_candidates():
    e, a1, a2, n1, n2, p = sibling_book()
    e.delete(a1 + a2)
    assert e.retrieve(2.0, budget=10) == []


def test_verbatim_channel():
    e, a1, a2, n1, n2, p = sibling_book()
    assert e.verbatim(11.0) == [a2[1]]
    assert e.verbatim(999.0) == []
    assert e.verbatim(2.0) == [a1[1]]


def test_verbatim_order_by_position():
    e, a1, a2, n1, n2, p = sibling_book()
    hits = e.verbatim(2.0)
    assert hits == [a1[1]]


def test_verbatim_skips_dead():
    e, a1, a2, n1, n2, p = sibling_book()
    e.delete([a2[1]])
    assert e.verbatim(11.0) == []
