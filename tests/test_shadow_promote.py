"""测试 I：召回影子注入 + 提升（路径分裂）+ 激活计数 + 从属标注。"""

import pytest

from core import Engine
from util import mk_engine, assert_dirty_exact, dirty_set, sibling_book, single_node_book


def kinds(emitted):
    return [k for k, _, _ in emitted]


def test_shadow_injected_after_covering_item():
    e, a1, a2, n1, n2, p = sibling_book()
    out = e.assemble(shadows=[n1.id], budget=100)
    assert kinds(out) == ["node", "shadow"]
    assert out[0][1] == p.id
    assert out[1][1] == n1.id and out[1][0] == "shadow"


def test_shadow_top_level_dedup():
    e, a1, a2, n1, n2, p = sibling_book()
    out = e.assemble(shadows=[p.id], budget=100)
    assert kinds(out) == ["node"]
    assert out[0][1] == p.id


def test_shadow_dedup_multiple_and_dead():
    e, a1, a2, n1, n2, p = sibling_book()
    e.delete([a2[0]])
    out = e.assemble(shadows=[p.id, n1.id, n2.id], budget=100)
    assert all(t[0] != "shadow" for t in out)
    assert out[0][1] == n1.id


def test_shadow_skips_dirty():
    e, a1, a2, n1, n2, p = sibling_book()
    e.delete([a1[0]])
    out = e.assemble(shadows=[n1.id, n2.id, p.id], budget=100)
    assert all(t[0] != "shadow" for t in out)
    assert out[0][0] == "atom"


def nested_book():
    e = mk_engine(decay=5.0)
    a1 = e.write([1.0, 2.0, 3.0])
    a2 = e.write([4.0, 5.0])
    a3 = e.write([10.0, 11.0, 12.0])
    n11, n1, n2, p = e._new_node(), e._new_node(), e._new_node(), e._new_node()
    for aid in a1:
        e.atoms[aid].owner = n11.id
        n11.children.append(aid)
    for aid in a2:
        e.atoms[aid].owner = n1.id
        n1.children.append(aid)
    for aid in a3:
        e.atoms[aid].owner = n2.id
        n2.children.append(aid)
    n11.parent = n1.id
    n1.children.insert(0, n11.id)
    for n, v in ((n1, 3.0), (n2, 11.0)):
        n.parent = p.id
        p.children.append(n.id)
        n.value = v
    n11.value = 2.0
    p.value = e._weighted_avg([n1.id, n2.id])
    e.tiling = [p.id]
    return e, n11, n1, n2, p


def test_shadow_nested_ordering_through_same_covering():
    e, n11, n1, n2, p = nested_book()
    out = e.assemble(shadows=[n11.id, n1.id], budget=100)
    ids = [t[1] for t in out if t[0] == "shadow"]
    assert ids == [n1.id, n11.id]
    assert out[0][1] == p.id


def test_shadow_nested_order_ancestor_first():
    e, a1, a2, n1, n2, p = sibling_book()
    out = e.assemble(shadows=[n1.id, p.id], budget=100)
    ids = [t[1] for t in out]
    assert ids == [p.id, n1.id]


def test_shadow_after_dirty_expansion_slots_at_own_child():
    e, n11, n1, n2, p = nested_book()
    e.delete([e.nodes[n11.id].children[0]])
    out = e.assemble(shadows=[n11.id], budget=100)
    assert "shadow" not in kinds(out)
    assert e.emission_atoms(out) == e.chain_ids()


def test_render_emission_marks_affiliation():
    e, a1, a2, n1, n2, p = sibling_book()
    out = e.assemble(shadows=[n2.id], budget=100)
    text = e.render_emission(out)
    lines = text.splitlines()
    assert any("⊂N" in ln for ln in lines), text
    assert any("召回影子" in ln for ln in lines), text
    assert any("⊂N9" in ln for ln in lines), text


def triple_book():
    e = mk_engine(decay=5.0)
    groups = []
    for vals in ([1.0, 2.0], [10.0, 11.0], [20.0, 21.0]):
        groups.append(e.write(vals))
    n1, n2, n3, p = e._new_node(), e._new_node(), e._new_node(), e._new_node()
    for n, ids, v in ((n1, groups[0], 1.5), (n2, groups[1], 10.5), (n3, groups[2], 20.5)):
        n.parent = p.id
        p.children.append(n.id)
        for aid in ids:
            e.atoms[aid].owner = n.id
            n.children.append(aid)
        n.value = v
    p.value = e._weighted_avg([n1.id, n2.id, n3.id])
    e.tiling = [p.id]
    return e, groups, n1, n2, n3, p


def test_promote_fragments_dirty_and_rebuildable():
    e, g, n1, n2, n3, p = triple_book()
    e.promote(n1.id)
    frags = [nid for nid, nd in e.nodes.items() if nd.alive and nd.dirty]
    assert frags
    assert e.nodes[n2.id].parent in frags
    e.merge_pass()
    assert dirty_set(e) == set()
    for nid, nd in e.nodes.items():
        if nd.alive and not nd.placeholder:
            assert abs(nd.value - e.oracle_value(nid)) < 1e-9
    assert e.emission_atoms(e._emit()) == e.chain_ids()


def test_promote_single_child_collapse():
    e, a1, a2, n1, n2, p = sibling_book()
    e.delete(a2)
    e.promote(n1.id)
    assert e.nodes[n1.id].parent is None
    assert not e.nodes[p.id].alive
    assert e.tiling == [n1.id, n2.id]
    e.merge_pass()
    assert e.tiling == [n1.id]
    assert e.emission_atoms(e._emit()) == e.chain_ids()


def test_promote_top_level_noop():
    e, a1, a2, n1, n2, p = sibling_book()
    e.promote(p.id)
    assert e.nodes[p.id].alive
    assert e.tiling == [p.id]


def test_promote_deep_path_to_top():
    e = mk_engine(decay=5.0)
    e.write([float(i) for i in range(10)])
    e.tick(15)
    e.merge_pass()
    n1 = e.tiling[0]
    e.write([float(i) for i in range(10, 20)])
    e.tick(15)
    e.merge_pass()
    p = e.tiling[0]
    e.write([float(i) for i in range(20, 30)])
    e.tick(15)
    e.merge_pass()
    top = e.tiling[0]
    e.promote(n1)
    assert e.nodes[n1].parent is None
    assert n1 in e.tiling
    for dead in (p, top):
        assert not e.nodes[dead].alive
    assert e.emission_atoms(e._emit()) == e.chain_ids()
    for nid, nd in e.nodes.items():
        if nd.alive and not nd.dirty and not nd.placeholder:
            assert abs(nd.value - e.oracle_value(nid)) < 1e-9


def test_promote_fragments_dirty_and_rebuildable():
    e, g, n1, n2, n3, p = triple_book()
    e.promote(n1.id)
    frags = [nid for nid, nd in e.nodes.items() if nd.alive and nd.dirty]
    assert frags
    assert e.nodes[n2.id].parent in frags
    e.merge_pass()
    assert dirty_set(e) == set()
    for nid, nd in e.nodes.items():
        if nd.alive and not nd.placeholder:
            assert abs(nd.value - e.oracle_value(nid)) < 1e-9
    assert e.emission_atoms(e._emit()) == e.chain_ids()


def test_activation_count_and_auto_promote():
    e, a1, a2, n1, n2, p = sibling_book()
    for _ in range(e.promote_threshold - 1):
        e.retrieve(2.0, budget=2)
    assert e.nodes[n1.id].parent is not None
    e.retrieve(2.0, budget=2)
    assert e.nodes[n1.id].parent is None
    assert n1.id in e.tiling
    assert any(ev[0] == "promote" for ev in e.events)


def test_activation_count_not_incremented_on_path_only():
    e, a1, a2, n1, n2, p = sibling_book()
    e.retrieve(2.0, budget=10)
    assert e.nodes[n1.id].activation_count == 1
    assert e.nodes[p.id].activation_count == 0


def test_promoted_node_merges_naturally_after_decay():
    e, a1, a2, n1, n2, p = sibling_book()
    e.promote(n1.id)
    e.tick(15)
    e.merge_pass()
    assert e.emission_atoms(e._emit()) == e.chain_ids()
    assert len(e.tiling) >= 1
