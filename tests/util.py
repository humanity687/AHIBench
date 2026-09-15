"""测试公用工具。"""

from core import Engine


def mk_engine(**kw):
    """机制测试用引擎：默认关闭注意力窗口的不对称衰减（倍率=1.0，衰减节奏与
    引入注意力前一致），注意力窗口自身的语义由 tests/test_attention.py 显式测试。"""
    defaults = dict(attention_in_decay=1.0, attention_out_decay=1.0)
    defaults.update(kw)
    return Engine(**defaults)


def dirty_set(e):
    return {nid for nid, nd in e.nodes.items() if nd.alive and nd.dirty}


def assert_dirty_exact(e, affected):
    expected = set()
    for aid in affected:
        expected.update(e._owner_chain(aid))
    assert dirty_set(e) == expected, f"expected {expected}, got {dirty_set(e)}"


def single_node_book(n=10):
    e = mk_engine(decay=5.0)
    ids = e.write([float(i) for i in range(n)])
    e.tick(15)
    e.merge_pass()
    assert len(e.tiling) == 1
    return e, ids, e.tiling[0]


def sibling_book():
    """白盒构造：P -> [N1, N2]；N1 含 3 原子（1,2,3），N2 含 3 原子（10,11,12）。"""
    e = mk_engine(decay=5.0)
    a1 = e.write([1.0, 2.0, 3.0])
    a2 = e.write([10.0, 11.0, 12.0])
    n1, n2, p = e._new_node(), e._new_node(), e._new_node()
    for n in (n1, n2):
        n.parent = p.id
        p.children.append(n.id)
    for aid in a1:
        e.atoms[aid].owner = n1.id
        n1.children.append(aid)
    for aid in a2:
        e.atoms[aid].owner = n2.id
        n2.children.append(aid)
    n1.value = 2.0
    n2.value = 11.0
    p.value = e._weighted_avg([n1.id, n2.id])
    e.tiling = [p.id]
    return e, a1, a2, n1, n2, p
