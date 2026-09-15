"""core.py — 水循环 v2 主数据结构桩（纯内存、零 LLM、数值替身）

替身约定（与真实实现的映射，见 AGENTS.md）：
- 每张卡一个数值 value；原子卡 value 由外部给定；摘要卡 value = 子树内
  存活原子值的加权平均（权重 = 存活原子数）。真实实现中为 LLM 摘要文本。
- 检索打分 = 1/(1+|value-q|)；预算单位 = 卡片数。真实实现中为相似度引擎。
- 编辑以显式区间给出（句级 diff 引擎后续接入）。

本桩的目的：验证数据结构的机制正确性（顺序链表、区间覆盖树、编辑失效
标记、懒重建、铺陈组装、检索路径去重、回滚），为后续接 LLM 蒸馏打地基。
"""
from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Optional

DECAY = 5.0
HP_MERGE_THRESHOLD = 30.0
PARENT_INIT_HP = 70.0
HOT_THRESHOLD = 60.0
RECALL_BOOST = 30.0
MAX_HP = 150.0
INITIAL_HP = 100.0
ASSEMBLY_BUDGET = 14000
RETRIEVAL_BUDGET = 200
MERGE_CAP_ATOMS = 500
MERGE_DEPTH_DIFF = 1
PROMOTE_THRESHOLD = 3
ATTENTION_CAP = 60        # 注意力窗口总名额（兼容保留；实际用分段参数）
ATTENTION_ATOM_CAP = 40   # 2026-08-29 拍：原子段名额
ATTENTION_NODE_CAP = 20   # 2026-08-30 改：节点段名额（全部存活节点，树内/影子平等参与）
ATTENTION_OUT = 45.0      # 出窗阈值：HP 低于此 → 窗口外（加速衰减）
ATTENTION_IN_DECAY = 0.5  # 窗口内衰减倍率（慢）
ATTENTION_OUT_DECAY = 3.0 # 窗口外衰减倍率（快，仅原子；节点出窗 = decay）
MERGE_FANOUT_CAP = 12     # 单父节点最大子卡数（防扇出爆炸：大量节点挂单一父下）
ATTENTION_PRESSURE = 2.0  # 窗外原子数 > attention_cap × 此值 → 触发 force_out 合并
MERGE_PRESSURE_THRESHOLD = 20  # 2026-08-30 拍：窗外节点+原子数 > 此值 → 常规 merge
ATOM_CHAR_CAP = 2000      # 原子卡文本大小上限（超限按行块再切，聊天适配 P0-1）
ATOM_WEIGHT_CHAR = 1000   # 文本超过此 → 原子卡计 weight=2（虚拟计数 + 加速衰减）


@dataclass
class Atom:
    id: int
    value: float
    alive: bool = True
    prev: Optional[int] = None
    next: Optional[int] = None
    owner: Optional[int] = None
    hp: float = INITIAL_HP
    created_round: int = 0
    weight: int = 1  # 虚拟原子计数（聊天适配 P0-1）：大卡 weight=2，只影响 HP 演化，不影响读
    metadata: dict = field(default_factory=dict)  # 元数据（source/doc 等）：蒸馏/组装可见，检索不可见


@dataclass
class Node:
    id: int
    parent: Optional[int] = None
    children: list = field(default_factory=list)
    value: Optional[float] = None
    hp: float = PARENT_INIT_HP
    dirty: bool = False
    placeholder: bool = False
    version: int = 0
    created_round: int = 0
    alive: bool = True
    activation_count: int = 0


class Engine:
    def __init__(self, **kwargs):
        self.atoms: dict[int, Atom] = {}
        self.nodes: dict[int, Node] = {}
        self.head: Optional[int] = None
        self.tail: Optional[int] = None
        self.tiling: list = []
        self.round = 0
        self.ledger: list = []
        self.archive: list = []
        self.events: list = []
        self._idc = 0
        self._rollback = False
        self.decay = kwargs.get("decay", DECAY)
        self.hp_merge_threshold = kwargs.get("hp_merge_threshold", HP_MERGE_THRESHOLD)
        self.parent_init_hp = kwargs.get("parent_init_hp", PARENT_INIT_HP)
        self.parent_hp_group_factor = kwargs.get("parent_hp_group_factor", 1.0)
        self.hot_threshold = kwargs.get("hot_threshold", HOT_THRESHOLD)
        self.recall_boost = kwargs.get("recall_boost", RECALL_BOOST)
        self.max_hp = kwargs.get("max_hp", MAX_HP)
        self.initial_hp = kwargs.get("initial_hp", INITIAL_HP)
        self.assembly_budget = kwargs.get("assembly_budget", ASSEMBLY_BUDGET)
        self.retrieval_budget = kwargs.get("retrieval_budget", RETRIEVAL_BUDGET)
        self.merge_cap_atoms = kwargs.get("merge_cap_atoms", MERGE_CAP_ATOMS)
        self.merge_depth_diff = kwargs.get("merge_depth_diff", MERGE_DEPTH_DIFF)
        self.promote_threshold = kwargs.get("promote_threshold", PROMOTE_THRESHOLD)
        # 注意力窗口（§4.3b）：只改变 HP 演化速率，不改任何顺序/组合逻辑
        # 2026-08-29：名额拆原子段 + 节点段（各自按 HP 排名互不挤占）
        self.attention_cap = kwargs.get("attention_cap", ATTENTION_CAP)
        self.attention_atom_cap = kwargs.get("attention_atom_cap", ATTENTION_ATOM_CAP)
        self.attention_node_cap = kwargs.get("attention_node_cap", ATTENTION_NODE_CAP)
        self.attention_out = kwargs.get("attention_out", ATTENTION_OUT)
        self.attention_in_decay = kwargs.get("attention_in_decay", ATTENTION_IN_DECAY)
        self.attention_out_decay = kwargs.get("attention_out_decay", ATTENTION_OUT_DECAY)
        self.merge_fanout_cap = kwargs.get("merge_fanout_cap", MERGE_FANOUT_CAP)
        self.pressure_threshold = kwargs.get("pressure_threshold", ATTENTION_PRESSURE)
        self.merge_pressure_threshold = kwargs.get("merge_pressure_threshold",
                                                   MERGE_PRESSURE_THRESHOLD)

    # ── 基础 ─────────────────────────────────────────────

    def _nid(self) -> int:
        self._idc += 1
        return self._idc

    def _is_atom(self, x) -> bool:
        return x in self.atoms

    def _is_node(self, x) -> bool:
        return x in self.nodes

    def _new_atom(self, value: float) -> Atom:
        a = Atom(id=self._nid(), value=float(value), hp=self.initial_hp, created_round=self.round)
        self.atoms[a.id] = a
        return a

    def _new_node(self) -> Node:
        n = Node(id=self._nid(), hp=self.parent_init_hp, created_round=self.round)
        self.nodes[n.id] = n
        return n

    def chain_ids(self) -> list:
        out = []
        cur = self.head
        while cur is not None:
            out.append(cur)
            cur = self.atoms[cur].next
        return out

    def _position_map(self) -> dict:
        pos = {}
        i = 0
        cur = self.head
        while cur is not None:
            pos[cur] = i
            i += 1
            cur = self.atoms[cur].next
        return pos

    def _unlink(self, aid: int) -> None:
        a = self.atoms[aid]
        left, right = a.prev, a.next
        if left is not None:
            self.atoms[left].next = right
        else:
            self.head = right
        if right is not None:
            self.atoms[right].prev = left
        else:
            self.tail = left
        a.prev = a.next = None

    # ── owner 链 / LCA ────────────────────────────────────

    def _owner_chain(self, atom_id: int) -> list:
        out = []
        a = self.atoms.get(atom_id)
        if a is None:
            return out
        nid = a.owner
        while nid is not None:
            out.append(nid)
            nid = self.nodes[nid].parent
        return out

    def _chain_from(self, node_id: int) -> list:
        out = []
        nid = node_id
        while nid is not None:
            out.append(nid)
            nid = self.nodes[nid].parent
        return out

    @staticmethod
    def _lca(chains_a: list, chains_b: list) -> Optional[int]:
        if not chains_a or not chains_b:
            return None
        sb = set(chains_b)
        for nid in chains_a:
            if nid in sb:
                return nid
        return None

    def _top_ancestor(self, atom_id: int):
        ch = self._owner_chain(atom_id)
        return ch[-1] if ch else atom_id

    def _child_containing(self, parent_id: int, atom_id: int):
        a = self.atoms[atom_id]
        nid = a.owner
        if nid is None:
            return None
        if nid == parent_id:
            return atom_id
        while nid is not None:
            p = self.nodes[nid].parent
            if p == parent_id:
                return nid
            nid = p
        return None

    def _adoption_index(self, parent_id: int, left_id, right_id) -> int:
        p = self.nodes[parent_id]
        if left_id is not None:
            lc = self._child_containing(parent_id, left_id)
            if lc is not None:
                return p.children.index(lc) + 1
        if right_id is not None:
            rc = self._child_containing(parent_id, right_id)
            if rc is not None:
                return p.children.index(rc)
        return 0

    # ── 编辑：插入 / 删除 / 替换 ──────────────────────────

    def write(self, values) -> list:
        return self.insert(self.tail, None, values)

    def insert(self, left_id, right_id, values=None, atoms=None) -> list:
        if atoms is None:
            new = [self._new_atom(v) for v in values]
        else:
            new = atoms
            for a in new:
                a.owner = None
        ids = [a.id for a in new]
        prev, nxt = left_id, right_id
        for a in new:
            a.alive = True
            a.prev, a.next = prev, nxt
            if prev is not None:
                self.atoms[prev].next = a.id
            else:
                self.head = a.id
            if nxt is not None:
                self.atoms[nxt].prev = a.id
            else:
                self.tail = a.id
            prev = a.id
        lch = self._owner_chain(left_id) if left_id is not None else []
        rch = self._owner_chain(right_id) if right_id is not None else []
        parent = self._lca(lch, rch)
        if parent is not None:
            p = self.nodes[parent]
            idx = self._adoption_index(parent, left_id, right_id)
            for a in new:
                a.owner = parent
            p.children[idx:idx] = ids
            self._mark_dirty_chain(self._chain_from(parent))
        else:
            tpos = self._tiling_index_after(left_id)
            for a in new:
                a.owner = None
            self.tiling[tpos:tpos] = ids
        if not self._rollback:
            self.ledger.append({"type": "insert", "atoms": ids})
        self._stale_archive(left_id, right_id)
        return ids

    def _tiling_index_after(self, left_id) -> int:
        if left_id is None:
            return 0
        top = self._top_ancestor(left_id)
        assert top in self.tiling, "top ancestor must be in tiling"
        return self.tiling.index(top) + 1

    def delete(self, atom_ids) -> list:
        dead = [aid for aid in atom_ids if self._is_atom(aid) and self.atoms[aid].alive]
        if not dead:
            return []
        if not self._rollback:
            self.ledger.append(
                {
                    "type": "delete",
                    "atoms": [(aid, self.atoms[aid].prev, self.atoms[aid].next) for aid in dead],
                }
            )
        left = self.atoms[dead[0]].prev
        right = self.atoms[dead[-1]].next
        for aid in dead:
            self._unlink(aid)
            self.atoms[aid].alive = False
        for aid in dead:
            if self.atoms[aid].owner is None and aid in self.tiling:
                self.tiling.remove(aid)
        self._mark_dirty_atoms(dead)
        self._stale_archive(left, right)
        return dead

    def replace(self, old_ids, new_values) -> list:
        alive_old = [aid for aid in old_ids if self._is_atom(aid) and self.atoms[aid].alive]
        if not alive_old:
            raise ValueError("replace requires at least one alive old atom")
        left = self.atoms[alive_old[0]].prev
        right = self.atoms[alive_old[-1]].next
        self.delete(alive_old)
        return self.insert(left, right, new_values)

    # ── 标记 ──────────────────────────────────────────────

    def _mark_dirty_chain(self, chain: list) -> None:
        for nid in chain:
            if nid in self.nodes:
                self.nodes[nid].dirty = True

    def _mark_dirty_atoms(self, atom_ids: list) -> None:
        for aid in atom_ids:
            self._mark_dirty_chain(self._owner_chain(aid))

    def _stale_archive(self, left_id, right_id) -> None:
        if not self.archive:
            return
        pos = self._position_map()
        lo = pos[left_id] + 1 if left_id is not None else 0
        hi = pos[right_id] - 1 if right_id is not None else len(pos)
        for e in self.archive:
            if e["stale"]:
                continue
            f, l = e["first"], e["last"]
            if f is None or l is None or f not in pos or l not in pos:
                e["stale"] = True
                continue
            if not (l < lo or f > hi):
                e["stale"] = True

    # ── 衰减 ──────────────────────────────────────────────

    def tick(self, n: int = 1) -> None:
        for _ in range(n):
            self.round += 1
            # 注意力窗口（§4.3b）：名额拆分——原子段 + 节点段各自按 HP 降序排名，
            # 互不挤占（2026-08-29 拍定）。只改变 HP 演化速率，不改变任何顺序/组合逻辑。
            in_decay = self.decay * self.attention_in_decay
            out_decay = self.decay * self.attention_out_decay
            # ── 原子段：全部存活原子（weight 计名额）──
            ranked = [aid for aid, a in self.atoms.items()
                      if a.alive and a.hp >= self.attention_out]
            ranked.sort(key=lambda aid: self.atoms[aid].hp, reverse=True)
            in_window = set()
            used = 0
            for aid in ranked:
                if used + self.atoms[aid].weight > self.attention_atom_cap:
                    break
                in_window.add(aid)
                used += self.atoms[aid].weight
            for a in self.atoms.values():
                if not a.alive:
                    continue
                rate = (in_decay if a.hp >= self.attention_out and a.id in in_window
                        else out_decay) * a.weight
                a.hp = max(0.0, a.hp - rate)
            # ── 节点段：全部存活节点按 HP 排名（2026-08-30 拍定：树内节点与
            # 铺陈节点平等参与——影子/被召回的树内节点也可入窗受保护，
            # 提取驱动巩固对节点与原子对称生效）──
            ranked_n = [nid for nid, nd in self.nodes.items()
                        if nd.alive and nd.hp >= self.attention_out]
            ranked_n.sort(key=lambda nid: self.nodes[nid].hp, reverse=True)
            in_window_n = set(ranked_n[:self.attention_node_cap])
            for nd in self.nodes.values():
                if not nd.alive:
                    continue
                if nd.id in in_window_n and nd.hp >= self.attention_out:
                    rate = in_decay
                else:
                    rate = self.decay
                nd.hp = max(0.0, nd.hp - rate)

    # ── merge pass（蒸馏替身）──────────────────────────────

    def attention_pressure(self) -> bool:
        """窗外原子数（仅统计铺陈 tiling 中的存活原子——已合并进树的子卡原子
        不再计数，否则窗外数只会随写入单调上涨，force 合并永远追不平）超过
        attention_cap × 阈值 → 应触发 force_out 合并。纯计算（零 LLM）。"""
        tiling_set = set(self.tiling)
        out_count = sum(self.atoms[aid].weight for aid in tiling_set
                        if aid in self.atoms and self.atoms[aid].alive
                        and self.atoms[aid].hp < self.attention_out)
        return out_count > self.attention_atom_cap * self.pressure_threshold

    def merge_pressure(self) -> bool:
        """常规 merge 触发信号（2026-08-30 拍定）：铺陈中窗外节点数 + 窗外
        原子数（weight 计）超过 merge_pressure_threshold → 应执行常规 merge。
        每轮调用（条件式触发替代时间式 merge_every）；merge 不只作用于原子，
        节点同样承压，故节点计入信号。纯计算（零 LLM）。"""
        tiling_set = set(self.tiling)
        out_atoms = sum(self.atoms[aid].weight for aid in tiling_set
                        if aid in self.atoms and self.atoms[aid].alive
                        and self.atoms[aid].hp < self.attention_out)
        out_nodes = sum(1 for nid in tiling_set
                        if nid in self.nodes and self.nodes[nid].alive
                        and self.nodes[nid].hp < self.attention_out)
        return out_atoms + out_nodes > self.merge_pressure_threshold

    def merge_pass(self, force_out: bool = False) -> None:
        while True:
            while True:
                dirty = [nid for nid, nd in self.nodes.items() if nd.alive and nd.dirty]
                if not dirty:
                    break
                dirty.sort(key=self._depth, reverse=True)
                for nid in dirty:
                    self._rebuild(nid)
            targets = [nid for nid, nd in self.nodes.items()
                       if nd.alive and nd.placeholder and self._cooled(nid)]
            if not targets:
                break
            for nid in targets:
                self._rebuild(nid)
        self._merge_runs(force_out)

    def _depth(self, nid: int) -> int:
        d = 0
        cur = self.nodes[nid].parent
        while cur is not None:
            d += 1
            cur = self.nodes[cur].parent
        return d

    def _child_alive(self, c) -> bool:
        if self._is_atom(c):
            return self.atoms[c].alive
        return self.nodes[c].alive

    def _is_hot(self, c) -> bool:
        if self._is_atom(c):
            return self.atoms[c].hp >= self.hot_threshold
        return self.nodes[c].hp >= self.hot_threshold

    def _cooled(self, nid: int) -> bool:
        nd = self.nodes[nid]
        if not nd.placeholder:
            return False
        kids = [c for c in nd.children if self._child_alive(c)]
        if not kids:
            return True
        return all(not self._is_hot(c) for c in kids)

    def _rebuild(self, nid: int) -> None:
        nd = self.nodes[nid]
        alive_children = [c for c in nd.children if self._child_alive(c)]
        if not alive_children:
            self._drop_node(nid)
            self._mark_parent_dirty(nid)
            return
        hot = [c for c in alive_children if self._is_hot(c)]
        old_value = nd.value
        if hot:
            nd.value = None
        else:
            nd.value = self._weighted_avg(alive_children)
        nd.children = alive_children
        nd.placeholder = bool(hot)
        nd.dirty = False
        old_version = nd.version
        nd.version += 1
        self._archive(nid, old_version, old_value, self._first_atom(nid), self._last_atom(nid))
        self._mark_parent_dirty(nid)
        self.events.append(("rebuild", nid, nd.placeholder))

    def _mark_parent_dirty(self, nid: int) -> None:
        nd = self.nodes[nid]
        if nd.parent is not None and nd.parent in self.nodes and self.nodes[nd.parent].alive:
            self.nodes[nd.parent].dirty = True

    def _drop_node(self, nid: int) -> None:
        nd = self.nodes[nid]
        if nd.parent is not None:
            p = self.nodes[nd.parent]
            if nid in p.children:
                p.children.remove(nid)
        if nid in self.tiling:
            self.tiling.remove(nid)
        self._archive(nid, nd.version, nd.value, self._first_atom(nid), self._last_atom(nid))
        nd.alive = False
        self.events.append(("drop", nid))

    def _archive(self, node_id, version, value, first, last) -> None:
        self.archive.append(
            {"node_id": node_id, "version": version, "value": value,
             "first": first, "last": last, "stale": False}
        )

    def _subtree_atom_count(self, item) -> int:
        if self._is_atom(item):
            return 1
        return sum(self._subtree_atom_count(c) for c in self.nodes[item].children if self._child_alive(c))

    def _effective_count(self, item) -> int:
        if self._is_atom(item):
            return 1 if self.atoms[item].alive else 0
        nd = self.nodes[item]
        if not nd.alive or nd.placeholder:
            return 0
        return sum(self._effective_count(c) for c in nd.children if self._child_alive(c))

    def _weighted_avg(self, children) -> Optional[float]:
        acc = 0.0
        wsum = 0.0
        for c in children:
            if self._is_atom(c):
                w, v = self._effective_count(c), self.atoms[c].value
            else:
                nd = self.nodes[c]
                w = self._effective_count(c)
                v = nd.value
            if v is None or w == 0:
                continue
            acc += w * v
            wsum += w
        if wsum == 0:
            return None
        return acc / wsum

    def _heights(self) -> dict:
        heights = {}
        stack = [(item, False) for item in reversed(self.tiling)]
        while stack:
            item, done = stack.pop()
            if self._is_atom(item):
                continue
            nd = self.nodes[item]
            if not nd.alive:
                continue
            if not done:
                stack.append((item, True))
                for c in reversed(nd.children):
                    if self._child_alive(c) and self._is_node(c):
                        stack.append((c, False))
            else:
                h = 0
                for c in nd.children:
                    if self._child_alive(c):
                        ch = 0 if self._is_atom(c) else heights.get(c, 0)
                        h = max(h, ch)
                heights[item] = h + 1
        return heights

    def _merge_runs(self, force_out: bool = False) -> None:
        """常规模式：连续"低 HP"（< hp_merge_threshold）组（原子与节点皆可）。
        force_out 模式（窗外压力合并）：**只合并纯原子段**（窗外原子，HP < attention_out），
        节点不参与——原子=无意义个体需尽快成组压缩；节点=概括卡代表一段上下文，
        只走常规低 HP 合并（自然老化后折叠，树由此长高）。两者都保持位置连续分组
        （父区间 = 子区间并集的不变量），与注意力窗口的"HP 接近"语义近似
        （同批老化原子天然相邻）。"""
        for run in self._find_merge_runs(force_out):
            self._merge_run(run)

    def _find_merge_runs(self, force_out: bool = False) -> list:
        """merge 候选组预测（纯计算零 LLM）：返回低 HP 连续组列表。
        供 merge 与预蒸馏预测共用（realengine.pre_distill_launch 以相同分组
        预测下轮候选 → 后台先蒸馏 → merge 时按组内卡 id 列表命中复用）。"""
        threshold = self.attention_out if force_out else self.hp_merge_threshold
        heights = self._heights()
        t = list(self.tiling)
        runs = []
        cur = []
        anchor = None
        for item in t:
            if force_out and self._is_node(item):
                # force 模式跳过节点：纯原子段，节点是段边界
                if len(cur) >= 2:
                    runs.append(cur)
                cur = []
                anchor = None
                continue
            d = heights.get(item, 0) if self._is_node(item) else 0
            if self._item_hp(item) < threshold and (
                anchor is None or abs(d - anchor) <= self.merge_depth_diff
            ):
                if anchor is None:
                    anchor = d
                cur.append(item)
            else:
                if len(cur) >= 2:
                    runs.append(cur)
                cur = []
                anchor = None
                if self._item_hp(item) < threshold:
                    cur.append(item)
                    anchor = d
        if len(cur) >= 2:
            runs.append(cur)
        return runs

    def _item_hp(self, item) -> float:
        if self._is_atom(item):
            return self.atoms[item].hp
        return self.nodes[item].hp

    def _chunk_run(self, run: list) -> list:
        """分批：同时受 子树原子数上限 merge_cap_atoms 与 单父扇出上限 merge_fanout_cap 约束
        （防止大量节点全挂单一父亲下 → 扇出爆炸、摘要过宽）。"""
        chunks = [[]]
        total = 0
        for item in run:
            w = self._subtree_atom_count(item)
            if (total + w > self.merge_cap_atoms or len(chunks[-1]) >= self.merge_fanout_cap) \
                    and chunks[-1]:
                chunks.append([])
                total = 0
            chunks[-1].append(item)
            total += w
        merged = []
        for chunk in chunks:
            if merged and (len(chunk) == 1 or len(merged[-1]) == 1):
                merged[-1] = merged[-1] + chunk
            else:
                merged.append(chunk)
        return [c for c in merged if len(c) >= 2]

    def _merge_run(self, run: list) -> None:
        for chunk in self._chunk_run(run):
            p = self._new_node()
            p.children = list(chunk)
            for c in chunk:
                if self._is_atom(c):
                    self.atoms[c].owner = p.id
                else:
                    self.nodes[c].parent = p.id
            p.value = self._weighted_avg(chunk)
            p.created_round = self.round
            # 2026-08-29 拍定（B2）：父 HP = 保底 + 系数×组内成员加权均 HP，封顶 max_hp。
            # 保底 70 = 老内容压缩通道；组均 HP 高（新鲜内容合并）→ 父 HP 高 →
            # 沉淀更久 → 中间层涌现。使用度沿树上继承（属性测试断言）。
            # parent_hp_group_factor = 继承系数（默认 1.0，敏感性分析可扫 0.5/1.5）。
            p.hp = min(self.max_hp,
                       self.parent_init_hp + self.parent_hp_group_factor * self._group_avg_hp(chunk))
            idx = self.tiling.index(chunk[0])
            self.tiling[idx:idx + len(chunk)] = [p.id]
            self.events.append(("merge", p.id, len(chunk), self._subtree_atom_count(p.id)))

    def _group_avg_hp(self, chunk) -> float:
        """组内成员 HP 加权均值（原子按 weight 加权，节点权重 1）。"""
        tot = 0.0
        wsum = 0.0
        for c in chunk:
            if self._is_atom(c):
                hp, w = self.atoms[c].hp, self.atoms[c].weight
            else:
                hp, w = self.nodes[c].hp, 1.0
            tot += hp * w
            wsum += w
        return tot / wsum if wsum else 0.0

    # ── 组装（clean tiling，纯计算）────────────────────────

    def _emit_item(self, item, out) -> None:
        if self._is_atom(item):
            a = self.atoms[item]
            if a.alive:
                out.append(("atom", item, a.value))
            return
        nd = self.nodes[item]
        if not nd.alive:
            return
        if nd.dirty:
            for c in nd.children:
                self._emit_item(c, out)
        else:
            out.append(("node", item, nd.value))

    def _emit(self) -> list:
        out = []
        for item in self.tiling:
            self._emit_item(item, out)
        return out

    def assemble(self, shadows=None, budget=None) -> list:
        if budget is None:
            budget = self.assembly_budget
        emitted = self._emit()
        if shadows:
            emitted = self._inject_shadows(emitted, shadows)
        if len(emitted) > budget:
            self.merge_pass()
            emitted = self._emit()
            if shadows:
                emitted = self._inject_shadows(emitted, shadows)
        if len(emitted) > budget:
            emitted = self._trim(emitted, budget)
        return emitted

    def _inject_shadows(self, emitted, shadow_ids) -> list:
        emitted_index = {e[1]: i for i, e in enumerate(emitted)}
        shadows = []
        for item in shadow_ids:
            if item in emitted_index:
                continue
            if self._is_atom(item):
                if item in self.atoms and self.atoms[item].alive:
                    shadows.append(item)
            elif item in self.nodes and self.nodes[item].alive and not self.nodes[item].dirty:
                shadows.append(item)
        if not shadows:
            return emitted
        slots = []
        for item in shadows:
            first = item if self._is_atom(item) else self._first_atom(item)
            if first is None:
                continue
            idx = self._shadow_slot(emitted_index, first)
            if idx is None:
                continue
            slots.append((idx, item))
        if not slots:
            return emitted
        slots.sort(key=lambda t: t[0])
        grouped = []
        for idx, item in slots:
            if grouped and grouped[-1][0] == idx:
                grouped[-1][1].append(item)
            else:
                grouped.append([idx, [item]])
        out = list(emitted)
        for idx, group in sorted(grouped, key=lambda g: -g[0]):
            group.sort(key=functools.cmp_to_key(self._order_shadow))
            for item in reversed(group):
                if self._is_atom(item):
                    out.insert(idx + 1, ("shadow", item, self.atoms[item].value))
                else:
                    out.insert(idx + 1, ("shadow", item, self.nodes[item].value))
        return out

    def _shadow_slot(self, emitted_index, first_atom):
        chain = self._owner_chain(first_atom)
        if not chain:
            return emitted_index.get(first_atom)
        nid = chain[-1]
        while True:
            nd = self.nodes[nid]
            if nd.dirty:
                child = None
                for m in chain:
                    if self._is_node(m) and self.nodes[m].parent == nid:
                        child = m
                        break
                if child is None:
                    return emitted_index.get(first_atom)
                nid = child
                continue
            return emitted_index.get(nid)

    def _order_shadow(self, a, b) -> int:
        if a == b:
            return 0
        c1 = list(reversed(self._item_chain(a)))
        c2 = list(reversed(self._item_chain(b)))
        n = min(len(c1), len(c2))
        i = 0
        while i < n and c1[i] == c2[i]:
            i += 1
        if i == n:
            if len(c1) == len(c2):
                return 0
            return -1 if len(c1) < len(c2) else 1
        p = self.nodes[c1[i]].parent
        kids = self.nodes[p].children if p is not None else self.tiling
        i1 = kids.index(c1[i])
        i2 = kids.index(c2[i])
        return -1 if i1 < i2 else 1

    def _shadow_content(self, item):
        if self._is_atom(item):
            return self.atoms[item].value
        return self.nodes[item].value

    def render_emission(self, emitted) -> str:
        lines = []
        for kind, item, value in emitted:
            if kind == "atom":
                parent = self.atoms[item].owner
                depth = len(self._owner_chain(item))
                mark = f" ⊂N{parent}" if parent is not None else ""
                lines.append(f"{'  ' * depth}#{item} 值={value} 原子{mark}")
            elif kind == "shadow" and self._is_atom(item):
                parent = self.atoms[item].owner
                depth = len(self._owner_chain(item))
                mark = f" ⊂N{parent}" if parent is not None else ""
                lines.append(f"{'  ' * depth}#{item} 值={value} 召回原文{mark}")
            else:
                nd = self.nodes[item]
                parent = nd.parent
                depth = self._depth(item)
                tag = "召回影子" if kind == "shadow" else "摘要"
                mark = f" ⊂N{parent}" if parent is not None else ""
                lines.append(f"{'  ' * depth}N{item} 值={value} {tag}{mark}")
        return "\n".join(lines)

    def _trim(self, emitted, budget) -> list:
        node_positions = [(i, e) for i, e in enumerate(emitted) if e[0] in ("node", "shadow")]
        node_positions.sort(key=lambda ip: self._item_hp(ip[1][1]))
        drop = set()
        total = len(emitted)
        for i, _ in node_positions:
            if total <= budget:
                break
            drop.add(i)
            total -= 1
        out = [e for i, e in enumerate(emitted) if i not in drop]
        if len(out) > budget:
            self.events.append(("hard_trim", len(out) - budget))
            out = out[:budget]
        return out

    # ── 检索（数值替身）────────────────────────────────────

    def _first_atom(self, node_id: int):
        stack = [node_id]
        while stack:
            c = stack.pop(0)
            if self._is_atom(c):
                if self.atoms[c].alive:
                    return c
                continue
            nd = self.nodes[c]
            if not nd.alive:
                continue
            stack[0:0] = nd.children
        return None

    def _last_atom(self, node_id: int):
        stack = [node_id]
        while stack:
            c = stack.pop(0)
            if self._is_atom(c):
                if self.atoms[c].alive:
                    return c
                continue
            nd = self.nodes[c]
            if not nd.alive:
                continue
            stack[0:0] = reversed(nd.children)
        return None

    def _leaf_atoms(self, node_id: int) -> list:
        """节点的全部**存活后代原子**（按位置序；区间 = [first_atom, last_atom]
        沿顺序链收集，跨子树无缝隙）。"""
        lo = self._first_atom(node_id)
        hi = self._last_atom(node_id)
        if lo is None or hi is None:
            return []
        chain = self.chain_ids()
        try:
            i1, i2 = chain.index(lo), chain.index(hi)
        except ValueError:
            return []
        if i1 > i2:
            i1, i2 = i2, i1
        return [aid for aid in chain[i1:i2 + 1] if self.atoms[aid].alive]

    def _item_chain(self, item) -> list:
        if self._is_atom(item):
            return self._owner_chain(item)
        return self._chain_from(item)

    def _retrieval_pool(self, q) -> dict:
        pool = {}
        for nid, nd in self.nodes.items():
            if nd.alive and not nd.dirty and not nd.placeholder and nd.value is not None:
                pool[nid] = self._score(nd, q)
        return pool

    def retrieve(self, q, budget=None) -> list:
        if budget is None:
            budget = self.retrieval_budget
        if budget <= 0:
            return []
        pos = self._position_map()
        pool = self._retrieval_pool(q)
        result = {}
        selected = []
        while pool and len(result) < budget:
            best = max(pool, key=pool.get)
            selected.append(best)
            if self._is_atom(best):
                path = [best]
            else:
                path = []
                cur = best
                while cur is not None:
                    if self.nodes[cur].dirty:
                        break
                    path.append(cur)
                    cur = self.nodes[cur].parent
            for item in path:
                result.setdefault(item, None)
                pool.pop(item, None)
        ordered = sorted(result, key=lambda item: pos.get(self._first_atom(item), -1))
        for item in result:
            if self._is_node(item):
                self.nodes[item].hp = min(self.nodes[item].hp + self.recall_boost, self.max_hp)
            else:
                self.atoms[item].hp = min(self.atoms[item].hp + self.recall_boost, self.max_hp)
        for item in selected:
            if self._is_node(item):
                nd = self.nodes[item]
                nd.activation_count += 1
                if nd.activation_count >= self.promote_threshold:
                    self.promote(item)
        out = []
        for item in ordered:
            if self._is_node(item):
                if item in self.nodes and self.nodes[item].alive:
                    out.append(item)
            elif item in self.atoms and self.atoms[item].alive:
                out.append(item)
        return out

    def promote(self, node_id: int) -> None:
        if node_id not in self.nodes:
            return
        nd = self.nodes[node_id]
        if not nd.alive:
            return
        while nd.parent is not None:
            p = self.nodes[nd.parent]
            idx = p.children.index(node_id)
            left, right = p.children[:idx], p.children[idx + 1:]
            parent_of_p = p.parent
            left_items = self._lift_side(p, left, parent_of_p)
            right_items = self._lift_side(p, right, parent_of_p)
            nd.parent = parent_of_p
            self._replace_in_parent(p, left_items + [node_id] + right_items, parent_of_p)
            self._archive(p.id, p.version, p.value, self._first_atom(p.id), self._last_atom(p.id))
            if p.id in self.tiling:
                self.tiling.remove(p.id)
            p.alive = False
            self.events.append(("promote", node_id))

    def _lift_side(self, p, children, parent_of_p) -> list:
        alive = [c for c in children if self._child_alive(c)]
        if len(alive) >= 2:
            frag = self._new_node()
            frag.children = list(alive)
            frag.parent = parent_of_p
            for c in alive:
                if self._is_atom(c):
                    self.atoms[c].owner = frag.id
                else:
                    self.nodes[c].parent = frag.id
            frag.dirty = True
            frag.value = None
            return [frag.id]
        if len(alive) == 1:
            c = alive[0]
            if self._is_atom(c):
                self.atoms[c].owner = parent_of_p
            else:
                self.nodes[c].parent = parent_of_p
            return [c]
        return []

    def _replace_in_parent(self, p, items, parent_of_p) -> None:
        if parent_of_p is not None:
            pp = self.nodes[parent_of_p]
            i = pp.children.index(p.id)
            pp.children[i:i + 1] = items
        else:
            i = self.tiling.index(p.id)
            self.tiling[i:i + 1] = items

    def _score(self, nd, q) -> float:
        return 1.0 / (1.0 + abs(nd.value - q))

    def _verbatim_match(self, a, q) -> bool:
        return a.value == q

    def verbatim(self, q: float, budget: int = 10) -> list:
        if budget <= 0:
            return []
        pos = self._position_map()
        hits = [aid for aid, a in self.atoms.items() if a.alive and self._verbatim_match(a, q)]
        hits.sort(key=lambda aid: pos.get(aid, -1))
        return hits[:budget]

    # ── 回滚 ──────────────────────────────────────────────

    def rollback(self, steps: int = 1) -> None:
        self._rollback = True
        try:
            for _ in range(steps):
                if not self.ledger:
                    break
                op = self.ledger.pop()
                if op["type"] == "insert":
                    self.delete(op["atoms"])
                else:
                    anchors = {aid: (p, n) for aid, p, n in op["atoms"]}
                    first = op["atoms"][0][0]
                    last = op["atoms"][-1][0]
                    prev, nxt = anchors[first][0], anchors[last][1]
                    self.insert(prev, nxt, atoms=[self.atoms[aid] for aid, _, _ in op["atoms"]])
        finally:
            self._rollback = False

    # ── 测试用 oracle / 工具 ───────────────────────────────

    def oracle_value(self, nid: int) -> Optional[float]:
        nd = self.nodes[nid]
        if nd.placeholder:
            return None
        kids = [c for c in nd.children if self._child_alive(c)]
        return self._weighted_avg(kids)

    def oracle_all(self) -> dict:
        res = {}

        def visit(item):
            if self._is_atom(item):
                a = self.atoms[item]
                return (a.value, 1) if a.alive else (0.0, 0)
            nd = self.nodes[item]
            if not nd.alive:
                return (0.0, 0)
            if nd.placeholder:
                for cc in nd.children:
                    if self._child_alive(cc):
                        visit(cc)
                return (0.0, 0)
            s = 0.0
            c = 0
            for cc in nd.children:
                if not self._child_alive(cc):
                    continue
                cs, cn = visit(cc)
                s += cs
                c += cn
            res[item] = (s / c if c else None, c)
            return (s, c)

        for item in self.tiling:
            visit(item)
        return res

    def _subtree_alive_atoms(self, item) -> list:
        out = []
        stack = [item]
        while stack:
            c = stack.pop()
            if self._is_atom(c):
                if self.atoms[c].alive:
                    out.append(c)
            else:
                nd = self.nodes[c]
                if nd.alive:
                    stack.extend(reversed(nd.children))
        return out

    def emission_atoms(self, emitted) -> list:
        out = []
        for kind, item, _ in emitted:
            if kind == "atom":
                out.append(item)
            else:
                out.extend(self._subtree_alive_atoms(item))
        return out

    def stat(self) -> dict:
        max_depth = max((self._depth(nid) for nid in self.nodes if self.nodes[nid].alive), default=0)
        return {
            "round": self.round,
            "atoms": {"alive": sum(1 for a in self.atoms.values() if a.alive),
                      "total": len(self.atoms)},
            "nodes": {"alive": sum(1 for n in self.nodes.values() if n.alive),
                      "total": len(self.nodes)},
            "max_depth": max_depth,
            "dirty": sum(1 for n in self.nodes.values() if n.alive and n.dirty),
            "placeholder": sum(1 for n in self.nodes.values() if n.alive and n.placeholder),
            "tiling_size": len(self.tiling),
            "archive": len(self.archive),
            "stale_archive": sum(1 for e in self.archive if e["stale"]),
            "ledger": len(self.ledger),
        }
