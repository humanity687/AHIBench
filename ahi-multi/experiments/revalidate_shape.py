#!/usr/bin/env python3
"""revalidate_shape.py — 30 分钟复验指标分析（2026-08-29 A+B2 验收）。

从 tree_snapshots.jsonl（10 轮/次）计算拍板复验指标：
1. 中间层节点数（depth 1-5）水位与波动
2. 节点寿命分布（created_round 距今，梯度分布 vs 单峰）
3. 节点段轮换率（窗口内节点身份变化：相邻快照的窗口成员交集/并集）
4. 原子周转率（铺陈原子吸收速率：相邻快照铺陈原子中"存活至下一快照"的比例）
5. 父 HP 继承分布（B2：新创建节点的 HP 分布，验证 70-100 区间 + 组均相关性）
"""
import argparse
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
REPO = os.path.dirname(PROJ)
sys.path.insert(0, REPO)


def load_snaps(agent):
    p = os.path.join(PROJ, "agents", agent, "data", "tree_snapshots.jsonl")
    if not os.path.isfile(p):
        print(f"无快照: {p}")
        sys.exit(1)
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def depth_dist(nodes):
    """节点 id → depth（到叶子的最大距离）。"""
    nodes = {str(k): v for k, v in nodes.items()}
    cache = {}

    def d(nid):
        nid = str(nid)
        if nid in cache:
            return cache[nid]
        ch = [c for c in nodes[nid].get("children", []) if str(c) in nodes]
        v = 0 if not ch else 1 + max(d(c) for c in ch)
        cache[nid] = v
        return v

    for n in nodes:
        d(n)
    return cache


def analyze(agent):
    snaps = load_snaps(agent)
    print(f"\n===== {agent}（{len(snaps)} 快照，round {snaps[0]['round']}-{snaps[-1]['round']}）=====")
    prev_tile = None
    prev_win = None
    mid_series = []
    rot_series = []
    turnover_series = []
    for s in snaps:
        nodes = s["nodes"]
        atoms = s["atoms"]
        tiling = s["tiling"]
        depths = depth_dist(nodes)
        # 1. 中间层（depth 1-5）
        mid = sum(1 for v in depths.values() if 1 <= v <= 5)
        mid_series.append(mid)
        # 2. 寿命分布（created_round 距今；快照轮次≈created 轮次差）
        ages = [s["round"] - nd.get("created_round", 0) for nd in nodes.values()]
        # 3. 节点段轮换率（窗口 = 铺陈中 HP≥45 节点）
        tile_nodes = [x for x in tiling if str(x) in nodes]
        win = {x for x in tile_nodes if nodes[str(x)].get("hp", 0) >= 45}
        if prev_win is not None and win or prev_win:
            rot_series.append(len(win ^ prev_win) / max(1, len(win | prev_win)))
        prev_win = win
        # 4. 原子周转率（铺陈原子在本快照的存活比例变化——简化：相邻快照铺陈原子交集）
        tile_atoms = {x for x in tiling if str(x) in atoms}
        if prev_tile is not None:
            inter = len(tile_atoms & prev_tile)
            turnover_series.append(1 - inter / max(1, len(prev_tile)))
        prev_tile = tile_atoms
    print(f"[1] 中间层节点数: {mid_series}")
    # 寿命分布（末快照）
    s = snaps[-1]
    ages = [s["round"] - nd.get("created_round", 0) for nd in s["nodes"].values()]
    if ages:
        ages_sorted = sorted(ages)
        n = len(ages_sorted)
        print(f"[2] 节点寿命分布(末快照): 中位 {ages_sorted[n//2]}, "
              f"p25 {ages_sorted[n//4]}, p75 {ages_sorted[3*n//4]}, max {ages_sorted[-1]}")
        bins = Counter(a // 20 * 20 for a in ages_sorted)
        print(f"    年龄桶: {dict(sorted(bins.items()))}")
    print(f"[3] 节点段轮换率序列: {[round(x,2) for x in rot_series]}")
    print(f"[4] 原子周转率序列(1-共存比例): {[round(x,2) for x in turnover_series]}")
    # 5. 父 HP 继承（新创建节点 HP 分布）
    hps = [nd.get("hp", 0) for nd in s["nodes"].values()]
    print(f"[5] 节点 HP 分布: >0 占比 {sum(1 for h in hps if h>0)/len(hps):.0%}, "
          f"均值 {sum(hps)/len(hps):.1f}, 70-100区间占比 {sum(1 for h in hps if 70<=h<=100)/len(hps):.0%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("agents", nargs="+", default=["lan", "neutral"])
    args = ap.parse_args()
    for a in args.agents:
        analyze(a)


if __name__ == "__main__":
    main()
