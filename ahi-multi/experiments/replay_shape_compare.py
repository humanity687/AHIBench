#!/usr/bin/env python3
"""replay_shape_compare.py — 树形状对比实验（2026-08-29 A+B2 实施后第一件事）。

同一份消息流（一天实验 7.4h）回放重建树，对比参数档位：
- OLD：旧行为（节点不进窗、父 HP 恒 70、max_hp=100）
- NEW×0.5 / NEW×1.0 / NEW×1.5：A+B2（原子40/节点20 拆分、父 HP=70+系数×组均、节点出窗无悬崖）

假 LLM（零 API 成本）：结构由机制决定，与摘要内容无关。
输出：每 50 tick 形状快照 JSON + 终态对比表 → experiments/out/shape_compare/
"""
import argparse
import datetime as _dt
import json
import os
import sqlite3
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
REPO = os.path.dirname(PROJ)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "realtest"))

from realtest.realengine import RealEngine
from realtest.chattext import split_chat

SYSDB = os.path.join(PROJ, "backup_day_20260829", "data", "system.db")
OUTDIR = os.path.join(HERE, "out", "shape_compare")


class FakeLLM:
    def __init__(self):
        self.calls = 0

    def chat(self, system, user):
        self.calls += 1
        return "【事实】摘要。", {"prompt_tokens": 1, "completion_tokens": 1}, 0.001


def load_stream(target="lan"):
    conn = sqlite3.connect(SYSDB)
    conn.row_factory = sqlite3.Row
    cur = conn.execute(
        "SELECT id, from_agent, to_target, content, msg_type, timestamp FROM messages "
        "WHERE to_target=? OR from_agent=? ORDER BY id",
        (f"agent:{target}", f"agent:{target}"))
    msgs = cur.fetchall()
    conn.close()
    seen, stream = set(), []
    for m in msgs:
        key = (m["from_agent"], m["to_target"], (m["content"] or "")[:80])
        if key in seen:
            continue
        seen.add(key)
        src = "ai" if m["from_agent"] == f"agent:{target}" else m["from_agent"]
        stream.append((m["id"], src, m["content"] or "", m["timestamp"]))
    return stream


def make_engine(kind, factor=1.0):
    """kind: 'OLD' | 'NEW'；NEW 带 parent_hp_group_factor。"""
    common = dict(
        decay=8.0, hp_merge_threshold=30.0, parent_init_hp=70.0,
        hot_threshold=60.0, recall_boost=30.0,
        merge_cap_atoms=500, merge_depth_diff=1, promote_threshold=3,
        attention_out=45.0, attention_in_decay=0.5, attention_out_decay=3.0,
        merge_fanout_cap=12, pressure_threshold=2.0,
        assembly_budget=16000, retrieval_budget=6, distill_parallel=1,
    )
    if kind == "OLD":
        common.update(
            attention_atom_cap=60, attention_node_cap=0,
            parent_hp_group_factor=0.0, max_hp=100.0)
    else:
        common.update(
            attention_atom_cap=40, attention_node_cap=20,
            parent_hp_group_factor=factor, max_hp=150.0)
    return RealEngine(FakeLLM(), **common)


def shape_snapshot(eng, tick_no):
    st = eng.stat()
    nodes = {nid: nd for nid, nd in eng.nodes.items() if nd.alive}
    # 深度分布
    depth_cnt = {}
    for nid in nodes:
        d = eng._depth(nid)
        depth_cnt[d] = depth_cnt.get(d, 0) + 1
    # 铺陈构成
    atom_ids = set(eng.atoms.keys())
    tile = [x for x in eng.tiling if x in atom_ids or x in nodes]
    n_atom = sum(1 for x in tile if x in atom_ids)
    n_node = len(tile) - n_atom
    # 节点 HP 分布
    hps = [nd.hp for nd in nodes.values()]
    hp_pos = sum(1 for h in hps if h > 0)
    hp_win = sum(1 for h in hps if h >= 45)
    # 中间层（depth 1-5）
    mid = sum(v for d, v in depth_cnt.items() if 1 <= d <= 5)
    # 节点寿命（created_round 距今）
    ages = [tick_no - nd.created_round for nd in nodes.values()]
    return {
        "tick": tick_no,
        "atoms": st["atoms"]["alive"],
        "nodes": st["nodes"]["alive"],
        "tiling": st["tiling_size"],
        "tile_atoms": n_atom,
        "tile_nodes": n_node,
        "max_depth": st["max_depth"],
        "depth_dist": depth_cnt,
        "mid_nodes": mid,
        "hp_pos_ratio": hp_pos / max(1, len(hps)),
        "hp_win_count": hp_win,
        "age_median": sorted(ages)[len(ages) // 2] if ages else 0,
        "age_max": max(ages) if ages else 0,
    }


def run(kind, factor, stream, outdir):
    eng = make_engine(kind, factor)
    name = f"{kind}" + (f"_f{factor}" if kind == "NEW" else "")
    snaps = []
    n_tick = 0
    last_tick_ts = None
    last_force_tick = -99
    t0 = time.time()
    for i, (_mid, src, content, mts) in enumerate(stream):
        if not content or not content.strip():
            continue
        units = split_chat(content, source=src)
        if units:
            eng.insert_units(units, left_id=eng.tail)
        ts = _dt.datetime.strptime(mts, "%Y-%m-%d %H:%M:%S")
        if last_tick_ts is None or (ts - last_tick_ts).total_seconds() >= 30:
            eng.tick(1)
            n_tick += 1
            last_tick_ts = ts
            if eng.attention_pressure() and (n_tick - last_force_tick) >= 8:
                eng.merge_pass(force_out=True)
                last_force_tick = n_tick
            elif n_tick % 4 == 0:
                eng.merge_pass()
            if n_tick % 50 == 0:
                snaps.append(shape_snapshot(eng, n_tick))
                print(f"  [{name}] tick={n_tick} atoms={snaps[-1]['atoms']} "
                      f"nodes={snaps[-1]['nodes']} tiling={snaps[-1]['tiling']} "
                      f"mid={snaps[-1]['mid_nodes']} maxd={snaps[-1]['max_depth']} "
                      f"({time.time()-t0:.0f}s)", flush=True)
    eng.tick(1)
    eng.merge_pass()
    final = shape_snapshot(eng, n_tick + 1)
    snaps.append(final)
    with open(os.path.join(outdir, f"snaps_{name}.json"), "w") as f:
        json.dump(snaps, f, ensure_ascii=False, indent=1)
    print(f"  [{name}] 完成 | {final['atoms']} atoms / {final['nodes']} nodes / "
          f"tiling {final['tiling']} ({final['tile_atoms']}原子+{final['tile_nodes']}节点) / "
          f"max_depth {final['max_depth']} / mid {final['mid_nodes']} / "
          f"hp>0 {final['hp_pos_ratio']:.0%} / 中位寿命 {final['age_median']} tick", flush=True)
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="lan")
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    stream = load_stream(args.target)
    print(f"消息流 {len(stream)} 条（去重后）· 目标 {args.target}")
    results = {}
    results["OLD"] = run("OLD", 0, stream, OUTDIR)
    for f in (0.5, 1.0, 1.5):
        results[f"NEW_f{f}"] = run("NEW", f, stream, OUTDIR)
    print("\n===== 终态对比 =====")
    print(f"{'档位':10s} {'原子':>7s} {'节点':>6s} {'铺陈':>6s} {'maxd':>5s} "
          f"{'mid':>5s} {'hp>0':>6s} {'中位寿命':>8s}")
    for k, s in results.items():
        print(f"{k:10s} {s['atoms']:7d} {s['nodes']:6d} {s['tiling']:6d} "
              f"{s['max_depth']:5d} {s['mid_nodes']:5d} {s['hp_pos_ratio']:6.0%} "
              f"{s['age_median']:8d}")
    with open(os.path.join(OUTDIR, "summary.json"), "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"\n产物: {OUTDIR}/")


if __name__ == "__main__":
    main()
