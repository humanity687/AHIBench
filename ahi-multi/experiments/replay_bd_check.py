#!/usr/bin/env python3
"""replay_bd_check.py — B+D 时序分流验证（2026-08-30）。

重放 1h v2 实验 lan 的消息流（真 LLM 蒸馏），检查事实变更（Rust→Go、
深圳→杭州）注入后的蒸馏卡叙述：旧值应以"历史态"（曾学/改学）出现，
不得以"当前态"（正在学/下周去）复述。
"""
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

from realtest.llmclient import make_client
from realtest.realengine import RealEngine
from realtest.chattext import split_chat

SYSDB = os.path.join(PROJ, "backup_v2_1h_20260830", "data", "system.db")
CFG = json.load(open(os.path.join(PROJ, "agents", "lan", "config.json"), encoding="utf-8"))
OUT = os.path.join(HERE, "out", "bd_check")
os.makedirs(OUT, exist_ok=True)


def main():
    conn = sqlite3.connect(SYSDB)
    conn.row_factory = sqlite3.Row
    msgs = conn.execute(
        "SELECT id, from_agent, to_target, content, msg_type, timestamp FROM messages "
        "WHERE to_target='agent:lan' OR from_agent='agent:lan' ORDER BY id").fetchall()
    conn.close()
    seen, stream = set(), []
    for m in msgs:
        key = (m["from_agent"], m["to_target"], (m["content"] or "")[:80])
        if key in seen:
            continue
        seen.add(key)
        src = "ai" if m["from_agent"] == "agent:lan" else m["from_agent"]
        stream.append((m["id"], src, m["content"] or "", m["timestamp"]))
    print(f"消息流 {len(stream)} 条")

    llm = make_client(CFG)
    eng = RealEngine(
        llm,
        decay=float(CFG.get("decay", 8.0)),
        hp_merge_threshold=float(CFG.get("hp_merge_threshold", 30.0)),
        parent_init_hp=float(CFG.get("parent_init_hp", 70.0)),
        hot_threshold=float(CFG.get("hot_threshold", 60.0)),
        recall_boost=float(CFG.get("recall_boost", 30.0)),
        merge_cap_atoms=int(CFG.get("merge_cap_atoms", 500)),
        merge_depth_diff=int(CFG.get("merge_depth_diff", 1)),
        promote_threshold=int(CFG.get("promote_threshold", 3)),
        attention_atom_cap=int(CFG.get("attention_atom_cap", 40)),
        attention_node_cap=int(CFG.get("attention_node_cap", 20)),
        attention_out=float(CFG.get("attention_out", 45.0)),
        attention_in_decay=float(CFG.get("attention_in_decay", 0.5)),
        attention_out_decay=float(CFG.get("attention_out_decay", 3.0)),
        merge_fanout_cap=int(CFG.get("merge_fanout_cap", 12)),
        pressure_threshold=float(CFG.get("pressure_threshold", 2.0)),
        assembly_budget=int(CFG.get("assembly_budget", 16000)),
        retrieval_budget=int(CFG.get("retrieval_budget", 6)),
        distill_parallel=4,
        style_alpha=1.0,
        max_hp=150.0,
        distill_snap_path=os.path.join(OUT, "distill_snapshots.jsonl"),
    )
    n_tick = 0
    last_ts = None
    last_force = -99
    t0 = time.time()
    for _mid, src, content, mts in stream:
        if not content.strip():
            continue
        units = split_chat(content, source=src)
        if units:
            eng.insert_units(units, left_id=eng.tail)
        ts = _dt.datetime.strptime(mts, "%Y-%m-%d %H:%M:%S")
        if last_ts is None or (ts - last_ts).total_seconds() >= 30:
            eng.tick(1)
            n_tick += 1
            last_ts = ts
            if eng.attention_pressure() and (n_tick - last_force) >= 8:
                eng.merge_pass(force_out=True)
                last_force = n_tick
            elif n_tick % 4 == 0:
                eng.merge_pass()
    eng.tick(1)
    eng.merge_pass()
    print(f"重放完成 {time.time()-t0:.0f}s | 蒸馏 {eng.distill['calls']} 次")

    # 验证：蒸馏快照中含 Rust/深圳 的卡，检查叙述态
    snaps = [json.loads(l) for l in open(os.path.join(OUT, "distill_snapshots.jsonl"))
             if l.strip()]
    print(f"\n蒸馏卡 {len(snaps)} 张。检查旧值叙述态：")
    rust_current = rust_history = 0
    sz_current = sz_history = 0
    rust_samples = []
    for s in snaps:
        fact = (s.get("fact") or "")
        if "Rust" in fact or "rust" in fact:
            if "曾学" in fact or "改学" in fact or "放下" in fact or "现学" in fact:
                rust_history += 1
            else:
                rust_current += 1
                if len(rust_samples) < 3:
                    rust_samples.append(fact[:120])
        if "深圳" in fact:
            if "改成杭州" in fact or "改去杭州" in fact or "曾" in fact:
                sz_history += 1
            else:
                sz_current += 1
    print(f"Rust: 历史态 {rust_history} / 当前态(疑似) {rust_current}")
    for x in rust_samples:
        print(f"  [当前态样例] {x}")
    print(f"深圳: 历史态 {sz_history} / 当前态(疑似) {sz_current}")
    # 变更注入时间之后才应出现历史态——统计变更后（ts>03:17 UTC）的卡
    cutoff = 1788063459  # 03:17:39 UTC 近似
    late = [s for s in snaps if s.get("ts", 0) > cutoff]
    late_rust = [s for s in late if "Rust" in (s.get("fact") or "")]
    print(f"\n变更后蒸馏卡 {len(late)} 张，其中含 Rust {len(late_rust)} 张：")
    for s in late_rust[:6]:
        print(f"  r{s['round']} {s['ts']:.0f}: {(s.get('fact') or '')[:130]}")


if __name__ == "__main__":
    main()
