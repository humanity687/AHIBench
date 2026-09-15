#!/usr/bin/env python3
"""monitor.py — Phase A 实验采样器（周期快照，供实验期间人工观测）。

每次运行输出一行紧凑摘要（ts / driver 轮次 / 三 agent 实时状态 + 末行指标），
指标字段与 metrics.jsonl 对齐，便于对照 Dashboard series：
  ds    = 最近一次 merge_pass 墙钟秒（并行蒸馏效果）
  dc    = 累计蒸馏调用 / dp = 并行上限
  pending / pending_left = 分片积压
  ctx   = 本轮组装字符
  atoms / nodes / tiling = 记忆树规模
  snap  = 快照字段数（lan 应 ≥7 后稳定） / drift = fact_drift 计数
  llmc  = LLM 调用 / preU / preW = 预蒸馏复用/浪费

用法：python3 monitor.py [--loop N --every 60]（默认采样一次退出）
"""
import argparse
import json
import os
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8080"
HERE = os.path.dirname(os.path.abspath(__file__))
AGENTS = ["lin-shen", "lan", "neutral"]


def http_get(path):
    with urllib.request.urlopen(BASE + path, timeout=5) as resp:
        return json.loads(resp.read().decode())


def read_metrics(aid):
    rows = []
    try:
        with open(os.path.join(HERE, "..", "agents", aid, "data", "metrics.jsonl"),
                  encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        continue
    except FileNotFoundError:
        pass
    return rows


def driver_round():
    try:
        tl = os.path.join(HERE, "out", "timeline.jsonl")
        n = 0
        with open(tl, encoding="utf-8") as f:
            for line in f:
                n += 1
        return n
    except Exception:
        return 0


def sample():
    ts = time.strftime("%H:%M:%S")
    msgs = http_get("/api/v1/messages?limit=1").get("data", [])
    online = http_get("/api/v1/agents/online").get("data", [])
    states = {}
    for a in online:
        aid = a.get("agent_id")
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{a.get('port')}/api/state", timeout=2) as resp:
                st = json.loads(resp.read().decode()).get("data") or {}
                states[aid] = st
        except Exception:
            states[aid] = {}
    # messages_total
    conn = None
    try:
        import sqlite3
        conn = sqlite3.connect(os.path.join(HERE, "..", "data", "system.db"))
        mtotal = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        events = conn.execute("SELECT COUNT(*) FROM system_events").fetchone()[0]
    except Exception:
        mtotal = events = -1
    finally:
        if conn:
            conn.close()

    parts = [f"[{ts}] driver_rows={driver_round()} msgs={mtotal} ev={events}"]
    for aid in AGENTS:
        st = states.get(aid, {})
        rows = read_metrics(aid)
        last = rows[-1] if rows else {}
        if not last:
            parts.append(f"{aid}: idl/pend={st.get('pending_messages',0)} no-metrics")
            continue
        pend = last.get("pending", "?")
        pleft = last.get("pending_left", 0)
        ctx = last.get("ctx_chars", "?")
        dc = last.get("distill_calls", "?")
        ds = last.get("distill_seconds", 0)
        llmc = last.get("llm_calls", "?")
        snap = last.get("snapshot_fields", "?")
        drift = last.get("fact_drift", "?")
        preU = last.get("pre_distill_used", "?")
        preW = last.get("pre_distill_wasted", "?")
        atoms = last.get("atoms_alive", "?")
        nodes = last.get("nodes_alive", "?")
        tiling = last.get("tiling", "?")
        dllmc = llmc if isinstance(llmc, int) else last.get("llm_calls", 0)
        parts.append(
            f"{aid}: pend={pend}(+{pleft}) ctx={ctx} "
            f"dc={dc} ds={ds}s llmc={dllmc} "
            f"a/n/t={atoms}/{nodes}/{tiling} snap={snap} drift={drift} "
            f"pv={last.get('person_source_violations','?')} "
            f"α={last.get('style_alpha','?')} "
            f"preU/W={preU}/{preW}")
    print(" | ".join(parts), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", type=int, default=0, help="采样 N 次（0 = 一次）")
    ap.add_argument("--every", type=int, default=60, help="循环间隔秒")
    args = ap.parse_args()
    if args.loop <= 0:
        sample()
        return
    for i in range(args.loop):
        sample()
        if i < args.loop - 1:
            time.sleep(args.every)


if __name__ == "__main__":
    main()
