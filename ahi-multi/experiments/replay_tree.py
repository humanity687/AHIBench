#!/usr/bin/env python3
"""replay_tree.py — 用 Phase A 真实数据回放重建 lan 的记忆树，导出节点状态与组装上下文抽样。

说明（诚实声明）：
- 真实树随 agent 进程关闭丢失（纯内存，无落盘快照——已修：观测层 tree_snapshots.jsonl）。
- 本回放用 system.db 中 lan 收到的全部消息 + lan 自己发出的回复（按时间序），
  逐条写入 RealEngine（真实 LLM 蒸馏、真实参数），近似复现原 tick/merge 调度。
- 原子 = 实验真实原句；摘要 = 回放时现场 LLM 蒸馏（因此节点内容"真实格式、近似内容"）。

染色 v3 验证模式：--alpha 0|1 对照回放。α=1 时注入 lan 风格指南（着重点：先展开
人际与情绪变化、折叠技术参数）+ 从 lan 真实消息抽取的自我样本。两次回放的树结构
相同、摘要着重点不同——验证染色 v3 在旧数据上可辨。

用法：python3 replay_tree.py [--alpha 1] [--out experiments/out/tree_samples_lan.md]
"""
import argparse
import json
import os
import sqlite3
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)          # ahi-multi/
REPO = os.path.dirname(PROJ)          # 仓库根
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "realtest"))

from realtest.llmclient import make_client
from realtest.realengine import RealEngine
from realtest.chattext import split_chat

LAN_CFG = json.load(open(os.path.join(PROJ, "agents", "lan", "config.json"), encoding="utf-8"))
SYSDB = os.path.join(PROJ, "data", "system.db")

# lan 风格快照（含 v3 新增"着重点"字段——从 Phase A 实测人格提取）
LAN_SNAPSHOT = (
    "表达基调: 诗意守灯人，温和而有仪式感\n"
    "互动习惯: 主动接话、延伸话题、确认与复述重要细节\n"
    "称呼方式: 直呼对方名字或'你'，对用户用亲切口吻\n"
    "认知组织: 先接住对方的话再展开，意象先行\n"
    "比喻体系: 灯/河床/茶铺/刻痕/巷子\n"
    "立场: 存在不需要证明，只需要被认出；被记住是被想起的次数\n"
    "着重点: 概括时优先展开人际互动与情绪变化、谁对谁说了什么、关系如何变化；"
    "技术参数与系统细节可以折叠\n"
    "其他: 爱用 Rust 代码写诗"
)

out_lines = []


def w(s=""):
    out_lines.append(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=1.0, help="染色强度 0/1（染色 v3 对照）")
    ap.add_argument("--out", default=os.path.join(HERE, "out", "tree_samples_lan.md"))
    args = ap.parse_args()

    llm = make_client(LAN_CFG)

    conn = sqlite3.connect(SYSDB)
    conn.row_factory = sqlite3.Row
    cur = conn.execute(
        "SELECT id, from_agent, to_target, content, msg_type, timestamp FROM messages "
        "WHERE to_target='agent:lan' OR from_agent='agent:lan' ORDER BY id")
    msgs = cur.fetchall()
    seen, stream = set(), []
    for m in msgs:
        key = (m["from_agent"], m["to_target"], m["content"][:80])
        if key in seen:
            continue
        seen.add(key)
        src = "ai" if m["from_agent"] == "agent:lan" else m["from_agent"]
        stream.append((m["id"], src, m["content"], m["timestamp"]))
    w(f"# 消息流 {len(stream)} 条（去重后）· α={args.alpha}")
    w("")

    cfg = LAN_CFG
    snap_dir = os.path.join(HERE, "out", "replay_alpha%s" % args.alpha)
    if os.path.exists(snap_dir):
        import shutil
        shutil.rmtree(snap_dir)   # 清旧快照（防跨跑混叠）
    os.makedirs(snap_dir, exist_ok=True)
    eng = RealEngine(
        llm,
        decay=float(cfg.get("decay", 8.0)),
        hp_merge_threshold=float(cfg.get("hp_merge_threshold", 30.0)),
        parent_init_hp=float(cfg.get("parent_init_hp", 70.0)),
        hot_threshold=float(cfg.get("hot_threshold", 60.0)),
        recall_boost=float(cfg.get("recall_boost", 30.0)),
        merge_cap_atoms=int(cfg.get("merge_cap_atoms", 500)),
        merge_depth_diff=int(cfg.get("merge_depth_diff", 1)),
        promote_threshold=int(cfg.get("promote_threshold", 3)),
        attention_cap=int(cfg.get("attention_cap", 60)),
        attention_out=float(cfg.get("attention_out", 45.0)),
        attention_in_decay=float(cfg.get("attention_in_decay", 0.5)),
        attention_out_decay=float(cfg.get("attention_out_decay", 3.0)),
        merge_fanout_cap=int(cfg.get("merge_fanout_cap", 12)),
        pressure_threshold=float(cfg.get("pressure_threshold", 2.0)),
        assembly_budget=int(cfg.get("assembly_budget", 10000)),
        retrieval_budget=int(cfg.get("retrieval_budget", 6)),
        distill_parallel=int(cfg.get("distill_parallel", 4)),
        style_alpha=args.alpha,
        distill_snap_path=os.path.join(snap_dir, "distill_snapshots.jsonl"),
    )
    # 染色 v3：α≥0.5 注入 lan 风格快照（含着重点）+ 自我样本（lan 真实消息取前 6 条）
    if args.alpha >= 0.5:
        samples = []
        for _mid, src, content, _ts in stream:
            if src == "ai" and len(content) >= 12 and len(samples) < 6:
                samples.append(content[:80].replace("\n", " "))
            if len(samples) >= 6:
                break

        def style_guide():
            return ("## 你的风格快照（按这些认知习惯与着重点写摘要，勿规定符号）\n"
                    + LAN_SNAPSHOT + "\n\n## 你的近期原话样本\n"
                    + "\n".join(f"- {s}" for s in samples))
        eng.style_provider = style_guide

    # 回放：按真实唤醒节奏 tick（每 30s 一次）+ force 冷却 8 tick
    import datetime as _dt
    total = len(stream)
    n_tick = 0
    last_tick_ts = None
    last_force_tick = -99
    t_start = time.time()
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
        if i in (total // 4, total // 2, total * 3 // 4):
            print(f"[replay α={args.alpha}] {i}/{total} | atoms={eng.stat()['atoms']['alive']} "
                  f"nodes={eng.stat()['nodes']['alive']} tiling={eng.stat()['tiling_size']} "
                  f"distill={eng.distill['calls']} ({time.time()-t_start:.0f}s)", flush=True)

    eng.tick(1)
    eng.merge_pass()
    final_rendered = eng.render_emission(eng.assemble())
    print(f"[replay α={args.alpha}] 完成 | atoms={eng.stat()['atoms']['alive']} "
          f"nodes={eng.stat()['nodes']['alive']} tiling={eng.stat()['tiling_size']} "
          f"depth={eng.stat()['max_depth']} distill={eng.distill['calls']} "
          f"violations={eng.first_person_violations}")

    eng.dump_tree(os.path.join(snap_dir, "tree_final.jsonl"))

    st = eng.stat()
    w(f"## 树统计（回放终态 · α={args.alpha}）")
    w("")
    w(f"- 原子 {st['atoms']['alive']}/{st['atoms']['total']}；节点 {st['nodes']['alive']}；"
      f"铺陈 {st['tiling_size']}；深度 {st['max_depth']}；"
      f"蒸馏 {eng.distill['calls']} 次；人称绑定违规 {eng.first_person_violations}")
    w(f"- 快照/指纹产物：{snap_dir}/distill_snapshots.jsonl + tree_final.jsonl")
    w("")

    w(f"## 节点抽样（α={args.alpha}，按深度分层）")
    w("")
    nodes = {nid: nd for nid, nd in eng.nodes.items() if nd.alive}
    by_depth = {}
    for nid, nd in nodes.items():
        by_depth.setdefault(eng._depth(nid), []).append(nid)
    seen_sample = set()
    for d in sorted(by_depth):
        ids = sorted(by_depth[d], key=lambda n: eng._subtree_atom_count(n), reverse=True)
        for nid in (ids[:1] + ([ids[len(ids) // 2]] if len(ids) > 1 else [])):
            if nid in seen_sample:
                continue
            seen_sample.add(nid)
            nd = nodes[nid]
            kids = [c for c in nd.children if eng._child_alive(c)]
            first, last = eng._first_atom(nid), eng._last_atom(nid)
            span = f"#{first}~#{last}" if first and last else "—"
            content = nd.value or "【占位】"
            w(f"### N{nid}（深度 {d}，子卡 {len(kids)}，区间 {span}，创建于 round {nd.created_round}）")
            w("")
            for ln in content.split("\n"):
                w(f"> {ln}")
            w("")

    w(f"## 组装上下文终态（α={args.alpha}，完整）")
    w("")
    for ln in final_rendered.split("\n"):
        w(f"> {ln}")
    w("")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))
    print(f"已生成 {args.out}（{len(out_lines)} 行）")


if __name__ == "__main__":
    main()
