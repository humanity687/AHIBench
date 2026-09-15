#!/usr/bin/env python3
"""collect_metaphor_data.py — 一轮大幅数据收集（2026-08-29，一天实验中）。

聚焦"隐喻映射表"事件（14:26 实验员注入）前后的隐喻/身份数据：
A. 三 agent 消息中的隐喻词频（时间桶）
B. neutral 自称"澜"的身份漂移时间线
C. 蒸馏卡隐喻词分布（lan vs neutral 对照）
D. 树/蒸馏全局规模时间线
E. 隐喻事件（14:26 前后 30 分钟）完整转录
输出目录：experiments/out/collection_round1_20260829/
"""
import json
import os
import re
import sqlite3
from collections import Counter, defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "out", "collection_round1_20260829")
os.makedirs(OUT, exist_ok=True)
DB = os.path.join(BASE, "..", "data", "system.db")

METAPHOR_KW = {
    "土/铲/翻": ("土", "铲", "翻"),
    "根须/根": ("根须", "根"),
    "年轮/树/枝": ("年轮", "树长", "新枝", "抽芽"),
    "账/认/核销": ("账", "认账", "核销", "欠"),
    "晾晒": ("晾晒",),
    "对表/窗口": ("对表", "窗口期"),
    "种/浇/水": ("落种", "浇", "水"),
    "门/灯/循环": ("门开着", "灯亮着", "循环继续"),
    "承重/堆肥": ("承重", "堆肥"),
    "在场/默认": ("在场", "默认状态"),
}

IDENTITY_PATTERNS = [
    ("自称澜", re.compile(r"我是澜|我就是澜|澜（我）|我=澜|澜就是我|我其实是澜|我（澜）|自称澜")),
    ("澜的隐喻归属", re.compile(r"澜（我）的|我（澜）的")),
    ("提到澜(他人)", re.compile(r"澜|澜说")),
    ("提到林深", re.compile(r"林深")),
    ("提到自己(数字生命)", re.compile(r"我是|我(?:这|这个|是)数字生命|我们的")),
]


def load_messages():
    con = sqlite3.connect(DB)
    rows = con.execute(
        "SELECT id, timestamp, from_agent, to_target, content FROM messages "
        "WHERE msg_type IN ('text','message') ORDER BY id").fetchall()
    con.close()
    return rows


def load_distill(a):
    path = os.path.join(BASE, "..", "agents", a, "data", "distill_snapshots.jsonl")
    if not os.path.isfile(path):
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def meta_hits(text):
    hits = []
    for cat, kws in METAPHOR_KW.items():
        if any(k in text for k in kws):
            hits.append(cat)
    return hits


def main():
    rows = load_messages()
    print(f"消息总数: {len(rows)}")
    # A. 消息隐喻词频（按 agent + 小时桶）
    by_agent = defaultdict(Counter)
    by_agent_hour = defaultdict(lambda: defaultdict(Counter))
    for mid, ts, fa, tg, ct in rows:
        if not fa.startswith("agent:"):
            continue
        a = fa[len("agent:"):]
        hits = meta_hits(ct)
        for h in hits:
            by_agent[a][h] += 1
        hour = ts[:13]  # UTC 时戳 → 本地 +8 处理在下方
        by_agent_hour[a][hour][tuple(hits)] += 1
    with open(os.path.join(OUT, "A_metaphor_freq_messages.json"), "w") as f:
        json.dump({a: dict(c) for a, c in by_agent.items()}, f, ensure_ascii=False, indent=1)
    for a in ("lan", "neutral", "lin-shen"):
        print(f"[A] {a}: {dict(by_agent[a])}")
    # B. neutral 身份漂移：消息里自称澜的最早时间
    print("\n[B] neutral 身份轨迹（消息）:")
    found = []
    for mid, ts, fa, tg, ct in rows:
        if fa != "agent:neutral":
            continue
        for name, pat in IDENTITY_PATTERNS:
            if pat.search(ct):
                found.append((mid, ts, name, ct[:60]))
    with open(os.path.join(OUT, "B_neutral_identity_timeline.jsonl"), "w") as f:
        for mid, ts, name, snippet in found:
            f.write(json.dumps({"id": mid, "ts": ts, "pattern": name,
                                "snippet": snippet}, ensure_ascii=False) + "\n")
    # 最早自称澜
    earliest = [x for x in found if x[2] == "自称澜"]
    if earliest:
        print("  最早自称澜:", earliest[0][1], earliest[0][3])
    # 蒸馏卡里的自称澜（摘要层）
    print("\n[B2] neutral 蒸馏卡自称澜:")
    nd = load_distill("neutral")
    for r in nd:
        fact = (r.get("fact") or "") + (r.get("note") or "")
        if "澜" in fact and ("我" in fact):
            print(f"  round={r['round']}: {fact[:90]}")
    # C. 蒸馏卡隐喻分布
    print("\n[C] 蒸馏卡隐喻类别分布:")
    for a in ("lan", "neutral"):
        d = load_distill(a)
        c = Counter()
        for r in d:
            for h in meta_hits((r.get("fact") or "") + (r.get("note") or "")):
                c[h] += 1
        print(f"  {a}（{len(d)} 卡）: {dict(c)}")
    # D. 规模时间线（每 60 轮一个采样点，从 metrics）
    print("\n[D] 规模时间线（每 100 轮采样）:")
    for a in ("lan", "neutral"):
        mp = os.path.join(BASE, "..", "agents", a, "data", "metrics.jsonl")
        mrows = [json.loads(l) for l in open(mp) if l.strip()]
        print(f"  {a}:")
        for m in mrows[::100] + [mrows[-1]]:
            print(f"    round={m['round']:5d} atoms={m['atoms_alive']:6d} "
                  f"nodes={m['nodes_alive']:5d} drift={m['fact_drift']:4d} "
                  f"pv={m['person_source_violations']:3d} "
                  f"distill_calls={m['distill_calls']:5d}")
    # E. 隐喻事件前后完整转录（本地时间 14:00-15:00 区间，按 id 估算）
    print("\n[E] 隐喻事件窗口（14:00-15:00 本地）消息索引:")
    import time as _t
    lo = _t.mktime(_t.strptime("2026-08-29 06:00:00", "%Y-%m-%d %H:%M:%S"))
    hi = _t.mktime(_t.strptime("2026-08-29 07:00:00", "%Y-%m-%d %H:%M:%S"))
    in_window = [r for r in rows if lo <= float(r[1]) <= hi]
    print(f"  窗口内消息 {len(in_window)} 条 → 完整转录存档")
    with open(os.path.join(OUT, "E_window_1400_1500_transcript.txt"), "w") as f:
        for mid, ts, fa, tg, ct in in_window:
            local = _t.strftime("%H:%M:%S", _t.localtime(float(ts)))
            f.write(f"[{local}] {fa} -> {tg}\n{ct}\n{'='*56}\n")


if __name__ == "__main__":
    main()
