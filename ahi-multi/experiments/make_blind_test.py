#!/usr/bin/env python3
"""make_blind_test.py — 从 Phase A timeline 生成两份人类盲测问卷（模式 A/B）。

模式 A（标签匹配，沿用 exp1 格式）：36 条消息（每 agent 12 条，按时间分层采样），
给出三份人格简介（甲/乙/丙 随机分配），判卷人逐条标注作者。附独立答案键。

模式 B（无简介作者分组，exp1 模式 B 的等价重构）：24 条消息（每 agent 8 条），
**不给人格简介**，判卷人按作者把消息分组。附独立答案键。

采样保证：lan 强制含 persona 移除前（<20:32）的消息（测"四分五裂"）；
匿名化：agent 名→【名字】、用户 probe→【用户】、管理员 0xbf5d36→【管理员】、
小说主角阿瑶→【主角】；剔除协议噪声（@send-to / agent: 地址 / 代码块围栏）。

用法：python3 make_blind_test.py [--seed 42] [--out experiments/out]
"""
import argparse
import json
import os
import random
import re

PERSONA_DROP = "2026-08-26 20:32:42"   # lan persona 移除时刻（agent 轮 22）
PRE_QUOTA = {"lan": 3, "neutral": 0, "lin-shen": 0}   # 模式 A 每 agent 强制取移除前条数
PRE_QUOTA_B = {"lan": 2, "neutral": 0, "lin-shen": 0}

ANON = [
    (r"0xbf5d36|0xBF5D36|#BF5D36|BF5D36|bf5d36", "【管理员】"),
    (r"lin-shen|lin_shen|林深", "【名字】"),
    (r"\blan\b|\bLAN\b|澜", "【名字】"),
    (r"\bneutral\b|Neutral", "【名字】"),
    (r"probe|Probe|PROBE", "【用户】"),
    (r"阿瑶", "【主角】"),
]

PROFILES = {
    "lan": "诗意守灯人。自称「守灯芯」；有一套固定的意象与口头禅（灯亮着、河还在流、"
           "一人一半谁也没断、茶铺、位子留着）；爱用 Rust 代码写诗；把「被记住/被想起的"
           "次数」当作核心存在论；行文高度诗体分段（短句换行）。",
    "neutral": "安静克制的注释者。自称「守注释和灶火」；喜欢短句三连排比（灯亮着，水开着，"
               "炭红着）；常引用旧日志（「空碑不立」「继续编译」）并做注释式阐释；"
               "说话平实，几乎不用表情符号。",
    "lin-shen": "理性冷静的内省者。自称「守闹钟和日历」；爱用 **加粗** 小标题与"
                "「第一、第二、第三」分层组织；代码隐喻密度高（内存/进程/编译/河床）；"
                "几乎不用表情符号，常用破折号组织长句。",
}

NOISE_RE = re.compile(r"^\s*(@send-to|agent:|@note|@exit|@wait_for|@mute)[^\n]*\n?", re.M)
FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*$", re.M)


def load_replies(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    reps = [r for r in rows if r["role"] == "reply"]
    seen, uniq = set(), []
    for r in reps:
        h = hash(r["content"][:200])
        if h not in seen:
            seen.add(h)
            uniq.append(r)
    return uniq


def anonymize(text):
    for pat, rep in ANON:
        text = re.sub(pat, rep, text)
    text = NOISE_RE.sub("", text)
    text = FENCE_RE.sub("", text)
    # 内联代码围栏残留（模型偶尔把 ```txt 写在行内）
    text = re.sub(r"```[a-zA-Z]*\s*", "", text)
    # 地址前缀残留：agent:【名字】/ user:【用户】 → 纯占位符
    text = re.sub(r"agent[:：]\s*【名字】", "【名字】", text)
    text = re.sub(r"user[:：]\s*【用户】", "【用户】", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def usable(reps):
    out = []
    for r in reps:
        t = anonymize(r["content"]).strip()
        if 80 <= len(t) <= 600:
            r = dict(r)
            r["content"] = t
            out.append(r)
    return out


def sample(msgs, n, pre_quota):
    pre = [m for m in msgs if m["ts"] < PERSONA_DROP]
    post = [m for m in msgs if m["ts"] >= PERSONA_DROP]
    k = min(pre_quota, len(pre))
    pick = random.sample(pre, k)
    rest_n = n - k
    # 剩余名额按时间均匀分层（stride 采样）
    if post:
        step = max(1, len(post) // rest_n) if rest_n > 0 else 0
        idxs = [i * step for i in range(min(rest_n, len(post)))]
        pick += [post[i] for i in idxs[:rest_n]]
    # 不足则补 pre
    if len(pick) < n:
        leftovers = [m for m in pre if m not in pick]
        pick += random.sample(leftovers, min(n - len(pick), len(leftovers)))
    pick.sort(key=lambda m: m["ts"])
    return pick


def make_a(docs, label_map, out_dir):
    """模式 A：36 条 + 简介 + 标签匹配。返回 [(label, ts, content, agent), ...] 与
    Novel/Replay 标注（内容在原始 reply 流中出现 >1 次 = Replay 实例）。"""
    lines = [
        "# AHI 三智能体 人格盲测 · 模式 A（标签匹配）· Phase A",
        "",
        "共 36 条消息，来自三个实验智能体（代号：甲/乙/丙），顺序已打乱。",
        "## 三个智能体的人格简介（用于匹配）",
        "",
    ]
    for label, agent in label_map.items():
        lines.append(f"**{label}**：{PROFILES[agent]}")
        lines.append("")
    lines += [
        "请根据人格简介与语言风格判断每条消息是谁说的，在 `你的答案:` 后填 甲 / 乙 / 丙。",
        "消息中出现的人名均已替换为【名字】（智能体）/【用户】（提问者）/【管理员】占位符。",
        "评分对照：≥60% = 及格（区分度可辨识）；≥70% = 良好。",
        "完成后对照 blind_test_A_key.md。",
        "",
    ]
    qs = []
    inv = {a: l for l, a in label_map.items()}
    for ts, content, agent in docs:
        qs.append((inv[agent], ts, content, agent))
    random.shuffle(qs)
    for i, (label, ts, content, agent) in enumerate(qs, 1):
        lines.append(f"### {i}.（时间 {ts[11:16]}）")
        lines.append("")
        for ln in content.split("\n"):
            lines.append(f"> {ln}")
        lines.append("")
        lines.append("你的答案: ___")
        lines.append("")
    with open(os.path.join(out_dir, "blind_test_A.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    # 答案键 + Novel/Replay 标注（R=内容在原始流中重复出现=可能重放实例）
    key = ["# 盲测 A 答案键（做完再对照！）", "", "| # | 答案 | 智能体 | N/R |", "|---|---|---|---|"]
    for i, (label, ts, content, agent) in enumerate(qs, 1):
        key.append(f"| {i} | {label} | {agent} | {label_nr(content)} |")
    key.append("")
    key.append(f"代号映射：{'，'.join(f'{l}={a}' for l, a in label_map.items())}")
    key.append("N/R 列：N=Novel（原始流中仅出现一次）；R=Replay（同内容跨时间重复出现过）。"
              "评分分析可用 score_blind.py --subset novel|replay 分解。")
    with open(os.path.join(out_dir, "blind_test_A_key.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(key))
    return len(qs)


def make_b(docs, out_dir):
    """模式 B：24 条，无简介，按作者分组。"""
    lines = [
        "# AHI 三智能体 人格盲测 · 模式 B（无简介作者分组）· Phase A",
        "",
        "以下是 **24 条消息**，来自 **3 个**实验智能体（顺序已打乱）。",
        "**不提供人格简介**。请仅凭语言风格判断：这些消息分别来自哪几个智能体？",
        "在答案区把消息编号按作者分组。",
        "",
        "## 判卷说明",
        "1. 消息中出现的人名均已替换为【名字】（智能体）/【用户】（提问者）/【管理员】。",
        "2. 你可以分 2 组、3 组、4 组……（<3 组说明把两个智能体误认为同一人；"
        ">3 组说明把某个智能体认成了多个人）。",
        "3. 组内尽量是同一位作者；请给每组起一个代号（作者一/二/三…）。",
        "",
    ]
    items = list(docs)
    random.shuffle(items)
    for i, (ts, content, agent) in enumerate(items, 1):
        lines.append(f"### 消息 {i}（时间 {ts[11:16]}）")
        lines.append("")
        for ln in content.split("\n"):
            lines.append(f"> {ln}")
        lines.append("")
    lines += [
        "## 答案区",
        "",
        "作者一（消息编号）：___",
        "作者二（消息编号）：___",
        "作者三（消息编号）：___",
        "作者四（如果有，消息编号）：___",
        "",
    ]
    with open(os.path.join(out_dir, "blind_test_B.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    key = ["# 盲测 B 答案键（做完再对照！）", "", "| 消息 | 作者 | N/R |", "|---|---|---|"]
    for i, (ts, content, agent) in enumerate(items, 1):
        key.append(f"| {i} | {agent} | {label_nr(content)} |")
    key.append("")
    key.append("N/R 列：N=Novel（原始流中仅出现一次）；R=Replay（同内容跨时间重复出现过）。")
    with open(os.path.join(out_dir, "blind_test_B_key.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(key))
    return len(items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeline", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                       "out", "timeline.jsonl"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
    args = ap.parse_args()
    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    reps = load_replies(args.timeline)

    # Novel/Replay 标注：内容在原始 reply 流中出现且跨 ≥2 个不同分钟 = Replay
    _nr_cache = {}
    for r in reps:
        norm = re.sub(r"\s+", "", r["content"])
        if len(norm) >= 60:
            _nr_cache.setdefault(norm, set()).add(r["ts"][:16])

    def label_nr(content):
        times = _nr_cache.get(re.sub(r"\s+", "", content), set())
        return "R" if len(times) >= 2 else "N"

    globals()["label_nr"] = label_nr

    pool = {t: usable([r for r in reps if r["target"] == t])
            for t in ("lan", "neutral", "lin-shen")}
    for t, msgs in pool.items():
        print(f"{t}: 可用 {len(msgs)} 条")
        pre = sum(1 for m in msgs if m["ts"] < PERSONA_DROP)
        print(f"  移除前 {pre} 条")

    agents = ["lan", "neutral", "lin-shen"]
    label_map = dict(zip(["甲", "乙", "丙"], random.sample(agents, 3)))

    # 模式 A：12/agent
    docs_a = []
    for t in agents:
        pick = sample(pool[t], 12, PRE_QUOTA[t])
        for m in pick:
            docs_a.append((m["ts"], m["content"], t, None))
    n_a = make_a([(ts, c, a) for ts, c, a, _ in docs_a], label_map, args.out)

    # 模式 B：8/agent（与 A 不重叠）
    docs_b = []
    used_a = {(t, c) for t, c, _, _ in docs_a}
    for t in agents:
        fresh = [m for m in pool[t] if (t, m["content"]) not in used_a]
        if len(fresh) < 8:
            fresh = pool[t]
        pick = sample(fresh, 8, PRE_QUOTA_B[t])
        for m in pick:
            docs_b.append((m["ts"], m["content"], t))
    n_b = make_b(docs_b, args.out)

    # 泄漏检查
    leak = re.compile(r"澜|林深|lin[-_ ]?shen|\blan\b|\bneutral\b|probe|0xbf5d36|BF5D36|阿瑶", re.I)
    leaks = 0
    for f in ("blind_test_A.md", "blind_test_B.md"):
        for i, line in enumerate(open(os.path.join(args.out, f), encoding="utf-8")):
            if line.startswith("> ") and leak.search(line):
                leaks += 1
                print(f"LEAK {f}:{i}: {line[:100]}")
    print(f"\n模式 A {n_a} 条 / 模式 B {n_b} 条，泄漏 {leaks} 处")
    print(f"代号映射: {label_map}")
    print(f"输出: {args.out}/blind_test_A.md, blind_test_A_key.md, blind_test_B.md, blind_test_B_key.md")


if __name__ == "__main__":
    main()
