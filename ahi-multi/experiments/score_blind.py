#!/usr/bin/env python3
"""score_blind.py — 盲测结果评分（模式 A 标签匹配 + 模式 B 无简介分组）。

输入：repo 内答案键 + 判卷人答案文件（--dir 指定结果目录）。
输出：总体正确率 / 混淆矩阵 / 每 agent 精确率·召回率 / 逐题错位清单
      / 模式 B 组数（碎片化）与每 agent 识别率（与 exp1 口径对齐）。
"""
import argparse
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))


def load_a_key(path):
    """解析 blind_test_A_key.md → [(label, agent, nr), ...]（36 条）。"""
    rows = []
    for line in open(path, encoding="utf-8"):
        m = re.match(r"\|\s*(\d+)\s*\|\s*([甲乙丙])\s*\|\s*(\w[\w-]*)\s*\|\s*([NR])\s*\|", line)
        if m:
            rows.append((m.group(2), m.group(3), m.group(4)))
            continue
        m = re.match(r"\|\s*(\d+)\s*\|\s*([甲乙丙])\s*\|\s*(\w[\w-]*)\s*\|", line)
        if m:
            rows.append((m.group(2), m.group(3), "N"))
    return rows


def load_b_key(path):
    """解析 blind_test_B_key.md → {消息号: (agent, nr)}。"""
    out = {}
    for line in open(path, encoding="utf-8"):
        m = re.match(r"\|\s*(\d+)\s*\|\s*(\w[\w-]*)\s*\|\s*([NR])\s*\|", line)
        if m:
            out[int(m.group(1))] = (m.group(2), m.group(3))
            continue
        m = re.match(r"\|\s*(\d+)\s*\|\s*(\w[\w-]*)\s*\|", line)
        if m:
            out[int(m.group(1))] = (m.group(2), "N")
    return out


def parse_a_answers(text):
    """判卷人 A 答案 → [label, ...]（36）。支持 裸行 / 'N. 标签' 两种格式。"""
    out = []
    for line in text.splitlines():
        line = line.replace("\ufeff", "").strip()
        if not line or line.startswith("简要") or line.startswith("- "):
            continue
        m = re.match(r"^\d+\.\s*([甲乙丙])$", line)
        if m:
            out.append(m.group(1))
            continue
        if re.match(r"^[甲乙丙]$", line):
            out.append(line)
    return out


def parse_b_answers(text):
    """判卷人 B 答案 → [[消息号,...], ...]（作者组）。"""
    groups = []
    for line in text.splitlines():
        line = line.replace("\ufeff", "").strip()
        if not line or line.startswith("简要") or line.startswith("- "):
            continue
        m = re.match(r"^作者[一二三四五六七八九十\d]*", line)
        if not m:
            continue
        body = re.split(r"[:：]", line, 1)
        if len(body) < 2:
            continue
        nums = [int(x) for x in re.findall(r"\d+", body[1])]
        if nums:
            groups.append(nums)
    return groups


def score_a(key, answers, subset="all"):
    """key 元素含 (label, agent, nr)；subset ∈ all/novel/replay 过滤后评分。
    返回 (正确率, 混淆矩阵, per-agent, 错位清单, 纳入题目数)。"""
    idx = [i for i, (_, _, nr) in enumerate(key)
           if subset == "all" or (subset == "novel" and nr == "N")
           or (subset == "replay" and nr == "R")]
    n = len(idx)
    assert len(answers) >= max(idx) + 1, f"A 答案不足（实际 {len(answers)}）"
    correct = sum(1 for i in idx if answers[i] == key[i][0])
    conf = defaultdict(lambda: defaultdict(int))
    for i in idx:
        conf[answers[i]][key[i][0]] += 1
    agents = {}
    for i in idx:
        agents.setdefault(key[i][1], []).append(answers[i])
    per = {}
    for a, preds in agents.items():
        true_label = [l for i in idx if key[i][1] == a for l in [key[i][0]]][0]
        correct_a = sum(1 for p in preds if p == true_label)
        per[a] = (correct_a, len(preds))
    wrong = [(i + 1, key[i][1], answers[i], key[i][0]) for i in idx if answers[i] != key[i][0]]
    return correct / n, dict(conf), per, wrong, n


def score_b(key, groups, subset="all"):
    """分组评分：每真实 agent 找其消息最集中的组（贪心，组不重复分配）。
    key: {mid: (agent, nr)}；subset 过滤。返回 (正确率, 组数, per-agent 识别率, 明细, 纳入数)。"""
    agent_msgs = defaultdict(set)
    for mid, (a, nr) in key.items():
        if subset == "all" or (subset == "novel" and nr == "N") or (subset == "replay" and nr == "R"):
            agent_msgs[a].add(mid)
    agents = sorted(agent_msgs)
    used = set()
    assign = {}
    total_correct = 0
    for a in agents:
        best, best_score = None, -1
        for gi, g in enumerate(groups):
            if gi in used:
                continue
            s = len(agent_msgs[a] & set(g))
            if s > best_score:
                best, best_score = gi, s
        if best is not None:
            used.add(best)
            assign[a] = best
            total_correct += best_score
    recog = {a: 0.0 for a in agents}
    for a, gi in assign.items():
        recog[a] = len(agent_msgs[a] & set(groups[gi])) / len(agent_msgs[a])
    detail = {a: (gi, len(groups[gi])) for a, gi in assign.items()}
    n_used = sum(len(v) for v in agent_msgs.values())
    return total_correct / max(1, n_used), len(groups), recog, detail, n_used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/Users/y/Desktop/exp-ctx-ahi-blindII",
                    help="结果目录（含 A/ B/ 子目录）")
    ap.add_argument("--a-key", default=os.path.join(HERE, "out", "blind_test_A_key.md"))
    ap.add_argument("--b-key", default=os.path.join(HERE, "out", "blind_test_B_key.md"))
    ap.add_argument("--subset", default="all", choices=["all", "novel", "replay"],
                    help="Novel/Replay 子集分解（污染校正评分）")
    args = ap.parse_args()

    a_key = load_a_key(args.a_key)
    b_key = load_b_key(args.b_key)
    print(f"A 键 {len(a_key)} 条 / B 键 {len(b_key)} 条（subset={args.subset}）")
    print()

    judges = ["chatgpt", "deepseek", "gemini", "glm", "minimax"]

    print("=" * 70)
    print("模式 A（标签匹配）")
    print("=" * 70)
    a_results = {}
    for j in judges:
        p = os.path.join(args.dir, "A", f"{j}.txt")
        if not os.path.isfile(p):
            continue
        text = open(p, encoding="utf-8").read()
        ans = parse_a_answers(text)
        acc, conf, per, wrong, n = score_a(a_key, ans, args.subset)
        a_results[j] = (acc, wrong)
        print(f"\n[{j}] 正确率 {acc*100:.1f}%（{int(acc*n)}/{n}）")
        for a, (c, nn) in per.items():
            print(f"    {a}: {c}/{nn}")
        print(f"    错位: " + ", ".join(f"#{i}({true}→{pred})" for i, true, pred, _ in wrong))

    print()
    print("=" * 70)
    print("模式 B（无简介分组）")
    print("=" * 70)
    for j in judges:
        p = os.path.join(args.dir, "B", f"{j}.txt")
        if not os.path.isfile(p):
            continue
        text = open(p, encoding="utf-8").read()
        groups = parse_b_answers(text)
        if not groups:
            print(f"[{j}] 无法解析分组")
            continue
        acc, ngroups, recog, detail, n_used = score_b(b_key, groups, args.subset)
        print(f"\n[{j}] 分 {ngroups} 组 | 最优分配正确率 {acc*100:.1f}%（{int(acc*n_used)}/{n_used}）")
        for a in recog:
            gi, gsize = detail.get(a, (None, None))
            print(f"    {a}: 识别率 {recog[a]*100:.0f}%（组{gi}，组大小 {gsize}）")
        # 组内纯度
        agent_msgs = defaultdict(set)
        for mid, a in b_key.items():
            agent_msgs[a].add(mid)
        print("    组内容:", " ".join(
            f"G{i+1}={sorted(g)}" for i, g in enumerate(groups)))

    # 汇总对照 exp1
    print()
    print("=" * 70)
    print("汇总（对照 exp1：A 七人 44-58%/机械 63.3%；B 三人 47-61%，")
    print("neutral 识别率 17-33%、lan 被认成 2-4 人、lin-shen 83-100%）")
    print("=" * 70)


if __name__ == "__main__":
    main()
