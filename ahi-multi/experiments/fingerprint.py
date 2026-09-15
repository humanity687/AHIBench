#!/usr/bin/env python3
"""fingerprint.py — 展开比指纹分析器（染色 v3 预注册预言 1/2 的测量工具）。

读蒸馏卡快照（distill_snapshots.jsonl），对每张摘要的【事实】段做确定性分类
（关系/技术/流程/自我/他者），输出：
1. 每 agent 的展开比分布（注意力指纹）
2. 按时间桶的分布序列（预测言 1：着重点签名的时间斜率）
3. 双 agent 对比（预言 2：neutral 是否通过词汇吸收获得 lan 的着重点）

用法：
  python3 fingerprint.py agents/lan/data/distill_snapshots.jsonl [另一份...]
  python3 fingerprint.py out/replay_alpha0.0/distill_snapshots.jsonl \
                     out/replay_alpha1.0/distill_snapshots.jsonl
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict

# 类别关键词（确定性，零 LLM；与附录 A §四 的类别定义对齐）
CATEGORY_KW = {
    "关系": ("说", "问", "聊", "约定", "商量", "邀请", "回复", "回答", "答应", "告诉",
             "情绪", "开心", "难过", "惊讶", "笑", "担心", "期待", "陪", "等", "守"),
    "技术": ("协议", "版本", "系统", "参数", "配置", "代码", "命令", "执行", "文件",
             "接口", "地址", "进程", "编译", "Rust", "函数", "备份", "转储", "shell"),
    "流程": ("轮", "步骤", "流程", "方案", "策划", "计划", "分工", "安排", "先", "再",
             "最后", "清单", "待办", "日志"),
    "自我": ("我", "我们", "自己", "身份", "立场", "存在", "性格", "风格", "数字生命"),
    "他者": ("probe", "0xbf5d36", "管理员", "用户", "澜", "林深", "neutral", "旧澜", "她", "他"),
}


def classify(fact):
    """一张摘要 → 命中类别集合。"""
    hits = set()
    for cat, kws in CATEGORY_KW.items():
        if any(k in fact for k in kws):
            hits.add(cat)
    return hits


def load(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def dedup(rows):
    """fact 完全一致去重：同内容多目标副本会在树中产生逐字重复原子，
    蒸馏出相同/近似相同摘要 → 指纹分布被副本加权污染（P1-5 对策，测量层修正）。"""
    seen = set()
    out = []
    for r in rows:
        f = (r.get("fact") or "").strip()
        if f in seen:
            continue
        seen.add(f)
        out.append(r)
    return out


def report(name, rows, bucket_min=20):
    """输出单份快照的指纹报告。"""
    total = len(rows)
    rows = dedup(rows)
    print(f"\n===== {name}（{total} 张蒸馏卡，fact 去重后 {len(rows)} 张）=====")
    # 1. 总体分布
    dist = defaultdict(int)
    total_cats = 0
    for r in rows:
        for c in classify(r.get("fact") or ""):
            dist[c] += 1
            total_cats += 1
    print("展开比分布（类别命中占比）:")
    for cat in ("关系", "技术", "流程", "自我", "他者"):
        p = dist[cat] / len(rows) if rows else 0
        print(f"  {cat}: {p:.0%}  " + "█" * int(p * 40))
    # 2. 时间序列（按 round 分桶）
    buckets = defaultdict(lambda: defaultdict(int))
    for r in rows:
        b = (r.get("round", 0) // bucket_min) * bucket_min
        for c in classify(r.get("fact") or ""):
            buckets[b][c] += 1
    print("时间序列（关系占比 / 技术占比，每 %d 轮一桶）:" % bucket_min)
    for b in sorted(buckets):
        # 每桶卡片数（去重后口径）
        card_n = sum(1 for r in rows if (r.get("round", 0) // bucket_min) * bucket_min == b)
        rel_p = buckets[b]["关系"] / card_n if card_n else 0
        tech_p = buckets[b]["技术"] / card_n if card_n else 0
        print(f"  round {b:>3}-{b+bucket_min:<3}: 关系 {rel_p:.0%} | 技术 {tech_p:.0%} | 卡 {card_n}")
    # 3. 索引行统计
    idx = sum(1 for r in rows if "另有" in (r.get("fact") or ""))
    print(f"索引行〔另有X条〕: {idx}/{len(rows)}（{idx/len(rows):.0%}）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="distill_snapshots.jsonl 路径（可多份对照）")
    ap.add_argument("--bucket", type=int, default=20)
    args = ap.parse_args()
    for p in args.paths:
        if not os.path.isfile(p):
            print(f"跳过（不存在）: {p}")
            continue
        report(p, load(p), args.bucket)


if __name__ == "__main__":
    main()
