#!/usr/bin/env python3
"""show_ctx.py — 演示：代码智能体实际上下文组成（实验组 vs 基线）。

用法: python3 show_ctx.py [--task T1] [--rounds 8] [--provider ollama]
跑真实任务若干轮，dump：
- 实验组每轮的 [WC_CTX] 记忆流槽位（组装流 + 笔记本 + 回执 + 进度）
- 基线每轮的对话消息结构（原生工具对话）
- 统计：字符数 / 原子数 / 节点数 / 铺陈项数 / 蒸馏次数
"""
import argparse
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tasks import TASK_BY_ID
from agent import CodeAgent
from baseline import BaselineAgent
from harness import prepare_task_dir, SANDBOX, HERE

C = lambda s, c: f"\033[{c}m{s}\033[0m"


def show_memory_group(task, cfg, max_rounds, show_rounds):
    wd = HERE / "work_show" / "memory" / task["id"]
    prepare_task_dir(task, HERE / "work_show" / "memory")
    desc = (wd / "TASK.md").read_text(encoding="utf-8")
    agent = CodeAgent(wd, cfg, verbose=False)
    slots = {}
    orig = agent._slot_text

    def wrapped(task_desc):
        txt = orig(task_desc)
        slots[agent.round_no] = txt
        return txt

    agent._slot_text = wrapped
    agent.run_task(desc, task_id=task["id"], max_rounds=max_rounds, max_seconds=1200)

    print("\n" + "=" * 70)
    print(C("实验组（记忆树）上下文组成", "1"))
    m = agent.eng.stat()
    print(f"统计: 原子存活 {m['atoms']['alive']} | 节点 {m['nodes']['alive']} | "
          f"铺陈项 {m['tiling_size']} | 蒸馏 {agent.eng.distill['calls']} 次 | "
          f"笔记本 {len(agent.eng.notebook)} 条")
    print("=" * 70)
    for rnd in sorted(slots):
        if rnd in show_rounds:
            txt = slots[rnd]
            print(f"\n{C(f'───── 第 {rnd} 轮 [WC_CTX] 槽位（{len(txt)} 字符）─────', '94')}")
            # 分段展示结构
            sections = txt.split("\n\n")
            for i, sec in enumerate(sections[:4]):
                print(C(f"── 段 {i + 1}（{len(sec)} 字符）", '90'))
                print(sec[:900] + ("..." if len(sec) > 900 else ""))
    return agent


def show_baseline_group(task, cfg, max_rounds):
    wd = HERE / "work_show" / "baseline" / task["id"]
    prepare_task_dir(task, HERE / "work_show" / "baseline")
    desc = (wd / "TASK.md").read_text(encoding="utf-8")
    agent = BaselineAgent(wd, cfg, verbose=False)
    agent.run_task(desc, task_id=task["id"], max_rounds=max_rounds, max_seconds=1200)
    print("\n" + "=" * 70)
    print(C("基线（无记忆）上下文组成", "1"))
    total = sum(len(m.get("content") or "") for m in agent.messages)
    print(f"统计: 消息 {len(agent.messages)} 条 | 总字符 {total} | "
          f"工具 {agent.tool_uses}")
    print("=" * 70)
    for i, m in enumerate(agent.messages):
        role = m.get("role", "?")
        if role == "system":
            nchars = len(m.get("content") or "")
            print(f"\n{C(f'[{i}] system（{nchars} 字符）', '90')}")
            print(m["content"][:200] + "...")
        elif role == "user":
            nchars = len(m.get("content") or "")
            print(f"\n{C(f'[{i}] user 任务（{nchars} 字符）', '90')}")
            print(m["content"][:150] + "...")
        elif role == "assistant":
            tc = m.get("tool_calls")
            if tc:
                print(f"\n[{i}] assistant 工具调用: {[t['function']['name'] if 'function' in t else t.get('name') for t in tc]}")
            else:
                print(f"\n[{i}] assistant 文本（{len(m.get('content') or '')} 字符）: {(m.get('content') or '')[:80]!r}")
        elif role == "tool":
            print(f"[{i}] tool 结果（{len(m.get('content') or '')} 字符）: {(m.get('content') or '')[:90]!r}")
    return agent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="T1")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--provider", default=None)
    ap.add_argument("--show-rounds", default="1,3")
    args = ap.parse_args()
    task = TASK_BY_ID[args.task]
    cfg = json.load(open(HERE / "config.json", encoding="utf-8"))
    if args.provider:
        cfg["provider"] = args.provider
    show_rounds = {int(x) for x in args.show_rounds.split(",")}
    print(f"模型: {cfg['provider']} / {cfg.get(cfg['provider'], {}).get('model')} | "
          f"任务 {task['id']}·{task['title']} | 轮数上限 {args.rounds}")
    agent = show_memory_group(task, cfg, args.rounds, show_rounds)
    # 顺带展示记忆树里的工具轨迹（持久化侧）
    print("\n" + "=" * 70)
    print(C("记忆树中的工具轨迹（source=tool 原子，跨轮持久）", "1"))
    n = 0
    for aid, a in agent.eng.atoms.items():
        if a.alive and (a.metadata or {}).get("source") == "tool":
            print(f"  [#{aid}·tool] {a.value[:70]}")
            n += 1
            if n >= 8:
                break
    print(f"  ... 共 {sum(1 for a in agent.eng.atoms.values() if a.alive and (a.metadata or {}).get('source') == 'tool')} 条 tool 原子")
    show_baseline_group(task, cfg, args.rounds)


if __name__ == "__main__":
    main()
