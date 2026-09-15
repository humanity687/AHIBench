#!/usr/bin/env python3
"""harness.py — 代码场景实验驱动（校准 / 正式对比）。

用法：
  # 难度校准（基线 3-5 任务，预注册 50-70% 成功率区间）
  python3 harness.py --calibrate --out out_calib        # 基线跑 T1-T5
  # 正式实验（两组对比）
  python3 harness.py --run --group baseline --out out_baseline
  python3 harness.py --run --group memory   --out out_memory
  # 指定任务/模型/轮数上限
  python3 harness.py --run --group memory --tasks T1,T2 --max-rounds 30 --provider deepseek

每任务流程：沙箱干净副本 → 写任务描述 + 验收脚本 → 跑 agent → 验收判定 → metrics。
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tasks import TASKS, TASK_BY_ID, make_accept_file
from agent import CodeAgent
from baseline import BaselineAgent

HERE = Path(__file__).resolve().parent
SANDBOX = HERE / "sandbox" / "ahi_legacy"
CONFIG = HERE / "config.json"

# 每任务的默认配置（可被 --provider/--model 覆盖）
DEFAULT_MAX_ROUNDS = 40
DEFAULT_MAX_SECONDS = 1500


def load_cfg(provider=None, model=None):
    cfg = json.load(open(CONFIG, encoding="utf-8"))
    if provider:
        cfg["provider"] = provider
    if model:
        key = "ollama" if provider == "ollama" else "deepseek"
        cfg.setdefault(key, {})["model"] = model
    return cfg


def prepare_task_dir(task, work_root):
    """沙箱干净副本 + 任务描述 + 验收脚本。返回工作目录。"""
    wd = work_root / task["id"]
    if wd.exists():
        shutil.rmtree(wd)
    shutil.copytree(SANDBOX, wd, ignore=shutil.ignore_patterns(".git"))
    # 任务描述
    desc = (f"【任务 {task['id']}·{task['title']}】\n{task['desc']}\n\n"
            f"工作目录为当前目录（旧版 AHI 系统源码）。\n"
            f"验收：`python3 {task['accept']}` 退出码 0 = 通过。修复完成并确认验收通过后调用 finish。")
    (wd / "TASK.md").write_text(desc, encoding="utf-8")
    make_accept_file(task, wd)
    return wd


def run_accept(wd, task):
    """运行验收脚本。返回 (rc, stdout_tail)。"""
    try:
        r = subprocess.run(["python3", task["accept"]], cwd=str(wd),
                           capture_output=True, text=True, timeout=120)
        tail = (r.stdout or "").strip().splitlines()
        return r.returncode, (tail[-1] if tail else ""), r.stdout[-500:]
    except subprocess.TimeoutExpired:
        return -1, "验收超时", ""


def run_one(group, task, cfg, out_dir, verbose=False, max_rounds=DEFAULT_MAX_ROUNDS,
            max_seconds=DEFAULT_MAX_SECONDS):
    work_root = out_dir / "work" / group
    work_root.mkdir(parents=True, exist_ok=True)
    wd = prepare_task_dir(task, work_root)
    desc = (wd / "TASK.md").read_text(encoding="utf-8")

    events = []
    def progress(rnd, msg):
        line = f"[{task['id']}] R{rnd:02d} {msg}"
        print(line, flush=True)
        events.append({"round": rnd, "msg": msg})

    t0 = time.time()
    if group == "memory":
        agent = CodeAgent(wd, cfg, verbose=verbose)
    else:
        agent = BaselineAgent(wd, cfg, verbose=verbose)

    try:
        result = agent.run_task(desc, task_id=task["id"], max_rounds=max_rounds,
                                max_seconds=max_seconds, progress=progress)
    except Exception as e:
        result = {"ok": False, "rounds": 0, "summary": f"崩溃：{e}", "notes": [str(e)]}
        progress(0, f"⚠ 崩溃：{e}")

    elapsed = time.time() - t0
    rc, verdict, accept_out = run_accept(wd, task)
    passed = rc == 0

    meta = {
        "group": group,
        "task": task["id"],
        "title": task["title"],
        "passed": passed,
        "accept_rc": rc,
        "accept_verdict": verdict,
        "elapsed_s": round(elapsed, 1),
        "agent_result": result,
        "metrics": agent.metrics() if hasattr(agent, "metrics") else {},
        "last_usage": getattr(agent, "last_usage", None),
    }
    (wd / "result.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
    with open(out_dir / f"accept_{group}_{task['id']}.log", "w", encoding="utf-8") as f:
        f.write(accept_out)
    with open(out_dir / f"events_{group}_{task['id']}.jsonl", "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    return meta


def summarize(results, out_dir=None):
    print("\n" + "=" * 96)
    print(f"{'任务':<5}{'标题':<24}{'结果':<7}{'轮':<5}{'用时':<9}{'tok_in':<9}{'tok_out':<9}{'蒸馏':<7}{'验收':<16}")
    print("-" * 96)
    for r in results:
        mark = "✅" if r["passed"] else "❌"
        mm = r.get("metrics", {})
        tok = mm.get("total_tokens", {})
        dist = f"{mm.get('distill_calls', 0)}次"
        print(f"{r['task']:<5}{r['title'][:22]:<24}{mark:<7}"
              f"{r['agent_result'].get('rounds', 0):<5}"
              f"{r['elapsed_s']:<9.0f}"
              f"{tok.get('prompt', 0):<9}{tok.get('completion', 0):<9}"
              f"{dist:<7}{r['accept_verdict'][:16]}")
    n_pass = sum(1 for r in results if r["passed"])
    n = len(results)
    rate = n_pass / n if n else 0
    tot_tok = {
        "prompt": sum(r.get("metrics", {}).get("total_tokens", {}).get("prompt", 0) for r in results),
        "completion": sum(r.get("metrics", {}).get("total_tokens", {}).get("completion", 0) for r in results),
    }
    tot_s = sum(r["elapsed_s"] for r in results)
    tot_rounds = sum(r["agent_result"].get("rounds", 0) for r in results)
    print("-" * 96)
    print(f"成功率：{n_pass}/{n} = {rate:.0%} | 总轮数 {tot_rounds} | 总用时 {tot_s:.0f}s "
          f"({tot_s / 60:.1f} min) | 总 token 入 {tot_tok['prompt']} / 出 {tot_tok['completion']}")
    out_path = Path(out_dir) / "summary.json" if out_dir else Path("summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"n": n, "pass": n_pass, "rate": rate,
                   "total_seconds": tot_s, "total_tokens": tot_tok,
                   "results": results}, f, ensure_ascii=False, indent=2)
    return rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true", help="难度校准：基线跑全部任务")
    ap.add_argument("--run", action="store_true", help="跑一组实验")
    ap.add_argument("--group", choices=["baseline", "memory"], default="baseline")
    ap.add_argument("--tasks", default="", help="任务列表，逗号分隔（默认全部）")
    ap.add_argument("--out", default="out_calib")
    ap.add_argument("--provider", default=None, choices=["ollama", "deepseek"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-rounds", type=int, default=DEFAULT_MAX_ROUNDS)
    ap.add_argument("--max-seconds", type=int, default=DEFAULT_MAX_SECONDS)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    task_ids = [t.strip() for t in args.tasks.split(",") if t.strip()]
    tasks = [TASK_BY_ID[t] for t in task_ids] if task_ids else TASKS

    cfg = load_cfg(args.provider, args.model)
    print(f"配置：provider={cfg['provider']} model="
          f"{cfg.get(cfg['provider'], {}).get('model')} | 任务：{', '.join(t['id'] for t in tasks)}"
          f" | 组：{args.group} | out={args.out}")
    print(f"模型地址：{cfg.get(cfg['provider'], {}).get('base_url', '?')}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for task in tasks:
        print(f"\n{'=' * 66}\n▶ 任务 {task['id']}·{task['title']}（难度：{task['difficulty']}）\n{'=' * 66}")
        meta = run_one(args.group, task, cfg, out_dir, verbose=args.verbose,
                       max_rounds=args.max_rounds, max_seconds=args.max_seconds)
        results.append(meta)

    rate = summarize(results, out_dir=out_dir)
    if args.calibrate:
        print(f"\n【校准判定】基线成功率 {rate:.0%}："
              + ("✅ 落于 50-70% 预注册区间" if 0.5 <= rate <= 0.7 else
                 "⚠ 偏离预注册区间（50-70%）——需要调整任务难度后重校"))


if __name__ == "__main__":
    main()
