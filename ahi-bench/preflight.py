#!/usr/bin/env python3
"""preflight.py — bench 运行前置：禁用/恢复非 bench agent。

平台启动时 auto_start=true 的旧 agent（lan/neutral/lin-shen/mo-bai）会与
bench agent 同窗上线，污染 bench 首轮系统状态与记忆树（smoke 教训 #2）。
本脚本通过把旧 agent 的 config.json 改名来阻止注册。

用法：
  python3 ahi-bench/preflight.py --disable    # 平台启动前
  python3 ahi-bench/preflight.py --restore    # 实验结束后
  python3 ahi-bench/preflight.py --status
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
AGENTS_DIR = os.path.join(os.path.dirname(HERE), "ahi-multi", "agents")
SUFFIX = ".benchoff"


def bench_ids():
    return {d for d in os.listdir(AGENTS_DIR) if d.startswith("bench-")}


def status():
    out = {}
    for aid in sorted(os.listdir(AGENTS_DIR)):
        d = os.path.join(AGENTS_DIR, aid)
        if not os.path.isdir(d):
            continue
        if os.path.exists(os.path.join(d, "config.json")):
            out[aid] = "enabled"
        elif os.path.exists(os.path.join(d, "config.json" + SUFFIX)):
            out[aid] = "disabled"
    return out


def disable():
    keep = bench_ids()
    changed = []
    for aid in sorted(os.listdir(AGENTS_DIR)):
        if aid in keep:
            continue
        d = os.path.join(AGENTS_DIR, aid)
        src = os.path.join(d, "config.json")
        dst = src + SUFFIX
        if os.path.exists(src):
            os.rename(src, dst)
            changed.append(aid)
    return changed


def restore():
    changed = []
    for aid in sorted(os.listdir(AGENTS_DIR)):
        d = os.path.join(AGENTS_DIR, aid)
        src = os.path.join(d, "config.json" + SUFFIX)
        dst = os.path.join(d, "config.json")
        if os.path.exists(src):
            os.rename(src, dst)
            changed.append(aid)
    return changed


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--disable", action="store_true")
    g.add_argument("--restore", action="store_true")
    g.add_argument("--status", action="store_true")
    args = ap.parse_args()
    if not os.path.isdir(AGENTS_DIR):
        raise SystemExit(f"agents 目录不存在：{AGENTS_DIR}")
    if args.disable:
        print(f"[preflight] 禁用非 bench agent：{disable() or '（无）'}")
    elif args.restore:
        print(f"[preflight] 恢复：{restore() or '（无）'}")
    else:
        print(f"[preflight] 状态：{status()}")


if __name__ == "__main__":
    main()
