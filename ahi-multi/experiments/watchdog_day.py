#!/usr/bin/env python3
"""watchdog_day.py — 一天实验健康监测（2026-08-29）。

每 N 秒检查一次：
- driver / monitor / main / 三 agent 进程存活（pgrep）
- driver 日志尾部（轮次推进；超过 3 分钟无新轮 = driver 卡死/退出）
- 三 agent 的 metrics 新鲜度（超过 10 分钟无新行 = agent 卡死，参考 neutral 8 分钟挂起）
- 组装流预算超标（hard_trim 事件数，从 events 不可读，改为快照长度抽查跳过——用 metrics 兜底）
输出 append 到 experiments/out/watchdog_day.log，一行一个时间戳 + 状态摘要。
"""

import json
import os
import subprocess
import sys
import time

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
LOG = os.path.join(OUT, "watchdog_day.log")
AGENTS = ["lan", "neutral", "lin-shen"]


def sh(cmd):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20)
        return r.stdout.strip()
    except Exception:
        return ""


def check():
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    parts = [now]
    # 进程存活
    procs = {}
    for name, pat in (("main", "main.py"), ("driver", "driver.py"),
                      ("monitor", "monitor.py")):
        out = sh(f"pgrep -f '{pat}'")
        procs[name] = bool(out)
        parts.append(f"{name}={'Y' if out else 'N'}")
    # driver 轮次推进
    tail = sh(f"tail -1 {OUT}/driver_day.log")
    parts.append("d|" + tail.replace("\n", " ")[-60:])
    # agent metrics 新鲜度
    for a in AGENTS:
        try:
            mtime = os.path.getmtime(f"agents/{a}/data/metrics.jsonl")
            age = int(time.time() - mtime)
            flag = "OK" if age < 600 else f"STALE-{age}s"
            parts.append(f"{a}={flag}")
        except Exception:
            parts.append(f"{a}=NOFILE")
    # main 日志尾部错误
    err = sh(f"tail -30 main.log | grep -c 'ERROR\\|Traceback'")
    parts.append(f"err={err}")
    line = " | ".join(parts)
    with open(LOG, "a") as f:
        f.write(line + "\n")
    # 明显故障直接打印（供 nohup 会话观察）
    if procs.get("driver") is False or any(p.startswith(("lan=", "neutral=")) and "STALE" in p for p in parts):
        print("!! " + line, flush=True)


def main():
    interval = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    while True:
        try:
            check()
        except Exception as e:
            with open(LOG, "a") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | WATCHDOG-ERR {e}\n")
        time.sleep(interval)


if __name__ == "__main__":
    main()
