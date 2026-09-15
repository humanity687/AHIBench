#!/usr/bin/env python3
"""monitor_absence.py — 缺席实验监测循环（每 interval 秒采样一次）。

监测点：
1. 驱动器进程存活（挂掉即告警）
2. 阶段标记（preheat/absent:xxx/observe:xxx）与轮数
3. 缺席状态 API（当前被隔离的 agent）
4. 三 agent 在线状态
5. 各 agent 树/组装快照是否在增长（观测层活着）
输出：追加到 experiments/out/absence/monitor.jsonl + stdout
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "http://127.0.0.1:8080/api/v1"
KEY = "0xbf5d36"
TL = os.path.join(HERE, "out", "absence",
                  sys.argv[2] if len(sys.argv) > 2 else "timeline_cold.jsonl")
MON = os.path.join(HERE, "out", "absence",
                   "monitor_" + ("hot" if "hot" in TL else "cold") + ".jsonl")
AGENTS = ["lan", "neutral", "lin-shen"]
EXPECTED_PHASES = 3 * 2  # 3 次缺席 + 3 次回归 = 6 个 phase 标记


def _http_get(path):
    try:
        with urllib.request.urlopen(BASE + path, timeout=5) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        return {"error": str(e)}


def _driver_alive():
    out = subprocess.run(["pgrep", "-f", "absence_driver.py"],
                         capture_output=True, text=True)
    return bool(out.stdout.strip())


def _count(path):
    try:
        with open(path, encoding="utf-8") as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0


def _snap_growth():
    """各 agent 快照文件行数（观测层活性）。"""
    g = {}
    for a in AGENTS:
        p = os.path.join(HERE, "agents", a, "data", "tree_snapshots.jsonl")
        g[a] = _count(p)
    return g


def _phase_state():
    """从 timeline 提取最新阶段与 phase 标记数。"""
    phases = []
    last_round = 0
    try:
        with open(TL, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if d.get("role") == "phase":
                    phases.append(d.get("content", ""))
                last_round = d.get("round", last_round)
    except FileNotFoundError:
        pass
    return {"round": last_round, "phase_markers": len(phases),
            "last_phase": phases[-1] if phases else "preheat"}


def main():
    interval = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    print(f"[monitor_absence] 每 {interval}s 采样一次，Ctrl-C 停止", flush=True)
    while True:
        st = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "driver_alive": _driver_alive(),
            "phase": _phase_state(),
            "absence": _http_get(f"/experiment/absence?key={KEY}").get("data", {}),
            "agents": {a: (a in {x.get("agent_id")
                                for x in _http_get("/agents/online").get("data", [])})
                       for a in AGENTS},
            "snap_rows": _snap_growth(),
        }
        with open(MON, "a", encoding="utf-8") as f:
            f.write(json.dumps(st, ensure_ascii=False) + "\n")
        flag = "⚠ " if not st["driver_alive"] else "  "
        print(f"{flag}{st['ts']} round={st['phase']['round']} "
              f"phase={st['phase']['last_phase']} "
              f"markers={st['phase']['phase_markers']}/{EXPECTED_PHASES} "
              f"absent={list(st['absence'].keys())} "
              f"online={[k for k, v in st['agents'].items() if v]}", flush=True)
        time.sleep(interval)


if __name__ == "__main__":
    main()
