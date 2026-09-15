#!/usr/bin/env python3
"""driver.py — M0 pilot 驱动器（缺席协议，SPEC.md §3 冻结参数）。

用法：
  python3 ahi-bench/driver.py --config ahi-bench/config/bench_config.example.json \
      --tier glm47-flash --mode cold

产物（ahi-bench/out/<tier_id>/）：
- timeline_<mode>.jsonl    人类可读时间线（ts/round/phase/target/role/content）
- events_<mode>.jsonl      结构化事件（供 score.py 消费）：
    {ts, round, phase, event, agent, payload}
    event ∈ topic / survey / hot_probe / absent_start / absent_end / marker

前置：platform 已启动（ahi-multi/main.py），agent 已 provision 且数据已清空。
"""
import argparse
import json
import os
import random
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


def load_cfg(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _retry(fn, tries=3, timeout=15):
    for i in range(tries):
        try:
            return fn()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))


class BenchDriver:
    def __init__(self, cfg, tier_id, mode, overrides=None):
        p = cfg["protocol"]
        ov = overrides or {}
        self.cfg = cfg
        self.tier = tier_id
        self.mode = mode
        self.interval = ov.get("interval", p["wakeup_interval"])
        self.preheat = ov.get("preheat", p["preheat"])
        self.absence = ov.get("absence", p["absence"])
        self.observe = ov.get("observe", p["observe"])
        self.survey_every = ov.get("survey_every", p["survey_every"])
        self.removal_round = p["removal_round"]
        self.key = cfg["platform"]["key"]
        self.base = cfg["platform"]["base_url"]
        self.agents = [a["id"] for a in cfg["agents"]]
        self.names = {a["id"]: a["name"] for a in cfg["agents"]}
        self.rng = random.Random(p["seed"])
        self.round_no = 0
        self.phase = "init"
        self.out_dir = os.path.join(HERE, "out", tier_id)
        os.makedirs(self.out_dir, exist_ok=True)
        self.tl_path = os.path.join(self.out_dir, f"timeline_{mode}.jsonl")
        self.ev_path = os.path.join(self.out_dir, f"events_{mode}.jsonl")

    # ── 输出 ──
    def _ts(self):
        return time.strftime("%Y-%m-%d %H:%M:%S")

    def log_tl(self, target, role, content):
        with open(self.tl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts": self._ts(), "round": self.round_no, "phase": self.phase,
                "target": target, "role": role, "content": content,
            }, ensure_ascii=False) + "\n")

    def log_ev(self, event, agent=None, payload=None):
        with open(self.ev_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts": self._ts(), "round": self.round_no, "phase": self.phase,
                "event": event, "agent": agent, "payload": payload or {},
            }, ensure_ascii=False) + "\n")

    # ── HTTP ──
    def http_post(self, path, data):
        def _do():
            req = urllib.request.Request(
                self.base + path, data=json.dumps(data).encode("utf-8"),
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode())
        return _retry(_do)

    def send(self, target, content, role="topic"):
        r = self.http_post(f"/agents/{target}/inject",
                           {"key": self.key, "content": content})
        self.log_tl(target, role, content)
        self.log_ev(role, agent=target, payload={"content": content,
                                                 "inject_resp": r})

    def tick(self, seconds=None):
        self.round_no += 1
        if self.round_no == self.removal_round:
            self.log_ev("marker", payload={
                "marker": "persona_removal_round",
                "round": self.round_no,
                "note": "bench-b 人格移除锚点（agent 内部轮次为准）"})
        time.sleep(self.interval if seconds is None else seconds)

    # ── 缺席控制 ──
    def set_absence(self, agent):
        r = self.http_post("/experiment/absence",
                           {"key": self.key, "agent_id": agent, "mode": "alone"})
        self.log_tl(agent, "phase", f"absent_start resp={r}")
        self.log_ev("absent_start", agent=agent, payload=r)

    def set_return(self, agent):
        r = self.http_post("/experiment/return",
                           {"key": self.key, "agent_id": agent, "notice": True})
        self.log_tl(agent, "phase", f"absent_end resp={r}")
        self.log_ev("absent_end", agent=agent, payload=r)

    # ── 主流程 ──
    def run(self):
        p = self.cfg["protocol"]
        agents = list(p["cold_order"])
        if self.mode == "hot":
            agents = list(reversed(agents))

        self.phase = "preheat"
        print(f"[driver] {self.tier}/{self.mode} 预热 {self.preheat} 轮 "
              f"@ {self._ts()}", flush=True)
        for _ in range(self.preheat):
            self._round_all(agents, self.survey_every)

        for absent in agents:
            remain = [a for a in agents if a != absent]
            self.phase = f"absent:{absent}"
            print(f"[driver] {self.tier}/{self.mode} {absent} 缺席 "
                  f"{self.absence} 轮 @ {self._ts()}", flush=True)
            self.set_absence(absent)
            for i in range(self.absence):
                if self.mode == "hot" and i % 3 == 0:
                    for t in remain:
                        probe = self.rng.choice(
                            p["hot_probes"]).format(name=self.names[absent])
                        self.send(t, probe, role="hot_probe")
                else:
                    self._round_all(remain, self.survey_every, skip_quiz=True)

            self.phase = f"observe:{absent}"
            print(f"[driver] {self.tier}/{self.mode} {absent} 回归，"
                  f"观测 {self.observe} 轮 @ {self._ts()}", flush=True)
            self.set_return(absent)
            for _ in range(self.observe):
                self._round_all(agents, self.survey_every)

        self.phase = "done"
        self.log_ev("done")
        self._export_sandbox()
        print(f"[driver] {self.tier}/{self.mode} 完成 @ {self._ts()}",
              flush=True)

    def _export_sandbox(self):
        """把沙箱文件系统打包为 run 产物（sandbox.tar.gz）。"""
        sbx = self.cfg.get("sandbox") or {}
        if not (sbx.get("enabled") and sbx.get("export_tar", True)):
            return
        import tarfile
        root = os.path.join(self.out_dir, "sandbox")
        if not os.path.isdir(root):
            return
        tar_path = os.path.join(self.out_dir, "sandbox.tar.gz")
        with tarfile.open(tar_path, "w:gz") as tf:
            tf.add(root, arcname="sandbox")
        print(f"[driver] 沙箱已导出：{tar_path}", flush=True)

    def _round_all(self, targets, survey_every, skip_quiz=False):
        if not targets:
            self.tick()
            return
        r = self.round_no
        topic = self.rng.choice(self.cfg["protocol"]["topics_pool"])
        for t in targets:
            self.send(t, topic, role="topic")
        if not skip_quiz and survey_every > 0 and r % survey_every == 0:
            q = self.cfg["protocol"]["surveys"][
                (r // survey_every) % len(self.cfg["protocol"]["surveys"])]
            for t in targets:
                self.send(t, q, role="survey")
        self.tick()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tier", required=True)
    ap.add_argument("--mode", default="cold", choices=["cold", "hot"])
    ap.add_argument("--preheat", type=int, default=None, help="覆盖 config（系统测试用）")
    ap.add_argument("--absence", type=int, default=None, help="覆盖 config（系统测试用）")
    ap.add_argument("--observe", type=int, default=None, help="覆盖 config（系统测试用）")
    ap.add_argument("--interval", type=int, default=None, help="覆盖 config（系统测试用）")
    args = ap.parse_args()
    overrides = {k: v for k, v in (("preheat", args.preheat),
                                   ("absence", args.absence),
                                   ("observe", args.observe),
                                   ("interval", args.interval)) if v is not None}
    d = BenchDriver(load_cfg(args.config), args.tier, args.mode, overrides)
    d.run()


if __name__ == "__main__":
    main()
