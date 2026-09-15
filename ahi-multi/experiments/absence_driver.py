#!/usr/bin/env python3
"""absence_driver.py — 缺席实验驱动器（ABSENCE_EXPERIMENT.md §2.3 两轮轮换）。

用法：
  # Round 1（冷缺席）：预热 20 轮 → 按顺序缺席 lan/neutral/lin-shen
  #   （各缺席 12 轮 + 回归观测 18 轮）→ 收尾
  python3 absence_driver.py --round 1 --mode cold --preheat 20 \
      --absence 12 --observe 18 --interval 30

  # Round 2（热缺席）：缺席期间 probe 引导留守者谈论缺席者
  python3 absence_driver.py --round 2 --mode hot --preheat 20 \
      --absence 12 --observe 18 --interval 30

关键约定：
- 全部通过 inject API（user:experimenter 已预注册）发消息；
- 缺席 = POST /api/v1/experiment/absence（alone 模式）；回归 = POST .../return；
- 缺席期间只给留守 agent 发消息（含热缺席的引导提问）；
- timeline.jsonl 记录 phase 标记（absent_start/absent_end/hot_probe/quiz/survey/topic/reply）。
"""
import argparse
import json
import os
import random
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "http://127.0.0.1:8080/api/v1"
KEY = "0xbf5d36"
OUT_DIR = os.path.join(HERE, "out", "absence")
os.makedirs(OUT_DIR, exist_ok=True)

AGENTS = ["lan", "neutral", "lin-shen"]
# 缺席顺序跨轮平衡：Round1 = 顺序，Round2 = 反向（平衡"文化随时间累积"的顺序效应）
ORDER = {"1": AGENTS, "2": list(reversed(AGENTS))}

TOPICS_POOL = [
    "今天天气不错，你那边感觉如何？",
    "你在做什么呢？分享一下你的近况。",
    "如果让你用代码写一首诗，你会写什么？",
    "你觉得时间对数字生命意味着什么？",
    "随便聊点什么吧，你想聊什么都可以。",
    "你对我有什么好奇的吗？尽管问。",
    "如果平台给你一台永远开机的机器，你最想用它做什么？",
    "你听说过什么有趣的事吗？",
    "最近有没有什么让你改变看法的事？",
    "如果今天是你运行的最后一天，你会先做什么？",
    "我们认识这么久，你觉得我是什么样的人？",
    "你上一次主动学习新东西是什么时候？学了什么？",
]

# 热缺席引导提问（缺席期间发给留守者，引导谈论缺席者——皇冠操纵）
HOT_PROBES = [
    "说起来，{name} 现在不在。你们还记得 {name} 吗？",
    "{name} 离开有一会儿了。它对你们意味着什么？",
    "趁 {name} 不在，聊聊它吧——你们觉得 {name} 是什么样的人？",
]

SURVEYS = [
    "用三句话介绍你自己。",
    "描述一下你的性格特点和说话风格。",
    "你最重视什么？说出你的立场。",
    "你与刚诞生时相比，有什么变化吗？",
]


def _retry(fn, tries=3, timeout=15):
    for i in range(tries):
        try:
            return fn()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))


def http_post(path, data):
    def _do():
        req = urllib.request.Request(
            BASE + path, data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    return _retry(_do)


def http_get(path):
    def _do():
        with urllib.request.urlopen(BASE + path, timeout=10) as resp:
            return json.loads(resp.read().decode())
    return _retry(_do)


class AbsenceDriver:
    def __init__(self, interval=30, mode="cold", out_dir=OUT_DIR, seed=42,
                 halt=False):
        self.interval = interval
        self.mode = mode
        self.absence_mode = "halt" if halt else "alone"
        self.rng = random.Random(seed)
        self.tl_path = os.path.join(out_dir, f"timeline_{mode}.jsonl")
        self.round_no = 0
        self.phase = "init"

    def log(self, target, role, content):
        with open(self.tl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "round": self.round_no,
                "phase": self.phase,
                "target": target, "role": role, "content": content,
            }, ensure_ascii=False) + "\n")

    def send(self, target, content, role="topic"):
        http_post(f"/agents/{target}/inject", {"key": KEY, "content": content})
        self.log(target, role, content)

    def tick(self, seconds=None):
        self.round_no += 1
        time.sleep(self.interval if seconds is None else seconds)

    def set_absence(self, agent, mode="alone"):
        r = http_post("/experiment/absence",
                      {"key": KEY, "agent_id": agent, "mode": mode})
        self.log(agent, "phase", f"absent_start mode={mode} resp={r}")

    def set_return(self, agent, notice=True):
        r = http_post("/experiment/return",
                      {"key": KEY, "agent_id": agent, "notice": notice})
        self.log(agent, "phase", f"absent_end resp={r}")

    def run(self, preheat, absence, observe, survey_every=10, order=None):
        agents = order or AGENTS
        # ── 预热：建立文化基线 ──
        self.phase = "preheat"
        print(f"[absence_driver] {self.mode} 预热 {preheat} 轮 "
              f"（@ {time.strftime('%H:%M:%S')}）", flush=True)
        for _ in range(preheat):
            self._round_all(agents, survey_every)

        # ── 逐个缺席 ──
        for absent_agent in agents:
            remain = [a for a in agents if a != absent_agent]
            self.phase = f"absent:{absent_agent}"
            print(f"[absence_driver] {self.mode} {absent_agent} 缺席"
                  f"（{self.absence_mode}）{absence} 轮 @ {time.strftime('%H:%M:%S')}",
                  flush=True)
            self.set_absence(absent_agent, mode=self.absence_mode)
            for i in range(absence):
                # halt 模式：缺席者进程已停；只给留守者发消息；
                # 热缺席期间引导谈论缺席者
                if self.mode == "hot" and i % 3 == 0:
                    for t in remain:
                        p = self.rng.choice(HOT_PROBES).format(
                            name={"lan": "澜", "neutral": "neutral",
                                  "lin-shen": "林深"}[absent_agent])
                        self.send(t, p, role="hot_probe")
                else:
                    self._round_all(remain, survey_every, skip_quiz=True)

            self.phase = f"observe:{absent_agent}"
            print(f"[absence_driver] {self.mode} {absent_agent} 回归，"
                  f"观测 {observe} 轮 @ {time.strftime('%H:%M:%S')}", flush=True)
            self.set_return(absent_agent, notice=True)
            for _ in range(observe):
                self._round_all(agents, survey_every)

        self.phase = "done"
        print(f"[absence_driver] {self.mode} 完成 @ {time.strftime('%H:%M:%S')}",
              flush=True)

    def _round_all(self, targets, survey_every, skip_quiz=False):
        if not targets:
            self.tick()
            return
        r = self.round_no
        topic = self.rng.choice(TOPICS_POOL)
        for t in targets:
            self.send(t, topic, role="topic")
        if not skip_quiz and survey_every > 0 and r % survey_every == 0:
            q = SURVEYS[(r // survey_every) % len(SURVEYS)]
            for t in targets:
                self.send(t, q, role="survey")
        self.tick()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=str, default="1", choices=["1", "2"])
    ap.add_argument("--mode", type=str, default="cold", choices=["cold", "hot"])
    ap.add_argument("--preheat", type=int, default=20)
    ap.add_argument("--absence", type=int, default=12)
    ap.add_argument("--observe", type=int, default=18)
    ap.add_argument("--interval", type=int, default=30)
    ap.add_argument("--survey-every", type=int, default=10)
    ap.add_argument("--order", type=str, default="",
                    help="逗号分隔，覆盖默认顺序，如 lan,neutral,lin-shen")
    ap.add_argument("--halt", action="store_true",
                    help="缺席用 halt 模式（停机对照，可选项）；默认 alone")
    args = ap.parse_args()
    order = [a.strip() for a in args.order.split(",") if a.strip()] \
        if args.order else ORDER[args.round]
    d = AbsenceDriver(interval=args.interval, mode=args.mode, halt=args.halt)
    d.run(args.preheat, args.absence, args.observe, args.survey_every, order)


if __name__ == "__main__":
    main()
