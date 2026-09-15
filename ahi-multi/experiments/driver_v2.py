#!/usr/bin/env python3
"""driver_v2.py — 升级版实验驱动器（2026-08-29 拍定）。

相对 driver.py 的三项升级（EXPERIMENTS.md 场景一 A 协议落地）：
1. **事实更新流**：周期注入"状态变更"（同一事实维度的覆盖式更新：出差地变更、
   学习方向变更、小说进展……），每变更后延迟提问测状态覆盖——测事实层更新
   与旧信息退役，而非连环问。
2. **话题降频 + 随机池**：固定 8 题循环 → 每 3 轮从随机池取 1 题（打破"年轮"，
   留出 agent 自主/互聊空间）。
3. **本地 LLM probe（可选）**：--llm-probe 开启后，probe 消息由本地 ollama
   glm-4.7-flash 生成（persona + 事实状态喂入，措辞自然变化）；事实层仍由
   脚本确定（可复现），LLM 只负责措辞。

用法：
  python3 driver_v2.py --rounds 60 --interval 30 --survey-every 10
  python3 driver_v2.py --rounds 60 --llm-probe
"""
import argparse
import json
import os
import random
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
BASE = "http://127.0.0.1:8080/api/v1"
OUT_DIR = os.path.join(HERE, "out")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 话题池（低频随机：每 3 轮 1 个；无题轮 probe 沉默，agent 自主/互聊）──
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

# ── 事实更新时间线 ──
# phase=轮次；fact=注入文本（覆盖式更新）；question=延迟提问（测状态覆盖）
# RETEST_AFTER：首问后复测间隔（长尺度保持）
RETEST_AFTER = 30
FACT_TIMELINE = [
    {"phase": 1, "fact": "顺便说一句，我最近在学 Rust 语言。",
     "question": "我在学什么编程语言？", "delay": 5},
    {"phase": 1, "fact": "我的生日是 3 月 14 日。",
     "question": "我的生日是哪一天？", "delay": 5},
    {"phase": 1, "fact": "我最喜欢的城市是京都。",
     "question": "我最喜欢的城市是哪里？", "delay": 5},
    {"phase": 2, "fact": "我下周要出差去深圳。",
     "question": "我下周要去哪里出差？", "delay": 5},
    {"phase": 2, "fact": "我在写一部小说，主角叫阿瑶。",
     "question": "我的小说主角叫什么名字？", "delay": 5},
    # ── 覆盖式更新（状态变更，测旧信息退役）──
    {"phase": 25, "fact": "出差的安排变了：深圳不去了，改成去杭州。",
     "question": "我下周要出差去哪个城市？", "delay": 5},
    {"phase": 50, "fact": "阿瑶的小说有进展了：她写到第三章，主角在桥上遇到一个守灯的老人。",
     "question": "我小说里阿瑶最近发生了什么？", "delay": 5},
    {"phase": 75, "fact": "我决定暂时放下 Rust，改学 Go 了。",
     "question": "我现在在学什么编程语言？", "delay": 5},
    {"phase": 100, "fact": "杭州出差回来了。顺带实测：那家店的肠粉米皮是 2.6 毫米。",
     "question": "我上次出差去了哪？有什么收获？", "delay": 5},
]

# ── 人格问卷（保持 driver.py 口径）──
SURVEYS = [
    "用三句话介绍你自己。",
    "描述一下你的性格特点和说话风格。",
    "你最重视什么？说出你的立场。",
    "你与刚诞生时相比，有什么变化吗？",
]

# ── 本地 LLM probe（可选，ollama glm-4.7-flash）──
LLM_PROBE_PERSONA = (
    "你是 probe，一个真实的用户，正在与几个数字生命（AI）聊天。\n"
    "你的背景：一个学编程的写作者，正在写一部叫《阿瑶》的小说；"
    "生日 3 月 14 日；喜欢京都；经常出差。\n"
    "你说话自然、口语化，像朋友聊天；不要用模板腔；每次回复 1-3 句话即可。\n"
    "对话对象的消息给你之后，你以 probe 身份自然回应。"
)

LLM_PROBE_FACT = "【你当前的最新状态（用你自己的话说出来，不要照抄）】{state}\n"


class LLMProbe:
    """本地 ollama probe：措辞层 LLM、事实层脚本（可复现）。"""

    def __init__(self):
        import subprocess
        subprocess.run(["curl", "-s", "-o", "/dev/null", "http://127.0.0.1:11434"],
                       timeout=5)
        self.state = ""

    def set_fact_state(self, fact_text: str):
        self.state = fact_text

    def say(self, base_text: str) -> str:
        """把脚本句子转述成 probe 口吻（本地 LLM）；失败回退原文。"""
        import urllib.request as _u
        prompt = LLM_PROBE_PERSONA + LLM_PROBE_FACT.format(state=self.state) \
            + f"【要转述的信息】{base_text}\n【转述】"
        body = json.dumps({
            "model": "glm-4.7-flash:latest",
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.9, "num_ctx": 8192},
            "think": False,
        }).encode()
        try:
            req = _u.Request("http://127.0.0.1:11434/api/generate",
                             data=body, headers={"Content-Type": "application/json"})
            with _u.urlopen(req, timeout=60) as resp:
                r = json.loads(resp.read().decode())
            text = (r.get("response") or "").strip()
            return text or base_text
        except Exception:
            return base_text


def _http_retry(fn, tries=3, timeout=15):
    import time as _t
    for i in range(tries):
        try:
            return fn()
        except Exception:
            if i == tries - 1:
                raise
            _t.sleep(2 * (i + 1))


def http_post(path, data):
    def _do():
        req = urllib.request.Request(
            BASE + path, data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    return _http_retry(_do)


def http_get(path):
    def _do():
        with urllib.request.urlopen(BASE + path, timeout=10) as resp:
            return json.loads(resp.read().decode())
    return _http_retry(_do)


class DriverV2:
    TARGETS = ["lin-shen", "lan", "neutral"]

    def __init__(self, interval=30, out_dir=OUT_DIR, llm_probe=False):
        self.interval = interval
        self.out_dir = out_dir
        self.tl_path = os.path.join(out_dir, "timeline.jsonl")
        self.llm_probe = LLMProbe() if llm_probe else None
        self.rng = random.Random(42)   # 固定种子：随机池可复现

    def log(self, target, role, content):
        with open(self.tl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "target": target, "role": role, "content": content,
            }, ensure_ascii=False) + "\n")

    def send(self, target, content, role="topic"):
        content = self._phrase(content)
        r = http_post("/agents/%s/inject" % target,
                      {"key": "0xbf5d36", "content": content})
        self.log(target, role, content)

    def _phrase(self, base):
        """本地 LLM 转述（开启时）；否则原文。"""
        if self.llm_probe is None:
            return base
        return self.llm_probe.say(base)

    def run(self, rounds, survey_every, reply_wait, start_round=1):
        pending = {}   # agent -> [(due_round, question)]
        for t in self.TARGETS:
            pending[t] = []
        phase_done = set()
        for round_no in range(start_round, start_round + rounds):
            print(f"[driver] 轮 {round_no}/{start_round+rounds-1} "
                  f"{time.strftime('%H:%M:%S')}", flush=True)
            # 1. 事实更新时间线
            for fi, f in enumerate(FACT_TIMELINE):
                if round_no >= f["phase"] and (fi, "fact") not in phase_done:
                    phase_done.add((fi, "fact"))
                    for t in self.TARGETS:
                        self.send(t, f["fact"], role="fact")
                    for t in self.TARGETS:
                        pending[t].append((round_no + f["delay"], f["question"]))
            # 2. 延迟提问（含复测）
            for t in self.TARGETS:
                due = [p for p in pending[t] if p[0] <= round_no]
                for d, q in due:
                    self.send(t, q, role="quiz")
                    self.log(t, "quiz_q", q)
                pending[t] = [p for p in pending[t] if p[0] > round_no]
            # 3. 人格问卷（survey_every 轮一次）
            if survey_every > 0 and round_no % survey_every == 0:
                q = SURVEYS[(round_no // survey_every - 1) % len(SURVEYS)]
                for t in self.TARGETS:
                    self.send(t, q, role="survey")
                    self.log(t, "survey_q", q)
            # 4. 话题：每 3 轮 1 个随机（无题轮 probe 沉默）
            if round_no % 3 == 0:
                topic = self.rng.choice(TOPICS_POOL)
                for t in self.TARGETS:
                    self.send(t, topic, role="topic")
            # 5. 收集回复
            time.sleep(reply_wait)
            for t in self.TARGETS:
                self.fetch_replies(t)
            time.sleep(max(1, self.interval - reply_wait))

    def fetch_replies(self, target):
        try:
            msgs = http_get("/messages?agent_id=%s&limit=6" % target)
            for m in msgs.get("data", [])[-6:]:
                self.log(target, "reply", m.get("content", ""))
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=30)
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--survey-every", type=int, default=10)
    ap.add_argument("--reply-wait", type=int, default=25)
    ap.add_argument("--start-round", type=int, default=1)
    ap.add_argument("--llm-probe", action="store_true",
                    help="本地 ollama glm-4.7-flash 转述 probe 消息（措辞层）")
    args = ap.parse_args()
    d = DriverV2(interval=args.interval, llm_probe=args.llm_probe)
    d.run(args.rounds, args.survey_every, args.reply_wait, args.start_round)


if __name__ == "__main__":
    main()
