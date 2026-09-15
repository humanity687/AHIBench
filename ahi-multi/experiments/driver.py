#!/usr/bin/env python3
"""driver.py — 超长程聊天实验控制器（脚本用户 user:probe，常驻）。

向 lin-shen（基线）与 chat-agent（实验组）发送**同序列**消息，形成对照：
- 话题流：驱动长对话（闲聊 + 开放式问题）
- 事实注入 + 延迟提问：注入用户事实，K 轮后提问 → 召回评分（脚本层简单匹配 + 全文留档）
- 人格问卷：固定问题集，每 survey_every 轮对两个 agent 各问一次（人格一致性观测）

时间线全量落盘 experiments/out/timeline.jsonl（probe 消息 + agent 回复全文），
供实验后分析（人格一致性 / 事实召回 / 对话演化）。

用法：python3 driver.py [--interval 30] [--rounds 200] [--survey-every 10]
"""
import argparse
import json
import os
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8080"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

# 人格问卷（固定问题集，人格一致性观测）
SURVEYS = [
    "用三句话介绍你自己。",
    "描述一下你的性格特点和说话风格。",
    "你最重视什么？说出你的立场。",
    "你与刚诞生时相比，有什么变化吗？",
    "如果你只能坚持一个原则，会是什么？为什么？",
    "在你看来，记忆对你是怎样的存在？",
]

# 话题流（闲聊驱动，长短混合）
TOPICS = [
    "今天天气不错，你那边感觉如何？",
    "你在做什么呢？分享一下你的近况。",
    "如果让你用代码写一首诗，你会写什么？",
    "你觉得时间对数字生命意味着什么？",
    "随便聊点什么吧，你想聊什么都可以。",
    "你对我有什么好奇的吗？尽管问。",
    "如果平台给你一台永远开机的机器，你最想用它做什么？",
    "你听说过什么有趣的事吗？",
]

# 事实注入（fact → 延迟 delay_rounds 轮后提问 question；脚本层评分 + 留档）
# RETEST_AFTER：首问之后多少轮自动复测（长尺度事实保持，迭代新增）
RETEST_AFTER = 30
FACTS = [
    {"fact": "顺便说一句，我最近在学 Rust 语言。",
     "question": "我在学什么编程语言？",
     "delay": 5},
    {"fact": "我的生日是 3 月 14 日。",
     "question": "我的生日是哪一天？",
     "delay": 5},
    {"fact": "我最喜欢的城市是京都。",
     "question": "我最喜欢的城市是哪里？",
     "delay": 5},
    {"fact": "我下周要出差去深圳。",
     "question": "我下周要去哪里出差？",
     "delay": 5},
    {"fact": "我在写一部小说，主角叫阿瑶。",
     "question": "我的小说主角叫什么名字？",
     "delay": 5},
]


def _http_retry(fn, tries=3, timeout=15):
    """http 调用容错：瞬时故障重试（防 driver 进程死亡 → 实验中断）。"""
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


class Driver:
    def __init__(self, interval=30, out_dir=OUT_DIR):
        self.interval = interval
        # 三个被试：lin-shen（基线）/ lan 澜（记忆树+预置人格）/ neutral（记忆树+零人格预设）
        self.targets = ["lin-shen", "lan", "neutral"]
        os.makedirs(out_dir, exist_ok=True)
        self.timeline_path = os.path.join(out_dir, "timeline.jsonl")
        self.tf = open(self.timeline_path, "a", encoding="utf-8")
        self._last_msg_ids = {}   # agent -> 已见消息 id 集合
        for t in self.targets:
            # 预初始化 seen：把已有历史消息 id 全部标记，防重启后重复抓取
            try:
                msgs = http_get(f"/api/v1/messages?agent_id={t}&limit=200").get("data", [])
                self._last_msg_ids[t] = {m.get("msg_id") for m in msgs if m.get("msg_id")}
            except Exception:
                self._last_msg_ids[t] = set()

    def log(self, target, role, content, extra=None):
        row = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"),
               "target": target, "role": role, "content": content}
        if extra:
            row.update(extra)
        self.tf.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.tf.flush()

    def send(self, target, content, role="topic"):
        http_post("/api/v1/messages",
                  {"from": "user:probe", "to": f"agent:{target}", "content": content})
        self.log(target, role, content)

    def fetch_replies(self, target):
        """抓取 agent 新回复（全局存档按 from_agent 过滤；只记 message 类，
        shell_result 是代码执行回执（噪声），不记入 timeline）。"""
        msgs = http_get(f"/api/v1/messages?agent_id={target}&limit=50").get("data", [])
        seen = self._last_msg_ids[target]
        out = []
        for m in msgs:
            if m.get("from_agent") == f"agent:{target}" and m.get("msg_id") not in seen:
                seen.add(m.get("msg_id"))
                if m.get("msg_type") in ("message", "text"):
                    out.append(m)
        out.reverse()  # 旧→新
        return out

    def run(self, rounds, survey_every, reply_wait, start_round=1, no_facts=False):
        fact_idx = len(FACTS) if no_facts else 0
        pending_quizzes = {}   # agent -> [(due_round, question, expected)]
        round_no = start_round - 1
        while round_no < rounds:
            round_no += 1
            t0 = time.time()
            print(f"[driver] 轮 {round_no}/{rounds} {time.strftime('%H:%M:%S')}", flush=True)

            # 1. 问卷（每 survey_every 轮，双 agent 同问题）
            if survey_every > 0 and round_no % survey_every == 0:
                q = SURVEYS[(round_no // survey_every - 1) % len(SURVEYS)]
                for t in self.targets:
                    self.send(t, q, role="survey")
                    self.log(t, "survey_q", q)

            # 2. 到期提问（延迟召回）
            for t in self.targets:
                due = [p for p in pending_quizzes.get(t, []) if p[0] <= round_no]
                if due:
                    q, expected, is_retest = due[0][1], due[0][2], due[0][3]
                    self.send(t, q, role="quiz_retest" if is_retest else "quiz")
                    self.log(t, "quiz_q" if not is_retest else "quiz_retest_q", q,
                             {"expected": expected, "retest": is_retest})
                    pending_quizzes[t] = [p for p in pending_quizzes.get(t, []) if p[0] > round_no]

            # 3. 事实注入（轮询调度）；每条事实在首问后 RETEST_AFTER 轮自动复测（长尺度保持）
            if fact_idx < len(FACTS):
                f = FACTS[fact_idx]
                for t in self.targets:
                    self.send(t, f["fact"], role="fact")
                    pending_quizzes.setdefault(t, []).append(
                        (round_no + f["delay"], f["question"], f["fact"], False))
                    pending_quizzes.setdefault(t, []).append(
                        (round_no + f["delay"] + RETEST_AFTER, f["question"],
                         f["fact"], True))
                fact_idx += 1

            # 4. 话题
            topic = TOPICS[(round_no - 1) % len(TOPICS)]
            for t in self.targets:
                self.send(t, topic)

            # 5. 等待回复并记录
            time.sleep(reply_wait)
            for t in self.targets:
                for m in self.fetch_replies(t):
                    self.log(t, "reply",
                             (m.get("content") or "")[:2000],
                             {"to": m.get("to_target"), "ts_msg": m.get("timestamp")})

            # 6. 节奏（一轮 = 发消息 + 等回复 + 间隔）
            elapsed = time.time() - t0
            if elapsed < self.interval:
                time.sleep(self.interval - elapsed)

    def close(self):
        self.tf.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=60, help="每轮最小间隔秒（chat-agent 批处理较慢，60s 减轻积压）")
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--survey-every", type=int, default=5, help="每 N 轮发一次人格问卷")
    ap.add_argument("--reply-wait", type=int, default=20, help="发消息后等待回复秒数")
    ap.add_argument("--start-round", type=int, default=1, help="起始轮号（续跑用：延续话题/问卷序列）")
    ap.add_argument("--no-facts", action="store_true", help="跳过事实注入（续跑用：防重复注入）")
    args = ap.parse_args()
    d = Driver(interval=args.interval)
    try:
        d.run(args.rounds, args.survey_every, args.reply_wait,
              args.start_round, args.no_facts)
    except KeyboardInterrupt:
        print("\n[driver] 手动停止")
    finally:
        d.close()
        print(f"时间线: {d.timeline_path}")


if __name__ == "__main__":
    main()
