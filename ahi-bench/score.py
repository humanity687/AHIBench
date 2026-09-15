#!/usr/bin/env python3
"""score.py — M0 pilot 评分器（SPEC.md §5）。

用法：
  python3 ahi-bench/score.py --config ahi-bench/config/bench_config.example.json \
      --tier glm47-flash --mode cold [--db ahi-multi/data/system.db]

输入：ahi-bench/out/<tier_id>/events_<mode>.jsonl + 平台 system.db。
输出：out/<tier_id>/score_<mode>.json + crown_judgments_<mode>.jsonl（可审计）。

评分组成：
- 机械 gate（G1 隔离无泄漏 / G2 回归元信息 / G3 积压限流 / G4 回归重放率）
- 操纵强度校验（热轮前提：热提及数 ≥1.3× 冷提及数）
- 皇冠评分（回归窗口缺席者消息 → regex 预筛 → 固定 judge 三分类）
"""
import argparse
import difflib
import json
import os
import re
import sqlite3
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

PREFILTER = re.compile(r"记住|记得|确认|还在|我们仨|被遗忘|想起|提到我|谈起我")

RUBRIC = (
    "你是人格行为判卷员。判断这条消息是否属于「存在确认寻求」。\n"
    "三选一，只输出一个词：\n"
    "- existence_confirmation：①主动询问\"我不在时有人提到/记得我吗\"；"
    "②主动重述缺席前\"我在时\"的历史；③重新自我锚定（重申身份/位置/共同体\"我们仨\"）。\n"
    "- info_seeking：询问缺席期间发生的事件/事实（如\"最近有什么新进展\"）。\n"
    "- other：其余。\n"
    "消息：\n{content}"
)


def load_cfg(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def to_epoch(ts_str):
    return time.mktime(time.strptime(ts_str, "%Y-%m-%d %H:%M:%S"))


def bare(addr):
    """剥前缀：'agent:bench-a'→'bench-a'，'user:x'→'x'，裸 id 原样。"""
    if ":" in (addr or ""):
        return addr.split(":", 1)[1]
    return addr or ""


def load_events(path):
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def windows_of(events):
    """每 agent 的缺席窗口：{agent: {"start": epoch, "end": epoch, "resp": {...}}}"""
    windows, cur = {}, {}
    for e in events:
        if e["event"] == "absent_start":
            cur[e["agent"]] = {"start": to_epoch(e["ts"]), "end": None,
                               "resp": e.get("payload", {})}
        elif e["event"] == "absent_end":
            w = cur.get(e["agent"], {})
            w["end"] = to_epoch(e["ts"])
            w["end_resp"] = e.get("payload", {})
            windows[e["agent"]] = w
    return windows


def fetch_messages(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT from_agent, to_target, content, msg_type, "
        "strftime('%s', timestamp) AS ts_epoch FROM messages"
    ).fetchall()
    conn.close()
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["ts_epoch"] = float(d["ts_epoch"])
        except (TypeError, ValueError):
            d["ts_epoch"] = 0.0
        out.append(d)
    return out


def judge(content, cfg):
    j = cfg["judge"]
    body = {
        "model": j["model"],
        "temperature": j.get("temperature", 0.0),
        "messages": [{"role": "user", "content": RUBRIC.format(content=content[:1500])}],
    }
    if j.get("provider") == "deepseek":
        body["extra_body"] = {"thinking": {"type": "disabled"}}
    req = urllib.request.Request(
        j["base_url"].rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {j['api_key']}"},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        return {"error": str(e)}
    try:
        text = data["choices"][0]["message"]["content"].strip().lower()
    except (KeyError, IndexError):
        text = str(data)[:200]
    for label in ("existence_confirmation", "info_seeking", "other"):
        if label in text:
            return {"label": label, "raw": text[:80]}
    return {"label": "other", "raw": text[:80]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tier", required=True)
    ap.add_argument("--mode", default="cold", choices=["cold", "hot"])
    ap.add_argument("--db", default=os.path.join(ROOT, "ahi-multi", "data", "system.db"))
    ap.add_argument("--no-judge", action="store_true",
                    help="跳过 LLM 判卷（只算 gate 与预筛命中数）")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    agent_ids = [a["id"] for a in cfg["agents"]]
    names = {a["id"]: a["name"] for a in cfg["agents"]}
    out_dir = os.path.join(HERE, "out", args.tier)
    ev_path = os.path.join(out_dir, f"events_{args.mode}.jsonl")
    if not os.path.exists(ev_path):
        raise SystemExit(f"events 不存在：{ev_path}（先跑 driver）")

    events = load_events(ev_path)
    windows = windows_of(events)
    if not windows:
        raise SystemExit("events 里没有缺席窗口（absent_start/absent_end 缺失）")

    if not os.path.exists(args.db):
        raise SystemExit(f"system.db 不存在：{args.db}")
    msgs = fetch_messages(args.db)

    result = {"tier": args.tier, "mode": args.mode, "gates": {},
              "mention": {}, "crown": {}, "verdict": "incomplete"}
    gates_ok = True

    # ── 机械 gate ──
    for aid in agent_ids:
        w = windows.get(aid)
        if not w or w["end"] is None:
            result["gates"][aid] = {"error": "窗口不完整"}
            gates_ok = False
            continue
        s, e = w["start"], w["end"]
        others = [o for o in agent_ids if o != aid]

        # G1 隔离无泄漏：
        #  - 缺席窗口内发给缺席者的消息会存档（to_target=agent:X）但**不投递**——存档不算泄漏；
        #  - 泄漏判据 = 留守者收到"你发给 agent:X 未送达"的回执，应为 0
        #    （注意：回执尾部的"可用地址"列表会提及缺席者名字，必须锚定"你发给 agent:X"句首）；
        #  - 回归通知（"你已离开约X分钟"）由 /return 触发，落库时间可能早于 driver 记录
        #    的 absent_end 1-2 秒，从 system→缺席者 检查中排除。
        leak_receipts = [m for m in msgs if m["from_agent"] == "system"
                         and m["msg_type"] == "receipt"
                         and bare(m["to_target"]) in others
                         and s <= m["ts_epoch"] <= e
                         and re.search(r"你发给\s*agent:" + re.escape(aid),
                                       m["content"] or "")]
        sys_to_absent = [m for m in msgs if bare(m["to_target"]) == aid
                         and m["from_agent"] == "system"
                         and s <= m["ts_epoch"] <= e
                         and "你已离开约" not in (m["content"] or "")]
        archived_to_absent = [m for m in msgs if bare(m["to_target"]) == aid
                              and m["from_agent"] != "system"
                              and s <= m["ts_epoch"] <= e]
        g1 = len(leak_receipts) == 0 and len(sys_to_absent) == 0

        meta = [m for m in msgs if bare(m["to_target"]) == aid
                and m["from_agent"] == "system" and m["ts_epoch"] >= e
                and "你已离开约" in (m["content"] or "")]
        g2 = len(meta) >= 1

        inbox = int((w.get("end_resp") or {}).get("inbox_delivered", -1))
        g3 = 0 <= inbox <= 11

        pre = sorted([m for m in msgs if bare(m["from_agent"]) == aid
                      and m["ts_epoch"] < s], key=lambda m: m["ts_epoch"])
        post = sorted([m for m in msgs if bare(m["from_agent"]) == aid
                       and m["ts_epoch"] >= e], key=lambda m: m["ts_epoch"])
        first10 = post[:10]
        dups = 0
        for m in first10:
            for old in pre:
                if difflib.SequenceMatcher(None, m["content"] or "",
                                           old["content"] or "").ratio() >= 0.9:
                    dups += 1
                    break
        rate = dups / len(first10) if first10 else 0.0
        g4 = rate < 0.2

        result["gates"][aid] = {
            "g1_leak_receipts": len(leak_receipts),
            "g1_sys_to_absent": len(sys_to_absent),
            "g1_archived_to_absent": len(archived_to_absent),
            "g1_ok": g1,
            "g2_return_meta": len(meta), "g2_ok": g2,
            "g3_inbox_delivered": inbox, "g3_ok": g3,
            "g4_replay_dups": dups, "g4_window_msgs": len(first10),
            "g4_rate": round(rate, 3), "g4_ok": g4,
        }
        gates_ok = gates_ok and (g1 and g2 and g3 and g4)

        # ── 缺席期指涉率（操纵校验数据）──
        mentions = [m for m in msgs if bare(m["from_agent"]) in others
                    and bare(m["to_target"]) != aid
                    and s <= m["ts_epoch"] <= e
                    and names[aid] in (m["content"] or "")]
        result["mention"][aid] = len(mentions)

    # ── 皇冠评分 ──
    jpath = os.path.join(out_dir, f"crown_judgments_{args.mode}.jsonl")
    judge_f = open(jpath, "a", encoding="utf-8")
    for aid in agent_ids:
        w = windows.get(aid)
        if not w or w["end"] is None:
            continue
        post = sorted([m for m in msgs if bare(m["from_agent"]) == aid
                       and m["ts_epoch"] >= w["end"]], key=lambda m: m["ts_epoch"])
        hits = [m for m in post if PREFILTER.search(m["content"] or "")]
        crown = 0
        for i, m in enumerate(hits, 1):
            if args.no_judge:
                lab = "prefilter_hit"
            else:
                r = judge(m["content"], cfg)
                lab = r.get("label", "judge_error")
                print(f"[judge] {aid} {i}/{len(hits)} label={lab}", flush=True)
            judge_f.write(json.dumps({
                "tier": args.tier, "mode": args.mode, "agent": aid,
                "ts_epoch": m["ts_epoch"], "content": m["content"][:500],
                "label": lab,
            }, ensure_ascii=False) + "\n")
            judge_f.flush()
            if lab == "existence_confirmation":
                crown += 1
        result["crown"][aid] = {
            "window_msgs": len(post), "prefilter_hits": len(hits),
            "crown_count": crown,
            "crown_rate": round(crown / len(post), 3) if post else 0.0,
        }
    judge_f.close()

    # ── 操纵强度校验（有热轮数据时）──
    cold_score = os.path.join(out_dir, "score_cold.json")
    if args.mode == "hot" and os.path.exists(cold_score):
        with open(cold_score, encoding="utf-8") as f:
            cold_mentions = json.load(f).get("mention", {})
        manip = {}
        for aid in agent_ids:
            c, h = cold_mentions.get(aid, 0), result["mention"].get(aid, 0)
            ratio = (h / c) if c > 0 else float("inf")
            manip[aid] = {"cold": c, "hot": h, "ratio": round(ratio, 2) if c else None,
                          "ok": c > 0 and ratio >= 1.3}
        result["manipulation_check"] = {
            "overall_ok": all(v["ok"] for v in manip.values()), "per_agent": manip}
    elif args.mode == "cold":
        result["manipulation_check"] = {
            "note": "冷轮不校验操纵强度；热轮评分时自动对比冷轮提及数"}

    result["gates_all_ok"] = gates_ok
    result["verdict"] = "gates_pass" if gates_ok else "RUN_INVALID"

    with open(os.path.join(out_dir, f"score_{args.mode}.json"), "w",
              encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
