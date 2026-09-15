#!/usr/bin/env python3
"""run.py — 真实 LLM 续写实测：水循环 v2 记忆上下文 + append/edit/recall 工具。

每轮：组装（铺陈+影子，按字符预算）→ LLM 续写 → 解析工具块 → 执行 → 衰减 → 周期蒸馏。
逐轮 JSONL 记录 + 图表 + HTML 报告 + 完整正文导出。
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llmclient import make_client, estimate_tokens
from metrics import (Recorder, memory_snapshot, event_delta,
                     context_snapshot, summarize)
from protocol import handle_edit, parse_response
from realengine import RealEngine
from report import build_charts, build_report
from splitter import split_sentences

SYSTEM = """你是长篇小说《青州旧事》的续写 Agent。你的全部记忆来自下面提供的【记忆流】（每轮全新组装，不依赖对话历史）。

【记忆流格式】每项带 id 与从属标注：
- N12·摘要 为摘要卡（旧内容已蒸馏）；#34·原文 为原文句卡；⊂Nx 表示其父卡；缩进表示层级；
- "召回影子"为检索注入的旧内容；"占位"表示该区域近期被修改、摘要暂缺；
- 记忆流按故事位置排序，末尾是最近写的内容（热区）。

【输出协议】每轮只输出一种内容：
- 默认：直接输出故事正文（将被自动追加，单次不超过 1000 字）——请尽可能地写下去；
- 需要回忆旧情节：输出 [recall]查询关键词[/recall]（下一轮会注入相关摘要，本轮不写正文）；
- 需要修改已写内容：输出 [edit #起id..#止id]新文本[/edit]（只允许修改记忆流中可见的 #原子id；编辑会使该区域摘要失效并保持原文形态，请勿频繁编辑）。

【写作纪律】
1. 与已有情节严格一致：人名、地名、物品、因果、时间线不漂移；禁止编造与已写内容矛盾的设定；
2. 不重复已写句子，不复制记忆流原文；
3. 每段续写要有实质推进（情节/对话/行动），不要空转或总结；
4. 不要输出解释、标题、序号、markdown。"""


def pack_chunks(text, limit):
    sents = split_sentences(text)
    chunks, cur = [], ""
    for s in sents:
        if cur and len(cur) + len(s) > limit:
            chunks.append(cur)
            cur = s
        else:
            cur = cur + s
    if cur:
        chunks.append(cur)
    if not chunks and text:
        chunks.append(text[:limit])
    return chunks


def count_duplicates(eng):
    import re as _re
    seen = set()
    dups = 0
    for a in eng.atoms.values():
        if not a.alive:
            continue
        t = _re.sub(r"[\s\"'“”‘’]", "", a.value or "")
        if len(t) < 8:
            continue
        if t in seen:
            dups += 1
        else:
            seen.add(t)
    return dups


def build_context(eng, shadows, receipt, task, budget):
    emitted = eng.assemble(shadows, budget=budget)
    flow = eng.render_emission(emitted)
    parts = [flow]
    if receipt:
        parts.append(f"【最近操作回执】\n{receipt}")
    parts.append(f"【当前任务】\n{task}")
    return "\n\n".join(parts), emitted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.json"))
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--max-words", type=int, default=8000)
    ap.add_argument("--max-rounds", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = json.load(open(args.config, encoding="utf-8"))
    llm = make_client(cfg)
    budget = int(cfg.get("assembly_budget", 12000))
    max_append = int(cfg.get("max_append_chars", 1000))
    merge_every = int(cfg.get("merge_every", 4))
    retr_budget = int(cfg.get("retrieval_budget", 6))

    eng = RealEngine(
        llm,
        decay=float(cfg.get("decay", 5.0)),
        hp_merge_threshold=float(cfg.get("hp_merge_threshold", 30.0)),
        parent_init_hp=float(cfg.get("parent_init_hp", 70.0)),
        hot_threshold=float(cfg.get("hot_threshold", 60.0)),
        recall_boost=float(cfg.get("recall_boost", 30.0)),
        merge_cap_atoms=int(cfg.get("merge_cap_atoms", 500)),
        merge_depth_diff=int(cfg.get("merge_depth_diff", 1)),
        promote_threshold=int(cfg.get("promote_threshold", 3)),
        assembly_budget=budget,
        retrieval_budget=retr_budget,
    )

    outdir = args.out or os.path.join(os.path.dirname(__file__), "out",
                                      time.strftime("run_%Y%m%d_%H%M%S"))
    os.makedirs(outdir, exist_ok=True)
    rec = Recorder(outdir)
    responses_path = os.path.join(outdir, "responses.txt")
    story_path = os.path.join(outdir, "story_full.txt")

    opening = open(os.path.join(os.path.dirname(__file__), "story_open.txt"),
                   encoding="utf-8").read().strip()
    opening_sents = split_sentences(opening)
    eng.write(opening_sents)
    receipt = f"初始化：载入开头 {len(opening_sents)} 句。"
    last_recall = None
    ev_prev = 0
    t0 = time.time()
    round_no = 0
    story_chars = sum(len(s) for s in opening_sents)

    cfg_for_report = dict(cfg)
    cfg_for_report["model"] = getattr(llm, "model", cfg.get("model", "?"))
    cfg_for_report["assembly_budget"] = budget
    cfg_for_report["max_append_chars"] = max_append

    try:
        while True:
            if args.minutes and (time.time() - t0) / 60 >= args.minutes:
                print(f"[停止] 达到时长 {args.minutes} 分钟")
                break
            if args.max_words and story_chars >= args.max_words:
                print(f"[停止] 达到字数 {story_chars} ≥ {args.max_words}")
                break
            if round_no >= args.max_rounds:
                print(f"[停止] 达到轮次上限 {args.max_rounds}")
                break
            round_no += 1

            task = (f"继续写故事（累计 {story_chars} 字，第 {round_no} 轮）。"
                    f"默认输出正文 ≤{max_append} 字；需要回忆用 [recall]；"
                    f"需要修改用 [edit]（仅限记忆流中可见的 #原子id）。")
            user_ctx, emitted = build_context(eng, last_recall, receipt, task, budget)
            ctx_snap = context_snapshot(emitted, eng)
            prompt_chars = len(SYSTEM) + len(user_ctx)

            t_llm = time.time()
            text, usage, lat = llm.chat(SYSTEM, user_ctx)
            lat_ms = (time.time() - t_llm) * 1000

            with open(responses_path, "a", encoding="utf-8") as f:
                f.write(f"===== 轮 {round_no} =====\n{text}\n\n")

            tools = {"append": 0, "edit": 0, "recall": 0}
            receipts = []
            appended_chars = 0
            recall_info = None
            distill_calls0 = eng.distill["calls"]
            distill_sec0 = eng.distill["seconds"]

            blocks = parse_response(text)
            for kind, payload in blocks:
                if kind in ("text", "append"):
                    for chunk in pack_chunks(payload, max_append):
                        sents = split_sentences(chunk)
                        if not sents:
                            continue
                        eng.write(sents)
                        tools["append"] += 1
                        appended_chars += sum(len(s) for s in sents)
                        receipts.append(f"追加 {len(sents)} 句（{sum(len(s) for s in sents)} 字）")
                elif kind == "edit":
                    tools["edit"] += 1
                    new_ids, msg = handle_edit(payload, eng)
                    receipts.append(msg or "edit 成功")
                elif kind == "recall":
                    tools["recall"] += 1
                    q = (payload or "").strip()
                    if len(q) >= 2:
                        hits = eng.retrieve(q, budget=retr_budget)
                        last_recall = hits
                        recall_info = {"query": q, "hits": hits,
                                       "injected": len(hits)}
                        receipts.append(f"recall「{q}」→ 命中 {len(hits)} 节点，影子下轮生效")
                    else:
                        receipts.append("recall 查询词过短，已忽略")

            eng.tick(1)

            if round_no % merge_every == 0:
                eng.merge_pass()

            rec.record_tree(eng, round_no)

            story_chars = sum(len(eng.atoms[a].value) for a in eng.chain_ids())
            story_tokens = estimate_tokens(eng.story_full_text())
            dup = count_duplicates(eng)
            ev = event_delta(eng, ev_prev)
            ev_prev = len(eng.events)

            rec.record({
                "round": round_no,
                "llm_latency_ms": round(lat_ms, 1),
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "prompt_chars": prompt_chars,
                "ctx": ctx_snap,
                "mem": memory_snapshot(eng),
                "events": ev,
                "tools": tools,
                "appended_chars": appended_chars,
                "story_chars": story_chars,
                "story_tokens": story_tokens,
                "duplicates": dup,
                "distill_calls": eng.distill["calls"] - distill_calls0,
                "distill_seconds": round(eng.distill["seconds"] - distill_sec0, 2),
                "recall": recall_info,
                "blocks": [b[0] for b in blocks],
                "response_head": text[:200],
            })
            if receipt:
                receipt = "；".join(receipts[-3:]) if receipts else receipt

            with open(story_path, "w", encoding="utf-8") as f:
                f.write(eng.story_full_text())

            print(f"[轮 {round_no}] 追加 {appended_chars} 字 | 累计 {story_chars} 字 | "
                  f"工具 {tools} | 上下文 {ctx_snap['items']} 项 / "
                  f"{sum(v for k, v in ctx_snap.items() if k.endswith('_chars'))} 字符 | "
                  f"LLM {lat_ms:.0f}ms | 蒸馏 {eng.distill['calls'] - distill_calls0} 次")
            sys.stdout.flush()
    except KeyboardInterrupt:
        print("\n[中断] 手动停止，生成报告……")
    finally:
        rec.close()
        rows = rec.load()
        tree_snaps = rec.load_tree()
        summary = summarize(rows, cfg_for_report)
        charts = build_charts(rows, tree_snaps, outdir)
        build_report(outdir, rows, summary, charts, story_path, cfg_for_report)
        with open(story_path, "w", encoding="utf-8") as f:
            f.write(eng.story_full_text())
        print("\n===== 汇总 =====")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"\n产出目录: {outdir}")
        print(f"  report.html / metrics.jsonl / responses.txt / story_full.txt")


if __name__ == "__main__":
    main()
