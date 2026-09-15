#!/usr/bin/env python3
"""run2.py — 无尽续写实测 v2：写前预想（[plan]）→ 预检索 → 区间回调（[span]）→ 笔记本（[note]）。

相对 run.py 的新机制：
1. [plan] 写前预想块：模型先想好"下一段要写什么"，系统以计划为目标驱动预检索
   （轨道B 状态条目按 plan 关键词选取注入 + 检索影子注入），下一轮组装后再写正文。
2. [span N] 区间全原子回调：把摘要卡 N 覆盖区间内的全部存活原子按位置序注入
   （全分辨率原文，带"区间回调原文"标注；受 SPAN_ATOM_CAP/SPAN_CHAR_CAP 限制）。
3. [note kind=...]key: 内容[/note] 笔记本（轨道B 最小实现）：constraint 常驻头部、
   state 按 plan 关键词精确选取、schedule 活动项注入；无衰减、无 LLM 参与写入。
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
from realengine import RealEngine, SPAN_ATOM_CAP, SPAN_CHAR_CAP
from report import build_charts, build_report
from splitter import split_sentences

SYSTEM = """你是长篇小说《青州旧事》的续写 Agent。你的全部记忆来自下面提供的【记忆流】（每轮全新组装，不依赖对话历史）。

【记忆流格式】每项带 id 与从属标注：
- N12·摘要 为摘要卡（旧内容已蒸馏）；#34·原文 为原文句卡；⊂Nx 表示其父卡；缩进表示层级；
- "召回影子"为检索注入的旧内容；"区间回调原文"为按区间注入的旧内容全文；
- 记忆流按故事位置排序，末尾是最近写的内容（热区）。

【写作流程（重要）】每轮按此顺序：
1. 先输出 [plan]：用 1~3 句说清"这一轮想写什么"（事件、涉及人物、地点、要推进的情节）。不要写正文细节。
2. 然后根据计划回想（可选）：需要确认旧情节/旧设定时，用 [recall]关键词[/recall]（下一轮注入相关摘要）或 [span N3]（回调摘要卡 N3 覆盖区间内的全部原文，下一轮生效）。只回想象中不足的部分，不要无谓回想。
3. 回想足够后输出正文（直接输出，或 [append]...[/append]）——正文最多 1000 字。
4. 状态记录（每轮必做）：本轮若有任何重要状态变化（人物死伤、物品归属转移、地点变更、时间推进、伏笔埋设、人物关系变化、对已写内容的事实性新信息），必须用 [note state]关键词: 内容[/note] 记入笔记本——笔记本是事实层，永不遗忘，且会按你的 plan 关键词注入下一轮上下文。例：[note state]玉坠: 已被顾七娘认领，背面刻"井底三尺，欠债还钱"[/note]。无变化可不写。
5. 需要修改已写内容：输出 [edit #起id..#止id]新文本[/edit]（只允许修改记忆流中可见的 #原子id；请勿频繁编辑）。

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


def render_notebook(rows):
    """笔记本注入：constraint 恒在头部；state/schedule 由 nb_render 按 plan 关键词选取。"""
    if not rows:
        return ""
    parts = []
    for kind, key, content in rows:
        if kind == "constraint":
            parts.append(f"[约束] {content}")
        elif kind == "state":
            parts.append(f"[状态·{key}] {content}")
        elif kind == "schedule":
            parts.append(f"[伏笔·{key}] {content}")
    return "\n".join(parts)


def build_context(eng, nb_rows, shadows, receipt, task, budget):
    emitted = eng.assemble(shadows, budget=budget)
    flow = eng.render_emission(emitted)
    nb = render_notebook(nb_rows)
    parts = []
    if nb:
        parts.append("【笔记本·事实层】（永不遗忘，最新状态以这里为准）\n" + nb)
    parts.append(flow)
    if receipt:
        parts.append(f"【最近操作回执】\n{receipt}")
    parts.append(f"【当前目标】\n{task}")
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
    print(f"[{time.strftime('%H:%M:%S')}] 启动 run2 | 配置: {cfg.get('provider')}/{cfg.get('model')} "
          f"budget={cfg.get('assembly_budget')} decay={cfg.get('decay')}", flush=True)
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
        attention_cap=int(cfg.get("attention_cap", 60)),
        attention_out=float(cfg.get("attention_out", 45.0)),
        attention_in_decay=float(cfg.get("attention_in_decay", 0.5)),
        attention_out_decay=float(cfg.get("attention_out_decay", 3.0)),
        merge_fanout_cap=int(cfg.get("merge_fanout_cap", 12)),
        pressure_threshold=float(cfg.get("pressure_threshold", 2.0)),
        expand_atom_cap=int(cfg.get("expand_atom_cap", 80)),
        expand_char_cap=int(cfg.get("expand_char_cap", 3000)),
        assembly_budget=budget,
        retrieval_budget=retr_budget,
    )

    outdir = args.out or os.path.join(os.path.dirname(__file__), "out",
                                      time.strftime("run2_%Y%m%d_%H%M%S"))
    os.makedirs(outdir, exist_ok=True)
    rec = Recorder(outdir)
    responses_path = os.path.join(outdir, "responses.txt")
    story_path = os.path.join(outdir, "story_full.txt")
    notebook_path = os.path.join(outdir, "notebook.json")
    log_path = os.path.join(outdir, "run.log")
    runlog = open(log_path, "a", encoding="utf-8")
    runlog.write(f"=== 启动 {time.strftime('%Y-%m-%d %H:%M:%S')} | "
                 f"budget={budget} decay={cfg.get('decay')} merge_every={merge_every} ===\n")
    runlog.flush()

    opening = open(os.path.join(os.path.dirname(__file__), "story_open.txt"),
                   encoding="utf-8").read().strip()
    opening_sents = split_sentences(opening)
    eng.write(opening_sents)
    print(f"[{time.strftime('%H:%M:%S')}] 引擎就绪，开头 {len(opening_sents)} 句载入 | "
          f"向量模型懒加载（首次检索时加载）", flush=True)
    receipt = f"初始化：载入开头 {len(opening_sents)} 句。"
    last_shadows = None
    pending_plan = None
    last_plan = None
    ev_prev = 0
    t0 = time.time()
    round_no = 0
    story_chars = sum(len(s) for s in opening_sents)

    cfg_for_report = dict(cfg)
    cfg_for_report["model"] = getattr(llm, "model", cfg.get("model", "?"))
    cfg_for_report["assembly_budget"] = budget
    cfg_for_report["max_append_chars"] = max_append
    cfg_for_report["notebook"] = "on"

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

            # 笔记本按当前 plan 关键词选取注入；预检索影子来自上一轮 plan/recall
            nb_rows = eng.nb_render(last_plan or "")
            nb_injected = [(k, key) for k, key, _ in nb_rows]
            task = (f"按流程写作：先 [plan] 想好写什么，需要回想用 [recall]/[span]，"
                    f"回想足够后写正文 ≤{max_append} 字；状态变化用 [note state] 记录。"
                    f"（累计 {story_chars} 字，第 {round_no} 轮；上一轮计划：{last_plan or '无'}）")
            user_ctx, emitted = build_context(eng, nb_rows, last_shadows, receipt, task, budget)
            ctx_snap = context_snapshot(emitted, eng)

            t_llm = time.time()
            print(f"[{time.strftime('%H:%M:%S')}] 轮 {round_no} 组装 {len(user_ctx)} 字符 → LLM 请求中…", flush=True)
            text, usage, lat = llm.chat(SYSTEM, user_ctx)
            lat_ms = (time.time() - t_llm) * 1000

            with open(responses_path, "a", encoding="utf-8") as f:
                f.write(f"===== 轮 {round_no} =====\n{text}\n\n")

            tools = {"append": 0, "edit": 0, "recall": 0, "plan": 0, "span": 0, "note": 0}
            receipts = []
            appended_chars = 0
            recall_info = None
            span_info = None
            distill_calls0 = eng.distill["calls"]
            distill_sec0 = eng.distill["seconds"]

            blocks = parse_response(text)
            this_plan = None
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
                        # C：命中概括卡 → 展开全部后代原子为影子（活一轮）
                        last_shadows = eng.expand_leaves(hits)
                        recall_info = {"query": q, "hits": hits,
                                       "injected": len(last_shadows)}
                        receipts.append(f"recall「{q}」→ 命中 {len(hits)} 项，"
                                        f"展开 {len(last_shadows)} 原子影子下轮生效")
                    else:
                        receipts.append("recall 查询词过短，已忽略")
                elif kind == "plan":
                    tools["plan"] += 1
                    this_plan = (payload or "").strip()
                    receipts.append(f"计划：「{this_plan[:60]}{'…' if len(this_plan) > 60 else ''}」")
                elif kind == "span":
                    tools["span"] += 1
                    spec, _ = payload
                    ids = [int(m) for m in __import__("re").findall(r"\d+", spec or "")]
                    if ids:
                        nid = ids[0]
                        atoms = eng.span_atoms(nid)
                        if atoms is None:
                            receipts.append(f"span N{nid}：节点不存在或已删除，已忽略")
                        else:
                            # C：区间回调 = 展开叶子（与检索展开同语义，全分辨率）
                            last_shadows = eng.expand_leaves([nid])
                            span_info = {"node": nid, "atoms": len(last_shadows),
                                         "chars": sum(len(eng.atoms[a].value or "") for a in last_shadows)}
                            receipts.append(f"span N{nid} → 展开 {len(last_shadows)} 个原子原文，下轮生效")
                    else:
                        receipts.append("span 参数不完整（需要节点 id），已忽略")
                elif kind == "note":
                    tools["note"] += 1
                    spec, body = payload
                    body = (body or "").strip()
                    kkind = (spec or "state").strip() or "state"
                    _re_ = __import__("re")
                    cm = _re_.split(r"[:：]", body, maxsplit=1)
                    if len(cm) == 2 and cm[0].strip():
                        key, content = cm[0].strip(), cm[1].strip()
                    else:
                        key, content = body[:8], body
                    key = eng.nb_set(key, kkind, content)
                    receipts.append(f"笔记本 {kkind}「{key}」已更新")

            eng.tick(1)

            # 窗外压力：窗外原子超限 → 立即 force_out 合并（不等 merge_every）
            if eng.attention_pressure():
                eng.merge_pass(force_out=True)
                receipts.append("窗外原子超限 → force 合并")
            elif round_no % merge_every == 0:
                eng.merge_pass()

            # 计划驱动：有 plan 时按计划预检索（下轮生效，与 recall/span 命中合并）
            if this_plan:
                last_plan = this_plan
                hits = eng.retrieve(this_plan, budget=retr_budget)
                expanded = eng.expand_leaves(hits)
                if last_shadows:
                    expanded = list(dict.fromkeys(list(expanded) + list(last_shadows)))
                last_shadows = expanded
                recall_info = {"query": f"[plan]{this_plan[:40]}", "hits": hits,
                               "injected": len(expanded)}
                receipts.append(f"按计划预检索 → {len(hits)} 项，展开 {len(expanded)} 原子影子注入下轮")
            else:
                last_shadows = None

            rec.record_tree(eng, round_no)

            story_chars = sum(len(eng.atoms[a].value) for a in eng.chain_ids())
            story_tokens = estimate_tokens(eng.story_full_text())
            dup = count_duplicates(eng)
            ev_start = ev_prev
            ev = event_delta(eng, ev_prev)
            ev_prev = len(eng.events)
            # hard_trim 明细（观测：被截了哪些项、砍掉多少字符、是不是热区）
            hard_trim_detail = []
            for e in eng.events[ev_start:]:
                if e[0] == "hard_trim":
                    hard_trim_detail.append({
                        "over_chars": e[1],
                        "dropped": [{"kind": d[0], "id": d[1], "hp": round(d[2], 1),
                                     "chars": d[3]} for d in e[2]],
                    })

            rec.record({
                "round": round_no,
                "llm_latency_ms": round(lat_ms, 1),
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "prompt_chars": len(SYSTEM) + len(user_ctx),
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
                "span": span_info,
                "plan": last_plan,
                "nb_injected": nb_injected,
                "notebook": {k: v["content"] for k, v in eng.notebook.items()},
                "blocks": [b[0] for b in blocks],
                "hard_trim": hard_trim_detail[-1] if hard_trim_detail else None,
                "response_head": text[:200],
            })
            if receipts:
                receipt = "；".join(receipts[-3:])

            with open(story_path, "w", encoding="utf-8") as f:
                f.write(eng.story_full_text())
            with open(notebook_path, "w", encoding="utf-8") as f:
                json.dump(eng.notebook, f, ensure_ascii=False, indent=1)

            mem = memory_snapshot(eng)
            print(f"[轮 {round_no}] 追加 {appended_chars} 字 | 累计 {story_chars} 字 | "
                  f"工具 {tools} | 上下文 {ctx_snap['items']} 项 / "
                  f"{sum(v for k, v in ctx_snap.items() if k.endswith('_chars'))} 字符 | "
                  f"LLM {lat_ms:.0f}ms | 蒸馏 {eng.distill['calls'] - distill_calls0} 次 | "
                  f"窗口内 {mem['attention_in']} / 窗外 {mem['attention_out']}")
            runlog.write(f"[轮 {round_no}] {time.strftime('%H:%M:%S')} 追加 {appended_chars} 字 | "
                         f"累计 {story_chars} 字 | 工具 {tools} | 上下文 {ctx_snap['items']} 项 / "
                         f"{sum(v for k, v in ctx_snap.items() if k.endswith('_chars'))} 字符 | "
                         f"LLM {lat_ms:.0f}ms | 蒸馏 {eng.distill['calls'] - distill_calls0} 次 | "
                         f"窗口内 {mem['attention_in']} / 窗外 {mem['attention_out']} | "
                         f"硬截 {hard_trim_detail[-1] if hard_trim_detail else '无'}\n")
            runlog.flush()
            sys.stdout.flush()
    except KeyboardInterrupt:
        print("\n[中断] 手动停止，生成报告……")
    finally:
        rec.close()
        runlog.write(f"=== 结束 {time.strftime('%Y-%m-%d %H:%M:%S')} | 累计 {story_chars} 字 ===\n")
        runlog.flush()
        runlog.close()
        rows = rec.load()
        tree_snaps = rec.load_tree()
        summary = summarize(rows, cfg_for_report)
        charts = build_charts(rows, tree_snaps, outdir)
        build_report(outdir, rows, summary, charts, story_path, cfg_for_report)
        with open(story_path, "w", encoding="utf-8") as f:
            f.write(eng.story_full_text())
        with open(notebook_path, "w", encoding="utf-8") as f:
            json.dump(eng.notebook, f, ensure_ascii=False, indent=1)
        print("\n===== 汇总 =====")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"\n产出目录: {outdir}")
        print(f"  report.html / metrics.jsonl / responses.txt / story_full.txt / notebook.json")


if __name__ == "__main__":
    main()
