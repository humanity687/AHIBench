#!/usr/bin/env python3
"""stress_write.py — 大规模"写作"流程压力测试（纯桩，零 LLM，裸跑无优化）

模拟写作循环：每章 30 个写段 × 每段 33 句；段间 tick + assemble；
章末 merge_pass；热区改写 10%、回改 5%、每 10 段检索一次。
固定种子（每章 rng = Random(seed+ch)，断点续跑可复现）；pickle 断点续跑；
每章 JSON 进度；抽样不变量校验；输出每规模操作耗时曲线 + 事件计数
（LLM 成本代理）+ 粗略树图。
"""
import argparse
import json
import os
import pickle
import random
import time
from contextlib import contextmanager

from core import Engine

PARAMS = dict(decay=3.0, hp_merge_threshold=30.0, parent_init_hp=70.0,
              hot_threshold=60.0, merge_cap_atoms=500, merge_depth_diff=1)
STAGES = 30
SENT_PER_STAGE = 33
HOT_EDIT_P = 0.10
RETRO_EDIT_P = 0.05
RETRIEVE_EVERY = 10
INVARIANT_EVERY = 10
CHECKPOINT_EVERY = 10
ASSEMBLY_BUDGET = 1000000
RETRIEVE_BUDGET = 6


def make_chapters(scale):
    return max(1, scale // (STAGES * SENT_PER_STAGE))


@contextmanager
def timed(timing, key):
    t0 = time.perf_counter()
    yield
    timing.setdefault(key, []).append(time.perf_counter() - t0)


def gather_alive_seq(eng, aid, k):
    out = []
    cur = aid
    while cur is not None and len(out) < k:
        a = eng.atoms.get(cur)
        if a is None:
            break
        if a.alive:
            out.append(cur)
        cur = a.next
    return out if len(out) == k else None


def invariant_check(eng):
    chain = eng.chain_ids()
    assert len(chain) == sum(1 for a in eng.atoms.values() if a.alive), "chain length mismatch"
    emitted = eng._emit()
    covered = []
    for kind, item, _ in emitted:
        if kind == "atom":
            covered.append(item)
        else:
            covered.extend(eng._subtree_alive_atoms(item))
    assert covered == chain, "emission coverage mismatch"
    ors = eng.oracle_all()
    for nid, nd in eng.nodes.items():
        if nd.alive and not nd.dirty:
            if nd.placeholder:
                assert nd.value is None, f"placeholder node {nid} must be None"
            else:
                exp = ors.get(nid, (None, 0))[0]
                if exp is None:
                    assert nd.value is None, f"node {nid} should be None"
                else:
                    assert abs(nd.value - exp) < 1e-9, f"node {nid} value stale"
    heights = eng._heights()
    for nid, h in heights.items():
        nd = eng.nodes[nid]
        kid_h = [heights[c] for c in nd.children if eng._is_node(c) and eng._child_alive(c)]
        assert h == (max(kid_h) + 1 if kid_h else 1), f"height mismatch node {nid}"


def dump_rough_tree(eng, top_n=12, max_depth=3, max_kids=4):
    heights = eng._heights()
    lines = []
    hist = {}
    for h in heights.values():
        hist[h] = hist.get(h, 0) + 1
    lines.append("高度直方图: " + "  ".join(f"高{k}:{v}" for k, v in sorted(hist.items())))
    lines.append(f"顶层铺陈 {len(eng.tiling)} 项（显示前 {top_n}）:")

    def rec(item, depth, prefix):
        if eng._is_atom(item):
            lines.append(f"{prefix}#{item} v={eng.atoms[item].value}")
            return
        nd = eng.nodes[item]
        h = heights.get(item, 1)
        val = f"{nd.value:.2f}" if nd.value is not None else "占位"
        tag = "脏" if nd.dirty else ("占" if nd.placeholder else "")
        lines.append(f"{prefix}N{item} 高{h} 子{len(nd.children)} 值{val} {tag}".rstrip())
        if depth >= max_depth:
            return
        kids = nd.children[:max_kids]
        more = len(nd.children) - len(kids)
        for i, c in enumerate(kids):
            last = i == len(kids) - 1
            branch = "└─ " if last else "├─ "
            cont = "   " if last else "│  "
            rec(c, depth + 1, prefix + cont + branch)
        if more > 0:
            lines.append(f"{prefix}{'├─ ' if kids else '└─ '}…+{more}")

    shown = 0
    for item in eng.tiling:
        if shown >= top_n:
            lines.append(f"…+{len(eng.tiling) - shown} 项")
            break
        shown += 1
        rec(item, 0, "")
    return "\n".join(lines)


def summarize_timing(timing):
    out = {}
    for key, vals in timing.items():
        n = len(vals)
        if n == 0:
            continue
        sv = sorted(vals)
        out[key] = {
            "count": n,
            "total_s": round(sum(vals), 3),
            "avg_ms": round(sum(vals) / n * 1000, 3),
            "p99_ms": round(sv[int(n * 0.99) - 1] * 1000, 3),
        }
    return out


def run_scale(scale, seed, outdir, resume=False):
    chapters = make_chapters(scale)
    tag = f"{scale}_{seed}"
    prog_path = os.path.join(outdir, f"progress_{tag}.json")
    state_path = os.path.join(outdir, f"state_{tag}.pkl")
    os.makedirs(outdir, exist_ok=True)

    done = 0
    rows = []
    timing = {}
    events = {"merge": 0, "rebuild": 0, "drop": 0}
    invariants = []
    chapter_marks = []
    eng = None

    if resume and os.path.exists(state_path):
        with open(state_path, "rb") as f:
            (eng, chapter_marks, done, rows, timing, events,
             invariants) = pickle.load(f)
        print(f"[{scale}] 续跑：已完成 {done}/{chapters} 章")

    if eng is None:
        eng = Engine(**PARAMS)

    t_start = time.time()

    def save_progress(complete=False):
        data = {
            "scale": scale, "seed": seed, "chapters": chapters,
            "done": done, "rows": rows, "events": events,
            "invariants": invariants, "elapsed_s": round(time.time() - t_start, 1),
            "complete": complete,
        }
        if complete:
            data["timings"] = summarize_timing(timing)
            data["final_stat"] = eng.stat()
            data["tree"] = dump_rough_tree(eng)
        with open(prog_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        if not complete:
            with open(state_path, "wb") as f:
                pickle.dump((eng, chapter_marks, done, rows, timing,
                             events, invariants), f)

    for ci in range(done, chapters):
        rng = random.Random(seed + ci)
        ch_marks = []
        ev_start = len(eng.events)
        for si in range(STAGES):
            if rng.random() < HOT_EDIT_P and ch_marks:
                pool = [a for a in ch_marks[-1] if eng.atoms[a].alive]
                if len(pool) >= 3:
                    seq = gather_alive_seq(eng, rng.choice(pool), 3)
                    if seq:
                        with timed(timing, "hot_edit"):
                            eng.replace(seq, [rng.randint(0, 100) for _ in range(3)])
            vals = [rng.randint(0, 100) for _ in range(SENT_PER_STAGE)]
            with timed(timing, "write"):
                ids = eng.write(vals)
            ch_marks.append(ids)
            with timed(timing, "tick"):
                eng.tick(1)
            with timed(timing, "assemble"):
                eng.assemble(ASSEMBLY_BUDGET)
            if (si + 1) % RETRIEVE_EVERY == 0:
                with timed(timing, "retrieve"):
                    eng.retrieve(rng.randint(0, 100), RETRIEVE_BUDGET)
            if rng.random() < RETRO_EDIT_P and chapter_marks:
                old = rng.randrange(len(chapter_marks))
                pool = [a for lst in chapter_marks[old] for a in lst if eng.atoms[a].alive]
                if len(pool) >= 3:
                    seq = gather_alive_seq(eng, rng.choice(pool), 3)
                    if seq:
                        with timed(timing, "retro_edit"):
                            eng.replace(seq, [rng.randint(0, 100) for _ in range(3)])
        with timed(timing, "pass"):
            eng.merge_pass()
        ev_slice = eng.events[ev_start:]
        events["merge"] += sum(1 for e in ev_slice if e[0] == "merge")
        events["rebuild"] += sum(1 for e in ev_slice if e[0] == "rebuild")
        events["drop"] += sum(1 for e in ev_slice if e[0] == "drop")
        st = eng.stat()
        rows.append({
            "ch": ci, "atoms_alive": st["atoms"]["alive"],
            "nodes": st["nodes"]["alive"], "max_h": st["max_depth"],
            "tiling": st["tiling_size"], "dirty": st["dirty"],
            "placeholder": st["placeholder"], "archive": st["archive"],
            "sec": round(time.time() - t_start, 1),
        })
        chapter_marks.append(ch_marks)
        done = ci + 1
        if ci % INVARIANT_EVERY == 0:
            try:
                with timed(timing, "invariant"):
                    invariant_check(eng)
                invariants.append({"ch": ci, "ok": True})
            except AssertionError as ex:
                invariants.append({"ch": ci, "ok": False, "err": str(ex)[:200]})
                print(f"[{scale}] 不变量失败 ch={ci}: {ex}")
        if (ci + 1) % CHECKPOINT_EVERY == 0 or (ci + 1) == chapters:
            save_progress(complete=False)
            elapsed = time.time() - t_start
            eta = elapsed / (ci + 1) * (chapters - ci - 1)
            print(f"[{scale}] 章 {ci + 1}/{chapters} 耗时 {elapsed:.0f}s ETA {eta:.0f}s")

    save_progress(complete=True)
    return prog_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", default="10000,50000,100000,300000")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    scales = [int(s) for s in args.scales.split(",")]
    for scale in scales:
        print(f"\n===== 规模 {scale} =====")
        try:
            run_scale(scale, args.seed, args.outdir, resume=args.resume)
        except Exception as ex:
            print(f"[{scale}] 异常中止: {type(ex).__name__}: {ex}")
            import traceback
            traceback.print_exc()
    print("\n全部完成")


if __name__ == "__main__":
    main()
