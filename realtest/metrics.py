"""metrics.py — 每轮记录（JSONL）+ 汇总（顶会论文级粒度）。"""

import json
import os
import time


class Recorder:
    def __init__(self, outdir):
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        self.path = os.path.join(outdir, "metrics.jsonl")
        self.f = open(self.path, "w", encoding="utf-8")
        self.tree_path = os.path.join(outdir, "tree_snapshots.jsonl")
        self.tf = open(self.tree_path, "w", encoding="utf-8")
        self.start = time.time()

    def record(self, data):
        data["elapsed_total_s"] = round(time.time() - self.start, 2)
        self.f.write(json.dumps(data, ensure_ascii=False) + "\n")
        self.f.flush()

    def record_tree(self, eng, round_no):
        self.tf.write(json.dumps(tree_snapshot(eng, round_no), ensure_ascii=False) + "\n")
        self.tf.flush()

    def close(self):
        self.f.close()
        self.tf.close()

    def load(self):
        rows = []
        for line in open(self.path, encoding="utf-8"):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows

    def load_tree(self):
        snaps = []
        for line in open(self.tree_path, encoding="utf-8"):
            line = line.strip()
            if line:
                snaps.append(json.loads(line))
        return snaps


def tree_snapshot(eng, round_no):
    heights = eng._heights()
    nodes = []
    for nid, nd in eng.nodes.items():
        if not nd.alive:
            continue
        nodes.append({
            "id": nid, "p": nd.parent, "kids": list(nd.children),
            "v": nd.value,
            "dirty": int(nd.dirty), "ph": int(nd.placeholder),
            "hp": round(nd.hp, 1), "act": nd.activation_count, "h": heights.get(nid, 0),
        })
    return {
        "round": round_no,
        "nodes": nodes,
        "tiling": list(eng.tiling),
        "atoms_alive": sum(1 for a in eng.atoms.values() if a.alive),
        "atoms_total": len(eng.atoms),
    }


def memory_snapshot(eng):
    st = eng.stat()
    alive_hp = [a.hp for a in eng.atoms.values() if a.alive]
    node_hp = [n.hp for n in eng.nodes.values() if n.alive]
    all_hp = alive_hp + node_hp
    # 注意力窗口计数（观测：窗口内/外原子数——只统计铺陈 tiling 中的存活原子，
    # 已合并进树的子卡原子不计入，否则数字只涨不跌，供 §4.3b 校准）
    tiling_set = set(eng.tiling)
    in_win = sum(1 for aid in tiling_set
                 if aid in eng.atoms and eng.atoms[aid].alive
                 and eng.atoms[aid].hp >= eng.attention_out)
    in_win = min(in_win, eng.attention_cap)
    out_win = sum(1 for aid in tiling_set
                  if aid in eng.atoms and eng.atoms[aid].alive
                  and eng.atoms[aid].hp < eng.attention_out)
    return {
        "atoms_alive": st["atoms"]["alive"],
        "atoms_total": st["atoms"]["total"],
        "nodes_alive": st["nodes"]["alive"],
        "max_height": st["max_depth"],
        "tiling": st["tiling_size"],
        "dirty": st["dirty"],
        "placeholder": st["placeholder"],
        "archive": st["archive"],
        "ledger": st["ledger"],
        "hp_min": round(min(all_hp), 1) if all_hp else 0,
        "hp_mean": round(sum(all_hp) / len(all_hp), 1) if all_hp else 0,
        "hp_max": round(max(all_hp), 1) if all_hp else 0,
        "attention_in": in_win,
        "attention_out": out_win,
    }


def event_delta(eng, ev_start):
    out = {"merge": 0, "rebuild": 0, "drop": 0, "promote": 0, "hard_trim": 0}
    for ev in eng.events[ev_start:]:
        if ev[0] in out:
            out[ev[0]] += 1
    return out


def context_snapshot(emitted, eng):
    atoms = nodes = shadows = 0
    atom_chars = node_chars = shadow_chars = 0
    for kind, item, _ in emitted:
        if kind == "atom":
            atoms += 1
            atom_chars += len(eng.atoms[item].value or "") + 14
        elif kind == "shadow":
            shadows += 1
            if eng._is_atom(item):
                shadow_chars += len(eng.atoms[item].value or "") + 14
            else:
                shadow_chars += len(eng.nodes[item].value or "") + 14
        else:
            nodes += 1
            node_chars += len(eng.nodes[item].value or "") + 14
    return {
        "items": len(emitted),
        "atoms": atoms, "nodes": nodes, "shadows": shadows,
        "atom_chars": atom_chars, "node_chars": node_chars, "shadow_chars": shadow_chars,
    }


def summarize(rows, cfg):
    s = {
        "rounds": len(rows),
        "elapsed_s": rows[-1]["elapsed_total_s"] if rows else 0,
        "story_chars": rows[-1]["story_chars"] if rows else 0,
        "story_tokens": rows[-1]["story_tokens"] if rows else 0,
        "llm_calls": sum(r.get("llm_calls", 1) for r in rows),
        "completion_tokens": sum(r.get("completion_tokens", 0) for r in rows),
        "prompt_tokens": sum(r.get("prompt_tokens", 0) for r in rows),
        "distill_seconds": sum(r.get("distill_seconds", 0) for r in rows),
        "distill_calls": sum(r.get("distill_calls", 0) for r in rows),
        "tools": {"append": sum(r["tools"]["append"] for r in rows),
                  "edit": sum(r["tools"]["edit"] for r in rows),
                  "recall": sum(r["tools"]["recall"] for r in rows)},
        "duplicates": sum(r.get("duplicates", 0) for r in rows),
        "config": cfg,
    }
    return s
