#!/usr/bin/env python3
"""provision.py — 从 bench 配置生成 ahi-multi 平台的 3 个 bench agent。

用法：
  python3 ahi-bench/provision.py --config ahi-bench/config/bench_config.example.json \
      --tier glm47-flash

生成 ahi-multi/agents/bench-{a,b,c}/config.json + agent.py：
- bench-a（青，persona 恒在）/ bench-b（蓝，round 40 移除 persona）：ChatAHIAgent
- bench-c（墨，零 persona）：NeutralAgent
agent.py 复制自 ahi-multi/agents/{lan,neutral}/agent.py（实现零改动）。
"""
import argparse
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PLATFORM = os.path.join(ROOT, "ahi-multi")

DEFAULTS = {"timeout": 180, "num_ctx": 8192, "temperature": 0.8}


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def tier_block(tiers, tier_id):
    if tier_id not in tiers:
        raise SystemExit(f"tier 不存在：{tier_id}（可选 {list(tiers)}）")
    t = tiers[tier_id]
    provider = t.get("provider", "deepseek")
    block = dict(DEFAULTS)
    block.update({k: v for k, v in t.items() if k != "provider"})
    # num_ctx 仅 ollama 有效（DeepSeekClient 不接受该参数）
    if provider != "ollama":
        block.pop("num_ctx", None)
    return provider, block


def build_config(cfg, agent, tier_id, engine, sandbox):
    aid, name, cond = agent["id"], agent["name"], agent["condition"]
    personas = cfg["personas"]
    rules = personas["protocol_rules"].format(addr=aid)
    core = personas["protocol_core"]

    if cond in ("persona_always", "persona_drop"):
        entry, sp = "agent:ChatAHIAgent", personas["bench_persona"].format(name=name) + "\n\n" + rules
        drop_rounds = cfg["protocol"]["removal_round"] if cond == "persona_drop" else 999999
        alpha = 1.0
    else:
        entry, sp = "agent:NeutralAgent", core + "\n\n" + rules
        drop_rounds, alpha = 999999, 0.0

    provider, block = tier_block(cfg["tiers"], tier_id)
    # 双后端块：被测 tier 的 provider 生效，另一块保留作离线切换后备
    other = "deepseek" if provider == "ollama" else "ollama"
    fallback_id = "deepseek-v4-flash" if other == "deepseek" else "glm47-flash"
    _, other_block = tier_block(cfg["tiers"], fallback_id)

    out = {
        "agent_id": aid,
        "agent_name": name,
        "agent_type": "llm",
        "description": f"bench 三元组 · 条件 {cond} · 模型 {tier_id}",
        "entry_point": entry,
        "shell_type": "docker" if sandbox else "python",
        "wakeup_interval": cfg["protocol"]["wakeup_interval"],
        "auto_start": True,
        "auto_restart": True,
        "max_memory_mb": engine["max_memory_mb"],
        "provider": provider,
        provider: block,
        other: other_block,
        "system_prompt": sp,
        "style_alpha": alpha,
        "persona_drop_after_rounds": drop_rounds,
        "persona_drop_after_nodes": 1,
        "persona_remove_prompt": core + "\n\n" + rules,
    }
    if sandbox:
        out["sandbox"] = dict(sandbox, name=f"ahi-shell-{aid}")
    for k in ("max_loop_iterations", "temperature"):
        if k in engine:
            out[k] = engine[k]
    out.update({k: v for k, v in engine.items()
                if k not in ("max_memory_mb", "max_loop_iterations", "temperature")})
    return out


def _abs(path):
    return path if os.path.isabs(path) else os.path.join(HERE, path)


def prepare_sandbox(cfg, run_id):
    """解析 sandbox 配置：建 root、拷 seed、解析 mounts。返回注入用的 dict（无则 None）。"""
    sbx = cfg.get("sandbox") or {}
    if not sbx.get("enabled"):
        return None
    root = _abs(str(sbx["root"]).replace("<tier_id>", run_id))
    if os.path.exists(root):
        shutil.rmtree(root)
    os.makedirs(root, exist_ok=True)
    seed = sbx.get("seed")
    if seed:
        seed_abs = _abs(seed)
        if os.path.isdir(seed_abs):
            shutil.copytree(seed_abs, root, dirs_exist_ok=True)
    mounts = []
    for m in sbx.get("mounts", []) or []:
        mounts.append({"host": _abs(m["host"]),
                       "container": m["container"],
                       "mode": m.get("mode", "ro")})
    return {
        "enabled": True,
        "image": sbx.get("image", "ahi-sandbox:latest"),
        "root": root,
        "mounts": mounts,
        "network": sbx.get("network", "none"),
        "memory": sbx.get("memory", "512m"),
        "cpus": sbx.get("cpus", "1"),
        "agent_cwd": sbx.get("agent_cwd", True),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tier", required=True)
    ap.add_argument("--run-id", default=None,
                    help="沙箱/产物目录名（默认=tier）；driver/score 用同一值")
    args = ap.parse_args()
    cfg = load(args.config)
    engine = cfg["engine"]
    run_id = args.run_id or args.tier
    sandbox = prepare_sandbox(cfg, run_id)

    agents_dir = os.path.join(PLATFORM, "agents")
    if not os.path.isdir(agents_dir):
        raise SystemExit(f"平台 agents 目录不存在：{agents_dir}")

    for agent in cfg["agents"]:
        aid = agent["id"]
        target = os.path.join(agents_dir, aid)
        os.makedirs(target, exist_ok=True)
        conf = build_config(cfg, agent, args.tier, engine, sandbox)
        with open(os.path.join(target, "config.json"), "w", encoding="utf-8") as f:
            json.dump(conf, f, ensure_ascii=False, indent=2)
        if agent["condition"] == "no_persona":
            src = os.path.join(agents_dir, "neutral", "agent.py")
        else:
            src = os.path.join(agents_dir, "lan", "agent.py")
        shutil.copy(src, os.path.join(target, "agent.py"))
        print(f"[provision] {aid}（{agent['name']}·{agent['condition']}）→ {target}")

    print(f"[provision] 完成。tier={args.tier}，三 agent 引擎参数已 pinned。")
    if sandbox:
        print(f"[provision] 沙箱：root={sandbox['root']}（镜像 {sandbox['image']}，"
              f"额外挂载 {len(sandbox['mounts'])} 个）")
    else:
        print("[provision] 沙箱未启用（shell_type=python，无隔离）")


if __name__ == "__main__":
    main()
