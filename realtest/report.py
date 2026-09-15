"""report.py — matplotlib 图表 + 自包含 HTML 报告。"""

import base64
import io
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Microsoft YaHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def _save_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def draw_tree(ax, snap):
    nodes = {n["id"]: n for n in snap["nodes"]}
    tiling = [t for t in snap["tiling"] if t in nodes]

    def depth(nid):
        d = 0
        cur = nodes[nid]["p"]
        while cur is not None and cur in nodes:
            d += 1
            cur = nodes[cur]["p"]
        return d

    ys = {}

    def layout(nid):
        n = nodes[nid]
        kids = [k for k in n["kids"] if k in nodes]
        if not kids:
            y = len(ys)
            ys[nid] = y
            return y
        vals = [layout(k) for k in kids]
        y = sum(vals) / len(vals)
        ys[nid] = y
        return y

    for tid in tiling:
        layout(tid)
    for nid, n in nodes.items():
        if n["p"] in nodes and n["p"] in ys:
            x0, x1 = depth(n["p"]), depth(nid)
            ax.plot([x0, x1], [ys[n["p"]], ys[nid]], color="#c8d3e0", lw=0.7, zorder=1)
    for nid, n in nodes.items():
        if nid not in ys:
            continue
        color = "#ffebe9" if n["dirty"] else ("#fff8c5" if n["ph"] else "#ddf4ff")
        ax.scatter(depth(nid), ys[nid], s=160, c=color,
                   edgecolors="#0969da", lw=0.8, zorder=2)
        label = f"N{nid}"
        if n["ph"]:
            label = f"N{nid}·占"
        ax.text(depth(nid), ys[nid], label, fontsize=5.5, ha="center",
                va="center", zorder=3)
    ax.set_xticks(sorted(set(depth(n) for n in nodes)))
    ax.set_xlabel("深度")
    ax.set_yticks([])
    ax.set_title(f"记忆树（轮 {snap['round']}，{len(nodes)} 节点，原子 {snap['atoms_alive']}）")
    ax.grid(alpha=.2, axis="x")


def build_charts(rows, tree_snaps, outdir):
    charts = {}
    xs = [r["round"] for r in rows]
    story = [r["story_chars"] for r in rows]
    ctx = [r["ctx"]["items"] for r in rows]
    ctx_chars = [r["ctx"]["atom_chars"] + r["ctx"]["node_chars"] + r["ctx"]["shadow_chars"] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(xs, story, color="#0969da", lw=1.6)
    ax.set_xlabel("轮次"); ax.set_ylabel("累计写作字数")
    ax.set_title("累计写作字数 vs 轮次")
    ax.grid(alpha=.3)
    charts["story"] = _save_png(fig)

    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(xs, ctx_chars, color="#1a7f37", lw=1.6, label="上下文字符")
    ax.plot(xs, ctx, color="#9a6700", lw=1.2, label="上下文项数")
    ax.set_xlabel("轮次"); ax.set_ylabel("字符 / 项")
    ax.set_title("每轮上下文规模（记忆流，含影子）")
    ax.legend(); ax.grid(alpha=.3)
    charts["ctx"] = _save_png(fig)

    fig, axes = plt.subplots(2, 1, figsize=(7, 5.2), sharex=True)
    axes[0].plot(xs, [r["mem"]["atoms_alive"] for r in rows], label="原子(存活)", color="#0969da")
    axes[0].plot(xs, [r["mem"]["nodes_alive"] for r in rows], label="摘要节点", color="#1a7f37")
    axes[0].plot(xs, [r["mem"]["tiling"] for r in rows], label="铺陈项", color="#9a6700")
    axes[0].set_ylabel("数量"); axes[0].legend(); axes[0].grid(alpha=.3)
    axes[0].set_title("记忆规模演化")
    axes[1].plot(xs, [r["mem"]["max_height"] for r in rows], color="#d1242f")
    axes[1].set_xlabel("轮次"); axes[1].set_ylabel("最大高度"); axes[1].grid(alpha=.3)
    charts["mem"] = _save_png(fig)

    fig, ax = plt.subplots(figsize=(7, 3.4))
    ev = {"merge": [], "rebuild": [], "promote": []}
    for r in rows:
        for k in ev:
            ev[k].append(r["events"].get(k, 0))
    ax.bar(xs, ev["merge"], label="merge", color="#0969da", alpha=.8)
    ax.bar(xs, ev["rebuild"], bottom=ev["merge"], label="rebuild", color="#9a6700", alpha=.8)
    ax.bar(xs, ev["promote"], label="promote", color="#8250df", alpha=.8)
    ax.set_xlabel("轮次"); ax.set_ylabel("事件数")
    ax.set_title("蒸馏/重建/提升事件")
    ax.legend(); ax.grid(alpha=.3)
    charts["events"] = _save_png(fig)

    fig, ax = plt.subplots(figsize=(7, 3.4))
    lat = [r.get("llm_latency_ms", 0) for r in rows]
    ax.bar(xs, lat, color="#57606a", alpha=.8)
    ax.set_xlabel("轮次"); ax.set_ylabel("ms")
    ax.set_title("每轮 LLM 延迟（正文/工具调用）")
    ax.grid(alpha=.3)
    charts["latency"] = _save_png(fig)

    fig, ax = plt.subplots(figsize=(7, 3.4))
    ac = [r["ctx"]["atom_chars"] for r in rows]
    nc = [r["ctx"]["node_chars"] for r in rows]
    sc = [r["ctx"]["shadow_chars"] for r in rows]
    ax.stackplot(xs, ac, nc, sc, labels=["原文", "摘要", "召回影子"], colors=["#0969da", "#9a6700", "#8250df"], alpha=.8)
    ax.set_xlabel("轮次"); ax.set_ylabel("字符")
    ax.set_title("上下文组成（原文/摘要/影子）")
    ax.legend(loc="upper left"); ax.grid(alpha=.3)
    charts["compose"] = _save_png(fig)

    if tree_snaps:
        picks = []
        n = len(tree_snaps)
        for idx in (0, n // 2, n - 1):
            picks.append(tree_snaps[idx])
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
        for ax, snap in zip(axes, picks):
            draw_tree(ax, snap)
        fig.tight_layout()
        charts["tree"] = _save_png(fig)

    return charts


def build_report(outdir, rows, summary, charts, story_path, cfg):
    def im(key):
        if key not in charts:
            return ""
        return f'<img src="data:image/png;base64,{charts[key]}" style="max-width:100%">'

    jsonl_html = ""
    for r in rows:
        jsonl_html += f"<details><summary>轮 {r['round']} · 工具 {r['tools']} · 字数 {r['story_chars']}</summary>" \
                      f"<pre>{json.dumps(r, ensure_ascii=False, indent=1)}</pre></details>"

    html = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>真实 LLM 续写实测报告</title>
<style>body{{font-family:-apple-system,'PingFang SC',sans-serif;max-width:1000px;margin:16px auto;padding:0 16px}}
h1{{font-size:20px}} h2{{font-size:15px;color:#57606a;border-bottom:1px solid #d0d7de;padding-bottom:4px;margin-top:28px}}
table{{border-collapse:collapse;font-size:13px}} td,th{{border:1px solid #d0d7de;padding:4px 10px}}
details{{font-size:12px;border:1px solid #e0e4e8;border-radius:6px;padding:6px;margin:4px 0}}</style></head><body>
<h1>真实 LLM 续写实测报告（水循环 v2）</h1>
<table>
<tr><td>模型</td><td>{cfg.get('provider')} / {cfg.get('model')}</td>
<td>组装预算</td><td>{cfg.get('assembly_budget')} 字符</td></tr>
<tr><td>运行时长</td><td>{summary['elapsed_s']}s（{summary['rounds']} 轮）</td>
<td>累计字数</td><td>{summary['story_chars']} 字（估 {summary['story_tokens']} tokens）</td></tr>
<tr><td>工具调用</td><td>append {summary['tools']['append']} · edit {summary['tools']['edit']} · recall {summary['tools']['recall']}</td>
<td>LLM 调用</td><td>{summary['llm_calls']} 次（正文+工具）</td></tr>
<tr><td>蒸馏</td><td>{summary['distill_calls']} 次 LLM，{summary['distill_seconds']:.1f}s</td>
<td>tokens</td><td>prompt {summary['prompt_tokens']} · completion {summary['completion_tokens']}</td></tr>
<tr><td>重复句</td><td>{summary['duplicates']}（启发式检测）</td>
<td>完整正文</td><td><a href="story_full.txt">story_full.txt</a></td></tr>
<tr><td>树快照</td><td><a href="tree_snapshots.jsonl">tree_snapshots.jsonl</a>（每轮完整树结构）</td>
<td>响应原文</td><td><a href="responses.txt">responses.txt</a></td></tr>
</table>
<h2>图表</h2>
{im('tree')}
{im('story')}
{im('ctx')}
{im('mem')}
{im('compose')}
{im('events')}
{im('latency')}
<h2>逐轮记录（JSONL 原始）</h2>
{jsonl_html}
</body></html>"""
    with open(os.path.join(outdir, "report.html"), "w", encoding="utf-8") as f:
        f.write(html)
    with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "charts": list(charts.keys())}, f, ensure_ascii=False, indent=1)
