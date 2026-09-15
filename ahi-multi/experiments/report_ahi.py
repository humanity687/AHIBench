#!/usr/bin/env python3
"""report_ahi.py — 实验后图表自动导出（后端 matplotlib，零浏览器依赖）。

读 agents/{lan,neutral}/data/metrics.jsonl → PNG 图 + 汇总 JSON → out/report/。
Phase B 使用：实验结束后一键归档图表（dashboard 是实时观察，此脚本是归档）。

用法：python3 report_ahi.py [--out experiments/out/report] [--watch 600]
  --watch N：实验期间每 N 秒增量出图（替代手动截图）
"""
import argparse
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
OUT_DEFAULT = os.path.join(HERE, "out", "report")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # macOS CJK 字体（防中文方框）
    matplotlib.rcParams["font.sans-serif"] = [
        "Arial Unicode MS", "PingFang SC", "Hiragino Sans GB", "Heiti SC",
        "STHeiti", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def load_metrics(agent):
    p = os.path.join(PROJ, "agents", agent, "data", "metrics.jsonl")
    rows = []
    if os.path.isfile(p):
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


def series(rows, key, default=None):
    return [r.get(key, default) for r in rows]


def make_charts(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    data = {a: load_metrics(a) for a in ("lan", "neutral")}
    summary = {}
    if not HAS_MPL:
        print("matplotlib 不可用，只导出汇总 JSON")
        for a, rows in data.items():
            if rows:
                summary[a] = rows[-1]
        with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=1)
        return summary

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()

    def plot(ax, key, title, ylabel):
        for a, rows in data.items():
            if not rows:
                continue
            ax.plot(series(rows, "round"), series(rows, key, 0), label=a, linewidth=1.2)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plot(axes[0], "pending", "pending 积压", "条")
    plot(axes[1], "ctx_chars", "组装上下文（字符）", "字符")
    plot(axes[2], "atoms_alive", "存活原子数", "个")
    plot(axes[3], "tiling", "铺陈项数", "项")
    plot(axes[4], "fact_drift", "fact_drift 累计", "次")
    plot(axes[5], "person_source_violations", "人称-来源绑定违规累计", "次")

    fig.suptitle("AHI 实验指标 · 自动导出", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    png = os.path.join(out_dir, "charts.png")
    fig.savefig(png, dpi=110)
    plt.close(fig)
    print(f"图表: {png}")

    for a, rows in data.items():
        if rows:
            summary[a] = rows[-1]
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--watch", type=int, default=0, help="每 N 秒增量出图（0=一次）")
    args = ap.parse_args()
    if args.watch <= 0:
        make_charts(args.out)
        return
    while True:
        try:
            make_charts(args.out)
        except Exception as e:
            print(f"出图失败: {e}")
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
