"""Figures for paper/glance.tex, drawn from committed results files only.

Palette: the validated categorical slots 1-3 (blue #2a78d6, orange #eb6834, aqua #1baf7a; all-pairs
CVD dE >= 9.2, normal-vision >= 24.0 in light mode). Aqua is below 3:1 contrast on the surface, so
every mark carries a direct label. Colour follows the entity (a width, a system), never its rank.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
OUT = Path("paper/figures")
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "font.family": "DejaVu Sans"})


def _axes(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def latency_depth():
    d = json.loads(Path("results/r0/latency_sweep.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.3), sharey=False)
    for ax, dev, title in [(axes[0], "mps", "Apple M4 Pro GPU (MPS, fp16)"), (axes[1], "cpu", "Apple M4 Pro CPU (4 threads, fp32)")]:
        for width, color in [(384, BLUE), (768, ORANGE)]:
            pts = sorted((r["depth"], r["p50_ms"]) for r in d[dev]["grid"] if r["width"] == width and r["tokens"] == 128)
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=4, markeredgecolor="white", markeredgewidth=0.8)
            ax.annotate(f"width {width}", (xs[-1], ys[-1]), xytext=(4, 0), textcoords="offset points",
                        va="center", color=INK, fontsize=7)
        f = d[dev]["fit"]
        ax.set_title(f"{title}\nfit: {f['a_ms_per_layer']:.2f} ms/layer + {f['b_ms_per_gflop']:.2f} ms/GFLOP (R$^2$ {f['r2']:.3f})",
                     fontsize=7.5, color=INK)
        ax.set_xlabel("sequential layers")
        ax.set_xticks([2, 4, 8, 12, 22])
        ax.set_xlim(0, 27)
        ax.set_ylim(0, None)
        _axes(ax)
    axes[0].set_ylabel("p50 latency (ms)")
    fig.tight_layout()
    fig.savefig(OUT / "latency_depth.pdf")


def snake_speed_quality():
    recs = [json.loads(l) for l in open("results/bench/snake_cpu.jsonl") if not json.loads(l).get("contended")]

    def latest(sel):
        return [r for r in recs if sel(r)][-1]

    points = [  # label, colour (entity: system), record, label offset (pt), alignment
        ("Laya-snake (322M)", ORANGE, latest(lambda r: r["runtime"] == "laya-torch" and "snake" in r["model"]
                                             and r["threads"] == 4 and "B_layastudio" in r), (-6, -13), "right"),
        ("C5 text (40M)", BLUE, latest(lambda r: r.get("run") == "C5_text_matched"), (0, -14), "center"),
        ("C5 image + facts", BLUE, latest(lambda r: r.get("run") == "C5_img_facts_matched"), (0, 8), "center"),
        ("C6s + aux\n(image only)", BLUE, latest(lambda r: r.get("run") == "C6s_img_aux"), (0, 9), "center"),
        ("C3 + aux\n(image only)", BLUE, latest(lambda r: r.get("run") == "C3_img_aux"), (0, 9), "center"),
    ]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for label, color, r, (dx, dy), ha in points:
        x, y = r["A_30s"]["p50_ms"], r["B_layastudio"]["moves_median"]
        ax.scatter([x], [y], s=34, color=color, edgecolor="white", linewidth=1.2, zorder=3)
        ax.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=6.5, color=INK, ha=ha)
    from matplotlib.lines import Line2D

    ax.legend(handles=[Line2D([], [], marker="o", linestyle="", color=c, markeredgecolor="white", label=n)
                       for n, c in [("Laya", ORANGE), ("Glance (ours)", BLUE)]],
              loc="center right", frameon=False, fontsize=7, labelcolor=INK)
    ax.set_xlabel("CPU latency per decision, p50 (ms)")
    ax.set_ylabel("moves survived, median of 10 games")
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 230)
    _axes(ax)
    fig.tight_layout()
    fig.savefig(OUT / "snake_speed_quality.pdf")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    latency_depth()
    snake_speed_quality()
    print("wrote", sorted(p.name for p in OUT.iterdir()))
