"""Table for results/bench/snake_cpu.jsonl next to the article's figures (cited, not measured)."""
import json
from pathlib import Path

from scripts.snake_cpu_bench import ARTICLE_JEV, ARTICLE_LAYA_MLX_M3MAX

NAMES = {"madhavbiplov/laya-snake-mlx": "Laya-snake 322M (Snake-tuned)",
         "convaiinnovations/laya-multilingual": "Laya multilingual 322M (untuned)",
         "convaiinnovations/laya": "Laya English 421M (untuned)", "teacher": "Planner (ceiling)",
         "C6": "Glance C6 6x512 (image in, untrained)", "C3": "Glance C3 12x768 (image in, untrained)",
         "C5": "Glance C5 fusion (image in, untrained)"}


def main(path="results/bench/snake_cpu.jsonl", out="results/bench/snake_cpu.md"):
    rows = [json.loads(l) for l in Path(path).open() if not json.loads(l).get("contended")]
    lines = [f"CPU only, {rows[0]['chip']}. A = 30 s continuous play (new game on death); B = LayaStudio's 10 games "
             "x <=500 ticks, unassisted top-1 move. 'glance' rows are untrained, timed on the planner's boards (latency only); "
             "'glance-snake' rows are Snake-tuned and play. Runs timed while another job used the machine are excluded.", "",
             "| model | runtime | threads | A: decisions/s | A: p50 / p95 ms | A: food in 30 s | A: best length | A: deaths "
             "| B: moves mean | B: score mean | B: legal | B: teacher agree |", "|" + "---|" * 12]
    for r in rows:
        a, b = r.get("A_30s", {}), r.get("B_layastudio", {})
        fa = lambda k, f="{:.1f}": f.format(a[k]) if k in a else "—"  # noqa: E731
        fb = lambda k, f="{:.1f}": f.format(b[k]) if k in b else "—"  # noqa: E731
        lat = f"{a['p50_ms']:.1f} / {a['p95_ms']:.1f}" if "p50_ms" in a else "—"
        name = NAMES.get(r["model"], r["model"]) + (f" [{r['run']}]" if r.get("run") else "")
        if r.get("readout", "").startswith("slot"):
            name += " (slot readout, superseded)"
        lines.append(f"| {name} | {r['runtime']} | {r['threads']} | {fa('decisions_per_s')} "
                     f"| {lat} | {fa('food_30s', '{}')} | {fa('best_length', '{}')} | {fa('deaths', '{}')} "
                     f"| {fb('moves_mean')} | {fb('score_mean')} | {fb('legal_rate', '{:.3f}')} | {fb('teacher_agreement', '{:.3f}')} |")
    for ref in (ARTICLE_LAYA_MLX_M3MAX, ARTICLE_JEV):
        lines.append(f"| {ref.get('model', 'Laya 421M')} | {ref['runtime']} | — | {ref['decisions_per_s']} | {ref['p50_ms']} / — "
                     f"| {ref['food_30s']} | {ref['length']} | — | — | — | — | — |")
    Path(out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
