"""Like-for-like Snake table: Laya-snake vs Snake-tuned Glance on the same CPU harness.

Same 12x8 engine, planner data and seed, same unassisted protocols, 4 CPU threads, quiet machine.
Uses the latest non-contended record per configuration from results/bench/snake_cpu.jsonl.
"""
import json
from pathlib import Path

ROWS = [  # (label, input, training boards, selector)
    ("Laya-snake (mmBERT-base, LoRA)", "text: ASCII board + facts", "2,339",
     lambda r: r["runtime"] == "laya-torch" and "snake" in r["model"] and r["threads"] == 4 and "B_layastudio" in r),
    ("Glance C5, text", "text: ASCII board + facts (same as Laya)", "2,339",
     lambda r: r.get("run") == "C5_text_matched"),
    ("Glance C5, image + facts", "board image + 2 fact lines", "2,339",
     lambda r: r.get("run") == "C5_img_facts_matched"),
    ("Glance C5, image + facts", "board image + 2 fact lines", "100k generated",
     lambda r: r.get("run") == "C5_img_facts"),
    ("Glance C6s + aux, image only", "board image only", "100k generated",
     lambda r: r.get("run") == "C6s_img_aux"),
    ("Glance C3 + aux, image only", "board image only", "100k generated",
     lambda r: r.get("run") == "C3_img_aux"),
]
PARAMS = {"Laya-snake (mmBERT-base, LoRA)": "322M", "Glance C5, text": "40M (text path; image features constant)",
          "Glance C5, image + facts": "134M", "Glance C6s + aux, image only": "113M", "Glance C3 + aux, image only": "183M"}


def main():
    recs = [json.loads(l) for l in open("results/bench/snake_cpu.jsonl") if not json.loads(l).get("contended")]
    lines = ["CPU only (Apple M4 Pro, 4 threads, quiet machine). Same engine, planner, seeds and unassisted top-1 "
             "protocols for every row. A = 30 s continuous play; B = 10 games x <=500 ticks.", "",
             "| model | input | params | training boards | p50 / p95 ms | decisions/s | A: food / deaths / best length "
             "| B: moves mean / median | B: score mean / best | B: legal | B: planner agreement |",
             "|" + "---|" * 11]
    for label, inp, boards, sel in ROWS:
        matches = [r for r in recs if sel(r)]
        if not matches:
            continue
        r = matches[-1]
        a, b = r["A_30s"], r.get("B_layastudio", {})
        lines.append(f"| {label} | {inp} | {PARAMS.get(label, '')} | {boards} | {a['p50_ms']:.1f} / {a['p95_ms']:.1f} "
                     f"| {a['decisions_per_s']:.1f} | {a['food_30s']} / {a['deaths']} / {a['best_length']} "
                     f"| {b.get('moves_mean', float('nan')):.1f} / {b.get('moves_median', float('nan')):.1f} "
                     f"| {b.get('score_mean', float('nan')):.1f} / {b.get('score_best', '—')} "
                     f"| {b.get('legal_rate', float('nan')):.3f} | {b.get('teacher_agreement', float('nan')):.3f} |")
    lines += ["", "Planner (the teacher, a ceiling): B moves 418.3 mean, score 34.4, 0.1 ms per decision."]
    Path("results/bench/snake_laya_vs_glance.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
