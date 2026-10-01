"""Breakdown of the SalArt clean-originals test (results/salart_clean_ai_items.jsonl; no new inference).

Per model: within-pair win rate (share of the 119 pairs where the flawed image gets the higher P(yes), with a
Wilson 95% interval: a paired statistic, unlike the pooled AUROC of flawed vs original); flawed vs clean AUROC
within Imagen 4 only (the pooled flawed-vs-clean-AI AUROC mixes generators, and 96 of the 119 clean originals
are Imagen 4); and, for calibrated models at the served threshold (P >= 0.5), recall on flawed images by
generator and by SalArt artifact type, specificity on clean generated originals, and on real photos.

    uv run python scripts/salart_breakdown.py     # -> results/salart_breakdown.{json,md}
"""
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from glance.data.benchmarks import EXTERNAL
from glance.metrics import auroc

GENS = ["imagen4", "flux2_klein", "flux1_dev", "z_turbo", "qwen"]


def generator(row_id: str) -> str:
    return next((g for g in GENS if g in row_id), "other")


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [c - h, c + h]


def main():
    meta = {}
    for line in open(EXTERNAL / "salart" / "data" / "test.jsonl"):
        r = json.loads(line)
        meta[r["row_id"]] = {"gen": generator(r["row_id"]), "type": r["artifact_type"]}
    items = [json.loads(l) for l in open("results/salart_clean_ai_items.jsonl")]
    by_model = defaultdict(list)
    for r in items:
        by_model[r["model"]].append(r)
    out = {}
    for model, rows in by_model.items():
        clean = {r["pair_id"]: r["p_yes"] for r in rows if r["group"] == "clean_ai"}
        pairs = [(r["p_yes"], clean[r["pair_id"]], meta[r["uid"]]["gen"]) for r in rows if r["group"] == "flawed" and r["pair_id"] in clean]
        wins = sum(a > b for a, b, _ in pairs)
        im_f = [r["p_yes"] for r in rows if r["group"] == "flawed" and meta[r["uid"]]["gen"] == "imagen4"]
        im_c = [r["p_yes"] for r in rows if r["group"] == "clean_ai" and meta[r["uid"]]["gen"] == "imagen4"]
        m = {"pairs": len(pairs), "pair_wins": wins, "pair_win_rate": wins / len(pairs), "pair_win_ci95": wilson(wins, len(pairs)),
             "imagen4_flawed_vs_clean_auroc": auroc(im_f + im_c, [1] * len(im_f) + [0] * len(im_c)),
             "n_imagen4": [len(im_f), len(im_c)]}
        if "no calibration" not in model:
            rec = defaultdict(list)
            typ = defaultdict(list)
            for r in rows:
                if r["group"] == "flawed":
                    rec[meta[r["uid"]]["gen"]].append(r["p_yes"] >= 0.5)
                    typ[meta[r["uid"]]["type"]].append(r["p_yes"] >= 0.5)
            m["recall_by_generator"] = {g: [float(np.mean(v)), len(v)] for g, v in sorted(rec.items())}
            m["recall_by_type"] = {t: [float(np.mean(v)), len(v)] for t, v in sorted(typ.items())}
            m["specificity_real"] = float(np.mean([r["p_yes"] < 0.5 for r in rows if r["group"] == "real"]))
            m["specificity_clean_ai"] = float(np.mean([r["p_yes"] < 0.5 for r in rows if r["group"] == "clean_ai"]))
            m["specificity_all_unflawed"] = float(np.mean([r["p_yes"] < 0.5 for r in rows if r["group"] != "flawed"]))
        out[model] = m
    Path("results/salart_breakdown.json").write_text(json.dumps(out, indent=1))
    L = ["SalArt clean-originals test, breakdown (served path; from results/salart_clean_ai_items.jsonl).", "",
         "| model | pair win rate [95% CI] | Imagen 4 flawed vs clean AUROC | specificity: real / clean AI / all unflawed |", "|---|---|---|---|"]
    for model, m in out.items():
        sp = f"{m['specificity_real']:.3f} / {m['specificity_clean_ai']:.3f} / {m['specificity_all_unflawed']:.3f}" if "specificity_real" in m else "—"
        L.append(f"| {model} | {m['pair_win_rate']:.3f} ({m['pair_wins']}/{m['pairs']}) [{m['pair_win_ci95'][0]:.2f}, {m['pair_win_ci95'][1]:.2f}] | "
                 f"{m['imagen4_flawed_vs_clean_auroc']:.3f} | {sp} |")
    for model, m in out.items():
        if "recall_by_generator" in m:
            L += ["", f"Recall on flawed images at the served threshold, {model}:", "",
                  "- by generator: " + "; ".join(f"{g} {v:.2f} (n={n})" for g, (v, n) in m["recall_by_generator"].items()),
                  "- by artifact type: " + "; ".join(f"{t} {v:.2f} (n={n})" for t, (v, n) in m["recall_by_type"].items())]
    Path("results/salart_breakdown.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
