"""Every run and recipe the paper compares, from the committed per-item files, with one set of rules.

For comparing runs with each other (not with published results): yes/no accuracy uses Platt scaling fitted
on the benchmark's own calibration split and is scored on its report split ("threshold fitted on the
benchmark"); choice accuracy is the argmax; AUROC is threshold-free. Items without a majority answer
(VQAv2's 5-5 splits) are left out of accuracy and AUROC. Recipes with several seeds give mean and sample sd.
SalArt and ArtiBench are scored on all items. Release-model numbers next to published results come from
scripts/comparison_report.py (package calibration) instead.

    uv run python scripts/recipe_table.py     # -> results/recipes.{md,json}
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from glance.metrics import auroc, fit_platt, majority

RUNS = {  # label -> run dirs (seeds)
    "R1 C1 (research)": ["results/r1/C1_ep2_lrA_s0"],
    "R1 C5 (research)": ["results/r1/C5_ep2_lrA_s0"],
    "R1 C7 (early fusion, small)": ["results/r1/C7_ep2_lrA_s0"],
    "R1 C8 (B/16, 256 tokens)": ["results/r1/C8_ep2_lrA_s0"],
    "C5 research, errors (round 2)": ["results/r2/C5_art2_ep2"],
    "C5 research, distilled from C1": ["results/r2/C5_distill_ep2"],
    "C5 research, same data no teacher": ["results/r2/C5_gold48k_ep2"],
    "C1 research, errors (teacher)": ["results/r2/C1_art2_ep2"],
    "C9 (C1 cut to 192 px, 6 layers)": ["results/r2/C9_distill_ep2"],
    "clean C5": ["results/r3/C5_clean_ep2"],
    "clean C5f6": ["results/r3/C5f6_clean_ep2"],
    "clean C5u4 (lr 5e-5)": ["results/r3/C5u4_clean_ep2"],
    "clean C5u4 (image lr 1e-5)": ["results/r3/C5u4_lrimg1e-5_clean_ep2"],
    "clean C5 + Qwen teacher": ["results/r3/C5_qwen_clean_ep2"],
    "clean errors, two-stage v1": ["results/r3/C5_art_clean_ep2"],
    "one-stage v2, Qwen teacher": [f"results/r4/C5_open_s{s}" for s in (0, 1, 2)],
    "one-stage v2, no teacher": ["results/r4/C5_open_noteacher_s0"],
    "one-stage v2, 4 epochs": ["results/r4/C5_open_ep4_s0"],
    "two-stage v2": [f"results/r4/C5_open_2stage_s{s}" for s in (0, 1, 2)],
    "two-stage v3 (templated wordings)": [f"results/r4/C5_open3_s{s}" for s in (0, 1, 2)],
    "two-stage v4 (+ face crops; released recipe)": [f"results/r4/C5_open4_s{s}" for s in (0, 1, 2)],
    "two-stage v4 seed 0": ["results/r4/C5_open4_s0"],
    "two-stage v5 (+ procedural local anomalies) seed 0": ["results/r4/C5_open5_s0"],
}
ACC = ["vqav2_val_yesno", "pope_adversarial", "aokvqa_val", "sugarcrepe_replace_rel", "sugarcrepe_swap_att"]
AUC = ["vqav2_val_yesno", "had_val_hands", "had_val_any", "richhf_test_artifacts", "had_val_any_anatomically_off", "salart_q1", "artibench"]
ALL_ITEMS = {"salart_q1", "artibench"}


def records(run: str, bench: str):
    cand = json.loads((Path(run) / "config.json").read_text())["cand"]
    path = Path(run) / "eval" / f"{cand}__{bench}.jsonl"
    if not path.exists():
        return None
    return [r for r in (json.loads(line) for line in open(path)) if majority(r["target"]) is not None or bench == "koniq_test"]


def metrics(run: str) -> dict:
    out = {}
    for b in ACC:
        recs = records(run, b)
        if not recs:
            continue
        rep = [r for r in recs if r["split"] == "report"]
        if len(rep[0]["logits"]) == 2:
            cal = [r for r in recs if r["split"] == "calib"]
            a, c = fit_platt([r["logits"] for r in cal], [r["target"] for r in cal])
            pred = [int(a * (r["logits"][0] - r["logits"][1]) + c < 0) for r in rep]
        else:
            pred = [int(np.argmax(r["logits"])) for r in rep]
        out[f"acc:{b}"] = float(np.mean([p == majority(r["target"]) for p, r in zip(pred, rep)]))
    for b in AUC:
        recs = records(run, b)
        if not recs:
            continue
        recs = recs if b in ALL_ITEMS else [r for r in recs if r["split"] == "report"]
        out[f"auroc:{b}"] = auroc([r["logits"][0] - r["logits"][1] for r in recs], [r["target"][0] > 0.5 for r in recs])
    recs = records(run, "koniq_test")
    if recs:
        rep = [r for r in recs if r["split"] == "report"]
        exp = []
        for r in rep:
            z = np.array(r["logits"], float)
            p = np.exp(z - z.max())
            exp.append(float(np.dot(np.arange(len(p)), p / p.sum())))
        out["srcc:koniq_test"] = float(spearmanr(exp, [r["mos"] for r in rep]).statistic)
    return out


def main():
    table = {}
    for label, runs in RUNS.items():
        per = [metrics(r) for r in runs if Path(r).exists()]
        keys = sorted({k for m in per for k in m})
        table[label] = {"runs": runs, "n_seeds": len(per),
                        **{k: {"mean": float(np.mean([m[k] for m in per if k in m])),
                               "sd": float(np.std([m[k] for m in per if k in m], ddof=1)) if len(per) > 1 else None,
                               "values": [m.get(k) for m in per]} for k in keys}}
    Path("results/recipes.json").write_text(json.dumps(table, indent=1))
    cols = [f"acc:{b}" for b in ACC] + ["srcc:koniq_test"] + [f"auroc:{b}" for b in AUC]
    short = {"acc:vqav2_val_yesno": "VQAv2 acc*", "acc:pope_adversarial": "POPE-adv acc*", "acc:aokvqa_val": "A-OKVQA",
             "acc:sugarcrepe_replace_rel": "SC-rel", "acc:sugarcrepe_swap_att": "SC-att", "srcc:koniq_test": "KonIQ",
             "auroc:vqav2_val_yesno": "VQAv2 AUC", "auroc:had_val_hands": "HAD hands", "auroc:had_val_any": "HAD any",
             "auroc:richhf_test_artifacts": "RichHF", "auroc:had_val_any_anatomically_off": "'anat. off'",
             "auroc:salart_q1": "SalArt", "auroc:artibench": "ArtiBench"}
    L = ["Runs compared with each other. *yes/no accuracy with a threshold fitted on the benchmark's calibration split; "
         "AUC columns threshold-free; items without a majority answer excluded; mean ± sd over seeds where several.", "",
         "| run | seeds | " + " | ".join(short[c] for c in cols) + " |", "|---|---|" + "---|" * len(cols)]
    for label, row in table.items():
        cells = []
        for c in cols:
            v = row.get(c)
            cells.append("—" if v is None else f"{v['mean']:.3f}" + (f" ± {v['sd']:.3f}" if v["sd"] is not None else ""))
        L.append(f"| {label} | {row['n_seeds']} | " + " | ".join(cells) + " |")
    Path("results/recipes.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
