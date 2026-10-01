"""How well calibrated is a release package on each benchmark, with its own (licence-clean) calibration?

Per benchmark: accuracy of the top answer, mean confidence (top probability), smooth ECE (glance.metrics.
smooth_ece), Brier score and, for yes/no benchmarks, recall and specificity of "yes" at the served
threshold, on the report split (SalArt and ArtiBench: all items, as in the comparison table).
Also the per-seed spread of the released recipe on the benchmarks the paper reports.

    uv run python scripts/release_calibration.py      # -> results/release_calibration.{md,json}
"""
import json
from pathlib import Path

import numpy as np

from glance.metrics import auroc, brier, majority, smooth_ece
from glance.serve import calibrated_probs

PACKAGE = Path("data/release/glance-c5-open-v4")
RUN = "results/r4/C5_open4_s{seed}/eval/C5__{b}.jsonl"
BENCHES = ["vqav2_val_yesno", "pope_adversarial", "pope_random", "aokvqa_val", "mmstar", "sugarcrepe_replace_rel",
           "sugarcrepe_swap_att", "salart_q1", "artibench", "had_val_hands", "had_val_any", "richhf_test_artifacts"]
ALL_ITEMS = {"salart_q1", "artibench"}


def load(seed, b):
    recs = [json.loads(line) for line in open(RUN.format(seed=seed, b=b))]
    return recs if b in ALL_ITEMS else [r for r in recs if r["split"] == "report"]


def main():
    calib = json.loads((PACKAGE / "calibration.json").read_text())["calibration"]
    rows, spread = {}, {}
    for b in BENCHES:
        recs = [r for r in load(0, b) if majority(r["target"]) is not None]
        ps = [calibrated_probs(calib, r["kind"], np.array(r["logits"], float)) for r in recs]
        ys = [r["target"] for r in recs]
        correct = [float(np.argmax(p) == majority(y)) for p, y in zip(ps, ys)]
        conf = [float(np.max(p)) for p in ps]
        rows[b] = {"n": len(recs), "acc": float(np.mean(correct)), "mean_conf": float(np.mean(conf)),
                   "smece": float(smooth_ece(np.array(conf), np.array(correct))), "brier": float(brier(ps, ys))}
        if recs[0]["kind"] == "noul":
            pos = [p[0] >= 0.5 for p, y in zip(ps, ys) if y[0] > 0.5]
            neg = [p[0] < 0.5 for p, y in zip(ps, ys) if y[0] <= 0.5]
            rows[b] |= {"recall_yes": float(np.mean(pos)), "specificity": float(np.mean(neg))}
        if len(recs[0]["logits"]) == 2:
            spread[b] = [auroc([r["logits"][0] - r["logits"][1] for r in recs_s], [r["target"][0] > 0.5 for r in recs_s])
                         for recs_s in ([r for r in load(s, b) if majority(r["target"]) is not None] for s in (0, 1, 2))]
    real = [json.loads(line) for line in open(RUN.format(seed=0, b="real_photos_no_errors"))]
    flagged = {}
    for r in real:  # every item is a real photo; a flag is a top answer other than the gold "no error" answer
        wrong = int(np.argmax(calibrated_probs(calib, r["kind"], np.array(r["logits"], float)))) != int(np.argmax(r["target"]))
        flagged[r["image_id"]] = flagged.get(r["image_id"], False) or wrong
        rows.setdefault("_real_photos", {"questions": 0, "flagged_questions": 0})
        rows["_real_photos"]["questions"] += 1
        rows["_real_photos"]["flagged_questions"] += int(wrong)
    rows["_real_photos"] |= {"photos": len(flagged), "flagged_photos": sum(flagged.values())}
    out = Path("results")
    (out / "release_calibration.json").write_text(json.dumps({"calibration": rows, "auroc_by_seed": spread}, indent=1))
    L = ["Release package glance-c5-open-v4 (seed 0) with its own calibration: accuracy, mean confidence, smooth ECE, Brier.", "",
         "| benchmark | n | accuracy | mean confidence | smECE | Brier | recall (yes) | specificity |", "|---|---|---|---|---|---|---|---|"]
    for b, r in rows.items():
        if b.startswith("_"):
            continue
        L.append(f"| {b} | {r['n']} | {r['acc']:.3f} | {r['mean_conf']:.3f} | {r['smece']:.3f} | {r['brier']:.3f} | "
                 + (f"{r['recall_yes']:.3f} | {r['specificity']:.3f} |" if "recall_yes" in r else "— | — |"))
    rp = rows["_real_photos"]
    L += ["", f"Real photographs (500 COCO, 500 KonIQ; three error questions each): {rp['flagged_questions']} of {rp['questions']} "
          f"questions flagged ({rp['flagged_questions'] / rp['questions']:.1%}), {rp['flagged_photos']} of {rp['photos']} photos flagged "
          f"by at least one question ({rp['flagged_photos'] / rp['photos']:.1%})."]
    L += ["", "AUROC by seed of the released recipe (v4, seeds 0 / 1 / 2):", "", "| benchmark | s0 | s1 | s2 |", "|---|---|---|---|"]
    for b, v in spread.items():
        L.append(f"| {b} | " + " | ".join(f"{x:.3f}" for x in v) + " |")
    (out / "release_calibration.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
