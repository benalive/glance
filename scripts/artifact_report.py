"""One table for the AI-image error benchmarks: every model with a per-item file for each benchmark
(zero-shot floors in results/r0/frontier/, trained Glance runs in results/r<n>/<run>/eval/).

Yes/no benchmarks: AUROC of the yes-vs-no logit gap on the report split (threshold-free, so
calibration does not matter), and accuracy after Platt scaling fitted on the calib split. The
plausibility score: Spearman correlation between the expected level and the raters' mean score.

    uv run python scripts/artifact_report.py   # -> results/everyday/artifacts.{md,json}
"""
import glob
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from glance.metrics import auroc, fit_platt

BENCHES = ["had_val_hands", "had_val_any", "richhf_test_artifacts", "richhf_test_plausibility",
           "had_val_hands_anatomy_correct", "had_val_any_looks_correct", "had_val_any_anatomy_correct",
           "richhf_test_looks_correct", "had_val_any_anatomically_off", "had_val_any_people_realistic",
           "real_photos_no_errors"]


def files(bench):
    out = {Path(f).name.split("__")[0]: f for f in glob.glob(f"results/r0/frontier/*__{bench}.jsonl")}
    for f in glob.glob(f"results/r[1-9]/*/eval/*__{bench}.jsonl"):
        out[f"glance {Path(f).parent.parent.name}"] = f
    return out


def score_file(path, bench):
    recs = [json.loads(line) for line in open(path)]
    if bench.endswith("plausibility"):
        exp = []
        for r in recs:
            z = np.array(r["logits"], dtype=np.float64)
            p = np.exp(z - z.max())
            exp.append(float(np.dot(np.arange(len(p)), p / p.sum())))
        return {"srcc": float(spearmanr(exp, [r["mos"] for r in recs]).statistic), "n": len(recs)}
    if bench == "real_photos_no_errors":  # one class only: share of real photos flagged as flawed (raw argmax)
        wrong = [int(np.argmax(r["logits"])) != int(np.argmax(r["target"])) for r in recs]
        return {"false_alarms": float(np.mean(wrong)), "n": len(recs)}
    rep = [r for r in recs if r["split"] == "report"]
    cal = [r for r in recs if r["split"] == "calib"]
    a, b = fit_platt([r["logits"] for r in cal], [r["target"] for r in cal])
    gap = [r["logits"][0] - r["logits"][1] for r in rep]
    yes = [r["target"][0] > 0.5 for r in rep]
    acc = np.mean([(1 / (1 + math.exp(-(a * g + b))) >= 0.5) == y for g, y in zip(gap, yes)])
    return {"auroc": auroc(gap, yes), "acc": float(acc), "share_yes": float(np.mean(yes)), "n": len(rep)}


def main():
    table = {b: {m: score_file(f, b) for m, f in sorted(files(b).items())} for b in BENCHES}
    models = sorted({m for t in table.values() for m in t})
    out = Path("results/everyday")
    out.mkdir(parents=True, exist_ok=True)
    (out / "artifacts.json").write_text(json.dumps(table, indent=1))
    lines = ["Visible errors in AI-generated images. Yes/no: AUROC / accuracy (Platt fitted on the calib split) on the "
             "report split; plausibility: SRCC with the raters' mean score. Chance: AUROC 0.5, SRCC 0.", "",
             "| benchmark | n | share yes | " + " | ".join(models) + " |", "|---|---|---|" + "---|" * len(models)]
    for b, t in table.items():
        any_row = next(iter(t.values()), {})
        n, share = any_row.get("n", 0), any_row.get("share_yes")
        cells = []
        for m in models:
            r = t.get(m)
            cells.append("—" if r is None else f"{r['srcc']:.3f}" if "srcc" in r else
                         f"{r['false_alarms']:.3f} false alarms" if "false_alarms" in r else f"{r['auroc']:.3f} / {r['acc']:.3f}")
        lines.append(f"| {b} | {n} | {'' if share is None else f'{share:.2f}'} | " + " | ".join(cells) + " |")
    (out / "artifacts.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
