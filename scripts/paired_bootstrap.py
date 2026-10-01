"""Paired, image-grouped bootstrap of the difference between two runs on one benchmark.

Both runs' per-item files (<run>/eval/<cand>__<bench>.jsonl) are matched by uid. Metric: AUROC of the raw
yes-vs-no logit gap (threshold-free; polarity comes from the gold label, so negated wordings are handled),
or accuracy with a stated calibration: "platt" fits Platt scaling on the benchmark's own calib split (a
threshold tuned on the benchmark: label it so, and never use it next to published rows), "package" uses a
release package's calibration.json, "argmax" none. Items are resampled by image, since questions on one
image are correlated. Reports both values, the difference and its 95% interval.

    uv run python scripts/paired_bootstrap.py --a results/r4/C5_open3_s0 --b results/r2/C5_distill_ep2 \
        --bench salart_q1 --metric auroc --split all
"""
import argparse
import json
from pathlib import Path

import numpy as np

from glance.metrics import auroc, fit_platt, majority
from glance.serve import calibrated_probs


def load(run: str, bench: str) -> dict:
    cand = json.loads((Path(run) / "config.json").read_text())["cand"]
    return {json.loads(l)["uid"]: json.loads(l) for l in open(Path(run) / "eval" / f"{cand}__{bench}.jsonl")}


def correct(recs: dict, uids: list, how: str, package: str | None) -> dict:
    if how == "platt":
        cal = [r for r in recs.values() if r["split"] == "calib"]
        a, b = fit_platt([r["logits"] for r in cal], [r["target"] for r in cal])
        calib = {"noul": {"a": a, "b": b}}
    elif how == "package":
        calib = json.loads((Path(package) / "calibration.json").read_text())["calibration"]
    else:
        calib = {}
    return {u: float(np.argmax(calibrated_probs(calib, recs[u]["kind"], np.array(recs[u]["logits"], float)))
                     == majority(recs[u]["target"])) for u in uids}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--metric", choices=["auroc", "platt", "package", "argmax"], default="auroc")
    ap.add_argument("--package", help="release package for --metric package")
    ap.add_argument("--split", choices=["report", "all"], default="report")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    A, B = load(a.a, a.bench), load(a.b, a.bench)
    uids = sorted(u for u in A if u in B and (a.split == "all" or A[u]["split"] == "report")
                  and majority(A[u]["target"]) is not None)  # items with a majority answer (VQAv2 has 5-5 ties)
    by_img = {}
    for u in uids:
        by_img.setdefault(A[u]["image_id"], []).append(u)
    keys = sorted(by_img)
    if a.metric == "auroc":
        def stat(recs, us):
            return auroc([recs[u]["logits"][0] - recs[u]["logits"][1] for u in us], [recs[u]["target"][0] > 0.5 for u in us])
        fa, fb = (lambda us: stat(A, us)), (lambda us: stat(B, us))
    else:
        ca, cb = correct(A, uids, a.metric, a.package), correct(B, uids, a.metric, a.package)
        fa, fb = (lambda us: float(np.mean([ca[u] for u in us]))), (lambda us: float(np.mean([cb[u] for u in us])))
    rng = np.random.default_rng(a.seed)
    diffs = []
    for _ in range(a.n):
        us = [u for k in rng.choice(keys, len(keys)) for u in by_img[k]]
        diffs.append(fa(us) - fb(us))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    va, vb = fa(uids), fb(uids)
    print(json.dumps({"bench": a.bench, "metric": a.metric, "split": a.split, "n_items": len(uids), "n_images": len(keys),
                      "a": [a.a, round(va, 4)], "b": [a.b, round(vb, 4)], "diff": round(va - vb, 4), "ci95": [round(lo, 4), round(hi, 4)]}))


if __name__ == "__main__":
    main()
