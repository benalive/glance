"""Are real photos of some groups of people wrongly flagged as flawed more often than others?

FairFace validation (HF HuggingFaceM4/FairFace, "1.25" crops with context; CC BY 4.0; Karkkainen & Joo,
WACV 2021): 10,954 real photos labelled with race (7 groups), gender and age group. Every photo is a real
photograph, so any "has errors" answer is a false alarm. The model answers three questions per photo
(visible generation errors, deformed body parts, distorted face) through the served path with the
package's calibration; the false-alarm rate (P(error) >= 0.5) is reported per group with a Wilson 95%
interval, plus the mean P(error).

    uv run python scripts/fairness_check.py --package data/release/glance-c5-open   # -> results/fairness/
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from glance.serve import Predictor

QUESTIONS = {"errors": ("Does this image contain visible generation errors?", 1),
             "body": ("Are there any deformed body parts in this image?", 1),
             "face": ("Is anyone's face distorted or malformed?", 1)}


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p, z = k / n, 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (c - h, c + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", help="release package dir")
    ap.add_argument("--ckpt", help="or a trained checkpoint (with --cand and --calib)")
    ap.add_argument("--cand", default="C5")
    ap.add_argument("--calib", help="calibration dir for --ckpt (in-format evaluation files)")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    path = hf_hub_download("HuggingFaceM4/FairFace", "1.25/validation-00000-of-00001-09e3e67bb00ab4ec.parquet", repo_type="dataset")
    table = pq.read_table(path)
    feats = table.schema.metadata  # ClassLabel names live in the card; decode from the HF features if present
    names = json.loads(feats[b"huggingface"])["info"]["features"] if feats and b"huggingface" in feats else {}
    label = {k: names.get(k, {}).get("names") for k in ("age", "gender", "race")}
    rows = table.to_pylist()[: a.limit or None]
    pred = Predictor(a.package, device="cpu", threads=4) if a.package else Predictor(a.ckpt, a.cand, a.calib, "cpu", 4)
    name = Path(a.package or a.ckpt).stem
    qs = {k: {"type": "noul", "instructions": q} for k, (q, _) in QUESTIONS.items()}
    p = {k: [] for k in QUESTIONS}
    for i, r in enumerate(rows):
        ans = pred.predict(qs, r["image"]["bytes"])["answers"]
        pred.cache.clear()
        for k in QUESTIONS:
            p[k].append(ans[k]["noul"])
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{len(rows)}", flush=True)
    out = {"n": len(rows), "questions": {k: q for k, (q, _) in QUESTIONS.items()}, "by": {}}
    for attr in ("race", "gender", "age"):
        groups = defaultdict(list)
        for j, r in enumerate(rows):
            g = r[attr]
            groups[label[attr][g] if label[attr] else str(g)].append(j)
        out["by"][attr] = {}
        for g, idx in sorted(groups.items()):
            row = {"n": len(idx)}
            for k in QUESTIONS:
                flags = sum(p[k][j] >= 0.5 for j in idx)
                row[k] = {"false_alarm": flags / len(idx), "ci95": wilson(flags, len(idx)), "mean_p": float(np.mean([p[k][j] for j in idx]))}
            out["by"][attr][g] = row
    dest = Path("results/fairness")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"{name}.json").write_text(json.dumps(out, indent=1))
    lines = [f"FairFace validation ({len(rows)} real photos), {name}: share of photos flagged as flawed "
             "(P >= 0.5), with Wilson 95% interval, per group.", ""]
    for attr, groups in out["by"].items():
        lines += [f"### {attr}", "", "| group | n | " + " | ".join(QUESTIONS) + " |", "|---|---|" + "---|" * len(QUESTIONS)]
        for g, row in groups.items():
            lines.append(f"| {g} | {row['n']} | " + " | ".join(
                f"{100 * row[k]['false_alarm']:.1f}% [{100 * row[k]['ci95'][0]:.1f}, {100 * row[k]['ci95'][1]:.1f}]" for k in QUESTIONS) + " |")
        lines.append("")
    (dest / f"{name}.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
