"""Side-by-side accuracy on everyday VQAv2 yes/no questions, split by phrasing, for every model that
has a per-item file for `vqav2_val_yesno` (zero-shot floors in results/r0/frontier/, trained runs in
results/r<n>/<run>/eval/). Each model gets Platt scaling fitted on the calib split; every number is on
the report split.

    uv run python scripts/everyday_report.py   # -> results/everyday/vqav2_val_yesno.{md,json}
"""
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from glance.data.benchmarks import vqav2_val_yesno
from glance.metrics import auroc, fit_platt
from scripts.api_benchmark import COCO, IS_THERE, PEOPLE

GROUPS = [
    ("all", lambda d: True),
    ("about people (man, woman, child, ...)", lambda d: bool(PEOPLE.search(d.question))),
    ("'is the man ...'", lambda d: d.meta["question_type"] == "is the man"),
    ("'is the woman ...'", lambda d: d.meta["question_type"] == "is the woman"),
    ("'Is there a X', X a COCO name", lambda d: _there(d) == "coco"),
    ("'Is there a X', other X", lambda d: _there(d) == "other"),
    ("'is this / is this a ...'", lambda d: d.meta["question_type"] in ("is this", "is this a", "is this an")),
    ("'are there ...'", lambda d: d.meta["question_type"] in ("are there", "are there any")),
    ("'does the / does this ...'", lambda d: d.meta["question_type"] in ("does the", "does this")),
    ("'is the ...' (other)", lambda d: d.meta["question_type"] == "is the"),
]


def _there(d):
    m = IS_THERE.match(d.question.strip())
    if not m:
        return None
    noun = m.group(3).strip().lower()
    return "coco" if noun in COCO or noun.rstrip("s") in COCO else "other"


def model_files():
    files = {Path(f).name.split("__")[0]: f for f in glob.glob("results/r0/frontier/*__vqav2_val_yesno.jsonl")}
    for f in glob.glob("results/r[1-9]/*/eval/*__vqav2_val_yesno.jsonl"):
        files[f"glance {Path(f).parent.parent.name}"] = f
    return files


def main():
    decisions = {d.uid: d for d in vqav2_val_yesno()}
    rows, table = {}, defaultdict(dict)
    for name, f in sorted(model_files().items()):
        recs = [json.loads(l) for l in open(f)]
        logits = {r["uid"]: r["logits"] for r in recs}
        cal = [r for r in recs if r["split"] == "calib"]
        a, b = fit_platt([r["logits"] for r in cal], [r["target"] for r in cal])
        p_yes = {u: 1 / (1 + math.exp(-(a * (l[0] - l[1]) + b))) for u, l in logits.items()}
        rep = [decisions[r["uid"]] for r in recs if r["split"] == "report"]
        rows[name] = {"n": len(rep), "auroc": auroc([p_yes[d.uid] for d in rep], [d.label == 0 for d in rep])}
        for g, keep in GROUPS:
            ds = [d for d in rep if keep(d)]
            if ds:
                table[g][name] = (len(ds), float(np.mean([(p_yes[d.uid] >= 0.5) == (d.label == 0) for d in ds])))
    names = list(rows)
    out = Path("results/everyday")
    out.mkdir(parents=True, exist_ok=True)
    (out / "vqav2_val_yesno.json").write_text(json.dumps({"models": rows, "groups": table}, indent=1))
    lines = ["Everyday VQAv2 validation yes/no questions on images outside the training mix; accuracy on the report "
             "split after Platt scaling fitted on the calib split (chance is about 0.5: the gold answers are balanced).", "",
             "| question group | n | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    for g, _ in GROUPS:
        n = max((v[0] for v in table[g].values()), default=0)
        lines.append(f"| {g} | {n} | " + " | ".join(f"{table[g][m][1]:.3f}" if m in table[g] else "—" for m in names) + " |")
    lines.append("| AUROC (all) | | " + " | ".join(f"{rows[m]['auroc']:.3f}" for m in names) + " |")
    (out / "vqav2_val_yesno.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
