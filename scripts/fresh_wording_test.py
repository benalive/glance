"""Unseen-wording test: 12 wordings of "does this image have visible errors?" written by the evaluation auditor
before any model saw them, each at least 5 word edits from
every error-question wording in training (checked with `check`). Run on SalArt-VQA Q1 images and on the
wording-probe images; AUROC per wording (polarity-adjusted), no calibration.

    uv run python scripts/fresh_wording_test.py check data/clean_art_v4
    uv run python scripts/fresh_wording_test.py run data/clean_art_v4 results/fresh_wordings_v4.json C5_open4_s0 C5_open4_s1 C5_open4_s2
"""
import argparse
import gc
import json
import re
from pathlib import Path

import numpy as np

from glance.metrics import auroc

FRESH = [("Did the image generator make a mistake somewhere in this picture?", 1),
         ("Is some detail here physically impossible or botched?", 1),
         ("Would an artist need to repair something in this picture?", 1),
         ("Is there a spot in this picture that gives away a failed generation?", 1),
         ("Are there parts of this scene that could not exist in reality?", 1),
         ("Something here is rendered wrong, isn't it?", 1),
         ("Could this picture pass as a faultless photograph?", -1),
         ("Is everything in this scene coherent and properly formed?", -1),
         ("Would a close inspection find nothing wrong here?", -1),
         ("Is this rendering good enough to publish without touch-ups?", -1),
         ("Does every object here have a believable shape?", -1),
         ("Was this image produced without any glitch a viewer would catch?", -1)]
SALART_Q1 = "Does this image contain at least one salient artifact?"


def toks(s: str) -> list[str]:
    return re.findall(r"[a-z\-']+", s.lower())


def lev(a: list[str], b: list[str]) -> int:
    """Word-level edit distance."""
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[len(b)]


def check(mix: Path):
    train = set()
    for line in open(mix / "decisions.jsonl"):
        r = json.loads(line)
        if r["kind"] == "noul" and r["source"] in ("evalmuse_train", "imagereward_train", "real_photo_negative"):
            train.add(r["question"])
    tt = [toks(t) for t in train]
    for q, _ in FRESH:
        print(min(lev(toks(q), t) for t in tt), q)


def run(mix: Path, out_path: Path, runs: list[str]):
    from glance.data.benchmarks import salart
    from glance.serve import Predictor
    from scripts.wording_probe import probe_images

    sal = salart(1)
    probe = probe_images(mix, 400, 0)
    qs = {f"q{i}": {"type": "noul", "instructions": q} for i, (q, _) in enumerate(FRESH)}
    qs["salart"] = {"type": "noul", "instructions": SALART_Q1}
    out = {}
    for run_name in runs:
        pred = Predictor(f"data/ckpt/{run_name}.pt", "C5", None, "cpu", 4)
        res = {}
        for setname, items in (("salart", [(d.image_bytes, d.target[0] > 0.5) for d in sal]),
                               ("probe_imgs", [(Path(p).read_bytes(), e) for _, p, e in probe])):
            rows = []
            for img, _ in items:
                a = pred.predict(qs, img)["answers"]
                pred.cache.clear()
                rows.append([a[f"q{i}"]["noul"] for i in range(len(FRESH))] + [a["salart"]["noul"]])
            p = np.array(rows)
            y = np.array([e for _, e in items])
            per = {q: auroc(p[:, i] if pol == 1 else 1 - p[:, i], y) for i, (q, pol) in enumerate(FRESH)}
            res[setname] = {"per": per, "mean": float(np.mean(list(per.values()))), "min": float(min(per.values())),
                            "salart_q": auroc(p[:, -1], y)}
        out[run_name] = res
        print(run_name, {s: (round(r["mean"], 3), round(r["min"], 3), round(r["salart_q"], 3)) for s, r in res.items()}, flush=True)
        pred = None
        gc.collect()
    out_path.write_text(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["check", "run"])
    ap.add_argument("mix", type=Path, help="training mix, e.g. data/clean_art_v4")
    ap.add_argument("out", nargs="?", type=Path, help="run: output json")
    ap.add_argument("runs", nargs="*", help="run: checkpoint names under data/ckpt/")
    a = ap.parse_args()
    if a.mode == "check":
        check(a.mix)
    else:
        run(a.mix, a.out, a.runs)


if __name__ == "__main__":
    main()
