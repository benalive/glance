"""Does a model's error detection survive wordings it never saw? A licence-clean check on held-out images.

Images: the calibration split of a licence-clean error mix (never trained on): AI-generated images whose
labels are clear (EvalMuse: a structural problem marked at least twice, or nothing marked; ImageRewardDB:
fidelity <= 3 or >= 6) and real photos (no generation errors). Questions: PROBE, unseen paraphrases of
"does this image have visible errors?" in both polarities; none is a training or evaluation wording
(asserted). Per wording: AUROC of the flaw-vs-clean ranking (polarity-adjusted), with no calibration.
SalArt-VQA is not used, so it stays a fair test of the chosen model.

    uv run python scripts/wording_probe.py --runs results/r4/C5_open_2stage_s0,results/r4/C5_open_2stage_s1
    # -> results/wording_probe.json
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

from glance.metrics import auroc
from glance.serve import Predictor
from scripts.build_clean_artifact_mix import ANY, ART_Q, FORBIDDEN, HAND, HAND_CATS, FACE_CATS, LIMB_CATS, OTHER_CATS, group_state
from scripts.build_artifact_mix import FACE, LIMB

PROBE = [("Does this picture contain an artifact?", 1), ("Is there a noticeable artifact anywhere in this image?", 1),
         ("Do you see any artifact in this image?", 1), ("Is there some visible anomaly in this picture?", 1),
         ("Does this image have a flaw that stands out?", 1), ("Has something in this image been rendered incorrectly?", 1),
         ("Is any part of this picture corrupted or malformed?", 1), ("Does this image contain a visual mistake?", 1),
         ("Would a careful viewer spot an error in this image?", 1), ("Is there an obvious defect in this picture?", 1),
         ("Is this picture artifact-free?", -1), ("Does this image look error-free to you?", -1),
         ("Is every part of this image rendered correctly?", -1), ("Would you say this picture has no visible flaws?", -1),
         ("Is this image without anomalies?", -1)]
_trained = {q.lower() for fam in (HAND, ANY, ART_Q, FACE, LIMB) for q, _ in fam} | {q.lower() for q in FORBIDDEN}
assert not {q.lower() for q, _ in PROBE} & _trained


def probe_images(mix: Path, per_group: int, seed: int) -> list[tuple[str, str, bool]]:
    """(image_id, path, has_error) for calibration images: flawed AI, clean AI, real photos."""
    index = json.loads((mix / "images.json").read_text())
    path_of, groups = {}, {"ai_flawed": [], "ai_clean": [], "real": []}
    labels = {}
    for src in ("evalmuse", "imagereward"):
        for line in open(Path("data/clean") / f"labels_{src}.jsonl"):
            r = json.loads(line)
            labels[r["image_id"]] = r
    for line in open(mix / "calib.jsonl"):
        r = json.loads(line)
        iid = r["image_id"]
        if iid in path_of or "#" in iid:
            continue
        path_of[iid] = index["blobs"][index["rows"][r["image_row"]][0]]
        if iid.startswith(("coco:", "oi:")):
            groups["real"].append(iid)
        elif iid.startswith("evalmuse:"):
            state = group_state(Counter(labels[iid]["fidelity_label"]), HAND_CATS | FACE_CATS | LIMB_CATS | OTHER_CATS)
            if state is not None:
                groups["ai_flawed" if state else "ai_clean"].append(iid)
        elif iid.startswith("imagereward:"):
            f = labels[iid]["fidelity_rating"]
            if f <= 3 or f >= 6:
                groups["ai_flawed" if f <= 3 else "ai_clean"].append(iid)
    rng = random.Random(seed)
    out = []
    for g, ids in groups.items():
        for iid in rng.sample(sorted(ids), min(per_group, len(ids))):
            out.append((iid, path_of[iid], g == "ai_flawed"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="comma list of run dirs (config.json names the candidate)")
    ap.add_argument("--mix", default="data/clean_art_v2")
    ap.add_argument("--per-group", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    images = probe_images(Path(a.mix), a.per_group, a.seed)
    print(f"{len(images)} images: {Counter(('flawed' if e else 'clean') for _, _, e in images)}", flush=True)
    out_path = Path("results/wording_probe.json")
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    qs = {f"q{i}": {"type": "noul", "instructions": q} for i, (q, _) in enumerate(PROBE)}
    for run in a.runs.split(","):
        run = Path(run)
        cid = json.loads((run / "config.json").read_text())["cand"]
        pred = Predictor(f"data/ckpt/{run.name}.pt", cid, None, "cpu", 4)
        p = np.zeros((len(images), len(PROBE)))
        for i, (_, path, _) in enumerate(images):
            ans = pred.predict(qs, Path(path).read_bytes())["answers"]
            p[i] = [ans[f"q{j}"]["noul"] for j in range(len(PROBE))]
            pred.cache.clear()
        err = [e for _, _, e in images]
        per = {q: auroc(list(p[:, j] if pol == 1 else 1 - p[:, j]), err) for j, (q, pol) in enumerate(PROBE)}
        results[run.name] = {"mean_auroc": float(np.mean(list(per.values()))), "min_auroc": float(min(per.values())),
                             "per_wording": per, "n_images": len(images)}
        print(f"{run.name}: mean AUROC {results[run.name]['mean_auroc']:.3f}, worst {results[run.name]['min_auroc']:.3f}", flush=True)
        out_path.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
