"""Can the frozen image features our candidates use tell images with generation errors from clean ones?

Linear probe: each image -> the frozen image tokens of a candidate (C5: SigLIP2-B/32 at 256 px; C1:
SigLIP2-B/16 at 512 px, pixel-shuffled), mean-pooled; logistic regression trained on 70% of each
benchmark (split by image hash) and scored by AUROC on the other 30%. A probe near 0.5 means the
frozen features carry little of the signal, so fine-tuning only the layers above them cannot learn
the task; the image encoder itself would have to be trained.

    uv run python scripts/artifact_probe.py --benches had_val_hands,had_val_any,richhf_test_artifacts
"""
import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from glance.data.benchmarks import ALL_BENCHMARKS
from glance.metrics import auroc
from glance.model.candidates import build_candidate, normalize

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


@torch.inference_mode()
def features(cand, decisions, batch=32):
    out = []
    for s in range(0, len(decisions), batch):
        chunk = decisions[s:s + batch]
        px = np.stack([np.asarray(Image.open(io.BytesIO(d.image_bytes)).convert("RGB").resize(
            (cand.image_size, cand.image_size), Image.Resampling.BICUBIC), dtype=np.uint8) for d in chunk])
        f = cand.image_features(normalize(torch.from_numpy(px)).to(DEVICE)).float()
        out.append(f.mean(1).cpu().numpy())
    return np.concatenate(out)


def probe(x, y, groups):
    test = np.array([int(hashlib.sha1(g.encode()).hexdigest()[:8], 16) % 10 < 3 for g in groups])
    sc = StandardScaler().fit(x[~test])
    best = None
    for c in (0.001, 0.01, 0.1, 1.0):  # C picked on a slice of the training part, never on the test part
        inner = np.arange(len(x))[~test]
        val = inner[np.array([int(hashlib.sha1(groups[i].encode()).hexdigest()[8:16], 16) % 5 == 0 for i in inner])]
        fit = np.setdiff1d(inner, val)
        m = LogisticRegression(C=c, max_iter=2000).fit(sc.transform(x[fit]), y[fit])
        a = auroc(m.predict_proba(sc.transform(x[val]))[:, 1], y[val])
        if best is None or a > best[0]:
            best = (a, c)
    m = LogisticRegression(C=best[1], max_iter=2000).fit(sc.transform(x[~test]), y[~test])
    return auroc(m.predict_proba(sc.transform(x[test]))[:, 1], y[test]), int(test.sum()), best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benches", default="had_val_hands,had_val_any,richhf_test_artifacts")
    ap.add_argument("--cands", default="C5,C1")
    a = ap.parse_args()
    data = {b: ALL_BENCHMARKS[b]() for b in a.benches.split(",")}
    results = {}
    for cid in a.cands.split(","):
        cand = build_candidate(cid).to(DEVICE).eval()
        for b, ds in data.items():
            x = features(cand, ds)
            y = np.array([d.label == 0 for d in ds], dtype=int)
            auc, n_test, c = probe(x, y, [d.image_id for d in ds])
            results[f"{cid} {b}"] = {"auroc": auc, "n_test": n_test, "positives": float(y.mean()), "C": c}
            print(f"{cid} {b}: probe AUROC {auc:.3f} (test n={n_test}, share yes {y.mean():.2f}, C={c})", flush=True)
        del cand
    Path("results/everyday").mkdir(parents=True, exist_ok=True)
    Path("results/everyday/artifact_probe.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
