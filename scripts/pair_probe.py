"""Diagnostic: do frozen image features carry SalArt's inpainted flaws at all?

SalArt-VQA has 119 pairs of a clean generated image and the same image with a flaw inpainted. Every Glance model
scores the two alike (scripts/salart_clean_ai.py). Before building training data for this, check whether the
image features a model can use register the flaw. For each image encoder, cached features as served
(Predictor._image_input) are probed with 5-fold cross-validation grouped by pair (never trained on the test pair):
  - pooled: logistic regression on mean- and max-pooled tokens, flawed vs clean, AUROC over held-out images;
  - paired: a linear direction fit on within-pair differences (flawed minus clean), accuracy of
    sign(w . (f_flawed - f_clean)) on held-out pairs (an upper bound: it sees both images of a pair);
  - control: the same pooled probe for SalArt's other flawed images against its real photos.
Also the pixel footprint of each inpainted flaw: share of pixels whose value changes by more than 8/255.
This is a diagnostic on the test benchmark and is not reported as a result of any model.

    uv run python scripts/pair_probe.py     # -> results/pair_probe.{json,md}
"""
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from glance.data.benchmarks import EXTERNAL
from glance.metrics import auroc
from glance.serve import Predictor

ENCODERS = {"C5: SigLIP2 B/32 @256 (released)": ("weights/glance-c5-open-v4", "C5", None),
            "C1: SigLIP2 B/16 @512 (ModernVBERT)": ("data/ckpt/C1_art2_ep2.pt", "C1", None)}


def rows():
    pairs, other, real = {}, [], []
    for line in open(EXTERNAL / "salart" / "data" / "test.jsonl"):
        r = json.loads(line)
        if r["row_id"].startswith("hadm__"):
            continue
        img = (EXTERNAL / "salart" / r["image_path"]).read_bytes()
        if r["row_id"].startswith("artifact_injection"):
            pairs.setdefault(r["pair_id"], {})["flawed" if r["image_role"] == "artifact" else "clean"] = img
        elif r["image_role"] == "artifact":
            other.append(img)
        elif r["image_role"] == "clean_reference":
            real.append(img)
    return [p for _, p in sorted(pairs.items()) if len(p) == 2], other, real


def footprint(a: bytes, b: bytes) -> float:
    x = np.asarray(Image.open(io.BytesIO(a)).convert("RGB"), dtype=np.int16)
    y = np.asarray(Image.open(io.BytesIO(b)).convert("RGB").resize(x.shape[1::-1]), dtype=np.int16)
    return float(np.mean(np.abs(x - y).max(axis=2) > 8))


def pooled(f: np.ndarray) -> np.ndarray:
    return np.concatenate([f.mean(0), f.max(0)])


def cv_pooled(X, y, groups, k=5, seed=0):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    fold = dict(zip(ug, rng.permutation(len(ug)) % k))
    score = np.zeros(len(y))
    for i in range(k):
        te = np.array([fold[g] == i for g in groups])
        sc = StandardScaler().fit(X[~te])
        clf = LogisticRegression(C=0.1, max_iter=2000).fit(sc.transform(X[~te]), y[~te])
        score[te] = clf.decision_function(sc.transform(X[te]))
    return auroc(list(score), list(y))


def cv_paired(D, k=5, seed=0):
    rng = np.random.default_rng(seed)
    fold = rng.permutation(len(D)) % k
    correct = []
    for i in range(k):
        tr, te = D[fold != i], D[fold == i]
        X = np.concatenate([tr, -tr])
        y = np.array([1] * len(tr) + [0] * len(tr))
        sc = StandardScaler(with_mean=False).fit(X)
        clf = LogisticRegression(C=0.1, max_iter=2000, fit_intercept=False).fit(sc.transform(X), y)
        correct += list(clf.decision_function(sc.transform(te)) > 0)
    return float(np.mean(correct))


def main():
    pairs, other, real = rows()
    fp = [footprint(p["flawed"], p["clean"]) for p in pairs]
    out = {"n_pairs": len(pairs), "n_other_flawed": len(other), "n_real": len(real),
           "flaw_footprint": {"median": float(np.median(fp)), "p10": float(np.percentile(fp, 10)), "p90": float(np.percentile(fp, 90))},
           "encoders": {}}
    for name, (ckpt, cand, calib) in ENCODERS.items():
        pred = Predictor(ckpt, cand, calib, "cpu", 4)

        def feats(img):
            f, _ = pred._image_input(img)
            pred.cache.clear()
            return f[0].float().cpu().numpy()

        F = [(feats(p["flawed"]), feats(p["clean"])) for p in pairs]
        X = np.array([pooled(f) for pair in F for f in pair])
        y = np.array([1, 0] * len(F))
        groups = np.repeat(np.arange(len(F)), 2)
        D = np.array([pooled(a) - pooled(b) for a, b in F])
        cos = [float(np.dot(a.mean(0), b.mean(0)) / np.linalg.norm(a.mean(0)) / np.linalg.norm(b.mean(0))) for a, b in F]
        tok = [float(np.mean(np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1))) for a, b in F]
        Xo = np.array([pooled(feats(i)) for i in other] + [pooled(feats(i)) for i in real])
        yo = np.array([1] * len(other) + [0] * len(real))
        res = {"pooled_auroc_flawed_vs_clean": cv_pooled(X, y, groups), "paired_accuracy": cv_paired(D),
               "control_pooled_auroc_flawed_vs_real": cv_pooled(Xo, yo, np.arange(len(yo))),
               "pair_cosine_mean_tokens": float(np.median(cos)), "pair_token_rel_change": float(np.median(tok))}
        out["encoders"][name] = res
        print(name, {k: round(v, 4) for k, v in res.items()}, flush=True)
        pred = None
    Path("results/pair_probe.json").write_text(json.dumps(out, indent=1))
    L = [f"Frozen-feature probes on SalArt's {len(pairs)} inpainted pairs (5-fold CV grouped by pair). Median share of pixels "
         f"changed by the inpainting: {out['flaw_footprint']['median']:.3f} (10-90%: {out['flaw_footprint']['p10']:.3f}-"
         f"{out['flaw_footprint']['p90']:.3f}).", "",
         "| encoder | pooled probe AUROC, flawed vs clean | paired direction accuracy | control: other flawed vs real AUROC | median cosine (mean token) | median relative token change |",
         "|---|---|---|---|---|---|"]
    for name, r in out["encoders"].items():
        L.append(f"| {name} | {r['pooled_auroc_flawed_vs_clean']:.3f} | {r['paired_accuracy']:.3f} | {r['control_pooled_auroc_flawed_vs_real']:.3f} | "
                 f"{r['pair_cosine_mean_tokens']:.4f} | {r['pair_token_rel_change']:.4f} |")
    Path("results/pair_probe.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
