"""Accuracy, proper scores and calibration for typed decisions.

Inputs are per-item probability vectors `p` and target distributions `y` over the same
candidates (lists of possibly different length K). Top-label calibration uses confidence = max p
and correctness = y[argmax p], which is 0/1 for gold labels and fractional for soft targets.

smECE follows Blasiok & Nakkiran (arXiv 2309.12236): kernel-smoothed residuals with a Gaussian
kernel reflected at 0 and 1, bandwidth chosen as the fixed point smECE_sigma = sigma.
"""
import math

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import pearsonr, spearmanr


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def top_label(ps, ys):
    conf = np.array([p.max() for p in ps])
    correct = np.array([y[int(p.argmax())] for p, y in zip(ps, ys)])
    return conf, correct


def majority(target) -> int | None:
    """Index of the gold answer, or None when the target has no single majority (e.g. 5 of 10 annotators
    said yes and 5 said no). Accuracy is undefined on such items and they are left out of it; counting them
    as either answer silently shifted VQAv2 accuracy by about 0.01 between two scripts (paper audit)."""
    t = list(target)
    top = max(t)
    return None if t.count(top) > 1 else t.index(top)


def accuracy(ps, ys) -> float:
    return float(np.mean([int(p.argmax()) == int(np.argmax(y)) for p, y in zip(ps, ys)]))


def nll(ps, ys) -> float:
    return float(np.mean([-(np.asarray(y) * np.log(np.clip(p, 1e-12, 1))).sum() for p, y in zip(ps, ys)]))


def brier(ps, ys) -> float:
    return float(np.mean([((p - np.asarray(y)) ** 2).sum() for p, y in zip(ps, ys)]))


def binned_ece(conf, correct, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def _smece_at(conf, resid, sigma, grid):
    def phi(d):
        return np.exp(-0.5 * (d / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))

    t = grid[:, None]
    f = conf[None, :]
    k = phi(t - f) + phi(t + f) + phi(t - (2 - f))  # reflect at 0 and 1
    smoothed = (k * resid[None, :]).mean(axis=1)
    return float(np.trapezoid(np.abs(smoothed), grid))


def smooth_ece(conf, correct, grid_points: int = 501) -> float:
    conf = np.asarray(conf, dtype=np.float64)
    resid = np.asarray(correct, dtype=np.float64) - conf
    grid = np.linspace(0, 1, grid_points)
    lo, hi = 1e-4, 1.0
    if _smece_at(conf, resid, lo, grid) <= lo:
        return _smece_at(conf, resid, lo, grid)
    for _ in range(24):  # smECE_sigma - sigma is decreasing in sigma; bisect the fixed point
        mid = 0.5 * (lo + hi)
        if _smece_at(conf, resid, mid, grid) > mid:
            lo = mid
        else:
            hi = mid
    return float(_smece_at(conf, resid, 0.5 * (lo + hi), grid))


def aurc(conf, correct) -> float:
    order = np.argsort(-conf, kind="stable")
    risk = 1 - np.asarray(correct, dtype=np.float64)[order]
    return float(np.mean(np.cumsum(risk) / np.arange(1, len(risk) + 1)))


def fit_temperature(logits_list, ys) -> float:
    """Scalar temperature minimising NLL (searched on log T)."""
    def loss(log_t):
        t = math.exp(log_t)
        return nll([softmax(l, t) for l in logits_list], ys)

    res = minimize_scalar(loss, bounds=(math.log(0.05), math.log(20.0)), method="bounded")
    return float(math.exp(res.x))


def summarize(logits_list, ys, temperature: float = 1.0, mos=None) -> dict:
    ps = [softmax(l, temperature) for l in logits_list]
    conf, correct = top_label(ps, ys)
    out = {"n": len(ps), "acc": accuracy(ps, ys), "nll": nll(ps, ys), "brier": brier(ps, ys),
           "ece15": binned_ece(conf, correct), "smece": smooth_ece(conf, correct), "aurc": aurc(conf, correct),
           "mean_conf": float(conf.mean()), "temperature": temperature}
    if mos is not None:  # ordinal: expected level vs mean opinion score
        expected = np.array([(p * np.arange(len(p))).sum() for p in ps])
        out["srcc"] = float(spearmanr(expected, mos).statistic)
        out["plcc"] = float(pearsonr(expected, mos).statistic)
    return out


def auroc(scores, labels) -> float:
    """Threshold-free ranking quality for binary items (Mann-Whitney U / (n_pos * n_neg))."""
    from scipy.stats import rankdata

    scores, labels = np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=bool)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def fit_platt(logits_list, ys) -> tuple[float, float]:
    """Binary items only: fit a, b so that P(first candidate) = sigmoid(a * (l0 - l1) + b), by NLL.
    Temperature scaling cannot move a decision threshold; zero-shot scorers (e.g. a SigLIP sigmoid)
    need the bias, so every method gets the same two-parameter family on binary benchmarks. The slope is kept
    non-negative: calibration may rescale and shift a model's ranking but not reverse it (a negative slope
    once turned a model that ranked backwards, AUROC 0.11, into 0.79 "accuracy"; release audit S4)."""
    from scipy.optimize import minimize

    d = np.array([l[0] - l[1] for l in logits_list], dtype=np.float64)
    y = np.array([t[0] for t in ys], dtype=np.float64)

    def loss(w):
        z = w[0] * d + w[1]
        return float(np.mean(np.logaddexp(0, z) - y * z))

    res = minimize(loss, x0=np.array([1.0, 0.0]), method="L-BFGS-B", bounds=[(0.0, None), (None, None)])
    return float(res.x[0]), float(res.x[1])


def grouped_bootstrap_ci(correct, groups, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% CI of mean correctness, resampling whole images (items on one image are correlated)."""
    rng = np.random.default_rng(seed)
    _, g = np.unique(np.asarray(groups), return_inverse=True)
    sums = np.bincount(g, weights=np.asarray(correct, dtype=np.float64))
    counts = np.bincount(g).astype(np.float64)
    idx = rng.integers(0, len(sums), size=(n, len(sums)))
    stats = sums[idx].sum(1) / counts[idx].sum(1)
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def paired_bootstrap(correct_a, correct_b, groups, n: int = 2000, seed: int = 0) -> dict:
    """Difference in mean correctness (a - b) on the same items with an image-grouped paired
    bootstrap 95% CI. Per-method CIs overlap far more than paired differences do."""
    rng = np.random.default_rng(seed)
    _, g = np.unique(np.asarray(groups), return_inverse=True)
    diff = np.asarray(correct_a, dtype=np.float64) - np.asarray(correct_b, dtype=np.float64)
    sums, counts = np.bincount(g, weights=diff), np.bincount(g).astype(np.float64)
    idx = rng.integers(0, len(sums), size=(n, len(sums)))
    stats = sums[idx].sum(1) / counts[idx].sum(1)
    return {"diff": float(diff.mean()), "ci95": (float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)))}


def summarize_run_file(path) -> list[dict]:
    """Metrics for one per-item jsonl (from frontier_eval or train.evaluate): one row per logit
    variant, raw and temperature-scaled, with the temperature fitted on the 'calib' split only
    and every metric computed on the 'report' split. Binary benchmarks also get a Platt-scaled
    row ("<variant>+platt") and AUROC; accuracy carries an image-grouped bootstrap 95% CI."""
    import json
    from pathlib import Path

    recs = [json.loads(l) for l in Path(path).open()]
    # constructions present on every item (a zero-shot scorer may fall back per item, e.g. SigLIP's
    # presence template vs its yes/no template); otherwise each item's default "logits"
    constructions = [k for k in recs[0] if k.startswith("logits_") and k != "logits_debiased"
                     and all(k in r for r in recs)]
    variants = constructions or ["logits"]
    variants += ["logits_debiased"] if "logits_debiased" in recs[0] else []
    cal = [r for r in recs if r["split"] == "calib"]
    rep = [r for r in recs if r["split"] == "report"]
    # Platt (slope + intercept) only for yes/no; on 2-way *choice* items (SugarCrepe) an intercept is
    # a fitted position prior (R0 eval audit), so those get temperature only like every other choice.
    binary = all(r["kind"] == "noul" for r in recs)
    ys = [r["target"] for r in rep]
    mos = [r["mos"] for r in rep] if "mos" in rep[0] else None
    groups = [r["image_id"] for r in rep]
    rows = []

    def finish(row, ps):
        correct = [float(int(p.argmax()) == int(np.argmax(y))) for p, y in zip(ps, ys)]
        row["scaled"]["acc_ci95"] = grouped_bootstrap_ci(correct, groups)
        if binary:
            row["scaled"]["auroc"] = auroc([p[0] for p in ps], [y[0] > 0.5 for y in ys])
        return row

    for v in variants:
        logits = [np.array(r[v]) for r in rep]
        raw = summarize(logits, ys, 1.0, mos)
        t = fit_temperature([np.array(r[v]) for r in cal], [r["target"] for r in cal])
        cal_nll = nll([softmax(np.array(r[v]), t) for r in cal], [r["target"] for r in cal])
        row = {"variant": v, "n_calib": len(cal), "calib_nll": cal_nll, "raw": raw, "scaled": summarize(logits, ys, t, mos)}
        if "mass" in rep[0]:
            row["mean_answer_mass"] = float(np.mean([r["mass"] for r in rep]))
        rows.append(finish(row, [softmax(l, t) for l in logits]))
        if binary:
            a, b = fit_platt([np.array(r[v]) for r in cal], [r["target"] for r in cal])
            platt_logits = [np.array([a * (l[0] - l[1]) + b, 0.0]) for l in logits]
            pl_cal = [np.array([a * (r[v][0] - r[v][1]) + b, 0.0]) for r in cal]
            prow = {"variant": v + "+platt", "n_calib": len(cal), "raw": raw,
                    "calib_nll": nll([softmax(l) for l in pl_cal], [r["target"] for r in cal]),
                    "scaled": summarize(platt_logits, ys, 1.0, mos), "platt": [a, b]}
            rows.append(finish(prow, [softmax(l) for l in platt_logits]))
    return rows
