import numpy as np
import pytest

from glance.metrics import aurc, binned_ece, brier, fit_temperature, nll, smooth_ece, softmax, top_label


def _synthetic(n=4000, miscal=1.0, seed=0):
    rng = np.random.default_rng(seed)
    true_p = rng.uniform(0.05, 0.95, n)
    y = (rng.uniform(size=n) < true_p).astype(float)
    conf = np.clip(0.5 + (true_p - 0.5) * miscal, 1e-3, 1 - 1e-3)
    return conf, y


def test_smooth_ece_matches_relplot():
    # Fixture keeps confidence inside [0.05, 0.95]: near 1.0, relplot's reflected kernel counts a
    # point once where ours (and a multi-reflection reference) counts it twice (R0 eval audit), so
    # the two legitimately differ there by 0.02-0.07.
    relplot = pytest.importorskip("relplot")
    for miscal in [1.0, 1.6, 0.5]:
        conf, y = _synthetic(miscal=miscal)
        ours, ref = smooth_ece(conf, y), float(relplot.smECE(conf, y))
        assert abs(ours - ref) < 3e-3, (miscal, ours, ref)


def test_smooth_ece_orders_calibration():
    good = smooth_ece(*_synthetic(miscal=1.0))
    bad = smooth_ece(*_synthetic(miscal=1.8))
    assert good < 0.03 < bad


def test_binned_ece_perfect_and_constant():
    conf = np.full(1000, 0.7)
    correct = np.r_[np.ones(700), np.zeros(300)]
    assert binned_ece(conf, correct) == pytest.approx(0.0)
    assert binned_ece(conf, np.ones(1000)) == pytest.approx(0.3)


def test_scores_on_known_values():
    ps = [np.array([0.8, 0.2]), np.array([0.5, 0.5])]
    ys = [[1.0, 0.0], [0.0, 1.0]]
    assert brier(ps, ys) == pytest.approx(((0.2**2 + 0.2**2) + (0.5**2 + 0.5**2)) / 2)
    assert nll(ps, ys) == pytest.approx(-(np.log(0.8) + np.log(0.5)) / 2)
    conf, correct = top_label(ps, ys)
    assert list(correct) == [1.0, 0.0]  # a tie resolves to index 0, whose target is 0
    assert aurc(np.array([0.9, 0.1]), np.array([1.0, 0.0])) == pytest.approx((0 + 0.5) / 2)


def test_temperature_recovers_overconfidence():
    rng = np.random.default_rng(1)
    logits, ys = [], []
    for _ in range(3000):
        true = rng.normal(size=4)
        p = softmax(true)
        c = rng.choice(4, p=p)
        logits.append(true * 3.0)  # 3x overconfident
        y = [0.0] * 4
        y[c] = 1.0
        ys.append(y)
    assert fit_temperature(logits, ys) == pytest.approx(3.0, rel=0.15)


def test_auroc_and_platt():
    from glance.metrics import auroc, fit_platt

    assert auroc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == 1.0
    assert auroc([0.1, 0.2, 0.8, 0.9], [True, True, False, False]) == 0.0
    rng = np.random.default_rng(0)
    d = rng.normal(size=4000)
    y = (rng.uniform(size=4000) < 1 / (1 + np.exp(-(2.0 * d - 1.5)))).astype(float)
    a, b = fit_platt([np.array([x, 0.0]) for x in d], [[t, 1 - t] for t in y])
    assert abs(a - 2.0) < 0.2 and abs(b + 1.5) < 0.2


def test_paired_bootstrap_detects_small_paired_gain():
    from glance.metrics import paired_bootstrap

    rng = np.random.default_rng(0)
    base = (rng.uniform(size=2000) < 0.6).astype(float)
    better = base.copy()
    flip = rng.choice(np.where(base == 0)[0], 60, replace=False)
    better[flip] = 1.0  # +3 points on the same items
    out = paired_bootstrap(better, base, groups=np.arange(2000))
    assert out["diff"] == pytest.approx(0.03) and out["ci95"][0] > 0
