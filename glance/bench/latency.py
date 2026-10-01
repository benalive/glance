"""Batch-1 latency protocol and cost accounting shared by every round.

Every timed iteration is bracketed by a device synchronize; we report p50/p95 over `iters`
after `warmup`. On this Mac the numbers only rank designs; absolute L4 numbers come from R2.
FLOPs are counted from the actual aten ops (torch FlopCounterMode), so they include attention
and whatever the HF implementations really run.
"""
import time

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


@torch.inference_mode()
def time_call(fn, device: torch.device, warmup: int = 15, iters: int = 60) -> dict:
    for _ in range(warmup):
        fn()
    sync(device)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        sync(device)
        times.append((time.perf_counter() - t0) * 1e3)
    a = np.asarray(times)
    return {"p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)),
            "mean_ms": float(a.mean()), "iters": iters, "warmup": warmup}


def _sdpa_flops(q, k, v, *args, **kwargs) -> int:
    """QK^T plus AV, for the fused SDPA kernels FlopCounterMode does not know."""
    b, h, lq, d = q
    return 2 * b * h * lq * k[-2] * d + 2 * b * h * lq * k[-2] * v[-1]


# On CPU and MPS (torch 2.14) SDPA reaches dispatch as the composite op, decomposes into fused
# kernels torch's flop registry does not cover, and counts 0; the meta device counts correctly.
# FlopCounterMode tests the OpOverload to decide whether to decompose, then looks the formula up
# by overload packet, so both keys are registered.
_SDPA = torch.ops.aten.scaled_dot_product_attention
_EXTRA_FLOPS = {_SDPA: _sdpa_flops, _SDPA.default: _sdpa_flops}


def count_flops(fn, inference: bool = True) -> float:
    """FLOPs of everything `fn` runs; pass inference=False to count a training step incl. backward."""
    with torch.inference_mode(inference), FlopCounterMode(display=False, custom_mapping=_EXTRA_FLOPS) as counter:
        fn()
    return float(counter.get_total_flops())


def l4_projection_ms(n_layers: int, flops: float, weight_bytes: float, kernels_per_layer: int = 9) -> float:
    """Rough compiled (CUDA graphs / TensorRT) batch-1 L4 estimate from the R0 literature review:
    0.15 ms per call + ~4 us per fused kernel + max(weight streaming at 250 GB/s, math at 50 TFLOPS).
    A projection for ranking, not a measurement."""
    return 0.15 + n_layers * kernels_per_layer * 0.004 + max(weight_bytes / 250e9, flops / 50e12) * 1e3


def fit_latency(layers, gflops, ms) -> dict:
    """Least-squares fit ms ~ c + a*layers + b*gflops."""
    X = np.column_stack([np.ones(len(ms)), layers, gflops])
    y = np.asarray(ms)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"c_ms": float(coef[0]), "a_ms_per_layer": float(coef[1]), "b_ms_per_gflop": float(coef[2]),
            "implied_tflops": float(1.0 / coef[2]) if coef[2] > 0 else None,
            "r2": 1 - ss_res / ss_tot, "mape": float(np.mean(np.abs(pred - y) / y))}
