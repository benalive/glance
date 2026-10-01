import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from glance.bench.latency import count_flops, fit_latency

DEVICES = ["cpu", "meta"] + (["mps"] if torch.backends.mps.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_flops_include_attention(device):
    torch.backends.mha.set_fastpath_enabled(False)
    layer = nn.TransformerEncoderLayer(384, 6, 1536, 0.0, "gelu", batch_first=True, norm_first=True)
    enc = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False).to(device).eval()
    x = torch.randn(1, 128, 384, device=device)
    assert count_flops(lambda: enc(x)) == 2 * 2 * (12 * 384**2 * 128 + 2 * 128**2 * 384)
    q = torch.randn(1, 6, 128, 64, device=device)
    mask = torch.ones(1, 1, 128, 128, dtype=torch.bool, device=device)
    assert count_flops(lambda: F.scaled_dot_product_attention(q, q, q, attn_mask=mask)) == 4 * 6 * 128 * 128 * 64


def test_fit_recovers_known_coefficients():
    layers = [2, 4, 8, 12, 22, 2, 8, 22]
    gflops = [1, 3, 2, 9, 5, 7, 11, 13]
    ms = [0.5 + 0.2 * l + 0.1 * g for l, g in zip(layers, gflops)]
    fit = fit_latency(layers, gflops, ms)
    assert abs(fit["a_ms_per_layer"] - 0.2) < 1e-9 and abs(fit["b_ms_per_gflop"] - 0.1) < 1e-9
