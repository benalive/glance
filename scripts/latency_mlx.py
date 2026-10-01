"""On-device latency (primary target since 2026-09-23): every candidate in MLX on this Mac.

Same protocol as R0 (batch 1; warmup; timed iterations each ending in mx.eval; p50/p95), fp16
weights, with and without mx.compile. Random weights at real shapes (latency does not depend on
values; parity of these modules with the torch models is tested in tests/test_mlx.py). A
Laya-shaped text-only model (ModernBERT-large, 28 layers, plus a 2-layer head) anchors the harness
against the published laya-mlx figures (p50 ~9 ms on M3 Max, 15.3 ms on M5 Pro).

    uv run python scripts/latency_mlx.py   # run on an otherwise idle machine
"""
import json
import platform
import subprocess
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from transformers import AutoConfig

from glance.mlx.modernbert import FusionTowerMLX, ModernBertMLX, modernbert_masks
from glance.mlx.tower import GlanceTowerMLX, PointerHeadMLX, text_block_mask
from glance.model.towers import TowerConfig, siglip_b32_config, tiny_cpu_config

Q_LEN, K, C_LEN = 24, 4, 6


def workload(n_q, vocab):
    rng = np.random.default_rng(n_q)
    ids, pos, blocks, cands = [], [], [], []
    for qi in range(n_q):
        ids += rng.integers(5, vocab - 5, Q_LEN + K * C_LEN).tolist()
        pos += list(range(Q_LEN)) + [Q_LEN + j for _ in range(K) for j in range(C_LEN)]
        blocks += [qi] * (Q_LEN + K * C_LEN)
        cands += [-1] * Q_LEN + [k for k in range(K) for _ in range(C_LEN)]
    a = lambda x: mx.array([x])  # noqa: E731
    return a(ids), a(pos), a(blocks), a(cands)


def timed(fn, warmup=15, iters=60):
    for _ in range(warmup):
        mx.eval(fn())
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append((time.perf_counter() - t0) * 1e3)
    a = np.asarray(ts)
    return {"p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95))}


def fp16(m):
    m.set_dtype(mx.float16)
    return m


def glance_tower(cfg: TowerConfig, n_q):
    t, h = fp16(GlanceTowerMLX(cfg)), fp16(PointerHeadMLX(cfg.width))
    px = mx.random.normal((1, 3, cfg.image_size, cfg.image_size)).astype(mx.float16)
    ids, pos, blocks, cands = workload(n_q, cfg.vocab_size)
    q_pos = mx.array([[0, i * (Q_LEN + K * C_LEN)] for i in range(n_q)])
    slot_pos = mx.array([[0, i * (Q_LEN + K * C_LEN) + Q_LEN + (k + 1) * C_LEN - 1] for i in range(n_q) for k in range(K)])
    slot_q = mx.array([i for i in range(n_q) for _ in range(K)])
    kv, _ = t.encode_image(px)
    mx.eval(kv)
    cold = lambda: h(t(px, ids, pos, blocks, cands)[1], q_pos, slot_pos, slot_q)  # noqa: E731
    cached = lambda: h(t.encode_text(ids, pos, blocks, kv, cands), q_pos, slot_pos, slot_q)  # noqa: E731
    return cold, cached, cfg.depth, cfg.depth


def fusion(n_q):
    vis = fp16(GlanceTowerMLX(siglip_b32_config()))  # its image path is SigLIP2-B/32's vision tower
    tcfg = AutoConfig.from_pretrained("jhu-clsp/ettin-encoder-32m", local_files_only=True)
    f = fp16(FusionTowerMLX(tcfg))
    px = mx.random.normal((1, 3, 256, 256)).astype(mx.float16)
    ids, pos, blocks, cands = workload(n_q, tcfg.vocab_size)
    img = f.project_image(vis.encode_image(px)[1])
    mx.eval(img)
    cold = lambda: f.encode_text(ids, pos, blocks, f.project_image(vis.encode_image(px)[1]), cands)  # noqa: E731
    cached = lambda: f.encode_text(ids, pos, blocks, img, cands)  # noqa: E731
    return cold, cached, 12 + tcfg.num_hidden_layers + 3, tcfg.num_hidden_layers + 3


class Connector(nn.Module):
    """ModernVBERT's pixel shuffle (r=4) + bias-free projection."""

    def __init__(self, d=768, r=4):
        super().__init__()
        self.r = r
        self.proj = nn.Linear(d * r * r, d, bias=False)

    def __call__(self, x):
        B, N, D = x.shape
        s, r = int(N ** 0.5), self.r
        x = x.reshape(B, s, s // r, D * r).transpose(0, 2, 1, 3).reshape(B, s // r, s // r, D * r * r)
        return self.proj(x.transpose(0, 2, 1, 3).reshape(B, N // (r * r), D * r * r))


def vbert(n_q, res, layer_ids=None):
    vcfg = TowerConfig(width=768, depth=12, heads=12, mlp_dim=3072, patch=16, image_size=res)
    vis, conn = fp16(GlanceTowerMLX(vcfg)), fp16(Connector())
    tcfg = AutoConfig.from_pretrained("ModernVBERT/modernvbert", local_files_only=True).text_config
    text = fp16(ModernBertMLX(tcfg, layer_ids))
    px = mx.random.normal((1, 3, res, res)).astype(mx.float16)
    ids, pos, blocks, cands = workload(n_q, tcfg.vocab_size)
    img = conn(vis.encode_image(px)[1])
    mx.eval(img)
    n_img = img.shape[1]
    all_pos = mx.concatenate([mx.arange(n_img)[None], pos + n_img], 1)
    all_blocks = mx.concatenate([mx.full((1, n_img), 10_000), blocks], 1)
    all_cands = mx.concatenate([mx.full((1, n_img), -1), cands], 1)
    masks = modernbert_masks(text_block_mask(all_blocks, all_cands), tcfg.local_attention, all_pos)

    def run(image_tokens):
        x = mx.concatenate([image_tokens, text.embeddings.tok_embeddings(ids)], 1)
        return text(masks, all_pos, inputs_embeds=x)

    L = len(text.layers)
    return (lambda: run(conn(vis.encode_image(px)[1]))), (lambda: run(img)), 12 + L, L


def laya_shape(n_tokens=128):
    """Text-only Laya: ModernBERT-large (28 x 1024) + a 2-layer transformer head, one question."""
    cfg = AutoConfig.from_pretrained("jhu-clsp/ettin-encoder-32m", local_files_only=True)
    cfg.hidden_size, cfg.num_hidden_layers, cfg.num_attention_heads, cfg.intermediate_size = 1024, 28, 16, 2624
    cfg.layer_types = ["full_attention" if i % 3 == 0 else "sliding_attention" for i in range(28)]
    enc = fp16(ModernBertMLX(cfg))
    head = [fp16(nn.TransformerEncoderLayer(1024, 16, 4096)) for _ in range(2)]
    ids = mx.random.randint(5, 50000, (1, n_tokens))
    pos = mx.arange(n_tokens)[None]
    masks = modernbert_masks(mx.ones((1, n_tokens, n_tokens), dtype=mx.bool_), 128, pos)

    def run():
        x = enc(masks, pos, input_ids=ids)
        for layer in head:
            x = layer(x, None)
        return x

    return run, run, 30, 30


def main():
    cands = {
        "C3/C4 single tower 12x768": lambda q: glance_tower(siglip_b32_config(), q),
        "C6 tiny tower 6x512": lambda q: glance_tower(tiny_cpu_config(), q),
        "C5 fusion (B/32 + Ettin-32M + 3 fusion)": fusion,
        "C1 ModernVBERT @512px": lambda q: vbert(q, 512),
        "C2 ModernVBERT text 11L @256px": lambda q: vbert(q, 256, list(range(0, 22, 2))),
    }
    rows = []
    for name, build in cands.items():
        for n_q in [1, 8]:
            cold, cached, dc, dk = build(n_q)
            for mode, fn in [("cold", cold), ("cached", cached)]:
                for compiled in [False, True]:
                    r = timed(mx.compile(fn) if compiled else fn)
                    r.update(name=name, n_questions=n_q, mode=mode, compiled=compiled, depth=dc if mode == "cold" else dk)
                    rows.append(r)
                    print(f"{name:42s} q={n_q} {mode:6s} {'compiled' if compiled else 'eager   '} "
                          f"p50 {r['p50_ms']:6.2f} ms  p95 {r['p95_ms']:6.2f}", flush=True)
    run, _, _, _ = laya_shape()
    for compiled in [False, True]:
        r = timed(mx.compile(run) if compiled else run)
        r.update(name="Laya-shape (ModernBERT-large 28L + 2L head, 128 tok, text-only)", n_questions=1, mode="cold",
                 compiled=compiled, depth=30)
        rows.append(r)
        print(f"{r['name']} {'compiled' if compiled else 'eager'} p50 {r['p50_ms']:.2f} ms", flush=True)
    meta = {"machine": platform.machine(), "chip": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                                                  capture_output=True, text=True).stdout.strip(),
            "mlx": mx.__version__, "dtype": "float16",
            "commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()}
    Path("results/r0/latency_mlx.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
