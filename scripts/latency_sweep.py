"""R0 latency screen: what dominates batch-1 latency here, and where each candidate lands.

Sections: (1) depth x width x tokens grid on a plain pre-norm encoder, (2) 256 px vision stems,
(3) every candidate/reference at real shape, cold vs cached image, 1 vs 8 questions,
(4) fixed-overhead probe, (5) per-device fit ms ~ c + a*layers + b*GFLOPs on the grid.
MPS runs fp16; CPU runs fp32 on 4 threads (the product's CPU target). Random weights throughout.

    uv run python scripts/latency_sweep.py [--smoke]
"""
import argparse
import datetime
import json
import platform
import subprocess
from pathlib import Path

import torch
import torch.nn as nn

from glance.bench.latency import count_flops, fit_latency, l4_projection_ms, time_call
from glance.bench.pipelines import build_all

CPU_THREADS = 4


def plain_encoder(depth, width, device, dtype):
    layer = nn.TransformerEncoderLayer(width, width // 64, 4 * width, dropout=0.0, activation="gelu",
                                       batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, depth, enable_nested_tensor=False).to(device, dtype).eval()


def timed(fn, device, smoke):
    if smoke:
        return time_call(fn, device, warmup=2, iters=5)
    probe = time_call(fn, device, warmup=2, iters=3)
    if probe["p50_ms"] > 150:  # keep the heavy CPU configs inside the time budget
        return time_call(fn, device, warmup=5, iters=25)
    return time_call(fn, device)


def grid(device, dtype, widths, smoke):
    rows = []
    depths = [2, 12] if smoke else [2, 4, 8, 12, 22]
    for depth in depths:
        for width in widths:
            for tokens in [128, 384]:
                model = plain_encoder(depth, width, device, dtype)
                x = torch.randn(1, tokens, width, device=device, dtype=dtype)
                fn = lambda: model(x)
                r = timed(fn, device, smoke)
                r.update(depth=depth, width=width, tokens=tokens, gflops=count_flops(fn) / 1e9)
                rows.append(r)
                print(f"  grid {device.type} d={depth:2d} w={width:4d} n={tokens}: {r['p50_ms']:.2f} ms", flush=True)
    return rows


def stems(device, dtype, smoke):
    from transformers import SiglipVisionConfig, SiglipVisionModel

    rows = []
    for patch in [16, 32]:
        for width in [384, 768]:
            cfg = SiglipVisionConfig(hidden_size=width, intermediate_size=4 * width, num_hidden_layers=12,
                                     num_attention_heads=width // 64, image_size=256, patch_size=patch, vision_use_head=False)
            model = SiglipVisionModel(cfg).to(device, dtype).eval()
            px = torch.randn(1, 3, 256, 256, device=device, dtype=dtype)
            fn = lambda: model(pixel_values=px)
            r = timed(fn, device, smoke)
            r.update(stem=f"ViT-12L w{width} patch{patch}", tokens=(256 // patch) ** 2, gflops=count_flops(fn) / 1e9)
            rows.append(r)
    conv = nn.Sequential(*[nn.Sequential(nn.Conv2d(a, b, 3, 2, 1), nn.BatchNorm2d(b), nn.GELU())
                           for a, b in [(3, 64), (64, 128), (128, 256), (256, 512)]],
                         nn.PixelUnshuffle(2), nn.Flatten(2)).to(device, dtype).eval()
    proj = nn.Linear(2048, 768).to(device, dtype)
    px = torch.randn(1, 3, 256, 256, device=device, dtype=dtype)
    fn = lambda: proj(conv(px).transpose(1, 2))
    r = timed(fn, device, smoke)
    r.update(stem="conv stem 4x(3x3 s2) + unshuffle", tokens=64, gflops=count_flops(fn) / 1e9)
    rows.append(r)
    for r in rows:
        print(f"  stem {device.type} {r['stem']}: {r['p50_ms']:.2f} ms", flush=True)
    return rows


def pipelines(device, dtype, smoke):
    rows = []
    for n_q in [1, 8]:
        for build in build_all(n_q, device, dtype, include_teacher=device.type == "mps"):
            p = build()
            for mode in ["cold", "cached"]:
                fn = getattr(p, mode)
                r = timed(fn, device, smoke)
                depth = p.depth_cold if mode == "cold" else p.depth_cached
                wbytes = p.bytes_cold if mode == "cold" else p.bytes_cached
                flops = count_flops(fn)
                r.update(name=p.name, mode=mode, n_questions=n_q, depth=depth, gflops=flops / 1e9,
                         params_m=p.params / 1e6, weight_mb=wbytes / 1e6, note=p.note,
                         l4_projection_ms=l4_projection_ms(depth, flops, wbytes * (2 / torch.tensor([], dtype=dtype).element_size())))
                rows.append(r)
                print(f"  pipe {device.type} q={n_q} {p.name:42s} {mode:6s}: {r['p50_ms']:7.2f} ms  "
                      f"{r['gflops']:7.1f} GF  L4~{r['l4_projection_ms']:.2f} ms", flush=True)
            del p
            if device.type == "mps":
                torch.mps.empty_cache()
    return rows


def overhead(device, dtype, smoke):
    rows = []
    for depth in [1, 2, 4, 8, 16]:
        model = plain_encoder(depth, 256, device, dtype)
        x = torch.randn(1, 16, 256, device=device, dtype=dtype)
        r = timed(lambda: model(x), device, smoke)
        r.update(depth=depth)
        rows.append(r)
    slope = (rows[-1]["p50_ms"] - rows[0]["p50_ms"]) / 15
    print(f"  overhead {device.type}: {slope * 1e3:.0f} us/layer, 1-layer call {rows[0]['p50_ms']:.3f} ms", flush=True)
    return {"rows": rows, "us_per_layer": slope * 1e3, "one_layer_call_ms": rows[0]["p50_ms"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="results/r0/latency_sweep.json")
    args = ap.parse_args()
    torch.set_num_threads(CPU_THREADS)
    # The fused CPU encoder fast path hides FLOPs and is not what our towers run; keep every
    # device on the same op-by-op path.
    torch.backends.mha.set_fastpath_enabled(False)
    out = {
        "meta": {"date": datetime.datetime.now().isoformat(timespec="seconds"), "torch": torch.__version__,
                 "machine": platform.machine(), "cpu_threads": CPU_THREADS, "smoke": args.smoke,
                 "commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
                 "protocol": "batch 1; warmup 15, 60 timed iters (25 if p50>150ms); device sync per iter; random weights"},
    }
    devices = [(torch.device("mps"), torch.float16, [256, 384, 512, 768, 1024]),
               (torch.device("cpu"), torch.float32, [384, 768])]
    with torch.inference_mode():
        for device, dtype, widths in devices:
            key = device.type
            print(f"== {key}", flush=True)
            out[key] = {"overhead": overhead(device, dtype, args.smoke),
                        "grid": grid(device, dtype, widths, args.smoke),
                        "stems": stems(device, dtype, args.smoke),
                        "pipelines": pipelines(device, dtype, args.smoke)}
            g = out[key]["grid"]
            out[key]["fit"] = fit_latency([r["depth"] for r in g], [r["gflops"] for r in g], [r["p50_ms"] for r in g])
            print(f"  fit {key}: {out[key]['fit']}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
