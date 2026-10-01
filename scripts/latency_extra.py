"""Measure additional pipeline shapes with the R0 latency protocol (run on an idle machine).

    uv run python scripts/latency_extra.py   # -> results/r0/latency_extra.json
"""
import json
from pathlib import Path

import torch

from glance.bench.latency import count_flops, l4_projection_ms, time_call
from glance.bench.pipelines import modernvbert

EXTRA = {"C2 ModernVBERT text 11L @256px": lambda n_q, dev, dt: modernvbert(n_q, dev, dt, text_layers=11, res=256)}


def main():
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    rows = []
    with torch.inference_mode():
        for dev, dt in [(torch.device("mps"), torch.float16), (torch.device("cpu"), torch.float32)]:
            for name, build in EXTRA.items():
                for n_q in [1, 8]:
                    p = build(n_q, dev, dt)
                    for mode in ["cold", "cached"]:
                        fn = getattr(p, mode)
                        r = time_call(fn, dev)
                        depth = p.depth_cold if mode == "cold" else p.depth_cached
                        wbytes = p.bytes_cold if mode == "cold" else p.bytes_cached
                        flops = count_flops(fn)
                        scale = 2 / torch.tensor([], dtype=dt).element_size()
                        r.update(device=dev.type, name=name, mode=mode, n_questions=n_q, depth=depth,
                                 gflops=flops / 1e9, params_m=p.params / 1e6,
                                 l4_projection_ms=l4_projection_ms(depth, flops, wbytes * scale))
                        rows.append(r)
                        print(f"{dev.type} q={n_q} {name} {mode}: {r['p50_ms']:.2f} ms {r['gflops']:.1f} GF", flush=True)
    Path("results/r0/latency_extra.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
