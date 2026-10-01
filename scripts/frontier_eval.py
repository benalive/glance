"""R0 quality floors: run each baseline over each benchmark, one jsonl per (method, bench).

Each line keeps the raw option logits (and CircularEval-debiased ones for VLM choice items), the
target distribution, the image-grouped calib/report split, and MOS for KonIQ. Metrics are
computed separately by scripts/summarize_frontier.py so they can be recomputed without rerunning.

    uv run python scripts/frontier_eval.py --methods siglip2_b32 --benches pope_adversarial [--limit 50]
"""
import argparse
import json
import time
from pathlib import Path

import torch

from glance.data.benchmarks import ALL_BENCHMARKS, R0_BENCHMARKS
from glance.data.schema import calib_split

METHODS = {
    "siglip2_b16": ("siglip", "google/siglip2-base-patch16-256"),
    "siglip2_b32": ("siglip", "google/siglip2-base-patch32-256"),
    "smolvlm_256m": ("vlm", "HuggingFaceTB/SmolVLM-256M-Instruct"),
    "qwen3vl_2b": ("vlm", "Qwen/Qwen3-VL-2B-Instruct"),
}


# Assistant prefill overrides, chosen per (method, benchmark) by *label-free* answer-token mass: on
# Qwen MMStar the unprefilled readout put 24% of its mass off the letters (math items start
# reasoning); on the 30 lowest-mass items "Answer: " raised mass 0.00 -> 0.31, the best of 4 tried.
# Everywhere else mass is >= 0.99 without a prefill.
PREFIX = {("qwen3vl_2b", "mmstar"): {"choice": "Answer: "}}


def load_method(name: str, device, bench: str | None = None):
    kind, repo = METHODS[name]
    if kind == "siglip":
        from glance.baselines.siglip_zs import SiglipZeroShot

        return SiglipZeroShot(repo, device)
    from glance.baselines.vlm_readout import VLMReadout

    return VLMReadout(repo, device, prefix=PREFIX.get((name, bench)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--benches", default=",".join(R0_BENCHMARKS))
    ap.add_argument("--limit", type=int, default=0, help="first N items per bench (smoke runs)")
    ap.add_argument("--out", default="results/r0/frontier")
    args = ap.parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    benches = {}
    for bench in args.benches.split(","):
        items = ALL_BENCHMARKS[bench]()
        benches[bench] = items[: args.limit] if args.limit else items
        print(f"{bench}: {len(benches[bench])} items", flush=True)
    for method in args.methods.split(","):
        model, model_prefix = None, None
        for bench, items in benches.items():
            path = out_dir / f"{method}__{bench}.jsonl"
            if path.exists() and sum(1 for _ in path.open()) == len(items):
                print(f"skip {path} (complete)", flush=True)
                continue
            if model is None or PREFIX.get((method, bench)) != model_prefix:
                model, model_prefix = load_method(method, device, bench), PREFIX.get((method, bench))
            t0 = time.time()
            with path.open("w") as fh:
                for i, d in enumerate(items):
                    rec = {"uid": d.uid, "image_id": d.image_id, "split": calib_split(d.image_id), "kind": d.kind,
                           "k": len(d.candidates), "target": d.target, **model.logits(d)}
                    if "mos" in d.meta:
                        rec["mos"] = d.meta["mos"]
                    fh.write(json.dumps(rec) + "\n")
                    if (i + 1) % 500 == 0:
                        print(f"  {method} {bench} {i + 1}/{len(items)} {(i + 1) / (time.time() - t0):.1f} it/s", flush=True)
            print(f"done {method} {bench}: {len(items)} items in {time.time() - t0:.0f}s", flush=True)
        del model
        if device.type == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
