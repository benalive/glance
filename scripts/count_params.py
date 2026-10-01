"""Parameter counts of every candidate as built for training, plus Laya-snake from its weights file.

    uv run python scripts/count_params.py   ->   results/params.json
"""
import gc
import glob
import json
import math
import os
import struct
from pathlib import Path


def safetensors_params(pattern):
    total = 0
    for f in glob.glob(os.path.expanduser(pattern)):
        with open(f, "rb") as fh:
            header = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))
        total += sum(math.prod(v["shape"]) for k, v in header.items() if k != "__metadata__")
    return total


def main():
    from glance.model.candidates import build_candidate

    out = {}
    for cid in ["C1", "C2", "C3", "C4", "C5", "C6", "C6s"]:
        c = build_candidate(cid)
        out[cid] = {"total": sum(p.numel() for p in c.parameters()),
                    "trainable": sum(p.numel() for p in c.parameters() if p.requires_grad)}
        if cid == "C5":  # frozen, cached image tower is outside the trainable module
            out[cid]["note"] = "trainable = the per-question path (text encoder, fusion, head); the rest is the frozen SigLIP2-B/32 vision tower"
        del c
        gc.collect()
    out["laya-snake"] = {"total": safetensors_params(
        "~/.cache/huggingface/hub/models--madhavbiplov--laya-snake-mlx/snapshots/*/*.safetensors"),
        "note": "sum of tensor sizes in the released safetensors file"}
    Path("results/params.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(k, {a: (f"{b / 1e6:.1f}M" if isinstance(b, int) else b) for a, b in v.items() if a != "note"})


if __name__ == "__main__":
    main()
