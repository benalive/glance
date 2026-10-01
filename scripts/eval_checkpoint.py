"""Evaluate a saved R1 checkpoint on named benchmarks, with the training code's own evaluation
(batched, same device and precision as the evaluation that runs after training).

    uv run python scripts/eval_checkpoint.py --run results/r1/C1_ep2_lrA_s0 \
        --benches vqav2_val_yesno,pope_random,pope_popular
    # checkpoint data/ckpt/<run name>.pt -> <run>/eval/<cand>__<bench>.jsonl, next to the standard ones
"""
import argparse
import json
import os
from pathlib import Path

from transformers import AutoTokenizer

from glance.model.candidates import build_candidate, load_trained, read_state
from glance.train import DEVICE, TrainConfig, evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="R1 run dir, e.g. results/r1/C1_ep2_lrA_s0")
    ap.add_argument("--benches", required=True)
    ap.add_argument("--ckpt", help="default: $GLANCE_DATA_DIR/ckpt/<run name>.pt")
    a = ap.parse_args()
    run = Path(a.run)
    cfg = TrainConfig(**{k: v for k, v in json.loads((run / "config.json").read_text()).items()
                         if k in TrainConfig.__dataclass_fields__})
    cand = build_candidate(cfg.cand)
    data_dir = Path(os.environ.get("GLANCE_DATA_DIR", "data"))
    ckpt = a.ckpt or data_dir / "ckpt" / f"{run.name}.pt"
    load_trained(cand, read_state(ckpt), str(ckpt))
    tok = AutoTokenizer.from_pretrained("jhu-clsp/ettin-encoder-32m")
    evaluate(cand.to(DEVICE), tok, run / "eval", cfg, a.benches.split(","))


if __name__ == "__main__":
    main()
