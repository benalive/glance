"""R1 quality screen queue, run sequentially and restartable (finished runs are skipped).

Runs are matched on data (epochs over the same training sequences), not FLOPs: with frozen,
cached image encoders a FLOP budget buys C1/C2/C5 ~8x more epochs than C3 (R1 audit fairness
finding), and on this Mac it made one C5 run at 1e16 ~15 h. FLOPs and wall time are still recorded.

Stage "sweep": every candidate for 0.5 epoch under two LR settings (same effort for all).
Stage "main": each candidate's better setting by dev NLL (2% image-grouped holdout of the
training mix; benchmarks are never used for selection) for 2 epochs.
Stage "control": shuffled-image runs for 2 epochs with the selected setting.

    uv run python scripts/run_r1.py --stage sweep|main|control [--cands C3,C5]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from glance.data.benchmarks import R0_BENCHMARKS

ORDER = ["C3", "C5", "C6", "C4", "C2", "C1"]
SMALL, LARGE = 0.5, 2.0  # epochs
LRS = {"lrA": (5e-5, 3e-4), "lrB": (2e-4, 1e-3)}  # (inherited, new)


def run_dir(cand, epochs, lr, seed=0, sanity=None):
    tag = f"ep{epochs:g}"
    return Path(f"results/r1/{cand}_{tag}_{lr}_s{seed}" + (f"_{sanity}" if sanity else ""))


def done(out: Path, cand: str) -> bool:
    return (out / "train_summary.json").exists() and all(
        (out / "eval" / f"{cand}__{b}.jsonl").exists() for b in R0_BENCHMARKS)


def best_lr(cand) -> str:
    scores = {}
    for lr in LRS:
        f = run_dir(cand, SMALL, lr) / "train_summary.json"
        if f.exists():
            scores[lr] = json.loads(f.read_text())["dev"]["nll"]
    assert scores, f"no sweep results for {cand}"
    return min(scores, key=scores.get)


def launch(cand, epochs, lr, sanity=None):
    """Each run in its own process: memory from earlier runs (notably the MPS caching allocator) is
    released, which a single long-lived queue process did not do (it grew to 13 GB)."""
    out = run_dir(cand, epochs, lr, sanity=sanity)
    if done(out, cand):
        print("skip", out, flush=True)
        return
    print("run", out, flush=True)
    li, ln = LRS[lr]
    cmd = [sys.executable, "scripts/train_candidate.py", "--cand", cand, "--epochs", str(epochs), "--out", str(out),
           "--lr-inherited", str(li), "--lr-new", str(ln)]
    if sanity:
        cmd += ["--sanity", sanity]
    if not (epochs == LARGE and sanity is None):
        cmd += ["--no-save-ckpt"]
    if subprocess.run(cmd).returncode != 0:
        print("FAILED", out, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["sweep", "main", "control"])
    ap.add_argument("--cands", default=",".join(ORDER))
    a = ap.parse_args()
    cands = a.cands.split(",")
    if a.stage == "sweep":
        for lr in LRS:
            for cand in cands:
                launch(cand, SMALL, lr)
    elif a.stage == "main":
        for cand in cands:
            launch(cand, LARGE, best_lr(cand))
    else:
        for cand in cands:
            launch(cand, LARGE, best_lr(cand), sanity="shuffle_images")


if __name__ == "__main__":
    main()
