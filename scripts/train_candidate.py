"""Train and evaluate one R1 candidate at one FLOP budget.

    uv run python scripts/train_candidate.py --cand C3 --budget 1e16 --seed 0
    uv run python scripts/train_candidate.py --cand C3 --sanity shuffle_images --budget 2e15
    uv run python scripts/train_candidate.py --cand C3 --sanity overfit
"""
import argparse

from glance.train import TrainConfig, train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--budget", type=float, default=1e16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--sanity", choices=["overfit", "shuffle_labels", "shuffle_images"])
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--eval-benches", default="")
    ap.add_argument("--epochs", type=float, default=0.0)
    ap.add_argument("--lr-image", type=float, default=0.0, help="trainable image-encoder layers (C5u<k>); 0 = --lr-inherited")
    ap.add_argument("--adam-eps", type=float, default=1e-6)
    ap.add_argument("--no-save-ckpt", action="store_true")
    ap.add_argument("--lr-inherited", type=float, default=5e-5)
    ap.add_argument("--lr-new", type=float, default=3e-4)
    ap.add_argument("--data-root", default="data/r1", help="training mix directory")
    ap.add_argument("--init-ckpt", default="", help="start from this trained checkpoint")
    ap.add_argument("--no-cache-features", action="store_true", help="compute frozen image features per batch")
    ap.add_argument("--distill-teacher", default="", help="teacher_<run>.jsonl in the data root")
    ap.add_argument("--distill-alpha", type=float, default=0.0, help="weight of the teacher in the targets")
    a = ap.parse_args()
    budget = 1e30 if a.sanity == "overfit" or a.epochs else a.budget
    tag = a.sanity or f"{a.budget:.0e}".replace("+", "")
    out = a.out or f"results/r1/{a.cand}_{tag}_s{a.seed}"
    print(train(TrainConfig(cand=a.cand, budget=budget, seed=a.seed, batch=a.batch, sanity=a.sanity, out=out,
                            amp=not a.no_amp, eval_benches=a.eval_benches,
                            lr_inherited=a.lr_inherited, lr_new=a.lr_new, lr_image=a.lr_image, adam_eps=a.adam_eps,
                            epochs=a.epochs,
                            save_ckpt=not a.no_save_ckpt, data_root=a.data_root, init_ckpt=a.init_ckpt,
                            cache_features=not a.no_cache_features, distill_teacher=a.distill_teacher,
                            distill_alpha=a.distill_alpha)))


if __name__ == "__main__":
    main()
