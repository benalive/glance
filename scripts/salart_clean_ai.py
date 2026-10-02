"""SalArt-VQA Q1 with its unflawed generated images: flawed AI vs clean AI, not only flawed AI vs real photos.

SalArt's Q1 has no gold answer for its 119 unflawed generated images, so the benchmark's Q1 score compares 437
flawed generated images with 318 real photographs. Here every model also answers Q1 on the 119 clean generated
images (answer "no"). HAD-derived rows are dropped, as in glance.data.benchmarks.salart. P(yes) goes through
the served path (Predictor.predict) with each model's own calibration.

The 119 clean images are the originals of SalArt's 119 "artifact_injection" flawed images (a flaw inpainted into
the clean picture), so the controlled flaw test is within those pairs: AUROC of injected vs original, and the
share of pairs where the injected image gets the higher P(yes). Also: AUROC flawed vs real, flawed vs clean AI,
the other 318 flawed images vs real photos; and at the served threshold (P(yes) >= 0.5) recall and the share of
real photos and of clean AI images flagged. Per-image probabilities: results/salart_clean_ai_items.jsonl.

    uv run python scripts/salart_clean_ai.py     # -> results/salart_clean_ai.{json,md}, salart_clean_ai_items.jsonl
"""
import gc
import json
from pathlib import Path

import numpy as np

from glance.data.benchmarks import EXTERNAL, SALART_Q
from glance.metrics import auroc
from glance.serve import Predictor

MODELS = {  # name -> Predictor args (ckpt, cand, calib_dir)
    "Glance C5 released (v4 seed 0, package)": ("weights/glance-c5-open-v4", "C5", None),
    "Glance C5 v4 seed 1 (no calibration)": ("data/ckpt/C5_open4_s1.pt", "C5", None),
    "Glance C5 v4 seed 2 (no calibration)": ("data/ckpt/C5_open4_s2.pt", "C5", None),
    "Glance C5 research (distilled)": ("data/ckpt/C5_distill_ep2.pt", "C5", "data/ckpt/C5_distill_ep2_calib"),
    "Glance C1 research (teacher)": ("data/ckpt/C1_art2_ep2.pt", "C1", "data/ckpt/C1_art2_ep2_calib"),
}
ROLE = {"artifact": "flawed", "clean_reference": "real", "paired_generated_counterpart": "clean_ai"}


def items():
    out = []
    for line in open(EXTERNAL / "salart" / "data" / "test.jsonl"):
        r = json.loads(line)
        if r["row_id"].startswith("hadm__"):
            continue
        out.append((r["row_id"], ROLE[r["image_role"]], (EXTERNAL / "salart" / r["image_path"]).read_bytes(), r["pair_id"]))
    return out


def auc2(pos, neg):
    return auroc(list(pos) + list(neg), [1] * len(pos) + [0] * len(neg))


def main():
    data = items()
    groups = {g: sum(1 for _, x, _, _ in data if x == g) for g in ROLE.values()}
    q = {"q": {"type": "noul", "instructions": SALART_Q[1]}}
    table, per_item = {}, []
    for name, (ckpt, cand, calib) in MODELS.items():
        pred = Predictor(ckpt, cand, calib, "cpu", 4)
        p = {}
        for uid, _, img, _ in data:
            p[uid] = pred.predict(q, img)["answers"]["q"]["noul"]
            pred.cache.clear()
        by = {g: np.array([p[u] for u, x, _, _ in data if x == g]) for g in ROLE.values()}
        clean_of = {pid: p[u] for u, x, _, pid in data if x == "clean_ai"}
        pairs = [(p[u], clean_of[pid]) for u, x, _, pid in data if x == "flawed" and pid in clean_of]
        other = [p[u] for u, x, _, pid in data if x == "flawed" and pid not in clean_of]
        per_item += [{"model": name, "uid": u, "group": x, "pair_id": pid, "p_yes": p[u]} for u, x, _, pid in data]
        row = {"auroc_flawed_vs_real": auc2(by["flawed"], by["real"]),
               "auroc_flawed_vs_clean_ai": auc2(by["flawed"], by["clean_ai"]),
               "auroc_injected_vs_original": auc2([a for a, _ in pairs], [b for _, b in pairs]),
               "pairs_injected_higher": float(np.mean([a > b for a, b in pairs])),
               "auroc_other_flawed_vs_real": auc2(other, by["real"])}
        if "no calibration" not in name:
            row |= {"recall_flawed": float(np.mean(by["flawed"] >= 0.5)),
                    "recall_injected": float(np.mean([a >= 0.5 for a, _ in pairs])),
                    "flagged_real": float(np.mean(by["real"] >= 0.5)),
                    "flagged_clean_ai": float(np.mean(by["clean_ai"] >= 0.5))}
        table[name] = row
        print(name, {k: round(v, 3) for k, v in row.items()}, flush=True)
        pred = None
        gc.collect()
    Path("results/salart_clean_ai.json").write_text(json.dumps({"n": groups, "n_pairs": len(pairs), "models": table}, indent=1))
    Path("results/salart_clean_ai_items.jsonl").write_text("".join(json.dumps(r) + "\n" for r in per_item))
    cols = list(next(iter(table.values())).keys() | {"recall_flawed", "recall_injected", "flagged_real", "flagged_clean_ai"})
    order = ["auroc_flawed_vs_real", "auroc_flawed_vs_clean_ai", "auroc_injected_vs_original", "pairs_injected_higher",
             "auroc_other_flawed_vs_real", "recall_flawed", "recall_injected", "flagged_real", "flagged_clean_ai"]
    cols = [c for c in order if c in cols]
    L = [f"SalArt-VQA Q1 on {groups['flawed']} flawed generated images, {groups['real']} real photos and "
         f"{groups['clean_ai']} clean generated images, the originals of {len(pairs)} flawed images made by inpainting a "
         "flaw (HAD-derived rows dropped). Recall and flagged shares at the served threshold (P(yes) >= 0.5).", "",
         "| model | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for name, row in table.items():
        L.append(f"| {name} | " + " | ".join(f"{row[c]:.3f}" if c in row else "—" for c in cols) + " |")
    Path("results/salart_clean_ai.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
