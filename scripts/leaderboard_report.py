"""Glance next to the published leaderboards of SalArt-VQA, ArtiBench and A-Bench (generative distortion).

Published rows are copied from the papers' tables (checked against the PDFs).
Our rows are computed here from per-item files with each paper's metrics and hard decisions:
  - Glance: probabilities calibrated as the served model calibrates them (Platt / temperature fitted on
    its own in-format benchmarks before these benchmarks were run: data/ckpt/<run>_calib), then yes if
    P(yes) >= 0.5, or the most probable option. Nothing is tuned on these benchmarks.
  - Qwen3-VL-2B (run here, logit readout of the answer tokens): raw argmax, single pass, fixed option
    order, which mirrors an exact-match answer.
Also reported for yes/no: AUROC (threshold-free).

    uv run python scripts/leaderboard_report.py   # -> results/external/leaderboard.{md,json}
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from glance.metrics import auroc
from glance.serve import Predictor, calibrated_probs

OURS = {  # name -> (per-item file pattern, calibration dir or None for raw argmax)
    "Glance C5 open-licence release (ours; v4 seed 0, package calibration)": (
        "results/r4/C5_open4_s0/eval/C5__{b}.jsonl", "data/release/glance-c5-open-v4/calibration.json"),
    "Glance C5 distilled from C1 (ours)": ("results/r2/C5_distill_ep2/eval/C5__{b}.jsonl", "data/ckpt/C5_distill_ep2_calib"),
    "Glance C5, same data without the teacher (ours; control)": ("results/r2/C5_gold48k_ep2/eval/C5__{b}.jsonl", "data/ckpt/C5_gold48k_ep2_calib"),
    "Glance C5 + AI-image errors (ours)": ("results/r2/C5_art2_ep2/eval/C5__{b}.jsonl", "data/ckpt/C5_art2_ep2_calib"),
    "Glance C1 + AI-image errors (ours; teacher, too slow to ship)": ("results/r2/C1_art2_ep2/eval/C1__{b}.jsonl", "data/ckpt/C1_art2_ep2_calib"),
    "Glance C1 (ours)": ("results/r1/C1_ep2_lrA_s0/eval/C1__{b}.jsonl", "data/ckpt/C1_ep2_lrA_s0_calib"),
    "Glance C5 (ours)": ("results/r1/C5_ep2_lrA_s0/eval/C5__{b}.jsonl", "data/ckpt/C5_ep2_lrA_s0_calib"),
    "Qwen3-VL-2B (run here)": ("results/r0/frontier/qwen3vl_2b__{b}.jsonl", None),
}
# SalArt-VQA Table 1: artifact Q1 recall, artifact AllQ1-4, real specificity, real AllQ1-4, overall Q1 acc, overall AllQ1-4
SALART_PUBLISHED = {
    "Gemini 3.1 Pro": (99.37, 53.26, 50.28, 7.87, 78.34, 33.81),
    "Gemini 3.1 Flash Lite": (83.37, 32.63, 93.26, 16.01, 87.61, 25.51),
    "Claude 4.6 Opus": (47.58, 11.79, 100.00, 96.35, 70.04, 48.01),
    "GPT-5.4": (4.63, 1.89, 100.00, 95.79, 45.49, 42.12),
    "Gemma-4-31B-it": (86.32, 33.26, 68.26, 3.93, 78.58, 20.70),
    "Qwen3.5-397B-A17B": (33.26, 12.63, 96.63, 39.33, 60.41, 24.07),
    "Qwen3-VL-235B-A22B-Instruct": (1.47, 0.42, 100.00, 92.70, 43.68, 39.95),
    "Qwen3-VL-8B-Instruct": (0.42, 0.21, 99.44, 91.85, 42.84, 39.47),
    "Llama-4-Maverick-17B-128E": (3.16, 0.00, 99.72, 75.84, 44.53, 32.49),
    "Random": (50.00, 0.40, 50.00, 0.40, 50.00, 0.40),
    "Human (mean of 3)": (100.00, 100.00, 99.53, 99.53, 99.80, 99.80),
}
ARTIBENCH_PUBLISHED = {  # Table 3(a): accuracy, macro-F1
    "GPT-4o": (0.619, 0.601), "GPT-5": (0.599, 0.577), "Gemini-2.5-Pro": (0.582, 0.575),
    "Qwen2.5-VL-7B": (0.501, 0.336), "InternVL3.5-8B": (0.498, 0.357), "Random": (0.500, 0.500),
}
ABENCH_PUBLISHED = {  # Table 2, generative-distortion accuracy (%), VAL + TEST
    "Qwen2-VL-72B": 70.23, "GPT-4o (2024-05-13)": 67.92, "LLaVA-NeXT (Qwen-110B)": 63.64, "MiniCPM-V2.6": 60.47,
    "Gemini 1.5 Pro": 59.07, "Qwen-VL-Max": 58.56, "LLaVA-OneVision-7B": 54.27, "InternVL2-40B": 50.10,
    "Random": 33.14, "Human (best)": 93.00,
}


def load(pattern, bench, calib):
    path = Path(pattern.format(b=bench))
    if not path.exists():
        return None
    out = {}
    for line in open(path):
        r = json.loads(line)
        z = np.array(r["logits"], dtype=np.float64)
        p = calibrated_probs(calib, r["kind"], z) if calib is not None else np.exp(z - z.max()) / np.exp(z - z.max()).sum()
        out[r["uid"]] = {"p": p, "correct": int(np.argmax(p)) == int(np.argmax(r["target"])), "yes": r["target"][0] > 0.5,
                         "gap": float(z[0] - z[1]) if len(z) == 2 else None}
    return out


def salart_meta():
    from glance.data.benchmarks import EXTERNAL
    meta = {}
    for line in open(EXTERNAL / "salart" / "data" / "test.jsonl"):
        r = json.loads(line)
        meta[r["row_id"]] = (r["image_role"], r["artifact_type"], f"salart:{r['pair_id'] or r['row_id']}")
    return meta


def salart_scores(res, meta, exclude=frozenset()):
    by_row = defaultdict(dict)
    for q in (1, 2, 3, 4):
        for uid, v in (res[q] or {}).items():
            row = uid.split("-", 2)[2]
            if meta[row][2] not in exclude:
                by_row[row][q] = v
    art = [r for r in by_row if meta[r][0] == "artifact"]
    real = [r for r in by_row if meta[r][0] == "clean_reference"]
    q1 = lambda rows: 100 * np.mean([by_row[r][1]["correct"] for r in rows if 1 in by_row[r]])  # noqa: E731
    all4 = lambda rows: 100 * np.mean([all(by_row[r].get(q, {"correct": False})["correct"] for q in (1, 2, 3, 4)) for r in rows])  # noqa: E731
    types = defaultdict(list)
    for r in art:
        types[meta[r][1]].append(by_row[r][1]["correct"])
    q1_items = [by_row[r][1] for r in art + real]
    return {"artifact_recall": q1(art), "artifact_all4": all4(art), "real_specificity": q1(real), "real_all4": all4(real),
            "overall_q1": q1(art + real), "overall_all4": all4(art + real),
            "q1_auroc": auroc([v["gap"] for v in q1_items], [v["yes"] for v in q1_items]),
            "recall_by_type": {t: 100 * float(np.mean(v)) for t, v in types.items()},
            **{f"q{q}_acc_artifact": 100 * np.mean([by_row[r][q]["correct"] for r in art if q in by_row[r]]) for q in (2, 3, 4)}}


def binary_scores(res):
    v = list(res.values())
    pred = [int(np.argmax(x["p"])) == 0 for x in v]
    gold = [x["yes"] for x in v]
    f1 = []
    for cls in (True, False):
        tp = sum(p == cls and g == cls for p, g in zip(pred, gold))
        fp = sum(p == cls and g != cls for p, g in zip(pred, gold))
        fn = sum(p != cls and g == cls for p, g in zip(pred, gold))
        f1.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return {"acc": float(np.mean([p == g for p, g in zip(pred, gold)])), "macro_f1": float(np.mean(f1)),
            "auroc": auroc([x["gap"] for x in v], gold), "said_yes": float(np.mean(pred))}


def main():
    meta = salart_meta()
    overlap = Path("results/external/overlap.json")
    exclude = frozenset(json.loads(overlap.read_text())["benches"]["salart_q1"]["ids"]) if overlap.exists() else frozenset()
    table = {"salart": {}, "salart_no_overlap": {}, "artibench": {}, "abench_generative": {}}
    for name, (pattern, calib_dir) in OURS.items():
        if calib_dir and calib_dir.endswith("calibration.json"):  # a release package: its own licence-clean calibration
            calib = json.loads(Path(calib_dir).read_text())["calibration"]
        else:
            calib = Predictor._fit_calibration(calib_dir) if calib_dir else None
        res = {q: load(pattern, f"salart_q{q}", calib) for q in (1, 2, 3, 4)}
        if res[1]:
            table["salart"][name] = salart_scores(res, meta)
            if exclude:
                table["salart_no_overlap"][name] = salart_scores(res, meta, exclude)
        a = load(pattern, "artibench", calib)
        if a:
            table["artibench"][name] = binary_scores(a)
        g = load(pattern, "abench_generative", calib)
        if g:
            table["abench_generative"][name] = {"acc": 100 * float(np.mean([x["correct"] for x in g.values()]))}
    out = Path("results/external")
    out.mkdir(parents=True, exist_ok=True)
    (out / "leaderboard.json").write_text(json.dumps(table, indent=1, default=float))

    L = ["## SalArt-VQA (arXiv 2606.12671), 437 flawed images + 318 real photos (38 HAD-derived pairs removed)", "",
         "Published rows: Table 1 (all 475 + 356 images). Q1: 'does the image contain a salient artifact?'; All-4: Q1-Q4 "
         "(yes/no, region, box, description) all right on the same image.", "",
         "| model | flawed: Q1 recall | flawed: All-4 | real: specificity | real: All-4 | overall Q1 acc | overall All-4 | Q1 AUROC |",
         "|---|---|---|---|---|---|---|---|"]
    for n, s in table["salart"].items():
        L.append(f"| **{n}** | {s['artifact_recall']:.1f} | {s['artifact_all4']:.1f} | {s['real_specificity']:.1f} | "
                 f"{s['real_all4']:.1f} | {s['overall_q1']:.1f} | {s['overall_all4']:.1f} | {s['q1_auroc']:.3f} |")
    for n, s in table["salart_no_overlap"].items():
        L.append(f"| **{n}, excluding the {len(exclude)} real photos that are near-duplicates of our training images** | "
                 f"{s['artifact_recall']:.1f} | {s['artifact_all4']:.1f} | {s['real_specificity']:.1f} | {s['real_all4']:.1f} | "
                 f"{s['overall_q1']:.1f} | {s['overall_all4']:.1f} | {s['q1_auroc']:.3f} |")
    for n, v in SALART_PUBLISHED.items():
        L.append(f"| {n} | " + " | ".join(f"{x:.1f}" for x in v) + " | — |")
    L += ["", "Q1 recall by flaw type (ours): " + "; ".join(
        f"{n}: " + ", ".join(f"{t} {x:.0f}" for t, x in sorted(s['recall_by_type'].items())) for n, s in table["salart"].items()), ""]
    L += ["## ArtiBench (arXiv 2602.20951), 1,000 generated images, half flawed", "",
          "| model | accuracy | macro-F1 | AUROC | answers yes |", "|---|---|---|---|---|"]
    for n, s in table["artibench"].items():
        L.append(f"| **{n}** | {s['acc']:.3f} | {s['macro_f1']:.3f} | {s['auroc']:.3f} | {s['said_yes']:.2f} |")
    for n, (a, f) in ARTIBENCH_PUBLISHED.items():
        L.append(f"| {n} | {a:.3f} | {f:.3f} | — | — |")
    L += ["", "## A-Bench generative distortion (arXiv 2406.03070); ours on the VAL half (250 questions), published on VAL + TEST", "",
          "| model | accuracy (%) |", "|---|---|"]
    for n, s in table["abench_generative"].items():
        L.append(f"| **{n}** | {s['acc']:.1f} |")
    for n, a in ABENCH_PUBLISHED.items():
        L.append(f"| {n} | {a:.2f} |")
    (out / "leaderboard.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
