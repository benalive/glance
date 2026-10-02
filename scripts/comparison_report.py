"""One comparison table: the open Glance release against our research models, open baselines, Laya and
published models, on accuracy and latency.

Every model is scored as it would be used:
  - Glance open release: its package calibration (fitted on the licence-clean calibration split only).
  - Research-only Glance C5 / C1: the served calibration, fitted on the calibration splits of POPE-adv,
    A-OKVQA, KonIQ, HAD and RichHF (so their accuracy there uses a threshold fitted on that benchmark).
  - Qwen3-VL-2B, SmolVLM-256M (zero-shot, run here): raw answers (argmax of the answer-token readout).
Accuracy is on each benchmark's report split (the part never used for any fitting) except SalArt and
ArtiBench, which are scored on all items like their published rows. AUROC (threshold-free) is given for
yes/no benchmarks. Published rows are copied from the papers (scripts/leaderboard_report.py).

    uv run python scripts/comparison_report.py    # -> results/comparison.{md,json}
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from glance.metrics import auroc, majority
from glance.serve import Predictor, calibrated_probs

MODELS = {  # name -> (per-item file pattern, calibration: package json / calib dir / None = raw)
    "Glance C5 open release (ours, Apache-2.0)": ("results/r4/C5_open4_s0/eval/C5__{b}.jsonl",
                                                  "weights/glance-c5-open-v4/calibration.json"),
    "Glance C5 research-only (distilled)": ("results/r2/C5_distill_ep2/eval/C5__{b}.jsonl", "data/ckpt/C5_distill_ep2_calib"),
    "Glance C1 research-only (teacher)": ("results/r2/C1_art2_ep2/eval/C1__{b}.jsonl", "data/ckpt/C1_art2_ep2_calib"),
    "Qwen3-VL-2B (zero-shot, run here)": ("results/r0/frontier/qwen3vl_2b__{b}.jsonl", None),
    "SmolVLM-256M (zero-shot, run here)": ("results/r0/frontier/smolvlm_256m__{b}.jsonl", None),
}
ALL_ITEMS = {"salart_q1", "artibench", "abench_generative"}
ACC = ["vqav2_val_yesno", "pope_adversarial", "pope_random", "aokvqa_val", "mmstar", "sugarcrepe_replace_rel",
       "sugarcrepe_swap_att", "salart_q1", "artibench"]
AUC = ["vqav2_val_yesno", "pope_adversarial", "salart_q1", "artibench", "had_val_hands", "had_val_any", "richhf_test_artifacts"]
LABEL = {"vqav2_val_yesno": "VQAv2 everyday yes/no", "pope_adversarial": "POPE adversarial", "pope_random": "POPE random",
         "aokvqa_val": "A-OKVQA (4-way)", "mmstar": "MMStar (chance .25)", "sugarcrepe_replace_rel": "SugarCrepe relation",
         "sugarcrepe_swap_att": "SugarCrepe attribute", "salart_q1": "SalArt-VQA Q1 (artifact?)", "artibench": "ArtiBench (flawed?)",
         "had_val_hands": "HAD malformed hands", "had_val_any": "HAD any anatomy error", "richhf_test_artifacts": "RichHF-18K artifacts",
         "koniq_test": "KonIQ quality (SRCC)", "real_photos_no_errors": "real photos flagged as flawed"}
PUBLISHED = {  # accuracy as published (long prompts, exact match on generated text); see leaderboard_report.py
    "salart_q1": {"Gemini 3.1 Pro": 0.783, "Gemini 3.1 Flash Lite": 0.876, "Claude 4.6 Opus": 0.700, "GPT-5.4": 0.455},
    "artibench": {"InternVL3.5-8B (fine-tuned by the ArtiBench authors)": 0.630, "Qwen2.5-VL-7B (fine-tuned by the ArtiBench authors)": 0.627,
                  "GPT-4o": 0.619, "GPT-5": 0.599, "Gemini-2.5-Pro": 0.582},
}
LATENCY = [  # model, machine, runtime, new image ms, question on a cached image ms, protocol / source
    ("Glance C5 open release", "Apple M4 Pro CPU, 4 threads", "ONNX Runtime", "18.3", "3.0",
     "served path, 40 COCO photos; results/latency/release_m4pro_onnx.json"),
    ("Glance C5 open release", "AMD EPYC 9V45 (x86) CPU, 4 threads", "ONNX Runtime", "26.2", "4.1",
     "served path, 30 COCO photos; results/latency/x86_azure_epyc9v45_onnx.json"),
    ("Glance C5 open release", "Apple M4 Pro CPU, 4 threads", "PyTorch fp32", "23.0", "5.1",
     "served path, 40 COCO photos; results/latency/release_m4pro_torch.json"),
    ("Glance C5 open release", "AMD EPYC 9V45 (x86) CPU, 4 threads", "PyTorch fp32", "57.3", "7.3",
     "served path, 30 COCO photos; results/latency/x86_azure_epyc9v45_onnx.json"),
    ("Glance C1 research (ModernVBERT, 252M)", "Apple M4 Pro CPU, 4 threads", "PyTorch fp32", "166.5", "24.1",
     "served path, 40 RichHF images; results/latency/serving_cpu.json"),
    ("SmolVLM-256M", "Apple M4 Pro CPU, 4 threads", "PyTorch fp32", "183.0", "23.0",
     "synthetic input, random weights, model only (R0); results/r0/latency_summary.md"),
    ("SmolVLM-256M", "Apple M4 Pro GPU (MPS)", "PyTorch fp16", "55.3", "12.7", "as above"),
    ("Qwen3-VL-2B", "Apple M4 Pro GPU (MPS)", "PyTorch fp16", "135.0", "81.1", "as above"),
    ("Laya-snake 322M", "Apple M4 Pro CPU, 4 threads", "PyTorch", "40.3 (text only)", "—",
     "one text decision on the Snake benchmark; no image input"),
    ("laya-vision", "NVIDIA L4 GPU", "PyTorch bf16", "32-41", "—", "published figure, not measured here"),
    ("Gemini / GPT / Claude", "cloud API", "—", "not measured", "—", "network round trip dominates"),
]


def calibration(spec):
    if spec is None:
        return None
    if spec.endswith("calibration.json"):
        return json.loads(Path(spec).read_text())["calibration"]
    return Predictor._fit_calibration(spec)


def load(pattern, bench):
    path = Path(pattern.format(b=bench))
    if not path.exists():
        return None
    recs = [json.loads(line) for line in open(path)]
    return recs if bench in ALL_ITEMS else [r for r in recs if r.get("split") == "report"]


def probs(r, calib):
    z = np.array(r["logits"], dtype=np.float64)
    if calib is None:
        e = np.exp(z - z.max())
        return e / e.sum()
    return calibrated_probs(calib, r["kind"], z)


def main():
    table = {}
    for name, (pattern, spec) in MODELS.items():
        calib, row = calibration(spec), {}
        for b in ACC:
            recs = load(pattern, b)
            if recs:
                row[f"acc:{b}"] = float(np.mean([np.argmax(probs(r, calib)) == majority(r["target"]) for r in recs
                                                 if majority(r["target"]) is not None]))
        for b in AUC:
            recs = load(pattern, b)
            if recs:
                recs = [r for r in recs if majority(r["target"]) is not None]
                row[f"auroc:{b}"] = auroc([r["logits"][0] - r["logits"][1] for r in recs], [r["target"][0] > 0.5 for r in recs])
        recs = load(pattern, "koniq_test")
        if recs:
            exp = [float(np.dot(np.arange(len(r["logits"])), probs(r, calib))) for r in recs]
            row["srcc:koniq_test"] = float(spearmanr(exp, [r["mos"] for r in recs]).statistic)
        path = Path(pattern.format(b="real_photos_no_errors"))
        if path.exists():
            recs = [json.loads(line) for line in open(path)]
            row["fa:real_photos_no_errors"] = float(np.mean([np.argmax(probs(r, calib)) != np.argmax(r["target"]) for r in recs]))
        table[name] = row
    names = list(MODELS)
    short = ["**Glance open**", "Glance C5 research", "Glance C1 research", "Qwen3-VL-2B", "SmolVLM-256M"]
    fmt = lambda v: "—" if v is None else f"{v:.3f}"  # noqa: E731
    L = ["# Glance open release vs other models", "",
         "Accuracy as each model is used (Glance open: its licence-clean calibration; research Glance: calibration fitted on the "
         "calibration splits of POPE-adv, A-OKVQA, KonIQ, HAD and RichHF; zero-shot models: raw answers). Glance rows use "
         "offline GPU logits; served CPU answers agree on 99.4-99.9% of items. Report splits, except SalArt and ArtiBench (all items, like the published rows). Published API rows used "
         "long prompts and graded generated text, so they are indicative, not like-for-like.", "",
         "## Everyday questions (accuracy)", "", "| benchmark | " + " | ".join(short) + " |", "|---|" + "---|" * len(short)]
    for b in ACC[:7]:
        L.append(f"| {LABEL[b]} | " + " | ".join(fmt(table[n].get(f'acc:{b}')) for n in names) + " |")
    L.append("| " + LABEL["koniq_test"] + " | " + " | ".join(fmt(table[n].get("srcc:koniq_test")) for n in names) + " |")
    L += ["", "## Visible errors in AI-generated images", "", "| benchmark | " + " | ".join(short) + " | published (accuracy) |",
          "|---|" + "---|" * (len(short) + 1)]
    for b in ("salart_q1", "artibench"):
        pub = "; ".join(f"{k} {v:.3f}" for k, v in PUBLISHED[b].items())
        L.append(f"| {LABEL[b]}, accuracy | " + " | ".join(fmt(table[n].get(f'acc:{b}')) for n in names) + f" | {pub} |")
        L.append(f"| {LABEL[b]}, AUROC | " + " | ".join(fmt(table[n].get(f'auroc:{b}')) for n in names) + " | not published |")
    for b in ("had_val_hands", "had_val_any", "richhf_test_artifacts"):
        L.append(f"| {LABEL[b]}, AUROC | " + " | ".join(fmt(table[n].get(f'auroc:{b}')) for n in names) + " | — |")
    L.append(f"| {LABEL['real_photos_no_errors']} | " + " | ".join(fmt(table[n].get("fa:real_photos_no_errors")) for n in names) + " | — |")
    L += ["", "HAD and RichHF are the research models' own training distributions (in-distribution for them, cross-dataset for the "
          "open release). The open release's threshold is conservative: at 97% specificity on SalArt it finds 45% of flawed images.",
          "", "## Latency (median, batch 1)", "", "| model | machine | runtime | new image, 1 question (ms) | "
          "question on a cached image (ms) | protocol / source |", "|---|---|---|---|---|---|"]
    for m in LATENCY:
        L.append("| " + " | ".join(m) + " |")
    L += ["", "## Laya (text decisions, Snake benchmark, same input and data)", "",
          "| model | input | p50 latency (M4 Pro CPU, 4 threads) | top-1 on held-out boards (report split) |", "|---|---|---|---|",
          "| Laya-snake 322M | text (ASCII board + facts) | 40.3 ms | 0.981 |",
          "| Glance C5 text path | identical text | 13.7 ms | 0.981 (same move on every board) |",
          "| Glance C5 image + facts | board image + fact lines | 31.4 ms | 0.981 |", "",
          "Laya is text-only (no image input); on identical input and data Glance is about 3x faster with equal accuracy. "
          "Jev was not compared (no API key)."]
    out = Path("results")
    (out / "comparison.json").write_text(json.dumps(table, indent=1))
    (out / "comparison.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
