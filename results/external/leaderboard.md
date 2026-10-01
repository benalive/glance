## SalArt-VQA (arXiv 2606.12671), 437 flawed images + 318 real photos (38 HAD-derived pairs removed)

Published rows: Table 1 (all 475 + 356 images). Q1: 'does the image contain a salient artifact?'; All-4: Q1-Q4 (yes/no, region, box, description) all right on the same image.

| model | flawed: Q1 recall | flawed: All-4 | real: specificity | real: All-4 | overall Q1 acc | overall All-4 | Q1 AUROC |
|---|---|---|---|---|---|---|---|
| **Glance C5 open-licence release (ours; v4 seed 0, package calibration)** | 44.9 | 0.5 | 97.2 | 0.0 | 66.9 | 0.3 | 0.911 |
| **Glance C5 distilled from C1 (ours)** | 79.2 | 0.9 | 88.4 | 0.3 | 83.0 | 0.7 | 0.904 |
| **Glance C5, same data without the teacher (ours; control)** | 73.5 | 3.0 | 73.3 | 0.0 | 73.4 | 1.7 | 0.821 |
| **Glance C5 + AI-image errors (ours)** | 88.8 | 2.3 | 65.4 | 0.0 | 78.9 | 1.3 | 0.879 |
| **Glance C1 + AI-image errors (ours; teacher, too slow to ship)** | 82.6 | 2.7 | 89.6 | 0.0 | 85.6 | 1.6 | 0.928 |
| **Glance C1 (ours)** | 1.4 | 0.0 | 97.5 | 0.0 | 41.9 | 0.0 | 0.476 |
| **Glance C5 (ours)** | 0.0 | 0.0 | 100.0 | 0.0 | 42.1 | 0.0 | 0.617 |
| **Qwen3-VL-2B (run here)** | 99.3 | 5.7 | 2.2 | 0.0 | 58.4 | 3.3 | 0.725 |
| **Glance C5 open-licence release (ours; v4 seed 0, package calibration), excluding the 87 real photos that are near-duplicates of our training images** | 44.9 | 0.5 | 96.5 | 0.0 | 62.7 | 0.3 | 0.905 |
| **Glance C5 distilled from C1 (ours), excluding the 87 real photos that are near-duplicates of our training images** | 79.2 | 0.9 | 89.2 | 0.0 | 82.6 | 0.6 | 0.910 |
| **Glance C5, same data without the teacher (ours; control), excluding the 87 real photos that are near-duplicates of our training images** | 73.5 | 3.0 | 71.9 | 0.0 | 72.9 | 1.9 | 0.823 |
| **Glance C5 + AI-image errors (ours), excluding the 87 real photos that are near-duplicates of our training images** | 88.8 | 2.3 | 67.1 | 0.0 | 81.3 | 1.5 | 0.884 |
| **Glance C1 + AI-image errors (ours; teacher, too slow to ship), excluding the 87 real photos that are near-duplicates of our training images** | 82.6 | 2.7 | 90.5 | 0.0 | 85.3 | 1.8 | 0.932 |
| **Glance C1 (ours), excluding the 87 real photos that are near-duplicates of our training images** | 1.4 | 0.0 | 97.8 | 0.0 | 34.7 | 0.0 | 0.468 |
| **Glance C5 (ours), excluding the 87 real photos that are near-duplicates of our training images** | 0.0 | 0.0 | 100.0 | 0.0 | 34.6 | 0.0 | 0.616 |
| **Qwen3-VL-2B (run here), excluding the 87 real photos that are near-duplicates of our training images** | 99.3 | 5.7 | 1.7 | 0.0 | 65.6 | 3.7 | 0.741 |
| Gemini 3.1 Pro | 99.4 | 53.3 | 50.3 | 7.9 | 78.3 | 33.8 | — |
| Gemini 3.1 Flash Lite | 83.4 | 32.6 | 93.3 | 16.0 | 87.6 | 25.5 | — |
| Claude 4.6 Opus | 47.6 | 11.8 | 100.0 | 96.3 | 70.0 | 48.0 | — |
| GPT-5.4 | 4.6 | 1.9 | 100.0 | 95.8 | 45.5 | 42.1 | — |
| Gemma-4-31B-it | 86.3 | 33.3 | 68.3 | 3.9 | 78.6 | 20.7 | — |
| Qwen3.5-397B-A17B | 33.3 | 12.6 | 96.6 | 39.3 | 60.4 | 24.1 | — |
| Qwen3-VL-235B-A22B-Instruct | 1.5 | 0.4 | 100.0 | 92.7 | 43.7 | 40.0 | — |
| Qwen3-VL-8B-Instruct | 0.4 | 0.2 | 99.4 | 91.8 | 42.8 | 39.5 | — |
| Llama-4-Maverick-17B-128E | 3.2 | 0.0 | 99.7 | 75.8 | 44.5 | 32.5 | — |
| Random | 50.0 | 0.4 | 50.0 | 0.4 | 50.0 | 0.4 | — |
| Human (mean of 3) | 100.0 | 100.0 | 99.5 | 99.5 | 99.8 | 99.8 | — |

Q1 recall by flaw type (ours): Glance C5 open-licence release (ours; v4 seed 0, package calibration): NA 0, anatomy_anomaly 32, count_anomaly 66, local_render_structure_anomaly 47, plausibility_anomaly 44, topology_anomaly 41; Glance C5 distilled from C1 (ours): NA 33, anatomy_anomaly 74, count_anomaly 95, local_render_structure_anomaly 79, plausibility_anomaly 72, topology_anomaly 75; Glance C5, same data without the teacher (ours; control): NA 33, anatomy_anomaly 66, count_anomaly 90, local_render_structure_anomaly 77, plausibility_anomaly 64, topology_anomaly 70; Glance C5 + AI-image errors (ours): NA 67, anatomy_anomaly 88, count_anomaly 98, local_render_structure_anomaly 84, plausibility_anomaly 79, topology_anomaly 88; Glance C1 + AI-image errors (ours; teacher, too slow to ship): NA 100, anatomy_anomaly 81, count_anomaly 95, local_render_structure_anomaly 82, plausibility_anomaly 72, topology_anomaly 78; Glance C1 (ours): NA 0, anatomy_anomaly 1, count_anomaly 2, local_render_structure_anomaly 5, plausibility_anomaly 0, topology_anomaly 0; Glance C5 (ours): NA 0, anatomy_anomaly 0, count_anomaly 0, local_render_structure_anomaly 0, plausibility_anomaly 0, topology_anomaly 0; Qwen3-VL-2B (run here): NA 100, anatomy_anomaly 98, count_anomaly 100, local_render_structure_anomaly 100, plausibility_anomaly 100, topology_anomaly 100

## ArtiBench (arXiv 2602.20951), 1,000 generated images, half flawed

| model | accuracy | macro-F1 | AUROC | answers yes |
|---|---|---|---|---|
| **Glance C5 open-licence release (ours; v4 seed 0, package calibration)** | 0.571 | 0.495 | 0.667 | 0.11 |
| **Glance C5 distilled from C1 (ours)** | 0.584 | 0.580 | 0.629 | 0.59 |
| **Glance C5, same data without the teacher (ours; control)** | 0.571 | 0.561 | 0.617 | 0.65 |
| **Glance C5 + AI-image errors (ours)** | 0.597 | 0.596 | 0.633 | 0.56 |
| **Glance C1 + AI-image errors (ours; teacher, too slow to ship)** | 0.593 | 0.592 | 0.645 | 0.55 |
| **Glance C1 (ours)** | 0.499 | 0.375 | 0.502 | 0.06 |
| **Glance C5 (ours)** | 0.500 | 0.333 | 0.517 | 0.00 |
| **Qwen3-VL-2B (run here)** | 0.469 | 0.424 | 0.409 | 0.22 |
| GPT-4o | 0.619 | 0.601 | — | — |
| GPT-5 | 0.599 | 0.577 | — | — |
| Gemini-2.5-Pro | 0.582 | 0.575 | — | — |
| Qwen2.5-VL-7B | 0.501 | 0.336 | — | — |
| InternVL3.5-8B | 0.498 | 0.357 | — | — |
| Random | 0.500 | 0.500 | — | — |

## A-Bench generative distortion (arXiv 2406.03070); ours on the VAL half (250 questions), published on VAL + TEST

| model | accuracy (%) |
|---|---|
| **Glance C5 open-licence release (ours; v4 seed 0, package calibration)** | 36.4 |
| **Glance C5 distilled from C1 (ours)** | 29.6 |
| **Glance C5, same data without the teacher (ours; control)** | 31.2 |
| **Glance C5 + AI-image errors (ours)** | 29.2 |
| **Glance C1 + AI-image errors (ours; teacher, too slow to ship)** | 34.4 |
| **Glance C1 (ours)** | 36.0 |
| **Glance C5 (ours)** | 24.4 |
| **Qwen3-VL-2B (run here)** | 62.8 |
| Qwen2-VL-72B | 70.23 |
| GPT-4o (2024-05-13) | 67.92 |
| LLaVA-NeXT (Qwen-110B) | 63.64 |
| MiniCPM-V2.6 | 60.47 |
| Gemini 1.5 Pro | 59.07 |
| Qwen-VL-Max | 58.56 |
| LLaVA-OneVision-7B | 54.27 |
| InternVL2-40B | 50.10 |
| Random | 33.14 |
| Human (best) | 93.00 |
