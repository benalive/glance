# Glance open release vs other models

Accuracy as each model is used (Glance open: its licence-clean calibration; research Glance: calibration fitted on the calibration splits of POPE-adv, A-OKVQA, KonIQ, HAD and RichHF; zero-shot models: raw answers). Glance rows use offline GPU logits; served CPU answers agree on 99.4-99.9% of items. Report splits, except SalArt and ArtiBench (all items, like the published rows). Published API rows used long prompts and graded generated text, so they are indicative, not like-for-like.

## Everyday questions (accuracy)

| benchmark | **Glance open** | Glance C5 research | Glance C1 research | Qwen3-VL-2B | SmolVLM-256M |
|---|---|---|---|---|---|
| VQAv2 everyday yes/no | 0.602 | 0.572 | 0.688 | 0.811 | 0.636 |
| POPE adversarial | 0.769 | 0.766 | 0.793 | 0.840 | 0.603 |
| POPE random | 0.887 | 0.905 | 0.912 | — | — |
| A-OKVQA (4-way) | 0.388 | 0.449 | 0.598 | 0.748 | 0.462 |
| MMStar (chance .25) | 0.280 | 0.235 | 0.324 | — | 0.316 |
| SugarCrepe relation | 0.619 | 0.628 | 0.706 | 0.890 | 0.535 |
| SugarCrepe attribute | 0.547 | 0.549 | 0.760 | 0.939 | 0.536 |
| KonIQ quality (SRCC) | 0.515 | 0.819 | 0.807 | 0.635 | 0.402 |

## Visible errors in AI-generated images

| benchmark | **Glance open** | Glance C5 research | Glance C1 research | Qwen3-VL-2B | SmolVLM-256M | published (accuracy) |
|---|---|---|---|---|---|---|
| SalArt-VQA Q1 (artifact?), accuracy | 0.669 | 0.830 | 0.856 | 0.584 | — | Gemini 3.1 Pro 0.783; Gemini 3.1 Flash Lite 0.876; Claude 4.6 Opus 0.700; GPT-5.4 0.455 |
| SalArt-VQA Q1 (artifact?), AUROC | 0.911 | 0.904 | 0.928 | 0.725 | — | not published |
| ArtiBench (flawed?), accuracy | 0.571 | 0.584 | 0.593 | 0.469 | — | InternVL3.5-8B (fine-tuned by the ArtiBench authors) 0.630; Qwen2.5-VL-7B (fine-tuned by the ArtiBench authors) 0.627; GPT-4o 0.619; GPT-5 0.599; Gemini-2.5-Pro 0.582 |
| ArtiBench (flawed?), AUROC | 0.667 | 0.629 | 0.645 | 0.409 | — | not published |
| HAD malformed hands, AUROC | 0.695 | 0.781 | 0.783 | 0.638 | — | — |
| HAD any anatomy error, AUROC | 0.702 | 0.802 | 0.799 | 0.593 | — | — |
| RichHF-18K artifacts, AUROC | 0.828 | 0.912 | 0.941 | 0.652 | — | — |
| real photos flagged as flawed | 0.008 | 0.018 | 0.018 | 0.148 | — | — |

HAD and RichHF are the research models' own training distributions (in-distribution for them, cross-dataset for the open release). The open release's threshold is conservative: at 97% specificity on SalArt it finds 45% of flawed images.

## Latency (median, batch 1)

| model | machine | runtime | new image, 1 question (ms) | question on a cached image (ms) | protocol / source |
|---|---|---|---|---|---|
| Glance C5 open release | Apple M4 Pro CPU, 4 threads | ONNX Runtime | 18.3 | 3.0 | served path, 40 COCO photos; results/latency/release_m4pro_onnx.json |
| Glance C5 open release | AMD EPYC 9V45 (x86) CPU, 4 threads | ONNX Runtime | 26.2 | 4.1 | served path, 30 COCO photos; results/latency/x86_azure_epyc9v45_onnx.json |
| Glance C5 open release | Apple M4 Pro CPU, 4 threads | PyTorch fp32 | 23.0 | 5.1 | served path, 40 COCO photos; results/latency/release_m4pro_torch.json |
| Glance C5 open release | AMD EPYC 9V45 (x86) CPU, 4 threads | PyTorch fp32 | 57.3 | 7.3 | served path, 30 COCO photos; results/latency/x86_azure_epyc9v45_onnx.json |
| Glance C1 research (ModernVBERT, 252M) | Apple M4 Pro CPU, 4 threads | PyTorch fp32 | 166.5 | 24.1 | served path, 40 RichHF images; results/latency/serving_cpu.json |
| SmolVLM-256M | Apple M4 Pro CPU, 4 threads | PyTorch fp32 | 183.0 | 23.0 | synthetic input, random weights, model only (R0); results/r0/latency_summary.md |
| SmolVLM-256M | Apple M4 Pro GPU (MPS) | PyTorch fp16 | 55.3 | 12.7 | as above |
| Qwen3-VL-2B | Apple M4 Pro GPU (MPS) | PyTorch fp16 | 135.0 | 81.1 | as above |
| Laya-snake 322M | Apple M4 Pro CPU, 4 threads | PyTorch | 40.3 (text only) | — | one text decision on the Snake benchmark; no image input |
| laya-vision | NVIDIA L4 GPU | PyTorch bf16 | 32-41 | — | published figure, not measured here |
| Gemini / GPT / Claude | cloud API | — | not measured | — | network round trip dominates |

## Laya (text decisions, Snake benchmark, same input and data)

| model | input | p50 latency (M4 Pro CPU, 4 threads) | top-1 on held-out boards (report split) |
|---|---|---|---|
| Laya-snake 322M | text (ASCII board + facts) | 40.3 ms | 0.981 |
| Glance C5 text path | identical text | 13.7 ms | 0.981 (same move on every board) |
| Glance C5 image + facts | board image + fact lines | 31.4 ms | 0.981 |

Laya is text-only (no image input); on identical input and data Glance is about 3x faster with equal accuracy. Jev was not compared (no API key).
