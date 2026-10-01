# Glance C5 (open) — model card

Glance C5 is a 134M-parameter vision-language model that answers typed questions about an image, with
a probability for every answer: **yes/no**, **choice** among options you give, and **score** on a scale you
define. It runs on a CPU (about 18-26 ms for a new image and 3-4 ms for a question about an image it has
already seen, with ONNX Runtime at 4 threads), and never generates text. It uses the same request format as Jev's `POST /v1/systemone`,
with an image added.

- **Weights, code, and this card:** Apache-2.0. Base models: SigLIP2-B/32 (Apache-2.0) and Ettin-32M (MIT).
- **Training data:** only images and annotations whose licences allow commercial use and adaptation (see
  "Training data").
- **Version:** `glance-c5-open-v4`. The run is `results/r4/C5_open4_s0` and the code commit is in `config.json`.

## What it is good for

- **A first-pass check for visible errors in AI-generated images** (malformed hands, faces, bodies and objects,
  rendering glitches), where missing some flaws is acceptable. Real photographs are rarely flagged: 0.8% of
  questions about everyday photos (2.0% of photos across three error questions), and 0.0–0.5% of close-up
  portraits in every demographic group tested. Read the caveat below: much of what it detects is that an image
  is generated.
- **Object presence:** "Is there a dog in the image?"
- **Quick, cheap yes/no and multiple-choice checks inside an agent loop.** You ask many questions about one
  image and pay for the image once.

## What it is not

- **Not a reliable flaw detector for images from newer generators.** On SalArt-VQA, where a flaw was inpainted
  into 119 generated images, it scores the flawed image higher in only 55% of pairs
  (95% interval 46–63%), and it flags 78% of
  the clean originals (84% of the Imagen 4 ones). The research models trained on non-commercial error data
  behave the same way. Among generated images alone, its flaw ranking is moderate (AUROC 0.67 on ArtiBench,
  0.70 on HAD, 0.83 on RichHF-18K). Asking about zoomed tiles does not help, and neither did retraining
  with procedural local anomalies (paper §7).
- **Not an AI-image detector either.** It was trained to answer "no errors" for clean generated images, and it
  does so for many older-generator images (96% specificity on ArtiBench), but not reliably.
- **Not a general visual question answerer.** On everyday yes/no questions about people and scenes it is
  right 60% of the time (Qwen3-VL-2B: 81%).
- **It cannot say where an error is**, and it does not handle multiple-choice options written as long
  descriptions.
- **English questions only.** Images are read at 256×256 px, so small details are lost.
- **Do not use it as a reward model or data filter to train a text-to-image generator.** Some training images
  come from Playground v2.5, whose licence forbids using outputs to improve another text-to-image model.

## How to use

```bash
python scripts/download_weights.py      # https://github.com/benalive/glance, release weights-v4
uv run python -m glance.serve --ckpt data/release/glance-c5-open-v4 --port 8089   # CPU by default; 8089 is the default port
curl -s localhost:8089/v1/systemone -H 'Content-Type: application/json' -d '{
  "image": "<base64 image or data URL>",
  "questions": {
    "flawed": {"type": "noul", "instructions": "Does this image contain visible generation errors?"},
    "hands":  {"type": "noul", "instructions": "Are any hands in this image malformed?"},
    "animal": {"type": "choice", "instructions": "Which animal is shown?", "criteria": ["cat", "dog", "horse"]}
  }}'
```

- **yes/no:** `noul` = P(yes).
- **choice:** the most likely option, plus a probability for every option.
- **score:** the expected level (0 = lowest), plus probabilities.

**Phrase error questions positively** ("Does this image contain …?"). Negated or unusual phrasings
("Could this pass as a faultless photograph?") can reverse the answer (see "Limitations").

## Performance

These are results for the released model on held-out sets, using its own calibration as it is served:
- The yes/no threshold was fitted only on the licence-clean calibration split, never on these benchmarks.
- "Report split" means the part of each benchmark not used for any fitting.
- 95% intervals resample images.

**Everyday questions**

| benchmark | result |
|---|---|
| VQAv2 validation yes/no (images not in training; 5-5 annotator ties excluded), accuracy | 0.602 [0.589, 0.615] |
| POPE object presence, adversarial / random / popular | 0.769 / 0.887 / 0.848 |
| A-OKVQA multiple choice (4 options) | 0.388 [0.356, 0.420] |
| MMStar (chance 0.25) | 0.280 |
| SugarCrepe replace-relation / swap-attribute | 0.619 / 0.547 |
| KonIQ-10k image quality (rank correlation) | 0.515 (weak) |

**Visible errors in AI-generated images**

| benchmark | result |
|---|---|
| SalArt-VQA Q1 ("salient artifact present?"), 437 flawed + 318 real | AUROC 0.911; recall 44.9%, specificity 97.2%, accuracy 66.9% |
| SalArt-VQA Q1, 119 inpainted flaws vs their clean generated originals | flawed scored higher in 55% of pairs [46, 63]; 78% of the clean originals flagged |
| ArtiBench (1,000 generated images, half flawed) | AUROC 0.667; accuracy 0.571, macro-F1 0.495 |
| HAD val: malformed hands / any anatomy error | AUROC 0.695 / 0.702 |
| RichHF-18K test: visible artifacts / plausibility | AUROC 0.828 / rank correlation 0.453 |
| Real everyday photos flagged as flawed | 0.8% |
| A-Bench generative distortion (descriptive options, 250 validation questions) | 36.4% (chance 34.9%): not supported |

**Reading these numbers:**
- **A conservative threshold.** At the served threshold the model flags few real photos: it finds 45% of
  SalArt's flawed images, 18% of ArtiBench's and 29% of HAD's malformed hands, while passing 97% of SalArt's
  real photos. If you need recall, compare P(yes) with a
  lower threshold of your own, or use the probabilities to rank images.
- **Published leaderboards are not like-for-like.** The published SalArt and ArtiBench rows used long
  instruction prompts and graded the generated text; this model gets the short question and reads the
  option probabilities. For reference: SalArt Q1 accuracy is Gemini 3.1 Pro 78.3, Claude 4.6 Opus 70.0,
  GPT-5.4 45.5; ArtiBench accuracy is InternVL3.5-8B 0.630 and
  Qwen2.5-VL-7B 0.627 (both fine-tuned by the ArtiBench authors), GPT-4o 0.619, GPT-5 0.599.
- **SalArt's usual score compares flawed generated images with real photographs.** Its 0.911 mostly measures
  telling generated from real; the paired row above is the controlled flaw test.
- **Calibration.** Probabilities are reliable on everyday yes/no and two-way questions (smooth ECE 0.02–0.08),
  less so on four-way multiple choice (0.11–0.15). On error benchmarks they range from reliable (HAD any error
  0.05, SalArt 0.09) to overconfident (HAD hands 0.21, ArtiBench 0.30, RichHF 0.35).
- **SalArt was consulted during development.** Its images and question were never trained on, but three data
  revisions targeted failures on its wording. Treat it as validation for wording robustness, not a clean test.
- **HAD and RichHF were not trained on** (their licences do not allow it). They are cross-dataset tests for
  this model.

## Limitations

- **Wording.** Error questions are trained with hundreds of generated phrasings. Still, a negated phrasing far
  from those can invert the answer. Across three training seeds of this recipe, SalArt's question scored
  AUROC 0.91 / 0.55 / 0.89, and this release is the first seed.
- **Weak areas.** Everyday reasoning questions, image quality and multiple choice with long options remain weak.
- **Generators.** Training flaws come from open image generators (Stable Diffusion 1.x–XL, PixArt, Kandinsky 3,
  Playground v2.5 and others). Images from generators outside training behave unpredictably: on SalArt it
  flags 84% of clean Imagen 4 images (and finds 75% of flawed ones), but finds only 20–38% of flawed FLUX,
  Z-Image and Qwen images.

## Fairness

FairFace (10,954 real portraits labelled with race, gender and age; CC BY 4.0; evaluation only) measures how
often a real portrait is flagged as flawed:

| question | every race, gender and age group |
|---|---|
| "Does this image contain visible generation errors?" | 0.0–0.3% by race and gender, 0.0–0.5% by age |
| "Are there any deformed body parts in this image?" | 0.0% |
| "Is anyone's face distorted or malformed?" | 0.0% |

An earlier candidate flagged 18–33% of portraits, most often of Black people (33%). Adding real close-up
portraits to training brought it down to the rates above; part of the drop may come from the threshold,
which was refitted at the same time (paper §8). At a matched operating point
(both models finding 45% of SalArt's flawed images) it flags 0.37% of portraits against 0.06% for the
research model, and 1.1% of portraits of Black people against 0.1–0.5% for other groups. Performance on
flawed images of different groups is not measured.

## Speed and size

134M parameters. The release package is 1.0 GB: fp32 safetensors (542 MB) plus the same model as ONNX
files (510 MB). The server runs the ONNX files through ONNX Runtime when `onnxruntime` is installed (CPU), and
PyTorch otherwise; the answers are the same (largest difference in P(yes) 0.0001). Median latency per request,
batch 1, 4 threads (the default), image ≤ 512 px, otherwise idle machine, no network:

| machine | runtime | new image, 1 question | new image, 4 questions | cached image, 1 question | cached image, 8 questions |
|---|---|---|---|---|---|
| x86: AMD EPYC 9V45 (Azure VM, AVX-512) | ONNX Runtime | 26.2 ms | 30.4 ms | 4.1 ms | 13.2 ms |
| x86: AMD EPYC 9V45 | PyTorch fp32 | 57.3 ms | 64.1 ms | 7.3 ms | 22.5 ms |
| Apple M4 Pro | ONNX Runtime | 18.3 ms | 19.1 ms | 3.0 ms | — |
| Apple M4 Pro | PyTorch fp32 | 23.0 ms | 24.4 ms | 5.1 ms | — |

- **Threads:** keep the default of 4. On the EPYC VM, ONNX Runtime with 32 threads was 5-7x *slower*
  (181 ms per new image).
- **Memory:** peak 959 MB for a serving process with ONNX Runtime, 1.45 GB with PyTorch.
- **Rejected speed-ups:** int8 quantisation is faster but changes 10-13% of answers, so it is not offered.
  bf16 halves new-image time in PyTorch on x86, but ONNX is faster and exact.
- **Variation:** runs vary by 10-20%. A 12 MP JPEG adds about 41 ms (up to 77 ms) to decode and resize, so resize on the
  client. Measurements: `results/latency/`.

## Training data

Only sources whose licences allow commercial use and adaptation. Every training image has an attribution
record in `attribution.jsonl`.

- **Photos:**
  - COCO train2014 photos under CC BY 2.0, "no known copyright restrictions" or US Government work (13,543
    photos; NonCommercial, NoDerivs and ShareAlike photos excluded);
  - Open Images V7 photos (CC BY 2.0), with face crops from its human-drawn boxes.
- **Questions:**
  - VQAv2 and GQA (CC BY 4.0), A-OKVQA (Apache-2.0);
  - COCO and Open Images object labels (CC BY 4.0);
  - synthetic image-quality distortions.
- **AI-generated images with human flaw labels:**
  - EvalMuse-40K (BSD-3-Clause), from 13 open generators whose licences leave outputs free for this use;
  - ImageRewardDB (Apache-2.0; Stable Diffusion images from DiffusionDB, CC0).

  Closed generators and those with output restrictions (Midjourney, Dreamina, SDXL-Turbo, SD3, DeepFloyd IF,
  Kolors, HunyuanDiT) are excluded after a review of each generator's output licence.
- **Checks:**
  - no training image is the same photo as any evaluation image (COCO id, Flickr id or perceptual hash);
  - every training image passes `scripts/check_clean_licences.py`.

This is an engineering reading of the licences, not legal advice.

## Citation

Glance: Fast Typed Decisions about Images on a CPU. Technical report, 2026 (`paper/glance.tex`).
