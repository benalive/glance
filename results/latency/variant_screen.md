Latency screen of in-budget candidates for a larger student (2026-09-26). Untrained weights, timed through
glance.serve.Predictor.predict on the CPU (M4 Pro, 4 threads, fp32 unless noted), median over 25 RichHF test
images after 5 warm-up images (ms). "new" = image encoder + questions; "same" = cached image, questions only.
int8 = torch.ao dynamic quantisation of every nn.Linear (qnnpack).

| model | params | new image, 1 q | same image, 1 q | new image, 4 q | same image, 4 q |
|---|---|---|---|---|---|
| C5 (SigLIP2-B/32 @256 + Ettin-32M + 3 fusion) | 134M | 23.7 | 4.7 | 24.8 | 6.4 |
| C5, int8 | 134M | 44.0 | 6.3 | 49.1 | 11.4 |
| C5 with 6 fusion layers | 141M | 25.1 | 6.0 | 26.9 | 8.2 |
| C5 with Ettin-68M | 176M | 28.7 | 10.1 | 32.0 | 13.5 |
| C5 with Ettin-150M | 273M | 37.4 | 18.3 | 47.4 | 28.8 |
| ModernVBERT @192 px (9 image tokens), 11 of 22 text layers | | 38.3 | 8.0 | 41.1 | 10.9 |
| ModernVBERT @192 px, 6 text layers (C9) | 172M | 35.0 | 4.6 | 36.4 | 6.4 |
| ModernVBERT @192 px, 4 text layers | | 33.6 | 3.3 | 34.8 | 4.6 |

ModernVBERT's vision tower + connector alone: 29.6 ms at 192 px, 39.4 ms at 256 px, 151.7 ms at 512 px
(224 px does not divide into its 4x4 pixel shuffle). int8 is slower than fp32 at batch 1 on this CPU for every model.
