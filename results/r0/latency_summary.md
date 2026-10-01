Source: `results/r0/latency_sweep.json` (commit 080a8f8, 2026-09-22T20:49:43). batch 1; warmup 15, 60 timed iters (25 if p50>150ms); device sync per iter; random weights. MPS fp16; CPU fp32 on 4 threads. L4 = analytic projection, not a measurement.

**mps fit** (plain encoder grid): ms = 0.11 + 0.244*layers + 0.159*GFLOPs; R^2 0.999, MAPE 4.4%, implied 6.30 TFLOPS; overhead probe 204 us/layer.
**cpu fit** (plain encoder grid): ms = -0.10 + 0.519*layers + 0.521*GFLOPs; R^2 0.999, MAPE 3.9%, implied 1.92 TFLOPS; overhead probe 177 us/layer.

| candidate | params (M) | q | depth cold/cached | MPS cold p50 | MPS cached p50 | CPU cold p50 | CPU cached p50 | GFLOPs cold/cached | L4 proj cold/cached |
|---|---|---|---|---|---|---|---|---|---|
| C3/C4 single tower 12x768 (B/32 shape) | 183 | 1 | 12/12 | 8.4 | 6.4 | 26.2 | 14.9 | 20/8 | 1.73/1.26 |
| C6 tiny tower 6x512 | 59 | 1 | 6/6 | 3.5 | 2.5 | 9.0 | 4.1 | 5/2 | 0.63/0.52 |
| C5 fusion (B/32 + Ettin-32M + 3 fusion) | 127 | 1 | 25/13 | 11.5 | 5.2 | 22.1 | 5.3 | 13/2 | 1.91/0.78 |
| C1 ModernVBERT @512px | 252 | 1 | 34/22 | 50.1 | 12.2 | 179.7 | 29.1 | 243/26 | 6.23/1.82 |
| C1 ModernVBERT @256px | 252 | 1 | 34/22 | 20.2 | 10.4 | 58.1 | 18.9 | 62/14 | 3.07/1.82 |
| C2 ModernVBERT text 11L @512px | 197 | 1 | 23/11 | 45.1 | 6.2 | 165.6 | 14.3 | 230/13 | 5.58/0.99 |
| C0 SigLIP2-B/32 dual encoder | 377 | 1 | 24/12 | 17.6 | 11.1 | 49.7 | 32.4 | 56/44 | 2.46/1.46 |
| C0 SmolVLM-256M causal readout | 256 | 1 | 42/30 | 55.3 | 12.7 | 183.0 | 23.0 | 239/11 | 6.45/2.31 |
| teacher Qwen3-VL-2B readout | 2128 | 1 | 52/28 | 135.0 | 81.1 | — | — | 494/319 | 16.53/12.43 |
| C3/C4 single tower 12x768 (B/32 shape) | 183 | 8 | 12/12 | 18.5 | 15.2 | 68.0 | 56.2 | 84/72 | 2.26/2.01 |
| C6 tiny tower 6x512 | 59 | 8 | 6/6 | 5.3 | 4.4 | 21.8 | 17.7 | 20/17 | 0.76/0.70 |
| C5 fusion (B/32 + Ettin-32M + 3 fusion) | 127 | 8 | 25/13 | 13.5 | 6.9 | 38.4 | 22.4 | 29/18 | 1.91/0.97 |
| C1 ModernVBERT @512px | 252 | 8 | 34/22 | 67.7 | 28.7 | 240.9 | 82.4 | 330/112 | 7.97/3.19 |
| C1 ModernVBERT @256px | 252 | 8 | 34/22 | 38.5 | 25.8 | 111.9 | 74.4 | 146/99 | 4.30/2.92 |
| C2 ModernVBERT text 11L @512px | 197 | 8 | 23/11 | 54.0 | 15.2 | 189.3 | 39.3 | 274/56 | 6.45/1.67 |
| C0 SigLIP2-B/32 dual encoder | 377 | 8 | 24/12 | 64.9 | 58.5 | 222.3 | 197.3 | 364/353 | 8.30/7.64 |
| C0 SmolVLM-256M causal readout | 256 | 8 | 42/30 | 73.3 | 26.9 | 233.3 | 80.8 | 324/93 | 8.14/3.10 |
| teacher Qwen3-VL-2B readout | 2128 | 8 | 52/28 | 289.9 | 235.7 | — | — | 1484/1309 | 31.70/27.35 |

| stem (256 px, 12 layers) | tokens | GFLOPs | MPS p50 | CPU p50 |
|---|---|---|---|---|
| ViT-12L w384 patch16 | 256 | 12.2 | 6.31 | 16.53 |
| ViT-12L w768 patch16 | 256 | 46.2 | 12.33 | 36.82 |
| ViT-12L w384 patch32 | 64 | 2.9 | 4.56 | 6.20 |
| ViT-12L w768 patch32 | 64 | 11.3 | 6.75 | 16.15 |
| conv stem 4x(3x3 s2) + unshuffle | 64 | 2.1 | 0.84 | 3.41 |
