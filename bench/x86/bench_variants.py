"""Which CPU speed-ups keep Glance's answers? Latency and fidelity of runtime variants against fp32.

Variants (each applied to glance.serve.Predictor, the served path):
  fp32         the baseline, as served today
  bf16         torch.autocast("cpu", bfloat16) around the image encoder and the question path
  int8         dynamic int8 quantisation of every nn.Linear (fbgemm on x86, qnnpack on ARM)
  compile      torch.compile on the image encoder and on the question path (dynamic shapes)
  onnx-image   the image encoder exported to ONNX and run with ONNX Runtime; question path in PyTorch
  onnx-full    the package's image.onnx and questions.onnx (both paths in ONNX Runtime; needs --package with them)
Latency: new image (1 and 4 questions) and cached image (1, 4 and 8 questions), median over --n photos, at
the thread counts given. Fidelity: on --check photos x 4 questions, the largest difference in P(yes) from
fp32 and the share of answers whose yes/no side flips, among answers fp32 gives with a margin over 0.05
(an untrained model answers ~50% everywhere, so use --package for a meaningful fidelity check). A variant that fails is reported, not fatal.

    ./run.sh --variants                                  # all variants, 4 threads and all threads
    ./run.sh --variants --package /path/to/glance-c5-open-v4
"""
import argparse
import glob
import json
import os
import tempfile
import time
import traceback
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from bench import QUESTIONS, cpu_info, ms, stats, untrained_predictor
from glance.serve import Predictor

VARIANTS = ["fp32", "bf16", "int8", "compile", "onnx-image", "onnx-full"]


def make(variant: str, package: str | None):
    """(predictor, context manager factory for each request)."""
    threads = torch.get_num_threads()  # Predictor sets its own thread count (default 4): keep the one under test
    if variant == "onnx-full":
        if not package:
            raise ValueError("onnx-full needs --package with image.onnx / questions.onnx")
        return Predictor(package, device="cpu", threads=threads, runtime="onnx"), nullcontext
    pred = Predictor(package, device="cpu", threads=threads, runtime="torch") if package else untrained_predictor()
    torch.set_num_threads(threads)
    ctx = nullcontext
    if variant == "bf16":
        ctx = lambda: torch.autocast("cpu", dtype=torch.bfloat16)  # noqa: E731
    elif variant == "int8":
        engines = torch.backends.quantized.supported_engines
        torch.backends.quantized.engine = "fbgemm" if "fbgemm" in engines else "qnnpack"
        pred.cand = torch.ao.quantization.quantize_dynamic(pred.cand, {torch.nn.Linear}, dtype=torch.qint8)
    elif variant == "compile":
        pred.cand.frozen_image = torch.compile(pred.cand.frozen_image)
        pred.cand.forward = torch.compile(pred.cand.forward, dynamic=True)
    elif variant == "onnx-image":
        import onnxruntime as ort

        img = pred.cand.frozen_image
        path = Path(tempfile.mkdtemp()) / "image.onnx"
        torch.onnx.export(img, (torch.randn(1, 3, pred.cand.image_size, pred.cand.image_size),), str(path),
                          input_names=["pixels"], output_names=["tokens"])
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = torch.get_num_threads()
        sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])

        def image_features(pixels):
            out = sess.run(None, {"pixels": pixels.numpy().astype(np.float32)})[0]
            return torch.from_numpy(out).half().float()

        pred.cand.image_features = image_features
    return pred, ctx


def answers(pred, ctx, images, qs) -> np.ndarray:
    out = []
    for img in images:
        pred.cache.clear()
        with ctx():
            a = pred.predict(qs, img)["answers"]
        out.append([a[k]["noul"] for k in qs])
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--check", type=int, default=20)
    ap.add_argument("--threads", default="4,all")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    a = ap.parse_args()
    if a.package and not (Path(a.package) / "model.safetensors").is_file():
        raise SystemExit(f"--package {a.package}: no model.safetensors there; copy the whole glance-c5-open-v4 folder "
                         "to the VM (scp -r) and pass its path")
    images = [Path(p).read_bytes() for p in sorted(glob.glob(str(Path(__file__).parent / "images" / "*.jpg")))]
    q4 = {f"q{i}": {"type": "noul", "instructions": QUESTIONS[i]} for i in range(4)}
    out = {"cpu": cpu_info(), "model": a.package or "untrained, public base models", "results": {}}
    ref = {}
    for t in a.threads.split(","):
        n_threads = os.cpu_count() if t == "all" else int(t)
        torch.set_num_threads(n_threads)
        for variant in a.variants.split(","):
            key = f"{variant} @ {n_threads} threads"
            try:
                t0 = time.perf_counter()
                pred, ctx = make(variant, a.package)
                for img in images[:5]:  # warm-up (compile happens here)
                    for nq in (1, 4, 8):
                        pred.cache.clear()
                        with ctx():
                            pred.predict({f"q{i}": {"type": "noul", "instructions": QUESTIONS[i]} for i in range(nq)}, img)
                res = {"setup_s": round(time.perf_counter() - t0, 1)}
                for nq in (1, 4):
                    qs = {f"q{i}": {"type": "noul", "instructions": QUESTIONS[i]} for i in range(nq)}
                    new = []
                    for img in images[5:5 + a.n]:
                        pred.cache.clear()
                        with ctx():
                            new.append(ms(lambda: pred.predict(qs, img)))
                    res[f"new_image_{nq}q"] = stats(new)
                for nq in (1, 4, 8):
                    qs = {f"q{i}": {"type": "noul", "instructions": QUESTIONS[i]} for i in range(nq)}
                    same = []
                    for img in images[5:5 + a.n]:
                        pred.cache.clear()
                        with ctx():
                            pred.predict(qs, img)
                            same.append(ms(lambda: pred.predict(qs, img)))
                    res[f"cached_image_{nq}q"] = stats(same)
                p = answers(pred, ctx, images[-a.check:], q4)
                if variant == "fp32":
                    ref[n_threads] = p
                if n_threads in ref:
                    d = np.abs(p - ref[n_threads])
                    sure = np.abs(ref[n_threads] - 0.5) > 0.05  # flips of near-50% answers mean nothing (untrained model)
                    flips = (p >= 0.5) != (ref[n_threads] >= 0.5)
                    res["fidelity_vs_fp32"] = {"max_abs_p_diff": round(float(d.max()), 4), "mean_abs_p_diff": round(float(d.mean()), 4),
                                               "flips_where_fp32_margin_over_0.05": round(float(flips[sure].mean()), 4) if sure.any() else None,
                                               "n_answers_with_margin": int(sure.sum())}
                out["results"][key] = res
                print(f"{key}: new 1q {res['new_image_1q']['median']} ms, cached 1q {res['cached_image_1q']['median']} ms, "
                      f"cached 8q {res['cached_image_8q']['median']} ms, fidelity {res.get('fidelity_vs_fp32')}", flush=True)
            except Exception as e:
                out["results"][key] = {"error": f"{type(e).__name__}: {e}"[:500]}
                print(f"{key}: FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
                traceback.print_exc(limit=2)
    (Path(__file__).parent / "result_variants.json").write_text(json.dumps(out, indent=1))
    print("\n=== paste everything below back ===")
    print(json.dumps({"cpu": {k: v for k, v in out["cpu"].items() if k != "torch_config"}, "model": out["model"],
                      "results": out["results"]}, indent=1))


if __name__ == "__main__":
    main()
