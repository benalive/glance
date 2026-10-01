"""Glance C5 serving latency on this machine's CPU, the way the server runs it (glance.serve.Predictor).

Loads the release package if --package points at one; otherwise builds the same architecture from the
public base models (SigLIP2-B/32 @256, Ettin-32M) with untrained fusion layers, which runs exactly as fast.
Times, for 1, 2, 4 and all threads: a new image (decode + resize + image encoder + questions) with 1 and 4
questions, and a cached image with 1, 2, 4 and 8 questions. Median and p95 over --n real photos after a
warm-up. Prints a report to paste back and writes result.json.

    ./run.sh                                  # sets up Python and runs this
    ./run.sh --package /path/to/glance-c5-open-v4
"""
import argparse
import glob
import json
import os
import platform
import statistics
import subprocess
import threading
import time
from collections import OrderedDict
from pathlib import Path

import torch

from glance.serve import Predictor

QUESTIONS = ["Does this image contain visible generation errors?", "Is there a person in the image?",
             "Are any hands in this image malformed?", "Is there a dog in the image?", "Is this photo taken outdoors?",
             "Is there a car in the image?", "Does this image look blurry?", "Is anyone smiling?"]


def cpu_info() -> dict:
    info = {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(),
            "torch": torch.__version__, "logical_cpus": os.cpu_count()}
    try:
        lscpu = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
        for line in lscpu.splitlines():
            k, _, v = line.partition(":")
            if k.strip() in ("Model name", "Thread(s) per core", "Core(s) per socket", "Socket(s)", "CPU max MHz", "Flags"):
                info[k.strip()] = v.strip() if k.strip() != "Flags" else " ".join(
                    f for f in v.split() if f.startswith(("avx512", "avx2", "amx", "fma")))
    except FileNotFoundError:
        pass
    info["torch_config"] = [ln.strip() for ln in torch.__config__.show().splitlines() if "CPU capability" in ln or "MKL" in ln or "oneDNN" in ln][:4]
    return info


def untrained_predictor() -> Predictor:
    """The release architecture from the public base models (untrained fusion layers and head): same speed."""
    from transformers import AutoTokenizer

    from glance.model.candidates import build_candidate

    p = object.__new__(Predictor)
    p.device, p.cand_id, p.ckpt, p.notes = torch.device("cpu"), "C5", "untrained (public base models)", ""
    p.cand = build_candidate("C5").eval()
    p.tok = AutoTokenizer.from_pretrained("jhu-clsp/ettin-encoder-32m")
    p.calib, p.cache, p.cache_size, p.lock = {}, OrderedDict(), 64, threading.Lock()
    p.onnx, p.runtime, p.image_size, p.cached_image = None, "torch", p.cand.image_size, p.cand.cached_image
    return p


def ms(f) -> float:
    t0 = time.perf_counter()
    f()
    return 1000 * (time.perf_counter() - t0)


def stats(xs: list[float]) -> dict:
    xs = sorted(xs)
    return {"median": round(statistics.median(xs), 2), "p95": round(xs[min(len(xs) - 1, int(0.95 * len(xs)))], 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", help="release package dir (optional)")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--threads", default="1,2,4,all")
    ap.add_argument("--runtime", default="auto", choices=["auto", "torch", "onnx"], help="with --package")
    a = ap.parse_args()
    if a.package and not (Path(a.package) / "model.safetensors").is_file():
        raise SystemExit(f"--package {a.package}: no model.safetensors there; copy the whole glance-c5-open-v4 folder "
                         "to the VM (scp -r) and pass its path")
    images = [Path(p).read_bytes() for p in sorted(glob.glob(str(Path(__file__).parent / "images" / "*.jpg")))]
    assert len(images) >= a.n + 5, "images/ missing"
    info = cpu_info()
    pred = Predictor(a.package, device="cpu", threads=os.cpu_count(), runtime=a.runtime) if a.package else untrained_predictor()
    info["model"] = (a.package or "untrained, public base models") + f" ({pred.runtime})"
    results = {}
    for t in a.threads.split(","):
        n_threads = os.cpu_count() if t == "all" else int(t)
        if n_threads > os.cpu_count():
            continue
        torch.set_num_threads(n_threads)
        res = {}
        for nq in (1, 4):
            qs = {f"q{i}": {"type": "noul", "instructions": QUESTIONS[i]} for i in range(nq)}
            for img in images[:5]:
                pred.cache.clear()
                pred.predict(qs, img)
            new = []
            for img in images[5:5 + a.n]:
                pred.cache.clear()
                new.append(ms(lambda: pred.predict(qs, img)))
            res[f"new_image_{nq}q"] = stats(new)
        for nq in (1, 2, 4, 8):
            qs = {f"q{i}": {"type": "noul", "instructions": QUESTIONS[i]} for i in range(nq)}
            same = []
            for img in images[5:5 + a.n]:
                pred.cache.clear()
                pred.predict(qs, img)  # encode the image once
                same.append(ms(lambda: pred.predict(qs, img)))
            res[f"cached_image_{nq}q"] = stats(same)
        results[f"{n_threads} threads"] = res
        print(f"{n_threads} threads: " + ", ".join(f"{k} {v['median']} ms" for k, v in res.items()), flush=True)
    out = {"cpu": info, "results": results, "n_images": a.n}
    (Path(__file__).parent / "result.json").write_text(json.dumps(out, indent=1))
    print("\n=== paste everything below back ===")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
