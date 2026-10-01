"""Serving latency of trained candidates on this machine, the way the server runs them.

Each model is loaded as glance.serve.Predictor (CPU, 4 threads by default), then answers questions
about real images: a new image (image encoder + questions) and the same image again (questions only).
Medians over --n images, after a warm-up. Run on an otherwise idle machine.

    uv run python scripts/latency_check.py --runs C5:data/ckpt/C5_ep2_lrA_s0.pt,C1:data/ckpt/C1_ep2_lrA_s0.pt
    # -> results/latency/serving_cpu.{json,md}
"""
import argparse
import glob
import json
import statistics
import time
from pathlib import Path

from glance.serve import Predictor

QUESTIONS = ["Is there a person in the image?", "Does this image contain visible generation errors?",
             "Is there a dog in the image?", "Does this image look correct?"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="comma list of CAND:checkpoint")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--images", default="data/artifacts/richhf/test/*.jpg", help="glob of test images")
    ap.add_argument("--runtime", default="auto", choices=["auto", "torch", "onnx"])
    ap.add_argument("--out", help="output name under results/latency/ (default serving_<device>)")
    a = ap.parse_args()
    images = [Path(p).read_bytes() for p in sorted(glob.glob(a.images))[: a.n + 5]]
    rows = {}
    for spec in a.runs.split(","):
        cand, ckpt = spec.split(":", 1)
        pred = Predictor(ckpt, cand, None, a.device, a.threads, runtime=a.runtime)
        res = {}
        for nq in (1, 4):
            qs = {f"q{i}": {"type": "noul", "instructions": QUESTIONS[i]} for i in range(nq)}
            for img in images[:5]:  # warm-up
                pred.predict(qs, img)
            new, cached = [], []
            for img in images[5:5 + a.n]:
                pred.cache.clear()
                t0 = time.perf_counter()
                pred.predict(qs, img)
                new.append(1000 * (time.perf_counter() - t0))
                t0 = time.perf_counter()
                pred.predict(qs, img)
                cached.append(1000 * (time.perf_counter() - t0))
            res[f"{nq}q"] = {"new_image_ms": statistics.median(new), "same_image_ms": statistics.median(cached)}
        rows[f"{cand} ({Path(ckpt).stem}, {pred.runtime})"] = res
        print(cand, Path(ckpt).stem, json.dumps({k: {m: round(v, 1) for m, v in d.items()} for k, d in res.items()}), flush=True)
        del pred
    out = Path("results/latency")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.out or 'serving_' + a.device}.json").write_text(json.dumps(rows, indent=1))
    lines = [f"Serving latency, {a.device}, {a.threads} threads, median over {a.n} images ({a.images}) (ms).", "",
             "| model | new image, 1 question | same image, 1 question | new image, 4 questions | same image, 4 questions |",
             "|---|---|---|---|---|"]
    for name, r in rows.items():
        lines.append(f"| {name} | {r['1q']['new_image_ms']:.1f} | {r['1q']['same_image_ms']:.1f} | "
                     f"{r['4q']['new_image_ms']:.1f} | {r['4q']['same_image_ms']:.1f} |")
    (out / f"{a.out or 'serving_' + a.device}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
