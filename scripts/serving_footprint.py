"""Serving footprint of a release package: peak memory per runtime, and the cost of decoding a large photo.

Memory: each runtime runs in a fresh process that loads the package, answers 20 requests (new images, 4
questions) and reports its peak resident set size. Decode: the server's own decode-and-resize step
(Image.open, convert, bicubic resize to the model's input) on 12 MP JPEGs, made by upscaling 10 COCO photos to
4000x3000 and saving at quality 90; median of 3 runs each. HTTP is not included.

    uv run python scripts/serving_footprint.py     # -> results/latency/release_footprint.{json,md}
"""
import io
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

PACKAGE = "weights/glance-c5-open-v4"
PHOTOS = sorted(Path("data/clean/coco").glob("*.jpg"))[:20]

CHILD = """
import resource, sys
from pathlib import Path
from glance.serve import Predictor
pred = Predictor(sys.argv[1], device="cpu", threads=4, runtime=sys.argv[2])
qs = {f"q{i}": {"type": "noul", "instructions": q} for i, q in enumerate([
    "Does this image contain visible generation errors?", "Is there a person in the image?",
    "Is it daytime?", "Are any hands in this image malformed?"])}
for p in sys.argv[3:]:
    pred.predict(qs, Path(p).read_bytes())
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(peak / 2**20 if sys.platform == "darwin" else peak / 2**10)
"""


def peak_mb(runtime: str) -> float:
    out = subprocess.run([sys.executable, "-c", CHILD, PACKAGE, runtime, *map(str, PHOTOS)], capture_output=True, text=True, check=True)
    return float(out.stdout.strip().splitlines()[-1])


def decode_ms(size: int) -> list[float]:
    times = []
    for p in PHOTOS[:10]:
        buf = io.BytesIO()
        Image.open(p).convert("RGB").resize((4000, 3000), Image.Resampling.BICUBIC).save(buf, "JPEG", quality=90)
        data = buf.getvalue()
        runs = []
        for _ in range(3):
            t = time.perf_counter()
            img = Image.open(io.BytesIO(data)).convert("RGB")
            np.asarray(img.resize((size, size), Image.Resampling.BICUBIC), dtype=np.uint8)
            runs.append(1000 * (time.perf_counter() - t))
        times.append(float(np.median(runs)))
    return times


def main():
    size = json.loads((Path(PACKAGE) / "config.json").read_text()).get("image_size", 256)
    res = {"machine": f"{platform.machine()} {platform.platform()}", "package": PACKAGE,
           "peak_rss_mb": {rt: peak_mb(rt) for rt in ("onnx", "torch")}}
    d = decode_ms(size)
    res["decode_12mp_ms"] = {"median": float(np.median(d)), "min": min(d), "max": max(d), "n_photos": len(d)}
    Path("results/latency/release_footprint.json").write_text(json.dumps(res, indent=1))
    L = [f"Serving footprint, {PACKAGE}, {res['machine']}.", "",
         f"- Peak RSS (load + 20 new-image requests, 4 questions): ONNX Runtime {res['peak_rss_mb']['onnx']:.0f} MB, "
         f"PyTorch {res['peak_rss_mb']['torch']:.0f} MB.",
         f"- Decode and resize a 12 MP JPEG to {size} px: median {res['decode_12mp_ms']['median']:.1f} ms "
         f"(range {min(d):.1f}-{max(d):.1f} over {len(d)} photos)."]
    Path("results/latency/release_footprint.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
