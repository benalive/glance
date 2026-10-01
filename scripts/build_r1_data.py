"""Materialise the R1 training mix on disk.

data/r1/decisions.jsonl   one Decision per line, image referenced by `image_row` (no bytes)
data/r1/images.bin        original JPEG bytes of every unique image, concatenated
data/r1/image_offsets.npy (N+1,) byte offsets into images.bin
data/r1/images_256.u8     (N, 256, 256, 3) uint8 memmap, squashed to 256x256 (SigLIP-style resize)
results/r1/data_manifest.json  counts, exclusions, near-duplicate report (tracked in git)
"""
import io
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from glance.data.train_mix import build_r1_mix

OUT = Path("data/r1")
SIZE = 256


def _resize(data: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(data)).convert("RGB").resize((SIZE, SIZE), Image.Resampling.BICUBIC)
    return np.asarray(img, dtype=np.uint8)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    decisions, manifest = build_r1_mix(seed=0)
    rows, blobs = {}, []
    for d in decisions:
        if d.image_id not in rows:
            rows[d.image_id] = len(blobs)
            blobs.append(d.image_bytes)
    offsets = np.zeros(len(blobs) + 1, dtype=np.int64)
    with open(OUT / "images.bin", "wb") as fh:
        for i, b in enumerate(blobs):
            fh.write(b)
            offsets[i + 1] = offsets[i] + len(b)
    np.save(OUT / "image_offsets.npy", offsets)
    store = np.lib.format.open_memmap(OUT / "images_256.u8.npy", mode="w+", dtype=np.uint8, shape=(len(blobs), SIZE, SIZE, 3))
    with ProcessPoolExecutor(8) as pool:
        for i, arr in enumerate(pool.map(_resize, blobs, chunksize=64)):
            store[i] = arr
    store.flush()
    with open(OUT / "decisions.jsonl", "w") as fh:
        for d in decisions:
            rec = asdict(d)
            rec.pop("image_bytes")
            rec["image_row"] = rows[d.image_id]
            fh.write(json.dumps(rec) + "\n")
    manifest.update(unique_images=len(blobs), decisions=len(decisions), image_store_gb=round(store.nbytes / 1e9, 2))
    Path("results/r1").mkdir(parents=True, exist_ok=True)
    Path("results/r1/data_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
