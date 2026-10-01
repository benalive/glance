"""Do any images of the external benchmarks also appear in our training data?

dHash (glance.data.train_mix.dhash) of every image in SalArt-VQA, ArtiBench and A-Bench (generative),
against every training image: the R1 mix (data/r1/images.bin), HAD train + val and RichHF-18K (all
splits). Near-duplicates are hash pairs within HAMMING_MAX bits; the coarse 8x8 hash over-reports, so
every hit is a candidate to inspect, not a confirmed duplicate.

    uv run python scripts/overlap_check.py   # -> results/external/overlap.json
"""
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from glance.data.benchmarks import ALL_BENCHMARKS
from glance.data.train_mix import HAMMING_MAX, dhash, near_duplicates


def _hash_file(path: str) -> int | None:
    try:
        return dhash(Path(path).read_bytes())
    except OSError:
        return None  # unreadable file; reported below


def _hash_blob(args) -> int:
    path, off, length = args
    with open(path, "rb") as fh:
        fh.seek(off)
        data = fh.read(length)
    try:
        return dhash(data)
    except OSError:
        return None


def main():
    art = Path("data/artifacts")
    files = sorted(str(p) for p in list((art / "had").glob("*/*.jpg")) + list((art / "richhf").glob("*/*.jpg")))
    offsets = np.load("data/r1/image_offsets.npy")
    blobs = [("data/r1/images.bin", int(offsets[i]), int(offsets[i + 1] - offsets[i])) for i in range(len(offsets) - 1)]
    with ProcessPoolExecutor(8) as ex:
        hashed = list(ex.map(_hash_file, files, chunksize=256)) + list(ex.map(_hash_blob, blobs, chunksize=256))
    names = files + [f"r1:{i}" for i in range(len(blobs))]
    unreadable = [n for n, h in zip(names, hashed) if h is None]
    train = [h for h in hashed if h is not None]
    print(f"{len(train)} training images hashed; unreadable: {unreadable[:10]} ({len(unreadable)})", flush=True)
    report = {"training_images": len(train), "unreadable": unreadable, "hamming_max": HAMMING_MAX, "benches": {}}
    for bench in ["salart_q1", "artibench", "abench_generative"]:
        ds = {d.image_id: d for d in ALL_BENCHMARKS[bench]()}
        ids = list(ds)
        hits = near_duplicates([dhash(ds[i].image_bytes) for i in ids], train)
        report["benches"][bench] = {"images": len(ids), "near_duplicates": len(hits), "ids": sorted(ids[k] for k in hits)}
        print(bench, report["benches"][bench]["images"], "images,", len(hits), "near-duplicates of training images", flush=True)
    Path("results/external").mkdir(parents=True, exist_ok=True)
    Path("results/external/overlap.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
