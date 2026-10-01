"""Fetch licence-clean AI-generated images with human labels of visible flaws, for the open Glance model.

  evalmuse     EvalMuse-40K (HF DY-Evalab/EvalMuse; annotations BSD-3-Clause), train list only (the test
               list has no labels). Three annotators per image mark structural problems (hands: deformed
               fingers, extra/missing fingers, malformed palm, overlapping hands; face: distorted features,
               missing/extra parts; limbs: twisted, extra/missing, out of proportion; animals; objects).
               Only images from generators whose licences leave outputs free for this use are kept
               (output licences reviewed per generator): SD 1.2/1.5/2.1, SDXL,
               SDXL-Lightning, LCM-SDXL, SSD-1B, LCM-SSD1B, PixArt-alpha/Sigma, LCM-PixArt, Kandinsky 3,
               Playground v2.5. Closed (Midjourney, Dreamina) and restricted (SDXL-Turbo, SD3, DeepFloyd IF,
               Kolors, HunyuanDiT) generators are left out. Images are read one by one from the 55 GB split
               zip through HTTP range requests; only the kept ones are downloaded.
  imagereward  ImageRewardDB (HF zai-org/ImageRewardDB, Apache-2.0; Stable Diffusion images from DiffusionDB,
               CC0) with annotators' fidelity rating 1-7 ("is the output true to the shape and
               characteristics the object should have"). Every image rated <= 3 and a random sample of the
               others (--per-rating per rating level 4-7), read from the per-shard zips by range requests.
Every image is stored as a <= 512 px JPEG under data/clean/{evalmuse,imagereward}/, with one line per image in
data/clean/labels_<source>.jsonl (labels) and data/clean/attribution_<source>.jsonl (licence record).

    uv run python scripts/fetch_clean_artifacts.py evalmuse
    uv run python scripts/fetch_clean_artifacts.py imagereward
"""
import argparse
import io
import json
import os
import random
import subprocess
import threading
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from glance.data.train_mix import dhash

DATA = Path(os.environ.get("GLANCE_DATA_DIR", "data"))
OUT = DATA / "clean"
SIZE, QUALITY = 512, 92
EVALMUSE = "https://huggingface.co/datasets/DY-Evalab/EvalMuse/resolve/main/"
IMAGEREWARD = "https://huggingface.co/datasets/zai-org/ImageRewardDB/resolve/main/"
# generator folder -> output licence (reviewed per generator); every one leaves outputs free for training a detector
EVALMUSE_GENERATORS = {
    "SD_v1.2": "CreativeML OpenRAIL-M", "SD_v1.5": "CreativeML OpenRAIL-M",
    "SD_v2.1": "CreativeML Open RAIL++-M", "SDXL": "CreativeML Open RAIL++-M",
    "SDXL-Lightning": "CreativeML Open RAIL++-M", "LCM-SDXL": "CreativeML Open RAIL++-M",
    "LCM-SSD1B": "CreativeML Open RAIL++-M", "PixArt-alpha": "CreativeML Open RAIL++-M",
    "LCM-PixArt": "CreativeML Open RAIL++-M", "PixArt-Sigma": "CreativeML Open RAIL++-M",
    "SSD1B": "Apache-2.0", "Kandinsky3": "Apache-2.0",
    "Playground_v2.5": "Playground v2.5 Community License (outputs not to be used to improve a text-to-image model)",
}


class RangeFile(io.RawIOBase):
    """Seekable read-only view of a remote file through HTTP range requests (curl retries every error;
    the HF CDN drops some connections)."""

    def __init__(self, url: str, size: int):
        self.url, self.size, self.pos = url, size, 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def read(self, n=-1):
        n = self.size - self.pos if n < 0 else min(n, self.size - self.pos)
        if n <= 0:
            return b""
        for _ in range(5):
            d = subprocess.run(["curl", "-sL", "--retry", "8", "--retry-all-errors", "--retry-delay", "2", "-m", "300",
                                "-r", f"{self.pos}-{self.pos + n - 1}", self.url], capture_output=True).stdout
            if len(d) == n:
                break
        self.pos += len(d)
        return d

    def readinto(self, b):
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)


class MultiRangeFile(io.RawIOBase):
    """One file split into parts (split -b): a seekable view over the concatenation."""

    def __init__(self, urls: list[str], sizes: list[int]):
        self.parts = [RangeFile(u, s) for u, s in zip(urls, sizes)]
        self.starts = [sum(sizes[:i]) for i in range(len(sizes))]
        self.size, self.pos = sum(sizes), 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def read(self, n=-1):
        n = self.size - self.pos if n < 0 else min(n, self.size - self.pos)
        out = []
        while n > 0:
            i = max(j for j, s in enumerate(self.starts) if s <= self.pos)
            part = self.parts[i]
            part.seek(self.pos - self.starts[i])
            d = part.read(min(n, part.size - part.pos))
            if not d:
                break
            out.append(d)
            self.pos += len(d)
            n -= len(d)
        return b"".join(out)

    def readinto(self, b):
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)


def remote_size(url: str) -> int:
    head = subprocess.run(["curl", "-sIL", "--retry", "8", "--retry-all-errors", url], capture_output=True, text=True).stdout
    return int([ln.split(":")[1] for ln in head.lower().splitlines() if ln.startswith("content-length")][-1])


def save_jpeg(data: bytes, path: Path) -> bytes:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img.thumbnail((SIZE, SIZE), Image.Resampling.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=QUALITY)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.getvalue())
    return buf.getvalue()


def fetch_members(opener, jobs: list[dict], source: str, workers: int):
    """jobs: {"image_id", "member", "file", "labels", "attribution"}; `opener()` returns a fresh ZipFile
    (one per thread: a zip reader is not thread-safe). Resumable through the labels file."""
    labels_path, att_path = OUT / f"labels_{source}.jsonl", OUT / f"attribution_{source}.jsonl"
    done = {json.loads(line)["image_id"] for line in open(labels_path)} if labels_path.exists() else set()
    todo = [j for j in jobs if j["image_id"] not in done]
    print(f"{source}: {len(jobs)} images, {len(done)} done, {len(todo)} to go", flush=True)
    local, lock, n, failed = threading.local(), threading.Lock(), [0], Counter()

    def one(job):
        try:
            if not hasattr(local, "zf"):
                local.zf = opener()
            data = save_jpeg(local.zf.read(job["member"]), job["file"])
        except Exception as e:
            failed[type(e).__name__] += 1
            return
        with lock:
            with open(labels_path, "a") as fh:
                fh.write(json.dumps({"image_id": job["image_id"], "path": str(job["file"]), **job["labels"]}) + "\n")
            with open(att_path, "a") as fh:
                fh.write(json.dumps({"image_id": job["image_id"], "path": str(job["file"]), "dhash": str(dhash(data)),
                                     **job["attribution"]}) + "\n")
            n[0] += 1
            if n[0] % 500 == 0:
                print(f"  {n[0]}/{len(todo)} fetched, failures {dict(failed)}", flush=True)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, todo))
    print(f"{source}: done, {n[0]} fetched, failures {dict(failed)}", flush=True)


def evalmuse(workers: int):
    rows = json.load(open(DATA / "raw" / "clean" / "evalmuse" / "train_list.json"))
    parts = [f"images.zip.part-a{c}" for c in "abcdef"]
    sizes = [remote_size(EVALMUSE + p) for p in parts]
    names = {}
    zf = zipfile.ZipFile(io.BufferedReader(MultiRangeFile([EVALMUSE + p for p in parts], sizes), buffer_size=1 << 20))
    for name in zf.namelist():  # the member for "SDXL/00110.png" may sit under a top folder
        key = "/".join(name.split("/")[-2:])
        names[key] = name
    kept = [r for r in rows if r["img_path"].split("/")[0] in EVALMUSE_GENERATORS]
    print(f"evalmuse: {len(rows)} labelled images, {len(kept)} from allowed generators, "
          f"{sum(r['img_path'] in names for r in kept)} found in the zip", flush=True)
    jobs = []
    for r in kept:
        if r["img_path"] not in names:
            continue
        gen, stem = r["img_path"].split("/")[0], Path(r["img_path"]).stem
        jobs.append({"image_id": f"evalmuse:{gen}/{stem}", "member": names[r["img_path"]],
                     "file": OUT / "evalmuse" / gen / f"{stem}.jpg",
                     "labels": {"generator": gen, "prompt": r["prompt"], "fidelity_label": r.get("fidelity_label") or [],
                                "total_score": r.get("total_score")},
                     "attribution": {"source": f"EvalMuse-40K ({gen})", "license": f"generator output: {EVALMUSE_GENERATORS[gen]}",
                                     "annotations": "EvalMuse-40K, BSD-3-Clause", "original": r["img_path"]}})

    def opener():
        return zipfile.ZipFile(io.BufferedReader(MultiRangeFile([EVALMUSE + p for p in parts], sizes), buffer_size=1 << 20))

    fetch_members(opener, jobs, "evalmuse", workers)


def imagereward(per_rating: int, workers: int, seed: int):
    import pandas as pd
    from huggingface_hub import hf_hub_download

    frames = []
    for split in ("train", "validation", "test"):
        p = hf_hub_download("zai-org/ImageRewardDB", f"metadata-{split}.parquet", repo_type="dataset",
                            local_dir=DATA / "raw" / "clean" / "imagereward")
        frames.append(pd.read_parquet(p).assign(split=split))
    df = pd.concat(frames)
    rng = random.Random(seed)
    pick = []
    for rating, g in df.groupby("fidelity_rating"):
        rows = g.to_dict("records")
        pick += rows if rating <= 3 else rng.sample(rows, min(per_rating, len(rows)))
    print(f"imagereward: {len(df)} rated images, picked {len(pick)}: "
          f"{dict(Counter(int(r['fidelity_rating']) for r in pick))}", flush=True)
    by_zip = defaultdict(list)
    for r in pick:
        shard = r["image_path"].split("/")[2]  # images/<split>/<split>_<n>/<uuid>.webp
        by_zip[f"images/{r['split']}/{shard}.zip"].append(r)
    for zpath, rows in sorted(by_zip.items()):
        url = IMAGEREWARD + zpath
        size = remote_size(url)
        jobs = [{"image_id": f"imagereward:{Path(r['image_path']).stem}", "member": Path(r["image_path"]).name,
                 "file": OUT / "imagereward" / f"{Path(r['image_path']).stem}.jpg",
                 "labels": {k: (int(r[k]) if k.endswith("rating") else r[k]) for k in
                            ("prompt", "fidelity_rating", "overall_rating", "image_text_alignment_rating", "split")},
                 "attribution": {"source": "ImageRewardDB (DiffusionDB, Stable Diffusion 1.x)",
                                 "license": "ImageRewardDB Apache-2.0; DiffusionDB images CC0", "original": r["image_path"]}}
                for r in rows]

        def opener(url=url, size=size):
            return zipfile.ZipFile(io.BufferedReader(RangeFile(url, size), buffer_size=1 << 20))

        fetch_members(opener, jobs, "imagereward", workers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["evalmuse", "imagereward"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--per-rating", type=int, default=1500, help="imagereward: images per fidelity rating 4-7")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.what == "evalmuse":
        evalmuse(a.workers)
    else:
        imagereward(a.per_rating, a.workers, a.seed)


if __name__ == "__main__":
    main()
