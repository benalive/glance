"""Fetch human-labelled datasets of visible errors in AI-generated images, storing each image as a
512 px JPEG plus one labels line, so the whole set stays small on disk.

  HAD, Human Artifact Dataset (github wangkaihong/HADM; arXiv 2411.13842): SDXL, Midjourney and DALL-E
    images of people with boxes on malformed body parts (hand, arm, leg, foot, face, torso...), each
    severe or mild; per-person "missing/extra <part>" tags; an image-level "good" tag. Official splits
    train_ALL (33,374) / val_ALL (4,180). Read member by member from the remote zip (HTTP range
    requests), so only the requested images are downloaded. Pass the direct zip URL (see the HADM
    README for the share link) and the annotations json (the zip's annotations/, one dict per split).
  RichHF-18K (google-research-datasets/richhf-18k): Pick-a-Pic images with per-image scores from three
    raters, including artifact_score in [0, 1] (1 = no visible artifacts) and an artifact heatmap.
    Official train / validation / test (15,810 / 995 / 955). Images from the mirror
    Exploration/richhf_18k_with_images, one parquet shard at a time, each deleted after use.

Neither dataset states a licence; both contain images from commercial generators. Research use only.

    uv run python scripts/fetch_artifacts.py had --url "$HAD_ZIP_URL" --ann annotations.json --split val_ALL
    uv run python scripts/fetch_artifacts.py richhf --split test
    # -> data/artifacts/{had,richhf}/<split>/*.jpg and data/artifacts/{had,richhf}/<split>.jsonl
"""
import argparse
import io
import json
import os
import subprocess
import tempfile
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

OUT = Path(os.environ.get("GLANCE_DATA_DIR", "data")) / "artifacts"
SIZE, QUALITY = 512, 92


def save_jpeg(data: bytes, path: Path):
    img = Image.open(io.BytesIO(data)).convert("RGB")
    if max(img.size) > SIZE:
        img.thumbnail((SIZE, SIZE), Image.Resampling.BICUBIC)
    img.save(path, format="JPEG", quality=QUALITY)


class RangeFile(io.RawIOBase):
    """Seekable read-only view of a remote file through HTTP range requests (curl)."""

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
        n = self.size - self.pos if n < 0 else n
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        for _ in range(4):
            d = subprocess.run(["curl", "-sL", "-m", "180", "-r", f"{self.pos}-{end}", self.url], capture_output=True).stdout
            if len(d) == end - self.pos + 1:
                break
        self.pos += len(d)
        return d

    def readinto(self, b):
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)


def remote_size(url: str) -> int:
    head = subprocess.run(["curl", "-sIL", url], capture_output=True, text=True).stdout
    return int([ln.split(":")[1] for ln in head.lower().splitlines() if ln.startswith("content-length")][-1])


def had_labels(ann: dict) -> dict:
    parts, severe = {}, {}
    for a in ann.get("annotation", []):
        p = a["body_parts"]
        parts[p] = parts.get(p, 0) + 1
        severe[p] = severe.get(p, 0) + (a.get("level") == "severe")
    human_tags = sorted({t for h in ann.get("human", []) for t in (h.get("tag") or h.get("tags") or [])}
                        if isinstance(ann.get("human"), list) else [])
    return {"tag": ann["image"].get("tag", "").strip(), "parts": parts, "severe": severe, "human_tags": human_tags,
            "prompt": ann["image"].get("prompt", "")}


def fetch_had(url: str, ann_path: str, split: str, limit: int, workers: int):
    anns = json.load(open(ann_path))[split]
    names = sorted(anns)[:limit] if limit else sorted(anns)
    out_dir = OUT / "had" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    size = remote_size(url)
    local = threading.local()

    def get(name):
        stem = name.removesuffix(".json")
        dst = out_dir / f"{stem}.jpg"
        if not dst.exists():
            if not hasattr(local, "zf"):
                local.zf = zipfile.ZipFile(io.BufferedReader(RangeFile(url, size), buffer_size=1 << 20))
            save_jpeg(local.zf.read(f"human_artifact_dataset/images/{split}/{anns[name]['image']['file_name']}"), dst)
        return {"file": f"{split}/{stem}.jpg", "source": stem.split("_")[0], **had_labels(anns[name])}

    rows = []
    with ThreadPoolExecutor(workers) as ex:
        for i, row in enumerate(ex.map(get, names)):
            rows.append(row)
            if (i + 1) % 250 == 0:
                print(f"had {split}: {i + 1}/{len(names)}", flush=True)
    (OUT / "had" / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"had {split}: {len(rows)} images -> {out_dir}", flush=True)


def fetch_richhf(split: str):
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    repo = "Exploration/richhf_18k_with_images"
    shards = sorted(s.rfilename for s in HfApi().dataset_info(repo).siblings if s.rfilename.startswith(f"data/{split}-"))
    out_dir = OUT / "richhf" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for shard in shards:
        with tempfile.TemporaryDirectory(dir=OUT) as tmp:
            path = hf_hub_download(repo, shard, repo_type="dataset", local_dir=tmp)
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=256):
                for r in batch.to_pylist():
                    img = r.get("image")
                    data = img["bytes"] if isinstance(img, dict) else img
                    stem = Path(r["filename"]).stem
                    save_jpeg(data, out_dir / f"{stem}.jpg")
                    rows.append({"file": f"{split}/{stem}.jpg", **{k: r[k] for k in ("artifact_score", "aesthetics_score",
                                                                                   "misalignment_score", "overall_score") if k in r}})
        print(f"richhf {split}: {shard} done, {len(rows)} images", flush=True)
    (OUT / "richhf" / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["had", "richhf"])
    ap.add_argument("--split", required=True)
    ap.add_argument("--url", default=os.environ.get("HAD_ZIP_URL"), help="HAD: direct URL of the dataset zip")
    ap.add_argument("--ann", help="HAD: annotations json with one dict per split")
    ap.add_argument("--limit", type=int, default=0, help="HAD: first N images (sorted by name)")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.dataset == "had":
        fetch_had(a.url, a.ann, a.split, a.limit, a.workers)
    else:
        fetch_richhf(a.split)


if __name__ == "__main__":
    main()
