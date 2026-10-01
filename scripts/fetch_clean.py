"""Fetch the licence-clean images for the open Glance model, with an attribution record for each.

Only images whose licence allows commercial use and adaptation, without share-alike, are kept:
  coco  COCO train2014 images (the VQAv2 training images) whose Flickr licence in the 2017
        annotations is id 4 (CC BY 2.0), 7 (no known copyright restrictions) or 8 (US Government
        work). NonCommercial, NoDerivs and ShareAlike images are left out.
  oi    Open Images V7 validation + test images (all listed as CC BY 2.0), sampled among those with
        at least one human-verified positive and one verified negative label of a common class
        (the labels give object-presence questions with real "no" answers). Images with a recorded
        rotation are skipped. Downloaded from Flickr's 640 px copy, or the S3 original if that is gone.
Every image is stored as a <= 512 px JPEG under data/clean/{coco,oi}/ and gets a line in
data/clean/attribution_<source>.jsonl (licence, original URL, author when known) plus its dHash
(glance.data.train_mix.dhash). Evaluation images are excluded by the mix builder, not here.

Needs the metadata in data/raw/clean/ (annotations_trainval2017.zip, v2_Questions_Train_mscoco.zip,
Open Images validation/test image CSVs and human image-label CSVs, oidv7-class-descriptions.csv).

    uv run python scripts/fetch_clean.py coco
    uv run python scripts/fetch_clean.py oi --n 30000
"""
import argparse
import csv
import io
import json
import os
import random
import threading
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageOps

from glance.data.train_mix import dhash

DATA = Path(os.environ.get("GLANCE_DATA_DIR", "data"))
RAW, OUT = DATA / "raw" / "clean", DATA / "clean"
SIZE, QUALITY = 512, 92
COCO_ALLOWED = {  # id -> licence URL as listed in the COCO annotations (checked on load)
    4: "http://creativecommons.org/licenses/by/2.0/",
    7: "http://flickr.com/commons/usage/",
    8: "http://www.usa.gov/copyright.shtml",
}
OI_LICENCE = "https://creativecommons.org/licenses/by/2.0/"
OI_MIN_POSITIVES = 200  # a class counts as common with at least this many verified positives (val + test)


def fetch(url: str, tries: int = 3) -> bytes:
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "glance-fetch/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))


def save_jpeg(data: bytes, path: Path) -> bytes:
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    img.thumbnail((SIZE, SIZE), Image.Resampling.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=QUALITY)
    path.write_bytes(buf.getvalue())
    return buf.getvalue()


def download_all(jobs: list[dict], out_dir: Path, attribution: Path, workers: int = 16):
    """jobs: {"image_id", "file", "urls": [...], **attribution fields}. Resumable: images already on
    disk and in the attribution file are skipped."""
    out_dir.mkdir(parents=True, exist_ok=True)
    done = set()
    if attribution.exists():
        done = {json.loads(line)["image_id"] for line in open(attribution)}
    todo = [j for j in jobs if j["image_id"] not in done]
    print(f"{len(jobs)} images, {len(done)} already fetched, {len(todo)} to go", flush=True)
    lock, failed, n = threading.Lock(), Counter(), [0]

    def one(job):
        path = out_dir / job["file"]
        for url in job["urls"]:
            try:
                data = save_jpeg(fetch(url), path)
                break
            except Exception as e:
                failed[type(e).__name__] += 1
        else:
            return
        rec = {k: v for k, v in job.items() if k not in ("urls", "file")} | {"path": str(path), "dhash": str(dhash(data))}
        with lock:
            with open(attribution, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            n[0] += 1
            if n[0] % 1000 == 0:
                print(f"  {n[0]}/{len(todo)} fetched, failures {dict(failed)}", flush=True)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, todo))
    print(f"done: {n[0]} fetched, failures {dict(failed)}", flush=True)


def coco_jobs() -> list[dict]:
    with zipfile.ZipFile(RAW / "annotations_trainval2017.zip") as z:
        caps = [json.loads(z.read(f"annotations/captions_{s}2017.json")) for s in ("train", "val")]
    licences = {l["id"]: l for l in caps[0]["licenses"]}
    for i, url in COCO_ALLOWED.items():
        assert licences[i]["url"] == url, (i, licences[i])
    with zipfile.ZipFile(RAW / "v2_Questions_Train_mscoco.zip") as z:
        train2014 = {q["image_id"] for q in json.loads(z.read("v2_OpenEnded_mscoco_train2014_questions.json"))["questions"]}
    images = [im for c in caps for im in c["images"]]
    by_lic = Counter(im["license"] for im in images if im["id"] in train2014)
    print("train2014 images by licence id:", dict(sorted(by_lic.items())), flush=True)
    jobs = []
    for im in images:
        if im["id"] in train2014 and im["license"] in COCO_ALLOWED:
            jobs.append({"image_id": f"coco:{im['id']}", "file": f"{im['id']:012d}.jpg",
                         "urls": [im["coco_url"]], "source": "COCO train2014 (Flickr)",
                         "license_id": im["license"], "license": licences[im["license"]]["url"],
                         "license_name": licences[im["license"]]["name"], "original": im["flickr_url"]})
    return jobs


def oi_labels() -> tuple[dict, dict, Counter]:
    """Human-verified image-level labels: (present classes per image, verified-absent classes per image,
    positives per class). The files write Confidence as "1", "1.0" or "0.0" (clean_v1 audit B1: a test for
    "1" alone had sent the "1.0" positives to the absent side)."""
    pos, neg, count = defaultdict(set), defaultdict(set), Counter()
    for f in ("oidv7-val-annotations-human-imagelabels.csv", "oidv7-test-annotations-human-imagelabels.csv"):
        for r in csv.DictReader(open(RAW / f)):
            conf = float(r["Confidence"])
            assert conf in (0.0, 1.0), r
            if conf == 1.0:
                pos[r["ImageID"]].add(r["LabelName"])
                count[r["LabelName"]] += 1
            else:
                neg[r["ImageID"]].add(r["LabelName"])
    return pos, neg, count


def oi_jobs(n: int, seed: int) -> list[dict]:
    pos, neg, count = oi_labels()
    common = {c for c, k in count.items() if k >= OI_MIN_POSITIVES}
    meta = []
    for f in ("validation-images-with-rotation.csv", "test-images-with-rotation.csv"):
        meta += list(csv.DictReader(open(RAW / f)))
    ok = [r for r in meta if r["License"] == OI_LICENCE and r["Rotation"] in ("", "0.0")
          and pos[r["ImageID"]] & common and neg[r["ImageID"]] & common]
    print(f"Open Images: {len(meta)} images, {len(common)} common classes, {len(ok)} eligible", flush=True)
    random.Random(seed).shuffle(ok)
    return [{"image_id": f"oi:{r['ImageID']}", "file": f"{r['ImageID']}.jpg",
             "urls": [u for u in (r["Thumbnail300KURL"],
                                  f"https://s3.amazonaws.com/open-images-dataset/{r['Subset']}/{r['ImageID']}.jpg") if u],
             "source": f"Open Images V7 {r['Subset']} (Flickr)", "license": r["License"],
             "original": r["OriginalLandingURL"], "author": r["Author"], "author_url": r["AuthorProfileURL"],
             "title": r["Title"]} for r in ok[:n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["coco", "oi"])
    ap.add_argument("--n", type=int, default=30000, help="Open Images: how many images")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.what == "coco":
        download_all(coco_jobs(), OUT / "coco", OUT / "attribution_coco.jsonl", a.workers)
    else:
        download_all(oi_jobs(a.n, a.seed), OUT / "oi", OUT / "attribution_oi.jsonl", a.workers)


if __name__ == "__main__":
    main()
