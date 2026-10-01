"""Real close-up portraits for the licence-clean error mix: crops around human faces in Open Images photos.

The first licence-clean release model flagged 18-33% of FairFace's real portraits as having generation
errors: its real-photo negatives were mostly wide scenes while many flawed training
images are people. These crops give it real close-ups. Source: Open Images V7 validation/test "Human face"
boxes (human-drawn, CC BY 4.0 annotations) on the CC BY 2.0 photos already fetched by fetch_clean.py; group
boxes, depictions (drawings, statues, screens) and faces under MIN_FACE px in the stored 512 px photo are
skipped. Each crop is a square of CONTEXT x the face size centred on the face, clipped to the photo, stored
as a <= 512 px JPEG with image id "<photo id>#face<k>" (so it follows its photo's calibration/dev split)
and an attribution record derived from the photo's.

With --fetch N, first fetches up to N more Open Images validation/test photos (CC BY 2.0, same rules as
fetch_clean.py) that have a face box at least FETCH_MIN_REL of the photo's width, into data/clean/oi_faces/.

    uv run python scripts/make_face_crops.py --fetch 6000   # -> data/clean/faces/, data/clean/attribution_faces.jsonl
"""
import argparse
import csv
import io
import json
import os
from collections import defaultdict
from pathlib import Path

from PIL import Image

from glance.data.train_mix import dhash

DATA = Path(os.environ.get("GLANCE_DATA_DIR", "data"))
RAW, CLEAN = DATA / "raw" / "clean", DATA / "clean"
FACE = "/m/0dzct"
MIN_FACE, CONTEXT, MAX_PER_PHOTO = 64, 2.2, 2
FETCH_MIN_REL = 0.15


def fetch_more(n: int, have: set):
    import random

    from scripts.fetch_clean import OI_LICENCE, download_all

    big = set()
    for f in ("validation-annotations-bbox.csv", "test-annotations-bbox.csv"):
        for r in csv.DictReader(open(RAW / f)):
            if (r["LabelName"] == FACE and r["IsGroupOf"] == "0" and r["IsDepiction"] == "0"
                    and float(r["XMax"]) - float(r["XMin"]) >= FETCH_MIN_REL):
                big.add(r["ImageID"])
    meta = []
    for f in ("validation-images-with-rotation.csv", "test-images-with-rotation.csv"):
        meta += [r for r in csv.DictReader(open(RAW / f)) if r["ImageID"] in big and r["ImageID"] not in have
                 and r["License"] == OI_LICENCE and r["Rotation"] in ("", "0.0")]
    random.Random(0).shuffle(meta)
    jobs = [{"image_id": f"oi:{r['ImageID']}", "file": f"{r['ImageID']}.jpg",
             "urls": [u for u in (r["Thumbnail300KURL"], f"https://s3.amazonaws.com/open-images-dataset/{r['Subset']}/{r['ImageID']}.jpg") if u],
             "source": f"Open Images V7 {r['Subset']} (Flickr)", "license": r["License"], "original": r["OriginalLandingURL"],
             "author": r["Author"], "author_url": r["AuthorProfileURL"], "title": r["Title"]} for r in meta[:n]]
    download_all(jobs, CLEAN / "oi_faces", CLEAN / "attribution_oi_faces.jsonl", 32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", type=int, default=0)
    a = ap.parse_args()
    photos = {}
    for line in open(CLEAN / "attribution_oi.jsonl"):
        r = json.loads(line)
        photos[r["image_id"].split(":")[1]] = r
    if a.fetch:
        fetch_more(a.fetch, set(photos))
    if (CLEAN / "attribution_oi_faces.jsonl").exists():
        for line in open(CLEAN / "attribution_oi_faces.jsonl"):
            r = json.loads(line)
            photos.setdefault(r["image_id"].split(":")[1], r)
    boxes = defaultdict(list)
    for f in ("validation-annotations-bbox.csv", "test-annotations-bbox.csv"):
        for r in csv.DictReader(open(RAW / f)):
            if r["LabelName"] == FACE and r["ImageID"] in photos and r["IsGroupOf"] == "0" and r["IsDepiction"] == "0":
                boxes[r["ImageID"]].append(tuple(float(r[k]) for k in ("XMin", "XMax", "YMin", "YMax")))
    out_dir = CLEAN / "faces"
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(CLEAN / "attribution_faces.jsonl", "w") as fh:
        for oid in sorted(boxes):
            img = Image.open(photos[oid]["path"]).convert("RGB")
            W, H = img.size
            faces = sorted(boxes[oid], key=lambda b: -(b[1] - b[0]) * (b[3] - b[2]))[:MAX_PER_PHOTO]
            for k, (x0, x1, y0, y1) in enumerate(faces):
                w, h = (x1 - x0) * W, (y1 - y0) * H
                if min(w, h) < MIN_FACE:
                    continue
                side = CONTEXT * max(w, h)
                cx, cy = (x0 + x1) / 2 * W, (y0 + y1) / 2 * H
                box = (max(0, cx - side / 2), max(0, cy - side / 2), min(W, cx + side / 2), min(H, cy + side / 2))
                crop = img.crop(tuple(round(v) for v in box))
                crop.thumbnail((512, 512), Image.Resampling.BICUBIC)
                buf = io.BytesIO()
                crop.save(buf, "JPEG", quality=92)
                path = out_dir / f"{oid}_face{k}.jpg"
                path.write_bytes(buf.getvalue())
                rec = {k2: v for k2, v in photos[oid].items() if k2 not in ("dhash", "path", "image_id")}
                fh.write(json.dumps(rec | {"image_id": f"oi:{oid}#face{k}", "path": str(path), "dhash": str(dhash(buf.getvalue())),
                                           "derived": f"crop around a human face of oi:{oid}"}) + "\n")
                n += 1
    print(f"{n} face crops from {len(boxes)} photos -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
