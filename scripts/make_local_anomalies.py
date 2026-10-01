"""Procedural local anomalies with matched controls, added to an error mix (default v4 -> data/clean_art_v5).

The released model cannot tell SalArt's inpainted flaws (mostly small count and structure anomalies) from the
clean originals, and SD 1.5 inpainting did not produce visible flaws. This makes local
structural anomalies procedurally, in the spirit of CutPaste (Li et al., CVPR 2021), on clean generated images:
  duplicate  a patch of the region pasted again with an offset (a doubled part: an extra finger, an extra object)
  swirl      a strong local twist (a melted or twisted part)
  tear       the region cut into strips shifted against each other (broken geometry)
  foreign    a patch from another generated image blended in (an out-of-place part)
Every source image also gets a control: the same region through the same feathered elliptical blend and resampling with
no structural change (zero offset, near-zero twist, zero shift, its own patch) plus a slight colour change, so
"was edited" cannot stand in for "is flawed". The flawed image is asked a "visible generation errors" wording
with answer "errors", the control with answer "none".

Sources: training images judged clean by their annotators (EvalMuse-40K with no structural mark, from cleared
generators except Playground v2.5; ImageRewardDB fidelity >= 6), never calibration-split images. 10% of source
images (by hash) are held out: their pairs are written to data/clean/anomalies/test.jsonl for evaluation only.

    uv run python scripts/make_local_anomalies.py --n 4000     # -> data/clean/anomalies/, data/clean_art_v5/
"""
import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import scripts.build_clean_artifact_mix as B
from scripts.build_clean_mix import is_calib

OUT = Path("data/clean/anomalies")
SKIP = {"Playground_v2.5", "HunyuanDiT", "SDXL-Turbo", "SD3", "IF", "Kolors", "Dreamina_v2.0Pro", "Midjourney_v6.1"}
OPS = ["duplicate", "swirl", "tear", "foreign"]


def split_of(image_id: str) -> str:
    return "test" if int(hashlib.sha1(f"anom:{image_id}".encode()).hexdigest(), 16) % 10 == 0 else "train"


def sources():
    lic = {}
    for f in ("data/clean/attribution_evalmuse.jsonl", "data/clean/attribution_imagereward.jsonl"):
        for line in open(f):
            r = json.loads(line)
            lic[r["image_id"]] = r["license"]
    out = []
    for line in open("data/clean/labels_evalmuse.jsonl"):
        r = json.loads(line)
        if r["generator"] not in SKIP and r["fidelity_label"] == [] and not is_calib(r["image_id"]):
            out.append((r["image_id"], r["path"], lic[r["image_id"]]))
    for line in open("data/clean/labels_imagereward.jsonl"):
        r = json.loads(line)
        if r.get("fidelity_rating", 0) >= 6 and r.get("split") == "train" and not is_calib(r["image_id"]):
            out.append((r["image_id"], r["path"], lic[r["image_id"]]))
    return out


def region(w, h, rng):
    area = rng.uniform(0.04, 0.12) * w * h
    ar = rng.uniform(0.6, 1.6)
    bw, bh = max(16, min(w // 2, int((area * ar) ** 0.5))), max(16, min(h // 2, int((area / ar) ** 0.5)))
    cx = min(max(int(rng.gauss(w / 2, w / 5)), bw), w - bw)
    cy = min(max(int(rng.gauss(h / 2, h / 5)), bh), h - bh)
    return cx - bw // 2, cy - bh // 2, bw, bh


def swirl(a: np.ndarray, strength: float) -> np.ndarray:
    h, w = a.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy, r = w / 2, h / 2, min(w, h) / 2
    dx, dy = xx - cx, yy - cy
    d = np.sqrt(dx * dx + dy * dy)
    t = np.arctan2(dy, dx) + strength * np.clip(1 - d / r, 0, 1) ** 2
    sx = np.clip(cx + d * np.cos(t), 0, w - 1).round().astype(int)
    sy = np.clip(cy + d * np.sin(t), 0, h - 1).round().astype(int)
    return a[sy, sx]


def edit(img: Image.Image, op: str, box, rng: random.Random, other: Image.Image, control: bool) -> Image.Image:
    x, y, bw, bh = box
    w, h = img.size
    a = np.asarray(img).copy()
    patch = a[y:y + bh, x:x + bw]
    tx, ty = x, y
    if op == "duplicate":
        if not control:
            tx = min(max(x + int(rng.choice([-1, 1]) * rng.uniform(0.35, 0.8) * bw), 0), w - bw)
            ty = min(max(y + int(rng.uniform(-0.3, 0.3) * bh), 0), h - bh)
    elif op == "swirl":
        patch = swirl(patch, 0.05 if control else rng.choice([-1, 1]) * rng.uniform(2.5, 5.0))
    elif op == "tear":
        k, s = rng.randint(2, 4), 0 if control else rng.uniform(0.12, 0.3)
        patch = patch.copy()
        cut = np.array_split(np.arange(bh if rng.random() < 0.5 else bw), k)
        vertical = len(cut[0]) and patch.shape[0] == sum(len(c) for c in cut)
        for i, idx in enumerate(cut):
            shift = int((1 if i % 2 else -1) * s * (bw if vertical else bh))
            if vertical:
                patch[idx] = np.roll(patch[idx], shift, axis=1)
            else:
                patch[:, idx] = np.roll(patch[:, idx], shift, axis=0)
    elif op == "foreign" and not control:
        o = np.asarray(other.resize((w, h)))
        ox, oy = rng.randrange(0, w - bw + 1), rng.randrange(0, h - bh + 1)
        patch = o[oy:oy + bh, ox:ox + bw]
    if control:
        patch = np.clip(patch.astype(np.float32) * rng.uniform(0.97, 1.03), 0, 255).astype(np.uint8)
    layer = Image.fromarray(a)
    layer.paste(Image.fromarray(patch), (tx, ty))
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).ellipse((tx + 4, ty + 4, tx + bw - 4, ty + bh - 4), fill=255)  # no rectangular seam
    return Image.composite(layer, img, mask.filter(ImageFilter.GaussianBlur(5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--base", default="data/clean_art_v4")
    ap.add_argument("--out", default="data/clean_art_v5")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    B.use_templated_wordings()
    from scripts.build_artifact_mix import yes_no

    rng = random.Random(a.seed)
    pool = sources()
    rng.shuffle(pool)
    pool = pool[: a.n]
    OUT.mkdir(parents=True, exist_ok=True)
    train, test, attrib = [], [], []
    for i, (iid, path, lic) in enumerate(pool):
        img = Image.open(path).convert("RGB")
        s = 512 / max(img.size)
        img = img.resize((max(64, int(img.width * s)), max(64, int(img.height * s))), Image.Resampling.BICUBIC)
        other = Image.open(pool[(i + 1) % len(pool)][1]).convert("RGB")
        op, box = rng.choice(OPS), region(*img.size, rng)
        stem = hashlib.sha1(iid.encode()).hexdigest()[:16]
        rec = {"source_id": iid, "op": op, "box": box, "split": split_of(iid)}
        for kind in ("flawed", "control"):
            dest = OUT / f"{stem}_{kind}.jpg"
            edit(img, op, box, random.Random(f"{a.seed}:{iid}"), other, kind == "control").save(dest, quality=95)
            rec[f"{kind}_path"] = str(dest)
        (test if rec["split"] == "test" else train).append(rec)
        for kind in ("flawed", "control"):
            attrib.append({"image_id": f"anomaly:{stem}_{kind}", "path": rec[f"{kind}_path"], "source": f"procedural edit of {iid}",
                           "license": lic, "original": path})
    (OUT / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in train))
    (OUT / "test.jsonl").write_text("".join(json.dumps(r) + "\n" for r in test))
    out, base = Path(a.out), Path(a.base)
    out.mkdir(parents=True, exist_ok=True)
    for f in ("calib.jsonl",):
        shutil.copy(base / f, out / f)
    idx = json.loads((base / "images.json").read_text())
    with open(out / "decisions.jsonl", "w") as fh:
        with open(base / "decisions.jsonl") as src:
            shutil.copyfileobj(src, fh)
        for rec in train:
            stem = Path(rec["flawed_path"]).stem.rsplit("_", 1)[0]
            for kind in ("flawed", "control"):
                idx["blobs"].append(rec[f"{kind}_path"])
                idx["rows"].append([len(idx["blobs"]) - 1, 0, -1])
                d = yes_no(f"anomaly-{stem}-{kind}", "local_anomaly", B.ART_Q, kind == "flawed", f"anomaly:{stem}_{kind}", rng,
                           {"op": rec["op"], "source_id": rec["source_id"]})
                row = {k: getattr(d, k) for k in ("uid", "source", "kind", "question", "candidates", "target", "image_id", "meta")}
                fh.write(json.dumps(row | {"image_row": len(idx["rows"]) - 1}) + "\n")
    (out / "images.json").write_text(json.dumps(idx))
    with open(out / "attribution.jsonl", "w") as fh:
        with open(base / "attribution.jsonl") as src:
            shutil.copyfileobj(src, fh)
        fh.writelines(json.dumps(r) + "\n" for r in attrib if not any(r["path"] == t[f"{k}_path"] for t in test for k in ("flawed", "control")))
    m = json.loads((base / "manifest.json").read_text())
    m["decisions"] += 2 * len(train)
    m["images"] += 2 * len(train)
    m["counts"]["local_anomaly:noul"] = 2 * len(train)
    m["stats"]["local_anomalies"] = {"sources": len(pool), "train_pairs": len(train), "test_pairs": len(test),
                                     "ops": {o: sum(r["op"] == o for r in train) for o in OPS}}
    (out / "manifest.json").write_text(json.dumps(m))
    print(json.dumps(m["stats"]["local_anomalies"]))


if __name__ == "__main__":
    main()
