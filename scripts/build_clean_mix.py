"""Build the licence-clean training mix for the open Glance model (default data/clean_v1).

Images: data/clean/{coco,oi}/ from scripts/fetch_clean.py (CC BY 2.0, no known copyright
restrictions, US Government work), each with an attribution record. Questions (CC BY 4.0 or
Apache-2.0 annotations):
  vqav2_train    VQAv2 train2014 questions on those COCO images (train_mix.vqav2_decision: yes/no ->
                 noul with the vote share; other -> 4-way choice with same-type distractors)
  aokvqa_train   A-OKVQA train (the official json carries the COCO id) -> 4-way choice
  gqa_train      GQA balanced train on Visual Genome images that are COCO images we hold (VG's
                 image_data.json gives the COCO id; images whose VG copy is cropped or rotated are left
                 out): yes/no -> noul; other -> 4-way choice, distractors of the same detailed type drawn
                 by answer frequency (never a plural or synonym of the answer), and for "X or Y?" the
                 other named option among them; at most GQA_PER_IMAGE questions per image
  coco_presence  COCO object presence on those images (train_mix.presence_decisions, POPE-style, as in R1)
  oi_presence    Open Images: "Is there a X?" for one human-verified present and one verified absent
                 concrete object class, balanced per class; which-of-4 with 1 present + 3 verified absent
  quality        a synthetic distortion (blur, noise, JPEG, exposure or pixelation) at one of five
                 graded levels on a sample of the images -> "How good is the technical quality of this
                 image?" over the KonIQ benchmark's five levels, soft target around the level
Exclusions (eval_exclusions; counts per benchmark in the manifest): any photo that is also an evaluation
image, matched by Flickr photo id (COCO ids and KonIQ file names map to one) or as a dHash near-duplicate
(train_mix.HAMMING_MAX), over every benchmark in EVAL_SETS; and second copies of a photo inside the mix
(an Open Images photo that is also a COCO photo, burst shots), so no photo straddles two splits.
Splits: 5% of images (by sha1, a distorted copy follows its source image) go to calib.jsonl, the
held-out set the release calibration is fitted on, never to training. train.py holds out 2% as dev.

    uv run python scripts/build_clean_mix.py            # -> data/clean_v1/
"""
import argparse
import csv
import hashlib
import io
import json
import os
import random
import tarfile
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from glance.data.benchmarks import ALL_BENCHMARKS, SCORE_LEVELS
from glance.data.schema import Decision
from glance.data.train_mix import HAMMING_MAX, _article, near_duplicates, presence_decisions, vqav2_decision

DATA = Path(os.environ.get("GLANCE_DATA_DIR", "data"))
RAW, CLEAN = DATA / "raw" / "clean", DATA / "clean"
EVAL_SETS = ["pope_adversarial", "pope_random", "pope_popular", "vqav2_val_yesno", "aokvqa_val",
             "sugarcrepe_replace_rel", "sugarcrepe_swap_att", "mmstar", "koniq_test", "real_photos_no_errors",
             "salart_q1", "salart_q2", "salart_q3", "salart_q4", "artibench", "abench_generative",
             "had_val_hands", "had_val_any", "richhf_test_artifacts", "richhf_test_plausibility"]
GQA_PER_IMAGE = 8
OI_MIN_POSITIVES = 200  # as in fetch_clean.py; and only Open Images' 600 boxable (concrete object) classes
OI_MAX_SKEW = 1.5  # per class, keep at most 1.5x as many presence answers of one polarity as of the other
SAME = [{"gray", "grey"}, {"black", "dark"}, {"man", "guy"}, {"woman", "lady"}, {"men", "people"}, {"man", "person"},
        {"woman", "person"}, {"tv", "television"}, {"cellphone", "phone"}, {"shirt", "t-shirt"}]  # not distractors of each other
N_QUALITY = 10_000
QUALITY_Q = "How good is the technical quality of this image?"
# severity per level: level 4 = untouched, 3 = mild ... 0 = severe
DISTORT = {"blur": {3: 1.2, 2: 2.5, 1: 4.0, 0: 7.0}, "noise": {3: 8, 2: 18, 1: 32, 0: 55},
           "jpeg": {3: 35, 2: 18, 1: 9, 0: 4}, "exposure": {3: 1.3, 2: 1.6, 1: 2.0, 0: 2.8},
           "pixelate": {3: 2, 2: 3, 1: 5, 0: 8}}


def is_calib(image_id: str) -> bool:
    group = image_id.split("#")[0]
    return int(hashlib.sha1(f"calib:{group}".encode()).hexdigest()[:8], 16) % 20 == 0


def flickr_id(url: str) -> str:
    """Flickr photo id from a static image URL (.../<id>_<secret>_x.jpg) or a photo page (.../photos/<user>/<id>/)."""
    parts = [p for p in url.rstrip("/").split("/") if p]
    last = parts[-1]
    return last.split("_")[0] if "." in last else last


def eval_exclusions(images: dict) -> tuple[set, dict]:
    """Image ids to drop: the same photo as any evaluation image (a COCO id or Flickr id a benchmark uses,
    or a dHash near-duplicate), and second copies of one photo inside the mix (same Flickr id, e.g. an Open
    Images photo that is also a COCO photo; or a dHash near-duplicate, e.g. burst shots)."""
    from glance.data.train_mix import dhash

    with zipfile.ZipFile(RAW / "annotations_trainval2017.zip") as z:
        coco_flickr = {im["id"]: flickr_id(im["flickr_url"]) for s in ("train", "val")
                       for im in json.loads(z.read(f"annotations/captions_{s}2017.json"))["images"]}
    eval_flickr, eval_hashes, per_bench = set(), {}, {}
    for name in EVAL_SETS:
        bench_ids = set()
        for d in ALL_BENCHMARKS[name]():
            bench_ids.add(d.image_id)
            if d.image_id.startswith("coco:"):
                eval_flickr.add(coco_flickr[int(d.image_id.split(":")[1])])
            elif d.image_id.startswith("koniq:"):  # KonIQ-10k files are named by Flickr photo id
                eval_flickr.add(d.image_id.split(":")[1].split(".")[0])
            if d.image_id not in eval_hashes:
                eval_hashes[d.image_id] = (name, dhash(d.image_bytes))
        per_bench[name] = len(bench_ids)
    fid = {i: flickr_id(r["original"]) for i, r in images.items()}
    ids = sorted(images, key=lambda i: (not i.startswith("coco:"), i))  # keep the COCO copy of a shared photo
    by_flickr = {i for i in ids if fid[i] in eval_flickr}
    hashes = [int(images[i]["dhash"]) for i in ids]
    ref = sorted({h for _, h in eval_hashes.values()})
    near = {ids[k] for k in near_duplicates(hashes, ref)}
    dropped = Counter()
    for i in near:
        h = int(images[i]["dhash"])
        for name, eh in eval_hashes.values():
            if bin(h ^ eh).count("1") <= HAMMING_MAX:
                dropped[name] += 1
                break
    all_coco_flickr = set(coco_flickr.values())
    seen_flickr, kept_hashes, twins = set(), [], set()
    for i, h in zip(ids, hashes):
        if i in by_flickr or i in near:
            continue
        if fid[i] in seen_flickr or (i.startswith("oi:") and fid[i] in all_coco_flickr):
            twins.add(i)
            continue
        if kept_hashes and near_duplicates([h], kept_hashes):
            twins.add(i)
            continue
        seen_flickr.add(fid[i])
        kept_hashes.append(h)
    return by_flickr | near | twins, {"eval_images": per_bench, "dropped_same_flickr_photo_as_eval": len(by_flickr),
                                      "dropped_near_duplicate_of_eval": len(near), "near_duplicate_by_bench": dict(dropped),
                                      "dropped_second_copy_in_mix": len(twins)}


def vqav2(keep: set, rng: random.Random, stats: dict, weighted: bool = False) -> list[Decision]:
    with zipfile.ZipFile(RAW / "v2_Questions_Train_mscoco.zip") as z:
        text = {q["question_id"]: q["question"] for q in json.loads(z.read("v2_OpenEnded_mscoco_train2014_questions.json"))["questions"]}
    with zipfile.ZipFile(RAW / "v2_Annotations_Train_mscoco.zip") as z:
        anns = json.loads(z.read("v2_mscoco_train2014_annotations.json"))["annotations"]
    vocab = defaultdict(Counter)
    for r in anns:
        vocab[(r["answer_type"], r["question_type"])][r["multiple_choice_answer"]] += 1
    out, skipped = [], Counter()
    for r in anns:
        iid = f"coco:{r['image_id']}"
        if iid not in keep:
            continue
        d, why = vqav2_decision(r | {"question": text[r["question_id"]]}, vocab, rng, b"", iid, weighted=weighted)
        if d is None:
            skipped[why] += 1
        else:
            out.append(d)
    stats["vqav2_train"] = {"questions": len(out), "skipped": dict(skipped)}
    return out


def aokvqa(keep: set, stats: dict) -> list[Decision]:
    with tarfile.open(RAW / "aokvqa_v1p0.tar.gz") as t:
        rows = json.load(t.extractfile("aokvqa_v1p0_train.json"))
    out = []
    for r in rows:
        iid = f"coco:{r['image_id']}"
        if iid in keep:
            t_ = [0.0] * len(r["choices"])
            t_[r["correct_choice_idx"]] = 1.0
            out.append(Decision(f"aokvqa-train-{r['question_id']}", "aokvqa_train", "choice", r["question"],
                                list(r["choices"]), t_, b"", iid))
    stats["aokvqa_train"] = {"questions": len(out), "of": len(rows)}
    return out


def _distractors(pool: Counter, gold: str, k: int, rng: random.Random, exclude=()) -> list[str] | None:
    """k answers drawn in proportion to how often they answer this question type (so the candidate set
    carries no frequency prior), never the gold answer, a plural of it, or a listed synonym."""
    def clash(a):
        a = a.strip().lower()
        return a == gold or a in exclude or a.rstrip("s") == gold.rstrip("s") or any({a, gold} <= s for s in SAME)
    items = [(a, c) for a, c in pool.most_common(200) if not clash(a)]
    out = []
    while len(out) < k and items:
        a = rng.choices([a for a, _ in items], weights=[c for _, c in items])[0]
        out.append(a.strip().lower())
        items = [(x, c) for x, c in items if x != a]
    return out if len(out) == k else None


def _named_alternative(question: str, answer: str, known: set | None = None) -> str | None:
    """The other option of an "X or Y?" question, when the answer is one of the two. With `known`
    (every GQA answer string), an alternative that is not itself a known answer is rejected: phrase
    options ("to the right of the table the pizza is on") do not parse cleanly (re-audit M2)."""
    q = question.rstrip("?").lower()
    if " or " not in q:
        return None
    left, right = q.rsplit(" or ", 1)
    right = right.strip()
    for article in ("the ", "a ", "an "):
        right = right.removeprefix(article)
    n = len(answer.split())
    alt = None
    if left.endswith(" " + answer):
        alt = right
    elif right == answer or right.startswith(answer + " "):
        alt = " ".join(left.split()[-n:])
    for article in ("the ", "a ", "an "):
        alt = alt.removeprefix(article) if alt else alt
    if not alt or (known is not None and alt not in known):
        return None
    return alt


def gqa(keep: set, rng: random.Random, stats: dict, k: int = 4, strict_alt: bool = False) -> list[Decision]:
    with zipfile.ZipFile(RAW / "image_data.json.zip") as z:
        vg = {str(r["image_id"]): r for r in json.loads(z.read("image_data.json")) if r["coco_id"]}
    with zipfile.ZipFile(RAW / "annotations_trainval2017.zip") as z:
        coco_size = {im["id"]: (im["width"], im["height"]) for s in ("train", "val")
                     for im in json.loads(z.read(f"annotations/captions_{s}2017.json"))["images"]}
    with zipfile.ZipFile(RAW / "questions1.2.zip") as z:
        qs = json.loads(z.read("train_balanced_questions.json"))
    vocab, by_image, reshaped = defaultdict(Counter), defaultdict(list), set()
    for qid, q in qs.items():
        vocab[q["types"]["detailed"]][q["answer"]] += 1
        r = vg.get(q["imageId"])
        if not r or f"coco:{r['coco_id']}" not in keep:
            continue
        w, h = coco_size[r["coco_id"]]
        if abs(r["width"] / r["height"] - w / h) > 0.02 * (w / h):  # VG's copy is cropped or rotated
            reshaped.add(r["coco_id"])
            continue
        by_image[f"coco:{r['coco_id']}"].append((qid, q))
    n_all = len(qs)
    del qs
    out, skipped = [], Counter()
    known = {a.strip().lower() for c in vocab.values() for a in c} if strict_alt else None
    for iid in sorted(by_image):
        items = sorted(by_image[iid])
        for qid, q in rng.sample(items, min(GQA_PER_IMAGE, len(items))):
            ans = q["answer"].strip().lower()
            meta = {"type": q["types"]["detailed"]}
            if ans in ("yes", "no"):
                out.append(Decision(f"gqa-{qid}", "gqa_train", "noul", q["question"], ["yes", "no"],
                                    [1.0, 0.0] if ans == "yes" else [0.0, 1.0], b"", iid, meta))
                continue
            alt = _named_alternative(q["question"], ans, known) if q["types"]["structural"] == "choose" else None
            rest = _distractors(vocab[q["types"]["detailed"]], ans, k - 1 - bool(alt), rng, (alt,) if alt else ())
            if rest is None:
                skipped["no_distractors"] += 1
                continue
            cands = [ans] + ([alt] if alt else []) + rest
            rng.shuffle(cands)
            out.append(Decision(f"gqa-{qid}", "gqa_train", "choice", q["question"], cands,
                                [1.0 if c == ans else 0.0 for c in cands], b"", iid, meta))
    stats["gqa_train"] = {"questions": len(out), "images": len(by_image), "balanced_train_questions": n_all,
                          "dropped_reshaped_vg_images": len(reshaped), "skipped": dict(skipped)}
    return out


def coco_presence(keep: set, rng: random.Random, stats: dict) -> list[Decision]:
    present, cats = defaultdict(set), None
    with zipfile.ZipFile(RAW / "annotations_trainval2017.zip") as z:
        for split in ("train", "val"):
            inst = json.loads(z.read(f"annotations/instances_{split}2017.json"))
            cats = cats or sorted(inst["categories"], key=lambda c: c["id"])
            index = {c["id"]: i for i, c in enumerate(cats)}
            for a in inst["annotations"]:
                present[a["image_id"]].add(index[a["category_id"]])
            del inst
    names = [c["name"] for c in cats]
    freq, cooc = Counter(), defaultdict(Counter)
    for objs in present.values():
        freq.update(objs)
        for a in objs:
            for b in objs:
                if a != b:
                    cooc[a][b] += 1
    out = []
    for iid in sorted(keep):
        if iid.startswith("coco:"):
            cid = int(iid.split(":")[1])
            out += presence_decisions(cid, present.get(cid, set()), names, freq, cooc, rng, b"", iid)
    stats["coco_presence"] = {"questions": len(out)}
    return out


def oi_presence(keep: set, rng: random.Random, stats: dict) -> list[Decision]:
    """Open Images object presence from human-verified labels, over concrete object classes (the 600
    boxable ones) with at least OI_MIN_POSITIVES verified positives. The "no" object is the image's
    verified-absent class that has so far been asked with "yes" most often more than with "no", and each
    class keeps at most OI_MAX_SKEW times as many answers of one polarity as of the other, so the question
    text alone does not give the answer away (clean_v1 audit S4). Which-of-4 distractors are
    verified-absent classes drawn in proportion to how often each class is present elsewhere."""
    from scripts.fetch_clean import oi_labels

    names = {r["LabelName"]: r["DisplayName"].strip().lower() for r in csv.DictReader(open(RAW / "oidv7-class-descriptions.csv"))}
    boxable = {r["LabelName"] for r in csv.DictReader(open(RAW / "oidv7-class-descriptions-boxable.csv"))}
    pos, neg, count = oi_labels()
    common = {c for c, k in count.items() if k >= OI_MIN_POSITIVES and c in boxable}
    order = sorted(i for i in keep if i.startswith("oi:"))
    rng.shuffle(order)
    yes_n, no_n, pairs = Counter(), Counter(), []
    for iid in order:
        oid = iid.split(":")[1]
        p, n = sorted(pos[oid] & common), sorted(neg[oid] & common)
        if not p or not n:
            continue
        y = rng.choice(p)
        best = max(yes_n[c] - no_n[c] for c in n)
        no = rng.choice([c for c in n if yes_n[c] - no_n[c] == best])
        yes_n[y] += 1
        no_n[no] += 1
        pairs.append((iid, oid, y, no, n))
    cap = {c: int(OI_MAX_SKEW * max(1, min(yes_n[c], no_n[c]))) for c in common}
    used_yes, used_no, out = Counter(), Counter(), []
    for iid, oid, y, no, n in pairs:
        for c, is_yes, used in ((y, True, used_yes), (no, False, used_no)):
            if used[c] >= cap[c]:
                continue
            used[c] += 1
            obj = names[c]
            out.append(Decision(f"oipres-{oid}-{obj}", "oi_presence", "noul", f"Is there {_article(obj)} {obj} in the image?",
                                ["yes", "no"], [1.0, 0.0] if is_yes else [0.0, 1.0], b"", iid))
        if len(n) >= 3:
            picks, pool = [], list(n)
            while len(picks) < 3:
                c = rng.choices(pool, weights=[count[x] for x in pool])[0]
                picks.append(c)
                pool.remove(c)
            cands = [names[y]] + [names[c] for c in picks]
            rng.shuffle(cands)
            out.append(Decision(f"oiwhich-{oid}", "oi_presence", "choice", "Which of these objects is in the image?",
                                cands, [1.0 if c == names[y] else 0.0 for c in cands], b"", iid))
    stats["oi_presence"] = {"questions": len(out), "classes": len(common),
                            "yes_share": round(sum(used_yes.values()) / max(1, sum(used_yes.values()) + sum(used_no.values())), 3)}
    return out


def distort(img: Image.Image, kind: str, level: int, rng: random.Random) -> tuple[bytes, int]:
    """(JPEG bytes, quality used). Level 4 leaves the image untouched."""
    q = 95
    if level < 4:
        s = DISTORT[kind][level]
        if kind == "blur":
            img = img.filter(ImageFilter.GaussianBlur(s))
        elif kind == "noise":
            a = np.asarray(img, dtype=np.float32) + np.random.default_rng(rng.randrange(2**32)).normal(0, s, (img.height, img.width, 3))
            img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
        elif kind == "jpeg":
            q = s
        elif kind == "exposure":
            img = ImageEnhance.Brightness(img).enhance(s if rng.random() < 0.5 else 1 / s)
        elif kind == "pixelate":
            w, h = img.size
            img = img.resize((max(1, w // s), max(1, h // s)), Image.Resampling.BILINEAR).resize((w, h), Image.Resampling.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=q)
    return buf.getvalue(), q


def quality_target(level: int) -> list[float]:
    if level == 4:
        return [0.0, 0.0, 0.0, 0.3, 0.7]  # an untouched photo is good or excellent, not always excellent
    t = [0.0] * 5
    t[level] = 0.7
    for j in (level - 1, level + 1):
        if 0 <= j < 5:
            t[j] = 0.15
    return [x / sum(t) for x in t]


def quality(keep: list[str], images: dict, rng: random.Random, stats: dict, out_dir: Path) -> tuple[list[Decision], dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out, paths = [], {}
    for src in rng.sample(keep, min(N_QUALITY, len(keep))):
        level, kind = rng.randrange(5), rng.choice(sorted(DISTORT))
        iid = f"{src}#quality"
        if level == 4:
            paths[iid] = images[src]["path"]
            kind = "none"
        else:
            path = out_dir / f"{src.replace(':', '_')}_{kind}{level}.jpg"
            if not path.exists():
                data, _ = distort(Image.open(images[src]["path"]).convert("RGB"), kind, level, rng)
                path.write_bytes(data)
            paths[iid] = str(path)
        out.append(Decision(f"quality-{src}-{kind}{level}", "quality_synthetic", "score", QUALITY_Q, list(SCORE_LEVELS),
                            quality_target(level), b"", iid, {"level": level, "distortion": kind}))
    stats["quality_synthetic"] = {"questions": len(out), "levels": dict(Counter(d.meta["level"] for d in out))}
    return out, paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DATA / "clean_v1"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shortcut-fixes", action="store_true",
                    help="re-audit fixes (clean_v2): VQAv2 distractors by answer frequency, GQA alternatives must be known answers")
    a = ap.parse_args()
    out, rng = Path(a.out), random.Random(a.seed)
    out.mkdir(parents=True, exist_ok=True)
    images = {}
    for f in ("attribution_coco.jsonl", "attribution_oi.jsonl"):
        for line in open(CLEAN / f):
            r = json.loads(line)
            images[r["image_id"]] = r
    drop, stats = eval_exclusions(images)
    keep = set(images) - drop
    stats["images_fetched"] = dict(Counter(i.split(":")[0] for i in images))
    stats["images_kept"] = dict(Counter(i.split(":")[0] for i in keep))
    print(json.dumps(stats), flush=True)

    decisions = vqav2(keep, rng, stats, a.shortcut_fixes) + aokvqa(keep, stats) + gqa(keep, rng, stats, strict_alt=a.shortcut_fixes)
    decisions += coco_presence(keep, rng, stats)
    decisions += oi_presence(keep, rng, stats)
    q, qpaths = quality(sorted(keep), images, rng, stats, CLEAN / "quality")
    decisions += q
    print(json.dumps(stats), flush=True)

    # one image row per image id; distorted copies are separate rows that point at their own file
    blobs, rows, row_of = [], [], {}
    for d in decisions:
        if d.image_id not in row_of:
            blobs.append(qpaths.get(d.image_id) or images[d.image_id]["path"])
            rows.append([len(blobs) - 1, 0, -1])
            row_of[d.image_id] = len(rows) - 1
    rng.shuffle(decisions)
    n_split = Counter()
    with open(out / "decisions.jsonl", "w") as train_fh, open(out / "calib.jsonl", "w") as calib_fh:
        for d in decisions:
            rec = asdict(d)
            rec.pop("image_bytes")
            rec["image_row"] = row_of[d.image_id]
            split = "calib" if is_calib(d.image_id) else "train"
            n_split[split] += 1
            (calib_fh if split == "calib" else train_fh).write(json.dumps(rec) + "\n")
    (out / "images.json").write_text(json.dumps({"blobs": blobs, "rows": rows}))
    with open(out / "attribution.jsonl", "w") as fh:
        for iid in row_of:
            src = iid.split("#")[0]
            rec = {k: v for k, v in images[src].items() if k != "dhash"} | {"image_id": iid, "path": blobs[rows[row_of[iid]][0]]}
            if iid != src:
                rec["derived"] = "synthetic quality distortion of " + src
            fh.write(json.dumps(rec) + "\n")
    manifest = {"decisions": n_split["train"], "calib_decisions": n_split["calib"], "images": len(rows),
                "counts": dict(Counter(f"{d.source}:{d.kind}" for d in decisions)), "stats": stats, "seed": a.seed}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
