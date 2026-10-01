"""R1 training mix (~100k gold decisions) with contamination exclusions.

Research use only: COCO images carry mixed Flickr licences (most are NonCommercial) and KonIQ-10k
labels have no licence, so weights trained on this mix cannot be released. The licence-clean mix is
built by scripts/build_clean_mix.py.

Sources: A-OKVQA train (choice, K=4); VQAv2 train subset (yes/no -> noul with the 10-answer vote
share as soft target; number/other -> choice with same-type distractors and vote-share targets);
COCO-2017 detection train (object presence yes/no and which-object choice, POPE-style popular /
adversarial negatives); KonIQ-10k train (score, rater histogram).

Exclusions, recorded in the manifest:
- POPE's 500 COCO val2014 images: by COCO id for VQAv2 (train2014, disjoint anyway) and COCO
  detection (train2017 re-uses val2014 images); by perceptual hash for A-OKVQA train, whose HF
  copy carries no COCO id.
- COCO val2017 (A-OKVQA val, SugarCrepe) never enters: every source here is train2014/train2017.
- MMStar images are also hashed against the whole mix and reported (not silently dropped).
"""
import glob
import io
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from PIL import Image

from glance.data import benchmarks
from glance.data.schema import Decision, bytes_id

HAMMING_MAX = 4


def dhash(image_bytes: bytes, size: int = 8) -> int:
    img = Image.open(io.BytesIO(image_bytes)).convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    a = np.asarray(img, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def near_duplicates(hashes: list[int], reference: list[int], max_dist: int = HAMMING_MAX) -> set[int]:
    """Indices i with hamming(hashes[i], some reference) <= max_dist."""
    if not reference or not hashes:
        return set()
    ref = np.array(reference, dtype=np.uint64)
    out = set()
    for i, h in enumerate(hashes):
        x = np.bitwise_xor(ref, np.uint64(h))
        dist = np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(1)
        if dist.min() <= max_dist:
            out.add(i)
    return out


def _local(repo: str, pattern: str) -> list[str]:
    """Files already in the HF cache matching `pattern`. We deliberately fetch only a subset of
    shards, so snapshot_download(local_files_only=True) would reject the snapshot as incomplete."""
    from huggingface_hub.constants import HF_HUB_CACHE

    root = os.path.join(HF_HUB_CACHE, "datasets--" + repo.replace("/", "--"), "snapshots", "*")
    files = sorted(glob.glob(os.path.join(root, pattern)))
    assert files, (repo, pattern)
    return files


def _article(noun: str) -> str:
    return "an" if noun[0] in "aeiou" else "a"


def aokvqa_train(pope_hashes: list[int], manifest: dict) -> list[Decision]:
    rows = [r for f in _local("HuggingFaceM4/A-OKVQA", "data/train-*.parquet") for r in pq.read_table(f).to_pylist()]
    hashes = [dhash(r["image"]["bytes"]) for r in rows]
    drop = near_duplicates(hashes, pope_hashes)
    manifest["aokvqa_train"] = {"rows": len(rows), "dropped_pope_phash": len(drop)}
    out = []
    for i, r in enumerate(rows):
        if i in drop:
            continue
        t = [0.0] * len(r["choices"])
        t[r["correct_choice_idx"]] = 1.0
        img = r["image"]["bytes"]
        out.append(Decision(f"aokvqa-train-{r['question_id']}", "aokvqa_train", "choice", r["question"],
                            list(r["choices"]), t, img, bytes_id(img), {"dhash": hashes[i]}))
    return out


def vqav2_decision(r: dict, vocab: dict, rng: random.Random, img: bytes, iid: str, k: int = 4,
                   source: str = "vqav2_train", weighted: bool = False) -> tuple[Decision | None, str]:
    """One VQAv2 question -> a decision, or (None, reason). `r` has question_id, question, answer_type,
    question_type, multiple_choice_answer and the 10 `answers` ({"answer": ...}); `vocab` maps
    (answer_type, question_type) to answer counts, the distractor pool. With `weighted`, distractors are
    drawn in proportion to how often they answer this question type, so frequent answers are as likely
    to be wrong options as right ones (uniform draws let the option text alone pick the answer 0.67 of
    the time on the clean mix; R1 keeps uniform draws)."""
    votes = Counter(a["answer"].strip().lower() for a in r["answers"])
    if r["answer_type"] == "yes/no":
        y, n = votes.get("yes", 0), votes.get("no", 0)
        if y + n < 5:
            return None, "yesno_unclear"
        return Decision(f"vqav2-{r['question_id']}", source, "noul", r["question"], ["yes", "no"],
                        [y / (y + n), n / (y + n)], img, iid), ""
    gold = r["multiple_choice_answer"].strip().lower()
    if votes.get(gold, 0) < 3:
        return None, "low_agreement"
    # Keep a second answer when annotators split (>= 2 votes); distractors never got a vote.
    voted = [gold] + [a for a, c in votes.most_common() if a != gold and c >= 2][:1]
    counts = vocab[(r["answer_type"], r["question_type"])]
    pool = [a for a, _ in counts.most_common(200) if a not in votes]
    if len(pool) < k - len(voted):
        return None, "no_distractors"
    if weighted:
        picks = []
        while len(picks) < k - len(voted):
            a = rng.choices(pool, weights=[counts[x] for x in pool])[0]
            picks.append(a)
            pool.remove(a)
    else:
        picks = rng.sample(pool, k - len(voted))
    cands = voted + picks
    rng.shuffle(cands)
    total = sum(votes[a] for a in voted)
    t = [votes.get(c, 0) / total if c in voted else 0.0 for c in cands]
    return Decision(f"vqav2-{r['question_id']}", source, "choice", r["question"], cands, t, img, iid,
                    {"answer_type": r["answer_type"]}), ""


def vqav2_train(pope_ids: set[int], manifest: dict, rng: random.Random, k: int = 4) -> list[Decision]:
    files = _local("pingzhili/vqa_v2", "data/train-*.parquet")
    text_cols = ["question_id", "question", "question_type", "answer_type", "multiple_choice_answer", "answers", "image_id"]
    vocab = defaultdict(Counter)  # (answer_type, question_type) -> answer counts, for distractors
    n_rows = 0
    for f in files:
        for r in pq.read_table(f, columns=text_cols).to_pylist():
            vocab[(r["answer_type"], r["question_type"])][r["multiple_choice_answer"]] += 1
            n_rows += 1
    images: dict[int, bytes] = {}  # one copy per image; each parquet row repeats the image
    rows = []
    for f in files:
        for r in pq.read_table(f).to_pylist():
            r["image"] = images.setdefault(r["image_id"], r["image"]["bytes"])
            rows.append(r)
    out, skipped, dropped = [], Counter(), 0
    for r in rows:
        if r["image_id"] in pope_ids:
            dropped += 1
            continue
        d, why = vqav2_decision(r, vocab, rng, r["image"], f"coco:{r['image_id']}", k)
        if d is None:
            skipped[why] += 1
        else:
            out.append(d)
    manifest["vqav2_train"] = {"rows": n_rows, "shards": len(files), "images": len(images),
                               "dropped_pope_id": dropped, "skipped": dict(skipped)}
    return out


def presence_decisions(key, present: set[int], names: list[str], freq: Counter, cooc: dict, rng: random.Random,
                       img: bytes, iid: str, source: str = "coco_presence") -> list[Decision]:
    """POPE-style object presence for one image: "Is there a X?" for one present and one absent object
    (yes / no), and "Which of these objects is in the image?" with 1 present + 3 absent. Absent objects
    are 50% adversarial (co-occur with the present ones), 30% popular (top 20), 20% random."""
    present = sorted(present)
    if not present:
        return []
    popular = [c for c, _ in freq.most_common(20)]
    absent = [c for c in range(len(names)) if c not in present]
    adv = [c for c, _ in sum((cooc[p] for p in present), Counter()).most_common() if c not in present]

    def negative():
        roll = rng.random()
        src = adv if roll < 0.5 and adv else [c for c in popular if c not in present] if roll < 0.8 else absent
        return rng.choice(src or absent)

    pos = names[rng.choice(present)]
    neg = names[negative()]
    out = [Decision(f"cocopres-{key}-{obj}", source, "noul", f"Is there {_article(obj)} {obj} in the image?",
                    ["yes", "no"], [1.0, 0.0] if yes else [0.0, 1.0], img, iid) for obj, yes in ((pos, True), (neg, False))]
    distract = []
    while len(distract) < 3:
        c = names[negative()]
        if c not in distract:
            distract.append(c)
    cands = [pos] + distract
    rng.shuffle(cands)
    out.append(Decision(f"cocowhich-{key}", source, "choice", "Which of these objects is in the image?", cands,
                        [1.0 if c == pos else 0.0 for c in cands], img, iid))
    return out


def coco_presence(pope_ids: set[int], manifest: dict, rng: random.Random) -> list[Decision]:
    files = _local("detection-datasets/coco", "data/train-*.parquet")
    # The parquet files carry no feature metadata; the ClassLabel names are pinned in the repo.
    names = json.loads((Path(__file__).parent / "coco_names.json").read_text())["names"]
    images = []
    for f in files:
        images += pq.read_table(f).to_pylist()
    freq, cooc = Counter(), defaultdict(Counter)
    for r in images:
        present = set(r["objects"]["category"])
        freq.update(present)
        for a in present:
            for b in present:
                if a != b:
                    cooc[a][b] += 1
    out, dropped = [], 0
    for r in images:
        if r["image_id"] in pope_ids:
            dropped += 1
            continue
        out += presence_decisions(r["image_id"], set(r["objects"]["category"]), names, freq, cooc, rng,
                                  r["image"]["bytes"], f"coco:{r['image_id']}")
    manifest["coco_presence"] = {"images": len(images), "dropped_pope_id": dropped}
    return out


def koniq_split(split: str) -> list[Decision]:
    import csv

    root = snapshot_download("chaofengc/IQA-PyTorch-Datasets-metainfo", repo_type="dataset",
                             allow_patterns=["meta_info_KonIQ10kDataset.csv"], local_files_only=True)
    out = []
    with open(os.path.join(root, "meta_info_KonIQ10kDataset.csv")) as fh:
        for r in csv.DictReader(fh):
            if r["official_split"] != split:
                continue
            hist = [float(r[f"c{i}"]) for i in range(1, 6)]
            img = (benchmarks.KONIQ_DIR / r["img_name"]).read_bytes()
            out.append(Decision(f"koniq-{r['img_name']}", f"koniq_{split}", "score",
                                "How good is the technical quality of this image?", list(benchmarks.SCORE_LEVELS),
                                [h / sum(hist) for h in hist], img, f"koniq:{r['img_name']}",
                                {"mos": float(r["mos"])}))
    return out


def build_r1_mix(seed: int = 0) -> tuple[list[Decision], dict]:
    rng = random.Random(seed)
    pope = benchmarks.pope("adversarial")
    pope_ids = {int(d.image_id.split(":")[1]) for d in pope}
    pope_hashes = list({d.image_id: dhash(d.image_bytes) for d in pope}.values())
    manifest = {"seed": seed, "pope_images_excluded": len(pope_ids)}
    mix = aokvqa_train(pope_hashes, manifest) + vqav2_train(pope_ids, manifest, rng) + coco_presence(pope_ids, manifest, rng)
    mix += koniq_split("train")
    manifest["counts"] = dict(Counter(d.source for d in mix))
    manifest["kinds"] = dict(Counter(d.kind for d in mix))
    manifest["images"] = len({d.image_id for d in mix})
    # Report (do not drop) MMStar near-duplicates in the mix; MMStar must stay unseen.
    ms = benchmarks.mmstar()
    mix_hashes = [d.meta.get("dhash") or dhash(d.image_bytes) for d in {d.image_id: d for d in mix}.values()]
    ms_dups = near_duplicates([dhash(d.image_bytes) for d in ms], mix_hashes)
    manifest["mmstar_near_duplicates_in_mix"] = len(ms_dups)
    return mix, manifest
