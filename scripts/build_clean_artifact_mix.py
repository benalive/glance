"""Licence-clean mix for spotting visible errors in AI-generated images (default data/clean_art_v1).

Replaces HAD and RichHF-18K (no licence) with human-labelled images from open generators
(scripts/fetch_clean_artifacts.py):
  evalmuse_train     EvalMuse-40K structural-problem marks (three annotators; hands, face, limbs,
                     animals, objects). A category counts as present when it was marked at least twice
                     and as absent when nothing in that group was marked; single marks are unclear and
                     asked about in no question. The "non-realistic scene" tag is not a flaw and is
                     ignored. Per image: a hands question, an any-anatomy-error question, a face or
                     limbs question, and a visible-generation-errors question (animals and objects count).
  imagereward_train  ImageRewardDB fidelity rating 1-7: <= 3 -> errors, >= 6 -> none (4-5 unclear), and
                     the 5-level plausibility score (1-2 severe, 3 clear, 4 some, 5 minor, 6-7 none).
Questions use the wording families of scripts/build_artifact_mix.py (several wordings per polarity;
the HELD_OUT wordings stay out of training). Every image of the licence-clean general mix
(data/clean_v1) is replayed with its questions plus one "no generation errors" question on the real
photo (REAL_NEGATIVES), as in the research mixes. Images that are dHash near-duplicates of an
evaluation image (the AI-image benchmarks and those checked for clean_v1) are dropped. 5% of the
new images go to calib.jsonl with clean_v1's calibration questions (build_clean_mix.is_calib).

    uv run python scripts/build_clean_artifact_mix.py     # -> data/clean_art_v1/
"""
import argparse
import json
import os
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from glance.data.benchmarks import ALL_BENCHMARKS, PLAUSIBILITY_LEVELS
from glance.data.schema import Decision
from glance.data.train_mix import dhash, near_duplicates
from scripts.build_artifact_mix import ANY as ANY_BASE
from scripts.build_artifact_mix import ART_Q as ART_Q_BASE
from scripts.build_artifact_mix import FACE, HELD_OUT, LIMB, PLAUS, yes_no
from scripts.build_artifact_mix import HAND as HAND_BASE
from scripts.build_clean_mix import EVAL_SETS, is_calib, quality_target

DATA = Path(os.environ.get("GLANCE_DATA_DIR", "data"))
CLEAN = DATA / "clean"
HAND_CATS = {"真实场景-人体-手部-手指变形", "真实场景-人体-手部-多指/少指", "真实场景-人体-手部-手掌不成型",
             "真实场景-人体-手部-多只手重叠混乱"}  # deformed fingers, extra/missing fingers, malformed palm, tangled hands
FACE_CATS = {"真实场景-人体-面部-五官扭曲/比例夸张", "真实场景-人体-面部-面部差缺/多余"}  # distorted features, missing/extra parts
LIMB_CATS = {"真实场景-人体-肢体-肢体扭曲/变形", "真实场景-人体-肢体-多/少肢体", "真实场景-人体-肢体-肢体比例不协调"}
OTHER_CATS = {"真实场景-动物", "真实场景-物体"}  # malformed animals, malformed objects
NOT_A_FLAW = {"非真实场景"}  # "non-realistic scene"
AI_EVAL_SETS = ["had_val_hands", "had_val_any", "richhf_test_artifacts", "salart_q1", "artibench", "abench_generative"]
# More generic paraphrases (v2): the first clean model tied the concept to the research wordings and
# failed on unseen phrasings (SalArt's question; the held-out "anything anatomically off"). No "anything
# ... off" phrasing is added, so that held-out wording still tests generalisation.
EXTRA_ART = [("Does this image contain any artifacts?", 1), ("Is there at least one visible flaw in this image?", 1),
             ("Are there any noticeable glitches or anomalies in this picture?", 1),
             ("Does something in this picture look wrong or broken?", 1),
             ("Is anything in this image distorted, warped or melted?", 1),
             ("Can you spot any rendering mistakes in this image?", 1), ("Does this image show defects from image generation?", 1),
             ("Is this image free of artifacts?", -1), ("Does everything in this picture look normal and intact?", -1),
             ("Is this picture without any visible defects?", -1), ("Is this image flawless?", -1),
             ("Does this image look clean, with nothing warped or broken?", -1)]
EXTRA_ANY = [("Is there anything wrong with the people in this image?", 1),
             ("Are any body parts in this image misshapen or duplicated?", 1),
             ("Does any figure in this image have extra or missing body parts?", 1),
             ("Does anyone in this picture have an impossible body?", 1),
             ("Do all the people in this picture look physically normal?", -1), ("Is everyone's anatomy in this image plausible?", -1),
             ("Are the bodies in this picture free of errors?", -1)]
EXTRA_HAND = [("Are there any weird-looking hands in this image?", 1), ("Are any hands in this image misshapen?", 1),
              ("Does a hand in this picture have the wrong shape?", 1),
              ("Do the hands in this picture look right?", -1), ("Are all the hands in this image normal?", -1)]
HAND, ANY, ART_Q = HAND_BASE + EXTRA_HAND, ANY_BASE + EXTRA_ANY, ART_Q_BASE + EXTRA_ART
FORBIDDEN = set(HELD_OUT) | {"Does this image contain at least one salient artifact?"}  # evaluation wordings
assert not {q.lower() for fam in (HAND, ANY, ART_Q, FACE, LIMB) for q, _ in fam} & {q.lower() for q in FORBIDDEN}
assert not any("salient" in q.lower() for fam in (HAND, ANY, ART_Q) for q, _ in fam)


def _templated(pos_frames, neg_frames, nouns, quantifiers, adjectives=(), adj_frames=()) -> list[tuple[str, int]]:
    """Every combination of frame x quantifier x noun (and frame x adjective), both polarities (v3).
    With ~20 wordings per family, whether a wording flips the answer was learned per wording and varied by
    seed (scripts/wording_probe.py); hundreds of combinations make the model read the question instead."""
    out = [(f.format(q=q, n=n).replace("  ", " "), 1) for f in pos_frames for q in quantifiers for n in nouns]
    out += [(f.format(q=q, n=n).replace("  ", " "), -1) for f in neg_frames for q in quantifiers for n in nouns]
    out += [(f.format(a=a), -1) for f in adj_frames for a in adjectives]
    return out


TEMPLATED_ART = _templated(
    ["Does this image contain {q} {n}?", "Does this picture have {q} {n}?", "Are there {q} {n} in this image?",
     "Can you see {q} {n} in this picture?", "Do you notice {q} {n} in this image?", "Is this image affected by {q} {n}?",
     "Does the image show {q} {n}?", "Would you say this image has {q} {n}?"],
    ["Is this picture free of {q} {n}?", "Would you say this picture has no {q} {n}?", "Is this image without {q} {n}?",
     "Does this image look free of {q} {n}?", "Is the image clean of {q} {n}?", "Does this image have no {q} {n}?"],
    ["errors", "artifacts", "flaws", "defects", "glitches", "anomalies", "mistakes", "distortions", "rendering errors",
     "generation errors", "visual errors", "imperfections"],
    ["any", "visible", "noticeable", "obvious", ""],
    ["error-free", "flawless", "artifact-free", "defect-free", "perfect", "clean and intact", "glitch-free"],
    ["Does this image look {a}?", "Is this picture {a}?", "Would you call this image {a}?"])
TEMPLATED_ANY = _templated(
    ["Do the people in this image have {q} {n}?", "Are there {q} {n} in this image?", "Does anyone in this image show {q} {n}?",
     "Can you see {q} {n} on the people here?", "Does this picture contain {q} {n}?"],
    ["Are the people in this image free of {q} {n}?", "Is everyone in this image free of {q} {n}?",
     "Would you say the people here have no {q} {n}?", "Is this picture without {q} {n}?"],
    ["anatomical errors", "deformed body parts", "malformed limbs", "anatomy mistakes", "distorted bodies",
     "body deformities", "extra or missing limbs", "misshapen body parts"],
    ["any", "visible", "obvious", ""],
    ["normal", "physically plausible", "well-formed", "natural"],
    ["Do the people in this picture look {a}?", "Are all the bodies in this image {a}?"])
TEMPLATED_HAND = _templated(
    ["Are there {q} {n} in this image?", "Does anyone in this image have {q} {n}?", "Do you see {q} {n} here?",
     "Does this picture show {q} {n}?"],
    ["Is this image free of {q} {n}?", "Would you say there are no {q} {n} here?", "Is this picture without {q} {n}?"],
    ["deformed hands", "malformed fingers", "extra fingers", "missing fingers", "mangled hands", "wrong finger counts",
     "twisted fingers", "fused fingers"],
    ["any", "visible", "obvious", ""],
    ["normal", "well-formed", "natural", "properly shaped"],
    ["Are all the hands in this picture {a}?", "Do the fingers here look {a}?"])


def _without(fam, banned):
    lower = {b.lower() for b in banned}
    return [(q, p) for q, p in fam if q.lower() not in lower and "salient" not in q.lower()]


def use_templated_wordings():
    """Switch the error families to the templated sets (plus the hand-written ones), minus evaluation and
    probe wordings."""
    global HAND, ANY, ART_Q, REAL_NEGATIVES
    from scripts.wording_probe import PROBE

    banned = FORBIDDEN | {q for q, _ in PROBE}
    HAND = _without(dict.fromkeys(HAND + TEMPLATED_HAND), banned)
    ANY = _without(dict.fromkeys(ANY + TEMPLATED_ANY), banned)
    ART_Q = _without(dict.fromkeys(ART_Q + TEMPLATED_ART), banned)
    REAL_NEGATIVES = [HAND, ANY, ART_Q]


REAL_NEGATIVES = [HAND, ANY, ART_Q]


def group_state(counts: Counter, cats: set) -> bool | None:
    """True if a category of the group was marked at least twice, False if none was marked, else None."""
    if max((counts[c] for c in cats), default=0) >= 2:
        return True
    return False if sum(counts[c] for c in cats) == 0 else None


def evalmuse(rows: list[dict], rng: random.Random) -> list[Decision]:
    out = []
    for r in rows:
        counts = Counter(r["fidelity_label"])
        unknown = set(counts) - HAND_CATS - FACE_CATS - LIMB_CATS - OTHER_CATS - NOT_A_FLAW
        assert not unknown, unknown
        uid, iid = f"evalmuse-{r['image_id'].split(':')[1]}", r["image_id"]
        hand, face, limb = group_state(counts, HAND_CATS), group_state(counts, FACE_CATS), group_state(counts, LIMB_CATS)
        body = group_state(counts, HAND_CATS | FACE_CATS | LIMB_CATS)
        anything = group_state(counts, HAND_CATS | FACE_CATS | LIMB_CATS | OTHER_CATS)
        for suffix, fam, state in (("-hand", HAND, hand), ("-any", ANY, body), ("-art", ART_Q, anything)):
            if state is not None:
                out.append(yes_no(uid + suffix, "evalmuse_train", fam, state, iid, rng, {"generator": r["generator"]}))
        part, state = rng.choice([(FACE, face), (LIMB, limb)])
        if state is not None:
            out.append(yes_no(uid + "-part", "evalmuse_train", part, state, iid, rng, {"generator": r["generator"]}))
    return out


def face_negatives(rng: random.Random, eval_hashes: set, stats: dict) -> dict:
    """Real close-up portraits (Open Images face crops): one 'no errors' question from a random real-photo
    family and one from the face family each. Crops near-duplicate to an evaluation image or to a FairFace
    image (kept for the fairness check) are dropped."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    ff = hf_hub_download("HuggingFaceM4/FairFace", "1.25/validation-00000-of-00001-09e3e67bb00ab4ec.parquet", repo_type="dataset")
    fair = sorted({dhash(r["image"]["bytes"]) for r in pq.read_table(ff, columns=["image"]).to_pylist()})
    crops = [json.loads(line) for line in open(CLEAN / "attribution_faces.jsonl")]
    hashes = [int(r["dhash"]) for r in crops]
    drop = near_duplicates(hashes, sorted(eval_hashes)) | near_duplicates(hashes, fair)
    out = {}
    for i, r in enumerate(crops):
        if i in drop:
            continue
        iid, uid = r["image_id"], "facecrop-" + r["image_id"].replace(":", "-").replace("#", "-")
        out[iid] = (r, [yes_no(uid + "-neg", "real_face_negative", rng.choice(REAL_NEGATIVES), False, iid, rng),
                        yes_no(uid + "-face", "real_face_negative", FACE, False, iid, rng)])
    stats["face_negatives"] = {"crops": len(crops), "dropped_near_duplicate": len(drop), "kept": len(out)}
    return out


def fidelity_level(rating: int) -> int:
    return {1: 0, 2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 4}[rating]


def imagereward(rows: list[dict], rng: random.Random) -> list[Decision]:
    out = []
    for r in rows:
        f, uid, iid = r["fidelity_rating"], f"imagereward-{r['image_id'].split(':')[1]}", r["image_id"]
        if f < 1:
            continue  # one image is rated 0, outside the 1-7 scale
        if f <= 3 or f >= 6:
            out.append(yes_no(uid + "-art", "imagereward_train", ART_Q, f <= 3, iid, rng, {"fidelity": f}))
        q, levels = rng.choice(PLAUS)
        level = fidelity_level(f)
        t = quality_target(level) if level < 4 else [0.0, 0.0, 0.0, 0.15, 0.85]
        out.append(Decision(uid + "-pl", "imagereward_train", "score", q, list(levels), t, b"", iid, {"fidelity": f}))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DATA / "clean_art_v1"))
    ap.add_argument("--general", default=str(DATA / "clean_v1"), help="licence-clean general mix to replay")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--templated-wordings", action="store_true", help="v3: hundreds of generated wordings per error family")
    ap.add_argument("--face-negatives", action="store_true",
                    help="v4: real close-up portraits (scripts/make_face_crops.py) with 'no errors' questions")
    a = ap.parse_args()
    if a.templated_wordings:
        use_templated_wordings()
    out, general, rng = Path(a.out), Path(a.general), random.Random(a.seed)
    out.mkdir(parents=True, exist_ok=True)
    assert PLAUS[0][1] == PLAUSIBILITY_LEVELS

    ai = {}
    for src in ("evalmuse", "imagereward"):
        att = {json.loads(line)["image_id"]: json.loads(line) for line in open(CLEAN / f"attribution_{src}.jsonl")}
        for line in open(CLEAN / f"labels_{src}.jsonl"):
            r = json.loads(line)
            ai[r["image_id"]] = r | {"attribution": att[r["image_id"]]}
    eval_hashes = set()
    for name in AI_EVAL_SETS + EVAL_SETS:
        seen = set()
        for d in ALL_BENCHMARKS[name]():
            if d.image_id not in seen:
                seen.add(d.image_id)
                eval_hashes.add(dhash(d.image_bytes))
    ids = sorted(ai)
    drop = {ids[k] for k in near_duplicates([int(ai[i]["attribution"]["dhash"]) for i in ids], sorted(eval_hashes))}
    stats = {"ai_images": dict(Counter(i.split(":")[0] for i in ai)), "dropped_near_duplicate": len(drop)}
    keep = [ai[i] for i in ids if i not in drop]

    decisions = evalmuse([r for r in keep if r["image_id"].startswith("evalmuse:")], rng)
    decisions += imagereward([r for r in keep if r["image_id"].startswith("imagereward:")], rng)

    # replay the general mix: its images, questions and calibration split, plus a real-photo negative
    gidx = json.loads((general / "images.json").read_text())
    blobs, rows, row_of = [], [], {}

    def row(image_id: str, path: str) -> int:
        if image_id not in row_of:
            blobs.append(path)
            rows.append([len(blobs) - 1, 0, -1])
            row_of[image_id] = len(rows) - 1
        return row_of[image_id]

    replay, calib = [], []
    for f, sink in (("decisions.jsonl", replay), ("calib.jsonl", calib)):
        for line in open(general / f):
            r = json.loads(line)
            r["image_row"] = row(r["image_id"], gidx["blobs"][gidx["rows"][r["image_row"]][0]])
            sink.append(r)
    real = sorted({r["image_id"] for r in replay + calib if "#" not in r["image_id"]})
    for iid in real:
        d = yes_no(f"realneg-{iid}", "real_photo_negative", rng.choice(REAL_NEGATIVES), False, iid, rng)
        rec = asdict(d)
        rec.pop("image_bytes")
        rec["image_row"] = row_of[iid]
        (calib if is_calib(iid) else replay).append(rec)
    for d in decisions:
        rec = asdict(d)
        rec.pop("image_bytes")
        rec["image_row"] = row(d.image_id, ai[d.image_id]["path"])
        (calib if is_calib(d.image_id) else replay).append(rec)
    faces = face_negatives(rng, eval_hashes, stats) if a.face_negatives else {}
    for iid, (r, ds) in faces.items():
        for d in ds:
            rec = asdict(d)
            rec.pop("image_bytes")
            rec["image_row"] = row(iid, r["path"])
            (calib if is_calib(iid) else replay).append(rec)

    rng.shuffle(replay)
    with open(out / "decisions.jsonl", "w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in replay)
    with open(out / "calib.jsonl", "w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in calib)
    (out / "images.json").write_text(json.dumps({"blobs": blobs, "rows": rows}))
    with open(out / "attribution.jsonl", "w") as fh:
        for line in open(general / "attribution.jsonl"):
            fh.write(line)
        for iid in row_of:
            if iid in ai:
                rec = {k: v for k, v in ai[iid]["attribution"].items() if k != "dhash"}
                fh.write(json.dumps(rec | {"image_id": iid, "path": ai[iid]["path"]}) + "\n")
            elif iid in faces:
                fh.write(json.dumps({k: v for k, v in faces[iid][0].items() if k != "dhash"}) + "\n")
    counts = Counter(f"{r['source']}:{r['kind']}" for r in replay)
    yes = Counter(r["source"] for r in replay if r["kind"] == "noul" and r["target"][0] > 0.5)
    noul = Counter(r["source"] for r in replay if r["kind"] == "noul")
    manifest = {"decisions": len(replay), "calib_decisions": len(calib), "images": len(rows), "counts": dict(counts),
                "yes_share": {s: round(yes[s] / n, 3) for s, n in noul.items()}, "stats": stats, "seed": a.seed}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
