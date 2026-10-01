"""Training mix for spotting visible errors in AI-generated images, plus a replay of the R1 mix.

  HAD train_ALL: malformed hands, any anatomy error, and one body-part question per image.
  RichHF-18K train: visible generation errors (clear cases only) and a five-level plausibility score.
  R1 replay: all questions on a sample of R1 training images, so earlier skills are kept.

Every question type is asked in several wordings with both polarities ("Are there deformed hands?"
and "Are the hands drawn correctly?"), one wording drawn per image. The wordings in HELD_OUT are never
used here; glance/data/benchmarks.py asks the validation/test images with them to check that answers
carry over to how people actually phrase things.

Images stay where they are (data/artifacts/..., data/r1/images.bin); data/r2art/images.json indexes them.

    uv run python scripts/build_artifact_mix.py   # -> data/r2art/{decisions.jsonl, images.json, manifest.json}
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from glance.data.benchmarks import ARTIFACT_Q, HAD_ANY_Q, HAD_HAND_Q, PLAUSIBILITY_LEVELS, PLAUSIBILITY_Q
from glance.data.schema import Decision

ART = Path("data/artifacts")
OUT = Path("data/r2art_v2")
R1 = Path("data/r1")
REPLAY_IMAGES = 24_000

# (wording, polarity): polarity +1 means "yes" = the error is present, -1 means "yes" = no error.
# Many wordings in both polarities, so the model learns the concept rather than a template; none of
# them may appear in HELD_OUT (checked below).
HAND = [(HAD_HAND_Q, 1), ("Does anyone in this image have an extra or missing finger?", 1),
        ("Are there deformed hands in this image?", 1), ("Do any hands have the wrong number of fingers?", 1),
        ("Is there something wrong with the hands in this image?", 1), ("Are any fingers fused, twisted or duplicated?", 1),
        ("Does this image show a hand with too many or too few fingers?", 1), ("Are the hands in this picture distorted?", 1),
        ("Are the hands in this image drawn correctly?", -1), ("Do all the hands in this image look normal?", -1),
        ("Does every hand in this image have the right number of fingers?", -1), ("Are the hands anatomically plausible?", -1),
        ("Do the fingers look natural?", -1), ("Are all hands in this picture well-formed?", -1),
        ("Is the hand anatomy in this image right?", -1), ("Do the hands have a correct number of fingers?", -1)]
ANY = [(HAD_ANY_Q, 1), ("Are there any deformed body parts in this image?", 1),
       ("Does this image contain anatomical errors?", 1), ("Is anyone's body distorted in this image?", 1),
       ("Are there anatomy mistakes in this image?", 1), ("Does anything about the people here look physically impossible?", 1),
       ("Are there visible errors in the human figures?", 1), ("Is there a problem with anyone's anatomy?", 1),
       ("Is this image free of anatomical errors?", -1), ("Do the people in this image look anatomically normal?", -1),
       ("Is the human anatomy in this image correct?", -1), ("Are the people in this image anatomically accurate?", -1),
       ("Does everyone's body look natural?", -1), ("Is this image anatomically plausible?", -1),
       ("Do the bodies in this image look right?", -1), ("Is this picture accurate in terms of anatomy?", -1)]
FACE = [("Is anyone's face distorted or malformed?", 1), ("Are there any deformed faces?", 1),
        ("Do all the faces in this image look normal?", -1), ("Are the faces anatomically correct?", -1)]
LIMB = [("Are any arms, legs or feet malformed, missing or extra?", 1), ("Does anyone have an extra or missing limb?", 1),
        ("Do all arms, legs and feet look normal?", -1), ("Are the limbs in this image correct?", -1)]
ART_Q = [(ARTIFACT_Q, 1), ("Are there visible artifacts or distortions in this image?", 1),
         ("Does anything in this image look malformed?", 1), ("Does this image have rendering glitches?", 1),
         ("Are there obvious AI-generation mistakes in this image?", 1), ("Does any object look warped or broken?", 1),
         ("Is this image free of visible errors?", -1), ("Do all objects in this image look well-formed?", -1),
         ("Does everything in this image look physically plausible?", -1), ("Is this image clean and artifact-free?", -1),
         ("Does this image look realistic and error-free?", -1), ("Is everything in this picture drawn correctly?", -1)]
PLAUS = [(PLAUSIBILITY_Q, PLAUSIBILITY_LEVELS),
         ("How well-formed does everything in this image look?",
          ["very malformed", "malformed", "somewhat odd", "mostly fine", "perfectly fine"])]
HELD_OUT = ["Does this image look correct?", "Is this image correct from a human anatomy perspective?",
            "Is the hand in this image anatomically correct?", "Is there anything anatomically off about this image?",
            "Do the people in this image look realistic?"]
assert not {q for fam in (HAND, ANY, FACE, LIMB, ART_Q) for q, _ in fam} & set(HELD_OUT)
REAL_NEGATIVES = [HAND, ANY, ART_Q]  # real photographs (the replayed R1 images) have no generation errors


def yes_no(uid, source, wordings, error: bool, img_id, rng, meta=None):
    q, pol = rng.choice(wordings)
    yes = error if pol == 1 else not error
    return Decision(uid, source, "noul", q, ["yes", "no"], [1.0, 0.0] if yes else [0.0, 1.0], b"", img_id, meta or {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--replay", type=int, default=REPLAY_IMAGES, help="R1 images to replay")
    ap.add_argument("--had", default=str(ART / "had" / "train_ALL.jsonl"), help="HAD labels (fetch_artifacts.py)")
    a = ap.parse_args()
    out = Path(a.out)
    rng = random.Random(0)
    out.mkdir(parents=True, exist_ok=True)
    blobs, rows, decisions, counts = [], [], [], Counter()

    def add_image(path: Path) -> int:
        blobs.append(str(path))
        rows.append([len(blobs) - 1, 0, -1])  # whole file
        return len(rows) - 1

    for line in open(a.had):
        r = json.loads(line)
        row, img_id = add_image(ART / "had" / r["file"]), f"had:{r['file']}"
        tags = " ".join(r["human_tags"])
        hand = r["parts"].get("hand", 0) > 0 or "hand" in tags
        face = r["parts"].get("face", 0) > 0 or "face" in tags
        limb = any(r["parts"].get(p, 0) > 0 or p in tags for p in ("arm", "leg", "feet", "foot"))
        anyerr = bool(r["parts"]) or bool(r["human_tags"])
        uid = f"had-{r['file']}"
        ds = [yes_no(uid + "-hand", "had_train", HAND, hand, img_id, rng)]
        if anyerr or r["tag"] == "good":  # untagged images with nothing marked are unclear
            ds.append(yes_no(uid + "-any", "had_train", ANY, anyerr, img_id, rng))
        part, err = rng.choice([(FACE, face), (LIMB, limb)])
        ds.append(yes_no(uid + "-part", "had_train", part, err, img_id, rng))
        for d in ds:
            d.meta["image_row"] = row
        decisions += ds
        counts["had_images"] += 1

    for line in open(ART / "richhf" / "train.jsonl"):
        r = json.loads(line)
        row, img_id = add_image(ART / "richhf" / r["file"]), f"richhf:{r['file']}"
        s, uid = r["artifact_score"], f"richhf-{r['file']}"
        ds = []
        if s <= 0.5 or s >= 0.917:
            ds.append(yes_no(uid + "-art", "richhf_train", ART_Q, s <= 0.5, img_id, rng))
        q, levels = rng.choice(PLAUS)
        t = [0.0] * 5
        t[min(4, round(4 * s))] = 1.0
        ds.append(Decision(uid + "-pl", "richhf_train", "score", q, list(levels), t, b"", img_id, {"mos": s}))
        for d in ds:
            d.meta["image_row"] = row
        decisions += ds
        counts["richhf_images"] += 1

    # replay: every R1 question on a random sample of R1 training images (not dev, not audited-out)
    from glance.train import R1Data

    r1 = R1Data(R1)
    by_row = defaultdict(list)
    for d in r1.decisions("train"):
        by_row[d.meta["image_row"]].append(d)
    blobs.append(str(R1 / "images.bin"))
    r1_blob = len(blobs) - 1
    for old_row in rng.sample(sorted(by_row), min(a.replay, len(by_row))):
        rows.append([r1_blob, int(r1.offsets[old_row]), int(r1.offsets[old_row + 1] - r1.offsets[old_row])])
        for d in by_row[old_row]:
            d.meta = {k: v for k, v in d.meta.items() if k != "image_row"} | {"image_row": len(rows) - 1}
            decisions.append(d)
        img_id = by_row[old_row][0].image_id
        neg = yes_no(f"real-{img_id}", "real_photo_negative", rng.choice(REAL_NEGATIVES), False, img_id, rng)
        neg.meta["image_row"] = len(rows) - 1
        decisions.append(neg)
        counts["replay_images"] += 1

    rng.shuffle(decisions)
    with open(out / "decisions.jsonl", "w") as fh:
        for d in decisions:
            rec = asdict(d)
            rec.pop("image_bytes")
            rec["image_row"] = rec["meta"].pop("image_row")
            fh.write(json.dumps(rec) + "\n")
    (out / "images.json").write_text(json.dumps({"blobs": blobs, "rows": rows}))
    counts.update(Counter(f"{d.source}:{d.kind}" for d in decisions))
    yes_share = np.mean([d.label == 0 for d in decisions if d.kind == "noul" and d.source != "vqav2_train"])
    manifest = {"decisions": len(decisions), "images": len(rows), "counts": dict(counts), "held_out_wordings": HELD_OUT,
                "artifact_yes_share": float(yes_share)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
