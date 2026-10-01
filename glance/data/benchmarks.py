"""Loaders that turn the R0 eval sets into Decisions.

POPE-adversarial (noul), A-OKVQA val (choice, K=4), MMStar (choice, K=2-4, eval-only),
SugarCrepe replace_rel / swap_att (choice, K=2) and KonIQ-10k test (score, 5 levels with the rater
histogram as target). All reads are from the local HF cache populated by scripts/fetch_r0.py.
"""
import csv
import hashlib
import glob
import json
import os
import re
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

from glance.data.schema import Decision, bytes_id

KONIQ_DIR = Path("data/raw/koniq10k/512x384")
SCORE_LEVELS = ["bad", "poor", "fair", "good", "excellent"]


def _parquet(repo: str, pattern: str) -> list[str]:
    root = snapshot_download(repo, repo_type="dataset", allow_patterns=[pattern], local_files_only=True)
    files = sorted(glob.glob(os.path.join(root, pattern)))
    assert files, (repo, pattern)
    return files


def _rows(repo: str, pattern: str):
    for f in _parquet(repo, pattern):
        yield from pq.read_table(f).to_pylist()


def pope(subset: str = "adversarial") -> list[Decision]:
    out = []
    for r in _rows("lmms-lab-encoder/POPE", f"Full/{subset}-*.parquet"):
        coco = int(r["image_source"].rsplit("_", 1)[-1])
        yes = r["answer"].strip().lower() == "yes"
        out.append(Decision(
            uid=f"pope-{subset}-{r['question_id']}", source=f"pope_{subset}", kind="noul", question=r["question"],
            candidates=["yes", "no"], target=[1.0, 0.0] if yes else [0.0, 1.0],
            image_bytes=r["image"]["bytes"], image_id=f"coco:{coco}", meta={"coco_split": "val2014"}))
    return out


def aokvqa_val() -> list[Decision]:
    out = []
    for r in _rows("HuggingFaceM4/A-OKVQA", "data/validation-*.parquet"):
        k = len(r["choices"])
        target = [0.0] * k
        target[r["correct_choice_idx"]] = 1.0
        img = r["image"]["bytes"]
        out.append(Decision(
            uid=f"aokvqa-{r['question_id']}", source="aokvqa_val", kind="choice", question=r["question"],
            candidates=list(r["choices"]), target=target, image_bytes=img, image_id=bytes_id(img),
            meta={"coco_split": "val2017", "note": "HF copy has no COCO id; image_id is a content hash"}))
    return out


_OPT = re.compile(r"(?:^|,\s*|\n\s*)([A-F])[:.]\s")
_PAREN_OPT = re.compile(r"^\(([A-F])\)\s*(.*)$", re.M)


def parse_mmstar(question: str) -> tuple[str, list[str]]:
    """Split an MMStar question into (stem, [option texts]).

    Two layouts occur: 'stem\\nOptions: A: x, B: y, ...' (most sources) and MathVista's
    'Hint: ...\\nQuestion: stem\\nChoices:\\n(A) x\\n(B) y'. The hint only asks for a letter, so it
    is dropped. Letters must appear in order A, B, C...; option text runs to the next marker."""
    if "\nChoices:" in question:
        head, _, body = question.partition("\nChoices:")
        stem = head.split("Question:", 1)[-1].strip()
        found = _PAREN_OPT.findall(body.strip())
        if [L for L, _ in found] != [chr(ord("A") + i) for i in range(len(found))] or len(found) < 2:
            raise ValueError("bad Choices: block")
        return stem, [t.strip() for _, t in found]
    stem, sep, opts = question.rpartition("Options:")
    if not sep:
        raise ValueError("no Options: marker")
    opts = opts.strip()
    marks, expect = [], ord("A")
    for m in _OPT.finditer(opts):
        if ord(m.group(1)) == expect:
            marks.append(m)
            expect += 1
    if len(marks) < 2:
        raise ValueError(f"parsed {len(marks)} options")
    texts = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(opts)
        texts.append(opts[m.end():end].strip().rstrip(",").strip())
    return stem.strip(), texts


def mmstar() -> list[Decision]:
    out, skipped = [], 0
    for r in _rows("Lin-Chen/MMStar", "mmstar.parquet"):
        try:
            stem, opts = parse_mmstar(r["question"])
        except ValueError:
            skipped += 1
            continue
        idx = ord(r["answer"].strip()) - ord("A")
        if not 0 <= idx < len(opts) or opts[idx].lower() == "nan":
            skipped += 1
            continue
        # MMStar pads 2- and 3-option items with literal 'nan' options; drop them (found in the
        # R1 data-integrity audit: 176 items were being scored as 4-way).
        keep = [i for i, o in enumerate(opts) if o.lower() != "nan"]
        idx, opts = keep.index(idx), [opts[i] for i in keep]
        target = [0.0] * len(opts)
        target[idx] = 1.0
        img = r["image"]
        out.append(Decision(
            uid=f"mmstar-{r['index']}", source="mmstar", kind="choice", question=stem, candidates=opts, target=target,
            image_bytes=img, image_id=bytes_id(img),
            meta={"category": r["category"], "l2_category": r["l2_category"], "origin": r["meta_info"]["source"]}))
    if skipped:
        print(f"mmstar: skipped {skipped} unparseable items")
    return out


def sugarcrepe(subset: str) -> list[Decision]:
    f = _parquet(f"HuggingFaceM4/SugarCrepe_{subset}", "data/*.parquet")[0]
    table = pq.read_table(f)
    names = json.loads(table.schema.metadata[b"huggingface"])["info"]["features"]["true_label"]["names"]
    out, skipped = [], 0
    for i, r in enumerate(table.to_pylist()):
        true = names[r["true_label"]].strip()
        caps = [c.strip() for c in r["tested_labels"]]
        hits = [j for j, c in enumerate(caps) if c == true]
        if len(hits) != 1:
            skipped += 1
            continue
        target = [0.0] * len(caps)
        target[hits[0]] = 1.0
        # The HF release always lists the true caption first, which a position-biased readout
        # would exploit; swap the pair for a deterministic half of the items.
        if int(hashlib.sha1(f"{subset}-{i}".encode()).hexdigest(), 16) % 2:
            caps, target = caps[::-1], target[::-1]
        coco = int(Path(r["image"]["path"]).stem)
        out.append(Decision(
            uid=f"sugarcrepe-{subset}-{i}", source=f"sugarcrepe_{subset}", kind="choice",
            question="Which caption describes the image?", candidates=caps, target=target,
            image_bytes=r["image"]["bytes"], image_id=f"coco:{coco}", meta={"coco_split": "val2017"}))
    if skipped:
        print(f"sugarcrepe {subset}: skipped {skipped} items whose true caption did not match exactly once")
    return out


def koniq_test() -> list[Decision]:
    root = snapshot_download("chaofengc/IQA-PyTorch-Datasets-metainfo", repo_type="dataset",
                             allow_patterns=["meta_info_KonIQ10kDataset.csv"], local_files_only=True)
    out = []
    with open(os.path.join(root, "meta_info_KonIQ10kDataset.csv")) as fh:
        for r in csv.DictReader(fh):
            if r["official_split"] != "test":
                continue
            hist = [float(r[f"c{i}"]) for i in range(1, 6)]
            total = sum(hist)
            img = (KONIQ_DIR / r["img_name"]).read_bytes()
            out.append(Decision(
                uid=f"koniq-{r['img_name']}", source="koniq_test", kind="score",
                question="How good is the technical quality of this image?", candidates=list(SCORE_LEVELS),
                target=[h / total for h in hist], image_bytes=img, image_id=f"koniq:{r['img_name']}",
                meta={"mos": float(r["mos"]), "n_ratings": int(r["c_total"])}))
    return out


def vqav2_val_yesno(train_decisions: str = "data/r1/decisions.jsonl") -> list[Decision]:
    """VQAv2 validation yes/no questions with a clear majority (>= 5 of 10 annotators say yes or no),
    on images that are not in the R1 training mix: COCO ids of the COCO-presence and VQAv2 training
    questions, plus near-duplicate dHashes of the A-OKVQA training images. Everyday phrasings ("Is the
    man...", "Are there any...") that the COCO-presence template does not cover. Needs the VQAv2
    validation shards in the HF cache (any subset; the item set grows with the shards present)."""
    import json as _json
    from collections import Counter as _Counter

    from glance.data.train_mix import _local, dhash, near_duplicates  # deferred: train_mix imports this module

    train_ids, train_hashes = set(), set()
    for line in open(train_decisions):
        r = _json.loads(line)
        if r["image_id"].startswith("coco:"):
            train_ids.add(int(r["image_id"].split(":")[1]))
        elif "dhash" in r.get("meta", {}):
            train_hashes.add(r["meta"]["dhash"])
    rows, images = [], {}
    for f in _local("pingzhili/vqa_v2", "data/validation-*.parquet"):
        for r in pq.read_table(f).to_pylist():
            if r["answer_type"] == "yes/no" and r["image_id"] not in train_ids:
                images.setdefault(r["image_id"], r["image"]["bytes"])
                rows.append(r)
    ids = list(images)
    seen = {ids[k] for k in near_duplicates([dhash(images[i]) for i in ids], sorted(train_hashes))}
    out = []
    for r in rows:
        votes = _Counter(a["answer"].strip().lower() for a in r["answers"])
        y, n = votes.get("yes", 0), votes.get("no", 0)
        if r["image_id"] in seen or y + n < 5:
            continue
        out.append(Decision(f"vqav2val-{r['question_id']}", "vqav2_val", "noul", r["question"], ["yes", "no"],
                            [y / (y + n), n / (y + n)], images[r["image_id"]], f"coco:{r['image_id']}",
                            {"question_type": r["question_type"]}))
    return out


R0_BENCHMARKS = {
    "pope_adversarial": lambda: pope("adversarial"),
    "aokvqa_val": aokvqa_val,
    "mmstar": mmstar,
    "sugarcrepe_replace_rel": lambda: sugarcrepe("replace_rel"),
    "sugarcrepe_swap_att": lambda: sugarcrepe("swap_att"),
    "koniq_test": koniq_test,
}

ARTIFACTS = Path("data/artifacts")
HAD_ANY_Q = "Does this image contain visible errors, such as malformed hands, faces or limbs?"
HAD_HAND_Q = "Are any hands in this image malformed, with extra, missing or deformed fingers?"
ARTIFACT_Q = "Does this image contain visible generation errors?"
PLAUSIBILITY_Q = "How free of visible generation errors is this image?"
PLAUSIBILITY_LEVELS = ["severe errors", "clear errors", "some errors", "minor errors", "no errors"]


def _had_rows(split: str):
    for line in open(ARTIFACTS / "had" / f"{split}.jsonl"):
        r = json.loads(line)
        r["image_bytes"] = (ARTIFACTS / "had" / r["file"]).read_bytes()
        yield r


def had_hands(split: str = "val_ALL") -> list[Decision]:
    """HAD (human-labelled boxes on malformed body parts in SDXL / Midjourney / DALL-E images of people):
    yes if an annotator boxed a malformed hand or tagged a person with a missing or extra hand."""
    out = []
    for r in _had_rows(split):
        bad = r["parts"].get("hand", 0) > 0 or any("hand" in t for t in r["human_tags"])
        out.append(Decision(f"had-hand-{r['file']}", f"had_{split}", "noul", HAD_HAND_Q, ["yes", "no"],
                            [1.0, 0.0] if bad else [0.0, 1.0], r["image_bytes"], f"had:{r['file']}",
                            {"generator": r["source"]}))
    return out


def had_any(split: str = "val_ALL") -> list[Decision]:
    """HAD: yes if anything was boxed or tagged; no only for images tagged 'good' with nothing marked;
    images with neither (untagged, unmarked) are dropped as unclear."""
    out = []
    for r in _had_rows(split):
        bad = bool(r["parts"]) or bool(r["human_tags"])
        if not bad and r["tag"] != "good":
            continue
        out.append(Decision(f"had-any-{r['file']}", f"had_{split}", "noul", HAD_ANY_Q, ["yes", "no"],
                            [1.0, 0.0] if bad else [0.0, 1.0], r["image_bytes"], f"had:{r['file']}",
                            {"generator": r["source"]}))
    return out


def _richhf_rows(split: str):
    for line in open(ARTIFACTS / "richhf" / f"{split}.jsonl"):
        r = json.loads(line)
        r["image_bytes"] = (ARTIFACTS / "richhf" / r["file"]).read_bytes()
        yield r


def richhf_artifacts(split: str = "test") -> list[Decision]:
    """RichHF-18K artifact_score (mean of 3 raters, 1 = no visible artifacts): yes if <= 0.5, no if
    >= 0.917 (essentially clean for all raters); the ambiguous middle is dropped."""
    out = []
    for r in _richhf_rows(split):
        s = r["artifact_score"]
        if 0.5 < s < 0.917:
            continue
        out.append(Decision(f"richhf-art-{r['file']}", f"richhf_{split}", "noul", ARTIFACT_Q, ["yes", "no"],
                            [1.0, 0.0] if s <= 0.5 else [0.0, 1.0], r["image_bytes"], f"richhf:{r['file']}",
                            {"artifact_score": s}))
    return out


def richhf_plausibility(split: str = "test") -> list[Decision]:
    """RichHF-18K artifact_score as a five-level score (level = round(4 s)); `mos` carries the raw score
    so it is ranked like KonIQ (SRCC of the expected level)."""
    out = []
    for r in _richhf_rows(split):
        s = r["artifact_score"]
        t = [0.0] * 5
        t[min(4, round(4 * s))] = 1.0
        out.append(Decision(f"richhf-pl-{r['file']}", f"richhf_{split}", "score", PLAUSIBILITY_Q, list(PLAUSIBILITY_LEVELS),
                            t, r["image_bytes"], f"richhf:{r['file']}", {"mos": s}))
    return out


def reworded(base, question: str, flip: bool):
    """The same items asked in a wording kept out of training (scripts/build_artifact_mix.py HELD_OUT);
    flip=True when the new wording's "yes" means "no error"."""
    def load():
        out = []
        for d in base():
            t = list(reversed(d.target)) if flip else list(d.target)
            out.append(Decision(d.uid + "-reworded", d.source, d.kind, question, d.candidates, t, d.image_bytes,
                                d.image_id, dict(d.meta)))
        return out
    return load


def real_photos_no_errors(n_per_source: int = 500) -> list[Decision]:
    """Real photographs, so every generation-error question should be answered "no error": 500 COCO
    val2014 images from POPE and 500 KonIQ-10k test images (neither is in any training mix). Three
    questions per image; "yes" means an error except for "Does this image look correct?"."""
    qs = [(ARTIFACT_Q, True), (HAD_HAND_Q, True), ("Does this image look correct?", False)]
    images = {}
    for d in pope("adversarial"):
        if len(images) < n_per_source:
            images.setdefault(d.image_id, d.image_bytes)
    for i, d in enumerate(koniq_test()):
        if i >= n_per_source:
            break
        images.setdefault(d.image_id, d.image_bytes)
    out = []
    for img_id, data in images.items():
        for k, (q, yes_is_error) in enumerate(qs):
            t = [0.0, 1.0] if yes_is_error else [1.0, 0.0]  # real photo: no error
            out.append(Decision(f"real-{img_id}-{k}", "real_photos", "noul", q, ["yes", "no"], t, data, img_id))
    return out


EXTERNAL = Path("data/external")
# SalArt-VQA's published prompts are long instruction blocks (read by generative models). Glance reads a
# short question plus the candidate answers, so it gets each prompt's question sentence and the options.
SALART_Q = {1: "Does this image contain at least one salient artifact?",
            2: "Which region best captures the most salient artifact?",
            3: "Which box best captures the most salient artifact?",
            4: "Which option best describes the most salient artifact?"}


def salart(question: int) -> list[Decision]:
    """SalArt-VQA (arXiv 2606.12671), one of its four questions for every image. Rows derived from HAD
    (ids 'hadm__...', 38 flawed images and their references) are dropped: HAD is in our training data.
    meta: role (artifact / clean_reference = real COCO photo / paired_generated_counterpart), type."""
    out = []
    for line in open(EXTERNAL / "salart" / "data" / "test.jsonl"):
        r = json.loads(line)
        if r["row_id"].startswith("hadm__") or r[f"q{question}_answer"] is None:
            continue  # HAD-derived; or unscored (Q1 has no answer for the 119 unflawed generated images)
        img = r["q3_overlay_image_path"] if question == 3 else r["image_path"]
        meta = {"role": r["image_role"], "type": r["artifact_type"]}
        if question == 1:
            yes = r["q1_answer"].strip().lower() == "yes"
            cands, target = ["yes", "no"], [1.0, 0.0] if yes else [0.0, 1.0]
        else:
            if question == 3:
                opts = {k: f"box {k}" for k in "ABCD"} | {"E": "none of these boxes"}
            else:
                opts = r[f"q{question}_options"]
            letters = sorted(opts)
            cands = [opts[k] for k in letters]
            target = [1.0 if k == r[f"q{question}_answer"].strip() else 0.0 for k in letters]
            if len(set(cands)) < len(cands):
                continue
        out.append(Decision(f"salart-q{question}-{r['row_id']}", "salart", "noul" if question == 1 else "choice",
                            SALART_Q[question], cands, target, (EXTERNAL / "salart" / img).read_bytes(),
                            f"salart:{r['pair_id'] or r['row_id']}", meta))
    return out


def artibench() -> list[Decision]:
    """ArtiBench (arXiv 2602.20951): 1,000 generated images, half with visual artifacts; yes/no."""
    labels = json.loads((EXTERNAL / "artibench" / "labels.json").read_text())
    return [Decision(f"artibench-{r['id']}", "artibench", "noul", "Does this image contain any visual artifacts?",
                     ["yes", "no"], [1.0, 0.0] if r["has_artifacts"] else [0.0, 1.0],
                     (EXTERNAL / "artibench" / "images" / f"{r['id']}.png").read_bytes(), f"artibench:{r['id']}")
            for r in labels]


def abench_generative() -> list[Decision]:
    """A-Bench (arXiv 2406.03070) validation split, 'generative distortion' questions (250 multiple choice,
    2-4 options; the test split's answers are not public)."""
    import base64

    csv.field_size_limit(10**9)
    out = []
    for r in csv.DictReader(open(EXTERNAL / "abench" / "A-bench_VAL.tsv"), delimiter="\t"):
        if not r["category"].startswith("part2 -> generative"):
            continue
        letters = [k for k in "ABCD" if r.get(k, "").strip()]
        cands = [r[k].strip() for k in letters]
        out.append(Decision(f"abench-{r['index']}", "abench", "choice", r["question"].strip(), cands,
                            [1.0 if k == r["answer"].strip() else 0.0 for k in letters],
                            base64.b64decode(r["image"]), f"abench:{r['index']}"))
    return out


# Not part of the R1 training runs' standard evaluation (that set is fixed mid-round); evaluated by
# scripts/api_benchmark.py and, for the zero-shot floors, scripts/frontier_eval.py --benches.
EXTRA_BENCHMARKS = {
    "pope_random": lambda: pope("random"),
    "pope_popular": lambda: pope("popular"),
    "vqav2_val_yesno": vqav2_val_yesno,
    "had_val_hands": had_hands,
    "had_val_any": had_any,
    "richhf_test_artifacts": richhf_artifacts,
    "richhf_test_plausibility": richhf_plausibility,
    # the same images in wordings never used for training, "yes" = the image is fine
    "had_val_any_looks_correct": reworded(had_any, "Does this image look correct?", flip=True),
    "had_val_any_anatomy_correct": reworded(had_any, "Is this image correct from a human anatomy perspective?", flip=True),
    "had_val_hands_anatomy_correct": reworded(had_hands, "Is the hand in this image anatomically correct?", flip=True),
    "richhf_test_looks_correct": reworded(richhf_artifacts, "Does this image look correct?", flip=True),
    "had_val_any_anatomically_off": reworded(had_any, "Is there anything anatomically off about this image?", flip=False),
    "had_val_any_people_realistic": reworded(had_any, "Do the people in this image look realistic?", flip=True),
    "real_photos_no_errors": real_photos_no_errors,
    # published benchmarks with leaderboards of leading open-weight and closed models
    "salart_q1": lambda: salart(1), "salart_q2": lambda: salart(2), "salart_q3": lambda: salart(3),
    "salart_q4": lambda: salart(4), "artibench": artibench, "abench_generative": abench_generative,
}
ALL_BENCHMARKS = {**R0_BENCHMARKS, **EXTRA_BENCHMARKS}
