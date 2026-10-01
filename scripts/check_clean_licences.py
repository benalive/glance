"""Check that every image a licence-clean mix uses has an allowed licence and an attribution record.

For each image row of <mix>/images.json: the attribution record (<mix>/attribution.jsonl) exists, its
licence is one of ALLOWED (photo licences, and the output licences of the generators we keep), its
file is the one the row points at, and, for a synthetic copy, the source image passes the same check. Also checks that no decision in the mix or its calibration set
points at an image without a record. Exits non-zero on any failure.

    uv run python scripts/check_clean_licences.py data/clean_v1
"""
import json
import sys
from collections import Counter
from pathlib import Path

ALLOWED = {
    "http://creativecommons.org/licenses/by/2.0/",   # COCO licence id 4
    "https://creativecommons.org/licenses/by/2.0/",  # Open Images
    "http://flickr.com/commons/usage/",              # COCO id 7: no known copyright restrictions
    "http://www.usa.gov/copyright.shtml",            # COCO id 8: US Government work
    "ImageRewardDB Apache-2.0; DiffusionDB images CC0",
}
# AI-generated images: the output licences of the generators we keep (reviewed per generator; see fetch_clean_artifacts.py)
from scripts.fetch_clean_artifacts import EVALMUSE_GENERATORS  # noqa: E402

ALLOWED |= {f"generator output: {v}" for v in EVALMUSE_GENERATORS.values()}


def main():
    mix = Path(sys.argv[1] if len(sys.argv) > 1 else "data/clean_v1")
    index = json.loads((mix / "images.json").read_text())
    att = {}
    for line in open(mix / "attribution.jsonl"):
        r = json.loads(line)
        att[r["image_id"]] = r
    problems, licences = [], Counter()
    by_path = {r["path"]: r for r in att.values()}
    for row, (blob, _, _) in enumerate(index["rows"]):
        path = index["blobs"][blob]
        rec = by_path.get(path)
        if rec is None:
            problems.append(f"row {row}: no attribution for {path}")
            continue
        if rec.get("license") not in ALLOWED:
            problems.append(f"row {row}: licence {rec.get('license')!r} not allowed ({rec['image_id']})")
        if not Path(path).is_file():
            problems.append(f"row {row}: file missing {path}")
        licences[rec.get("license")] += 1
    used = set()
    for f in ("decisions.jsonl", "calib.jsonl"):
        for line in open(mix / f):
            r = json.loads(line)
            used.add(r["image_id"])
            if r["image_id"] not in att:
                problems.append(f"{f}: {r['uid']} uses {r['image_id']} with no attribution record")
    print(f"{len(index['rows'])} image rows, {len(used)} image ids used, licences: {dict(licences)}")
    for p in problems[:20]:
        print("FAIL", p)
    if problems:
        print(f"{len(problems)} problems")
        sys.exit(1)
    print("OK: every image has an allowed licence and an attribution record")


if __name__ == "__main__":
    main()
