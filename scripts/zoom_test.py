"""Inference-only test: do SalArt's small inpainted flaws become visible to the released model if it looks at
zoomed tiles?

SalArt's injected flaws are mostly small (an extra finger, a wrong count), a few pixels at the model's 256 px
input. Without retraining, each image is also cut into k x k overlapping tiles (each tile 1/k of the side plus
25% overlap, resized to the model's input like any image), the error question is asked of the whole image and
of every tile, and the image score is the maximum P(yes). Compared with the whole-image score on: SalArt pairs
(flawed vs its clean original), SalArt flawed vs real photos, ArtiBench (flawed vs clean generated), and
1,000 real photographs (share flagged, at the whole-image served threshold and at the tile-max threshold that
keeps SalArt real-photo flags equal to the whole-image rate). Cost: 1 + k^2 image encodings per image.

    uv run python scripts/zoom_test.py     # -> results/zoom_test.{json,md}
"""
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from glance.data.benchmarks import EXTERNAL, SALART_Q, artibench, real_photos_no_errors
from glance.metrics import auroc
from glance.serve import Predictor

PACKAGE = "data/release/glance-c5-open-v4"
GRIDS = (2, 3)


def tiles(data: bytes, k: int) -> list[bytes]:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    w, h = img.size
    tw, th = int(w / k * 1.25), int(h / k * 1.25)
    out = []
    for i in range(k):
        for j in range(k):
            x = min(int(i * (w - tw) / max(k - 1, 1)), w - tw)
            y = min(int(j * (h - th) / max(k - 1, 1)), h - th)
            buf = io.BytesIO()
            img.crop((x, y, x + tw, y + th)).save(buf, "PNG")
            out.append(buf.getvalue())
    return out


def main():
    pred = Predictor(PACKAGE, device="cpu", threads=4)

    def p(data: bytes, question: str) -> float:
        a = pred.predict({"q": {"type": "noul", "instructions": question}}, data)["answers"]["q"]["noul"]
        pred.cache.clear()
        return a

    def scores(data: bytes, question: str) -> dict:
        s = {"whole": p(data, question)}
        for k in GRIDS:
            s[f"tiles{k}"] = max([s["whole"]] + [p(t, question) for t in tiles(data, k)])
        return s

    sal = {"flawed": [], "clean": [], "real": []}
    pair_of = {}
    for line in open(EXTERNAL / "salart" / "data" / "test.jsonl"):
        r = json.loads(line)
        if r["row_id"].startswith("hadm__"):
            continue
        role = {"artifact": "flawed", "paired_generated_counterpart": "clean", "clean_reference": "real"}[r["image_role"]]
        s = scores((EXTERNAL / "salart" / r["image_path"]).read_bytes(), SALART_Q[1])
        sal[role].append(s)
        if r["row_id"].startswith("artifact_injection"):
            pair_of.setdefault(r["pair_id"], {})[role] = s
    pairs = [v for v in pair_of.values() if len(v) == 2]
    arti = [(scores(d.image_bytes, d.question), d.target[0] > 0.5) for d in artibench()]
    real = [scores(d.image_bytes, d.question) for d in real_photos_no_errors() if d.question != "Does this image look correct?"]
    res = {"n_pairs": len(pairs), "methods": {}}
    for m in ["whole"] + [f"tiles{k}" for k in GRIDS]:
        fl, cl, rl = [s[m] for s in sal["flawed"]], [s[m] for s in sal["clean"]], [s[m] for s in sal["real"]]
        thr = float(np.quantile(rl, 1 - np.mean([s["whole"] >= 0.5 for s in sal["real"]]))) if m != "whole" else 0.5
        res["methods"][m] = {
            "encodings_per_image": 1 + (int(m[5:]) ** 2 if m != "whole" else 0),
            "salart_pairs_auroc": auroc([v["flawed"][m] for v in pairs] + [v["clean"][m] for v in pairs], [1] * len(pairs) + [0] * len(pairs)),
            "salart_flawed_vs_real_auroc": auroc(fl + rl, [1] * len(fl) + [0] * len(rl)),
            "salart_flawed_vs_clean_ai_auroc": auroc(fl + cl, [1] * len(fl) + [0] * len(cl)),
            "artibench_auroc": auroc([s[m] for s, _ in arti], [y for _, y in arti]),
            "threshold_matched_on_salart_real": thr,
            "salart_recall_at_threshold": float(np.mean([x >= thr for x in fl])),
            "real_photo_questions_flagged_at_threshold": float(np.mean([s[m] >= thr for s in real])),
        }
        print(m, {k: round(v, 3) for k, v in res["methods"][m].items()}, flush=True)
    Path("results/zoom_test.json").write_text(json.dumps(res, indent=1))
    cols = list(next(iter(res["methods"].values())))
    L = [f"Zoomed tiles, inference only, released model ({PACKAGE}); image score = max P(yes) over the whole image and its tiles. "
         f"SalArt pairs: {len(pairs)} inpainted flaws vs their clean originals. Threshold for tiles chosen so SalArt's real "
         "photos are flagged as often as with the whole image at 0.5.", "", "| method | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for m, r in res["methods"].items():
        L.append(f"| {m} | " + " | ".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
    Path("results/zoom_test.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
