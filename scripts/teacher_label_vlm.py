"""Label a training mix's questions with Qwen3-VL's probabilities, for distillation into Glance.

Qwen3-VL (Apache-2.0; its outputs may be used for training) reads each question once
(glance.baselines.vlm_readout.VLMReadout: yes/no from the yes/no token log-probs, choice from the
option letters, one pass, no rotation debiasing). Only the chosen sources are labelled; by default the
everyday-question sets, since Qwen is poor at spotting generation errors (AUROC 0.59-0.65).

Raw readouts are overconfident (fitted temperatures 2-12 in R0), so the script also reads the mix's
held-out calibration questions (calib.jsonl, never trained on) and fits Platt scaling for yes/no and a
temperature for choice against their gold answers; the written probabilities are calibrated.

Resumable: raw readouts are appended to <data_root>/teacher_<name>.raw.jsonl as they are made; the
final file <data_root>/teacher_<name>.jsonl ({"uid", "p"}, train questions only) is written at the end.

    uv run python scripts/teacher_label_vlm.py --data-root data/clean_v1
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from glance.baselines.vlm_readout import VLMReadout
from glance.data.schema import Decision
from glance.metrics import fit_platt, fit_temperature
from glance.train import R1Data

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def calib_decisions(data: R1Data) -> list[Decision]:
    out = []
    for line in open(data.root / "calib.jsonl"):
        r = json.loads(line)
        row = r.pop("image_row")
        d = Decision(**r, image_bytes=b"")
        d.meta = {**d.meta, "image_row": row}
        out.append(d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--name", default="qwen3vl2b")
    ap.add_argument("--sources", default="vqav2_train,gqa_train,aokvqa_train")
    ap.add_argument("--limit", type=int, default=0, help="label at most this many train questions (0 = all)")
    ap.add_argument("--reuse-from", help="another mix dir: copy its raw readouts for questions with the same uid, "
                    "text and options (so a rebuilt mix only relabels what changed)")
    a = ap.parse_args()
    data = R1Data(Path(a.data_root))
    sources = set(a.sources.split(","))
    train = [d for d in data.decisions("train") if d.source in sources and d.kind in ("noul", "choice")]
    calib = [d for d in calib_decisions(data) if d.source in sources and d.kind in ("noul", "choice")]
    if a.limit:
        train = train[:a.limit]
    raw_path = data.root / f"teacher_{a.name}.raw.jsonl"
    if a.reuse_from and not raw_path.exists():
        old = Path(a.reuse_from)
        was = {}
        for f in ("decisions.jsonl", "calib.jsonl"):
            for line in open(old / f):
                r = json.loads(line)
                was[r["uid"]] = (r["question"], r["candidates"])
        now = {d.uid: (d.question, d.candidates, split) for split, ds in (("train", train), ("calib", calib)) for d in ds}
        kept = 0
        with open(raw_path, "w") as fh:
            for line in open(old / f"teacher_{a.name}.raw.jsonl"):
                r = json.loads(line)
                if r["uid"] in now and was.get(r["uid"]) == now[r["uid"]][:2]:
                    q, c, split = now[r["uid"]]
                    fh.write(json.dumps(r | {"split": split, "q": q, "c": c}) + "\n")
                    kept += 1
        print(f"reused {kept} readouts from {old}", flush=True)
    done = set()
    if raw_path.exists():
        done = {json.loads(line)["uid"] for line in open(raw_path)}
    todo = [("calib", d) for d in calib if d.uid not in done] + [("train", d) for d in train if d.uid not in done]
    print(f"{len(train)} train + {len(calib)} calib questions; {len(done)} done, {len(todo)} to go", flush=True)
    todo.sort(key=lambda x: x[1].meta["image_row"])  # neighbouring reads
    reader = VLMReadout(a.model, DEVICE)
    t0 = time.time()
    with open(raw_path, "a") as fh:
        for i, (split, d) in enumerate(todo):
            d.image_bytes = data.image_bytes(d.meta["image_row"])
            out = reader.logits(d, debias=False)
            fh.write(json.dumps({"uid": d.uid, "split": split, "kind": d.kind, "logits": out["logits"],
                                 "mass": round(out["mass"], 4), "q": d.question, "c": d.candidates}) + "\n")
            d.image_bytes = b""
            if (i + 1) % 2000 == 0:
                fh.flush()
                rate = (i + 1) / (time.time() - t0)
                print(f"  {i + 1}/{len(todo)} ({rate:.1f}/s, {(len(todo) - i - 1) / rate / 3600:.1f} h left)", flush=True)

    raw = {}
    for line in open(raw_path):
        r = json.loads(line)
        raw[r["uid"]] = r
    gold = {d.uid: d.target for d in calib}
    fit = {}
    noul = [(raw[u]["logits"], gold[u]) for u in gold if u in raw and raw[u]["kind"] == "noul"]
    choice = [(raw[u]["logits"], gold[u]) for u in gold if u in raw and raw[u]["kind"] == "choice"]
    fit["noul"] = fit_platt([l for l, _ in noul], [y for _, y in noul])
    fit["choice"] = fit_temperature([l for l, _ in choice], [y for _, y in choice])
    print(f"calibration from {len(noul)} yes/no and {len(choice)} choice calib questions: {fit}", flush=True)
    with open(data.root / f"teacher_{a.name}.jsonl", "w") as fh:
        for d in train:
            z = np.array(raw[d.uid]["logits"], dtype=np.float64)
            if d.kind == "noul":
                pa, pb = fit["noul"]
                p1 = 1 / (1 + np.exp(-(pa * (z[0] - z[1]) + pb)))
                p = np.array([p1, 1 - p1])
            else:
                e = np.exp((z - z.max()) / fit["choice"])
                p = e / e.sum()
            fh.write(json.dumps({"uid": d.uid, "p": [round(float(x), 5) for x in p], "c": d.candidates}) + "\n")
    (data.root / f"teacher_{a.name}.calibration.json").write_text(json.dumps(
        {"model": a.model, "noul_platt": fit["noul"], "choice_temperature": fit["choice"],
         "n_calib_noul": len(noul), "n_calib_choice": len(choice), "sources": sorted(sources)}, indent=1))
    print(f"wrote teacher_{a.name}.jsonl ({len(train)} questions)", flush=True)


if __name__ == "__main__":
    main()
