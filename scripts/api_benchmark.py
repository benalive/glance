"""Run benchmark items through a running glance server, the way users' requests arrive, and score them.

Two purposes:
  1. Serving parity. Every R0 benchmark goes through `POST /v1/systemone`; accuracy (KonIQ: SRCC)
     and per-item top answers are compared with the offline evaluation of the same checkpoint, with
     the same calibration. A gap means the served path differs from training/eval (image decoding,
     resizing, packing, calibration, device precision).
  2. Everyday questions. POPE random/popular, and VQAv2 validation yes/no questions on images that
     never appear in the training mix, broken down by question phrasing: VQAv2's question types,
     questions about people ("man", "woman", "child"...), and "Is there a X?" with X inside vs
     outside COCO's 80 object names (the only nouns in the yes/no training questions).

    uv run python -m glance.serve --ckpt data/ckpt/C5_ep2_lrA_s0.pt        # in another terminal
    uv run python scripts/api_benchmark.py --offline results/r1/C5_ep2_lrA_s0/eval
    # -> results/api_benchmark/<run>/{summary.md, summary.json, items.jsonl}
"""
import argparse
import base64
import json
import re
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from glance.data.benchmarks import ALL_BENCHMARKS
from glance.data.schema import Decision, calib_split
from glance.serve import Predictor, calibrated_probs

PEOPLE = re.compile(r"\b(man|men|woman|women|boy|boys|girl|girls|child|children|kid|kids|person|people|guy|guys|"
                    r"lady|ladies|player|players|baby|adult|adults|someone|anyone|he|she)\b", re.I)
IS_THERE = re.compile(r"^(is|are) there (a |an |any )?(.+?)( in (the|this) (image|picture|photo))?\??$", re.I)
COCO = set(json.loads((Path("glance/data/coco_names.json")).read_text())["names"])


BENCHES = ALL_BENCHMARKS


def spec(d: Decision) -> dict:
    if d.kind == "noul":
        return {"type": "noul", "instructions": d.question}
    if d.kind == "choice":
        return {"type": "choice", "instructions": d.question, "criteria": {c: "" for c in d.candidates}}
    return {"type": "score", "instructions": d.question, "criteria": list(d.candidates)}


def probs_of(d: Decision, ans: dict) -> list[float]:
    p = ans["probabilities"]
    if d.kind == "score":
        return [p[str(i)] for i in range(len(d.candidates))]
    return [p[c] for c in d.candidates]


def run_bench(url: str, decisions: list[Decision]):
    by_img = defaultdict(list)
    for d in decisions:
        by_img[d.image_id].append(d)
    served, timings = {}, []
    for items in by_img.values():
        body = {"image": base64.b64encode(items[0].image_bytes).decode(),
                "questions": {str(i): spec(d) for i, d in enumerate(items)}}
        req = urllib.request.Request(f"{url}/v1/systemone", json.dumps(body).encode(), {"Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(req, timeout=120).read())
        timings.append((len(items), r["image_cached"], r["timing_ms"]["total"]))
        for i, d in enumerate(items):
            served[d.uid] = probs_of(d, r["answers"][str(i)])
    return served, timings


def score(decisions, probs):
    """Accuracy on the report split (KonIQ: SRCC of the expected level against MOS, all items)."""
    if decisions[0].kind == "score":
        exp = [float(np.dot(np.arange(len(probs[d.uid])), probs[d.uid])) for d in decisions]
        return float(spearmanr(exp, [d.meta["mos"] for d in decisions]).statistic)
    rep = [d for d in decisions if calib_split(d.image_id) == "report"] or decisions
    return float(np.mean([int(np.argmax(probs[d.uid])) == d.label for d in rep]))


def breakdown(decisions, probs):
    groups = defaultdict(list)
    for d in decisions:
        groups[f"type: {d.meta['question_type']}"].append(d)
        if PEOPLE.search(d.question):
            groups["about people (man, woman, child, ...)"].append(d)
        m = IS_THERE.match(d.question.strip())
        if m:
            noun = m.group(3).strip().lower()
            groups["'Is there a X', X a COCO name" if noun in COCO or noun.rstrip("s") in COCO
                   else "'Is there a X', X not a COCO name"].append(d)
    rows = []
    for g, ds in groups.items():
        yes = [probs[d.uid][0] for d in ds if d.label == 0]
        no = [probs[d.uid][0] for d in ds if d.label == 1]
        rows.append({"group": g, "n": len(ds), "acc": float(np.mean([int(np.argmax(probs[d.uid])) == d.label for d in ds])),
                     "gold_yes_share": len(yes) / len(ds), "p_yes_when_yes": float(np.mean(yes)) if yes else None,
                     "p_yes_when_no": float(np.mean(no)) if no else None})
    return sorted(rows, key=lambda r: -r["n"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8089")
    ap.add_argument("--offline", help="offline eval dir of the served checkpoint, e.g. results/r1/C5_ep2_lrA_s0/eval")
    ap.add_argument("--benches", default=",".join(BENCHES))
    ap.add_argument("--package", help="release package the server runs: its calibration.json is used for the offline side")
    a = ap.parse_args()
    health = json.loads(urllib.request.urlopen(f"{a.url}/health").read())
    run = Path(a.offline).parent.name if a.offline else health["checkpoint"].removesuffix(".pt")
    out = Path("results/api_benchmark") / run
    out.mkdir(parents=True, exist_ok=True)
    if a.package:
        calib = json.loads((Path(a.package) / "calibration.json").read_text())["calibration"]
    else:
        calib = Predictor._fit_calibration(a.offline) if a.offline else {}
    summary, all_timings = {"server": health, "benches": {}}, []
    with open(out / "items.jsonl", "w") as items_fh:
        for bench in a.benches.split(","):
            decisions = BENCHES[bench]()
            t0 = time.time()
            served, timings = run_bench(a.url, decisions)
            all_timings += timings
            row = {"n": len(decisions), "served": score(decisions, served), "wall_s": round(time.time() - t0, 1)}
            offline_file = Path(a.offline or "-") / f"{health['model'].split('-')[-1]}__{bench}.jsonl"
            if offline_file.exists():
                logits = {json.loads(l)["uid"]: json.loads(l)["logits"] for l in offline_file.open()}
                off = {d.uid: list(calibrated_probs(calib, d.kind, np.array(logits[d.uid]))) for d in decisions}
                row["offline"] = score(decisions, off)
                row["top_answer_agreement"] = float(np.mean([np.argmax(served[d.uid]) == np.argmax(off[d.uid]) for d in decisions]))
                row["max_abs_prob_diff"] = float(max(np.max(np.abs(np.array(served[d.uid]) - np.array(off[d.uid]))) for d in decisions))
            if bench == "vqav2_val_yesno":
                row["breakdown"] = breakdown(decisions, served)
            summary["benches"][bench] = row
            print(bench, {k: v for k, v in row.items() if k != "breakdown"}, flush=True)
            for d in decisions:
                items_fh.write(json.dumps({"bench": bench, "uid": d.uid, "label": d.label, "probs": [round(p, 5) for p in served[d.uid]]}) + "\n")
    lat = {"new_image": [t for _, c, t in all_timings if not c], "cached": [t for _, c, t in all_timings if c]}
    summary["latency_ms"] = {k: {"n": len(v), "p50": float(np.median(v)), "p95": float(np.percentile(v, 95))} for k, v in lat.items() if v}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    lines = [f"Server: {health['model']} ({health['checkpoint']}) on {health['device']}, calibrated: {', '.join(health['calibrated'])}. "
             "Metric: accuracy on the report split; KonIQ: SRCC of the expected level vs MOS. Offline = the same checkpoint's "
             "evaluation file with the same calibration.", "",
             "| benchmark | n | served | offline | top-answer agreement | max abs prob diff |", "|---|---|---|---|---|---|"]
    for b, r in summary["benches"].items():
        f = lambda k, fmt="{:.3f}": fmt.format(r[k]) if k in r else "—"
        lines.append(f"| {b} | {r['n']} | {r['served']:.3f} | {f('offline')} | {f('top_answer_agreement')} | {f('max_abs_prob_diff', '{:.4f}')} |")
    if "vqav2_val_yesno" in summary["benches"]:
        lines += ["", "VQAv2 validation yes/no (images not in training), by phrasing; groups with n >= 40:", "",
                  "| group | n | accuracy | share gold yes | mean P(yes), gold yes | mean P(yes), gold no |", "|---|---|---|---|---|---|"]
        for r in summary["benches"]["vqav2_val_yesno"]["breakdown"]:
            if r["n"] >= 40:
                g = lambda k: f"{r[k]:.2f}" if r[k] is not None else "—"
                lines.append(f"| {r['group']} | {r['n']} | {r['acc']:.3f} | {r['gold_yes_share']:.2f} | {g('p_yes_when_yes')} | {g('p_yes_when_no')} |")
    lines += ["", "Latency per request (ms): " + "; ".join(f"{k} p50 {v['p50']:.1f} / p95 {v['p95']:.1f} (n={v['n']})"
                                                     for k, v in summary["latency_ms"].items())]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
