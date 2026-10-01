"""Does training on procedural local anomalies (scripts/make_local_anomalies.py) teach flaw detection that
transfers, or only edit detection?

Per checkpoint (raw P(yes) of the "visible generation errors" question, no calibration; AUROC is threshold-free):
  held-out anomaly pairs   flawed vs its control, all and per operation (learned the task?)
  edit confound            control vs the untouched original (should stay near 0.5: an edit alone is not a flaw)
  SalArt pairs             inpainted flaw vs clean original (transfer to real generator flaws)
  SalArt                   flawed vs real photos, flawed vs clean generated
Paired image-grouped bootstrap intervals (2,000 resamples) for the pair AUROCs.

    uv run python scripts/anomaly_eval.py data/ckpt/C5_open4_s0.pt data/ckpt/C5_open5_s0.pt   # -> results/anomaly_eval.{json,md}
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import io

from glance.data.benchmarks import EXTERNAL, SALART_Q
from glance.metrics import auroc
from glance.serve import Predictor

Q = "Does this image contain visible generation errors?"


def pair_auroc(pairs, rng_seed=0, n=2000):
    a, b = [x for x, _ in pairs], [y for _, y in pairs]
    val = auroc(a + b, [1] * len(a) + [0] * len(b))
    rng = np.random.default_rng(rng_seed)
    boots = []
    for _ in range(n):
        idx = rng.integers(0, len(pairs), len(pairs))
        boots.append(auroc([a[i] for i in idx] + [b[i] for i in idx], [1] * len(idx) + [0] * len(idx)))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"auroc": val, "ci95": [float(lo), float(hi)], "n_pairs": len(pairs)}


def resized_original(path: str) -> bytes:
    img = Image.open(path).convert("RGB")
    s = 512 / max(img.size)
    img = img.resize((max(64, int(img.width * s)), max(64, int(img.height * s))), Image.Resampling.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=95)
    return buf.getvalue()


def main():
    test = [json.loads(l) for l in open("data/clean/anomalies/test.jsonl")]
    src_path = {}
    for f in ("data/clean/labels_evalmuse.jsonl", "data/clean/labels_imagereward.jsonl"):
        for line in open(f):
            r = json.loads(line)
            src_path[r["image_id"]] = r["path"]
    sal = {}
    for line in open(EXTERNAL / "salart" / "data" / "test.jsonl"):
        r = json.loads(line)
        if not r["row_id"].startswith("hadm__"):
            sal[r["row_id"]] = r
    out = {}
    for ckpt in sys.argv[1:]:
        pred = Predictor(ckpt, "C5", None, "cpu", 4)

        def p(data: bytes, q: str = Q) -> float:
            a = pred.predict({"q": {"type": "noul", "instructions": q}}, data)["answers"]["q"]["noul"]
            pred.cache.clear()
            return a

        rows = [(r["op"], p(Path(r["flawed_path"]).read_bytes()), p(Path(r["control_path"]).read_bytes()),
                 p(resized_original(src_path[r["source_id"]]))) for r in test]
        res = {"anomaly_pairs": pair_auroc([(f, c) for _, f, c, _ in rows]),
               "anomaly_pairs_by_op": {op: auroc([f for o, f, _, _ in rows if o == op] + [c for o, _, c, _ in rows if o == op],
                                                 [1] * sum(o == op for o, *_ in rows) + [0] * sum(o == op for o, *_ in rows))
                                       for op in sorted({o for o, *_ in rows})},
               "edit_confound_control_vs_original": pair_auroc([(c, o) for _, _, c, o in rows])}
        sp = {k: p((EXTERNAL / "salart" / r["image_path"]).read_bytes(), SALART_Q[1]) for k, r in sal.items()}
        role = {k: r["image_role"] for k, r in sal.items()}
        pairs = {}
        for k, r in sal.items():
            if k.startswith("artifact_injection"):
                pairs.setdefault(r["pair_id"], {})[r["image_role"]] = sp[k]
        res["salart_pairs"] = pair_auroc([(v["artifact"], v["paired_generated_counterpart"]) for v in pairs.values() if len(v) == 2])
        fl = [sp[k] for k in sp if role[k] == "artifact"]
        for name, neg in (("salart_flawed_vs_real", "clean_reference"), ("salart_flawed_vs_clean_ai", "paired_generated_counterpart")):
            ng = [sp[k] for k in sp if role[k] == neg]
            res[name] = auroc(fl + ng, [1] * len(fl) + [0] * len(ng))
        out[Path(ckpt).stem] = res
        print(Path(ckpt).stem, json.dumps(res), flush=True)
        pred = None
    Path("results/anomaly_eval.json").write_text(json.dumps(out, indent=1))
    fmt = lambda r: f"{r['auroc']:.3f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]"  # noqa: E731
    L = ["Procedural local anomalies: held-out pairs, edit confound, and transfer to SalArt (raw P(yes), AUROC).", "",
         "| checkpoint | anomaly pairs (flawed vs control) | by op | control vs original | SalArt pairs | SalArt flawed vs real | SalArt flawed vs clean AI |",
         "|---|---|---|---|---|---|---|"]
    for k, r in out.items():
        ops = ", ".join(f"{o} {v:.2f}" for o, v in r["anomaly_pairs_by_op"].items())
        L.append(f"| {k} | {fmt(r['anomaly_pairs'])} | {ops} | {fmt(r['edit_confound_control_vs_original'])} | {fmt(r['salart_pairs'])} | "
                 f"{r['salart_flawed_vs_real']:.3f} | {r['salart_flawed_vs_clean_ai']:.3f} |")
    Path("results/anomaly_eval.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
