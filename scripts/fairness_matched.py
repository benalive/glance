"""FairFace false alarms at a matched operating point, and threshold-free.

scripts/fairness_check.py reports each model at its own served threshold, but the release and the research
model run at different operating points (SalArt recall 45% vs 79%), so their FairFace rates are not directly
comparable. Here both answer "Does this image contain visible generation errors?" on FairFace's 10,954 real
portraits and on SalArt's 437 flawed generated images (HAD-derived rows dropped). Per race, gender and age group:
  - flagged at the served threshold (P >= 0.5);
  - flagged at the threshold where the model finds 45% of SalArt's flawed images with this question (matched);
  - AUROC of SalArt flawed images against that group's portraits (threshold-free).
Per-image probabilities: results/fairness/matched_items.jsonl.

    uv run python scripts/fairness_matched.py     # -> results/fairness/matched.{json,md}
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from glance.data.benchmarks import EXTERNAL
from glance.metrics import auroc
from glance.serve import Predictor

MODELS = {"released C5 (v4 seed 0)": ("weights/glance-c5-open-v4", "C5", None),
          "research C5 (distilled)": ("data/ckpt/C5_distill_ep2.pt", "C5", "data/ckpt/C5_distill_ep2_calib")}
QUESTION = "Does this image contain visible generation errors?"
RECALL = 0.45


def main():
    path = hf_hub_download("HuggingFaceM4/FairFace", "1.25/validation-00000-of-00001-09e3e67bb00ab4ec.parquet", repo_type="dataset")
    table = pq.read_table(path)
    meta = table.schema.metadata
    names = json.loads(meta[b"huggingface"])["info"]["features"] if meta and b"huggingface" in meta else {}
    label = {k: names.get(k, {}).get("names") for k in ("age", "gender", "race")}
    faces = table.to_pylist()
    flawed = []
    for line in open(EXTERNAL / "salart" / "data" / "test.jsonl"):
        r = json.loads(line)
        if r["image_role"] == "artifact" and not r["row_id"].startswith("hadm__"):
            flawed.append((r["row_id"], (EXTERNAL / "salart" / r["image_path"]).read_bytes()))
    q = {"q": {"type": "noul", "instructions": QUESTION}}
    out, items = {"question": QUESTION, "matched_recall": RECALL, "n_faces": len(faces), "n_flawed": len(flawed), "models": {}}, []
    for name, (ckpt, cand, calib) in MODELS.items():
        pred = Predictor(ckpt, cand, calib, "cpu", 4)

        def p_yes(img):
            a = pred.predict(q, img)["answers"]["q"]["noul"]
            pred.cache.clear()
            return a

        pf = np.array([p_yes(img) for _, img in flawed])
        pp = np.array([p_yes(r["image"]["bytes"]) for r in faces])
        items += [{"model": name, "set": "salart_flawed", "uid": u, "p_yes": float(x)} for (u, _), x in zip(flawed, pf)]
        items += [{"model": name, "set": "fairface", "i": i, "p_yes": float(x)} for i, x in enumerate(pp)]
        t = float(np.quantile(pf, 1 - RECALL))
        m = {"salart_recall_served": float(np.mean(pf >= 0.5)), "matched_threshold": t,
             "all": {"served": float(np.mean(pp >= 0.5)), "matched": float(np.mean(pp >= t)), "auroc": auroc(list(pf) + list(pp), [1] * len(pf) + [0] * len(pp))},
             "by": {}}
        for attr in ("race", "gender", "age"):
            groups = defaultdict(list)
            for j, r in enumerate(faces):
                groups[label[attr][r[attr]] if label[attr] else str(r[attr])].append(j)
            m["by"][attr] = {g: {"n": len(idx), "served": float(np.mean(pp[idx] >= 0.5)), "matched": float(np.mean(pp[idx] >= t)),
                                 "auroc": auroc(list(pf) + list(pp[idx]), [1] * len(pf) + [0] * len(idx))}
                             for g, idx in sorted(groups.items())}
        out["models"][name] = m
        print(name, round(m["salart_recall_served"], 3), round(t, 3), {k: round(v, 4) for k, v in m["all"].items()}, flush=True)
        pred = None
    dest = Path("results/fairness")
    (dest / "matched.json").write_text(json.dumps(out, indent=1))
    (dest / "matched_items.jsonl").write_text("".join(json.dumps(r) + "\n" for r in items))
    L = [f"FairFace ({len(faces)} real portraits) with \"{QUESTION}\": share flagged at the served threshold (P >= 0.5) "
         f"and at the threshold where the model finds {RECALL:.0%} of SalArt's {len(flawed)} flawed images (matched); AUROC "
         "of SalArt flawed images against each group's portraits.", ""]
    for name, m in out["models"].items():
        L += [f"### {name}: SalArt recall at 0.5 = {m['salart_recall_served']:.3f}; matched threshold P >= {m['matched_threshold']:.3f}", "",
              "| group | n | flagged (served) | flagged (matched) | AUROC flawed vs group |", "|---|---|---|---|---|",
              f"| all | {len(faces)} | {m['all']['served']:.2%} | {m['all']['matched']:.2%} | {m['all']['auroc']:.3f} |"]
        for attr, groups in m["by"].items():
            for g, r in groups.items():
                L.append(f"| {attr}: {g} | {r['n']} | {r['served']:.2%} | {r['matched']:.2%} | {r['auroc']:.3f} |")
        L.append("")
    (dest / "matched.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
