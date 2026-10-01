"""R1 tables: every trained run next to the zero-shot floors, with latency.

Rules (R0 eval audit):
- same calibration family for every method: temperature on choice/score, Platt on yes/no (POPE),
  fitted on the calib split, reported on the report split;
- in-domain (POPE in-format, A-OKVQA, KonIQ: trained candidates saw these task formats) and
  out-of-domain (MMStar, SugarCrepe) are separate blocks;
- KonIQ is ranked by SRCC from raw logits; no KonIQ accuracy;
- differences between runs use a paired, image-grouped bootstrap on common items;
- latency is the R0 protocol on this Mac (p50, one question); L4 figures are a compiled best-case
  projection, not a measurement.
"""
import json
import re
from pathlib import Path

import numpy as np

from glance.metrics import paired_bootstrap, softmax, summarize_run_file

RUN = re.compile(r"(?P<cand>C\d\w*)_(?P<budget>ep[\d.]+|[\de\-+]+)_(?P<lr>lr[A-Z])_s(?P<seed>\d+)(?:_(?P<sanity>\w+))?$")
LATENCY_NAME = {"C1": "C1 ModernVBERT @512px", "C2": "C2 ModernVBERT text 11L @256px",
                "C3": "C3/C4 single tower 12x768 (B/32 shape)", "C4": "C3/C4 single tower 12x768 (B/32 shape)",
                "C5": "C5 fusion (B/32 + Ettin-32M + 3 fusion)", "C6": "C6 tiny tower 6x512",
                "C0d": "C0 SigLIP2-B/32 dual encoder"}
IN_DOMAIN = ["pope_adversarial", "aokvqa_val", "koniq_test"]
OUT_DOMAIN = ["mmstar", "sugarcrepe_replace_rel", "sugarcrepe_swap_att"]
SHORT = {"pope_adversarial": "POPE*", "aokvqa_val": "A-OKVQA", "koniq_test": "KonIQ SRCC", "mmstar": "MMStar",
         "sugarcrepe_replace_rel": "SC-rel", "sugarcrepe_swap_att": "SC-att"}


def latency_table():
    out = {}
    for f in ["results/r0/latency_sweep.json", "results/r0/latency_extra.json"]:
        if not Path(f).exists():
            continue
        d = json.loads(Path(f).read_text())
        rows = d if isinstance(d, list) else [dict(r, device=dev) for dev in ["mps", "cpu"] for r in d[dev]["pipelines"]]
        for r in rows:
            if r["n_questions"] == 1:
                out[(r["name"], r["device"], r["mode"])] = r["p50_ms"]
    return out


def pick(rows):
    """Platt row on yes/no benchmarks, the single-pass/temperature row otherwise."""
    platt = [r for r in rows if r["variant"].endswith("+platt")]
    return (platt or [rows[0]])[0]


def correctness(path: Path, row: dict) -> dict:
    """uid -> (correct under the chosen calibration, image_id), for paired comparisons."""
    recs = [json.loads(l) for l in path.open() if json.loads(l)["split"] == "report"]
    v = row["variant"].removesuffix("+platt")
    out = {}
    for r in recs:
        if "platt" in row:
            a, b = row["platt"]
            p = np.array([1 / (1 + np.exp(-(a * (r[v][0] - r[v][1]) + b))), 0.0])
            p[1] = 1 - p[0]
        else:
            p = softmax(np.array(r[v]), row["scaled"]["temperature"])
        out[r["uid"]] = (float(int(p.argmax()) == int(np.argmax(r["target"]))), r["image_id"])
    return out


def cell(bench, s):
    return f"{s['srcc_raw']:.3f}" if bench == "koniq_test" else f"{s['acc']:.3f} ({s['smece']:.3f})"


def main():
    lat = latency_table()
    entries = []  # (label, cand, {bench: (stats, path, row)}, meta)
    for run in sorted(Path("results/r1").iterdir()):
        m = RUN.match(run.name)
        if not m or not (run / "train_summary.json").exists():
            continue
        summ = json.loads((run / "train_summary.json").read_text())
        benches = {}
        for f in sorted((run / "eval").glob("*.jsonl")):
            rows = summarize_run_file(f)
            row = pick(rows)
            stats = dict(row["scaled"], srcc_raw=rows[0]["raw"].get("srcc", float("nan")))
            benches[f.stem.split("__")[1]] = (stats, f, row)
        g = m.groupdict()
        label = f"{g['cand']} {g['budget']} {g['lr']} s{g['seed']}" + (f" [{g['sanity']}]" if g["sanity"] else "")
        entries.append((label, g["cand"], benches, {"dev": summ["dev"], "epochs": summ["epochs"],
                                                     "pflop": summ["train_flops"] / 1e15, "wall_h": summ["wall_s"] / 3600}))
    lines = ["Accuracy (smECE after calibration); KonIQ = SRCC from raw logits. *POPE is in-format for trained runs "
             "(training used its question template on disjoint images). Zero-shot floors: results/r0/frontier_summary.md.", ""]
    for title, block in [("In-domain", IN_DOMAIN), ("Out-of-domain", OUT_DOMAIN)]:
        lines += [f"**{title}**", "", "| run | dev NLL | dev acc | epochs | PFLOP | wall h | " + " | ".join(SHORT[b] for b in block)
                  + " | MPS cold/cached ms | CPU cold ms |", "|" + "---|" * (6 + len(block) + 2)]
        for label, cand, benches, meta in entries:
            cells = [cell(b, benches[b][0]) if b in benches else "—" for b in block]
            name = LATENCY_NAME.get(cand, "")
            L = [lat.get((name, d, mo)) for d, mo in [("mps", "cold"), ("mps", "cached"), ("cpu", "cold")]]
            fmt = lambda x: "—" if x is None else f"{x:.1f}"  # noqa: E731
            lines.append(f"| {label} | {meta['dev']['nll']:.3f} | {meta['dev']['acc']:.3f} | {meta['epochs']:.2f} "
                         f"| {meta['pflop']:.1f} | {meta['wall_h']:.2f} | " + " | ".join(cells)
                         + f" | {fmt(L[0])}/{fmt(L[1])} | {fmt(L[2])} |")
        lines.append("")
    # paired comparisons on common items between the best-dev run of each candidate
    best = {}
    for label, cand, benches, meta in entries:
        if "[" in label:
            continue
        if cand not in best or meta["dev"]["nll"] < best[cand][2]["dev"]["nll"]:
            best[cand] = (label, benches, meta)
    lines += ["**Paired differences (accuracy, image-grouped bootstrap 95% CI), best-dev run per candidate vs C3**", "",
              "| comparison | " + " | ".join(SHORT[b] for b in IN_DOMAIN + OUT_DOMAIN if b != "koniq_test") + " |",
              "|" + "---|" * 6]
    if "C3" in best:
        ref_label, ref_b, _ = best["C3"]
        for cand, (label, benches, _) in sorted(best.items()):
            if cand == "C3":
                continue
            cells = []
            for b in [x for x in IN_DOMAIN + OUT_DOMAIN if x != "koniq_test"]:
                if b in benches and b in ref_b:
                    ca, cb = correctness(benches[b][1], benches[b][2]), correctness(ref_b[b][1], ref_b[b][2])
                    common = sorted(set(ca) & set(cb))
                    pb = paired_bootstrap([ca[u][0] for u in common], [cb[u][0] for u in common], [ca[u][1] for u in common])
                    cells.append(f"{pb['diff'] * 100:+.1f} [{pb['ci95'][0] * 100:+.1f}, {pb['ci95'][1] * 100:+.1f}]")
                else:
                    cells.append("—")
            lines.append(f"| {label} − {ref_label} | " + " | ".join(cells) + " |")
    Path("results/r1/summary.md").write_text("\n".join(lines) + "\n")
    Path("results/r1/summary.json").write_text(json.dumps(
        [{"run": l, "cand": c, "meta": m, "bench": {b: v[0] for b, v in bs.items()}} for l, c, bs, m in entries], indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
