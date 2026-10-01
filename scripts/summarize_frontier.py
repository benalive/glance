"""Metrics for every results/r0/frontier/<method>__<bench>.jsonl, raw and temperature-scaled.

Temperature is fitted on the image-grouped 'calib' split only; every reported metric is on the
'report' split. Writes results/r0/frontier_summary.json and a markdown table.
"""
import argparse
import json
from pathlib import Path

from glance.metrics import summarize_run_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/r0/frontier")
    ap.add_argument("--out", default="results/r0/frontier_summary")
    args = ap.parse_args()
    rows = []
    for path in sorted(Path(args.dir).glob("*.jsonl")):
        method, bench = path.stem.split("__")
        file_rows = summarize_run_file(path)
        # One construction per (method, bench), chosen by calib NLL (SigLIP has several); the
        # rotation-averaged and Platt rows are reported alongside, labelled.
        base = [r for r in file_rows if r["variant"] != "logits_debiased" and not r["variant"].endswith("+platt")]
        chosen = min(base, key=lambda r: r["calib_nll"])["variant"]
        for r in file_rows:
            root = r["variant"].removesuffix("+platt")
            r["chosen"] = root in (chosen, "logits_debiased")
            rows.append({"method": method, "bench": bench, **r})
    Path(args.out + ".json").write_text(json.dumps(rows, indent=1))
    lines = ["Temperature (or Platt, yes/no only) fitted on the 20% image-grouped calib split; everything else on "
             "the report split. KonIQ accuracy is omitted (constant 'fair' scores 0.511); KonIQ SRCC is from raw "
             "logits. 'rotation Kx' = probabilities averaged over K option rotations (K forward passes).", "",
             "| method | bench | readout | n | acc [95% CI] | AUROC | smECE raw | smECE cal | Brier cal | NLL cal | T | SRCC (raw) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if not r["chosen"]:
            continue
        s, a = r["scaled"], r["raw"]
        lo, hi = s["acc_ci95"]
        v = r["variant"].replace("logits_debiased", "rotation Kx").replace("logits_", "").replace("logits", "single pass")
        acc = "—" if r["bench"] == "koniq_test" else f"{s['acc']:.3f} [{lo:.3f}, {hi:.3f}]"
        lines.append(f"| {r['method']} | {r['bench']} | {v} | {s['n']} | {acc} | {s.get('auroc', float('nan')):.3f} "
                     f"| {a['smece']:.3f} | {s['smece']:.3f} | {s['brier']:.3f} | {s['nll']:.3f} "
                     f"| {s['temperature']:.2f} | {a.get('srcc', float('nan')):.3f} |")
    Path(args.out + ".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
