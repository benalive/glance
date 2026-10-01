"""Tables from results/r0/latency_sweep.json: per-candidate latency and the per-device fits."""
import json
import sys
from pathlib import Path


def main(path="results/r0/latency_sweep.json", out="results/r0/latency_summary.md"):
    d = json.loads(Path(path).read_text())
    lines = [f"Source: `{path}` (commit {d['meta']['commit']}, {d['meta']['date']}). {d['meta']['protocol']}. "
             f"MPS fp16; CPU fp32 on {d['meta']['cpu_threads']} threads. L4 = analytic projection, not a measurement.", ""]
    for dev in ["mps", "cpu"]:
        f, o = d[dev]["fit"], d[dev]["overhead"]
        lines.append(f"**{dev} fit** (plain encoder grid): ms = {f['c_ms']:.2f} + {f['a_ms_per_layer']:.3f}*layers + "
                     f"{f['b_ms_per_gflop']:.3f}*GFLOPs; R^2 {f['r2']:.3f}, MAPE {f['mape']:.1%}, implied "
                     f"{f['implied_tflops'] or float('nan'):.2f} TFLOPS; overhead probe {o['us_per_layer']:.0f} us/layer.")
    lines += ["", "| candidate | params (M) | q | depth cold/cached | MPS cold p50 | MPS cached p50 | CPU cold p50 | CPU cached p50 | GFLOPs cold/cached | L4 proj cold/cached |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    rows = {}
    for dev in ["mps", "cpu"]:
        for r in d[dev]["pipelines"]:
            rows.setdefault((r["name"], r["n_questions"]), {})[(dev, r["mode"])] = r
    for (name, q), m in rows.items():
        mc, mk = m[("mps", "cold")], m[("mps", "cached")]
        cc, ck = m.get(("cpu", "cold")), m.get(("cpu", "cached"))
        cpu = (f"{cc['p50_ms']:.1f}", f"{ck['p50_ms']:.1f}") if cc else ("—", "—")
        lines.append(f"| {name} | {mc['params_m']:.0f} | {q} | {mc['depth']}/{mk['depth']} | {mc['p50_ms']:.1f} | {mk['p50_ms']:.1f} "
                     f"| {cpu[0]} | {cpu[1]} | {mc['gflops']:.0f}/{mk['gflops']:.0f} "
                     f"| {mc['l4_projection_ms']:.2f}/{mk['l4_projection_ms']:.2f} |")
    lines += ["", "| stem (256 px, 12 layers) | tokens | GFLOPs | MPS p50 | CPU p50 |", "|---|---|---|---|---|"]
    for a, b in zip(d["mps"]["stems"], d["cpu"]["stems"]):
        lines.append(f"| {a['stem']} | {a['tokens']} | {a['gflops']:.1f} | {a['p50_ms']:.2f} | {b['p50_ms']:.2f} |")
    Path(out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:])
