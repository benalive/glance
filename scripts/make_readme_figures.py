"""README hero chart: latency per new image, Glance against other image-question models, as static SVGs for
GitHub's light and dark themes (docs/assets/latency-{light,dark}.svg). Values are the ones in the README's
speed table (results/latency/, results/comparison.md); bars for other models are reference points measured
under their own conditions or published.

    uv run python scripts/make_readme_figures.py
"""
from pathlib import Path

ROWS = [  # label, detail, ms per new image (one question), shown value, is Glance
    ("Glance", "Apple M4 Pro CPU", 18.3, "18 ms", True),
    ("Glance", "AMD EPYC 9V45 CPU (x86)", 26.2, "26 ms", True),
    ("laya-vision", "NVIDIA L4 GPU, published", 32.0, "32–41 ms", False),
    ("SmolVLM-256M", "Apple M4 Pro GPU", 55.3, "55 ms", False),
    ("Qwen3-VL-2B", "Apple M4 Pro GPU", 135.0, "135 ms", False),
    ("SmolVLM-256M", "Apple M4 Pro CPU", 183.0, "183 ms", False),
]
THEMES = {
    "light": {"text": "#1f2328", "muted": "#59636e", "grid": "#d1d9e0", "glance": "#2a78d6", "other": "#8f8d87"},
    "dark": {"text": "#f0f6fc", "muted": "#9198a1", "grid": "#3d444d", "glance": "#3987e5", "other": "#6e6d68"},
}
W, LEFT, RIGHT, TOP, ROW, BAR = 860, 250, 70, 64, 40, 18
XMAX = 200.0


def svg(theme: str) -> str:
    c = THEMES[theme]
    plot_w = W - LEFT - RIGHT
    h = TOP + ROW * len(ROWS) + 46
    x = lambda v: LEFT + plot_w * v / XMAX  # noqa: E731
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{h}" viewBox="0 0 {W} {h}" '
           f'font-family="-apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif" role="img" '
           f'aria-label="Latency per new image: Glance 18 ms on an Apple M4 Pro CPU and 26 ms on an x86 CPU; '
           f'other image-question models 32 to 183 ms">',
           f'<text x="0" y="22" font-size="17" font-weight="600" fill="{c["text"]}">Milliseconds per new image, one question, batch 1</text>',
           f'<text x="0" y="44" font-size="13" fill="{c["muted"]}">Lower is faster. Glance answers more questions about a cached image in 3–4 ms each.</text>']
    for v in (0, 50, 100, 150, 200):
        out.append(f'<line x1="{x(v):.1f}" y1="{TOP - 6}" x2="{x(v):.1f}" y2="{TOP + ROW * len(ROWS)}" stroke="{c["grid"]}" stroke-width="1"/>')
        out.append(f'<text x="{x(v):.1f}" y="{TOP + ROW * len(ROWS) + 18}" font-size="12" fill="{c["muted"]}" text-anchor="middle">{v}</text>')
    for i, (name, detail, ms, shown, ours) in enumerate(ROWS):
        y = TOP + i * ROW + (ROW - BAR) / 2
        weight = "700" if ours else "500"
        out.append(f'<text x="{LEFT - 12}" y="{y + 8}" font-size="14" font-weight="{weight}" fill="{c["text"]}" text-anchor="end">{name}</text>')
        out.append(f'<text x="{LEFT - 12}" y="{y + 23}" font-size="11.5" fill="{c["muted"]}" text-anchor="end">{detail}</text>')
        bw = x(ms) - LEFT
        fill = c["glance"] if ours else c["other"]
        # bar anchored at the baseline, 4 px rounding only on the data end
        out.append(f'<path d="M{LEFT},{y} h{bw - 4:.1f} a4,4 0 0 1 4,4 v{BAR - 8} a4,4 0 0 1 -4,4 h{-(bw - 4):.1f} z" fill="{fill}"/>')
        out.append(f'<text x="{x(ms) + 8:.1f}" y="{y + 13.5}" font-size="13.5" font-weight="{weight}" fill="{c["text"]}">{shown}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main():
    dest = Path("docs/assets")
    dest.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        (dest / f"latency-{theme}.svg").write_text(svg(theme))
    print("wrote", ", ".join(str(dest / f"latency-{t}.svg") for t in THEMES))


if __name__ == "__main__":
    main()
