"""End-to-end check of the playground page against a running server, in headless Chrome.

Does what a person does: loads the page, uploads an image, adds an example question, asks, and
then checks what is on screen: one result per question, every probability bar drawn at the width
its percentage says, the timing line, and the warning shown when a question is typed into the
context box instead of the question box. Saves screenshots for a visual look.

    uv run python -m glance.serve --ckpt data/ckpt/C5_ep2_lrA_s0.pt        # in another terminal
    uv run --with playwright python scripts/ui_check.py --image photo.jpg   # uses the installed Chrome
"""
import argparse
import sys
from pathlib import Path


def main():
    from playwright.sync_api import sync_playwright

    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8089")
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="results/ui_check")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    failures = []

    def check(ok, what):
        print(("PASS " if ok else "FAIL ") + what, flush=True)
        if not ok:
            failures.append(what)

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 1000})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(a.url)
        page.wait_for_function("document.getElementById('status').textContent !== 'connecting…'")
        status = page.inner_text("#status")
        check("calibrated" in status and "unreachable" not in status, f"status pill shows a calibrated model ({status!r})")

        page.set_input_files("#file", a.image)
        page.wait_for_selector("#drop img")
        check(page.is_visible("#drop img"), "uploaded image is previewed")

        page.click("#examples button:has-text('Is there a person?')")
        page.click("#examples button:has-text('Which room?')")
        n_q = page.locator(".q").count()
        page.click("#ask")
        page.wait_for_function(f"document.querySelectorAll('.result').length === {n_q}", timeout=30000)
        check(page.locator(".result").count() == n_q, f"one result per question ({n_q})")

        bars = page.eval_on_selector_all(".bar", """els => els.map(b => ({
            pct: parseFloat(b.querySelector('.pct').textContent),
            fill: b.querySelector('.fill').getBoundingClientRect().width,
            track: b.querySelector('.track').getBoundingClientRect().width}))""")
        bad = [b for b in bars if abs(b["fill"] - b["track"] * b["pct"] / 100) > 3]
        check(bars and not bad, f"all {len(bars)} bars drawn at their percentage (off by >3 px: {len(bad)})")
        check(all(b["track"] > 50 for b in bars), "bar tracks have a visible width")
        timing = page.inner_text("#timing")
        check("ms" in timing, f"timing line shown ({timing.strip()[:60]}...)")
        page.screenshot(path=str(out / "results.png"), full_page=True)

        # A question typed into the context box, with a non-question in the question box, must be flagged.
        page.click("#ctxBox summary")
        page.fill("#state", "are there humans in the image?")
        page.locator(".q input[data-field='text']").first.fill("hallucination")
        page.click("#ask")
        page.wait_for_selector("#notice:not([hidden])", timeout=10000)
        check(page.is_visible("#notice"), "misplaced-question warning shown")
        page.screenshot(path=str(out / "misplaced_question.png"), full_page=True)

        check(not errors, f"no JavaScript errors ({errors[:1]})")
        browser.close()
    print(f"{'OK' if not failures else 'FAILED'}: screenshots in {out}/", flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
