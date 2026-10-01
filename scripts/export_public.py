"""Build the public copy of this repository from the committed tree at HEAD: only what is needed to use the
released model and to trace the paper's numbers, as a fresh git repository with one commit.

Shipped: the code (glance/, scripts/, tests/, bench/), the paper, the model card, the top-level files, and the
summary result files the paper's tables come from (results/**/*.md and *.json, plus the paired intervals).
Not shipped: per-item model outputs, working notes, audits and local tooling.

    uv run python scripts/export_public.py ../glance-public
"""
import argparse
import io
import subprocess
import tarfile
from pathlib import Path

TOP = {".gitignore", "README.md", "LICENSE", "NOTICE", "CITATION.cff", "pyproject.toml", "uv.lock", "docs/model_card.md"}
DIRS = ("glance/", "scripts/", "tests/", "bench/", "paper/", "docs/assets/")
RESULTS = {"results/paper_intervals.jsonl"}  # a summary despite its extension


def shipped(name: str) -> bool:
    if name in TOP or name in RESULTS or name.startswith(DIRS):
        return True
    return name.startswith("results/") and name.endswith((".md", ".json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dest", type=Path)
    a = ap.parse_args()
    if a.dest.exists():
        raise SystemExit(f"{a.dest} exists; remove it first")
    tar = subprocess.run(["git", "archive", "--format=tar", "HEAD"], capture_output=True, check=True).stdout
    kept = skipped = 0
    with tarfile.open(fileobj=io.BytesIO(tar)) as t:
        for m in t.getmembers():
            if not m.isfile():
                continue
            if not shipped(m.name):
                skipped += 1
                continue
            dest = a.dest / m.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(t.extractfile(m).read())
            if m.mode & 0o111:
                dest.chmod(0o755)
            kept += 1
    print(f"{kept} files shipped, {skipped} left out")
    run = lambda *c: subprocess.run(c, cwd=a.dest, check=True)  # noqa: E731
    run("git", "init", "-q", "-b", "main")
    run("git", "add", "-A")
    run("git", "commit", "-q", "-m", "Glance: code, paper, model card and summary results for the open-licence release")
    print(f"initialised {a.dest} with one commit")


if __name__ == "__main__":
    main()
