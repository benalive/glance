"""Download the released Glance weights from this repository's GitHub release and check their SHA-256 sums.

    python scripts/download_weights.py          # release package -> data/release/glance-c5-open-v4
    python scripts/download_weights.py --all    # plus the trained checkpoints -> data/ckpt/*.safetensors

Standard library only, so it runs before `uv sync`. Files already present with the right checksum are skipped.
"""
import argparse
import hashlib
import sys
import tarfile
import urllib.request
from pathlib import Path

REPO = "benalive/glance"  # GitHub owner/name of this repository
TAG = "weights-v4"
PACKAGE = "glance-c5-open-v4.tar"
CHECKPOINTS = ["C5_open4_s0", "C5_open4_s1", "C5_open4_s2", "C5_general_v2_s0", "C5_general_v2_s1", "C5_general_v2_s2",
               "C5_open5_s0"]


def url(name: str) -> str:
    return f"https://github.com/{REPO}/releases/download/{TAG}/{name}"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str, dest: Path, sums: dict) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and sha256(dest) == sums[name]:
        print(f"{name}: present")
        return dest
    print(f"{name}: downloading", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url(name), tmp)
    got = sha256(tmp)
    if got != sums[name]:
        tmp.unlink()
        sys.exit(f"{name}: checksum mismatch ({got} != {sums[name]})")
    tmp.replace(dest)
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="also download the trained checkpoints")
    ap.add_argument("--data", default="data", help="local data directory")
    a = ap.parse_args()
    data = Path(a.data)
    with urllib.request.urlopen(url("SHA256SUMS")) as r:
        sums = {line.split()[1]: line.split()[0] for line in r.read().decode().splitlines() if line.strip()}
    tar = fetch(PACKAGE, data / "downloads" / PACKAGE, sums)
    out = data / "release"
    if not (out / "glance-c5-open-v4" / "model.safetensors").exists():
        with tarfile.open(tar) as t:
            t.extractall(out, filter="data")
    print(f"release package: {out / 'glance-c5-open-v4'}")
    if a.all:
        for name in CHECKPOINTS:
            fetch(f"{name}.safetensors", data / "ckpt" / f"{name}.safetensors", sums)
        print(f"checkpoints: {data / 'ckpt'}")


if __name__ == "__main__":
    main()
