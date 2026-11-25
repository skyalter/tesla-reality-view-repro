#!/usr/bin/env python3
"""Download the pretrained weights this project starts from, pinned to exact revisions, and verify them.

Usage:
    python scripts/download_weights.py --dest <dir> [--set stage1] [--dry-run]

No Hugging Face login is needed and no token is sent. Files land in <dest>/<name>/ and
<dest>/WEIGHTS.lock.tsv records repository, revision, size and SHA-256 of every verified file.
Safe to re-run: verified files are skipped and interrupted downloads resume.

Only the Python standard library is used, so this runs before the training environment exists.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path

# name -> repository, pinned revision, {file: (bytes, sha256 or None for small text files)}
WEIGHTS = {
    "da3-large-1.1": {
        "repo": "depth-anything/DA3-LARGE-1.1",
        "revision": "0e109ae307c5982f319a67cf6f9f99ccdc0ec97c",
        "files": {
            "model.safetensors": (1643843860, "739905c423cf0d6ccaf9e61a8401d82ba1ac32d7f4d3ee6dca8f92b377633f64"),
            "config.json": (1213, None),
        },
        "use": "backbone and depth head to fine-tune",
    },
    "da3-base": {
        "repo": "depth-anything/DA3-BASE",
        "revision": "f4a6c9b3c95e41c82048423d3493a81ec3fa810e",
        "files": {
            "model.safetensors": (541518028, "e01067dc1659613083d9145a9a2547ccdbe6ccbbf83c4fe7b3e8a4e2bdae78b5"),
            "config.json": (1205, None),
        },
        "use": "small backbone for debugging",
    },
    "da3metric-large": {
        "repo": "depth-anything/DA3METRIC-LARGE",
        "revision": "4010e39f3634a45bc60553321fb49fb760bd594e",
        "files": {
            "model.safetensors": (1336734448, "bbea5b0b3ee389849cffa7ddae89de064a90abd2b055fc5aa99aac68db324776"),
            "config.json": (847, None),
        },
        "use": "sky scores during preprocessing",
    },
}
SETS = {
    "debug": ["da3-base"],
    "stage1": ["da3-large-1.1", "da3-base", "da3metric-large"],
}
CHUNK = 1 << 20


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def is_verified(path: Path, size: int, sha: str | None) -> bool:
    if not path.is_file() or path.stat().st_size != size:
        return False
    return sha is None or sha256_of(path) == sha


def fetch(url: str, path: Path, size: int) -> None:
    """Download url to path through a .part file, resuming if a partial file exists."""
    part = path.with_name(path.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    if have > size:
        part.unlink()
        have = 0
    if have < size:
        req = urllib.request.Request(url, headers={"User-Agent": "weights-downloader"})
        if have:
            req.add_header("Range", f"bytes={have}-")
        with urllib.request.urlopen(req, timeout=60) as resp:
            # A server that ignores Range answers 200 with the whole file: start over.
            mode = "ab" if have and resp.status == 206 else "wb"
            if mode == "wb":
                have = 0
            with part.open(mode) as out:
                done = have
                while chunk := resp.read(CHUNK):
                    out.write(chunk)
                    done += len(chunk)
                    if size >= 100 * CHUNK and done // (200 * CHUNK) != (done - len(chunk)) // (200 * CHUNK):
                        print(f"    {done / 1e9:.2f} / {size / 1e9:.2f} GB", flush=True)
    part.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", required=True, type=Path)
    ap.add_argument("--set", default="stage1", choices=sorted(SETS), dest="which")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    names = SETS[args.which]
    total = sum(size for n in names for size, _ in WEIGHTS[n]["files"].values())
    print(f"set={args.which}: {len(names)} models, {total / 1e9:.2f} GB -> {args.dest}")
    lock_rows = []
    failed = 0
    for name in names:
        spec = WEIGHTS[name]
        print(f"{name}  ({spec['repo']} @ {spec['revision'][:12]}, {spec['use']})")
        for fname, (size, sha) in spec["files"].items():
            path = args.dest / name / fname
            url = f"https://huggingface.co/{spec['repo']}/resolve/{spec['revision']}/{fname}"
            if is_verified(path, size, sha):
                print(f"  ok       {fname}  {size} bytes")
            elif args.dry_run:
                print(f"  would download {fname}  {size} bytes")
                continue
            else:
                print(f"  download {fname}  {size} bytes")
                path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    fetch(url, path, size)
                except (urllib.error.URLError, OSError) as e:
                    print(f"  FAILED   {fname}: {e}", file=sys.stderr)
                    failed += 1
                    continue
                if not is_verified(path, size, sha):
                    print(f"  FAILED   {fname}: size or SHA-256 does not match the pinned value", file=sys.stderr)
                    failed += 1
                    continue
            lock_rows.append((name, spec["repo"], spec["revision"], fname, str(size), sha or sha256_of(path)))

    if args.dry_run:
        print("dry run: nothing downloaded")
        return 0
    if lock_rows:
        lock = args.dest / "WEIGHTS.lock.tsv"
        header = "name\trepo\trevision\tfile\tbytes\tsha256\n"
        lock.write_text(header + "".join("\t".join(r) + "\n" for r in lock_rows))
        print(f"lock file: {lock}")
    if failed:
        print(f"{failed} file(s) failed; run the same command again to resume", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
