#!/usr/bin/env python3
"""Download selected AMI Meeting Corpus audio files directly.

Defaults to a small subset for smoke-testing. Uses `aria2c` for downloads.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

BASE_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"

DEFAULT_MEETINGS = [
    "ES2002a",
    "ES2002b",
    "ES2002c",
    "ES2002d",
    "ES2003a",
    "ES2003b",
    "ES2003c",
    "ES2003d",
    "IS1000a",
    "IS1000b",
    "IS1000c",
    "IS1000d",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _download_one(*, meeting_id: str, output_dir: Path, connections: int) -> bool:
    url = f"{BASE_URL}/{meeting_id}/audio/{meeting_id}.Mix-Headset.wav"
    output_path = output_dir / f"{meeting_id}.Mix-Headset.wav"

    if output_path.exists() and output_path.stat().st_size > 1_000_000:
        print(f"[SKIP] {meeting_id}")
        return True

    cmd = [
        "aria2c",
        "-x",
        str(connections),
        "-s",
        str(connections),
        "-c",
        "--file-allocation=none",
        "--console-log-level=error",
        "-d",
        str(output_dir),
        "-o",
        output_path.name,
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"[OK] {meeting_id}")
        return True

    stderr = (result.stderr or "").strip()
    print(f"[FAIL] {meeting_id}: {stderr[:200]}")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_repo_root() / "data" / "ami" / "audio",
        help="Output directory for downloaded .wav files",
    )
    parser.add_argument(
        "--connections",
        type=int,
        default=8,
        help="aria2c connections per download",
    )
    parser.add_argument(
        "--meetings",
        nargs="*",
        default=DEFAULT_MEETINGS,
        help="Meeting IDs to download (e.g., ES2002a)",
    )
    args = parser.parse_args()

    if shutil.which("aria2c") is None:
        raise SystemExit("aria2c not found in PATH")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading AMI audio files (subset)...")
    print("Note: full AMI corpus is ~100GB")
    ok = 0
    fail = 0
    for meeting_id in args.meetings:
        if _download_one(
            meeting_id=meeting_id,
            output_dir=args.output_dir,
            connections=args.connections,
        ):
            ok += 1
        else:
            fail += 1

    print(f"Done. OK={ok} FAIL={fail}")
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
