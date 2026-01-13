#!/usr/bin/env python3
"""Download the full AMI Meeting Corpus audio (Mix-Headset wav files).

Uses `aria2c` and downloads all meeting IDs in the common AMI meeting series.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

BASE_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"

# Full list of AMI meetings (ES=Edinburgh Scenario, IS=Interactive Scenario, IN=Induced, TS=Training Scenario)
# Each meeting has parts a,b,c,d
MEETING_SERIES = [
    # Edinburgh Scenario meetings
    "ES2002",
    "ES2003",
    "ES2004",
    "ES2005",
    "ES2006",
    "ES2007",
    "ES2008",
    "ES2009",
    "ES2010",
    "ES2011",
    "ES2012",
    "ES2013",
    "ES2014",
    "ES2015",
    "ES2016",
    # Interactive Scenario meetings
    "IS1000",
    "IS1001",
    "IS1002",
    "IS1003",
    "IS1004",
    "IS1005",
    "IS1006",
    "IS1007",
    "IS1008",
    "IS1009",
    # Induced meetings
    "IN1001",
    "IN1002",
    "IN1005",
    "IN1007",
    "IN1008",
    "IN1009",
    "IN1012",
    "IN1013",
    "IN1014",
    "IN1016",
    # Training Scenario meetings
    "TS3003",
    "TS3004",
    "TS3005",
    "TS3006",
    "TS3007",
    "TS3008",
    "TS3009",
    "TS3010",
    "TS3011",
    "TS3012",
]

PARTS = ["a", "b", "c", "d"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _download_meeting(*, meeting_id: str, output_dir: Path, connections: int) -> str:
    url = f"{BASE_URL}/{meeting_id}/audio/{meeting_id}.Mix-Headset.wav"
    output_path = output_dir / f"{meeting_id}.Mix-Headset.wav"

    if output_path.exists() and output_path.stat().st_size > 1_000_000:
        return "skip"

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
    return "ok" if result.returncode == 0 else "fail"


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
        default=4,
        help="aria2c connections per download",
    )
    args = parser.parse_args()

    if shutil.which("aria2c") is None:
        raise SystemExit("aria2c not found in PATH")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading full AMI corpus (Mix-Headset)...")
    downloaded, skipped, failed = 0, 0, 0

    for series in MEETING_SERIES:
        for part in PARTS:
            meeting_id = f"{series}{part}"
            idx = downloaded + skipped + failed + 1
            print(f"[{idx}] {meeting_id}...", end=" ", flush=True)
            result = _download_meeting(
                meeting_id=meeting_id,
                output_dir=args.output_dir,
                connections=args.connections,
            )
            if result == "ok":
                print("OK")
                downloaded += 1
            elif result == "skip":
                print("SKIP")
                skipped += 1
            else:
                print("FAIL")
                failed += 1

    print(f"\nDone. Downloaded={downloaded} Skipped={skipped} Failed={failed}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
