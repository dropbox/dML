#!/usr/bin/env python3
"""
Download People's Speech dataset using aria2c for reliable multi-connection downloads.
"""

import argparse
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from shutil import which

from huggingface_hub import list_repo_files

HF_BASE = "https://huggingface.co/datasets/MLCommons/peoples_speech/resolve/main"
HF_REPO = "MLCommons/peoples_speech"


@dataclass(frozen=True)
class DownloadSpec:
    prefix: str
    include_metadata: bool


def get_files_to_download(spec: DownloadSpec) -> list[str]:
    """Get list of files from HuggingFace repo."""
    print("Fetching file list from HuggingFace...")
    files = list_repo_files(HF_REPO, repo_type="dataset")

    parquet = [f for f in files if f.startswith(spec.prefix) and f.endswith(".parquet")]
    if spec.include_metadata:
        metadata = [f for f in files if f.endswith((".md", ".json"))]
        return sorted(set(parquet + metadata))

    return sorted(parquet)


def get_existing_files(output_dir: Path) -> set[str]:
    """Get set of already downloaded files."""
    existing = set()
    for f in output_dir.rglob("*.parquet"):
        rel = f.relative_to(output_dir)
        existing.add(str(rel))
    return existing


def download_file(url: str, output_path: Path, connections: int) -> tuple[bool, str]:
    """Download a file using aria2c."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "aria2c",
        "-x", str(connections),
        "-s", str(connections),
        "-k", "10M",
        "-c",  # Continue
        "--file-allocation=none",
        "--auto-file-renaming=false",
        "--console-log-level=error",
        "-d", str(output_path.parent),
        "-o", output_path.name,
        url,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return True, ""

    err = (result.stderr or result.stdout or "").strip()
    return False, err.splitlines()[-1] if err else "aria2c failed"


def main():
    parser = argparse.ArgumentParser(description="Download People's Speech via aria2c")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/peoples_speech"),
        help="Destination directory (default: data/peoples_speech)",
    )
    parser.add_argument(
        "--prefix",
        default="clean/train",
        help="Repo path prefix to download (default: clean/train)",
    )
    parser.add_argument("--connections", type=int, default=8, help="aria2c connections per file")
    parser.add_argument("--retries", type=int, default=3, help="Retries per file (default: 3)")
    parser.add_argument("--max-files", type=int, default=0, help="Limit number of files (0 = no limit)")
    parser.add_argument("--include-metadata", action="store_true", help="Also download .md/.json files")
    args = parser.parse_args()

    if which("aria2c") is None:
        raise SystemExit("ERROR: aria2c not found. Install with: brew install aria2")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    spec = DownloadSpec(prefix=args.prefix, include_metadata=args.include_metadata)
    all_files = get_files_to_download(spec)
    print(f"Total files matching prefix '{args.prefix}': {len(all_files)}")

    existing = get_existing_files(args.output_dir)
    print(f"Already downloaded: {len(existing)}")

    # Filter to only files we need
    to_download = [f for f in all_files if f.endswith(".parquet") and f not in existing]
    if args.max_files:
        to_download = to_download[: args.max_files]
    print(f"Need to download: {len(to_download)} parquet files")

    if not to_download:
        print("All files already downloaded!")
        return

    # Download in batches
    failed = []
    for i, f in enumerate(to_download):
        url = f"{HF_BASE}/{f}"
        output_path = args.output_dir / f

        print(f"[{i+1}/{len(to_download)}] {f}...", end=" ", flush=True)
        ok = False
        last_err = ""
        for attempt in range(1, args.retries + 1):
            ok, last_err = download_file(url, output_path, connections=args.connections)
            if ok:
                break
            time.sleep(min(30, 2 ** (attempt - 1)))

        if ok:
            print("OK")
        else:
            print(f"FAILED ({last_err})")
            failed.append(f)

        # Progress checkpoint every 50 files
        if (i + 1) % 50 == 0:
            print(f"--- Progress: {i+1}/{len(to_download)} ({len(failed)} failed) ---")

    print(f"\nComplete! {len(to_download) - len(failed)} succeeded, {len(failed)} failed")
    if failed:
        print("Failed files:")
        for f in failed[:10]:
            print(f"  - {f}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")


if __name__ == "__main__":
    main()
