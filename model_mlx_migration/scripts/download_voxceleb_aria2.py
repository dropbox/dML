#!/usr/bin/env python3
"""
Download VoxCeleb dataset using aria2c for reliable multi-connection downloads.
Uses HuggingFace URLs with aria2c for better resume support and speed.
"""

import subprocess
import sys
from pathlib import Path


# HuggingFace dataset URL pattern
HF_BASE = "https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/main"

# Files to download - prioritize audio files needed for speaker embeddings
VOX1_FILES = [
    "vox1/vox1_dev_wav.zip",
    "vox1/vox1_test_wav.zip",
    "vox1/vox1_meta.csv",
    "vox1/vox1_dev_txt.zip",
    "vox1/vox1_test_txt.zip",
]

VOX2_FILES = [
    # AAC audio (smaller, good for training)
    "vox2/vox2_aac_1.zip",
    "vox2/vox2_aac_2.zip",
    "vox2/vox2_test_aac.zip",
    # Metadata
    "vox2/vox2_meta.csv",
    "vox2/vox2_dev_txt.zip",
    "vox2/vox2_test_txt.zip",
    # MP4 video parts (optional, large)
    "vox2/vox2_dev_mp4_partaa",
    "vox2/vox2_dev_mp4_partab",
    "vox2/vox2_dev_mp4_partac",
    "vox2/vox2_dev_mp4_partad",
    "vox2/vox2_dev_mp4_partae",
    "vox2/vox2_dev_mp4_partaf",
    "vox2/vox2_dev_mp4_partag",
    "vox2/vox2_dev_mp4_partah",
    "vox2/vox2_dev_mp4_partai",
    "vox2/vox2_test_mp4.zip",
]


def get_existing_files(output_dir: Path) -> set:
    """Get set of files that already exist and are complete."""
    existing = set()
    for f in VOX1_FILES + VOX2_FILES:
        local_path = output_dir / f.replace("vox1/", "vox1/").replace("vox2/", "vox2/")
        if local_path.exists():
            # Check if file has reasonable size (not truncated)
            size = local_path.stat().st_size
            if size > 1000:  # At least 1KB
                existing.add(f)
    return existing


def download_file(url: str, output_path: Path, connections: int = 16) -> bool:
    """Download a file using aria2c with multiple connections."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "aria2c",
        "-x", str(connections),  # Max connections per server
        "-s", str(connections),  # Split file into N parts
        "-k", "10M",  # Min split size
        "-c",  # Continue download
        "--file-allocation=none",
        "--auto-file-renaming=false",
        "-d", str(output_path.parent),
        "-o", output_path.name,
        url,
    ]

    print(f"Downloading: {output_path.name}")
    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    output_dir = Path("data/voxceleb_hf")
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = get_existing_files(output_dir)
    print(f"Found {len(existing)} existing files")

    # Determine which files to download
    all_files = VOX1_FILES + VOX2_FILES
    to_download = [f for f in all_files if f not in existing]

    if not to_download:
        print("All files already downloaded!")
        return

    print(f"Need to download {len(to_download)} files:")
    for f in to_download:
        print(f"  - {f}")

    # Download each file
    failed = []
    for f in to_download:
        url = f"{HF_BASE}/{f}"
        output_path = output_dir / f

        if not download_file(url, output_path):
            failed.append(f)
            print(f"FAILED: {f}")
        else:
            print(f"OK: {f}")

    if failed:
        print(f"\n{len(failed)} files failed:")
        for f in failed:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("\nAll downloads complete!")


if __name__ == "__main__":
    main()
