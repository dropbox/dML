#!/usr/bin/env python3
"""
Download VoxCeleb from HuggingFace

The dataset is available at: https://huggingface.co/datasets/ProgramComputer/voxceleb
License: CC-BY-4.0

Usage:
    python scripts/download_voxceleb_hf.py
    python scripts/download_voxceleb_hf.py --audio-only
    python scripts/download_voxceleb_hf.py --vox1-only
    python scripts/download_voxceleb_hf.py --extract
    python scripts/download_voxceleb_hf.py --method aria2 --audio-only --extract
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from shutil import which

try:
    from huggingface_hub import snapshot_download, list_repo_files
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
    sys.exit(1)

# Paths
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
VOX1_DIR = DATA_DIR / "voxceleb1"
VOX2_DIR = DATA_DIR / "voxceleb2"

# HuggingFace dataset
HF_REPO = "ProgramComputer/voxceleb"
HF_BASE = "https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/main"


# File lists for aria2 "audio-only" downloads (avoid multi-hundred-GB mp4 parts by default)
VOX1_AUDIO_FILES = [
    "vox1/vox1_dev_wav.zip",
    "vox1/vox1_test_wav.zip",
    "vox1/vox1_meta.csv",
    "vox1/vox1_dev_txt.zip",
    "vox1/vox1_test_txt.zip",
]

VOX2_AUDIO_FILES = [
    "vox2/vox2_aac_1.zip",
    "vox2/vox2_aac_2.zip",
    "vox2/vox2_test_aac.zip",
    "vox2/vox2_meta.csv",
    "vox2/vox2_dev_txt.zip",
    "vox2/vox2_test_txt.zip",
]

VOX2_MP4_FILES = [
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


def log(msg: str, level: str = "INFO"):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}")


def list_files():
    """List all files in the HF repo."""
    log("Listing files in HuggingFace repo...")
    files = list_repo_files(HF_REPO, repo_type="dataset")
    for f in sorted(files):
        print(f"  {f}")
    return files


def download_voxceleb():
    """Download VoxCeleb from HuggingFace."""
    log("=== Downloading VoxCeleb from HuggingFace ===")
    log(f"Repo: {HF_REPO}")
    log(f"Output: {DATA_DIR}")

    # Create directories
    VOX1_DIR.mkdir(parents=True, exist_ok=True)
    VOX2_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # Download everything
        log("Starting download (this may take several hours)...")
        snapshot_download(
            repo_id=HF_REPO,
            repo_type="dataset",
            local_dir=str(DATA_DIR / "voxceleb_hf"),
            resume_download=True,
            max_workers=4,
        )
        log("Download complete!")
        return True
    except Exception as e:
        log(f"Download failed: {e}", "ERROR")
        return False


def download_specific_files(patterns: list):
    """Download specific files matching patterns."""
    log(f"Downloading files matching: {patterns}")

    try:
        snapshot_download(
            repo_id=HF_REPO,
            repo_type="dataset",
            local_dir=str(DATA_DIR / "voxceleb_hf"),
            resume_download=True,
            allow_patterns=patterns,
            max_workers=4,
        )
        return True
    except Exception as e:
        log(f"Download failed: {e}", "ERROR")
        return False


def _aria2_download_file(url: str, output_path: Path, connections: int) -> tuple[bool, str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "aria2c",
        "-x",
        str(connections),
        "-s",
        str(connections),
        "-k",
        "10M",
        "-c",
        "--file-allocation=none",
        "--auto-file-renaming=false",
        "--console-log-level=error",
        "-d",
        str(output_path.parent),
        "-o",
        output_path.name,
        url,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return True, ""

    err = (result.stderr or result.stdout or "").strip()
    return False, err.splitlines()[-1] if err else "aria2c failed"


def download_audio_only_via_aria2(connections: int, vox1_only: bool, vox2_only: bool, include_mp4: bool) -> bool:
    """Download audio-only file set via aria2 (fast + resumable)."""
    if which("aria2c") is None:
        log("aria2c not found. Install with: brew install aria2", "ERROR")
        return False

    hf_dir = DATA_DIR / "voxceleb_hf"
    hf_dir.mkdir(parents=True, exist_ok=True)

    files: list[str] = []
    if vox1_only:
        files.extend(VOX1_AUDIO_FILES)
    elif vox2_only:
        files.extend(VOX2_AUDIO_FILES)
        if include_mp4:
            files.extend(VOX2_MP4_FILES)
    else:
        files.extend(VOX1_AUDIO_FILES)
        files.extend(VOX2_AUDIO_FILES)
        if include_mp4:
            files.extend(VOX2_MP4_FILES)

    log(f"Downloading {len(files)} files via aria2c (connections={connections})")

    failed: list[str] = []
    for f in files:
        url = f"{HF_BASE}/{f}"
        out = hf_dir / f
        log(f"Downloading: {f}")
        ok, err = _aria2_download_file(url, out, connections=connections)
        if not ok:
            log(f"FAILED: {f} ({err})", "ERROR")
            failed.append(f)

    if failed:
        log(f"{len(failed)} files failed. First 10:", "ERROR")
        for f in failed[:10]:
            log(f"  - {f}", "ERROR")
        return False

    return True


def extract_and_organize():
    """Extract and organize downloaded files."""
    hf_dir = DATA_DIR / "voxceleb_hf"

    if not hf_dir.exists():
        log("No HuggingFace download directory found", "ERROR")
        return False

    log("=== Organizing downloaded files ===")

    # Prefer canonical HF layout: voxceleb_hf/vox1/* and voxceleb_hf/vox2/*
    vox1_src = hf_dir / "vox1"
    vox2_src = hf_dir / "vox2"

    if vox1_src.exists():
        for f in vox1_src.iterdir():
            dest = VOX1_DIR / f.name
            if not dest.exists():
                log(f"Moving vox1/{f.name} to voxceleb1/")
                f.rename(dest)

    if vox2_src.exists():
        for f in vox2_src.iterdir():
            dest = VOX2_DIR / f.name
            if not dest.exists():
                log(f"Moving vox2/{f.name} to voxceleb2/")
                f.rename(dest)

    # Concatenate parts if needed
    log("=== Concatenating file parts ===")

    # VoxCeleb1
    vox1_parts = sorted(VOX1_DIR.glob("vox1_dev*part*"))
    if vox1_parts and not (VOX1_DIR / "vox1_dev_wav.zip").exists():
        log("Concatenating VoxCeleb1 dev parts...")
        with open(VOX1_DIR / "vox1_dev_wav.zip", "wb") as outf:
            for part in vox1_parts:
                log(f"  Adding {part.name}")
                with open(part, "rb") as inf:
                    outf.write(inf.read())

    # VoxCeleb2 AAC (only concatenate if legacy part files exist)
    vox2_aac_parts = sorted(VOX2_DIR.glob("vox2_dev_aac_part*"))
    if vox2_aac_parts and not (VOX2_DIR / "vox2_dev_aac.zip").exists():
        log("Concatenating VoxCeleb2 AAC parts...")
        with open(VOX2_DIR / "vox2_dev_aac.zip", "wb") as outf:
            for part in vox2_aac_parts:
                log(f"  Adding {part.name}")
                with open(part, "rb") as inf:
                    outf.write(inf.read())

    # Extract zips
    log("=== Extracting archives ===")

    import zipfile

    for zip_path in VOX1_DIR.glob("*.zip"):
        log(f"Extracting {zip_path.name}...")
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(VOX1_DIR)
        except Exception as e:
            log(f"Failed to extract {zip_path.name}: {e}", "ERROR")

    for zip_path in VOX2_DIR.glob("*.zip"):
        log(f"Extracting {zip_path.name}...")
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(VOX2_DIR)
        except Exception as e:
            log(f"Failed to extract {zip_path.name}: {e}", "ERROR")

    return True


def print_stats():
    """Print download statistics."""
    log("=== Statistics ===")

    for name, path in [("VoxCeleb1", VOX1_DIR), ("VoxCeleb2", VOX2_DIR)]:
        if not path.exists():
            log(f"{name}: Not found")
            continue

        wav_files = list(path.rglob("*.wav"))
        m4a_files = list(path.rglob("*.m4a"))
        zip_files = list(path.glob("*.zip"))

        speakers = set()
        for f in wav_files + m4a_files:
            try:
                rel = f.relative_to(path)
                if len(rel.parts) >= 1:
                    speakers.add(rel.parts[0])
            except:
                pass

        log(f"{name}:")
        log(f"  ZIP files: {len(zip_files)}")
        log(f"  WAV files: {len(wav_files)}")
        log(f"  M4A files: {len(m4a_files)}")
        log(f"  Speakers: {len(speakers)}")

        # Size
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        log(f"  Total size: {total / (1024**3):.1f} GB")


def main():
    parser = argparse.ArgumentParser(description="Download VoxCeleb from HuggingFace")
    parser.add_argument("--list", action="store_true", help="List available files")
    parser.add_argument("--vox1-only", action="store_true", help="Download only VoxCeleb1")
    parser.add_argument("--vox2-only", action="store_true", help="Download only VoxCeleb2")
    parser.add_argument("--extract", action="store_true", help="Extract and organize after download")
    parser.add_argument("--stats", action="store_true", help="Print statistics only")
    parser.add_argument(
        "--method",
        choices=["snapshot", "aria2"],
        default="snapshot",
        help="Download method (default: snapshot)",
    )
    parser.add_argument(
        "--audio-only",
        action="store_true",
        help="Avoid downloading huge mp4 parts (recommended for speaker/audio training)",
    )
    parser.add_argument("--include-mp4", action="store_true", help="Include mp4 parts (very large)")
    parser.add_argument("--connections", type=int, default=16, help="aria2c connections per file (aria2 mode)")
    args = parser.parse_args()

    if args.stats:
        print_stats()
        return

    if args.list:
        list_files()
        return

    # Determine what to download
    if args.method == "aria2":
        if not args.audio_only:
            log("aria2 method only supports --audio-only (use snapshot for full repo)", "ERROR")
            sys.exit(2)
        ok = download_audio_only_via_aria2(
            connections=args.connections,
            vox1_only=args.vox1_only,
            vox2_only=args.vox2_only,
            include_mp4=args.include_mp4,
        )
        if not ok:
            sys.exit(1)
    else:
        if args.audio_only:
            patterns = []
            if args.vox1_only:
                patterns = ["vox1/*"]
            elif args.vox2_only:
                patterns = ["vox2/*aac*", "vox2/*meta*", "vox2/*txt*"]
                if args.include_mp4:
                    patterns.append("vox2/*mp4*")
            else:
                patterns = ["vox1/*", "vox2/*aac*", "vox2/*meta*", "vox2/*txt*"]
                if args.include_mp4:
                    patterns.append("vox2/*mp4*")

            download_specific_files(patterns)
        else:
            if args.vox1_only:
                download_specific_files(["*vox1*"])
            elif args.vox2_only:
                download_specific_files(["*vox2*"])
            else:
                download_voxceleb()

    if args.extract:
        extract_and_organize()

    print_stats()


if __name__ == "__main__":
    main()
