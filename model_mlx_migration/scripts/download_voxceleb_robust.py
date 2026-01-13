#!/usr/bin/env python3
"""
Robust VoxCeleb Download Script

Downloads VoxCeleb1 and VoxCeleb2 with:
- Resume capability (won't re-download completed files)
- Multiple source fallback (HuggingFace, official)
- Progress tracking and verification
- Automatic extraction

Usage:
    # Download from HuggingFace (recommended - more reliable)
    python scripts/download_voxceleb_robust.py --source huggingface

    # Download from official (requires credentials)
    VOXCELEB_BASE_URL="..." VOXCELEB_KEY="..." python scripts/download_voxceleb_robust.py --source official

    # Resume interrupted download
    python scripts/download_voxceleb_robust.py --resume

    # Download only VoxCeleb1
    python scripts/download_voxceleb_robust.py --voxceleb1-only
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from datetime import datetime

# Try to import optional dependencies
try:
    from huggingface_hub import hf_hub_download, snapshot_download, login
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


# Constants
DATA_DIR = Path(__file__).parent.parent / "data"
VOXCELEB1_DIR = DATA_DIR / "voxceleb1"
VOXCELEB2_DIR = DATA_DIR / "voxceleb2"
STATE_FILE = DATA_DIR / ".voxceleb_download_state.json"

# Expected sizes (approximate, in bytes)
EXPECTED_SIZES = {
    "vox1_dev_wav.zip": 30_000_000_000,  # ~30GB
    "vox1_test_wav.zip": 1_200_000_000,   # ~1.2GB
    "vox2_dev_aac.zip": 75_000_000_000,   # ~75GB
    "vox2_test_aac.zip": 1_000_000_000,   # ~1GB
}

# MD5 checksums
CHECKSUMS = {
    "vox1_dev_wav.zip": "ae63e55b951748cc486645f532ba230b",
    "vox1_test_wav.zip": "185fdc63c3c739954633d50379a3d102",
    "vox2_dev_aac.zip": "bbc063c46078a602ca71605645c2a402",
    "vox2_test_aac.zip": "0d2b3ea430a821c33263b5ea37ede312",
}

# HuggingFace dataset info
HF_DATASETS = {
    "voxceleb1": "ProgramComputer/voxceleb",
    "voxceleb2": "ProgramComputer/voxceleb",
}


def log(msg: str, level: str = "INFO"):
    """Print timestamped log message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}")


def load_state() -> dict:
    """Load download state from file."""
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"completed": [], "in_progress": None, "errors": []}


def save_state(state: dict):
    """Save download state to file."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def compute_md5(filepath: Path, chunk_size: int = 8192) -> str:
    """Compute MD5 hash of file."""
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        if TQDM_AVAILABLE:
            total = filepath.stat().st_size
            with tqdm(total=total, unit="B", unit_scale=True, desc="Verifying") as pbar:
                for chunk in iter(lambda: f.read(chunk_size), b""):
                    md5.update(chunk)
                    pbar.update(len(chunk))
        else:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                md5.update(chunk)
    return md5.hexdigest()


def download_with_wget(url: str, output: Path, resume: bool = True) -> bool:
    """Download file using wget with resume capability."""
    cmd = ["wget", "-c" if resume else "", "-O", str(output), url]
    cmd = [c for c in cmd if c]  # Remove empty strings

    log(f"Downloading: {url}")
    log(f"Output: {output}")

    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        log(f"wget failed: {e}", "ERROR")
        return False


def download_with_curl(url: str, output: Path, resume: bool = True) -> bool:
    """Download file using curl with resume capability."""
    cmd = ["curl", "-L", "--fail", "--retry", "5", "--retry-delay", "2"]
    if resume and output.exists():
        cmd.extend(["-C", "-"])  # Resume
    cmd.extend(["-o", str(output), url])

    log(f"Downloading: {url}")
    log(f"Output: {output}")

    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        log(f"curl failed: {e}", "ERROR")
        return False


def download_from_huggingface(dataset_name: str, output_dir: Path, subset: str = None) -> bool:
    """Download dataset from HuggingFace."""
    if not HF_AVAILABLE:
        log("huggingface_hub not installed. Install with: pip install huggingface_hub", "ERROR")
        return False

    log(f"Downloading from HuggingFace: {dataset_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Try snapshot download first
        snapshot_download(
            repo_id=dataset_name,
            repo_type="dataset",
            local_dir=str(output_dir),
            resume_download=True,
            max_workers=4,
        )
        return True
    except Exception as e:
        log(f"HuggingFace download failed: {e}", "ERROR")
        return False


def download_voxceleb1_hf() -> bool:
    """Download VoxCeleb1 from HuggingFace."""
    log("=== Downloading VoxCeleb1 from HuggingFace ===")

    # VoxCeleb1 is available at multiple HF repos
    hf_repos = [
        "ProgramComputer/voxceleb",
        "speechbrain/voxceleb1",
    ]

    for repo in hf_repos:
        log(f"Trying HuggingFace repo: {repo}")
        try:
            snapshot_download(
                repo_id=repo,
                repo_type="dataset",
                local_dir=str(VOXCELEB1_DIR),
                resume_download=True,
                allow_patterns=["*vox1*", "*VoxCeleb1*"],
                max_workers=4,
            )
            log(f"Successfully downloaded from {repo}")
            return True
        except Exception as e:
            log(f"Failed to download from {repo}: {e}", "WARNING")
            continue

    return False


def download_voxceleb2_hf() -> bool:
    """Download VoxCeleb2 from HuggingFace."""
    log("=== Downloading VoxCeleb2 from HuggingFace ===")

    hf_repos = [
        "ProgramComputer/voxceleb",
    ]

    for repo in hf_repos:
        log(f"Trying HuggingFace repo: {repo}")
        try:
            snapshot_download(
                repo_id=repo,
                repo_type="dataset",
                local_dir=str(VOXCELEB2_DIR),
                resume_download=True,
                allow_patterns=["*vox2*", "*VoxCeleb2*"],
                max_workers=4,
            )
            log(f"Successfully downloaded from {repo}")
            return True
        except Exception as e:
            log(f"Failed to download from {repo}: {e}", "WARNING")
            continue

    return False


def download_official(base_url: str, key: str, voxceleb1: bool = True, voxceleb2: bool = True) -> bool:
    """Download from official VoxCeleb source."""
    state = load_state()
    success = True

    if voxceleb1:
        log("=== Downloading VoxCeleb1 (Official) ===")
        VOXCELEB1_DIR.mkdir(parents=True, exist_ok=True)

        # Download parts
        parts = ["partaa", "partab", "partac", "partad"]
        for part in parts:
            filename = f"vox1_dev_wav_{part}"
            output = VOXCELEB1_DIR / filename

            if filename in state["completed"]:
                log(f"Skipping {filename} (already completed)")
                continue

            url = f"{base_url}?key={key}&file={filename}"
            state["in_progress"] = filename
            save_state(state)

            if download_with_wget(url, output) or download_with_curl(url, output):
                state["completed"].append(filename)
                state["in_progress"] = None
                save_state(state)
            else:
                success = False
                state["errors"].append({"file": filename, "time": str(datetime.now())})
                save_state(state)

        # Download test set
        filename = "vox1_test_wav.zip"
        output = VOXCELEB1_DIR / filename
        if filename not in state["completed"]:
            url = f"{base_url}?key={key}&file={filename}"
            if download_with_wget(url, output) or download_with_curl(url, output):
                state["completed"].append(filename)
                save_state(state)

    if voxceleb2:
        log("=== Downloading VoxCeleb2 (Official) ===")
        VOXCELEB2_DIR.mkdir(parents=True, exist_ok=True)

        # Download parts (8 parts for VoxCeleb2)
        parts = ["partaa", "partab", "partac", "partad", "partae", "partaf", "partag", "partah"]
        for part in parts:
            filename = f"vox2_dev_aac_{part}"
            output = VOXCELEB2_DIR / filename

            if filename in state["completed"]:
                log(f"Skipping {filename} (already completed)")
                continue

            url = f"{base_url}?key={key}&file={filename}"
            state["in_progress"] = filename
            save_state(state)

            if download_with_wget(url, output) or download_with_curl(url, output):
                state["completed"].append(filename)
                state["in_progress"] = None
                save_state(state)
            else:
                success = False
                state["errors"].append({"file": filename, "time": str(datetime.now())})
                save_state(state)

        # Download test set
        filename = "vox2_test_aac.zip"
        output = VOXCELEB2_DIR / filename
        if filename not in state["completed"]:
            url = f"{base_url}?key={key}&file={filename}"
            if download_with_wget(url, output) or download_with_curl(url, output):
                state["completed"].append(filename)
                save_state(state)

    return success


def concatenate_and_extract_voxceleb1():
    """Concatenate VoxCeleb1 parts and extract."""
    log("=== Processing VoxCeleb1 ===")

    parts = sorted(VOXCELEB1_DIR.glob("vox1_dev_wav_part*"))
    if not parts:
        log("No VoxCeleb1 parts found", "WARNING")
        return False

    combined = VOXCELEB1_DIR / "vox1_dev_wav.zip"

    if not combined.exists():
        log(f"Concatenating {len(parts)} parts...")
        with open(combined, "wb") as outfile:
            for part in parts:
                log(f"  Adding {part.name}...")
                with open(part, "rb") as infile:
                    shutil.copyfileobj(infile, outfile)

    # Verify checksum
    expected_md5 = CHECKSUMS.get("vox1_dev_wav.zip")
    if expected_md5:
        log("Verifying checksum...")
        actual_md5 = compute_md5(combined)
        if actual_md5 != expected_md5:
            log(f"Checksum mismatch! Expected {expected_md5}, got {actual_md5}", "ERROR")
            return False
        log("Checksum verified!")

    # Extract
    log("Extracting VoxCeleb1 dev...")
    with zipfile.ZipFile(combined, "r") as zf:
        zf.extractall(VOXCELEB1_DIR)

    # Extract test set
    test_zip = VOXCELEB1_DIR / "vox1_test_wav.zip"
    if test_zip.exists():
        log("Extracting VoxCeleb1 test...")
        with zipfile.ZipFile(test_zip, "r") as zf:
            zf.extractall(VOXCELEB1_DIR)

    return True


def concatenate_and_extract_voxceleb2():
    """Concatenate VoxCeleb2 parts and extract."""
    log("=== Processing VoxCeleb2 ===")

    parts = sorted(VOXCELEB2_DIR.glob("vox2_dev_aac_part*"))
    if not parts:
        log("No VoxCeleb2 parts found", "WARNING")
        return False

    combined = VOXCELEB2_DIR / "vox2_aac.zip"

    if not combined.exists():
        log(f"Concatenating {len(parts)} parts...")
        with open(combined, "wb") as outfile:
            for part in parts:
                log(f"  Adding {part.name}...")
                with open(part, "rb") as infile:
                    shutil.copyfileobj(infile, outfile)

    # Verify checksum
    expected_md5 = CHECKSUMS.get("vox2_dev_aac.zip")
    if expected_md5:
        log("Verifying checksum...")
        actual_md5 = compute_md5(combined)
        if actual_md5 != expected_md5:
            log(f"Checksum mismatch! Expected {expected_md5}, got {actual_md5}", "ERROR")
            return False
        log("Checksum verified!")

    # Extract
    log("Extracting VoxCeleb2 dev...")
    with zipfile.ZipFile(combined, "r") as zf:
        zf.extractall(VOXCELEB2_DIR)

    # Extract test set
    test_zip = VOXCELEB2_DIR / "vox2_test_aac.zip"
    if test_zip.exists():
        log("Extracting VoxCeleb2 test...")
        with zipfile.ZipFile(test_zip, "r") as zf:
            zf.extractall(VOXCELEB2_DIR)

    return True


def convert_aac_to_wav(input_dir: Path, sample_rate: int = 16000):
    """Convert AAC/M4A files to WAV using ffmpeg."""
    log("=== Converting AAC to WAV ===")

    m4a_files = list(input_dir.rglob("*.m4a"))
    if not m4a_files:
        log("No M4A files found to convert")
        return

    log(f"Found {len(m4a_files)} M4A files to convert")

    for i, m4a in enumerate(m4a_files):
        wav = m4a.with_suffix(".wav")
        if wav.exists():
            continue

        if i % 1000 == 0:
            log(f"Converting {i}/{len(m4a_files)}...")

        cmd = [
            "ffmpeg", "-i", str(m4a),
            "-ar", str(sample_rate),
            "-ac", "1",
            "-y", str(wav)
        ]
        subprocess.run(cmd, capture_output=True)


def verify_download() -> dict:
    """Verify downloaded data and return statistics."""
    stats = {
        "voxceleb1": {"speakers": 0, "utterances": 0, "size_gb": 0},
        "voxceleb2": {"speakers": 0, "utterances": 0, "size_gb": 0},
    }

    # Check VoxCeleb1
    if VOXCELEB1_DIR.exists():
        wav_files = list(VOXCELEB1_DIR.rglob("*.wav"))
        speakers = set()
        for f in wav_files:
            # VoxCeleb structure: id10001/video_id/00001.wav
            parts = f.relative_to(VOXCELEB1_DIR).parts
            if len(parts) >= 1:
                speakers.add(parts[0])

        total_size = sum(f.stat().st_size for f in wav_files)
        stats["voxceleb1"] = {
            "speakers": len(speakers),
            "utterances": len(wav_files),
            "size_gb": total_size / (1024**3),
        }

    # Check VoxCeleb2
    if VOXCELEB2_DIR.exists():
        wav_files = list(VOXCELEB2_DIR.rglob("*.wav"))
        m4a_files = list(VOXCELEB2_DIR.rglob("*.m4a"))
        audio_files = wav_files + m4a_files

        speakers = set()
        for f in audio_files:
            parts = f.relative_to(VOXCELEB2_DIR).parts
            if len(parts) >= 1:
                speakers.add(parts[0])

        total_size = sum(f.stat().st_size for f in audio_files)
        stats["voxceleb2"] = {
            "speakers": len(speakers),
            "utterances": len(audio_files),
            "size_gb": total_size / (1024**3),
        }

    return stats


def main():
    parser = argparse.ArgumentParser(description="Download VoxCeleb datasets")
    parser.add_argument("--source", choices=["huggingface", "official", "auto"], default="auto",
                        help="Download source (default: auto)")
    parser.add_argument("--voxceleb1-only", action="store_true", help="Download only VoxCeleb1")
    parser.add_argument("--voxceleb2-only", action="store_true", help="Download only VoxCeleb2")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted download")
    parser.add_argument("--extract-only", action="store_true", help="Only extract (skip download)")
    parser.add_argument("--convert-aac", action="store_true", help="Convert AAC to WAV after download")
    parser.add_argument("--verify", action="store_true", help="Verify downloaded data")
    parser.add_argument("--hf-token", type=str, help="HuggingFace token for gated datasets")
    args = parser.parse_args()

    # Determine what to download
    download_vox1 = not args.voxceleb2_only
    download_vox2 = not args.voxceleb1_only

    log("VoxCeleb Robust Downloader")
    log(f"VoxCeleb1: {download_vox1}, VoxCeleb2: {download_vox2}")
    log(f"Source: {args.source}")

    # Create directories
    VOXCELEB1_DIR.mkdir(parents=True, exist_ok=True)
    VOXCELEB2_DIR.mkdir(parents=True, exist_ok=True)

    # HuggingFace login if token provided
    if args.hf_token and HF_AVAILABLE:
        log("Logging into HuggingFace...")
        login(token=args.hf_token)

    if not args.extract_only:
        # Determine source
        source = args.source
        if source == "auto":
            # Check if credentials are available for official
            if os.environ.get("VOXCELEB_BASE_URL") and os.environ.get("VOXCELEB_KEY"):
                source = "official"
            elif HF_AVAILABLE:
                source = "huggingface"
            else:
                log("No download source available!", "ERROR")
                log("Either set VOXCELEB_BASE_URL/VOXCELEB_KEY or install huggingface_hub")
                sys.exit(1)

        # Download
        if source == "huggingface":
            if download_vox1:
                download_voxceleb1_hf()
            if download_vox2:
                download_voxceleb2_hf()
        else:
            base_url = os.environ.get("VOXCELEB_BASE_URL")
            key = os.environ.get("VOXCELEB_KEY")
            if not base_url or not key:
                log("VOXCELEB_BASE_URL and VOXCELEB_KEY environment variables required", "ERROR")
                sys.exit(1)
            download_official(base_url, key, download_vox1, download_vox2)

    # Extract
    if download_vox1:
        concatenate_and_extract_voxceleb1()
    if download_vox2:
        concatenate_and_extract_voxceleb2()

    # Convert AAC to WAV
    if args.convert_aac and download_vox2:
        convert_aac_to_wav(VOXCELEB2_DIR)

    # Verify
    if args.verify or True:  # Always verify
        stats = verify_download()
        log("=== Download Statistics ===")
        for name, s in stats.items():
            log(f"{name}: {s['speakers']} speakers, {s['utterances']} utterances, {s['size_gb']:.1f}GB")

    log("=== Done ===")


if __name__ == "__main__":
    main()
