#!/usr/bin/env python3
"""
Download All Missing Datasets

Downloads all critical datasets needed for SOTA++ training.
Uses HuggingFace datasets library for reliable downloads with resume.

Usage:
    python scripts/download_all_datasets.py --all
    python scripts/download_all_datasets.py --tedlium
    python scripts/download_all_datasets.py --ami
    python scripts/download_all_datasets.py --peoples-speech
"""

import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Try imports
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

try:
    from huggingface_hub import snapshot_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# Paths
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
LOG_DIR = PROJECT_DIR / "logs"


def log(msg: str, level: str = "INFO"):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}")


# Dataset configurations
DATASETS = {
    "tedlium3": {
        "name": "TEDLIUM-3",
        "hf_repo": "LIUM/tedlium",
        "hf_name": "release3",
        "output_dir": DATA_DIR / "tedlium3",
        "description": "TED talk transcriptions for punctuation training",
        "size": "~50GB",
        "priority": "HIGH",
    },
    "ami": {
        "name": "AMI Meeting Corpus",
        "hf_repo": "edinburghcstr/ami",
        "hf_name": "ihm",
        "output_dir": DATA_DIR / "ami",
        "description": "Multi-speaker meeting recordings for diarization",
        "size": "~100GB",
        "priority": "MEDIUM",
    },
    "peoples_speech": {
        "name": "People's Speech",
        "hf_repo": "MLCommons/peoples_speech",
        "hf_name": "clean",
        "output_dir": DATA_DIR / "peoples_speech",
        "description": "Large-scale English ASR (30,000+ hours)",
        "size": "~350GB",
        "priority": "HIGH",
    },
    "librispeech": {
        "name": "LibriSpeech",
        "hf_repo": "openslr/librispeech_asr",
        "hf_name": None,
        "output_dir": DATA_DIR / "LibriSpeech",
        "description": "Clean English speech for ASR baseline",
        "size": "~60GB",
        "priority": "HIGH",
    },
    "commonvoice_en": {
        "name": "CommonVoice English",
        "hf_repo": "mozilla-foundation/common_voice_16_1",
        "hf_name": "en",
        "output_dir": DATA_DIR / "commonvoice" / "en",
        "description": "Community-contributed English speech",
        "size": "~90GB",
        "priority": "HIGH",
    },
    "mls_english": {
        "name": "MLS English",
        "hf_repo": "facebook/multilingual_librispeech",
        "hf_name": "english",
        "output_dir": DATA_DIR / "mls" / "english",
        "description": "Multilingual LibriSpeech - English subset",
        "size": "~100GB",
        "priority": "HIGH",
    },
    "voxpopuli": {
        "name": "VoxPopuli",
        "hf_repo": "facebook/voxpopuli",
        "hf_name": "en",
        "output_dir": DATA_DIR / "voxpopuli",
        "description": "European Parliament recordings",
        "size": "~100GB",
        "priority": "MEDIUM",
    },
    "gigaspeech": {
        "name": "GigaSpeech",
        "hf_repo": "speechcolab/gigaspeech",
        "hf_name": "xs",  # Start with XS subset
        "output_dir": DATA_DIR / "gigaspeech",
        "description": "Large-scale English speech (10,000 hours)",
        "size": "~250GB (xs: 10GB)",
        "priority": "LOW",  # License may be non-commercial
    },
    "fleurs": {
        "name": "FLEURS",
        "hf_repo": "google/fleurs",
        "hf_name": "en_us",
        "output_dir": DATA_DIR / "fleurs",
        "description": "Multilingual evaluation benchmark",
        "size": "~10GB",
        "priority": "MEDIUM",
    },
    "covost2": {
        "name": "CoVoST 2",
        "hf_repo": "facebook/covost2",
        "hf_name": "en_de",
        "output_dir": DATA_DIR / "covost2",
        "description": "Speech translation dataset",
        "size": "~20GB",
        "priority": "LOW",
    },
}


def download_with_datasets(config: dict) -> bool:
    """Download using HuggingFace datasets library."""
    if not DATASETS_AVAILABLE:
        log("datasets library not available", "ERROR")
        return False

    log(f"Downloading {config['name']} using datasets library...")
    log(f"  Repo: {config['hf_repo']}")
    log(f"  Config: {config['hf_name']}")
    log(f"  Output: {config['output_dir']}")

    try:
        config['output_dir'].mkdir(parents=True, exist_ok=True)

        # Load dataset (this downloads it)
        if config['hf_name']:
            ds = load_dataset(
                config['hf_repo'],
                config['hf_name'],
                cache_dir=str(config['output_dir']),
            )
        else:
            ds = load_dataset(
                config['hf_repo'],
                cache_dir=str(config['output_dir']),
            )

        log(f"  Downloaded: {ds}")
        return True

    except Exception as e:
        log(f"  datasets.load_dataset failed: {e}", "WARNING")

        # Newer `datasets` versions can no longer load script-based datasets.
        # Fall back to `snapshot_download` to at least fetch the repo contents.
        if HF_AVAILABLE:
            log("Falling back to huggingface_hub snapshot_download...", "INFO")
            return download_with_snapshot(config)

        log("huggingface_hub not available for fallback", "ERROR")
        return False


def download_with_snapshot(config: dict) -> bool:
    """Download using huggingface_hub snapshot_download."""
    if not HF_AVAILABLE:
        log("huggingface_hub not available", "ERROR")
        return False

    log(f"Downloading {config['name']} using snapshot_download...")

    try:
        config['output_dir'].mkdir(parents=True, exist_ok=True)

        snapshot_download(
            repo_id=config['hf_repo'],
            repo_type="dataset",
            local_dir=str(config['output_dir']),
            resume_download=True,
        )
        return True

    except Exception as e:
        log(f"  Failed: {e}", "ERROR")
        return False


def download_tedlium3():
    """Download TEDLIUM-3 from OpenSLR."""
    log("=== Downloading TEDLIUM-3 ===")
    output_dir = DATA_DIR / "tedlium3"
    output_dir.mkdir(parents=True, exist_ok=True)

    url = "https://www.openslr.org/resources/51/TEDLIUM_release-3.tgz"
    output_file = output_dir / "TEDLIUM_release-3.tgz"

    if output_file.exists():
        log("Already downloaded, extracting...")
    else:
        log(f"Downloading from {url}...")
        # OpenSLR occasionally returns 403 to wget's default user-agent.
        # Try curl with a browser-ish UA and retries, then fall back to wget.
        curl_cmd = [
            "curl",
            "-L",
            "--fail",
            "--retry",
            "5",
            "--retry-delay",
            "2",
            "-A",
            "Mozilla/5.0",
            "-o",
            str(output_file),
            url,
        ]
        wget_cmd = [
            "wget",
            "-c",
            "--user-agent=Mozilla/5.0",
            "-O",
            str(output_file),
            url,
        ]

        try:
            subprocess.run(curl_cmd, check=True)
        except subprocess.CalledProcessError:
            subprocess.run(wget_cmd, check=True)

    # Extract
    log("Extracting...")
    cmd = ["tar", "-xzf", str(output_file), "-C", str(output_dir)]
    subprocess.run(cmd, check=True)

    log("TEDLIUM-3 download complete!")
    return True


def download_ami():
    """Download AMI Meeting Corpus."""
    log("=== Downloading AMI Meeting Corpus ===")

    config = DATASETS["ami"]
    return download_with_datasets(config)


def download_peoples_speech():
    """Download People's Speech."""
    log("=== Downloading People's Speech ===")

    config = DATASETS["peoples_speech"]
    return download_with_datasets(config)


def check_dataset_status(name: str) -> dict:
    """Check if dataset exists and its size."""
    config = DATASETS.get(name)
    if not config:
        return {"exists": False, "size": 0}

    output_dir = config["output_dir"]
    if not output_dir.exists():
        return {"exists": False, "size": 0}

    # Count files and size
    files = list(output_dir.rglob("*"))
    total_size = sum(f.stat().st_size for f in files if f.is_file())

    return {
        "exists": True,
        "files": len([f for f in files if f.is_file()]),
        "size_gb": total_size / (1024**3),
    }


def print_status():
    """Print status of all datasets."""
    log("=== Dataset Status ===")
    print(f"{'Dataset':<20} {'Priority':<10} {'Status':<15} {'Size':<10}")
    print("-" * 60)

    for name, config in DATASETS.items():
        status = check_dataset_status(name)
        if status["exists"]:
            status_str = f"{status['size_gb']:.1f}GB ({status['files']} files)"
        else:
            status_str = "NOT DOWNLOADED"

        print(f"{config['name']:<20} {config['priority']:<10} {status_str:<15}")


def main():
    parser = argparse.ArgumentParser(description="Download missing datasets")
    parser.add_argument("--all", action="store_true", help="Download all missing datasets")
    parser.add_argument("--tedlium", action="store_true", help="Download TEDLIUM-3")
    parser.add_argument("--ami", action="store_true", help="Download AMI Corpus")
    parser.add_argument("--peoples-speech", action="store_true", help="Download People's Speech")
    parser.add_argument("--librispeech", action="store_true", help="Download LibriSpeech")
    parser.add_argument("--status", action="store_true", help="Print status only")
    parser.add_argument("--high-priority", action="store_true", help="Download HIGH priority only")
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if args.status:
        print_status()
        return

    downloads = []

    if args.all or args.high_priority:
        for name, config in DATASETS.items():
            if args.high_priority and config["priority"] != "HIGH":
                continue
            status = check_dataset_status(name)
            if not status["exists"] or status.get("size_gb", 0) < 1:
                downloads.append(name)

    if args.tedlium:
        downloads.append("tedlium3")
    if args.ami:
        downloads.append("ami")
    if args.peoples_speech:
        downloads.append("peoples_speech")
    if args.librispeech:
        downloads.append("librispeech")

    if not downloads:
        log("No datasets to download. Use --status to see current state.")
        print_status()
        return

    log(f"Downloading {len(downloads)} datasets: {downloads}")

    for name in downloads:
        if name == "tedlium3":
            download_tedlium3()
        else:
            config = DATASETS.get(name)
            if config:
                download_with_datasets(config)

    print_status()


if __name__ == "__main__":
    main()
