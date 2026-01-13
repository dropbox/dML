#!/usr/bin/env python3
"""
Download datasets for multi-speaker separation and diarization.

Priority datasets:
1. MUSAN (~11G) - Noise augmentation
2. RIRS_NOISES (~1.2G) - Room impulse responses
3. VoxConverse (~30G) - Diarization benchmark
4. LibriMix (~100G) - Source separation training
5. LibriCSS (~13G) - Overlap evaluation

Usage:
    python scripts/download_separation_datasets.py --dataset musan
    python scripts/download_separation_datasets.py --dataset all
    python scripts/download_separation_datasets.py --list
"""

import argparse
import subprocess
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data"


DATASETS = {
    "musan": {
        "name": "MUSAN (Music, Speech, Noise)",
        "size": "11G",
        "url": "https://www.openslr.org/resources/17/musan.tar.gz",
        "output_dir": "augmentation/musan",
        "license": "CC BY 4.0",
        "priority": 1,
    },
    "rirs_noises": {
        "name": "RIRS_NOISES (Room Impulse Responses)",
        "size": "1.2G",
        "url": "https://www.openslr.org/resources/28/rirs_noises.zip",
        "output_dir": "augmentation/rirs_noises",
        "license": "Apache 2.0",
        "priority": 1,
    },
    "voxconverse": {
        "name": "VoxConverse (Diarization Benchmark)",
        "size": "30G",
        "url": "https://github.com/joonson/voxconverse",  # Git clone + download
        "output_dir": "diarization/voxconverse",
        "license": "CC BY 4.0",
        "priority": 2,
        "special": "voxconverse",
    },
    "libricss": {
        "name": "LibriCSS (Continuous Speech Separation)",
        "size": "13G",
        "url": "https://www.openslr.org/resources/131/libri_css.tar.gz",
        "output_dir": "overlap/libricss",
        "license": "CC BY 4.0",
        "priority": 2,
    },
    "librimix": {
        "name": "LibriMix (Source Separation)",
        "size": "100G",
        "url": "https://github.com/JorisCos/LibriMix",  # Requires generation
        "output_dir": "separation/librimix",
        "license": "CC BY 4.0",
        "priority": 1,
        "special": "librimix",
    },
}


def download_file(url: str, output_path: Path, description: str = ""):
    """Download a file using wget or curl."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Downloading: {description or url}")
    print(f"Output: {output_path}")
    print(f"{'='*60}\n")

    # Try wget first, fall back to curl
    if subprocess.run(["which", "wget"], capture_output=True).returncode == 0:
        cmd = ["wget", "-c", "--progress=bar:force", "-O", str(output_path), url]
    else:
        cmd = ["curl", "-L", "-C", "-", "-o", str(output_path), url]

    result = subprocess.run(cmd)
    return result.returncode == 0


def extract_archive(archive_path: Path, output_dir: Path):
    """Extract tar.gz or zip archive."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nExtracting {archive_path.name} to {output_dir}...")

    if archive_path.suffix == ".gz" or str(archive_path).endswith(".tar.gz"):
        cmd = ["tar", "-xzf", str(archive_path), "-C", str(output_dir)]
    elif archive_path.suffix == ".zip":
        cmd = ["unzip", "-o", str(archive_path), "-d", str(output_dir)]
    else:
        print(f"Unknown archive format: {archive_path}")
        return False

    result = subprocess.run(cmd)
    return result.returncode == 0


def download_musan():
    """Download MUSAN dataset."""
    info = DATASETS["musan"]
    output_dir = DATA_ROOT / info["output_dir"]
    archive_path = output_dir.parent / "musan.tar.gz"

    if (output_dir / "music").exists():
        print(f"MUSAN already exists at {output_dir}")
        return True

    if not archive_path.exists():
        if not download_file(info["url"], archive_path, info["name"]):
            return False

    if not extract_archive(archive_path, output_dir.parent):
        return False

    # Clean up archive
    archive_path.unlink()
    print(f"\nMUSAN downloaded to {output_dir}")
    return True


def download_rirs_noises():
    """Download RIRS_NOISES dataset."""
    info = DATASETS["rirs_noises"]
    output_dir = DATA_ROOT / info["output_dir"]
    archive_path = output_dir.parent / "rirs_noises.zip"

    if (output_dir / "real_rirs_isotropic_noises").exists():
        print(f"RIRS_NOISES already exists at {output_dir}")
        return True

    if not archive_path.exists():
        if not download_file(info["url"], archive_path, info["name"]):
            return False

    if not extract_archive(archive_path, output_dir.parent):
        return False

    archive_path.unlink()
    print(f"\nRIRS_NOISES downloaded to {output_dir}")
    return True


def download_libricss():
    """Download LibriCSS dataset."""
    info = DATASETS["libricss"]
    output_dir = DATA_ROOT / info["output_dir"]
    archive_path = output_dir.parent / "libri_css.tar.gz"

    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"LibriCSS already exists at {output_dir}")
        return True

    if not archive_path.exists():
        if not download_file(info["url"], archive_path, info["name"]):
            return False

    if not extract_archive(archive_path, output_dir.parent):
        return False

    archive_path.unlink()
    print(f"\nLibriCSS downloaded to {output_dir}")
    return True


def download_voxconverse():
    """Download VoxConverse dataset."""
    info = DATASETS["voxconverse"]
    output_dir = DATA_ROOT / info["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clone the repo for metadata
    repo_dir = output_dir / "voxconverse"
    if not repo_dir.exists():
        print("Cloning VoxConverse repository...")
        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/joonson/voxconverse.git",
            str(repo_dir)
        ])

    # Download audio files from VoxCeleb
    # VoxConverse uses VoxCeleb2 test set audio
    print("\nVoxConverse requires VoxCeleb2 audio files.")
    print("Audio URLs are in the repository. Manual download may be required.")
    print(f"Repository cloned to: {repo_dir}")
    print("\nTo complete setup:")
    print("1. Download VoxCeleb2 test set audio")
    print("2. Place in data/diarization/voxconverse/audio/")

    return True


def download_librimix():
    """Download/generate LibriMix dataset."""
    info = DATASETS["librimix"]
    output_dir = DATA_ROOT / info["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clone LibriMix repo
    repo_dir = output_dir / "LibriMix"
    if not repo_dir.exists():
        print("Cloning LibriMix repository...")
        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/JorisCos/LibriMix.git",
            str(repo_dir)
        ])

    print("\nLibriMix requires generation from LibriSpeech + WHAM noise.")
    print(f"Repository cloned to: {repo_dir}")
    print("\nTo generate LibriMix:")
    print("1. Ensure LibriSpeech is downloaded (data/LibriSpeech/)")
    print("2. Download WHAM noise: http://wham.whisper.ai/")
    print("3. Run: cd {repo_dir} && ./generate_librimix.sh")
    print("\nOr use pre-generated from HuggingFace:")
    print("  huggingface-cli download JorisCos/LibriMix --local-dir {output_dir}")

    # Try HuggingFace download
    print("\nAttempting HuggingFace download (Libri2Mix-16k-min)...")
    try:
        result = subprocess.run([
            "huggingface-cli", "download",
            "JorisCos/LibriMix",
            "--include", "Libri2Mix/wav16k/min/*",
            "--local-dir", str(output_dir / "hf_cache"),
        ], capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("HuggingFace download started. Check progress...")
        else:
            print(f"HuggingFace CLI not available or failed: {result.stderr}")
    except Exception as e:
        print(f"Could not start HuggingFace download: {e}")

    return True


def list_datasets():
    """List all available datasets."""
    print("\nAvailable Datasets for Multi-Speaker Pipeline:")
    print("=" * 70)

    for key, info in sorted(DATASETS.items(), key=lambda x: x[1]["priority"]):
        status = "PRIORITY" if info["priority"] == 1 else "PLANNED"
        print(f"\n[{status}] {info['name']}")
        print(f"  Key: {key}")
        print(f"  Size: {info['size']}")
        print(f"  License: {info['license']}")
        print(f"  Output: data/{info['output_dir']}")


def main():
    parser = argparse.ArgumentParser(description="Download separation/diarization datasets")
    parser.add_argument("--dataset", "-d", type=str,
                       help="Dataset to download (musan, rirs_noises, voxconverse, libricss, librimix, all)")
    parser.add_argument("--list", "-l", action="store_true", help="List available datasets")
    parser.add_argument("--priority-only", "-p", action="store_true",
                       help="Download only priority=1 datasets")
    args = parser.parse_args()

    if args.list:
        list_datasets()
        return

    if not args.dataset:
        parser.print_help()
        return

    download_funcs = {
        "musan": download_musan,
        "rirs_noises": download_rirs_noises,
        "libricss": download_libricss,
        "voxconverse": download_voxconverse,
        "librimix": download_librimix,
    }

    if args.dataset == "all":
        datasets_to_download = list(download_funcs.keys())
    elif args.dataset == "priority":
        datasets_to_download = [k for k, v in DATASETS.items() if v["priority"] == 1]
    elif args.dataset in download_funcs:
        datasets_to_download = [args.dataset]
    else:
        print(f"Unknown dataset: {args.dataset}")
        print(f"Available: {list(download_funcs.keys())}")
        return

    print(f"\nWill download: {datasets_to_download}")

    def parse_size(s):
        s = s.rstrip('G')
        try:
            return float(s)
        except ValueError:
            return 0

    total_size = sum(parse_size(DATASETS[d]['size']) for d in datasets_to_download)
    print(f"Total estimated size: ~{total_size:.1f}G")

    for dataset in datasets_to_download:
        print(f"\n{'#'*60}")
        print(f"# Downloading {dataset}")
        print(f"{'#'*60}")

        success = download_funcs[dataset]()
        if not success:
            print(f"Failed to download {dataset}")

    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
