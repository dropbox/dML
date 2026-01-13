#!/usr/bin/env python3
# Copyright 2024-2025 Andrew Yates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Download all datasets for multi-head training.

Supports:
    - Expresso: 40h, 34 expressive styles (HuggingFace)
    - RAVDESS: 24.8GB, 8 emotions + singing (Zenodo)
    - VocalSet: 2.1GB, 10h singing techniques (Zenodo)
    - MELD: 13k utterances, 7 emotions (University of Michigan)

Usage:
    # Download all datasets
    python scripts/download_datasets.py --all

    # Download specific datasets
    python scripts/download_datasets.py --expresso
    python scripts/download_datasets.py --ravdess
    python scripts/download_datasets.py --vocalset
    python scripts/download_datasets.py --meld
"""

import argparse
import zipfile
from pathlib import Path

try:
    import requests
    from tqdm import tqdm
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


def download_file(url: str, output_path: Path, chunk_size: int = 8192) -> bool:
    """Download a file with progress bar."""
    if not HAS_REQUESTS:
        print("ERROR: requests and tqdm required. Install: pip install requests tqdm")
        return False

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='iB', unit_scale=True, desc=output_path.name) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    size = f.write(chunk)
                    pbar.update(size)
        return True
    except Exception as e:
        print(f"ERROR downloading {url}: {e}")
        return False


def extract_zip(zip_path: Path, output_dir: Path):
    """Extract zip file."""
    print(f"Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(output_dir)
    print(f"  Extracted to {output_dir}")


def download_expresso(output_dir: str = "data/expresso"):
    """
    Download Expresso from HuggingFace.

    40 hours of professional studio recordings with 34 expressive styles.
    Automatically cached by HuggingFace datasets library.
    """
    print("\n" + "=" * 70)
    print("Downloading Expresso (34 expressive styles)")
    print("=" * 70)

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: datasets library required. Install: pip install datasets")
        return False

    print("Loading Expresso from HuggingFace...")
    print("This will cache ~5.76GB to your HuggingFace cache directory.")
    print("First download may take 10-30 minutes depending on connection.")
    print()

    try:
        # Load the read speech config (main config with styles)
        dataset = load_dataset("ylacombe/expresso", "read")
        print("Successfully loaded Expresso!")
        print(f"  Train samples: {len(dataset.get('train', []))}")
        print("  Cache location: ~/.cache/huggingface/datasets/")

        # Show style distribution
        styles = {}
        for split in dataset:
            for item in dataset[split]:
                style = item.get("style", "unknown")
                styles[style] = styles.get(style, 0) + 1

        print(f"\nStyles found: {len(styles)}")
        for style, count in sorted(styles.items(), key=lambda x: -x[1])[:10]:
            print(f"  {style}: {count}")

        return True
    except Exception as e:
        print(f"ERROR loading Expresso: {e}")
        print("Try: pip install datasets soundfile librosa")
        return False


def download_ravdess(output_dir: str = "data/ravdess"):
    """
    Download RAVDESS from Zenodo.

    24.8GB of professional actor recordings with 8 emotions + singing.
    """
    print("\n" + "=" * 70)
    print("Downloading RAVDESS (8 emotions + singing)")
    print("=" * 70)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    downloads_path = output_path / "downloads"
    downloads_path.mkdir(exist_ok=True)

    # RAVDESS download URLs
    base_url = "https://zenodo.org/record/1188976/files"
    files = [
        "Audio_Speech_Actors_01-24.zip",
        "Audio_Song_Actors_01-24.zip",
    ]

    success = True
    for filename in files:
        url = f"{base_url}/{filename}"
        zip_path = downloads_path / filename

        if zip_path.exists():
            print(f"Already downloaded: {filename}")
        else:
            print(f"Downloading: {filename}")
            if not download_file(url, zip_path):
                success = False
                continue

        # Extract
        extract_zip(zip_path, output_path)

    # Verify
    if success:
        wav_files = list(output_path.rglob("*.wav"))
        print(f"\nTotal audio files: {len(wav_files)}")

    return success


def download_vocalset(output_dir: str = "data/vocalset"):
    """
    Download VocalSet from Zenodo.

    2.1GB of professional singing with various techniques.
    """
    print("\n" + "=" * 70)
    print("Downloading VocalSet (singing techniques)")
    print("=" * 70)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    url = "https://zenodo.org/records/1193957/files/VocalSet.zip"
    zip_path = output_path / "VocalSet.zip"

    if zip_path.exists():
        print("Already downloaded: VocalSet.zip")
    else:
        print("Downloading VocalSet.zip (2.1GB)...")
        if not download_file(url, zip_path):
            return False

    # Extract
    extract_zip(zip_path, output_path)

    # Verify
    wav_files = list(output_path.rglob("*.wav"))
    print(f"\nTotal audio files: {len(wav_files)}")

    return True


def download_meld(output_dir: str = "data/meld"):
    """
    Download MELD (Friends TV show emotions).

    13k utterances with 7 emotions in conversational context.
    """
    print("\n" + "=" * 70)
    print("Downloading MELD (7 emotions in conversations)")
    print("=" * 70)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    url = "http://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz"
    tar_path = output_path / "MELD.Raw.tar.gz"

    if tar_path.exists():
        print("Already downloaded: MELD.Raw.tar.gz")
    else:
        print("Downloading MELD.Raw.tar.gz...")
        if not download_file(url, tar_path):
            # Try HuggingFace mirror
            print("Trying HuggingFace mirror...")
            url_hf = "https://huggingface.co/datasets/declare-lab/MELD/resolve/main/MELD.Raw.tar.gz"
            if not download_file(url_hf, tar_path):
                return False

    # Extract tar.gz
    import tarfile
    print(f"Extracting {tar_path.name}...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(output_path)

    print(f"Extracted to {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download datasets for multi-head training")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--expresso", action="store_true", help="Download Expresso (34 styles, HuggingFace)")
    parser.add_argument("--ravdess", action="store_true", help="Download RAVDESS (8 emotions + singing)")
    parser.add_argument("--vocalset", action="store_true", help="Download VocalSet (singing techniques)")
    parser.add_argument("--meld", action="store_true", help="Download MELD (7 emotions in conversations)")
    parser.add_argument("--output-dir", type=str, default="data", help="Base output directory")

    args = parser.parse_args()

    # If no specific dataset selected, show help
    if not any([args.all, args.expresso, args.ravdess, args.vocalset, args.meld]):
        parser.print_help()
        print("\n" + "=" * 70)
        print("RECOMMENDED: Start with Expresso (SOTA, automatic download)")
        print("=" * 70)
        print("\n  python scripts/download_datasets.py --expresso\n")
        return

    print("=" * 70)
    print("Multi-Head Training Dataset Downloader")
    print("=" * 70)

    results = {}

    if args.all or args.expresso:
        results["Expresso"] = download_expresso(f"{args.output_dir}/expresso")

    if args.all or args.ravdess:
        results["RAVDESS"] = download_ravdess(f"{args.output_dir}/ravdess")

    if args.all or args.vocalset:
        results["VocalSet"] = download_vocalset(f"{args.output_dir}/vocalset")

    if args.all or args.meld:
        results["MELD"] = download_meld(f"{args.output_dir}/meld")

    # Summary
    print("\n" + "=" * 70)
    print("Download Summary")
    print("=" * 70)
    for name, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  {name}: {status}")

    # Next steps
    print("\n" + "=" * 70)
    print("Next Steps: Train Multi-Head Model")
    print("=" * 70)

    if results.get("Expresso"):
        print("\n# RECOMMENDED: Train with Expresso (34 styles)")
        print("python -m tools.whisper_mlx.train_multi_head --expresso")

    if results.get("RAVDESS"):
        print("\n# Train with RAVDESS (8 emotions + singing)")
        print(f"python -m tools.whisper_mlx.train_multi_head --ravdess-dir {args.output_dir}/ravdess")

    if results.get("VocalSet"):
        print("\n# Add VocalSet for better singing detection")
        print(f"python -m tools.whisper_mlx.train_multi_head --vocalset-dir {args.output_dir}/vocalset")

    print()


if __name__ == "__main__":
    main()
