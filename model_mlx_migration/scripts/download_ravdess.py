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
Download RAVDESS dataset for emotion/singing training.

RAVDESS: Ryerson Audio-Visual Database of Emotional Speech and Song
- 24 professional actors (12 female, 12 male)
- 8 emotions (speech): neutral, calm, happy, sad, angry, fearful, disgust, surprise
- 5 emotions (song): calm, happy, sad, angry, fearful
- Both speech and SONG audio!

Source: https://zenodo.org/record/1188976
License: CC BY-NC-SA 4.0 (non-commercial)

Usage:
    python scripts/download_ravdess.py
    python scripts/download_ravdess.py --output-dir data/ravdess
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


# RAVDESS download URLs (audio-only versions)
# Each actor has their own zip file on Zenodo
RAVDESS_BASE_URL = "https://zenodo.org/record/1188976/files"

# Actor files (Audio_Speech_Actors_01-24.zip and Audio_Song_Actors_01-24.zip)
RAVDESS_FILES = [
    # Speech files
    "Audio_Speech_Actors_01-24.zip",
    # Song files (for singing detection!)
    "Audio_Song_Actors_01-24.zip",
]

# Alternative: individual actor files if main files don't work
ACTOR_FILES = {
    "speech": [f"Actor_{i:02d}.zip" for i in range(1, 25)],
    "song": [f"Actor_{i:02d}.zip" for i in range(1, 25)],
}


def download_file(url: str, output_path: Path, chunk_size: int = 8192):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))

    with open(output_path, 'wb') as f:
        with tqdm(total=total_size, unit='iB', unit_scale=True, desc=output_path.name) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                size = f.write(chunk)
                pbar.update(size)


def extract_zip(zip_path: Path, output_dir: Path):
    """Extract zip file."""
    print(f"Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(output_dir)
    print(f"  Extracted to {output_dir}")


def download_ravdess(output_dir: str = "data/ravdess"):
    """Download and extract RAVDESS dataset."""
    if not HAS_REQUESTS:
        print("ERROR: requests and tqdm required. Install: pip install requests tqdm")
        return False

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    downloads_path = output_path / "downloads"
    downloads_path.mkdir(exist_ok=True)

    print("=" * 70)
    print("RAVDESS Dataset Download")
    print("=" * 70)
    print(f"Output: {output_path}")
    print()

    # Download main files
    success = True
    for filename in RAVDESS_FILES:
        url = f"{RAVDESS_BASE_URL}/{filename}"
        zip_path = downloads_path / filename

        if zip_path.exists():
            print(f"Already downloaded: {filename}")
        else:
            print(f"Downloading: {filename}")
            try:
                download_file(url, zip_path)
            except Exception as e:
                print(f"  ERROR: {e}")
                success = False
                continue

        # Extract
        extract_zip(zip_path, output_path)

    # Verify
    print()
    print("=" * 70)
    print("Verification")
    print("=" * 70)

    wav_files = list(output_path.rglob("*.wav"))
    print(f"Total audio files: {len(wav_files)}")

    # Count by type
    speech_files = [f for f in wav_files if "Speech" in str(f) or "-01-" in f.name]
    song_files = [f for f in wav_files if "Song" in str(f) or "-02-" in f.name]
    print(f"Speech files: {len(speech_files)}")
    print(f"Song files: {len(song_files)}")

    # Count emotions
    emotion_map = {1: "neutral", 2: "calm", 3: "happy", 4: "sad",
                   5: "angry", 6: "fearful", 7: "disgust", 8: "surprise"}
    emotion_counts = {e: 0 for e in emotion_map.values()}

    for f in wav_files:
        parts = f.stem.split("-")
        if len(parts) >= 3:
            try:
                emotion_id = int(parts[2])
                if emotion_id in emotion_map:
                    emotion_counts[emotion_map[emotion_id]] += 1
            except ValueError:
                pass

    print("\nEmotion distribution:")
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: -x[1]):
        print(f"  {emotion}: {count}")

    return success


def main():
    parser = argparse.ArgumentParser(description="Download RAVDESS dataset")
    parser.add_argument("--output-dir", type=str, default="data/ravdess",
                       help="Output directory")
    args = parser.parse_args()

    success = download_ravdess(args.output_dir)

    if success:
        print()
        print("=" * 70)
        print("Download Complete!")
        print("=" * 70)
        print(f"Dataset ready at: {args.output_dir}")
        print()
        print("To train multi-head model:")
        print("  python -m tools.whisper_mlx.train_multi_head \\")
        print(f"      --ravdess-dir {args.output_dir} \\")
        print("      --output-dir checkpoints/multi_head_large_v3")
    else:
        print()
        print("Some downloads failed. Please retry or download manually from:")
        print("  https://zenodo.org/record/1188976")


if __name__ == "__main__":
    main()
