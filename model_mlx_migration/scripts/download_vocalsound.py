#!/usr/bin/env python3
"""
Download VocalSound dataset with proper labels.

VocalSound has 6 classes: laughter, sigh, cough, throat_clearing, sneeze, sniff
Labels are encoded in filenames: f0003_0_cough.wav

Source: https://github.com/YuanGongND/vocalsound
"""

import json
from pathlib import Path

DATA_ROOT = Path("/Users/ayates/model_mlx_migration/data")
OUTPUT_DIR = DATA_ROOT / "paralinguistics/vocalsound_labeled"

# VocalSound class mapping
VOCALSOUND_CLASSES = {
    'laughter': 'laugh',
    'sigh': 'sigh',
    'cough': 'cough',
    'throat_clearing': 'throat_clear',
    'sneeze': 'sneeze',
    'sniff': 'sniff',
}

def extract_label_from_filename(filename: str) -> str:
    """
    Extract label from VocalSound filename.
    Format: f0003_0_cough.wav -> cough
    """
    # Remove extension
    name = Path(filename).stem
    # Split by underscore, last part is label
    parts = name.split('_')
    if len(parts) >= 3:
        label = parts[-1].lower()
        return VOCALSOUND_CLASSES.get(label, label)
    return 'unknown'


def scan_existing_files():
    """
    Check if original VocalSound files exist anywhere.
    """
    search_dirs = [
        DATA_ROOT / "paralinguistics/vocalsound",
        DATA_ROOT / "paralinguistics/vocalsound_extra",
        DATA_ROOT / "downloads",
        Path("/Users/ayates/Downloads"),
    ]

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        # Look for files with original naming pattern
        for wav in search_dir.rglob("*.wav"):
            name = wav.name
            # Original format: f0003_0_cough.wav or m0001_0_sigh.wav
            if '_' in name and not name.startswith('test_'):
                parts = name.replace('.wav', '').split('_')
                if len(parts) >= 3:
                    label = parts[-1].lower()
                    if label in VOCALSOUND_CLASSES or label in VOCALSOUND_CLASSES.values():
                        print(f"Found original VocalSound file: {wav}")
                        return search_dir

    return None


def create_manifest_from_original(source_dir: Path):
    """
    Create manifest from original VocalSound files.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    samples = []
    label_counts = {}

    for wav in source_dir.rglob("*.wav"):
        if wav.name.startswith('test_') or 'unknown' in wav.name:
            continue

        label = extract_label_from_filename(wav.name)
        if label == 'unknown':
            continue

        samples.append({
            'audio_path': str(wav),
            'text': '',
            'emotion': 'neutral',  # Default emotion
            'para': label,
            'language': 'en',
            'source': 'vocalsound',
        })

        label_counts[label] = label_counts.get(label, 0) + 1

    print(f"\nFound {len(samples)} VocalSound samples:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")

    # Save manifest
    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(samples, f, indent=2)

    print(f"\nSaved manifest to {manifest_path}")
    return samples


def main():
    print("Searching for original VocalSound files...")

    source_dir = scan_existing_files()

    if source_dir:
        print(f"\nFound original files in: {source_dir}")
        samples = create_manifest_from_original(source_dir)
    else:
        print("\n" + "="*60)
        print("ORIGINAL VOCALSOUND FILES NOT FOUND")
        print("="*60)
        print("""
The VocalSound files were renamed during download and labels are lost.

To get VocalSound with labels, download from:
  https://github.com/YuanGongND/vocalsound

Steps:
1. Download the dataset (requires registration)
2. Extract to: data/paralinguistics/vocalsound_labeled/
3. Re-run this script to create manifest

Expected filename format: f0003_0_cough.wav
- f/m = female/male
- 0003 = speaker ID
- 0 = recording index
- cough = class label (laughter, sigh, cough, throat_clearing, sneeze, sniff)
""")


if __name__ == "__main__":
    main()
