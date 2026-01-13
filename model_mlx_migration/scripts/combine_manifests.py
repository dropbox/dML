#!/usr/bin/env python3
"""
Combine training manifest with pseudo-labeled LibriSpeech manifest.

This script creates a merged manifest for V4 training that includes:
1. Original labeled data (emotion, paralinguistics, etc.)
2. Pseudo-labeled LibriSpeech samples (high-confidence predictions)

The combined manifest uses a unified format compatible with train_rich_decoder_v3.py.

Usage:
    python scripts/combine_manifests.py \
        --train-manifest data/v3_multitask/train_manifest.json \
        --pseudo-manifest data/pseudo_labels/librispeech_manifest.json \
        --output data/v4_combined/train_manifest.json \
        --min-confidence 0.9

Note: Pseudo-labeled samples point to pre-extracted features, not audio files.
The training script must support loading from both audio paths and feature paths.
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def load_manifest(path: str) -> list:
    """Load JSON manifest file."""
    with open(path) as f:
        return json.load(f)


def convert_pseudo_to_train_format(pseudo_entry: dict, emotion_classes: list) -> dict:
    """
    Convert pseudo-label entry to training manifest format.

    Pseudo format:
        path: str (path to .npz features)
        utterance_id: str
        emotion_id: int
        emotion_label: str
        confidence: float
        transcript: str
        speaker: str
        duration_s: float
        source: str

    Training format:
        audio_path: str (or features_path for pre-extracted)
        text: str
        emotion: str
        para: str
        language: str
        source: str
    """
    return {
        "features_path": pseudo_entry["path"],  # Pre-extracted features
        "text": pseudo_entry.get("transcript", ""),
        "emotion": pseudo_entry["emotion_label"],
        "para": "speech",  # LibriSpeech is always speech
        "language": "en",  # LibriSpeech is English
        "source": pseudo_entry["source"],
        "confidence": pseudo_entry.get("confidence", 1.0),  # Keep confidence for filtering
        "is_pseudo_label": True,  # Mark as pseudo-label
    }


def analyze_distribution(entries: list, key: str = "emotion") -> dict:
    """Analyze class distribution for a key."""
    counts = Counter(entry.get(key) for entry in entries)
    total = sum(counts.values())
    return {k: (v, v / total * 100) for k, v in sorted(counts.items(), key=lambda x: -x[1])}


def main():
    parser = argparse.ArgumentParser(description="Combine training and pseudo-labeled manifests")
    parser.add_argument(
        "--train-manifest",
        type=str,
        default="data/v3_multitask/train_manifest.json",
        help="Path to original training manifest",
    )
    parser.add_argument(
        "--pseudo-manifest",
        type=str,
        default="data/pseudo_labels/librispeech_manifest.json",
        help="Path to pseudo-labeled manifest",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/v4_combined/train_manifest.json",
        help="Output path for combined manifest",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.9,
        help="Minimum confidence threshold for pseudo-labels (default: 0.9)",
    )
    parser.add_argument(
        "--max-pseudo",
        type=int,
        default=None,
        help="Maximum number of pseudo-labels to include (default: all)",
    )
    parser.add_argument(
        "--balance-classes",
        action="store_true",
        help="Balance pseudo-labels to match original class distribution",
    )

    args = parser.parse_args()

    # Load manifests
    print(f"Loading training manifest: {args.train_manifest}")
    train_entries = load_manifest(args.train_manifest)
    print(f"  Loaded {len(train_entries)} training samples")

    print(f"\nLoading pseudo-label manifest: {args.pseudo_manifest}")
    pseudo_entries = load_manifest(args.pseudo_manifest)
    print(f"  Loaded {len(pseudo_entries)} pseudo-labeled samples")

    # Filter pseudo-labels by confidence
    filtered_pseudo = [
        e for e in pseudo_entries
        if e.get("confidence", 0) >= args.min_confidence
    ]
    print(f"  After confidence filter (>= {args.min_confidence}): {len(filtered_pseudo)} samples")

    # Optionally limit number of pseudo-labels
    if args.max_pseudo and len(filtered_pseudo) > args.max_pseudo:
        # Sort by confidence and take top N
        filtered_pseudo = sorted(filtered_pseudo, key=lambda x: -x.get("confidence", 0))[:args.max_pseudo]
        print(f"  Limited to top {args.max_pseudo} by confidence")

    # Convert pseudo-labels to training format
    converted_pseudo = [convert_pseudo_to_train_format(e, []) for e in filtered_pseudo]

    # Analyze original distribution
    print("\nOriginal training distribution (emotion):")
    train_dist = analyze_distribution(train_entries, "emotion")
    for emotion, (count, pct) in train_dist.items():
        print(f"  {emotion}: {count} ({pct:.1f}%)")

    # Analyze pseudo-label distribution
    print("\nPseudo-label distribution (emotion):")
    pseudo_dist = analyze_distribution(converted_pseudo, "emotion")
    for emotion, (count, pct) in pseudo_dist.items():
        print(f"  {emotion}: {count} ({pct:.1f}%)")

    # Combine manifests
    combined = train_entries + converted_pseudo

    # Analyze combined distribution
    print("\nCombined distribution (emotion):")
    combined_dist = analyze_distribution(combined, "emotion")
    for emotion, (count, pct) in combined_dist.items():
        print(f"  {emotion}: {count} ({pct:.1f}%)")

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save combined manifest
    print(f"\nSaving combined manifest: {args.output}")
    with open(output_path, "w") as f:
        json.dump(combined, f, indent=2)

    print("\nSummary:")
    print(f"  Original training samples: {len(train_entries)}")
    print(f"  Pseudo-labeled samples: {len(converted_pseudo)}")
    print(f"  Combined total: {len(combined)}")
    print(f"  Data increase: {len(converted_pseudo) / len(train_entries) * 100:.1f}%")

    # Save stats
    stats_path = output_path.parent / f"{output_path.stem}_stats.json"
    with open(stats_path, "w") as f:
        json.dump({
            "train_samples": len(train_entries),
            "pseudo_samples": len(converted_pseudo),
            "combined_samples": len(combined),
            "confidence_threshold": args.min_confidence,
            "train_distribution": {k: v[0] for k, v in train_dist.items()},
            "pseudo_distribution": {k: v[0] for k, v in pseudo_dist.items()},
            "combined_distribution": {k: v[0] for k, v in combined_dist.items()},
        }, f, indent=2)
    print(f"Saved statistics to {stats_path}")


if __name__ == "__main__":
    main()
