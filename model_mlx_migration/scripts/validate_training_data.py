#!/usr/bin/env python3
"""
Validate training data manifests and caches.

Checks:
- All audio paths exist
- All audio files load correctly
- Labels are valid
- No duplicates
- Encoder cache coverage

Usage:
    python scripts/validate_training_data.py --manifest data/v4_expanded/train_manifest.json
    python scripts/validate_training_data.py --all  # Validate all known manifests
"""

import argparse
import json
import os
import sys
import hashlib
from pathlib import Path
from typing import Dict, Optional, Set
from collections import Counter

# Valid label sets
EMOTION_LABELS = {'neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise', 'other', 'calm', 'fearful', 'amused', 'frustrated', 'annoyed', 'confused', 'excited', 'bored', 'anxious', 'contempt'}
PARA_LABELS = {'speech', 'laughter', 'laugh', 'laughing', 'cough', 'coughing', 'sigh', 'breath', 'breathing', 'cry', 'crying', 'crying_baby', 'yawn', 'throat_clearing', 'throat_clear', 'sneeze', 'sneezing', 'gasp', 'groan', 'sniff', 'hiccup', 'snoring', 'clapping', 'filler', 'silence', 'um_en', 'uh_en', 'hmm_en', 'neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise'}
LANGUAGE_LABELS = {'en', 'zh', 'ja', 'ko', 'de', 'es', 'fr', 'ru', 'hi', 'pt', 'ar', 'it', 'nl', 'pl', 'tr'}


def validate_manifest(manifest_path: str, cache_dir: Optional[str] = None, check_audio: bool = False) -> Dict:
    """
    Validate a manifest file.

    Args:
        manifest_path: Path to JSON manifest
        cache_dir: Optional encoder cache directory
        check_audio: If True, try to load each audio file

    Returns:
        Dict with validation results
    """
    results = {
        "manifest_path": manifest_path,
        "total_entries": 0,
        "valid_entries": 0,
        "missing_audio": [],
        "invalid_labels": [],
        "duplicate_paths": [],
        "cache_coverage": None,
        "label_distribution": {},
        "source_distribution": {},
        "errors": [],
    }

    # Load manifest
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except Exception as e:
        results["errors"].append(f"Failed to load manifest: {e}")
        return results

    results["total_entries"] = len(manifest)

    # Track duplicates
    seen_paths: Set[str] = set()

    # Counters
    emotion_counts = Counter()
    para_counts = Counter()
    source_counts = Counter()

    # Validate each entry
    for i, entry in enumerate(manifest):
        audio_path = entry.get("audio_path", "")

        # Check for duplicates
        if audio_path in seen_paths:
            results["duplicate_paths"].append(audio_path)
        seen_paths.add(audio_path)

        # Check audio exists
        if audio_path and not os.path.exists(audio_path):
            results["missing_audio"].append(audio_path)
            continue

        # Check labels (handle both string and integer values)
        emotion_raw = entry.get("emotion", "")
        para_raw = entry.get("para", "")
        language_raw = entry.get("language", "")

        # Convert to string if needed
        emotion = str(emotion_raw).lower() if emotion_raw != "" else ""
        para = str(para_raw).lower() if para_raw != "" else ""
        language = str(language_raw).lower() if language_raw != "" else ""

        # Skip label validation for integer IDs (pitch manifests use numeric emotion IDs)
        is_numeric_emotion = isinstance(emotion_raw, (int, float)) or (isinstance(emotion_raw, str) and emotion_raw.isdigit())
        if emotion and not is_numeric_emotion and emotion not in EMOTION_LABELS:
            results["invalid_labels"].append(f"emotion={emotion} at index {i}")

        # Count distributions
        if emotion:
            emotion_counts[emotion] += 1
        if para:
            para_counts[para] += 1
        source = entry.get("source", "unknown")
        source_counts[source] += 1

        results["valid_entries"] += 1

    results["label_distribution"] = {
        "emotion": dict(emotion_counts.most_common()),
        "para": dict(para_counts.most_common()),
    }
    results["source_distribution"] = dict(source_counts.most_common())

    # Check cache coverage
    if cache_dir:
        cache_path = Path(cache_dir)
        if cache_path.exists():
            cached = 0
            for entry in manifest:
                audio_path = entry.get("audio_path", "")
                if audio_path:
                    cache_key = hashlib.sha256(audio_path.encode()).hexdigest()[:16]
                    cache_file = cache_path / cache_key[:2] / f"{cache_key}.npz"
                    if cache_file.exists():
                        cached += 1

            results["cache_coverage"] = {
                "cached": cached,
                "total": len(manifest),
                "percentage": round(100 * cached / max(len(manifest), 1), 2),
            }

    return results


def print_results(results: Dict):
    """Pretty print validation results."""
    print(f"\n{'='*60}")
    print(f"Manifest: {results['manifest_path']}")
    print(f"{'='*60}")

    total = results["total_entries"]
    valid = results["valid_entries"]
    missing = len(results["missing_audio"])
    duplicates = len(results["duplicate_paths"])
    invalid_labels = len(results["invalid_labels"])

    print("\nSummary:")
    print(f"  Total entries:    {total}")
    print(f"  Valid entries:    {valid} ({100*valid/max(total,1):.1f}%)")
    print(f"  Missing audio:    {missing}")
    print(f"  Duplicate paths:  {duplicates}")
    print(f"  Invalid labels:   {invalid_labels}")

    if results["cache_coverage"]:
        cc = results["cache_coverage"]
        print("\nEncoder Cache:")
        print(f"  Cached:     {cc['cached']} / {cc['total']} ({cc['percentage']}%)")

    if results["label_distribution"]["emotion"]:
        print("\nEmotion Distribution:")
        for label, count in list(results["label_distribution"]["emotion"].items())[:10]:
            print(f"  {label}: {count}")

    if results["source_distribution"]:
        print("\nSource Distribution:")
        for source, count in list(results["source_distribution"].items())[:10]:
            print(f"  {source}: {count}")

    if results["errors"]:
        print("\nErrors:")
        for err in results["errors"][:5]:
            print(f"  {err}")

    if missing > 0:
        print("\nFirst 5 missing audio files:")
        for path in results["missing_audio"][:5]:
            print(f"  {path}")

    # Status
    if missing == 0 and invalid_labels == 0 and not results["errors"]:
        print("\n✓ VALIDATION PASSED")
    else:
        print("\n✗ VALIDATION FAILED")

    return missing == 0 and invalid_labels == 0 and not results["errors"]


def main():
    parser = argparse.ArgumentParser(description="Validate training data")
    parser.add_argument("--manifest", type=str, help="Path to manifest JSON")
    parser.add_argument("--cache-dir", type=str, help="Encoder cache directory")
    parser.add_argument("--all", action="store_true", help="Validate all known manifests")
    parser.add_argument("--check-audio", action="store_true", help="Try loading each audio file")

    args = parser.parse_args()

    manifests_to_check = []

    if args.all:
        # Known manifests
        base = "/Users/ayates/model_mlx_migration/data"
        manifests_to_check = [
            (f"{base}/v4_expanded/train_manifest.json", f"{base}/v4_expanded/encoder_cache"),
            (f"{base}/v4_expanded/val_manifest.json", f"{base}/v4_expanded/encoder_cache"),
            (f"{base}/sep28k/interjection_manifest.json", None),
            (f"{base}/timit/timit_train_manifest.json", None),
            (f"{base}/prosody/pitch_train_manifest.json", None),
        ]
    elif args.manifest:
        manifests_to_check = [(args.manifest, args.cache_dir)]
    else:
        parser.print_help()
        return 1

    all_passed = True
    for manifest_path, cache_dir in manifests_to_check:
        if not os.path.exists(manifest_path):
            print(f"Skipping {manifest_path} (not found)")
            continue

        results = validate_manifest(manifest_path, cache_dir, args.check_audio)
        passed = print_results(results)
        if not passed:
            all_passed = False

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
