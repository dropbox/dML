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
Populate encoder cache from pre-extracted LibriSpeech features.

Converts LibriSpeech encoder features (.npz files) into the encoder cache
format expected by train_rich_decoder_v3.py. This allows pseudo-labeled
LibriSpeech samples to be used with the existing training infrastructure.

The encoder cache uses SHA256 hashes of audio paths as keys. This script:
1. Maps utterance IDs to original audio file paths
2. Computes the cache key (SHA256 hash)
3. Copies features to the correct cache location

Usage:
    python -m tools.whisper_mlx.populate_encoder_cache \
        --features-dir data/emotion_punctuation/librispeech_features \
        --audio-dir data/LibriSpeech/train-clean-100 \
        --cache-dir data/v3_multitask/encoder_cache \
        --limit 1000
"""

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np


def get_cache_key(audio_path: str) -> str:
    """Generate cache key from audio path using SHA256 hash."""
    return hashlib.sha256(audio_path.encode()).hexdigest()[:16]


def get_cache_path(cache_dir: Path, cache_key: str) -> Path:
    """Get file path for cache key."""
    # Use 2-level directory structure to avoid too many files in one dir
    return cache_dir / cache_key[:2] / f"{cache_key}.npz"


def find_audio_path(utterance_id: str, audio_dir: Path) -> Path | None:
    """Find the audio file path for a given utterance ID.

    LibriSpeech utterance IDs have format: speaker-chapter-utterance
    e.g., 103-1240-0000 -> 103/1240/103-1240-0000.flac
    """
    parts = utterance_id.split("-")
    if len(parts) != 3:
        return None

    speaker, chapter, _ = parts
    audio_path = audio_dir / speaker / chapter / f"{utterance_id}.flac"

    if audio_path.exists():
        return audio_path
    return None


def populate_cache(
    features_dir: str,
    audio_dir: str,
    cache_dir: str,
    limit: int | None = None,
) -> dict:
    """
    Populate encoder cache from pre-extracted features.

    Args:
        features_dir: Directory with pre-extracted .npz features
        audio_dir: Directory with original LibriSpeech audio files
        cache_dir: Target encoder cache directory
        limit: Optional limit on number of files to process

    Returns:
        Statistics dictionary
    """
    features_path = Path(features_dir)
    audio_path = Path(audio_dir)
    cache_path = Path(cache_dir)

    # Find all feature files
    npz_files = sorted(features_path.glob("*.npz"))
    if limit:
        npz_files = npz_files[:limit]

    total = len(npz_files)
    print(f"Processing {total} feature files...")
    print(f"  Features dir: {features_path}")
    print(f"  Audio dir: {audio_path}")
    print(f"  Cache dir: {cache_path}")

    stats = {
        "total": total,
        "processed": 0,
        "cached": 0,
        "skipped_audio_not_found": 0,
        "skipped_already_cached": 0,
        "errors": 0,
    }

    for i, npz_file in enumerate(npz_files):
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{total} ({stats['cached']} cached)")

        try:
            # Get utterance ID from filename
            utterance_id = npz_file.stem

            # Find original audio path
            audio_file = find_audio_path(utterance_id, audio_path)
            if audio_file is None:
                stats["skipped_audio_not_found"] += 1
                continue

            # Generate cache key and path
            cache_key = get_cache_key(str(audio_file))
            cache_file = get_cache_path(cache_path, cache_key)

            # Skip if already cached
            if cache_file.exists():
                stats["skipped_already_cached"] += 1
                stats["processed"] += 1
                continue

            # Load features
            data = np.load(npz_file)
            encoder_features = data["encoder_features"]

            # Create cache directory
            cache_file.parent.mkdir(parents=True, exist_ok=True)

            # Save to cache in the expected format
            # The cache expects: encoder_out (T, 1280), seq_len (int)
            np.savez_compressed(
                cache_file,
                encoder_out=encoder_features,
                seq_len=encoder_features.shape[0],
            )

            stats["cached"] += 1
            stats["processed"] += 1

        except Exception as e:
            print(f"  Error processing {npz_file.name}: {e}")
            stats["errors"] += 1
            continue

    print("\nCache population complete:")
    print(f"  Total files: {stats['total']}")
    print(f"  Processed: {stats['processed']}")
    print(f"  Newly cached: {stats['cached']}")
    print(f"  Skipped (audio not found): {stats['skipped_audio_not_found']}")
    print(f"  Skipped (already cached): {stats['skipped_already_cached']}")
    print(f"  Errors: {stats['errors']}")

    return stats


def create_manifest(
    pseudo_labels_path: str,
    audio_dir: str,
    output_manifest: str,
) -> dict:
    """
    Create training manifest from pseudo-labels.

    Converts pseudo-label JSON to training manifest format with audio paths.

    Args:
        pseudo_labels_path: Path to pseudo-label JSON
        audio_dir: Directory with original LibriSpeech audio files
        output_manifest: Path to save training manifest

    Returns:
        Statistics dictionary
    """
    import json

    audio_path = Path(audio_dir)

    with open(pseudo_labels_path) as f:
        pseudo_labels = json.load(f)

    print(f"Converting {len(pseudo_labels)} pseudo-labels to manifest...")

    manifest = []
    stats = {
        "total": len(pseudo_labels),
        "converted": 0,
        "skipped_audio_not_found": 0,
    }

    for item in pseudo_labels:
        # Get utterance ID from the feature file path
        features_path = Path(item["path"])
        utterance_id = features_path.stem

        # Find original audio path
        audio_file = find_audio_path(utterance_id, audio_path)
        if audio_file is None:
            stats["skipped_audio_not_found"] += 1
            continue

        # Create manifest entry
        manifest.append({
            "audio_path": str(audio_file),
            "text": item.get("transcript", ""),
            "emotion": item["emotion"],
            "para": "speech",  # Default to speech
            "language": "en",  # LibriSpeech is English
            "source": "pseudo_librispeech",
            "confidence": item.get("confidence", 0.0),
        })
        stats["converted"] += 1

    # Save manifest
    output_path = Path(output_manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("\nManifest creation complete:")
    print(f"  Total pseudo-labels: {stats['total']}")
    print(f"  Converted: {stats['converted']}")
    print(f"  Skipped (audio not found): {stats['skipped_audio_not_found']}")
    print(f"  Saved to: {output_manifest}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Populate encoder cache from pre-extracted LibriSpeech features",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Cache population command
    cache_parser = subparsers.add_parser("cache", help="Populate encoder cache")
    cache_parser.add_argument(
        "--features-dir",
        type=str,
        required=True,
        help="Directory with pre-extracted .npz features",
    )
    cache_parser.add_argument(
        "--audio-dir",
        type=str,
        required=True,
        help="Directory with original LibriSpeech audio files",
    )
    cache_parser.add_argument(
        "--cache-dir",
        type=str,
        required=True,
        help="Target encoder cache directory",
    )
    cache_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of files to process (for testing)",
    )

    # Manifest creation command
    manifest_parser = subparsers.add_parser("manifest", help="Create training manifest")
    manifest_parser.add_argument(
        "--pseudo-labels",
        type=str,
        required=True,
        help="Path to pseudo-label JSON file",
    )
    manifest_parser.add_argument(
        "--audio-dir",
        type=str,
        required=True,
        help="Directory with original LibriSpeech audio files",
    )
    manifest_parser.add_argument(
        "--output-manifest",
        type=str,
        required=True,
        help="Path to save training manifest",
    )

    args = parser.parse_args()

    if args.command == "cache":
        populate_cache(
            features_dir=args.features_dir,
            audio_dir=args.audio_dir,
            cache_dir=args.cache_dir,
            limit=args.limit,
        )
    elif args.command == "manifest":
        create_manifest(
            pseudo_labels_path=args.pseudo_labels,
            audio_dir=args.audio_dir,
            output_manifest=args.output_manifest,
        )
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
