#!/usr/bin/env python3
"""
Add prosody features to existing encoder cache files.

This script augments pre-extracted encoder features with prosody features
(F0, energy, deltas) without needing to re-run the expensive encoder forward pass.

Usage:
    python scripts/add_prosody_to_cache.py \
        --manifest data/v3_multitask/train_manifest.json \
        --cache-dir data/v3_multitask/encoder_cache

Time: ~37ms per sample (prosody extraction only)
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm


def get_cache_path(audio_path: str, cache_dir: Path) -> Path:
    """Get cache file path for audio."""
    cache_key = hashlib.sha256(audio_path.encode()).hexdigest()[:16]
    return cache_dir / cache_key[:2] / f"{cache_key}.npz"


def add_prosody_to_cache(
    manifest_path: str,
    cache_dir: str,
    sample_rate: int = 16000,
    max_audio_len: float = 30.0,
    skip_existing: bool = True,
):
    """
    Add prosody features to existing cache files.
    """
    from tools.whisper_mlx.audio import load_audio
    from tools.whisper_mlx.prosody_features import extract_prosody_features

    cache_dir = Path(cache_dir)

    # Load manifest
    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"Manifest: {len(manifest)} samples")
    print(f"Cache dir: {cache_dir}")

    # Process samples
    processed = 0
    skipped = 0
    errors = 0
    no_cache = 0

    start_time = time.time()

    for sample in tqdm(manifest, desc="Adding prosody"):
        audio_path = sample.get("audio_path", sample.get("path", ""))

        if not audio_path:
            errors += 1
            continue

        cache_path = get_cache_path(audio_path, cache_dir)

        # Check if cache file exists
        if not cache_path.exists():
            no_cache += 1
            continue

        try:
            # Load existing cache
            data = dict(np.load(cache_path))

            # Skip if prosody already exists
            if skip_existing and 'prosody' in data:
                skipped += 1
                continue

            # Load audio
            if not Path(audio_path).exists():
                errors += 1
                continue

            audio = load_audio(audio_path, sample_rate=sample_rate)

            # Truncate to max length
            target_samples = int(max_audio_len * sample_rate)
            audio = audio[:target_samples]

            # Extract prosody features
            prosody = extract_prosody_features(audio, sr=sample_rate)

            # Get encoder output length for alignment
            encoder_out = data['encoder_output']
            encoder_len = encoder_out.shape[0]

            # Align prosody to encoder frame rate
            from tools.whisper_mlx.prosody_features import align_prosody_to_encoder
            prosody = align_prosody_to_encoder(prosody, encoder_len)

            # Add prosody to cache
            data['prosody'] = prosody.astype(np.float16)

            # Save updated cache
            np.savez_compressed(cache_path, **data)

            processed += 1

        except Exception as e:
            print(f"Error: {audio_path}: {e}")
            errors += 1

    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print("Prosody augmentation complete!")
    print(f"  Processed: {processed}")
    print(f"  Skipped (existing): {skipped}")
    print(f"  No cache file: {no_cache}")
    print(f"  Errors: {errors}")
    if processed > 0:
        print(f"  Time: {elapsed:.1f}s ({elapsed/processed*1000:.1f}ms per sample)")


def main():
    parser = argparse.ArgumentParser(description="Add prosody to encoder cache")
    parser.add_argument("--manifest", required=True, help="Path to manifest JSON")
    parser.add_argument("--cache-dir", required=True, help="Encoder cache directory")
    parser.add_argument("--no-skip-existing", action="store_true",
                       help="Re-extract prosody even if already exists")

    args = parser.parse_args()

    add_prosody_to_cache(
        manifest_path=args.manifest,
        cache_dir=args.cache_dir,
        skip_existing=not args.no_skip_existing,
    )


if __name__ == "__main__":
    main()
