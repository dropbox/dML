#!/usr/bin/env python3
"""
FAST Pre-extract Whisper encoder features.

Optimizations over original:
1. TRUE BATCHING - process multiple samples per encoder call (4-8x speedup)
2. NO COMPRESSION - use np.savez instead of np.savez_compressed (2x speedup)
3. PARALLEL AUDIO LOADING - ThreadPool for I/O-bound audio loading

Expected: 22h -> ~3-4h
"""

import argparse
import json
import gc
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from tqdm import tqdm

import mlx.core as mx


def load_whisper_encoder(model_name: str = "mlx-community/whisper-large-v3-mlx"):
    """Load Whisper model and return encoder."""
    from tools.whisper_mlx.model import WhisperMLX
    print(f"Loading {model_name}...")
    model = WhisperMLX.from_pretrained(model_name)
    return model


def load_and_preprocess_audio(
    audio_path: str,
    max_audio_len: float = 30.0,
    sample_rate: int = 16000,
) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
    """Load audio and compute mel spectrogram (CPU-bound, can parallelize)."""
    from tools.whisper_mlx.audio import load_audio, log_mel_spectrogram

    try:
        audio = load_audio(audio_path, sample_rate=sample_rate)
        original_audio = audio.copy()

        target_samples = int(max_audio_len * sample_rate)
        actual_frames = min((len(audio) // 160) // 2, 1500)

        if len(audio) > target_samples:
            audio = audio[:target_samples]
        elif len(audio) < target_samples:
            audio = np.pad(audio, (0, target_samples - len(audio)), mode='constant')

        mel = log_mel_spectrogram(audio)
        return mel, original_audio[:target_samples], actual_frames
    except Exception:
        return None


def extract_prosody_single(audio: np.ndarray, sample_rate: int = 16000) -> Optional[np.ndarray]:
    """Extract prosody features for single audio (CPU-bound)."""
    try:
        from tools.whisper_mlx.prosody_features import extract_prosody_features
        return extract_prosody_features(audio, sample_rate=sample_rate, align_to_encoder=True)
    except:
        return None


def get_cache_path(audio_path: str, output_dir: Path) -> Path:
    """Get cache file path for audio."""
    cache_key = hashlib.sha256(audio_path.encode()).hexdigest()[:16]
    return output_dir / cache_key[:2] / f"{cache_key}.npz"


def preextract_features_fast(
    manifest_path: str,
    output_dir: str,
    model_name: str = "mlx-community/whisper-large-v3-mlx",
    batch_size: int = 8,  # TRUE batch size now!
    max_audio_len: float = 30.0,
    skip_existing: bool = True,
    extract_prosody: bool = True,
    num_workers: int = 4,  # Parallel audio loading
):
    """Fast pre-extraction with batching and parallelism."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest
    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"Manifest: {len(manifest)} samples")
    print(f"Batch size: {batch_size}")
    print(f"Workers: {num_workers}")

    # Filter to samples that need processing
    samples_to_process = []
    skipped = 0
    for sample in manifest:
        audio_path = sample.get("audio_path", sample.get("path", ""))
        if not audio_path:
            continue
        cache_path = get_cache_path(audio_path, output_dir)
        if skip_existing and cache_path.exists():
            skipped += 1
            continue
        if not Path(audio_path).exists():
            continue
        samples_to_process.append(sample)

    print(f"To process: {len(samples_to_process)} (skipped {skipped} cached)")

    if not samples_to_process:
        print("Nothing to process!")
        return

    # Load model
    model = load_whisper_encoder(model_name)

    processed = 0
    errors = 0
    start_time = time.time()

    # Process in batches
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for batch_start in tqdm(range(0, len(samples_to_process), batch_size), desc="Batches"):
            batch_samples = samples_to_process[batch_start:batch_start + batch_size]
            audio_paths = [s.get("audio_path", s.get("path", "")) for s in batch_samples]

            # PARALLEL: Load and preprocess audio
            futures = [
                executor.submit(load_and_preprocess_audio, path, max_audio_len)
                for path in audio_paths
            ]
            preprocess_results = [f.result() for f in futures]

            # Filter valid results
            valid_indices = []
            mels = []
            original_audios = []
            actual_frames_list = []

            for i, result in enumerate(preprocess_results):
                if result is not None:
                    mel, orig_audio, frames = result
                    valid_indices.append(i)
                    mels.append(mel)
                    original_audios.append(orig_audio)
                    actual_frames_list.append(frames)
                else:
                    errors += 1

            if not mels:
                continue

            # BATCH: Stack mels and run encoder ONCE
            mel_batch = mx.array(np.stack(mels))  # [B, 128, 3000]
            encoder_out_batch = model.encoder(mel_batch)  # [B, 1500, 1280]
            mx.eval(encoder_out_batch)  # Force computation

            # Convert to numpy
            encoder_out_np = np.array(encoder_out_batch).astype(np.float16)

            # PARALLEL: Extract prosody (CPU-bound)
            prosody_list = [None] * len(valid_indices)
            if extract_prosody:
                prosody_futures = [
                    executor.submit(extract_prosody_single, audio)
                    for audio in original_audios
                ]
                prosody_list = [f.result() for f in prosody_futures]

            # Save results (fast, no compression)
            for j, idx in enumerate(valid_indices):
                audio_path = audio_paths[idx]
                cache_path = get_cache_path(audio_path, output_dir)
                cache_path.parent.mkdir(parents=True, exist_ok=True)

                save_dict = {
                    'encoder_output': encoder_out_np[j],
                    'actual_frames': np.array(actual_frames_list[j]),
                }
                if prosody_list[j] is not None:
                    save_dict['prosody'] = prosody_list[j].astype(np.float16)

                # NO COMPRESSION - much faster!
                np.savez(cache_path, **save_dict)
                processed += 1

            # Clear GPU memory periodically
            if processed % 500 == 0:
                mx.clear_cache()  # Correct API (not mx.metal.clear_cache)
                gc.collect()

    elapsed = time.time() - start_time
    samples_per_sec = processed / max(elapsed, 1)

    print(f"\n{'='*60}")
    print("Pre-extraction complete!")
    print(f"  Processed: {processed}")
    print(f"  Skipped (cached): {skipped}")
    print(f"  Errors: {errors}")
    print(f"  Time: {elapsed:.1f}s ({samples_per_sec:.2f} samples/sec)")
    print(f"  Speedup vs 1.2s/sample: {1.2 * samples_per_sec:.1f}x")
    print(f"  Output: {output_dir}")

    total_size = sum(f.stat().st_size for f in output_dir.rglob("*.npz"))
    print(f"  Disk usage: {total_size / 1e9:.2f} GB")


def main():
    parser = argparse.ArgumentParser(description="FAST Pre-extract encoder features")
    parser.add_argument("--manifest", required=True, help="Path to manifest JSON")
    parser.add_argument("--output-dir", required=True, help="Output directory for cache")
    parser.add_argument("--model", default="mlx-community/whisper-large-v3-mlx")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for encoder")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers for audio loading")
    parser.add_argument("--max-audio-len", type=float, default=30.0)
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--no-prosody", action="store_true")

    args = parser.parse_args()

    preextract_features_fast(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        model_name=args.model,
        batch_size=args.batch_size,
        max_audio_len=args.max_audio_len,
        skip_existing=not args.no_skip_existing,
        extract_prosody=not args.no_prosody,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    main()
