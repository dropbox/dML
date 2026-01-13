#!/usr/bin/env python3
"""
Pre-extract Whisper encoder features for RichDecoder training.

This eliminates the encoder forward pass during training (3-4x speedup).
The encoder is frozen anyway, so we can compute features once and reuse.

Now includes prosody feature extraction (F0, energy, deltas).

Usage:
    python scripts/preextract_encoder_features.py \
        --manifest data/v3_multitask/train_manifest.json \
        --output-dir data/v3_multitask/encoder_cache

Memory: ~4.5MB per sample (uncompressed), ~1.5MB (compressed)
Time: ~0.5s per sample on M1 Max (+ ~37ms for prosody)

Features:
- Pre-filters manifest (accurate progress bar, skips missing/cached)
- Lazy module imports (prosody, audio)
- Progress checkpointing to JSON with error tracking
- Incremental disk usage tracking
- Configurable compression (--compress for smaller files)
"""

import argparse
import hashlib
import json
import gc
import time
from pathlib import Path
import os
from typing import Tuple, Dict

import numpy as np
from tqdm import tqdm

import mlx.core as mx

# Lazy imports (at module level to avoid repeated imports)
_prosody_module = None
_audio_module = None

def _get_prosody_functions():
    """Lazy load prosody functions once."""
    global _prosody_module
    if _prosody_module is None:
        from tools.whisper_mlx import prosody_features
        _prosody_module = prosody_features
    return _prosody_module.extract_prosody_features, _prosody_module.align_prosody_to_encoder

def _get_audio_functions():
    """Lazy load audio functions once."""
    global _audio_module
    if _audio_module is None:
        from tools.whisper_mlx import audio
        _audio_module = audio
    return _audio_module.load_audio, _audio_module.log_mel_spectrogram


def load_whisper_encoder(model_name: str = "mlx-community/whisper-large-v3-mlx"):
    """Load Whisper model and return encoder."""
    from tools.whisper_mlx.model import WhisperMLX
    print(f"Loading {model_name}...")
    model = WhisperMLX.from_pretrained(model_name)
    return model


SAMPLE_RATE = 16000  # Whisper expects 16kHz audio

def extract_single_feature(
    model,
    audio_path: str,
    target_samples: int,
    extract_prosody: bool = True,
    load_audio_fn=None,
    log_mel_fn=None,
    extract_prosody_fn=None,
    align_prosody_fn=None,
) -> Tuple:
    """
    Extract encoder features and prosody for a single audio file.

    Returns:
        (encoder_output_np, actual_frames, prosody_features) tuple
        Returns (None, 0, None) on error
    """
    try:
        # Load audio
        audio = load_audio_fn(audio_path, sample_rate=SAMPLE_RATE)

        # Keep original audio for prosody extraction (only copy if needed)
        original_audio = audio.copy() if extract_prosody else None

        # Calculate actual frames before any padding/truncation
        # Whisper uses 160 hop_length, 2x downsampling in conv layers
        actual_frames = min(
            (len(audio) // 160) // 2,
            1500  # Max frames for 30s audio
        )

        # Pad or truncate to exactly 30s
        if len(audio) > target_samples:
            audio = audio[:target_samples]
        elif len(audio) < target_samples:
            audio = np.pad(audio, (0, target_samples - len(audio)), mode='constant')

        # Compute mel spectrogram (now always 30s)
        mel = log_mel_fn(audio)

        # Free audio array after mel computation (keep original_audio for prosody)
        del audio

        # Add batch dimension and run encoder
        mel = mx.expand_dims(mel, axis=0)
        encoder_out = model.encoder(mel)

        # Free mel array after encoder forward pass
        del mel

        encoder_out = encoder_out[0]

        # Force evaluation to prevent memory buildup from lazy ops
        mx.eval(encoder_out)

        # Convert to numpy once (avoid double conversion)
        encoder_np = np.array(encoder_out).astype(np.float16)

        # Free MLX array after numpy conversion
        del encoder_out

        # Extract prosody features if enabled
        prosody = None
        if extract_prosody and extract_prosody_fn is not None and original_audio is not None:
            prosody_raw = extract_prosody_fn(
                original_audio[:target_samples],
                sr=SAMPLE_RATE,
            )
            # Free original_audio after prosody extraction
            del original_audio
            # Align to encoder output length
            prosody = align_prosody_fn(prosody_raw, encoder_np.shape[0])
            del prosody_raw
        elif original_audio is not None:
            # Free original_audio if not used for prosody
            del original_audio

        return (encoder_np, actual_frames, prosody)

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return (None, 0, None)


def _save_progress(progress_file: Path, processed: int, skipped: int, errors: int,
                   error_files: list, start_time: float, final: bool = False):
    """Save extraction progress to JSON file."""
    elapsed = time.time() - start_time
    progress = {
        "processed": processed,
        "skipped": skipped,
        "errors": errors,
        "elapsed_seconds": round(elapsed, 1),
        "samples_per_second": round(processed / max(elapsed, 1), 2),
        "status": "complete" if final else "in_progress",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "error_files": error_files[-100:],  # Keep last 100 errors
        "total_errors_count": len(error_files),
    }
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)


def preextract_features(
    manifest_path: str,
    output_dir: str,
    model_name: str = "mlx-community/whisper-large-v3-mlx",
    max_audio_len: float = 30.0,
    skip_existing: bool = True,
    extract_prosody: bool = True,
    compress: bool = False,  # Uncompressed is faster
    limit: int = None,  # Limit samples for testing
) -> Dict[str, int]:
    """
    Pre-extract encoder features for all samples in manifest.

    Returns:
        Dict with keys: processed, skipped, errors
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Apply limit if specified
    if limit is not None:
        manifest = manifest[:limit]

    print(f"Manifest: {len(manifest)} samples")

    # Pre-create all cache subdirectories (avoid mkdir in loop)
    for i in range(256):
        subdir = output_dir / f"{i:02x}"
        subdir.mkdir(parents=True, exist_ok=True)

    # Compute target_samples once (constant for all files)
    target_samples = int(max_audio_len * SAMPLE_RATE)

    # Pre-filter: build list of (audio_path, cache_path) for files that need processing
    # This gives accurate progress bar and avoids duplicate existence checks
    to_process = []
    skipped = 0
    missing = 0
    error_files = []

    for sample in tqdm(manifest, desc="Pre-filtering", leave=False):
        audio_path = sample.get("audio_path", sample.get("path", ""))
        if not audio_path:
            error_files.append({"path": "", "error": "empty audio_path"})
            continue

        cache_key = hashlib.sha256(audio_path.encode()).hexdigest()[:16]
        cache_path = output_dir / cache_key[:2] / f"{cache_key}.npz"

        if skip_existing and cache_path.exists():
            skipped += 1
            continue

        if not os.path.exists(audio_path):
            missing += 1
            error_files.append({"path": audio_path, "error": "file not found"})
            continue

        to_process.append((audio_path, cache_path))

    print(f"  Already cached: {skipped}")
    print(f"  Missing files: {missing}")
    print(f"  To process: {len(to_process)}")

    if not to_process:
        print("Nothing to process!")
        return {"processed": 0, "skipped": skipped, "errors": len(error_files)}

    # Load model only if there's work to do
    model = load_whisper_encoder(model_name)

    # Get audio/prosody functions once
    load_audio_fn, log_mel_fn = _get_audio_functions()
    extract_prosody_fn, align_prosody_fn = (None, None)
    if extract_prosody:
        extract_prosody_fn, align_prosody_fn = _get_prosody_functions()

    # Process samples
    processed = 0
    errors = len(error_files)  # Start with pre-filter errors
    total_bytes = 0  # Track disk usage incrementally
    start_time = time.time()
    progress_file = output_dir / "extraction_progress.json"

    # Choose save function
    save_fn = np.savez_compressed if compress else np.savez

    for audio_path, cache_path in tqdm(to_process, desc="Extracting features"):
        encoder_np, actual_frames, prosody = extract_single_feature(
            model=model,
            audio_path=audio_path,
            target_samples=target_samples,
            extract_prosody=extract_prosody,
            load_audio_fn=load_audio_fn,
            log_mel_fn=log_mel_fn,
            extract_prosody_fn=extract_prosody_fn,
            align_prosody_fn=align_prosody_fn,
        )

        if encoder_np is None:
            errors += 1
            error_files.append({"path": audio_path, "error": "extraction failed"})
            continue

        # Build save dict (actual_frames is scalar, no need for np.array)
        save_dict = {
            'encoder_output': encoder_np,
            'actual_frames': actual_frames,
        }
        if prosody is not None:
            save_dict['prosody'] = prosody.astype(np.float16)

        save_fn(cache_path, **save_dict)
        processed += 1

        # Track file size incrementally
        try:
            total_bytes += cache_path.stat().st_size
        except OSError:
            pass

        # Clear memory periodically (every 100 for more aggressive cleanup)
        if processed % 100 == 0:
            if hasattr(mx, 'clear_cache'):
                mx.clear_cache()
            gc.collect()
            # Save progress every 500 (less frequent I/O)
            if processed % 500 == 0:
                _save_progress(progress_file, processed, skipped, errors, error_files, start_time)

    elapsed = time.time() - start_time

    # Save final progress
    _save_progress(progress_file, processed, skipped, errors, error_files, start_time, final=True)

    print(f"\n{'='*60}")
    print("Pre-extraction complete!")
    print(f"  Processed: {processed}")
    print(f"  Skipped (cached): {skipped}")
    print(f"  Errors: {errors}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/max(processed,1):.2f}s per sample)")
    print(f"  Output: {output_dir}")
    print(f"  Progress file: {progress_file}")
    print(f"  New data written: {total_bytes / 1e9:.2f} GB")

    # Report errors if any
    if error_files:
        print("\nFirst 10 errors:")
        for err in error_files[:10]:
            print(f"  {err['error']}: {err['path'][:80]}")

    return {"processed": processed, "skipped": skipped, "errors": errors}


def main():
    parser = argparse.ArgumentParser(description="Pre-extract encoder features")
    parser.add_argument("--manifest", required=True, help="Path to manifest JSON")
    parser.add_argument("--output-dir", required=True, help="Output directory for cache")
    parser.add_argument("--model", default="mlx-community/whisper-large-v3-mlx")
    parser.add_argument("--max-audio-len", type=float, default=30.0)
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--no-prosody", action="store_true",
                       help="Skip prosody feature extraction")
    parser.add_argument("--compress", action="store_true",
                       help="Use compressed npz (slower but smaller)")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of samples to process (for testing)")

    args = parser.parse_args()

    preextract_features(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        model_name=args.model,
        max_audio_len=args.max_audio_len,
        skip_existing=not args.no_skip_existing,
        extract_prosody=not args.no_prosody,
        compress=args.compress,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
