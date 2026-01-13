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
Fast parallel F0 extraction using joblib multiprocessing.
Uses librosa's yin (faster than pyin) with parallel processing.
"""

import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm
import librosa
from joblib import Parallel, delayed

SAMPLE_RATE = 16000
HOP_LENGTH = 160  # 10ms at 16kHz
N_JOBS = 8  # Use 8 parallel workers


def extract_f0_yin(wav_path: Path) -> dict:
    """Extract F0 contour using yin (faster than pyin)."""
    try:
        audio, sr = librosa.load(wav_path, sr=SAMPLE_RATE)

        if len(audio) < SAMPLE_RATE * 0.3:  # Skip < 0.3s
            return None

        # Use yin (faster than pyin, still good quality)
        f0 = librosa.yin(
            audio,
            fmin=librosa.note_to_hz('C2'),  # ~65 Hz
            fmax=librosa.note_to_hz('C7'),  # ~2093 Hz
            sr=sr,
            hop_length=HOP_LENGTH,
        )

        # yin returns inf for unvoiced, replace with 0
        f0 = np.where(np.isinf(f0) | np.isnan(f0), 0, f0)

        # Skip if mostly unvoiced
        voiced_ratio = np.sum(f0 > 0) / len(f0)
        if voiced_ratio < 0.2:
            return None

        # Get time axis
        times = librosa.times_like(f0, sr=sr, hop_length=HOP_LENGTH)

        # Get stats for voiced regions
        f0_nonzero = f0[f0 > 0]
        if len(f0_nonzero) < 10:
            return None

        f0_min = float(np.min(f0_nonzero))
        f0_max = float(np.max(f0_nonzero))

        # Normalize: map f0 range to 0-1, keep 0 as unvoiced
        f0_normalized = np.zeros_like(f0)
        voiced_mask = f0 > 0
        if f0_max > f0_min:
            f0_normalized[voiced_mask] = (f0[voiced_mask] - f0_min) / (f0_max - f0_min)
        else:
            f0_normalized[voiced_mask] = 0.5

        return {
            'audio_path': str(wav_path),
            'f0_contour': f0_normalized.tolist(),
            'f0_hz': f0.tolist(),
            'f0_time': times.tolist(),
            'f0_min': f0_min,
            'f0_max': f0_max,
            'duration_s': len(audio) / sr,
            'voiced_ratio': float(voiced_ratio),
            'prosody_type': 14,  # Match MIR-1K format
            'text': '',
            'emotion': 'neutral',
        }
    except Exception:
        return None


def process_files_parallel(audio_files: list, desc: str = "Extracting F0") -> list:
    """Process files in parallel using joblib."""
    results = Parallel(n_jobs=N_JOBS, verbose=0)(
        delayed(extract_f0_yin)(f) for f in tqdm(audio_files, desc=desc)
    )
    return [r for r in results if r is not None]


def process_vocalset():
    """Extract F0 from VocalSet."""
    input_dir = Path("data/singing/vocalset/FULL")
    output_dir = Path("data/pitch/vocalset_f0")

    if not input_dir.exists():
        print(f"VocalSet not found at {input_dir}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    audio_files = list(input_dir.rglob("*.wav"))
    print(f"Found {len(audio_files)} VocalSet files")

    samples = process_files_parallel(audio_files, "VocalSet F0")

    # Save
    train = samples[:int(len(samples) * 0.9)]
    val = samples[int(len(samples) * 0.9):]

    with open(output_dir / "contours_train.json", 'w') as f:
        json.dump(train, f)
    with open(output_dir / "contours_val.json", 'w') as f:
        json.dump(val, f)

    print(f"VocalSet: {len(samples)} samples (train={len(train)}, val={len(val)})")
    return len(samples)


def process_librispeech(max_samples=3000):
    """Extract F0 from LibriSpeech."""
    input_dir = Path("data/benchmarks/librispeech/LibriSpeech")
    output_dir = Path("data/pitch/libri_f0")

    # Try different potential locations
    train_dir = input_dir / "train-clean-100"
    if not train_dir.exists():
        train_dir = input_dir / "dev-clean"
    if not train_dir.exists():
        print("LibriSpeech training data not found")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    audio_files = list(train_dir.rglob("*.flac"))
    print(f"Found {len(audio_files)} LibriSpeech files")

    # Limit samples
    if len(audio_files) > max_samples:
        import random
        random.seed(42)
        audio_files = random.sample(audio_files, max_samples)
        print(f"Sampled {len(audio_files)} files")

    samples = process_files_parallel(audio_files, "LibriSpeech F0")

    # Save
    train = samples[:int(len(samples) * 0.9)]
    val = samples[int(len(samples) * 0.9):]

    with open(output_dir / "contours_train.json", 'w') as f:
        json.dump(train, f)
    with open(output_dir / "contours_val.json", 'w') as f:
        json.dump(val, f)

    print(f"LibriSpeech: {len(samples)} samples (train={len(train)}, val={len(val)})")
    return len(samples)


def main():
    print("=" * 60)
    print("Fast Parallel F0 Extraction (yin + joblib)")
    print(f"Using {N_JOBS} parallel workers")
    print("=" * 60)

    total = 0

    # Process VocalSet by default
    if "--vocalset" in sys.argv or "--all" in sys.argv or len(sys.argv) == 1:
        total += process_vocalset()

    if "--librispeech" in sys.argv or "--all" in sys.argv:
        total += process_librispeech()

    print(f"\nTotal samples extracted: {total}")


if __name__ == "__main__":
    main()
