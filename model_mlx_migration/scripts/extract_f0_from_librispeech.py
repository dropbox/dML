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
Extract F0 contours from LibriSpeech using CREPE.

This expands the pitch training dataset beyond MIR-1K.
Target: 5000+ additional samples with F0 labels.
"""

import json
from pathlib import Path

from tqdm import tqdm

# Try to import CREPE
try:
    import crepe
    HAS_CREPE = True
except ImportError:
    HAS_CREPE = False
    print("WARNING: CREPE not installed. Run: pip install crepe")

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("WARNING: librosa not installed. Run: pip install librosa")


INPUT_DIR = Path("data/LibriSpeech_full/train-clean-100")
OUTPUT_DIR = Path("data/pitch/libri_f0")
MAX_SAMPLES = 5000
SAMPLE_RATE = 16000


def extract_f0(wav_path: Path) -> dict:
    """Extract F0 contour from a wav file using CREPE."""
    try:
        # Load audio
        audio, sr = librosa.load(wav_path, sr=SAMPLE_RATE)

        if len(audio) < SAMPLE_RATE * 0.5:  # Skip files < 0.5s
            return None

        # Run CREPE (use 'tiny' for speed, 'full' for accuracy)
        time, frequency, confidence, _ = crepe.predict(
            audio, sr,
            model_capacity='tiny',
            viterbi=True,
            step_size=10  # 10ms hop
        )

        # Filter low-confidence predictions
        frequency[confidence < 0.5] = 0  # Mark as unvoiced

        return {
            'path': str(wav_path),
            'f0_hz': frequency.tolist(),
            'f0_time': time.tolist(),
            'f0_confidence': confidence.tolist(),
            'duration': len(audio) / sr,
        }
    except Exception as e:
        print(f"Error processing {wav_path}: {e}")
        return None


def main():
    if not HAS_CREPE or not HAS_LIBROSA:
        print("Missing dependencies. Install with:")
        print("  pip install crepe librosa")
        return

    if not INPUT_DIR.exists():
        print(f"LibriSpeech not found at {INPUT_DIR}")
        print("Download with: python3 -m tools.whisper_mlx.download_librispeech")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find all wav/flac files
    audio_files = list(INPUT_DIR.rglob("*.flac")) + list(INPUT_DIR.rglob("*.wav"))
    print(f"Found {len(audio_files)} audio files")

    # Limit to MAX_SAMPLES
    if len(audio_files) > MAX_SAMPLES:
        import random
        random.seed(42)
        audio_files = random.sample(audio_files, MAX_SAMPLES)

    print(f"Processing {len(audio_files)} files...")

    samples = []

    # Process files (sequential for CREPE GPU usage)
    for wav_path in tqdm(audio_files, desc="Extracting F0"):
        result = extract_f0(wav_path)
        if result:
            samples.append(result)

    # Save metadata
    metadata = {
        'source': 'LibriSpeech train-clean-100',
        'total_samples': len(samples),
        'sample_rate': SAMPLE_RATE,
        'f0_method': 'CREPE tiny',
        'samples': samples,
    }

    output_file = OUTPUT_DIR / "metadata.json"
    with open(output_file, 'w') as f:
        json.dump(metadata, f)

    print(f"\nExtracted F0 for {len(samples)} files")
    print(f"Saved to {output_file}")

    # Create train/val split
    train_samples = samples[:int(len(samples) * 0.9)]
    val_samples = samples[int(len(samples) * 0.9):]

    with open(OUTPUT_DIR / "contours_train.json", 'w') as f:
        json.dump(train_samples, f)

    with open(OUTPUT_DIR / "contours_val.json", 'w') as f:
        json.dump(val_samples, f)

    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")


if __name__ == "__main__":
    main()
