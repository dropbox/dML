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
Extract F0 contours from VocalSet singing data using CREPE.

VocalSet has professional singers with clear pitch - excellent for pitch training.
"""

import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

try:
    import crepe
    import librosa
except ImportError:
    print("Install dependencies: pip install crepe librosa")
    exit(1)


INPUT_DIR = Path("data/singing/vocalset/FULL")
OUTPUT_DIR = Path("data/pitch/vocalset_f0")
SAMPLE_RATE = 16000


def extract_f0(wav_path: Path) -> dict:
    """Extract F0 contour from a wav file."""
    try:
        audio, sr = librosa.load(wav_path, sr=SAMPLE_RATE)

        if len(audio) < SAMPLE_RATE * 0.3:
            return None

        time, frequency, confidence, _ = crepe.predict(
            audio, sr,
            model_capacity='tiny',
            viterbi=True,
            step_size=10
        )

        # Singing usually has high confidence - use stricter threshold
        frequency[confidence < 0.6] = 0

        # Skip if mostly unvoiced
        voiced_ratio = np.sum(frequency > 0) / len(frequency)
        if voiced_ratio < 0.3:
            return None

        return {
            'path': str(wav_path),
            'f0_hz': frequency.tolist(),
            'f0_time': time.tolist(),
            'f0_confidence': confidence.tolist(),
            'duration': len(audio) / sr,
            'voiced_ratio': voiced_ratio,
        }
    except Exception as e:
        print(f"Error: {wav_path}: {e}")
        return None


def main():
    if not INPUT_DIR.exists():
        print(f"VocalSet not found at {INPUT_DIR}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    audio_files = list(INPUT_DIR.rglob("*.wav"))
    print(f"Found {len(audio_files)} audio files in VocalSet")

    samples = []
    for wav_path in tqdm(audio_files, desc="Extracting F0 from VocalSet"):
        result = extract_f0(wav_path)
        if result:
            samples.append(result)

    # Save
    metadata = {
        'source': 'VocalSet',
        'total_samples': len(samples),
        'sample_rate': SAMPLE_RATE,
        'samples': samples,
    }

    with open(OUTPUT_DIR / "metadata.json", 'w') as f:
        json.dump(metadata, f)

    # Train/val split
    train = samples[:int(len(samples) * 0.9)]
    val = samples[int(len(samples) * 0.9):]

    with open(OUTPUT_DIR / "contours_train.json", 'w') as f:
        json.dump(train, f)
    with open(OUTPUT_DIR / "contours_val.json", 'w') as f:
        json.dump(val, f)

    print(f"\nExtracted {len(samples)} samples")
    print(f"Train: {len(train)}, Val: {len(val)}")


if __name__ == "__main__":
    main()
