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

"""Generate pitch labels for MIR-1K dataset using librosa.pyin.

This script processes MIR-1K audio files and extracts F0 contours using
librosa's probabilistic YIN algorithm.
"""

import json
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm

MIR1K_DIR = Path("data/pitch/mir1k")
OUTPUT_FILE = MIR1K_DIR / "pitch_labels.json"

def extract_pitch(wav_path: str, sr: int = 16000) -> dict:
    """Extract pitch contour from audio file using pyin."""
    try:
        # Load audio
        audio, file_sr = librosa.load(wav_path, sr=sr)

        # Run pyin pitch detection
        # fmin=65 (C2), fmax=2093 (C7) covers human voice range
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),  # ~65 Hz
            fmax=librosa.note_to_hz('C7'),  # ~2093 Hz
            sr=sr,
            frame_length=2048,
            hop_length=160,  # 10ms at 16kHz
        )

        # Replace NaN with 0 for unvoiced frames
        f0_clean = np.nan_to_num(f0, nan=0.0)

        # Time array (10ms steps)
        times = librosa.times_like(f0, sr=sr, hop_length=160)

        return {
            "pitch_hz": f0_clean.tolist(),
            "pitch_confidence": voiced_probs.tolist(),
            "pitch_time": times.tolist(),
            "voiced_frames": int(np.sum(voiced_flag)),
            "total_frames": len(f0),
            "f0_mean": float(np.mean(f0_clean[f0_clean > 0])) if np.any(f0_clean > 0) else 0.0,
            "f0_std": float(np.std(f0_clean[f0_clean > 0])) if np.any(f0_clean > 0) else 0.0,
        }
    except Exception as e:
        return {
            "pitch_hz": [],
            "pitch_confidence": [],
            "pitch_time": [],
            "error": str(e),
        }


def main():
    print("Loading MIR-1K metadata...")

    with open(MIR1K_DIR / "metadata.json") as f:
        metadata = json.load(f)

    samples = metadata["samples"]
    print(f"Processing {len(samples)} audio files...")

    # Process each sample
    for i, sample in enumerate(tqdm(samples, desc="Extracting pitch")):
        wav_path = sample["path"]

        # Handle relative paths
        if not Path(wav_path).is_absolute():
            wav_path = MIR1K_DIR / Path(wav_path).name

        pitch_data = extract_pitch(str(wav_path))

        # Update sample with pitch data
        sample.update(pitch_data)

        if (i + 1) % 200 == 0:
            print(f"  Processed {i+1}/{len(samples)}, mean F0: {pitch_data.get('f0_mean', 0):.1f} Hz")

    # Save updated metadata
    output_data = {
        "samples": samples,
        "total": len(samples),
        "pitch_method": "librosa.pyin",
        "sr": 16000,
        "hop_length": 160,
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output_data, f, indent=2)

    # Stats
    voiced_counts = [s.get("voiced_frames", 0) for s in samples]
    total_counts = [s.get("total_frames", 0) for s in samples]
    f0_means = [s.get("f0_mean", 0) for s in samples if s.get("f0_mean", 0) > 0]

    print(f"\nSaved pitch labels to {OUTPUT_FILE}")
    print(f"Total samples: {len(samples)}")
    print(f"Average voiced ratio: {sum(voiced_counts)/sum(total_counts):.2%}")
    print(f"Average F0 (voiced): {np.mean(f0_means):.1f} Hz")


if __name__ == "__main__":
    main()
