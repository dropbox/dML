#!/usr/bin/env python3.12
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

"""Generate F0 labels for MIR-1K dataset using CREPE."""
import crepe
import numpy as np
from pathlib import Path
import json
import librosa
from tqdm import tqdm
import sys

MIR1K_DIR = Path("/Users/ayates/model_mlx_migration/data/pitch/mir1k")
OUTPUT_FILE = MIR1K_DIR / "pitch_labels.json"

print("=" * 50)
print("Generating F0 labels for MIR-1K with CREPE")
print("=" * 50)

# Find all wav files
wav_files = sorted(MIR1K_DIR.glob("*.wav"))
print(f"Found {len(wav_files)} wav files")

samples = []
errors = []

for i, wav_path in enumerate(tqdm(wav_files, desc="Processing")):
    try:
        # Load audio at 16kHz
        audio, sr = librosa.load(str(wav_path), sr=16000)

        # Run CREPE (tiny model for speed)
        time, frequency, confidence, _ = crepe.predict(
            audio, sr,
            model_capacity='tiny',
            viterbi=True,
            step_size=10,  # 10ms
            verbose=0
        )

        # Calculate stats for verification
        valid_f0 = frequency[confidence > 0.5]

        sample = {
            "path": str(wav_path),
            "f0_hz": frequency.tolist(),
            "f0_confidence": confidence.tolist(),
            "f0_time": time.tolist(),
            "duration_sec": len(audio) / sr,
            "mean_f0": float(np.mean(valid_f0)) if len(valid_f0) > 0 else 0.0,
            "voiced_ratio": float(len(valid_f0) / len(frequency)) if len(frequency) > 0 else 0.0
        }
        samples.append(sample)

        # Progress update every 100 files
        if (i + 1) % 100 == 0:
            print(f"\n  Processed {i+1}/{len(wav_files)} files")
            print(f"  Last file: mean F0 = {sample['mean_f0']:.1f} Hz, voiced = {sample['voiced_ratio']*100:.1f}%")
            sys.stdout.flush()

    except Exception as e:
        errors.append({"path": str(wav_path), "error": str(e)})
        print(f"\nError on {wav_path.name}: {e}")

# Save results
result = {
    "total": len(samples),
    "errors": len(errors),
    "samples": samples
}

with open(OUTPUT_FILE, 'w') as f:
    json.dump(result, f)

print("\n" + "=" * 50)
print(f"COMPLETE: Saved to {OUTPUT_FILE}")
print(f"  Samples with F0: {len(samples)}")
print(f"  Errors: {len(errors)}")

# Stats
if samples:
    mean_f0s = [s['mean_f0'] for s in samples if s['mean_f0'] > 0]
    voiced_ratios = [s['voiced_ratio'] for s in samples]
    print(f"  Average mean F0: {np.mean(mean_f0s):.1f} Hz")
    print(f"  Average voiced ratio: {np.mean(voiced_ratios)*100:.1f}%")
print("=" * 50)
