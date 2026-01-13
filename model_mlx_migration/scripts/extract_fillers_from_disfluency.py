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
Extract filler segments from DisfluencySpeech dataset.

DisfluencySpeech annotates fillers with {F ... } tags in transcript_annotated.
We extract these segments for paralinguistics training.

Output: data/paralinguistics/fillers/ with .wav files and labels.json
"""

import json
import re
from pathlib import Path
from datasets import load_dataset
import numpy as np
import soundfile as sf

OUTPUT_DIR = Path("data/paralinguistics/fillers")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Regex to find filler markers: {F text, }
FILLER_PATTERN = re.compile(r'\{F\s+([^}]+)\}')

def extract_fillers():
    print("Loading DisfluencySpeech dataset...")
    ds = load_dataset('amaai-lab/DisfluencySpeech', split='train')

    samples = []
    saved = 0

    print(f"Processing {len(ds)} samples...")

    for idx, item in enumerate(ds):
        if idx % 500 == 0:
            print(f"  Progress: {idx}/{len(ds)}")

        transcript = item.get('transcript_annotated', '')

        # Find fillers in transcript
        fillers = FILLER_PATTERN.findall(transcript)

        if not fillers:
            continue

        # Get audio
        audio_data = item.get('audio', {})
        if not audio_data:
            continue

        try:
            # Handle AudioDecoder format
            if hasattr(audio_data, 'get_all_samples'):
                decoded = audio_data.get_all_samples()
                audio = decoded.data.numpy().flatten()
                sr = int(decoded.sample_rate)
            else:
                audio = np.array(audio_data.get('array', []))
                sr = audio_data.get('sampling_rate', 16000)

            if len(audio) == 0:
                continue

            # Resample to 16kHz if needed
            if sr != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                sr = 16000

            # Save full audio with filler annotation
            output_path = OUTPUT_DIR / f"filler_{saved:05d}.wav"
            sf.write(str(output_path), audio, sr)

            samples.append({
                "path": str(output_path),
                "fillers": fillers,
                "transcript": transcript,
                "label": "filler",
                "label_id": 8  # filler class
            })
            saved += 1

            if saved >= 5000:  # Limit to 5000 samples
                break

        except Exception:
            continue

    # Save metadata
    metadata_path = OUTPUT_DIR / "labels.json"
    with open(metadata_path, 'w') as f:
        json.dump(samples, f, indent=2)

    print(f"\nExtracted {saved} filler samples to {OUTPUT_DIR}")
    print(f"Metadata saved to {metadata_path}")

if __name__ == "__main__":
    extract_fillers()
