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

"""Create unified singing dataset from multiple sources."""

import json
from pathlib import Path
from collections import defaultdict

OUTPUT_DIR = Path("data/singing/unified_singing")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Track sources
sources = defaultdict(int)
samples = []

# 1. Vocalset FULL (main singing dataset)
vocalset_dir = Path("data/singing/FULL")
if vocalset_dir.exists():
    for wav_file in vocalset_dir.rglob("*.wav"):
        samples.append({
            "path": str(wav_file),
            "source": "vocalset",
            "label": "singing",
            "label_id": 1
        })
        sources["vocalset"] += 1
    print(f"Vocalset: {sources['vocalset']} files")
else:
    print("Vocalset FULL not found")

# 2. RAVDESS Song (emotional singing)
ravdess_dir = Path("data/singing/ravdess_song")
if ravdess_dir.exists():
    for wav_file in ravdess_dir.rglob("*.wav"):
        samples.append({
            "path": str(wav_file),
            "source": "ravdess_song",
            "label": "singing",
            "label_id": 1
        })
        sources["ravdess_song"] += 1
    print(f"RAVDESS Song: {sources['ravdess_song']} files")
else:
    print("RAVDESS Song not found")

# 3. Vocalset from data/vocalset/FULL (skip - duplicate of data/singing/FULL)
# vocalset_alt = Path("data/vocalset/FULL")
# Both paths contain the same VocalSet data, so we skip to avoid duplicates

# 4. LibriSpeech for speech contrast (limit to 2000)
librispeech_dir = Path("data/LibriSpeech/dev-clean")
if librispeech_dir.exists():
    speech_files = list(librispeech_dir.rglob("*.flac"))[:2000]
    for audio_file in speech_files:
        samples.append({
            "path": str(audio_file),
            "source": "librispeech",
            "label": "speech",
            "label_id": 0
        })
        sources["librispeech"] += 1
    print(f"LibriSpeech (speech): {sources['librispeech']} files")
else:
    print("LibriSpeech not found")

# Calculate totals
singing_count = sum(1 for s in samples if s["label"] == "singing")
speech_count = sum(1 for s in samples if s["label"] == "speech")

print(f"\nTotal: {len(samples)} samples")
print(f"  Singing: {singing_count}")
print(f"  Speech: {speech_count}")

# Save metadata
metadata = {
    "total_samples": len(samples),
    "singing_samples": singing_count,
    "speech_samples": speech_count,
    "sources": dict(sources),
    "label_map": {"speech": 0, "singing": 1}
}

with open(OUTPUT_DIR / "metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"\nSaved metadata to {OUTPUT_DIR}/metadata.json")

with open(OUTPUT_DIR / "samples.json", 'w') as f:
    json.dump(samples, f)
print(f"Saved samples to {OUTPUT_DIR}/samples.json")

print(f"\nUnified singing dataset created: {len(samples)} total samples")
