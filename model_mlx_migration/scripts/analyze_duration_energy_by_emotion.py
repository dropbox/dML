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
Analyze duration and energy by emotion from training data.

Computes data-driven multipliers for the Duration/Energy models
by analyzing the multilingual training data.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# Prosody type IDs (from prosody_types.h)
PROSODY_NAMES = {
    0: "NEUTRAL",
    40: "ANGRY",
    41: "SAD",
    42: "EXCITED",
    45: "CALM",
    48: "FRUSTRATED",
    49: "NERVOUS",
    50: "SURPRISED",
}


def analyze_data(data_path: Path):
    """Analyze duration and energy by emotion."""
    print(f"Loading data from {data_path}...")

    with open(data_path) as f:
        data = json.load(f)

    print(f"Total samples: {len(data)}")

    # Collect stats per emotion
    duration_by_emotion = defaultdict(list)
    energy_by_emotion = defaultdict(list)

    # Also collect text length for duration normalization
    duration_per_char_by_emotion = defaultdict(list)

    for sample in data:
        prosody_type = sample.get("prosody_type", 0)
        duration = sample.get("duration_s", 0)
        energy = sample.get("energy_rms", 0)
        text = sample.get("text", "")

        if prosody_type not in PROSODY_NAMES:
            continue

        if duration > 0 and energy > 0:
            duration_by_emotion[prosody_type].append(duration)
            energy_by_emotion[prosody_type].append(energy)

            # Duration per character (to normalize for text length)
            if len(text) > 0:
                duration_per_char_by_emotion[prosody_type].append(duration / len(text))

    # Compute statistics
    print("\n" + "=" * 80)
    print("Duration and Energy by Emotion")
    print("=" * 80)

    # Get neutral baseline
    neutral_dur = np.mean(duration_by_emotion[0]) if 0 in duration_by_emotion else 1.0
    neutral_energy = np.mean(energy_by_emotion[0]) if 0 in energy_by_emotion else 1.0
    neutral_dur_per_char = np.mean(duration_per_char_by_emotion[0]) if 0 in duration_per_char_by_emotion else 1.0

    results = {}

    print("\nNeutral baseline:")
    print(f"  Duration: {neutral_dur:.3f}s (mean)")
    print(f"  Energy: {neutral_energy:.6f} RMS")
    print(f"  Duration/char: {neutral_dur_per_char:.4f}s")

    print(f"\n{'Emotion':<12} {'Count':<8} {'Dur (s)':<10} {'Dur Mult':<10} {'Dur/Char Mult':<14} {'Energy RMS':<12} {'Energy Mult':<12}")
    print("-" * 88)

    for pid in sorted(PROSODY_NAMES.keys()):
        name = PROSODY_NAMES[pid]

        if pid not in duration_by_emotion:
            print(f"{name:<12} {'N/A':<8}")
            continue

        dur_list = duration_by_emotion[pid]
        energy_list = energy_by_emotion[pid]
        dur_per_char_list = duration_per_char_by_emotion[pid]

        count = len(dur_list)
        mean_dur = np.mean(dur_list)
        mean_energy = np.mean(energy_list)
        mean_dur_per_char = np.mean(dur_per_char_list) if dur_per_char_list else 0

        # Compute multipliers relative to neutral
        dur_mult = mean_dur / neutral_dur if neutral_dur > 0 else 1.0
        dur_per_char_mult = mean_dur_per_char / neutral_dur_per_char if neutral_dur_per_char > 0 else 1.0
        energy_mult = mean_energy / neutral_energy if neutral_energy > 0 else 1.0

        results[pid] = {
            "name": name,
            "count": count,
            "duration_mean": mean_dur,
            "duration_mult": dur_mult,
            "duration_per_char_mult": dur_per_char_mult,
            "energy_mean": mean_energy,
            "energy_mult": energy_mult,
        }

        print(f"{name:<12} {count:<8} {mean_dur:<10.3f} {dur_mult:<10.2f} {dur_per_char_mult:<14.2f} {mean_energy:<12.6f} {energy_mult:<12.2f}")

    # Print summary table in Python format
    print("\n" + "=" * 80)
    print("Data-Driven Multipliers (copy-paste format)")
    print("=" * 80)

    print("\n# Duration multipliers (normalized by text length)")
    print("DURATION_MULTIPLIERS = {")
    for pid in sorted(results.keys()):
        r = results[pid]
        print(f"    {pid}: {r['duration_per_char_mult']:.2f},  # {r['name']}")
    print("}")

    print("\n# Energy multipliers (RMS)")
    print("ENERGY_MULTIPLIERS = {")
    for pid in sorted(results.keys()):
        r = results[pid]
        print(f"    {pid}: {r['energy_mult']:.2f},  # {r['name']}")
    print("}")

    # Detailed analysis
    print("\n" + "=" * 80)
    print("Detailed Analysis")
    print("=" * 80)

    # Group by high/low arousal
    high_arousal = [40, 42, 49, 50]  # ANGRY, EXCITED, NERVOUS, SURPRISED
    low_arousal = [41, 48]  # SAD, FRUSTRATED
    neutral_like = [0, 45]  # NEUTRAL, CALM

    print("\nHigh Arousal (expect: shorter duration, higher energy):")
    for pid in high_arousal:
        if pid in results:
            r = results[pid]
            print(f"  {r['name']}: dur={r['duration_per_char_mult']:.2f}x, energy={r['energy_mult']:.2f}x")

    print("\nLow Arousal (expect: longer duration, lower energy):")
    for pid in low_arousal:
        if pid in results:
            r = results[pid]
            print(f"  {r['name']}: dur={r['duration_per_char_mult']:.2f}x, energy={r['energy_mult']:.2f}x")

    print("\nNeutral-Like (expect: 1.0x for both):")
    for pid in neutral_like:
        if pid in results:
            r = results[pid]
            print(f"  {r['name']}: dur={r['duration_per_char_mult']:.2f}x, energy={r['energy_mult']:.2f}x")

    return results


if __name__ == "__main__":
    # Try different data files
    data_files = [
        Path("data/prosody/multilingual_train.json"),
        Path("data/prosody/train.json"),
        Path("data/prosody/train_split.json"),
    ]

    for data_path in data_files:
        if data_path.exists():
            results = analyze_data(data_path)
            break
    else:
        print("No training data found!")
