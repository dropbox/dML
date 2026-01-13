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
Debug script to analyze VAD probabilities around chunk 312 for file 0004.
This is where C++ and Python Silero diverge causing different segment detection.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from tools.whisper_mlx.audio import load_audio

TEST_FILE = "data/librispeech/dev-clean/1272/128104/1272-128104-0004.flac"


def main():
    if not os.path.exists(TEST_FILE):
        print(f"Test file not found: {TEST_FILE}")
        return

    # Load audio
    audio = load_audio(TEST_FILE, sample_rate=16000)
    print(f"Audio: {len(audio)} samples = {len(audio)/16000:.2f}s")

    # Load Silero model
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False
    )

    # Process in 512-sample chunks
    chunk_size = 512
    num_chunks = len(audio) // chunk_size
    print(f"Total chunks: {num_chunks}")

    # Parameters matching C++
    threshold = 0.5
    neg_threshold = threshold - 0.15  # 0.35

    print("\nThresholds:")
    print(f"  speech_threshold: {threshold}")
    print(f"  neg_threshold (end trigger): {neg_threshold}")

    # Segment 2 ends around 10.046s = chunk ~313
    # Let's look at chunks 300-330
    print(f"\n{'='*70}")
    print("VAD Probabilities around chunk 312 (segment 2 end region)")
    print(f"{'='*70}")
    print(f"{'Chunk':>6} | {'Time':>8} | {'Prob':>8} | {'> thr':>6} | {'> neg':>6} | Notes")
    print(f"{'-'*70}")

    model.reset_states()
    probs = []

    for i in range(num_chunks):
        chunk = audio[i*chunk_size:(i+1)*chunk_size]
        chunk_tensor = torch.tensor(chunk)
        prob = model(chunk_tensor, 16000).item()
        probs.append(prob)

        # Print detailed info for chunks 300-330
        if 300 <= i <= 330:
            time_s = i * chunk_size / 16000
            above_thresh = "YES" if prob >= threshold else "no"
            above_neg = "YES" if prob >= neg_threshold else "NO"

            notes = ""
            if i == 312:
                notes = "<<< CRITICAL (C++: 0.3485, Py: 0.3527)"
            elif 0.34 <= prob <= 0.36:
                notes = "Near neg_threshold boundary"

            print(f"{i:>6} | {time_s:>7.3f}s | {prob:>8.4f} | {above_thresh:>6} | {above_neg:>6} | {notes}")

    # Analyze where segment 2 would end
    print(f"\n{'='*70}")
    print("Segment End Detection Analysis")
    print(f"{'='*70}")

    # Find where prob first drops below neg_threshold after segment 2 starts (~chunk 168)
    in_speech = False
    seg2_start = 168  # approximately where seg 2 starts
    temp_end_chunk = None

    for i in range(seg2_start, min(num_chunks, 350)):
        prob = probs[i]
        if prob >= threshold:
            in_speech = True
        elif in_speech and prob < neg_threshold and temp_end_chunk is None:
            temp_end_chunk = i
            time_s = i * chunk_size / 16000
            print(f"\nPython temp_end would be set at chunk {i} ({time_s:.3f}s)")
            print(f"  Probability: {prob:.4f}")
            print(f"  neg_threshold: {neg_threshold}")
            break

    # Check what C++ gets
    print("\nC++ (from report 1501):")
    print(f"  Chunk 312 probability: 0.3485 (< {neg_threshold}) → triggers temp_end")
    print(f"  Python chunk 312 probability: {probs[312]:.4f} ({'>=' if probs[312] >= neg_threshold else '<'} {neg_threshold})")

    diff = probs[312] - 0.3485
    print(f"\nProbability difference at chunk 312: {diff:.4f}")

    # Suggested fix
    print(f"\n{'='*70}")
    print("Suggested Fix")
    print(f"{'='*70}")
    if diff > 0:
        suggested_margin = diff + 0.001
        new_neg_threshold = neg_threshold - suggested_margin
        print(f"Add margin to neg_threshold: {neg_threshold} → {new_neg_threshold:.4f}")
        print("This would make C++ behavior match Python at chunk 312")

    # Verify with adjusted threshold
    print(f"\n{'='*70}")
    print("Verification with adjusted neg_threshold")
    print(f"{'='*70}")

    for margin in [0.005, 0.01, 0.015]:
        adj_neg = neg_threshold - margin
        cpp_312 = 0.3485
        py_312 = probs[312]
        cpp_triggers = cpp_312 < adj_neg
        py_triggers = py_312 < adj_neg

        status = "MATCH" if cpp_triggers == py_triggers else "DIVERGE"
        print(f"margin={margin:.3f}: adj_neg={adj_neg:.4f} | C++={'triggers' if cpp_triggers else 'no'}, Py={'triggers' if py_triggers else 'no'} → {status}")


if __name__ == "__main__":
    main()
