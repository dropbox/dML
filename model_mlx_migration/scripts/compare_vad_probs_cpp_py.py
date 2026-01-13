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
Compare C++ and Python VAD probabilities chunk by chunk.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from tools.whisper_mlx.audio import load_audio

TEST_FILE = "data/librispeech/dev-clean/1272/128104/1272-128104-0004.flac"
CPP_PROBS_FILE = "/tmp/cpp_vad_probs.json"


def get_python_probs():
    """Get per-chunk probabilities using Python Silero VAD."""
    audio = load_audio(TEST_FILE, sample_rate=16000)

    # Load Silero model
    model, _ = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False
    )
    model.reset_states()

    chunk_size = 512
    num_chunks = len(audio) // chunk_size

    probs = []
    for i in range(num_chunks):
        chunk = audio[i*chunk_size:(i+1)*chunk_size]
        chunk_tensor = torch.tensor(chunk)
        prob = model(chunk_tensor, 16000).item()
        probs.append(prob)

    return np.array(probs)


def main():
    print("=" * 70)
    print("C++ vs Python VAD Probability Comparison")
    print("=" * 70)

    # Load C++ probabilities
    if not os.path.exists(CPP_PROBS_FILE):
        print(f"C++ probabilities not found: {CPP_PROBS_FILE}")
        print("Run: ./build/test_mlx_engine --vad-probs <audio> > /tmp/cpp_vad_probs.json")
        return

    with open(CPP_PROBS_FILE) as f:
        cpp_data = json.load(f)
    cpp_probs = np.array(cpp_data["probabilities"])

    print(f"C++ chunks: {len(cpp_probs)}")

    # Get Python probabilities
    print("Computing Python probabilities...")
    py_probs = get_python_probs()
    print(f"Python chunks: {len(py_probs)}")

    # Align lengths
    min_len = min(len(cpp_probs), len(py_probs))
    cpp_probs = cpp_probs[:min_len]
    py_probs = py_probs[:min_len]

    # Calculate differences
    diff = py_probs - cpp_probs
    abs_diff = np.abs(diff)

    print(f"\n{'='*70}")
    print("Overall Statistics")
    print(f"{'='*70}")
    print(f"Max absolute difference: {abs_diff.max():.6f}")
    print(f"Mean absolute difference: {abs_diff.mean():.6f}")
    print(f"Median absolute difference: {np.median(abs_diff):.6f}")
    print(f"Std dev of difference: {diff.std():.6f}")

    # Find chunks where difference exceeds threshold
    threshold = 0.04  # Focus on significant differences
    significant_chunks = np.where(abs_diff > threshold)[0]

    print(f"\n{'='*70}")
    print(f"Chunks with difference > {threshold}")
    print(f"{'='*70}")
    print(f"{'Chunk':>6} | {'Time':>8} | {'C++':>8} | {'Python':>8} | {'Diff':>8}")
    print("-" * 50)

    for chunk in significant_chunks[:30]:  # Show first 30
        time_s = chunk * 512 / 16000
        print(f"{chunk:>6} | {time_s:>7.3f}s | {cpp_probs[chunk]:>8.4f} | {py_probs[chunk]:>8.4f} | {diff[chunk]:>+8.4f}")

    # Focus on chunk 312 (critical for file 0004)
    print(f"\n{'='*70}")
    print("Critical Region: Chunk 312 (segment 2 end detection)")
    print(f"{'='*70}")
    neg_threshold = 0.35  # Original
    neg_threshold_fixed = 0.345  # With fix

    for i in range(max(0, 312 - 5), min(len(cpp_probs), 312 + 5)):
        time_s = i * 512 / 16000
        cpp_triggers_old = cpp_probs[i] < neg_threshold
        py_triggers_old = py_probs[i] < neg_threshold
        cpp_triggers_new = cpp_probs[i] < neg_threshold_fixed
        py_triggers_new = py_probs[i] < neg_threshold_fixed

        marker = ""
        if i == 312:
            marker = " <<< CRITICAL"

        match_old = "MATCH" if cpp_triggers_old == py_triggers_old else "DIFFER"
        match_new = "MATCH" if cpp_triggers_new == py_triggers_new else "DIFFER"

        print(f"Chunk {i} ({time_s:.3f}s): C++={cpp_probs[i]:.4f}, Py={py_probs[i]:.4f}, "
              f"diff={diff[i]:+.4f}, old_thr={match_old}, new_thr={match_new}{marker}")

    # Analyze segment detection
    print(f"\n{'='*70}")
    print("Segment Detection Analysis")
    print(f"{'='*70}")

    speech_threshold = 0.5
    neg_threshold = 0.345  # Using fixed threshold

    def detect_segments(probs, speech_thr, neg_thr):
        in_speech = False
        segments = []
        speech_start = 0

        for i, prob in enumerate(probs):
            if not in_speech and prob >= speech_thr:
                in_speech = True
                speech_start = i
            elif in_speech and prob < neg_thr:
                # Simplified - just track where we'd detect end
                # In full algo, need silence duration check too
                pass

        return segments

    # Compare where temp_end would be set
    print("\nChunks where temp_end differs (C++ triggers but Python doesn't):")
    for i in range(len(cpp_probs)):
        cpp_below = cpp_probs[i] < neg_threshold_fixed
        py_below = py_probs[i] < neg_threshold_fixed
        if cpp_below != py_below:
            time_s = i * 512 / 16000
            print(f"  Chunk {i} ({time_s:.3f}s): C++={cpp_probs[i]:.4f} ({'<' if cpp_below else '>='} {neg_threshold_fixed}), "
                  f"Py={py_probs[i]:.4f} ({'<' if py_below else '>='} {neg_threshold_fixed})")


if __name__ == "__main__":
    main()
