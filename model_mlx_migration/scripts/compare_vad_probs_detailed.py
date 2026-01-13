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
Compare per-chunk VAD probabilities between Python and C++ implementations.
This helps identify exactly where and why they diverge.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from tools.whisper_mlx.audio import load_audio
from tools.whisper_mlx.silero_vad import SileroVADProcessor

TEST_FILE = "data/librispeech/dev-clean/1272/128104/1272-128104-0004.flac"


def get_python_per_chunk_probs(audio_path: str):
    """Get per-chunk probabilities using Python Silero VAD."""
    audio = load_audio(audio_path, sample_rate=16000)

    # Load Silero model
    processor = SileroVADProcessor(aggressiveness=2)
    model, utils = processor._load_model()

    # Reset state
    model.reset_states()

    # Process in 512-sample chunks
    chunk_size = 512
    num_chunks = len(audio) // chunk_size

    probs = []
    for i in range(num_chunks):
        chunk = audio[i*chunk_size:(i+1)*chunk_size]
        chunk_tensor = torch.tensor(chunk)
        prob = model(chunk_tensor, 16000).item()
        probs.append(prob)

    return np.array(probs)


def find_speech_segments(probs, threshold=0.5, min_speech_chunks=8, min_silence_chunks=10):
    """
    Find speech segments from probabilities (matching C++ algorithm).
    min_speech_chunks = 250ms / 32ms = ~8
    min_silence_chunks = 300ms / 32ms = ~10
    """
    segments = []
    in_speech = False
    speech_start = 0
    silence_count = 0

    for i, prob in enumerate(probs):
        is_speech = prob > threshold

        if not in_speech:
            if is_speech:
                in_speech = True
                speech_start = i
                silence_count = 0
        else:
            if not is_speech:
                silence_count += 1
                if silence_count >= min_silence_chunks:
                    speech_end = i - silence_count
                    speech_len = speech_end - speech_start
                    if speech_len >= min_speech_chunks:
                        segments.append((speech_start, speech_end))
                    in_speech = False
                    silence_count = 0
            else:
                silence_count = 0

    # Handle final segment
    if in_speech:
        speech_end = len(probs)
        speech_len = speech_end - speech_start
        if speech_len >= min_speech_chunks:
            segments.append((speech_start, speech_end))

    return segments


def main():
    if not os.path.exists(TEST_FILE):
        print(f"Test file not found: {TEST_FILE}")
        return

    print("=" * 70)
    print("Per-Chunk VAD Probability Analysis")
    print("=" * 70)

    # Load audio
    audio = load_audio(TEST_FILE, sample_rate=16000)
    print(f"Audio: {len(audio)} samples = {len(audio)/16000:.2f}s")

    # Get Python probabilities
    print("\nComputing Python VAD probabilities...")
    probs = get_python_per_chunk_probs(TEST_FILE)
    print(f"Generated {len(probs)} probabilities")

    # Find segments using same algorithm as C++
    segments = find_speech_segments(probs)
    print("\nPython VAD Segments (using C++ algorithm):")
    for i, (start, end) in enumerate(segments):
        start_time = start * 512 / 16000
        end_time = end * 512 / 16000
        duration = (end - start) * 512 / 16000
        print(f"  {i}: {start_time:.3f}s - {end_time:.3f}s ({duration:.3f}s) [chunks {start}-{end}]")

    # Show probability distribution
    print("\nProbability Distribution:")
    print(f"  Min: {probs.min():.4f}")
    print(f"  Max: {probs.max():.4f}")
    print(f"  Mean: {probs.mean():.4f}")
    print(f"  Speech chunks (>0.5): {np.sum(probs > 0.5)}/{len(probs)}")

    # Show first transition region
    print("\nFirst speech transition (chunks 0-25):")
    for i in range(min(25, len(probs))):
        marker = "***" if probs[i] > 0.5 else "   "
        time_s = i * 512 / 16000
        print(f"  {i:3d} ({time_s:5.3f}s): {probs[i]:.4f} {marker}")

    # Show transition around segment 1->2 (around chunk 83-93)
    gap_start = segments[0][1]  # End of first segment
    gap_end = segments[1][0]    # Start of second segment
    print(f"\nGap between segments 0 and 1 (chunks {gap_start-3}-{gap_end+3}):")
    for i in range(max(0, gap_start-3), min(len(probs), gap_end+3)):
        marker = "***" if probs[i] > 0.5 else "   "
        time_s = i * 512 / 16000
        print(f"  {i:3d} ({time_s:5.3f}s): {probs[i]:.4f} {marker}")

    # Compare with known C++ segments
    print("\n" + "=" * 70)
    print("Comparison with C++ Segments")
    print("=" * 70)

    cpp_segments = [
        (0.576, 2.624),
        (3.008, 4.960),
        (5.408, 9.952),
        (11.520, 18.656),
        (19.776, 29.376),
    ]

    py_segments_times = [(s*512/16000, e*512/16000) for s, e in segments]

    print("\n| Seg | Python Start | C++ Start | Diff   | Python End | C++ End | Diff   |")
    print("|-----|-------------|-----------|--------|-----------|---------|--------|")
    for i, ((py_s, py_e), (cpp_s, cpp_e)) in enumerate(zip(py_segments_times, cpp_segments)):
        s_diff = (cpp_s - py_s) * 1000
        e_diff = (cpp_e - py_e) * 1000
        print(f"| {i}   | {py_s:11.3f} | {cpp_s:9.3f} | {s_diff:+6.1f}ms | {py_e:9.3f} | {cpp_e:7.3f} | {e_diff:+6.1f}ms |")


if __name__ == "__main__":
    main()
