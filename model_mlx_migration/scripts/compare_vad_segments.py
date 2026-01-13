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
Compare VAD segments between Python and C++ Silero VAD implementations.
This helps diagnose why file 0004 produces different transcriptions.
"""

import os
import sys
from pathlib import Path
import subprocess
import json

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

# Test file
TEST_FILE = "data/librispeech/dev-clean/1272/128104/1272-128104-0004.flac"


def get_python_vad_segments(audio_path: str):
    """Get VAD segments using Python Silero VAD."""
    from tools.whisper_mlx.silero_vad import SileroVADProcessor
    from tools.whisper_mlx.audio import load_audio

    # Load audio
    audio = load_audio(audio_path, sample_rate=16000)
    print(f"Audio: {len(audio)} samples = {len(audio)/16000:.2f}s at 16kHz")

    # Create processor with same parameters as C++ defaults
    processor = SileroVADProcessor(
        aggressiveness=2,  # threshold=0.5, min_silence=300ms
        sample_rate=16000,
        min_speech_duration_ms=250,
    )

    # Get segments
    result = processor.get_speech_segments(audio)

    print(f"\nPython VAD Segments ({len(result.segments)}):")
    for i, seg in enumerate(result.segments):
        print(f"  {i}: {seg.start:.3f}s - {seg.end:.3f}s ({seg.duration:.3f}s)")

    print("\nPython VAD Stats:")
    print(f"  Total duration: {result.total_duration:.2f}s")
    print(f"  Speech duration: {result.speech_duration:.2f}s")
    print(f"  Speech ratio: {result.speech_ratio:.1%}")

    return result


def get_cpp_vad_segments(audio_path: str):
    """Get VAD segments using C++ Silero VAD via test_engine --vad-debug."""
    # Check if test_engine supports --vad-debug
    test_engine = Path(__file__).parent.parent / "build" / "test_mlx_engine"
    if not test_engine.exists():
        print("C++ test_engine not found at build/test_mlx_engine")
        return None

    # Try to get VAD debug output
    # First, check if we have a VAD-specific debug command
    cmd = [
        str(test_engine),
        "--vad-segments",
        os.path.abspath(audio_path),
    ]

    env = os.environ.copy()
    env["DYLD_LIBRARY_PATH"] = os.path.expanduser("~/.local/lib")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, env=env)
        if result.returncode == 0:
            # Parse JSON output
            stdout = result.stdout
            json_start = stdout.find("{")
            json_end = stdout.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                data = json.loads(stdout[json_start:json_end])
                return data.get("vad_segments", [])
        else:
            print(f"C++ VAD debug not available (returncode={result.returncode})")
            print(f"stderr: {result.stderr[:500]}")
            return None
    except Exception as e:
        print(f"C++ VAD debug failed: {e}")
        return None


def compare_vad_per_chunk(audio_path: str):
    """Compare per-chunk VAD probabilities between Python and C++."""
    from tools.whisper_mlx.silero_vad import SileroVADProcessor
    from tools.whisper_mlx.audio import load_audio
    import torch

    # Load audio
    audio = load_audio(audio_path, sample_rate=16000)

    # Get Python Silero model
    processor = SileroVADProcessor(aggressiveness=2)
    model, utils = processor._load_model()

    # Process in 512-sample chunks (32ms at 16kHz)
    chunk_size = 512
    num_chunks = len(audio) // chunk_size

    print(f"\nPer-chunk VAD comparison ({num_chunks} chunks):")
    print("Chunk | Time     | Python Prob")
    print("-" * 40)

    probs = []
    for i in range(min(num_chunks, 50)):  # First 50 chunks (~1.6s)
        chunk = audio[i*chunk_size:(i+1)*chunk_size]
        chunk_tensor = torch.tensor(chunk)

        # Get probability
        prob = model(chunk_tensor, 16000).item()
        probs.append(prob)

        time_s = i * chunk_size / 16000
        print(f"  {i:3d} | {time_s:6.3f}s | {prob:.4f}")

    # Show where speech starts/ends
    threshold = 0.5
    in_speech = False
    for i, prob in enumerate(probs):
        if not in_speech and prob > threshold:
            print(f"\n  -> Speech START at chunk {i} ({i*chunk_size/16000:.3f}s)")
            in_speech = True
        elif in_speech and prob < threshold:
            print(f"  -> Speech END at chunk {i} ({i*chunk_size/16000:.3f}s)")
            in_speech = False


def get_audio_stats(audio_path: str):
    """Get basic audio statistics."""
    from tools.whisper_mlx.audio import load_audio

    audio = load_audio(audio_path, sample_rate=16000)
    sr = 16000

    print(f"\nAudio Statistics for {os.path.basename(audio_path)}:")
    print(f"  Sample rate: {sr} Hz")
    print(f"  Duration: {len(audio)/sr:.2f}s")
    print(f"  Samples: {len(audio)}")
    print(f"  Min value: {audio.min():.4f}")
    print(f"  Max value: {audio.max():.4f}")
    print(f"  Mean: {audio.mean():.6f}")
    print(f"  RMS: {np.sqrt(np.mean(audio**2)):.4f}")

    return audio


def main():
    if not os.path.exists(TEST_FILE):
        print(f"Test file not found: {TEST_FILE}")
        return

    print("=" * 60)
    print("VAD Segment Comparison: Python vs C++")
    print("=" * 60)

    # Audio stats
    audio = get_audio_stats(TEST_FILE)

    # Python VAD
    print("\n" + "=" * 60)
    print("Python Silero VAD")
    print("=" * 60)
    py_result = get_python_vad_segments(TEST_FILE)

    # C++ VAD (if available)
    print("\n" + "=" * 60)
    print("C++ Silero VAD")
    print("=" * 60)
    cpp_segments = get_cpp_vad_segments(TEST_FILE)

    if cpp_segments is None:
        print("C++ VAD segments not available.")
        print("\nTo enable C++ VAD debug output, add --vad-segments flag to test_engine")

    # Per-chunk comparison
    print("\n" + "=" * 60)
    print("Per-Chunk VAD Analysis (first 50 chunks)")
    print("=" * 60)
    compare_vad_per_chunk(TEST_FILE)


if __name__ == "__main__":
    main()
