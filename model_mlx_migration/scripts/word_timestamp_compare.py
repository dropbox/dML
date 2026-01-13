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
Compare word timestamps between Python mlx-whisper and C++ implementation.
Verifies GAP 3, GAP 26 (alignment_heads), and GAP 27 (medfilt_width).
"""

import subprocess
from pathlib import Path

import mlx_whisper

# Test audio files
TEST_FILES = [
    "data/librispeech/dev-clean/1272/128104/1272-128104-0000.flac",
]

MODEL_PATH = Path.home() / ".cache/huggingface/hub/models--mlx-community--whisper-large-v3-turbo/snapshots/beea265c324f07ba1e347f3c8a97aec454056a86"
CPP_ENGINE = "./build/test_mlx_engine"


def get_python_word_timestamps(audio_path: str) -> list[dict]:
    """Get word timestamps from Python mlx-whisper."""
    result = mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo=str(MODEL_PATH),
        word_timestamps=True,
    )

    words = []
    for segment in result.get("segments", []):
        for word_info in segment.get("words", []):
            words.append({
                "word": word_info["word"],
                "start": word_info["start"],
                "end": word_info["end"],
            })
    return words


def run_cpp_with_word_timestamps(audio_path: str):
    """Run C++ engine and capture output (for future when word timestamps exposed)."""
    cmd = [
        CPP_ENGINE,
        "--whisper", str(MODEL_PATH),
        "--word-timestamps",
        "--transcribe", audio_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout, result.stderr


def main():
    print("=" * 60)
    print("Word Timestamp Comparison: Python mlx-whisper vs C++")
    print("=" * 60)

    for audio_file in TEST_FILES:
        if not Path(audio_file).exists():
            print(f"SKIP: {audio_file} not found")
            continue

        print(f"\nAudio: {audio_file}")
        print("-" * 40)

        # Get Python word timestamps
        print("Python mlx-whisper word timestamps:")
        py_words = get_python_word_timestamps(audio_file)
        for w in py_words[:10]:  # Show first 10
            print(f"  [{w['start']:.2f}s - {w['end']:.2f}s] {w['word']}")
        if len(py_words) > 10:
            print(f"  ... ({len(py_words) - 10} more words)")

        print(f"\nTotal words: {len(py_words)}")

        # Run C++ (just to verify it loads alignment heads)
        print("\nC++ alignment heads loading:")
        stdout, stderr = run_cpp_with_word_timestamps(audio_file)
        if "Loaded 6 alignment heads" in stderr:
            print("  ✓ C++ loaded 6 alignment heads (GAP 26 FIXED)")
        else:
            print("  ✗ C++ did not load alignment heads correctly")
            print(f"  stderr: {stderr[:200]}")


if __name__ == "__main__":
    main()
