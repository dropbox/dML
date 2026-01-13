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

"""Test Python and C++ transcription with VAD disabled."""

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx

TEST_FILE = "data/librispeech/dev-clean/1272/128104/1272-128104-0004.flac"
WHISPER_MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--mlx-community--whisper-large-v3-turbo/snapshots/beea265c324f07ba1e347f3c8a97aec454056a86/"
)


def transcribe_python_no_vad() -> str:
    """Transcribe with Python WITHOUT VAD."""
    from tools.whisper_mlx import WhisperMLX
    import tools.whisper_mlx.model as model_module

    # Load model
    model = WhisperMLX.from_pretrained("mlx-community/whisper-large-v3-turbo", dtype=mx.float16)

    # Patch VAD to be a no-op
    original_func = model_module.preprocess_audio_with_vad

    def no_vad_preprocess(audio, aggressiveness=2, padding_ms=50, verbose=False):
        from tools.whisper_mlx.silero_vad import VADResult
        duration = len(audio) / 16000.0
        result = VADResult(segments=[], speech_ratio=1.0, total_duration=duration, speech_duration=duration)
        return audio, result

    model_module.preprocess_audio_with_vad = no_vad_preprocess

    try:
        result = model.transcribe(
            TEST_FILE,
            language="en",
            task="transcribe",
            temperature=0.0,
        )
        return result.get("text", "")
    finally:
        model_module.preprocess_audio_with_vad = original_func


def transcribe_cpp_no_vad() -> str:
    """Transcribe with C++ WITHOUT VAD."""
    test_engine = "src/mlx_inference_engine/build/test_mlx_engine"

    cmd = [
        test_engine,
        "--whisper", WHISPER_MODEL_PATH,
        "--transcribe", os.path.abspath(TEST_FILE),
        "--no-vad"  # Disable VAD in C++
    ]

    env = os.environ.copy()
    env["DYLD_LIBRARY_PATH"] = os.path.expanduser("~/.local/lib")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180, env=env, cwd="/Users/ayates/model_mlx_migration")
    stdout = result.stdout

    # Parse JSON from output
    json_start = stdout.find("{")
    json_end = stdout.rfind("}") + 1
    if json_start >= 0 and json_end > json_start:
        data = json.loads(stdout[json_start:json_end])
        return data.get("text", "")
    return ""


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    import re
    text = text.strip().upper()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def main():
    print("=" * 60)
    print("NO-VAD COMPARISON - File 0004")
    print("=" * 60)

    # Python without VAD
    print("\nGetting Python transcription (NO VAD)...")
    py_text = transcribe_python_no_vad()
    py_norm = normalize_text(py_text)
    print(f"Python: {py_text[:100]}...")

    # C++ without VAD
    print("\nGetting C++ transcription (NO VAD)...")
    cpp_text = transcribe_cpp_no_vad()
    cpp_norm = normalize_text(cpp_text)
    print(f"C++: {cpp_text[:100]}...")

    # Compare
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    match = (py_norm == cpp_norm)
    print(f"\nNormalized text match: {'YES' if match else 'NO'}")

    if not match:
        # Show differences
        py_words = py_norm.split()
        cpp_words = cpp_norm.split()

        print(f"\nPython words: {len(py_words)}")
        print(f"C++ words: {len(cpp_words)}")

        # Find first difference
        for i in range(max(len(py_words), len(cpp_words))):
            py_w = py_words[i] if i < len(py_words) else "---"
            cpp_w = cpp_words[i] if i < len(cpp_words) else "---"
            if py_w != cpp_w:
                print(f"\nFirst difference at word {i}:")
                print(f"  Python: '{py_w}'")
                print(f"  C++:    '{cpp_w}'")
                print(f"\nContext (words {max(0,i-3)} to {i+5}):")
                for j in range(max(0, i-3), min(len(py_words), i+5)):
                    py_w = py_words[j] if j < len(py_words) else "---"
                    cpp_w = cpp_words[j] if j < len(cpp_words) else "---"
                    marker = ">>>" if j == i else "   "
                    print(f"  {marker} [{j:2d}] Py: {py_w:20s} C++: {cpp_w}")
                break
    else:
        print("\n*** SUCCESS: Python and C++ match without VAD! ***")
        print("This confirms VAD segment differences cause the divergence.")

    # Print full text
    print("\n" + "=" * 60)
    print("FULL TEXT")
    print("=" * 60)
    print(f"\nPython ({len(py_text)} chars):")
    print(f"  {py_text}")
    print(f"\nC++ ({len(cpp_text)} chars):")
    print(f"  {cpp_text}")


if __name__ == "__main__":
    main()
