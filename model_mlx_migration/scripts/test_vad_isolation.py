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
Test VAD isolation for Gate 0 debugging.
Compare Python and C++ transcription with and without VAD on file 0004.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
from tools.whisper_mlx import WhisperMLX


# Test file 0004 - the problematic one
TEST_FILE = "data/librispeech/dev-clean/1272/128104/1272-128104-0004.flac"
REFERENCE = "LINNELL'S PICTURES ARE A SORT OF UP GUARDS AND AT THEM PAINTING AND MASSON'S EXQUISITE IDYLLS ARE AS NATIONAL AS A JINGO POEM MISTER BIRKET FOSTER'S LANDSCAPES SMILE AT ONE MUCH IN THE SAME WAY THAT MISTER CARKER USED TO FLASH HIS TEETH AND MISTER JOHN COLLIER GIVES HIS SITTER A CHEERFUL SLAP ON THE BACK BEFORE HE SAYS LIKE A SHAMPOOER IN A TURKISH BATH NEXT MAN"

WHISPER_MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--mlx-community--whisper-large-v3-turbo/snapshots/beea265c324f07ba1e347f3c8a97aec454056a86/"
)


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    import re
    text = text.strip().upper()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def transcribe_python_no_vad(model, audio_path: str) -> Dict:
    """Transcribe using Python WITHOUT VAD preprocessing."""
    # Import to patch
    import tools.whisper_mlx.model as model_module

    # Temporarily disable VAD by making preprocess_audio_with_vad a no-op
    original_func = model_module.preprocess_audio_with_vad

    def no_vad_preprocess(audio, aggressiveness=2, padding_ms=50, verbose=False):
        """Return audio unchanged with dummy VAD result."""
        from tools.whisper_mlx.silero_vad import VADResult
        duration = len(audio) / 16000.0
        result = VADResult(
            segments=[],
            speech_ratio=1.0,  # Pretend all audio is speech
            total_duration=duration,
            speech_duration=duration,
        )
        return audio, result

    model_module.preprocess_audio_with_vad = no_vad_preprocess

    try:
        result = model.transcribe(
            audio_path,
            language="en",
            task="transcribe",
            temperature=0.0,
        )
        return {
            "text": result["text"],
            "tokens": result.get("tokens", []),
        }
    finally:
        # Restore original VAD function
        model_module.preprocess_audio_with_vad = original_func


def transcribe_python_with_vad(model, audio_path: str) -> Dict:
    """Transcribe using Python WITH VAD preprocessing (normal)."""
    result = model.transcribe(
        audio_path,
        language="en",
        task="transcribe",
        temperature=0.0,
    )
    return {
        "text": result["text"],
        "tokens": result.get("tokens", []),
    }


def transcribe_cpp(audio_path: str, use_vad: bool = True) -> Dict:
    """Transcribe using C++ test_engine."""
    audio_path = os.path.abspath(audio_path)
    test_engine = "./build/test_mlx_engine"

    if not os.path.exists(test_engine):
        test_engine = "./src/mlx_inference_engine/build/test_mlx_engine"

    cmd = [
        test_engine,
        "--whisper", WHISPER_MODEL_PATH,
        "--transcribe", audio_path
    ]

    if not use_vad:
        cmd.append("--no-vad")

    env = os.environ.copy()
    env["DYLD_LIBRARY_PATH"] = os.path.expanduser("~/.local/lib")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180, env=env)
        stdout = result.stdout

        json_start = stdout.find("{")
        json_end = stdout.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            data = json.loads(stdout[json_start:json_end])
            return {
                "text": data.get("text", ""),
                "tokens": data.get("tokens", []),
            }
        else:
            return {"text": f"[Parse error: {stdout[:200]}]", "tokens": []}
    except Exception as e:
        return {"text": f"[Error: {e}]", "tokens": []}


def main():
    print("=== VAD Isolation Test for Gate 0 ===\n")
    print(f"Test file: {TEST_FILE}")
    print(f"Reference: {REFERENCE[:60]}...\n")

    # Load model
    print("Loading Python WhisperMLX...")
    model = WhisperMLX.from_pretrained("mlx-community/whisper-large-v3-turbo", dtype=mx.float16)

    # Test 1: Python WITH VAD (reference)
    print("\n1. Python WITH VAD:")
    py_vad = transcribe_python_with_vad(model, TEST_FILE)
    print(f"   Text: {py_vad['text'][:80]}...")
    print(f"   Tokens: {len(py_vad['tokens'])}")

    # Test 2: Python WITHOUT VAD
    print("\n2. Python WITHOUT VAD:")
    py_no_vad = transcribe_python_no_vad(model, TEST_FILE)
    print(f"   Text: {py_no_vad['text'][:80]}...")
    print(f"   Tokens: {len(py_no_vad['tokens'])}")

    # Test 3: C++ WITH VAD
    print("\n3. C++ WITH VAD:")
    cpp_vad = transcribe_cpp(TEST_FILE, use_vad=True)
    print(f"   Text: {cpp_vad['text'][:80]}...")
    print(f"   Tokens: {len(cpp_vad['tokens'])}")

    # Test 4: C++ WITHOUT VAD
    print("\n4. C++ WITHOUT VAD:")
    cpp_no_vad = transcribe_cpp(TEST_FILE, use_vad=False)
    print(f"   Text: {cpp_no_vad['text'][:80]}...")
    print(f"   Tokens: {len(cpp_no_vad['tokens'])}")

    # Compare
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    py_vad_norm = normalize_text(py_vad['text'])
    py_no_vad_norm = normalize_text(py_no_vad['text'])
    cpp_vad_norm = normalize_text(cpp_vad['text'])
    cpp_no_vad_norm = normalize_text(cpp_no_vad['text'])

    print(f"\nPython VAD == Python no-VAD: {py_vad_norm == py_no_vad_norm}")
    print(f"C++ VAD == C++ no-VAD: {cpp_vad_norm == cpp_no_vad_norm}")
    print(f"\nPython VAD == C++ VAD: {py_vad_norm == cpp_vad_norm}")
    print(f"Python no-VAD == C++ no-VAD: {py_no_vad_norm == cpp_no_vad_norm}")

    # Key question: Does disabling VAD make them match?
    if py_no_vad_norm == cpp_no_vad_norm:
        print("\n*** VAD IS THE ISSUE: Python and C++ match when VAD is disabled! ***")
    else:
        print("\n*** VAD is NOT the only issue: Mismatch persists without VAD ***")
        print(f"\nPython no-VAD: {py_no_vad_norm[:100]}...")
        print(f"C++ no-VAD:    {cpp_no_vad_norm[:100]}...")

    # Print full texts for analysis
    print("\n" + "=" * 60)
    print("FULL TEXTS (no-VAD)")
    print("=" * 60)
    print(f"\nPython no-VAD (full):\n{py_no_vad['text']}")
    print(f"\nC++ no-VAD (full):\n{cpp_no_vad['text']}")
    print(f"\nPython no-VAD normalized (full):\n{py_no_vad_norm}")
    print(f"\nC++ no-VAD normalized (full):\n{cpp_no_vad_norm}")

    # Token comparison for no-VAD case
    print("\n" + "=" * 60)
    print("TOKEN COMPARISON (no-VAD)")
    print("=" * 60)

    py_tokens = py_no_vad['tokens']
    cpp_tokens = cpp_no_vad['tokens']

    print(f"Python tokens: {len(py_tokens)}")
    print(f"C++ tokens: {len(cpp_tokens)}")

    # Find first difference
    min_len = min(len(py_tokens), len(cpp_tokens))
    first_diff = None
    for i in range(min_len):
        if py_tokens[i] != cpp_tokens[i]:
            first_diff = i
            break

    if first_diff is not None:
        print(f"\nFirst difference at token {first_diff}:")
        start = max(0, first_diff - 3)
        end = min(min_len, first_diff + 5)
        print(f"  Python: {py_tokens[start:end]}")
        print(f"  C++:    {cpp_tokens[start:end]}")
    elif len(py_tokens) != len(cpp_tokens):
        print(f"\nTokens match for first {min_len}, but lengths differ")
    else:
        print("\n*** TOKENS MATCH EXACTLY ***")


if __name__ == "__main__":
    main()
