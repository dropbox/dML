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

"""Debug token divergence for file 0004 between C++ and Python."""

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
from transformers import WhisperTokenizer

# Test file 0004
TEST_FILE = "data/librispeech/dev-clean/1272/128104/1272-128104-0004.flac"
WHISPER_MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--mlx-community--whisper-large-v3-turbo/snapshots/beea265c324f07ba1e347f3c8a97aec454056a86/"
)


def get_python_tokens_with_vad() -> list:
    """Get Python tokens WITH VAD (default behavior)."""
    from tools.whisper_mlx import WhisperMLX

    model = WhisperMLX.from_pretrained("mlx-community/whisper-large-v3-turbo", dtype=mx.float16)

    # Transcribe with VAD (default)
    result = model.transcribe(
        TEST_FILE,
        language="en",
        task="transcribe",
        temperature=0.0,
    )

    # Get tokens from segments
    tokens = []
    for seg in result.get("segments", []):
        tokens.extend(seg.get("tokens", []))

    return tokens, result.get("text", "")


def get_cpp_tokens_with_vad() -> tuple:
    """Get C++ tokens WITH VAD."""
    test_engine = "./src/mlx_inference_engine/build/test_mlx_engine"
    if not os.path.exists(test_engine):
        test_engine = "./build/test_mlx_engine"

    cmd = [
        test_engine,
        "--whisper", WHISPER_MODEL_PATH,
        "--transcribe", os.path.abspath(TEST_FILE),
        # Note: VAD is enabled by default (no --no-vad flag)
    ]

    env = os.environ.copy()
    env["DYLD_LIBRARY_PATH"] = os.path.expanduser("~/.local/lib")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180, env=env)
    stdout = result.stdout

    # Parse JSON from output
    json_start = stdout.find("{")
    json_end = stdout.rfind("}") + 1
    if json_start >= 0 and json_end > json_start:
        data = json.loads(stdout[json_start:json_end])
        return data.get("tokens", []), data.get("text", "")
    return [], ""


def main():
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v3")

    print("=" * 60)
    print("TOKEN COMPARISON - File 0004 WITH VAD")
    print("=" * 60)

    # Get Python tokens
    print("\nGetting Python tokens (with VAD)...")
    py_tokens, py_text = get_python_tokens_with_vad()
    print(f"Python: {len(py_tokens)} tokens")
    print(f"Python text: {py_text[:100]}...")

    # Get C++ tokens
    print("\nGetting C++ tokens (with VAD)...")
    cpp_tokens, cpp_text = get_cpp_tokens_with_vad()
    print(f"C++: {len(cpp_tokens)} tokens")
    print(f"C++ text: {cpp_text[:100]}...")

    # Find first difference
    print("\n" + "=" * 60)
    print("SIDE-BY-SIDE COMPARISON")
    print("=" * 60)

    max_len = max(len(py_tokens), len(cpp_tokens))
    first_diff = None

    for i in range(max_len):
        py_tok = py_tokens[i] if i < len(py_tokens) else None
        cpp_tok = cpp_tokens[i] if i < len(cpp_tokens) else None

        if py_tok != cpp_tok and first_diff is None:
            first_diff = i

        # Print around the first difference, or all if no diff
        should_print = (first_diff is not None and i >= first_diff - 5 and i <= first_diff + 25)
        should_print = should_print or (first_diff is None and i < 30)

        if should_print:
            py_str = tokenizer.decode([py_tok]) if py_tok is not None else "---"
            cpp_str = tokenizer.decode([cpp_tok]) if cpp_tok is not None else "---"

            match = "  " if py_tok == cpp_tok else "!!"

            py_tok_str = f"{py_tok:6d}" if py_tok is not None else "  None"
            cpp_tok_str = f"{cpp_tok:6d}" if cpp_tok is not None else "  None"

            print(f"[{i:3d}] {match} Py:{py_tok_str} '{py_str:20s}'  | C++:{cpp_tok_str} '{cpp_str}'")

    if first_diff is not None:
        print(f"\n*** FIRST DIFFERENCE at index {first_diff} ***")
        print(f"    Python: {py_tokens[first_diff] if first_diff < len(py_tokens) else 'END'}")
        print(f"    C++:    {cpp_tokens[first_diff] if first_diff < len(cpp_tokens) else 'END'}")
    else:
        print("\n*** No differences found - tokens match! ***")

    # Print full text comparison
    print("\n" + "=" * 60)
    print("FULL TEXT COMPARISON")
    print("=" * 60)
    print(f"Python ({len(py_text)} chars):")
    print(f"  {py_text}")
    print(f"\nC++ ({len(cpp_text)} chars):")
    print(f"  {cpp_text}")


if __name__ == "__main__":
    main()
