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

"""Debug tokens for file 0004."""

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import WhisperTokenizer

# Test file
TEST_FILE = "data/librispeech/dev-clean/1272/128104/1272-128104-0004.flac"
WHISPER_MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--mlx-community--whisper-large-v3-turbo/snapshots/beea265c324f07ba1e347f3c8a97aec454056a86/"
)

def get_cpp_tokens_no_vad() -> list:
    """Get C++ tokens without VAD."""
    test_engine = "./build/test_mlx_engine"
    if not os.path.exists(test_engine):
        test_engine = "./src/mlx_inference_engine/build/test_mlx_engine"

    cmd = [
        test_engine,
        "--whisper", WHISPER_MODEL_PATH,
        "--transcribe", os.path.abspath(TEST_FILE),
        "--no-vad"
    ]

    env = os.environ.copy()
    env["DYLD_LIBRARY_PATH"] = os.path.expanduser("~/.local/lib")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180, env=env)
    stdout = result.stdout

    json_start = stdout.find("{")
    json_end = stdout.rfind("}") + 1
    if json_start >= 0 and json_end > json_start:
        data = json.loads(stdout[json_start:json_end])
        return data.get("tokens", [])
    return []


def main():
    # Load tokenizer
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v3")

    # Get C++ tokens
    print("Getting C++ tokens (no VAD)...")
    cpp_tokens = get_cpp_tokens_no_vad()

    print(f"\nTotal tokens: {len(cpp_tokens)}")
    print(f"\nTokens: {cpp_tokens}")

    # Decode each token
    print("\nDecoded tokens:")
    for i, tok in enumerate(cpp_tokens):
        try:
            decoded = tokenizer.decode([tok])
            print(f"  [{i:2d}] {tok:6d} -> '{decoded}'")
        except Exception as e:
            print(f"  [{i:2d}] {tok:6d} -> ERROR: {e}")

    # Find repetitions
    print("\n" + "=" * 60)
    print("REPETITION ANALYSIS")
    print("=" * 60)

    # Filter to text tokens only (exclude timestamps)
    text_tokens = [t for t in cpp_tokens if t < 50257]  # EOT and timestamps are >= 50257

    # Look for 2-gram repetitions
    for i in range(1, len(text_tokens)):
        if text_tokens[i] == text_tokens[i-1]:
            decoded = tokenizer.decode([text_tokens[i]])
            print(f"2-gram repetition at {i}: token {text_tokens[i]} ('{decoded}')")

    # Look for 3-gram repetitions
    print("\n3-gram patterns:")
    for i in range(2, len(text_tokens)):
        if text_tokens[i] == text_tokens[i-1] == text_tokens[i-2]:
            decoded = tokenizer.decode([text_tokens[i]])
            print(f"3-gram at {i}: '{decoded}' repeated 3 times")

    # Look for word repetitions in decoded text
    decoded_text = tokenizer.decode(text_tokens)
    print(f"\nDecoded text: {decoded_text}")


if __name__ == "__main__":
    main()
