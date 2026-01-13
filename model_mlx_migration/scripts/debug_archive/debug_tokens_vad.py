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

"""Debug C++ tokens WITH VAD enabled for file 0004."""

import json
import os
import subprocess
from transformers import WhisperTokenizer

TEST_FILE = "data/librispeech/dev-clean/1272/128104/1272-128104-0004.flac"
WHISPER_MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--mlx-community--whisper-large-v3-turbo/snapshots/beea265c324f07ba1e347f3c8a97aec454056a86/"
)

def get_cpp_tokens_with_vad() -> list:
    """Get C++ tokens WITH VAD (default)."""
    test_engine = "./build/test_mlx_engine"
    if not os.path.exists(test_engine):
        test_engine = "./src/mlx_inference_engine/build/test_mlx_engine"

    cmd = [
        test_engine,
        "--whisper", WHISPER_MODEL_PATH,
        "--transcribe", os.path.abspath(TEST_FILE)
        # No --no-vad flag = VAD enabled
    ]

    env = os.environ.copy()
    env["DYLD_LIBRARY_PATH"] = os.path.expanduser("~/.local/lib")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180, env=env)
    stdout = result.stdout

    json_start = stdout.find("{")
    json_end = stdout.rfind("}") + 1
    if json_start >= 0 and json_end > json_start:
        data = json.loads(stdout[json_start:json_end])
        return data.get("tokens", []), data.get("text", "")
    return [], ""


def main():
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v3")

    print("Getting C++ tokens WITH VAD...")
    cpp_tokens, cpp_text = get_cpp_tokens_with_vad()

    print(f"\nTotal tokens: {len(cpp_tokens)}")
    print(f"\nText: {cpp_text}")

    # Filter to text tokens only
    text_tokens = [t for t in cpp_tokens if t < 50257]

    print(f"\nText tokens: {len(text_tokens)}")

    # Decode and show
    print("\nDecoded tokens:")
    for i, tok in enumerate(cpp_tokens):
        decoded = tokenizer.decode([tok])
        print(f"  [{i:3d}] {tok:6d} -> '{decoded}'")

    # Look for repetitions
    print("\n" + "=" * 60)
    print("REPETITION ANALYSIS")
    print("=" * 60)

    # Look for 2-gram repetitions
    print("\n2-gram repetitions:")
    for i in range(1, len(text_tokens)):
        if text_tokens[i] == text_tokens[i-1]:
            decoded = tokenizer.decode([text_tokens[i]])
            print(f"  Position {i}: token {text_tokens[i]} ('{decoded}')")

    # Look for phrase repetitions (4+ tokens)
    print("\n4-gram phrase search:")
    for i in range(4, len(text_tokens)):
        phrase = tuple(text_tokens[i-3:i+1])
        # Search for this phrase earlier
        for j in range(4, i-3):
            earlier = tuple(text_tokens[j-3:j+1])
            if phrase == earlier:
                decoded = tokenizer.decode(list(phrase))
                print(f"  Phrase at {i}: '{decoded}' (also at {j})")


if __name__ == "__main__":
    main()
