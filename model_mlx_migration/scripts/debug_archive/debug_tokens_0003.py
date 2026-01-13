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

"""Debug C++ tokens for file 0003."""

import json
import os
import subprocess
from transformers import WhisperTokenizer

TEST_FILE = "data/librispeech/dev-clean/1272/128104/1272-128104-0003.flac"
WHISPER_MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--mlx-community--whisper-large-v3-turbo/snapshots/beea265c324f07ba1e347f3c8a97aec454056a86/"
)

def get_cpp_tokens() -> list:
    """Get C++ tokens."""
    test_engine = "./build/test_mlx_engine"
    if not os.path.exists(test_engine):
        test_engine = "./src/mlx_inference_engine/build/test_mlx_engine"

    cmd = [
        test_engine,
        "--whisper", WHISPER_MODEL_PATH,
        "--transcribe", os.path.abspath(TEST_FILE)
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

    print("Getting C++ tokens for file 0003...")
    cpp_tokens, cpp_text = get_cpp_tokens()

    print(f"Text: {cpp_text}")
    print(f"\nTotal tokens: {len(cpp_tokens)}")

    print("\nDecoded tokens:")
    for i, tok in enumerate(cpp_tokens):
        decoded = tokenizer.decode([tok])
        print(f"  [{i:3d}] {tok:6d} -> '{decoded}'")


if __name__ == "__main__":
    main()
