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

"""Compare Python and C++ tokens in detail for file 0004."""

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
from tools.whisper_mlx import WhisperMLX
from transformers import WhisperTokenizer

TEST_FILE = "data/librispeech/dev-clean/1272/128104/1272-128104-0004.flac"
WHISPER_MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--mlx-community--whisper-large-v3-turbo/snapshots/beea265c324f07ba1e347f3c8a97aec454056a86/"
)


def get_python_tokens_no_vad(model) -> list:
    """Get Python tokens without VAD."""
    import tools.whisper_mlx.model as model_module

    original_func = model_module.preprocess_audio_with_vad

    def no_vad_preprocess(audio, aggressiveness=2, padding_ms=50, verbose=False):
        from tools.whisper_mlx.silero_vad import VADResult
        duration = len(audio) / 16000.0
        result = VADResult(segments=[], speech_ratio=1.0, total_duration=duration, speech_duration=duration)
        return audio, result

    model_module.preprocess_audio_with_vad = no_vad_preprocess

    try:
        # Use transcribe with word timestamps to get tokens
        result = model.transcribe(
            TEST_FILE,
            language="en",
            task="transcribe",
            temperature=0.0,
            word_timestamps=True,
        )

        # Tokens should be in the result
        tokens = result.get("tokens", [])
        if not tokens:
            # Fall back to decoding from segments if available
            segments = result.get("segments", [])
            for seg in segments:
                tokens.extend(seg.get("tokens", []))

        return tokens
    finally:
        model_module.preprocess_audio_with_vad = original_func


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
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v3")

    # Load model
    print("Loading Python WhisperMLX...")
    model = WhisperMLX.from_pretrained("mlx-community/whisper-large-v3-turbo", dtype=mx.float16)

    # Get Python tokens
    print("\nGetting Python tokens (no VAD)...")
    py_tokens = get_python_tokens_no_vad(model)
    print(f"Python tokens: {len(py_tokens)}")

    # Get C++ tokens
    print("Getting C++ tokens (no VAD)...")
    cpp_tokens = get_cpp_tokens_no_vad()
    print(f"C++ tokens: {len(cpp_tokens)}")

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

        # Only print around the first difference
        if first_diff is not None and i >= first_diff - 5 and i <= first_diff + 20:
            py_str = tokenizer.decode([py_tok]) if py_tok is not None else "---"
            cpp_str = tokenizer.decode([cpp_tok]) if cpp_tok is not None else "---"

            match = "  " if py_tok == cpp_tok else "!!"

            print(f"[{i:3d}] {match} Py:{py_tok:6d} '{py_str:15s}'  | C++:{cpp_tok if cpp_tok else '---':6} '{cpp_str}'")

    if first_diff is not None:
        print(f"\nFirst difference at index {first_diff}")
    else:
        print("\nNo differences found - tokens match!")


if __name__ == "__main__":
    main()
