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

"""Compare Python and C++ tokens without VAD for file 0004."""

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import numpy as np
from transformers import WhisperTokenizer

TEST_FILE = "data/librispeech/dev-clean/1272/128104/1272-128104-0004.flac"
WHISPER_MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--mlx-community--whisper-large-v3-turbo/snapshots/beea265c324f07ba1e347f3c8a97aec454056a86/"
)


def get_python_tokens_no_vad() -> list:
    """Get Python tokens without VAD by accessing internal decode method."""
    from tools.whisper_mlx import WhisperMLX
    from tools.whisper_mlx.audio import load_audio, log_mel_spectrogram

    # Load model
    model = WhisperMLX.from_pretrained("mlx-community/whisper-large-v3-turbo", dtype=mx.float16)

    # Load audio directly without VAD
    audio = load_audio(TEST_FILE, sample_rate=16000)
    print(f"Python: Loaded {len(audio)} samples ({len(audio)/16000:.2f}s)")

    # Pad to 30 seconds
    padded_audio = np.zeros(480000, dtype=np.float32)
    padded_audio[:min(len(audio), 480000)] = audio[:min(len(audio), 480000)]

    # Compute mel spectrogram
    n_mels = 128  # large-v3
    mel = log_mel_spectrogram(padded_audio, n_mels=n_mels)
    print(f"Python: Mel shape {mel.shape}")

    # Add batch dimension
    mel = mx.expand_dims(mel, axis=0)

    # Encode audio
    audio_features = model.embed_audio(mel)
    mx.eval(audio_features)
    print(f"Python: Encoder output shape {audio_features.shape}")

    # Get tokenizer
    from tools.whisper_mlx.tokenizer import get_whisper_tokenizer
    tokenizer = get_whisper_tokenizer(
        multilingual=True,
        num_languages=100,
        language="en",
        task="transcribe",
    )

    # Decode with greedy (temperature=0)
    # This directly calls the internal decode method
    tokens, segments, avg_logprob, no_speech_prob = model._decode_with_metrics(
        audio_features,
        tokenizer,
        temperature=0.0,
        no_speech_threshold=0.6,
        prompt_tokens=None,
    )

    return tokens


def get_cpp_tokens_no_vad() -> list:
    """Get C++ tokens without VAD."""
    test_engine = "src/mlx_inference_engine/build/test_mlx_engine"

    cmd = [
        test_engine,
        "--whisper", WHISPER_MODEL_PATH,
        "--transcribe", os.path.abspath(TEST_FILE),
        "--no-vad"
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
        return data.get("tokens", [])
    return []


def main():
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v3")

    print("=" * 70)
    print("TOKEN COMPARISON - File 0004 WITHOUT VAD")
    print("=" * 70)

    # Get Python tokens
    print("\nGetting Python tokens (no VAD)...")
    py_tokens = get_python_tokens_no_vad()
    print(f"Python: {len(py_tokens)} tokens")

    # Get C++ tokens
    print("\nGetting C++ tokens (no VAD)...")
    cpp_tokens = get_cpp_tokens_no_vad()
    print(f"C++: {len(cpp_tokens)} tokens")

    # Find first difference
    print("\n" + "=" * 70)
    print("SIDE-BY-SIDE COMPARISON")
    print("=" * 70)

    max_len = max(len(py_tokens), len(cpp_tokens))
    first_diff = None

    for i in range(max_len):
        py_tok = py_tokens[i] if i < len(py_tokens) else None
        cpp_tok = cpp_tokens[i] if i < len(cpp_tokens) else None

        if py_tok != cpp_tok and first_diff is None:
            first_diff = i

        # Print around first difference or first 30 if no diff
        should_print = False
        if first_diff is not None:
            should_print = (i >= first_diff - 5 and i <= first_diff + 30)
        else:
            should_print = (i < 30)

        if should_print:
            py_str = tokenizer.decode([py_tok]) if py_tok is not None else "---"
            cpp_str = tokenizer.decode([cpp_tok]) if cpp_tok is not None else "---"

            match = "  " if py_tok == cpp_tok else "!!"

            py_tok_str = f"{py_tok:6d}" if py_tok is not None else "  None"
            cpp_tok_str = f"{cpp_tok:6d}" if cpp_tok is not None else "  None"

            print(f"[{i:3d}] {match} Py:{py_tok_str} '{py_str:20s}'  | C++:{cpp_tok_str} '{cpp_str}'")

    if first_diff is not None:
        print(f"\n*** FIRST DIFFERENCE at index {first_diff} ***")
        py_t = py_tokens[first_diff] if first_diff < len(py_tokens) else "END"
        cpp_t = cpp_tokens[first_diff] if first_diff < len(cpp_tokens) else "END"
        print(f"    Python: {py_t}")
        print(f"    C++:    {cpp_t}")
        if first_diff < len(py_tokens) and first_diff < len(cpp_tokens):
            print(f"    Python text: '{tokenizer.decode([py_tokens[first_diff]])}'")
            print(f"    C++    text: '{tokenizer.decode([cpp_tokens[first_diff]])}'")
    else:
        print("\n*** SUCCESS: All tokens match! ***")

    # Decode full text
    print("\n" + "=" * 70)
    print("FULL DECODED TEXT")
    print("=" * 70)

    # Filter out special tokens for text decoding
    py_text_tokens = [t for t in py_tokens if t < 50257]
    cpp_text_tokens = [t for t in cpp_tokens if t < 50257]

    py_text = tokenizer.decode(py_text_tokens)
    cpp_text = tokenizer.decode(cpp_text_tokens)

    print(f"\nPython ({len(py_text)} chars): {py_text}")
    print(f"\nC++ ({len(cpp_text)} chars): {cpp_text}")


if __name__ == "__main__":
    main()
