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
Compare tokens between Python (no VAD) and C++ to find exact divergence.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
from tools.whisper_mlx import WhisperMLX
from tools.whisper_mlx.audio import load_audio, log_mel_spectrogram
from tools.whisper_mlx.tokenizer import get_whisper_tokenizer

AUDIO_FILE = "data/librispeech/dev-clean/1272/128104/1272-128104-0004.flac"

# C++ tokens from previous output
CPP_TOKENS = [50258, 50259, 50359, 50364, 9355, 8903, 311, 5242, 366, 257, 1333, 295, 493, 17652, 293, 7938, 14880, 293, 25730, 311, 454, 34152, 4496, 34353, 82, 366, 382, 4048, 382, 257, 361, 18459, 13065, 33660, 2779, 74, 325, 17114, 311, 29822, 29822, 7563, 293, 275, 22943, 275, 22943, 275, 50257]

def main():
    print("Loading model and tokenizer...")
    model = WhisperMLX.from_pretrained("mlx-community/whisper-large-v3-turbo", dtype=mx.float16)
    tokenizer = get_whisper_tokenizer(multilingual=True)

    # Load raw audio and process
    audio = load_audio(AUDIO_FILE)
    print(f"Audio: {len(audio)/16000:.2f}s")

    # Compute mel spectrogram manually (no VAD)
    mel = log_mel_spectrogram(audio, model.config.n_mels, model.config.n_fft, model.config.hop_length)
    n_frames = 3000  # 30s
    if mel.shape[0] < n_frames:
        mel = mx.pad(mel, [(0, n_frames - mel.shape[0]), (0, 0)])
    mel = mel[:n_frames]

    # Encode
    audio_features = model.encoder(mel[None])
    mx.eval(audio_features)

    # Decode
    tokens, _, _, _ = model._decode_with_metrics(
        audio_features,
        tokenizer,
        temperature=0.0,
        audio_duration=len(audio) / 16000,
    )

    # Add SOT sequence (which _decode_with_metrics returns without)
    # Actually, let's get the full tokens including SOT
    sot_sequence = list(tokenizer.sot_sequence)
    py_tokens = sot_sequence + tokens

    print("\n=== TOKEN COMPARISON ===")
    print(f"C++ tokens: {len(CPP_TOKENS)}")
    print(f"Python tokens: {len(py_tokens)}")

    # Compare token by token
    print("\n{:>4} {:>6} {:>6} {:>15} {:>15} {:>5}".format(
        "Pos", "C++", "Py", "C++ text", "Py text", "Match"
    ))
    print("-" * 70)

    max_len = max(len(CPP_TOKENS), len(py_tokens))
    first_diff = None

    for i in range(min(50, max_len)):  # Show first 50
        cpp_tok = CPP_TOKENS[i] if i < len(CPP_TOKENS) else None
        py_tok = py_tokens[i] if i < len(py_tokens) else None

        cpp_text = tokenizer.decode([cpp_tok]) if cpp_tok else "N/A"
        py_text = tokenizer.decode([py_tok]) if py_tok else "N/A"

        match = "✓" if cpp_tok == py_tok else "✗"
        if cpp_tok != py_tok and first_diff is None:
            first_diff = i

        print("{:>4} {:>6} {:>6} {:>15} {:>15} {:>5}".format(
            i, cpp_tok or "-", py_tok or "-",
            repr(cpp_text)[:15], repr(py_text)[:15], match
        ))

    if first_diff is not None:
        print(f"\n*** FIRST DIFFERENCE at position {first_diff} ***")
        print(f"  C++ token: {CPP_TOKENS[first_diff]} = {repr(tokenizer.decode([CPP_TOKENS[first_diff]]))}")
        print(f"  Py token:  {py_tokens[first_diff]} = {repr(tokenizer.decode([py_tokens[first_diff]]))}")

        # Show context
        print("\n=== C++ Context around divergence ===")
        start = max(0, first_diff - 5)
        end = min(len(CPP_TOKENS), first_diff + 10)
        cpp_context_tokens = CPP_TOKENS[start:end]
        print(f"Tokens: {cpp_context_tokens}")
        print(f"Text: {tokenizer.decode(cpp_context_tokens)}")

        print("\n=== Python Context around divergence ===")
        end = min(len(py_tokens), first_diff + 10)
        py_context_tokens = py_tokens[start:end]
        print(f"Tokens: {py_context_tokens}")
        print(f"Text: {tokenizer.decode(py_context_tokens)}")

    # Check what "idles" and "idylls" encode to
    print("\n=== ENCODING TEST ===")
    print(f"'idles' encodes to: {tokenizer.encode('idles')}")
    print(f"'idylls' encodes to: {tokenizer.encode('idylls')}")
    print(f"' idles' encodes to: {tokenizer.encode(' idles')}")
    print(f"' idylls' encodes to: {tokenizer.encode(' idylls')}")

if __name__ == "__main__":
    main()
