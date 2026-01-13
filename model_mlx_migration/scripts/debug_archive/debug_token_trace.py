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
Compare token-by-token: C++ vs Python forced decode with "idylls" prefix.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
from tools.whisper_mlx import WhisperMLX
from tools.whisper_mlx.audio import load_audio, log_mel_spectrogram
from tools.whisper_mlx.tokenizer import get_whisper_tokenizer
from tools.whisper_mlx.decoding import (
    DecodingOptions, build_logit_filters, apply_filters
)

AUDIO_FILE = "data/librispeech/dev-clean/1272/128104/1272-128104-0004.flac"

# C++ full sequence
CPP_TOKENS = [50258, 50259, 50359, 50364, 9355, 8903, 311, 5242, 366, 257, 1333, 295, 493, 17652, 293, 7938, 14880, 293, 25730, 311, 454, 34152, 4496, 34353, 82, 366, 382, 4048, 382, 257, 361, 18459, 13065, 33660, 2779, 74, 325, 17114, 311, 29822, 29822, 7563, 293, 275, 22943, 275, 22943, 275, 50257]

# Python prefix (same as C++ up to "idylls")
PY_PREFIX = CPP_TOKENS[:25]  # Up to token 82 ('s')

def main():
    print("Loading model...")
    model = WhisperMLX.from_pretrained("mlx-community/whisper-large-v3-turbo", dtype=mx.float16)
    tokenizer = get_whisper_tokenizer(multilingual=True)

    # Load and process audio
    audio = load_audio(AUDIO_FILE)
    mel = log_mel_spectrogram(audio, model.config.n_mels, model.config.n_fft, model.config.hop_length)
    n_frames = 3000
    if mel.shape[0] < n_frames:
        mel = mx.pad(mel, [(0, n_frames - mel.shape[0]), (0, 0)])
    mel = mel[:n_frames]

    # Encode
    audio_features = model.encoder(mel[None])
    mx.eval(audio_features)

    # Build logit filters
    sample_begin = 4  # After SOT sequence
    options = DecodingOptions(
        temperature=0.0,
        max_initial_timestamp=1.0,
        suppress_blank=True,
        suppress_tokens="-1",
        without_timestamps=False,
    )
    logit_filters = build_logit_filters(
        tokenizer,
        options,
        sample_begin=sample_begin,
        n_vocab=model.config.n_vocab,
        precision=0.02,
        audio_duration=len(audio) / 16000,
    )

    # Decode with forced prefix
    tokens = mx.array([PY_PREFIX])
    kv_cache = None

    # Run prefix through decoder
    logits, kv_cache, _, _ = model.decoder(tokens, audio_features, kv_cache=kv_cache)
    mx.eval(kv_cache)

    # Continue decoding
    generated_tokens = list(PY_PREFIX)
    max_tokens = 100

    for step in range(max_tokens):
        logits = logits[:, -1].astype(mx.float32)
        current_tokens = mx.array([generated_tokens])
        filtered_logits = apply_filters(logits, current_tokens[:, sample_begin:], logit_filters)

        next_token = int(mx.argmax(filtered_logits, axis=-1).item())
        generated_tokens.append(next_token)

        if next_token == tokenizer.eot:
            break

        # Get next logits
        logits, kv_cache, _, _ = model.decoder(
            mx.array([[next_token]]),
            audio_features,
            kv_cache=kv_cache
        )
        mx.eval(logits)

    # Compare with C++ tokens
    print("\n=== Token-by-Token Comparison (positions 20-50) ===")
    print(f"{'Pos':>4} {'C++':>6} {'Py':>6} {'C++ text':>15} {'Py text':>15} {'Match':>5}")
    print("-" * 60)

    for i in range(20, min(50, max(len(CPP_TOKENS), len(generated_tokens)))):
        cpp_tok = CPP_TOKENS[i] if i < len(CPP_TOKENS) else None
        py_tok = generated_tokens[i] if i < len(generated_tokens) else None

        cpp_text = tokenizer.decode([cpp_tok])[:12] if cpp_tok else "N/A"
        py_text = tokenizer.decode([py_tok])[:12] if py_tok else "N/A"

        match = "✓" if cpp_tok == py_tok else "✗"
        print(f"{i:>4} {cpp_tok or '-':>6} {py_tok or '-':>6} {repr(cpp_text):>15} {repr(py_text):>15} {match:>5}")

    # Find first divergence
    print("\n=== First Divergence ===")
    for i in range(len(min(CPP_TOKENS, generated_tokens, key=len))):
        if CPP_TOKENS[i] != generated_tokens[i]:
            print(f"Position {i}: C++ selected {CPP_TOKENS[i]} ({repr(tokenizer.decode([CPP_TOKENS[i]]))})")
            print(f"Position {i}: Py  selected {generated_tokens[i]} ({repr(tokenizer.decode([generated_tokens[i]]))})")
            break
    else:
        print("No divergence in overlapping tokens!")

if __name__ == "__main__":
    main()
