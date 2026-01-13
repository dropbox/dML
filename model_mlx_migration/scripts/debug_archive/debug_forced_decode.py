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
Test: Force Python to decode with the same initial tokens as C++ to see if repetition occurs.

If Python also enters repetition when forced to use "idylls" instead of "idles",
then the issue is the model's behavior on this audio, not C++ implementation.
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

# C++ token sequence up to position 30 (after "idylls")
CPP_PREFIX = [50258, 50259, 50359, 50364, 9355, 8903, 311, 5242, 366, 257, 1333, 295, 493, 17652, 293, 7938, 14880, 293, 25730, 311, 454, 34152, 4496, 34353, 82]  # ends with "idylls"

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

    # Force decode with C++ prefix, then continue naturally
    print("\n=== Forced decode with 'idylls' prefix ===")
    print(f"Prefix: {tokenizer.decode(CPP_PREFIX[sample_begin:])}")

    # Initialize tokens with forced prefix
    tokens = mx.array([CPP_PREFIX])
    kv_cache = None

    # Run full prefix through decoder to build KV cache
    logits, kv_cache, _, _ = model.decoder(
        tokens, audio_features, kv_cache=kv_cache
    )
    mx.eval(kv_cache)

    print(f"\nKV cache built for {len(CPP_PREFIX)} tokens")

    # Show what tokens the model would naturally select after "idylls"
    logits = logits[:, -1].astype(mx.float32)
    current_tokens = mx.array([CPP_PREFIX])
    filtered_logits = apply_filters(logits, current_tokens[:, sample_begin:], logit_filters)

    # Get top 5 candidates
    top_indices = mx.argsort(filtered_logits, axis=-1)[:, -5:]
    mx.eval(top_indices)
    print("\nTop 5 token candidates after 'idylls':")
    for i in range(5):
        idx = int(top_indices[0, -(i+1)].item())
        score = float(filtered_logits[0, idx].item())
        text = tokenizer.decode([idx])
        is_ts = idx >= tokenizer.timestamp_begin
        print(f"  {i+1}. Token {idx} ({repr(text)}) logit={score:.2f} {'[TIMESTAMP]' if is_ts else ''}")

    # Now continue decoding naturally
    generated_tokens = list(CPP_PREFIX)
    max_tokens = 100

    for step in range(max_tokens):
        # Get logits for last position
        logits, kv_cache, _, _ = model.decoder(
            mx.array([[generated_tokens[-1]]]),
            audio_features,
            kv_cache=kv_cache
        )
        mx.eval(logits)

        logits = logits[:, -1].astype(mx.float32)

        # Apply filters
        current_tokens = mx.array([generated_tokens])
        filtered_logits = apply_filters(logits, current_tokens[:, sample_begin:], logit_filters)

        # Greedy select
        next_token = int(mx.argmax(filtered_logits, axis=-1).item())

        generated_tokens.append(next_token)

        if next_token == tokenizer.eot:
            break

        # Check for repetition
        if len(generated_tokens) >= 2:
            if generated_tokens[-1] == generated_tokens[-2] and generated_tokens[-1] < tokenizer.timestamp_begin:
                text = tokenizer.decode([generated_tokens[-1]])
                print(f"  REPETITION at step {step}: token {generated_tokens[-1]} ({repr(text)})")

    print(f"\n=== Result ({len(generated_tokens)} tokens) ===")
    full_text = tokenizer.decode(generated_tokens[sample_begin:])
    print(f"Text: {full_text}")

    # Count words
    words = full_text.split()
    print(f"Word count: {len(words)}")

if __name__ == "__main__":
    main()
