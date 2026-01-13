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
Compare logits at position 33 for BOTH paths:
1. Python path: "idles" (tokens 4496, 904)
2. C++ path: "idylls" (tokens 4496, 34353, 82)
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

# Common prefix up to "id" (position 22)
COMMON_PREFIX = [50258, 50259, 50359, 50364, 9355, 8903, 311, 5242, 366, 257, 1333, 295, 493, 17652, 293, 7938, 14880, 293, 25730, 311, 454, 34152, 4496]

# Python path: "idles" = 4496 + 904
PYTHON_CONTINUATION = [904, 366, 382, 4048, 382, 257, 361, 18459, 13065]  # les are as national as a jingo poem

# C++ path: "idylls" = 4496 + 34353 + 82
CPP_CONTINUATION = [34353, 82, 366, 382, 4048, 382, 257, 361, 18459, 13065]  # ylls are as national as a jingo poem

def test_path(model, audio_features, tokenizer, logit_filters, prefix, path_name):
    tokens = mx.array([prefix])
    logits, kv_cache, _, _ = model.decoder(tokens, audio_features)
    mx.eval(logits)

    raw_logits = logits[:, -1].astype(mx.float32)
    mx.eval(raw_logits)

    current_tokens = mx.array([prefix])
    sample_begin = 4
    filtered_logits = apply_filters(raw_logits, current_tokens[:, sample_begin:], logit_filters)
    mx.eval(filtered_logits)

    print(f"\n=== {path_name} path (after '{tokenizer.decode([prefix[-1]])}') ===")
    print(f"Prefix length: {len(prefix)} tokens")

    tokens_to_check = [13, 33660, 2221]  # '.', ' mr', ' Mr'
    for tok in tokens_to_check:
        filt = float(filtered_logits[0, tok].item())
        text = tokenizer.decode([tok])
        print(f"  Token {tok:5d} ({repr(text):>12}): filtered={filt:8.4f}")

    # Get top 3
    top_indices = mx.argsort(filtered_logits, axis=-1)[:, -3:]
    mx.eval(top_indices)
    print("  Top 3:")
    for i in range(3):
        idx = int(top_indices[0, -(i+1)].item())
        filt = float(filtered_logits[0, idx].item())
        text = tokenizer.decode([idx])
        print(f"    {i+1}. Token {idx} ({repr(text)}): {filt:.4f}")

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

    audio_features = model.encoder(mel[None])
    mx.eval(audio_features)

    # Build logit filters
    sample_begin = 4
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

    # Test Python path (with "idles")
    python_prefix = COMMON_PREFIX + PYTHON_CONTINUATION
    test_path(model, audio_features, tokenizer, logit_filters, python_prefix, "Python (idles)")

    # Test C++ path (with "idylls")
    cpp_prefix = COMMON_PREFIX + CPP_CONTINUATION
    test_path(model, audio_features, tokenizer, logit_filters, cpp_prefix, "C++ (idylls)")

if __name__ == "__main__":
    main()
