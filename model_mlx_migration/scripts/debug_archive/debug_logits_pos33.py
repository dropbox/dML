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
Check logits at position 33 (after "poem") to see if period and ' mr' are close.
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

# Tokens up to position 32 (ends with " poem")
PREFIX = [50258, 50259, 50359, 50364, 9355, 8903, 311, 5242, 366, 257, 1333, 295, 493, 17652, 293, 7938, 14880, 293, 25730, 311, 454, 34152, 4496, 34353, 82, 366, 382, 4048, 382, 257, 361, 18459, 13065]

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

    # Run prefix through decoder
    tokens = mx.array([PREFIX])
    logits, kv_cache, _, _ = model.decoder(tokens, audio_features)
    mx.eval(logits)

    # Get logits for position 33 (after "poem")
    raw_logits = logits[:, -1].astype(mx.float32)
    mx.eval(raw_logits)

    # Apply filters
    current_tokens = mx.array([PREFIX])
    filtered_logits = apply_filters(raw_logits, current_tokens[:, sample_begin:], logit_filters)
    mx.eval(filtered_logits)

    # Check specific tokens
    print("\n=== Logits at position 33 (after 'poem') ===")
    print(f"Last token: {PREFIX[-1]} = {repr(tokenizer.decode([PREFIX[-1]]))}")

    tokens_to_check = [13, 33660, 366, 257, 382, 2221]  # '.', ' mr', ' are', ' a', ' as', ' Mr'
    for tok in tokens_to_check:
        raw = float(raw_logits[0, tok].item())
        filt = float(filtered_logits[0, tok].item())
        text = tokenizer.decode([tok])
        print(f"  Token {tok:5d} ({repr(text):>12}): raw={raw:8.4f}, filtered={filt:8.4f}")

    # Get top 10 candidates
    print("\n=== Top 10 candidates (filtered logits) ===")
    top_indices = mx.argsort(filtered_logits, axis=-1)[:, -10:]
    mx.eval(top_indices)

    for i in range(10):
        idx = int(top_indices[0, -(i+1)].item())
        raw = float(raw_logits[0, idx].item())
        filt = float(filtered_logits[0, idx].item())
        text = tokenizer.decode([idx])
        print(f"  {i+1:2d}. Token {idx:5d} ({repr(text):>15}): raw={raw:8.4f}, filtered={filt:8.4f}")

    # Check if period is in top 20
    top20 = mx.argsort(filtered_logits, axis=-1)[:, -20:]
    mx.eval(top20)
    top20_list = [int(x) for x in top20[0].tolist()]
    print("\n=== Period token (13) rank ===")
    if 13 in top20_list:
        rank = 20 - top20_list.index(13)
        print(f"  Period (13) is rank {rank} in top 20")
    else:
        # Find actual rank
        all_ranks = mx.argsort(filtered_logits, axis=-1)
        mx.eval(all_ranks)
        all_list = [int(x) for x in all_ranks[0].tolist()]
        if 13 in all_list:
            rank = len(all_list) - all_list.index(13)
            print(f"  Period (13) is rank {rank}")

if __name__ == "__main__":
    main()
