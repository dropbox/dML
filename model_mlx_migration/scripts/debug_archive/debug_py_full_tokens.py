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
Get complete Python tokens with timestamps
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
from tools.whisper_mlx import WhisperMLX
from tools.whisper_mlx.audio import load_audio, log_mel_spectrogram
from tools.whisper_mlx.tokenizer import get_whisper_tokenizer

AUDIO_FILE = "data/librispeech/dev-clean/1272/128104/1272-128104-0004.flac"

def main():
    print("Loading model...")
    model = WhisperMLX.from_pretrained("mlx-community/whisper-large-v3-turbo", dtype=mx.float16)
    tokenizer = get_whisper_tokenizer(multilingual=True)

    # Load and process audio
    audio = load_audio(AUDIO_FILE)

    # Compute mel
    mel = log_mel_spectrogram(audio, model.config.n_mels, model.config.n_fft, model.config.hop_length)
    n_frames = 3000
    if mel.shape[0] < n_frames:
        mel = mx.pad(mel, [(0, n_frames - mel.shape[0]), (0, 0)])
    mel = mel[:n_frames]

    # Encode
    audio_features = model.encoder(mel[None])
    mx.eval(audio_features)

    # Decode
    tokens, segments, _, _ = model._decode_with_metrics(
        audio_features,
        tokenizer,
        temperature=0.0,
        audio_duration=len(audio) / 16000,
    )

    # Add SOT sequence
    sot_sequence = list(tokenizer.sot_sequence)
    full_tokens = sot_sequence + tokens

    # Print all tokens
    timestamp_begin = tokenizer.timestamp_begin
    eot = tokenizer.eot

    print(f"\n=== Python Tokens ({len(full_tokens)}) ===")
    print(f"timestamp_begin={timestamp_begin}, eot={eot}")

    for i, t in enumerate(full_tokens):
        text = tokenizer.decode([t])
        if t >= timestamp_begin and t < eot:
            ts_value = (t - timestamp_begin) * 0.02
            print(f"{i:3d}: {t:5d} = TIMESTAMP {ts_value:.2f}s")
        elif t == eot:
            print(f"{i:3d}: {t:5d} = EOT")
        else:
            print(f"{i:3d}: {t:5d} = {repr(text)}")

    # Show full decoded text
    print("\n=== Full Text ===")
    print(tokenizer.decode(tokens))

    # Check for repetitions
    print("\n=== Repetition Check ===")
    for i in range(len(full_tokens) - 1):
        if full_tokens[i] == full_tokens[i+1] and full_tokens[i] < timestamp_begin:
            text = tokenizer.decode([full_tokens[i]])
            print(f"  Position {i}-{i+1}: token {full_tokens[i]} ({repr(text)}) repeated")

if __name__ == "__main__":
    main()
