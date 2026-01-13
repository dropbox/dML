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
Debug logits at "teeth" divergence point for file 0004.
Python predicts comma (11), C++ predicts teeth (7798).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
from tools.whisper_mlx import WhisperMLX
from tools.whisper_mlx.audio import load_audio, log_mel_spectrogram
from transformers import WhisperTokenizer

AUDIO_FILE = "data/librispeech/dev-clean/1272/128104/1272-128104-0004.flac"

def main():
    print("Loading model...")
    model = WhisperMLX.from_pretrained("mlx-community/whisper-large-v3-turbo", dtype=mx.float16)
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v3")

    # Load and process audio (full 30s)
    audio = load_audio(AUDIO_FILE)
    mel = log_mel_spectrogram(audio, model.config.n_mels, model.config.n_fft, model.config.hop_length)
    n_frames = 3000  # 30s at 100fps
    if mel.shape[0] < n_frames:
        mel = mx.pad(mel, [(0, n_frames - mel.shape[0]), (0, 0)])
    mel = mel[:n_frames]

    print(f"Mel shape: {mel.shape}")

    # Encode
    audio_features = model.encoder(mel[None])
    mx.eval(audio_features)
    print(f"Encoder output shape: {audio_features.shape}")
    print(f"Encoder output stats: min={float(audio_features.min()):.4f}, max={float(audio_features.max()):.4f}")

    # Build prefix up to "teeth" - need to find exact tokens from full transcription
    # Start with standard prefix
    prefix = [
        50258,  # <|startoftranscript|>
        50259,  # <|en|>
        50360,  # <|transcribe|>
        50365,  # <|notimestamps|>
    ]

    # Now we need to decode step by step and find where "teeth" appears
    # Let's do greedy decode and track logits
    print("\n=== Step-by-step decode (greedy) ===")

    tokens = list(prefix)
    kv_cache = None

    for step in range(80):
        tokens_mx = mx.array([tokens])
        if kv_cache is None:
            logits, kv_cache, _, _ = model.decoder(tokens_mx, audio_features)
        else:
            # Only pass last token for incremental decode
            logits, kv_cache, _, _ = model.decoder(tokens_mx[:, -1:], audio_features, kv_cache=kv_cache)

        mx.eval(logits)

        # Get logits for last position
        last_logits = logits[0, -1]

        # Greedy: pick argmax
        next_token = int(mx.argmax(last_logits).item())

        # Check logits for key tokens at this step
        token_11 = float(last_logits[11].item())   # comma
        token_7798 = float(last_logits[7798].item())  # teeth
        token_13 = float(last_logits[13].item())   # period
        token_293 = float(last_logits[293].item())   # and

        # Decode current token
        token_text = tokenizer.decode([next_token])

        # Check if we're at "teeth" position
        is_teeth = (next_token == 7798)  # teeth token

        if step >= 55 or is_teeth:  # Print around teeth position (narrower range)
            print(f"Step {step}: tok={next_token:5d} ({token_text:15s})")
            print(f"         logits: comma(11)={token_11:.2f}, teeth(7798)={token_7798:.2f}, period(13)={token_13:.2f}, and(293)={token_293:.2f}")

            if is_teeth:
                # This is where divergence happens in C++
                # Get top 5 tokens
                top_indices = mx.argsort(last_logits)[-5:]
                print("         Top 5 candidates:")
                for i, idx in enumerate(reversed(top_indices.tolist())):
                    val = float(last_logits[idx].item())
                    txt = tokenizer.decode([idx])
                    print(f"           {i+1}. {idx} ({txt}): {val:.2f}")

        tokens.append(next_token)

        # Stop at EOT
        if next_token == 50257:
            break

    print("\n=== Final transcription ===")
    text = tokenizer.decode(tokens[4:])  # Skip prefix
    print(text[:500])

    # Now find the exact position where "teeth" first appears
    print("\n=== Token sequence around 'teeth' ===")
    for i, tok in enumerate(tokens):
        if tok == 7798:
            print(f"Position {i}: teeth token found")
            context = tokens[max(0,i-5):i+5]
            print(f"  Context: {context}")
            print(f"  Text: {tokenizer.decode(context)}")

if __name__ == "__main__":
    main()
