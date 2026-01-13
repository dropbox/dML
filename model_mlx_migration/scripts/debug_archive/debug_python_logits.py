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
Export Python decoder logits at each step for comparison with C++.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
from tools.whisper_mlx import WhisperMLX
from tools.whisper_mlx.audio import load_audio, log_mel_spectrogram

AUDIO_FILE = "data/librispeech/dev-clean/1272/128104/1272-128104-0004.flac"

def main():
    print("Loading model...")
    model = WhisperMLX.from_pretrained("mlx-community/whisper-large-v3-turbo", dtype=mx.float16)

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
    print(f"Encoder output: shape={audio_features.shape}, min={float(audio_features.min()):.4f}, max={float(audio_features.max()):.4f}")

    # Build prefix - same as C++
    prefix = [
        50258,  # <|startoftranscript|>
        50259,  # <|en|>
        50360,  # <|transcribe|>
        50365,  # <|notimestamps|>
    ]

    print("\n=== Step-by-step decode (greedy) ===")

    tokens = list(prefix)
    kv_cache = None

    for step in range(100):
        tokens_mx = mx.array([tokens])
        if kv_cache is None:
            logits, kv_cache, _, _ = model.decoder(tokens_mx, audio_features)
        else:
            # Only pass last token for incremental decode
            logits, kv_cache, _, _ = model.decoder(tokens_mx[:, -1:], audio_features, kv_cache=kv_cache)

        mx.eval(logits)

        # Get logits for last position
        last_logits = logits[0, -1]

        # Convert to float32 like C++ does
        last_logits = last_logits.astype(mx.float32)
        mx.eval(last_logits)

        # Greedy: pick argmax
        next_token = int(mx.argmax(last_logits).item())

        # Print key token logits (same format as C++)
        token_293 = float(last_logits[293].item())   # and
        token_7798 = float(last_logits[7798].item())  # teeth
        token_11 = float(last_logits[11].item())   # comma

        print(f"[PYTHON] Step {step}: and(293)={token_293:.5f} teeth(7798)={token_7798:.5f} comma(11)={token_11:.5f} -> tok={next_token}")

        tokens.append(next_token)

        # Stop at EOT
        if next_token == 50257:
            break

    print("\n=== Final transcription ===")
    from transformers import WhisperTokenizer
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v3")
    text = tokenizer.decode(tokens[4:])  # Skip prefix
    print(text[:500])

if __name__ == "__main__":
    main()
