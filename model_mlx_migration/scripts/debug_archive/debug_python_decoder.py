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
Debug Python decoder embeddings and intermediate values.
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

    # Check token embedding
    te = model.decoder.token_embedding.weight
    mx.eval(te)
    print(f"[DEBUG PYTHON] token_embedding shape={te.shape} min={float(te.min()):.6f} max={float(te.max()):.6f}")

    # Check positional embedding
    pe = model.decoder.positional_embedding
    mx.eval(pe)
    print(f"[DEBUG PYTHON] positional_embedding shape={pe.shape} min={float(pe.min()):.6f} max={float(pe.max()):.6f}")

    # Test with prompt tokens
    prefix = [50258, 50259, 50360, 50365]
    tokens = mx.array([prefix])

    # Token lookup
    x = te[tokens]
    mx.eval(x)
    print(f"[DEBUG PYTHON] after token lookup x shape={x.shape} min={float(x.min()):.6f} max={float(x.max()):.6f}")

    # Add positional embedding
    pos_emb = pe[:4]
    x = x + pos_emb
    mx.eval(x)
    print(f"[DEBUG PYTHON] after pos_emb x min={float(x.min()):.6f} max={float(x.max()):.6f}")

    # Now run full decode and check output at step 0
    audio = load_audio(AUDIO_FILE)
    mel = log_mel_spectrogram(audio, model.config.n_mels, model.config.n_fft, model.config.hop_length)
    n_frames = 3000
    if mel.shape[0] < n_frames:
        mel = mx.pad(mel, [(0, n_frames - mel.shape[0]), (0, 0)])
    mel = mel[:n_frames]

    audio_features = model.encoder(mel[None])
    mx.eval(audio_features)
    print(f"[DEBUG PYTHON] encoder_output min={float(audio_features.min()):.6f} max={float(audio_features.max()):.6f}")

    # Single decode step
    logits, kv_cache, _, _ = model.decoder(tokens, audio_features)
    mx.eval(logits)

    last_logits = logits[0, -1].astype(mx.float32)
    mx.eval(last_logits)

    print(f"[DEBUG PYTHON] logits step 0: and(293)={float(last_logits[293]):.5f} teeth(7798)={float(last_logits[7798]):.5f} comma(11)={float(last_logits[11]):.5f}")

if __name__ == "__main__":
    main()
