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
Save encoder output from Python for comparison with C++
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import mlx.core as mx
from tools.whisper_mlx import WhisperMLX
from tools.whisper_mlx.audio import load_audio, log_mel_spectrogram

AUDIO_FILE = "data/librispeech/dev-clean/1272/128104/1272-128104-0004.flac"

def main():
    print("Loading model...")
    model = WhisperMLX.from_pretrained("mlx-community/whisper-large-v3-turbo", dtype=mx.float16)

    # Load and process audio
    audio = load_audio(AUDIO_FILE)
    print(f"Audio length: {len(audio)/16000:.2f}s")

    # Compute mel spectrogram
    mel = log_mel_spectrogram(audio, model.config.n_mels, model.config.n_fft, model.config.hop_length)
    print(f"Mel shape: {mel.shape}")

    # Pad to 30s
    n_frames = 3000
    if mel.shape[0] < n_frames:
        mel = mx.pad(mel, [(0, n_frames - mel.shape[0]), (0, 0)])
    mel = mel[:n_frames]
    print(f"Padded mel shape: {mel.shape}")

    # Encode
    audio_features = model.encoder(mel[None])
    mx.eval(audio_features)

    print(f"Encoder output shape: {audio_features.shape}")
    print(f"Encoder output dtype: {audio_features.dtype}")

    # Convert to numpy and save
    enc_np = np.array(audio_features)
    np.save("/tmp/python_encoder_output.npy", enc_np)
    print("Saved encoder output to /tmp/python_encoder_output.npy")

    # Print statistics
    print("\nEncoder output stats:")
    print(f"  Min: {enc_np.min():.6f}")
    print(f"  Max: {enc_np.max():.6f}")
    print(f"  Mean: {enc_np.mean():.6f}")
    print(f"  Std: {enc_np.std():.6f}")

    # Print first few values
    print("\nFirst 10 values at position [0,0,:]:")
    print(f"  {enc_np[0, 0, :10]}")

    print("\nFirst 10 values at position [0,500,:]:")
    print(f"  {enc_np[0, 500, :10]}")

    # Also save mel for comparison
    mel_np = np.array(mel)
    np.save("/tmp/python_mel.npy", mel_np)
    print("\nSaved mel to /tmp/python_mel.npy")
    print(f"Mel stats: min={mel_np.min():.4f}, max={mel_np.max():.4f}, mean={mel_np.mean():.4f}")

if __name__ == "__main__":
    main()
