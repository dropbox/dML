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
Test if scaling magnitude produces audible audio.
"""

import sys

sys.path.insert(0, "/Users/ayates/model_mlx_migration")

import mlx.core as mx
import numpy as np
import scipy.io.wavfile as wav

from tools.pytorch_to_mlx.converters.kokoro_converter import KokoroConverter


def main():
    print("Loading model...")
    converter = KokoroConverter()
    model, config, _ = converter.load_from_hf("hexgrad/Kokoro-82M")

    # Load voice
    voice = converter.load_voice("af_heart")

    # Input
    test_ids = [0, 50, 47, 54, 54, 57, 0]
    input_ids = mx.array([test_ids])

    # Generate audio
    audio = model(input_ids, voice)
    mx.eval(audio)

    rms = float(mx.sqrt(mx.mean(audio**2)))
    print("\nOriginal audio:")
    print(f"  Shape: {audio.shape}")
    print(f"  RMS: {rms:.6f}")
    print(f"  Range: [{float(audio.min()):.4f}, {float(audio.max()):.4f}]")

    # Scale up to audible level
    target_rms = 0.1
    scale = target_rms / (rms + 1e-8)
    scaled_audio = audio * scale
    mx.eval(scaled_audio)

    scaled_rms = float(mx.sqrt(mx.mean(scaled_audio**2)))
    print(f"\nScaled audio (target RMS={target_rms}):")
    print(f"  Scale factor: {scale:.2f}")
    print(f"  RMS: {scaled_rms:.6f}")
    print(
        f"  Range: [{float(scaled_audio.min()):.4f}, {float(scaled_audio.max()):.4f}]"
    )

    # Save to WAV file
    audio_np = np.array(scaled_audio)[0]
    # Normalize to [-1, 1] for WAV
    audio_np = np.clip(audio_np, -1, 1)
    wav.write("/tmp/kokoro_test_scaled.wav", 24000, (audio_np * 32767).astype(np.int16))
    print("\nSaved scaled audio to /tmp/kokoro_test_scaled.wav")

    # Also save original
    audio_orig_np = np.array(audio)[0]
    audio_orig_np = np.clip(audio_orig_np, -1, 1)
    wav.write(
        "/tmp/kokoro_test_original.wav", 24000, (audio_orig_np * 32767).astype(np.int16)
    )
    print("Saved original audio to /tmp/kokoro_test_original.wav")


if __name__ == "__main__":
    main()
