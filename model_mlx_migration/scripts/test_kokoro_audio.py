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
Test Kokoro audio output quality after noise_convs fix.
"""

import sys

sys.path.insert(0, "/Users/ayates/model_mlx_migration")

import mlx.core as mx

# Load the model
from tools.pytorch_to_mlx.converters.kokoro_converter import KokoroConverter


def test_audio():
    print("Loading Kokoro model...")
    converter = KokoroConverter()

    # Use load_from_hf to load model
    print("Loading from HuggingFace...")
    model, config, state_dict = converter.load_from_hf("hexgrad/Kokoro-82M")
    print("Model loaded successfully")

    # Load voice with phoneme_length (use 50 as default for testing)
    # Note: This test doesn't use voice_pack for synthesis, just checks loading
    voice_pack = converter.load_voice("af_heart", phoneme_length=50)
    print(f"Voice loaded: shape={voice_pack.shape}")

    # Simple test: Generate audio from dummy features
    print("\nTesting Generator directly...")

    # Get decoder and generator
    decoder = model.decoder
    generator = decoder.generator

    # Create test inputs
    batch = 1
    length = 20
    channels = 512  # Generator input channels
    style_dim = 128

    # Test with controlled input
    x = mx.random.normal((batch, length, channels)) * 0.1  # Small std
    s = mx.random.normal((batch, style_dim)) * 0.1
    f0 = mx.ones((batch, length)) * 200.0  # 200 Hz

    print(
        f"Input x: shape={x.shape}, mean={float(x.mean()):.4f}, std={float(x.std()):.4f}"
    )
    print(
        f"Input s: shape={s.shape}, mean={float(s.mean()):.4f}, std={float(s.std()):.4f}"
    )
    print(f"Input f0: shape={f0.shape}, mean={float(f0.mean()):.4f}")

    # Run generator
    audio = generator(x, s, f0)
    mx.eval(audio)

    rms = float(mx.sqrt(mx.mean(audio**2)))
    max_val = float(mx.max(mx.abs(audio)))

    print("\nGenerator output:")
    print(f"  Shape: {audio.shape}")
    print(f"  RMS: {rms:.6f}")
    print(f"  Max: {max_val:.6f}")
    print(f"  Min: {float(mx.min(audio)):.6f}")
    print(f"  Mean: {float(mx.mean(audio)):.6f}")

    if rms < 0.01:
        print("\nWARNING: Audio is near-silent (RMS < 0.01)")
    else:
        print(f"\nAudio has reasonable level (RMS = {rms:.4f})")

    # Test with fresh weights for comparison
    print("\n--- Testing with FRESH (random) weights ---")
    from tools.pytorch_to_mlx.converters.models.kokoro import Generator

    fresh_gen = Generator(model.config)

    fresh_audio = fresh_gen(x, s, f0)
    mx.eval(fresh_audio)

    fresh_rms = float(mx.sqrt(mx.mean(fresh_audio**2)))
    print(f"Fresh generator RMS: {fresh_rms:.6f}")
    print(f"Loaded generator RMS: {rms:.6f}")
    print(f"Ratio (loaded/fresh): {rms / fresh_rms if fresh_rms > 0 else 'N/A':.6f}")

    # Assertion for pytest (test functions should not return values)
    assert rms > 0.01, f"Audio is near-silent (RMS = {rms})"


if __name__ == "__main__":
    test_audio()
