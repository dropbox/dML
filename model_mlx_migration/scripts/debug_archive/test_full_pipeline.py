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
Test full Kokoro pipeline with text-to-audio.
"""

import sys

sys.path.insert(0, "/Users/ayates/model_mlx_migration")

import mlx.core as mx

from tools.pytorch_to_mlx.converters.kokoro_converter import KokoroConverter


def main():
    print("Loading Kokoro model...")
    converter = KokoroConverter()
    model, config, _ = converter.load_from_hf("hexgrad/Kokoro-82M")

    # Load voice (full 256-dim)
    voice = converter.load_voice("af_heart")
    print(f"Voice shape: {voice.shape}")

    # Create phoneme input
    # Use a simple test: "Hello"
    # From config vocab, we can map phonemes to IDs
    # For now, use dummy IDs that are valid

    # Create a simple test sequence with valid token IDs
    # IDs 16-68 are common phonemes (space, letters)
    test_ids = [0, 50, 47, 54, 54, 57, 0]  # [BOS, h, e, l, l, o, EOS] roughly
    input_ids = mx.array([test_ids])
    print(f"Input IDs: {test_ids}, shape={input_ids.shape}")

    # Run full model with tracing
    print("\nRunning full model pipeline...")
    try:
        # Manual pipeline to trace values (following model.__call__)
        attention_mask = mx.ones_like(input_ids)

        # Split voice
        style = voice[:, :128]
        speaker = voice[:, 128:]

        # BERT + bert_encoder
        bert_out = model.bert(input_ids, attention_mask)
        bert_enc = model.bert_encoder(bert_out)
        mx.eval(bert_enc)

        # TextEncoder (takes input_ids, not bert_enc!)
        text_enc = model.text_encoder(input_ids, attention_mask)
        mx.eval(text_enc)

        # Combine
        combined = bert_enc + text_enc
        mx.eval(combined)
        print(
            f"\nCombined encoding: shape={combined.shape}, mean={float(combined.mean()):.4f}"
        )

        # Predictor (uses speaker, not style!)
        duration, f0, noise = model.predictor(combined, speaker, attention_mask)
        mx.eval(f0, noise)
        print(
            f"Predictor F0: mean={float(f0.mean()):.2f}Hz, range=[{float(f0.min()):.2f}, {float(f0.max()):.2f}]"
        )
        print(f"Predictor noise: mean={float(noise.mean()):.4f}")

        # Decoder (uses style, not speaker!)
        audio = model.decoder(combined, f0, noise, style)
        mx.eval(audio)

        rms = float(mx.sqrt(mx.mean(audio**2)))
        print("\nAudio output:")
        print(f"  Shape: {audio.shape}")
        print(f"  RMS: {rms:.6f}")
        print(f"  Range: [{float(audio.min()):.4f}, {float(audio.max()):.4f}]")

        if rms < 0.01:
            print("\nWARNING: Audio is near-silent")
        else:
            print(f"\nSUCCESS: Audio has reasonable level (RMS = {rms:.4f})")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()

    # Also test decoder directly with different input scales
    print("\n=== Testing Decoder with various input scales ===")
    decoder = model.decoder

    # Style from voice
    style = voice[:, :128]
    print(f"Style shape: {style.shape}")

    # Create synthetic ASR features at different scales
    for scale in [0.01, 0.1, 1.0, 10.0]:
        asr = mx.random.normal((1, 50, 512)) * scale
        f0 = mx.ones((1, 50)) * 200.0  # 200 Hz
        noise = mx.random.normal((1, 50)) * 0.1

        audio = decoder(asr, f0, noise, style)
        mx.eval(audio)

        rms = float(mx.sqrt(mx.mean(audio**2)))
        print(f"  ASR scale={scale:.2f}: RMS={rms:.6f}")


if __name__ == "__main__":
    main()
