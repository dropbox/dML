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
Compare full Kokoro pipeline: MLX vs PyTorch step by step.
"""

import sys

sys.path.insert(0, "/Users/ayates/model_mlx_migration")

import mlx.core as mx
import numpy as np
import torch

from tools.pytorch_to_mlx.converters.kokoro_converter import KokoroConverter


def main():
    print("=== Loading Models ===")

    # MLX
    converter = KokoroConverter()
    mlx_model, config, _ = converter.load_from_hf("hexgrad/Kokoro-82M")

    # PyTorch weights
    torch.load(
        "/Users/ayates/models/kokoro/kokoro-v1_0.pth",
        map_location="cpu",
        weights_only=True,
    )

    # Load voice
    voice = converter.load_voice("af_heart")
    np.array(voice)
    print(f"Voice shape: {voice.shape}")

    # Test input
    test_ids = [0, 50, 47, 54, 54, 57, 0]

    # ============ MLX Forward ============
    print("\n=== MLX Forward Pass ===")
    input_ids = mx.array([test_ids])
    attention_mask = mx.ones_like(input_ids)

    # BERT
    bert_out = mlx_model.bert(input_ids, attention_mask)
    bert_enc = mlx_model.bert_encoder(bert_out)
    mx.eval(bert_enc)
    print(
        f"BERT encoder: mean={float(bert_enc.mean()):.6f}, std={float(bert_enc.std()):.6f}"
    )

    # TextEncoder
    text_enc = mlx_model.text_encoder(input_ids, attention_mask)
    mx.eval(text_enc)
    print(
        f"TextEncoder: mean={float(text_enc.mean()):.6f}, std={float(text_enc.std()):.6f}"
    )

    # Combined
    combined = bert_enc + text_enc
    mx.eval(combined)
    print(
        f"Combined: mean={float(combined.mean()):.6f}, std={float(combined.std()):.6f}"
    )

    # Split voice
    style = voice[:, :128]
    speaker = voice[:, 128:]

    # Predictor
    duration, f0, noise = mlx_model.predictor(combined, speaker, attention_mask)
    mx.eval(duration, f0, noise)
    print("\nPredictor outputs:")
    print(f"  duration: mean={float(duration.mean()):.4f}")
    print(
        f"  f0: mean={float(f0.mean()):.2f}Hz, range=[{float(f0.min()):.2f}, {float(f0.max()):.2f}]"
    )
    print(f"  noise: mean={float(noise.mean()):.6f}, std={float(noise.std()):.6f}")

    # Decoder
    audio = mlx_model.decoder(combined, f0, noise, style)
    mx.eval(audio)

    rms = float(mx.sqrt(mx.mean(audio**2)))
    print("\nMLX Audio:")
    print(f"  Shape: {audio.shape}")
    print(f"  RMS: {rms:.6f}")
    print(f"  Range: [{float(audio.min()):.4f}, {float(audio.max()):.4f}]")

    # Save intermediate values
    combined_np = np.array(combined)
    f0_np = np.array(f0)
    noise_np = np.array(noise)
    style_np = np.array(style)

    # ============ Check with fixed inputs ============
    print("\n=== Comparing Decoder with Fixed Inputs ===")

    # Use simple synthetic inputs to test decoder in isolation
    np.random.seed(42)
    batch = 1
    seq_len = 7
    hidden_dim = 512
    style_dim = 128

    syn_asr = np.random.randn(batch, seq_len, hidden_dim).astype(np.float32) * 0.1
    syn_f0 = np.abs(np.random.randn(batch, seq_len * 4).astype(np.float32)) * 100 + 200
    syn_noise = np.random.randn(batch, seq_len * 4).astype(np.float32) * 0.1
    syn_style = np.random.randn(batch, style_dim).astype(np.float32) * 0.1

    print("Synthetic inputs:")
    print(f"  asr: shape={syn_asr.shape}, mean={syn_asr.mean():.4f}")
    print(f"  f0: shape={syn_f0.shape}, mean={syn_f0.mean():.2f}Hz")
    print(f"  noise: shape={syn_noise.shape}, mean={syn_noise.mean():.4f}")
    print(f"  style: shape={syn_style.shape}, mean={syn_style.mean():.4f}")

    # MLX decoder
    mlx_syn_audio = mlx_model.decoder(
        mx.array(syn_asr), mx.array(syn_f0), mx.array(syn_noise), mx.array(syn_style)
    )
    mx.eval(mlx_syn_audio)

    syn_rms = float(mx.sqrt(mx.mean(mlx_syn_audio**2)))
    print("\nSynthetic Input Audio:")
    print(f"  Shape: {mlx_syn_audio.shape}")
    print(f"  RMS: {syn_rms:.6f}")
    print(
        f"  Range: [{float(mlx_syn_audio.min()):.4f}, {float(mlx_syn_audio.max()):.4f}]"
    )

    # ============ Test with real inputs but from model ============
    print("\n=== Testing Real vs Synthetic ===")

    # Use real model outputs
    real_audio = mlx_model.decoder(
        mx.array(combined_np), mx.array(f0_np), mx.array(noise_np), mx.array(style_np)
    )
    mx.eval(real_audio)

    real_rms = float(mx.sqrt(mx.mean(real_audio**2)))
    print(f"Real input audio RMS: {real_rms:.6f}")

    # Compare
    print("\nComparison:")
    print(f"  Synthetic input RMS: {syn_rms:.6f}")
    print(f"  Real input RMS: {real_rms:.6f}")
    print(f"  Ratio: {real_rms / syn_rms:.2f}x")


if __name__ == "__main__":
    main()
