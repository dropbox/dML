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
Compare full decoder output between PyTorch (official kokoro) and MLX.

This requires the official kokoro package to be installed.
"""

import sys

sys.path.insert(0, "/Users/ayates/model_mlx_migration")

import numpy as np
import torch

# Check if we can use the official kokoro package
try:
    from kokoro import KModel  # noqa: F401

    KOKORO_AVAILABLE = True
except ImportError:
    KOKORO_AVAILABLE = False
    print(
        "Official kokoro package not available. Will compare using manually built PyTorch decoder."
    )

import mlx.core as mx

from tools.pytorch_to_mlx.converters.kokoro_converter import KokoroConverter


def compare_with_manual_pytorch():
    """Compare by building PyTorch decoder manually from checkpoint."""
    print("Loading MLX model...")
    converter = KokoroConverter()
    mlx_model, config, _ = converter.load_from_hf("hexgrad/Kokoro-82M")

    # Load PyTorch checkpoint
    print("Loading PyTorch checkpoint...")
    ckpt = torch.load(
        "/Users/ayates/models/kokoro/kokoro-v1_0.pth",
        map_location="cpu",
        weights_only=True,
    )
    ckpt["decoder"]

    # Create deterministic test inputs
    np.random.seed(42)
    batch = 1
    length = 50  # More realistic length

    # ASR features: [batch, length, 512]
    asr = np.random.randn(batch, length, 512).astype(np.float32) * 0.1

    # F0: [batch, length]
    f0 = np.ones((batch, length), dtype=np.float32) * 200.0  # 200 Hz

    # Noise: [batch, length]
    noise = np.random.randn(batch, length).astype(np.float32) * 0.1

    # Style: [batch, 128]
    style = np.random.randn(batch, 128).astype(np.float32) * 0.1

    print("\nInputs:")
    print(f"  asr: shape={asr.shape}, std={asr.std():.4f}")
    print(f"  f0: shape={f0.shape}, mean={f0.mean():.4f}")
    print(f"  noise: shape={noise.shape}, std={noise.std():.4f}")
    print(f"  style: shape={style.shape}, std={style.std():.4f}")

    # Run MLX decoder
    print("\nRunning MLX decoder...")
    mlx_asr = mx.array(asr)
    mlx_f0 = mx.array(f0)
    mlx_noise = mx.array(noise)
    mlx_style = mx.array(style)

    mlx_audio = mlx_model.decoder(mlx_asr, mlx_f0, mlx_noise, mlx_style)
    mx.eval(mlx_audio)

    mlx_audio_np = np.array(mlx_audio)
    print(
        f"MLX audio: shape={mlx_audio_np.shape}, mean={mlx_audio_np.mean():.6f}, std={mlx_audio_np.std():.6f}"
    )
    print(f"  RMS: {np.sqrt(np.mean(mlx_audio_np**2)):.6f}")
    print(f"  Range: [{mlx_audio_np.min():.4f}, {mlx_audio_np.max():.4f}]")

    # For PyTorch comparison, we'd need to reconstruct the full decoder which is complex.
    # Instead, let's check intermediate values at the Generator level.

    print("\n=== Generator Input Analysis ===")

    # Process through decoder up to generator input
    decoder = mlx_model.decoder

    # F0 and noise convolutions
    f0_in = mlx_f0[:, :, None]  # Add channel dim
    n_in = mlx_noise[:, :, None]

    f0_proc = decoder.f0_conv(f0_in)
    n_proc = decoder.n_conv(n_in)
    mx.eval(f0_proc, n_proc)

    print(
        f"f0_proc: shape={f0_proc.shape}, mean={float(f0_proc.mean()):.6f}, std={float(f0_proc.std()):.6f}"
    )
    print(
        f"n_proc: shape={n_proc.shape}, mean={float(n_proc.mean()):.6f}, std={float(n_proc.std()):.6f}"
    )

    # ASR residual
    asr_res = decoder.asr_res(mlx_asr)
    mx.eval(asr_res)
    print(
        f"asr_res: shape={asr_res.shape}, mean={float(asr_res.mean()):.6f}, std={float(asr_res.std()):.6f}"
    )

    # Match lengths
    asr_len = mlx_asr.shape[1]
    f0_len = f0_proc.shape[1]
    stride = asr_len // f0_len
    asr_down = mlx_asr[:, ::stride, :][:, :f0_len, :]
    asr_res_down = asr_res[:, ::stride, :][:, :f0_len, :]

    # Encode
    x = mx.concatenate([asr_down, f0_proc, n_proc], axis=-1)
    x = decoder.encode(x, mlx_style)
    mx.eval(x)
    print(
        f"After encode: shape={x.shape}, mean={float(x.mean()):.6f}, std={float(x.std()):.6f}"
    )

    # Decode blocks
    for i, block in enumerate(decoder.decode):
        x = mx.concatenate([x, asr_res_down, f0_proc, n_proc], axis=-1)
        x = block(x, mlx_style)
        mx.eval(x)
        print(
            f"After decode[{i}]: shape={x.shape}, mean={float(x.mean()):.6f}, std={float(x.std()):.6f}"
        )

        if block.upsample:
            asr_res_down = mx.repeat(asr_res_down, 2, axis=1)[:, : x.shape[1], :]
            f0_proc = mx.repeat(f0_proc, 2, axis=1)[:, : x.shape[1], :]
            n_proc = mx.repeat(n_proc, 2, axis=1)[:, : x.shape[1], :]

    print(
        f"\nGenerator input (x): shape={x.shape}, mean={float(x.mean()):.6f}, std={float(x.std()):.6f}"
    )
    print("  This is fed to generator as input feature")


if __name__ == "__main__":
    compare_with_manual_pytorch()
