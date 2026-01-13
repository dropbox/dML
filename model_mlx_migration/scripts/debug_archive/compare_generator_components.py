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
Compare Kokoro Generator components between PyTorch and MLX.
Focus on: noise_convs, source STFT, ISTFT.
"""

import sys

sys.path.insert(0, "/Users/ayates/model_mlx_migration")

import mlx.core as mx
import numpy as np
import torch
import torch.nn as nn

from tools.pytorch_to_mlx.converters.kokoro_converter import KokoroConverter

# Load models
print("Loading MLX model...")
converter = KokoroConverter()
mlx_model, config, _ = converter.load_from_hf("hexgrad/Kokoro-82M")

print("Loading PyTorch checkpoint...")
ckpt = torch.load(
    "/Users/ayates/models/kokoro/kokoro-v1_0.pth", map_location="cpu", weights_only=True
)
pt_decoder = ckpt["decoder"]

mlx_gen = mlx_model.decoder.generator


def test_source_stft():
    """Compare source STFT between PyTorch and MLX."""
    print("\n" + "=" * 50)
    print("=== Source STFT Comparison ===")
    print("=" * 50)

    np.random.seed(42)
    batch, samples = 1, 8400
    x_np = np.random.randn(batch, samples).astype(np.float32) * 0.1

    print(f"Input: shape={x_np.shape}, mean={x_np.mean():.6f}")

    # PyTorch STFT
    n_fft, hop = 20, 5
    window = torch.hann_window(n_fft)
    pt_x = torch.from_numpy(x_np)

    pt_stft = torch.stft(
        pt_x,
        n_fft,
        hop,
        n_fft,
        window=window,
        return_complex=True,
        center=True,
        pad_mode="reflect",
    )
    pt_mag = torch.abs(pt_stft)
    pt_phase = torch.angle(pt_stft)
    pt_har = torch.cat([pt_mag, pt_phase], dim=1).transpose(1, 2)  # [B, F, 22]
    pt_np = pt_har.numpy()

    print(f"PyTorch STFT: shape={pt_np.shape}")
    print(
        f"  mag mean={pt_np[..., :11].mean():.6f}, phase mean={pt_np[..., 11:].mean():.6f}"
    )

    # MLX STFT
    mlx_x = mx.array(x_np)
    mlx_har = mlx_gen._source_stft(mlx_x)
    mx.eval(mlx_har)
    mlx_np = np.array(mlx_har)

    print(f"MLX STFT: shape={mlx_np.shape}")
    print(
        f"  mag mean={mlx_np[..., :11].mean():.6f}, phase mean={mlx_np[..., 11:].mean():.6f}"
    )

    # Compare magnitudes (should match exactly)
    min_frames = min(pt_np.shape[1], mlx_np.shape[1])
    mag_diff = np.abs(pt_np[:, :min_frames, :11] - mlx_np[:, :min_frames, :11])
    print(f"\nComparison (first {min_frames} frames):")
    print(f"  Magnitude max diff: {mag_diff.max():.8f}")

    # Phase can differ by ±2π (wrapping) but complex values should match
    # Check complex spectrum equivalence instead of raw phase
    pt_spec = pt_np[:, :min_frames, :11] * np.exp(1j * pt_np[:, :min_frames, 11:])
    mlx_spec = mlx_np[:, :min_frames, :11] * np.exp(1j * mlx_np[:, :min_frames, 11:])
    complex_diff = np.abs(pt_spec - mlx_spec)
    print(f"  Complex spectrum max diff: {complex_diff.max():.8f}")
    print("  (Phase ±π wrapping is expected and doesn't affect reconstruction)")


def test_noise_conv():
    """Compare noise_conv outputs between PyTorch and MLX."""
    print("\n" + "=" * 50)
    print("=== noise_conv Comparison ===")
    print("=" * 50)

    np.random.seed(42)
    batch, frames, channels = 1, 1201, 22
    x_np = np.random.randn(batch, frames, channels).astype(np.float32) * 0.1

    print(f"Input: shape={x_np.shape}, mean={x_np.mean():.6f}")

    # Test noise_conv[0] (kernel=12, stride=6)
    print("\n--- noise_conv[0] (kernel=12, stride=6) ---")

    # PyTorch
    pt_w = pt_decoder["module.generator.noise_convs.0.weight"]  # [out, in, k]
    pt_b = pt_decoder["module.generator.noise_convs.0.bias"]

    pt_x = torch.from_numpy(x_np).transpose(1, 2)  # NCL
    padding = (6 + 1) // 2  # = 3
    pt_out = nn.functional.conv1d(pt_x, pt_w, pt_b, stride=6, padding=padding)
    pt_np = pt_out.numpy().transpose(0, 2, 1)  # NLC

    print(f"PyTorch: shape={pt_np.shape}, mean={pt_np.mean():.6f}")

    # MLX
    mlx_x = mx.array(x_np)
    mlx_out = mlx_gen.noise_convs[0](mlx_x)
    mx.eval(mlx_out)
    mlx_np = np.array(mlx_out)

    print(f"MLX: shape={mlx_np.shape}, mean={mlx_np.mean():.6f}")
    print(f"Max diff: {np.abs(pt_np - mlx_np).max():.8f}")

    # Test noise_conv[1] (kernel=1, stride=1)
    print("\n--- noise_conv[1] (kernel=1, stride=1) ---")

    # Input for stage 1 is output of noise_res[0], use random for simplicity
    x1_np = np.random.randn(batch, 200, 22).astype(np.float32) * 0.1

    # PyTorch
    pt_w = pt_decoder["module.generator.noise_convs.1.weight"]  # [out, in, k]
    pt_b = pt_decoder["module.generator.noise_convs.1.bias"]

    pt_x1 = torch.from_numpy(x1_np).transpose(1, 2)  # NCL
    pt_out1 = nn.functional.conv1d(
        pt_x1, pt_w, pt_b, stride=1, padding=0
    )  # padding=0 for kernel=1
    pt_np1 = pt_out1.numpy().transpose(0, 2, 1)  # NLC

    print(f"PyTorch: shape={pt_np1.shape}, mean={pt_np1.mean():.6f}")

    # MLX
    mlx_x1 = mx.array(x1_np)
    mlx_out1 = mlx_gen.noise_convs[1](mlx_x1)
    mx.eval(mlx_out1)
    mlx_np1 = np.array(mlx_out1)

    print(f"MLX: shape={mlx_np1.shape}, mean={mlx_np1.mean():.6f}")
    print(f"Max diff: {np.abs(pt_np1 - mlx_np1).max():.8f}")


def test_istft():
    """Compare ISTFT outputs between PyTorch and MLX."""
    print("\n" + "=" * 50)
    print("=== ISTFT Comparison ===")
    print("=" * 50)

    np.random.seed(42)
    n_fft, hop = 20, 5
    batch, frames, n_bins = 1, 100, n_fft // 2 + 1

    mag_np = np.random.rand(batch, frames, n_bins).astype(np.float32) * 0.5
    phase_np = (
        (np.random.rand(batch, frames, n_bins).astype(np.float32) - 0.5) * 2 * np.pi
    )

    print(f"Input mag: shape={mag_np.shape}, mean={mag_np.mean():.6f}")

    # PyTorch ISTFT
    window = torch.hann_window(n_fft)
    pt_mag = torch.from_numpy(mag_np).transpose(1, 2)  # [B, F, T] -> [B, T, F] is wrong
    pt_phase = torch.from_numpy(phase_np).transpose(1, 2)
    spec = pt_mag * torch.exp(pt_phase * 1j)

    pt_audio = torch.istft(
        spec,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        center=True,
        return_complex=False,
    )
    pt_np = pt_audio.numpy()

    print(f"PyTorch ISTFT: shape={pt_np.shape}")
    print(f"  RMS: {np.sqrt((pt_np**2).mean()):.6f}")

    # MLX ISTFT
    mlx_mag = mx.array(mag_np)
    mlx_phase = mx.array(phase_np)
    mlx_audio = mlx_gen._istft_synthesis(mlx_mag, mlx_phase)
    mx.eval(mlx_audio)
    mlx_np = np.array(mlx_audio)

    print(f"MLX ISTFT: shape={mlx_np.shape}")
    print(f"  RMS: {np.sqrt((mlx_np**2).mean()):.6f}")

    # Compare
    min_len = min(pt_np.shape[-1], mlx_np.shape[-1])
    diff = np.abs(pt_np[..., :min_len] - mlx_np[..., :min_len])
    print(f"\nComparison (first {min_len} samples):")
    print(f"  Max diff: {diff.max():.8f}")
    print(f"  Mean diff: {diff.mean():.8f}")


def main():
    test_noise_conv()
    test_source_stft()
    test_istft()

    print("\n" + "=" * 50)
    print("Done!")


if __name__ == "__main__":
    main()
