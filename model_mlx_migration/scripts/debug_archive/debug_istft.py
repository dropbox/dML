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

"""Debug ISTFT differences between PyTorch and MLX."""

import sys

sys.path.insert(0, "/Users/ayates/model_mlx_migration")

import math

import mlx.core as mx
import numpy as np
import torch


def pytorch_istft(mag, phase, n_fft, hop_length):
    """Reference PyTorch ISTFT."""
    window = torch.hann_window(n_fft)
    spec = mag * torch.exp(phase * 1j)
    audio = torch.istft(
        spec,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=True,
        return_complex=False,
    )
    return audio


def mlx_istft(mag, phase, n_fft, hop_length):
    """Current MLX ISTFT implementation."""
    batch, n_bins, frames = mag.shape

    # Complex spectrum
    spectrum = mag * mx.exp(1j * phase)

    # Transpose to [batch, frames, n_bins] for irfft
    spectrum = mx.transpose(spectrum, (0, 2, 1))

    # IRFFT per frame
    time_frames = mx.fft.irfft(spectrum, n=n_fft, axis=-1)
    # time_frames: [batch, frames, n_fft]

    # Hann window (periodic=True in PyTorch)
    n = mx.arange(n_fft, dtype=mx.float32)
    window = 0.5 * (1 - mx.cos(2 * math.pi * n / n_fft))

    # Apply window to each frame
    time_frames = time_frames * window

    # Output length for center=True (matches PyTorch)
    output_length = (frames - 1) * hop_length

    # Initialize output buffer
    audio = mx.zeros((batch, output_length + n_fft))
    window_sum = mx.zeros((output_length + n_fft,))

    # Overlap-add each frame
    for i in range(frames):
        start = i * hop_length
        end = start + n_fft
        audio = audio.at[:, start:end].add(time_frames[:, i, :])
        window_sum = window_sum.at[start:end].add(window**2)

    # Normalize by window sum
    window_sum = mx.maximum(window_sum, 1e-8)
    audio = audio / window_sum

    # Remove padding (center=True)
    pad = n_fft // 2
    audio = audio[:, pad : pad + output_length]

    return audio


def main():
    np.random.seed(42)

    n_fft = 20
    hop_length = 5
    frames = 100
    n_bins = n_fft // 2 + 1
    batch = 1

    # Create test input
    mag_np = np.random.rand(batch, n_bins, frames).astype(np.float32) * 0.5
    phase_np = (
        (np.random.rand(batch, n_bins, frames).astype(np.float32) - 0.5) * 2 * np.pi
    )

    print(f"Input mag: shape={mag_np.shape}, mean={mag_np.mean():.6f}")
    print(f"Input phase: shape={phase_np.shape}, mean={phase_np.mean():.6f}")

    # PyTorch
    pt_mag = torch.from_numpy(mag_np)
    pt_phase = torch.from_numpy(phase_np)
    pt_audio = pytorch_istft(pt_mag, pt_phase, n_fft, hop_length)

    print("\nPyTorch ISTFT:")
    print(f"  Shape: {tuple(pt_audio.shape)}")
    print(f"  RMS: {(pt_audio**2).mean().sqrt().item():.6f}")
    print(f"  Range: [{pt_audio.min().item():.4f}, {pt_audio.max().item():.4f}]")

    # MLX
    mlx_mag = mx.array(mag_np)
    mlx_phase = mx.array(phase_np)
    mlx_audio = mlx_istft(mlx_mag, mlx_phase, n_fft, hop_length)
    mx.eval(mlx_audio)

    mlx_np = np.array(mlx_audio)
    print("\nMLX ISTFT:")
    print(f"  Shape: {mlx_np.shape}")
    print(f"  RMS: {np.sqrt((mlx_np**2).mean()):.6f}")
    print(f"  Range: [{mlx_np.min():.4f}, {mlx_np.max():.4f}]")

    # Compare
    pt_np = pt_audio.numpy()
    min_len = min(pt_np.shape[-1], mlx_np.shape[-1])

    pt_trimmed = pt_np[..., :min_len]
    mlx_trimmed = mlx_np[..., :min_len]

    diff = np.abs(pt_trimmed - mlx_trimmed)
    print(f"\nComparison (first {min_len} samples):")
    print(f"  Max diff: {diff.max():.6f}")
    print(f"  Mean diff: {diff.mean():.6f}")
    print(f"  PT RMS (trimmed): {np.sqrt((pt_trimmed**2).mean()):.6f}")
    print(f"  MLX RMS (trimmed): {np.sqrt((mlx_trimmed**2).mean()):.6f}")

    # Check what torch.istft actually returns for shape
    print("\n=== Shape Analysis ===")
    print(f"frames={frames}, hop={hop_length}, n_fft={n_fft}")
    print(f"PyTorch output length: {pt_np.shape[-1]}")
    print(f"  Expected (frames-1)*hop + n_fft = {(frames - 1) * hop_length + n_fft}")
    print(f"  Expected frames*hop = {frames * hop_length}")
    print(f"MLX output length: {mlx_np.shape[-1]}")

    # Check PyTorch COLA (constant overlap-add) constraint
    print("\n=== Window Analysis ===")
    window = torch.hann_window(n_fft)
    # Check if COLA is satisfied
    cola_check = torch.zeros(n_fft + hop_length * (frames - 1))
    for i in range(frames):
        start = i * hop_length
        cola_check[start : start + n_fft] += window**2
    print(
        f"COLA check - min: {cola_check.min().item():.6f}, max: {cola_check.max().item():.6f}"
    )
    # For proper reconstruction, COLA should be constant


if __name__ == "__main__":
    main()
