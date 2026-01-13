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
Compare MLX and PyTorch ISTFT implementations directly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import math

import mlx.core as mx
import numpy as np
import torch
import torch.nn as nn


class PyTorchSTFT(nn.Module):
    def __init__(self, filter_length=20, hop_length=5, win_length=20):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = torch.hann_window(win_length, periodic=True, dtype=torch.float32)

    def inverse(self, magnitude, phase):
        inverse_transform = torch.istft(
            magnitude * torch.exp(phase * 1j),
            self.filter_length,
            self.hop_length,
            self.win_length,
            window=self.window.to(magnitude.device),
        )
        return inverse_transform


def mlx_istft(
    mag: mx.array, phase: mx.array, n_fft: int = 20, hop: int = 5
) -> mx.array:
    """MLX ISTFT matching the Generator's implementation."""
    batch, frames, n_bins = mag.shape

    # Construct complex spectrum
    spectrum = mag * mx.exp(1j * phase)

    # IRFFT per frame
    time_frames = mx.fft.irfft(spectrum, n=n_fft, axis=-1)

    # Hann window
    n = mx.arange(n_fft, dtype=mx.float32)
    window = 0.5 * (1 - mx.cos(2 * math.pi * n / n_fft))

    # Apply window
    time_frames = time_frames * window

    # Overlap-add
    output_length = (frames - 1) * hop
    audio = mx.zeros((batch, output_length + n_fft))
    window_sum = mx.zeros((output_length + n_fft,))

    for i in range(frames):
        start = i * hop
        end = start + n_fft
        audio = audio.at[:, start:end].add(time_frames[:, i, :])
        window_sum = window_sum.at[start:end].add(window**2)

    # Normalize
    window_sum = mx.maximum(window_sum, 1e-8)
    audio = audio / window_sum

    # Remove padding
    pad = n_fft // 2
    audio = audio[:, pad : pad + output_length]

    return audio


def compare():
    print("=" * 60)
    print("ISTFT Comparison: PyTorch vs MLX")
    print("=" * 60)

    np.random.seed(42)
    torch.manual_seed(42)

    n_fft = 20
    hop = 5
    n_bins = n_fft // 2 + 1  # 11
    frames = 100

    # Create test magnitude and phase
    log_mag_np = np.random.randn(1, frames, n_bins).astype(np.float32) * 2  # ~N(0, 2)
    phase_np = np.random.randn(1, frames, n_bins).astype(np.float32)

    # Clip log_mag like the generator does
    log_mag_np = np.clip(log_mag_np, -10, 10)
    mag_np = np.exp(log_mag_np)
    phase_rad_np = np.sin(phase_np)  # Generator uses sin(phase_logits)

    print("\nTest input:")
    print(
        f"  log_mag: mean={log_mag_np.mean():.4f}, range=[{log_mag_np.min():.4f}, {log_mag_np.max():.4f}]"
    )
    print(
        f"  mag (exp): mean={mag_np.mean():.4f}, range=[{mag_np.min():.4f}, {mag_np.max():.4f}]"
    )
    print(
        f"  phase (sin): mean={phase_rad_np.mean():.4f}, range=[{phase_rad_np.min():.4f}, {phase_rad_np.max():.4f}]"
    )

    # PyTorch ISTFT
    # Note: PyTorch expects [batch, freq, time], MLX uses [batch, time, freq]
    mag_pt = torch.from_numpy(mag_np.transpose(0, 2, 1))  # [1, 11, 100]
    phase_pt = torch.from_numpy(phase_rad_np.transpose(0, 2, 1))  # [1, 11, 100]

    stft_pt = PyTorchSTFT(n_fft, hop, n_fft)
    audio_pt = stft_pt.inverse(mag_pt, phase_pt)
    audio_pt_np = audio_pt.numpy()

    print("\nPyTorch ISTFT:")
    print(f"  shape: {audio_pt.shape}")
    print(f"  RMS: {np.sqrt(np.mean(audio_pt_np**2)):.6f}")
    print(f"  range: [{audio_pt_np.min():.4f}, {audio_pt_np.max():.4f}]")

    # MLX ISTFT
    mag_mx = mx.array(mag_np)
    phase_mx = mx.array(phase_rad_np)

    audio_mx = mlx_istft(mag_mx, phase_mx, n_fft, hop)
    mx.eval(audio_mx)
    audio_mx_np = np.array(audio_mx)

    print("\nMLX ISTFT:")
    print(f"  shape: {audio_mx.shape}")
    print(f"  RMS: {np.sqrt(np.mean(audio_mx_np**2)):.6f}")
    print(f"  range: [{audio_mx_np.min():.4f}, {audio_mx_np.max():.4f}]")

    # Compare
    # Trim to same length
    min_len = min(audio_pt_np.shape[-1], audio_mx_np.shape[-1])
    audio_pt_trim = audio_pt_np[..., :min_len].flatten()
    audio_mx_trim = audio_mx_np[..., :min_len].flatten()

    diff = audio_pt_trim - audio_mx_trim
    max_diff = np.max(np.abs(diff))
    mean_diff = np.mean(np.abs(diff))

    if np.std(audio_pt_trim) > 0 and np.std(audio_mx_trim) > 0:
        corr = np.corrcoef(audio_pt_trim, audio_mx_trim)[0, 1]
    else:
        corr = 0.0

    print("\nComparison:")
    print(f"  Max diff: {max_diff:.6f}")
    print(f"  Mean diff: {mean_diff:.6f}")
    print(f"  Correlation: {corr:.6f}")

    # Now test with Generator's typical output values
    print("\n" + "=" * 60)
    print("Test with Generator-like values")
    print("=" * 60)

    # From debug: conv_post range [-44.98, 14.45] but clipped to [-10, 10]
    # Mean is probably around -5 to 5
    log_mag_gen = np.random.randn(1, frames, n_bins).astype(np.float32) * 3 + 2
    log_mag_gen = np.clip(log_mag_gen, -10, 10)
    mag_gen = np.exp(log_mag_gen)
    phase_gen = np.sin(np.random.randn(1, frames, n_bins).astype(np.float32))

    print("\nGenerator-like input:")
    print(f"  log_mag: mean={log_mag_gen.mean():.4f}")
    print(f"  mag: mean={mag_gen.mean():.4f}")

    # PyTorch
    mag_pt = torch.from_numpy(mag_gen.transpose(0, 2, 1))
    phase_pt = torch.from_numpy(phase_gen.transpose(0, 2, 1))
    audio_pt = stft_pt.inverse(mag_pt, phase_pt).numpy()

    # MLX
    audio_mx = mlx_istft(mx.array(mag_gen), mx.array(phase_gen), n_fft, hop)
    mx.eval(audio_mx)
    audio_mx_np = np.array(audio_mx)

    print(
        f"\nPyTorch: RMS={np.sqrt(np.mean(audio_pt**2)):.4f}, range=[{audio_pt.min():.4f}, {audio_pt.max():.4f}]"
    )
    print(
        f"MLX: RMS={np.sqrt(np.mean(audio_mx_np**2)):.4f}, range=[{audio_mx_np.min():.4f}, {audio_mx_np.max():.4f}]"
    )


if __name__ == "__main__":
    compare()
