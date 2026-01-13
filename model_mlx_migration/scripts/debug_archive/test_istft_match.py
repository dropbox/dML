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
Test ISTFT implementation matches PyTorch.
"""

import math
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


def mlx_istft(mag, phase, n_fft=20, hop=5):
    """MLX ISTFT implementation."""
    batch, frames, n_bins = mag.shape

    # Complex spectrum
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


def main():
    # Create test input
    np.random.seed(42)
    batch = 1
    frames = 100
    n_fft = 20
    n_bins = n_fft // 2 + 1  # 11
    hop = 5

    # Create random magnitude and phase
    log_mag_np = np.random.randn(batch, frames, n_bins).astype(np.float32) * 2
    phase_logits_np = np.random.randn(batch, frames, n_bins).astype(np.float32) * 3

    # Clip log_mag
    log_mag_np = np.clip(log_mag_np, -10, 10)
    mag_np = np.exp(log_mag_np)

    # Apply sin to phase
    phase_np = np.sin(phase_logits_np)

    print(f"mag shape: {mag_np.shape}")
    print(f"mag range: [{mag_np.min():.4f}, {mag_np.max():.4f}]")
    print(f"phase range: [{phase_np.min():.4f}, {phase_np.max():.4f}]")

    # PyTorch ISTFT
    print("\n=== PyTorch ISTFT ===")
    mag_pt = torch.tensor(mag_np)
    phase_pt = torch.tensor(phase_np)

    # Transpose to [batch, n_bins, frames] for torch.istft
    mag_pt_ncl = mag_pt.transpose(1, 2)
    phase_pt_ncl = phase_pt.transpose(1, 2)

    # Construct complex spectrum
    spectrum_pt = mag_pt_ncl * torch.exp(1j * phase_pt_ncl)

    # PyTorch ISTFT
    window = torch.hann_window(n_fft)
    audio_pt = torch.istft(
        spectrum_pt,
        n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        center=True,
        return_complex=False,
    )

    print(f"PT audio shape: {audio_pt.shape}")
    print(f"PT audio range: [{audio_pt.min():.4f}, {audio_pt.max():.4f}]")
    print(f"PT audio std: {audio_pt.std():.4f}")

    # MLX ISTFT
    print("\n=== MLX ISTFT ===")
    mag_mx = mx.array(mag_np)
    phase_mx = mx.array(phase_np)

    audio_mx = mlx_istft(mag_mx, phase_mx, n_fft=n_fft, hop=hop)
    mx.eval(audio_mx)

    print(f"MLX audio shape: {audio_mx.shape}")
    print(
        f"MLX audio range: [{float(mx.min(audio_mx)):.4f}, {float(mx.max(audio_mx)):.4f}]"
    )
    print(f"MLX audio std: {float(mx.std(audio_mx)):.4f}")

    # Compare
    audio_mx_np = np.array(audio_mx)
    audio_pt_np = audio_pt.numpy()

    # Match lengths
    min_len = min(audio_mx_np.shape[1], audio_pt_np.shape[1])
    audio_mx_flat = audio_mx_np[0, :min_len]
    audio_pt_flat = audio_pt_np[0, :min_len]

    print(f"\n=== Comparison (first {min_len} samples) ===")
    diff = np.abs(audio_mx_flat - audio_pt_flat)
    print(f"Max diff: {diff.max():.6f}")
    print(f"Mean diff: {diff.mean():.6f}")
    corr = np.corrcoef(audio_mx_flat, audio_pt_flat)[0, 1]
    print(f"Correlation: {corr:.6f}")

    # Check magnitude ratio
    ratio = np.std(audio_mx_flat) / np.std(audio_pt_flat)
    print(f"Std ratio (MLX/PT): {ratio:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
