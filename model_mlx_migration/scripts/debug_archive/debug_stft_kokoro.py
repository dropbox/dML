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
Debug Source STFT for actual Kokoro sizes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import math

import mlx.core as mx
import numpy as np
import torch


def compare_stft_kokoro_size():
    """Compare STFT for Kokoro-like input sizes."""
    print("=" * 60)
    print("Source STFT - Kokoro Size Debug")
    print("=" * 60)

    # Kokoro uses:
    # - n_fft = 20
    # - hop = 5
    # - Sample rate = 24000
    # - Total upsample = 300
    # For 10 F0 frames: 10 * 300 = 3000 samples

    n_fft = 20
    hop = 5
    samples = 3000

    print(f"Input samples: {samples}")
    print(f"n_fft: {n_fft}, hop: {hop}")

    # Generate test signal
    np.random.seed(42)
    x_np = np.sin(np.linspace(0, 100 * np.pi, samples)).astype(np.float32)[None, :]

    # PyTorch STFT
    x_pt = torch.from_numpy(x_np)
    window = torch.hann_window(n_fft, periodic=True)

    # PyTorch default: center=True, pad_mode='reflect'
    spec_pt = torch.stft(
        x_pt,
        n_fft,
        hop,
        n_fft,
        window=window,
        return_complex=True,
        center=True,
        pad_mode="reflect",
    )
    print(f"\nPyTorch STFT output: shape={spec_pt.shape}")  # [batch, freq, frames]

    # Expected PyTorch frames with center=True:
    # frames = ceil((samples + 2*(n_fft//2)) / hop) - ceil is for torch.stft
    # Actually torch.stft formula: (input_len + 2*pad - n_fft) // hop + 1 when center=True
    pad_amount = n_fft // 2
    expected_frames = (samples + 2 * pad_amount - n_fft) // hop + 1
    print(f"Expected frames: {expected_frames}")

    # MLX STFT (from kokoro.py)
    x_mx = mx.array(x_np)

    # Reflect padding
    if pad_amount > 0:
        left_pad = x_mx[:, 1 : pad_amount + 1][:, ::-1]
        right_pad = x_mx[:, -(pad_amount + 1) : -1][:, ::-1]
        x_padded = mx.concatenate([left_pad, x_mx, right_pad], axis=1)
    else:
        x_padded = x_mx

    mx.eval(x_padded)
    print(f"MLX padded: shape={x_padded.shape}")

    # MLX frame calculation
    padded_len = x_padded.shape[1]
    num_frames = (padded_len - n_fft) // hop + 1
    print(f"MLX num_frames: {num_frames}")

    # Do MLX FFT
    indices = mx.arange(n_fft)[None, :] + mx.arange(num_frames)[:, None] * hop
    frames = x_padded[:, indices.flatten()].reshape(1, num_frames, n_fft)

    n = mx.arange(n_fft, dtype=mx.float32)
    window_mx = 0.5 * (1 - mx.cos(2 * math.pi * n / n_fft))
    frames = frames * window_mx

    spectrum_mx = mx.fft.rfft(frames, axis=-1)
    mx.eval(spectrum_mx)
    print(f"MLX STFT output: shape={spectrum_mx.shape}")  # [batch, frames, freq]

    # Compare
    print("\n=== Comparison ===")
    spec_pt_np = spec_pt.numpy()  # [batch, freq, frames]
    spec_mx_np = np.array(spectrum_mx)  # [batch, frames, freq]

    # Transpose MLX to match
    spec_mx_ncl = spec_mx_np.transpose(0, 2, 1)

    print(f"PyTorch shape: {spec_pt_np.shape}")
    print(f"MLX shape (transposed): {spec_mx_ncl.shape}")

    min_frames = min(spec_pt_np.shape[2], spec_mx_ncl.shape[2])
    spec_pt_trim = spec_pt_np[:, :, :min_frames]
    spec_mx_trim = spec_mx_ncl[:, :, :min_frames]

    diff = np.abs(spec_pt_trim - spec_mx_trim).flatten()
    print(f"\nMax diff: {diff.max():.6f}")
    print(f"Mean diff: {diff.mean():.6f}")

    # Correlation on magnitudes
    pt_mag = np.abs(spec_pt_trim).flatten()
    mx_mag = np.abs(spec_mx_trim).flatten()
    corr = np.corrcoef(pt_mag, mx_mag)[0, 1]
    print(f"Magnitude correlation: {corr:.6f}")

    # Check phase
    pt_phase = np.angle(spec_pt_trim).flatten()
    mx_phase = np.angle(spec_mx_trim).flatten()
    phase_corr = np.corrcoef(pt_phase, mx_phase)[0, 1]
    print(f"Phase correlation: {phase_corr:.6f}")

    # Check if frame counts match
    if spec_pt_np.shape[2] != spec_mx_ncl.shape[2]:
        print("\nWARNING: Frame count mismatch!")
        print(f"  PyTorch: {spec_pt_np.shape[2]}")
        print(f"  MLX: {spec_mx_ncl.shape[2]}")

        # Debug padding difference
        print(f"\nPyTorch input len: {samples}")
        print(f"PyTorch with center padding: {samples + 2 * pad_amount}")

        print(f"MLX padded len: {padded_len}")
        print(f"MLX left_pad len: {pad_amount}")
        print(f"MLX right_pad len: {pad_amount}")


if __name__ == "__main__":
    compare_stft_kokoro_size()
