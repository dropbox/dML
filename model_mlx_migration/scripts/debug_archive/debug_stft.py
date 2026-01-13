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
Debug Source STFT mismatch between PyTorch and MLX.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import math

import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as F


def pytorch_stft_detailed(x, n_fft=20, hop=5):
    """PyTorch STFT with detailed output."""
    print(f"\nPyTorch STFT input: shape={x.shape}")

    window = torch.hann_window(n_fft, periodic=True, dtype=torch.float32)

    # torch.stft with center=True:
    # - Pads input by n_fft//2 on each side with reflect mode
    # - Returns complex spectrum

    # Let's manually do what torch.stft does
    pad_amount = n_fft // 2
    x_padded = F.pad(x, (pad_amount, pad_amount), mode="reflect")
    print(f"PyTorch padded: shape={x_padded.shape}")

    # STFT
    spec = torch.stft(
        x_padded,
        n_fft,
        hop,
        n_fft,
        window=window,
        return_complex=True,
        center=False,  # We already padded
    )
    print(f"PyTorch STFT output: shape={spec.shape}")

    return spec


def mlx_stft_detailed(x, n_fft=20, hop=5):
    """MLX STFT with detailed output."""
    print(f"\nMLX STFT input: shape={x.shape}")
    batch, samples = x.shape

    # Manual reflect padding to match torch.stft center=True
    pad_amount = n_fft // 2  # 10

    # PyTorch reflect: for input [a,b,c,d] with pad(2,2):
    # Result: [c,b,a,b,c,d,c,b]
    # The reflection doesn't repeat the edge value

    # Current MLX implementation:
    # left_pad = x[:, 1:pad_amount + 1][:, ::-1]  # indices [1, 2, ..., pad_amount] reversed
    # right_pad = x[:, -(pad_amount + 1):-1][:, ::-1]  # indices [-(pad+1), -pad, ..., -2] reversed

    # For x = [0,1,2,3,4,5] with pad_amount=2:
    # left_pad = x[:, 1:3][:, ::-1] = [2,1] reversed = [2,1]
    # Wait, [::-1] reverses - so [1,2][::-1] = [2,1]
    # Expected: [1,0] for PyTorch reflect

    # Let me check PyTorch reflect behavior:
    # F.pad([0,1,2,3,4,5], (2,2), mode='reflect') should give:
    # left: reflection of [0,1,2,3,4,5] from index 1 going left = [2,1]
    # right: reflection of [0,1,2,3,4,5] from index 4 going right = [4,3]
    # Result: [2,1,0,1,2,3,4,5,4,3]

    # Hmm, my understanding might be off. Let me verify with PyTorch.
    test_x = torch.arange(6, dtype=torch.float32).unsqueeze(0)
    test_padded = F.pad(test_x, (2, 2), mode="reflect")
    print(
        f"PyTorch reflect test: {test_x.squeeze().tolist()} -> {test_padded.squeeze().tolist()}"
    )

    # Now implement correctly for MLX
    # For reflect padding:
    # Left pad: reflect indices [pad_amount, pad_amount-1, ..., 1]
    # Right pad: reflect indices [-2, -3, ..., -(pad_amount+1)]

    if pad_amount > 0:
        # PyTorch reflect pad: x[pad-1], x[pad-2], ..., x[0], x, x[-1], x[-2], ..., x[-pad]
        # Actually from the test: [0,1,2,3,4,5] pad(2,2) -> [2,1,0,1,2,3,4,5,4,3]
        # So left pad is: x[2], x[1] = [2,1]
        # Right pad is: x[-2], x[-3] = [4,3]

        # Current implementation:
        # left_pad = x[:, 1:pad_amount + 1][:, ::-1]
        # For [0,1,2,3,4,5] with pad=2: x[:,1:3][:,::-1] = [1,2][::-1] = [2,1] ✓

        # right_pad = x[:, -(pad_amount + 1):-1][:, ::-1]
        # For [0,1,2,3,4,5] with pad=2: x[:,-(2+1):-1][:,::-1] = x[:,-3:-1][::-1] = [3,4][::-1] = [4,3] ✓

        left_pad = x[:, 1 : pad_amount + 1][:, ::-1]
        right_pad = x[:, -(pad_amount + 1) : -1][:, ::-1]
        x_padded = mx.concatenate([left_pad, x, right_pad], axis=1)
    else:
        x_padded = x

    print(f"MLX padded: shape={x_padded.shape}")

    # Test reflect padding correctness
    test_mx = mx.arange(6, dtype=mx.float32)[None, :]
    if pad_amount > 0:
        test_left = test_mx[:, 1 : pad_amount + 1][:, ::-1]
        test_right = test_mx[:, -(pad_amount + 1) : -1][:, ::-1]
        test_padded_mx = mx.concatenate([test_left, test_mx, test_right], axis=1)
        mx.eval(test_padded_mx)
        print(
            f"MLX reflect test: {list(np.array(test_mx).squeeze())} -> {list(np.array(test_padded_mx).squeeze())}"
        )

    # Frame the signal
    padded_len = x_padded.shape[1]
    num_frames = (padded_len - n_fft) // hop + 1
    print(f"MLX num_frames: {num_frames}, padded_len={padded_len}")

    # PyTorch num_frames formula:
    # num_frames = (input_len + 2*pad - n_fft) / hop + 1
    # For input=100, pad=10, n_fft=20, hop=5:
    # num_frames = (100 + 20 - 20) / 5 + 1 = 21
    pt_num_frames = (samples + 2 * pad_amount - n_fft) // hop + 1
    print(f"Expected PyTorch num_frames: {pt_num_frames}")

    indices = mx.arange(n_fft)[None, :] + mx.arange(num_frames)[:, None] * hop
    frames = x_padded[:, indices.flatten()].reshape(batch, num_frames, n_fft)

    # Hann window
    n = mx.arange(n_fft, dtype=mx.float32)
    # PyTorch hann_window(n, periodic=True) = 0.5 * (1 - cos(2*pi*i / n))
    window = 0.5 * (1 - mx.cos(2 * math.pi * n / n_fft))
    frames = frames * window

    # FFT
    spectrum = mx.fft.rfft(frames, axis=-1)
    print(f"MLX spectrum: shape={spectrum.shape}")

    return spectrum


def compare_stft():
    """Compare STFT implementations."""
    print("=" * 60)
    print("Source STFT Debug")
    print("=" * 60)

    # Simple test signal
    np.random.seed(42)
    samples = 100
    x_np = np.sin(np.linspace(0, 10 * np.pi, samples)).astype(np.float32)[None, :]

    n_fft = 20
    hop = 5

    # PyTorch
    x_pt = torch.from_numpy(x_np)
    spec_pt = pytorch_stft_detailed(x_pt, n_fft, hop)

    # MLX
    x_mx = mx.array(x_np)
    spec_mx = mlx_stft_detailed(x_mx, n_fft, hop)
    mx.eval(spec_mx)

    print("\n" + "=" * 60)
    print("Comparison")
    print("=" * 60)

    # Convert to comparable format
    # PyTorch: [batch, n_fft//2+1, frames]
    # MLX: [batch, frames, n_fft//2+1]
    spec_pt_np = spec_pt.numpy()  # [1, 11, frames]
    spec_mx_np = np.array(spec_mx)  # [1, frames, 11]

    print(f"\nPyTorch spectrum shape: {spec_pt_np.shape}")
    print(f"MLX spectrum shape: {spec_mx_np.shape}")

    # Transpose MLX to match
    spec_mx_ncl = spec_mx_np.transpose(0, 2, 1)

    # Match frame count
    min_frames = min(spec_pt_np.shape[2], spec_mx_ncl.shape[2])
    spec_pt_trim = spec_pt_np[:, :, :min_frames]
    spec_mx_trim = spec_mx_ncl[:, :, :min_frames]

    # Compare real and imag parts
    diff_real = np.abs(spec_pt_trim.real - spec_mx_trim.real).flatten()
    diff_imag = np.abs(spec_pt_trim.imag - spec_mx_trim.imag).flatten()

    print(
        f"\nReal part: max_diff={diff_real.max():.6f}, mean_diff={diff_real.mean():.6f}"
    )
    print(
        f"Imag part: max_diff={diff_imag.max():.6f}, mean_diff={diff_imag.mean():.6f}"
    )

    # Correlation
    pt_flat = np.abs(spec_pt_trim).flatten()
    mx_flat = np.abs(spec_mx_trim).flatten()

    if np.std(pt_flat) > 0 and np.std(mx_flat) > 0:
        corr = np.corrcoef(pt_flat, mx_flat)[0, 1]
        print(f"Magnitude correlation: {corr:.6f}")


if __name__ == "__main__":
    compare_stft()
