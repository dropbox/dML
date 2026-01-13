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
Full Generator comparison: MLX vs PyTorch

Loads the same weights, runs the same random input through both,
and compares outputs at every stage.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import math

import mlx.core as mx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# PyTorch SourceModule (minimal)
class PyTorchSourceModule(nn.Module):
    def __init__(self, sampling_rate=24000, harmonic_num=8, sine_amp=0.1):
        super().__init__()
        self.sample_rate = sampling_rate
        self.harmonic_num = harmonic_num
        self.sine_amp = sine_amp
        self.l_linear = nn.Linear(harmonic_num + 1, 1)

    def forward(self, f0, upp):
        batch, length = f0.shape

        # Upsample F0
        f0_up = F.interpolate(
            f0[:, None, :], scale_factor=float(upp), mode="nearest"
        ).squeeze(1)

        # Generate harmonics
        samples = f0_up.shape[1]
        harmonics = []
        for h in range(1, self.harmonic_num + 2):
            phase = torch.cumsum(f0_up * h / self.sample_rate, dim=1) * 2 * math.pi
            sine = torch.sin(phase) * self.sine_amp
            harmonics.append(sine)

        harmonics_stack = torch.stack(harmonics, dim=-1)

        # UV mask
        uv = (f0_up > 0).float().unsqueeze(-1)
        harmonics_stack = harmonics_stack * uv

        # Linear + tanh
        har_source = torch.tanh(self.l_linear(harmonics_stack))

        # Noise
        noise = torch.randn(batch, samples, 1) * self.sine_amp / 3

        return har_source, noise, uv


def pytorch_source_stft(x, n_fft=20, hop=5):
    """STFT for source signal (PyTorch)."""
    batch, samples = x.shape

    # Reflect padding
    pad_amount = n_fft // 2
    x_padded = F.pad(x, (pad_amount, pad_amount), mode="reflect")

    # STFT
    window = torch.hann_window(n_fft, periodic=True, dtype=torch.float32)
    spec = torch.stft(x_padded, n_fft, hop, n_fft, window=window, return_complex=True)
    # spec: [batch, n_fft//2+1, frames]

    magnitude = torch.abs(spec)
    phase = torch.angle(spec)

    # Concatenate: [batch, 22, frames]
    har = torch.cat([magnitude, phase], dim=1)

    return har


def mlx_source_stft(x, n_fft=20, hop=5):
    """STFT for source signal (MLX)."""
    batch, samples = x.shape

    # Reflect padding
    pad_amount = n_fft // 2
    if pad_amount > 0:
        left_pad = x[:, 1 : pad_amount + 1][:, ::-1]
        right_pad = x[:, -(pad_amount + 1) : -1][:, ::-1]
        x_padded = mx.concatenate([left_pad, x, right_pad], axis=1)
    else:
        x_padded = x

    # Frame
    padded_len = x_padded.shape[1]
    num_frames = (padded_len - n_fft) // hop + 1

    indices = mx.arange(n_fft)[None, :] + mx.arange(num_frames)[:, None] * hop
    frames = x_padded[:, indices.flatten()].reshape(batch, num_frames, n_fft)

    # Window
    n = mx.arange(n_fft, dtype=mx.float32)
    window = 0.5 * (1 - mx.cos(2 * math.pi * n / n_fft))
    frames = frames * window

    # FFT
    spectrum = mx.fft.rfft(frames, axis=-1)

    magnitude = mx.abs(spectrum)
    phase = mx.arctan2(spectrum.imag, spectrum.real)

    # Concatenate: [batch, frames, 22] (NLC format)
    har = mx.concatenate([magnitude, phase], axis=-1)

    return har


def compare_source_module():
    """Compare SourceModule outputs."""
    print("=" * 60)
    print("SourceModule Comparison")
    print("=" * 60)

    # Load PyTorch weights
    ckpt = torch.load(
        "/Users/ayates/models/kokoro/kokoro-v1_0.pth",
        map_location="cpu",
        weights_only=True,
    )
    pt_gen = ckpt["decoder"]

    # Create PyTorch source module
    pt_source = PyTorchSourceModule(24000, 8, 0.1)
    pt_source.l_linear.weight.data = pt_gen["module.generator.m_source.l_linear.weight"]
    pt_source.l_linear.bias.data = pt_gen["module.generator.m_source.l_linear.bias"]
    pt_source.eval()

    # Fixed F0 input
    np.random.seed(42)
    f0_np = np.abs(np.random.randn(1, 10).astype(np.float32)) * 100 + 200

    upp = 300

    # PyTorch
    f0_pt = torch.from_numpy(f0_np)
    with torch.no_grad():
        har_pt, noise_pt, uv_pt = pt_source(f0_pt, upp)
    har_pt_np = har_pt.numpy()

    print("\nPyTorch harmonic source:")
    print(f"  shape: {har_pt.shape}")
    print(f"  range: [{har_pt.min():.4f}, {har_pt.max():.4f}]")

    # MLX
    from tools.pytorch_to_mlx.converters import KokoroConverter

    converter = KokoroConverter()
    model, config, _ = converter.load_from_hf()

    f0_mx = mx.array(f0_np)
    har_mx, noise_mx, uv_mx = model.decoder.generator.m_source(f0_mx, upp)
    mx.eval(har_mx)
    har_mx_np = np.array(har_mx)

    print("\nMLX harmonic source:")
    print(f"  shape: {har_mx.shape}")
    print(f"  range: [{har_mx.min():.4f}, {har_mx.max():.4f}]")

    # Compare
    # PyTorch: [batch, samples, 1], MLX: [batch, samples, 1] (both should be same)
    min_len = min(har_pt_np.shape[1], har_mx_np.shape[1])
    har_pt_flat = har_pt_np[:, :min_len, :].flatten()
    har_mx_flat = har_mx_np[:, :min_len, :].flatten()

    diff = np.abs(har_pt_flat - har_mx_flat)
    print("\nComparison:")
    print(f"  Max diff: {diff.max():.6f}")
    print(f"  Mean diff: {diff.mean():.6f}")

    if np.std(har_pt_flat) > 0 and np.std(har_mx_flat) > 0:
        corr = np.corrcoef(har_pt_flat, har_mx_flat)[0, 1]
        print(f"  Correlation: {corr:.6f}")

    # Now compare STFT
    print("\n" + "=" * 60)
    print("Source STFT Comparison")
    print("=" * 60)

    # Use same harmonic source for STFT
    har_1d_pt = har_pt.squeeze(-1)  # [1, samples]
    har_1d_mx = har_mx.squeeze(-1)  # [1, samples]

    # PyTorch STFT
    stft_pt = pytorch_source_stft(har_1d_pt)
    stft_pt_np = stft_pt.numpy()

    print("\nPyTorch STFT:")
    print(f"  shape: {stft_pt.shape} (NCL format)")
    print(f"  range: [{stft_pt.min():.4f}, {stft_pt.max():.4f}]")

    # MLX STFT
    stft_mx = mlx_source_stft(har_1d_mx)
    mx.eval(stft_mx)
    stft_mx_np = np.array(stft_mx)

    print("\nMLX STFT:")
    print(f"  shape: {stft_mx.shape} (NLC format)")
    print(f"  range: [{stft_mx.min():.4f}, {stft_mx.max():.4f}]")

    # Transpose MLX to match PyTorch for comparison
    stft_mx_ncl = stft_mx_np.transpose(0, 2, 1)  # [batch, 22, frames]

    min_frames = min(stft_pt_np.shape[2], stft_mx_ncl.shape[2])
    stft_pt_trim = stft_pt_np[:, :, :min_frames].flatten()
    stft_mx_trim = stft_mx_ncl[:, :, :min_frames].flatten()

    diff = np.abs(stft_pt_trim - stft_mx_trim)
    print("\nComparison:")
    print(f"  Max diff: {diff.max():.6f}")
    print(f"  Mean diff: {diff.mean():.6f}")

    if np.std(stft_pt_trim) > 0 and np.std(stft_mx_trim) > 0:
        corr = np.corrcoef(stft_pt_trim, stft_mx_trim)[0, 1]
        print(f"  Correlation: {corr:.6f}")


if __name__ == "__main__":
    compare_source_module()
