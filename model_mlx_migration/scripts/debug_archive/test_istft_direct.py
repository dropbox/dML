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
Test ISTFT implementation directly with controlled inputs.
"""

import sys

sys.path.insert(0, "/Users/ayates/model_mlx_migration")

import mlx.core as mx
import numpy as np

from tools.pytorch_to_mlx.converters.kokoro_converter import KokoroConverter


def main():
    print("Loading MLX model...")
    converter = KokoroConverter()
    mlx_model, config, _ = converter.load_from_hf("hexgrad/Kokoro-82M")

    generator = mlx_model.decoder.generator

    # Test ISTFT with controlled inputs
    print("\n=== Testing ISTFT with controlled inputs ===")

    # Parameters
    n_fft = generator.istft_n_fft  # 20
    hop = generator.istft_hop_size  # 5
    n_bins = n_fft // 2 + 1  # 11

    frames = 100
    batch = 1

    # Test 1: Unit magnitude, zero phase (should produce impulses)
    print("\nTest 1: Unit magnitude, zero phase")
    mag = mx.ones((batch, frames, n_bins))
    phase = mx.zeros((batch, frames, n_bins))
    audio = generator._istft_synthesis(mag, phase)
    mx.eval(audio)
    print(f"  Audio: shape={audio.shape}, RMS={float(mx.sqrt(mx.mean(audio**2))):.6f}")
    print(f"  Range: [{float(audio.min()):.4f}, {float(audio.max()):.4f}]")

    # Test 2: Varying magnitude (more realistic)
    print("\nTest 2: Varying magnitude (sine wave envelope)")
    t = mx.arange(frames, dtype=mx.float32) / frames
    envelope = 0.5 + 0.5 * mx.sin(2 * np.pi * 2 * t)  # 2 cycles
    mag = mx.ones((batch, frames, n_bins)) * envelope[:, None]
    phase = mx.zeros((batch, frames, n_bins))
    audio = generator._istft_synthesis(mag, phase)
    mx.eval(audio)
    print(f"  Audio: shape={audio.shape}, RMS={float(mx.sqrt(mx.mean(audio**2))):.6f}")
    print(f"  Range: [{float(audio.min()):.4f}, {float(audio.max()):.4f}]")

    # Test 3: Small magnitude (like what we see in practice)
    print("\nTest 3: Small magnitude (0.001)")
    mag = mx.ones((batch, frames, n_bins)) * 0.001
    phase = mx.zeros((batch, frames, n_bins))
    audio = generator._istft_synthesis(mag, phase)
    mx.eval(audio)
    print(f"  Audio: shape={audio.shape}, RMS={float(mx.sqrt(mx.mean(audio**2))):.6f}")
    print(f"  Range: [{float(audio.min()):.4f}, {float(audio.max()):.4f}]")

    # Test 4: Very small magnitude (what we actually get)
    print("\nTest 4: Very small magnitude (0.00005, like exp(-10))")
    mag = mx.ones((batch, frames, n_bins)) * 0.00005
    phase = mx.zeros((batch, frames, n_bins))
    audio = generator._istft_synthesis(mag, phase)
    mx.eval(audio)
    print(f"  Audio: shape={audio.shape}, RMS={float(mx.sqrt(mx.mean(audio**2))):.6f}")
    print(f"  Range: [{float(audio.min()):.4f}, {float(audio.max()):.4f}]")

    # Test 5: Compare with PyTorch ISTFT
    print("\n=== Comparing with PyTorch ISTFT ===")
    import torch

    # Create test spectrogram
    np.random.seed(42)
    mag_np = np.random.uniform(0.1, 1.0, (batch, frames, n_bins)).astype(np.float32)
    phase_np = np.random.uniform(-np.pi, np.pi, (batch, frames, n_bins)).astype(
        np.float32
    )

    # PyTorch ISTFT
    spec_pt = torch.from_numpy(mag_np) * torch.exp(1j * torch.from_numpy(phase_np))
    spec_pt = spec_pt.transpose(1, 2)  # [batch, n_bins, frames] for PyTorch

    window = torch.hann_window(n_fft)
    audio_pt = torch.istft(
        spec_pt,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        center=True,
        onesided=True,
    )

    print("PyTorch ISTFT:")
    print(
        f"  Audio: shape={audio_pt.shape}, RMS={audio_pt.pow(2).mean().sqrt().item():.6f}"
    )
    print(f"  Range: [{audio_pt.min().item():.4f}, {audio_pt.max().item():.4f}]")

    # MLX ISTFT
    mag_mlx = mx.array(mag_np)
    phase_mlx = mx.array(phase_np)
    audio_mlx = generator._istft_synthesis(mag_mlx, phase_mlx)
    mx.eval(audio_mlx)

    print("MLX ISTFT:")
    print(
        f"  Audio: shape={audio_mlx.shape}, RMS={float(mx.sqrt(mx.mean(audio_mlx**2))):.6f}"
    )
    print(f"  Range: [{float(audio_mlx.min()):.4f}, {float(audio_mlx.max()):.4f}]")

    # Check length
    print("\nLength comparison:")
    print(f"  PyTorch: {audio_pt.shape[-1]}")
    print(f"  MLX: {audio_mlx.shape[-1]}")


if __name__ == "__main__":
    main()
