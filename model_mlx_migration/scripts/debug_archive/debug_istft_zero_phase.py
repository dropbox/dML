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

"""Debug ISTFT with zero phase."""

import sys

sys.path.insert(0, "/Users/ayates/model_mlx_migration")

import mlx.core as mx
import numpy as np
import torch


def main():
    n_fft = 20
    hop = 5
    n_bins = n_fft // 2 + 1  # 11
    frames = 10
    batch = 1

    # Unit magnitude, zero phase
    mag = np.ones((batch, frames, n_bins), dtype=np.float32)
    phase = np.zeros((batch, frames, n_bins), dtype=np.float32)

    # PyTorch version
    print("=== PyTorch ===")
    spec_pt = torch.from_numpy(mag) * torch.exp(1j * torch.from_numpy(phase))
    spec_pt = spec_pt.transpose(1, 2)  # [batch, n_bins, frames]
    print(f"Spectrum: shape={spec_pt.shape}, dtype={spec_pt.dtype}")
    print(f"  First frame: {spec_pt[0, :, 0].numpy()}")

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
    print(
        f"Audio: shape={audio_pt.shape}, RMS={audio_pt.pow(2).mean().sqrt().item():.6f}"
    )
    print(f"  First 20 samples: {audio_pt[0, :20].numpy()}")

    # MLX version - debug step by step
    print("\n=== MLX Debug ===")
    mag_mlx = mx.array(mag)
    phase_mlx = mx.array(phase)

    # Create spectrum
    spectrum = mag_mlx * mx.exp(1j * phase_mlx)
    mx.eval(spectrum)
    print(f"Spectrum: shape={spectrum.shape}")
    print(f"  First frame: {np.array(spectrum[0, 0, :])}")

    # IRFFT
    time_frames = mx.fft.irfft(spectrum, n=n_fft, axis=-1)
    mx.eval(time_frames)
    print(f"\nTime frames after irfft: shape={time_frames.shape}")
    print(f"  First frame: {np.array(time_frames[0, 0, :])}")
    print(f"  Frame max: {float(time_frames.max()):.6f}")

    # Hann window
    n = mx.arange(n_fft, dtype=mx.float32)
    window = 0.5 * (1 - mx.cos(2 * np.pi * n / n_fft))
    print(f"\nWindow: {np.array(window)}")

    # Apply window
    windowed = time_frames * window
    mx.eval(windowed)
    print(f"\nWindowed frame 0: {np.array(windowed[0, 0, :])}")
    print(f"  Windowed max: {float(windowed.max()):.6f}")

    # Check PyTorch's irfft behavior
    print("\n=== PyTorch irfft check ===")
    spec_pt_single = torch.ones(n_bins, dtype=torch.complex64)
    time_pt = torch.fft.irfft(spec_pt_single, n=n_fft)
    print(f"PyTorch irfft of all-ones spectrum: {time_pt.numpy()}")

    # MLX irfft
    spec_mlx_single = mx.ones((n_bins,), dtype=mx.complex64)
    time_mlx = mx.fft.irfft(spec_mlx_single, n=n_fft)
    mx.eval(time_mlx)
    print(f"MLX irfft of all-ones spectrum: {np.array(time_mlx)}")


if __name__ == "__main__":
    main()
