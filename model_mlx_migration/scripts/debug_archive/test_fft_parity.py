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
Direct comparison of MLX FFT vs PyTorch FFT to identify phase differences.

This test is designed to prove (or disprove) the "framework limitation" claim
by testing FFT behavior on identical inputs.
"""

import sys

import mlx.core as mx
import numpy as np
import torch


def test_rfft_basic():
    """Test basic rfft on identical inputs."""
    print("=" * 72)
    print("TEST 1: Basic rfft on identical inputs")
    print("=" * 72)

    # Create deterministic input
    np.random.seed(42)
    signal = np.random.randn(800).astype(np.float32)

    # PyTorch
    pt_input = torch.from_numpy(signal)
    pt_fft = torch.fft.rfft(pt_input)
    pt_mag = torch.abs(pt_fft).numpy()
    pt_phase = torch.angle(pt_fft).numpy()

    # MLX
    mx_input = mx.array(signal)
    mx_fft = mx.fft.rfft(mx_input)
    mx.eval(mx_fft)
    mx_mag = np.array(mx.abs(mx_fft))
    mx_phase = np.array(mx.arctan2(mx_fft.imag, mx_fft.real))

    # Compare magnitude
    mag_diff = np.abs(pt_mag - mx_mag)
    print(f"Magnitude: max_diff={mag_diff.max():.2e}, mean_diff={mag_diff.mean():.2e}")

    # Compare phase
    phase_diff = np.abs(pt_phase - mx_phase)
    print(f"Phase (raw): max_diff={phase_diff.max():.4f}, mean_diff={phase_diff.mean():.4f}")

    # Identify phase boundary differences (±2π)
    boundary_mask = phase_diff > 3.0  # More than π, likely a 2π wrap
    n_boundary = boundary_mask.sum()
    print(f"Phase boundary wraps: {n_boundary}/{len(phase_diff)} bins ({100*n_boundary/len(phase_diff):.1f}%)")

    if n_boundary > 0:
        # Examine a few boundary cases
        boundary_indices = np.where(boundary_mask)[0][:5]
        print("\nSample boundary cases:")
        for idx in boundary_indices:
            print(f"  bin[{idx}]: PT={pt_phase[idx]:.6f}, MLX={mx_phase[idx]:.6f}, diff={phase_diff[idx]:.4f}")
            # Get raw FFT values
            pt_real, pt_imag = pt_fft[idx].real.item(), pt_fft[idx].imag.item()
            mx_fft_np = np.array(mx_fft)
            mx_real, mx_imag = mx_fft_np[idx].real, mx_fft_np[idx].imag
            print(f"          PT:  real={pt_real:.2e}, imag={pt_imag:.2e}")
            print(f"          MLX: real={mx_real:.2e}, imag={mx_imag:.2e}")

    return n_boundary


def test_rfft_synthetic_boundary():
    """Test rfft with synthetic signal that creates phase boundaries."""
    print("\n" + "=" * 72)
    print("TEST 2: Synthetic boundary test")
    print("=" * 72)

    # Create signal designed to produce real < 0, imag ≈ 0 in FFT
    # This is the boundary condition where arctan2 returns ±π
    n = 800
    t = np.arange(n, dtype=np.float32) / n

    # Mix of frequencies to create various FFT magnitudes
    signal = (np.sin(2 * np.pi * 5 * t) +
              0.5 * np.cos(2 * np.pi * 10 * t) +
              0.25 * np.sin(2 * np.pi * 20 * t)).astype(np.float32)

    # PyTorch
    pt_input = torch.from_numpy(signal)
    pt_fft = torch.fft.rfft(pt_input)

    # MLX
    mx_input = mx.array(signal)
    mx_fft = mx.fft.rfft(mx_input)
    mx.eval(mx_fft)

    # Find bins where real < 0 and |imag| is small
    pt_real = pt_fft.real.numpy()
    pt_imag = pt_fft.imag.numpy()
    mx_real_arr = np.array(mx_fft.real)
    mx_imag_arr = np.array(mx_fft.imag)

    # Near-boundary conditions: real < 0 and |imag| < threshold
    threshold = 1e-5
    pt_boundary = (pt_real < 0) & (np.abs(pt_imag) < threshold)
    mx_boundary = (mx_real_arr < 0) & (np.abs(mx_imag_arr) < threshold)

    print(f"PyTorch bins at boundary (real<0, |imag|<{threshold}): {pt_boundary.sum()}")
    print(f"MLX bins at boundary: {mx_boundary.sum()}")

    # Check if imaginary signs differ at boundaries
    both_boundary = pt_boundary & mx_boundary
    if both_boundary.sum() > 0:
        boundary_idx = np.where(both_boundary)[0][:5]
        print("\nBins at boundary in both frameworks:")
        for idx in boundary_idx:
            pt_angle = np.arctan2(pt_imag[idx], pt_real[idx])
            mx_angle = np.arctan2(mx_imag_arr[idx], mx_real_arr[idx])
            sign_match = "SAME" if np.sign(pt_imag[idx]) == np.sign(mx_imag_arr[idx]) else "DIFFER"
            print(f"  bin[{idx}]: PT imag={pt_imag[idx]:.2e}, MLX imag={mx_imag_arr[idx]:.2e} [{sign_match}]")
            print(f"           PT angle={pt_angle:.6f}, MLX angle={mx_angle:.6f}")


def test_phase_normalization():
    """Test if phase normalization can fix the boundary issue."""
    print("\n" + "=" * 72)
    print("TEST 3: Phase normalization strategies")
    print("=" * 72)

    np.random.seed(42)
    signal = np.random.randn(800).astype(np.float32)

    # PyTorch
    pt_input = torch.from_numpy(signal)
    pt_fft = torch.fft.rfft(pt_input)
    pt_phase = torch.angle(pt_fft).numpy()

    # MLX
    mx_input = mx.array(signal)
    mx_fft = mx.fft.rfft(mx_input)
    mx.eval(mx_fft)
    mx_phase = np.array(mx.arctan2(mx_fft.imag, mx_fft.real))

    # Strategy 1: Normalize both to [0, 2π)
    pt_phase_norm = np.where(pt_phase < 0, pt_phase + 2*np.pi, pt_phase)
    mx_phase_norm = np.where(mx_phase < 0, mx_phase + 2*np.pi, mx_phase)
    diff_norm = np.abs(pt_phase_norm - mx_phase_norm)

    # Handle wraparound at 2π
    diff_norm = np.minimum(diff_norm, 2*np.pi - diff_norm)
    print(f"Strategy 1 [0, 2π): max_diff={diff_norm.max():.6f}")

    # Strategy 2: Use cos/sin comparison (magnitude-weighted)
    pt_mag = np.abs(pt_fft.numpy())  # used for weighting below

    # Compare phase through real/imag rather than angle
    pt_unit_real = np.cos(pt_phase)
    pt_unit_imag = np.sin(pt_phase)
    mx_unit_real = np.cos(mx_phase)
    mx_unit_imag = np.sin(mx_phase)

    unit_diff = np.sqrt((pt_unit_real - mx_unit_real)**2 + (pt_unit_imag - mx_unit_imag)**2)
    # Weight by magnitude (phase doesn't matter when magnitude is small)
    weighted_unit_diff = unit_diff * pt_mag / (pt_mag.max() + 1e-8)
    print(f"Strategy 2 (unit vector): max_diff={unit_diff.max():.6f}, weighted_max={weighted_unit_diff.max():.6f}")

    # Strategy 3: Compare complex values directly
    complex_diff = np.abs(pt_fft.numpy() - np.array(mx_fft.real) - 1j*np.array(mx_fft.imag))
    print(f"Strategy 3 (complex): max_diff={complex_diff.max():.2e}")

    return diff_norm.max()


def test_stft_direct_comparison():
    """Test STFT with identical inputs."""
    print("\n" + "=" * 72)
    print("TEST 4: Direct STFT comparison")
    print("=" * 72)

    # Load the same STFT implementation
    sys.path.insert(0, "tools/pytorch_to_mlx")
    from converters.models.stft import TorchSTFT as MLX_TorchSTFT

    # Create 1 second of test audio
    np.random.seed(42)
    sr = 24000
    audio = np.random.randn(sr).astype(np.float32)

    # MLX STFT
    mlx_stft = MLX_TorchSTFT(filter_length=800, hop_length=200, win_length=800)
    mlx_audio = mx.array(audio)[None, :]  # [1, sr]
    mlx_mag, mlx_phase = mlx_stft.transform(mlx_audio)
    mx.eval(mlx_mag, mlx_phase)
    mlx_mag_np = np.array(mlx_mag).squeeze()  # [n_fft//2+1, num_frames]
    mlx_phase_np = np.array(mlx_phase).squeeze()

    # PyTorch STFT
    pt_audio = torch.from_numpy(audio)[None, :]  # [1, sr]
    window = torch.hann_window(800)
    pt_stft = torch.stft(
        pt_audio.squeeze(),
        n_fft=800,
        hop_length=200,
        win_length=800,
        window=window,
        return_complex=True,
        center=True,
        pad_mode='reflect'  # Note: MLX uses constant padding
    )
    pt_mag_np = torch.abs(pt_stft).numpy()  # [n_fft//2+1, num_frames]
    pt_phase_np = torch.angle(pt_stft).numpy()

    # Compare
    min_frames = min(mlx_mag_np.shape[1], pt_mag_np.shape[1])
    mlx_mag_cmp = mlx_mag_np[:, :min_frames]
    pt_mag_cmp = pt_mag_np[:, :min_frames]
    mlx_phase_cmp = mlx_phase_np[:, :min_frames]
    pt_phase_cmp = pt_phase_np[:, :min_frames]

    mag_diff = np.abs(mlx_mag_cmp - pt_mag_cmp)
    phase_diff = np.abs(mlx_phase_cmp - pt_phase_cmp)

    print(f"Shapes: MLX={mlx_mag_np.shape}, PT={pt_mag_np.shape}")
    print(f"Magnitude: max_diff={mag_diff.max():.6f}, mean_diff={mag_diff.mean():.6f}")
    print(f"Phase (raw): max_diff={phase_diff.max():.4f}")

    # Phase with boundary handling
    phase_diff_bounded = np.minimum(phase_diff, 2*np.pi - phase_diff)
    print(f"Phase (bounded): max_diff={phase_diff_bounded.max():.6f}")

    # Count boundary wraps
    boundary_wraps = phase_diff > 3.0
    print(f"Phase boundary wraps: {boundary_wraps.sum()}/{phase_diff.size} ({100*boundary_wraps.sum()/phase_diff.size:.1f}%)")

    # Note difference in padding
    print("\nNote: PyTorch uses reflect padding, MLX uses constant padding.")
    print("This affects edge frames differently.")


def test_istft_roundtrip():
    """Test STFT -> ISTFT roundtrip with phase fix."""
    print("\n" + "=" * 72)
    print("TEST 5: STFT/ISTFT roundtrip comparison")
    print("=" * 72)

    sys.path.insert(0, "tools/pytorch_to_mlx")
    from converters.models.stft import TorchSTFT as MLX_TorchSTFT

    np.random.seed(42)
    sr = 24000
    original = np.random.randn(sr).astype(np.float32)

    # MLX roundtrip
    mlx_stft = MLX_TorchSTFT(filter_length=800, hop_length=200, win_length=800)
    mlx_audio = mx.array(original)[None, :]
    mlx_mag, mlx_phase = mlx_stft.transform(mlx_audio)
    mlx_recon = mlx_stft.inverse(mlx_mag, mlx_phase)
    mx.eval(mlx_recon)
    mlx_recon_np = np.array(mlx_recon).squeeze()

    # PyTorch roundtrip
    pt_audio = torch.from_numpy(original)
    window = torch.hann_window(800)
    pt_stft = torch.stft(
        pt_audio, n_fft=800, hop_length=200, win_length=800,
        window=window, return_complex=True, center=True
    )
    pt_recon = torch.istft(
        pt_stft, n_fft=800, hop_length=200, win_length=800,
        window=window, center=True
    )
    pt_recon_np = pt_recon.numpy()

    # Compare reconstructions
    min_len = min(len(mlx_recon_np), len(pt_recon_np), len(original))

    # Skip edges
    start, end = 800, min_len - 800

    mlx_error = np.abs(mlx_recon_np[start:end] - original[start:end])
    pt_error = np.abs(pt_recon_np[start:end] - original[start:end])
    cross_error = np.abs(mlx_recon_np[start:end] - pt_recon_np[start:end])

    print(f"MLX reconstruction error: max={mlx_error.max():.6f}, mean={mlx_error.mean():.6f}")
    print(f"PT reconstruction error: max={pt_error.max():.6f}, mean={pt_error.mean():.6f}")
    print(f"Cross-framework error: max={cross_error.max():.6f}, mean={cross_error.mean():.6f}")


def main():
    print("FFT Parity Test - MLX vs PyTorch")
    print("=" * 72)
    print(f"PyTorch version: {torch.__version__}")
    print(f"MLX version: {mx.__version__}")
    print()

    n_boundary_basic = test_rfft_basic()
    test_rfft_synthetic_boundary()
    normalized_max = test_phase_normalization()
    test_stft_direct_comparison()
    test_istft_roundtrip()

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)

    if n_boundary_basic > 0:
        print(f"Phase boundary wraps detected: {n_boundary_basic} bins")
        print("Root cause: arctan2 returns ±π when imag≈0 and real<0")
        print("This is mathematically correct but non-deterministic across frameworks.")

        if normalized_max < 0.001:
            print(f"\nPhase normalization achieves max_diff={normalized_max:.6f}")
            print("FIX AVAILABLE: Normalize phase to [0, 2π) before comparison/storage")
        else:
            print(f"\nPhase normalization insufficient: max_diff={normalized_max:.6f}")
    else:
        print("No phase boundary issues detected!")


if __name__ == "__main__":
    main()
