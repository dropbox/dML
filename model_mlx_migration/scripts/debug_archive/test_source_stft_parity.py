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
Test _source_stft (n_fft=20) parity between PyTorch and MLX.

This is the critical path causing the 0.066 max_abs error in audio output.
"""

import math

import mlx.core as mx
import numpy as np
import torch


def pytorch_source_stft(x: torch.Tensor, n_fft: int = 20, hop: int = 5) -> torch.Tensor:
    """
    PyTorch version of _source_stft.

    Args:
        x: [batch, samples] - Input waveform

    Returns:
        har: [batch, frames, 22] - Concatenated magnitude and phase
    """
    window = torch.hann_window(n_fft)

    # STFT with center=True (default PyTorch behavior)
    stft = torch.stft(
        x.squeeze(0) if x.dim() > 1 else x,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        return_complex=True,
        center=True,
        pad_mode='reflect'
    )
    # stft: [n_fft//2+1, frames]

    magnitude = torch.abs(stft)  # [11, frames]
    phase = torch.angle(stft)  # [11, frames]

    # Transpose and concatenate: [frames, 22]
    har = torch.cat([magnitude.T, phase.T], dim=-1)
    return har.unsqueeze(0)  # [1, frames, 22]


def mlx_source_stft(x: mx.array, n_fft: int = 20, hop: int = 5) -> mx.array:
    """
    MLX version of _source_stft (simplified from kokoro.py Generator).
    """
    batch, samples = x.shape

    # Create Hann window
    n = mx.arange(n_fft, dtype=mx.float32)
    window = 0.5 * (1 - mx.cos(2 * math.pi * n / n_fft))

    # Reflect padding
    pad_amount = n_fft // 2
    if pad_amount > 0:
        left_pad = x[:, 1:pad_amount + 1][:, ::-1]
        right_pad = x[:, -(pad_amount + 1):-1][:, ::-1]
        x_padded = mx.concatenate([left_pad, x, right_pad], axis=1)
    else:
        x_padded = x

    # Frame
    padded_len = x_padded.shape[1]
    num_frames = (padded_len - n_fft) // hop + 1
    indices = mx.arange(n_fft)[None, :] + mx.arange(num_frames)[:, None] * hop
    frames = x_padded[:, indices.flatten()].reshape(batch, num_frames, n_fft)

    # Window
    frames = frames * window

    # FFT
    spectrum = mx.fft.rfft(frames, axis=-1)

    # Magnitude and phase
    magnitude = mx.abs(spectrum)
    spectrum_real = spectrum.real + 0.0
    spectrum_imag = spectrum.imag + 0.0
    phase = mx.arctan2(spectrum_imag, spectrum_real)

    # Concatenate: [batch, frames, 22]
    har = mx.concatenate([magnitude, phase], axis=-1)

    return har


def test_source_stft():
    """Compare _source_stft between PyTorch and MLX."""
    print("=" * 72)
    print("Test: _source_stft (n_fft=20) parity")
    print("=" * 72)

    # Create test signal: sine wave like what SourceModule produces
    np.random.seed(42)
    sr = 24000
    f0 = 200  # Hz
    duration = 0.1  # 100ms = 2400 samples

    t = np.arange(int(sr * duration), dtype=np.float32) / sr
    signal = np.sin(2 * np.pi * f0 * t).astype(np.float32)

    # Add some noise like real source signal
    signal += 0.01 * np.random.randn(len(signal)).astype(np.float32)

    print(f"Input: {len(signal)} samples ({duration*1000:.0f}ms)")

    # PyTorch
    pt_signal = torch.from_numpy(signal)[None, :]
    pt_har = pytorch_source_stft(pt_signal)
    pt_har_np = pt_har.numpy().squeeze()

    # MLX
    mx_signal = mx.array(signal)[None, :]
    mx_har = mlx_source_stft(mx_signal)
    mx.eval(mx_har)
    mx_har_np = np.array(mx_har).squeeze()

    print(f"Output: PyTorch={pt_har_np.shape}, MLX={mx_har_np.shape}")

    # Compare
    min_frames = min(pt_har_np.shape[0], mx_har_np.shape[0])
    pt_cmp = pt_har_np[:min_frames, :]
    mx_cmp = mx_har_np[:min_frames, :]

    # Split magnitude and phase (11 bins each)
    pt_mag = pt_cmp[:, :11]
    pt_phase = pt_cmp[:, 11:]
    mx_mag = mx_cmp[:, :11]
    mx_phase = mx_cmp[:, 11:]

    mag_diff = np.abs(pt_mag - mx_mag)
    phase_diff = np.abs(pt_phase - mx_phase)

    print("\nMagnitude:")
    print(f"  max_diff: {mag_diff.max():.6f}")
    print(f"  mean_diff: {mag_diff.mean():.6f}")

    print("\nPhase (raw):")
    print(f"  max_diff: {phase_diff.max():.4f} rad ({phase_diff.max() * 180/np.pi:.2f} deg)")
    print(f"  mean_diff: {phase_diff.mean():.4f} rad")

    # Check for boundary wraps
    boundary_mask = phase_diff > 3.0
    n_boundary = boundary_mask.sum()
    print(f"  boundary wraps: {n_boundary}/{phase_diff.size} ({100*n_boundary/phase_diff.size:.1f}%)")

    # Phase with boundary handling (min of diff and 2π-diff)
    phase_diff_bounded = np.minimum(phase_diff, 2*np.pi - phase_diff)
    print("\nPhase (bounded):")
    print(f"  max_diff: {phase_diff_bounded.max():.6f} rad")
    print(f"  mean_diff: {phase_diff_bounded.mean():.6f} rad")

    # Combined error
    har_diff = np.abs(pt_cmp - mx_cmp)
    print("\nCombined (magnitude + phase):")
    print(f"  max_diff: {har_diff.max():.4f}")
    print(f"  mean_diff: {har_diff.mean():.4f}")

    # Show some boundary examples
    if n_boundary > 0:
        boundary_locs = np.where(boundary_mask)
        print("\nSample boundary wraps:")
        for i in range(min(5, n_boundary)):
            frame, bin_idx = boundary_locs[0][i], boundary_locs[1][i]
            print(f"  frame[{frame}] bin[{bin_idx}]: PT={pt_phase[frame, bin_idx]:.4f}, MLX={mx_phase[frame, bin_idx]:.4f}")

    return mag_diff.max(), phase_diff_bounded.max(), n_boundary


def test_with_deterministic_har_source():
    """Test with an actual har_source from reference if available."""
    print("\n" + "=" * 72)
    print("Test: Using actual har_source from reference")
    print("=" * 72)

    ref_path = "/tmp/kokoro_ref_deterministic_source/tensors.npz"
    try:
        ref = np.load(ref_path)
    except FileNotFoundError:
        print(f"Reference not found: {ref_path}")
        return

    if "gen_har_source" not in ref:
        print("gen_har_source not in reference")
        return

    har_source = ref["gen_har_source"].astype(np.float32)
    print(f"har_source shape: {har_source.shape}")

    # Convert from [batch, 1, samples] to [batch, samples] if needed
    if har_source.ndim == 3:
        if har_source.shape[1] == 1:
            har_source = har_source[:, 0, :]
        elif har_source.shape[2] == 1:
            har_source = har_source[:, :, 0]

    print(f"har_source (2D) shape: {har_source.shape}")

    # PyTorch
    pt_signal = torch.from_numpy(har_source)
    pt_har = pytorch_source_stft(pt_signal)
    pt_har_np = pt_har.numpy().squeeze()

    # MLX
    mx_signal = mx.array(har_source)
    mx_har = mlx_source_stft(mx_signal)
    mx.eval(mx_har)
    mx_har_np = np.array(mx_har).squeeze()

    print(f"Output: PyTorch={pt_har_np.shape}, MLX={mx_har_np.shape}")

    # Compare
    min_frames = min(pt_har_np.shape[0], mx_har_np.shape[0])
    pt_cmp = pt_har_np[:min_frames, :]
    mx_cmp = mx_har_np[:min_frames, :]

    pt_mag = pt_cmp[:, :11]
    pt_phase = pt_cmp[:, 11:]
    mx_mag = mx_cmp[:, :11]
    mx_phase = mx_cmp[:, 11:]

    mag_diff = np.abs(pt_mag - mx_mag)
    phase_diff = np.abs(pt_phase - mx_phase)
    phase_diff_bounded = np.minimum(phase_diff, 2*np.pi - phase_diff)

    boundary_mask = phase_diff > 3.0
    n_boundary = boundary_mask.sum()

    print(f"\nMagnitude: max_diff={mag_diff.max():.6f}")
    print(f"Phase (bounded): max_diff={phase_diff_bounded.max():.6f}")
    print(f"Phase boundary wraps: {n_boundary}/{phase_diff.size} ({100*n_boundary/phase_diff.size:.2f}%)")

    # Check the reference gen_har if available
    if "gen_har" in ref:
        ref_har = ref["gen_har"].astype(np.float32)
        print(f"\nReference gen_har shape: {ref_har.shape}")

        # gen_har is [batch, 22, frames] NCL -> transpose to NLC
        ref_har_nlc = ref_har.transpose(0, 2, 1).squeeze()

        # Compare to our PyTorch result
        min_frames = min(ref_har_nlc.shape[0], pt_har_np.shape[0])
        ref_cmp = ref_har_nlc[:min_frames, :]
        pt_cmp2 = pt_har_np[:min_frames, :]

        diff = np.abs(ref_cmp - pt_cmp2)
        print(f"gen_har vs our PyTorch STFT: max_diff={diff.max():.6f}")

        # Compare MLX to reference
        diff_mlx = np.abs(ref_cmp - mx_cmp[:min_frames, :])
        print(f"gen_har vs MLX STFT: max_diff={diff_mlx.max():.4f}")


def main():
    mag_max, phase_max, n_boundary = test_source_stft()
    test_with_deterministic_har_source()

    print("\n" + "=" * 72)
    print("CONCLUSION")
    print("=" * 72)
    if phase_max < 0.001 and mag_max < 0.001:
        print("Both magnitude and phase differences < 0.001")
        print("The _source_stft achieves parity!")
    else:
        print(f"Magnitude max_diff: {mag_max:.6f}")
        print(f"Phase max_diff (bounded): {phase_max:.6f}")
        print(f"Phase boundary wraps: {n_boundary}")
        print("\nThe phase boundary wraps are the root cause of the error.")
        print("These wraps occur at bins where |imag| ≈ 0 and real < 0,")
        print("causing arctan2 to return +π or -π non-deterministically.")


if __name__ == "__main__":
    main()
