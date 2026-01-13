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
Debug the MLX STFT/ISTFT roundtrip bug.

Test shows:
- MLX roundtrip: max_error=1.12 (BAD)
- PyTorch roundtrip: max_error=0.000001 (GOOD)

This proves there's a bug in the MLX STFT implementation.
"""

import sys

import mlx.core as mx
import numpy as np
import torch

sys.path.insert(0, "tools/pytorch_to_mlx")
from converters.models.stft import TorchSTFT as MLX_TorchSTFT
from converters.models.stft import get_hann_window, overlap_add


def test_window_sum_normalization():
    """Test the overlap-add window sum normalization."""
    print("=" * 72)
    print("TEST: Window sum normalization")
    print("=" * 72)

    hop_length = 200
    win_length = 800  # also used as n_fft

    window = get_hann_window(win_length)
    mx.eval(window)
    window_np = np.array(window)

    # Expected window sum for overlap-add with Hann window
    # For hop_length=200, win_length=800, overlap ratio = 200/800 = 0.25
    # Number of overlapping windows = 800/200 = 4
    # Sum should be ~constant in the middle

    num_frames = 10
    output_length = (num_frames - 1) * hop_length + win_length

    # Compute window sum the same way as overlap_add
    window_sum = np.zeros(output_length)
    for i in range(num_frames):
        start = i * hop_length
        end = start + win_length
        window_sum[start:end] += window_np[:end-start]

    print(f"Output length: {output_length}")
    print(f"Window sum stats: min={window_sum.min():.4f}, max={window_sum.max():.4f}")
    print(f"Window sum in middle: {window_sum[800:1000].mean():.4f}")

    # For proper reconstruction, window_sum should be constant
    # With Hann window and hop=n_fft/4, the sum should be exactly 1.5
    # With hop=n_fft/2, sum = 1.0
    # With hop=n_fft/4 (200 for n_fft=800), sum = 2.0 (4 windows each contributing 0.5 avg)

    # Check if the issue is in window**2 vs window
    window_squared_sum = np.zeros(output_length)
    for i in range(num_frames):
        start = i * hop_length
        end = start + win_length
        window_squared_sum[start:end] += window_np[:end-start] ** 2

    print(f"Window squared sum in middle: {window_squared_sum[800:1000].mean():.4f}")


def test_overlap_add_implementation():
    """Test the overlap_add function directly."""
    print("\n" + "=" * 72)
    print("TEST: overlap_add implementation")
    print("=" * 72)

    n_fft = 800
    hop_length = 200
    num_frames = 10
    batch = 1

    # Create simple test frames
    frames = mx.ones((batch, num_frames, n_fft))
    window = get_hann_window(n_fft)
    mx.eval(window)

    # Run our overlap_add
    result = overlap_add(frames, hop_length, window)
    mx.eval(result)
    result_np = np.array(result).squeeze()

    # The middle of the output should be ~constant if overlap-add is correct
    middle = result_np[800:1000]
    print(f"Output shape: {result_np.shape}")
    print(f"Middle region mean: {middle.mean():.4f}, std: {middle.std():.6f}")

    # With constant input of 1.0 and Hann window normalization,
    # output should be ~1.0 in the middle


def test_pytorch_stft_parameters():
    """Compare PyTorch STFT parameter behavior."""
    print("\n" + "=" * 72)
    print("TEST: PyTorch STFT parameter comparison")
    print("=" * 72)

    np.random.seed(42)
    signal = np.random.randn(24000).astype(np.float32)
    pt_signal = torch.from_numpy(signal)

    window = torch.hann_window(800)

    # Test with center=True (default, adds padding)
    stft_center = torch.stft(
        pt_signal, n_fft=800, hop_length=200, win_length=800,
        window=window, return_complex=True, center=True
    )
    recon_center = torch.istft(
        stft_center, n_fft=800, hop_length=200, win_length=800,
        window=window, center=True
    )

    error_center = np.abs(recon_center.numpy() - signal).max()

    print(f"center=True: max_error={error_center:.6f}")
    print(f"STFT shape: center={stft_center.shape}")


def test_pytorch_istft_onesided():
    """Test PyTorch ISTFT behavior with one-sided FFT."""
    print("\n" + "=" * 72)
    print("TEST: PyTorch ISTFT with one-sided FFT")
    print("=" * 72)

    np.random.seed(42)
    signal = np.random.randn(24000).astype(np.float32)
    pt_signal = torch.from_numpy(signal)
    window = torch.hann_window(800)

    # Standard roundtrip
    stft = torch.stft(
        pt_signal, n_fft=800, hop_length=200, win_length=800,
        window=window, return_complex=True, center=True
    )

    # Reconstruct
    recon = torch.istft(
        stft, n_fft=800, hop_length=200, win_length=800,
        window=window, center=True
    )

    # Check if onesided=True is default
    print(f"STFT output shape: {stft.shape}")  # Should be [401, frames] for onesided
    print(f"Expected onesided bins: {800//2 + 1} = 401")

    error = np.abs(recon.numpy() - signal).max()
    print(f"Roundtrip error: {error:.10f}")


def test_mlx_stft_step_by_step():
    """Step through MLX STFT to find the bug."""
    print("\n" + "=" * 72)
    print("TEST: MLX STFT step-by-step debug")
    print("=" * 72)

    np.random.seed(42)
    sr = 24000
    signal = np.random.randn(sr).astype(np.float32)

    stft = MLX_TorchSTFT(filter_length=800, hop_length=200, win_length=800)

    # Step 1: Pad the signal
    x = mx.array(signal)[None, :]  # [1, sr]
    pad_amount = stft.filter_length // 2  # 400
    x_padded = mx.pad(x, [(0, 0), (pad_amount, pad_amount)])
    mx.eval(x_padded)
    print(f"Step 1 - Padded signal: {x.shape} -> {x_padded.shape}")

    # Step 2: Frame the signal
    from converters.models.stft import frame_signal
    frames = frame_signal(x_padded, stft.win_length, stft.hop_length)
    mx.eval(frames)
    print(f"Step 2 - Frames: {frames.shape}")

    # Step 3: Apply window
    windowed_frames = frames * stft.window
    mx.eval(windowed_frames)
    print(f"Step 3 - Windowed: {windowed_frames.shape}")

    # Step 4: FFT
    spectrum = mx.fft.rfft(windowed_frames, axis=-1)
    mx.eval(spectrum)
    print(f"Step 4 - Spectrum: {spectrum.shape}")

    # Step 5: Magnitude and phase
    mag = mx.abs(spectrum)
    phase = mx.arctan2(spectrum.imag, spectrum.real)
    mx.eval(mag, phase)
    print(f"Step 5 - Mag/Phase: {mag.shape}, {phase.shape}")

    # Step 6: Inverse - reconstruct spectrum
    real = mag * mx.cos(phase)
    imag = mag * mx.sin(phase)
    spectrum_recon = real + 1j * imag
    mx.eval(spectrum_recon)

    # Verify spectrum reconstruction is lossless
    spectrum_np = np.array(spectrum)
    spectrum_recon_np = np.array(spectrum_recon)
    spec_diff = np.abs(spectrum_np - spectrum_recon_np)
    print(f"Step 6 - Spectrum recon error: max={spec_diff.max():.2e}")

    # Step 7: Inverse FFT
    frames_recon = mx.fft.irfft(spectrum_recon, n=stft.filter_length, axis=-1)
    mx.eval(frames_recon)
    print(f"Step 7 - IFFT frames: {frames_recon.shape}")

    # Compare to original windowed frames
    windowed_np = np.array(windowed_frames)
    frames_recon_np = np.array(frames_recon)
    ifft_diff = np.abs(windowed_np - frames_recon_np)
    print(f"Step 7 - IFFT error: max={ifft_diff.max():.6f}")

    # Step 8: Apply window again (for overlap-add)
    frames_windowed = frames_recon * stft.window
    mx.eval(frames_windowed)

    # Step 9: Overlap-add
    signal_recon = overlap_add(frames_windowed, stft.hop_length, stft.window)
    mx.eval(signal_recon)
    print(f"Step 9 - Overlap-add: {signal_recon.shape}")

    # Step 10: Remove padding
    signal_unpadded = signal_recon[:, pad_amount:-pad_amount]
    mx.eval(signal_unpadded)
    print(f"Step 10 - Unpadded: {signal_unpadded.shape}")

    # Compare to original
    orig_len = len(signal)
    recon_len = signal_unpadded.shape[1]
    min_len = min(orig_len, recon_len)

    signal_recon_np = np.array(signal_unpadded).squeeze()[:min_len]
    signal_orig = signal[:min_len]

    # Skip edges
    start, end = 800, min_len - 800
    error = np.abs(signal_recon_np[start:end] - signal_orig[start:end])
    print(f"\nFinal reconstruction error (middle): max={error.max():.6f}, mean={error.mean():.6f}")

    # Compare to PyTorch
    pt_signal = torch.from_numpy(signal)
    pt_window = torch.hann_window(800)
    pt_stft = torch.stft(pt_signal, n_fft=800, hop_length=200, win_length=800,
                         window=pt_window, return_complex=True, center=True)
    pt_recon = torch.istft(pt_stft, n_fft=800, hop_length=200, win_length=800,
                           window=pt_window, center=True)
    pt_error = np.abs(pt_recon.numpy() - signal).max()
    print(f"PyTorch reconstruction error: max={pt_error:.6f}")


def main():
    test_window_sum_normalization()
    test_overlap_add_implementation()
    test_pytorch_stft_parameters()
    test_pytorch_istft_onesided()
    test_mlx_stft_step_by_step()


if __name__ == "__main__":
    main()
