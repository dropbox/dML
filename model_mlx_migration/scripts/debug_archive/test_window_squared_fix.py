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
Test the window squared fix for overlap-add normalization.

The issue: Window is applied twice (STFT and ISTFT), so normalization
should use window**2, not window.
"""

import sys

import mlx.core as mx
import numpy as np
import torch

sys.path.insert(0, "tools/pytorch_to_mlx")
from converters.models.stft import frame_signal, get_hann_window


def overlap_add_fixed(frames: mx.array, hop_length: int, window: mx.array) -> mx.array:
    """
    Fixed overlap-add using window**2 for normalization.
    """
    batch, num_frames, frame_length = frames.shape
    output_length = (num_frames - 1) * hop_length + frame_length

    # Overlap-add each frame
    contributions = []
    for i in range(num_frames):
        start = i * hop_length
        pre_pad = mx.zeros((batch, start))
        post_pad_len = output_length - start - frame_length
        if post_pad_len > 0:
            post_pad = mx.zeros((batch, post_pad_len))
            contribution = mx.concatenate([pre_pad, frames[:, i, :], post_pad], axis=1)
        else:
            contribution = mx.concatenate(
                [pre_pad, frames[:, i, : output_length - start]], axis=1
            )
        contributions.append(contribution)

    output = sum(contributions, mx.zeros((batch, output_length)))

    # FIX: Compute window SQUARED sum for normalization
    # Because window is applied in both STFT and ISTFT
    window_sq = window * window
    window_sq_sum = mx.zeros((output_length,))
    for i in range(num_frames):
        start = i * hop_length
        end = min(start + frame_length, output_length)
        actual_len = end - start
        pre = mx.zeros((start,))
        if output_length - end > 0:
            post = mx.zeros((output_length - end,))
            ws_contrib = mx.concatenate([pre, window_sq[:actual_len], post], axis=0)
        else:
            ws_contrib = mx.concatenate([pre, window_sq[:actual_len]], axis=0)

        if i == 0:
            window_sq_sum = ws_contrib
        else:
            window_sq_sum = window_sq_sum + ws_contrib

    # Normalize by window squared sum
    window_sq_sum = mx.maximum(window_sq_sum, 1e-8)
    output = output / window_sq_sum

    return output


def test_fix():
    """Test the fixed overlap-add."""
    print("=" * 72)
    print("TEST: Fixed overlap-add with window**2 normalization")
    print("=" * 72)

    np.random.seed(42)
    sr = 24000
    signal = np.random.randn(sr).astype(np.float32)

    filter_length = 800
    hop_length = 200
    win_length = 800

    window = get_hann_window(win_length)
    mx.eval(window)

    # Step 1: Pad
    x = mx.array(signal)[None, :]
    pad_amount = filter_length // 2
    x_padded = mx.pad(x, [(0, 0), (pad_amount, pad_amount)])

    # Step 2: Frame
    frames = frame_signal(x_padded, win_length, hop_length)

    # Step 3: Window
    windowed_frames = frames * window

    # Step 4: FFT
    spectrum = mx.fft.rfft(windowed_frames, axis=-1)

    # Step 5: Magnitude and phase
    mag = mx.abs(spectrum)
    phase = mx.arctan2(spectrum.imag, spectrum.real)

    # Step 6: Inverse - reconstruct spectrum
    real = mag * mx.cos(phase)
    imag = mag * mx.sin(phase)
    spectrum_recon = real + 1j * imag

    # Step 7: IFFT
    frames_recon = mx.fft.irfft(spectrum_recon, n=filter_length, axis=-1)

    # Step 8: Apply window again
    frames_windowed = frames_recon * window
    mx.eval(frames_windowed)

    # Step 9a: Original overlap_add
    from converters.models.stft import overlap_add as overlap_add_orig
    signal_orig = overlap_add_orig(frames_windowed, hop_length, window)
    mx.eval(signal_orig)

    # Step 9b: Fixed overlap_add
    signal_fixed = overlap_add_fixed(frames_windowed, hop_length, window)
    mx.eval(signal_fixed)

    # Step 10: Remove padding
    signal_orig_np = np.array(signal_orig)[:, pad_amount:-pad_amount].squeeze()
    signal_fixed_np = np.array(signal_fixed)[:, pad_amount:-pad_amount].squeeze()

    # Compare
    min_len = min(len(signal), len(signal_orig_np), len(signal_fixed_np))
    start, end = 800, min_len - 800

    error_orig = np.abs(signal_orig_np[start:end] - signal[start:end])
    error_fixed = np.abs(signal_fixed_np[start:end] - signal[start:end])

    print(f"Original overlap_add: max_error={error_orig.max():.6f}, mean_error={error_orig.mean():.6f}")
    print(f"Fixed overlap_add:    max_error={error_fixed.max():.6f}, mean_error={error_fixed.mean():.6f}")

    # Compare to PyTorch
    pt_signal = torch.from_numpy(signal)
    pt_window = torch.hann_window(800)
    pt_stft = torch.stft(pt_signal, n_fft=800, hop_length=200, win_length=800,
                         window=pt_window, return_complex=True, center=True)
    pt_recon = torch.istft(pt_stft, n_fft=800, hop_length=200, win_length=800,
                           window=pt_window, center=True)
    pt_error = np.abs(pt_recon.numpy() - signal).max()
    print(f"PyTorch roundtrip:    max_error={pt_error:.6f}")

    if error_fixed.max() < 0.001:
        print("\nFIX SUCCESSFUL! Window**2 normalization achieves < 0.001 error.")
        return True
    else:
        print(f"\nFix reduced error from {error_orig.max():.4f} to {error_fixed.max():.4f}")
        return False


if __name__ == "__main__":
    success = test_fix()
    sys.exit(0 if success else 1)
