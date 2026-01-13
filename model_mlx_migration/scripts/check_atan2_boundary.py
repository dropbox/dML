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

"""Check arctan2 behavior at ±π boundary in PyTorch vs MLX."""

from pathlib import Path

import mlx.core as mx
import numpy as np


def main():
    ref_dir = Path("/tmp/kokoro_ref_decoder")
    gen_ref = np.load(ref_dir / "generator_intermediates.npz")

    # Get PyTorch har_source and compute STFT with MLX
    pt_har_source = gen_ref["gen_har_source"]
    pt_har = gen_ref["gen_har"]

    n_bins = pt_har.shape[-1] // 2  # 11
    pt_phase = pt_har[..., n_bins:]

    # Load MLX model and compute STFT
    from tools.pytorch_to_mlx.converters import KokoroConverter
    converter = KokoroConverter()
    mlx_model, _, _ = converter.load_from_hf("hexgrad/Kokoro-82M")
    generator = mlx_model.decoder.generator

    # Manually compute STFT to get intermediate complex values
    har_1d = mx.array(pt_har_source.astype(np.float32)).squeeze(-1)

    # Replicate _source_stft logic
    batch, samples = har_1d.shape
    n_fft = generator.source_stft_n_fft
    hop = generator.source_stft_hop

    # Padding
    pad_amount = n_fft // 2
    x = har_1d
    left_pad = x[:, 1 : pad_amount + 1][:, ::-1]
    right_pad = x[:, -(pad_amount + 1) : -1][:, ::-1]
    x_padded = mx.concatenate([left_pad, x, right_pad], axis=1)

    # Frame
    padded_len = x_padded.shape[1]
    num_frames = (padded_len - n_fft) // hop + 1
    indices = mx.arange(n_fft)[None, :] + mx.arange(num_frames)[:, None] * hop
    frames = x_padded[:, indices.flatten()].reshape(batch, num_frames, n_fft)

    # Window
    frames = frames * generator._source_window

    # FFT
    spectrum = mx.fft.rfft(frames, axis=-1)
    mx.eval(spectrum)

    # Get real and imag
    real = np.array(spectrum.real)
    imag = np.array(spectrum.imag)

    # Compute MLX phase
    mlx_phase = np.array(mx.arctan2(spectrum.imag, spectrum.real))

    # Find positions where phase differs by ~2π
    phase_diff = np.abs(mlx_phase - pt_phase)
    large_diff_mask = phase_diff > 3.0

    print(f"Positions with >π phase diff: {large_diff_mask.sum()}")

    # Analyze the complex values at these positions
    print("\n=== Analyzing complex values at boundary positions ===")
    indices = np.where(large_diff_mask)
    for i in range(min(20, len(indices[0]))):
        b, f, c = indices[0][i], indices[1][i], indices[2][i]
        r, im = real[b, f, c], imag[b, f, c]
        mlx_p = mlx_phase[b, f, c]
        pt_p = pt_phase[b, f, c]
        print(f"  [{b},{f},{c}]: real={r:10.6f}, imag={im:12.9f}, MLX={mlx_p:8.4f}, PT={pt_p:8.4f}")

    # Check: is imag always very small at these positions?
    imag_at_diff = imag[large_diff_mask]
    real_at_diff = real[large_diff_mask]
    print("\nAt boundary positions:")
    print(f"  |imag| max: {np.abs(imag_at_diff).max():.9f}")
    print(f"  |imag| mean: {np.abs(imag_at_diff).mean():.9f}")
    print(f"  real min: {real_at_diff.min():.6f}")
    print(f"  real max: {real_at_diff.max():.6f}")

    # Pattern: imag sign determines which way atan2 goes at boundary
    print("\n=== Checking imag sign pattern ===")
    mlx_pos_pi = mlx_phase > 3.0  # MLX chose +π
    mlx_neg_pi = mlx_phase < -3.0  # MLX chose -π
    pt_pos_pi = pt_phase > 3.0  # PT chose +π
    pt_neg_pi = pt_phase < -3.0  # PT chose -π

    # Where MLX=+π and PT=-π
    case1 = mlx_pos_pi & pt_neg_pi
    # Where MLX=-π and PT=+π
    case2 = mlx_neg_pi & pt_pos_pi

    print(f"MLX=+π, PT=-π: {case1.sum()} positions")
    print(f"MLX=-π, PT=+π: {case2.sum()} positions")

    if case1.any():
        idx = np.where(case1)
        print("\n  Case 1 samples (MLX=+π, PT=-π):")
        for i in range(min(5, len(idx[0]))):
            b, f, c = idx[0][i], idx[1][i], idx[2][i]
            print(f"    [{b},{f},{c}]: imag={imag[b,f,c]:12.9f}")

    if case2.any():
        idx = np.where(case2)
        print("\n  Case 2 samples (MLX=-π, PT=+π):")
        for i in range(min(5, len(idx[0]))):
            b, f, c = idx[0][i], idx[1][i], idx[2][i]
            print(f"    [{b},{f},{c}]: imag={imag[b,f,c]:12.9f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
