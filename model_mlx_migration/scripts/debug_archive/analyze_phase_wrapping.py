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

"""Analyze phase wrapping differences between MLX and PyTorch STFT."""

from pathlib import Path

import mlx.core as mx
import numpy as np


def main():
    ref_dir = Path("/tmp/kokoro_ref_decoder")
    gen_ref = np.load(ref_dir / "generator_intermediates.npz")

    # Load har (STFT output) from both
    pt_har = gen_ref["gen_har"]  # PyTorch STFT output
    print(f"PyTorch har shape: {pt_har.shape}")  # Expected: [batch, frames, 22]

    # Split into magnitude and phase (11 bins each)
    n_bins = pt_har.shape[-1] // 2
    pt_mag = pt_har[..., :n_bins]
    pt_phase = pt_har[..., n_bins:]

    print("\nPyTorch phase stats:")
    print(f"  Min: {pt_phase.min():.6f}")
    print(f"  Max: {pt_phase.max():.6f}")
    print(f"  Mean: {pt_phase.mean():.6f}")

    # Load MLX har_source and compute STFT
    from tools.pytorch_to_mlx.converters import KokoroConverter
    converter = KokoroConverter()
    mlx_model, _, _ = converter.load_from_hf("hexgrad/Kokoro-82M")
    mlx_model.set_deterministic(True)

    generator = mlx_model.decoder.generator

    # Use MLX har_source
    ref = np.load(ref_dir / "tensors.npz")
    f0 = mx.array(ref["F0_pred"].astype(np.float32))

    # Compute har_source
    total_upp = 1
    for r in generator.config.istft_upsample_rates:
        total_upp *= r
    total_upp *= generator.istft_hop_size

    mlx_har_source, _, _ = generator.m_source(f0, total_upp)
    mx.eval(mlx_har_source)

    # Get PyTorch har_source
    pt_har_source = gen_ref["gen_har_source"]

    print("\n=== har_source comparison ===")
    mlx_har_np = np.array(mlx_har_source)
    diff = np.abs(mlx_har_np - pt_har_source)
    print(f"Max diff: {diff.max():.6f}")

    # Find where differences occur
    large_diff_mask = diff > 0.1
    print(f"Positions with >0.1 diff: {large_diff_mask.sum()} / {diff.size}")

    # Run STFT on both har_sources
    print("\n=== STFT comparison ===")

    # MLX STFT
    har_1d = mlx_har_source.squeeze(-1)
    mlx_har = generator._source_stft(har_1d)
    mx.eval(mlx_har)
    mlx_har_np = np.array(mlx_har)

    mlx_mag = mlx_har_np[..., :n_bins]
    mlx_phase = mlx_har_np[..., n_bins:]

    print("MLX phase stats:")
    print(f"  Min: {mlx_phase.min():.6f}")
    print(f"  Max: {mlx_phase.max():.6f}")
    print(f"  Mean: {mlx_phase.mean():.6f}")

    # Compare magnitude
    mag_diff = np.abs(mlx_mag - pt_mag)
    print(f"\nMagnitude diff: max={mag_diff.max():.6f}, mean={mag_diff.mean():.6f}")

    # Compare phase
    phase_diff = np.abs(mlx_phase - pt_phase)
    print(f"Phase diff (raw): max={phase_diff.max():.6f}, mean={phase_diff.mean():.6f}")

    # Check for 2π differences
    large_phase_diff = phase_diff > 3.0  # > π difference
    print(f"Positions with >π diff: {large_phase_diff.sum()} / {phase_diff.size}")

    # These are likely 2π wrapping issues
    # Normalize phase difference to [-π, π]
    phase_diff_normalized = np.abs(mlx_phase - pt_phase)
    phase_diff_normalized = np.minimum(phase_diff_normalized, 2*np.pi - phase_diff_normalized)
    print(f"Phase diff (normalized ±2π): max={phase_diff_normalized.max():.6f}, mean={phase_diff_normalized.mean():.6f}")

    # Test: what if we use PyTorch's har_source through MLX STFT?
    print("\n=== Test: PyTorch har_source through MLX STFT ===")
    pt_har_source_mx = mx.array(pt_har_source.astype(np.float32))
    har_1d_pt = pt_har_source_mx.squeeze(-1)
    mlx_har_from_pt = generator._source_stft(har_1d_pt)
    mx.eval(mlx_har_from_pt)
    mlx_har_from_pt_np = np.array(mlx_har_from_pt)

    mlx_from_pt_phase = mlx_har_from_pt_np[..., n_bins:]

    phase_diff_2 = np.abs(mlx_from_pt_phase - pt_phase)
    print(f"Phase diff (PT source, MLX STFT): max={phase_diff_2.max():.6f}")

    large_phase_diff_2 = phase_diff_2 > 3.0
    print(f"Positions with >π diff: {large_phase_diff_2.sum()} / {phase_diff_2.size}")

    # Normalize
    phase_diff_2_norm = np.minimum(phase_diff_2, 2*np.pi - phase_diff_2)
    print(f"Phase diff (normalized ±2π): max={phase_diff_2_norm.max():.6f}")

    # Where exactly do the large differences occur?
    print("\n=== Analyzing 2π differences ===")
    if large_phase_diff_2.any():
        indices = np.where(large_phase_diff_2)
        print("Sample positions with >π diff:")
        for i in range(min(10, len(indices[0]))):
            b, f, c = indices[0][i], indices[1][i], indices[2][i]
            mlx_val = mlx_from_pt_phase[b, f, c]
            pt_val = pt_phase[b, f, c]
            print(f"  [{b},{f},{c}]: MLX={mlx_val:.4f}, PT={pt_val:.4f}, diff={abs(mlx_val-pt_val):.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
