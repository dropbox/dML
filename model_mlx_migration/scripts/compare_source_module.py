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
Compare SourceModule outputs (har_source, noise, uv) between MLX and PyTorch.
"""

from pathlib import Path

import mlx.core as mx
import numpy as np


def main():
    ref_dir = Path("/tmp/kokoro_ref_seed0")
    ref = np.load(ref_dir / "tensors.npz")

    # Load MLX model
    from tools.pytorch_to_mlx.converters import KokoroConverter
    converter = KokoroConverter()
    mlx_model, _, _ = converter.load_from_hf("hexgrad/Kokoro-82M")
    mlx_model.set_deterministic(True)

    generator = mlx_model.decoder.generator

    # Get F0 input
    f0 = mx.array(ref["F0_pred"].astype(np.float32))

    # Calculate upp
    total_upp = 1
    for r in generator.config.istft_upsample_rates:
        total_upp *= r
    total_upp *= generator.istft_hop_size

    # Run MLX source module
    har_source_mlx, noise_mlx, uv_mlx = generator.m_source(f0, total_upp)
    mx.eval(har_source_mlx, noise_mlx, uv_mlx)

    # PyTorch references
    pt_har = ref["gen_har_source"].astype(np.float32)
    pt_noise = ref["gen_noi_source"].astype(np.float32)
    pt_uv = ref["gen_uv"].astype(np.float32)

    # Convert MLX to numpy
    mlx_har = np.array(har_source_mlx)
    mlx_noise = np.array(noise_mlx)
    mlx_uv = np.array(uv_mlx)

    print("=" * 72)
    print("SourceModule Output Comparison")
    print("=" * 72)

    # har_source
    print("\nhar_source:")
    print(f"  MLX shape: {mlx_har.shape}, PyTorch shape: {pt_har.shape}")
    if mlx_har.shape == pt_har.shape:
        diff = np.abs(mlx_har - pt_har)
        print(f"  Max diff: {diff.max():.6f}")
        print(f"  Mean diff: {diff.mean():.6f}")
        print(f"  MLX range: [{mlx_har.min():.4f}, {mlx_har.max():.4f}]")
        print(f"  PyTorch range: [{pt_har.min():.4f}, {pt_har.max():.4f}]")

        # Find where differences are
        if diff.max() > 0.001:
            flat_diff = diff.reshape(-1)
            large_diff_idx = np.where(flat_diff > 0.01)[0]
            print(f"  Samples with diff > 0.01: {len(large_diff_idx)}")
            if len(large_diff_idx) > 0:
                print(f"  First 10: {large_diff_idx[:10]}")
    else:
        print("  Shape mismatch!")

    # noise
    print("\nnoise:")
    print(f"  MLX shape: {mlx_noise.shape}, PyTorch shape: {pt_noise.shape}")
    if mlx_noise.shape == pt_noise.shape:
        diff = np.abs(mlx_noise - pt_noise)
        print(f"  Max diff: {diff.max():.6f}")
        print(f"  Mean diff: {diff.mean():.6f}")
    else:
        print("  Shape mismatch!")
    print("  Note: Noise is stochastic - diffs expected even with same seed")

    # uv
    print("\nuv (unvoiced/voiced mask):")
    print(f"  MLX shape: {mlx_uv.shape}, PyTorch shape: {pt_uv.shape}")
    if mlx_uv.shape == pt_uv.shape:
        diff = np.abs(mlx_uv - pt_uv)
        print(f"  Max diff: {diff.max():.6f}")
        print(f"  Mean diff: {diff.mean():.6f}")
        # UV should be binary (0 or 1)
        print(f"  MLX unique values: {np.unique(mlx_uv)[:5]}")
        print(f"  PyTorch unique values: {np.unique(pt_uv)[:5]}")
    else:
        print("  Shape mismatch!")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
