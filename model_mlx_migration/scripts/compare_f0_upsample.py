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
Compare F0 upsampling between MLX and PyTorch.
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

    # F0 from predictor
    f0 = mx.array(ref["F0_pred"].astype(np.float32))  # [B, T]
    print(f"F0 input shape: {f0.shape}")

    # PyTorch f0_up
    pt_f0_up = ref["gen_f0_up"].astype(np.float32)  # [B, T_up, 1]
    print(f"PyTorch gen_f0_up shape: {pt_f0_up.shape}")

    # Get upsample factor (same calculation as in Generator)
    total_upp = 1
    for r in generator.config.istft_upsample_rates:
        total_upp *= r
    total_upp *= generator.istft_hop_size
    print(f"Total upsample factor: {total_upp}")

    # The SourceModule does its own F0 upsampling using mx.repeat
    # Trace through m_source to get f0_up
    batch, length = f0.shape

    # This is how SourceModule does F0 upsampling (line 757-758 in kokoro.py)
    f0_up = mx.repeat(f0[:, :, None], total_upp, axis=1).squeeze(-1)  # [batch, samples]
    mx.eval(f0_up)
    print(f"MLX f0_up (via SourceModule pattern): {f0_up.shape}")

    # Convert to [B, T, 1] for comparison with PyTorch
    mlx_f0_up = np.array(f0_up)[:, :, None]  # [B, T_up, 1]
    print(f"MLX f0_up shape: {mlx_f0_up.shape}")

    # Compare
    min_len = min(mlx_f0_up.shape[1], pt_f0_up.shape[1])
    diff = np.abs(mlx_f0_up[:, :min_len, :] - pt_f0_up[:, :min_len, :])

    print("\nF0 upsample comparison:")
    print(f"  Max diff: {diff.max():.6f}")
    print(f"  Mean diff: {diff.mean():.6f}")
    print(f"  MLX range: [{mlx_f0_up.min():.4f}, {mlx_f0_up.max():.4f}]")
    print(f"  PyTorch range: [{pt_f0_up.min():.4f}, {pt_f0_up.max():.4f}]")

    # Check at specific locations
    if diff.max() > 0.001:
        flat_diff = diff.reshape(-1)
        top_idx = np.argsort(flat_diff)[-5:][::-1]
        print("\nTop 5 differences:")
        for idx in top_idx:
            loc = np.unravel_index(idx, diff.shape)
            t = loc[1]
            print(f"  Sample {t}: mlx={mlx_f0_up[0,t,0]:.6f}, pt={pt_f0_up[0,t,0]:.6f}, diff={diff[loc]:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
