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
Compare individual resblock outputs between PyTorch and MLX.
"""

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.pytorch_to_mlx.converters import KokoroConverter


def main():
    # Load internal traces
    internal_path = Path("/tmp/kokoro_ref/generator_internal_traces.npz")
    if not internal_path.exists():
        print("Internal traces not found")
        return 1
    internal = np.load(internal_path)

    print("Available resblock traces:")
    for k in sorted(internal.keys()):
        if "resblock" in k:
            print(f"  {k}: {internal[k].shape}")

    # Load MLX model
    print("\n=== Loading MLX Model ===")
    converter = KokoroConverter()
    model, config, _ = converter.load_from_hf()
    mx.eval(model)

    generator = model.decoder.generator

    # Load reference
    ref = np.load("/tmp/kokoro_ref/tensors.npz")
    style_128 = ref["style_128"]
    style_mx = mx.array(style_128)

    # Test individual resblocks with PyTorch input/output
    print("\n=== Testing Individual Resblocks ===")

    for i in range(6):  # 6 resblocks total
        in_key = f"resblock_{i}_in"
        out_key = f"resblock_{i}_out"

        if in_key not in internal or out_key not in internal:
            print(f"Resblock {i}: traces not found")
            continue

        pt_in = internal[in_key]  # NCL
        pt_out = internal[out_key]  # NCL

        # Convert to MLX NLC
        mlx_in = mx.array(pt_in).transpose(0, 2, 1)  # NLC

        # Run MLX resblock
        resblock = getattr(generator, f"resblocks_{i}")
        mlx_out = resblock(mlx_in, style_mx)
        mx.eval(mlx_out)

        # Convert back to NCL for comparison
        mlx_out_ncl = np.array(mlx_out.transpose(0, 2, 1))

        # Compare
        corr = np.corrcoef(mlx_out_ncl.flatten(), pt_out.flatten())[0, 1]
        max_diff = np.abs(mlx_out_ncl - pt_out).max()
        mean_diff = np.abs(mlx_out_ncl - pt_out).mean()

        pt_std = pt_out.std()
        mlx_std = mlx_out_ncl.std()

        print(f"\nResblock {i}:")
        print(f"  Input shape: {pt_in.shape}")
        print(
            f"  PT output: range [{pt_out.min():.4f}, {pt_out.max():.4f}], std={pt_std:.4f}"
        )
        print(
            f"  MLX output: range [{mlx_out_ncl.min():.4f}, {mlx_out_ncl.max():.4f}], std={mlx_std:.4f}"
        )
        print(f"  Correlation: {corr:.6f}")
        print(f"  Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")

    # Now test what happens when we chain resblocks with PT input
    print("\n\n=== Testing Resblock Chain (Stage 0: resblocks 0, 1, 2) ===")

    # PT stage 0 input (after ups_0 + source add)
    # We need to compute this from the trace
    # after_source_add_0 would be x before resblocks

    # Actually, let's use ups_0_out and noise_res_0_out to compute the combined input
    ups_0_out = internal["ups_0_out"]  # NCL [1, 256, 1260]
    noise_res_0_out = internal["noise_res_0_out"]  # NCL [1, 256, 1260]

    # They should be the same shape
    print(f"ups_0_out shape: {ups_0_out.shape}")
    print(f"noise_res_0_out shape: {noise_res_0_out.shape}")

    # Add them (with length matching)
    combined = ups_0_out + noise_res_0_out
    print(
        f"Combined (stage 0 input): range [{combined.min():.4f}, {combined.max():.4f}]"
    )

    # Convert to MLX
    combined_mx = mx.array(combined).transpose(0, 2, 1)  # NLC

    # Run all 3 resblocks and average
    num_kernels = generator.num_kernels  # Should be 3
    xs = None
    for j in range(num_kernels):
        block_idx = 0 * num_kernels + j  # Stage 0
        resblock = getattr(generator, f"resblocks_{block_idx}")
        rb_out = resblock(combined_mx, style_mx)
        if xs is None:
            xs = rb_out
        else:
            xs = xs + rb_out
    result = xs / num_kernels
    mx.eval(result)

    result_ncl = np.array(result.transpose(0, 2, 1))

    # Compare with PT after_resblocks_0 equivalent
    # We need to compute what PT's after_resblocks would be
    # Unfortunately we don't have that trace, but we have individual resblock outputs
    # Let's compute it the same way
    pt_xs = None
    for j in range(3):
        rb_out = internal[f"resblock_{j}_out"]
        if pt_xs is None:
            pt_xs = rb_out
        else:
            pt_xs = pt_xs + rb_out
    pt_result = pt_xs / 3

    corr = np.corrcoef(result_ncl.flatten(), pt_result.flatten())[0, 1]
    print("\nStage 0 combined output:")
    print(
        f"  PT: range [{pt_result.min():.4f}, {pt_result.max():.4f}], std={pt_result.std():.4f}"
    )
    print(
        f"  MLX: range [{result_ncl.min():.4f}, {result_ncl.max():.4f}], std={result_ncl.std():.4f}"
    )
    print(f"  Correlation: {corr:.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
