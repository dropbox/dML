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
Trace through encode block step-by-step to find exact error source.
"""

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def main():
    ref_dir = Path("/tmp/kokoro_ref_decoder")
    ref = np.load(ref_dir / "tensors.npz")

    from tools.pytorch_to_mlx.converters import KokoroConverter
    converter = KokoroConverter()
    mlx_model, _, _ = converter.load_from_hf("hexgrad/Kokoro-82M")
    mlx_model.set_deterministic(True)

    decoder = mlx_model.decoder
    encode = decoder.encode

    # Get inputs
    x_input = mx.array(ref["decoder_input_concat"].astype(np.float32))
    style = mx.array(ref["style_128"].astype(np.float32))

    print("=" * 72)
    print("Encode Block Step-by-Step Trace")
    print("=" * 72)
    print(f"Input x: {x_input.shape}")
    print(f"Style: {style.shape}")

    # Verify input matches
    x_np = np.array(x_input)
    pt_input = ref["decoder_input_concat"]
    input_diff = np.abs(x_np - pt_input).max()
    print(f"Input diff from PyTorch: {input_diff:.6f}")

    # Trace through encode block
    # AdainResBlk1d structure:
    # - norm1 -> leaky_relu(0.2) -> [pool if upsample] -> conv1
    # - norm2 -> leaky_relu(0.2) -> conv2
    # - shortcut: [upsample] -> conv1x1
    # - output = (h + skip) * rsqrt(2)

    print("\n--- Step-by-step trace ---")

    # Check encode block config
    print(f"encode.upsample: {encode.upsample}")
    print(f"encode.downsample: {encode.downsample}")
    print(f"encode.conv1x1 is None: {encode.conv1x1 is None}")

    # Shortcut path
    x_short = x_input
    if encode.upsample:
        x_short = mx.repeat(x_input, 2, axis=1)

    if encode.conv1x1 is not None:
        skip = encode.conv1x1(x_short)
    else:
        skip = x_short
    mx.eval(skip)
    print("\nShortcut:")
    print(f"  skip shape: {skip.shape}")
    print(f"  skip range: [{np.array(skip).min():.4f}, {np.array(skip).max():.4f}]")

    # Step 1: norm1
    h = encode.norm1(x_input, style)
    mx.eval(h)
    h_np = np.array(h)
    print("\nAfter norm1:")
    print(f"  h shape: {h.shape}")
    print(f"  h range: [{h_np.min():.4f}, {h_np.max():.4f}]")
    print(f"  h mean: {h_np.mean():.4f}, std: {h_np.std():.4f}")

    # Step 2: leaky_relu
    h = nn.leaky_relu(h, 0.2)
    mx.eval(h)
    h_np = np.array(h)
    print("\nAfter leaky_relu(0.2):")
    print(f"  h range: [{h_np.min():.4f}, {h_np.max():.4f}]")

    # Step 3: conv1
    h = encode.conv1(h)
    mx.eval(h)
    h_np = np.array(h)
    print("\nAfter conv1:")
    print(f"  h shape: {h.shape}")
    print(f"  h range: [{h_np.min():.4f}, {h_np.max():.4f}]")

    # Step 4: norm2
    h = encode.norm2(h, style)
    mx.eval(h)
    h_np = np.array(h)
    print("\nAfter norm2:")
    print(f"  h range: [{h_np.min():.4f}, {h_np.max():.4f}]")

    # Step 5: leaky_relu
    h = nn.leaky_relu(h, 0.2)
    mx.eval(h)
    h_np = np.array(h)
    print("\nAfter leaky_relu(0.2):")
    print(f"  h range: [{h_np.min():.4f}, {h_np.max():.4f}]")

    # Step 6: conv2
    h = encode.conv2(h)
    mx.eval(h)
    h_np = np.array(h)
    print("\nAfter conv2:")
    print(f"  h shape: {h.shape}")
    print(f"  h range: [{h_np.min():.4f}, {h_np.max():.4f}]")

    # Step 7: Residual
    out = (h + skip) * (2**-0.5)
    mx.eval(out)
    out_np = np.array(out)
    print("\nFinal output (h + skip) * rsqrt(2):")
    print(f"  out shape: {out.shape}")
    print(f"  out range: [{out_np.min():.4f}, {out_np.max():.4f}]")

    # Compare with PyTorch
    pt_encode = ref["encode_output"]
    diff = np.abs(out_np - pt_encode)
    print("\nComparison with PyTorch encode_output:")
    print(f"  Max diff: {diff.max():.6f}")
    print(f"  Mean diff: {diff.mean():.6f}")

    # Find where largest diffs are
    flat_diff = diff.reshape(-1)
    top_idx = np.argsort(flat_diff)[-5:][::-1]
    print("\nTop 5 differences:")
    for idx in top_idx:
        loc = np.unravel_index(idx, diff.shape)
        print(f"  {loc}: mlx={out_np[loc]:.6f}, pt={pt_encode[loc]:.6f}, diff={flat_diff[idx]:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
