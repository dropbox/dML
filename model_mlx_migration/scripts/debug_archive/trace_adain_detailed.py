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
Trace AdaIN operation in detail to find exact source of 0.038 error.
"""

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def sample_variance(x: mx.array, axis: int, keepdims: bool = False) -> mx.array:
    """Same as in kokoro_modules.py"""
    mean = mx.mean(x, axis=axis, keepdims=True)
    diff = x - mean
    n = x.shape[axis]
    var = mx.sum(diff * diff, axis=axis, keepdims=keepdims) / (n - 1)  # ddof=1
    return var


def main():
    ref_dir = Path("/tmp/kokoro_ref_decoder")
    ref = np.load(ref_dir / "tensors.npz")

    from tools.pytorch_to_mlx.converters import KokoroConverter
    converter = KokoroConverter()
    mlx_model, _, _ = converter.load_from_hf("hexgrad/Kokoro-82M")
    mlx_model.set_deterministic(True)

    decoder = mlx_model.decoder
    encode_block = decoder.encode

    # Load inputs
    asr_nlc = mx.array(ref["asr_nlc"].astype(np.float32))
    f0_proc = mx.array(ref["F0_proc"].astype(np.float32))
    n_proc = mx.array(ref["N_proc"].astype(np.float32))
    style_128 = mx.array(ref["style_128"].astype(np.float32))

    # Initial concat - this is the input to encode block
    x = mx.concatenate([asr_nlc, f0_proc, n_proc], axis=-1)
    mx.eval(x)

    print("=" * 72)
    print("AdaIN Detailed Trace - Encode Block norm1")
    print("=" * 72)

    print(f"Input x shape: {x.shape}")
    print(f"Style s shape: {style_128.shape}")

    # Get norm1 (AdaIN)
    norm1 = encode_block.norm1
    print(f"\nnorm1 type: {type(norm1)}")
    print(f"norm1.channels: {norm1.channels}")

    # Step 1: fc layer
    h = norm1.fc(style_128)  # [batch, channels * 2]
    mx.eval(h)
    print(f"\nfc output h shape: {h.shape}")
    print(f"  range: [{np.array(h).min():.6f}, {np.array(h).max():.6f}]")

    # Step 2: Split gamma/beta
    gamma, beta = mx.split(h, 2, axis=-1)
    gamma = gamma[:, None, :]  # [batch, 1, channels]
    beta = beta[:, None, :]
    mx.eval(gamma, beta)
    print(f"\ngamma shape: {gamma.shape}, range: [{np.array(gamma).min():.6f}, {np.array(gamma).max():.6f}]")
    print(f"beta shape: {beta.shape}, range: [{np.array(beta).min():.6f}, {np.array(beta).max():.6f}]")

    # Step 3: Instance norm - mean
    mean = mx.mean(x, axis=1, keepdims=True)
    mx.eval(mean)
    print(f"\nmean shape: {mean.shape}")
    print(f"  channel 0 mean: {np.array(mean)[0, 0, 0]:.6f}")

    # Step 4: Instance norm - variance
    var = sample_variance(x, axis=1, keepdims=True)
    mx.eval(var)
    print(f"\nvar shape: {var.shape}")
    print(f"  channel 0 var: {np.array(var)[0, 0, 0]:.6f}")

    # Step 5: Normalize
    x_norm = (x - mean) / mx.sqrt(var + 1e-5)
    mx.eval(x_norm)
    x_norm_np = np.array(x_norm)
    print(f"\nx_norm shape: {x_norm.shape}")
    print(f"  range: [{x_norm_np.min():.6f}, {x_norm_np.max():.6f}]")

    # Step 6: Apply gamma/beta
    out = (1 + gamma) * x_norm + beta
    mx.eval(out)
    out_np = np.array(out)
    print(f"\nAdaIN output shape: {out.shape}")
    print(f"  range: [{out_np.min():.6f}, {out_np.max():.6f}]")

    # Compare with PyTorch if available
    if "encode_norm1_output" in ref:
        pt_norm1 = ref["encode_norm1_output"].astype(np.float32)
        print("\nPyTorch norm1 output available!")
        print(f"  shape: {pt_norm1.shape}")
        print(f"  channel 872 at pos 56: {pt_norm1[0, 56, 872]:.6f}")

        diff = np.abs(out_np - pt_norm1)
        print("\nComparison:")
        print(f"  max diff: {diff.max():.6f}")
        print(f"  mean diff: {diff.mean():.6f}")
    else:
        print("\n(No PyTorch norm1 output available - need to export)")

    # Now trace through full norm1 -> leaky_relu -> conv1
    print("\n" + "=" * 72)
    print("Full path: norm1 -> leaky_relu -> conv1")
    print("=" * 72)

    h = norm1(x, style_128)
    mx.eval(h)
    h_np = np.array(h)
    print(f"After norm1: shape={h.shape}, range=[{h_np.min():.4f}, {h_np.max():.4f}]")

    h = nn.leaky_relu(h, 0.2)
    mx.eval(h)
    h_np = np.array(h)
    print(f"After leaky_relu: shape={h.shape}, range=[{h_np.min():.4f}, {h_np.max():.4f}]")

    h = encode_block.conv1(h)
    mx.eval(h)
    h_np = np.array(h)
    print(f"After conv1: shape={h.shape}, range=[{h_np.min():.4f}, {h_np.max():.4f}]")

    # Compare with PyTorch encode output
    pt_encode = ref["encode_output"].astype(np.float32)
    print(f"\nPyTorch encode_output: shape={pt_encode.shape}")

    # Full encode block
    x_out = encode_block(x, style_128)
    mx.eval(x_out)
    x_out_np = np.array(x_out)

    diff = np.abs(x_out_np - pt_encode)
    print("\nEncode block comparison:")
    print(f"  max diff: {diff.max():.6f}")
    print(f"  mean diff: {diff.mean():.6f}")

    # Find where largest diff is
    flat_diff = diff.reshape(-1)
    top_idx = np.argsort(flat_diff)[-5:][::-1]
    print("\nTop 5 diffs in encode output:")
    for idx in top_idx:
        loc = np.unravel_index(idx, diff.shape)
        print(f"  {loc}: mlx={x_out_np[loc]:.6f}, pt={pt_encode[loc]:.6f}, diff={flat_diff[idx]:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
