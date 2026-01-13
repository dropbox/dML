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
Compare MLX vs PyTorch encode block step by step.
"""

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def compare(name, mlx_val, pt_val, format_ncl=False):
    """Compare and print differences."""
    if format_ncl:
        # PyTorch stores NCL, MLX stores NLC, convert PyTorch to NLC
        pt_val = pt_val.transpose(0, 2, 1)

    # Always convert to numpy first
    if hasattr(mlx_val, 'tolist'):
        mlx_np = np.array(mlx_val)
    else:
        mlx_np = mlx_val

    diff = np.abs(mlx_np - pt_val)
    max_diff = diff.max()
    mean_diff = diff.mean()

    status = "PASS" if max_diff < 0.001 else "FAIL"
    print(f"{name:<25} max={max_diff:.6f} mean={mean_diff:.6f} {status}")

    if max_diff > 0.001:
        flat_diff = diff.reshape(-1)
        top_idx = np.argsort(flat_diff)[-3:][::-1]
        for idx in top_idx:
            loc = np.unravel_index(idx, diff.shape)
            print(f"    {loc}: mlx={float(mlx_np[loc]):.6f}, pt={float(pt_val[loc]):.6f}, diff={float(flat_diff[idx]):.6f}")

    return max_diff


def sample_variance(x: mx.array, axis: int, keepdims: bool = False) -> mx.array:
    """Same as in kokoro_modules.py"""
    mean = mx.mean(x, axis=axis, keepdims=True)
    diff = x - mean
    n = x.shape[axis]
    var = mx.sum(diff * diff, axis=axis, keepdims=keepdims) / (n - 1)
    return var


def main():
    ref_dir = Path("/tmp/kokoro_ref_decoder")
    ref = np.load(ref_dir / "tensors.npz")
    encode_ref = np.load(ref_dir / "encode_intermediates.npz")

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

    x = mx.concatenate([asr_nlc, f0_proc, n_proc], axis=-1)
    mx.eval(x)

    print("=" * 72)
    print("Encode Block Step-by-Step Comparison")
    print("=" * 72)

    # Input verification
    compare("encode_input", x, encode_ref["encode_input"])

    # === SHORTCUT PATH ===
    print("\n--- Shortcut Path ---")

    if encode_block.conv1x1 is not None:
        skip = encode_block.conv1x1(x)
        mx.eval(skip)
        compare("skip (conv1x1)", skip, encode_ref["encode_skip_ncl"], format_ncl=True)
    else:
        skip = x

    # === RESIDUAL PATH ===
    print("\n--- Residual Path ---")

    # Get norm1
    norm1 = encode_block.norm1

    # norm1 fc
    h_fc = norm1.fc(style_128)
    mx.eval(h_fc)
    print(f"MLX norm1.fc output range: [{np.array(h_fc).min():.4f}, {np.array(h_fc).max():.4f}]")

    # gamma/beta
    gamma, beta = mx.split(h_fc, 2, axis=-1)
    gamma_exp = gamma[:, None, :]
    beta_exp = beta[:, None, :]
    mx.eval(gamma_exp, beta_exp)

    # Compare gamma/beta
    pt_gamma = encode_ref["norm1_gamma_ncl"]  # [batch, channels]
    pt_beta = encode_ref["norm1_beta_ncl"]
    compare("norm1 gamma", gamma, pt_gamma)
    compare("norm1 beta", beta, pt_beta)

    # Instance norm
    mean = mx.mean(x, axis=1, keepdims=True)
    var = sample_variance(x, axis=1, keepdims=True)
    mx.eval(mean, var)

    # PyTorch mean/var are in NCL format, need to compare correctly
    pt_mean = encode_ref["norm1_mean_ncl"]  # [batch, channels, 1]
    pt_var = encode_ref["norm1_var_ncl"]

    # Convert PT to NLC
    pt_mean_nlc = pt_mean.transpose(0, 2, 1)  # [batch, 1, channels]
    pt_var_nlc = pt_var.transpose(0, 2, 1)

    compare("norm1 mean", mean, pt_mean_nlc)
    compare("norm1 var", var, pt_var_nlc)

    # x_norm
    x_norm = (x - mean) / mx.sqrt(var + 1e-5)
    mx.eval(x_norm)
    compare("norm1 x_norm", x_norm, encode_ref["norm1_x_norm_ncl"], format_ncl=True)

    # Apply scale/shift
    norm1_out = (1 + gamma_exp) * x_norm + beta_exp
    mx.eval(norm1_out)
    compare("norm1 out (manual)", norm1_out, encode_ref["norm1_out_manual_ncl"], format_ncl=True)

    # Full norm1 call
    h = norm1(x, style_128)
    mx.eval(h)
    compare("norm1 out (actual)", h, encode_ref["encode_norm1_out_ncl"], format_ncl=True)

    # leaky_relu
    h = nn.leaky_relu(h, 0.2)
    mx.eval(h)
    compare("actv1 out", h, encode_ref["encode_actv1_out_ncl"], format_ncl=True)

    # conv1
    h = encode_block.conv1(h)
    mx.eval(h)
    compare("conv1 out", h, encode_ref["encode_conv1_out_ncl"], format_ncl=True)

    # norm2
    h = encode_block.norm2(h, style_128)
    mx.eval(h)
    compare("norm2 out", h, encode_ref["encode_norm2_out_ncl"], format_ncl=True)

    # actv2
    h = nn.leaky_relu(h, 0.2)
    mx.eval(h)
    compare("actv2 out", h, encode_ref["encode_actv2_out_ncl"], format_ncl=True)

    # conv2
    h = encode_block.conv2(h)
    mx.eval(h)
    compare("conv2 out", h, encode_ref["encode_conv2_out_ncl"], format_ncl=True)

    # final combine
    out = (h + skip) * (2**-0.5)
    mx.eval(out)
    compare("encode out", out, encode_ref["encode_output_nlc"])

    # Compare with final reference
    print("\n--- Final Comparison ---")
    full_out = encode_block(x, style_128)
    mx.eval(full_out)
    compare("full encode", full_out, ref["encode_output"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
