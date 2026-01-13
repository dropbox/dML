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
Verify weight normalization convolution matches PyTorch.
"""

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.pytorch_to_mlx.converters import KokoroConverter


def main():
    # Load PyTorch checkpoint
    from huggingface_hub import hf_hub_download

    ckpt_path = hf_hub_download("hexgrad/Kokoro-82M", "kokoro-v1_0.pth")
    pt_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    dec_state = pt_ckpt["decoder"]

    # Load MLX model
    print("=== Loading MLX Model ===")
    converter = KokoroConverter()
    model, config, _ = converter.load_from_hf()
    mx.eval(model)

    generator = model.decoder.generator

    # Get a specific conv weight - resblocks_0.convs1_0
    pt_wg = dec_state[
        "module.generator.resblocks.0.convs1.0.weight_g"
    ].numpy()  # [256, 1, 1]
    pt_wv = dec_state[
        "module.generator.resblocks.0.convs1.0.weight_v"
    ].numpy()  # [256, 256, 7]
    pt_bias = dec_state["module.generator.resblocks.0.convs1.0.bias"].numpy()  # [256]

    mlx_conv = generator.resblocks_0.convs1_0
    mlx_wg = np.array(mlx_conv.weight_g)
    mlx_wv = np.array(mlx_conv.weight_v)
    np.array(mlx_conv.bias)

    print(f"PT weight_g: {pt_wg.shape}, range [{pt_wg.min():.4f}, {pt_wg.max():.4f}]")
    print(
        f"MLX weight_g: {mlx_wg.shape}, range [{mlx_wg.min():.4f}, {mlx_wg.max():.4f}]"
    )
    print(f"weight_g diff: {np.abs(pt_wg - mlx_wg).max():.6e}")

    print(f"\nPT weight_v: {pt_wv.shape}, range [{pt_wv.min():.4f}, {pt_wv.max():.4f}]")
    print(
        f"MLX weight_v: {mlx_wv.shape}, range [{mlx_wv.min():.4f}, {mlx_wv.max():.4f}]"
    )
    print(f"weight_v diff: {np.abs(pt_wv - mlx_wv).max():.6e}")

    # Compute the actual weight from weight_g and weight_v
    # PyTorch weight_norm: w = g * v / ||v||
    # v shape: [out, in, kernel], g shape: [out, 1, 1]
    v_norm = np.sqrt(np.sum(pt_wv**2, axis=(1, 2), keepdims=True))  # [256, 1, 1]
    pt_weight = pt_wg * pt_wv / v_norm

    print(
        f"\nPT effective weight: {pt_weight.shape}, range [{pt_weight.min():.4f}, {pt_weight.max():.4f}]"
    )

    # Test convolution output with same input
    print("\n=== Testing Convolution ===")

    # Create test input
    np.random.seed(42)
    x_np = np.random.randn(1, 1260, 256).astype(np.float32)  # NLC

    # PyTorch conv
    pt_conv = weight_norm(nn.Conv1d(256, 256, 7, padding=3, dilation=1))
    pt_conv.weight_g.data = torch.tensor(pt_wg)
    pt_conv.weight_v.data = torch.tensor(pt_wv)
    pt_conv.bias.data = torch.tensor(pt_bias)

    x_pt = torch.tensor(x_np.transpose(0, 2, 1))  # NCL
    with torch.no_grad():
        out_pt = pt_conv(x_pt)
    print(
        f"PT conv output: {out_pt.shape}, range [{out_pt.min():.4f}, {out_pt.max():.4f}]"
    )

    # MLX conv
    x_mx = mx.array(x_np)
    out_mx = mlx_conv(x_mx)
    mx.eval(out_mx)
    print(
        f"MLX conv output: {out_mx.shape}, range [{float(mx.min(out_mx)):.4f}, {float(mx.max(out_mx)):.4f}]"
    )

    # Compare
    out_mx_ncl = np.array(out_mx.transpose(0, 2, 1))
    out_pt_np = out_pt.numpy()

    diff = np.abs(out_mx_ncl - out_pt_np)
    print(f"\nMax diff: {diff.max():.6e}")
    print(f"Mean diff: {diff.mean():.6e}")
    corr = np.corrcoef(out_mx_ncl.flatten(), out_pt_np.flatten())[0, 1]
    print(f"Correlation: {corr:.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
