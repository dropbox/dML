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
Trace through each iteration of AdaINResBlock1dStyled separately.
Find exactly where magnitude diverges between PyTorch and MLX.
"""

import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as mnn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))


def pytorch_resblock_single_iter(x_pt, s_pt, weights, i):
    """Run a single iteration of PyTorch AdaINResBlock1."""
    channels = x_pt.shape[1]

    # Get weights for iteration i
    adain1_w = weights[f"adain1.{i}.fc.weight"]
    adain1_b = weights[f"adain1.{i}.fc.bias"]
    adain2_w = weights[f"adain2.{i}.fc.weight"]
    adain2_b = weights[f"adain2.{i}.fc.bias"]
    alpha1 = weights[f"alpha1.{i}"]
    alpha2 = weights[f"alpha2.{i}"]
    conv1_wv = weights[f"convs1.{i}.weight_v"]
    conv1_wg = weights[f"convs1.{i}.weight_g"]
    conv1_b = weights[f"convs1.{i}.bias"]
    conv2_wv = weights[f"convs2.{i}.weight_v"]
    conv2_wg = weights[f"convs2.{i}.weight_g"]
    conv2_b = weights[f"convs2.{i}.bias"]

    dilations = [1, 3, 5]
    dilation = dilations[i]

    with torch.no_grad():
        xt = x_pt

        # === AdaIN1 ===
        adain1_fc = nn.Linear(s_pt.shape[1], channels * 2)
        adain1_fc.weight.data = adain1_w.clone()
        adain1_fc.bias.data = adain1_b.clone()

        style_out = adain1_fc(s_pt)
        gamma = style_out[:, :channels].unsqueeze(-1)
        beta = style_out[:, channels:].unsqueeze(-1)

        mean = xt.mean(dim=2, keepdim=True)
        var = xt.var(dim=2, keepdim=True, unbiased=False)
        xt_norm = (xt - mean) / torch.sqrt(var + 1e-5)
        xt = (1 + gamma) * xt_norm + beta

        # === Snake1D ===
        xt = xt + (1 / alpha1) * (torch.sin(alpha1 * xt) ** 2)

        # === Conv1 with weight norm ===
        norm = conv1_wv.norm(dim=(1, 2), keepdim=True)
        w = conv1_wg * conv1_wv / norm

        kernel_size = w.shape[2]
        padding = (kernel_size - 1) * dilation // 2
        xt = F.conv1d(xt, w, conv1_b, padding=padding, dilation=dilation)

        # === AdaIN2 ===
        adain2_fc = nn.Linear(s_pt.shape[1], channels * 2)
        adain2_fc.weight.data = adain2_w.clone()
        adain2_fc.bias.data = adain2_b.clone()

        style_out = adain2_fc(s_pt)
        gamma = style_out[:, :channels].unsqueeze(-1)
        beta = style_out[:, channels:].unsqueeze(-1)

        mean = xt.mean(dim=2, keepdim=True)
        var = xt.var(dim=2, keepdim=True, unbiased=False)
        xt_norm = (xt - mean) / torch.sqrt(var + 1e-5)
        xt = (1 + gamma) * xt_norm + beta

        # === Snake2D ===
        xt = xt + (1 / alpha2) * (torch.sin(alpha2 * xt) ** 2)

        # === Conv2 with weight norm (dilation=1 for convs2) ===
        norm2 = conv2_wv.norm(dim=(1, 2), keepdim=True)
        w2 = conv2_wg * conv2_wv / norm2

        kernel_size2 = w2.shape[2]
        padding2 = (kernel_size2 - 1) // 2  # convs2 uses dilation=1
        xt = F.conv1d(xt, w2, conv2_b, padding=padding2, dilation=1)

        # Residual
        out = xt + x_pt

    return out


def mlx_resblock_single_iter(x_mx, s_mx, weights, i, channels):
    """Run a single iteration of MLX AdaINResBlock1dStyled."""
    style_dim = s_mx.shape[1]

    # Get weights for iteration i (convert from PyTorch tensors)
    adain1_w = mx.array(weights[f"adain1.{i}.fc.weight"].numpy())
    adain1_b = mx.array(weights[f"adain1.{i}.fc.bias"].numpy())
    adain2_w = mx.array(weights[f"adain2.{i}.fc.weight"].numpy())
    adain2_b = mx.array(weights[f"adain2.{i}.fc.bias"].numpy())
    alpha1 = mx.array(weights[f"alpha1.{i}"].numpy())  # [1, C, 1]
    alpha2 = mx.array(weights[f"alpha2.{i}"].numpy())
    conv1_wv = mx.array(weights[f"convs1.{i}.weight_v"].numpy())  # [out, in, k]
    conv1_wg = mx.array(weights[f"convs1.{i}.weight_g"].numpy())  # [out, 1, 1]
    conv1_b = mx.array(weights[f"convs1.{i}.bias"].numpy())
    conv2_wv = mx.array(weights[f"convs2.{i}.weight_v"].numpy())
    conv2_wg = mx.array(weights[f"convs2.{i}.weight_g"].numpy())
    conv2_b = mx.array(weights[f"convs2.{i}.bias"].numpy())

    dilations = [1, 3, 5]
    dilation = dilations[i]

    # x_mx is NLC
    xt = x_mx

    # === AdaIN1 ===
    adain1_fc = mnn.Linear(style_dim, channels * 2)
    adain1_fc.weight = adain1_w
    adain1_fc.bias = adain1_b

    style_out = adain1_fc(s_mx)
    gamma = style_out[:, :channels][:, None, :]  # [batch, 1, channels]
    beta = style_out[:, channels:][:, None, :]

    mean = mx.mean(xt, axis=1, keepdims=True)
    var = mx.var(xt, axis=1, keepdims=True)
    xt_norm = (xt - mean) / mx.sqrt(var + 1e-5)
    xt = (1 + gamma) * xt_norm + beta

    # === Snake1D ===
    alpha1_nlc = alpha1.transpose(0, 2, 1)  # [1, 1, channels]
    xt = xt + (1 / alpha1_nlc) * (mx.sin(alpha1_nlc * xt) ** 2)

    # === Conv1 with weight norm ===
    # MLX format: [out, k, in]
    wv_mx = conv1_wv.transpose(0, 2, 1)
    wg_mx = conv1_wg.reshape(-1, 1, 1)
    norm_mx = mx.sqrt(mx.sum(wv_mx**2, axis=(1, 2), keepdims=True) + 1e-12)
    w_mx = wg_mx * wv_mx / norm_mx

    kernel_size = w_mx.shape[1]
    padding = (kernel_size - 1) * dilation // 2

    conv1 = mnn.Conv1d(
        channels, channels, kernel_size, padding=padding, dilation=dilation
    )
    conv1.weight = w_mx
    conv1.bias = conv1_b
    xt = conv1(xt)

    # === AdaIN2 ===
    adain2_fc = mnn.Linear(style_dim, channels * 2)
    adain2_fc.weight = adain2_w
    adain2_fc.bias = adain2_b

    style_out = adain2_fc(s_mx)
    gamma = style_out[:, :channels][:, None, :]
    beta = style_out[:, channels:][:, None, :]

    mean = mx.mean(xt, axis=1, keepdims=True)
    var = mx.var(xt, axis=1, keepdims=True)
    xt_norm = (xt - mean) / mx.sqrt(var + 1e-5)
    xt = (1 + gamma) * xt_norm + beta

    # === Snake2D ===
    alpha2_nlc = alpha2.transpose(0, 2, 1)
    xt = xt + (1 / alpha2_nlc) * (mx.sin(alpha2_nlc * xt) ** 2)

    # === Conv2 with weight norm (dilation=1) ===
    wv2_mx = conv2_wv.transpose(0, 2, 1)
    wg2_mx = conv2_wg.reshape(-1, 1, 1)
    norm2_mx = mx.sqrt(mx.sum(wv2_mx**2, axis=(1, 2), keepdims=True) + 1e-12)
    w2_mx = wg2_mx * wv2_mx / norm2_mx

    kernel_size2 = w2_mx.shape[1]
    padding2 = (kernel_size2 - 1) // 2

    conv2 = mnn.Conv1d(channels, channels, kernel_size2, padding=padding2, dilation=1)
    conv2.weight = w2_mx
    conv2.bias = conv2_b
    xt = conv2(xt)

    # Residual
    out = xt + x_mx
    mx.eval(out)

    return out


def main():
    from huggingface_hub import hf_hub_download

    print("=== Loading checkpoint ===")
    ckpt_path = hf_hub_download("hexgrad/Kokoro-82M", "kokoro-v1_0.pth")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    dec_state = ckpt["decoder"]

    # Extract resblock 0 weights
    prefix = "module.generator.resblocks.0"
    weights = {}
    for key in dec_state:
        if key.startswith(prefix):
            short_key = key[len(prefix) + 1 :]  # Remove prefix + dot
            weights[short_key] = dec_state[key]

    print(f"Loaded {len(weights)} weight keys")

    # Create test input matching generator dimensions
    np.random.seed(42)
    channels = 256
    style_dim = 128
    length = 126  # Match generator stage 0

    x_np = np.random.randn(1, channels, length).astype(np.float32) * 0.1
    s_np = np.random.randn(1, style_dim).astype(np.float32)

    x_pt = torch.tensor(x_np)
    s_pt = torch.tensor(s_np)
    x_mx = mx.array(x_np.transpose(0, 2, 1))  # NLC
    s_mx = mx.array(s_np)

    print(f"\nInput: {x_np.shape}, range [{x_np.min():.4f}, {x_np.max():.4f}]")

    # Run each iteration separately
    print("\n=== Iteration-by-Iteration Comparison ===")

    x_pt_curr = x_pt.clone()
    x_mx_curr = mx.array(x_mx)

    for i in range(3):
        print(f"\n--- Iteration {i} (dilation={[1, 3, 5][i]}) ---")

        # Before this iteration
        print(f"Input to iter {i}:")
        print(
            f"  PT: range [{x_pt_curr.min():.4f}, {x_pt_curr.max():.4f}], std={x_pt_curr.std():.4f}"
        )
        x_mx_ncl = np.array(x_mx_curr.transpose(0, 2, 1))
        print(
            f"  MLX: range [{x_mx_ncl.min():.4f}, {x_mx_ncl.max():.4f}], std={x_mx_ncl.std():.4f}"
        )
        corr = np.corrcoef(x_pt_curr.numpy().flatten(), x_mx_ncl.flatten())[0, 1]
        print(f"  Correlation: {corr:.6f}")

        # Run single iteration
        x_pt_curr = pytorch_resblock_single_iter(x_pt_curr, s_pt, weights, i)
        x_mx_curr = mlx_resblock_single_iter(x_mx_curr, s_mx, weights, i, channels)

        # After this iteration
        print(f"Output from iter {i}:")
        print(
            f"  PT: range [{x_pt_curr.min():.4f}, {x_pt_curr.max():.4f}], std={x_pt_curr.std():.4f}"
        )
        x_mx_ncl = np.array(x_mx_curr.transpose(0, 2, 1))
        print(
            f"  MLX: range [{x_mx_ncl.min():.4f}, {x_mx_ncl.max():.4f}], std={x_mx_ncl.std():.4f}"
        )
        corr = np.corrcoef(x_pt_curr.numpy().flatten(), x_mx_ncl.flatten())[0, 1]
        print(f"  Correlation: {corr:.6f}")
        diff = np.abs(x_pt_curr.numpy() - x_mx_ncl)
        print(f"  Max diff: {diff.max():.6e}")

    print("\n=== Final Comparison ===")
    x_pt_final = x_pt_curr.numpy()
    x_mx_final = np.array(x_mx_curr.transpose(0, 2, 1))

    print(f"PT final: std={x_pt_final.std():.4f}")
    print(f"MLX final: std={x_mx_final.std():.4f}")
    print(f"Ratio: {x_pt_final.std() / x_mx_final.std():.4f}")
    print(
        f"Final correlation: {np.corrcoef(x_pt_final.flatten(), x_mx_final.flatten())[0, 1]:.6f}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
