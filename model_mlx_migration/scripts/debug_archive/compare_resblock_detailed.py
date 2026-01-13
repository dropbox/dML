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
Detailed comparison of single resblock between PyTorch and MLX.
Uses exact same inputs and weights to find divergence point.
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


def main():
    from huggingface_hub import hf_hub_download

    print("=== Loading checkpoint ===")
    ckpt_path = hf_hub_download("hexgrad/Kokoro-82M", "kokoro-v1_0.pth")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    dec_state = ckpt["decoder"]

    # Get resblock 0 weights
    prefix = "module.generator.resblocks.0"

    # Create test inputs
    np.random.seed(42)
    channels = 256
    style_dim = 128
    length = 63

    x_np = np.random.randn(1, channels, length).astype(np.float32) * 0.1  # NCL
    s_np = np.random.randn(1, style_dim).astype(np.float32)

    print(f"Input x shape: {x_np.shape}, range: [{x_np.min():.4f}, {x_np.max():.4f}]")
    print(f"Input s shape: {s_np.shape}, range: [{s_np.min():.4f}, {s_np.max():.4f}]")

    # ===== PyTorch Implementation (matching StyleTTS2 exactly) =====
    print("\n=== PyTorch (StyleTTS2 style) ===")

    x_pt = torch.tensor(x_np)
    s_pt = torch.tensor(s_np)

    # Load weights for first dilation layer (index 0)
    adain1_w = dec_state[f"{prefix}.adain1.0.fc.weight"].numpy()
    adain1_b = dec_state[f"{prefix}.adain1.0.fc.bias"].numpy()
    adain2_w = dec_state[f"{prefix}.adain2.0.fc.weight"].numpy()
    adain2_b = dec_state[f"{prefix}.adain2.0.fc.bias"].numpy()
    alpha1 = dec_state[f"{prefix}.alpha1.0"].numpy()  # [1, channels, 1]
    alpha2 = dec_state[f"{prefix}.alpha2.0"].numpy()
    conv1_wv = dec_state[f"{prefix}.convs1.0.weight_v"].numpy()  # [out, in, k]
    conv1_wg = dec_state[f"{prefix}.convs1.0.weight_g"].numpy()  # [out, 1, 1]
    conv1_b = dec_state[f"{prefix}.convs1.0.bias"].numpy()
    conv2_wv = dec_state[f"{prefix}.convs2.0.weight_v"].numpy()
    conv2_wg = dec_state[f"{prefix}.convs2.0.weight_g"].numpy()
    conv2_b = dec_state[f"{prefix}.convs2.0.bias"].numpy()

    print(
        f"alpha1 shape: {alpha1.shape}, range: [{alpha1.min():.4f}, {alpha1.max():.4f}]"
    )
    print(f"conv1_wv shape: {conv1_wv.shape}, conv1_wg shape: {conv1_wg.shape}")

    with torch.no_grad():
        xt = x_pt

        # === AdaIN1 ===
        adain1_fc = nn.Linear(style_dim, channels * 2)
        adain1_fc.weight.data = torch.tensor(adain1_w)
        adain1_fc.bias.data = torch.tensor(adain1_b)

        style_out = adain1_fc(s_pt)  # [1, 512]
        gamma1 = style_out[:, :channels].unsqueeze(-1)  # [1, C, 1]
        beta1 = style_out[:, channels:].unsqueeze(-1)

        # Instance norm (over L dimension)
        mean1 = xt.mean(dim=2, keepdim=True)
        var1 = xt.var(dim=2, keepdim=True, unbiased=False)
        xt_norm = (xt - mean1) / torch.sqrt(var1 + 1e-5)
        xt = (1 + gamma1) * xt_norm + beta1  # StyleTTS2 uses (1+gamma)

        print(
            f"After AdaIN1: range [{xt.min():.4f}, {xt.max():.4f}], std={xt.std():.4f}"
        )

        # === Snake1D ===
        alpha1_t = torch.tensor(alpha1)
        xt = xt + (1 / alpha1_t) * (torch.sin(alpha1_t * xt) ** 2)

        print(
            f"After Snake1D: range [{xt.min():.4f}, {xt.max():.4f}], std={xt.std():.4f}"
        )

        # === Conv1 with weight norm ===
        # Compute effective weight: w = g * v / ||v||
        wv = torch.tensor(conv1_wv)
        wg = torch.tensor(conv1_wg)
        # Weight norm: w = g * (v / ||v||_2), where norm is over (in_ch, kernel)
        norm = wv.norm(dim=(1, 2), keepdim=True)
        w = wg * wv / norm
        b = torch.tensor(conv1_b)

        # Conv1d with dilation=1, kernel=3
        kernel_size = w.shape[2]
        padding = (kernel_size - 1) // 2  # Same padding for dilation=1
        xt = F.conv1d(xt, w, b, padding=padding, dilation=1)

        print(
            f"After Conv1: range [{xt.min():.4f}, {xt.max():.4f}], std={xt.std():.4f}"
        )

        # === AdaIN2 ===
        adain2_fc = nn.Linear(style_dim, channels * 2)
        adain2_fc.weight.data = torch.tensor(adain2_w)
        adain2_fc.bias.data = torch.tensor(adain2_b)

        style_out = adain2_fc(s_pt)
        gamma2 = style_out[:, :channels].unsqueeze(-1)
        beta2 = style_out[:, channels:].unsqueeze(-1)

        mean2 = xt.mean(dim=2, keepdim=True)
        var2 = xt.var(dim=2, keepdim=True, unbiased=False)
        xt_norm = (xt - mean2) / torch.sqrt(var2 + 1e-5)
        xt = (1 + gamma2) * xt_norm + beta2

        print(
            f"After AdaIN2: range [{xt.min():.4f}, {xt.max():.4f}], std={xt.std():.4f}"
        )

        # === Snake2D ===
        alpha2_t = torch.tensor(alpha2)
        xt = xt + (1 / alpha2_t) * (torch.sin(alpha2_t * xt) ** 2)

        print(
            f"After Snake2D: range [{xt.min():.4f}, {xt.max():.4f}], std={xt.std():.4f}"
        )

        # === Conv2 with weight norm ===
        # Per ISTFTNet: convs2 uses dilation=1
        wv2 = torch.tensor(conv2_wv)
        wg2 = torch.tensor(conv2_wg)
        norm2 = wv2.norm(dim=(1, 2), keepdim=True)
        w2 = wg2 * wv2 / norm2
        b2 = torch.tensor(conv2_b)

        kernel_size2 = w2.shape[2]
        padding2 = (kernel_size2 - 1) // 2
        xt = F.conv1d(xt, w2, b2, padding=padding2, dilation=1)

        print(
            f"After Conv2: range [{xt.min():.4f}, {xt.max():.4f}], std={xt.std():.4f}"
        )

        # === Residual ===
        x_out_pt = xt + x_pt

        print(
            f"\nPT output: range [{x_out_pt.min():.4f}, {x_out_pt.max():.4f}], std={x_out_pt.std():.4f}"
        )

    # ===== MLX Implementation =====
    print("\n=== MLX ===")

    # NLC format: [1, length, channels]
    x_mx = mx.array(x_np.transpose(0, 2, 1))  # NCL -> NLC
    s_mx = mx.array(s_np)

    xt = x_mx

    # === AdaIN1 ===
    adain1_fc_mx = mnn.Linear(style_dim, channels * 2)
    adain1_fc_mx.weight = mx.array(adain1_w)
    adain1_fc_mx.bias = mx.array(adain1_b)

    style_out = adain1_fc_mx(s_mx)  # [1, 512]
    gamma1 = style_out[:, :channels][:, None, :]  # [1, 1, C]
    beta1 = style_out[:, channels:][:, None, :]

    # Instance norm (over L dimension = axis 1 for NLC)
    mean1 = mx.mean(xt, axis=1, keepdims=True)
    var1 = mx.var(xt, axis=1, keepdims=True)
    xt_norm = (xt - mean1) / mx.sqrt(var1 + 1e-5)
    xt = (1 + gamma1) * xt_norm + beta1
    mx.eval(xt)

    print(
        f"After AdaIN1: range [{float(mx.min(xt)):.4f}, {float(mx.max(xt)):.4f}], std={float(mx.std(xt)):.4f}"
    )

    # === Snake1D ===
    # alpha is [1, C, 1] NCL -> [1, 1, C] NLC
    alpha1_mx = mx.array(alpha1).transpose(0, 2, 1)
    xt = xt + (1 / alpha1_mx) * (mx.sin(alpha1_mx * xt) ** 2)
    mx.eval(xt)

    print(
        f"After Snake1D: range [{float(mx.min(xt)):.4f}, {float(mx.max(xt)):.4f}], std={float(mx.std(xt)):.4f}"
    )

    # === Conv1 with weight norm ===
    # MLX Conv1d expects weight [out_ch, kernel, in_ch] for NLC format
    # PyTorch: [out, in, k] -> MLX: [out, k, in]
    wv_mx = mx.array(conv1_wv).transpose(0, 2, 1)  # [out, k, in]
    wg_mx = mx.array(conv1_wg).reshape(-1, 1, 1)  # [out, 1, 1]

    # Weight norm: w = g * (v / ||v||)
    # For MLX NLC format, normalize over (k, in) = axes (1, 2)
    norm_mx = mx.sqrt(mx.sum(wv_mx**2, axis=(1, 2), keepdims=True) + 1e-12)
    w_mx = wg_mx * wv_mx / norm_mx
    b_mx = mx.array(conv1_b)

    # Create Conv1d - for dilation=1, kernel=3, padding=1
    kernel_size = w_mx.shape[1]
    padding = (kernel_size - 1) // 2

    # Manual convolution in NLC
    mx.pad(xt, [(0, 0), (padding, padding), (0, 0)])
    # Use unfold-style conv
    # Actually, let's use MLX conv1d properly
    conv1_mx = mnn.Conv1d(channels, channels, kernel_size, padding=padding)
    conv1_mx.weight = w_mx
    conv1_mx.bias = b_mx
    xt = conv1_mx(xt)
    mx.eval(xt)

    print(
        f"After Conv1: range [{float(mx.min(xt)):.4f}, {float(mx.max(xt)):.4f}], std={float(mx.std(xt)):.4f}"
    )

    # === AdaIN2 ===
    adain2_fc_mx = mnn.Linear(style_dim, channels * 2)
    adain2_fc_mx.weight = mx.array(adain2_w)
    adain2_fc_mx.bias = mx.array(adain2_b)

    style_out = adain2_fc_mx(s_mx)
    gamma2 = style_out[:, :channels][:, None, :]
    beta2 = style_out[:, channels:][:, None, :]

    mean2 = mx.mean(xt, axis=1, keepdims=True)
    var2 = mx.var(xt, axis=1, keepdims=True)
    xt_norm = (xt - mean2) / mx.sqrt(var2 + 1e-5)
    xt = (1 + gamma2) * xt_norm + beta2
    mx.eval(xt)

    print(
        f"After AdaIN2: range [{float(mx.min(xt)):.4f}, {float(mx.max(xt)):.4f}], std={float(mx.std(xt)):.4f}"
    )

    # === Snake2D ===
    alpha2_mx = mx.array(alpha2).transpose(0, 2, 1)
    xt = xt + (1 / alpha2_mx) * (mx.sin(alpha2_mx * xt) ** 2)
    mx.eval(xt)

    print(
        f"After Snake2D: range [{float(mx.min(xt)):.4f}, {float(mx.max(xt)):.4f}], std={float(mx.std(xt)):.4f}"
    )

    # === Conv2 with weight norm ===
    wv2_mx = mx.array(conv2_wv).transpose(0, 2, 1)
    wg2_mx = mx.array(conv2_wg).reshape(-1, 1, 1)
    norm2_mx = mx.sqrt(mx.sum(wv2_mx**2, axis=(1, 2), keepdims=True) + 1e-12)
    w2_mx = wg2_mx * wv2_mx / norm2_mx
    b2_mx = mx.array(conv2_b)

    kernel_size2 = w2_mx.shape[1]
    padding2 = (kernel_size2 - 1) // 2

    conv2_mx = mnn.Conv1d(channels, channels, kernel_size2, padding=padding2)
    conv2_mx.weight = w2_mx
    conv2_mx.bias = b2_mx
    xt = conv2_mx(xt)
    mx.eval(xt)

    print(
        f"After Conv2: range [{float(mx.min(xt)):.4f}, {float(mx.max(xt)):.4f}], std={float(mx.std(xt)):.4f}"
    )

    # === Residual ===
    x_out_mx = xt + x_mx
    mx.eval(x_out_mx)

    print(
        f"\nMLX output: range [{float(mx.min(x_out_mx)):.4f}, {float(mx.max(x_out_mx)):.4f}], std={float(mx.std(x_out_mx)):.4f}"
    )

    # === Compare ===
    # Convert MLX NLC to NCL for comparison
    x_out_mx_ncl = np.array(x_out_mx.transpose(0, 2, 1))
    x_out_pt_np = x_out_pt.numpy()

    diff = np.abs(x_out_mx_ncl - x_out_pt_np)
    print("\n=== Comparison ===")
    print(f"Max diff: {diff.max():.6e}")
    print(f"Mean diff: {diff.mean():.6e}")
    print(
        f"Correlation: {np.corrcoef(x_out_mx_ncl.flatten(), x_out_pt_np.flatten())[0, 1]:.6f}"
    )


if __name__ == "__main__":
    main()
