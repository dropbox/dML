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
Trace single AdaIN step in both PyTorch and MLX with same inputs.
"""

import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as mnn
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    # Load real weights from checkpoint
    from huggingface_hub import hf_hub_download

    ckpt_path = hf_hub_download("hexgrad/Kokoro-82M", "kokoro-v1_0.pth")
    pt_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    dec_state = pt_ckpt["decoder"]

    # Get resblocks.0.adain1.0 weights
    adain_w = dec_state["module.generator.resblocks.0.adain1.0.fc.weight"].numpy()
    adain_b = dec_state["module.generator.resblocks.0.adain1.0.fc.bias"].numpy()
    alpha1 = dec_state["module.generator.resblocks.0.alpha1.0"].numpy()

    print(f"AdaIN weight shape: {adain_w.shape}")  # [512, 128]
    print(f"AdaIN bias shape: {adain_b.shape}")  # [512]
    print(f"Alpha shape: {alpha1.shape}")  # [1, 256, 1]

    # Create fixed test inputs (same for both)
    np.random.seed(42)
    channels = 256
    style_dim = 128
    length = 63

    x_np = np.random.randn(1, length, channels).astype(np.float32)  # NLC
    s_np = np.random.randn(1, style_dim).astype(np.float32)

    print(f"\nInput x shape: {x_np.shape}, range: [{x_np.min():.4f}, {x_np.max():.4f}]")
    print(f"Input s shape: {s_np.shape}, range: [{s_np.min():.4f}, {s_np.max():.4f}]")

    # ===== PyTorch =====
    print("\n=== PyTorch ===")
    x_pt = torch.tensor(x_np.transpose(0, 2, 1))  # NCL: [1, 256, 63]
    s_pt = torch.tensor(s_np)

    # AdaIN fc layer
    pt_fc = nn.Linear(style_dim, channels * 2)
    pt_fc.weight.data = torch.tensor(adain_w)
    pt_fc.bias.data = torch.tensor(adain_b)

    with torch.no_grad():
        # Style projection
        style_out = pt_fc(s_pt)  # [1, 512]
        gamma = style_out[:, :channels]  # [1, 256]
        beta = style_out[:, channels:]  # [1, 256]

        print(
            f"style_out shape: {style_out.shape}, range: [{style_out.min():.4f}, {style_out.max():.4f}]"
        )
        print(
            f"gamma shape: {gamma.shape}, range: [{gamma.min():.4f}, {gamma.max():.4f}]"
        )
        print(f"beta shape: {beta.shape}, range: [{beta.min():.4f}, {beta.max():.4f}]")

        # Instance norm (normalize over L dimension for NCL format)
        # PyTorch InstanceNorm1d with affine=False
        # InstanceNorm1d normalizes over the last dimension (L) for each (N, C) slice
        mean = x_pt.mean(dim=2, keepdim=True)  # [1, 256, 1]
        var = x_pt.var(dim=2, keepdim=True, unbiased=False)  # [1, 256, 1]
        x_normed = (x_pt - mean) / torch.sqrt(var + 1e-5)  # [1, 256, 63]

        print(f"mean shape: {mean.shape}, range: [{mean.min():.4f}, {mean.max():.4f}]")
        print(f"var shape: {var.shape}, range: [{var.min():.4f}, {var.max():.4f}]")
        print(
            f"x_normed shape: {x_normed.shape}, range: [{x_normed.min():.4f}, {x_normed.max():.4f}]"
        )

        # AdaIN: scale by (1 + gamma) and shift by beta
        # gamma/beta are [1, 256], need to expand to [1, 256, 1] for NCL
        gamma_exp = gamma.unsqueeze(-1)  # [1, 256, 1]
        beta_exp = beta.unsqueeze(-1)  # [1, 256, 1]
        x_adain = (1 + gamma_exp) * x_normed + beta_exp

        print(
            f"PT AdaIN output shape: {x_adain.shape}, range: [{x_adain.min():.4f}, {x_adain.max():.4f}]"
        )

        # Snake1D activation
        alpha = torch.tensor(alpha1)  # [1, 256, 1]
        x_snake_pt_out = x_adain + (1 / alpha) * (torch.sin(alpha * x_adain) ** 2)
        print(
            f"PT Snake output shape: {x_snake_pt_out.shape}, range: [{x_snake_pt_out.min():.4f}, {x_snake_pt_out.max():.4f}]"
        )

    # ===== MLX =====
    print("\n=== MLX ===")
    x_mx = mx.array(x_np)  # NLC: [1, 63, 256]
    s_mx = mx.array(s_np)

    # AdaIN fc layer
    mx_fc = mnn.Linear(style_dim, channels * 2)
    mx_fc.weight = mx.array(adain_w)
    mx_fc.bias = mx.array(adain_b)

    # Style projection
    style_out = mx_fc(s_mx)  # [1, 512]
    gamma = style_out[:, :channels]  # [1, 256]
    beta = style_out[:, channels:]  # [1, 256]

    print(
        f"style_out shape: {style_out.shape}, range: [{float(mx.min(style_out)):.4f}, {float(mx.max(style_out)):.4f}]"
    )
    print(
        f"gamma shape: {gamma.shape}, range: [{float(mx.min(gamma)):.4f}, {float(mx.max(gamma)):.4f}]"
    )
    print(
        f"beta shape: {beta.shape}, range: [{float(mx.min(beta)):.4f}, {float(mx.max(beta)):.4f}]"
    )

    # Instance norm (normalize over length=axis 1 for NLC format)
    mean = mx.mean(x_mx, axis=1, keepdims=True)  # [1, 1, 256]
    var = mx.var(x_mx, axis=1, keepdims=True)  # [1, 1, 256]
    x_normed = (x_mx - mean) / mx.sqrt(var + 1e-5)  # [1, 63, 256]

    print(
        f"mean shape: {mean.shape}, range: [{float(mx.min(mean)):.4f}, {float(mx.max(mean)):.4f}]"
    )
    print(
        f"var shape: {var.shape}, range: [{float(mx.min(var)):.4f}, {float(mx.max(var)):.4f}]"
    )
    print(
        f"x_normed shape: {x_normed.shape}, range: [{float(mx.min(x_normed)):.4f}, {float(mx.max(x_normed)):.4f}]"
    )

    # AdaIN: scale by (1 + gamma) and shift by beta
    # gamma/beta are [1, 256], need to expand to [1, 1, 256] for NLC
    gamma_exp = gamma[:, None, :]  # [1, 1, 256]
    beta_exp = beta[:, None, :]  # [1, 1, 256]
    x_adain = (1 + gamma_exp) * x_normed + beta_exp

    print(
        f"MLX AdaIN output shape: {x_adain.shape}, range: [{float(mx.min(x_adain)):.4f}, {float(mx.max(x_adain)):.4f}]"
    )

    # Snake1D activation
    # alpha is [1, 256, 1] (NCL format), need to transpose to [1, 1, 256] for NLC
    alpha_mx = mx.array(alpha1).transpose(0, 2, 1)  # [1, 1, 256]
    x_snake = x_adain + (1 / alpha_mx) * (mx.sin(alpha_mx * x_adain) ** 2)
    print(
        f"MLX Snake output shape: {x_snake.shape}, range: [{float(mx.min(x_snake)):.4f}, {float(mx.max(x_snake)):.4f}]"
    )

    # Compare outputs (transpose MLX to NCL for comparison)
    x_snake_ncl = np.array(x_snake.transpose(0, 2, 1))  # [1, 256, 63]
    x_snake_pt = x_snake_pt_out.numpy()

    diff = np.abs(x_snake_ncl - x_snake_pt)
    print("\n=== Comparison ===")
    print(f"Max diff: {diff.max():.6e}")
    print(f"Mean diff: {diff.mean():.6e}")
    print(
        f"Correlation: {np.corrcoef(x_snake_ncl.flatten(), x_snake_pt.flatten())[0, 1]:.6f}"
    )


if __name__ == "__main__":
    main()
