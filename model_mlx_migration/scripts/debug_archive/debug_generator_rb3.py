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
Debug Generator Resblock 3 - Compare Python vs C++ step-by-step.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

MODEL_PATH = Path.home() / "model_mlx_migration" / "kokoro_cpp_export"


def load_weights():
    weights = {}
    with safe_open(MODEL_PATH / "weights.safetensors", framework="pt") as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)
    return weights


def print_stats(name, t):
    t_np = t.detach().cpu().numpy().flatten()
    print(f"{name}: [{t_np.min():.4f}, {t_np.max():.4f}], mean={t_np.mean():.4f}")


def adain(x, style, fc_weight, fc_bias):
    """Adaptive Instance Normalization."""
    fc_out = F.linear(style, fc_weight, fc_bias)
    channels = x.shape[-1]
    gamma = fc_out[:, :channels].unsqueeze(1)
    beta = fc_out[:, channels:].unsqueeze(1)

    mean = x.mean(dim=1, keepdim=True)
    var = x.var(dim=1, keepdim=True, unbiased=False)
    x_norm = (x - mean) / torch.sqrt(var + 1e-5)

    return (1 + gamma) * x_norm + beta


def snake1d(x, alpha):
    """Snake1D activation: x + sin²(αx) / α"""
    # x: [batch, time, channels]
    # alpha: [1, channels, 1] -> transpose to [1, 1, channels]
    alpha_t = alpha.permute(0, 2, 1)
    ax = x * alpha_t
    sin_ax = torch.sin(ax)
    return x + sin_ax * sin_ax / alpha_t


def conv1d_dilated(x, weight, bias, dilation):
    """Conv1d with dilation. x: [batch, time, channels]"""
    x_t = x.transpose(1, 2)  # [batch, channels, time]
    kernel = weight.shape[2]
    padding = (kernel - 1) * dilation // 2
    out = F.conv1d(x_t, weight, bias, dilation=dilation, padding=padding)
    return out.transpose(1, 2)


def resblock(x_in, style, weights, block_idx=3, dilations=[1, 3, 5]):
    """Generator resblock with debug output."""
    x = x_in.clone()
    prefix = f"decoder.generator.resblocks_{block_idx}"

    for d in range(3):
        dilation = dilations[d]
        print(f"\n--- d={d} (dilation={dilation}) ---")

        conv1_w = weights[f"{prefix}.convs1_{d}.weight"]
        conv1_b = weights[f"{prefix}.convs1_{d}.bias"]
        conv2_w = weights[f"{prefix}.convs2_{d}.weight"]
        conv2_b = weights[f"{prefix}.convs2_{d}.bias"]
        adain1_fc_w = weights[f"{prefix}.adain1_{d}.fc.weight"]
        adain1_fc_b = weights[f"{prefix}.adain1_{d}.fc.bias"]
        adain2_fc_w = weights[f"{prefix}.adain2_{d}.fc.weight"]
        adain2_fc_b = weights[f"{prefix}.adain2_{d}.fc.bias"]
        alpha1 = weights[f"{prefix}.alpha1_{d}"]
        alpha2 = weights[f"{prefix}.alpha2_{d}"]

        print_stats("input", x)

        # AdaIN -> Snake -> Conv1 -> AdaIN -> Snake -> Conv2 -> Residual
        xt = adain(x, style, adain1_fc_w, adain1_fc_b)
        print_stats("after_adain1", xt)

        xt = snake1d(xt, alpha1)
        print_stats("after_snake1", xt)

        xt = conv1d_dilated(xt, conv1_w, conv1_b, dilation)
        print_stats("after_conv1", xt)

        xt = adain(xt, style, adain2_fc_w, adain2_fc_b)
        print_stats("after_adain2", xt)

        xt = snake1d(xt, alpha2)
        print_stats("after_snake2", xt)

        xt = conv1d_dilated(xt, conv2_w, conv2_b, 1)
        print_stats("after_conv2", xt)

        x = xt + x
        print_stats("after_resid", x)

    return x


def main():
    print("=" * 60)
    print("Generator Resblock 3 Debug")
    print("=" * 60)

    weights = load_weights()

    # Try to load tensors from C++
    try:
        before_rb345 = torch.from_numpy(np.load("/tmp/cpp_before_rb345.npy"))
        style = torch.from_numpy(np.load("/tmp/cpp_generator_style.npy"))
        print("Loaded from C++:")
        print(f"  before_rb345: {before_rb345.shape}")
        print(f"  style: {style.shape}")
    except Exception as e:
        print(f"Error loading C++ tensors: {e}")
        # Fallback to voice file
        with safe_open(
            MODEL_PATH / "voices" / "af_bella.safetensors", framework="pt"
        ) as f:
            voice_embed = f.get_tensor("embedding")
        style = voice_embed[:, :128]
        before_rb345 = torch.randn(1, 7441, 128) * 2.0

    print_stats("before_resblocks_345 input", before_rb345)
    print_stats("style", style)

    print("\n" + "=" * 60)
    print("Running Resblock 3...")
    print("=" * 60)

    with torch.no_grad():
        rb3_out = resblock(before_rb345, style, weights, block_idx=3)

    print("\n" + "=" * 60)
    print_stats("rb3_output", rb3_out)
    print("=" * 60)


if __name__ == "__main__":
    main()
