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
Trace through a full resblock (all 3 dilations) step by step.
Compare PyTorch vs MLX at each iteration.
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

    # Load traces
    internal = np.load("/tmp/kokoro_ref/generator_internal_traces.npz")
    ref = np.load("/tmp/kokoro_ref/tensors.npz")

    # Get resblock 0 input from PT trace
    pt_input_ncl = internal["resblock_0_in"]  # NCL [1, 256, 1260]
    pt_output_ncl = internal["resblock_0_out"]
    style = ref["style_128"]

    print(f"Input shape: {pt_input_ncl.shape}")
    print(f"Input range: [{pt_input_ncl.min():.4f}, {pt_input_ncl.max():.4f}]")
    print(
        f"Expected output range: [{pt_output_ncl.min():.4f}, {pt_output_ncl.max():.4f}]"
    )

    # Load checkpoint
    ckpt_path = hf_hub_download("hexgrad/Kokoro-82M", "kokoro-v1_0.pth")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    dec_state = ckpt["decoder"]

    prefix = "module.generator.resblocks.0"
    channels = 256
    style_dim = 128
    dilations = [1, 3, 5]
    kernel_size = 3

    # ===== PyTorch =====
    print("\n=== PyTorch Full Resblock ===")
    x_pt = torch.tensor(pt_input_ncl)
    s_pt = torch.tensor(style)

    with torch.no_grad():
        for dil_idx, dilation in enumerate(dilations):
            print(f"\n--- Dilation {dilation} (idx {dil_idx}) ---")

            # Load weights
            adain1_w = dec_state[f"{prefix}.adain1.{dil_idx}.fc.weight"].numpy()
            adain1_b = dec_state[f"{prefix}.adain1.{dil_idx}.fc.bias"].numpy()
            adain2_w = dec_state[f"{prefix}.adain2.{dil_idx}.fc.weight"].numpy()
            adain2_b = dec_state[f"{prefix}.adain2.{dil_idx}.fc.bias"].numpy()
            alpha1 = dec_state[f"{prefix}.alpha1.{dil_idx}"].numpy()
            alpha2 = dec_state[f"{prefix}.alpha2.{dil_idx}"].numpy()
            conv1_wv = dec_state[f"{prefix}.convs1.{dil_idx}.weight_v"].numpy()
            conv1_wg = dec_state[f"{prefix}.convs1.{dil_idx}.weight_g"].numpy()
            conv1_b = dec_state[f"{prefix}.convs1.{dil_idx}.bias"].numpy()
            conv2_wv = dec_state[f"{prefix}.convs2.{dil_idx}.weight_v"].numpy()
            conv2_wg = dec_state[f"{prefix}.convs2.{dil_idx}.weight_g"].numpy()
            conv2_b = dec_state[f"{prefix}.convs2.{dil_idx}.bias"].numpy()

            xt = x_pt

            # AdaIN1
            adain1_fc = nn.Linear(style_dim, channels * 2)
            adain1_fc.weight.data = torch.tensor(adain1_w)
            adain1_fc.bias.data = torch.tensor(adain1_b)

            style_out = adain1_fc(s_pt)
            gamma1 = style_out[:, :channels].unsqueeze(-1)
            beta1 = style_out[:, channels:].unsqueeze(-1)

            mean1 = xt.mean(dim=2, keepdim=True)
            var1 = xt.var(dim=2, keepdim=True, unbiased=False)
            xt = (xt - mean1) / torch.sqrt(var1 + 1e-5)
            xt = (1 + gamma1) * xt + beta1

            print(
                f"After AdaIN1: range [{xt.min():.4f}, {xt.max():.4f}], std={xt.std():.4f}"
            )

            # Snake1D
            alpha1_t = torch.tensor(alpha1)
            xt = xt + (1 / alpha1_t) * (torch.sin(alpha1_t * xt) ** 2)

            print(
                f"After Snake1D: range [{xt.min():.4f}, {xt.max():.4f}], std={xt.std():.4f}"
            )

            # Conv1 with dilation
            wv = torch.tensor(conv1_wv)
            wg = torch.tensor(conv1_wg)
            norm = wv.norm(dim=(1, 2), keepdim=True)
            w = wg * wv / norm
            b = torch.tensor(conv1_b)
            padding = dilation * (kernel_size - 1) // 2
            xt = F.conv1d(xt, w, b, padding=padding, dilation=dilation)

            print(
                f"After Conv1 (d={dilation}): range [{xt.min():.4f}, {xt.max():.4f}], std={xt.std():.4f}"
            )

            # AdaIN2
            adain2_fc = nn.Linear(style_dim, channels * 2)
            adain2_fc.weight.data = torch.tensor(adain2_w)
            adain2_fc.bias.data = torch.tensor(adain2_b)

            style_out = adain2_fc(s_pt)
            gamma2 = style_out[:, :channels].unsqueeze(-1)
            beta2 = style_out[:, channels:].unsqueeze(-1)

            mean2 = xt.mean(dim=2, keepdim=True)
            var2 = xt.var(dim=2, keepdim=True, unbiased=False)
            xt = (xt - mean2) / torch.sqrt(var2 + 1e-5)
            xt = (1 + gamma2) * xt + beta2

            print(
                f"After AdaIN2: range [{xt.min():.4f}, {xt.max():.4f}], std={xt.std():.4f}"
            )

            # Snake2D
            alpha2_t = torch.tensor(alpha2)
            xt = xt + (1 / alpha2_t) * (torch.sin(alpha2_t * xt) ** 2)

            print(
                f"After Snake2D: range [{xt.min():.4f}, {xt.max():.4f}], std={xt.std():.4f}"
            )

            # Conv2 - per StyleTTS2, convs2 uses dilation=1
            wv2 = torch.tensor(conv2_wv)
            wg2 = torch.tensor(conv2_wg)
            norm2 = wv2.norm(dim=(1, 2), keepdim=True)
            w2 = wg2 * wv2 / norm2
            b2 = torch.tensor(conv2_b)
            padding2 = (kernel_size - 1) // 2  # dilation=1 for conv2
            xt = F.conv1d(xt, w2, b2, padding=padding2, dilation=1)

            print(
                f"After Conv2 (d=1): range [{xt.min():.4f}, {xt.max():.4f}], std={xt.std():.4f}"
            )

            # Residual
            x_pt = xt + x_pt

            print(
                f"After residual: range [{x_pt.min():.4f}, {x_pt.max():.4f}], std={x_pt.std():.4f}"
            )

    print(
        f"\nPT final: range [{x_pt.min():.4f}, {x_pt.max():.4f}], std={x_pt.std():.4f}"
    )

    # Compare with expected
    pt_final_np = x_pt.numpy()
    corr_pt = np.corrcoef(pt_final_np.flatten(), pt_output_ncl.flatten())[0, 1]
    print(f"Correlation with expected: {corr_pt:.6f}")

    # ===== MLX =====
    print("\n\n=== MLX Full Resblock ===")
    x_mx = mx.array(pt_input_ncl).transpose(0, 2, 1)  # NLC
    s_mx = mx.array(style)

    for dil_idx, dilation in enumerate(dilations):
        print(f"\n--- Dilation {dilation} (idx {dil_idx}) ---")

        # Load weights
        adain1_w = dec_state[f"{prefix}.adain1.{dil_idx}.fc.weight"].numpy()
        adain1_b = dec_state[f"{prefix}.adain1.{dil_idx}.fc.bias"].numpy()
        adain2_w = dec_state[f"{prefix}.adain2.{dil_idx}.fc.weight"].numpy()
        adain2_b = dec_state[f"{prefix}.adain2.{dil_idx}.fc.bias"].numpy()
        alpha1 = dec_state[f"{prefix}.alpha1.{dil_idx}"].numpy()
        alpha2 = dec_state[f"{prefix}.alpha2.{dil_idx}"].numpy()
        conv1_wv = dec_state[f"{prefix}.convs1.{dil_idx}.weight_v"].numpy()
        conv1_wg = dec_state[f"{prefix}.convs1.{dil_idx}.weight_g"].numpy()
        conv1_b = dec_state[f"{prefix}.convs1.{dil_idx}.bias"].numpy()
        conv2_wv = dec_state[f"{prefix}.convs2.{dil_idx}.weight_v"].numpy()
        conv2_wg = dec_state[f"{prefix}.convs2.{dil_idx}.weight_g"].numpy()
        conv2_b = dec_state[f"{prefix}.convs2.{dil_idx}.bias"].numpy()

        xt = x_mx

        # AdaIN1
        adain1_fc = mnn.Linear(style_dim, channels * 2)
        adain1_fc.weight = mx.array(adain1_w)
        adain1_fc.bias = mx.array(adain1_b)

        style_out = adain1_fc(s_mx)
        gamma1 = style_out[:, :channels][:, None, :]  # [1, 1, C]
        beta1 = style_out[:, channels:][:, None, :]

        mean1 = mx.mean(xt, axis=1, keepdims=True)
        var1 = mx.var(xt, axis=1, keepdims=True)
        xt = (xt - mean1) / mx.sqrt(var1 + 1e-5)
        xt = (1 + gamma1) * xt + beta1
        mx.eval(xt)

        print(
            f"After AdaIN1: range [{float(mx.min(xt)):.4f}, {float(mx.max(xt)):.4f}], std={float(mx.std(xt)):.4f}"
        )

        # Snake1D
        alpha1_mx = mx.array(alpha1).transpose(0, 2, 1)  # [1, 1, C]
        xt = xt + (1 / alpha1_mx) * (mx.sin(alpha1_mx * xt) ** 2)
        mx.eval(xt)

        print(
            f"After Snake1D: range [{float(mx.min(xt)):.4f}, {float(mx.max(xt)):.4f}], std={float(mx.std(xt)):.4f}"
        )

        # Conv1 with dilation
        wv_mx = mx.array(conv1_wv).transpose(0, 2, 1)  # [out, k, in]
        wg_mx = mx.array(conv1_wg).reshape(-1, 1, 1)
        norm_mx = mx.sqrt(mx.sum(wv_mx**2, axis=(1, 2), keepdims=True) + 1e-12)
        w_mx = wg_mx * wv_mx / norm_mx
        b_mx = mx.array(conv1_b)

        padding = dilation * (kernel_size - 1) // 2
        conv1 = mnn.Conv1d(
            channels, channels, kernel_size, padding=padding, dilation=dilation
        )
        conv1.weight = w_mx
        conv1.bias = b_mx
        xt = conv1(xt)
        mx.eval(xt)

        print(
            f"After Conv1 (d={dilation}): range [{float(mx.min(xt)):.4f}, {float(mx.max(xt)):.4f}], std={float(mx.std(xt)):.4f}"
        )

        # AdaIN2
        adain2_fc = mnn.Linear(style_dim, channels * 2)
        adain2_fc.weight = mx.array(adain2_w)
        adain2_fc.bias = mx.array(adain2_b)

        style_out = adain2_fc(s_mx)
        gamma2 = style_out[:, :channels][:, None, :]
        beta2 = style_out[:, channels:][:, None, :]

        mean2 = mx.mean(xt, axis=1, keepdims=True)
        var2 = mx.var(xt, axis=1, keepdims=True)
        xt = (xt - mean2) / mx.sqrt(var2 + 1e-5)
        xt = (1 + gamma2) * xt + beta2
        mx.eval(xt)

        print(
            f"After AdaIN2: range [{float(mx.min(xt)):.4f}, {float(mx.max(xt)):.4f}], std={float(mx.std(xt)):.4f}"
        )

        # Snake2D
        alpha2_mx = mx.array(alpha2).transpose(0, 2, 1)
        xt = xt + (1 / alpha2_mx) * (mx.sin(alpha2_mx * xt) ** 2)
        mx.eval(xt)

        print(
            f"After Snake2D: range [{float(mx.min(xt)):.4f}, {float(mx.max(xt)):.4f}], std={float(mx.std(xt)):.4f}"
        )

        # Conv2 - use dilation=1 (per StyleTTS2)
        wv2_mx = mx.array(conv2_wv).transpose(0, 2, 1)
        wg2_mx = mx.array(conv2_wg).reshape(-1, 1, 1)
        norm2_mx = mx.sqrt(mx.sum(wv2_mx**2, axis=(1, 2), keepdims=True) + 1e-12)
        w2_mx = wg2_mx * wv2_mx / norm2_mx
        b2_mx = mx.array(conv2_b)

        padding2 = (kernel_size - 1) // 2  # dilation=1
        conv2 = mnn.Conv1d(
            channels, channels, kernel_size, padding=padding2, dilation=1
        )
        conv2.weight = w2_mx
        conv2.bias = b2_mx
        xt = conv2(xt)
        mx.eval(xt)

        print(
            f"After Conv2 (d=1): range [{float(mx.min(xt)):.4f}, {float(mx.max(xt)):.4f}], std={float(mx.std(xt)):.4f}"
        )

        # Residual
        x_mx = xt + x_mx
        mx.eval(x_mx)

        print(
            f"After residual: range [{float(mx.min(x_mx)):.4f}, {float(mx.max(x_mx)):.4f}], std={float(mx.std(x_mx)):.4f}"
        )

    print(
        f"\nMLX final: range [{float(mx.min(x_mx)):.4f}, {float(mx.max(x_mx)):.4f}], std={float(mx.std(x_mx)):.4f}"
    )

    # Compare
    mlx_final_ncl = np.array(x_mx.transpose(0, 2, 1))
    corr = np.corrcoef(mlx_final_ncl.flatten(), pt_output_ncl.flatten())[0, 1]
    print(f"Correlation with expected: {corr:.6f}")

    corr_pt_mlx = np.corrcoef(mlx_final_ncl.flatten(), pt_final_np.flatten())[0, 1]
    print(f"Correlation PT vs MLX: {corr_pt_mlx:.6f}")


if __name__ == "__main__":
    main()
