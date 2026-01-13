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
Trace through a full resblock with matching dilations for convs2.
Test hypothesis: Kokoro uses dilation=(1,3,5) for BOTH convs1 and convs2.
"""

import sys
from pathlib import Path

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

    pt_input_ncl = internal["resblock_0_in"]  # NCL [1, 256, 1260]
    pt_output_ncl = internal["resblock_0_out"]
    style = ref["style_128"]

    print(f"Input shape: {pt_input_ncl.shape}")
    print(
        f"Expected output range: [{pt_output_ncl.min():.4f}, {pt_output_ncl.max():.4f}], std={pt_output_ncl.std():.4f}"
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

    # Check conv2 weight shapes to infer expected padding/dilation
    for dil_idx in range(3):
        conv2_wv = dec_state[f"{prefix}.convs2.{dil_idx}.weight_v"].numpy()
        print(f"convs2.{dil_idx}.weight_v shape: {conv2_wv.shape}")

    print("\n=== Test 1: conv2 with dilation=1 (StyleTTS2 original) ===")
    result1 = run_resblock_pt(
        pt_input_ncl,
        style,
        dec_state,
        prefix,
        channels,
        style_dim,
        dilations,
        kernel_size,
        conv2_dilation=1,
    )
    corr1 = np.corrcoef(result1.flatten(), pt_output_ncl.flatten())[0, 1]
    print(
        f"Result: range [{result1.min():.4f}, {result1.max():.4f}], std={result1.std():.4f}"
    )
    print(f"Correlation with expected: {corr1:.6f}")

    print("\n=== Test 2: conv2 with matching dilation (1,3,5) ===")
    result2 = run_resblock_pt(
        pt_input_ncl,
        style,
        dec_state,
        prefix,
        channels,
        style_dim,
        dilations,
        kernel_size,
        conv2_dilation="matching",
    )
    corr2 = np.corrcoef(result2.flatten(), pt_output_ncl.flatten())[0, 1]
    print(
        f"Result: range [{result2.min():.4f}, {result2.max():.4f}], std={result2.std():.4f}"
    )
    print(f"Correlation with expected: {corr2:.6f}")

    print("\n=== Test 3: conv2 with different padding formula ===")
    # Maybe the padding is computed differently
    result3 = run_resblock_pt(
        pt_input_ncl,
        style,
        dec_state,
        prefix,
        channels,
        style_dim,
        dilations,
        kernel_size,
        conv2_dilation="matching",
        alt_padding=True,
    )
    corr3 = np.corrcoef(result3.flatten(), pt_output_ncl.flatten())[0, 1]
    print(
        f"Result: range [{result3.min():.4f}, {result3.max():.4f}], std={result3.std():.4f}"
    )
    print(f"Correlation with expected: {corr3:.6f}")

    # If none match, check what the actual PyTorch implementation does
    print("\n=== Analyzing expected output statistics ===")
    print(
        f"Expected per-channel std: min={pt_output_ncl.std(axis=2).min():.4f}, max={pt_output_ncl.std(axis=2).max():.4f}"
    )
    print(
        f"Input per-channel std: min={pt_input_ncl.std(axis=2).min():.4f}, max={pt_input_ncl.std(axis=2).max():.4f}"
    )
    amplification = pt_output_ncl.std() / pt_input_ncl.std()
    print(f"Overall amplification ratio: {amplification:.4f}")


def run_resblock_pt(
    input_ncl,
    style,
    dec_state,
    prefix,
    channels,
    style_dim,
    dilations,
    kernel_size,
    conv2_dilation=1,
    alt_padding=False,
):
    """Run resblock in PyTorch with specified conv2 dilation."""
    x = torch.tensor(input_ncl)
    s = torch.tensor(style)

    with torch.no_grad():
        for dil_idx, dilation in enumerate(dilations):
            # Determine conv2 dilation
            if conv2_dilation == "matching":
                c2_dilation = dilation
            else:
                c2_dilation = conv2_dilation

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

            xt = x

            # AdaIN1
            adain1_fc = nn.Linear(style_dim, channels * 2)
            adain1_fc.weight.data = torch.tensor(adain1_w)
            adain1_fc.bias.data = torch.tensor(adain1_b)

            style_out = adain1_fc(s)
            gamma1 = style_out[:, :channels].unsqueeze(-1)
            beta1 = style_out[:, channels:].unsqueeze(-1)

            mean1 = xt.mean(dim=2, keepdim=True)
            var1 = xt.var(dim=2, keepdim=True, unbiased=False)
            xt = (xt - mean1) / torch.sqrt(var1 + 1e-5)
            xt = (1 + gamma1) * xt + beta1

            # Snake1D
            alpha1_t = torch.tensor(alpha1)
            xt = xt + (1 / alpha1_t) * (torch.sin(alpha1_t * xt) ** 2)

            # Conv1
            wv = torch.tensor(conv1_wv)
            wg = torch.tensor(conv1_wg)
            norm = wv.norm(dim=(1, 2), keepdim=True)
            w = wg * wv / norm
            b = torch.tensor(conv1_b)
            padding1 = dilation * (kernel_size - 1) // 2
            xt = F.conv1d(xt, w, b, padding=padding1, dilation=dilation)

            # AdaIN2
            adain2_fc = nn.Linear(style_dim, channels * 2)
            adain2_fc.weight.data = torch.tensor(adain2_w)
            adain2_fc.bias.data = torch.tensor(adain2_b)

            style_out = adain2_fc(s)
            gamma2 = style_out[:, :channels].unsqueeze(-1)
            beta2 = style_out[:, channels:].unsqueeze(-1)

            mean2 = xt.mean(dim=2, keepdim=True)
            var2 = xt.var(dim=2, keepdim=True, unbiased=False)
            xt = (xt - mean2) / torch.sqrt(var2 + 1e-5)
            xt = (1 + gamma2) * xt + beta2

            # Snake2D
            alpha2_t = torch.tensor(alpha2)
            xt = xt + (1 / alpha2_t) * (torch.sin(alpha2_t * xt) ** 2)

            # Conv2
            wv2 = torch.tensor(conv2_wv)
            wg2 = torch.tensor(conv2_wg)
            norm2 = wv2.norm(dim=(1, 2), keepdim=True)
            w2 = wg2 * wv2 / norm2
            b2 = torch.tensor(conv2_b)

            if alt_padding:
                # Alternative: use kernel-based padding
                padding2 = c2_dilation * (kernel_size - 1) // 2
            else:
                padding2 = c2_dilation * (kernel_size - 1) // 2

            xt = F.conv1d(xt, w2, b2, padding=padding2, dilation=c2_dilation)

            # Residual
            x = xt + x

    return x.numpy()


if __name__ == "__main__":
    main()
