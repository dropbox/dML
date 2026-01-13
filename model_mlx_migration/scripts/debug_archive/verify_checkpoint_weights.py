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
Verify checkpoint weights and test forward pass with raw weights.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    from huggingface_hub import hf_hub_download

    # Load checkpoint
    ckpt_path = hf_hub_download("hexgrad/Kokoro-82M", "kokoro-v1_0.pth")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    dec_state = ckpt["decoder"]

    # List all keys related to resblocks.0
    rb0_keys = sorted([k for k in dec_state.keys() if "resblocks.0" in k])
    print("Resblock 0 weights in checkpoint:")
    for k in rb0_keys:
        shape = dec_state[k].shape
        print(f"  {k}: {shape}")

    # Check for any unexpected keys
    print("\nLooking for extra weights not accounted for...")
    for k in rb0_keys:
        if "conv" not in k and "adain" not in k and "alpha" not in k:
            print(f"  Unknown key: {k}")

    # Check weight statistics
    print("\n=== Weight Statistics ===")
    prefix = "module.generator.resblocks.0"
    for dil_idx in range(3):
        conv1_wv = dec_state[f"{prefix}.convs1.{dil_idx}.weight_v"]
        conv1_wg = dec_state[f"{prefix}.convs1.{dil_idx}.weight_g"]
        conv2_wv = dec_state[f"{prefix}.convs2.{dil_idx}.weight_v"]
        conv2_wg = dec_state[f"{prefix}.convs2.{dil_idx}.weight_g"]

        # Compute effective weights
        norm1 = conv1_wv.norm(dim=(1, 2), keepdim=True)
        w1 = conv1_wg * conv1_wv / norm1
        norm2 = conv2_wv.norm(dim=(1, 2), keepdim=True)
        w2 = conv2_wg * conv2_wv / norm2

        print(f"\nDilation {dil_idx}:")
        print(
            f"  conv1: wg range [{conv1_wg.min():.4f}, {conv1_wg.max():.4f}], eff_w std={w1.std():.6f}"
        )
        print(
            f"  conv2: wg range [{conv2_wg.min():.4f}, {conv2_wg.max():.4f}], eff_w std={w2.std():.6f}"
        )

        alpha1 = dec_state[f"{prefix}.alpha1.{dil_idx}"]
        alpha2 = dec_state[f"{prefix}.alpha2.{dil_idx}"]
        print(f"  alpha1: range [{alpha1.min():.4f}, {alpha1.max():.4f}]")
        print(f"  alpha2: range [{alpha2.min():.4f}, {alpha2.max():.4f}]")

    # Test the actual forward with just convolutions (no AdaIN or Snake)
    print("\n=== Testing Conv-only forward ===")
    internal = np.load("/tmp/kokoro_ref/generator_internal_traces.npz")
    rb0_in = internal["resblock_0_in"]  # [1, 256, 1260]

    x = torch.tensor(rb0_in)

    with torch.no_grad():
        # Just apply conv1.0 -> conv2.0 -> residual
        conv1_wv = dec_state[f"{prefix}.convs1.0.weight_v"]
        conv1_wg = dec_state[f"{prefix}.convs1.0.weight_g"]
        conv1_b = dec_state[f"{prefix}.convs1.0.bias"]
        norm1 = conv1_wv.norm(dim=(1, 2), keepdim=True)
        w1 = conv1_wg * conv1_wv / norm1

        # Different padding tests
        for pad in [0, 1, 2]:
            out = F.conv1d(x, w1, conv1_b, padding=pad, dilation=1)
            print(f"conv1.0 with padding={pad}: shape {out.shape}, std={out.std():.4f}")

    # Check if there's something special about the model structure
    print("\n=== Looking for other model components ===")
    # Check for any additional layers that might affect the output
    all_keys = list(dec_state.keys())
    unique_prefixes = set()
    for k in all_keys:
        parts = k.split(".")
        if len(parts) >= 4:
            unique_prefixes.add(".".join(parts[:4]))

    print("Unique prefixes (first 20):")
    for p in sorted(unique_prefixes)[:20]:
        print(f"  {p}")


if __name__ == "__main__":
    main()
