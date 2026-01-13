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
Verify resblock AdaIN weights match between PyTorch checkpoint and loaded MLX model.
"""

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.pytorch_to_mlx.converters import KokoroConverter


def main():
    print("=== Loading MLX Model ===")
    converter = KokoroConverter()
    model, config, _ = converter.load_from_hf()
    mx.eval(model)

    print("\n=== Loading PyTorch Checkpoint ===")
    from huggingface_hub import hf_hub_download

    ckpt_path = hf_hub_download("hexgrad/Kokoro-82M", "kokoro-v1_0.pth")
    pt_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    # The checkpoint is nested - generator weights are under 'decoder' key
    pt_state = pt_ckpt.get("decoder", {})

    generator = model.decoder.generator

    print("\n=== Comparing Generator resblocks AdaIN weights ===")

    # Check resblocks (there should be num_upsamples * num_kernels resblocks)
    # With num_upsamples=6 and num_kernels=3, that's 18 resblocks
    num_resblocks = generator._num_resblocks
    print(f"Number of resblocks: {num_resblocks}")

    all_match = True
    for i in range(num_resblocks):
        resblock = getattr(generator, f"resblocks_{i}")
        res_prefix = f"module.generator.resblocks.{i}"

        # Check adain1 weights (3 layers: adain1_0, adain1_1, adain1_2)
        for j in range(3):
            pt_weight_key = f"{res_prefix}.adain1.{j}.fc.weight"
            pt_bias_key = f"{res_prefix}.adain1.{j}.fc.bias"

            if pt_weight_key not in pt_state:
                print(f"WARNING: {pt_weight_key} not in checkpoint")
                continue

            pt_weight = pt_state[pt_weight_key].numpy()
            pt_bias = pt_state[pt_bias_key].numpy()

            adain_block = getattr(resblock, f"adain1_{j}", None)
            if adain_block is None:
                print(f"WARNING: resblocks_{i}.adain1_{j} not found in MLX model")
                continue

            mlx_weight = np.array(adain_block.fc.weight)
            mlx_bias = np.array(adain_block.fc.bias)

            weight_diff = np.abs(pt_weight - mlx_weight).max()
            bias_diff = np.abs(pt_bias - mlx_bias).max()

            if weight_diff > 1e-6 or bias_diff > 1e-6:
                all_match = False
                print(f"\nresblocks_{i}.adain1_{j}.fc MISMATCH:")
                print(
                    f"  PT weight shape: {pt_weight.shape}, range: [{pt_weight.min():.4f}, {pt_weight.max():.4f}]"
                )
                print(
                    f"  MLX weight shape: {mlx_weight.shape}, range: [{mlx_weight.min():.4f}, {mlx_weight.max():.4f}]"
                )
                print(f"  Weight max diff: {weight_diff:.6e}")
                print(f"  Bias max diff: {bias_diff:.6e}")

        # Check adain2 weights
        for j in range(3):
            pt_weight_key = f"{res_prefix}.adain2.{j}.fc.weight"
            pt_bias_key = f"{res_prefix}.adain2.{j}.fc.bias"

            if pt_weight_key not in pt_state:
                continue

            pt_weight = pt_state[pt_weight_key].numpy()
            pt_bias = pt_state[pt_bias_key].numpy()

            adain_block = getattr(resblock, f"adain2_{j}", None)
            if adain_block is None:
                continue

            mlx_weight = np.array(adain_block.fc.weight)
            mlx_bias = np.array(adain_block.fc.bias)

            weight_diff = np.abs(pt_weight - mlx_weight).max()
            bias_diff = np.abs(pt_bias - mlx_bias).max()

            if weight_diff > 1e-6 or bias_diff > 1e-6:
                all_match = False
                print(f"\nresblocks_{i}.adain2_{j}.fc MISMATCH:")
                print(f"  Weight max diff: {weight_diff:.6e}")
                print(f"  Bias max diff: {bias_diff:.6e}")

    if all_match:
        print("\nAll resblock AdaIN weights match!")

    # Also verify convs1 and convs2 weights
    print("\n=== Comparing Generator resblocks Conv weights ===")
    for i in range(num_resblocks):
        resblock = getattr(generator, f"resblocks_{i}")
        res_prefix = f"module.generator.resblocks.{i}"

        for j in range(3):
            # Check convs1
            pt_wv_key = f"{res_prefix}.convs1.{j}.weight_v"
            pt_wg_key = f"{res_prefix}.convs1.{j}.weight_g"

            if pt_wv_key in pt_state:
                conv_block = getattr(resblock, f"convs1_{j}", None)
                if conv_block is not None:
                    pt_wv = pt_state[pt_wv_key].numpy()
                    pt_wg = pt_state[pt_wg_key].numpy()
                    mlx_wv = np.array(conv_block.weight_v)
                    mlx_wg = np.array(conv_block.weight_g)

                    wv_diff = np.abs(pt_wv - mlx_wv).max()
                    wg_diff = np.abs(pt_wg - mlx_wg).max()

                    if wv_diff > 1e-6 or wg_diff > 1e-6:
                        all_match = False
                        print(
                            f"resblocks_{i}.convs1_{j} MISMATCH: wv_diff={wv_diff:.6e}, wg_diff={wg_diff:.6e}"
                        )

    if all_match:
        print("All resblock conv weights match!")

    # Verify alpha values
    print("\n=== Comparing Generator resblocks alpha values ===")
    for i in range(num_resblocks):
        resblock = getattr(generator, f"resblocks_{i}")
        res_prefix = f"module.generator.resblocks.{i}"

        for j in range(3):
            pt_alpha1_key = f"{res_prefix}.alpha1.{j}"
            if pt_alpha1_key in pt_state:
                pt_alpha = pt_state[pt_alpha1_key].numpy()
                mlx_alpha = np.array(getattr(resblock, f"alpha1_{j}"))
                diff = np.abs(pt_alpha - mlx_alpha).max()
                if diff > 1e-6:
                    print(f"resblocks_{i}.alpha1_{j} MISMATCH: diff={diff:.6e}")
                    print(f"  PT: {pt_alpha}, MLX: {mlx_alpha}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
