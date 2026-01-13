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
Debug weight format differences between PyTorch and MLX.
"""

import sys

sys.path.insert(0, "tools/pytorch_to_mlx")

from pathlib import Path

import mlx.core as mx
import torch


def debug_weights():
    """Check weight format differences."""
    print("=" * 60)
    print("Weight Format Debug")
    print("=" * 60)

    # Load PyTorch state dict
    weights_path = Path.home() / "models" / "kokoro" / "kokoro-v1_0.pth"
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    decoder_state = state_dict.get("decoder", {})

    print("\nPyTorch conv_post weights:")
    key_v = "module.generator.conv_post.weight_v"
    key_g = "module.generator.conv_post.weight_g"
    if key_v in decoder_state:
        pt_v = decoder_state[key_v]
        pt_g = decoder_state[key_g]
        print(f"  weight_v shape: {pt_v.shape}")  # PyTorch NCL: [out_ch, in_ch, kernel]
        print(f"  weight_g shape: {pt_g.shape}")
        print(f"  weight_v[:3,:3,0]: \n{pt_v[:3, :3, 0]}")

    # Check resblock weights
    print("\nPyTorch resblock.0.convs1.0 weights:")
    key_v = "module.generator.resblocks.0.convs1.0.weight_v"
    key_g = "module.generator.resblocks.0.convs1.0.weight_g"
    if key_v in decoder_state:
        pt_v = decoder_state[key_v]
        pt_g = decoder_state[key_g]
        print(f"  weight_v shape: {pt_v.shape}")
        print(f"  weight_g shape: {pt_g.shape}")

    # Check noise_conv weights
    print("\nPyTorch noise_convs.0 weights:")
    key = "module.generator.noise_convs.0.weight"
    if key in decoder_state:
        pt_w = decoder_state[key]
        print(
            f"  weight shape: {pt_w.shape}"
        )  # Should be [out, in, kernel] for PyTorch NCL

    # Check upsample weights
    print("\nPyTorch ups.0 weights:")
    key = "module.generator.ups.0.weight"
    if key in decoder_state:
        pt_w = decoder_state[key]
        print(f"  weight shape: {pt_w.shape}")

    # Now load MLX model and compare
    print("\n" + "=" * 60)
    print("MLX Model Weights")
    print("=" * 60)

    from converters.kokoro_converter import KokoroConverter

    converter = KokoroConverter()
    model, config, _ = converter.load_from_hf()
    mx.eval(model)

    gen = model.decoder.generator

    print("\nMLX conv_post weights:")
    print(f"  weight_v shape: {gen.conv_post.weight_v.shape}")
    print(f"  weight_g shape: {gen.conv_post.weight_g.shape}")
    # MLX NLC format: [out_ch, kernel, in_ch] for conv (transposed vs PyTorch)

    print("\nMLX resblock.0.convs1.0 weights:")
    print(f"  weight_v shape: {gen.resblocks[0].convs1[0].weight_v.shape}")
    print(f"  weight_g shape: {gen.resblocks[0].convs1[0].weight_g.shape}")

    print("\nMLX noise_convs.0 weights:")
    print(f"  weight_v shape: {gen.noise_convs[0].weight_v.shape}")

    print("\nMLX ups.0 weights:")
    print(f"  weight shape: {gen.ups[0].weight.shape}")

    # Now let's verify the WeightNormConv1d format
    print("\n" + "=" * 60)
    print("Verifying WeightNormConv1d weight format")
    print("=" * 60)

    from converters.models.kokoro_modules import WeightNormConv1d

    # Create test conv and check expected format
    test_conv = WeightNormConv1d(in_channels=22, out_channels=128, kernel_size=5)
    print("\nTest WeightNormConv1d(22, 128, 5):")
    print(f"  weight_v shape: {test_conv.weight_v.shape}")
    # Expected for MLX NLC: [out_ch, kernel, in_ch] = [128, 5, 22]

    # Check if PyTorch weights need transposition
    print("\n" + "=" * 60)
    print("Weight format comparison")
    print("=" * 60)

    # PyTorch Conv1d weight: [out_ch, in_ch, kernel] for NCL input
    # MLX Conv1d weight: [out_ch, kernel, in_ch] for NLC input

    # When loading PyTorch NCL weights into MLX NLC conv:
    # Need to transpose: [out, in, kernel] -> [out, kernel, in]
    # This is a permutation of axes 1 and 2

    key_v = "module.generator.noise_convs.0.weight"
    if key_v in decoder_state:
        pt_w = decoder_state[key_v]
        print(f"\nPyTorch noise_convs.0.weight: {pt_w.shape}")
        # Should be [out, in, kernel] = [256, 22, 12] for NCL

        # What MLX expects for NLC: [out, kernel, in] = [256, 12, 22]
        expected_mlx = pt_w.permute(0, 2, 1)
        print(f"Expected MLX shape (transposed): {expected_mlx.shape}")

        # What we actually loaded:
        mlx_actual = gen.noise_convs[0].weight_v
        print(f"Actual MLX weight_v shape: {mlx_actual.shape}")

        if list(pt_w.shape) == list(mlx_actual.shape):
            print("\n>>> MISMATCH: Weights loaded without transposition!")
            print(
                ">>> Need to transpose PyTorch [out,in,kernel] to MLX [out,kernel,in]"
            )


if __name__ == "__main__":
    debug_weights()
