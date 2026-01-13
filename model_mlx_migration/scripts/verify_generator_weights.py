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
Verify generator weights match between MLX and PyTorch.
This helps identify if there's a weight loading issue.
"""

import numpy as np


def main():
    # Load MLX model
    from tools.pytorch_to_mlx.converters import KokoroConverter
    converter = KokoroConverter()
    mlx_model, config, state_dict = converter.load_from_hf("hexgrad/Kokoro-82M")

    print(f"State dict top-level keys: {list(state_dict.keys())}")

    # The weights are nested under component names
    decoder_state = state_dict.get("decoder", {})
    print(f"\nDecoder keys sample: {list(decoder_state.keys())[:10]}")

    # Find generator keys
    gen_keys = [k for k in decoder_state.keys() if k.startswith("module.generator.")]
    print(f"\nGenerator keys: {len(gen_keys)}")

    if gen_keys:
        print(f"Sample generator keys: {gen_keys[:10]}")

    print("\n" + "=" * 72)
    print("Conv_post Weight Comparison")
    print("=" * 72)

    # Check conv_post weights
    # PyTorch path: decoder.module.generator.conv_post.*
    # MLX path: decoder.generator.conv_post.*

    pt_weight_g = decoder_state.get("module.generator.conv_post.weight_g")
    pt_weight_v = decoder_state.get("module.generator.conv_post.weight_v")
    pt_bias = decoder_state.get("module.generator.conv_post.bias")

    if pt_weight_g is not None:
        print(f"PyTorch conv_post.weight_g shape: {pt_weight_g.shape}")
        print(f"PyTorch conv_post.weight_v shape: {pt_weight_v.shape}")
        if pt_bias is not None:
            print(f"PyTorch conv_post.bias shape: {pt_bias.shape}")

        # MLX weights
        mlx_conv_post = mlx_model.decoder.generator.conv_post
        print(f"\nMLX conv_post type: {type(mlx_conv_post)}")

        if hasattr(mlx_conv_post, 'weight_g'):
            mlx_wg = np.array(mlx_conv_post.weight_g)
            mlx_wv = np.array(mlx_conv_post.weight_v)
            pt_wg = pt_weight_g.numpy()
            pt_wv = pt_weight_v.numpy()

            print(f"MLX weight_g shape: {mlx_wg.shape}")
            print(f"MLX weight_v shape: {mlx_wv.shape}")

            # Compare - need to handle transposition
            # PyTorch Conv1d: [out_channels, in_channels, kernel_size]
            # MLX Conv1d: [out_channels, kernel_size, in_channels] ? or same?
            wg_diff = np.abs(pt_wg.reshape(-1) - mlx_wg.reshape(-1)).max()
            wv_diff = np.abs(pt_wv - mlx_wv.transpose(0, 2, 1)).max() if pt_wv.shape != mlx_wv.shape else np.abs(pt_wv - mlx_wv).max()

            print(f"\nweight_g max diff: {wg_diff:.2e}")
            print(f"weight_v max diff: {wv_diff:.2e}")

            if pt_bias is not None and hasattr(mlx_conv_post, 'bias') and mlx_conv_post.bias is not None:
                mlx_bias = np.array(mlx_conv_post.bias)
                pt_b = pt_bias.numpy()
                bias_diff = np.abs(pt_b - mlx_bias).max()
                print(f"bias max diff: {bias_diff:.2e}")
        else:
            print("MLX conv_post doesn't have weight_g - checking plain weight")
            if hasattr(mlx_conv_post, 'weight'):
                print(f"MLX weight shape: {mlx_conv_post.weight.shape}")
    else:
        print("PyTorch conv_post weights not found in expected path")
        print(f"Looking for 'module.generator.conv_post.*' in {list(decoder_state.keys())[:20]}")

    # Check ups.0 weights
    print("\n" + "=" * 72)
    print("Ups.0 Weight Comparison")
    print("=" * 72)

    pt_ups0_wg = decoder_state.get("module.generator.ups.0.weight_g")
    pt_ups0_wv = decoder_state.get("module.generator.ups.0.weight_v")

    if pt_ups0_wg is not None:
        print(f"PyTorch ups.0.weight_g shape: {pt_ups0_wg.shape}")
        print(f"PyTorch ups.0.weight_v shape: {pt_ups0_wv.shape}")

        mlx_ups0 = mlx_model.decoder.generator.ups_0
        if hasattr(mlx_ups0, 'weight_g'):
            mlx_wg = np.array(mlx_ups0.weight_g)
            mlx_wv = np.array(mlx_ups0.weight_v)
            print(f"MLX ups_0.weight_g shape: {mlx_wg.shape}")
            print(f"MLX ups_0.weight_v shape: {mlx_wv.shape}")

            pt_wg = pt_ups0_wg.numpy()
            pt_wv = pt_ups0_wv.numpy()

            # Check transposition
            if pt_wv.shape != mlx_wv.shape:
                print("Shape mismatch - checking transpose")
                # ConvTranspose1d has different weight layout
                # PyTorch: [in_ch, out_ch, kernel]
                # MLX: might be [out_ch, kernel, in_ch] or [in_ch, kernel, out_ch]
                for perm in [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]:
                    transposed = pt_wv.transpose(*perm)
                    if transposed.shape == mlx_wv.shape:
                        diff = np.abs(transposed - mlx_wv).max()
                        print(f"  Permutation {perm}: shape matches, diff={diff:.2e}")
    else:
        print("PyTorch ups.0 weights not found")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
