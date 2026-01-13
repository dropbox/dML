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
Compare full generator (to conv_post) between PyTorch and MLX.
"""

import sys

sys.path.insert(0, "/Users/ayates/model_mlx_migration")

import mlx.core as mx
import mlx.nn as mnn
import numpy as np
import torch
import torch.nn as nn

from tools.pytorch_to_mlx.converters.kokoro_converter import KokoroConverter


def main():
    print("Loading MLX model...")
    converter = KokoroConverter()
    mlx_model, config, _ = converter.load_from_hf("hexgrad/Kokoro-82M")

    print("Loading PyTorch checkpoint...")
    ckpt = torch.load(
        "/Users/ayates/models/kokoro/kokoro-v1_0.pth",
        map_location="cpu",
        weights_only=True,
    )
    pt_decoder = ckpt["decoder"]

    mlx_gen = mlx_model.decoder.generator

    # Create test input
    np.random.seed(42)
    batch = 1
    length = 20
    channels = 512

    x_np = np.random.randn(batch, length, channels).astype(np.float32) * 0.1

    print(f"\nInput: shape={x_np.shape}, mean={x_np.mean():.6f}")

    # MLX path
    mlx_x = mx.array(x_np)

    # Run through ups and resblocks (matching the Generator forward without source)
    for i in range(2):
        mlx_x = mnn.leaky_relu(mlx_x, 0.1)

        # Get PT ups weights
        prefix = f"module.generator.ups.{i}"
        pt_w_g = pt_decoder[f"{prefix}.weight_g"]
        pt_w_v = pt_decoder[f"{prefix}.weight_v"]
        pt_bias = pt_decoder.get(f"{prefix}.bias")
        v_norm = pt_w_v.norm(dim=[1, 2], keepdim=True)
        pt_eff_w = pt_w_g * pt_w_v / (v_norm + 1e-12)

        # ups
        mlx_x = mlx_gen.ups[i](mlx_x)
        mx.eval(mlx_x)

        # resblocks
        xs = None
        for j in range(3):
            block_idx = i * 3 + j
            out = mlx_gen.resblocks[block_idx](mlx_x)
            if xs is None:
                xs = out
            else:
                xs = xs + out
        mlx_x = xs / 3
        mx.eval(mlx_x)
        print(
            f"Stage {i}: shape={mlx_x.shape}, mean={float(mlx_x.mean()):.4f}, std={float(mlx_x.std()):.4f}"
        )

    # conv_post
    mlx_x = mnn.leaky_relu(mlx_x, 0.1)
    mlx_x = mlx_gen.conv_post(mlx_x)
    mx.eval(mlx_x)

    print(f"\nMLX conv_post output: shape={mlx_x.shape}")
    print(f"  mean={float(mlx_x.mean()):.4f}, std={float(mlx_x.std()):.4f}")
    print(f"  range=[{float(mlx_x.min()):.4f}, {float(mlx_x.max()):.4f}]")

    n_bins = mlx_gen.post_n_fft // 2 + 1
    mlx_log_mag = np.array(mlx_x[..., :n_bins])
    mlx_phase = np.array(mlx_x[..., n_bins:])

    print(
        f"\nMLX log_mag: mean={mlx_log_mag.mean():.4f}, range=[{mlx_log_mag.min():.4f}, {mlx_log_mag.max():.4f}]"
    )
    print(
        f"MLX phase: mean={mlx_phase.mean():.4f}, range=[{mlx_phase.min():.4f}, {mlx_phase.max():.4f}]"
    )

    # Now run PyTorch path
    print("\n=== PyTorch Path ===")
    pt_x = torch.from_numpy(x_np).transpose(1, 2)  # NCL

    upsample_rates = [10, 6]
    upsample_kernel_sizes = [20, 12]
    kernel_sizes = [3, 7, 11]
    dilations = [1, 3, 5]

    for i in range(2):
        pt_x = nn.functional.leaky_relu(pt_x, 0.1)

        # ups
        prefix = f"module.generator.ups.{i}"
        pt_w_g = pt_decoder[f"{prefix}.weight_g"]
        pt_w_v = pt_decoder[f"{prefix}.weight_v"]
        pt_bias = pt_decoder.get(f"{prefix}.bias")
        v_norm = pt_w_v.norm(dim=[1, 2], keepdim=True)
        pt_eff_w = pt_w_g * pt_w_v / (v_norm + 1e-12)

        rate = upsample_rates[i]
        kernel = upsample_kernel_sizes[i]
        padding = (kernel - rate) // 2

        pt_x = nn.functional.conv_transpose1d(
            pt_x, pt_eff_w, bias=pt_bias, stride=rate, padding=padding
        )

        # resblocks
        xs = None
        for j in range(3):
            block_idx = i * 3 + j
            rb_x = pt_x

            for dil_idx, dil in enumerate(dilations):
                ks = kernel_sizes[j]
                padding1 = dil * (ks - 1) // 2
                padding2 = (ks - 1) // 2

                # convs1
                prefix = f"module.generator.resblocks.{block_idx}.convs1.{dil_idx}"
                pt_w_g = pt_decoder[f"{prefix}.weight_g"]
                pt_w_v = pt_decoder[f"{prefix}.weight_v"]
                pt_bias = pt_decoder[f"{prefix}.bias"]
                v_norm = pt_w_v.norm(dim=[1, 2], keepdim=True)
                pt_eff_w = pt_w_g * pt_w_v / (v_norm + 1e-12)

                h = nn.functional.leaky_relu(rb_x, 0.1)
                h = nn.functional.conv1d(
                    h, pt_eff_w, pt_bias, padding=padding1, dilation=dil
                )

                # convs2
                prefix = f"module.generator.resblocks.{block_idx}.convs2.{dil_idx}"
                pt_w_g = pt_decoder[f"{prefix}.weight_g"]
                pt_w_v = pt_decoder[f"{prefix}.weight_v"]
                pt_bias = pt_decoder[f"{prefix}.bias"]
                v_norm = pt_w_v.norm(dim=[1, 2], keepdim=True)
                pt_eff_w = pt_w_g * pt_w_v / (v_norm + 1e-12)

                h = nn.functional.leaky_relu(h, 0.1)
                h = nn.functional.conv1d(h, pt_eff_w, pt_bias, padding=padding2)

                rb_x = rb_x + h

            if xs is None:
                xs = rb_x
            else:
                xs = xs + rb_x
        pt_x = xs / 3

        print(
            f"Stage {i}: shape={tuple(pt_x.shape)}, mean={pt_x.mean().item():.4f}, std={pt_x.std().item():.4f}"
        )

    # conv_post
    pt_x = nn.functional.leaky_relu(pt_x, 0.1)

    prefix = "module.generator.conv_post"
    pt_w_g = pt_decoder[f"{prefix}.weight_g"]
    pt_w_v = pt_decoder[f"{prefix}.weight_v"]
    pt_bias = pt_decoder[f"{prefix}.bias"]
    v_norm = pt_w_v.norm(dim=[1, 2], keepdim=True)
    pt_eff_w = pt_w_g * pt_w_v / (v_norm + 1e-12)

    pt_x = nn.functional.conv1d(pt_x, pt_eff_w, pt_bias, padding=3)
    pt_np = pt_x.detach().numpy().transpose(0, 2, 1)  # NLC

    print(f"\nPT conv_post output: shape={pt_np.shape}")
    print(f"  mean={pt_np.mean():.4f}, std={pt_np.std():.4f}")
    print(f"  range=[{pt_np.min():.4f}, {pt_np.max():.4f}]")

    pt_log_mag = pt_np[..., :n_bins]
    pt_phase = pt_np[..., n_bins:]

    print(
        f"\nPT log_mag: mean={pt_log_mag.mean():.4f}, range=[{pt_log_mag.min():.4f}, {pt_log_mag.max():.4f}]"
    )
    print(
        f"PT phase: mean={pt_phase.mean():.4f}, range=[{pt_phase.min():.4f}, {pt_phase.max():.4f}]"
    )

    # Comparison
    print("\n=== Comparison ===")
    print(f"conv_post max diff: {np.abs(np.array(mlx_x) - pt_np).max():.6f}")
    print(f"log_mag max diff: {np.abs(mlx_log_mag - pt_log_mag).max():.6f}")
    print(f"phase max diff: {np.abs(mlx_phase - pt_phase).max():.6f}")


if __name__ == "__main__":
    main()
