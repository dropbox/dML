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
Compare PyTorch vs MLX ResBlock outputs.
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
    model, config, state_dict = converter.load_from_hf("hexgrad/Kokoro-82M")

    # Get MLX ResBlock 4 (kernel=7, 128 channels)
    mlx_resblock = model.decoder.generator.resblocks[4]

    # Create PyTorch ResBlock with same architecture
    class ResBlock1dPT(nn.Module):
        def __init__(self, channels, kernel_size, dilations):
            super().__init__()
            self.convs1 = nn.ModuleList()
            self.convs2 = nn.ModuleList()
            for d in dilations:
                self.convs1.append(
                    nn.utils.parametrizations.weight_norm(
                        nn.Conv1d(
                            channels,
                            channels,
                            kernel_size,
                            dilation=d,
                            padding=(kernel_size * d - d) // 2,
                        )
                    )
                )
                self.convs2.append(
                    nn.utils.parametrizations.weight_norm(
                        nn.Conv1d(
                            channels,
                            channels,
                            kernel_size,
                            dilation=1,
                            padding=(kernel_size - 1) // 2,
                        )
                    )
                )

        def forward(self, x):
            for c1, c2 in zip(self.convs1, self.convs2):
                xt = torch.nn.functional.leaky_relu(x, 0.1)
                xt = c1(xt)
                xt = torch.nn.functional.leaky_relu(xt, 0.1)
                xt = c2(xt)
                x = xt + x
            return x

    # Create PyTorch model
    pt_resblock = ResBlock1dPT(128, 7, [1, 3, 5])

    # Load PyTorch checkpoint directly
    ckpt = torch.load(
        "/Users/ayates/models/kokoro/kokoro-v1_0.pth",
        map_location="cpu",
        weights_only=True,
    )
    decoder_state = ckpt["decoder"]

    # Copy weights from checkpoint
    for i in range(3):
        prefix = f"module.generator.resblocks.4.convs1.{i}"
        pt_resblock.convs1[i].parametrizations.weight.original0.data = decoder_state[
            f"{prefix}.weight_g"
        ]
        pt_resblock.convs1[i].parametrizations.weight.original1.data = decoder_state[
            f"{prefix}.weight_v"
        ]
        pt_resblock.convs1[i].bias.data = decoder_state[f"{prefix}.bias"]

        prefix = f"module.generator.resblocks.4.convs2.{i}"
        pt_resblock.convs2[i].parametrizations.weight.original0.data = decoder_state[
            f"{prefix}.weight_g"
        ]
        pt_resblock.convs2[i].parametrizations.weight.original1.data = decoder_state[
            f"{prefix}.weight_v"
        ]
        pt_resblock.convs2[i].bias.data = decoder_state[f"{prefix}.bias"]

    pt_resblock.eval()

    # Create test input - use same seed for reproducibility
    np.random.seed(42)
    test_input = np.random.randn(1, 128, 200).astype(np.float32) * 0.1

    # Run through PyTorch (NCL format)
    pt_input = torch.from_numpy(test_input)
    with torch.no_grad():
        pt_output = pt_resblock(pt_input)

    # Run through MLX (NLC format)
    mlx_input = mx.array(test_input.transpose(0, 2, 1))  # Convert NCL to NLC
    mlx_output = mlx_resblock(mlx_input)
    mx.eval(mlx_output)

    # Convert MLX output back to NCL for comparison
    mlx_output_np = np.array(mlx_output).transpose(0, 2, 1)
    pt_output_np = pt_output.numpy()

    print(f"\nInput: shape={test_input.shape}, std={test_input.std():.4f}")
    print("\nPyTorch output:")
    print(f"  mean={pt_output_np.mean():.4f}, std={pt_output_np.std():.4f}")
    print(f"  range=[{pt_output_np.min():.4f}, {pt_output_np.max():.4f}]")

    print("\nMLX output:")
    print(f"  mean={mlx_output_np.mean():.4f}, std={mlx_output_np.std():.4f}")
    print(f"  range=[{mlx_output_np.min():.4f}, {mlx_output_np.max():.4f}]")

    print("\nDifference:")
    diff = np.abs(pt_output_np - mlx_output_np)
    print(f"  max_diff={diff.max():.6f}")
    print(f"  mean_diff={diff.mean():.6f}")

    # Also trace step by step
    print("\n=== Step-by-step trace ===")
    pt_x = pt_input.clone()
    mlx_x = mx.array(test_input.transpose(0, 2, 1))

    for i, (c1_pt, c2_pt, c1_mlx, c2_mlx) in enumerate(
        zip(
            pt_resblock.convs1,
            pt_resblock.convs2,
            mlx_resblock.convs1,
            mlx_resblock.convs2,
        )
    ):
        print(f"\nDilation {i}:")

        # LeakyReLU
        pt_xt = torch.nn.functional.leaky_relu(pt_x, 0.1)
        mlx_xt = mnn.leaky_relu(mlx_x, 0.1)

        # c1
        with torch.no_grad():
            pt_xt = c1_pt(pt_xt)
        mlx_xt = c1_mlx(mlx_xt)
        mx.eval(mlx_xt)

        pt_np = pt_xt.numpy()
        mlx_np = np.array(mlx_xt).transpose(0, 2, 1)
        print("  After c1:")
        print(f"    PT:  mean={pt_np.mean():.4f}, std={pt_np.std():.4f}")
        print(f"    MLX: mean={mlx_np.mean():.4f}, std={mlx_np.std():.4f}")
        diff = np.abs(pt_np - mlx_np)
        print(f"    max_diff={diff.max():.6f}")

        # LeakyReLU + c2
        pt_xt = torch.nn.functional.leaky_relu(pt_xt, 0.1)
        mlx_xt = mnn.leaky_relu(mlx_xt, 0.1)

        with torch.no_grad():
            pt_xt = c2_pt(pt_xt)
        mlx_xt = c2_mlx(mlx_xt)
        mx.eval(mlx_xt)

        pt_np = pt_xt.numpy()
        mlx_np = np.array(mlx_xt).transpose(0, 2, 1)
        print("  After c2:")
        print(f"    PT:  mean={pt_np.mean():.4f}, std={pt_np.std():.4f}")
        print(f"    MLX: mean={mlx_np.mean():.4f}, std={mlx_np.std():.4f}")
        diff = np.abs(pt_np - mlx_np)
        print(f"    max_diff={diff.max():.6f}")

        # Residual
        pt_x = pt_xt + pt_x
        mlx_x = mlx_xt + mlx_x


if __name__ == "__main__":
    main()
