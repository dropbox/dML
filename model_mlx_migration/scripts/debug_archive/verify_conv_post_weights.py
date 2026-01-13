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
Verify conv_post weights match between PyTorch and MLX.
"""

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from huggingface_hub import hf_hub_download

from tools.pytorch_to_mlx.converters import KokoroConverter


def main():
    # Load PyTorch checkpoint
    ckpt_path = hf_hub_download("hexgrad/Kokoro-82M", "kokoro-v1_0.pth")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    dec_state = ckpt["decoder"]

    # Get conv_post weights from checkpoint
    pt_wv = dec_state["module.generator.conv_post.weight_v"].numpy()  # [out, in, k]
    pt_wg = dec_state["module.generator.conv_post.weight_g"].numpy()  # [out, 1, 1]
    pt_b = dec_state["module.generator.conv_post.bias"].numpy()

    print("PT conv_post weights:")
    print(f"  weight_v shape: {pt_wv.shape}")  # [22, 128, 7]
    print(f"  weight_g shape: {pt_wg.shape}")  # [22, 1, 1]
    print(f"  bias shape: {pt_b.shape}")

    # Compute effective weight
    norm = np.sqrt(np.sum(pt_wv**2, axis=(1, 2), keepdims=True))
    pt_w_eff = pt_wg * pt_wv / norm

    print(f"  effective weight shape: {pt_w_eff.shape}")
    print(f"  effective weight range: [{pt_w_eff.min():.6f}, {pt_w_eff.max():.6f}]")
    print(f"  effective weight std: {pt_w_eff.std():.6f}")
    print(f"  bias range: [{pt_b.min():.6f}, {pt_b.max():.6f}]")

    # Load MLX model
    print("\n=== Loading MLX Model ===")
    converter = KokoroConverter()
    model, config, _ = converter.load_from_hf()
    mx.eval(model)

    generator = model.decoder.generator

    # Get MLX conv_post weights
    mlx_wv = np.array(generator.conv_post.weight_v)  # MLX NLC format: [out, k, in]
    mlx_wg = np.array(generator.conv_post.weight_g)
    mlx_b = np.array(generator.conv_post.bias)

    print("\nMLX conv_post weights:")
    print(f"  weight_v shape: {mlx_wv.shape}")
    print(f"  weight_g shape: {mlx_wg.shape}")
    print(f"  bias shape: {mlx_b.shape}")

    # Compute effective weight
    # For MLX NLC format, norm over (k, in) = axes (1, 2)
    mlx_norm = np.sqrt(np.sum(mlx_wv**2, axis=(1, 2), keepdims=True))
    mlx_w_eff = mlx_wg.reshape(-1, 1, 1) * mlx_wv / mlx_norm

    print(f"  effective weight shape: {mlx_w_eff.shape}")
    print(f"  effective weight range: [{mlx_w_eff.min():.6f}, {mlx_w_eff.max():.6f}]")
    print(f"  effective weight std: {mlx_w_eff.std():.6f}")
    print(f"  bias range: [{mlx_b.min():.6f}, {mlx_b.max():.6f}]")

    # Compare weights (need to transpose MLX [out, k, in] to PT [out, in, k])
    mlx_w_eff_pt_fmt = mlx_w_eff.transpose(0, 2, 1)  # [out, in, k]

    print("\n=== Weight Comparison ===")
    print(f"MLX (transposed to PT format) shape: {mlx_w_eff_pt_fmt.shape}")
    print(f"PT shape: {pt_w_eff.shape}")

    diff = np.abs(mlx_w_eff_pt_fmt - pt_w_eff)
    print(f"Max diff: {diff.max():.6e}")
    print(f"Mean diff: {diff.mean():.6e}")

    bias_diff = np.abs(mlx_b - pt_b)
    print(f"Bias max diff: {bias_diff.max():.6e}")

    # Test conv_post with same input
    print("\n=== Testing conv_post with Identical Input ===")
    np.random.seed(42)
    test_input = np.random.randn(1, 7561, 128).astype(np.float32)  # NLC

    # PyTorch conv
    import torch.nn.functional as F

    test_pt = torch.tensor(test_input.transpose(0, 2, 1))  # NCL
    pt_w = torch.tensor(pt_w_eff)
    pt_bias = torch.tensor(pt_b)
    pt_out = F.conv1d(test_pt, pt_w, pt_bias, padding=3)  # padding=(kernel-1)//2

    print(
        f"PT output: range [{pt_out.min():.4f}, {pt_out.max():.4f}], std={pt_out.std():.4f}"
    )

    # MLX conv
    test_mx = mx.array(test_input)
    mlx_out = generator.conv_post(test_mx)
    mx.eval(mlx_out)

    print(
        f"MLX output: range [{float(mx.min(mlx_out)):.4f}, {float(mx.max(mlx_out)):.4f}], std={float(mx.std(mlx_out)):.4f}"
    )

    # Compare
    mlx_out_ncl = np.array(mlx_out.transpose(0, 2, 1))
    pt_out_np = pt_out.numpy()
    corr = np.corrcoef(mlx_out_ncl.flatten(), pt_out_np.flatten())[0, 1]
    output_diff = np.abs(mlx_out_ncl - pt_out_np)
    print(f"Output correlation: {corr:.6f}")
    print(f"Output max diff: {output_diff.max():.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
