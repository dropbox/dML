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
Trace the ACTUAL loaded model's resblock to find where divergence happens.
"""

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.pytorch_to_mlx.converters import KokoroConverter


def main():
    from huggingface_hub import hf_hub_download

    # Load PyTorch checkpoint
    print("=== Loading PyTorch checkpoint ===")
    ckpt_path = hf_hub_download("hexgrad/Kokoro-82M", "kokoro-v1_0.pth")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    dec_state = ckpt["decoder"]

    # Load MLX model
    print("=== Loading MLX model ===")
    converter = KokoroConverter()
    model, config, _ = converter.load_from_hf()
    mx.eval(model)

    generator = model.decoder.generator
    resblock = generator.resblocks_0

    # Create test input
    np.random.seed(42)
    channels = 256
    style_dim = 128
    length = 126

    x_np = np.random.randn(1, channels, length).astype(np.float32) * 0.1
    s_np = np.random.randn(1, style_dim).astype(np.float32)

    x_mx = mx.array(x_np.transpose(0, 2, 1))  # NLC
    s_mx = mx.array(s_np)

    print(f"\nInput shape: {x_mx.shape}")
    print(f"Style shape: {s_mx.shape}")

    # Compare weights between model and checkpoint
    prefix = "module.generator.resblocks.0"
    print("\n=== Weight Comparison (iter 0 only) ===")

    # adain1_0
    pt_adain1_w = dec_state[f"{prefix}.adain1.0.fc.weight"].numpy()
    dec_state[f"{prefix}.adain1.0.fc.bias"].numpy()
    mlx_adain1_w = np.array(resblock.adain1_0.fc.weight)
    np.array(resblock.adain1_0.fc.bias)

    print("adain1_0.weight:")
    print(f"  PT shape: {pt_adain1_w.shape}, MLX shape: {mlx_adain1_w.shape}")
    print(f"  PT range: [{pt_adain1_w.min():.4f}, {pt_adain1_w.max():.4f}]")
    print(f"  MLX range: [{mlx_adain1_w.min():.4f}, {mlx_adain1_w.max():.4f}]")
    if pt_adain1_w.shape == mlx_adain1_w.shape:
        print(f"  Max diff: {np.abs(pt_adain1_w - mlx_adain1_w).max():.6e}")

    # alpha1_0
    pt_alpha1 = dec_state[f"{prefix}.alpha1.0"].numpy()  # [1, C, 1]
    mlx_alpha1 = np.array(resblock.alpha1_0)

    print("\nalpha1_0:")
    print(f"  PT shape: {pt_alpha1.shape}, MLX shape: {mlx_alpha1.shape}")
    print(f"  PT range: [{pt_alpha1.min():.4f}, {pt_alpha1.max():.4f}]")
    print(f"  MLX range: [{mlx_alpha1.min():.4f}, {mlx_alpha1.max():.4f}]")

    # convs1_0 weight - WeightNormConv1d has weight_v and weight_g
    pt_conv1_wv = dec_state[f"{prefix}.convs1.0.weight_v"].numpy()  # [out, in, k]
    pt_conv1_wg = dec_state[f"{prefix}.convs1.0.weight_g"].numpy()  # [out, 1, 1]
    pt_conv1_b = dec_state[f"{prefix}.convs1.0.bias"].numpy()

    mlx_conv1_wv = np.array(resblock.convs1_0.weight_v)  # [out, in, k]
    mlx_conv1_wg = np.array(resblock.convs1_0.weight_g)  # [out, 1, 1]

    print("\nconvs1_0.weight_v:")
    print(f"  PT shape: {pt_conv1_wv.shape}, MLX shape: {mlx_conv1_wv.shape}")
    print(f"  PT range: [{pt_conv1_wv.min():.4f}, {pt_conv1_wv.max():.4f}]")
    print(f"  MLX range: [{mlx_conv1_wv.min():.4f}, {mlx_conv1_wv.max():.4f}]")
    if pt_conv1_wv.shape == mlx_conv1_wv.shape:
        print(f"  Max diff: {np.abs(pt_conv1_wv - mlx_conv1_wv).max():.6e}")

    print("\nconvs1_0.weight_g:")
    print(f"  PT shape: {pt_conv1_wg.shape}, MLX shape: {mlx_conv1_wg.shape}")
    print(f"  PT range: [{pt_conv1_wg.min():.4f}, {pt_conv1_wg.max():.4f}]")
    print(f"  MLX range: [{mlx_conv1_wg.min():.4f}, {mlx_conv1_wg.max():.4f}]")

    # convs1_0 bias
    mlx_conv1_b = np.array(resblock.convs1_0.bias)
    print("\nconvs1_0.bias:")
    print(f"  PT range: [{pt_conv1_b.min():.4f}, {pt_conv1_b.max():.4f}]")
    print(f"  MLX range: [{mlx_conv1_b.min():.4f}, {mlx_conv1_b.max():.4f}]")
    print(f"  Max diff: {np.abs(pt_conv1_b - mlx_conv1_b).max():.6e}")

    # Check convs1_0 dilation and padding
    print("\nconvs1_0 config:")
    print(f"  kernel_size: {resblock.convs1_0.kernel_size}")
    print(f"  dilation: {resblock.convs1_0.dilation}")
    print(f"  padding: {resblock.convs1_0.padding}")

    # Check convs2_0 dilation
    print("\nconvs2_0 config:")
    print(f"  kernel_size: {resblock.convs2_0.kernel_size}")
    print(f"  dilation: {resblock.convs2_0.dilation}")
    print(f"  padding: {resblock.convs2_0.padding}")

    # Run through model's resblock
    print("\n=== Running Model's Resblock ===")
    out_mx = resblock(x_mx, s_mx)
    mx.eval(out_mx)

    out_mx_ncl = np.array(out_mx.transpose(0, 2, 1))
    print(
        f"MLX output: range [{out_mx_ncl.min():.4f}, {out_mx_ncl.max():.4f}], std={out_mx_ncl.std():.4f}"
    )

    # Run through manual implementation for comparison
    from trace_resblock_iterations import pytorch_resblock_single_iter

    # Get weights
    prefix = "module.generator.resblocks.0"
    weights = {}
    for key in dec_state:
        if key.startswith(prefix):
            short_key = key[len(prefix) + 1 :]
            weights[short_key] = dec_state[key]

    x_pt = torch.tensor(x_np)
    s_pt = torch.tensor(s_np)

    for i in range(3):
        x_pt = pytorch_resblock_single_iter(x_pt, s_pt, weights, i)

    print(
        f"PT output: range [{x_pt.min():.4f}, {x_pt.max():.4f}], std={x_pt.std():.4f}"
    )

    # Correlation
    corr = np.corrcoef(out_mx_ncl.flatten(), x_pt.numpy().flatten())[0, 1]
    print(f"\nCorrelation: {corr:.6f}")
    print(f"Std ratio (PT/MLX): {float(x_pt.std()) / out_mx_ncl.std():.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
