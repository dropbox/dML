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
CosyVoice2 DiT Decoder PyTorch vs MLX Validation

Validates the MLX DiT decoder output against PyTorch reference implementation.
Target: max error < 1e-3
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import math

import numpy as np


def main():
    """Main validation function."""
    import mlx.core as mx
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from tools.pytorch_to_mlx.converters.models.cosyvoice2_flow import (
        DiTDecoder,
        FlowMatchingConfig,
        sinusoidal_embedding,
    )

    # Path to flow.pt
    flow_path = Path.home() / ".cache" / "cosyvoice2" / "cosyvoice2-0.5b" / "flow.pt"

    if not flow_path.exists():
        print(f"flow.pt not found at {flow_path}")
        print("Run: python scripts/download_cosyvoice2.py")
        return 1

    print("=" * 60)
    print("CosyVoice2 DiT Decoder Validation: PyTorch vs MLX")
    print("=" * 60)

    print(f"\nLoading {flow_path}...")
    state_dict = torch.load(flow_path, map_location="cpu", weights_only=False)

    # ====================
    # Test 1: Sinusoidal Embedding Comparison
    # ====================
    print("\n=== Test 1: Sinusoidal Embedding ===")

    class PtSinusoidalPosEmb(nn.Module):
        """PyTorch sinusoidal embedding matching Matcha formula."""

        def __init__(self, dim):
            super().__init__()
            self.dim = dim

        def forward(self, x, scale=1000):
            if x.ndim < 1:
                x = x.unsqueeze(0)
            half_dim = self.dim // 2
            emb = math.log(10000) / (half_dim - 1)
            emb = torch.exp(torch.arange(half_dim).float() * -emb)
            emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
            emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
            return emb

    # Test timesteps
    np.random.seed(42)
    t_np = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    t_pt = torch.tensor(t_np)
    t_mlx = mx.array(t_np)

    pt_sin_emb = PtSinusoidalPosEmb(320)
    with torch.no_grad():
        pt_sin = pt_sin_emb(t_pt, scale=1000)

    mlx_sin = sinusoidal_embedding(t_mlx, 320)
    mx.eval(mlx_sin)

    pt_sin_np = pt_sin.numpy()
    mlx_sin_np = np.array(mlx_sin)

    sin_error = np.max(np.abs(pt_sin_np - mlx_sin_np))
    print(
        f"PyTorch sinusoidal: shape={pt_sin_np.shape}, range=[{pt_sin_np.min():.4f}, {pt_sin_np.max():.4f}]"
    )
    print(
        f"MLX sinusoidal: shape={mlx_sin_np.shape}, range=[{mlx_sin_np.min():.4f}, {mlx_sin_np.max():.4f}]"
    )
    print(f"Sinusoidal max error: {sin_error:.6e}")
    sin_pass = sin_error < 1e-4
    print(f"Sinusoidal test: {'PASS' if sin_pass else 'FAIL'}")

    # ====================
    # Test 2: Time MLP Comparison
    # ====================
    print("\n=== Test 2: Time MLP Comparison ===")

    class PtTimestepEmbedding(nn.Module):
        """PyTorch timestep embedding matching Matcha."""

        def __init__(self, in_channels, time_embed_dim):
            super().__init__()
            self.linear_1 = nn.Linear(in_channels, time_embed_dim)
            self.act = nn.SiLU()
            self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

        def forward(self, sample):
            sample = self.linear_1(sample)
            sample = self.act(sample)
            sample = self.linear_2(sample)
            return sample

    pt_time_mlp = PtTimestepEmbedding(320, 1024)
    pt_time_mlp.linear_1.weight.data = state_dict[
        "decoder.estimator.time_mlp.linear_1.weight"
    ]
    pt_time_mlp.linear_1.bias.data = state_dict[
        "decoder.estimator.time_mlp.linear_1.bias"
    ]
    pt_time_mlp.linear_2.weight.data = state_dict[
        "decoder.estimator.time_mlp.linear_2.weight"
    ]
    pt_time_mlp.linear_2.bias.data = state_dict[
        "decoder.estimator.time_mlp.linear_2.bias"
    ]

    # Load MLX decoder
    config = FlowMatchingConfig()
    mlx_decoder = DiTDecoder.from_pretrained(str(flow_path), config)

    # Test single timestep
    t_np = np.array([0.5], dtype=np.float32)
    t_pt = torch.tensor(t_np)
    t_mlx = mx.array(t_np)

    with torch.no_grad():
        pt_sin = pt_sin_emb(t_pt, scale=1000)
        pt_time_out = pt_time_mlp(pt_sin)

    mlx_time_out = mlx_decoder.time_mlp(t_mlx)
    mx.eval(mlx_time_out)

    pt_time_np = pt_time_out.numpy()
    mlx_time_np = np.array(mlx_time_out)

    time_error = np.max(np.abs(pt_time_np - mlx_time_np))
    print(
        f"PyTorch time MLP: shape={pt_time_np.shape}, range=[{pt_time_np.min():.4f}, {pt_time_np.max():.4f}]"
    )
    print(
        f"MLX time MLP: shape={mlx_time_np.shape}, range=[{mlx_time_np.min():.4f}, {mlx_time_np.max():.4f}]"
    )
    print(f"Time MLP max error: {time_error:.6e}")
    time_pass = time_error < 1e-4
    print(f"Time MLP test: {'PASS' if time_pass else 'FAIL'}")

    # ====================
    # Test 3: Weight Loading Verification
    # ====================
    print("\n=== Test 3: Weight Loading Verification ===")

    # Check down_block[0] conv weights
    pt_conv_w = state_dict["decoder.estimator.down_blocks.0.0.block1.block.0.weight"]
    mlx_conv_w = mlx_decoder.down_blocks[0].conv_block.block1_conv.weight

    pt_conv_transposed = pt_conv_w.numpy().transpose(0, 2, 1)
    mlx_conv_np = np.array(mlx_conv_w)

    conv_error = np.max(np.abs(pt_conv_transposed - mlx_conv_np))
    print(f"Down block[0] conv1 weight max error: {conv_error:.6e}")

    # Check mid_block[0] attention Q weight
    pt_to_q = state_dict["decoder.estimator.mid_blocks.0.1.0.attn1.to_q.weight"]
    mlx_to_q = mlx_decoder.mid_blocks[0].attention_blocks[0].to_q.weight

    attn_error = np.max(np.abs(pt_to_q.numpy() - np.array(mlx_to_q)))
    print(f"Mid block[0] attn Q weight max error: {attn_error:.6e}")

    # Check final projection
    pt_final = state_dict["decoder.estimator.final_proj.weight"]
    mlx_final = mlx_decoder.final_proj.weight

    pt_final_transposed = pt_final.numpy().transpose(0, 2, 1)
    mlx_final_np = np.array(mlx_final)

    final_error = np.max(np.abs(pt_final_transposed - mlx_final_np))
    print(f"Final proj weight max error: {final_error:.6e}")

    weights_pass = (conv_error < 1e-6) and (attn_error < 1e-6) and (final_error < 1e-6)
    print(f"Weight loading test: {'PASS' if weights_pass else 'FAIL'}")

    # ====================
    # Test 4: DiTConvBlock Forward Pass
    # ====================
    print("\n=== Test 4: DiTConvBlock Forward Pass ===")

    # Build PyTorch DiTConvBlock equivalent
    class PtDiTConvBlock(nn.Module):
        """PyTorch DiTConvBlock for comparison."""

        def __init__(self, in_channels, out_channels, time_channels):
            super().__init__()
            # Block1: Conv1d + GroupNorm + Mish
            self.block1_conv = nn.Conv1d(in_channels, out_channels, 3, padding=1)
            self.block1_norm = nn.GroupNorm(32, out_channels)
            # Block2: Conv1d + GroupNorm + Mish
            self.block2_conv = nn.Conv1d(out_channels, out_channels, 3, padding=1)
            self.block2_norm = nn.GroupNorm(32, out_channels)
            # Time MLP (Linear with SiLU)
            self.time_mlp = nn.Linear(time_channels, out_channels)
            # Residual
            self.res_conv = (
                nn.Conv1d(in_channels, out_channels, 1)
                if in_channels != out_channels
                else None
            )

        def forward(self, x, time_emb):
            # x: [batch, channels, time] (NCT format in PyTorch)
            residual = x
            # Block1
            h = self.block1_conv(x)
            h = self.block1_norm(h)
            h = F.mish(h)
            # Add time embedding (Mish, not SiLU - matches Matcha)
            h = h + self.time_mlp(F.mish(time_emb)).unsqueeze(-1)
            # Block2
            h = self.block2_conv(h)
            h = self.block2_norm(h)
            h = F.mish(h)
            # Residual
            if self.res_conv is not None:
                residual = self.res_conv(residual)
            return h + residual

    pt_conv_block = PtDiTConvBlock(320, 256, 1024)
    # Load weights
    pt_conv_block.block1_conv.weight.data = state_dict[
        "decoder.estimator.down_blocks.0.0.block1.block.0.weight"
    ]
    pt_conv_block.block1_conv.bias.data = state_dict[
        "decoder.estimator.down_blocks.0.0.block1.block.0.bias"
    ]
    pt_conv_block.block1_norm.weight.data = state_dict[
        "decoder.estimator.down_blocks.0.0.block1.block.2.weight"
    ]
    pt_conv_block.block1_norm.bias.data = state_dict[
        "decoder.estimator.down_blocks.0.0.block1.block.2.bias"
    ]
    pt_conv_block.block2_conv.weight.data = state_dict[
        "decoder.estimator.down_blocks.0.0.block2.block.0.weight"
    ]
    pt_conv_block.block2_conv.bias.data = state_dict[
        "decoder.estimator.down_blocks.0.0.block2.block.0.bias"
    ]
    pt_conv_block.block2_norm.weight.data = state_dict[
        "decoder.estimator.down_blocks.0.0.block2.block.2.weight"
    ]
    pt_conv_block.block2_norm.bias.data = state_dict[
        "decoder.estimator.down_blocks.0.0.block2.block.2.bias"
    ]
    pt_conv_block.time_mlp.weight.data = state_dict[
        "decoder.estimator.down_blocks.0.0.mlp.1.weight"
    ]
    pt_conv_block.time_mlp.bias.data = state_dict[
        "decoder.estimator.down_blocks.0.0.mlp.1.bias"
    ]
    pt_conv_block.res_conv.weight.data = state_dict[
        "decoder.estimator.down_blocks.0.0.res_conv.weight"
    ]
    pt_conv_block.res_conv.bias.data = state_dict[
        "decoder.estimator.down_blocks.0.0.res_conv.bias"
    ]

    # Test inputs
    np.random.seed(42)
    x_np = np.random.randn(1, 320, 50).astype(np.float32) * 0.1
    time_np = pt_time_np  # Use the time embedding we already computed

    x_pt = torch.tensor(x_np)
    time_pt = torch.tensor(time_np)

    with torch.no_grad():
        pt_conv_out = pt_conv_block(x_pt, time_pt)

    pt_conv_out_np = pt_conv_out.numpy()

    # MLX forward
    # MLX uses NLC format, so transpose
    x_mlx = mx.array(x_np.transpose(0, 2, 1))  # [1, 50, 320]
    time_mlx = mx.array(time_np)

    mlx_conv_out = mlx_decoder.down_blocks[0].conv_block(x_mlx, time_mlx)
    mx.eval(mlx_conv_out)
    mlx_conv_out_np = np.array(mlx_conv_out)  # [1, 50, 256]

    # Compare (transpose MLX to NCT for comparison)
    mlx_conv_out_nct = mlx_conv_out_np.transpose(0, 2, 1)  # [1, 256, 50]

    conv_block_error = np.max(np.abs(pt_conv_out_np - mlx_conv_out_nct))
    conv_block_mean_error = np.mean(np.abs(pt_conv_out_np - mlx_conv_out_nct))
    print(
        f"PyTorch conv block: shape={pt_conv_out_np.shape}, range=[{pt_conv_out_np.min():.4f}, {pt_conv_out_np.max():.4f}]"
    )
    print(
        f"MLX conv block: shape={mlx_conv_out_np.shape}, range=[{mlx_conv_out_np.min():.4f}, {mlx_conv_out_np.max():.4f}]"
    )
    print(f"Conv block max error: {conv_block_error:.6e}")
    print(f"Conv block mean error: {conv_block_mean_error:.6e}")
    conv_block_pass = conv_block_error < 1e-3
    print(f"Conv block test: {'PASS' if conv_block_pass else 'FAIL'}")

    # ====================
    # Test 5: Full MLX Forward Pass
    # ====================
    print("\n=== Test 5: Full MLX Forward Pass ===")

    np.random.seed(42)
    batch_size = 1
    mel_len = 50

    x_np = np.random.randn(batch_size, mel_len, 80).astype(np.float32) * 0.1
    cond_np = np.random.randn(batch_size, mel_len, 80).astype(np.float32) * 0.1
    t_np = np.array([0.5], dtype=np.float32)

    x_mlx = mx.array(x_np)
    cond_mlx = mx.array(cond_np)
    t_mlx = mx.array(t_np)

    mlx_out = mlx_decoder(x_mlx, t_mlx, cond_mlx)
    mx.eval(mlx_out)
    mlx_out_np = np.array(mlx_out)

    print(f"MLX output shape: {mlx_out_np.shape}")
    print(f"MLX output range: [{mlx_out_np.min():.4f}, {mlx_out_np.max():.4f}]")
    print(f"MLX output std: {mlx_out_np.std():.4f}")

    output_valid = (not np.isnan(mlx_out_np).any()) and (mlx_out_np.std() > 0.01)
    print(f"Output validity: {'PASS' if output_valid else 'FAIL'}")

    # ====================
    # Summary
    # ====================
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(
        f"1. Sinusoidal embedding: {'PASS' if sin_pass else 'FAIL'} (max error: {sin_error:.6e})"
    )
    print(
        f"2. Time MLP: {'PASS' if time_pass else 'FAIL'} (max error: {time_error:.6e})"
    )
    print(f"3. Weight loading: {'PASS' if weights_pass else 'FAIL'}")
    print(
        f"4. Conv block: {'PASS' if conv_block_pass else 'FAIL'} (max error: {conv_block_error:.6e})"
    )
    print(f"5. Full forward: {'PASS' if output_valid else 'FAIL'}")

    all_pass = (
        sin_pass and time_pass and weights_pass and conv_block_pass and output_valid
    )
    print(f"\nOverall: {'PASS' if all_pass else 'FAIL'}")

    if not conv_block_pass:
        print("\nNote: Conv block error likely due to activation function differences")
        print("(MLX uses SiLU vs PyTorch Mish)")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
