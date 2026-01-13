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

"""Check F0 block structure and behavior."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "pytorch_to_mlx"))

import mlx.core as mx
import torch

# Check PyTorch F0 block structure
print("=== PyTorch F0 block keys ===")
model_path = Path.home() / "models" / "kokoro" / "kokoro-v1_0.pth"
checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
pred_state = checkpoint.get("predictor", {})

# Check F0.0, F0.1, F0.2 structure
for block_idx in range(3):
    print(f"\n--- F0.{block_idx} ---")
    keys = [k for k in sorted(pred_state.keys()) if f"module.F0.{block_idx}." in k]
    for key in keys[:20]:  # First 20 keys
        val = pred_state[key]
        print(f"  {key}: {val.shape}")

# Check for pool layer
print("\n=== Pool layers in F0 blocks ===")
for key in sorted(pred_state.keys()):
    if "F0" in key and "pool" in key:
        val = pred_state[key]
        print(f"{key}: {val.shape}")

# Load MLX model
print("\n=== Loading MLX model ===")
from converters.kokoro_converter import KokoroConverter

converter = KokoroConverter()
model, config, pt_state = converter.load_from_hf("hexgrad/Kokoro-82M")
mx.eval(model.parameters())

# Check F0 block attributes
print("\n=== MLX F0 blocks ===")
for i in range(3):
    block = getattr(model.predictor, f"F0_{i}")
    print(f"\nF0_{i} attributes:")
    print(f"  in_channels: {block.in_channels}")
    print(f"  out_channels: {block.out_channels}")
    print(f"  upsample: {block.upsample}")
    print(f"  downsample: {block.downsample}")
    print(f"  conv1x1: {'yes' if block.conv1x1 is not None else 'no'}")
    print(f"  pool: {'yes' if block.pool is not None else 'no'}")
    if block.pool is not None:
        print(f"    pool type: {type(block.pool)}")

# Test F0 block behavior
print("\n=== F0 block behavior test ===")
test_input = mx.random.normal((1, 168, 512))
speaker = mx.random.normal((1, 128))
mx.eval(test_input, speaker)

print(f"Input shape: {test_input.shape}")

x = test_input
for i in range(3):
    block = getattr(model.predictor, f"F0_{i}")
    x = block(x, speaker)
    mx.eval(x)
    print(
        f"After F0_{i}: shape={x.shape}, range=[{float(x.min()):.4f}, {float(x.max()):.4f}]"
    )

f0 = model.predictor.F0_proj(x).squeeze(-1)
mx.eval(f0)
print(
    f"After F0_proj: shape={f0.shape}, range=[{float(f0.min()):.4f}, {float(f0.max()):.4f}]"
)
