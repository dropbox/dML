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

"""Check F0_1 pool (depthwise ConvTranspose) weights."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "pytorch_to_mlx"))

import mlx.core as mx
import torch

print("=== PyTorch F0.1.pool weights ===")
model_path = Path.home() / "models" / "kokoro" / "kokoro-v1_0.pth"
checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
pred_state = checkpoint.get("predictor", {})

pt_pool_w_g = pred_state["module.F0.1.pool.weight_g"]
pt_pool_w_v = pred_state["module.F0.1.pool.weight_v"]
pt_pool_b = pred_state["module.F0.1.pool.bias"]

print(f"weight_g: shape={pt_pool_w_g.shape}")
print(f"weight_v: shape={pt_pool_w_v.shape}")
print(f"bias: shape={pt_pool_b.shape}")

print(
    f"\nweight_v range: [{pt_pool_w_v.min().item():.6f}, {pt_pool_w_v.max().item():.6f}]"
)
print(f"weight_v[:5, 0, :]: {pt_pool_w_v[:5, 0, :].numpy()}")

# Load MLX model
print("\n=== Loading MLX model ===")
from converters.kokoro_converter import KokoroConverter

converter = KokoroConverter()
model, config, pt_state = converter.load_from_hf("hexgrad/Kokoro-82M")
mx.eval(model.parameters())

pool = model.predictor.F0_1.pool
print("\n=== MLX F0_1.pool weights ===")
print(f"weight_g shape: {pool.weight_g.shape}")
print(f"weight_v shape: {pool.weight_v.shape}")
print(f"bias shape: {pool.bias.shape}")

print(
    f"\nMLX weight_v range: [{float(pool.weight_v.min()):.6f}, {float(pool.weight_v.max()):.6f}]"
)

# Check the actual implementation
print("\n=== Check WeightNormConvTranspose1d implementation ===")
from converters.models.kokoro_modules import WeightNormConvTranspose1d

# Create a test
test_conv = WeightNormConvTranspose1d(
    in_channels=512,
    out_channels=512,
    kernel_size=3,
    stride=2,
    padding=1,
    output_padding=1,
    groups=512,
)
print("Created test ConvTranspose with groups=512")
print(f"Test weight_v shape: {test_conv.weight_v.shape}")

# The pool layer should be depthwise (groups = in_channels)
# Check if the loaded pool has groups attribute
print("\nChecking pool attributes:")
print(f"  Has groups: {hasattr(pool, 'groups')}")
if hasattr(pool, "groups"):
    print(f"  groups value: {pool.groups}")

# Test the pool behavior
print("\n=== Test pool behavior ===")
test_input = mx.random.normal((1, 168, 512))
mx.eval(test_input)
print(f"Input shape: {test_input.shape}")

# Run through pool
output = pool(test_input)
mx.eval(output)
print(f"Output shape: {output.shape}")
print("Expected shape: (1, 336, 512) for 2x upsample")

# Also test full F0_1 block
print("\n=== Test full F0_1 block ===")
speaker = mx.random.normal((1, 128))
mx.eval(speaker)

x = mx.random.normal((1, 168, 512)) * 0.1
mx.eval(x)
print(f"Input: shape={x.shape}, range=[{float(x.min()):.4f}, {float(x.max()):.4f}]")

# F0_1 should upsample 168 -> 336
out = model.predictor.F0_1(x, speaker)
mx.eval(out)
print(
    f"F0_1 output: shape={out.shape}, range=[{float(out.min()):.4f}, {float(out.max()):.4f}]"
)
