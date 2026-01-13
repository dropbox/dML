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

"""Check F0 projection weights."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "pytorch_to_mlx"))

import mlx.core as mx
import torch

print("=== PyTorch F0_proj weights ===")
model_path = Path.home() / "models" / "kokoro" / "kokoro-v1_0.pth"
checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

pred_state = checkpoint.get("predictor", {})
for key in sorted(pred_state.keys()):
    if "F0_proj" in key:
        val = pred_state[key]
        print(f"{key}: shape={val.shape}")
        print(f"  range: [{val.min().item():.6f}, {val.max().item():.6f}]")
        print(f"  mean: {val.mean().item():.6f}")
        if val.numel() < 20:
            print(f"  values: {val.flatten().numpy()}")

print("\n=== Loading MLX model ===")
from converters.kokoro_converter import KokoroConverter

converter = KokoroConverter()
model, config, pt_state = converter.load_from_hf("hexgrad/Kokoro-82M")
mx.eval(model.parameters())

print("\n=== MLX F0_proj weights ===")
predictor = model.predictor
print(f"F0_proj weight shape: {predictor.F0_proj.weight.shape}")
print(f"F0_proj weight: {predictor.F0_proj.weight}")
print(f"F0_proj bias shape: {predictor.F0_proj.bias.shape}")
print(f"F0_proj bias: {predictor.F0_proj.bias}")

# Compare
pt_weight = pred_state["module.F0_proj.weight"]
pt_bias = pred_state["module.F0_proj.bias"]

print("\n=== Comparison ===")
print(f"PyTorch F0_proj.weight: {pt_weight.flatten().numpy()[:5]}")
mlx_w = predictor.F0_proj.weight.flatten()[:5].tolist()
print(f"MLX F0_proj.weight: {mlx_w}")

print(f"\nPyTorch F0_proj.bias: {pt_bias.numpy()}")
print(f"MLX F0_proj.bias: {predictor.F0_proj.bias.tolist()}")

# Also check F0 block weights
print("\n=== F0_0 conv1 weights ===")
pt_f0_0_conv1 = pred_state.get("module.F0.0.conv1.weight_v")
if pt_f0_0_conv1 is not None:
    print(f"PyTorch shape: {pt_f0_0_conv1.shape}")
    print(
        f"PyTorch range: [{pt_f0_0_conv1.min().item():.6f}, {pt_f0_0_conv1.max().item():.6f}]"
    )

# Check MLX F0_0 weights
f0_0 = predictor.F0_0
print(f"\nMLX F0_0.conv1.weight_v shape: {f0_0.conv1.weight_v.shape}")
print(
    f"MLX F0_0.conv1.weight_v range: [{float(f0_0.conv1.weight_v.min()):.6f}, {float(f0_0.conv1.weight_v.max()):.6f}]"
)
