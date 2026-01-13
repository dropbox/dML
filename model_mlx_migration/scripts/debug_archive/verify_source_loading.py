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

"""Verify that m_source weights are correctly loaded from checkpoint."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "pytorch_to_mlx"))

import mlx.core as mx
import torch

# Load PyTorch checkpoint
model_path = Path.home() / "models" / "kokoro" / "kokoro-v1_0.pth"
checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

print("=== PyTorch checkpoint structure ===")
print(f"Top-level keys: {list(checkpoint.keys())}")

dec_state = checkpoint.get("decoder", {})
print(f"\ndecoder dict has {len(dec_state)} keys")

# Check for m_source weights
l_linear_w_key = "module.generator.m_source.l_linear.weight"
l_linear_b_key = "module.generator.m_source.l_linear.bias"

print("\n=== m_source.l_linear weights ===")
if l_linear_w_key in dec_state:
    w = dec_state[l_linear_w_key]
    print(f"Weight found: shape={w.shape}")
    print(f"Weight values: {w.numpy()}")
else:
    print(f"WARNING: {l_linear_w_key} NOT FOUND in decoder dict")

if l_linear_b_key in dec_state:
    b = dec_state[l_linear_b_key]
    print(f"Bias found: shape={b.shape}")
    print(f"Bias values: {b.numpy()}")
else:
    print(f"WARNING: {l_linear_b_key} NOT FOUND in decoder dict")

# Now try loading via KokoroConverter
print("\n=== Loading via KokoroConverter ===")
from converters.kokoro_converter import KokoroConverter

converter = KokoroConverter()

# Load model using load_from_hf
print("Loading model...")
model, config, pt_state = converter.load_from_hf("hexgrad/Kokoro-82M")
mx.eval(model.parameters())

# Check m_source weights
gen = model.decoder.generator
print("\nGenerator m_source.l_linear weight after loading:")
print(f"  shape: {gen.m_source.l_linear.weight.shape}")
print(f"  values: {gen.m_source.l_linear.weight}")

print("\nGenerator m_source.l_linear bias after loading:")
print(f"  shape: {gen.m_source.l_linear.bias.shape}")
print(f"  values: {gen.m_source.l_linear.bias}")

# Compare with PyTorch
if l_linear_w_key in dec_state:
    pt_w = dec_state[l_linear_w_key].numpy()
    mlx_w = gen.m_source.l_linear.weight
    mx.eval(mlx_w)
    mlx_w_np = mlx_w.__array__().flatten()
    pt_w_flat = pt_w.flatten()

    print("\n=== Comparison ===")
    print(f"PyTorch weight: {pt_w_flat}")
    print(f"MLX weight: {mlx_w_np}")
    print(f"Match: {(abs(pt_w_flat - mlx_w_np) < 1e-6).all()}")
