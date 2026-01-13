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

"""Check resblock alpha and other weights."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "pytorch_to_mlx"))

import mlx.core as mx
import torch
from converters.kokoro_converter import KokoroConverter

print("=== Loading PyTorch weights ===")
model_path = Path.home() / "models" / "kokoro" / "kokoro-v1_0.pth"
checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
dec_state = checkpoint.get("decoder", {})

print("\n=== PyTorch resblocks.0 alpha values ===")
for key in sorted(dec_state.keys()):
    if "resblocks.0.alpha" in key:
        val = dec_state[key]
        print(
            f"{key}: shape={val.shape}, range=[{val.min().item():.4f}, {val.max().item():.4f}]"
        )
        if val.numel() < 10:
            print(f"  values: {val.flatten().numpy()[:10]}")

print("\n=== PyTorch noise_res.0 alpha values ===")
for key in sorted(dec_state.keys()):
    if "noise_res.0.alpha" in key:
        val = dec_state[key]
        print(
            f"{key}: shape={val.shape}, range=[{val.min().item():.4f}, {val.max().item():.4f}]"
        )

print("\n=== Loading MLX model ===")
converter = KokoroConverter()
model, config, pt_state = converter.load_from_hf("hexgrad/Kokoro-82M")
mx.eval(model.parameters())

gen = model.decoder.generator

print("\n=== MLX resblocks_0 alpha values ===")
resblock = gen.resblocks_0
for i in range(3):
    alpha1 = getattr(resblock, f"alpha1_{i}")
    alpha2 = getattr(resblock, f"alpha2_{i}")
    mx.eval(alpha1, alpha2)
    print(
        f"alpha1_{i}: shape={alpha1.shape}, range=[{float(alpha1.min()):.4f}, {float(alpha1.max()):.4f}]"
    )
    print(
        f"alpha2_{i}: shape={alpha2.shape}, range=[{float(alpha2.min()):.4f}, {float(alpha2.max()):.4f}]"
    )

print("\n=== MLX noise_res_0 alpha values ===")
noise_res = gen.noise_res_0
for i in range(3):
    alpha1 = getattr(noise_res, f"alpha1_{i}")
    alpha2 = getattr(noise_res, f"alpha2_{i}")
    mx.eval(alpha1, alpha2)
    print(
        f"alpha1_{i}: shape={alpha1.shape}, range=[{float(alpha1.min()):.4f}, {float(alpha1.max()):.4f}]"
    )
    print(
        f"alpha2_{i}: shape={alpha2.shape}, range=[{float(alpha2.min()):.4f}, {float(alpha2.max()):.4f}]"
    )

# Compare with PyTorch
print("\n=== Direct comparison ===")
pt_alpha = dec_state["module.generator.resblocks.0.alpha1.0"]
mlx_alpha = gen.resblocks_0.alpha1_0
print(f"PyTorch alpha1.0: {pt_alpha.flatten()[:5].numpy()}")
print(f"MLX alpha1_0: {mlx_alpha.flatten()[:5].tolist()}")
