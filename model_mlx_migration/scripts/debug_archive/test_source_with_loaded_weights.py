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

"""Test SourceModule with correctly loaded weights."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "pytorch_to_mlx"))

import mlx.core as mx
from converters.kokoro_converter import KokoroConverter

print("=== Loading model with KokoroConverter ===")
converter = KokoroConverter()
model, config, pt_state = converter.load_from_hf("hexgrad/Kokoro-82M")
mx.eval(model.parameters())

# Get the generator's source module
source = model.decoder.generator.m_source

print("\n=== Loaded l_linear weights ===")
print(f"weight: {source.l_linear.weight}")
print(f"bias: {source.l_linear.bias}")

# Test with different F0 values
print("\n=== Testing source module with different F0 ===")
upp = 600  # Upsampling factor

for f0_val in [100, 200, 300, 400, 500]:
    f0 = mx.full((1, 10), float(f0_val))
    har, noise, uv = source(f0, upp)
    mx.eval(har, noise, uv)

    har_min = float(har.min())
    har_max = float(har.max())
    har_std = float(mx.std(har))
    har_mean = float(mx.mean(har))

    print(
        f"F0={f0_val}Hz: har range=[{har_min:.4f}, {har_max:.4f}], "
        f"mean={har_mean:.4f}, std={har_std:.4f}"
    )

# Now test through full synthesis
print("\n=== Testing full synthesis ===")
import torch

# Load voice
voice_path = Path.home() / "models" / "kokoro" / "voices" / "af_heart.pt"
voice_pack = torch.load(voice_path, map_location="cpu", weights_only=False)
style = mx.array(voice_pack[5].numpy())  # Use 6th style

# Simple tokens
tokens = mx.array([[0, 47, 44, 51, 51, 54, 0]])

# Run synthesis
audio = model.synthesize(tokens, style)
mx.eval(audio)

print(f"Audio shape: {audio.shape}")
print(f"Audio range: [{float(audio.min()):.4f}, {float(audio.max()):.4f}]")
print(f"Audio RMS: {float(mx.sqrt(mx.mean(audio**2))):.4f}")

# Check for clipping
clipped = (mx.abs(audio) > 0.99).astype(mx.float32)
clip_pct = float(mx.mean(clipped)) * 100
print(f"Clipping percentage: {clip_pct:.2f}%")
