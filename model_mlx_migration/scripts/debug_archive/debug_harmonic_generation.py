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

"""Debug harmonic generation step by step."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "pytorch_to_mlx"))

import math

import mlx.core as mx

# Test parameters
sample_rate = 24000
sine_amp = 0.1
num_harmonics = 9

print("=== Step-by-step harmonic generation ===")

# F0 = 200 Hz, 10 time steps, upsample by 600
f0 = mx.full((1, 10), 200.0)
upp = 600

print(f"\nInput F0 shape: {f0.shape}, value: 200 Hz")
print(f"Upsampling factor: {upp}")

# Step 1: Upsample F0
f0_up = mx.repeat(f0[:, :, None], upp, axis=1).squeeze(-1)
f0_hz = mx.maximum(f0_up, 0.0)
mx.eval(f0_hz)

print("\nStep 1: Upsample F0")
print(f"  f0_up shape: {f0_up.shape}")
print(f"  f0_up[0, :10] (first 10 samples): {f0_up[0, :10].tolist()}")
print(f"  f0_hz range: [{float(f0_hz.min())}, {float(f0_hz.max())}]")

# Step 2: Generate harmonics
print("\nStep 2: Generate harmonics")
samples = 10 * upp  # 6000 samples

# Show what happens for harmonic 1
h = 1
# Phase accumulation: phase[t] = sum_{i=0}^{t} f0_hz[i] * h / sample_rate * 2*pi
# Since f0_hz is constant 200, phase[t] = (t+1) * 200 / 24000 * 2*pi

phase_rate = 200.0 * h / sample_rate  # = 0.00833... cycles per sample
print(f"  Phase rate for h={h}: {phase_rate} cycles/sample")
print(f"  Phase increment per sample: {phase_rate * 2 * math.pi} radians")

# Do cumsum
phase_per_sample = f0_hz * h / sample_rate  # [batch, samples]
mx.eval(phase_per_sample)
print(f"  phase_per_sample[0, :10]: {phase_per_sample[0, :10].tolist()}")

# Cumsum
cumulative_phase = mx.cumsum(phase_per_sample, axis=1)  # cycles
mx.eval(cumulative_phase)
print(f"  cumulative_phase[0, :10] (cycles): {cumulative_phase[0, :10].tolist()}")

# Convert to radians
phase_radians = cumulative_phase * 2 * math.pi
mx.eval(phase_radians)
print(f"  phase_radians[0, :10]: {phase_radians[0, :10].tolist()}")

# Sine
sine = mx.sin(phase_radians) * sine_amp
mx.eval(sine)
print(f"  sine[0, :10]: {sine[0, :10].tolist()}")
print(f"  sine range: [{float(sine.min())}, {float(sine.max())}]")
print(f"  sine std: {float(mx.std(sine))}")

# For 200 Hz, one period = 24000/200 = 120 samples
# So in 6000 samples, we should see 50 complete periods
print(f"\n  Expected periods in 6000 samples: {6000 / 120}")
print(f"  Expected sine range: [-{sine_amp}, {sine_amp}]")

# Check if sin is working correctly
test_angles = mx.array([0, math.pi / 2, math.pi, 3 * math.pi / 2, 2 * math.pi])
test_sin = mx.sin(test_angles)
mx.eval(test_sin)
print(f"\nSanity check - sin([0, pi/2, pi, 3pi/2, 2pi]): {test_sin.tolist()}")

# Now generate all harmonics
print(f"\nStep 3: Generate all {num_harmonics} harmonics")
harmonics = []
for h in range(1, num_harmonics + 1):
    phase = mx.cumsum(f0_hz * h / sample_rate, axis=1) * 2 * math.pi
    sine = mx.sin(phase) * sine_amp
    mx.eval(sine)
    harmonics.append(sine)
    print(
        f"  Harmonic {h}f: range=[{float(sine.min()):.4f}, {float(sine.max()):.4f}], "
        f"std={float(mx.std(sine)):.4f}"
    )

# Stack harmonics
harmonics_stack = mx.stack(harmonics, axis=-1)
mx.eval(harmonics_stack)
print(f"\nStacked harmonics shape: {harmonics_stack.shape}")

# Apply UV mask
uv = (f0_hz > 0).astype(mx.float32)[:, :, None]
harmonics_masked = harmonics_stack * uv
mx.eval(harmonics_masked)
print(f"UV mask mean (should be 1.0 for all voiced): {float(mx.mean(uv))}")

# Apply l_linear + tanh (using actual weights)
# Load weights
from converters.kokoro_converter import KokoroConverter

converter = KokoroConverter()
model, config, pt_state = converter.load_from_hf("hexgrad/Kokoro-82M")
mx.eval(model.parameters())

l_linear = model.decoder.generator.m_source.l_linear

print("\nStep 4: Apply l_linear")
print(f"  l_linear weight: {l_linear.weight}")
print(f"  l_linear bias: {l_linear.bias}")

# Apply linear: [batch, samples, 9] @ [9, 1] + [1] -> [batch, samples, 1]
# Note: nn.Linear in MLX does x @ W.T + b
linear_out = l_linear(harmonics_masked)
mx.eval(linear_out)
print(f"  Linear output range: [{float(linear_out.min())}, {float(linear_out.max())}]")
print(f"  Linear output std: {float(mx.std(linear_out))}")

# Apply tanh
tanh_out = mx.tanh(linear_out)
mx.eval(tanh_out)
print("\nStep 5: Apply tanh")
print(f"  Tanh output range: [{float(tanh_out.min())}, {float(tanh_out.max())}]")
print(f"  Tanh output std: {float(mx.std(tanh_out))}")

# Check some samples
print(f"\n  tanh_out[0, :10, 0]: {tanh_out[0, :10, 0].tolist()}")

# Compare with what source module does
print("\n=== Compare with source module ===")
source = model.decoder.generator.m_source
har, noise, uv_out = source(f0, upp)
mx.eval(har)
print(f"Source module output range: [{float(har.min())}, {float(har.max())}]")
print(f"Source module output std: {float(mx.std(har))}")
print(f"har[0, :10, 0]: {har[0, :10, 0].tolist()}")
