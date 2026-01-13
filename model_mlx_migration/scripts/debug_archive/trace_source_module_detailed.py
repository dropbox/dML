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
Detailed trace of SourceModule to find divergence.
"""

import math
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    # Load reference
    ref_path = Path("/tmp/kokoro_ref/tensors.npz")
    ref = np.load(ref_path)

    internal_path = Path("/tmp/kokoro_ref/generator_internal_traces.npz")
    internal = np.load(internal_path)

    # Get F0
    F0_np = ref["F0_pred"]  # [1, 126]
    print(f"F0 shape: {F0_np.shape}")
    print(f"F0 range: [{F0_np.min():.4f}, {F0_np.max():.4f}]")
    print(f"F0 first 10 values: {F0_np[0, :10]}")
    print(f"F0 negative values: {np.sum(F0_np < 0)} / {F0_np.size}")

    # Get PyTorch output for comparison
    pt_har = internal["m_source_out_0"]  # [1, 37800, 1]
    print(f"\nPyTorch har_source shape: {pt_har.shape}")
    print(f"PyTorch har_source range: [{pt_har.min():.6f}, {pt_har.max():.6f}]")
    print(f"PyTorch har_source mean: {pt_har.mean():.6f}")
    print(f"PyTorch har_source std: {pt_har.std():.6f}")

    # Get l_linear weights from checkpoint
    from huggingface_hub import hf_hub_download

    ckpt_path = hf_hub_download("hexgrad/Kokoro-82M", "kokoro-v1_0.pth")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    dec_state = ckpt["decoder"]

    l_linear_w = dec_state["module.generator.m_source.l_linear.weight"].numpy()
    l_linear_b = dec_state["module.generator.m_source.l_linear.bias"].numpy()
    print(f"\nl_linear weight shape: {l_linear_w.shape}")  # [1, 9]
    print(f"l_linear bias shape: {l_linear_b.shape}")  # [1]

    # Now run MLX implementation step by step
    print("\n=== MLX Source Module Step by Step ===")

    F0_mx = mx.array(F0_np)
    batch, length = F0_mx.shape
    upp = 300
    samples = length * upp
    sample_rate = 24000
    sine_amp = 0.1
    num_harmonics = 9

    print(f"batch={batch}, length={length}, upp={upp}, samples={samples}")

    # F0 upsample - nearest neighbor
    # PyTorch: F.interpolate(f0.unsqueeze(1), scale_factor=upp, mode='nearest').squeeze(1)
    f0_up = mx.repeat(F0_mx[:, :, None], upp, axis=1).squeeze(-1)
    mx.eval(f0_up)
    print(
        f"\nf0_up shape: {f0_up.shape}, range: [{float(mx.min(f0_up)):.4f}, {float(mx.max(f0_up)):.4f}]"
    )

    # UV mask
    uv = (f0_up > 10.0).astype(mx.float32)
    mx.eval(uv)
    print(f"uv shape: {uv.shape}, sum: {float(mx.sum(uv))}")

    # Generate harmonics with anti-aliasing
    print("\n--- Generating harmonics ---")
    harmonics = []

    for h in range(1, num_harmonics + 1):
        # rad_values = (f0 * h / sample_rate) % 1.0
        rad_values = (f0_up * h / sample_rate) % 1.0  # [batch, samples]

        # Downsample with linear interpolation
        t_down = mx.arange(length) * (samples - 1) / (length - 1)
        t_floor = mx.floor(t_down).astype(mx.int32)
        t_ceil = mx.minimum(t_floor + 1, samples - 1)
        t_frac = t_down - t_floor.astype(mx.float32)

        rad_floor = rad_values[:, t_floor]
        rad_ceil = rad_values[:, t_ceil]
        rad_values_down = rad_floor * (1 - t_frac) + rad_ceil * t_frac

        # Cumsum at low rate
        phase_low = mx.cumsum(rad_values_down, axis=1) * 2 * math.pi

        # Scale and upsample
        phase_scaled = phase_low * upp

        # Upsample with linear interpolation
        t_up = mx.arange(samples) * (length - 1) / (samples - 1)
        t_floor = mx.floor(t_up).astype(mx.int32)
        t_ceil = mx.minimum(t_floor + 1, length - 1)
        t_frac = t_up - t_floor.astype(mx.float32)

        phase_floor = phase_scaled[:, t_floor]
        phase_ceil = phase_scaled[:, t_ceil]
        phase = phase_floor * (1 - t_frac) + phase_ceil * t_frac

        # Sine
        sine = mx.sin(phase) * sine_amp
        harmonics.append(sine)
        mx.eval(sine)

        if h <= 3:
            print(
                f"Harmonic {h}: range [{float(mx.min(sine)):.6f}, {float(mx.max(sine)):.6f}]"
            )

    # Stack
    harmonics_stack = mx.stack(harmonics, axis=-1)  # [batch, samples, 9]
    mx.eval(harmonics_stack)
    print(f"\nharmonics_stack shape: {harmonics_stack.shape}")
    print(
        f"harmonics_stack range: [{float(mx.min(harmonics_stack)):.6f}, {float(mx.max(harmonics_stack)):.6f}]"
    )

    # Apply UV mask
    uv_expanded = uv[:, :, None]
    sine_waves = harmonics_stack * uv_expanded
    mx.eval(sine_waves)
    print(
        f"sine_waves (after UV) range: [{float(mx.min(sine_waves)):.6f}, {float(mx.max(sine_waves)):.6f}]"
    )

    # l_linear + tanh
    import mlx.nn as nn

    l_linear = nn.Linear(num_harmonics, 1)
    l_linear.weight = mx.array(l_linear_w)
    l_linear.bias = mx.array(l_linear_b)

    har_source = mx.tanh(l_linear(sine_waves))  # [batch, samples, 1]
    mx.eval(har_source)
    print(f"\nhar_source shape: {har_source.shape}")
    print(
        f"har_source range: [{float(mx.min(har_source)):.6f}, {float(mx.max(har_source)):.6f}]"
    )
    print(f"har_source mean: {float(mx.mean(har_source)):.6f}")
    print(f"har_source std: {float(mx.std(har_source)):.6f}")

    # Compare
    mlx_har = np.array(har_source)
    corr = np.corrcoef(mlx_har.flatten(), pt_har.flatten())[0, 1]
    diff = np.abs(mlx_har - pt_har)
    print("\n=== Comparison ===")
    print(f"Max diff: {diff.max():.6f}")
    print(f"Mean diff: {diff.mean():.6f}")
    print(f"Correlation: {corr:.6f}")

    # Check specific samples
    print(f"\nFirst 10 MLX values: {mlx_har[0, :10, 0]}")
    print(f"First 10 PT values: {pt_har[0, :10, 0]}")

    # Check if the issue is random phase
    print("\n=== Checking PyTorch SineGen behavior ===")
    print("PyTorch uses random initial phase for harmonics > 1")
    print("This would cause significant divergence even with correct logic")

    # Let's see what we get with just harmonic 1 (no random phase)
    corr_h1 = np.corrcoef(np.array(harmonics[0]).flatten(), pt_har[:, :, 0].flatten())[
        0, 1
    ]
    print(f"\nHarmonic 1 only vs full PT output correlation: {corr_h1:.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
