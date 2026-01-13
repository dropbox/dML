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

"""Debug SourceModule l_linear weights and harmonic generation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "pytorch_to_mlx"))

import math

import mlx.core as mx
import torch


def debug_pytorch_source_weights():
    """Check PyTorch source module weights."""
    print("\n=== PyTorch Source Module Weights ===")
    model_path = Path.home() / "models" / "kokoro" / "kokoro-v1_0.pth"
    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)

    # Find l_linear weights
    for key in state_dict.keys():
        if "l_linear" in key:
            weight = state_dict[key]
            print(f"{key}: shape={weight.shape}, dtype={weight.dtype}")
            print(f"  range: [{weight.min().item():.6f}, {weight.max().item():.6f}]")
            print(f"  mean: {weight.mean().item():.6f}")
            if weight.numel() < 20:
                print(f"  values: {weight.numpy().flatten()}")


def debug_raw_harmonics():
    """Test raw harmonic generation without l_linear."""
    print("\n=== Raw Harmonic Generation Test ===")

    sample_rate = 24000
    sine_amp = 0.1
    num_harmonics = 9

    # Test F0 = 200 Hz
    f0 = mx.full((1, 10), 200.0)
    upp = 600

    # Upsample F0
    f0_up = mx.repeat(f0[:, :, None], upp, axis=1).squeeze(-1)
    f0_hz = mx.maximum(f0_up, 0.0)

    print(f"F0 upsampled shape: {f0_up.shape}")
    print(f"F0 Hz range: [{float(f0_hz.min()):.2f}, {float(f0_hz.max()):.2f}]")

    # Generate individual harmonics
    10 * upp  # 6000 samples
    for h in [1, 3, 9]:
        phase = mx.cumsum(f0_hz * h / sample_rate, axis=1) * 2 * math.pi
        sine = mx.sin(phase) * sine_amp
        mx.eval(sine)
        print(
            f"Harmonic {h}f: range=[{float(sine.min()):.4f}, {float(sine.max()):.4f}], "
            f"std={float(mx.std(sine)):.4f}"
        )

    # Generate all harmonics and stack
    harmonics = []
    for h in range(1, num_harmonics + 1):
        phase = mx.cumsum(f0_hz * h / sample_rate, axis=1) * 2 * math.pi
        sine = mx.sin(phase) * sine_amp
        harmonics.append(sine)

    harmonics_stack = mx.stack(harmonics, axis=-1)
    mx.eval(harmonics_stack)

    print(f"\nStacked harmonics shape: {harmonics_stack.shape}")
    print(
        f"Stacked range: [{float(harmonics_stack.min()):.4f}, {float(harmonics_stack.max()):.4f}]"
    )

    # Check individual harmonic channels
    for i in range(num_harmonics):
        h_slice = harmonics_stack[:, :, i]
        print(
            f"  Harmonic {i + 1}: mean={float(mx.mean(h_slice)):.6f}, std={float(mx.std(h_slice)):.6f}"
        )

    # Test with random weights (what MLX is using now)
    l_linear_rand = mx.random.normal((1, num_harmonics)) * 0.1
    l_linear_bias = mx.zeros((1,))

    combined_rand = mx.tanh(harmonics_stack @ l_linear_rand.T + l_linear_bias)
    mx.eval(combined_rand)
    print(
        f"\nWith random l_linear: range=[{float(combined_rand.min()):.4f}, {float(combined_rand.max()):.4f}]"
    )

    # Load actual PyTorch weights and test
    model_path = Path.home() / "models" / "kokoro" / "kokoro-v1_0.pth"
    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)

    # Find the actual l_linear weights
    l_linear_weight = None
    l_linear_bias_pt = None
    for key in state_dict.keys():
        if "decoder.generator.m_source.l_linear.weight" in key:
            l_linear_weight = state_dict[key]
        if "decoder.generator.m_source.l_linear.bias" in key:
            l_linear_bias_pt = state_dict[key]

    if l_linear_weight is not None:
        print("\n=== Using actual PyTorch weights ===")
        print(f"l_linear weight shape: {l_linear_weight.shape}")
        print(f"l_linear weight: {l_linear_weight.numpy()}")

        # Convert to MLX
        w_mlx = mx.array(l_linear_weight.numpy())  # [1, 9]
        b_mlx = (
            mx.array(l_linear_bias_pt.numpy())
            if l_linear_bias_pt is not None
            else mx.zeros((1,))
        )

        # Apply
        # harmonics_stack: [batch, samples, 9]
        # w_mlx: [1, 9]
        # Result should be: [batch, samples, 1]
        combined = mx.tanh(harmonics_stack @ w_mlx.T + b_mlx)
        mx.eval(combined)

        print(
            f"With actual weights: range=[{float(combined.min()):.4f}, {float(combined.max()):.4f}]"
        )
        print(f"Combined std: {float(mx.std(combined)):.4f}")

        # Test at multiple F0 values
        print("\nF0 sweep with actual weights:")
        for f0_val in [100, 200, 300, 400, 500]:
            f0_test = mx.full((1, 10), float(f0_val))
            f0_up_test = mx.repeat(f0_test[:, :, None], upp, axis=1).squeeze(-1)

            harmonics_test = []
            for h in range(1, num_harmonics + 1):
                phase = mx.cumsum(f0_up_test * h / sample_rate, axis=1) * 2 * math.pi
                sine = mx.sin(phase) * sine_amp
                harmonics_test.append(sine)

            harmonics_test_stack = mx.stack(harmonics_test, axis=-1)
            combined_test = mx.tanh(harmonics_test_stack @ w_mlx.T + b_mlx)
            mx.eval(combined_test)

            print(
                f"  F0={f0_val}Hz: range=[{float(combined_test.min()):.4f}, {float(combined_test.max()):.4f}], "
                f"std={float(mx.std(combined_test)):.4f}"
            )

    else:
        print("ERROR: Could not find l_linear weights in state_dict")


def check_source_weight_loading():
    """Check if MLX SourceModule loads weights correctly."""
    print("\n=== MLX SourceModule Weight Loading Check ===")

    from converters.models.kokoro import SourceModule

    source = SourceModule(sample_rate=24000, num_harmonics=9)

    # Check initial weights (should be random)
    print(f"l_linear weight shape: {source.l_linear.weight.shape}")
    print(f"l_linear initial weight: {source.l_linear.weight}")
    print(f"l_linear initial bias: {source.l_linear.bias}")

    # Check if weights are what we expect (0 mean, small values)
    w_mean = float(mx.mean(source.l_linear.weight))
    w_std = float(mx.std(source.l_linear.weight))
    print(f"l_linear weight stats: mean={w_mean:.6f}, std={w_std:.6f}")


def main():
    debug_pytorch_source_weights()
    debug_raw_harmonics()
    check_source_weight_loading()


if __name__ == "__main__":
    main()
