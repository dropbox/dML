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
Debug SourceModule weights and compare with PyTorch.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import math

import mlx.core as mx
import numpy as np
import torch


def debug_source_module():
    """Debug SourceModule weights and harmonic generation."""
    from tools.pytorch_to_mlx.converters import KokoroConverter

    print("=" * 60)
    print("SourceModule Weight Debug")
    print("=" * 60)

    # Load MLX model
    converter = KokoroConverter()
    model, config, _ = converter.load_from_hf()

    # Get generator's source module
    gen = model.decoder.generator
    m_source = gen.m_source

    print("\n=== MLX SourceModule Weights ===")
    print(f"l_linear.weight shape: {m_source.l_linear.weight.shape}")
    print(f"l_linear.weight: {m_source.l_linear.weight}")
    print(f"l_linear.bias shape: {m_source.l_linear.bias.shape}")
    print(f"l_linear.bias: {m_source.l_linear.bias}")

    # Generate harmonics manually to debug
    print("\n=== Harmonic Generation Debug ===")

    # Test F0 at 200Hz
    f0 = mx.full((1, 10), 200.0)  # 10 samples at 200Hz
    upp = 300  # Total upsample

    batch, length = f0.shape
    length * upp  # 3000

    # Upsample F0
    f0_up = mx.repeat(f0[:, :, None], upp, axis=1).squeeze(-1)  # [1, 3000]
    print(f"F0 upsampled: shape={f0_up.shape}, value={float(f0_up[0, 0])}")

    # F0 Hz (already in Hz)
    f0_hz = mx.maximum(f0_up, 0.0)

    # Generate harmonics
    sample_rate = 24000
    sine_amp = 0.1
    num_harmonics = 9

    harmonics = []
    for h in range(1, num_harmonics + 1):
        phase = mx.cumsum(f0_hz * h / sample_rate, axis=1) * 2 * math.pi
        sine = mx.sin(phase) * sine_amp
        harmonics.append(sine)
        if h <= 3:
            mx.eval(sine)
            print(
                f"  Harmonic {h}: range=[{float(sine.min()):.4f}, {float(sine.max()):.4f}]"
            )

    harmonics_stack = mx.stack(harmonics, axis=-1)  # [1, 3000, 9]
    mx.eval(harmonics_stack)
    print(
        f"Harmonics stack: shape={harmonics_stack.shape}, range=[{float(harmonics_stack.min()):.4f}, {float(harmonics_stack.max()):.4f}]"
    )

    # UV mask
    uv = (f0_hz > 0).astype(mx.float32)[:, :, None]
    harmonics_stack = harmonics_stack * uv

    # Apply linear + tanh
    har_source = mx.tanh(m_source.l_linear(harmonics_stack))
    mx.eval(har_source)
    print(
        f"After l_linear + tanh: shape={har_source.shape}, range=[{float(har_source.min()):.4f}, {float(har_source.max()):.4f}]"
    )

    # What if weights are wrong?
    print("\n=== Testing with Identity Linear ===")
    # With identity, each harmonic contributes equally
    identity_out = mx.sum(harmonics_stack, axis=-1, keepdims=True) / num_harmonics
    identity_tanh = mx.tanh(identity_out)
    mx.eval(identity_tanh)
    print(
        f"Identity sum/tanh: range=[{float(identity_tanh.min()):.4f}, {float(identity_tanh.max()):.4f}]"
    )

    # Compare with PyTorch reference implementation
    print("\n=== PyTorch Reference ===")

    # Load PyTorch weights directly
    weights_path = Path.home() / "models" / "kokoro" / "kokoro-v1_0.pth"
    if weights_path.exists():
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
        if "decoder" in state_dict:
            dec_state = state_dict["decoder"]

            for key in dec_state:
                if "m_source" in key and "l_linear" in key:
                    print(f"Found: {key}: {dec_state[key].shape}")

            # Get the actual keys
            weight_key = None
            bias_key = None
            for key in dec_state:
                if (
                    "generator.m_source.l_linear.weight" in key
                    or key == "generator.m_source.l_linear.weight"
                ):
                    weight_key = key
                if (
                    "generator.m_source.l_linear.bias" in key
                    or key == "generator.m_source.l_linear.bias"
                ):
                    bias_key = key

            if weight_key:
                pt_weight = dec_state[weight_key]
                print(f"\nPyTorch l_linear.weight: shape={pt_weight.shape}")
                print(f"PyTorch l_linear.weight: {pt_weight}")

            if bias_key:
                pt_bias = dec_state[bias_key]
                print(f"PyTorch l_linear.bias: shape={pt_bias.shape}")
                print(f"PyTorch l_linear.bias: {pt_bias}")

            # Test with PyTorch weights
            if weight_key and bias_key:
                print("\n=== Test with PyTorch Weights ===")
                # Convert harmonics to numpy/torch
                harmonics_np = np.array(harmonics_stack)
                harmonics_torch = torch.from_numpy(harmonics_np)

                # Apply PyTorch linear
                pt_out = torch.nn.functional.linear(harmonics_torch, pt_weight, pt_bias)
                pt_tanh = torch.tanh(pt_out)
                print(
                    f"PyTorch output: range=[{pt_tanh.min().item():.4f}, {pt_tanh.max().item():.4f}]"
                )
    else:
        print(f"PyTorch weights not found at {weights_path}")


def check_weight_loading():
    """Check if weights are being loaded correctly."""
    print("\n" + "=" * 60)
    print("Weight Loading Check")
    print("=" * 60)

    from tools.pytorch_to_mlx.converters import KokoroConverter

    converter = KokoroConverter()
    model, config, _ = converter.load_from_hf()

    # Check some key weights
    gen = model.decoder.generator

    print("\n=== Key Generator Weights ===")

    # m_source.l_linear
    w = gen.m_source.l_linear.weight
    b = gen.m_source.l_linear.bias
    mx.eval(w, b)
    print(f"m_source.l_linear.weight: shape={w.shape}, sum={float(mx.sum(w)):.6f}")
    print(f"m_source.l_linear.bias: shape={b.shape}, value={float(b):.6f}")

    # Check if weights are just initialized (not loaded)
    # MLX Linear default init is small values
    if abs(float(mx.sum(w))) < 0.01:
        print("WARNING: l_linear weights appear to be default initialized, not loaded!")

    # ups weights
    for i, up in enumerate(gen.ups):
        w = up.weight_v
        mx.eval(w)
        print(f"ups[{i}].weight_v: shape={w.shape}, sum={float(mx.sum(w)):.6f}")

    # noise_convs weights
    for i, nc in enumerate(gen.noise_convs):
        w = nc.weight
        mx.eval(w)
        print(f"noise_convs[{i}].weight: shape={w.shape}, sum={float(mx.sum(w)):.6f}")


if __name__ == "__main__":
    debug_source_module()
    check_weight_loading()
