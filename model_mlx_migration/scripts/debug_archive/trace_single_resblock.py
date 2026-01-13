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
Trace a single resblock with actual input values.
"""

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.pytorch_to_mlx.converters import KokoroConverter


def main():
    # Load traces
    traces_path = Path("/tmp/kokoro_ref/generator_internal_traces.npz")
    if not traces_path.exists():
        print("No internal traces")
        return 1
    pt_traces = np.load(traces_path)

    print("=== Loading MLX Model ===")
    converter = KokoroConverter()
    model, config, _ = converter.load_from_hf()
    mx.eval(model)

    generator = model.decoder.generator

    # Load reference tensors for style
    ref = np.load("/tmp/kokoro_ref/tensors.npz")
    style_128 = ref["style_128"]
    style_mx = mx.array(style_128)

    # Get resblock_0 input from PyTorch traces
    if "resblock_0_in" not in pt_traces:
        print("No resblock_0_in trace")
        return 1

    resblock_0_in = pt_traces["resblock_0_in"]  # [1, 256, 1260] NCL
    resblock_0_out = pt_traces["resblock_0_out"]  # [1, 256, 1260] NCL

    print(
        f"PT resblock_0_in: {resblock_0_in.shape}, range [{resblock_0_in.min():.4f}, {resblock_0_in.max():.4f}]"
    )
    print(
        f"PT resblock_0_out: {resblock_0_out.shape}, range [{resblock_0_out.min():.4f}, {resblock_0_out.max():.4f}]"
    )

    # Convert to MLX NLC
    x = mx.array(resblock_0_in).transpose(0, 2, 1)  # [1, 1260, 256]
    mx.eval(x)
    print(
        f"\nMLX input: {x.shape}, range [{float(mx.min(x)):.4f}, {float(mx.max(x)):.4f}]"
    )

    # Run through resblocks_0
    resblock = generator.resblocks_0
    out = resblock(x, style_mx)
    mx.eval(out)
    print(
        f"MLX output: {out.shape}, range [{float(mx.min(out)):.4f}, {float(mx.max(out)):.4f}]"
    )

    # Convert to NCL for comparison
    out_ncl = np.array(out.transpose(0, 2, 1))

    corr = np.corrcoef(out_ncl.flatten(), resblock_0_out.flatten())[0, 1]
    print(f"\nCorrelation: {corr:.6f}")
    print(f"Max diff: {np.abs(out_ncl - resblock_0_out).max():.6e}")

    # Check intermediate values
    print("\n=== Detailed trace through resblock ===")
    x_trace = mx.array(resblock_0_in).transpose(0, 2, 1)
    channels = 256

    for i in range(3):
        adain1 = getattr(resblock, f"adain1_{i}")
        adain2 = getattr(resblock, f"adain2_{i}")
        alpha1 = getattr(resblock, f"alpha1_{i}")
        alpha2 = getattr(resblock, f"alpha2_{i}")
        conv1 = getattr(resblock, f"convs1_{i}")
        conv2 = getattr(resblock, f"convs2_{i}")

        # AdaIN1
        style_out = adain1(style_mx)
        gamma = style_out[:, :channels][:, None, :]
        beta = style_out[:, channels:][:, None, :]
        mean = mx.mean(x_trace, axis=1, keepdims=True)
        var = mx.var(x_trace, axis=1, keepdims=True)
        xt = (x_trace - mean) / mx.sqrt(var + 1e-5)
        xt = (1 + gamma) * xt + beta
        mx.eval(xt)
        print(
            f"Layer {i} after adain1: [{float(mx.min(xt)):.4f}, {float(mx.max(xt)):.4f}]"
        )

        # Snake1
        alpha1_nlc = alpha1.transpose(0, 2, 1)
        xt = xt + (1 / alpha1_nlc) * (mx.sin(alpha1_nlc * xt) ** 2)
        mx.eval(xt)
        print(
            f"Layer {i} after snake1: [{float(mx.min(xt)):.4f}, {float(mx.max(xt)):.4f}]"
        )

        # Conv1
        xt = conv1(xt)
        mx.eval(xt)
        print(
            f"Layer {i} after conv1: [{float(mx.min(xt)):.4f}, {float(mx.max(xt)):.4f}]"
        )

        # AdaIN2
        style_out = adain2(style_mx)
        gamma = style_out[:, :channels][:, None, :]
        beta = style_out[:, channels:][:, None, :]
        mean = mx.mean(xt, axis=1, keepdims=True)
        var = mx.var(xt, axis=1, keepdims=True)
        xt = (xt - mean) / mx.sqrt(var + 1e-5)
        xt = (1 + gamma) * xt + beta
        mx.eval(xt)
        print(
            f"Layer {i} after adain2: [{float(mx.min(xt)):.4f}, {float(mx.max(xt)):.4f}]"
        )

        # Snake2
        alpha2_nlc = alpha2.transpose(0, 2, 1)
        xt = xt + (1 / alpha2_nlc) * (mx.sin(alpha2_nlc * xt) ** 2)
        mx.eval(xt)
        print(
            f"Layer {i} after snake2: [{float(mx.min(xt)):.4f}, {float(mx.max(xt)):.4f}]"
        )

        # Conv2
        xt = conv2(xt)
        mx.eval(xt)
        print(
            f"Layer {i} after conv2: [{float(mx.min(xt)):.4f}, {float(mx.max(xt)):.4f}]"
        )

        # Residual
        x_trace = xt + x_trace
        mx.eval(x_trace)
        print(
            f"Layer {i} after residual: [{float(mx.min(x_trace)):.4f}, {float(mx.max(x_trace)):.4f}]"
        )
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
