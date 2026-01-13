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
Trace the stage just before conv_post to find magnitude divergence.
"""

import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.pytorch_to_mlx.converters import KokoroConverter


def main():
    ref = np.load("/tmp/kokoro_ref/tensors.npz")
    gen_traces = np.load("/tmp/kokoro_ref/generator_traces.npz")
    internal = np.load("/tmp/kokoro_ref/generator_internal_traces.npz")

    print("=== Loading MLX Model ===")
    converter = KokoroConverter()
    model, config, _ = converter.load_from_hf()
    mx.eval(model)

    F0_mx = mx.array(ref["F0_pred"])
    style_mx = mx.array(ref["style_128"])

    pt_gen_input = gen_traces["generator_input_ncl"]
    gen_input = mx.array(pt_gen_input).transpose(0, 2, 1)

    generator = model.decoder.generator

    # Check if we have conv_post_in trace
    if "conv_post_in" in internal:
        pt_pre_conv = internal["conv_post_in"]  # NCL
        print(f"\nPT conv_post_in shape: {pt_pre_conv.shape}")
        print(
            f"PT conv_post_in range: [{pt_pre_conv.min():.4f}, {pt_pre_conv.max():.4f}]"
        )
        print(f"PT conv_post_in std: {pt_pre_conv.std():.4f}")

    # Run MLX Generator
    x = gen_input

    total_upp = 1
    for r in generator.config.istft_upsample_rates:
        total_upp *= r
    total_upp *= generator.istft_hop_size

    har_source, noise_src, uv = generator.m_source(F0_mx, total_upp)
    har_source_1d = har_source.squeeze(-1)
    source = generator._source_stft(har_source_1d)
    mx.eval(source)

    print("\n=== Stage-by-stage Comparison ===")

    for i in range(generator.num_upsamples):
        up = getattr(generator, f"ups_{i}")
        noise_conv = getattr(generator, f"noise_convs_{i}")
        noise_res = getattr(generator, f"noise_res_{i}")

        x = nn.leaky_relu(x, 0.1)

        x_source = noise_conv(source)
        x_source = noise_res(x_source, style_mx)

        x = up(x)

        if i == generator.num_upsamples - 1:
            x = mx.concatenate([x[:, 1:2, :], x], axis=1)

        if x_source.shape[1] < x.shape[1]:
            pad_len = x.shape[1] - x_source.shape[1]
            x_source = mx.pad(x_source, [(0, 0), (0, pad_len), (0, 0)])
        elif x_source.shape[1] > x.shape[1]:
            x_source = x_source[:, : x.shape[1], :]

        # Check x_source contribution
        mx.eval([x, x_source])

        x_pre_add = x
        x = x + x_source

        xs = None
        for j in range(generator.num_kernels):
            block_idx = i * generator.num_kernels + j
            if block_idx < generator._num_resblocks:
                resblock = getattr(generator, f"resblocks_{block_idx}")
                if xs is None:
                    xs = resblock(x, style_mx)
                else:
                    xs = xs + resblock(x, style_mx)
        if xs is not None:
            x = xs / generator.num_kernels
        mx.eval(x)

        print(f"\nStage {i}:")
        print(
            f"  Main path (x): range [{float(mx.min(x_pre_add)):.4f}, {float(mx.max(x_pre_add)):.4f}], std={float(mx.std(x_pre_add)):.4f}"
        )
        print(
            f"  Noise path (x_source): range [{float(mx.min(x_source)):.4f}, {float(mx.max(x_source)):.4f}], std={float(mx.std(x_source)):.4f}"
        )
        print(
            f"  After resblocks: range [{float(mx.min(x)):.4f}, {float(mx.max(x)):.4f}], std={float(mx.std(x)):.4f}"
        )

        # Compare with PT if available
        key = f"noise_res_{i}_out"
        if key in internal:
            pt_noise = internal[key]  # NCL
            mlx_noise_ncl = np.array(x_source.transpose(0, 2, 1))
            min_len = min(mlx_noise_ncl.size, pt_noise.size)
            corr = np.corrcoef(
                mlx_noise_ncl.flatten()[:min_len], pt_noise.flatten()[:min_len]
            )[0, 1]
            print(f"  Noise path correlation with PT: {corr:.4f}")

    # Before conv_post: leaky_relu
    x = nn.leaky_relu(x, 0.1)
    mx.eval(x)

    print("\nBefore conv_post (after leaky_relu):")
    print(f"  MLX range: [{float(mx.min(x)):.4f}, {float(mx.max(x)):.4f}]")
    print(f"  MLX std: {float(mx.std(x)):.4f}")

    if "conv_post_in" in internal:
        pt_pre = internal["conv_post_in"]  # NCL
        mlx_pre_ncl = np.array(x.transpose(0, 2, 1))
        min_len = min(mlx_pre_ncl.size, pt_pre.size)
        corr = np.corrcoef(mlx_pre_ncl.flatten()[:min_len], pt_pre.flatten()[:min_len])[
            0, 1
        ]
        print(f"  Correlation with PT: {corr:.4f}")
        print(f"  PT range: [{pt_pre.min():.4f}, {pt_pre.max():.4f}]")
        print(f"  PT std: {pt_pre.std():.4f}")

        # Magnitude ratio
        ratio = float(mx.std(x)) / pt_pre.std()
        print(f"  Std ratio (MLX/PT): {ratio:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
