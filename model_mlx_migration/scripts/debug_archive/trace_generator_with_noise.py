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
Trace Generator WITH noise path to find actual divergence.
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

    pt_gen_input_ncl = gen_traces["generator_input_ncl"]
    gen_input = mx.array(pt_gen_input_ncl).transpose(0, 2, 1)

    generator = model.decoder.generator

    x = gen_input

    # WITH noise path
    print("\n=== Trace WITH Noise Path ===")

    total_upp = 1
    for r in generator.config.istft_upsample_rates:
        total_upp *= r
    total_upp *= generator.istft_hop_size

    har_source, noise_src, uv = generator.m_source(F0_mx, total_upp)
    har_source_1d = har_source.squeeze(-1)
    source = generator._source_stft(har_source_1d)
    mx.eval(source)

    print(f"source STFT: {source.shape}")
    print(f"  range: [{float(mx.min(source)):.4f}, {float(mx.max(source)):.4f}]")

    for i in range(generator.num_upsamples):
        print(f"\n--- Stage {i} ---")

        up = getattr(generator, f"ups_{i}")
        noise_conv = getattr(generator, f"noise_convs_{i}")
        noise_res = getattr(generator, f"noise_res_{i}")

        x = nn.leaky_relu(x, 0.1)
        mx.eval(x)

        # Noise path
        x_source = noise_conv(source)
        mx.eval(x_source)
        key = f"noise_conv_{i}_out"
        if key in internal:
            pt_val = internal[key]
            mlx_ncl = np.array(x_source.transpose(0, 2, 1))
            min_len = min(mlx_ncl.size, pt_val.size)
            corr = np.corrcoef(mlx_ncl.flatten()[:min_len], pt_val.flatten()[:min_len])[
                0, 1
            ]
            print(f"noise_conv_{i}: corr={corr:.6f}")

        x_source = noise_res(x_source, style_mx)
        mx.eval(x_source)
        key = f"noise_res_{i}_out"
        if key in internal:
            pt_val = internal[key]
            mlx_ncl = np.array(x_source.transpose(0, 2, 1))
            min_len = min(mlx_ncl.size, pt_val.size)
            corr = np.corrcoef(mlx_ncl.flatten()[:min_len], pt_val.flatten()[:min_len])[
                0, 1
            ]
            print(f"noise_res_{i}: corr={corr:.6f}")

        # Upsample main
        x = up(x)
        mx.eval(x)
        key = f"ups_{i}_out"
        if key in internal:
            pt_val = internal[key]
            mlx_ncl = np.array(x.transpose(0, 2, 1))
            min_len = min(mlx_ncl.size, pt_val.size)
            corr = np.corrcoef(mlx_ncl.flatten()[:min_len], pt_val.flatten()[:min_len])[
                0, 1
            ]
            print(f"ups_{i}_out: corr={corr:.6f}")

        # Reflection pad
        if i == generator.num_upsamples - 1:
            x = mx.concatenate([x[:, 1:2, :], x], axis=1)
            mx.eval(x)

        # Match lengths
        if x_source.shape[1] < x.shape[1]:
            pad_len = x.shape[1] - x_source.shape[1]
            x_source = mx.pad(x_source, [(0, 0), (0, pad_len), (0, 0)])
        elif x_source.shape[1] > x.shape[1]:
            x_source = x_source[:, : x.shape[1], :]

        # Add noise to main
        x = x + x_source
        mx.eval(x)

        key = f"after_source_add_{i}"
        if key in internal:
            pt_val = internal[key]
            mlx_ncl = np.array(x.transpose(0, 2, 1))
            min_len = min(mlx_ncl.size, pt_val.size)
            corr = np.corrcoef(mlx_ncl.flatten()[:min_len], pt_val.flatten()[:min_len])[
                0, 1
            ]
            print(f"after_source_add_{i}: corr={corr:.6f}")
            print(f"  MLX range: [{float(mx.min(x)):.4f}, {float(mx.max(x)):.4f}]")
            print(f"  PT range: [{pt_val.min():.4f}, {pt_val.max():.4f}]")

        # Resblocks
        xs = None
        for j in range(generator.num_kernels):
            block_idx = i * generator.num_kernels + j
            if block_idx < generator._num_resblocks:
                resblock = getattr(generator, f"resblocks_{block_idx}")
                key_in = f"resblock_{block_idx}_in"
                if key_in in internal:
                    pt_rb_in = internal[key_in]
                    mlx_ncl = np.array(x.transpose(0, 2, 1))
                    min_len = min(mlx_ncl.size, pt_rb_in.size)
                    corr = np.corrcoef(
                        mlx_ncl.flatten()[:min_len], pt_rb_in.flatten()[:min_len]
                    )[0, 1]
                    if j == 0:
                        print(f"resblock_{block_idx}_in: corr={corr:.6f}")

                rb_out = resblock(x, style_mx)
                mx.eval(rb_out)

                key_out = f"resblock_{block_idx}_out"
                if key_out in internal:
                    pt_rb_out = internal[key_out]
                    mlx_rb_ncl = np.array(rb_out.transpose(0, 2, 1))
                    min_len = min(mlx_rb_ncl.size, pt_rb_out.size)
                    corr = np.corrcoef(
                        mlx_rb_ncl.flatten()[:min_len], pt_rb_out.flatten()[:min_len]
                    )[0, 1]
                    print(f"resblock_{block_idx}_out: corr={corr:.6f}")

                if xs is None:
                    xs = rb_out
                else:
                    xs = xs + rb_out

        if xs is not None:
            x = xs / generator.num_kernels
        mx.eval(x)

        key = f"after_resblocks_{i}"
        if key in internal:
            pt_val = internal[key]
            mlx_ncl = np.array(x.transpose(0, 2, 1))
            min_len = min(mlx_ncl.size, pt_val.size)
            corr = np.corrcoef(mlx_ncl.flatten()[:min_len], pt_val.flatten()[:min_len])[
                0, 1
            ]
            print(f"after_resblocks_{i}: corr={corr:.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
