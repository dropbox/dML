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
Trace through Generator stage by stage to find where magnitude diverges.
"""

import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.pytorch_to_mlx.converters import KokoroConverter


def main():
    # Load reference
    ref_path = Path("/tmp/kokoro_ref/tensors.npz")
    if not ref_path.exists():
        print("Reference not found")
        return 1
    ref = np.load(ref_path)

    # Load generator internal traces if available
    traces_path = Path("/tmp/kokoro_ref/generator_internal_traces.npz")
    has_traces = traces_path.exists()
    if has_traces:
        pt_traces = np.load(traces_path)
        print("PyTorch internal traces available")

    # Load MLX model
    print("=== Loading MLX Model ===")
    converter = KokoroConverter()
    model, config, _ = converter.load_from_hf()
    mx.eval(model)

    # Get inputs
    asr_ncl = ref["asr_ncl"]  # [1, 512, 63]
    F0_pred = ref["F0_pred"]  # [1, 126]
    N_pred = ref["N_pred"]  # [1, 126]
    style_128 = ref["style_128"]  # [1, 128]

    # Convert to MLX (NLC format for Generator)
    mx.transpose(mx.array(asr_ncl), (0, 2, 1))  # [1, 63, 512]
    F0_mx = mx.array(F0_pred)
    mx.array(N_pred)
    style_mx = mx.array(style_128)

    # Access generator
    decoder = model.decoder
    generator = decoder.generator

    # Get the generator input from reference (ups_0_in is the generator input)
    if has_traces and "ups_0_in" in pt_traces:
        gen_input = mx.array(pt_traces["ups_0_in"])
        # PyTorch is NCL [1, 512, 126], convert to NLC [1, 126, 512]
        gen_input = gen_input.transpose(0, 2, 1)
        print("\nUsing PyTorch ups_0_in as generator input")
    else:
        print("\nNo ups_0_in trace available")
        return 0

    x = gen_input
    mx.eval(x)
    print(
        f"Generator input: {x.shape}, range: [{float(mx.min(x)):.4f}, {float(mx.max(x)):.4f}]"
    )

    # Now run generator stages
    print("\n=== Generator stages ===")

    # Calculate upp
    total_upp = 1
    for r in generator.config.istft_upsample_rates:
        total_upp *= r
    total_upp *= generator.istft_hop_size
    print(f"Total upsampling factor: {total_upp}")

    # Source module
    har_source, noise_src, uv = generator.m_source(F0_mx, total_upp)
    mx.eval([har_source, noise_src, uv])
    print(
        f"m_source har: {har_source.shape}, range: [{float(mx.min(har_source)):.4f}, {float(mx.max(har_source)):.4f}]"
    )

    if has_traces and "m_source_out_0" in pt_traces:
        pt_msrc = pt_traces["m_source_out_0"]  # har_source
        corr = np.corrcoef(np.array(har_source).flatten(), pt_msrc.flatten())[0, 1]
        print(
            f"  vs PyTorch: range [{pt_msrc.min():.4f}, {pt_msrc.max():.4f}], corr: {corr:.6f}"
        )

    # STFT of source
    har_source_1d = har_source.squeeze(-1)
    source = generator._source_stft(har_source_1d)
    mx.eval(source)
    print(
        f"source STFT: {source.shape}, range: [{float(mx.min(source)):.4f}, {float(mx.max(source)):.4f}]"
    )

    # Upsample stages
    for i in range(generator.num_upsamples):
        up = getattr(generator, f"ups_{i}")
        noise_conv = getattr(generator, f"noise_convs_{i}")
        noise_res = getattr(generator, f"noise_res_{i}")

        x = nn.leaky_relu(x, 0.1)

        # Noise path
        x_source = noise_conv(source)
        x_source = noise_res(x_source, style_mx)

        # Upsample main
        x = up(x)

        # Reflection pad at last stage
        if i == generator.num_upsamples - 1:
            x = mx.concatenate([x[:, 1:2, :], x], axis=1)

        # Match lengths
        if x_source.shape[1] < x.shape[1]:
            pad_len = x.shape[1] - x_source.shape[1]
            x_source = mx.pad(x_source, [(0, 0), (0, pad_len), (0, 0)])
        elif x_source.shape[1] > x.shape[1]:
            x_source = x_source[:, : x.shape[1], :]

        x = x + x_source

        # ResBlocks
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
        print(
            f"ups_{i} out: {x.shape}, range: [{float(mx.min(x)):.4f}, {float(mx.max(x)):.4f}]"
        )

        if has_traces:
            key = f"ups_{i}_out"
            if key in pt_traces:
                pt_val = pt_traces[key]  # NCL
                # Convert MLX NLC to NCL for comparison
                mlx_ncl = np.array(x.transpose(0, 2, 1))
                min_len = min(mlx_ncl.flatten().size, pt_val.flatten().size)
                corr = np.corrcoef(
                    mlx_ncl.flatten()[:min_len], pt_val.flatten()[:min_len]
                )[0, 1]
                print(
                    f"  vs PyTorch: range [{pt_val.min():.4f}, {pt_val.max():.4f}], corr: {corr:.6f}"
                )

    # conv_post
    x = nn.leaky_relu(x, 0.1)
    x = generator.conv_post(x)
    mx.eval(x)
    print(
        f"conv_post out: {x.shape}, range: [{float(mx.min(x)):.4f}, {float(mx.max(x)):.4f}]"
    )

    if has_traces and "conv_post_out" in pt_traces:
        pt_val = pt_traces["conv_post_out"]  # NCL
        mlx_ncl = np.array(x.transpose(0, 2, 1))
        min_len = min(mlx_ncl.flatten().size, pt_val.flatten().size)
        corr = np.corrcoef(mlx_ncl.flatten()[:min_len], pt_val.flatten()[:min_len])[
            0, 1
        ]
        print(
            f"  vs PyTorch: range [{pt_val.min():.4f}, {pt_val.max():.4f}], corr: {corr:.6f}"
        )

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
