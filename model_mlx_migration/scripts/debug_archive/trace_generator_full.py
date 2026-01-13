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
Full Generator trace - compare PyTorch vs MLX at each stage.
"""

import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as mnn
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.pytorch_to_mlx.converters import KokoroConverter


def main():
    # Load reference traces
    ref_path = Path("/tmp/kokoro_ref/tensors.npz")
    if not ref_path.exists():
        print("Reference not found - run export_kokoro_reference.py first")
        return 1
    ref = np.load(ref_path)

    # Load generator traces if available
    gen_traces_path = Path("/tmp/kokoro_ref/generator_traces.npz")
    if gen_traces_path.exists():
        np.load(gen_traces_path)
        print("Generator traces available")
    else:
        print("No generator traces")

    # Load internal traces if available
    internal_traces_path = Path("/tmp/kokoro_ref/generator_internal_traces.npz")
    if internal_traces_path.exists():
        internal_traces = np.load(internal_traces_path)
        print("Internal traces available:")
        for k in sorted(internal_traces.keys()):
            print(f"  {k}: {internal_traces[k].shape}")
    else:
        internal_traces = None
        print("No internal traces")

    print("\n=== Loading MLX Model ===")
    converter = KokoroConverter()
    model, config, _ = converter.load_from_hf()
    mx.eval(model)

    decoder = model.decoder
    generator = decoder.generator

    # Get inputs from reference
    F0_pred = ref["F0_pred"]
    style_128 = ref["style_128"]

    F0_mx = mx.array(F0_pred)
    style_mx = mx.array(style_128)

    print(
        f"F0: {F0_mx.shape}, range: [{float(mx.min(F0_mx)):.4f}, {float(mx.max(F0_mx)):.4f}]"
    )
    print(f"style: {style_mx.shape}")

    # If we have ups_0_in, use it as generator input
    if internal_traces is not None and "ups_0_in" in internal_traces:
        # PyTorch format: NCL
        pt_gen_input_ncl = internal_traces["ups_0_in"]
        print(f"\nPT generator input (NCL): {pt_gen_input_ncl.shape}")
        print(f"  range: [{pt_gen_input_ncl.min():.4f}, {pt_gen_input_ncl.max():.4f}]")

        # Convert to MLX NLC
        gen_input = mx.array(pt_gen_input_ncl).transpose(0, 2, 1)  # NLC
        mx.eval(gen_input)
    else:
        print("Using random input for generator")
        np.random.seed(42)
        gen_input = mx.array(np.random.randn(1, 126, 512).astype(np.float32) * 0.1)

    x = gen_input
    print(f"\nMLX generator input (NLC): {x.shape}")
    print(f"  range: [{float(mx.min(x)):.4f}, {float(mx.max(x)):.4f}]")

    # Calculate total upsampling
    total_upp = 1
    for r in generator.config.istft_upsample_rates:
        total_upp *= r
    total_upp *= generator.istft_hop_size
    print(f"\nTotal upsampling: {total_upp}")

    # Run source module
    har_source, noise_src, uv = generator.m_source(F0_mx, total_upp)
    mx.eval([har_source, noise_src, uv])
    print(
        f"\nm_source har: {har_source.shape}, range: [{float(mx.min(har_source)):.4f}, {float(mx.max(har_source)):.4f}]"
    )

    if internal_traces is not None and "m_source_out_0" in internal_traces:
        pt_har = internal_traces["m_source_out_0"]
        corr = np.corrcoef(np.array(har_source).flatten(), pt_har.flatten())[0, 1]
        print(f"  vs PT: correlation {corr:.6f}")

    # STFT of source
    har_source_1d = har_source.squeeze(-1)
    source = generator._source_stft(har_source_1d)
    mx.eval(source)
    print(
        f"\nsource STFT: {source.shape}, range: [{float(mx.min(source)):.4f}, {float(mx.max(source)):.4f}]"
    )

    if internal_traces is not None and "source_stft" in internal_traces:
        pt_source = internal_traces["source_stft"]  # NCL
        mlx_source_ncl = np.array(source.transpose(0, 2, 1))
        corr = np.corrcoef(mlx_source_ncl.flatten(), pt_source.flatten())[0, 1]
        print(f"  vs PT: correlation {corr:.6f}")

    # Upsample stages
    print("\n=== Upsample stages ===")
    for i in range(generator.num_upsamples):
        up = getattr(generator, f"ups_{i}")
        noise_conv = getattr(generator, f"noise_convs_{i}")
        noise_res = getattr(generator, f"noise_res_{i}")

        x = mnn.leaky_relu(x, 0.1)

        # Compare after leaky_relu
        if internal_traces is not None:
            key = f"after_leaky_relu_{i}"
            if key in internal_traces:
                pt_val = internal_traces[key]  # NCL
                mlx_ncl = np.array(x.transpose(0, 2, 1))
                corr = np.corrcoef(mlx_ncl.flatten()[: pt_val.size], pt_val.flatten())[
                    0, 1
                ]
                print(
                    f"after_leaky_relu_{i}: corr {corr:.6f}, range [{float(mx.min(x)):.4f}, {float(mx.max(x)):.4f}]"
                )

        # Noise path
        x_source = noise_conv(source)
        mx.eval(x_source)

        if internal_traces is not None:
            key = f"noise_conv_{i}_out"
            if key in internal_traces:
                pt_val = internal_traces[key]  # NCL
                mlx_ncl = np.array(x_source.transpose(0, 2, 1))
                min_len = min(mlx_ncl.size, pt_val.size)
                corr = np.corrcoef(
                    mlx_ncl.flatten()[:min_len], pt_val.flatten()[:min_len]
                )[0, 1]
                print(f"noise_conv_{i}: corr {corr:.6f}")

        x_source = noise_res(x_source, style_mx)
        mx.eval(x_source)

        if internal_traces is not None:
            key = f"noise_res_{i}_out"
            if key in internal_traces:
                pt_val = internal_traces[key]
                mlx_ncl = np.array(x_source.transpose(0, 2, 1))
                min_len = min(mlx_ncl.size, pt_val.size)
                corr = np.corrcoef(
                    mlx_ncl.flatten()[:min_len], pt_val.flatten()[:min_len]
                )[0, 1]
                print(f"noise_res_{i}: corr {corr:.6f}")

        # Upsample main
        x = up(x)
        mx.eval(x)

        if internal_traces is not None:
            key = f"ups_{i}_out"
            if key in internal_traces:
                pt_val = internal_traces[key]
                mlx_ncl = np.array(x.transpose(0, 2, 1))
                min_len = min(mlx_ncl.size, pt_val.size)
                corr = np.corrcoef(
                    mlx_ncl.flatten()[:min_len], pt_val.flatten()[:min_len]
                )[0, 1]
                print(
                    f"ups_{i}: corr {corr:.6f}, shape {x.shape}, range [{float(mx.min(x)):.4f}, {float(mx.max(x)):.4f}]"
                )

        # Reflection pad at last stage
        if i == generator.num_upsamples - 1:
            x = mx.concatenate([x[:, 1:2, :], x], axis=1)
            mx.eval(x)

        # Match lengths
        if x_source.shape[1] < x.shape[1]:
            pad_len = x.shape[1] - x_source.shape[1]
            x_source = mx.pad(x_source, [(0, 0), (0, pad_len), (0, 0)])
        elif x_source.shape[1] > x.shape[1]:
            x_source = x_source[:, : x.shape[1], :]

        x = x + x_source
        mx.eval(x)

        if internal_traces is not None:
            key = f"after_source_add_{i}"
            if key in internal_traces:
                pt_val = internal_traces[key]
                mlx_ncl = np.array(x.transpose(0, 2, 1))
                min_len = min(mlx_ncl.size, pt_val.size)
                corr = np.corrcoef(
                    mlx_ncl.flatten()[:min_len], pt_val.flatten()[:min_len]
                )[0, 1]
                print(f"after_add_{i}: corr {corr:.6f}")

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
            f"after_resblocks_{i}: shape {x.shape}, range [{float(mx.min(x)):.4f}, {float(mx.max(x)):.4f}], std={float(mx.std(x)):.4f}"
        )

        if internal_traces is not None:
            key = f"after_resblocks_{i}"
            if key in internal_traces:
                pt_val = internal_traces[key]
                mlx_ncl = np.array(x.transpose(0, 2, 1))
                min_len = min(mlx_ncl.size, pt_val.size)
                corr = np.corrcoef(
                    mlx_ncl.flatten()[:min_len], pt_val.flatten()[:min_len]
                )[0, 1]
                print(
                    f"  vs PT: corr {corr:.6f}, PT range [{pt_val.min():.4f}, {pt_val.max():.4f}], PT std={pt_val.std():.4f}"
                )

    # conv_post
    x = mnn.leaky_relu(x, 0.1)
    x = generator.conv_post(x)
    mx.eval(x)
    print(
        f"\nconv_post: shape {x.shape}, range [{float(mx.min(x)):.4f}, {float(mx.max(x)):.4f}]"
    )

    if internal_traces is not None and "conv_post_out" in internal_traces:
        pt_val = internal_traces["conv_post_out"]
        mlx_ncl = np.array(x.transpose(0, 2, 1))
        min_len = min(mlx_ncl.size, pt_val.size)
        corr = np.corrcoef(mlx_ncl.flatten()[:min_len], pt_val.flatten()[:min_len])[
            0, 1
        ]
        print(
            f"  vs PT: corr {corr:.6f}, PT range [{pt_val.min():.4f}, {pt_val.max():.4f}]"
        )

    # ISTFT
    n_bins = generator.post_n_fft // 2 + 1
    log_mag = mx.clip(x[..., :n_bins], -10, 10)
    mag = mx.exp(log_mag)
    phase = mx.sin(x[..., n_bins:])
    audio = generator._istft_synthesis(mag, phase)
    audio = mx.clip(audio, -1.0, 1.0)
    mx.eval(audio)

    print(
        f"\nFinal audio: {audio.shape}, range [{float(mx.min(audio)):.4f}, {float(mx.max(audio)):.4f}], std={float(mx.std(audio)):.4f}"
    )

    # Compare with reference
    ref_audio = ref["audio"]
    mlx_audio = np.array(audio).flatten()
    ref_flat = ref_audio.flatten()
    min_len = min(len(mlx_audio), len(ref_flat))
    corr = np.corrcoef(mlx_audio[:min_len], ref_flat[:min_len])[0, 1]
    print(
        f"Reference audio: range [{ref_audio.min():.4f}, {ref_audio.max():.4f}], std={ref_audio.std():.4f}"
    )
    print(f"Correlation with reference: {corr:.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
