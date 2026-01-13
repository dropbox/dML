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
Trace Generator with actual decoder output to find divergence point.
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

    # Load generator traces
    traces_path = Path("/tmp/kokoro_ref/generator_traces.npz")
    has_gen_traces = traces_path.exists()
    if has_gen_traces:
        gen_traces = np.load(traces_path)
        print("Generator traces available:")
        for k in sorted(gen_traces.keys()):
            print(f"  {k}: {gen_traces[k].shape}")

    print("\n=== Loading MLX Model ===")
    converter = KokoroConverter()
    model, config, _ = converter.load_from_hf()
    mx.eval(model)

    # Get inputs
    asr_ncl = ref["asr_ncl"]
    F0_pred = ref["F0_pred"]
    N_pred = ref["N_pred"]
    style_128 = ref["style_128"]

    # Convert to MLX
    mx.array(asr_ncl).transpose(0, 2, 1)
    F0_mx = mx.array(F0_pred)
    mx.array(N_pred)
    style_mx = mx.array(style_128)

    decoder = model.decoder
    generator = decoder.generator

    # Get generator input by running decode blocks
    # First, we need to understand how the decoder processes inputs

    # Looking at generator_traces.npz, we have 'generator_input_ncl' which is the
    # input to the generator from the decode blocks

    if has_gen_traces and "generator_input_ncl" in gen_traces:
        pt_gen_input = gen_traces["generator_input_ncl"]  # NCL
        print(
            f"\nPyTorch generator input: {pt_gen_input.shape}, range: [{pt_gen_input.min():.4f}, {pt_gen_input.max():.4f}]"
        )

        # Use this as input to MLX generator
        gen_input = mx.array(pt_gen_input).transpose(0, 2, 1)  # NLC
        mx.eval(gen_input)
        print(f"MLX generator input (from PT): {gen_input.shape}")

        # Run through Generator manually
        print("\n=== Running Generator with PyTorch decode output ===")
        x = gen_input

        # Calculate upp
        total_upp = 1
        for r in generator.config.istft_upsample_rates:
            total_upp *= r
        total_upp *= generator.istft_hop_size
        print(f"Total upsampling: {total_upp}")

        # Source module
        har_source, noise_src, uv = generator.m_source(F0_mx, total_upp)
        mx.eval([har_source, noise_src, uv])
        print(
            f"har_source: range [{float(mx.min(har_source)):.4f}, {float(mx.max(har_source)):.4f}]"
        )

        if "generator_audio" in gen_traces:
            gen_traces.get("generator_input_ncl", None)

        # STFT of source
        har_source_1d = har_source.squeeze(-1)
        source = generator._source_stft(har_source_1d)
        mx.eval(source)
        print(
            f"source STFT: {source.shape}, range [{float(mx.min(source)):.4f}, {float(mx.max(source)):.4f}]"
        )

        # Run through upsample stages
        for i in range(generator.num_upsamples):
            up = getattr(generator, f"ups_{i}")
            noise_conv = getattr(generator, f"noise_convs_{i}")
            noise_res = getattr(generator, f"noise_res_{i}")

            x = nn.leaky_relu(x, 0.1)
            print(
                f"\nAfter leaky_relu_{i}: range [{float(mx.min(x)):.4f}, {float(mx.max(x)):.4f}]"
            )

            # Noise path
            x_source = noise_conv(source)
            x_source = noise_res(x_source, style_mx)
            mx.eval(x_source)
            print(
                f"noise_res_{i}: range [{float(mx.min(x_source)):.4f}, {float(mx.max(x_source)):.4f}]"
            )

            # Upsample main
            x = up(x)
            mx.eval(x)
            print(
                f"ups_{i}: {x.shape}, range [{float(mx.min(x)):.4f}, {float(mx.max(x)):.4f}]"
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
            print(
                f"after add_{i}: range [{float(mx.min(x)):.4f}, {float(mx.max(x)):.4f}]"
            )

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
                f"after resblocks_{i}: range [{float(mx.min(x)):.4f}, {float(mx.max(x)):.4f}]"
            )

        # conv_post
        x = nn.leaky_relu(x, 0.1)
        x = generator.conv_post(x)
        mx.eval(x)
        print(
            f"\nconv_post: {x.shape}, range [{float(mx.min(x)):.4f}, {float(mx.max(x)):.4f}]"
        )

        if "generator_audio" in gen_traces:
            pt_audio = gen_traces["generator_audio"]
            print(
                f"PT generator_audio: {pt_audio.shape}, range [{pt_audio.min():.4f}, {pt_audio.max():.4f}]"
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
            f"\nFinal audio: {audio.shape}, range [{float(mx.min(audio)):.4f}, {float(mx.max(audio)):.4f}]"
        )

        # Compare with reference
        ref_audio = ref["audio"]
        mlx_audio = np.array(audio).flatten()
        ref_flat = ref_audio.flatten()
        min_len = min(len(mlx_audio), len(ref_flat))
        corr = np.corrcoef(mlx_audio[:min_len], ref_flat[:min_len])[0, 1]
        print(f"Correlation with reference: {corr:.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
