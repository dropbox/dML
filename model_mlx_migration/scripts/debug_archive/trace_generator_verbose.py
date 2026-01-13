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
Verbose trace of Generator forward to find connection issues.
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

    mx.array(ref["F0_pred"])
    style_mx = mx.array(ref["style_128"])

    pt_gen_input_ncl = gen_traces["generator_input_ncl"]
    gen_input = mx.array(pt_gen_input_ncl).transpose(0, 2, 1)  # NLC [1, 126, 512]

    generator = model.decoder.generator

    print("\n=== Input ===")
    print(f"gen_input (NLC): {gen_input.shape}")
    print(f"  range: [{float(mx.min(gen_input)):.4f}, {float(mx.max(gen_input)):.4f}]")
    print(f"PT input (NCL): {pt_gen_input_ncl.shape}")
    print(f"  range: [{pt_gen_input_ncl.min():.4f}, {pt_gen_input_ncl.max():.4f}]")

    # Check if transposition is correct
    gen_input_back = np.array(gen_input.transpose(0, 2, 1))
    diff = np.abs(gen_input_back - pt_gen_input_ncl).max()
    print(f"  Round-trip diff: {diff:.6e}")

    x = gen_input

    # Skip noise path for clarity
    print("\n=== Trace WITHOUT Noise Path ===")

    for i in range(generator.num_upsamples):
        print(f"\n--- Stage {i} ---")

        # LeakyReLU
        x = nn.leaky_relu(x, 0.1)
        mx.eval(x)
        print(f"After leaky_relu: shape {x.shape}")
        print(f"  range: [{float(mx.min(x)):.4f}, {float(mx.max(x)):.4f}]")

        # PT comparison
        key = f"ups_{i}_in"
        if key in internal:
            pt_val = internal[key]  # NCL
            mlx_ncl = np.array(x.transpose(0, 2, 1))
            diff = np.abs(mlx_ncl - pt_val).max()
            corr = np.corrcoef(mlx_ncl.flatten(), pt_val.flatten())[0, 1]
            print(f"  vs PT {key}: corr={corr:.6f}, max_diff={diff:.4f}")

        # Upsample
        up = getattr(generator, f"ups_{i}")
        x = up(x)
        mx.eval(x)
        print(f"After ups_{i}: shape {x.shape}")
        print(f"  range: [{float(mx.min(x)):.4f}, {float(mx.max(x)):.4f}]")

        key = f"ups_{i}_out"
        if key in internal:
            pt_val = internal[key]
            mlx_ncl = np.array(x.transpose(0, 2, 1))
            min_len = min(mlx_ncl.size, pt_val.size)
            diff = np.abs(
                mlx_ncl.flatten()[:min_len] - pt_val.flatten()[:min_len]
            ).max()
            corr = np.corrcoef(mlx_ncl.flatten()[:min_len], pt_val.flatten()[:min_len])[
                0, 1
            ]
            print(f"  vs PT {key}: corr={corr:.6f}, max_diff={diff:.4f}")
            print(
                f"  PT shape: {pt_val.shape}, range: [{pt_val.min():.4f}, {pt_val.max():.4f}]"
            )

        # Reflection pad at last stage
        if i == generator.num_upsamples - 1:
            x = mx.concatenate([x[:, 1:2, :], x], axis=1)
            mx.eval(x)
            print(f"After reflect pad: shape {x.shape}")

        # Resblocks (without noise addition)
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
        print(f"After resblocks: shape {x.shape}")
        print(f"  range: [{float(mx.min(x)):.4f}, {float(mx.max(x)):.4f}]")
        print(f"  std: {float(mx.std(x)):.4f}")

    # Final conv
    x = nn.leaky_relu(x, 0.1)
    conv_post_out = generator.conv_post(x)
    mx.eval(conv_post_out)
    print(f"\nconv_post output: shape {conv_post_out.shape}")
    print(
        f"  range: [{float(mx.min(conv_post_out)):.4f}, {float(mx.max(conv_post_out)):.4f}]"
    )

    # Compare with PT (but PT has noise path)
    if "conv_post_out" in internal:
        pt_val = internal["conv_post_out"]
        mlx_ncl = np.array(conv_post_out.transpose(0, 2, 1))
        corr = np.corrcoef(mlx_ncl.flatten(), pt_val.flatten())[0, 1]
        print(f"  vs PT: corr={corr:.6f} (NOTE: PT has noise path, we don't)")

    # ISTFT
    n_bins = generator.post_n_fft // 2 + 1
    log_mag = mx.clip(conv_post_out[..., :n_bins], -10, 10)
    mag = mx.exp(log_mag)
    phase = mx.sin(conv_post_out[..., n_bins:])
    audio = generator._istft_synthesis(mag, phase)
    audio_pre_clip = audio
    audio = mx.clip(audio, -1.0, 1.0)
    mx.eval([audio, audio_pre_clip])

    print("\nAudio (no noise):")
    print(
        f"  Pre-clip range: [{float(mx.min(audio_pre_clip)):.4f}, {float(mx.max(audio_pre_clip)):.4f}]"
    )
    print(
        f"  Post-clip range: [{float(mx.min(audio)):.4f}, {float(mx.max(audio)):.4f}]"
    )
    print(f"  Std: {float(mx.std(audio)):.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
