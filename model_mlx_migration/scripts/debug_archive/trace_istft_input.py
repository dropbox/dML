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
Trace ISTFT input to understand magnitude differences.
"""

import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.pytorch_to_mlx.converters import KokoroConverter


def main():
    # Load MLX model
    print("=== Loading MLX Model ===")
    converter = KokoroConverter()
    model, config, _ = converter.load_from_hf()
    mx.eval(model)

    # Load reference
    ref = np.load("/tmp/kokoro_ref/tensors.npz")

    # Load generator traces
    gen_traces = np.load("/tmp/kokoro_ref/generator_traces.npz")
    internal_traces = np.load("/tmp/kokoro_ref/generator_internal_traces.npz")

    F0_mx = mx.array(ref["F0_pred"])
    style_mx = mx.array(ref["style_128"])

    # Get PT generator input and use it
    pt_gen_input = gen_traces["generator_input_ncl"]  # [1, 512, 126]
    gen_input = mx.array(pt_gen_input).transpose(0, 2, 1)  # NLC

    generator = model.decoder.generator

    print("\n=== Running Generator Step by Step ===")

    # Run through Generator manually to get conv_post output
    x = gen_input

    # Calculate upp
    total_upp = 1
    for r in generator.config.istft_upsample_rates:
        total_upp *= r
    total_upp *= generator.istft_hop_size

    # Source module
    har_source, noise_src, uv = generator.m_source(F0_mx, total_upp)
    har_source_1d = har_source.squeeze(-1)
    source = generator._source_stft(har_source_1d)
    mx.eval(source)

    # Upsample stages
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

    # conv_post
    x = nn.leaky_relu(x, 0.1)
    conv_post_out = generator.conv_post(x)
    mx.eval(conv_post_out)

    print("\nconv_post output:")
    print(f"  shape: {conv_post_out.shape}")
    print(
        f"  range: [{float(mx.min(conv_post_out)):.4f}, {float(mx.max(conv_post_out)):.4f}]"
    )
    print(f"  std: {float(mx.std(conv_post_out)):.4f}")

    # Compare with PT
    if "conv_post_out" in internal_traces:
        pt_conv_post = internal_traces["conv_post_out"]  # NCL
        print("\nPT conv_post output:")
        print(f"  shape: {pt_conv_post.shape}")
        print(f"  range: [{pt_conv_post.min():.4f}, {pt_conv_post.max():.4f}]")
        print(f"  std: {pt_conv_post.std():.4f}")

        # Correlation
        mlx_ncl = np.array(conv_post_out.transpose(0, 2, 1))
        corr = np.corrcoef(mlx_ncl.flatten(), pt_conv_post.flatten())[0, 1]
        print(f"  Correlation: {corr:.6f}")

    # ISTFT processing
    n_bins = generator.post_n_fft // 2 + 1  # 11
    log_mag = mx.clip(conv_post_out[..., :n_bins], -10, 10)
    mag = mx.exp(log_mag)
    phase = mx.sin(conv_post_out[..., n_bins:])
    mx.eval([log_mag, mag, phase])

    print("\nlog_mag:")
    print(f"  range: [{float(mx.min(log_mag)):.4f}, {float(mx.max(log_mag)):.4f}]")
    print(f"  std: {float(mx.std(log_mag)):.4f}")

    print("\nmag (exp(log_mag)):")
    print(f"  range: [{float(mx.min(mag)):.4f}, {float(mx.max(mag)):.4f}]")
    print(f"  std: {float(mx.std(mag)):.4f}")

    # Compare with PT mag
    if "log_mag" in internal_traces:
        pt_log_mag = internal_traces["log_mag"]  # [1, 11, frames]
        pt_mag = internal_traces.get("mag_out", np.exp(pt_log_mag))
        print("\nPT log_mag:")
        print(f"  range: [{pt_log_mag.min():.4f}, {pt_log_mag.max():.4f}]")
        print(f"  std: {pt_log_mag.std():.4f}")
        print("PT mag:")
        print(f"  range: [{pt_mag.min():.4f}, {pt_mag.max():.4f}]")
        print(f"  std: {pt_mag.std():.4f}")

    # Run ISTFT
    audio = generator._istft_synthesis(mag, phase)
    audio_clipped = mx.clip(audio, -1.0, 1.0)
    mx.eval([audio, audio_clipped])

    print("\nPre-clip audio:")
    print(f"  range: [{float(mx.min(audio)):.4f}, {float(mx.max(audio)):.4f}]")
    print(f"  std: {float(mx.std(audio)):.4f}")

    print("\nPost-clip audio:")
    print(
        f"  range: [{float(mx.min(audio_clipped)):.4f}, {float(mx.max(audio_clipped)):.4f}]"
    )
    print(f"  std: {float(mx.std(audio_clipped)):.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
