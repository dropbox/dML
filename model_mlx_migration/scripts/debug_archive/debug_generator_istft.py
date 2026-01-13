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
Debug the Generator's ISTFT conversion with PT decode output.
"""

import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.pytorch_to_mlx.converters import KokoroConverter


def main():
    gen_traces = np.load("/tmp/kokoro_ref/generator_traces.npz")
    ref = np.load("/tmp/kokoro_ref/tensors.npz")

    print("=== Loading MLX Model ===")
    converter = KokoroConverter()
    model, config, _ = converter.load_from_hf()
    mx.eval(model)

    generator = model.decoder.generator
    style_mx = mx.array(ref["style_128"])
    F0_mx = mx.array(ref["F0_pred"])

    # Use PT decode output as generator input
    gen_input = mx.array(gen_traces["generator_input_ncl"]).transpose(0, 2, 1)  # NLC
    mx.eval(gen_input)
    print(
        f"Generator input: {gen_input.shape}, range [{float(mx.min(gen_input)):.4f}, {float(mx.max(gen_input)):.4f}]"
    )

    x = gen_input

    # Run through generator stages until conv_post
    total_upp = 1
    for r in generator.config.istft_upsample_rates:
        total_upp *= r
    total_upp *= generator.istft_hop_size

    har_source, noise_src, uv = generator.m_source(F0_mx, total_upp)
    har_source_1d = har_source.squeeze(-1)
    source = generator._source_stft(har_source_1d)
    mx.eval([har_source, source])
    print(f"source: {source.shape}")

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

    x = nn.leaky_relu(x, 0.1)
    conv_post_out = generator.conv_post(x)
    mx.eval(conv_post_out)

    print(f"\nconv_post output: {conv_post_out.shape}")
    print(
        f"Full range: [{float(mx.min(conv_post_out)):.4f}, {float(mx.max(conv_post_out)):.4f}]"
    )

    # Split into mag and phase components
    n_bins = generator.post_n_fft // 2 + 1  # 11
    log_mag_part = conv_post_out[..., :n_bins]
    phase_part = conv_post_out[..., n_bins:]

    print("\nlog_mag part (first 11 channels):")
    print(
        f"  Range: [{float(mx.min(log_mag_part)):.4f}, {float(mx.max(log_mag_part)):.4f}]"
    )
    print(f"  Mean: {float(mx.mean(log_mag_part)):.4f}")

    print("\nphase part (next 11 channels):")
    print(
        f"  Range: [{float(mx.min(phase_part)):.4f}, {float(mx.max(phase_part)):.4f}]"
    )
    print(f"  Mean: {float(mx.mean(phase_part)):.4f}")

    # After clipping log_mag to [-10, 10]
    log_mag_clipped = mx.clip(log_mag_part, -10, 10)
    print("\nlog_mag after clip to [-10, 10]:")
    print(
        f"  Range: [{float(mx.min(log_mag_clipped)):.4f}, {float(mx.max(log_mag_clipped)):.4f}]"
    )

    mag = mx.exp(log_mag_clipped)
    print("\nmag (exp of clipped log_mag):")
    print(f"  Range: [{float(mx.min(mag)):.4f}, {float(mx.max(mag)):.4f}]")

    phase = mx.sin(phase_part)
    print("\nphase (sin of phase_part):")
    print(f"  Range: [{float(mx.min(phase)):.4f}, {float(mx.max(phase)):.4f}]")

    # Run ISTFT
    audio = generator._istft_synthesis(mag, phase)
    mx.eval(audio)
    print(f"\nAudio (before clamp): {audio.shape}")
    print(f"  Range: [{float(mx.min(audio)):.4f}, {float(mx.max(audio)):.4f}]")
    print(f"  Std: {float(mx.std(audio)):.4f}")

    # Reference
    ref_audio = ref["audio"]
    print(
        f"\nReference audio: range [{ref_audio.min():.4f}, {ref_audio.max():.4f}], std {ref_audio.std():.4f}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
