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
Debug the noise path contribution in Generator.
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

    x = gen_input
    mx.eval(x)

    # Source
    total_upp = 1
    for r in generator.config.istft_upsample_rates:
        total_upp *= r
    total_upp *= generator.istft_hop_size

    har_source, noise_src, uv = generator.m_source(F0_mx, total_upp)
    har_source_1d = har_source.squeeze(-1)
    source = generator._source_stft(har_source_1d)
    mx.eval([har_source, source])

    print(
        f"Generator input: {x.shape}, range [{float(mx.min(x)):.4f}, {float(mx.max(x)):.4f}]"
    )
    print(
        f"source (STFT of har): {source.shape}, range [{float(mx.min(source)):.4f}, {float(mx.max(source)):.4f}]"
    )

    # Run through stages WITHOUT noise path
    print("\n=== WITHOUT noise path ===")
    x_no_noise = gen_input

    for i in range(generator.num_upsamples):
        up = getattr(generator, f"ups_{i}")

        x_no_noise = nn.leaky_relu(x_no_noise, 0.1)
        x_no_noise = up(x_no_noise)
        if i == generator.num_upsamples - 1:
            x_no_noise = mx.concatenate([x_no_noise[:, 1:2, :], x_no_noise], axis=1)

        # Skip noise path - just apply resblocks
        xs = None
        for j in range(generator.num_kernels):
            block_idx = i * generator.num_kernels + j
            if block_idx < generator._num_resblocks:
                resblock = getattr(generator, f"resblocks_{block_idx}")
                if xs is None:
                    xs = resblock(x_no_noise, style_mx)
                else:
                    xs = xs + resblock(x_no_noise, style_mx)
        if xs is not None:
            x_no_noise = xs / generator.num_kernels

        mx.eval(x_no_noise)
        print(
            f"Stage {i}: {x_no_noise.shape}, range [{float(mx.min(x_no_noise)):.4f}, {float(mx.max(x_no_noise)):.4f}]"
        )

    x_no_noise = nn.leaky_relu(x_no_noise, 0.1)
    x_no_noise = generator.conv_post(x_no_noise)
    mx.eval(x_no_noise)
    print(
        f"conv_post (no noise): range [{float(mx.min(x_no_noise)):.4f}, {float(mx.max(x_no_noise)):.4f}]"
    )

    # ISTFT
    n_bins = generator.post_n_fft // 2 + 1
    log_mag = mx.clip(x_no_noise[..., :n_bins], -10, 10)
    mag = mx.exp(log_mag)
    phase = mx.sin(x_no_noise[..., n_bins:])
    audio_no_noise = generator._istft_synthesis(mag, phase)
    audio_no_noise = mx.clip(audio_no_noise, -1.0, 1.0)
    mx.eval(audio_no_noise)
    print(
        f"Audio (no noise): range [{float(mx.min(audio_no_noise)):.4f}, {float(mx.max(audio_no_noise)):.4f}], std {float(mx.std(audio_no_noise)):.4f}"
    )

    # Now WITH noise path
    print("\n=== WITH noise path ===")
    x_with_noise = gen_input

    for i in range(generator.num_upsamples):
        up = getattr(generator, f"ups_{i}")
        noise_conv = getattr(generator, f"noise_convs_{i}")
        noise_res = getattr(generator, f"noise_res_{i}")

        x_with_noise = nn.leaky_relu(x_with_noise, 0.1)

        # Noise path
        x_source = noise_conv(source)
        x_source = noise_res(x_source, style_mx)
        mx.eval(x_source)
        print(
            f"Stage {i} noise_res: range [{float(mx.min(x_source)):.4f}, {float(mx.max(x_source)):.4f}]"
        )

        x_with_noise = up(x_with_noise)
        if i == generator.num_upsamples - 1:
            x_with_noise = mx.concatenate(
                [x_with_noise[:, 1:2, :], x_with_noise], axis=1
            )

        # Match lengths
        if x_source.shape[1] < x_with_noise.shape[1]:
            pad_len = x_with_noise.shape[1] - x_source.shape[1]
            x_source = mx.pad(x_source, [(0, 0), (0, pad_len), (0, 0)])
        elif x_source.shape[1] > x_with_noise.shape[1]:
            x_source = x_source[:, : x_with_noise.shape[1], :]

        x_with_noise = x_with_noise + x_source
        mx.eval(x_with_noise)
        print(
            f"Stage {i} after add: range [{float(mx.min(x_with_noise)):.4f}, {float(mx.max(x_with_noise)):.4f}]"
        )

        xs = None
        for j in range(generator.num_kernels):
            block_idx = i * generator.num_kernels + j
            if block_idx < generator._num_resblocks:
                resblock = getattr(generator, f"resblocks_{block_idx}")
                if xs is None:
                    xs = resblock(x_with_noise, style_mx)
                else:
                    xs = xs + resblock(x_with_noise, style_mx)
        if xs is not None:
            x_with_noise = xs / generator.num_kernels

        mx.eval(x_with_noise)
        print(
            f"Stage {i} after resblocks: range [{float(mx.min(x_with_noise)):.4f}, {float(mx.max(x_with_noise)):.4f}]"
        )

    x_with_noise = nn.leaky_relu(x_with_noise, 0.1)
    x_with_noise = generator.conv_post(x_with_noise)
    mx.eval(x_with_noise)
    print(
        f"conv_post (with noise): range [{float(mx.min(x_with_noise)):.4f}, {float(mx.max(x_with_noise)):.4f}]"
    )

    # Compare
    ref_audio = ref["audio"]
    print(
        f"\nReference audio: range [{ref_audio.min():.4f}, {ref_audio.max():.4f}], std {ref_audio.std():.4f}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
