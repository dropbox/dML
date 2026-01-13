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
Compare conv_post output between loaded and fresh models.
"""

import sys

sys.path.insert(0, "/Users/ayates/model_mlx_migration")

import mlx.core as mx
import mlx.nn as nn

from tools.pytorch_to_mlx.converters.kokoro_converter import KokoroConverter
from tools.pytorch_to_mlx.converters.models.kokoro import Generator


def compare():
    print("Loading Kokoro model...")
    converter = KokoroConverter()
    model, config, state_dict = converter.load_from_hf("hexgrad/Kokoro-82M")

    loaded_gen = model.decoder.generator

    # Create fresh generator
    fresh_gen = Generator(config)

    # Create identical inputs
    mx.random.seed(42)
    batch = 1
    length = 20
    channels = 512
    style_dim = 128

    x = mx.random.normal((batch, length, channels)) * 0.1
    mx.random.normal((batch, style_dim)) * 0.1
    f0 = mx.ones((batch, length)) * 200.0

    print(f"Input x: mean={float(x.mean()):.4f}, std={float(x.std()):.4f}")

    # Run both generators step by step and compare conv_post input
    def trace_to_conv_post(gen, x, s, f0, name):
        print(f"\n=== {name} ===")

        # Source generation
        total_upp = 1
        for r in config.istft_upsample_rates:
            total_upp *= r
        total_upp *= gen.istft_hop_size

        har_source, noise, uv = gen.m_source(f0, total_upp)
        har_source_1d = har_source.squeeze(-1)
        source = gen._source_stft(har_source_1d)

        # Main signal path
        current = x
        for i, (up, noise_conv, noise_res) in enumerate(
            zip(gen.ups, gen.noise_convs, gen.noise_res)
        ):
            current = nn.leaky_relu(current, 0.1)
            x_source = noise_conv(source)
            x_source = noise_res(x_source, s)
            current = up(current)

            if x_source.shape[1] < current.shape[1]:
                pad_len = current.shape[1] - x_source.shape[1]
                x_source = mx.pad(x_source, [(0, 0), (0, pad_len), (0, 0)])
            elif x_source.shape[1] > current.shape[1]:
                x_source = x_source[:, : current.shape[1], :]

            current = current + x_source

            xs = None
            for j in range(gen.num_kernels):
                block_idx = i * gen.num_kernels + j
                if block_idx < len(gen.resblocks):
                    if xs is None:
                        xs = gen.resblocks[block_idx](current)
                    else:
                        xs = xs + gen.resblocks[block_idx](current)
            if xs is not None:
                current = xs / gen.num_kernels

        mx.eval(current)
        print(
            f"Before conv_post: mean={float(current.mean()):.4f}, std={float(current.std()):.4f}"
        )

        # Check conv_post weights
        print(
            f"conv_post.weight_g: shape={gen.conv_post.weight_g.shape}, mean={float(gen.conv_post.weight_g.mean()):.4f}, std={float(gen.conv_post.weight_g.std()):.4f}"
        )
        print(
            f"conv_post.weight_v: shape={gen.conv_post.weight_v.shape}, mean={float(gen.conv_post.weight_v.mean()):.4f}, std={float(gen.conv_post.weight_v.std()):.4f}"
        )
        print(
            f"conv_post.bias: mean={float(gen.conv_post.bias.mean()):.4f}, std={float(gen.conv_post.bias.std()):.4f}"
        )

        current = nn.leaky_relu(current, 0.1)
        output = gen.conv_post(current)
        mx.eval(output)

        n_bins = gen.post_n_fft // 2 + 1
        log_mag = output[..., :n_bins]
        phase_logits = output[..., n_bins:]

        print(
            f"\nconv_post output: mean={float(output.mean()):.4f}, std={float(output.std()):.4f}"
        )
        print(
            f"log_mag: mean={float(log_mag.mean()):.4f}, range=[{float(log_mag.min()):.4f}, {float(log_mag.max()):.4f}]"
        )
        print(
            f"phase_logits: mean={float(phase_logits.mean()):.4f}, range=[{float(phase_logits.min()):.4f}, {float(phase_logits.max()):.4f}]"
        )

        # Check effective weights (weight normalized)
        v_norm = mx.sqrt(
            mx.sum(gen.conv_post.weight_v**2, axis=(1, 2), keepdims=True) + 1e-12
        )
        eff_weight = gen.conv_post.weight_g * gen.conv_post.weight_v / v_norm
        print(
            f"\nEffective weight: mean={float(eff_weight.mean()):.6f}, std={float(eff_weight.std()):.6f}"
        )

        return output

    # Reset RNG for fair comparison
    mx.random.seed(42)
    x_test = mx.random.normal((batch, length, channels)) * 0.1
    s_test = mx.random.normal((batch, style_dim)) * 0.1

    trace_to_conv_post(loaded_gen, x_test, s_test, f0, "LOADED MODEL")

    mx.random.seed(42)
    x_test = mx.random.normal((batch, length, channels)) * 0.1
    s_test = mx.random.normal((batch, style_dim)) * 0.1

    trace_to_conv_post(fresh_gen, x_test, s_test, f0, "FRESH MODEL")


if __name__ == "__main__":
    compare()
