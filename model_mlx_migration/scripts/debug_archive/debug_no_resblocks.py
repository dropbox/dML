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
Debug Generator without resblocks to isolate their effect.
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

    # Run through stages WITHOUT resblocks
    print("\n=== WITHOUT resblocks ===")

    for i in range(generator.num_upsamples):
        up = getattr(generator, f"ups_{i}")
        noise_conv = getattr(generator, f"noise_convs_{i}")
        noise_res = getattr(generator, f"noise_res_{i}")

        x = nn.leaky_relu(x, 0.1)

        # Noise path
        x_source = noise_conv(source)
        x_source = noise_res(x_source, style_mx)

        # Upsample
        x = up(x)
        if i == generator.num_upsamples - 1:
            x = mx.concatenate([x[:, 1:2, :], x], axis=1)

        # Match lengths
        if x_source.shape[1] < x.shape[1]:
            pad_len = x.shape[1] - x_source.shape[1]
            x_source = mx.pad(x_source, [(0, 0), (0, pad_len), (0, 0)])
        elif x_source.shape[1] > x.shape[1]:
            x_source = x_source[:, : x.shape[1], :]

        x = x + x_source

        # SKIP RESBLOCKS!

        mx.eval(x)
        print(
            f"Stage {i}: {x.shape}, range [{float(mx.min(x)):.4f}, {float(mx.max(x)):.4f}]"
        )

    x = nn.leaky_relu(x, 0.1)
    x = generator.conv_post(x)
    mx.eval(x)
    print(f"conv_post: range [{float(mx.min(x)):.4f}, {float(mx.max(x)):.4f}]")

    # ISTFT
    n_bins = generator.post_n_fft // 2 + 1
    log_mag = mx.clip(x[..., :n_bins], -10, 10)
    mag = mx.exp(log_mag)
    phase = mx.sin(x[..., n_bins:])
    audio = generator._istft_synthesis(mag, phase)
    audio = mx.clip(audio, -1.0, 1.0)
    mx.eval(audio)
    print(
        f"Audio: range [{float(mx.min(audio)):.4f}, {float(mx.max(audio)):.4f}], std {float(mx.std(audio)):.4f}"
    )

    # Compare
    ref_audio = ref["audio"]
    print(
        f"\nReference audio: range [{ref_audio.min():.4f}, {ref_audio.max():.4f}], std {ref_audio.std():.4f}"
    )

    # Correlation
    mlx_audio = np.array(audio).flatten()
    ref_flat = ref_audio.flatten()
    min_len = min(len(mlx_audio), len(ref_flat))
    corr = np.corrcoef(mlx_audio[:min_len], ref_flat[:min_len])[0, 1]
    print(f"Correlation: {corr:.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
