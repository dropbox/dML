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

"""Trace generator with PyTorch's source to find remaining error."""

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def compare(name, mlx_val, pt_val, threshold=0.001):
    """Compare and print differences."""
    mlx_np = np.array(mlx_val) if hasattr(mlx_val, 'tolist') else mlx_val
    pt_np = pt_val.astype(np.float32) if pt_val.dtype != np.float32 else pt_val

    if mlx_np.shape != pt_np.shape:
        min_shape = tuple(min(m, p) for m, p in zip(mlx_np.shape, pt_np.shape))
        slices = tuple(slice(0, s) for s in min_shape)
        mlx_np = mlx_np[slices]
        pt_np = pt_np[slices]

    diff = np.abs(mlx_np - pt_np)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())

    status = "PASS" if max_diff < threshold else "FAIL"
    print(f"{name:<30} max={max_diff:.6f} mean={mean_diff:.6f} {status}")
    return max_diff


def main():
    ref_dir = Path("/tmp/kokoro_ref_decoder")
    ref = np.load(ref_dir / "tensors.npz")
    gen_ref = np.load(ref_dir / "generator_intermediates.npz")

    from tools.pytorch_to_mlx.converters import KokoroConverter
    converter = KokoroConverter()
    mlx_model, _, _ = converter.load_from_hf("hexgrad/Kokoro-82M")
    mlx_model.set_deterministic(True)

    generator = mlx_model.decoder.generator

    print("=" * 72)
    print("Generator Trace with PyTorch Source")
    print("=" * 72)

    # Get inputs
    x_nlc = mx.array(ref["generator_input_x"].astype(np.float32))
    style = mx.array(ref["style_128"].astype(np.float32))

    # Use PyTorch's source (STFT output)
    source = mx.array(gen_ref["gen_har"].astype(np.float32))

    print(f"Input x shape: {x_nlc.shape}")
    print(f"Source shape: {source.shape}")
    print(f"Style shape: {style.shape}")

    # Run MLX generator step by step using PyTorch source
    x = x_nlc

    print("\n--- Generator Loop (with PT source) ---")
    for i in range(generator.num_upsamples):
        up = getattr(generator, f"ups_{i}")
        noise_conv = getattr(generator, f"noise_convs_{i}")
        noise_res = getattr(generator, f"noise_res_{i}")

        # leaky_relu before ups
        x = nn.leaky_relu(x, 0.1)
        mx.eval(x)
        compare(f"pre_lrelu_{i}", x, gen_ref[f"gen_pre_lrelu_{i}"])

        # noise_convs(source) - using PyTorch source
        x_source = noise_conv(source)
        mx.eval(x_source)
        compare(f"noise_convs_{i}", x_source, gen_ref[f"gen_noise_convs_{i}_out"])

        # noise_res
        x_source = noise_res(x_source, style)
        mx.eval(x_source)
        compare(f"noise_res_{i}", x_source, gen_ref[f"gen_noise_res_{i}_out"])

        # ups
        x = up(x)
        mx.eval(x)
        compare(f"ups_{i}", x, gen_ref[f"gen_ups_{i}_out"])

        # reflection pad at last stage
        if i == generator.num_upsamples - 1:
            x = mx.concatenate([x[:, 1:2, :], x], axis=1)
            mx.eval(x)

        # add
        x = x + x_source
        mx.eval(x)
        compare(f"after_add_{i}", x, gen_ref[f"gen_after_add_{i}"])

        # resblocks
        xs = None
        for j in range(generator.num_kernels):
            block_idx = i * generator.num_kernels + j
            if block_idx < generator._num_resblocks:
                resblock = getattr(generator, f"resblocks_{block_idx}")
                if xs is None:
                    xs = resblock(x, style)
                else:
                    xs = xs + resblock(x, style)
        if xs is not None:
            x = xs / generator.num_kernels
        mx.eval(x)
        compare(f"resblocks_{i}", x, gen_ref[f"gen_resblocks_{i}_out"])

    # Final leaky_relu
    x = nn.leaky_relu(x)
    mx.eval(x)
    compare("final_lrelu", x, gen_ref["gen_final_lrelu"])

    # conv_post
    x = generator.conv_post(x)
    mx.eval(x)
    compare("conv_post", x, gen_ref["gen_conv_post_out"])

    # spec/phase split
    n_bins = generator.post_n_fft // 2 + 1
    raw_spec = x[..., :n_bins]
    raw_phase = x[..., n_bins:]

    compare("raw_spec", raw_spec, gen_ref["gen_raw_spec"])
    compare("raw_phase", raw_phase, gen_ref["gen_raw_phase"])

    # exp/sin
    spec = mx.exp(raw_spec)
    phase = mx.sin(raw_phase)
    mx.eval(spec, phase)

    compare("spec (after exp)", spec, gen_ref["gen_spec"])
    compare("phase (after sin)", phase, gen_ref["gen_phase"])

    # ISTFT
    print("\n--- ISTFT ---")
    # spec and phase are [batch, frames, n_bins], already in correct format for _istft_synthesis
    audio = generator._istft_synthesis(spec, phase)
    mx.eval(audio)

    pt_audio = gen_ref["gen_audio"]
    compare("audio", audio, pt_audio)

    # Also test full generator with debug override
    print("\n--- Full Generator with PT source ---")
    from tools.pytorch_to_mlx.converters import KokoroConverter
    f0 = mx.array(ref["F0_pred"].astype(np.float32))
    x_nlc = mx.array(ref["generator_input_x"].astype(np.float32))
    debug_overrides = {"source": source}
    audio_full = generator(x_nlc, style, f0, _debug_overrides=debug_overrides)
    mx.eval(audio_full)
    compare("audio (full gen)", audio_full, pt_audio)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
