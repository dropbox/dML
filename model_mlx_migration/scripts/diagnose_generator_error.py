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
Diagnose where error accumulates in the Generator by tracing step-by-step.

This script:
1. Loads PyTorch reference tensors
2. Runs MLX generator with source override
3. Captures intermediate outputs at each stage
4. Identifies which stage introduces the most error
"""

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def compare_arrays(name: str, a: np.ndarray, b: np.ndarray) -> dict:
    """Compare two arrays and return statistics."""
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    min_len = min(len(a_flat), len(b_flat))
    a_flat = a_flat[:min_len]
    b_flat = b_flat[:min_len]

    diff = np.abs(a_flat - b_flat)
    return {
        "name": name,
        "max_abs": float(diff.max()) if len(diff) > 0 else 0.0,
        "mean_abs": float(diff.mean()) if len(diff) > 0 else 0.0,
        "a_shape": list(a.shape),
        "b_shape": list(b.shape),
    }


def trace_generator_stages(mlx_model, ref: dict, style: mx.array):
    """Trace through generator stages to find error source."""
    generator = mlx_model.decoder.generator

    # Get the decoder input x from a forward pass
    # We need to run the decoder up to the generator call
    asr_nlc = mx.array(ref["asr_nlc"].astype(np.float32))
    f0 = mx.array(ref["F0_pred"].astype(np.float32))
    n = mx.array(ref["N_pred"].astype(np.float32))
    style_128 = mx.array(ref["style_128"].astype(np.float32))

    decoder = mlx_model.decoder

    # Manually trace through decoder to get generator input x
    f0_in = f0[:, :, None]
    n_in = n[:, :, None]

    f0_proc = decoder.f0_conv(f0_in)
    n_proc = decoder.n_conv(n_in)
    asr_res = decoder.asr_res(asr_nlc)

    x = mx.concatenate([asr_nlc, f0_proc, n_proc], axis=-1)
    x = decoder.encode(x, style_128)

    asr_res_down = asr_res
    decode_blocks = [decoder.decode_0, decoder.decode_1, decoder.decode_2, decoder.decode_3]
    for block in decode_blocks:
        x = mx.concatenate([x, asr_res_down, f0_proc, n_proc], axis=-1)
        x = block(x, style_128)
        if block.upsample:
            new_len = x.shape[1]
            asr_res_down = mx.repeat(asr_res_down, 2, axis=1)[:, :new_len, :]
            f0_proc = mx.repeat(f0_proc, 2, axis=1)[:, :new_len, :]
            n_proc = mx.repeat(n_proc, 2, axis=1)[:, :new_len, :]

    mx.eval(x)
    gen_input_x = x  # [batch, 126, 512]

    print(f"Generator input x shape: {gen_input_x.shape}")

    # Now trace through generator with source override
    gen_har = ref["gen_har"].astype(np.float32)  # [batch, 22, frames] NCL
    gen_har_nlc = gen_har.transpose(0, 2, 1)  # [batch, frames, 22] NLC
    source = mx.array(gen_har_nlc)

    print(f"Source (gen_har) shape: {source.shape}")

    # Trace generator stages
    x = gen_input_x
    s = style_128

    config = generator.config
    total_upp = 1
    for r in config.istft_upsample_rates:
        total_upp *= r
    total_upp *= generator.istft_hop_size

    print("\nGenerator stages:")
    print(f"  num_upsamples: {generator.num_upsamples}")
    print(f"  num_kernels: {generator.num_kernels}")

    stages = []

    for i in range(generator.num_upsamples):
        up = getattr(generator, f"ups_{i}")
        noise_conv = getattr(generator, f"noise_convs_{i}")
        noise_res = getattr(generator, f"noise_res_{i}")

        x = nn.leaky_relu(x, 0.1)
        stages.append(("leaky_relu_pre_" + str(i), np.array(x)))

        x_source = noise_conv(source)
        stages.append((f"noise_conv_{i}", np.array(x_source)))

        x_source = noise_res(x_source, s)
        stages.append((f"noise_res_{i}", np.array(x_source)))

        x = up(x)
        stages.append((f"up_{i}", np.array(x)))

        if i == generator.num_upsamples - 1:
            x = mx.concatenate([x[:, 1:2, :], x], axis=1)
            stages.append((f"reflect_pad_{i}", np.array(x)))

        # Check shape match
        if x_source.shape[1] != x.shape[1]:
            print(f"  WARNING: Shape mismatch at stage {i}: x={x.shape}, x_source={x_source.shape}")
            # Pad or trim
            if x_source.shape[1] < x.shape[1]:
                pad_len = x.shape[1] - x_source.shape[1]
                x_source = mx.pad(x_source, [(0,0), (0, pad_len), (0,0)])
            else:
                x_source = x_source[:, :x.shape[1], :]

        x = x + x_source
        stages.append((f"add_{i}", np.array(x)))

        # ResBlocks
        xs = None
        for j in range(generator.num_kernels):
            block_idx = i * generator.num_kernels + j
            if block_idx < generator._num_resblocks:
                resblock = getattr(generator, f"resblocks_{block_idx}")
                if xs is None:
                    xs = resblock(x, s)
                else:
                    xs = xs + resblock(x, s)
        if xs is not None:
            x = xs / generator.num_kernels
        stages.append((f"resblocks_{i}", np.array(x)))

    # Final processing
    x = nn.leaky_relu(x)  # default slope 0.01
    stages.append(("leaky_relu_final", np.array(x)))

    x = generator.conv_post(x)
    stages.append(("conv_post", np.array(x)))

    n_bins = generator.post_n_fft // 2 + 1
    log_mag = x[..., :n_bins]
    mag = mx.exp(log_mag)
    phase_logits = x[..., n_bins:]
    phase = mx.sin(phase_logits)

    stages.append(("mag", np.array(mag)))
    stages.append(("phase", np.array(phase)))

    audio = generator._istft_synthesis(mag, phase)
    mx.eval(audio)
    stages.append(("audio", np.array(audio)))

    return stages


def main():
    ref_dir = Path("/tmp/kokoro_ref_seed0")
    if not ref_dir.exists():
        print(f"Reference directory not found: {ref_dir}")
        return 1

    ref = np.load(ref_dir / "tensors.npz")

    # Load MLX model
    from tools.pytorch_to_mlx.converters import KokoroConverter
    converter = KokoroConverter()
    mlx_model, _, _ = converter.load_from_hf("hexgrad/Kokoro-82M")
    mlx_model.set_deterministic(True)

    style_128 = mx.array(ref["style_128"].astype(np.float32))

    print("=" * 72)
    print("Generator Stage Trace")
    print("=" * 72)

    stages = trace_generator_stages(mlx_model, ref, style_128)

    print("\n" + "=" * 72)
    print("Stage Shapes")
    print("=" * 72)
    for name, arr in stages:
        print(f"  {name}: {arr.shape}")

    # Compare final audio
    audio_ref = ref["audio"].astype(np.float32)
    audio_mlx = stages[-1][1].reshape(-1)

    print("\n" + "=" * 72)
    print("Final Audio Comparison")
    print("=" * 72)
    stats = compare_arrays("audio", audio_ref, audio_mlx)
    print(f"  Max abs error: {stats['max_abs']:.6f}")
    print(f"  Mean abs error: {stats['mean_abs']:.6f}")
    print(f"  PyTorch shape: {audio_ref.shape}")
    print(f"  MLX shape: {audio_mlx.shape}")

    # Check if mag/phase look reasonable
    mag_arr = None
    phase_arr = None
    for name, arr in stages:
        if name == "mag":
            mag_arr = arr
        elif name == "phase":
            phase_arr = arr

    if mag_arr is not None:
        print(f"\n  Mag stats: min={mag_arr.min():.4f}, max={mag_arr.max():.4f}, mean={mag_arr.mean():.4f}")
    if phase_arr is not None:
        print(f"  Phase stats: min={phase_arr.min():.4f}, max={phase_arr.max():.4f}, mean={phase_arr.mean():.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
