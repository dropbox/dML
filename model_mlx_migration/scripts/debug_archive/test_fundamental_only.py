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
Test Generator with only fundamental harmonic to see if phase is the issue.
"""

import math
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import scipy.io.wavfile as wavfile

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.pytorch_to_mlx.converters import KokoroConverter


def modified_source_module(
    f0,
    l_linear_weight,
    l_linear_bias,
    sample_rate=24000,
    sine_amp=0.1,
    num_harmonics=9,
    upp=300,
):
    """
    Modified SourceModule that uses zero initial phase for ALL harmonics.
    This matches what should happen if PyTorch had same random seed.
    """
    batch, length = f0.shape
    samples = length * upp

    # Upsample F0
    f0_up = mx.repeat(f0[:, :, None], upp, axis=1).squeeze(-1)

    # UV mask
    uv = (f0_up > 10.0).astype(mx.float32)

    # Generate harmonics
    harmonics = []
    for h in range(1, num_harmonics + 1):
        rad_values = (f0_up * h / sample_rate) % 1.0

        # Downsample
        t_down = mx.arange(length) * (samples - 1) / (length - 1)
        t_floor = mx.floor(t_down).astype(mx.int32)
        t_ceil = mx.minimum(t_floor + 1, samples - 1)
        t_frac = t_down - t_floor.astype(mx.float32)

        rad_floor = rad_values[:, t_floor]
        rad_ceil = rad_values[:, t_ceil]
        rad_values_down = rad_floor * (1 - t_frac) + rad_ceil * t_frac

        # Cumsum
        phase_low = mx.cumsum(rad_values_down, axis=1) * 2 * math.pi

        # Upsample
        phase_scaled = phase_low * upp

        t_up = mx.arange(samples) * (length - 1) / (samples - 1)
        t_floor = mx.floor(t_up).astype(mx.int32)
        t_ceil = mx.minimum(t_floor + 1, length - 1)
        t_frac = t_up - t_floor.astype(mx.float32)

        phase_floor = phase_scaled[:, t_floor]
        phase_ceil = phase_scaled[:, t_ceil]
        phase = phase_floor * (1 - t_frac) + phase_ceil * t_frac

        sine = mx.sin(phase) * sine_amp
        harmonics.append(sine)

    # Stack
    harmonics_stack = mx.stack(harmonics, axis=-1)

    # Apply UV mask
    uv_expanded = uv[:, :, None]
    sine_waves = harmonics_stack * uv_expanded

    # Combine with l_linear
    # l_linear: [num_harmonics] -> [1]
    fc = nn.Linear(num_harmonics, 1)
    fc.weight = mx.array(l_linear_weight)
    fc.bias = mx.array(l_linear_bias)

    har_source = mx.tanh(fc(sine_waves))

    return har_source, uv_expanded


def main():
    from huggingface_hub import hf_hub_download

    ref = np.load("/tmp/kokoro_ref/tensors.npz")
    gen_traces = np.load("/tmp/kokoro_ref/generator_traces.npz")

    print("=== Loading MLX Model ===")
    converter = KokoroConverter()
    model, config, _ = converter.load_from_hf()
    mx.eval(model)

    F0_mx = mx.array(ref["F0_pred"])
    style_mx = mx.array(ref["style_128"])

    pt_gen_input = gen_traces["generator_input_ncl"]
    gen_input = mx.array(pt_gen_input).transpose(0, 2, 1)

    generator = model.decoder.generator

    # Get l_linear weights
    ckpt_path = hf_hub_download("hexgrad/Kokoro-82M", "kokoro-v1_0.pth")
    import torch

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    l_linear_w = ckpt["decoder"]["module.generator.m_source.l_linear.weight"].numpy()
    l_linear_b = ckpt["decoder"]["module.generator.m_source.l_linear.bias"].numpy()

    print("\n=== Testing with Modified SourceModule ===")

    x = gen_input

    total_upp = 1
    for r in generator.config.istft_upsample_rates:
        total_upp *= r
    total_upp *= generator.istft_hop_size

    # Use modified source module
    har_source, uv = modified_source_module(
        F0_mx,
        l_linear_w,
        l_linear_b,
        sample_rate=24000,
        sine_amp=0.1,
        num_harmonics=9,
        upp=total_upp,
    )
    mx.eval(har_source)

    print(
        f"har_source: range [{float(mx.min(har_source)):.4f}, {float(mx.max(har_source)):.4f}]"
    )

    # Compare with PT
    internal = np.load("/tmp/kokoro_ref/generator_internal_traces.npz")
    if "m_source_out_0" in internal:
        pt_har = internal["m_source_out_0"]
        corr = np.corrcoef(np.array(har_source).flatten(), pt_har.flatten())[0, 1]
        print(f"Correlation with PT har_source: {corr:.6f}")

    # Continue with generator
    har_source_1d = har_source.squeeze(-1)
    source = generator._source_stft(har_source_1d)
    mx.eval(source)

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

    x = nn.leaky_relu(x, 0.1)
    conv_post_out = generator.conv_post(x)
    mx.eval(conv_post_out)

    # ISTFT
    n_bins = generator.post_n_fft // 2 + 1
    log_mag = mx.clip(conv_post_out[..., :n_bins], -10, 10)
    mag = mx.exp(log_mag)
    phase = mx.sin(conv_post_out[..., n_bins:])
    audio = generator._istft_synthesis(mag, phase)
    audio = mx.clip(audio, -1.0, 1.0)
    mx.eval(audio)

    audio_np = np.array(audio).flatten()
    print(
        f"\nAudio: range [{audio_np.min():.4f}, {audio_np.max():.4f}], std={audio_np.std():.4f}"
    )

    # Save audio
    output_path = Path("/tmp/kokoro_ref/mlx_audio_modified.wav")
    audio_int = np.clip(audio_np * 32767, -32768, 32767).astype(np.int16)
    wavfile.write(output_path, 24000, audio_int)
    print(f"Saved: {output_path}")

    # Try transcription
    try:
        import whisper

        model_w = whisper.load_model("base")
        result = model_w.transcribe(str(output_path), language="en")
        print(f"\nTranscription: {result['text']}")
    except Exception:
        print("Whisper not available")

    return 0


if __name__ == "__main__":
    sys.exit(main())
