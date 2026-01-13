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
Debug Decoder signal flow to find where audio is attenuated.
"""

import sys

sys.path.insert(0, "tools/pytorch_to_mlx")

import mlx.core as mx
from converters.kokoro_converter import KokoroConverter


def debug_decoder():
    """Trace signal through Decoder to find attenuation."""
    print("=" * 60)
    print("Decoder Signal Flow Debug")
    print("=" * 60)

    print("\nLoading model with weights...")
    converter = KokoroConverter()
    model, config, state_dict = converter.load_from_hf()
    mx.eval(model)
    print("Model loaded")

    decoder = model.decoder
    generator = decoder.generator

    # Create test inputs matching what Decoder receives
    batch = 1
    seq_length = 32
    hidden_dim = config.hidden_dim  # 512

    # Simulate ASR features (output from predictor)
    asr_features = mx.random.normal((batch, seq_length, hidden_dim)) * 0.1
    f0 = mx.full((batch, seq_length), 1.0)  # Normalized F0
    noise = mx.random.normal((batch, seq_length)) * 0.1
    style = mx.random.normal((batch, config.style_dim)) * 0.1

    print("\nInputs:")
    print(
        f"  asr_features shape: {asr_features.shape}, rms: {float(mx.sqrt(mx.mean(asr_features**2))):.6f}"
    )
    print(f"  f0 shape: {f0.shape}, mean: {float(f0.mean()):.4f}")
    print(f"  noise shape: {noise.shape}, rms: {float(mx.sqrt(mx.mean(noise**2))):.6f}")
    print(f"  style shape: {style.shape}, rms: {float(mx.sqrt(mx.mean(style**2))):.6f}")

    # Step through Decoder manually
    print("\n" + "=" * 60)
    print("Step-by-step Decoder trace")
    print("=" * 60)

    # 1. F0 and noise convolutions
    f0_orig = f0
    f0_in = f0[:, :, None]
    n_in = noise[:, :, None]

    f0_proc = decoder.f0_conv(f0_in)
    n_proc = decoder.n_conv(n_in)
    mx.eval(f0_proc, n_proc)
    print("\n1. F0/Noise convs:")
    print(
        f"   f0_proc shape: {f0_proc.shape}, rms: {float(mx.sqrt(mx.mean(f0_proc**2))):.6f}"
    )
    print(
        f"   n_proc shape: {n_proc.shape}, rms: {float(mx.sqrt(mx.mean(n_proc**2))):.6f}"
    )

    # 2. ASR residual
    asr_res = decoder.asr_res(asr_features)
    mx.eval(asr_res)
    print("\n2. ASR residual:")
    print(
        f"   asr_res shape: {asr_res.shape}, rms: {float(mx.sqrt(mx.mean(asr_res**2))):.6f}"
    )

    # 3. Match lengths
    asr_len = asr_features.shape[1]
    f0_len = f0_proc.shape[1]

    if asr_len > f0_len:
        stride = asr_len // f0_len
        asr_down = asr_features[:, ::stride, :][:, :f0_len, :]
        asr_res_down = asr_res[:, ::stride, :][:, :f0_len, :]
    else:
        asr_down = asr_features
        asr_res_down = asr_res

    # 4. Concatenate for encode
    x_encode = mx.concatenate([asr_down, f0_proc, n_proc], axis=-1)
    print("\n3. Encode input:")
    print(
        f"   x_encode shape: {x_encode.shape}, rms: {float(mx.sqrt(mx.mean(x_encode**2))):.6f}"
    )

    # 5. Encode block
    x = decoder.encode(x_encode, style)
    mx.eval(x)
    print("\n4. After encode:")
    print(f"   x shape: {x.shape}, rms: {float(mx.sqrt(mx.mean(x**2))):.6f}")

    # 6. Decode blocks
    for i, block in enumerate(decoder.decode):
        # Resample F0/N to match current length if needed
        curr_len = x.shape[1]
        f0_len = f0_proc.shape[1]
        if curr_len != f0_len:
            # Upsample
            f0_up = mx.repeat(f0_proc, curr_len // f0_len, axis=1)[:, :curr_len, :]
            n_up = mx.repeat(n_proc, curr_len // f0_len, axis=1)[:, :curr_len, :]
            asr_res_up = mx.repeat(asr_res_down, curr_len // f0_len, axis=1)[
                :, :curr_len, :
            ]
        else:
            f0_up = f0_proc
            n_up = n_proc
            asr_res_up = asr_res_down

        # Concatenate
        x_in = mx.concatenate([x, asr_res_up, f0_up, n_up], axis=-1)
        x = block(x_in, style)
        mx.eval(x)
        print(f"\n5.{i} After decode[{i}]:")
        print(f"   x shape: {x.shape}, rms: {float(mx.sqrt(mx.mean(x**2))):.6f}")

    # 7. Generator
    print("\n" + "=" * 60)
    print("Generator trace")
    print("=" * 60)

    # Scale F0 to Hz
    gen = generator
    f0_hz = f0_orig * 200.0
    print(f"\nF0 to Hz: mean={float(f0_hz.mean()):.1f} Hz")

    # Upsample F0
    f0_up = mx.repeat(f0_hz[:, :, None], gen.f0_upsample, axis=1).squeeze(-1)
    print(f"F0 upsampled: shape={f0_up.shape}")

    # Source module
    har_source, noi_source, uv = gen.m_source(f0_up)
    mx.eval(har_source)
    print("\nSourceModule:")
    print(f"  har_source shape: {har_source.shape}")
    print(f"  har_source rms: {float(mx.sqrt(mx.mean(har_source**2))):.6f}")
    print(
        f"  har_source range: [{float(har_source.min()):.6f}, {float(har_source.max()):.6f}]"
    )

    # Check SineGen output
    sine_waves, uv2, _ = gen.m_source.l_sin_gen(f0_up)
    mx.eval(sine_waves)
    print("\n  SineGen output:")
    print(f"    shape: {sine_waves.shape}")
    print(f"    rms: {float(mx.sqrt(mx.mean(sine_waves**2))):.6f}")
    print(f"    range: [{float(sine_waves.min()):.6f}, {float(sine_waves.max()):.6f}]")

    # Check l_linear weights
    print("\n  l_linear weights:")
    print(f"    weight shape: {gen.m_source.l_linear.weight.shape}")
    print(f"    weight values: {gen.m_source.l_linear.weight}")
    print(f"    bias: {gen.m_source.l_linear.bias}")

    # Check before and after tanh
    linear_out = gen.m_source.l_linear(sine_waves)
    mx.eval(linear_out)
    print("\n  After l_linear (before tanh):")
    print(f"    rms: {float(mx.sqrt(mx.mean(linear_out**2))):.6f}")
    print(f"    range: [{float(linear_out.min()):.6f}, {float(linear_out.max()):.6f}]")

    # Now run full generator
    print("\n" + "=" * 60)
    print("Full Generator output")
    print("=" * 60)

    audio = gen(x, style, f0_orig)
    mx.eval(audio)
    print(f"Audio shape: {audio.shape}")
    print(f"Audio RMS: {float(mx.sqrt(mx.mean(audio**2))):.6f}")
    print(f"Audio max: {float(mx.abs(audio).max()):.6f}")

    # Now compare with random-weight generator
    print("\n" + "=" * 60)
    print("Comparing with random-weight generator")
    print("=" * 60)

    from converters.models.kokoro import Generator as GenClass

    random_gen = GenClass(config)

    random_audio = random_gen(x, style, f0_orig)
    mx.eval(random_audio)
    print("Random generator audio:")
    print(f"  Shape: {random_audio.shape}")
    print(f"  RMS: {float(mx.sqrt(mx.mean(random_audio**2))):.6f}")
    print(f"  Max: {float(mx.abs(random_audio).max()):.6f}")


if __name__ == "__main__":
    debug_decoder()
