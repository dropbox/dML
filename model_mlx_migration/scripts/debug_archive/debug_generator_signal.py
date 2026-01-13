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
Debug Generator signal flow to find where audio is being attenuated.
"""

import sys

sys.path.insert(0, "tools/pytorch_to_mlx")

import mlx.core as mx
from converters.models.kokoro import Generator
from converters.models.kokoro_modules import KokoroConfig


def debug_signal_flow():
    """Trace signal through Generator to find attenuation."""
    print("=" * 60)
    print("Generator Signal Flow Debug")
    print("=" * 60)

    config = KokoroConfig()
    gen = Generator(config)

    # Print config values
    print("\nConfig:")
    print(f"  total_upsample: {gen.total_upsample}")
    print(f"  f0_upsample: {gen.f0_upsample}")
    print(f"  source_n_fft: {gen.source_n_fft}")
    print(f"  source_hop_size: {gen.source_hop_size}")

    # Create test inputs
    batch = 1
    length = 64  # Token sequence length
    channels = 512  # Initial channel count

    x = mx.random.normal((batch, length, channels)) * 0.1  # Small init
    s = mx.random.normal((batch, config.style_dim)) * 0.1
    f0 = mx.full((batch, length // 2), 1.0)  # Normalized F0 around 1.0

    print("\nInputs:")
    print(f"  x shape: {x.shape}, rms: {float(mx.sqrt(mx.mean(x**2))):.6f}")
    print(f"  s shape: {s.shape}, rms: {float(mx.sqrt(mx.mean(s**2))):.6f}")
    print(f"  f0 shape: {f0.shape}, mean: {float(f0.mean()):.6f}")

    # Step through Generator manually
    print("\n" + "=" * 60)
    print("Step-by-step signal trace")
    print("=" * 60)

    # 1. F0 scaling
    f0_hz = f0 * 200.0
    print(f"\n1. F0 to Hz: mean={float(f0_hz.mean()):.1f} Hz")

    # 2. F0 upsampling
    f0_up = mx.repeat(f0_hz[:, :, None], gen.f0_upsample, axis=1).squeeze(-1)
    print(f"2. F0 upsampled: shape={f0_up.shape}")

    # 3. Source module
    har_source, noi_source, uv = gen.m_source(f0_up)
    mx.eval(har_source, uv)
    print("\n3. SourceModule output:")
    print(f"   har_source shape: {har_source.shape}")
    print(f"   har_source rms: {float(mx.sqrt(mx.mean(har_source**2))):.6f}")
    print(f"   har_source max: {float(mx.abs(har_source).max()):.6f}")
    print(f"   har_source min: {float(har_source.min()):.6f}")
    print(f"   uv (voiced ratio): {float(uv.mean()):.4f}")

    # 4. STFT
    har_source_1d = har_source.squeeze(-1)
    har_spec, har_phase = gen.stft.transform(har_source_1d)
    mx.eval(har_spec, har_phase)
    print("\n4. STFT output:")
    print(f"   har_spec shape: {har_spec.shape}")
    print(f"   har_spec rms: {float(mx.sqrt(mx.mean(har_spec**2))):.6f}")
    print(f"   har_spec max: {float(mx.abs(har_spec).max()):.6f}")
    print(f"   har_phase shape: {har_phase.shape}")
    print(
        f"   har_phase range: [{float(har_phase.min()):.4f}, {float(har_phase.max()):.4f}]"
    )

    # 5. Concatenate
    har = mx.concatenate([har_spec, har_phase], axis=1)
    har = har.transpose(0, 2, 1)
    print(f"\n5. Concatenated har (NCL->NLC): shape={har.shape}")
    print(f"   rms: {float(mx.sqrt(mx.mean(har**2))):.6f}")

    # 6. Progressive upsampling
    print("\n6. Upsampling stages:")
    for i, (up, noise_conv, noise_res) in enumerate(
        zip(gen.ups, gen.noise_convs, gen.noise_res)
    ):
        x = mx.maximum(x, x * 0.1)  # leaky_relu
        x = up(x)
        mx.eval(x)

        print(f"\n   Stage {i}:")
        print(
            f"     After upsample: shape={x.shape}, rms={float(mx.sqrt(mx.mean(x**2))):.6f}"
        )

        # Match source length
        har_len = har.shape[1]
        target_len = x.shape[1]

        if har_len > target_len:
            stride = har_len // target_len
            har_down = har[:, ::stride, :][:, :target_len, :]
        elif har_len < target_len:
            repeat_factor = (target_len + har_len - 1) // har_len
            har_down = mx.repeat(har, repeat_factor, axis=1)[:, :target_len, :]
        else:
            har_down = har

        print(
            f"     har_down shape: {har_down.shape}, rms={float(mx.sqrt(mx.mean(har_down**2))):.6f}"
        )

        # Process through noise_conv
        x_source = noise_conv(har_down)
        mx.eval(x_source)
        print(
            f"     After noise_conv: shape={x_source.shape}, rms={float(mx.sqrt(mx.mean(x_source**2))):.6f}"
        )

        # Match lengths
        x_source_len = x_source.shape[1]
        if x_source_len > target_len:
            x_source = x_source[:, :target_len, :]
        elif x_source_len < target_len:
            repeat_factor = (target_len + x_source_len - 1) // x_source_len
            x_source = mx.repeat(x_source, repeat_factor, axis=1)[:, :target_len, :]

        # Style residual
        x_source = noise_res(x_source, s)
        mx.eval(x_source)
        print(
            f"     After noise_res: shape={x_source.shape}, rms={float(mx.sqrt(mx.mean(x_source**2))):.6f}"
        )

        x = x + x_source

        # ResBlocks
        xs = None
        for j in range(gen.num_kernels):
            block_idx = i * gen.num_kernels + j
            if block_idx < len(gen.resblocks):
                if xs is None:
                    xs = gen.resblocks[block_idx](x)
                else:
                    xs = xs + gen.resblocks[block_idx](x)
        if xs is not None:
            x = xs / gen.num_kernels
        mx.eval(x)
        print(
            f"     After ResBlocks: shape={x.shape}, rms={float(mx.sqrt(mx.mean(x**2))):.6f}"
        )

    # 7. Final conv
    print("\n7. Final convolution:")
    x = mx.maximum(x, x * 0.1)  # leaky_relu
    x = gen.conv_post(x)
    mx.eval(x)
    print(
        f"   After conv_post: shape={x.shape}, rms={float(mx.sqrt(mx.mean(x**2))):.6f}"
    )

    # 8. Spectrogram extraction
    n_bins = gen.post_n_fft // 2 + 1
    log_mag = mx.clip(x[..., :n_bins], -10, 10)
    mag = mx.exp(log_mag)
    phase = mx.sin(x[..., n_bins:]) * 3.14159
    mx.eval(mag, phase)

    print("\n8. Spectrogram:")
    print(f"   log_mag range: [{float(log_mag.min()):.4f}, {float(log_mag.max()):.4f}]")
    print(f"   mag range: [{float(mag.min()):.4f}, {float(mag.max()):.4f}]")
    print(f"   mag rms: {float(mx.sqrt(mx.mean(mag**2))):.6f}")
    print(f"   phase range: [{float(phase.min()):.4f}, {float(phase.max()):.4f}]")

    # 9. ISTFT
    mag_t = mag.transpose(0, 2, 1)
    phase_t = phase.transpose(0, 2, 1)
    audio = gen.post_stft.inverse(mag_t, phase_t)
    mx.eval(audio)

    print("\n9. Final audio:")
    print(f"   shape: {audio.shape}")
    print(f"   rms: {float(mx.sqrt(mx.mean(audio**2))):.6f}")
    print(f"   max: {float(mx.abs(audio).max()):.6f}")

    print("\n" + "=" * 60)

    # Now compare with loaded weights
    print("\nComparing with loaded weights...")

    from pathlib import Path

    from converters.kokoro_converter import KokoroConverter

    weights_path = Path.home() / "models" / "kokoro" / "kokoro-v1_0.pth"
    if not weights_path.exists():
        print(f"Weights not found at {weights_path}")
        return

    converter = KokoroConverter()
    model, _, _ = converter.load_from_hf()
    mx.eval(model)

    # Check m_source weights
    print("\nLoaded m_source weights:")
    print(
        f"  l_linear.weight shape: {model.decoder.generator.m_source.l_linear.weight.shape}"
    )
    print(f"  l_linear.weight: {model.decoder.generator.m_source.l_linear.weight}")
    print(f"  l_linear.bias: {model.decoder.generator.m_source.l_linear.bias}")

    # Test with loaded model
    print("\nTesting with loaded model:")
    audio = model.decoder.generator(x, s, f0)
    mx.eval(audio)
    print(f"  Audio shape: {audio.shape}")
    print(f"  Audio RMS: {float(mx.sqrt(mx.mean(audio**2))):.6f}")
    print(f"  Audio max: {float(mx.abs(audio).max()):.6f}")

    # Debug source module specifically
    print("\n" + "=" * 60)
    print("Debugging SourceModuleHnNSF")
    print("=" * 60)

    source = model.decoder.generator.m_source

    # Test with clear F0 signal
    test_f0 = mx.full((1, 1000), 200.0)  # 200 Hz
    sine_out, noise_out, uv = source(test_f0)
    mx.eval(sine_out, uv)

    print("\nSourceModule with 200 Hz F0:")
    print(f"  Input shape: {test_f0.shape}")
    print(f"  sine_out shape: {sine_out.shape}")
    print(f"  sine_out rms: {float(mx.sqrt(mx.mean(sine_out**2))):.6f}")
    print(f"  sine_out max: {float(mx.abs(sine_out).max()):.6f}")
    print(f"  uv (voiced): {float(uv.mean()):.4f}")

    # Check l_sin_gen output
    sine_waves, uv2, _ = source.l_sin_gen(test_f0)
    mx.eval(sine_waves)
    print("\n  SineGen output:")
    print(f"    sine_waves shape: {sine_waves.shape}")
    print(f"    sine_waves rms: {float(mx.sqrt(mx.mean(sine_waves**2))):.6f}")
    print(f"    sine_waves max: {float(mx.abs(sine_waves).max()):.6f}")

    # Check l_linear application
    linear_out = source.l_linear(sine_waves)
    mx.eval(linear_out)
    print("\n  After l_linear (before tanh):")
    print(f"    shape: {linear_out.shape}")
    print(f"    rms: {float(mx.sqrt(mx.mean(linear_out**2))):.6f}")
    print(f"    max: {float(mx.abs(linear_out).max()):.6f}")

    tanh_out = mx.tanh(linear_out)
    mx.eval(tanh_out)
    print("\n  After tanh:")
    print(f"    rms: {float(mx.sqrt(mx.mean(tanh_out**2))):.6f}")
    print(f"    max: {float(mx.abs(tanh_out).max()):.6f}")


if __name__ == "__main__":
    debug_signal_flow()
