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
Debug Generator upsampling stages with loaded weights vs random weights.
"""

import sys

sys.path.insert(0, "tools/pytorch_to_mlx")

import mlx.core as mx
from converters.kokoro_converter import KokoroConverter
from converters.models.kokoro import Generator


def debug_generator_stages():
    """Compare loaded vs random weight Generator stages."""
    print("=" * 60)
    print("Generator Stage-by-Stage Debug")
    print("=" * 60)

    print("\nLoading model with weights...")
    converter = KokoroConverter()
    model, config, state_dict = converter.load_from_hf()
    mx.eval(model)
    print("Loaded model ready")

    gen_loaded = model.decoder.generator
    gen_random = Generator(config)

    # Create test inputs
    batch = 1
    length = 32
    channels = 512

    # Use same inputs for both
    x_input = mx.random.normal((batch, length, channels)) * 0.1
    s = mx.random.normal((batch, config.style_dim)) * 0.1
    f0 = mx.full((batch, length // 2), 1.0)

    print("\nInputs:")
    print(f"  x shape: {x_input.shape}, rms: {float(mx.sqrt(mx.mean(x_input**2))):.6f}")

    for name, gen in [("Loaded", gen_loaded), ("Random", gen_random)]:
        print(f"\n{'=' * 40}")
        print(f"Processing with {name} weights")
        print(f"{'=' * 40}")

        x = x_input

        # Scale F0 and upsample
        f0_hz = f0 * 200.0
        f0_up = mx.repeat(f0_hz[:, :, None], gen.f0_upsample, axis=1).squeeze(-1)

        # Source module
        har_source, noi_source, uv = gen.m_source(f0_up)
        mx.eval(har_source)
        print("\nSource module:")
        print(f"  har_source rms: {float(mx.sqrt(mx.mean(har_source**2))):.6f}")

        # STFT
        har_source_1d = har_source.squeeze(-1)
        har_spec, har_phase = gen.stft.transform(har_source_1d)
        mx.eval(har_spec)
        print("\nSTFT:")
        print(
            f"  har_spec shape: {har_spec.shape}, rms: {float(mx.sqrt(mx.mean(har_spec**2))):.6f}"
        )

        # Concatenate and transpose
        har = mx.concatenate([har_spec, har_phase], axis=1)
        har = har.transpose(0, 2, 1)
        print(
            f"  har (NLC) shape: {har.shape}, rms: {float(mx.sqrt(mx.mean(har**2))):.6f}"
        )

        # Upsampling stages
        for i, (up, noise_conv, noise_res) in enumerate(
            zip(gen.ups, gen.noise_convs, gen.noise_res)
        ):
            print(f"\n--- Stage {i} ---")
            print(f"x before: shape={x.shape}, rms={float(mx.sqrt(mx.mean(x**2))):.6f}")

            # Check up layer weights
            print("\nUpsample layer weights:")
            print(f"  weight shape: {up.weight.shape}")
            w_rms = float(mx.sqrt(mx.mean(up.weight**2)))
            print(f"  weight rms: {w_rms:.6f}")

            x = mx.maximum(x, x * 0.1)  # leaky_relu
            x = up(x)
            mx.eval(x)
            print(
                f"After upsample: shape={x.shape}, rms={float(mx.sqrt(mx.mean(x**2))):.6f}"
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

            # Process through noise_conv
            print("\nNoise conv weights:")
            print(f"  weight_v shape: {noise_conv.weight_v.shape}")
            w_rms = float(mx.sqrt(mx.mean(noise_conv.weight_v**2)))
            print(f"  weight_v rms: {w_rms:.6f}")

            x_source = noise_conv(har_down)
            mx.eval(x_source)
            print(
                f"After noise_conv: shape={x_source.shape}, rms={float(mx.sqrt(mx.mean(x_source**2))):.6f}"
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
                f"After noise_res: shape={x_source.shape}, rms={float(mx.sqrt(mx.mean(x_source**2))):.6f}"
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
                f"After ResBlocks: shape={x.shape}, rms={float(mx.sqrt(mx.mean(x**2))):.6f}"
            )

        # Final conv
        print("\n--- Final conv ---")
        print(f"conv_post weight_v shape: {gen.conv_post.weight_v.shape}")
        w_rms = float(mx.sqrt(mx.mean(gen.conv_post.weight_v**2)))
        print(f"conv_post weight_v rms: {w_rms:.6f}")

        x = mx.maximum(x, x * 0.1)  # leaky_relu
        x = gen.conv_post(x)
        mx.eval(x)
        print(
            f"After conv_post: shape={x.shape}, rms={float(mx.sqrt(mx.mean(x**2))):.6f}"
        )

        # Spectrogram
        n_bins = gen.post_n_fft // 2 + 1
        log_mag = mx.clip(x[..., :n_bins], -10, 10)
        mag = mx.exp(log_mag)
        phase = mx.sin(x[..., n_bins:]) * 3.14159
        mx.eval(mag, phase)

        print("\nSpectrogram:")
        print(
            f"  log_mag range: [{float(log_mag.min()):.4f}, {float(log_mag.max()):.4f}]"
        )
        print(f"  mag rms: {float(mx.sqrt(mx.mean(mag**2))):.6f}")

        # ISTFT
        mag_t = mag.transpose(0, 2, 1)
        phase_t = phase.transpose(0, 2, 1)
        audio = gen.post_stft.inverse(mag_t, phase_t)
        mx.eval(audio)

        print("\nFinal audio:")
        print(f"  shape: {audio.shape}")
        print(f"  rms: {float(mx.sqrt(mx.mean(audio**2))):.6f}")
        print(f"  max: {float(mx.abs(audio).max()):.6f}")


if __name__ == "__main__":
    debug_generator_stages()
