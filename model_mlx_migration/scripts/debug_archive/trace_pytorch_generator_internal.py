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
Trace through PyTorch Generator internals to find where MLX diverges.

Run in kokoro PyTorch environment.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


def main():
    ref_path = Path("/tmp/kokoro_ref/generator_traces.npz")
    if not ref_path.exists():
        print("ERROR: Run export_generator_intermediates.py first")
        return 1

    pt_traces = dict(np.load(ref_path))
    ref = np.load("/tmp/kokoro_ref/tensors.npz")

    import torch
    import torch.nn.functional as F

    try:
        from kokoro import KModel
    except ImportError:
        print("ERROR: kokoro package not found. Run in kokoro environment.")
        return 1

    print("=== Loading Kokoro Model ===")
    model = KModel().to("cpu").eval()
    gen = model.decoder.generator

    # Generator input from previous trace
    gen_input_ncl = pt_traces["generator_input_ncl"]  # [1, 512, 126]
    style = ref["style_128"]  # [1, 128]
    f0 = ref["F0_pred"]  # [1, 126]

    gen_input_pt = torch.tensor(gen_input_ncl)
    style_pt = torch.tensor(style)
    f0_pt = torch.tensor(f0)

    print(f"Generator input: {gen_input_pt.shape}")
    print(f"Style: {style_pt.shape}")
    print(f"F0: {f0_pt.shape}")

    traces = {}

    with torch.no_grad():
        # The Generator.forward() signature and internals
        # Let's trace through it step by step

        x = gen_input_pt  # [1, 512, 126] NCL format

        print("\n=== Generator Structure ===")
        for name, mod in gen.named_children():
            print(f"  {name}: {type(mod).__name__}")

        # Check upsample parameters
        print("\n=== Upsample Rates ===")
        if hasattr(gen, "ups"):
            for i, up in enumerate(gen.ups):
                print(f"  ups[{i}]: {up}")

        # Source module parameters
        print("\n=== Source Module ===")
        m_source = gen.m_source
        print(f"  m_source type: {type(m_source)}")
        for name, param in m_source.named_parameters():
            print(f"    {name}: {param.shape}")

        # Call m_source to get harmonic source
        # It expects (f0_frames) - let's see signature
        # The Generator.forward likely upsamples F0 first

        # Check f0_upsamp
        f0_upsamp = gen.f0_upsamp
        print(f"\nf0_upsamp scale_factor: {f0_upsamp.scale_factor}")

        # Upsample F0
        f0_up = f0_upsamp(f0_pt.unsqueeze(1))  # [1, 1, upsampled_len]
        print(f"F0 upsampled: {f0_up.shape}")
        traces["f0_up"] = f0_up.numpy()

        # Call m_source
        # The actual signature depends on implementation
        # Let's check what m_source expects
        har = m_source(f0_up)  # SourceModuleHnNSF expects [B, 1, samples]
        print(f"m_source output shape: {har.shape}")
        traces["m_source_out"] = har.numpy()

        # STFT on harmonic source
        stft = gen.stft
        print(f"\nSTFT module: {type(stft)}")

        # Apply STFT
        har_spec = stft(har)  # [B, n_fft//2+1, frames] complex
        print(f"STFT output shape: {har_spec.shape}")
        traces["stft_out"] = har_spec.numpy()

        # Get magnitude and phase
        spec_real = har_spec.real
        spec_imag = har_spec.imag
        mag = torch.sqrt(spec_real**2 + spec_imag**2)
        phase = torch.atan2(spec_imag, spec_real)
        print(f"Mag shape: {mag.shape}, Phase shape: {phase.shape}")

        # Concatenate to get noise_conv input
        source = torch.cat([mag, phase], dim=1)  # [B, 22, frames]
        print(f"Source (mag+phase) shape: {source.shape}")
        print(f"Source range: [{source.min():.4f}, {source.max():.4f}]")
        traces["source_concat"] = source.numpy()

        # Now trace through progressive upsampling
        print("\n=== Progressive Upsampling ===")

        for i in range(len(gen.ups)):
            x = F.leaky_relu(x, 0.1)
            traces[f"stage{i}_after_leaky"] = x.numpy()
            print(
                f"Stage {i} after leaky_relu: {x.shape}, range: [{x.min():.4f}, {x.max():.4f}]"
            )

            # Noise conv
            x_source = gen.noise_convs[i](source)
            traces[f"stage{i}_noise_conv"] = x_source.numpy()
            print(
                f"Stage {i} noise_conv: {x_source.shape}, range: [{x_source.min():.4f}, {x_source.max():.4f}]"
            )

            # Noise res (AdaIN resblock with style)
            x_source = gen.noise_res[i](x_source, style_pt)
            traces[f"stage{i}_noise_res"] = x_source.numpy()
            print(
                f"Stage {i} noise_res: {x_source.shape}, range: [{x_source.min():.4f}, {x_source.max():.4f}]"
            )

            # Upsample
            x = gen.ups[i](x)
            traces[f"stage{i}_ups"] = x.numpy()
            print(f"Stage {i} ups: {x.shape}, range: [{x.min():.4f}, {x.max():.4f}]")

            # Reflection pad on last stage
            if i == len(gen.ups) - 1:
                x = gen.reflection_pad(x)
                traces[f"stage{i}_reflect_pad"] = x.numpy()
                print(f"Stage {i} reflect_pad: {x.shape}")

            # Align lengths
            if x_source.shape[2] < x.shape[2]:
                x_source = F.pad(x_source, (0, x.shape[2] - x_source.shape[2]))
            elif x_source.shape[2] > x.shape[2]:
                x_source = x_source[..., : x.shape[2]]
            traces[f"stage{i}_x_source_aligned"] = x_source.numpy()

            # Add
            x = x + x_source
            traces[f"stage{i}_after_add"] = x.numpy()
            print(
                f"Stage {i} after add: {x.shape}, range: [{x.min():.4f}, {x.max():.4f}]"
            )

            # Resblocks
            xs = None
            for j in range(len(gen.resblocks) // len(gen.ups)):
                idx = i * (len(gen.resblocks) // len(gen.ups)) + j
                if idx < len(gen.resblocks):
                    if xs is None:
                        xs = gen.resblocks[idx](x, style_pt)
                    else:
                        xs = xs + gen.resblocks[idx](x, style_pt)
            if xs is not None:
                x = xs / (len(gen.resblocks) // len(gen.ups))
            traces[f"stage{i}_after_resblocks"] = x.numpy()
            print(
                f"Stage {i} after resblocks: {x.shape}, range: [{x.min():.4f}, {x.max():.4f}]"
            )

        # Final conv_post
        x = F.leaky_relu(x, 0.1)
        traces["before_conv_post"] = x.numpy()

        x = gen.conv_post(x)
        traces["conv_post_out"] = x.numpy()
        print(f"\nconv_post: {x.shape}, range: [{x.min():.4f}, {x.max():.4f}]")

        # Split to mag and phase
        n_fft = 20
        n_bins = n_fft // 2 + 1  # 11
        log_mag = x[:, :n_bins, :].clamp(-10, 10)
        mag_out = torch.exp(log_mag)
        phase_out = torch.sin(x[:, n_bins:, :])

        traces["log_mag"] = log_mag.numpy()
        traces["mag_out"] = mag_out.numpy()
        traces["phase_out"] = phase_out.numpy()
        print(
            f"Mag: {mag_out.shape}, range: [{mag_out.min():.4f}, {mag_out.max():.4f}]"
        )
        print(
            f"Phase: {phase_out.shape}, range: [{phase_out.min():.4f}, {phase_out.max():.4f}]"
        )

    # Save traces
    output_path = Path("/tmp/kokoro_ref/generator_internal_traces.npz")
    np.savez_compressed(output_path, **traces)
    print(f"\nSaved {len(traces)} traces to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
