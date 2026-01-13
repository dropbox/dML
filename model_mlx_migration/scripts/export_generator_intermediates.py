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
Export PyTorch Generator intermediate tensors for detailed comparison.
Must run with: /tmp/kokoro_env/bin/python scripts/export_generator_intermediates.py
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().float().numpy()


def main():
    # Load existing reference
    ref_dir = Path("/tmp/kokoro_ref_decoder")
    ref = np.load(ref_dir / "tensors.npz")

    from kokoro import KModel
    model = KModel().eval()

    decoder = model.decoder
    gen = decoder.generator

    # Get inputs from reference
    x_nlc = torch.from_numpy(ref["generator_input_x"].astype(np.float32))  # [1, 126, 512]
    f0 = torch.from_numpy(ref["F0_pred"].astype(np.float32))  # [1, 63] or [1, 126]
    style = torch.from_numpy(ref["style_128"].astype(np.float32))  # [1, 128]

    print("=" * 72)
    print("PyTorch Generator Intermediate Export")
    print("=" * 72)
    print(f"Input x shape: {x_nlc.shape}")
    print(f"F0 shape: {f0.shape}")
    print(f"Style shape: {style.shape}")

    intermediates = {}
    intermediates["gen_input_x"] = _to_numpy(x_nlc)

    with torch.no_grad():
        # Convert to NCL for PyTorch
        x = x_nlc.transpose(1, 2)  # [B, C, T]
        print(f"x (NCL): {x.shape}, range=[{x.min():.4f}, {x.max():.4f}]")

        # Source module (from forward: with torch.no_grad())
        f0_up = gen.f0_upsamp(f0[:, None]).transpose(1, 2)  # [B, T_up, 1]
        har_source, noi_source, uv = gen.m_source(f0_up)
        har_squeezed = har_source.transpose(1, 2).squeeze(1)  # [B, samples]
        har_spec, har_phase = gen.stft.transform(har_squeezed)
        har = torch.cat([har_spec, har_phase], dim=1)  # [B, 22, frames]

        intermediates["gen_har_source"] = _to_numpy(har_source)
        intermediates["gen_har"] = _to_numpy(har.transpose(1, 2))
        print(f"har (STFT output): {har.shape}")

        # Generator main loop (matching forward exactly)
        for i in range(gen.num_upsamples):
            # Step 1: leaky_relu BEFORE ups
            x = F.leaky_relu(x, negative_slope=0.1)
            intermediates[f"gen_pre_lrelu_{i}"] = _to_numpy(x.transpose(1, 2))
            print(f"Stage {i} - after leaky_relu: {x.shape}, range=[{x.min():.4f}, {x.max():.4f}]")

            # Step 2: x_source = noise_convs(har) then noise_res(x_source, s)
            x_source = gen.noise_convs[i](har)
            intermediates[f"gen_noise_convs_{i}_out"] = _to_numpy(x_source.transpose(1, 2))
            print(f"  noise_convs_{i}: {x_source.shape}")

            x_source = gen.noise_res[i](x_source, style)
            intermediates[f"gen_noise_res_{i}_out"] = _to_numpy(x_source.transpose(1, 2))
            print(f"  noise_res_{i}: {x_source.shape}")

            # Step 3: x = ups[i](x)
            x = gen.ups[i](x)
            intermediates[f"gen_ups_{i}_out"] = _to_numpy(x.transpose(1, 2))
            print(f"  ups_{i}: {x.shape}")

            # Step 4: reflection pad at last stage
            if i == gen.num_upsamples - 1:
                x = gen.reflection_pad(x)
                print(f"  reflection_pad: {x.shape}")

            # Step 5: x = x + x_source
            x = x + x_source
            intermediates[f"gen_after_add_{i}"] = _to_numpy(x.transpose(1, 2))
            print(f"  after add: {x.shape}")

            # Step 6: resblocks (summed and averaged)
            xs = None
            for j in range(gen.num_kernels):
                rb_idx = i * gen.num_kernels + j
                if xs is None:
                    xs = gen.resblocks[rb_idx](x, style)
                else:
                    xs = xs + gen.resblocks[rb_idx](x, style)
            x = xs / gen.num_kernels
            intermediates[f"gen_resblocks_{i}_out"] = _to_numpy(x.transpose(1, 2))
            print(f"  resblocks_{i}: {x.shape}, range=[{x.min():.4f}, {x.max():.4f}]")

        # Final leaky_relu (default slope 0.01)
        x = F.leaky_relu(x)
        intermediates["gen_final_lrelu"] = _to_numpy(x.transpose(1, 2))
        print(f"After final leaky_relu: {x.shape}")

        # conv_post
        x = gen.conv_post(x)
        intermediates["gen_conv_post_out"] = _to_numpy(x.transpose(1, 2))
        print(f"After conv_post: {x.shape}")

        # Extract spec and phase (pre-exp/sin)
        n_fft = gen.post_n_fft
        raw_spec = x[:, :n_fft // 2 + 1, :]
        raw_phase = x[:, n_fft // 2 + 1:, :]
        intermediates["gen_raw_spec"] = _to_numpy(raw_spec.transpose(1, 2))
        intermediates["gen_raw_phase"] = _to_numpy(raw_phase.transpose(1, 2))

        # Apply exp/sin (as in forward)
        spec = torch.exp(raw_spec)
        phase = torch.sin(raw_phase)
        intermediates["gen_spec"] = _to_numpy(spec.transpose(1, 2))
        intermediates["gen_phase"] = _to_numpy(phase.transpose(1, 2))
        print(f"Spec: {spec.shape}, Phase: {phase.shape}")

        # Full generator forward for audio
        x_in = x_nlc.transpose(1, 2)
        audio = gen(x_in, style, f0)
        intermediates["gen_audio"] = _to_numpy(audio)
        print(f"Audio shape: {audio.shape}")

    # Save
    out_path = ref_dir / "generator_intermediates.npz"
    np.savez(out_path, **intermediates)
    print(f"\nSaved to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
