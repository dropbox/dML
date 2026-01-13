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
Export PyTorch ISTFT intermediate tensors for comparison.
Must run with: /tmp/kokoro_env/bin/python scripts/export_istft_intermediates.py
"""

from pathlib import Path

import numpy as np
import torch


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().float().numpy()


def main():
    ref_dir = Path("/tmp/kokoro_ref_decoder")
    gen_ref = np.load(ref_dir / "generator_intermediates.npz")

    # Get PyTorch spec and phase
    pt_spec = torch.from_numpy(gen_ref["gen_spec"].astype(np.float32))  # [1, frames, 11]
    pt_phase = torch.from_numpy(gen_ref["gen_phase"].astype(np.float32))  # sin(raw_phase)
    pt_raw_phase = torch.from_numpy(gen_ref["gen_raw_phase"].astype(np.float32))

    from kokoro import KModel
    model = KModel().eval()
    gen = model.decoder.generator

    print("=" * 72)
    print("PyTorch ISTFT Export")
    print("=" * 72)

    # Check generator's ISTFT parameters
    print(f"post_n_fft: {gen.post_n_fft}")
    # Find the hop parameter - might be named differently
    hop_attr = None
    for attr in dir(gen):
        if 'hop' in attr.lower():
            print(f"  {attr}: {getattr(gen, attr, None)}")
            if 'istft' in attr.lower() or hop_attr is None:
                hop_attr = attr

    n_fft = gen.post_n_fft
    # Use hop_size attribute if available, otherwise default to 5
    hop = getattr(gen, 'istft_hop_size', getattr(gen, 'hop_size', 5))

    # CRITICAL: Check what PyTorch's forward method actually does for ISTFT
    # Look at the generator forward code to understand the ISTFT call

    # Transpose to NCL format for PyTorch
    spec_ncl = pt_spec.transpose(1, 2)  # [1, 11, frames]
    phase_ncl = pt_phase.transpose(1, 2)  # [1, 11, frames] - this is sin(raw_phase)
    raw_phase_ncl = pt_raw_phase.transpose(1, 2)

    print(f"\nspec shape: {spec_ncl.shape}")
    print(f"phase shape: {phase_ncl.shape}")

    intermediates = {}

    with torch.no_grad():
        # The actual ISTFTNet forward does (from StyleTTS2):
        # mag_out = torch.exp(spec_out)  # spec_out is raw logspec
        # pha_out = torch.sin(pha_out)   # pha_out is raw phase output

        # Then torch.istft is called with a complex tensor:
        # stft_out = mag_out * torch.cos(pha_out) + 1j * mag_out * torch.sin(pha_out)
        # OR: stft_out = mag_out * torch.exp(1j * pha_out) where pha_out is raw_phase

        # Test: what if phase passed to istft is sin(raw_phase)?
        # Then: stft = spec * cos(sin(raw_phase)) + j * spec * sin(sin(raw_phase))
        # This is weird...

        # Let's check what the actual formula is by looking at different approaches

        # Approach 1: Use raw_phase as angle
        print("\n--- Approach 1: raw_phase as angle ---")
        cos_raw = torch.cos(raw_phase_ncl)
        sin_raw = torch.sin(raw_phase_ncl)
        real_1 = spec_ncl * cos_raw
        imag_1 = spec_ncl * sin_raw
        stft_1 = torch.complex(real_1, imag_1)  # [1, 11, frames]
        audio_1 = torch.istft(stft_1, n_fft=n_fft, hop_length=hop,
                              win_length=n_fft, window=torch.hann_window(n_fft),
                              center=True, return_complex=False)
        print(f"Audio 1 shape: {audio_1.shape}, range: [{audio_1.min():.4f}, {audio_1.max():.4f}]")
        intermediates["istft_audio_raw_phase"] = _to_numpy(audio_1)

        # Approach 2: Use sin(raw_phase) directly as phase angle
        print("\n--- Approach 2: sin(raw_phase) as angle ---")
        cos_sin = torch.cos(phase_ncl)  # cos(sin(raw_phase))
        sin_sin = torch.sin(phase_ncl)  # sin(sin(raw_phase))
        real_2 = spec_ncl * cos_sin
        imag_2 = spec_ncl * sin_sin
        stft_2 = torch.complex(real_2, imag_2)
        audio_2 = torch.istft(stft_2, n_fft=n_fft, hop_length=hop,
                              win_length=n_fft, window=torch.hann_window(n_fft),
                              center=True, return_complex=False)
        print(f"Audio 2 shape: {audio_2.shape}, range: [{audio_2.min():.4f}, {audio_2.max():.4f}]")
        intermediates["istft_audio_sin_phase"] = _to_numpy(audio_2)

        # Approach 3: mag*phase for imag (common in some vocoder implementations)
        print("\n--- Approach 3: real=mag, imag=mag*sin(raw_phase) ---")
        real_3 = spec_ncl
        imag_3 = spec_ncl * phase_ncl  # mag * sin(raw_phase)
        stft_3 = torch.complex(real_3, imag_3)
        audio_3 = torch.istft(stft_3, n_fft=n_fft, hop_length=hop,
                              win_length=n_fft, window=torch.hann_window(n_fft),
                              center=True, return_complex=False)
        print(f"Audio 3 shape: {audio_3.shape}, range: [{audio_3.min():.4f}, {audio_3.max():.4f}]")
        intermediates["istft_audio_simple"] = _to_numpy(audio_3)

        # Compare with actual generator output
        pt_audio = gen_ref["gen_audio"]
        print(f"\nTarget audio range: [{pt_audio.min():.4f}, {pt_audio.max():.4f}]")

        for name, audio in [("raw_phase", audio_1), ("sin_phase", audio_2), ("simple", audio_3)]:
            audio_np = _to_numpy(audio)
            # Handle shape difference
            min_len = min(audio_np.size, pt_audio.size)
            audio_flat = audio_np.flatten()[:min_len]
            target_flat = pt_audio.flatten()[:min_len]
            diff = np.abs(audio_flat - target_flat)
            print(f"  {name}: max_diff={diff.max():.6f}, mean_diff={diff.mean():.6f}")

    # Save
    out_path = ref_dir / "istft_intermediates.npz"
    np.savez(out_path, **intermediates)
    print(f"\nSaved to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
