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
Test torch.istft with exact parameters to match MLX.
Must run with: /tmp/kokoro_env/bin/python scripts/test_torch_istft_match.py
"""

from pathlib import Path

import numpy as np
import torch


def main():
    ref_dir = Path("/tmp/kokoro_ref_decoder")
    gen_ref = np.load(ref_dir / "generator_intermediates.npz")

    # Get spec and phase
    spec = torch.from_numpy(gen_ref["gen_spec"].astype(np.float32))  # [1, frames, 11]
    phase = torch.from_numpy(gen_ref["gen_phase"].astype(np.float32))  # sin(raw_phase)

    print(f"Spec shape: {spec.shape}")
    print(f"Phase shape: {phase.shape}")

    # Transpose to NCL for PyTorch ISTFT
    spec_ncl = spec.transpose(1, 2)  # [1, 11, frames]
    phase_ncl = phase.transpose(1, 2)

    # Parameters
    n_fft = 20
    hop = 5
    win_length = 20
    window = torch.hann_window(win_length)

    # Build complex spectrum as per PyTorch's stft.inverse:
    # magnitude * torch.exp(phase * 1j)
    stft_complex = spec_ncl * torch.exp(phase_ncl * 1j)

    print(f"\nComplex STFT shape: {stft_complex.shape}")

    # Test different torch.istft variations
    print("\n=== Testing torch.istft variations ===")

    # Default: center=True, normalized=False, return_complex=False
    audio1 = torch.istft(stft_complex, n_fft, hop, win_length, window=window, center=True)
    print(f"center=True: shape={audio1.shape}, range=[{audio1.min():.4f}, {audio1.max():.4f}]")

    # center=False - skip due to overlap constraint issues
    # audio2 = torch.istft(stft_complex, n_fft, hop, win_length, window=window, center=False)
    audio2 = None

    # Load target audio
    pt_audio = gen_ref["gen_audio"]
    print(f"\nTarget audio shape: {pt_audio.shape}, range=[{pt_audio.min():.4f}, {pt_audio.max():.4f}]")

    # Compare each
    for name, audio in [("center=True", audio1)]:
        audio_np = audio.numpy()
        min_len = min(audio_np.size, pt_audio.size)
        audio_flat = audio_np.flatten()[:min_len]
        target_flat = pt_audio.flatten()[:min_len]
        diff = np.abs(audio_flat - target_flat)
        corr = np.corrcoef(audio_flat, target_flat)[0, 1]
        print(f"  {name}: max_diff={diff.max():.6f}, mean_diff={diff.mean():.6f}, corr={corr:.6f}")

    # Also save intermediate values for MLX comparison
    print("\n=== Saving intermediate values ===")

    # We need to understand what torch.istft does internally
    # Let's manually do IRFFT and overlap-add to match

    # Manual IRFFT
    stft_for_irfft = stft_complex.transpose(1, 2)  # [1, frames, 11]
    time_frames = torch.fft.irfft(stft_for_irfft, n=n_fft, dim=-1)
    print(f"IRFFT output shape: {time_frames.shape}")
    print(f"Time frames range: [{time_frames.min():.6f}, {time_frames.max():.6f}]")

    # Apply window
    windowed = time_frames * window
    print(f"Windowed range: [{windowed.min():.6f}, {windowed.max():.6f}]")

    # Save for comparison
    np.savez(
        ref_dir / "torch_istft_intermediates.npz",
        irfft_output=time_frames.numpy(),
        windowed=windowed.numpy(),
        audio_center_true=audio1.numpy(),
        audio_center_false=audio2.numpy(),
    )
    print(f"\nSaved to {ref_dir / 'torch_istft_intermediates.npz'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
