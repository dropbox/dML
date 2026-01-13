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
Diagnose if the error is in ISTFT by comparing:
1. PyTorch audio -> STFT -> expected mag/phase
2. MLX conv_post output -> ISTFT -> audio

If ISTFT is correct, the error must be before conv_post.
"""

from pathlib import Path

import mlx.core as mx
import numpy as np


def main():
    ref_dir = Path("/tmp/kokoro_ref_seed0")
    if not ref_dir.exists():
        print(f"Reference directory not found: {ref_dir}")
        return 1

    ref = np.load(ref_dir / "tensors.npz")
    audio_ref = ref["audio"].astype(np.float32)

    # Load MLX model
    from tools.pytorch_to_mlx.converters import KokoroConverter
    converter = KokoroConverter()
    mlx_model, _, _ = converter.load_from_hf("hexgrad/Kokoro-82M")
    mlx_model.set_deterministic(True)

    generator = mlx_model.decoder.generator
    n_fft = generator.istft_n_fft  # 20
    hop = generator.istft_hop_size  # 5

    print("=" * 72)
    print("ISTFT Roundtrip Test")
    print("=" * 72)
    print(f"n_fft: {n_fft}")
    print(f"hop_size: {hop}")
    print(f"audio_ref shape: {audio_ref.shape}")

    # Test ISTFT roundtrip: STFT(audio) -> ISTFT -> audio_reconstructed
    # Note: audio_mx was used for testing MLX STFT, keeping for reference
    _ = mx.array(audio_ref.reshape(1, -1))

    # Do STFT using np/scipy for reference
    from scipy.signal import stft

    # PyTorch STFT parameters
    # window: hann, n_fft=20, hop_length=5, center=True, pad_mode=reflect
    # We need to match these for proper comparison

    # Use scipy STFT as reference
    f, t, Zxx = stft(audio_ref, fs=24000, window='hann', nperseg=n_fft, noverlap=n_fft-hop,
                    nfft=n_fft, boundary='zeros', padded=True)
    print(f"scipy STFT output shape: {Zxx.shape}")  # (n_fft//2+1, frames)

    # Get magnitude and phase
    scipy_mag = np.abs(Zxx).T  # [frames, bins]
    scipy_phase = np.angle(Zxx).T  # [frames, bins]

    print(f"scipy mag shape: {scipy_mag.shape}")
    print(f"scipy mag range: {scipy_mag.min():.4f} to {scipy_mag.max():.4f}")
    print(f"scipy phase range: {scipy_phase.min():.4f} to {scipy_phase.max():.4f}")

    # Now run MLX ISTFT on scipy mag/phase
    mag_mx = mx.array(scipy_mag[None, :, :].astype(np.float32))
    phase_mx = mx.array(scipy_phase[None, :, :].astype(np.float32))

    # MLX ISTFT expects phase as sin(phase), not raw phase
    # Actually no - looking at the code, MLX ISTFT expects raw phase (radians)
    audio_recon = generator._istft_synthesis(mag_mx, mx.sin(phase_mx))
    mx.eval(audio_recon)
    audio_recon_np = np.array(audio_recon).reshape(-1)

    min_len = min(len(audio_ref), len(audio_recon_np))
    diff = np.abs(audio_ref[:min_len] - audio_recon_np[:min_len])
    print("\nMLX ISTFT roundtrip (with sin(phase)):")
    print(f"  Max abs error: {diff.max():.6f}")
    print(f"  Mean abs error: {diff.mean():.6f}")
    print(f"  Ref length: {len(audio_ref)}, Recon length: {len(audio_recon_np)}")

    # Try with raw phase
    audio_recon2 = generator._istft_synthesis(mag_mx, phase_mx)
    mx.eval(audio_recon2)
    audio_recon2_np = np.array(audio_recon2).reshape(-1)

    min_len2 = min(len(audio_ref), len(audio_recon2_np))
    diff2 = np.abs(audio_ref[:min_len2] - audio_recon2_np[:min_len2])
    print("\nMLX ISTFT roundtrip (raw phase):")
    print(f"  Max abs error: {diff2.max():.6f}")
    print(f"  Mean abs error: {diff2.mean():.6f}")

    # Now let's check what the conv_post output looks like vs expected
    # Expected: log(mag) for first 11 channels, arcsin(sin(phase)) for last 11
    print("\n" + "=" * 72)
    print("Conv_post expected output (if ISTFT is correct)")
    print("=" * 72)

    expected_log_mag = np.log(scipy_mag + 1e-10)  # Avoid log(0)
    expected_phase_logits = scipy_phase  # Generator applies sin()

    print(f"Expected log_mag range: {expected_log_mag.min():.4f} to {expected_log_mag.max():.4f}")
    print(f"Expected phase_logits range: {expected_phase_logits.min():.4f} to {expected_phase_logits.max():.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
