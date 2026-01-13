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

"""Check the ISTFT formula used by PyTorch ISTFTNet."""

from pathlib import Path

import mlx.core as mx
import numpy as np


def main():
    ref_dir = Path("/tmp/kokoro_ref_decoder")
    gen_ref = np.load(ref_dir / "generator_intermediates.npz")

    # Get PyTorch values
    pt_spec = gen_ref["gen_spec"]  # [1, frames, 11] - exp(raw_spec) = magnitude
    pt_phase = gen_ref["gen_phase"]  # [1, frames, 11] - sin(raw_phase)
    pt_audio = gen_ref["gen_audio"]
    pt_raw_spec = gen_ref["gen_raw_spec"]  # [1, frames, 11] - raw network output
    pt_raw_phase = gen_ref["gen_raw_phase"]  # [1, frames, 11] - raw network output

    print("PyTorch values:")
    print(f"  raw_spec range: [{pt_raw_spec.min():.4f}, {pt_raw_spec.max():.4f}]")
    print(f"  raw_phase range: [{pt_raw_phase.min():.4f}, {pt_raw_phase.max():.4f}]")
    print(f"  spec (exp) range: [{pt_spec.min():.4f}, {pt_spec.max():.4f}]")
    print(f"  phase (sin) range: [{pt_phase.min():.4f}, {pt_phase.max():.4f}]")
    print(f"  audio range: [{pt_audio.min():.4f}, {pt_audio.max():.4f}]")
    print(f"  audio shape: {pt_audio.shape}")

    # Load MLX generator to get ISTFT params
    from tools.pytorch_to_mlx.converters import KokoroConverter
    converter = KokoroConverter()
    mlx_model, _, _ = converter.load_from_hf("hexgrad/Kokoro-82M")
    gen = mlx_model.decoder.generator

    n_fft = gen.istft_n_fft
    hop = gen.istft_hop_size
    print(f"\nISTFT params: n_fft={n_fft}, hop={hop}")

    # Test different ISTFT formulas
    spec_mx = mx.array(pt_spec.astype(np.float32))
    phase_mx = mx.array(pt_phase.astype(np.float32))
    raw_phase_mx = mx.array(pt_raw_phase.astype(np.float32))

    # Formula 1: Current MLX implementation
    # spectrum = mag * exp(1j * phase) where phase is sin(raw_phase)
    print("\n=== Formula 1: mag * exp(1j * sin(raw_phase)) ===")
    spectrum1 = spec_mx * mx.exp(1j * phase_mx)
    time_frames1 = mx.fft.irfft(spectrum1, n=n_fft, axis=-1)
    print(f"Time frames range: [{float(time_frames1.min()):.6f}, {float(time_frames1.max()):.6f}]")

    # Formula 2: Use raw_phase directly as phase in radians
    # spectrum = mag * exp(1j * raw_phase)
    print("\n=== Formula 2: mag * exp(1j * raw_phase) ===")
    spectrum2 = spec_mx * mx.exp(1j * raw_phase_mx)
    time_frames2 = mx.fft.irfft(spectrum2, n=n_fft, axis=-1)
    print(f"Time frames range: [{float(time_frames2.min()):.6f}, {float(time_frames2.max()):.6f}]")

    # Formula 3: real = mag * cos(phase_sin), imag = mag * phase_sin
    # This is a common ISTFTNet variant where sin(phase) directly multiplies magnitude
    print("\n=== Formula 3: real=mag, imag=mag*sin(raw_phase) ===")
    real3 = spec_mx
    imag3 = spec_mx * phase_mx
    spectrum3 = real3 + 1j * imag3
    time_frames3 = mx.fft.irfft(spectrum3, n=n_fft, axis=-1)
    print(f"Time frames range: [{float(time_frames3.min()):.6f}, {float(time_frames3.max()):.6f}]")

    # Formula 4: ISTFTNet formula using cos(raw_phase) and sin(raw_phase)
    print("\n=== Formula 4: real=mag*cos(raw_phase), imag=mag*sin(raw_phase) ===")
    cos_phase = mx.cos(raw_phase_mx)
    sin_phase = mx.sin(raw_phase_mx)  # This equals phase_mx
    spectrum4 = spec_mx * (cos_phase + 1j * sin_phase)
    time_frames4 = mx.fft.irfft(spectrum4, n=n_fft, axis=-1)
    print(f"Time frames range: [{float(time_frames4.min()):.6f}, {float(time_frames4.max()):.6f}]")

    # Test full ISTFT with formula 4
    print("\n=== Testing full ISTFT with formula 4 ===")
    window = gen._istft_window
    time_frames4_windowed = time_frames4 * window

    batch, frames, _ = time_frames4_windowed.shape
    weight = gen._istft_identity_kernel.astype(time_frames4_windowed.dtype)
    audio4 = mx.conv_transpose1d(time_frames4_windowed, weight, stride=hop)[:, :, 0]

    ones_input = mx.ones((1, frames, 1), dtype=time_frames4_windowed.dtype)
    window_kernel = gen._istft_window_kernel.astype(time_frames4_windowed.dtype)
    window_sum = mx.conv_transpose1d(ones_input, window_kernel, stride=hop)[0, :, 0]
    window_sum = mx.maximum(window_sum, 1e-8)
    audio4 = audio4 / window_sum

    pad = n_fft // 2
    output_length = (frames - 1) * hop
    audio4 = audio4[:, pad : pad + output_length]
    mx.eval(audio4)

    audio4_np = np.array(audio4)
    diff = np.abs(audio4_np - pt_audio)
    print(f"Audio diff: max={diff.max():.6f}, mean={diff.mean():.6f}")

    # Also check if we need to use raw_spec instead of exp(raw_spec)
    print("\n=== Formula 5: exp(raw_spec)*exp(1j*raw_phase) ===")
    raw_spec_mx = mx.array(pt_raw_spec.astype(np.float32))
    # This is equivalent to formula 4 since spec = exp(raw_spec)
    spectrum5 = mx.exp(raw_spec_mx) * mx.exp(1j * raw_phase_mx)
    time_frames5 = mx.fft.irfft(spectrum5, n=n_fft, axis=-1)
    time_frames5_windowed = time_frames5 * window

    audio5 = mx.conv_transpose1d(time_frames5_windowed, weight, stride=hop)[:, :, 0]
    audio5 = audio5 / window_sum
    audio5 = audio5[:, pad : pad + output_length]
    mx.eval(audio5)

    audio5_np = np.array(audio5)
    diff5 = np.abs(audio5_np - pt_audio)
    print(f"Audio diff: max={diff5.max():.6f}, mean={diff5.mean():.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
