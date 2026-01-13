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

"""Debug _istft_synthesis vs manual implementation."""

from pathlib import Path

import mlx.core as mx
import numpy as np


def main():
    ref_dir = Path("/tmp/kokoro_ref_decoder")
    gen_ref = np.load(ref_dir / "generator_intermediates.npz")

    from tools.pytorch_to_mlx.converters import KokoroConverter
    converter = KokoroConverter()
    mlx_model, _, _ = converter.load_from_hf("hexgrad/Kokoro-82M")
    gen = mlx_model.decoder.generator

    n_fft = gen.istft_n_fft
    hop = gen.istft_hop_size
    print(f"ISTFT params: n_fft={n_fft}, hop={hop}")

    # Use PyTorch's spec and phase
    spec = mx.array(gen_ref["gen_spec"].astype(np.float32))  # [1, frames, 11]
    phase = mx.array(gen_ref["gen_phase"].astype(np.float32))  # sin(raw_phase)

    print(f"Input spec shape: {spec.shape}")
    print(f"Input phase shape: {phase.shape}")

    # Call _istft_synthesis
    print("\n=== Calling _istft_synthesis ===")
    audio_istft = gen._istft_synthesis(spec, phase)
    mx.eval(audio_istft)
    print(f"Output shape: {audio_istft.shape}")
    print(f"Output range: [{float(audio_istft.min()):.6f}, {float(audio_istft.max()):.6f}]")

    # Manual implementation
    print("\n=== Manual implementation ===")
    cos_phase = mx.cos(phase)
    sin_phase = mx.sin(phase)
    spectrum = spec * (cos_phase + 1j * sin_phase)

    time_frames = mx.fft.irfft(spectrum, n=n_fft, axis=-1)
    window = gen._istft_window
    time_frames_windowed = time_frames * window

    batch, frames, _ = time_frames_windowed.shape
    weight = gen._istft_identity_kernel.astype(time_frames_windowed.dtype)
    audio_raw = mx.conv_transpose1d(time_frames_windowed, weight, stride=hop)[:, :, 0]

    ones_input = mx.ones((1, frames, 1), dtype=time_frames_windowed.dtype)
    window_kernel = gen._istft_window_kernel.astype(time_frames_windowed.dtype)
    window_sum = mx.conv_transpose1d(ones_input, window_kernel, stride=hop)[0, :, 0]
    window_sum_safe = mx.maximum(window_sum, 1e-8)
    audio_normalized = audio_raw / window_sum_safe

    pad = n_fft // 2
    output_length = (frames - 1) * hop
    audio_manual = audio_normalized[:, pad : pad + output_length]
    mx.eval(audio_manual)

    print(f"Output shape: {audio_manual.shape}")
    print(f"Output range: [{float(audio_manual.min()):.6f}, {float(audio_manual.max()):.6f}]")

    # Compare
    audio_istft_np = np.array(audio_istft)
    audio_manual_np = np.array(audio_manual)

    diff = np.abs(audio_istft_np - audio_manual_np)
    print(f"\nDifference: max={diff.max():.6f}, mean={diff.mean():.6f}")

    # Check intermediate step in _istft_synthesis
    print("\n=== Checking _istft_synthesis formula ===")
    # _istft_synthesis does: spectrum = mag * mx.exp(1j * phase)
    spectrum_istft = spec * mx.exp(1j * phase)
    mx.eval(spectrum_istft)

    # Manual does: spectrum = spec * (cos(phase) + j*sin(phase))
    # which equals: spec * exp(j*phase)
    # They should be identical!

    # Check if they're the same
    diff_spectrum = mx.abs(spectrum_istft - spectrum)
    mx.eval(diff_spectrum)
    print(f"Spectrum diff: max={float(diff_spectrum.max()):.10f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
