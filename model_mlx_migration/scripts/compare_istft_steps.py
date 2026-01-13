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

"""Compare ISTFT step by step between MLX and PyTorch."""

from pathlib import Path

import mlx.core as mx
import numpy as np


def main():
    ref_dir = Path("/tmp/kokoro_ref_decoder")
    gen_ref = np.load(ref_dir / "generator_intermediates.npz")
    istft_ref = np.load(ref_dir / "istft_intermediates.npz")

    # Load MLX generator
    from tools.pytorch_to_mlx.converters import KokoroConverter
    converter = KokoroConverter()
    mlx_model, _, _ = converter.load_from_hf("hexgrad/Kokoro-82M")
    gen = mlx_model.decoder.generator

    n_fft = gen.istft_n_fft
    hop = gen.istft_hop_size
    print(f"ISTFT params: n_fft={n_fft}, hop={hop}")

    # Get spec and phase (both are sin(raw_phase) already applied)
    pt_spec = gen_ref["gen_spec"]  # [1, frames, 11]
    pt_phase = gen_ref["gen_phase"]  # sin(raw_phase)

    spec_mx = mx.array(pt_spec.astype(np.float32))
    phase_mx = mx.array(pt_phase.astype(np.float32))

    print(f"\nSpec shape: {spec_mx.shape}")
    print(f"Phase shape: {phase_mx.shape}")

    # Step 1: Construct complex spectrum using sin(raw_phase) as angle
    # PyTorch approach 2: cos(sin(raw_phase)) + j*sin(sin(raw_phase))
    print("\n=== Step 1: Complex spectrum ===")
    cos_phase = mx.cos(phase_mx)  # cos(sin(raw_phase))
    sin_phase = mx.sin(phase_mx)  # sin(sin(raw_phase))
    spectrum = spec_mx * (cos_phase + 1j * sin_phase)
    mx.eval(spectrum)
    print(f"Spectrum shape: {spectrum.shape}")
    print(f"Spectrum real range: [{float(spectrum.real.min()):.6f}, {float(spectrum.real.max()):.6f}]")

    # Step 2: IRFFT
    print("\n=== Step 2: IRFFT ===")
    time_frames = mx.fft.irfft(spectrum, n=n_fft, axis=-1)
    mx.eval(time_frames)
    print(f"Time frames shape: {time_frames.shape}")
    print(f"Time frames range: [{float(time_frames.min()):.6f}, {float(time_frames.max()):.6f}]")

    # Step 3: Window
    print("\n=== Step 3: Window ===")
    window = gen._istft_window
    time_frames_windowed = time_frames * window
    mx.eval(time_frames_windowed)
    print(f"Windowed range: [{float(time_frames_windowed.min()):.6f}, {float(time_frames_windowed.max()):.6f}]")

    # Step 4: Overlap-add using conv_transpose1d
    print("\n=== Step 4: Overlap-add ===")
    batch, frames, _ = time_frames_windowed.shape
    weight = gen._istft_identity_kernel.astype(time_frames_windowed.dtype)
    audio_raw = mx.conv_transpose1d(time_frames_windowed, weight, stride=hop)[:, :, 0]
    mx.eval(audio_raw)
    print(f"Audio raw shape: {audio_raw.shape}")
    print(f"Audio raw range: [{float(audio_raw.min()):.6f}, {float(audio_raw.max()):.6f}]")

    # Step 5: Window sum normalization
    print("\n=== Step 5: Window normalization ===")
    ones_input = mx.ones((1, frames, 1), dtype=time_frames_windowed.dtype)
    window_kernel = gen._istft_window_kernel.astype(time_frames_windowed.dtype)
    window_sum = mx.conv_transpose1d(ones_input, window_kernel, stride=hop)[0, :, 0]
    mx.eval(window_sum)
    print(f"Window sum shape: {window_sum.shape}")
    print(f"Window sum range: [{float(window_sum.min()):.6f}, {float(window_sum.max()):.6f}]")

    window_sum_safe = mx.maximum(window_sum, 1e-8)
    audio_normalized = audio_raw / window_sum_safe
    mx.eval(audio_normalized)
    print(f"Audio normalized range: [{float(audio_normalized.min()):.6f}, {float(audio_normalized.max()):.6f}]")

    # Step 6: Remove padding
    print("\n=== Step 6: Remove padding ===")
    pad = n_fft // 2
    output_length = (frames - 1) * hop
    audio_final = audio_normalized[:, pad : pad + output_length]
    mx.eval(audio_final)
    print(f"Audio final shape: {audio_final.shape}")
    print(f"Audio final range: [{float(audio_final.min()):.6f}, {float(audio_final.max()):.6f}]")

    # Compare with PyTorch
    print("\n=== Comparison ===")
    pt_audio_sin = istft_ref["istft_audio_sin_phase"]  # PyTorch with same formula
    pt_audio_target = gen_ref["gen_audio"]  # Target from full generator

    mlx_audio = np.array(audio_final)

    # Compare with approach 2 audio
    min_len = min(mlx_audio.size, pt_audio_sin.size)
    mlx_flat = mlx_audio.flatten()[:min_len]
    pt_sin_flat = pt_audio_sin.flatten()[:min_len]
    pt_target_flat = pt_audio_target.flatten()[:min_len]

    diff_sin = np.abs(mlx_flat - pt_sin_flat)
    diff_target = np.abs(mlx_flat - pt_target_flat)

    print(f"vs PyTorch approach 2 (sin_phase): max={diff_sin.max():.6f}, mean={diff_sin.mean():.6f}")
    print(f"vs PyTorch target audio: max={diff_target.max():.6f}, mean={diff_target.mean():.6f}")

    # Check if PyTorch approach 2 matches target
    diff_pt = np.abs(pt_sin_flat - pt_target_flat)
    print(f"PyTorch approach 2 vs target: max={diff_pt.max():.6f}, mean={diff_pt.mean():.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
