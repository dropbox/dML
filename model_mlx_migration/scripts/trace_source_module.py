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
Trace through SourceModule step-by-step to find where har_source error originates.

The SourceModule generates harmonics from F0:
1. Upsample F0
2. Generate rad_values = (f0 * h / sample_rate) % 1
3. Interpolation downsample
4. cumsum to accumulate phase
5. Scale phase by upp
6. Interpolation upsample
7. sin() and amplitude
8. Linear combination via l_linear
"""

import math
from pathlib import Path

import mlx.core as mx
import numpy as np


def main():
    ref_dir = Path("/tmp/kokoro_ref_seed0")
    ref = np.load(ref_dir / "tensors.npz")

    # Load MLX model
    from tools.pytorch_to_mlx.converters import KokoroConverter
    converter = KokoroConverter()
    mlx_model, _, _ = converter.load_from_hf("hexgrad/Kokoro-82M")
    mlx_model.set_deterministic(True)

    generator = mlx_model.decoder.generator
    m_source = generator.m_source

    # Get F0 input
    f0 = mx.array(ref["F0_pred"].astype(np.float32))

    # Calculate upp
    total_upp = 1
    for r in generator.config.istft_upsample_rates:
        total_upp *= r
    total_upp *= generator.istft_hop_size

    print("=" * 72)
    print("SourceModule Step-by-Step Trace")
    print("=" * 72)
    print(f"F0 shape: {f0.shape}, upp={total_upp}")
    print(f"num_harmonics: {m_source.num_harmonics}")
    print(f"sample_rate: {m_source.sample_rate}")

    length = f0.shape[1]
    samples = length * total_upp
    print(f"length={length}, samples={samples}")

    # Step 1: Upsample F0
    f0_up = mx.repeat(f0[:, :, None], total_upp, axis=1).squeeze(-1)
    mx.eval(f0_up)
    print(f"\nStep 1 - F0 upsample: {f0_up.shape}")

    # Step 2: UV mask
    uv = (f0_up > m_source.voiced_threshold).astype(mx.float32)
    mx.eval(uv)
    print(f"Step 2 - UV mask: {uv.shape}, voiced={int((np.array(uv) > 0.5).sum())}")

    # Step 3: rad_values for all harmonics
    h_factors = mx.arange(1, m_source.num_harmonics + 1, dtype=mx.float32)[None, None, :]
    f0_expanded = f0_up[:, :, None]
    rad_values = (f0_expanded * h_factors / m_source.sample_rate) % 1.0
    mx.eval(rad_values)
    print(f"Step 3 - rad_values: {rad_values.shape}")
    print(f"  First harmonic rad_values[:, :10]: {np.array(rad_values[0, :10, 0])}")

    # Step 4: Interpolation downsample (samples -> length)
    t_down = (mx.arange(length) + 0.5) * samples / length - 0.5
    t_down = mx.clip(t_down, 0, samples - 1)
    t_floor_down = mx.floor(t_down).astype(mx.int32)
    t_ceil_down = mx.minimum(t_floor_down + 1, samples - 1)
    t_frac_down = t_down - t_floor_down.astype(mx.float32)

    rad_floor = rad_values[:, t_floor_down, :]
    rad_ceil = rad_values[:, t_ceil_down, :]
    rad_values_down = (
        rad_floor * (1 - t_frac_down[None, :, None])
        + rad_ceil * t_frac_down[None, :, None]
    )
    mx.eval(rad_values_down)
    print(f"Step 4 - rad_values_down: {rad_values_down.shape}")

    # Step 5: Cumulative sum
    phase_low = mx.cumsum(rad_values_down, axis=1) * 2 * math.pi
    mx.eval(phase_low)
    print(f"Step 5 - phase_low (cumsum): {phase_low.shape}")
    print(f"  First harmonic phase_low[:, :10]: {np.array(phase_low[0, :10, 0])}")
    print(f"  First harmonic phase_low[:, -5:]: {np.array(phase_low[0, -5:, 0])}")

    # Check cumsum accumulation
    phase_low_np = np.array(phase_low)
    print(f"  Phase range (h=1): [{phase_low_np[0, :, 0].min():.2f}, {phase_low_np[0, :, 0].max():.2f}]")

    # Step 6: Scale phase by upp
    phase_scaled = phase_low * total_upp
    mx.eval(phase_scaled)
    print(f"Step 6 - phase_scaled: {phase_scaled.shape}")

    # Step 7: Interpolation upsample (length -> samples)
    t_up = (mx.arange(samples) + 0.5) * length / samples - 0.5
    t_up = mx.clip(t_up, 0, length - 1)
    t_floor_up = mx.floor(t_up).astype(mx.int32)
    t_ceil_up = mx.minimum(t_floor_up + 1, length - 1)
    t_frac_up = t_up - t_floor_up.astype(mx.float32)

    phase_floor = phase_scaled[:, t_floor_up, :]
    phase_ceil = phase_scaled[:, t_ceil_up, :]
    phase = (
        phase_floor * (1 - t_frac_up[None, :, None])
        + phase_ceil * t_frac_up[None, :, None]
    )
    mx.eval(phase)
    print(f"Step 7 - phase (upsampled): {phase.shape}")

    # Step 8: sin()
    sines = mx.sin(phase)  # [batch, samples, num_harmonics]
    mx.eval(sines)
    print(f"Step 8 - sines: {sines.shape}")
    print(f"  sines range: [{np.array(sines).min():.4f}, {np.array(sines).max():.4f}]")

    # Step 9: Apply amplitude and UV mask
    sines_amp = sines * m_source.sine_amp * uv[:, :, None]  # [batch, samples, 9]
    mx.eval(sines_amp)
    print(f"Step 9 - sines_amp (with UV): {sines_amp.shape}")

    # Step 10: Linear combination
    har_source = m_source.l_linear(sines_amp)  # [batch, samples, 1]
    mx.eval(har_source)
    print(f"Step 10 - har_source: {har_source.shape}")

    # Compare with PyTorch
    pt_har = ref["gen_har_source"].astype(np.float32)
    mlx_har = np.array(har_source)

    diff = np.abs(mlx_har - pt_har)
    print("\nFinal comparison:")
    print(f"  Max diff: {diff.max():.6f}")
    print(f"  Mean diff: {diff.mean():.6f}")

    # Find where largest diffs are
    flat_diff = diff.reshape(-1)
    top_idx = np.argsort(flat_diff)[-5:][::-1]
    print("\nTop 5 differences:")
    for idx in top_idx:
        t = idx
        print(f"  Sample {t} ({t/24000:.4f}s): mlx={mlx_har.flat[idx]:.6f}, pt={pt_har.flat[idx]:.6f}, diff={flat_diff[idx]:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
