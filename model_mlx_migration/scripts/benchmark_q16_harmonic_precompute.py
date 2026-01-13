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
Q16: Source Module Harmonic Precomputation Benchmark

Evaluates whether precomputing harmonic multipliers and vectorizing the
harmonic generation loop provides any speedup for CosyVoice3 vocoder.

Current implementation:
    for h in range(nb_harmonics):
        harmonic_phase = phase * (h + 1)
        harmonics.append(mx.sin(harmonic_phase))
        harmonics.append(mx.cos(harmonic_phase))

Proposed optimization:
    harmonic_ratios = mx.array([1, 2, ..., 8])  # precomputed
    harmonic_phases = phase * harmonic_ratios   # vectorized
    sin_harmonics = mx.sin(harmonic_phases)
    cos_harmonics = mx.cos(harmonic_phases)
"""

import time
import math
import mlx.core as mx


def current_source_module(f0_up: mx.array, nb_harmonics: int = 8) -> mx.array:
    """Current loop-based harmonic generation."""
    B = f0_up.shape[0]
    L_audio = f0_up.shape[2]
    sample_rate = 24000

    # Generate phase
    phase_inc = f0_up / sample_rate
    phase = mx.cumsum(phase_inc, axis=2) * 2 * math.pi  # [B, 1, L_audio]

    # Generate harmonics (sin and cos) - CURRENT LOOP
    harmonics = []
    for h in range(nb_harmonics):
        harmonic_phase = phase * (h + 1)
        harmonics.append(mx.sin(harmonic_phase))
        harmonics.append(mx.cos(harmonic_phase))

    # Add noise channels
    noise = mx.random.normal((B, 2, L_audio)) * 0.003

    # Stack all: [B, nb_harmonics * 2 + 2, L_audio]
    source = mx.concatenate(harmonics + [noise], axis=1)

    return source


def optimized_source_module(
    f0_up: mx.array,
    harmonic_ratios: mx.array,
    nb_harmonics: int = 8
) -> mx.array:
    """Optimized vectorized harmonic generation with precomputed ratios."""
    B = f0_up.shape[0]
    L_audio = f0_up.shape[2]
    sample_rate = 24000

    # Generate phase
    phase_inc = f0_up / sample_rate
    phase = mx.cumsum(phase_inc, axis=2) * 2 * math.pi  # [B, 1, L_audio]

    # Vectorized harmonic generation
    # phase: [B, 1, L_audio]
    # harmonic_ratios: [nb_harmonics]
    # harmonic_phases: [B, nb_harmonics, L_audio]
    harmonic_phases = phase * harmonic_ratios[None, :, None]  # broadcast

    # Generate all sin and cos at once
    sin_harmonics = mx.sin(harmonic_phases)  # [B, nb_harmonics, L_audio]
    cos_harmonics = mx.cos(harmonic_phases)  # [B, nb_harmonics, L_audio]

    # Interleave sin and cos: [B, 2*nb_harmonics, L_audio]
    # We need sin1, cos1, sin2, cos2, ... order
    harmonics = []
    for h in range(nb_harmonics):
        harmonics.append(sin_harmonics[:, h:h+1, :])
        harmonics.append(cos_harmonics[:, h:h+1, :])

    # Add noise channels
    noise = mx.random.normal((B, 2, L_audio)) * 0.003

    # Stack all: [B, nb_harmonics * 2 + 2, L_audio]
    source = mx.concatenate(harmonics + [noise], axis=1)

    return source


def optimized_source_module_v2(
    f0_up: mx.array,
    harmonic_ratios: mx.array,
    nb_harmonics: int = 8
) -> mx.array:
    """
    Fully vectorized harmonic generation - no interleaving loop.
    Uses reshape instead of loop for sin/cos interleaving.
    """
    B = f0_up.shape[0]
    L_audio = f0_up.shape[2]
    sample_rate = 24000

    # Generate phase
    phase_inc = f0_up / sample_rate
    phase = mx.cumsum(phase_inc, axis=2) * 2 * math.pi  # [B, 1, L_audio]

    # Vectorized harmonic generation
    harmonic_phases = phase * harmonic_ratios[None, :, None]  # [B, nb_harmonics, L_audio]

    # Generate all sin and cos at once
    sin_harmonics = mx.sin(harmonic_phases)  # [B, nb_harmonics, L_audio]
    cos_harmonics = mx.cos(harmonic_phases)  # [B, nb_harmonics, L_audio]

    # Stack and interleave: [B, 2, nb_harmonics, L_audio] -> [B, 2*nb_harmonics, L_audio]
    stacked = mx.stack([sin_harmonics, cos_harmonics], axis=2)  # [B, nb_harmonics, 2, L_audio]
    harmonics_interleaved = stacked.reshape(B, nb_harmonics * 2, L_audio)

    # Add noise channels
    noise = mx.random.normal((B, 2, L_audio)) * 0.003

    # Concatenate: [B, nb_harmonics * 2 + 2, L_audio]
    source = mx.concatenate([harmonics_interleaved, noise], axis=1)

    return source


def benchmark_source_module():
    """Benchmark current vs optimized source module."""
    print("=" * 60)
    print("Q16: Source Module Harmonic Precomputation Benchmark")
    print("=" * 60)

    # Test parameters (typical vocoder usage)
    B = 1
    nb_harmonics = 8
    mel_len = 100  # 100 mel frames
    upsample_factor = 120  # total upsample in CausalHiFT
    L_audio = mel_len * upsample_factor  # 12000 samples = 0.5s at 24kHz

    print("\nTest Configuration:")
    print(f"  Batch size: {B}")
    print(f"  Harmonics: {nb_harmonics}")
    print(f"  Mel frames: {mel_len}")
    print(f"  Audio length: {L_audio} samples ({L_audio/24000:.2f}s)")

    # Create test F0 (upsampled)
    mx.random.seed(42)
    f0_base = 200.0 + mx.random.normal((B, 1, mel_len)) * 50  # F0 around 200Hz
    f0_up = mx.repeat(f0_base, upsample_factor, axis=2)  # [B, 1, L_audio]

    # Precompute harmonic ratios for optimized version
    harmonic_ratios = mx.array([float(h + 1) for h in range(nb_harmonics)])

    # Warmup
    print("\nWarming up...")
    for _ in range(3):
        out_current = current_source_module(f0_up, nb_harmonics)
        mx.eval(out_current)
        out_opt = optimized_source_module(f0_up, harmonic_ratios, nb_harmonics)
        mx.eval(out_opt)
        out_opt_v2 = optimized_source_module_v2(f0_up, harmonic_ratios, nb_harmonics)
        mx.eval(out_opt_v2)

    # Benchmark current implementation
    n_runs = 20
    print(f"\nBenchmarking ({n_runs} runs each)...")

    times_current = []
    for _ in range(n_runs):
        mx.random.seed(42)  # Reset seed for fair comparison
        start = time.perf_counter()
        out = current_source_module(f0_up, nb_harmonics)
        mx.eval(out)
        times_current.append((time.perf_counter() - start) * 1000)

    times_opt = []
    for _ in range(n_runs):
        mx.random.seed(42)
        start = time.perf_counter()
        out = optimized_source_module(f0_up, harmonic_ratios, nb_harmonics)
        mx.eval(out)
        times_opt.append((time.perf_counter() - start) * 1000)

    times_opt_v2 = []
    for _ in range(n_runs):
        mx.random.seed(42)
        start = time.perf_counter()
        out = optimized_source_module_v2(f0_up, harmonic_ratios, nb_harmonics)
        mx.eval(out)
        times_opt_v2.append((time.perf_counter() - start) * 1000)

    # Calculate statistics
    current_mean = sum(times_current) / len(times_current)
    opt_mean = sum(times_opt) / len(times_opt)
    opt_v2_mean = sum(times_opt_v2) / len(times_opt_v2)

    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    print("\n| Configuration | Latency (ms) | Speedup |")
    print("|---------------|--------------|---------|")
    print(f"| Current (loop) | {current_mean:.3f} | 1.00x |")
    print(f"| Q16 (vectorized + loop interleave) | {opt_mean:.3f} | {current_mean/opt_mean:.2f}x |")
    print(f"| Q16 v2 (fully vectorized) | {opt_v2_mean:.3f} | {current_mean/opt_v2_mean:.2f}x |")

    # Verify correctness
    print("\nVerifying correctness...")
    mx.random.seed(42)
    out_current = current_source_module(f0_up, nb_harmonics)
    mx.eval(out_current)

    mx.random.seed(42)
    out_opt = optimized_source_module(f0_up, harmonic_ratios, nb_harmonics)
    mx.eval(out_opt)

    mx.random.seed(42)
    out_opt_v2 = optimized_source_module_v2(f0_up, harmonic_ratios, nb_harmonics)
    mx.eval(out_opt_v2)

    # Compare outputs (ignore noise channels which are random)
    harmonics_only = nb_harmonics * 2
    diff_opt = mx.abs(out_current[:, :harmonics_only] - out_opt[:, :harmonics_only])
    diff_v2 = mx.abs(out_current[:, :harmonics_only] - out_opt_v2[:, :harmonics_only])

    print(f"  Q16 vs current max diff (harmonics): {float(mx.max(diff_opt)):.2e}")
    print(f"  Q16 v2 vs current max diff (harmonics): {float(mx.max(diff_v2)):.2e}")

    # Context: vocoder timing
    print("\n" + "=" * 60)
    print("E2E Impact Analysis:")
    print("=" * 60)
    vocoder_time_ms = 238.0  # From H2+I1 benchmark (Worker #1319)
    source_pct_current = (current_mean / vocoder_time_ms) * 100
    source_savings = current_mean - opt_v2_mean

    print(f"\n  Vocoder total time (H2+I1): {vocoder_time_ms:.1f} ms")
    print(f"  Source module time (current): {current_mean:.2f} ms ({source_pct_current:.2f}%)")
    print(f"  Source module time (Q16 v2): {opt_v2_mean:.2f} ms")
    print(f"  Potential savings: {source_savings:.2f} ms ({source_savings/vocoder_time_ms*100:.2f}% of vocoder)")

    # Final verdict
    print("\n" + "=" * 60)
    print("Verdict:")
    print("=" * 60)

    if opt_v2_mean < current_mean * 0.95:  # >5% improvement
        if source_savings > 1.0:  # >1ms savings
            print(f"\n  WORTH IMPLEMENTING: {source_savings:.2f}ms savings")
        else:
            print(f"\n  NOT WORTH: Only {source_savings:.2f}ms savings (<1ms)")
    else:
        print(f"\n  NOT WORTH: No significant speedup ({current_mean/opt_v2_mean:.2f}x)")

    return {
        'current_ms': current_mean,
        'opt_ms': opt_mean,
        'opt_v2_ms': opt_v2_mean,
        'speedup': current_mean / opt_v2_mean,
        'savings_ms': source_savings,
        'pct_of_vocoder': source_savings / vocoder_time_ms * 100
    }


if __name__ == "__main__":
    results = benchmark_source_module()
