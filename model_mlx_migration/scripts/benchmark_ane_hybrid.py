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
Benchmark CoreML/ANE encoder vs MLX encoder.

Compares:
1. CoreML encoder with different compute units (CPU_AND_NE, CPU_AND_GPU, CPU_ONLY)
2. MLX encoder (GPU)

Usage:
    python scripts/benchmark_ane_hybrid.py
    python scripts/benchmark_ane_hybrid.py --iterations 20
    python scripts/benchmark_ane_hybrid.py --audio test.wav
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_mlx_encoder():
    """Load MLX encoder for comparison."""
    from tools.whisper_mlx import WhisperMLX

    print("Loading MLX WhisperMLX model (large-v3)...")
    model = WhisperMLX.from_pretrained("large-v3")
    return model.encoder


def load_coreml_encoder(compute_units: str = "CPU_AND_NE"):
    """Load CoreML encoder."""
    from tools.whisper_ane import CoreMLEncoder

    print(f"Loading CoreML encoder (compute_units={compute_units})...")
    encoder = CoreMLEncoder.from_pretrained(
        "large-v3",
        compute_units=compute_units,
        auto_download=False,
    )
    return encoder


def generate_test_mel(n_frames: int = 3000, n_mels: int = 128) -> np.ndarray:
    """Generate random mel spectrogram for testing."""
    return np.random.randn(n_frames, n_mels).astype(np.float32)


def benchmark_mlx_encoder(encoder, mel: np.ndarray, n_iterations: int = 10, warmup: int = 3) -> dict:
    """Benchmark MLX encoder."""
    import mlx.core as mx

    mel_mx = mx.array(mel)

    # Warmup
    for _ in range(warmup):
        output = encoder(mel_mx, variable_length=True)
        mx.eval(output)

    # Benchmark
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        output = encoder(mel_mx, variable_length=True)
        mx.eval(output)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "backend": "MLX (GPU)",
        "mean_ms": np.mean(times) * 1000,
        "std_ms": np.std(times) * 1000,
        "min_ms": np.min(times) * 1000,
        "max_ms": np.max(times) * 1000,
        "n_iterations": n_iterations,
        "output_shape": tuple(output.shape),
    }


def benchmark_coreml_encoder(
    encoder,
    mel: np.ndarray,
    n_iterations: int = 10,
    warmup: int = 3,
    label: str = "CoreML"
) -> dict:
    """Benchmark CoreML encoder."""
    # Add batch dimension for CoreML
    mel_batch = mel[np.newaxis, ...]  # (1, n_frames, n_mels)

    # Warmup
    for _ in range(warmup):
        output = encoder(mel_batch)

    # Benchmark
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        output = encoder(mel_batch)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "backend": label,
        "mean_ms": np.mean(times) * 1000,
        "std_ms": np.std(times) * 1000,
        "min_ms": np.min(times) * 1000,
        "max_ms": np.max(times) * 1000,
        "n_iterations": n_iterations,
        "output_shape": tuple(output.shape),
    }


def verify_numerical_equivalence(mlx_encoder, coreml_encoder, mel: np.ndarray) -> dict:
    """Compare outputs from MLX and CoreML encoders."""
    import mlx.core as mx

    # Run MLX encoder
    mel_mx = mx.array(mel)
    mlx_output = mlx_encoder(mel_mx, variable_length=True)
    mx.eval(mlx_output)
    mlx_output_np = np.array(mlx_output)

    # Run CoreML encoder
    mel_batch = mel[np.newaxis, ...]
    coreml_output = coreml_encoder(mel_batch)

    # Compare - need to handle shape differences
    # MLX: (1, seq_len, n_state)
    # CoreML: (1, seq_len, n_state) after our conversion
    if mlx_output_np.ndim == 2:
        mlx_output_np = mlx_output_np[np.newaxis, ...]

    # Get actual seq_len from MLX (which handles variable length)
    mlx_seq_len = mlx_output_np.shape[1]
    coreml_seq_len = coreml_output.shape[1]

    # For 3000 frames, MLX gives (3000+1)//2 = 1500
    # CoreML is fixed at 1500

    # Compare only the overlapping region
    min_seq_len = min(mlx_seq_len, coreml_seq_len)
    mlx_slice = mlx_output_np[:, :min_seq_len, :]
    coreml_slice = coreml_output[:, :min_seq_len, :]

    # Compute differences
    abs_diff = np.abs(mlx_slice - coreml_slice)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)
    rel_diff = np.mean(abs_diff / (np.abs(mlx_slice) + 1e-8))

    return {
        "mlx_shape": tuple(mlx_output_np.shape),
        "coreml_shape": tuple(coreml_output.shape),
        "compared_region": (1, min_seq_len, mlx_output_np.shape[2]),
        "max_abs_diff": float(max_diff),
        "mean_abs_diff": float(mean_diff),
        "mean_rel_diff": float(rel_diff),
        "mlx_mean": float(np.mean(mlx_slice)),
        "coreml_mean": float(np.mean(coreml_slice)),
        "mlx_std": float(np.std(mlx_slice)),
        "coreml_std": float(np.std(coreml_slice)),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark ANE hybrid encoder")
    parser.add_argument("--iterations", type=int, default=10, help="Number of benchmark iterations")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup iterations")
    parser.add_argument("--audio", type=str, default=None, help="Path to audio file (optional)")
    parser.add_argument("--duration", type=float, default=30.0, help="Audio duration in seconds (default: 30)")
    parser.add_argument("--skip-mlx", action="store_true", help="Skip MLX benchmark")
    parser.add_argument("--skip-coreml", action="store_true", help="Skip CoreML benchmark")
    parser.add_argument("--compare", action="store_true", help="Compare numerical outputs")

    args = parser.parse_args()

    print("=" * 60)
    print("ANE Hybrid Encoder Benchmark")
    print("=" * 60)

    # Generate or load mel spectrogram
    n_frames = int(args.duration * 100)  # 100 frames per second
    mel = generate_test_mel(n_frames=n_frames, n_mels=128)
    print(f"\nTest mel spectrogram: {mel.shape} ({args.duration}s audio)")

    results = []

    # Benchmark MLX encoder
    if not args.skip_mlx:
        try:
            mlx_encoder = load_mlx_encoder()
            print("\nBenchmarking MLX encoder...")
            mlx_result = benchmark_mlx_encoder(
                mlx_encoder, mel,
                n_iterations=args.iterations,
                warmup=args.warmup
            )
            results.append(mlx_result)
            print(f"  Mean: {mlx_result['mean_ms']:.1f}ms, Std: {mlx_result['std_ms']:.1f}ms")
        except Exception as e:
            print(f"MLX benchmark failed: {e}")
            mlx_encoder = None
    else:
        mlx_encoder = None

    # Benchmark CoreML encoder with different compute units
    if not args.skip_coreml:
        compute_units_list = ["CPU_AND_NE", "CPU_AND_GPU", "CPU_ONLY"]
        coreml_encoder = None

        for compute_units in compute_units_list:
            try:
                encoder = load_coreml_encoder(compute_units=compute_units)
                if coreml_encoder is None:
                    coreml_encoder = encoder  # Keep first for comparison

                print(f"\nBenchmarking CoreML encoder ({compute_units})...")
                result = benchmark_coreml_encoder(
                    encoder, mel,
                    n_iterations=args.iterations,
                    warmup=args.warmup,
                    label=f"CoreML ({compute_units})"
                )
                results.append(result)
                print(f"  Mean: {result['mean_ms']:.1f}ms, Std: {result['std_ms']:.1f}ms")
            except Exception as e:
                print(f"CoreML ({compute_units}) benchmark failed: {e}")
    else:
        coreml_encoder = None

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"\nInput: {mel.shape} ({args.duration}s audio, {n_frames} frames)")
    print(f"Iterations: {args.iterations} (warmup: {args.warmup})")
    print()

    for result in results:
        print(f"{result['backend']:25} | Mean: {result['mean_ms']:6.1f}ms | Std: {result['std_ms']:5.1f}ms | Output: {result['output_shape']}")

    # Calculate speedups relative to MLX
    if results and not args.skip_mlx:
        mlx_time = results[0]["mean_ms"]
        print("\nSpeedup vs MLX:")
        for result in results[1:]:
            speedup = mlx_time / result["mean_ms"]
            print(f"  {result['backend']:25} {speedup:.2f}x")

    # Numerical comparison
    if args.compare and mlx_encoder is not None and coreml_encoder is not None:
        print("\n" + "=" * 60)
        print("NUMERICAL COMPARISON")
        print("=" * 60)

        comparison = verify_numerical_equivalence(mlx_encoder, coreml_encoder, mel)
        print(f"\nMLX output shape:    {comparison['mlx_shape']}")
        print(f"CoreML output shape: {comparison['coreml_shape']}")
        print(f"Compared region:     {comparison['compared_region']}")
        print()
        print(f"Max absolute diff:   {comparison['max_abs_diff']:.6f}")
        print(f"Mean absolute diff:  {comparison['mean_abs_diff']:.6f}")
        print(f"Mean relative diff:  {comparison['mean_rel_diff']:.6f}")
        print()
        print(f"MLX mean/std:        {comparison['mlx_mean']:.4f} / {comparison['mlx_std']:.4f}")
        print(f"CoreML mean/std:     {comparison['coreml_mean']:.4f} / {comparison['coreml_std']:.4f}")

        # Assess equivalence
        if comparison['max_abs_diff'] < 1e-3:
            print("\nNumerical equivalence: PASS (<1e-3 max diff)")
        elif comparison['max_abs_diff'] < 1e-2:
            print("\nNumerical equivalence: ACCEPTABLE (<1e-2 max diff)")
        else:
            print(f"\nNumerical equivalence: FAIL (>{comparison['max_abs_diff']:.2e} max diff)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
