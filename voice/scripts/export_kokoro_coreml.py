#!/usr/bin/env python3
"""
Export Kokoro TTS to CoreML for Apple Neural Engine acceleration.

This script converts the Kokoro TorchScript model to CoreML format,
which can run on:
- Apple Neural Engine (ANE) - fastest, potentially 2-3x speedup
- GPU (Metal) - similar to MPS
- CPU - fallback

Usage:
    python scripts/export_kokoro_coreml.py
    python scripts/export_kokoro_coreml.py --compute-units all
    python scripts/export_kokoro_coreml.py --compute-units neural-engine
    python scripts/export_kokoro_coreml.py --benchmark

Copyright 2025 Andrew Yates. All rights reserved.
"""

import argparse
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")


def convert_to_coreml(torchscript_path: str, output_path: str, compute_units: str = "all"):
    """Convert TorchScript model to CoreML format.

    Args:
        torchscript_path: Path to the TorchScript model
        output_path: Output path for the CoreML model
        compute_units: Target compute units - 'all', 'cpu-and-gpu', 'cpu-only', 'neural-engine'
    """
    import torch
    import coremltools as ct

    print(f"[INFO] coremltools version: {ct.__version__}")
    print(f"[INFO] Loading TorchScript model from {torchscript_path}...")

    # Load the TorchScript model
    model = torch.jit.load(torchscript_path, map_location="cpu")
    model.eval()

    # Define input shapes for tracing
    # From export_kokoro_torchscript.py:
    # ids: [1, seq_len] int64 tensor of phoneme token IDs
    # ref: [1, 256] float32 voice embedding
    # speed: [1] float32 speed factor

    # Use flexible sequence length
    seq_len = 50  # Example sequence length (typical range: 10-200)

    print("[INFO] Preparing input specifications...")

    # Create example inputs for conversion
    example_ids = torch.randint(0, 100, (1, seq_len), dtype=torch.long)
    example_ref = torch.randn(1, 256, dtype=torch.float32)
    example_speed = torch.tensor([1.0], dtype=torch.float32)

    # Map compute units string to ct enum
    compute_unit_map = {
        "all": ct.ComputeUnit.ALL,
        "cpu-and-gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu-only": ct.ComputeUnit.CPU_ONLY,
        "neural-engine": ct.ComputeUnit.CPU_AND_NE,
    }
    target_compute_unit = compute_unit_map.get(compute_units, ct.ComputeUnit.ALL)

    print(f"[INFO] Target compute units: {compute_units} -> {target_compute_unit}")
    print(f"[INFO] Converting to CoreML (this may take several minutes)...")

    start_time = time.time()

    try:
        # Try using unified conversion API
        traced_model = torch.jit.trace(model, (example_ids, example_ref, example_speed))

        # Define input shapes with flexible dimensions
        mlmodel = ct.convert(
            traced_model,
            inputs=[
                ct.TensorType(name="ids", shape=(1, ct.RangeDim(1, 500, 50)), dtype=int),
                ct.TensorType(name="ref", shape=(1, 256)),
                ct.TensorType(name="speed", shape=(1,)),
            ],
            compute_units=target_compute_unit,
            minimum_deployment_target=ct.target.macOS14,  # For ANE support
        )

        conversion_time = time.time() - start_time
        print(f"[INFO] Conversion completed in {conversion_time:.1f}s")

        # Save the model
        print(f"[INFO] Saving CoreML model to {output_path}...")
        mlmodel.save(output_path)

        # Get model size
        if output_path.endswith('.mlpackage'):
            # mlpackage is a directory
            import subprocess
            size_output = subprocess.check_output(['du', '-sh', output_path]).decode()
            model_size = size_output.split()[0]
        else:
            model_size = f"{os.path.getsize(output_path) / (1024*1024):.1f} MB"
        print(f"[INFO] Model size: {model_size}")

        return mlmodel

    except Exception as e:
        print(f"[ERROR] Conversion failed: {e}")
        print("\n[DEBUG] Attempting to get more details...")

        # Try to understand what ops are problematic
        import traceback
        traceback.print_exc()

        return None


def benchmark_coreml(coreml_path: str, iterations: int = 10):
    """Benchmark CoreML model inference speed."""
    import coremltools as ct
    import numpy as np

    print(f"\n[BENCHMARK] Loading CoreML model from {coreml_path}...")
    mlmodel = ct.models.MLModel(coreml_path)

    # Prepare test input
    seq_len = 50
    test_ids = np.random.randint(0, 100, (1, seq_len)).astype(np.int32)
    test_ref = np.random.randn(1, 256).astype(np.float32)
    test_speed = np.array([1.0], dtype=np.float32)

    # Warmup
    print(f"[BENCHMARK] Warming up ({iterations} warmup iterations)...")
    for _ in range(iterations):
        _ = mlmodel.predict({
            "ids": test_ids,
            "ref": test_ref,
            "speed": test_speed
        })

    # Benchmark
    print(f"[BENCHMARK] Running {iterations} benchmark iterations...")
    times = []
    for i in range(iterations):
        start = time.time()
        output = mlmodel.predict({
            "ids": test_ids,
            "ref": test_ref,
            "speed": test_speed
        })
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)
        print(f"  Iteration {i+1}: {elapsed:.1f}ms")

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print(f"\n[BENCHMARK] Results:")
    print(f"  Average: {avg_time:.1f}ms")
    print(f"  Min: {min_time:.1f}ms")
    print(f"  Max: {max_time:.1f}ms")

    return avg_time


def compare_with_mps(torchscript_path: str, iterations: int = 10):
    """Benchmark MPS (TorchScript) for comparison."""
    import torch

    print(f"\n[BENCHMARK MPS] Loading TorchScript model from {torchscript_path}...")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[BENCHMARK MPS] Using device: {device}")

    model = torch.jit.load(torchscript_path, map_location=device)
    model.eval()

    # Prepare test input
    seq_len = 50
    test_ids = torch.randint(0, 100, (1, seq_len), dtype=torch.long, device=device)
    test_ref = torch.randn(1, 256, dtype=torch.float32, device=device)
    test_speed = torch.tensor([1.0], dtype=torch.float32, device=device)

    # Warmup
    print(f"[BENCHMARK MPS] Warming up ({iterations} warmup iterations)...")
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(test_ids, test_ref, test_speed)
            if device == "mps":
                torch.mps.synchronize()

    # Benchmark
    print(f"[BENCHMARK MPS] Running {iterations} benchmark iterations...")
    times = []
    with torch.no_grad():
        for i in range(iterations):
            start = time.time()
            output = model(test_ids, test_ref, test_speed)
            if device == "mps":
                torch.mps.synchronize()
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)
            print(f"  Iteration {i+1}: {elapsed:.1f}ms")

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print(f"\n[BENCHMARK MPS] Results:")
    print(f"  Average: {avg_time:.1f}ms")
    print(f"  Min: {min_time:.1f}ms")
    print(f"  Max: {max_time:.1f}ms")

    return avg_time


def main():
    parser = argparse.ArgumentParser(description='Export Kokoro to CoreML')
    parser.add_argument('--input', default='models/kokoro/kokoro_mps.pt',
                       help='Input TorchScript model (default: models/kokoro/kokoro_mps.pt)')
    parser.add_argument('--output', default='models/kokoro/kokoro.mlpackage',
                       help='Output CoreML model (default: models/kokoro/kokoro.mlpackage)')
    parser.add_argument('--compute-units', choices=['all', 'cpu-and-gpu', 'cpu-only', 'neural-engine'],
                       default='all', help='Target compute units (default: all)')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run benchmark after conversion')
    parser.add_argument('--benchmark-only', action='store_true',
                       help='Only run benchmark (skip conversion)')
    parser.add_argument('--compare-mps', action='store_true',
                       help='Also benchmark MPS for comparison')
    parser.add_argument('--iterations', type=int, default=10,
                       help='Number of benchmark iterations (default: 10)')
    args = parser.parse_args()

    if not args.benchmark_only:
        # Check input exists
        if not os.path.exists(args.input):
            print(f"[ERROR] Input model not found: {args.input}")
            sys.exit(1)

        # Convert to CoreML
        mlmodel = convert_to_coreml(args.input, args.output, args.compute_units)

        if mlmodel is None:
            print("[ERROR] Conversion failed")
            sys.exit(1)

    # Benchmark
    if args.benchmark or args.benchmark_only:
        if not os.path.exists(args.output):
            print(f"[ERROR] CoreML model not found: {args.output}")
            sys.exit(1)

        coreml_time = benchmark_coreml(args.output, args.iterations)

        if args.compare_mps and os.path.exists(args.input):
            mps_time = compare_with_mps(args.input, args.iterations)

            print(f"\n[COMPARISON]")
            print(f"  CoreML: {coreml_time:.1f}ms")
            print(f"  MPS:    {mps_time:.1f}ms")
            speedup = mps_time / coreml_time
            if speedup > 1:
                print(f"  CoreML is {speedup:.2f}x FASTER")
            else:
                print(f"  MPS is {1/speedup:.2f}x FASTER")


if __name__ == '__main__':
    main()
