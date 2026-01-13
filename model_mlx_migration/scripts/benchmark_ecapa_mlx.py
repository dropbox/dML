#!/usr/bin/env python3
"""Benchmark ECAPA-TDNN MLX vs PyTorch performance.

This script measures inference latency and throughput for both implementations.

Usage:
    python scripts/benchmark_ecapa_mlx.py
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn.functional as F

# Add tools to path
sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "whisper_mlx"))


def create_test_inputs(batch_sizes: list, sequence_length: int = 300, n_mels: int = 60):
    """Create test inputs for benchmarking."""
    inputs = {}
    for bs in batch_sizes:
        inputs[bs] = np.random.randn(bs, n_mels, sequence_length).astype(np.float32)
    return inputs


def benchmark_pytorch_cpu(inputs: Dict[int, np.ndarray], weights_path: str, num_iterations: int = 100) -> Dict[str, Any]:
    """Benchmark PyTorch CPU inference (Block 0 only for fair comparison)."""
    weights = torch.load(weights_path, map_location="cpu", weights_only=False)

    results = {}
    for batch_size, input_np in inputs.items():
        x = torch.from_numpy(input_np)

        # Warm up
        for _ in range(10):
            with torch.no_grad():
                out = F.conv1d(x, weights["blocks.0.conv.conv.weight"],
                              weights["blocks.0.conv.conv.bias"], padding=2)
                out = F.batch_norm(out,
                    weights["blocks.0.norm.norm.running_mean"],
                    weights["blocks.0.norm.norm.running_var"],
                    weights["blocks.0.norm.norm.weight"],
                    weights["blocks.0.norm.norm.bias"], False)
                out = F.relu(out)

        # Benchmark
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            with torch.no_grad():
                out = F.conv1d(x, weights["blocks.0.conv.conv.weight"],
                              weights["blocks.0.conv.conv.bias"], padding=2)
                out = F.batch_norm(out,
                    weights["blocks.0.norm.norm.running_mean"],
                    weights["blocks.0.norm.norm.running_var"],
                    weights["blocks.0.norm.norm.weight"],
                    weights["blocks.0.norm.norm.bias"], False)
                out = F.relu(out)
            times.append(time.perf_counter() - start)

        avg_time = np.mean(times) * 1000
        std_time = np.std(times) * 1000
        results[batch_size] = {
            "avg_ms": avg_time,
            "std_ms": std_time,
            "throughput": batch_size / (avg_time / 1000),
        }

    return results


def benchmark_pytorch_mps(inputs: Dict[int, np.ndarray], weights_path: str, num_iterations: int = 100) -> Dict[str, Any]:
    """Benchmark PyTorch MPS (Metal) inference."""
    if not torch.backends.mps.is_available():
        return {"error": "MPS not available"}

    device = torch.device("mps")
    weights = torch.load(weights_path, map_location="cpu", weights_only=False)

    # Move weights to MPS
    w_conv = weights["blocks.0.conv.conv.weight"].to(device)
    b_conv = weights["blocks.0.conv.conv.bias"].to(device)
    rm = weights["blocks.0.norm.norm.running_mean"].to(device)
    rv = weights["blocks.0.norm.norm.running_var"].to(device)
    w_bn = weights["blocks.0.norm.norm.weight"].to(device)
    b_bn = weights["blocks.0.norm.norm.bias"].to(device)

    results = {}
    for batch_size, input_np in inputs.items():
        x = torch.from_numpy(input_np).to(device)

        # Warm up
        for _ in range(10):
            with torch.no_grad():
                out = F.conv1d(x, w_conv, b_conv, padding=2)
                out = F.batch_norm(out, rm, rv, w_bn, b_bn, False)
                out = F.relu(out)
            torch.mps.synchronize()

        # Benchmark
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            with torch.no_grad():
                out = F.conv1d(x, w_conv, b_conv, padding=2)
                out = F.batch_norm(out, rm, rv, w_bn, b_bn, False)
                out = F.relu(out)
            torch.mps.synchronize()
            times.append(time.perf_counter() - start)

        avg_time = np.mean(times) * 1000
        std_time = np.std(times) * 1000
        results[batch_size] = {
            "avg_ms": avg_time,
            "std_ms": std_time,
            "throughput": batch_size / (avg_time / 1000),
        }

    return results


def benchmark_mlx(inputs: Dict[int, np.ndarray], num_iterations: int = 100) -> Dict[str, Any]:
    """Benchmark MLX inference (Block 0 only)."""
    import mlx.core as mx
    from sota.ecapa_tdnn import ECAPATDNNForLanguageID
    from sota.ecapa_config import ECAPATDNNConfig

    # Create and load model
    config = ECAPATDNNConfig.voxlingua107()
    model = ECAPATDNNForLanguageID(config)

    weights = mx.load("models/sota/ecapa-tdnn-mlx/weights.npz")
    model.load_weights(list(weights.items()))

    results = {}
    for batch_size, input_np in inputs.items():
        # Convert to MLX native format (B, T, C) from PyTorch format (B, C, T)
        x = mx.array(input_np.transpose(0, 2, 1))

        # Warm up
        for _ in range(10):
            out = model.embedding_model.blocks_0(x)
            mx.eval(out)

        # Benchmark
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            out = model.embedding_model.blocks_0(x)
            mx.eval(out)
            times.append(time.perf_counter() - start)

        avg_time = np.mean(times) * 1000
        std_time = np.std(times) * 1000
        results[batch_size] = {
            "avg_ms": avg_time,
            "std_ms": std_time,
            "throughput": batch_size / (avg_time / 1000),
        }

    return results


def benchmark_mlx_full(inputs: Dict[int, np.ndarray], num_iterations: int = 50, use_compile: bool = True) -> Dict[str, Any]:
    """Benchmark full MLX model inference."""
    import mlx.core as mx
    from sota.ecapa_tdnn import ECAPATDNNForLanguageID
    from sota.ecapa_config import ECAPATDNNConfig

    # Create and load model
    config = ECAPATDNNConfig.voxlingua107()
    model = ECAPATDNNForLanguageID(config)

    weights = mx.load("models/sota/ecapa-tdnn-mlx/weights.npz")
    model.load_weights(list(weights.items()))

    results = {}
    for batch_size, input_np in inputs.items():
        # Input can be either format - model handles conversion
        x = mx.array(input_np)

        # Warm up
        for _ in range(5):
            logits, pred = model(x, use_compile=use_compile)
            mx.eval(logits, pred)

        # Benchmark
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            logits, pred = model(x, use_compile=use_compile)
            mx.eval(logits, pred)
            times.append(time.perf_counter() - start)

        avg_time = np.mean(times) * 1000
        std_time = np.std(times) * 1000
        results[batch_size] = {
            "avg_ms": avg_time,
            "std_ms": std_time,
            "throughput": batch_size / (avg_time / 1000),
        }

    return results


def main():
    """Main benchmark function."""
    print("=" * 60)
    print("ECAPA-TDNN Performance Benchmark")
    print("=" * 60)

    # Test parameters
    batch_sizes = [1, 4, 8]
    sequence_length = 300  # ~3 seconds of 16kHz audio at 10ms hop
    num_iterations = 100

    print("\nParameters:")
    print(f"  Batch sizes: {batch_sizes}")
    print(f"  Sequence length: {sequence_length} frames (~3 seconds)")
    print(f"  Iterations: {num_iterations}")

    # Create inputs
    inputs = create_test_inputs(batch_sizes, sequence_length)

    results = {}

    # Benchmark PyTorch CPU
    print("\n" + "-" * 40)
    print("PyTorch CPU (Block 0)")
    print("-" * 40)
    results["pytorch_cpu"] = benchmark_pytorch_cpu(
        inputs, "models/sota/ecapa-tdnn/embedding_model.ckpt", num_iterations
    )
    for bs, res in results["pytorch_cpu"].items():
        print(f"  Batch {bs}: {res['avg_ms']:.2f} ms (±{res['std_ms']:.2f}), "
              f"throughput: {res['throughput']:.1f} samples/s")

    # Benchmark PyTorch MPS
    print("\n" + "-" * 40)
    print("PyTorch MPS (Block 0)")
    print("-" * 40)
    results["pytorch_mps"] = benchmark_pytorch_mps(
        inputs, "models/sota/ecapa-tdnn/embedding_model.ckpt", num_iterations
    )
    if "error" in results["pytorch_mps"]:
        print(f"  {results['pytorch_mps']['error']}")
    else:
        for bs, res in results["pytorch_mps"].items():
            print(f"  Batch {bs}: {res['avg_ms']:.2f} ms (±{res['std_ms']:.2f}), "
                  f"throughput: {res['throughput']:.1f} samples/s")

    # Benchmark MLX (Block 0 only)
    print("\n" + "-" * 40)
    print("MLX (Block 0)")
    print("-" * 40)
    results["mlx_block0"] = benchmark_mlx(inputs, num_iterations)
    for bs, res in results["mlx_block0"].items():
        print(f"  Batch {bs}: {res['avg_ms']:.2f} ms (±{res['std_ms']:.2f}), "
              f"throughput: {res['throughput']:.1f} samples/s")

    # Benchmark MLX full model (compiled)
    print("\n" + "-" * 40)
    print("MLX Full Model (with mx.compile)")
    print("-" * 40)
    results["mlx_full_compiled"] = benchmark_mlx_full(inputs, num_iterations=50, use_compile=True)
    for bs, res in results["mlx_full_compiled"].items():
        print(f"  Batch {bs}: {res['avg_ms']:.2f} ms (±{res['std_ms']:.2f}), "
              f"throughput: {res['throughput']:.1f} samples/s")

    # Benchmark MLX full model (uncompiled)
    print("\n" + "-" * 40)
    print("MLX Full Model (without mx.compile)")
    print("-" * 40)
    results["mlx_full_uncompiled"] = benchmark_mlx_full(inputs, num_iterations=50, use_compile=False)
    for bs, res in results["mlx_full_uncompiled"].items():
        print(f"  Batch {bs}: {res['avg_ms']:.2f} ms (±{res['std_ms']:.2f}), "
              f"throughput: {res['throughput']:.1f} samples/s")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\nBlock 0 Speedup (MLX vs PyTorch CPU):")
    for bs in batch_sizes:
        if bs in results["pytorch_cpu"] and bs in results["mlx_block0"]:
            speedup = results["pytorch_cpu"][bs]["avg_ms"] / results["mlx_block0"][bs]["avg_ms"]
            print(f"  Batch {bs}: {speedup:.2f}x")

    if "error" not in results["pytorch_mps"]:
        print("\nBlock 0 Speedup (MLX vs PyTorch MPS):")
        for bs in batch_sizes:
            if bs in results["pytorch_mps"] and bs in results["mlx_block0"]:
                speedup = results["pytorch_mps"][bs]["avg_ms"] / results["mlx_block0"][bs]["avg_ms"]
                print(f"  Batch {bs}: {speedup:.2f}x")

    print("\nmx.compile() Speedup (full model):")
    for bs in batch_sizes:
        if bs in results["mlx_full_compiled"] and bs in results["mlx_full_uncompiled"]:
            speedup = results["mlx_full_uncompiled"][bs]["avg_ms"] / results["mlx_full_compiled"][bs]["avg_ms"]
            print(f"  Batch {bs}: {speedup:.2f}x")

    # Save results
    output_path = Path("models/sota/ecapa-tdnn-mlx/benchmark_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
