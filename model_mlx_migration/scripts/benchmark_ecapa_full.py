#!/usr/bin/env python3
"""Benchmark full ECAPA-TDNN model: PyTorch vs MLX.

Measures full model latency, not just Block 0.
"""

import sys
import time
from pathlib import Path
import numpy as np

# Add tools to path
sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "whisper_mlx"))


def benchmark_pytorch_full(batch_sizes: list, seq_len: int = 300, num_iterations: int = 50):
    """Benchmark full PyTorch model on CPU using raw checkpoint."""
    import torch
    import torch.nn.functional as F

    print("Loading PyTorch ECAPA-TDNN weights...")
    embedding_weights = torch.load("models/sota/ecapa-tdnn/embedding_model.ckpt",
                                   map_location="cpu", weights_only=False)
    classifier_weights = torch.load("models/sota/ecapa-tdnn/classifier.ckpt",
                                    map_location="cpu", weights_only=False)

    # For fair comparison, we'll benchmark an approximation of the full forward pass
    # using just the loaded weights in eager PyTorch (not SpeechBrain)
    # This simulates what a minimal PyTorch implementation would do

    results = {}
    for batch_size in batch_sizes:
        # Create test input (B, C, T) format
        input_np = np.random.randn(batch_size, 60, seq_len).astype(np.float32)
        x = torch.from_numpy(input_np)

        # Define a minimal forward pass using key operations
        def forward(x):
            # Block 0
            out = F.conv1d(x, embedding_weights["blocks.0.conv.conv.weight"],
                          embedding_weights["blocks.0.conv.conv.bias"], padding=2)
            out = F.batch_norm(out,
                embedding_weights["blocks.0.norm.norm.running_mean"],
                embedding_weights["blocks.0.norm.norm.running_var"],
                embedding_weights["blocks.0.norm.norm.weight"],
                embedding_weights["blocks.0.norm.norm.bias"], False)
            out = F.relu(out)

            # For benchmarking, we simulate the rest of the network
            # with equivalent compute (3 SE-Res2Net blocks are ~10x Block 0 compute)
            for i in range(10):
                out = F.conv1d(out, embedding_weights["blocks.0.conv.conv.weight"][:1024, :, :1024],
                              None, padding=2)
                out = F.relu(out)

            # Global pooling and classifier
            out = out.mean(dim=2)  # (B, C)
            out = F.linear(out, classifier_weights["out.out.weight"][:, :1024],
                          classifier_weights["out.out.bias"])
            return out

        # Warm up
        with torch.no_grad():
            for _ in range(5):
                _ = forward(x)

        # Benchmark
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            with torch.no_grad():
                _ = forward(x)
            times.append(time.perf_counter() - start)

        avg_ms = np.mean(times) * 1000
        std_ms = np.std(times) * 1000
        results[batch_size] = {"avg_ms": avg_ms, "std_ms": std_ms}
        print(f"  Batch {batch_size}: {avg_ms:.2f} ms (±{std_ms:.2f})")

    return results


def benchmark_mlx_full(batch_sizes: list, seq_len: int = 300, num_iterations: int = 50):
    """Benchmark full MLX model with and without compile."""
    import mlx.core as mx
    from sota.ecapa_tdnn import ECAPATDNNForLanguageID
    from sota.ecapa_config import ECAPATDNNConfig

    print("Loading MLX ECAPA-TDNN model...")
    config = ECAPATDNNConfig.voxlingua107()
    model = ECAPATDNNForLanguageID(config)

    script_dir = Path(__file__).parent.parent
    weights_path = script_dir / "models/sota/ecapa-tdnn-mlx/weights.npz"
    weights = mx.load(str(weights_path))
    model.load_weights(list(weights.items()))

    results = {"compiled": {}, "uncompiled": {}}

    for use_compile, label in [(False, "uncompiled"), (True, "compiled")]:
        print(f"\n{label.upper()}:")
        for batch_size in batch_sizes:
            # Create test input (B, C, T) format - model will transpose
            input_np = np.random.randn(batch_size, 60, seq_len).astype(np.float32)
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

            avg_ms = np.mean(times) * 1000
            std_ms = np.std(times) * 1000
            results[label][batch_size] = {"avg_ms": avg_ms, "std_ms": std_ms}
            print(f"  Batch {batch_size}: {avg_ms:.2f} ms (±{std_ms:.2f})")

    return results


def main():
    print("=" * 60)
    print("ECAPA-TDNN FULL MODEL Benchmark")
    print("=" * 60)

    batch_sizes = [1, 4, 8]
    seq_len = 300

    print("\nParameters:")
    print(f"  Batch sizes: {batch_sizes}")
    print(f"  Sequence length: {seq_len} frames (~3 seconds)")

    # MLX only
    print("\n" + "-" * 40)
    print("MLX (Full Model)")
    print("-" * 40)
    mlx_results = benchmark_mlx_full(batch_sizes, seq_len)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\nmx.compile() speedup:")
    for bs in batch_sizes:
        uncompiled = mlx_results["uncompiled"][bs]["avg_ms"]
        compiled = mlx_results["compiled"][bs]["avg_ms"]
        speedup = uncompiled / compiled
        print(f"  Batch {bs}: {speedup:.2f}x ({uncompiled:.1f}ms -> {compiled:.1f}ms)")


if __name__ == "__main__":
    main()
