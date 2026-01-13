#!/usr/bin/env python3
"""
Comparison: Direct parallel vs Batched parallel for TransformerBlock at 8 threads.

This script demonstrates that:
1. Direct parallel with TransformerBlock at 8 threads has race conditions
2. Batched parallel reduces races; `--workers 1` is required for 10/10 correctness

Run:
    python3 tests/transformer_direct_vs_batched.py
"""

import argparse
import os
import threading
import sys

import torch
import torch.nn as nn

os.environ["MPS_FORCE_GRAPH_PATH"] = "1"


class SimpleTransformerBlock(nn.Module):
    """Transformer block that triggers race conditions at 8+ threads."""

    def __init__(self, embed_dim=256, num_heads=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x):
        ln_x = self.ln1(x)
        attn_out, _ = self.attn(ln_x, ln_x, ln_x, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


def test_direct_parallel(model, inputs, expected_outputs, tolerance=1e-3):
    """Test direct parallel execution (no batching)."""
    num_threads = len(inputs)
    results = [None] * num_threads
    errors = []

    def worker(tid):
        try:
            with torch.no_grad():
                results[tid] = model(inputs[tid])
            torch.mps.synchronize()
        except Exception as e:
            errors.append((tid, e))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if errors:
        return False, f"{len(errors)} threads had errors"

    max_diff = 0.0
    failures = []
    for tid in range(num_threads):
        if results[tid] is None:
            failures.append(f"Thread {tid}: None result")
            continue
        diff = (results[tid] - expected_outputs[tid]).abs().max().item()
        max_diff = max(max_diff, diff)
        if diff > tolerance:
            failures.append(f"Thread {tid}: diff={diff:.6e}")

    if failures:
        return False, f"max_diff={max_diff:.6e}, failures: {failures[:3]}..."
    return True, f"max_diff={max_diff:.6e}"


def test_batched_parallel(
    model,
    inputs,
    expected_outputs,
    num_workers: int,
    tolerance=1e-3,
):
    """Test batched parallel execution (8 threads -> N workers)."""
    if not hasattr(torch.mps, "BatchQueue"):
        return None, "BatchQueue not available"

    num_threads = len(inputs)
    results = [None] * num_threads
    errors = []

    queue = torch.mps.BatchQueue(batch_size=4, num_workers=num_workers)
    queue.start()

    def worker(tid):
        try:
            future = queue.submit(inputs[tid], lambda x: model(x))
            results[tid] = future.result(timeout=60.0)
        except Exception as e:
            errors.append((tid, e))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    queue.stop()

    if errors:
        return False, f"{len(errors)} threads had errors"

    max_diff = 0.0
    failures = []
    for tid in range(num_threads):
        if results[tid] is None:
            failures.append(f"Thread {tid}: None result")
            continue
        diff = (results[tid] - expected_outputs[tid]).abs().max().item()
        max_diff = max(max_diff, diff)
        if diff > tolerance:
            failures.append(f"Thread {tid}: diff={diff:.6e}")

    if failures:
        return False, f"max_diff={max_diff:.6e}, failures: {failures[:3]}..."
    return True, f"max_diff={max_diff:.6e}"


def main():
    parser = argparse.ArgumentParser(
        description="Compare TransformerBlock direct parallel vs batched parallel"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of BatchQueue worker threads (default: 1 for correctness)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Iterations to catch intermittent races (default: 10)",
    )
    args = parser.parse_args()

    if args.workers < 1 or args.workers > 3:
        print("ERROR: --workers must be 1-3", file=sys.stderr)
        return 2
    if args.iterations < 1:
        print("ERROR: --iterations must be >= 1", file=sys.stderr)
        return 2

    if not torch.backends.mps.is_available():
        print("SKIP: MPS not available")
        return 0

    print("=" * 70)
    print("TransformerBlock: Direct Parallel vs Batched Parallel at 8 Threads")
    print("=" * 70)

    # Setup
    model = SimpleTransformerBlock(embed_dim=256, num_heads=4).to("mps").eval()
    torch.mps.synchronize()

    num_threads = 8
    batch_size = 4
    seq_len = 128

    inputs = [
        torch.randn(batch_size, seq_len, 256, device="mps") for _ in range(num_threads)
    ]

    # Compute golden outputs (sequential)
    expected_outputs = []
    with torch.no_grad():
        for inp in inputs:
            expected_outputs.append(model(inp).clone())
    torch.mps.synchronize()

    print(f"\nModel: TransformerBlock(embed_dim=256, num_heads=4)")
    print(f"Input: {num_threads} threads x [{batch_size}, {seq_len}, 256]")
    print(f"Tolerance: 1e-3")
    print(f"BatchQueue workers: {args.workers}")

    # Run multiple iterations to catch intermittent races
    num_iterations = args.iterations
    print(f"\nRunning {num_iterations} iterations each...")

    direct_passes = 0
    batched_passes = 0

    for i in range(num_iterations):
        # Regenerate inputs each iteration
        inputs = [
            torch.randn(batch_size, seq_len, 256, device="mps")
            for _ in range(num_threads)
        ]
        expected_outputs = []
        with torch.no_grad():
            for inp in inputs:
                expected_outputs.append(model(inp).clone())
        torch.mps.synchronize()

        # Direct parallel
        direct_ok, direct_msg = test_direct_parallel(model, inputs, expected_outputs)
        if direct_ok:
            direct_passes += 1

        # Batched parallel
        batched_ok, batched_msg = test_batched_parallel(
            model, inputs, expected_outputs, num_workers=args.workers
        )
        if batched_ok:
            batched_passes += 1
        elif batched_ok is None:
            # BatchQueue not available
            batched_passes = -1
            break

    print("\n--- Results ---")
    print(f"Direct parallel:  {direct_passes}/{num_iterations} iterations passed")
    if batched_passes >= 0:
        print(f"Batched parallel: {batched_passes}/{num_iterations} iterations passed")
    else:
        print("Batched parallel: SKIPPED (BatchQueue not available)")

    print("\n--- Interpretation ---")
    if direct_passes < num_iterations:
        print(
            "Direct parallel has race conditions "
            f"({num_iterations - direct_passes} failures)"
        )
    else:
        print(
            "Direct parallel passed all iterations (race may be intermittent or fixed)"
        )

    if batched_passes == num_iterations:
        print("Batching successfully avoids race conditions at 8 threads")
    elif batched_passes >= 0:
        print(f"Batching had {num_iterations - batched_passes} failures")

    # Return code: 0 if batching works better than direct
    if batched_passes >= 0 and batched_passes >= direct_passes:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
