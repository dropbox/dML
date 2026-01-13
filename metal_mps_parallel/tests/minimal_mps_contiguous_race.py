#!/usr/bin/env python3
"""
Minimal reproduction: MPS .contiguous() race condition with parallel threads

This script demonstrates a race condition in PyTorch's MPS backend when
.contiguous() is called on complex reshaped tensors from multiple threads.

Environment:
- macOS with Apple Silicon (M1/M2/M3/M4)
- PyTorch 2.x with MPS backend

Issue Summary:
- Calling .contiguous() on tensors with complex stride patterns
  (from unflatten/transpose operations) fails intermittently when
  called from multiple threads simultaneously.
- The same operations WITHOUT .contiguous() pass 100%.
- This affects nn.MultiheadAttention which uses this pattern internally.

To reproduce:
    python3 minimal_mps_contiguous_race.py

Expected: Both tests should pass 30/30
Actual: "With .contiguous()" typically fails ~10-15 out of 30 iterations
"""

import os
import threading
import sys

# Force graph path mode (consistent with production usage)
os.environ["MPS_FORCE_GRAPH_PATH"] = "1"

import torch
import torch.nn.functional as F


def test_contiguous_race(use_contiguous: bool, iterations: int = 30, threads: int = 8):
    """
    Test parallel .contiguous() calls on MPS.

    Args:
        use_contiguous: If True, call .contiguous() after reshape ops
        iterations: Number of test iterations
        threads: Number of parallel threads

    Returns:
        Tuple of (passed_count, total_count, max_difference)
    """
    if not torch.backends.mps.is_available():
        print("SKIP: MPS not available")
        return iterations, iterations, 0.0

    # Setup: Create weight matrices on MPS
    embed_dim = 256
    batch_size = 4
    seq_len = 128

    torch.manual_seed(42)
    weight = torch.randn(3 * embed_dim, embed_dim, device="mps")
    bias = torch.randn(3 * embed_dim, device="mps")
    torch.mps.synchronize()

    num_heads = 4
    head_dim = embed_dim // num_heads

    def projection_op(x):
        """
        Mimics PyTorch's _in_projection_packed pattern:
        1. Linear projection to 3*embed_dim
        2. Reshape to separate Q, K, V
        3. Optionally call .contiguous() (this is where the race occurs)
        4. Run SDPA (to mimic full MHA path)
        """
        proj = F.linear(x, weight, bias)

        # PyTorch's reshape pattern (from _in_projection_packed)
        proj = (
            proj.unflatten(-1, (3, embed_dim))
            .unsqueeze(0)
            .transpose(0, -2)
            .squeeze(-2)
        )

        if use_contiguous:
            proj = proj.contiguous()  # <-- RACE CONDITION TRIGGER

        # Extract Q, K, V
        q, k, v = proj[0], proj[1], proj[2]

        # Reshape for multi-head attention
        if use_contiguous:
            q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        else:
            q = q.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            k = k.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            v = v.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

        # Run SDPA (mimics full MHA path)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        return out

    passed = 0
    max_diff = 0.0

    for iteration in range(iterations):
        # Create unique inputs for each thread
        inputs = []
        for tid in range(threads):
            torch.manual_seed(iteration * 1000 + tid)
            inputs.append(torch.randn(batch_size, seq_len, embed_dim, device="mps"))
        torch.mps.synchronize()

        # Compute expected results serially
        expected = []
        with torch.no_grad():
            for inp in inputs:
                expected.append(projection_op(inp).clone())
        torch.mps.synchronize()

        # Run in parallel
        results = [None] * threads

        def worker(tid):
            with torch.no_grad():
                results[tid] = projection_op(inputs[tid])
            torch.mps.synchronize()

        worker_threads = [threading.Thread(target=worker, args=(i,)) for i in range(threads)]
        for t in worker_threads:
            t.start()
        for t in worker_threads:
            t.join()

        # Check results
        iteration_ok = True
        for tid in range(threads):
            if results[tid] is None:
                iteration_ok = False
                continue
            diff = (results[tid] - expected[tid]).abs().max().item()
            max_diff = max(max_diff, diff)
            if diff > 1e-3:
                iteration_ok = False

        if iteration_ok:
            passed += 1

    return passed, iterations, max_diff


def main():
    print("=" * 70)
    print("MPS .contiguous() Race Condition Reproduction")
    print("=" * 70)
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print()

    if not torch.backends.mps.is_available():
        print("SKIP: MPS not available")
        return 0

    # Test WITHOUT .contiguous() - should always pass
    print("Test 1: WITHOUT .contiguous() (expected: PASS)")
    passed, total, diff = test_contiguous_race(use_contiguous=False)
    status1 = "PASS" if passed == total else "FAIL"
    print(f"  Result: {status1} ({passed}/{total}), max_diff={diff:.2e}")
    print()

    # Test WITH .contiguous() - demonstrates the race
    print("Test 2: WITH .contiguous() (demonstrates race condition)")
    passed, total, diff = test_contiguous_race(use_contiguous=True)
    status2 = "PASS" if passed == total else "FAIL"
    print(f"  Result: {status2} ({passed}/{total}), max_diff={diff:.2e}")
    print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if status1 == "PASS" and status2 == "FAIL":
        print("BUG REPRODUCED: .contiguous() triggers race condition")
        print()
        print("Root cause: When multiple threads call .contiguous() on tensors")
        print("with complex stride patterns (from unflatten/transpose), there is")
        print("a race condition in MPS memory allocation or copy operations.")
        print()
        print("Workaround: Avoid .contiguous() on complex reshaped tensors.")
        print("Use .reshape() instead of .view() to handle non-contiguous tensors.")
        return 1
    elif status1 == "PASS" and status2 == "PASS":
        print("Bug not reproduced this run (race is intermittent)")
        print("Try running multiple times or increasing iterations")
        return 0
    else:
        print("Unexpected result - both tests should pass without contiguous()")
        return 2


if __name__ == "__main__":
    sys.exit(main())
