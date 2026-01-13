#!/usr/bin/env python3
"""
Isolate which MPS operation causes the parallel race condition.

Tests individual operations in parallel to narrow down:
1. LayerNorm alone
2. SDPA alone
3. Linear/MLP alone
4. LayerNorm + Linear combo
5. Full TransformerBlock

Goal: Identify the SMALLEST operation that reproduces the race.
"""

import argparse
import os
import threading
import sys
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["MPS_FORCE_GRAPH_PATH"] = "1"


@dataclass
class TestResult:
    name: str
    passed: int
    total: int
    max_diff: float
    failures: list


def run_parallel_test(
    name: str,
    model_factory: Callable[[], nn.Module],
    input_factory: Callable[[int], torch.Tensor],
    num_threads: int = 8,
    iterations: int = 20,
    tolerance: float = 1e-3,
) -> TestResult:
    """Run a parallel test on the given model."""
    model = model_factory().to("mps").eval()
    torch.mps.synchronize()

    passed = 0
    max_diff_overall = 0.0
    all_failures = []

    for it in range(iterations):
        # Create fresh inputs each iteration
        inputs = [input_factory(tid).to("mps") for tid in range(num_threads)]

        # Compute golden outputs sequentially
        expected = []
        with torch.no_grad():
            for inp in inputs:
                expected.append(model(inp).clone())
        torch.mps.synchronize()

        # Run in parallel
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
            all_failures.append(f"iter {it}: {len(errors)} errors")
            continue

        # Check results
        iteration_ok = True
        for tid in range(num_threads):
            if results[tid] is None:
                iteration_ok = False
                all_failures.append(f"iter {it} tid {tid}: None")
                continue
            diff = (results[tid] - expected[tid]).abs().max().item()
            max_diff_overall = max(max_diff_overall, diff)
            if diff > tolerance:
                iteration_ok = False
                all_failures.append(f"iter {it} tid {tid}: diff={diff:.2e}")

        if iteration_ok:
            passed += 1

    return TestResult(
        name=name,
        passed=passed,
        total=iterations,
        max_diff=max_diff_overall,
        failures=all_failures[:5],  # Only keep first 5 failures
    )


def main():
    parser = argparse.ArgumentParser(description="Isolate failing MPS operation")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--test", choices=["all", "layernorm", "linear", "sdpa", "mlp", "ln_linear", "transformer"], default="all")
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        print("SKIP: MPS not available")
        return 0

    print("=" * 70)
    print("MPS Operation Isolation Test")
    print(f"Threads: {args.threads}, Iterations: {args.iterations}, Tolerance: {args.tolerance}")
    print("=" * 70)

    # Define test configurations
    embed_dim = 256
    seq_len = 128
    batch_size = 4
    num_heads = 4

    tests_to_run = []

    def make_input(tid):
        torch.manual_seed(tid * 1000 + 42)
        return torch.randn(batch_size, seq_len, embed_dim)

    # Test 1: LayerNorm alone
    if args.test in ["all", "layernorm"]:
        tests_to_run.append((
            "LayerNorm only",
            lambda: nn.LayerNorm(embed_dim),
            make_input,
        ))

    # Test 2: Linear layer alone
    if args.test in ["all", "linear"]:
        tests_to_run.append((
            "Linear only",
            lambda: nn.Linear(embed_dim, embed_dim),
            make_input,
        ))

    # Test 3: MLP (Linear + GELU + Linear)
    if args.test in ["all", "mlp"]:
        def make_mlp():
            return nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Linear(embed_dim * 4, embed_dim),
            )
        tests_to_run.append((
            "MLP (Linear+GELU+Linear)",
            make_mlp,
            make_input,
        ))

    # Test 4: SDPA alone (via MultiheadAttention)
    if args.test in ["all", "sdpa"]:
        class SDPAOnly(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

            def forward(self, x):
                out, _ = self.attn(x, x, x, need_weights=False)
                return out

        tests_to_run.append((
            "SDPA (MultiheadAttention)",
            SDPAOnly,
            make_input,
        ))

    # Test 5: LayerNorm + Linear combo
    if args.test in ["all", "ln_linear"]:
        class LNLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.ln = nn.LayerNorm(embed_dim)
                self.linear = nn.Linear(embed_dim, embed_dim)

            def forward(self, x):
                return self.linear(self.ln(x))

        tests_to_run.append((
            "LayerNorm + Linear",
            LNLinear,
            make_input,
        ))

    # Test 6: Full TransformerBlock
    if args.test in ["all", "transformer"]:
        class TransformerBlock(nn.Module):
            def __init__(self):
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

        tests_to_run.append((
            "Full TransformerBlock",
            TransformerBlock,
            make_input,
        ))

    # Run all tests
    results = []
    for name, model_factory, input_factory in tests_to_run:
        print(f"\nTesting: {name}...")
        result = run_parallel_test(
            name=name,
            model_factory=model_factory,
            input_factory=input_factory,
            num_threads=args.threads,
            iterations=args.iterations,
            tolerance=args.tolerance,
        )
        results.append(result)
        status = "PASS" if result.passed == result.total else "FAIL"
        print(f"  {status}: {result.passed}/{result.total} iterations, max_diff={result.max_diff:.2e}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Operation':<30} {'Pass/Total':<15} {'Max Diff':<12} Status")
    print("-" * 70)

    for r in results:
        status = "OK" if r.passed == r.total else "RACE"
        print(f"{r.name:<30} {r.passed}/{r.total:<12} {r.max_diff:<12.2e} {status}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    failing_ops = [r for r in results if r.passed < r.total]
    passing_ops = [r for r in results if r.passed == r.total]

    if not failing_ops:
        print("All operations passed! Race condition may be intermittent.")
        print("Try running with more iterations: --iterations 100")
    else:
        print("Operations with race conditions:")
        for r in failing_ops:
            failure_rate = (r.total - r.passed) / r.total * 100
            print(f"  - {r.name}: {failure_rate:.0f}% failure rate")
            if r.failures:
                print(f"    Sample failures: {r.failures[:3]}")

    if passing_ops:
        print("\nOperations that passed:")
        for r in passing_ops:
            print(f"  - {r.name}")

    # Return 0 if all pass, 1 if any fail
    return 0 if not failing_ops else 1


if __name__ == "__main__":
    sys.exit(main())
