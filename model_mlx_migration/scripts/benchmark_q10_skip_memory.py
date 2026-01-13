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
Q10: Skip Connection Memory Reuse Benchmark

Evaluates whether preallocating skip buffers provides any benefit
for CosyVoice3 DiT model.

Current implementation:
    h_init = h  # Reference assignment
    for i, block in enumerate(blocks):
        h = block(h)
        if i == depth // 2:
            h = skip_proj(concat([h, h_init]))

Proposed optimization:
    self._skip_buffer = preallocated
    for i, block in enumerate(blocks):
        if i == 0:
            self._skip_buffer = h  # "Reuse" buffer
        ...

In MLX, arrays are IMMUTABLE. h_init = h is already a reference (O(1)).
There's no copy to optimize.
"""

import time
import mlx.core as mx


def verify_reference_semantics():
    """Verify that MLX array assignment is reference, not copy."""
    print("=" * 60)
    print("Verifying MLX Reference Semantics")
    print("=" * 60)

    # Create array
    a = mx.random.normal((100, 100, 100))
    mx.eval(a)

    # Time assignment
    n_runs = 1000
    start = time.perf_counter()
    for _ in range(n_runs):
        b = a  # Should be O(1) reference
    elapsed = (time.perf_counter() - start) * 1000 / n_runs

    print(f"\n  Assignment time: {elapsed*1000:.4f} microseconds")
    print(f"  Array size: {a.nbytes / 1024 / 1024:.1f} MB")

    # Verify same memory (via id - won't work for MLX but check values)
    b = a
    print(f"\n  a.sum(): {float(a.sum()):.4f}")
    print(f"  b.sum(): {float(b.sum()):.4f}")
    print(f"  Same values: {bool(mx.all(a == b))}")

    # Check if modifying b affects a (it shouldn't in MLX)
    c = a
    c = c + 1  # This creates new array
    print("\n  After c = a; c = c + 1:")
    print(f"  a.sum(): {float(a.sum()):.4f} (unchanged)")
    print(f"  c.sum(): {float(c.sum()):.4f} (new array)")


def benchmark_skip_patterns():
    """Benchmark different skip connection patterns."""
    print("\n" + "=" * 60)
    print("Q10: Skip Connection Pattern Benchmark")
    print("=" * 60)

    # Typical DiT dimensions
    B = 1
    L = 100
    D = 1024
    depth = 22

    print(f"\n  Batch: {B}, Length: {L}, Dim: {D}, Depth: {depth}")

    # Create input
    h = mx.random.normal((B, L, D))
    mx.eval(h)

    # Pattern 1: Current (reference assignment)
    def pattern_current():
        x = h
        h_init = x  # Reference
        for i in range(depth):
            x = x * 1.0001  # Simulate block computation
            if i == depth // 2:
                skip = mx.concatenate([x, h_init], axis=-1)
                x = skip[:, :, :D]  # Simulate skip_proj
        return x

    # Pattern 2: Preallocated buffer (Q10)
    skip_buffer = mx.zeros_like(h)

    def pattern_preallocated():
        nonlocal skip_buffer
        x = h
        skip_buffer = x  # "Reuse" buffer
        for i in range(depth):
            x = x * 1.0001
            if i == depth // 2:
                skip = mx.concatenate([x, skip_buffer], axis=-1)
                x = skip[:, :, :D]
        return x

    # Pattern 3: Direct reference (most explicit)
    def pattern_direct():
        x = h
        # No separate variable, access h directly at skip point
        for i in range(depth):
            x = x * 1.0001
            if i == depth // 2:
                skip = mx.concatenate([x, h], axis=-1)
                x = skip[:, :, :D]
        return x

    # Warmup
    for _ in range(3):
        out = pattern_current()
        mx.eval(out)
        out = pattern_preallocated()
        mx.eval(out)
        out = pattern_direct()
        mx.eval(out)

    # Benchmark
    n_runs = 50

    times_current = []
    for _ in range(n_runs):
        start = time.perf_counter()
        out = pattern_current()
        mx.eval(out)
        times_current.append((time.perf_counter() - start) * 1000)

    times_prealloc = []
    for _ in range(n_runs):
        start = time.perf_counter()
        out = pattern_preallocated()
        mx.eval(out)
        times_prealloc.append((time.perf_counter() - start) * 1000)

    times_direct = []
    for _ in range(n_runs):
        start = time.perf_counter()
        out = pattern_direct()
        mx.eval(out)
        times_direct.append((time.perf_counter() - start) * 1000)

    current_mean = sum(times_current) / len(times_current)
    prealloc_mean = sum(times_prealloc) / len(times_prealloc)
    direct_mean = sum(times_direct) / len(times_direct)

    print("\n" + "-" * 60)
    print("Results:")
    print("-" * 60)
    print("\n| Pattern | Latency (ms) | vs Current |")
    print("|---------|--------------|------------|")
    print(f"| Current (h_init = h) | {current_mean:.3f} | 1.00x |")
    print(f"| Q10 Preallocated | {prealloc_mean:.3f} | {current_mean/prealloc_mean:.2f}x |")
    print(f"| Direct reference (h) | {direct_mean:.3f} | {current_mean/direct_mean:.2f}x |")

    # Verdict
    print("\n" + "=" * 60)
    print("Verdict:")
    print("=" * 60)
    print("\n  Q10 is NOT APPLICABLE to MLX:")
    print("  1. MLX arrays are immutable - assignments are references (O(1))")
    print("  2. No memory copy occurs with h_init = h")
    print("  3. Preallocating a buffer provides no benefit")
    print("  4. The optimization assumes mutable arrays (NumPy/PyTorch pattern)")

    return {
        'current_ms': current_mean,
        'prealloc_ms': prealloc_mean,
        'direct_ms': direct_mean,
    }


if __name__ == "__main__":
    verify_reference_semantics()
    results = benchmark_skip_patterns()
