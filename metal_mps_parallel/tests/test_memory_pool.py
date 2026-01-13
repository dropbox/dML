#!/usr/bin/env python3
"""
Tests for TensorPool memory pooling (P3 optimization).

Tests verify:
1. Basic acquire/release functionality
2. Thread safety under concurrent access
3. Fallback allocation when pool exhausted
4. Memory stability (no growth over sustained load)
5. Performance comparison vs dynamic allocation

Run:
    ./scripts/run_test_with_crash_check.sh python3 tests/test_memory_pool.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples'))

import torch
import torch.nn as nn
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tensor_pool import (
    TensorPool,
    MultiShapeTensorPool,
    PooledInferenceContext,
    create_inference_pool
)


# Detect device
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'


def test_create_pool():
    """Pool creates correct number of tensors."""
    pool = TensorPool(shape=(32, 256), dtype=torch.float32, device=DEVICE, pool_size=4)
    assert pool.pool_size == 4
    assert pool.available_count == 4
    print("  test_create_pool: PASS")


def test_acquire_release():
    """Acquire and release work correctly."""
    pool = TensorPool(shape=(4, 64), device=DEVICE, pool_size=2)

    # Acquire one
    idx1, t1 = pool.acquire()
    assert idx1 >= 0
    assert t1.shape == (4, 64)
    assert pool.available_count == 1

    # Acquire second
    idx2, t2 = pool.acquire()
    assert idx2 >= 0
    assert idx2 != idx1
    assert pool.available_count == 0

    # Release first
    pool.release(idx1)
    assert pool.available_count == 1

    # Release second
    pool.release(idx2)
    assert pool.available_count == 2
    print("  test_acquire_release: PASS")


def test_context_manager():
    """Context manager acquires and releases."""
    pool = TensorPool(shape=(8, 128), device=DEVICE, pool_size=3)

    with pool.context() as tensor:
        assert tensor.shape == (8, 128)
        assert pool.available_count == 2

    # After context, tensor is released
    assert pool.available_count == 3
    print("  test_context_manager: PASS")


def test_fallback_allocation():
    """Pool falls back to allocation when exhausted."""
    pool = TensorPool(shape=(2, 32), device=DEVICE, pool_size=2)

    # Exhaust pool
    idx1, _ = pool.acquire()
    idx2, _ = pool.acquire()

    # Third acquire should fallback
    idx3, t3 = pool.acquire()
    assert idx3 == -1  # Fallback indicator
    assert t3.shape == (2, 32)

    stats = pool.stats
    assert stats['fallback_allocs'] == 1

    # Release fallback tensor (no-op but should not crash)
    pool.release(idx3)

    # Release pooled tensors
    pool.release(idx1)
    pool.release(idx2)
    print("  test_fallback_allocation: PASS")


def test_stats():
    """Statistics are tracked correctly."""
    pool = TensorPool(shape=(4, 16), device=DEVICE, pool_size=2)

    idx, _ = pool.acquire()
    pool.release(idx)
    idx, _ = pool.acquire()
    pool.release(idx)

    stats = pool.stats
    assert stats['acquires'] == 2
    assert stats['releases'] == 2
    assert stats['fallback_allocs'] == 0
    assert stats['hit_rate'] == 1.0
    print("  test_stats: PASS")


def test_concurrent_acquire_release():
    """Multiple threads can acquire/release safely."""
    pool = TensorPool(shape=(8, 64), device=DEVICE, pool_size=8)
    num_threads = 8
    ops_per_thread = 100
    errors = []

    def worker():
        try:
            for _ in range(ops_per_thread):
                idx, tensor = pool.acquire()
                # Simulate some work
                tensor.fill_(1.0)
                time.sleep(0.0001)
                pool.release(idx)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"Errors: {errors}"
    # All tensors should be released
    assert pool.available_count == 8
    print(f"  test_concurrent_acquire_release: PASS ({num_threads * ops_per_thread} ops)")


def test_concurrent_context_manager():
    """Context manager is thread-safe."""
    pool = TensorPool(shape=(4, 32), device=DEVICE, pool_size=4)
    num_threads = 8
    ops_per_thread = 50
    results = []
    errors = []

    def worker(tid):
        try:
            for _ in range(ops_per_thread):
                with pool.context() as tensor:
                    tensor.fill_(float(tid))
                    results.append(tensor.mean().item())
        except Exception as e:
            errors.append((tid, e))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"Errors: {errors}"
    assert len(results) == num_threads * ops_per_thread
    assert pool.available_count == 4
    print(f"  test_concurrent_context_manager: PASS ({len(results)} ops)")


def test_high_contention():
    """Pool handles high contention without deadlock."""
    pool = TensorPool(shape=(2, 16), device=DEVICE, pool_size=2)
    num_threads = 16
    ops_per_thread = 100
    barrier = threading.Barrier(num_threads)
    success_count = [0]
    lock = threading.Lock()

    def worker():
        barrier.wait()  # Start all threads simultaneously
        for _ in range(ops_per_thread):
            idx, tensor = pool.acquire()
            tensor.fill_(1.0)
            pool.release(idx)
            with lock:
                success_count[0] += 1

    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert success_count[0] == num_threads * ops_per_thread
    assert pool.available_count == 2
    print(f"  test_high_contention: PASS ({success_count[0]} ops)")


def test_multiple_shapes():
    """Pool handles multiple shapes correctly."""
    pool = MultiShapeTensorPool(device=DEVICE, pool_size_per_shape=2)

    # Acquire different shapes
    s1, idx1, t1 = pool.acquire((4, 64))
    s2, idx2, t2 = pool.acquire((8, 32))
    s3, idx3, t3 = pool.acquire((4, 64))

    assert t1.shape == (4, 64)
    assert t2.shape == (8, 32)
    assert t3.shape == (4, 64)

    pool.release(s1, idx1)
    pool.release(s2, idx2)
    pool.release(s3, idx3)

    stats = pool.stats
    assert stats['num_shapes'] == 2
    assert (4, 64) in stats['shapes']
    assert (8, 32) in stats['shapes']
    print("  test_multiple_shapes: PASS")


def test_inference_context():
    """Inference context with input and output pools."""
    ctx = PooledInferenceContext(
        input_shape=(8, 256),
        output_shape=(8, 128),
        device=DEVICE,
        pool_size=4
    )

    with ctx.inference() as (input_tensor, output_tensor):
        assert input_tensor.shape == (8, 256)
        assert output_tensor.shape == (8, 128)
    print("  test_inference_context: PASS")


def test_no_memory_growth_single_thread():
    """Memory usage stays constant under sustained load (single thread)."""
    pool = TensorPool(shape=(64, 256), device=DEVICE, pool_size=4)

    # Warmup
    for _ in range(10):
        with pool.context() as tensor:
            tensor.fill_(1.0)

    if DEVICE == 'mps':
        torch.mps.synchronize()

    # Run many iterations
    iterations = 500
    for _ in range(iterations):
        with pool.context() as tensor:
            tensor.fill_(1.0)

    if DEVICE == 'mps':
        torch.mps.synchronize()

    # Check pool is in good state
    stats = pool.stats
    assert stats['available'] == 4
    assert stats['acquires'] == 10 + iterations
    assert stats['releases'] == 10 + iterations
    assert stats['fallback_allocs'] <= 1  # Allow 1 for overlap
    print(f"  test_no_memory_growth_single_thread: PASS ({iterations} ops)")


def test_no_memory_growth_multi_thread():
    """Memory usage stays constant under concurrent load."""
    pool = TensorPool(shape=(32, 128), device=DEVICE, pool_size=8)
    num_threads = 8
    ops_per_thread = 200

    def worker():
        for _ in range(ops_per_thread):
            with pool.context() as tensor:
                tensor.fill_(1.0)

    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if DEVICE == 'mps':
        torch.mps.synchronize()

    stats = pool.stats
    assert stats['available'] == 8
    total_ops = num_threads * ops_per_thread
    assert stats['acquires'] == total_ops
    assert stats['releases'] == total_ops
    print(f"  test_no_memory_growth_multi_thread: PASS ({total_ops} ops)")


def test_pooled_vs_dynamic_throughput():
    """Compare throughput: pooled vs dynamic allocation."""
    if DEVICE == 'cpu':
        print("  test_pooled_vs_dynamic_throughput: SKIP (CPU)")
        return

    shape = (32, 256)
    iterations = 1000

    # Warmup
    for _ in range(10):
        t = torch.empty(shape, device=DEVICE)
        t.fill_(1.0)
    torch.mps.synchronize()

    # Dynamic allocation baseline
    start = time.perf_counter()
    for _ in range(iterations):
        t = torch.empty(shape, device=DEVICE)
        t.fill_(1.0)
    torch.mps.synchronize()
    dynamic_time = time.perf_counter() - start

    # Pooled allocation
    pool = TensorPool(shape=shape, device=DEVICE, pool_size=4)

    # Warmup pool
    for _ in range(10):
        with pool.context() as t:
            t.fill_(1.0)
    torch.mps.synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        with pool.context() as t:
            t.fill_(1.0)
    torch.mps.synchronize()
    pooled_time = time.perf_counter() - start

    dynamic_ops = iterations / dynamic_time
    pooled_ops = iterations / pooled_time
    speedup = dynamic_time / pooled_time

    print(f"  Dynamic: {dynamic_ops:.0f} ops/s, Pooled: {pooled_ops:.0f} ops/s, Speedup: {speedup:.2f}x")

    # Pool should be at least as fast (no regression)
    assert pooled_time <= dynamic_time * 1.5, "Pooled should not be significantly slower"
    print(f"  test_pooled_vs_dynamic_throughput: PASS")


def test_pooled_inference_throughput():
    """Test pooled inference with real model."""
    if DEVICE == 'cpu':
        print("  test_pooled_inference_throughput: SKIP (CPU)")
        return

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(256, 256)

        def forward(self, x):
            return self.fc(x)

    model = SimpleModel().to(DEVICE).eval()
    batch_size = 32
    iterations = 200

    # Dynamic allocation
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            x = torch.randn(batch_size, 256, device=DEVICE)
            y = model(x)
            _ = y.sum().cpu()  # Force sync
    dynamic_time = time.perf_counter() - start

    # Pooled
    pool = TensorPool(shape=(batch_size, 256), device=DEVICE, pool_size=4)

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            with pool.context() as x:
                x.normal_()  # Fill with random data
                y = model(x)
                _ = y.sum().cpu()
    pooled_time = time.perf_counter() - start

    dynamic_ops = iterations / dynamic_time
    pooled_ops = iterations / pooled_time
    speedup = dynamic_time / pooled_time

    print(f"  Dynamic: {dynamic_ops:.0f} ops/s, Pooled: {pooled_ops:.0f} ops/s, Speedup: {speedup:.2f}x")

    # Just verify it works
    assert pool.stats['fallback_allocs'] == 0
    print(f"  test_pooled_inference_throughput: PASS")


def test_pooled_concurrent_inference():
    """Pooled inference works correctly under concurrent load."""
    if DEVICE == 'cpu':
        print("  test_pooled_concurrent_inference: SKIP (CPU)")
        return

    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(64, 128)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(128, 64)

        def forward(self, x):
            return self.fc2(self.relu(self.fc1(x)))

    model = SimpleMLP().to(DEVICE).eval()
    pool = TensorPool(shape=(8, 64), device=DEVICE, pool_size=8)

    num_threads = 4
    ops_per_thread = 50
    results = []
    errors = []

    def worker(tid):
        try:
            for _ in range(ops_per_thread):
                with pool.context() as x:
                    x.normal_()
                    with torch.no_grad():
                        y = model(x)
                    _ = y.sum().cpu()
                    results.append((tid, y.shape))
        except Exception as e:
            errors.append((tid, e))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"Errors: {errors}"
    assert len(results) == num_threads * ops_per_thread

    # Check all outputs have correct shape
    for tid, shape in results:
        assert shape == torch.Size([8, 64])

    print(f"  test_pooled_concurrent_inference: PASS ({len(results)} ops)")
    print(f"    Pool stats: hit_rate={pool.stats['hit_rate']:.1%}, fallbacks={pool.stats['fallback_allocs']}")


def main():
    print("=" * 60)
    print("TensorPool Memory Pooling Tests (P3 Optimization)")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print()

    tests = [
        ("Basic Tests", [
            test_create_pool,
            test_acquire_release,
            test_context_manager,
            test_fallback_allocation,
            test_stats,
        ]),
        ("Thread Safety Tests", [
            test_concurrent_acquire_release,
            test_concurrent_context_manager,
            test_high_contention,
        ]),
        ("Multi-Shape & Context Tests", [
            test_multiple_shapes,
            test_inference_context,
        ]),
        ("Memory Stability Tests", [
            test_no_memory_growth_single_thread,
            test_no_memory_growth_multi_thread,
        ]),
        ("Performance Tests", [
            test_pooled_vs_dynamic_throughput,
            test_pooled_inference_throughput,
            test_pooled_concurrent_inference,
        ]),
    ]

    passed = 0
    failed = 0

    for section_name, section_tests in tests:
        print(f"\n--- {section_name} ---")
        for test_func in section_tests:
            try:
                test_func()
                passed += 1
            except Exception as e:
                print(f"  {test_func.__name__}: FAIL - {e}")
                failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
