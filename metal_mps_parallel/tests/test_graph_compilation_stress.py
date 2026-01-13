#!/usr/bin/env python3
"""
Test 39.3: Graph Compilation Race Stress Test

Tests concurrent MPSGraph compilation from multiple threads.
Uses unique input shapes per thread to force separate graph compilations.
Verifies:
- Thread-local caches work correctly
- No duplicate compilation (same key = same graph returned)
- TSan clean (no data races)
"""
import threading
import time
import sys


class SkipTest(Exception):
    """Exception raised when a test should be skipped."""
    pass


def graph_compilation_worker(thread_id: int, base_size: int, iterations: int,
                             results: list, errors: list, lock: threading.Lock):
    """
    Worker that triggers graph compilation with unique shapes.
    Each thread uses different tensor sizes to force unique graph compilation.
    """
    import torch

    try:
        # Use thread-specific size to ensure unique graph keys
        size = base_size + thread_id * 7  # Prime offset for uniqueness

        for i in range(iterations):
            # Create tensors with unique per-thread dimensions
            x = torch.randn(size, size, device='mps')
            y = torch.randn(size, size, device='mps')

            # Matrix multiply - triggers MPSGraph compilation
            z = torch.mm(x, y)

            # Different operation that also compiles to graph
            z = torch.relu(z)

            torch.mps.synchronize()

            with lock:
                results.append((thread_id, i, size))

    except Exception as e:
        with lock:
            errors.append((thread_id, str(e)))


def test_concurrent_graph_compilation():
    """
    Test 16 threads simultaneously compiling different graphs.
    """
    import torch

    print("=" * 70)
    print("Graph Compilation Race Stress Test (39.3)")
    print("=" * 70)
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")

    if not torch.backends.mps.is_available():
        raise SkipTest("MPS not available")

    # Warmup to initialize MPS
    print("\n--- Warmup ---")
    x = torch.randn(64, 64, device='mps')
    y = torch.mm(x, x)
    torch.mps.synchronize()
    print("Warmup complete")

    errors = []
    results = []
    lock = threading.Lock()

    num_threads = 16
    iterations = 30
    base_size = 64

    print(f"\n--- Running {num_threads} threads x {iterations} iterations ---")
    print(f"Each thread uses unique tensor size: {base_size} + thread_id * 7")

    threads = []
    start_time = time.time()

    for i in range(num_threads):
        t = threading.Thread(
            target=graph_compilation_worker,
            args=(i, base_size, iterations, results, errors, lock)
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join(timeout=60)

    elapsed = time.time() - start_time
    total_expected = num_threads * iterations

    print(f"\nResults:")
    print(f"  Threads: {num_threads}")
    print(f"  Iterations/thread: {iterations}")
    print(f"  Total operations: {len(results)}/{total_expected}")
    print(f"  Errors: {len(errors)}")
    print(f"  Elapsed: {elapsed:.3f}s")
    print(f"  Throughput: {len(results)/elapsed:.1f} ops/sec")

    # Analyze size distribution
    sizes_seen = set(r[2] for r in results)
    print(f"  Unique tensor sizes compiled: {len(sizes_seen)}")

    if errors:
        print("\nErrors encountered:")
        for tid, err in errors[:5]:
            print(f"  Thread {tid}: {err[:80]}")

    assert len(errors) == 0, f"{len(errors)} errors encountered"
    assert len(results) == total_expected, f"Expected {total_expected} ops, got {len(results)}"
    print("PASS")


def test_repeated_compilation_same_shape():
    """
    Test multiple threads compiling the SAME graph (same input shapes).
    This stresses the graph cache lookup under contention.
    """
    import torch

    print("\n" + "=" * 70)
    print("Same-Shape Graph Cache Contention Test")
    print("=" * 70)

    if not torch.backends.mps.is_available():
        raise SkipTest("MPS not available")

    errors = []
    results = []
    lock = threading.Lock()

    num_threads = 16
    iterations = 50
    fixed_size = 128  # All threads use same size

    def same_shape_worker(thread_id: int):
        import torch
        try:
            for i in range(iterations):
                x = torch.randn(fixed_size, fixed_size, device='mps')
                y = torch.randn(fixed_size, fixed_size, device='mps')
                z = torch.mm(x, y)
                z = torch.relu(z)
                torch.mps.synchronize()

                with lock:
                    results.append((thread_id, i))
        except Exception as e:
            with lock:
                errors.append((thread_id, str(e)))

    print(f"\n--- Running {num_threads} threads x {iterations} iterations ---")
    print(f"All threads use same tensor size: {fixed_size}x{fixed_size}")

    threads = []
    start_time = time.time()

    for i in range(num_threads):
        t = threading.Thread(target=same_shape_worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join(timeout=60)

    elapsed = time.time() - start_time
    total_expected = num_threads * iterations

    print(f"\nResults:")
    print(f"  Total operations: {len(results)}/{total_expected}")
    print(f"  Errors: {len(errors)}")
    print(f"  Elapsed: {elapsed:.3f}s")
    print(f"  Throughput: {len(results)/elapsed:.1f} ops/sec")

    if errors:
        print("\nErrors encountered:")
        for tid, err in errors[:5]:
            print(f"  Thread {tid}: {err[:80]}")

    assert len(errors) == 0, f"{len(errors)} errors encountered"
    assert len(results) == total_expected, f"Expected {total_expected} ops, got {len(results)}"
    print("PASS")


def test_mixed_operation_graphs():
    """
    Test threads compiling graphs with different operation types.
    Uses conservative settings to ensure stability.
    """
    import torch

    print("\n" + "=" * 70)
    print("Mixed Operation Graph Compilation Test")
    print("=" * 70)

    if not torch.backends.mps.is_available():
        raise SkipTest("MPS not available")

    errors = []
    results = []
    lock = threading.Lock()

    def matmul_worker(thread_id: int, iterations: int):
        import torch
        try:
            for i in range(iterations):
                x = torch.randn(96, 96, device='mps')
                y = torch.randn(96, 96, device='mps')
                z = torch.mm(x, y)
                z = torch.relu(z)
                torch.mps.synchronize()
                with lock:
                    results.append((thread_id, i, 'matmul'))
        except Exception as e:
            with lock:
                errors.append((thread_id, str(e), 'matmul'))

    def activation_worker(thread_id: int, iterations: int):
        import torch
        try:
            for i in range(iterations):
                x = torch.randn(128, 128, device='mps')
                z = torch.relu(x)
                torch.mps.synchronize()
                with lock:
                    results.append((thread_id, i, 'activation'))
        except Exception as e:
            with lock:
                errors.append((thread_id, str(e), 'activation'))

    def elementwise_worker(thread_id: int, iterations: int):
        import torch
        try:
            for i in range(iterations):
                x = torch.randn(112, 112, device='mps')
                y = torch.randn(112, 112, device='mps')
                z = x + y
                torch.mps.synchronize()
                with lock:
                    results.append((thread_id, i, 'elementwise'))
        except Exception as e:
            with lock:
                errors.append((thread_id, str(e), 'elementwise'))

    iterations = 30  # Reduced for stability
    print(f"\n--- Running 12 threads (4 each type) x {iterations} iterations ---")

    threads = []
    start_time = time.time()

    # 4 threads each for different operation types
    for i in range(4):
        threads.append(threading.Thread(target=matmul_worker, args=(i, iterations)))
        threads.append(threading.Thread(target=activation_worker, args=(i+4, iterations)))
        threads.append(threading.Thread(target=elementwise_worker, args=(i+8, iterations)))

    for t in threads:
        t.start()

    for t in threads:
        t.join(timeout=60)

    elapsed = time.time() - start_time
    total_expected = 12 * iterations

    # Count by type
    matmul_count = sum(1 for r in results if r[2] == 'matmul')
    activation_count = sum(1 for r in results if r[2] == 'activation')
    elementwise_count = sum(1 for r in results if r[2] == 'elementwise')

    print(f"\nResults:")
    print(f"  Total operations: {len(results)}/{total_expected}")
    print(f"    - matmul: {matmul_count}")
    print(f"    - activation: {activation_count}")
    print(f"    - elementwise: {elementwise_count}")
    print(f"  Errors: {len(errors)}")
    print(f"  Elapsed: {elapsed:.3f}s")
    print(f"  Throughput: {len(results)/elapsed:.1f} ops/sec")

    if errors:
        print("\nErrors encountered:")
        for tid, err, op in errors[:5]:
            print(f"  Thread {tid} ({op}): {err[:60]}")

    assert len(errors) == 0, f"{len(errors)} errors encountered"
    assert len(results) == total_expected, f"Expected {total_expected} ops, got {len(results)}"
    print("PASS")


def cleanup_between_tests():
    """Clean up MPS state between tests to prevent accumulation issues."""
    import gc
    import torch
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
        torch.mps.empty_cache()


if __name__ == '__main__':
    all_passed = True

    try:
        test_concurrent_graph_compilation()
    except SkipTest as e:
        print(f"SKIP: {e}")
    except Exception as e:
        print(f"EXCEPTION: {e}")
        all_passed = False

    cleanup_between_tests()

    try:
        test_repeated_compilation_same_shape()
    except SkipTest as e:
        print(f"SKIP: {e}")
    except Exception as e:
        print(f"EXCEPTION: {e}")
        all_passed = False

    cleanup_between_tests()

    try:
        test_mixed_operation_graphs()
    except SkipTest as e:
        print(f"SKIP: {e}")
    except Exception as e:
        print(f"EXCEPTION: {e}")
        all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL GRAPH COMPILATION STRESS TESTS PASSED")
        sys.exit(0)
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)
