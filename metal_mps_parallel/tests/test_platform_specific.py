#!/usr/bin/env python3
"""
P5: Platform-Specific Tests

Tests for platform-specific behavior differences across Apple Silicon generations.
These tests help identify chip-specific issues that might not manifest on all platforms.

Categories:
1. Dynamic Caching interaction tests (M3+)
2. High-core-count contention tests
3. Memory bandwidth sensitivity tests
4. Timing-dependent race detection

Exit codes:
    0: All tests passed
    1: Test failure (but not skip)
"""

import sys
import time
import threading
import gc
import json
from dataclasses import dataclass
from typing import List, Optional, Tuple, Callable

# Import platform utilities
sys.path.insert(0, '.')
from platform_utils import (
    get_platform_info, get_chip_generation, has_dynamic_caching,
    get_gpu_core_count, requires_dynamic_caching, requires_gpu_cores,
    PlatformInfo, print_platform_info
)

try:
    import torch
    HAS_TORCH = torch.backends.mps.is_available()
except ImportError:
    HAS_TORCH = False


@dataclass
class TestResult:
    name: str
    passed: bool
    skipped: bool
    message: str
    duration_ms: float
    platform: str


class PlatformSpecificTests:
    """Platform-specific test suite for MPS parallel inference."""

    def __init__(self):
        self.results: List[TestResult] = []
        self.platform = get_platform_info()

    def run_test(self, name: str, test_func: Callable, skip_condition: Optional[Tuple[bool, str]] = None):
        """Run a single test and record result."""
        platform_str = f"M{self.platform.chip_generation}" if self.platform.chip_generation else "Unknown"

        if skip_condition and skip_condition[0]:
            self.results.append(TestResult(name, False, True, skip_condition[1], 0.0, platform_str))
            return True

        start = time.perf_counter()
        try:
            test_func()
            duration_ms = (time.perf_counter() - start) * 1000
            self.results.append(TestResult(name, True, False, "PASS", duration_ms, platform_str))
            return True
        except AssertionError as e:
            duration_ms = (time.perf_counter() - start) * 1000
            self.results.append(TestResult(name, False, False, str(e), duration_ms, platform_str))
            return False
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            self.results.append(TestResult(name, False, False, f"Exception: {e}", duration_ms, platform_str))
            return False

    # ==================== Dynamic Caching Tests (M3+) ====================

    def test_dynamic_caching_memory_patterns(self):
        """
        Test that Dynamic Caching (M3+) doesn't affect parallel inference correctness.

        Dynamic Caching dynamically allocates local memory per GPU thread,
        which could potentially interact with our synchronization primitives.
        """
        if not HAS_TORCH:
            raise AssertionError("PyTorch not available")

        # Create varying workload sizes to exercise dynamic memory allocation
        sizes = [(128, 128), (256, 256), (512, 512), (1024, 1024)]
        errors = []
        results = []
        lock = threading.Lock()

        def varied_workload(tid, sizes, results, errors, lock):
            for size in sizes:
                try:
                    x = torch.randn(size[0], size[1], device='mps')
                    y = torch.randn(size[0], size[1], device='mps')
                    z = torch.matmul(x, y)
                    # Verify result shape
                    assert z.shape == (size[0], size[1]), f"Shape mismatch: {z.shape}"
                    torch.mps.synchronize()
                    with lock:
                        results.append((tid, size))
                except Exception as e:
                    with lock:
                        errors.append((tid, size, str(e)))

        threads = [threading.Thread(target=varied_workload, args=(i, sizes, results, errors, lock))
                   for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        expected = 4 * len(sizes)
        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results) == expected, f"Only {len(results)}/{expected} operations completed"

    def test_dynamic_caching_rapid_reallocation(self):
        """
        Test rapid memory allocation/deallocation patterns.

        Dynamic Caching might behave differently under rapid allocation changes.
        """
        if not HAS_TORCH:
            raise AssertionError("PyTorch not available")

        errors = []
        results = []
        lock = threading.Lock()

        def rapid_alloc_worker(tid, iterations, results, errors, lock):
            for i in range(iterations):
                try:
                    # Rapidly allocate varying sizes
                    tensors = []
                    for size in [64, 128, 256, 512, 256, 128, 64]:
                        t = torch.randn(size, size, device='mps')
                        tensors.append(t)

                    # Do computation
                    result = tensors[0]
                    for t in tensors[1:]:
                        result = torch.matmul(result[:64, :64], t[:64, :64])

                    torch.mps.synchronize()

                    # Release
                    del tensors

                    with lock:
                        results.append((tid, i))
                except Exception as e:
                    with lock:
                        errors.append((tid, i, str(e)))

        threads = [threading.Thread(target=rapid_alloc_worker, args=(i, 20, results, errors, lock))
                   for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        expected = 4 * 20
        assert len(errors) == 0, f"Errors: {errors[:5]}"
        assert len(results) >= expected * 0.9, f"Only {len(results)}/{expected} completed"

    # ==================== High-Core-Count Contention Tests ====================

    def test_high_contention_matmul(self):
        """
        Test behavior under high GPU parallelism matching core count.

        More GPU cores = more potential for true parallel execution,
        which could expose races not visible on lower-core chips.
        """
        if not HAS_TORCH:
            raise AssertionError("PyTorch not available")

        gpu_cores = self.platform.gpu_cores
        # Use thread count based on GPU cores, capped at 16
        thread_count = min(gpu_cores // 2, 16) if gpu_cores > 0 else 4

        errors = []
        results = []
        lock = threading.Lock()

        def contention_worker(tid, iterations, results, errors, lock):
            for i in range(iterations):
                try:
                    x = torch.randn(256, 256, device='mps')
                    y = torch.matmul(x, x)
                    # Verify sum is not NaN/Inf (corruption indicator)
                    s = y.sum().item()
                    if not (-1e10 < s < 1e10):
                        raise AssertionError(f"Suspicious sum: {s}")
                    torch.mps.synchronize()
                    with lock:
                        results.append((tid, i))
                except Exception as e:
                    with lock:
                        errors.append((tid, i, str(e)))

        threads = [threading.Thread(target=contention_worker, args=(i, 50, results, errors, lock))
                   for i in range(thread_count)]

        start = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        elapsed = time.time() - start

        expected = thread_count * 50
        assert len(errors) == 0, f"Errors: {errors[:5]}"
        assert len(results) >= expected * 0.95, f"Only {len(results)}/{expected} completed"

    def test_maximum_concurrent_streams(self):
        """
        Test maximum concurrent stream utilization.

        Different GPU core counts may handle stream saturation differently.
        """
        if not HAS_TORCH:
            raise AssertionError("PyTorch not available")

        # Use 16 threads to stress the stream pool
        thread_count = 16
        iterations = 30

        errors = []
        results = []
        lock = threading.Lock()

        def stream_stress_worker(tid, iterations, results, errors, lock):
            for i in range(iterations):
                try:
                    # Create work that keeps stream busy
                    x = torch.randn(512, 512, device='mps')
                    for _ in range(3):
                        x = torch.matmul(x, x.T)
                    torch.mps.synchronize()
                    with lock:
                        results.append((tid, i))
                except Exception as e:
                    with lock:
                        errors.append((tid, i, str(e)))

        threads = [threading.Thread(target=stream_stress_worker, args=(i, iterations, results, errors, lock))
                   for i in range(thread_count)]

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)

        expected = thread_count * iterations
        assert len(errors) == 0, f"Errors: {errors[:5]}"
        assert len(results) >= expected * 0.90, f"Only {len(results)}/{expected} completed"

    # ==================== Memory Bandwidth Sensitivity Tests ====================

    def test_bandwidth_sensitivity_race(self):
        """
        Test for timing-dependent races that might manifest differently
        based on memory bandwidth.

        Faster bandwidth = smaller race windows = races might not trigger.
        Slower bandwidth = larger race windows = races more likely.
        """
        if not HAS_TORCH:
            raise AssertionError("PyTorch not available")

        # Run many iterations to catch timing-dependent issues
        iterations = 100
        errors = []
        results = []
        lock = threading.Lock()

        def timing_sensitive_worker(tid, iterations, results, errors, lock):
            for i in range(iterations):
                try:
                    # Operations that could have timing-dependent behavior
                    a = torch.randn(128, 128, device='mps')
                    b = torch.randn(128, 128, device='mps')

                    # Interleaved operations
                    c = torch.matmul(a, b)
                    d = torch.matmul(b, a)

                    # Cross-result operation
                    e = c + d

                    torch.mps.synchronize()

                    # Verify mathematical property: c + d should be valid
                    s = e.sum().item()
                    if not (-1e8 < s < 1e8):
                        raise AssertionError(f"Computation error: sum={s}")

                    with lock:
                        results.append((tid, i))
                except Exception as e:
                    with lock:
                        errors.append((tid, i, str(e)))

        # Use 8 threads for good coverage
        threads = [threading.Thread(target=timing_sensitive_worker, args=(i, iterations, results, errors, lock))
                   for i in range(8)]

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        expected = 8 * iterations
        assert len(errors) == 0, f"Errors: {errors[:5]}"
        assert len(results) >= expected * 0.95, f"Only {len(results)}/{expected} completed"

    def test_shared_tensor_cross_thread(self):
        """
        Test behavior when tensors created in one thread are used in another.

        Memory bandwidth affects how quickly data becomes visible across threads.
        """
        if not HAS_TORCH:
            raise AssertionError("PyTorch not available")

        # Shared storage for cross-thread tensor access
        shared = {}
        shared_lock = threading.Lock()
        errors = []
        results = []
        results_lock = threading.Lock()

        def producer(iteration):
            """Create tensor and store in shared dict."""
            try:
                x = torch.randn(256, 256, device='mps')
                torch.mps.synchronize()
                with shared_lock:
                    shared[iteration] = x
                return True
            except Exception as e:
                with results_lock:
                    errors.append(('producer', iteration, str(e)))
                return False

        def consumer(iteration):
            """Read tensor from shared dict and compute."""
            try:
                # Wait for tensor to be available
                tensor = None
                for _ in range(100):  # 100ms max wait
                    with shared_lock:
                        tensor = shared.get(iteration)
                    if tensor is not None:
                        break
                    time.sleep(0.001)

                if tensor is None:
                    raise AssertionError(f"Tensor {iteration} never became available")

                # Compute on shared tensor
                y = torch.matmul(tensor, tensor)
                torch.mps.synchronize()
                s = y.sum().item()

                with results_lock:
                    results.append(('consumer', iteration, s))
                return True
            except Exception as e:
                with results_lock:
                    errors.append(('consumer', iteration, str(e)))
                return False

        # Run producer-consumer pairs
        for i in range(20):
            p = threading.Thread(target=producer, args=(i,))
            c = threading.Thread(target=consumer, args=(i,))
            p.start()
            c.start()
            p.join(timeout=5)
            c.join(timeout=5)

        assert len(errors) == 0, f"Errors: {errors[:5]}"
        assert len(results) >= 18, f"Only {len(results)}/20 producer-consumer pairs succeeded"

    # ==================== Chip Generation Comparison Tests ====================

    def test_consistent_numerical_results(self):
        """
        Verify numerical results are consistent regardless of chip.

        Different GPU architectures might have subtle numerical differences.
        We use seeded random for reproducibility.
        """
        if not HAS_TORCH:
            raise AssertionError("PyTorch not available")

        torch.manual_seed(42)

        # Create deterministic inputs
        x = torch.randn(100, 100, device='mps')

        # Compute result
        y = torch.matmul(x, x)
        torch.mps.synchronize()

        # Check checksum - should be similar across chips (within tolerance)
        checksum = y.sum().item()

        # Note: Exact value may vary slightly across architectures
        # This test mainly verifies no catastrophic failures
        assert not torch.isnan(y).any(), "NaN detected in result"
        assert not torch.isinf(y).any(), "Inf detected in result"
        assert abs(checksum) < 1e8, f"Checksum out of expected range: {checksum}"

    def test_parallel_correctness_all_sizes(self):
        """
        Comprehensive parallel correctness test across various tensor sizes.

        Different chips may handle different sizes with different efficiency.
        """
        if not HAS_TORCH:
            raise AssertionError("PyTorch not available")

        sizes = [(32, 32), (64, 64), (128, 128), (256, 256), (512, 512)]
        errors = []
        results = []
        lock = threading.Lock()

        def size_test_worker(tid, sizes, results, errors, lock):
            for size in sizes:
                try:
                    # Set seed for reproducibility
                    torch.manual_seed(tid * 1000 + size[0])

                    x = torch.randn(size[0], size[1], device='mps')
                    y = torch.matmul(x, x)
                    torch.mps.synchronize()

                    # Verify basic sanity
                    assert y.shape == size, f"Shape mismatch"
                    assert not torch.isnan(y).any(), "NaN in result"
                    assert not torch.isinf(y).any(), "Inf in result"

                    with lock:
                        results.append((tid, size))
                except Exception as e:
                    with lock:
                        errors.append((tid, size, str(e)))

        threads = [threading.Thread(target=size_test_worker, args=(i, sizes, results, errors, lock))
                   for i in range(4)]

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        expected = 4 * len(sizes)
        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results) == expected, f"Only {len(results)}/{expected} completed"

    def run_all(self):
        """Run all platform-specific tests."""
        print("=" * 70)
        print("  Platform-Specific Test Suite (P5)")
        print("=" * 70)
        print()
        print_platform_info()
        print()
        print("=" * 70)

        # Define tests with skip conditions
        tests = [
            # Dynamic Caching tests
            ("DC: memory_patterns", self.test_dynamic_caching_memory_patterns, None),
            ("DC: rapid_reallocation", self.test_dynamic_caching_rapid_reallocation, None),

            # High contention tests
            ("Contention: high_matmul", self.test_high_contention_matmul, None),
            ("Contention: max_streams", self.test_maximum_concurrent_streams, None),

            # Bandwidth sensitivity tests
            ("Bandwidth: timing_race", self.test_bandwidth_sensitivity_race, None),
            ("Bandwidth: cross_thread", self.test_shared_tensor_cross_thread, None),

            # Cross-chip comparison tests
            ("CrossChip: numerical", self.test_consistent_numerical_results, None),
            ("CrossChip: all_sizes", self.test_parallel_correctness_all_sizes, None),
        ]

        # Run tests
        for name, test_func, skip_cond in tests:
            passed = self.run_test(name, test_func, skip_cond)
            result = self.results[-1]

            if result.skipped:
                status = "SKIP"
            elif result.passed:
                status = "PASS"
            else:
                status = "FAIL"

            print(f"[{status}] {name} ({result.duration_ms:.1f}ms)")

        # Summary
        print()
        print("=" * 70)
        print("  Test Summary")
        print("=" * 70)

        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed and not r.skipped)
        skipped = sum(1 for r in self.results if r.skipped)

        print(f"Platform: M{self.platform.chip_generation} ({self.platform.chip_name})")
        print(f"GPU Cores: {self.platform.gpu_cores}")
        print(f"Dynamic Caching: {'Yes' if self.platform.has_dynamic_caching else 'No'}")
        print()
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Skipped: {skipped}")

        if failed > 0:
            print()
            print("Failed tests:")
            for r in self.results:
                if not r.passed and not r.skipped:
                    print(f"  - {r.name}: {r.message}")

        print("=" * 70)

        return failed == 0


def main():
    if not HAS_TORCH:
        print("ERROR: PyTorch with MPS not available")
        return 2

    tests = PlatformSpecificTests()
    success = tests.run_all()

    # Output JSON for cross-platform comparison
    results_json = {
        "platform": tests.platform.to_dict(),
        "total": len(tests.results),
        "passed": sum(1 for r in tests.results if r.passed),
        "failed": sum(1 for r in tests.results if not r.passed and not r.skipped),
        "skipped": sum(1 for r in tests.results if r.skipped),
        "results": [
            {
                "name": r.name,
                "passed": r.passed,
                "skipped": r.skipped,
                "message": r.message,
                "duration_ms": r.duration_ms,
                "platform": r.platform
            }
            for r in tests.results
        ]
    }

    print()
    print("JSON Output:")
    print(json.dumps(results_json, indent=2))

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
