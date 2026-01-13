#!/usr/bin/env python3
"""
R6: Backwards Compatibility Test Suite

Verifies that MPS parallel inference patch maintains backwards compatibility:
1. Single-threaded operations work identically
2. Basic MPS operations produce correct results
3. No performance regression in single-threaded code
4. Synchronization APIs work correctly

Exit codes:
    0: All tests passed
    1: Test failure
"""

import sys
import time
import threading
import subprocess
import json
from dataclasses import dataclass
from typing import List, Tuple, Optional

from platform_utils import get_platform_info, print_platform_info

try:
    import torch
    if not torch.backends.mps.is_available():
        print("ERROR: MPS backend not available", file=sys.stderr)
        sys.exit(2)
except ImportError:
    print("ERROR: PyTorch not found", file=sys.stderr)
    sys.exit(2)


@dataclass
class TestResult:
    name: str
    passed: bool
    message: str
    duration_ms: float


def time_operation(func, *args, **kwargs):
    """Time a function call in milliseconds."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    duration_ms = (time.perf_counter() - start) * 1000
    return result, duration_ms


class BackwardsCompatibilityTests:
    """Test suite for backwards compatibility."""

    def __init__(self):
        self.results: List[TestResult] = []

    def run_test(self, name: str, test_func):
        """Run a single test and record result."""
        start = time.perf_counter()
        try:
            test_func()
            duration_ms = (time.perf_counter() - start) * 1000
            self.results.append(TestResult(name, True, "PASS", duration_ms))
            return True
        except AssertionError as e:
            duration_ms = (time.perf_counter() - start) * 1000
            self.results.append(TestResult(name, False, str(e), duration_ms))
            return False
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            self.results.append(TestResult(name, False, f"Exception: {e}", duration_ms))
            return False

    # ==================== Basic Operations ====================

    def test_zeros_and_ones(self):
        """Test torch.zeros and torch.ones on MPS."""
        z = torch.zeros(10, 10, device='mps')
        assert z.sum().item() == 0, "zeros should sum to 0"

        o = torch.ones(10, 10, device='mps')
        assert o.sum().item() == 100, "10x10 ones should sum to 100"

    def test_tensor_creation(self):
        """Test various tensor creation methods."""
        # randn
        x = torch.randn(100, 100, device='mps')
        assert x.shape == (100, 100), "randn shape mismatch"

        # arange
        a = torch.arange(100, device='mps')
        assert a.sum().item() == 4950, "arange(100) should sum to 4950"

        # linspace
        l = torch.linspace(0, 10, 11, device='mps')
        assert len(l) == 11, "linspace should create 11 elements"

    def test_tensor_operations(self):
        """Test basic tensor operations."""
        x = torch.randn(100, 100, device='mps')
        y = torch.randn(100, 100, device='mps')

        # Addition
        z = x + y
        assert z.shape == (100, 100), "addition shape mismatch"

        # Multiplication
        m = x * y
        assert m.shape == (100, 100), "multiplication shape mismatch"

        # Matmul
        mm = torch.matmul(x, y)
        assert mm.shape == (100, 100), "matmul shape mismatch"

    def test_reductions(self):
        """Test reduction operations."""
        x = torch.ones(100, 100, device='mps')

        assert x.sum().item() == 10000, "sum should be 10000"
        assert x.mean().item() == 1.0, "mean should be 1.0"
        assert x.max().item() == 1.0, "max should be 1.0"
        assert x.min().item() == 1.0, "min should be 1.0"

    # ==================== Neural Network Operations ====================

    def test_linear_layer(self):
        """Test nn.Linear on MPS."""
        layer = torch.nn.Linear(256, 128).to('mps')
        x = torch.randn(32, 256, device='mps')
        y = layer(x)
        assert y.shape == (32, 128), f"Expected (32, 128), got {y.shape}"

    def test_conv2d_layer(self):
        """Test nn.Conv2d on MPS."""
        layer = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1).to('mps')
        x = torch.randn(1, 3, 32, 32, device='mps')
        y = layer(x)
        assert y.shape == (1, 64, 32, 32), f"Expected (1, 64, 32, 32), got {y.shape}"

    def test_relu(self):
        """Test ReLU activation on MPS."""
        x = torch.randn(100, device='mps')
        y = torch.nn.functional.relu(x)
        assert (y >= 0).all(), "ReLU should produce non-negative values"

    def test_softmax(self):
        """Test softmax on MPS."""
        x = torch.randn(10, 10, device='mps')
        y = torch.nn.functional.softmax(x, dim=-1)
        # Check rows sum to 1 (within tolerance)
        row_sums = y.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(10, device='mps'), atol=1e-5), \
            "softmax rows should sum to 1"

    # ==================== Synchronization ====================

    def test_synchronize(self):
        """Test torch.mps.synchronize()."""
        x = torch.randn(1000, 1000, device='mps')
        y = torch.matmul(x, x)

        # Synchronize should not raise
        torch.mps.synchronize()

        # After sync, result should be accessible
        _ = y.cpu()

    def test_empty_cache(self):
        """Test torch.mps.empty_cache()."""
        # Create some tensors
        tensors = [torch.randn(100, 100, device='mps') for _ in range(10)]

        # Clear tensors
        del tensors

        # empty_cache should not raise
        torch.mps.empty_cache()

    def test_current_allocated_memory(self):
        """Test torch.mps.current_allocated_memory()."""
        # Should return a non-negative integer
        mem = torch.mps.current_allocated_memory()
        assert isinstance(mem, int), "current_allocated_memory should return int"
        assert mem >= 0, "allocated memory should be non-negative"

    # ==================== Device Transfer ====================

    def test_cpu_to_mps(self):
        """Test CPU to MPS transfer."""
        x_cpu = torch.randn(100, 100)
        x_mps = x_cpu.to('mps')
        assert x_mps.device.type == 'mps', "Should be on MPS device"
        assert torch.allclose(x_cpu, x_mps.cpu(), atol=1e-6), "Values should match"

    def test_mps_to_cpu(self):
        """Test MPS to CPU transfer."""
        x_mps = torch.randn(100, 100, device='mps')
        x_cpu = x_mps.to('cpu')
        assert x_cpu.device.type == 'cpu', "Should be on CPU device"

    def test_device_properties(self):
        """Test MPS device properties."""
        device = torch.device('mps')
        assert torch.mps.is_available(), "MPS should be available"

    # ==================== Gradients ====================

    def test_backward_pass(self):
        """Test backward pass on MPS."""
        x = torch.randn(10, 10, device='mps', requires_grad=True)
        y = x.sum()
        y.backward()
        assert x.grad is not None, "Gradient should be computed"
        assert x.grad.shape == (10, 10), "Gradient shape should match input"

    def test_gradient_accumulation(self):
        """Test gradient accumulation on MPS."""
        x = torch.randn(10, device='mps', requires_grad=True)

        # First backward
        y1 = x.sum()
        y1.backward()
        grad1 = x.grad.clone()

        # Second backward (should accumulate)
        y2 = x.sum()
        y2.backward()

        # Gradient should be accumulated
        assert torch.allclose(x.grad, grad1 * 2, atol=1e-6), \
            "Gradients should accumulate"

    # ==================== Single-Thread Performance ====================

    def test_single_thread_matmul_performance(self):
        """Verify single-threaded matmul is reasonably fast."""
        x = torch.randn(512, 512, device='mps')

        # Warmup
        for _ in range(5):
            _ = torch.matmul(x, x)
        torch.mps.synchronize()

        # Time 10 iterations
        start = time.perf_counter()
        for _ in range(10):
            _ = torch.matmul(x, x)
        torch.mps.synchronize()
        elapsed = time.perf_counter() - start

        # Should complete in reasonable time (< 1 second for 10 matmuls)
        assert elapsed < 1.0, f"Single-thread matmul too slow: {elapsed:.3f}s for 10 ops"

    # ==================== Thread Safety (Compatibility) ====================

    def test_single_thread_sequential_ops(self):
        """Test that sequential operations work correctly (as before patch)."""
        results = []
        for i in range(10):
            x = torch.randn(100, 100, device='mps')
            y = torch.matmul(x, x)
            torch.mps.synchronize()
            results.append(y.sum().item())

        # All operations should complete without error
        assert len(results) == 10, "All 10 operations should complete"

    def test_multi_thread_basic(self):
        """Test basic multi-threaded operation (new capability)."""
        errors = []
        results = []
        lock = threading.Lock()

        def worker(tid):
            try:
                x = torch.randn(100, 100, device='mps')
                y = torch.matmul(x, x)
                torch.mps.synchronize()
                with lock:
                    results.append((tid, y.sum().item()))
            except Exception as e:
                with lock:
                    errors.append((tid, str(e)))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Multi-thread errors: {errors}"
        assert len(results) == 4, "All 4 threads should complete"

    def run_all(self):
        """Run all backwards compatibility tests."""
        print("=" * 60)
        print("  MPS Backwards Compatibility Test Suite (R6)")
        print("=" * 60)
        print(f"PyTorch version: {torch.__version__}")
        print(f"MPS available: {torch.backends.mps.is_available()}")
        print("=" * 60)
        print()

        # Define test methods
        tests = [
            # Basic operations
            ("Basic: zeros_and_ones", self.test_zeros_and_ones),
            ("Basic: tensor_creation", self.test_tensor_creation),
            ("Basic: tensor_operations", self.test_tensor_operations),
            ("Basic: reductions", self.test_reductions),

            # Neural network operations
            ("NN: linear_layer", self.test_linear_layer),
            ("NN: conv2d_layer", self.test_conv2d_layer),
            ("NN: relu", self.test_relu),
            ("NN: softmax", self.test_softmax),

            # Synchronization
            ("Sync: synchronize", self.test_synchronize),
            ("Sync: empty_cache", self.test_empty_cache),
            ("Sync: current_allocated_memory", self.test_current_allocated_memory),

            # Device transfer
            ("Device: cpu_to_mps", self.test_cpu_to_mps),
            ("Device: mps_to_cpu", self.test_mps_to_cpu),
            ("Device: device_properties", self.test_device_properties),

            # Gradients
            ("Grad: backward_pass", self.test_backward_pass),
            ("Grad: gradient_accumulation", self.test_gradient_accumulation),

            # Performance
            ("Perf: single_thread_matmul", self.test_single_thread_matmul_performance),

            # Thread safety
            ("Thread: single_thread_sequential", self.test_single_thread_sequential_ops),
            ("Thread: multi_thread_basic", self.test_multi_thread_basic),
        ]

        # Run tests
        for name, test_func in tests:
            passed = self.run_test(name, test_func)
            status = "PASS" if passed else "FAIL"
            print(f"[{status}] {name}")

        # Summary
        print()
        print("=" * 60)
        print("  Test Summary")
        print("=" * 60)
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")

        if failed > 0:
            print()
            print("Failed tests:")
            for r in self.results:
                if not r.passed:
                    print(f"  - {r.name}: {r.message}")

        print("=" * 60)

        return failed == 0


def main():
    print_platform_info()
    tests = BackwardsCompatibilityTests()
    success = tests.run_all()

    # Output JSON for CI
    results_json = {
        "platform": get_platform_info().to_dict(),
        "pytorch_version": torch.__version__,
        "total": len(tests.results),
        "passed": sum(1 for r in tests.results if r.passed),
        "failed": sum(1 for r in tests.results if not r.passed),
        "results": [
            {
                "name": r.name,
                "passed": r.passed,
                "message": r.message,
                "duration_ms": r.duration_ms
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
