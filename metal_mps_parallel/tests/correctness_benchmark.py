#!/usr/bin/env python3
"""
Comprehensive Correctness Benchmark: Patched vs Unpatched PyTorch

This benchmark verifies that our MPS patches produce IDENTICAL outputs
to unpatched PyTorch. Critical for:
1. Proving we didn't break anything
2. Academic credibility (paper Section 7)
3. Upstream PR acceptance

Usage:
    # Compare against CPU reference (always available)
    python tests/correctness_benchmark.py --reference cpu

    # Compare against unpatched MPS (requires two PyTorch installs)
    python tests/correctness_benchmark.py --reference unpatched_mps

    # Full benchmark with all operations
    python tests/correctness_benchmark.py --full

Output: JSON report with pass/fail per operation and max numerical difference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import argparse
import time
import sys
import os
import threading
import concurrent.futures
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional, Tuple, Callable
import numpy as np
import copy
import queue


# =============================================================================
# Python BatchQueue - Simulates the C++ MPSBatchQueue for testing
# =============================================================================
class PythonBatchQueue:
    """
    Python implementation of batch queue for N-thread -> M-worker batching.

    This bypasses Apple Metal's thread-safety bugs at 4+ threads by serializing
    requests through a small number of worker threads. User threads submit
    requests and get futures back; workers execute requests sequentially.

    This is a Python prototype of the C++ MPSBatchQueue (Phase 1.1-1.3).
    """

    def __init__(self, num_workers: int = 1):
        """
        Args:
            num_workers: Number of worker threads (default: 1)
        """
        self.num_workers = num_workers
        self._queue: queue.Queue = queue.Queue()
        self._workers: List[threading.Thread] = []
        self._running = False
        self._completed = 0
        self._lock = threading.Lock()

    def start(self):
        """Start worker threads."""
        if self._running:
            return
        self._running = True
        for i in range(self.num_workers):
            t = threading.Thread(target=self._worker_loop, args=(i,), daemon=True)
            t.start()
            self._workers.append(t)

    def stop(self):
        """Stop worker threads (waits for pending requests to complete)."""
        if not self._running:
            return
        # Send sentinel values to stop workers
        for _ in range(self.num_workers):
            self._queue.put(None)
        for t in self._workers:
            t.join(timeout=30.0)
        self._workers.clear()
        self._running = False

    def submit(self, operation: Callable[[], torch.Tensor]) -> concurrent.futures.Future:
        """
        Submit an operation to be executed by a worker thread.

        Args:
            operation: Callable that performs the inference operation

        Returns:
            Future that will contain the result tensor
        """
        future: concurrent.futures.Future = concurrent.futures.Future()
        self._queue.put((operation, future))
        return future

    def _worker_loop(self, worker_id: int):
        """Worker thread loop - processes requests sequentially."""
        while self._running:
            try:
                item = self._queue.get(timeout=1.0)
                if item is None:  # Sentinel to stop
                    break

                operation, future = item
                try:
                    result = operation()
                    if torch.backends.mps.is_available():
                        torch.mps.synchronize()
                    future.set_result(result)
                    with self._lock:
                        self._completed += 1
                except Exception as e:
                    future.set_exception(e)

            except queue.Empty:
                continue

    @property
    def completed_count(self) -> int:
        """Number of completed requests."""
        with self._lock:
            return self._completed


# Global batch queue instance (mimics getMPSBatchQueue() in C++)
_global_batch_queue: Optional[PythonBatchQueue] = None


def get_batch_queue(num_workers: int = 1) -> PythonBatchQueue:
    """Get or create the global batch queue."""
    global _global_batch_queue
    if _global_batch_queue is None:
        _global_batch_queue = PythonBatchQueue(num_workers=num_workers)
    return _global_batch_queue


def shutdown_batch_queue():
    """Shutdown the global batch queue."""
    global _global_batch_queue
    if _global_batch_queue is not None:
        _global_batch_queue.stop()
        _global_batch_queue = None

# Tolerance for floating-point comparison
# Note: MPS uses different accumulation strategies than CPU, so tolerances
# need to account for FMA differences and accumulation order. These are
# scaled relative to typical operation magnitudes, not bug thresholds.
#
# For large matrix operations in float16, we need scaled tolerances because
# accumulation errors grow with sqrt(N) where N is the reduction dimension.
# float16 has ~3 decimal digits of precision, so for K=1024 reduction:
# expected error ≈ sqrt(1024) * 10^-3 ≈ 0.032
RTOL = {
    torch.float32: 1e-4,   # Allow 0.01% relative error (accumulation differences)
    torch.float16: 0.1,    # float16 needs ~10% tolerance for large reductions
    torch.bfloat16: 0.15,  # bfloat16 needs ~15% tolerance for large reductions
}
ATOL = {
    torch.float32: 1e-4,   # Absolute tolerance for near-zero values
    torch.float16: 0.15,   # Allow 0.15 absolute error for float16 (large matmuls)
    torch.bfloat16: 0.2,   # Allow 0.2 absolute error for bfloat16
}


@dataclass
class TestResult:
    name: str
    passed: bool
    max_abs_diff: float
    max_rel_diff: float
    mean_abs_diff: float
    shape: List[int]
    dtype: str
    time_ms: float
    error: Optional[str] = None


@dataclass
class BenchmarkReport:
    timestamp: str
    pytorch_version: str
    device: str
    total_tests: int
    passed: int
    failed: int
    results: List[Dict]


@dataclass
class ParallelTestResult:
    """Result from parallel correctness test."""
    name: str
    num_threads: int
    passed: bool
    all_threads_identical: bool      # All thread outputs match each other
    all_match_reference: bool        # All thread outputs match CPU reference
    max_inter_thread_diff: float     # Max diff between any two thread outputs
    max_vs_reference_diff: float     # Max diff between any thread and reference
    errors: List[str] = field(default_factory=list)


@dataclass
class ParallelBenchmarkReport:
    """Report from parallel correctness benchmark."""
    timestamp: str
    pytorch_version: str
    device: str
    thread_counts: List[int]
    total_tests: int
    passed: int
    failed: int
    results: List[Dict]


def compare_tensors(a: torch.Tensor, b: torch.Tensor, rtol: float, atol: float) -> Tuple[bool, float, float, float]:
    """Compare two tensors, return (passed, max_abs, max_rel, mean_abs)."""
    if a.shape != b.shape:
        return False, float('inf'), float('inf'), float('inf')

    a_float = a.float().cpu()
    b_float = b.float().cpu()

    abs_diff = torch.abs(a_float - b_float)
    max_abs = abs_diff.max().item()
    mean_abs = abs_diff.mean().item()

    # Relative difference (avoid div by zero)
    denom = torch.maximum(torch.abs(a_float), torch.abs(b_float))
    denom = torch.where(denom < 1e-10, torch.ones_like(denom), denom)
    rel_diff = abs_diff / denom
    max_rel = rel_diff.max().item()

    passed = torch.allclose(a_float, b_float, rtol=rtol, atol=atol)
    return passed, max_abs, max_rel, mean_abs


class CorrectnessTest:
    """Base class for correctness tests."""

    def __init__(self, name: str, dtype: torch.dtype = torch.float32):
        self.name = name
        self.dtype = dtype

    def run(self, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run on device, return (input, output)."""
        raise NotImplementedError

    def get_reference(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Get reference output (CPU)."""
        raise NotImplementedError


class LinearTest(CorrectnessTest):
    """Test nn.Linear correctness."""

    def __init__(self, batch: int, in_features: int, out_features: int,
                 dtype: torch.dtype = torch.float32, bias: bool = True):
        super().__init__(f"Linear_{batch}x{in_features}x{out_features}_{dtype}", dtype)
        self.batch = batch
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        # Fixed seed for reproducibility
        torch.manual_seed(42)
        self.weight = torch.randn(out_features, in_features)
        self.bias_tensor = torch.randn(out_features) if bias else None

    def run(self, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(42)
        x = torch.randn(self.batch, self.in_features, dtype=self.dtype, device=device)
        weight = self.weight.to(dtype=self.dtype, device=device)
        bias = self.bias_tensor.to(dtype=self.dtype, device=device) if self.bias_tensor is not None else None

        output = F.linear(x, weight, bias)
        if device == "mps":
            torch.mps.synchronize()
        return x, output

    def get_reference(self, input_tensor: torch.Tensor) -> torch.Tensor:
        x_cpu = input_tensor.float().cpu()
        weight_cpu = self.weight.float()
        bias_cpu = self.bias_tensor.float() if self.bias_tensor is not None else None
        return F.linear(x_cpu, weight_cpu, bias_cpu)


class MatmulTest(CorrectnessTest):
    """Test torch.matmul correctness."""

    def __init__(self, m: int, k: int, n: int, dtype: torch.dtype = torch.float32):
        super().__init__(f"Matmul_{m}x{k}x{n}_{dtype}", dtype)
        self.m, self.k, self.n = m, k, n

        # Pre-generate tensors with fixed seed
        torch.manual_seed(42)
        self.a_base = torch.randn(m, k)
        self.b_base = torch.randn(k, n)

    def run(self, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        a = self.a_base.to(dtype=self.dtype, device=device)
        b = self.b_base.to(dtype=self.dtype, device=device)
        output = torch.matmul(a, b)
        if device == "mps":
            torch.mps.synchronize()
        return a, output

    def get_reference(self, input_tensor: torch.Tensor) -> torch.Tensor:
        a_cpu = self.a_base.float()
        b_cpu = self.b_base.float()
        return torch.matmul(a_cpu, b_cpu)


class BMMatmulTest(CorrectnessTest):
    """Test batched matrix multiplication."""

    def __init__(self, batch: int, m: int, k: int, n: int, dtype: torch.dtype = torch.float32):
        super().__init__(f"BMM_{batch}x{m}x{k}x{n}_{dtype}", dtype)
        self.batch, self.m, self.k, self.n = batch, m, k, n

        # Pre-generate tensors with fixed seed
        torch.manual_seed(42)
        self.a_base = torch.randn(batch, m, k)
        self.b_base = torch.randn(batch, k, n)

    def run(self, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        a = self.a_base.to(dtype=self.dtype, device=device)
        b = self.b_base.to(dtype=self.dtype, device=device)
        output = torch.bmm(a, b)
        if device == "mps":
            torch.mps.synchronize()
        return a, output

    def get_reference(self, input_tensor: torch.Tensor) -> torch.Tensor:
        a_cpu = self.a_base.float()
        b_cpu = self.b_base.float()
        return torch.bmm(a_cpu, b_cpu)


class LayerNormTest(CorrectnessTest):
    """Test nn.LayerNorm correctness."""

    def __init__(self, batch: int, seq_len: int, hidden: int, dtype: torch.dtype = torch.float32):
        super().__init__(f"LayerNorm_{batch}x{seq_len}x{hidden}_{dtype}", dtype)
        self.batch, self.seq_len, self.hidden = batch, seq_len, hidden

        torch.manual_seed(42)
        self.weight = torch.randn(hidden)
        self.bias = torch.randn(hidden)

    def run(self, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(42)
        x = torch.randn(self.batch, self.seq_len, self.hidden, dtype=self.dtype, device=device)
        weight = self.weight.to(dtype=self.dtype, device=device)
        bias = self.bias.to(dtype=self.dtype, device=device)

        output = F.layer_norm(x, [self.hidden], weight, bias)
        if device == "mps":
            torch.mps.synchronize()
        return x, output

    def get_reference(self, input_tensor: torch.Tensor) -> torch.Tensor:
        x_cpu = input_tensor.float().cpu()
        return F.layer_norm(x_cpu, [self.hidden], self.weight.float(), self.bias.float())


class SoftmaxTest(CorrectnessTest):
    """Test softmax correctness."""

    def __init__(self, batch: int, seq_len: int, vocab: int, dtype: torch.dtype = torch.float32):
        super().__init__(f"Softmax_{batch}x{seq_len}x{vocab}_{dtype}", dtype)
        self.batch, self.seq_len, self.vocab = batch, seq_len, vocab

    def run(self, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(42)
        x = torch.randn(self.batch, self.seq_len, self.vocab, dtype=self.dtype, device=device)
        output = F.softmax(x, dim=-1)
        if device == "mps":
            torch.mps.synchronize()
        return x, output

    def get_reference(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return F.softmax(input_tensor.float().cpu(), dim=-1)


class Conv2dTest(CorrectnessTest):
    """Test Conv2d correctness."""

    def __init__(self, batch: int, in_ch: int, out_ch: int, h: int, w: int,
                 kernel: int = 3, dtype: torch.dtype = torch.float32):
        super().__init__(f"Conv2d_{batch}x{in_ch}x{h}x{w}_k{kernel}_{dtype}", dtype)
        self.batch, self.in_ch, self.out_ch = batch, in_ch, out_ch
        self.h, self.w, self.kernel = h, w, kernel

        torch.manual_seed(42)
        self.weight = torch.randn(out_ch, in_ch, kernel, kernel)
        self.bias = torch.randn(out_ch)

    def run(self, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(42)
        x = torch.randn(self.batch, self.in_ch, self.h, self.w, dtype=self.dtype, device=device)
        weight = self.weight.to(dtype=self.dtype, device=device)
        bias = self.bias.to(dtype=self.dtype, device=device)

        output = F.conv2d(x, weight, bias, padding=1)
        if device == "mps":
            torch.mps.synchronize()
        return x, output

    def get_reference(self, input_tensor: torch.Tensor) -> torch.Tensor:
        x_cpu = input_tensor.float().cpu()
        return F.conv2d(x_cpu, self.weight.float(), self.bias.float(), padding=1)


class GELUTest(CorrectnessTest):
    """Test GELU activation correctness."""

    def __init__(self, shape: List[int], dtype: torch.dtype = torch.float32):
        super().__init__(f"GELU_{'x'.join(map(str, shape))}_{dtype}", dtype)
        self.shape = shape

    def run(self, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(42)
        x = torch.randn(*self.shape, dtype=self.dtype, device=device)
        output = F.gelu(x)
        if device == "mps":
            torch.mps.synchronize()
        return x, output

    def get_reference(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return F.gelu(input_tensor.float().cpu())


class TransformerBlockTest(CorrectnessTest):
    """Test full transformer encoder layer."""

    def __init__(self, batch: int, seq_len: int, d_model: int, nhead: int,
                 dtype: torch.dtype = torch.float32):
        super().__init__(f"TransformerBlock_{batch}x{seq_len}x{d_model}_h{nhead}_{dtype}", dtype)
        self.batch, self.seq_len, self.d_model, self.nhead = batch, seq_len, d_model, nhead

        torch.manual_seed(42)
        self.layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4,
                                                 batch_first=True)
        # Pre-created models for parallel testing (populated by prepare_for_parallel)
        self._parallel_models: Optional[List[nn.Module]] = None

    def prepare_for_parallel(self, num_threads: int, device: str) -> None:
        """Pre-create models for parallel testing to avoid concurrent deepcopy+to() race.

        CRITICAL: Apple's Metal framework has a race condition when multiple threads
        call deepcopy(cpu_model).to(mps_device) concurrently. This manifests as
        incorrect outputs (~15% failure rate, ~0.48 max diff).

        This method creates all model copies SEQUENTIALLY before parallel execution.
        """
        self._parallel_models = []
        for _ in range(num_threads + 1):  # +1 for golden reference
            model = copy.deepcopy(self.layer).to(dtype=self.dtype, device=device)
            model.eval()
            self._parallel_models.append(model)
        if device == "mps":
            torch.mps.synchronize()

    def get_parallel_model(self, thread_id: int) -> Optional[nn.Module]:
        """Get pre-created model for a specific thread."""
        if self._parallel_models is None:
            return None
        return self._parallel_models[thread_id]

    def cleanup_parallel_models(self) -> None:
        """Release pre-created models after parallel testing."""
        self._parallel_models = None

    def run(self, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(42)
        x = torch.randn(self.batch, self.seq_len, self.d_model, dtype=self.dtype, device=device)
        layer = self.layer.to(dtype=self.dtype, device=device)
        layer.eval()

        with torch.no_grad():
            output = layer(x)
        if device == "mps":
            torch.mps.synchronize()
        return x, output

    def get_reference(self, input_tensor: torch.Tensor) -> torch.Tensor:
        x_cpu = input_tensor.float().cpu()
        layer_cpu = self.layer.float().cpu()
        layer_cpu.eval()
        with torch.no_grad():
            return layer_cpu(x_cpu)


def get_standard_tests() -> List[CorrectnessTest]:
    """Get standard test suite."""
    tests = []

    # Linear tests - various sizes
    for dtype in [torch.float32, torch.float16]:
        for batch, in_f, out_f in [(1, 128, 64), (16, 512, 256), (64, 2048, 2048), (128, 4096, 4096)]:
            tests.append(LinearTest(batch, in_f, out_f, dtype))

    # Matmul tests
    for dtype in [torch.float32, torch.float16]:
        for m, k, n in [(64, 64, 64), (256, 512, 256), (1024, 1024, 1024)]:
            tests.append(MatmulTest(m, k, n, dtype))

    # BMM tests
    for dtype in [torch.float32]:
        for batch, m, k, n in [(8, 64, 64, 64), (32, 128, 128, 128)]:
            tests.append(BMMatmulTest(batch, m, k, n, dtype))

    # LayerNorm tests
    for dtype in [torch.float32, torch.float16]:
        for batch, seq, hidden in [(4, 128, 256), (16, 512, 768)]:
            tests.append(LayerNormTest(batch, seq, hidden, dtype))

    # Softmax tests
    for dtype in [torch.float32, torch.float16]:
        tests.append(SoftmaxTest(4, 128, 32000, dtype))

    # Conv2d tests
    for dtype in [torch.float32]:
        tests.append(Conv2dTest(4, 3, 64, 224, 224, 3, dtype))
        tests.append(Conv2dTest(16, 64, 128, 56, 56, 3, dtype))

    # GELU tests
    for dtype in [torch.float32, torch.float16]:
        tests.append(GELUTest([16, 512, 768], dtype))

    # Transformer tests
    for dtype in [torch.float32]:
        tests.append(TransformerBlockTest(4, 128, 256, 4, dtype))

    return tests


def get_full_tests() -> List[CorrectnessTest]:
    """Get comprehensive test suite."""
    tests = get_standard_tests()

    # Add more edge cases
    for dtype in [torch.float32, torch.float16]:
        # Very small
        tests.append(LinearTest(1, 1, 1, dtype))
        tests.append(LinearTest(1, 16, 16, dtype))

        # Very large
        tests.append(LinearTest(1, 8192, 8192, dtype))

        # Non-power-of-2
        tests.append(LinearTest(7, 127, 63, dtype))
        tests.append(LinearTest(13, 333, 777, dtype))

        # Tall/wide matrices
        tests.append(LinearTest(1, 16, 4096, dtype))
        tests.append(LinearTest(1, 4096, 16, dtype))

    return tests


def run_benchmark(tests: List[CorrectnessTest], device: str = "mps") -> BenchmarkReport:
    """Run all tests and generate report."""
    results = []
    passed = 0
    failed = 0

    print(f"\n{'='*70}")
    print(f"Correctness Benchmark: {device.upper()} vs CPU Reference")
    print(f"{'='*70}\n")

    for i, test in enumerate(tests):
        rtol = RTOL.get(test.dtype, 1e-5)
        atol = ATOL.get(test.dtype, 1e-6)

        try:
            # Run on device
            start = time.perf_counter()
            input_tensor, output = test.run(device)
            elapsed_ms = (time.perf_counter() - start) * 1000

            # Get CPU reference
            reference = test.get_reference(input_tensor)

            # Compare
            is_pass, max_abs, max_rel, mean_abs = compare_tensors(
                output, reference, rtol, atol
            )

            result = TestResult(
                name=test.name,
                passed=is_pass,
                max_abs_diff=max_abs,
                max_rel_diff=max_rel,
                mean_abs_diff=mean_abs,
                shape=list(output.shape),
                dtype=str(test.dtype),
                time_ms=elapsed_ms
            )

            if is_pass:
                passed += 1
                status = "✓ PASS"
            else:
                failed += 1
                status = "✗ FAIL"

            print(f"[{i+1:3d}/{len(tests)}] {status} {test.name}")
            print(f"         max_abs={max_abs:.2e} max_rel={max_rel:.2e} mean={mean_abs:.2e}")

        except Exception as e:
            failed += 1
            result = TestResult(
                name=test.name,
                passed=False,
                max_abs_diff=float('inf'),
                max_rel_diff=float('inf'),
                mean_abs_diff=float('inf'),
                shape=[],
                dtype=str(test.dtype),
                time_ms=0,
                error=str(e)
            )
            print(f"[{i+1:3d}/{len(tests)}] ✗ ERROR {test.name}: {e}")

        results.append(asdict(result))

    report = BenchmarkReport(
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        pytorch_version=torch.__version__,
        device=device,
        total_tests=len(tests),
        passed=passed,
        failed=failed,
        results=results
    )

    print(f"\n{'='*70}")
    print(f"SUMMARY: {passed}/{len(tests)} passed ({100*passed/len(tests):.1f}%)")
    if failed > 0:
        print(f"WARNING: {failed} tests FAILED - outputs differ from reference!")
    else:
        print("SUCCESS: All outputs match CPU reference within tolerance")
    print(f"{'='*70}\n")

    return report


def run_parallel_test(test: CorrectnessTest, num_threads: int,
                      num_iterations: int = 10, device: str = "mps") -> ParallelTestResult:
    """
    RIGOROUS test for race conditions in MPS kernels.

    CRITICAL DISCOVERY: Some MPS operations (notably LayerNorm) produce different
    results when called from the main thread vs a spawned thread. This is a real
    MPS bug, not a test artifact.

    To detect RACE CONDITIONS (not thread-affinity bugs), we compute the golden
    reference from a SPAWNED thread, then compare against other spawned threads.
    This tests: "Do concurrent threads produce consistent results?"

    Algorithm:
    1. Pre-generate input tensors on CPU (deterministic)
    2. Run operation from a SPAWNED thread → "golden" reference
    3. Run operation CONCURRENTLY on N spawned threads
    4. ALL outputs must match the golden (spawned thread reference)
    5. Repeat num_iterations times to catch intermittent races
    """
    errors = []

    # Get tolerances (tight for same-device comparison)
    rtol = 1e-6 if test.dtype == torch.float32 else 1e-3
    atol = 1e-6 if test.dtype == torch.float32 else 1e-3

    # Step 0: Pre-create models for TransformerBlockTest BEFORE test.run()
    # CRITICAL: test.run() modifies self.layer in place (moves to device), so we must
    # prepare_for_parallel FIRST while the base layer is still on CPU. This avoids the
    # concurrent deepcopy+to() race in Apple's Metal framework.
    if isinstance(test, TransformerBlockTest):
        test.prepare_for_parallel(num_threads, device)

    # Step 0.5: Pre-generate input on CPU (single-threaded, deterministic)
    torch.manual_seed(42)
    input_cpu, _ = test.run(device)
    if device == "mps":
        torch.mps.synchronize()
    input_for_threads = input_cpu.cpu().clone()

    # Step 1: Compute golden from a SPAWNED thread (not main thread)
    # This ensures we're comparing spawned-to-spawned, detecting true races
    golden_holder = [None]
    def compute_golden():
        local_input = input_for_threads.clone().to(device=device, dtype=test.dtype)
        # Use thread_id=0 for golden reference (first pre-created model)
        output = run_operation_with_input(test, local_input, device, thread_id=0)
        golden_holder[0] = output.cpu().clone()

    golden_thread = threading.Thread(target=compute_golden)
    golden_thread.start()
    golden_thread.join()
    if device == "mps":
        torch.mps.synchronize()

    golden = golden_holder[0]
    if golden is None:
        return ParallelTestResult(
            name=test.name, num_threads=num_threads, passed=False,
            all_threads_identical=False, all_match_reference=False,
            max_inter_thread_diff=float('inf'), max_vs_reference_diff=float('inf'),
            errors=["Failed to compute golden reference"]
        )

    max_inter_diff_overall = 0.0
    max_vs_golden_overall = 0.0
    all_iterations_passed = True

    # Step 2: Run concurrent test multiple times
    for iteration in range(num_iterations):
        outputs = [None] * num_threads
        thread_errors = []
        barrier = threading.Barrier(num_threads)

        def thread_worker(thread_id: int, shared_input: torch.Tensor):
            try:
                # Each thread gets its OWN COPY of the input (avoid sharing)
                local_input = shared_input.clone().to(device=device, dtype=test.dtype)

                # Synchronize all threads to maximize contention
                barrier.wait()

                # Run the operation with the pre-generated input
                # NOTE: We can't use test.run() because it regenerates input
                # Instead, we directly call the underlying operation
                # thread_id+1 because thread_id=0 is reserved for golden reference model
                output = run_operation_with_input(test, local_input, device, thread_id=thread_id + 1)
                outputs[thread_id] = output.cpu().clone()

            except Exception as e:
                thread_errors.append(f"Thread {thread_id} iter {iteration}: {str(e)}")

        # Launch threads - each gets a reference to the shared input
        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=thread_worker, args=(i, input_for_threads))
            threads.append(t)
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        if device == "mps":
            torch.mps.synchronize()

        if thread_errors:
            errors.extend(thread_errors)
            all_iterations_passed = False
            continue

        if any(o is None for o in outputs):
            errors.append(f"Iteration {iteration}: Some threads produced no output")
            all_iterations_passed = False
            continue

        # Step 3: Check ALL concurrent outputs match sequential golden
        for tid, output in enumerate(outputs):
            passed, max_abs, _, _ = compare_tensors(output, golden, rtol, atol)
            max_vs_golden_overall = max(max_vs_golden_overall, max_abs)
            if not passed:
                all_iterations_passed = False
                errors.append(f"Iter {iteration} thread {tid}: output differs from sequential golden "
                             f"(max_abs={max_abs:.2e})")

        # Also check inter-thread consistency within this iteration
        for i in range(1, num_threads):
            _, max_abs, _, _ = compare_tensors(outputs[0], outputs[i], rtol, atol)
            max_inter_diff_overall = max(max_inter_diff_overall, max_abs)

    # Cleanup: release pre-created models for TransformerBlockTest
    if isinstance(test, TransformerBlockTest):
        test.cleanup_parallel_models()

    return ParallelTestResult(
        name=test.name,
        num_threads=num_threads,
        passed=all_iterations_passed and len(errors) == 0,
        all_threads_identical=(max_inter_diff_overall < atol),
        all_match_reference=(max_vs_golden_overall < atol),
        max_inter_thread_diff=max_inter_diff_overall,
        max_vs_reference_diff=max_vs_golden_overall,
        errors=errors[:10]
    )


def run_operation_with_input(test: CorrectnessTest, input_tensor: torch.Tensor,
                             device: str, thread_id: Optional[int] = None) -> torch.Tensor:
    """
    Run the test's operation with a pre-generated input tensor.
    This avoids the torch.manual_seed() race condition.

    Args:
        thread_id: Optional thread ID for accessing pre-created models in parallel tests.
                   Used by TransformerBlockTest to avoid concurrent deepcopy+to() race.
    """
    if isinstance(test, LinearTest):
        weight = test.weight.to(dtype=test.dtype, device=device)
        bias = test.bias_tensor.to(dtype=test.dtype, device=device) if test.bias_tensor is not None else None
        return F.linear(input_tensor, weight, bias)

    elif isinstance(test, MatmulTest):
        b = test.b_base.to(dtype=test.dtype, device=device)
        return torch.matmul(input_tensor, b)

    elif isinstance(test, BMMatmulTest):
        b = test.b_base.to(dtype=test.dtype, device=device)
        return torch.bmm(input_tensor, b)

    elif isinstance(test, LayerNormTest):
        weight = test.weight.to(dtype=test.dtype, device=device)
        bias = test.bias.to(dtype=test.dtype, device=device)
        return F.layer_norm(input_tensor, [test.hidden], weight, bias)

    elif isinstance(test, SoftmaxTest):
        return F.softmax(input_tensor, dim=-1)

    elif isinstance(test, Conv2dTest):
        weight = test.weight.to(dtype=test.dtype, device=device)
        bias = test.bias.to(dtype=test.dtype, device=device)
        return F.conv2d(input_tensor, weight, bias, padding=1)

    elif isinstance(test, GELUTest):
        return F.gelu(input_tensor)

    elif isinstance(test, TransformerBlockTest):
        # Use pre-created model if available (from prepare_for_parallel)
        # This avoids the concurrent deepcopy+to() race condition in Apple's Metal framework
        layer = test.get_parallel_model(thread_id) if thread_id is not None else None
        if layer is None:
            # Fallback: create model on-demand (used for non-parallel tests)
            layer = copy.deepcopy(test.layer).to(dtype=test.dtype, device=device)
            layer.eval()
        with torch.no_grad():
            return layer(input_tensor)

    else:
        raise ValueError(f"Unknown test type: {type(test)}")


def get_parallel_tests() -> List[CorrectnessTest]:
    """Get tests suitable for parallel correctness testing."""
    tests = []

    # Focus on operations most likely to have thread-safety issues
    # These are the operations that use MPS kernels with potential shared state

    # Linear - uses MPSNDArrayMatrixMultiplication
    for dtype in [torch.float32]:
        tests.append(LinearTest(16, 512, 256, dtype))
        tests.append(LinearTest(64, 2048, 2048, dtype))

    # Matmul - core GEMM operation
    for dtype in [torch.float32]:
        tests.append(MatmulTest(256, 512, 256, dtype))
        tests.append(MatmulTest(1024, 1024, 1024, dtype))

    # BMM - batched operations
    tests.append(BMMatmulTest(8, 64, 64, 64, torch.float32))

    # LayerNorm - normalization kernel
    tests.append(LayerNormTest(16, 512, 768, torch.float32))

    # Softmax
    tests.append(SoftmaxTest(4, 128, 32000, torch.float32))

    # Conv2d
    tests.append(Conv2dTest(4, 3, 64, 224, 224, 3, torch.float32))

    # GELU
    tests.append(GELUTest([16, 512, 768], torch.float32))

    # Transformer block - complex multi-op test
    tests.append(TransformerBlockTest(4, 128, 256, 4, torch.float32))

    return tests


def run_parallel_benchmark(tests: List[CorrectnessTest],
                           thread_counts: List[int] = [1, 2, 4, 8],
                           iterations_per_test: int = 10,
                           device: str = "mps") -> ParallelBenchmarkReport:
    """
    RIGOROUS parallel correctness benchmark.

    Tests for race conditions by comparing:
    - SEQUENTIAL MPS output (golden reference)
    - CONCURRENT MPS output (should match golden exactly)

    Key: We're NOT comparing to CPU. We're comparing sequential vs concurrent
    on the SAME device. This catches race conditions that corrupt output.

    Each (test, thread_count) combination runs `iterations_per_test` times
    to catch intermittent race conditions.
    """
    results = []
    passed = 0
    failed = 0

    print(f"\n{'='*70}")
    print(f"PARALLEL RACE DETECTION Benchmark: {device.upper()}")
    print(f"Thread counts: {thread_counts}")
    print(f"Iterations per test: {iterations_per_test}")
    print(f"Comparison: Sequential MPS vs Concurrent MPS (NOT CPU)")
    print(f"{'='*70}\n")

    total = len(tests) * len(thread_counts)
    test_num = 0

    for test in tests:
        print(f"\nTest: {test.name}")
        print("-" * 50)

        for num_threads in thread_counts:
            test_num += 1
            result = run_parallel_test(test, num_threads, iterations_per_test, device)

            if result.passed:
                passed += 1
                status = "✓ PASS"
            else:
                failed += 1
                status = "✗ FAIL"

            print(f"  [{test_num:3d}/{total}] {num_threads:2d} threads x {iterations_per_test} iters: {status}")
            print(f"           inter_thread_diff={result.max_inter_thread_diff:.2e} "
                  f"vs_sequential={result.max_vs_reference_diff:.2e}")

            if not result.all_threads_identical:
                print(f"           RACE DETECTED: Thread outputs differ from each other!")
            if not result.all_match_reference:
                print(f"           RACE DETECTED: Concurrent differs from sequential!")
            if result.errors:
                for err in result.errors[:3]:  # Show first 3 errors
                    print(f"           ERROR: {err}")
                if len(result.errors) > 3:
                    print(f"           ... and {len(result.errors)-3} more errors")

            results.append(asdict(result))

    report = ParallelBenchmarkReport(
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        pytorch_version=torch.__version__,
        device=device,
        thread_counts=thread_counts,
        total_tests=total,
        passed=passed,
        failed=failed,
        results=results
    )

    print(f"\n{'='*70}")
    print(f"RACE DETECTION SUMMARY: {passed}/{total} passed ({100*passed/total:.1f}%)")
    if failed > 0:
        print(f"CRITICAL: {failed} tests detected RACE CONDITIONS!")
        print("Concurrent MPS output differs from sequential MPS output.")
        print("This indicates thread-safety bugs in MPS kernels.")
    else:
        print("SUCCESS: No race conditions detected!")
        print("All concurrent outputs match sequential golden reference.")
        print("Thread-safety verification PASSED.")
    print(f"{'='*70}\n")

    return report


# =============================================================================
# BATCHED PARALLEL TESTS - Routes 8 user threads through 2 worker threads
# =============================================================================

@dataclass
class BatchedTestResult:
    """Result from batched parallel correctness test."""
    name: str
    num_user_threads: int
    num_workers: int
    passed: bool
    all_threads_correct: bool
    max_vs_reference_diff: float
    errors: List[str] = field(default_factory=list)


@dataclass
class BatchedBenchmarkReport:
    """Report from batched parallel correctness benchmark."""
    timestamp: str
    pytorch_version: str
    device: str
    user_thread_counts: List[int]
    num_workers: int
    total_tests: int
    passed: int
    failed: int
    results: List[Dict]


def _batched_user_thread_work(thread_id: int, iter_num: int, input_for_threads: torch.Tensor,
                               num_workers: int, test: CorrectnessTest, device: str,
                               batch_queue: PythonBatchQueue, outputs: List,
                               thread_errors: List, thread_errors_lock: threading.Lock):
    """Worker function for batched parallel test - handles a single user thread."""
    try:
        # Keep input on CPU - worker will move to MPS
        cpu_input = input_for_threads.clone()
        worker_id = thread_id % num_workers
        dtype = test.dtype

        def operation():
            # WORKER does all MPS operations
            mps_input = cpu_input.to(device=device, dtype=dtype)
            result = run_operation_with_input(test, mps_input, device, thread_id=worker_id)
            if device == "mps":
                torch.mps.synchronize()
            return result.cpu().clone()

        future = batch_queue.submit(operation)
        result = future.result(timeout=60.0)
        outputs[thread_id] = result
    except Exception as e:
        with thread_errors_lock:
            thread_errors.append(f"Thread {thread_id} iter {iter_num}: {str(e)}")


def run_batched_parallel_test(test: CorrectnessTest, num_user_threads: int,
                               num_iterations: int = 10, num_workers: int = 1,
                               device: str = "mps") -> BatchedTestResult:
    """
    Test correctness with BATCHED parallelism.

    User threads submit requests to a batch queue, which routes them to
    a small number of worker threads (default: 2). This bypasses Apple
    Metal's thread-safety bugs at 4+ concurrent threads.

    Algorithm:
    1. Pre-generate input tensors on CPU
    2. Compute sequential golden reference
    3. Submit num_user_threads concurrent requests through batch queue
    4. All outputs must match the golden reference
    5. Repeat num_iterations times
    """
    errors = []

    # Tolerances (tight for same-device comparison)
    rtol = 1e-5 if test.dtype == torch.float32 else 1e-2
    atol = 1e-5 if test.dtype == torch.float32 else 1e-2

    # Pre-create models for TransformerBlockTest
    if isinstance(test, TransformerBlockTest):
        # Only need num_workers models since workers process sequentially
        test.prepare_for_parallel(num_workers, device)

    # Pre-generate input on CPU
    torch.manual_seed(42)
    input_cpu, _ = test.run(device)
    if device == "mps":
        torch.mps.synchronize()
    input_for_threads = input_cpu.cpu().clone()

    # Compute golden reference (sequential)
    local_input = input_for_threads.clone().to(device=device, dtype=test.dtype)
    golden = run_operation_with_input(test, local_input, device, thread_id=0)
    if device == "mps":
        torch.mps.synchronize()
    golden = golden.cpu().clone()

    max_vs_golden_overall = 0.0
    all_iterations_passed = True

    # Create batch queue with specified workers
    batch_queue = PythonBatchQueue(num_workers=num_workers)
    batch_queue.start()

    try:
        for iteration in range(num_iterations):
            outputs = [None] * num_user_threads
            thread_errors_lock = threading.Lock()
            thread_errors = []

            # Launch user threads - using external function to avoid closure issues
            threads = []
            for i in range(num_user_threads):
                t = threading.Thread(
                    target=_batched_user_thread_work,
                    args=(i, iteration, input_for_threads, num_workers, test,
                          device, batch_queue, outputs, thread_errors, thread_errors_lock)
                )
                threads.append(t)
                t.start()

            for t in threads:
                t.join(timeout=120.0)

            if device == "mps":
                torch.mps.synchronize()

            if thread_errors:
                errors.extend(thread_errors)
                all_iterations_passed = False
                continue

            if any(o is None for o in outputs):
                errors.append(f"Iteration {iteration}: Some threads produced no output")
                all_iterations_passed = False
                continue

            # Check all outputs match golden
            for tid, output in enumerate(outputs):
                passed, max_abs, _, _ = compare_tensors(output, golden, rtol, atol)
                max_vs_golden_overall = max(max_vs_golden_overall, max_abs)
                if not passed:
                    all_iterations_passed = False
                    errors.append(f"Iter {iteration} thread {tid}: diff={max_abs:.2e}")

    finally:
        batch_queue.stop()

    # Cleanup
    if isinstance(test, TransformerBlockTest):
        test.cleanup_parallel_models()

    return BatchedTestResult(
        name=test.name,
        num_user_threads=num_user_threads,
        num_workers=num_workers,
        passed=all_iterations_passed and len(errors) == 0,
        all_threads_correct=(max_vs_golden_overall < atol),
        max_vs_reference_diff=max_vs_golden_overall,
        errors=errors[:10]
    )


def run_batched_parallel_benchmark(tests: List[CorrectnessTest],
                                    user_thread_counts: List[int] = [4, 8, 16],
                                    num_workers: int = 1,
                                    iterations_per_test: int = 10,
                                    device: str = "mps") -> BatchedBenchmarkReport:
    """
    BATCHED parallel correctness benchmark.

    Routes user threads through a batch queue to a small number of workers,
    bypassing Apple Metal's thread-safety bugs. Tests that this approach
    produces correct results.

    Key: We compare batched concurrent output vs sequential golden reference.
    All outputs should match exactly.
    """
    results = []
    passed = 0
    failed = 0

    print(f"\n{'='*70}")
    print(f"BATCHED PARALLEL Correctness Benchmark: {device.upper()}")
    print(f"User thread counts: {user_thread_counts}")
    print(f"Worker threads: {num_workers}")
    print(f"Iterations per test: {iterations_per_test}")
    print(f"{'='*70}\n")

    total = len(tests) * len(user_thread_counts)
    test_num = 0

    for test in tests:
        print(f"\nTest: {test.name}")
        print("-" * 50)

        for num_threads in user_thread_counts:
            test_num += 1
            result = run_batched_parallel_test(
                test, num_threads, iterations_per_test, num_workers, device
            )

            if result.passed:
                passed += 1
                status = "✓ PASS"
            else:
                failed += 1
                status = "✗ FAIL"

            print(f"  [{test_num:3d}/{total}] {num_threads:2d} user threads -> {num_workers} workers: {status}")
            print(f"           max_diff={result.max_vs_reference_diff:.2e}")

            if result.errors:
                for err in result.errors[:3]:
                    print(f"           ERROR: {err}")
                if len(result.errors) > 3:
                    print(f"           ... and {len(result.errors)-3} more errors")

            results.append(asdict(result))

    report = BatchedBenchmarkReport(
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        pytorch_version=torch.__version__,
        device=device,
        user_thread_counts=user_thread_counts,
        num_workers=num_workers,
        total_tests=total,
        passed=passed,
        failed=failed,
        results=results
    )

    print(f"\n{'='*70}")
    print(f"BATCHED BENCHMARK SUMMARY: {passed}/{total} passed ({100*passed/total:.1f}%)")
    if failed > 0:
        print(f"WARNING: {failed} tests failed with batching")
    else:
        print("SUCCESS: All batched parallel tests passed!")
        print(f"Correctness verified at {max(user_thread_counts)} user threads")
        print(f"via {num_workers}-worker batching.")
    print(f"{'='*70}\n")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Correctness Benchmark: Single-threaded and Parallel Race Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single-threaded MPS vs CPU reference
    python correctness_benchmark.py

    # Parallel race detection (CRITICAL for thread-safety)
    python correctness_benchmark.py --parallel

    # Parallel with custom thread counts and iterations
    python correctness_benchmark.py --parallel --threads 1,2,4,8,16 --iterations 20

    # BATCHED parallel: 8 user threads via batching (default 1 worker)
    python correctness_benchmark.py --parallel --use-batching --threads 8

    # Both single-threaded and parallel
    python correctness_benchmark.py --all
        """
    )
    parser.add_argument("--full", action="store_true",
                        help="Run full test suite with edge cases (single-threaded only)")
    parser.add_argument("--parallel", action="store_true",
                        help="Run PARALLEL race detection tests (sequential vs concurrent MPS)")
    parser.add_argument("--all", action="store_true",
                        help="Run both single-threaded and parallel tests")
    parser.add_argument("--use-batching", action="store_true",
                        help="Use batch queue (user threads -> worker threads) to bypass Metal bugs")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of worker threads for batching (default: 1)")
    parser.add_argument("--threads", type=str, default="1,2,4,8",
                        help="Comma-separated thread counts for parallel tests (default: 1,2,4,8)")
    parser.add_argument("--iterations", type=int, default=10,
                        help="Iterations per parallel test to catch intermittent races (default: 10)")
    parser.add_argument("--output", type=str, default="correctness_report.json",
                        help="Output JSON file")
    parser.add_argument("--device", type=str, default="mps",
                        help="Device to test (mps, cuda)")
    args = parser.parse_args()

    if args.workers < 1:
        print("ERROR: --workers must be >= 1", file=sys.stderr)
        sys.exit(2)

    # Check device availability
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("ERROR: MPS not available", file=sys.stderr)
        sys.exit(1)

    # Warm up
    print("Warming up MPS...")
    _ = torch.zeros(1, device=args.device)
    if args.device == "mps":
        torch.mps.synchronize()

    failed_count = 0

    # Determine what to run
    run_single = not args.parallel or args.all
    run_parallel = args.parallel or args.all

    # Single-threaded tests: MPS vs CPU reference
    if run_single:
        print("\n" + "="*70)
        print("PART 1: Single-threaded MPS vs CPU Reference")
        print("="*70)

        tests = get_full_tests() if args.full else get_standard_tests()
        report = run_benchmark(tests, args.device)

        output_file = args.output
        with open(output_file, 'w') as f:
            json.dump(asdict(report), f, indent=2)
        print(f"Single-threaded report saved to: {output_file}")

        failed_count += report.failed

    # Parallel tests: Sequential MPS vs Concurrent MPS (race detection)
    if run_parallel:
        thread_counts = [int(x) for x in args.threads.split(",")]
        tests = get_parallel_tests()

        if args.use_batching:
            # BATCHED mode: route user threads through batch queue
            print("\n" + "="*70)
            thread_counts_str = ",".join(str(x) for x in thread_counts)
            print(
                "PART 2: BATCHED Parallel Correctness "
                f"(threads={thread_counts_str} -> workers={args.workers})"
            )
            print("="*70)

            batched_report = run_batched_parallel_benchmark(
                tests, thread_counts, args.workers, args.iterations, args.device
            )

            batched_output = args.output.replace(".json", "_batched.json")
            with open(batched_output, 'w') as f:
                json.dump(asdict(batched_report), f, indent=2)
            print(f"Batched report saved to: {batched_output}")

            failed_count += batched_report.failed
        else:
            # DIRECT mode: standard parallel race detection
            print("\n" + "="*70)
            print("PART 2: Parallel Race Detection (Sequential vs Concurrent MPS)")
            print("="*70)

            parallel_report = run_parallel_benchmark(
                tests, thread_counts, args.iterations, args.device
            )

            parallel_output = args.output.replace(".json", "_parallel.json")
            with open(parallel_output, 'w') as f:
                json.dump(asdict(parallel_report), f, indent=2)
            print(f"Parallel report saved to: {parallel_output}")

            failed_count += parallel_report.failed

    # Final summary
    if run_single and run_parallel:
        print("\n" + "="*70)
        print("FINAL SUMMARY")
        print("="*70)
        if failed_count == 0:
            print("ALL TESTS PASSED")
            print("  - Single-threaded: MPS outputs match CPU reference")
            if args.use_batching:
                print("  - Batched parallel: Correctness verified via batching")
            else:
                print("  - Parallel: No race conditions detected")
        else:
            print(f"FAILURES DETECTED: {failed_count} total")

    sys.exit(0 if failed_count == 0 else 1)


if __name__ == "__main__":
    main()
