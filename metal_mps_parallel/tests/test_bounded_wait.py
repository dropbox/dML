#!/usr/bin/env python3
"""
Bounded Wait Verification Test

This test measures actual wait times when threads compete for MPS streams
and verifies that no thread waits beyond a configured threshold.

Purpose:
- Phase 3 aspirational property verification
- Connect TLA+ bounded wait spec to runtime behavior
- Detect pathological lock/queue waits that indicate scaling issues

Test Methods:
1. Wait histogram - measures distribution of wait times across threads
2. Max wait threshold - fails if any thread waits too long
3. P99 latency - tracks 99th percentile wait time

Usage:
    python tests/test_bounded_wait.py
    python tests/test_bounded_wait.py --threads 8 --iterations 100
    python tests/test_bounded_wait.py --max-wait-ms 1000  # 1 second threshold

Output:
    - Wait time histogram (ms buckets)
    - Max observed wait time
    - P50, P95, P99 latencies
    - PASS/FAIL based on threshold

Created: 2025-12-19 by Worker N=1301
"""

import argparse
import json
import os
import queue
import statistics
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Callable, Any

# Try to import torch - gracefully handle if not available
try:
    import torch
    HAS_TORCH = True
    HAS_MPS = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
except ImportError:
    HAS_TORCH = False
    HAS_MPS = False


@dataclass
class WaitMeasurement:
    """Records a single wait time measurement."""
    thread_id: int
    iteration: int
    wait_time_ns: int
    acquired: bool
    timestamp_ns: int

    @property
    def wait_time_ms(self) -> float:
        return self.wait_time_ns / 1_000_000


@dataclass
class WaitStatistics:
    """Aggregated wait statistics."""
    total_measurements: int
    total_waits: int
    min_wait_ns: int
    max_wait_ns: int
    mean_wait_ns: float
    median_wait_ns: float
    p95_wait_ns: float
    p99_wait_ns: float
    histogram_ms: Dict[str, int]

    @property
    def min_wait_ms(self) -> float:
        return self.min_wait_ns / 1_000_000

    @property
    def max_wait_ms(self) -> float:
        return self.max_wait_ns / 1_000_000

    @property
    def mean_wait_ms(self) -> float:
        return self.mean_wait_ns / 1_000_000

    @property
    def median_wait_ms(self) -> float:
        return self.median_wait_ns / 1_000_000

    @property
    def p95_wait_ms(self) -> float:
        return self.p95_wait_ns / 1_000_000

    @property
    def p99_wait_ms(self) -> float:
        return self.p99_wait_ns / 1_000_000


class BoundedWaitMonitor:
    """
    Monitors wait times for bounded wait verification.

    This class provides instrumentation to measure how long threads wait
    when competing for resources (streams, locks, queue slots).
    """

    def __init__(self, max_wait_threshold_ms: float = 5000.0):
        """
        Args:
            max_wait_threshold_ms: Maximum allowed wait time in milliseconds.
                                   Test fails if any wait exceeds this.
        """
        self.max_wait_threshold_ms = max_wait_threshold_ms
        self.max_wait_threshold_ns = int(max_wait_threshold_ms * 1_000_000)
        self.measurements: List[WaitMeasurement] = []
        self._lock = threading.Lock()
        self._violations: List[WaitMeasurement] = []

    def record_wait(self, thread_id: int, iteration: int,
                    wait_time_ns: int, acquired: bool = True) -> None:
        """Record a wait time measurement."""
        measurement = WaitMeasurement(
            thread_id=thread_id,
            iteration=iteration,
            wait_time_ns=wait_time_ns,
            acquired=acquired,
            timestamp_ns=time.time_ns()
        )

        with self._lock:
            self.measurements.append(measurement)
            if wait_time_ns > self.max_wait_threshold_ns:
                self._violations.append(measurement)

    def get_statistics(self) -> WaitStatistics:
        """Compute aggregate statistics from measurements."""
        with self._lock:
            wait_times = [m.wait_time_ns for m in self.measurements if m.wait_time_ns > 0]

        if not wait_times:
            return WaitStatistics(
                total_measurements=len(self.measurements),
                total_waits=0,
                min_wait_ns=0,
                max_wait_ns=0,
                mean_wait_ns=0,
                median_wait_ns=0,
                p95_wait_ns=0,
                p99_wait_ns=0,
                histogram_ms={}
            )

        # Compute percentiles
        sorted_waits = sorted(wait_times)
        p95_idx = int(len(sorted_waits) * 0.95)
        p99_idx = int(len(sorted_waits) * 0.99)

        # Build histogram (ms buckets: 0-1, 1-10, 10-100, 100-1000, 1000+)
        histogram: Dict[str, int] = {
            "0-1ms": 0,
            "1-10ms": 0,
            "10-100ms": 0,
            "100-1000ms": 0,
            "1000ms+": 0
        }
        for wait_ns in wait_times:
            wait_ms = wait_ns / 1_000_000
            if wait_ms < 1:
                histogram["0-1ms"] += 1
            elif wait_ms < 10:
                histogram["1-10ms"] += 1
            elif wait_ms < 100:
                histogram["10-100ms"] += 1
            elif wait_ms < 1000:
                histogram["100-1000ms"] += 1
            else:
                histogram["1000ms+"] += 1

        return WaitStatistics(
            total_measurements=len(self.measurements),
            total_waits=len(wait_times),
            min_wait_ns=min(wait_times),
            max_wait_ns=max(wait_times),
            mean_wait_ns=statistics.mean(wait_times),
            median_wait_ns=statistics.median(wait_times),
            p95_wait_ns=sorted_waits[p95_idx] if sorted_waits else 0,
            p99_wait_ns=sorted_waits[p99_idx] if sorted_waits else 0,
            histogram_ms=histogram
        )

    def get_violations(self) -> List[WaitMeasurement]:
        """Return list of measurements that exceeded the threshold."""
        with self._lock:
            return list(self._violations)

    def check_bounded_wait(self) -> bool:
        """Check if all waits were bounded. Returns True if no violations."""
        return len(self._violations) == 0


class SimulatedResourcePool:
    """
    Simulated resource pool for bounded wait testing.

    Models a pool of N resources that threads compete for.
    Used when MPS is not available.
    """

    def __init__(self, pool_size: int = 4, hold_time_ms: float = 10.0):
        self.pool_size = pool_size
        self.hold_time_ms = hold_time_ms
        self._available = pool_size
        self._condition = threading.Condition()
        self._acquired_by: Dict[int, int] = {}  # thread_id -> slot

    def acquire(self, thread_id: int, timeout_ms: float = 5000.0) -> tuple[int, bool]:
        """
        Acquire a resource from the pool.

        Returns:
            (wait_time_ns, success)
        """
        start_ns = time.time_ns()
        timeout_s = timeout_ms / 1000.0

        with self._condition:
            deadline = time.time() + timeout_s

            while self._available <= 0:
                remaining = deadline - time.time()
                if remaining <= 0:
                    return (time.time_ns() - start_ns, False)
                self._condition.wait(timeout=remaining)

            # Got one
            self._available -= 1
            slot = self.pool_size - self._available
            self._acquired_by[thread_id] = slot

        wait_time_ns = time.time_ns() - start_ns
        return (wait_time_ns, True)

    def release(self, thread_id: int) -> None:
        """Release a resource back to the pool."""
        with self._condition:
            if thread_id in self._acquired_by:
                del self._acquired_by[thread_id]
                self._available += 1
                self._condition.notify()


class MPSResourcePool:
    """
    MPS stream pool for bounded wait testing.

    Uses actual PyTorch MPS operations to measure real wait times.
    """

    def __init__(self):
        if not HAS_MPS:
            raise RuntimeError("MPS not available")

    def acquire(self, thread_id: int, timeout_ms: float = 5000.0) -> tuple[int, bool]:
        """
        Acquire an MPS stream by performing an operation.

        Wait time is measured as time from request to completion.
        """
        start_ns = time.time_ns()

        try:
            # Simple MPS operation to trigger stream acquisition
            x = torch.randn(64, 64, device='mps')
            y = torch.mm(x, x)
            torch.mps.synchronize()

            wait_time_ns = time.time_ns() - start_ns
            return (wait_time_ns, True)
        except Exception as e:
            wait_time_ns = time.time_ns() - start_ns
            return (wait_time_ns, False)

    def release(self, thread_id: int) -> None:
        """No-op for MPS - streams are managed internally."""
        pass


def run_bounded_wait_test(
    num_threads: int = 8,
    iterations_per_thread: int = 50,
    max_wait_ms: float = 5000.0,
    pool_size: int = 4,
    hold_time_ms: float = 10.0,
    use_mps: bool = True,
    verbose: bool = True
) -> tuple[WaitStatistics, bool, List[WaitMeasurement]]:
    """
    Run the bounded wait test.

    Args:
        num_threads: Number of concurrent threads competing for resources
        iterations_per_thread: Number of acquire/release cycles per thread
        max_wait_ms: Maximum allowed wait time (test fails if exceeded)
        pool_size: Size of resource pool (for simulated pool)
        hold_time_ms: Time to hold resource (for simulated pool)
        use_mps: Use real MPS if available
        verbose: Print progress

    Returns:
        (statistics, passed, violations)
    """
    monitor = BoundedWaitMonitor(max_wait_threshold_ms=max_wait_ms)

    # Choose resource pool
    if use_mps and HAS_MPS:
        pool = MPSResourcePool()
        if verbose:
            print(f"Using MPS resource pool")
    else:
        pool = SimulatedResourcePool(pool_size=pool_size, hold_time_ms=hold_time_ms)
        if verbose:
            print(f"Using simulated resource pool (size={pool_size})")

    completed = [0]
    lock = threading.Lock()

    def worker(thread_id: int):
        for i in range(iterations_per_thread):
            # Acquire resource and measure wait time
            wait_ns, acquired = pool.acquire(thread_id, timeout_ms=max_wait_ms)
            monitor.record_wait(thread_id, i, wait_ns, acquired)

            if acquired:
                # Simulate some work
                if isinstance(pool, SimulatedResourcePool):
                    time.sleep(hold_time_ms / 1000.0)

                pool.release(thread_id)

            with lock:
                completed[0] += 1

    # Run test
    if verbose:
        print(f"Starting bounded wait test: {num_threads} threads, "
              f"{iterations_per_thread} iterations each")
        print(f"Max wait threshold: {max_wait_ms:.1f}ms")

    start_time = time.time()

    threads = []
    for t in range(num_threads):
        thread = threading.Thread(target=worker, args=(t,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    elapsed = time.time() - start_time

    # Get results
    stats = monitor.get_statistics()
    passed = monitor.check_bounded_wait()
    violations = monitor.get_violations()

    if verbose:
        print(f"\n{'='*60}")
        print("BOUNDED WAIT TEST RESULTS")
        print(f"{'='*60}")
        print(f"Elapsed time: {elapsed:.2f}s")
        print(f"Total measurements: {stats.total_measurements}")
        print(f"Total waits with contention: {stats.total_waits}")
        print(f"\nWait Time Statistics:")
        print(f"  Min:    {stats.min_wait_ms:8.2f} ms")
        print(f"  Max:    {stats.max_wait_ms:8.2f} ms")
        print(f"  Mean:   {stats.mean_wait_ms:8.2f} ms")
        print(f"  Median: {stats.median_wait_ms:8.2f} ms")
        print(f"  P95:    {stats.p95_wait_ms:8.2f} ms")
        print(f"  P99:    {stats.p99_wait_ms:8.2f} ms")
        print(f"\nWait Time Histogram:")
        for bucket, count in stats.histogram_ms.items():
            bar = '#' * min(50, count)
            print(f"  {bucket:>12}: {count:5} {bar}")
        print(f"\nThreshold: {max_wait_ms:.1f} ms")
        print(f"Violations: {len(violations)}")
        print(f"\n{'PASS' if passed else 'FAIL'}: Bounded wait property "
              f"{'SATISFIED' if passed else 'VIOLATED'}")

    return stats, passed, violations


def main():
    parser = argparse.ArgumentParser(
        description='Bounded Wait Verification Test',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--threads', type=int, default=8,
                        help='Number of concurrent threads (default: 8)')
    parser.add_argument('--iterations', type=int, default=50,
                        help='Iterations per thread (default: 50)')
    parser.add_argument('--max-wait-ms', type=float, default=5000.0,
                        help='Max allowed wait time in ms (default: 5000)')
    parser.add_argument('--pool-size', type=int, default=4,
                        help='Simulated pool size (default: 4)')
    parser.add_argument('--hold-time-ms', type=float, default=10.0,
                        help='Simulated hold time in ms (default: 10)')
    parser.add_argument('--no-mps', action='store_true',
                        help='Force simulated pool even if MPS available')
    parser.add_argument('--json', type=str, metavar='FILE',
                        help='Output results to JSON file')
    parser.add_argument('--quiet', action='store_true',
                        help='Quiet mode (less output)')

    args = parser.parse_args()

    stats, passed, violations = run_bounded_wait_test(
        num_threads=args.threads,
        iterations_per_thread=args.iterations,
        max_wait_ms=args.max_wait_ms,
        pool_size=args.pool_size,
        hold_time_ms=args.hold_time_ms,
        use_mps=not args.no_mps,
        verbose=not args.quiet
    )

    # Output JSON if requested
    if args.json:
        result = {
            "test": "bounded_wait",
            "passed": passed,
            "config": {
                "num_threads": args.threads,
                "iterations_per_thread": args.iterations,
                "max_wait_threshold_ms": args.max_wait_ms,
                "pool_size": args.pool_size,
                "used_mps": not args.no_mps and HAS_MPS,
            },
            "statistics": asdict(stats),
            "violations_count": len(violations),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        with open(args.json, 'w') as f:
            json.dump(result, f, indent=2)
        if not args.quiet:
            print(f"\nResults written to: {args.json}")

    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
