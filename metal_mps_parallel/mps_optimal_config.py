#!/usr/bin/env python3
"""
MPS Optimal Configuration Calculator

Created by Andrew Yates

This module provides the optimal configuration for ANY job on MPS,
given the constraints and requirements.

Usage:
    from mps_optimal_config import get_optimal_config

    config = get_optimal_config(
        num_requests=1000,
        max_latency_ms=50,
        num_processes_available=4
    )
    print(config)
"""

from dataclasses import dataclass
from typing import Optional
import math

# Measured performance characteristics (from benchmark_scaling_curves.py)
# These are empirical measurements on Apple Silicon M-series

MEASURED_PERFORMANCE = {
    # Single operation baseline (batch=1, single thread)
    'single_op_rate': 10698,  # ops/s

    # Threading scaling (efficiency drops as threads increase)
    'threading': {
        1: {'efficiency': 0.067, 'ops_per_sec': 719},
        2: {'efficiency': 0.122, 'ops_per_sec': 2605},
        4: {'efficiency': 0.064, 'ops_per_sec': 2752},
        8: {'efficiency': 0.032, 'ops_per_sec': 2728},
        16: {'efficiency': 0.016, 'ops_per_sec': 2720},
    },

    # Batching scaling (efficiency stays high)
    'batching': {
        1: {'efficiency': 0.862, 'samples_per_sec': 9218},
        2: {'efficiency': 0.836, 'samples_per_sec': 17892},
        4: {'efficiency': 0.907, 'samples_per_sec': 38823},
        8: {'efficiency': 0.962, 'samples_per_sec': 82315},
        16: {'efficiency': 0.947, 'samples_per_sec': 162060},
        32: {'efficiency': 0.899, 'samples_per_sec': 307829},
        64: {'efficiency': 0.882, 'samples_per_sec': 603591},
        128: {'efficiency': 0.722, 'samples_per_sec': 988226},
        256: {'efficiency': 0.507, 'samples_per_sec': 1388672},
    },

    # Process pool scaling (each process has own Metal context)
    'process_pool': {
        1: {'efficiency': 0.188, 'ops_per_sec': 2007},
        2: {'efficiency': 0.206, 'ops_per_sec': 4414},
        4: {'efficiency': 0.214, 'ops_per_sec': 9178},
        # 8 processes has issues with queue timeouts
    },
}

# Theoretical "fixed" performance (if Apple fixed the driver)
THEORETICAL_FIXED = {
    'threading': {
        # Assume 85% efficiency like batching achieves
        1: {'efficiency': 0.85, 'ops_per_sec': 9093},
        2: {'efficiency': 0.85, 'ops_per_sec': 18187},
        4: {'efficiency': 0.85, 'ops_per_sec': 36374},
        8: {'efficiency': 0.85, 'ops_per_sec': 72748},
        16: {'efficiency': 0.80, 'ops_per_sec': 136893},
    },
}


@dataclass
class OptimalConfig:
    """Optimal configuration for MPS inference."""
    method: str  # 'batching', 'process_pool', 'threading'
    batch_size: int
    num_workers: int  # threads or processes
    expected_throughput: float  # samples/sec
    expected_latency_ms: float  # per batch
    efficiency: float  # vs theoretical ideal
    reasoning: str


def get_optimal_config(
    num_requests: int = 1,
    max_latency_ms: Optional[float] = None,
    num_processes_available: int = 1,
    prefer_throughput: bool = True
) -> OptimalConfig:
    """
    Get the optimal MPS configuration for a given workload.

    Args:
        num_requests: Number of inference requests to process
        max_latency_ms: Maximum acceptable latency per request (None = no limit)
        num_processes_available: Number of processes that can be spawned
        prefer_throughput: If True, optimize for throughput; else latency

    Returns:
        OptimalConfig with the recommended settings
    """

    # RULE 1: Never use threading for MPS parallelism
    # Threading efficiency is 3-12%, batching is 50-96%

    if max_latency_ms is not None and max_latency_ms < 1.0:
        # Ultra-low latency: use small batch
        batch_size = 1
        throughput = MEASURED_PERFORMANCE['batching'][1]['samples_per_sec']
        latency = 1000.0 / throughput
        return OptimalConfig(
            method='batching',
            batch_size=1,
            num_workers=1,
            expected_throughput=throughput,
            expected_latency_ms=latency,
            efficiency=MEASURED_PERFORMANCE['batching'][1]['efficiency'],
            reasoning="Ultra-low latency requires batch_1"
        )

    if prefer_throughput:
        # For throughput: use largest batch that fits constraints
        best_config = None
        best_throughput = 0

        for batch_size in [256, 128, 64, 32, 16, 8, 4, 2, 1]:
            if batch_size not in MEASURED_PERFORMANCE['batching']:
                continue

            perf = MEASURED_PERFORMANCE['batching'][batch_size]
            throughput = perf['samples_per_sec']
            latency = batch_size * 1000.0 / throughput

            # Check latency constraint
            if max_latency_ms is not None and latency > max_latency_ms:
                continue

            # Scale by available processes
            if num_processes_available > 1 and batch_size <= 64:
                # Each process can run independently
                throughput *= min(num_processes_available, 4)
                # Latency stays the same (parallel, not serial)

            if throughput > best_throughput:
                best_throughput = throughput
                best_config = OptimalConfig(
                    method='batching' + ('+process_pool' if num_processes_available > 1 else ''),
                    batch_size=batch_size,
                    num_workers=min(num_processes_available, 4),
                    expected_throughput=throughput,
                    expected_latency_ms=latency,
                    efficiency=perf['efficiency'],
                    reasoning=f"batch_{batch_size} × {min(num_processes_available, 4)} processes for max throughput"
                )

        if best_config:
            return best_config

    # Default: batch_64 single process
    perf = MEASURED_PERFORMANCE['batching'][64]
    return OptimalConfig(
        method='batching',
        batch_size=64,
        num_workers=1,
        expected_throughput=perf['samples_per_sec'],
        expected_latency_ms=64 * 1000.0 / perf['samples_per_sec'],
        efficiency=perf['efficiency'],
        reasoning="Default: batch_64 for good throughput/latency balance"
    )


def print_configuration_table():
    """Print the optimal configuration for all common scenarios."""

    print("=" * 90)
    print("MPS OPTIMAL CONFIGURATION TABLE")
    print("=" * 90)
    print()

    scenarios = [
        ("Single request, low latency", {"num_requests": 1, "max_latency_ms": 5}),
        ("Batch processing, max throughput", {"num_requests": 10000, "prefer_throughput": True}),
        ("4 processes available", {"num_requests": 1000, "num_processes_available": 4}),
        ("Real-time (10ms latency)", {"max_latency_ms": 10}),
        ("Real-time (50ms latency)", {"max_latency_ms": 50}),
        ("Offline batch job", {"num_requests": 100000, "num_processes_available": 4}),
    ]

    print(f"{'Scenario':<35} {'Method':<25} {'Batch':<8} {'Workers':<8} {'Throughput':<15} {'Latency':<10}")
    print("-" * 90)

    for name, kwargs in scenarios:
        config = get_optimal_config(**kwargs)
        print(f"{name:<35} {config.method:<25} {config.batch_size:<8} {config.num_workers:<8} {config.expected_throughput:<15.0f} {config.expected_latency_ms:<10.1f}ms")

    print()
    print("=" * 90)
    print("CURRENT vs FIXED COMPARISON")
    print("=" * 90)
    print()

    print(f"{'N':<6} {'Current Threading':<25} {'IF FIXED Threading':<25} {'Gap':<15}")
    print("-" * 70)

    for n in [1, 2, 4, 8, 16]:
        current = MEASURED_PERFORMANCE['threading'][n]
        fixed = THEORETICAL_FIXED['threading'][n]
        gap = fixed['ops_per_sec'] / current['ops_per_sec']

        print(f"{n:<6} {current['ops_per_sec']:<10.0f} ({current['efficiency']:<5.1%})   "
              f"{fixed['ops_per_sec']:<10.0f} ({fixed['efficiency']:<5.1%})   "
              f"{gap:<.0f}x lost")

    print()
    print("TOTAL PERFORMANCE LEFT ON TABLE:")
    current_8t = MEASURED_PERFORMANCE['threading'][8]['ops_per_sec']
    fixed_8t = THEORETICAL_FIXED['threading'][8]['ops_per_sec']
    print(f"  At 8 threads: {current_8t:.0f} → {fixed_8t:.0f} ops/s ({fixed_8t/current_8t:.0f}x potential)")

    batching_8 = MEASURED_PERFORMANCE['batching'][8]['samples_per_sec']
    print(f"  Batching (batch_8) already achieves: {batching_8:.0f} samples/s")
    print(f"  Batching efficiency: {MEASURED_PERFORMANCE['batching'][8]['efficiency']:.0%}")
    print()


def visualize_scaling_comparison():
    """Show visual comparison of current vs fixed scaling."""

    print("=" * 90)
    print("SCALING CURVES: CURRENT vs IF-FIXED")
    print("=" * 90)
    print()

    print("THREADING EFFICIENCY:")
    print()
    print("100% ┬" + "─" * 70)

    # Show batching for reference
    print("    │" + " " * 5 + "████████████████████████████ Batching (for reference)")
    print(" 80%┤" + " " * 5 + "████████████████████████████████████")
    print("    │")
    print("    │" + " " * 20 + "▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ IF FIXED (theoretical)")
    print(" 60%┤")
    print("    │")
    print(" 40%┤")
    print("    │")
    print(" 20%┤  ░░")
    print("    │░░░░░░")
    print(" 10%┤░░░░░░░░░░░░ CURRENT (Apple bug)")
    print("    │    ░░░░░░░░░░░░░░░░")
    print("  0%┴───┬───┬───┬───┬───┬───┬───┬───┬"+ "─" * 30)
    print("        1   2   4   8  16  32  64 128")
    print("              Threads / Batch Size")
    print()
    print("Legend:")
    print("  ░░░░ = Current threading (3-12% efficiency)")
    print("  ▓▓▓▓ = IF FIXED threading (85% efficiency, theoretical)")
    print("  ████ = Batching (86-96% efficiency, working NOW)")
    print()


if __name__ == "__main__":
    print_configuration_table()
    visualize_scaling_comparison()

    print("=" * 90)
    print("RECOMMENDATION")
    print("=" * 90)
    print("""
For ANY MPS workload, use this decision tree:

1. Need lowest latency?
   → Use batch_1 (single sample, ~9K samples/s)

2. Need highest throughput?
   → Use batch_256 (~1.4M samples/s)
   → Add process pool (4 processes) for ~2.4M samples/s

3. Need parallel inference streams?
   → DO NOT use threading (3% efficiency)
   → USE process pool + batching per process

4. Balanced latency/throughput?
   → Use batch_32 to batch_64 (88-90% efficiency)

THE KEY INSIGHT:
  Threading is BROKEN on MPS (Apple driver serialization)
  Batching WORKS (86-96% efficiency)
  Use batching, not threading
""")
