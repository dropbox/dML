"""
Performance Baseline Smoke Tests - Regression Detection

Validates that TTS synthesis latency stays within acceptable bounds.
If performance regresses significantly, these tests will fail.

This tests COLD-START one-shot mode (model loading + synthesis per invocation).
This is different from warm/daemon mode benchmarks which keep models loaded.

Cold-start baselines established 2025-12-09 (#433):
- Short text (~5 words): P50 ~540ms (dominated by model loading)
- Medium text (~15 words): P50 ~540ms
- Long text (~30 words): P50 ~560ms

Warm-mode baselines (perf_snapshot tool, 2025-12-08 #338):
- Short text: P50 50ms
- Medium text: P50 111ms
- Long text: P50 215ms

Test thresholds are set with 50% margin to allow for CI variance.
These tests detect major regressions (e.g., 2x+ slowdown).

Run: pytest tests/smoke/test_performance_baseline.py -v
"""

import os
import subprocess
import tempfile
import time
import pytest
from pathlib import Path


# Test sentences of varying lengths (same as perf_snapshot.cpp)
TEST_SENTENCES = {
    "short": "Hello world.",
    "medium": "The quick brown fox jumps over the lazy dog near the river.",
    "long": "In a distant galaxy far away, an ancient civilization built magnificent cities that reached toward the stars with towers of crystal and light.",
}

# Cold-start performance thresholds in milliseconds
# These include model loading (~300ms) + warmup + synthesis
# Cold-start baseline (2025-12-09): ~540ms for all text lengths
# Thresholds set to detect 2x+ regressions (e.g., model loading failure)
LATENCY_THRESHOLDS = {
    "short": 1200,   # Baseline ~540ms, threshold 1200ms (2.2x margin)
    "medium": 1200,  # Baseline ~540ms, threshold 1200ms (2.2x margin)
    "long": 1500,    # Baseline ~560ms, threshold 1500ms (2.7x margin)
}

# Minimum number of iterations for statistical significance
NUM_ITERATIONS = 5


@pytest.fixture(scope="module")
def cpp_binary():
    """Path to stream-tts-cpp binary."""
    binary = Path(__file__).parent.parent.parent / "stream-tts-cpp" / "build" / "stream-tts-cpp"
    if not binary.exists():
        pytest.skip(f"TTS binary not found: {binary}")
    return binary


def get_tts_env():
    """Get environment for TTS processes."""
    env = os.environ.copy()
    return env


def measure_synthesis_latency(binary: Path, text: str, iterations: int = NUM_ITERATIONS) -> list[float]:
    """
    Measure TTS synthesis latency for given text.

    Returns list of latency values in milliseconds.
    Uses one-shot mode (--speak --lang en) with --dry-run to measure synthesis time.
    """
    latencies = []
    env = get_tts_env()

    for i in range(iterations):
        start_time = time.perf_counter()

        result = subprocess.run(
            [str(binary), "--speak", text, "--lang", "en", "--dry-run"],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )

        end_time = time.perf_counter()

        if result.returncode != 0:
            pytest.fail(f"TTS synthesis failed: {result.stderr}")

        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)

    return latencies


def percentile(values: list[float], p: int) -> float:
    """Calculate the p-th percentile of a list of values."""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = min(int(p * (len(sorted_values) - 1) / 100), len(sorted_values) - 1)
    return sorted_values[idx]


class TestPerformanceBaseline:
    """Performance regression detection tests."""

    def test_short_text_latency(self, cpp_binary):
        """Short text synthesis should complete within threshold."""
        text = TEST_SENTENCES["short"]
        threshold = LATENCY_THRESHOLDS["short"]

        latencies = measure_synthesis_latency(cpp_binary, text)
        p50 = percentile(latencies, 50)

        assert p50 < threshold, (
            f"Short text P50 latency ({p50:.0f}ms) exceeds threshold ({threshold}ms). "
            f"Cold-start baseline: ~540ms. Latencies: {[f'{l:.0f}' for l in latencies]}ms"
        )

    def test_medium_text_latency(self, cpp_binary):
        """Medium text synthesis should complete within threshold."""
        text = TEST_SENTENCES["medium"]
        threshold = LATENCY_THRESHOLDS["medium"]

        latencies = measure_synthesis_latency(cpp_binary, text)
        p50 = percentile(latencies, 50)

        assert p50 < threshold, (
            f"Medium text P50 latency ({p50:.0f}ms) exceeds threshold ({threshold}ms). "
            f"Cold-start baseline: ~540ms. Latencies: {[f'{l:.0f}' for l in latencies]}ms"
        )

    def test_long_text_latency(self, cpp_binary):
        """Long text synthesis should complete within threshold."""
        text = TEST_SENTENCES["long"]
        threshold = LATENCY_THRESHOLDS["long"]

        latencies = measure_synthesis_latency(cpp_binary, text)
        p50 = percentile(latencies, 50)

        assert p50 < threshold, (
            f"Long text P50 latency ({p50:.0f}ms) exceeds threshold ({threshold}ms). "
            f"Cold-start baseline: ~560ms. Latencies: {[f'{l:.0f}' for l in latencies]}ms"
        )

    def test_synthesis_consistent(self, cpp_binary):
        """Synthesis latency should be consistent (low variance)."""
        text = TEST_SENTENCES["medium"]

        latencies = measure_synthesis_latency(cpp_binary, text, iterations=10)

        mean_latency = sum(latencies) / len(latencies)
        variance = sum((l - mean_latency) ** 2 for l in latencies) / len(latencies)
        stddev = variance ** 0.5
        cv = stddev / mean_latency if mean_latency > 0 else 0

        # Coefficient of variation should be <50% (reasonable consistency)
        assert cv < 0.5, (
            f"Latency variance too high. CV: {cv:.2%}, StdDev: {stddev:.0f}ms, "
            f"Mean: {mean_latency:.0f}ms. Latencies: {[f'{l:.0f}' for l in latencies]}ms"
        )
