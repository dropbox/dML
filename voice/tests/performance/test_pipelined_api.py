"""
Pytest wrapper for PipelinedTTSEngine performance measurement.

Worker #243: Added proper pytest test for pipelined API now that MPS concurrency is fixed (Worker #242).

This test:
1. Runs the C++ test_pipelined_tts binary
2. Parses and validates the output
3. Reports key metrics (throughput, latency, prefetch hit rate)
4. Fails if performance requirements are not met
"""

import subprocess
import re
import pytest
from pathlib import Path


# Test configuration
BINARY_PATH = Path(__file__).parent.parent.parent / "stream-tts-cpp" / "build" / "test_pipelined_tts"
CONFIG_PATH = Path(__file__).parent.parent.parent / "stream-tts-cpp" / "config" / "kokoro-mps-en.yaml"

# Performance thresholds
MIN_BASELINE_THROUGHPUT = 2.0  # req/s (conservative)
MIN_PIPELINED_THROUGHPUT = 3.0  # req/s (conservative)
MAX_AVG_LATENCY_MS = 500  # ms (post-warmup)
MIN_IMPROVEMENT_PCT = 10  # % improvement required
MIN_BURST_PREFETCH_HIT_RATE = 50  # % (burst mode should have high prefetch hits)


def parse_summary_metrics(output: str) -> dict:
    """Parse key metrics from test_pipelined_tts output."""
    metrics = {}

    # Parse throughput values
    patterns = {
        "baseline_throughput": r"Baseline throughput:\s+(\d+\.\d+) req/s",
        "baseline_latency": r"Baseline avg latency:\s+(\d+\.\d+) ms/request",
        "pipelined_seq_throughput": r"Pipelined seq throughput:\s+(\d+\.\d+) req/s",
        "pipelined_seq_latency": r"Pipelined seq avg latency:\s+(\d+\.\d+) ms/request",
        "pipelined_burst_throughput": r"Pipelined burst throughput:\s+(\d+\.\d+) req/s",
        "pipelined_seq_prefetch_hit": r"Pipelined seq prefetch hit:\s+(\d+\.\d+)%",
        "pipelined_burst_prefetch_hit": r"Pipelined burst prefetch hit:\s+(\d+\.\d+)%",
        "pipelined_seq_improvement": r"Pipelined seq improvement:\s+([+-]?\d+\.\d+)%",
        "pipelined_burst_improvement": r"Pipelined burst improvement:\s+([+-]?\d+\.\d+)%",
        "avg_g2p_time_ms": r"Avg G2P time:\s+(\d+) ms",
        "avg_inference_time_ms": r"Avg inference time:\s+(\d+) ms",
        "total_prep_overlap_ms": r"Total prep overlap:\s+(\d+) ms",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            metrics[key] = float(match.group(1))

    # Check for test pass/fail
    metrics["passed"] = "ALL TESTS PASSED" in output

    return metrics


@pytest.fixture(scope="module")
def pipelined_test_output():
    """Run test_pipelined_tts once and cache the output."""
    if not BINARY_PATH.exists():
        pytest.skip(f"Binary not found: {BINARY_PATH}")

    if not CONFIG_PATH.exists():
        pytest.skip(f"Config not found: {CONFIG_PATH}")

    result = subprocess.run(
        [str(BINARY_PATH), str(CONFIG_PATH)],
        capture_output=True,
        text=True,
        timeout=300  # 5 minutes max
    )

    # Combine stdout and stderr (spdlog writes to stdout by default)
    output = result.stdout + result.stderr

    return {
        "returncode": result.returncode,
        "output": output,
        "metrics": parse_summary_metrics(output)
    }


class TestPipelinedTTSAPI:
    """Test suite for PipelinedTTSEngine performance."""

    def test_binary_runs_successfully(self, pipelined_test_output):
        """Verify the C++ test binary runs without crashing."""
        assert pipelined_test_output["returncode"] == 0, \
            f"test_pipelined_tts exited with code {pipelined_test_output['returncode']}"

    def test_all_cpp_tests_pass(self, pipelined_test_output):
        """Verify all C++ internal tests pass."""
        assert pipelined_test_output["metrics"]["passed"], \
            "C++ test reported failures"

    def test_baseline_throughput(self, pipelined_test_output):
        """Verify baseline sequential throughput meets minimum."""
        metrics = pipelined_test_output["metrics"]
        assert "baseline_throughput" in metrics, "Could not parse baseline throughput"

        throughput = metrics["baseline_throughput"]
        print(f"Baseline throughput: {throughput:.2f} req/s")

        assert throughput >= MIN_BASELINE_THROUGHPUT, \
            f"Baseline throughput {throughput:.2f} < {MIN_BASELINE_THROUGHPUT} req/s"

    def test_pipelined_throughput(self, pipelined_test_output):
        """Verify pipelined sequential throughput meets minimum."""
        metrics = pipelined_test_output["metrics"]
        assert "pipelined_seq_throughput" in metrics, "Could not parse pipelined throughput"

        throughput = metrics["pipelined_seq_throughput"]
        print(f"Pipelined sequential throughput: {throughput:.2f} req/s")

        assert throughput >= MIN_PIPELINED_THROUGHPUT, \
            f"Pipelined throughput {throughput:.2f} < {MIN_PIPELINED_THROUGHPUT} req/s"

    def test_latency_requirement(self, pipelined_test_output):
        """Verify both baseline and pipelined meet latency requirement."""
        metrics = pipelined_test_output["metrics"]

        if "baseline_latency" in metrics:
            assert metrics["baseline_latency"] <= MAX_AVG_LATENCY_MS, \
                f"Baseline latency {metrics['baseline_latency']:.1f}ms > {MAX_AVG_LATENCY_MS}ms"

        if "pipelined_seq_latency" in metrics:
            assert metrics["pipelined_seq_latency"] <= MAX_AVG_LATENCY_MS, \
                f"Pipelined latency {metrics['pipelined_seq_latency']:.1f}ms > {MAX_AVG_LATENCY_MS}ms"

    def test_pipelined_improves_throughput(self, pipelined_test_output):
        """Verify pipelined mode shows throughput improvement over baseline."""
        metrics = pipelined_test_output["metrics"]
        assert "pipelined_seq_improvement" in metrics, "Could not parse improvement percentage"

        improvement = metrics["pipelined_seq_improvement"]
        print(f"Pipelined improvement: {improvement:+.1f}%")

        # Allow for some variance - improvement should be positive
        assert improvement >= MIN_IMPROVEMENT_PCT, \
            f"Pipelined improvement {improvement:.1f}% < {MIN_IMPROVEMENT_PCT}%"

    def test_burst_prefetch_hit_rate(self, pipelined_test_output):
        """Verify burst mode achieves high prefetch hit rate."""
        metrics = pipelined_test_output["metrics"]
        assert "pipelined_burst_prefetch_hit" in metrics, "Could not parse prefetch hit rate"

        hit_rate = metrics["pipelined_burst_prefetch_hit"]
        print(f"Burst mode prefetch hit rate: {hit_rate:.1f}%")

        assert hit_rate >= MIN_BURST_PREFETCH_HIT_RATE, \
            f"Burst prefetch hit rate {hit_rate:.1f}% < {MIN_BURST_PREFETCH_HIT_RATE}%"

    def test_report_all_metrics(self, pipelined_test_output):
        """Report all parsed metrics (always passes, for visibility)."""
        metrics = pipelined_test_output["metrics"]

        print("\n" + "=" * 60)
        print("  PipelinedTTSEngine Performance Report")
        print("=" * 60)

        print("\n  Baseline (KokoroTorchScriptTTS):")
        if "baseline_throughput" in metrics:
            print(f"    Throughput:         {metrics['baseline_throughput']:.2f} req/s")
        if "baseline_latency" in metrics:
            print(f"    Avg latency:        {metrics['baseline_latency']:.1f} ms")

        print("\n  Pipelined Sequential:")
        if "pipelined_seq_throughput" in metrics:
            print(f"    Throughput:         {metrics['pipelined_seq_throughput']:.2f} req/s")
        if "pipelined_seq_latency" in metrics:
            print(f"    Avg latency:        {metrics['pipelined_seq_latency']:.1f} ms")
        if "pipelined_seq_improvement" in metrics:
            print(f"    Improvement:        {metrics['pipelined_seq_improvement']:+.1f}%")
        if "pipelined_seq_prefetch_hit" in metrics:
            print(f"    Prefetch hit rate:  {metrics['pipelined_seq_prefetch_hit']:.1f}%")

        print("\n  Pipelined Burst:")
        if "pipelined_burst_throughput" in metrics:
            print(f"    Throughput:         {metrics['pipelined_burst_throughput']:.2f} req/s")
        if "pipelined_burst_improvement" in metrics:
            print(f"    Improvement:        {metrics['pipelined_burst_improvement']:+.1f}%")
        if "pipelined_burst_prefetch_hit" in metrics:
            print(f"    Prefetch hit rate:  {metrics['pipelined_burst_prefetch_hit']:.1f}%")

        print("\n  Internal Metrics:")
        if "avg_g2p_time_ms" in metrics:
            print(f"    Avg G2P time:       {metrics['avg_g2p_time_ms']:.0f} ms")
        if "avg_inference_time_ms" in metrics:
            print(f"    Avg inference time: {metrics['avg_inference_time_ms']:.0f} ms")
        if "total_prep_overlap_ms" in metrics:
            print(f"    Total prep overlap: {metrics['total_prep_overlap_ms']:.0f} ms")

        print("=" * 60)
