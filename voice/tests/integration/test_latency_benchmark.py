"""
Latency Benchmark Tests - Quick performance validation for CI

This module provides latency benchmarks that can run periodically
to catch performance regressions. Each test spawns subprocess calls
to the TTS binary, so measurements include model loading overhead (~7s).

IMPORTANT: These tests measure subprocess cold-start latency, NOT warm
TTS latency. For true warm latency (50-230ms), use the perf_snapshot C++
tool which keeps the model loaded between requests.

Repo Audit Item #12: Integrate latency benchmarks into periodic testing.

Usage:
    pytest tests/integration/test_latency_benchmark.py -v
    make test-latency-quick

Metrics tracked:
    - Cold-start subprocess latency: ~7-8s per request (includes model loading)
    - End-to-end subprocess latency: Time from subprocess spawn to completion
    - Latency percentiles: P50, P90, P99

For warm latency (model already loaded):
    ./stream-tts-cpp/build/perf_snapshot --capture baseline.json
"""

import json
import os
import pytest
import statistics
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
STREAM_TTS_CPP = PROJECT_ROOT / "stream-tts-cpp"
BUILD_DIR = STREAM_TTS_CPP / "build"
CONFIG_DIR = STREAM_TTS_CPP / "config"
REPORTS_DIR = PROJECT_ROOT / "reports" / "benchmarks"


def get_tts_env():
    """Get environment variables for TTS subprocess."""
    env = os.environ.copy()
    # Note: PYTORCH_ENABLE_MPS_FALLBACK not needed with PyTorch 2.9.1+
    # torch.angle() and other ops now have native MPS support
    # Fix OpenMP duplicate library crash when llama.cpp + libtorch both link OpenMP
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    return env


def run_tts_benchmark(
    binary: Path,
    text: str,
    config: Path,
    output_path: Optional[Path] = None,
    timeout: int = 60
) -> Tuple[bool, float, str]:
    """
    Run a single TTS synthesis and measure end-to-end latency.

    Returns:
        Tuple of (success, latency_seconds, error_message)
    """
    escaped_text = text.replace('"', '\\"').replace('\n', '\\n')
    input_json = f'{{"type":"content_block_delta","delta":{{"type":"text_delta","text":"{escaped_text}"}}}}'

    cmd = [str(binary)]
    if output_path:
        cmd.extend(["--save-audio", str(output_path)])
    cmd.append(str(config))

    start_time = time.perf_counter()
    try:
        result = subprocess.run(
            cmd,
            input=input_json.encode('utf-8'),
            capture_output=True,
            timeout=timeout,
            cwd=str(STREAM_TTS_CPP),
            env=get_tts_env()
        )
        elapsed = time.perf_counter() - start_time

        if result.returncode != 0:
            return False, elapsed, result.stderr.decode('utf-8', errors='replace')[:200]

        if output_path and (not output_path.exists() or output_path.stat().st_size < 100):
            return False, elapsed, "Audio file not generated or too small"

        return True, elapsed, ""

    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - start_time
        return False, elapsed, f"Timeout after {timeout}s"
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        return False, elapsed, str(e)


def calculate_percentiles(latencies: List[float]) -> Dict[str, float]:
    """Calculate latency percentiles."""
    if not latencies:
        return {}

    sorted_latencies = sorted(latencies)
    n = len(sorted_latencies)

    return {
        "min": sorted_latencies[0],
        "p50": sorted_latencies[n // 2],
        "p90": sorted_latencies[int(n * 0.9)],
        "p95": sorted_latencies[int(n * 0.95)] if n >= 20 else sorted_latencies[-1],
        "p99": sorted_latencies[int(n * 0.99)] if n >= 100 else sorted_latencies[-1],
        "max": sorted_latencies[-1],
        "mean": statistics.mean(sorted_latencies),
        "stddev": statistics.stdev(sorted_latencies) if n > 1 else 0.0,
    }


def save_benchmark_report(results: Dict, output_dir: Optional[Path] = None) -> Optional[Path]:
    """Save benchmark results to JSON file for tracking.

    Args:
        results: Benchmark results dict to save
        output_dir: Directory to save report. If None, uses temp directory
                   unless SAVE_BENCHMARK_TO_REPO env var is set.

    Returns:
        Path to saved report, or None if save failed
    """
    # Default to temp dir for hermetic tests (TF2 fix)
    # Set SAVE_BENCHMARK_TO_REPO=1 to save to reports/benchmarks/
    if output_dir is None:
        if os.environ.get("SAVE_BENCHMARK_TO_REPO"):
            output_dir = REPORTS_DIR
        else:
            output_dir = Path(tempfile.gettempdir()) / "voice_benchmarks"

    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    report_path = output_dir / f"latency_benchmark_{timestamp}.json"

    results["timestamp"] = timestamp
    results["generated_at"] = datetime.now().isoformat()

    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    return report_path


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def tts_binary():
    """Path to stream-tts-cpp binary."""
    binary = BUILD_DIR / "stream-tts-cpp"
    if not binary.exists():
        pytest.fail(
            f"TTS binary not found at {binary}.\n"
            "Run 'make build' to compile the C++ code."
        )
    return binary


@pytest.fixture(scope="module")
def english_config():
    """Path to English TTS config."""
    config = CONFIG_DIR / "kokoro-mps-en.yaml"
    if not config.exists():
        pytest.fail(f"English config not found: {config}")
    return config


# =============================================================================
# Latency Benchmark Tests
# =============================================================================

# Benchmark sentences - varied lengths for comprehensive testing
BENCHMARK_SENTENCES = [
    "Hello.",                           # Very short (warm-up)
    "Hello world.",                     # Short
    "The quick brown fox jumps.",       # Medium short
    "Testing the text to speech system today.",  # Medium
    "This is a longer sentence designed to test the TTS pipeline thoroughly.",  # Long
]


@pytest.mark.integration
class TestLatencyBenchmark:
    """
    Quick latency benchmarks for periodic CI runs.

    These tests are faster than full stress tests but still provide
    meaningful performance metrics for regression detection.
    """

    def test_warmup_latency(self, tts_binary, english_config, tmp_path):
        """
        Measure warm-up latency (first request after cold start).

        This is typically the worst-case latency as models are loaded.
        """
        wav_file = tmp_path / "warmup.wav"

        success, latency, error = run_tts_benchmark(
            tts_binary, "Hello world.", english_config, wav_file, timeout=120
        )

        print(f"\nWarm-up latency: {latency:.3f}s")

        assert success, f"Warmup TTS failed: {error}"
        # Warm-up can be slow due to model loading, but should complete
        assert latency < 60, f"Warmup took too long: {latency:.1f}s"

    def test_subprocess_latency_10_requests(self, tts_binary, english_config, tmp_path):
        """
        Measure latency over 10 sequential subprocess calls.

        IMPORTANT: Each call spawns a new process, so this measures cold-start
        latency (model loading + synthesis), NOT warm latency. For true warm
        latency measurement, use the perf_snapshot C++ tool which keeps the
        model loaded between requests.

        Target: P50 < 10s (cold start per request), P99 < 15s
        """
        latencies = []
        errors = []

        # Run 10 requests sequentially
        print("\n=== Warm Latency Benchmark (10 requests) ===")
        for i, text in enumerate(BENCHMARK_SENTENCES * 2):  # 10 requests
            wav_file = tmp_path / f"warm_{i}.wav"

            success, latency, error = run_tts_benchmark(
                tts_binary, text, english_config, wav_file, timeout=60
            )

            if success:
                latencies.append(latency)
                print(f"  [{i}] {latency:.3f}s - '{text[:30]}...'")
            else:
                errors.append(f"[{i}] {error}")
                print(f"  [{i}] FAIL - {error}")

        # Calculate percentiles
        if latencies:
            percentiles = calculate_percentiles(latencies)
            print(f"\nLatency Statistics:")
            print(f"  P50:  {percentiles['p50']:.3f}s")
            print(f"  P90:  {percentiles['p90']:.3f}s")
            print(f"  P99:  {percentiles['p99']:.3f}s (same as max for N=10)")
            print(f"  Mean: {percentiles['mean']:.3f}s")
            print(f"  Min:  {percentiles['min']:.3f}s")
            print(f"  Max:  {percentiles['max']:.3f}s")

        # Assert reasonable performance
        success_rate = len(latencies) / 10
        assert success_rate >= 0.9, f"Too many failures: {1 - success_rate:.0%}"

        if latencies:
            percentiles = calculate_percentiles(latencies)
            # Note: Each subprocess cold-starts the model (~7s), so thresholds
            # reflect cold-start overhead, not warm TTS latency
            assert percentiles["p50"] < 10.0, f"P50 too high: {percentiles['p50']:.1f}s (cold start)"
            assert percentiles["max"] < 20.0, f"Max latency too high: {percentiles['max']:.1f}s"

    def test_short_text_latency(self, tts_binary, english_config, tmp_path):
        """
        Measure latency for short text (minimum viable input).

        Note: Each request spawns a subprocess, so this measures cold-start
        overhead (~7s) plus short text synthesis time.
        """
        short_texts = ["Hi.", "Yes.", "No.", "Ok.", "Hello."]
        latencies = []

        print("\n=== Short Text Latency ===")
        for i, text in enumerate(short_texts):
            wav_file = tmp_path / f"short_{i}.wav"
            success, latency, error = run_tts_benchmark(
                tts_binary, text, english_config, wav_file, timeout=30
            )
            if success:
                latencies.append(latency)
                print(f"  '{text}': {latency:.3f}s")
            else:
                print(f"  '{text}': FAIL - {error}")

        assert len(latencies) >= 4, f"Too many short text failures"

        mean_latency = statistics.mean(latencies)
        print(f"\nMean short text latency: {mean_latency:.3f}s")

        # Each subprocess cold-starts (~7s), so threshold accounts for that
        assert mean_latency < 10.0, f"Short text too slow: {mean_latency:.1f}s (includes cold start)"

    def test_latency_benchmark_report(self, tts_binary, english_config, tmp_path):
        """
        Run full benchmark and generate JSON report for tracking.

        This test produces a machine-readable report that can be used
        for historical tracking and regression detection.
        """
        print("\n=== Full Latency Benchmark Report ===")

        # Collect 20 samples for better statistics
        latencies_by_length = {"short": [], "medium": [], "long": []}

        sentences = {
            "short": ["Hello.", "Testing.", "Nice work.", "Good job.", "Well done."],
            "medium": ["The quick brown fox jumps.", "Testing the TTS system.",
                       "How are you today?", "What time is it?", "Where are we?"],
            "long": [
                "This is a longer sentence to test the system thoroughly.",
                "The text to speech pipeline handles various inputs well.",
                "Performance benchmarks help identify regressions early.",
                "Quality assurance is important for production systems.",
                "Machine learning models require careful optimization.",
            ],
        }

        for category, texts in sentences.items():
            for i, text in enumerate(texts):
                wav_file = tmp_path / f"bench_{category}_{i}.wav"
                success, latency, error = run_tts_benchmark(
                    tts_binary, text, english_config, wav_file, timeout=60
                )
                if success:
                    latencies_by_length[category].append(latency)

        # Build report
        report = {
            "test_type": "latency_benchmark",
            "samples_per_category": 5,
            "categories": {},
            "overall": {},
        }

        all_latencies = []
        for category, latencies in latencies_by_length.items():
            if latencies:
                percentiles = calculate_percentiles(latencies)
                report["categories"][category] = {
                    "samples": len(latencies),
                    "percentiles": percentiles,
                }
                all_latencies.extend(latencies)
                print(f"\n{category.upper()} ({len(latencies)} samples):")
                print(f"  P50: {percentiles['p50']:.3f}s, Mean: {percentiles['mean']:.3f}s")

        if all_latencies:
            report["overall"] = {
                "total_samples": len(all_latencies),
                "percentiles": calculate_percentiles(all_latencies),
            }

            print(f"\nOVERALL ({len(all_latencies)} samples):")
            overall = report["overall"]["percentiles"]
            print(f"  P50: {overall['p50']:.3f}s")
            print(f"  P90: {overall['p90']:.3f}s")
            print(f"  Mean: {overall['mean']:.3f}s")

        # Save report
        report_path = save_benchmark_report(report)
        print(f"\nReport saved: {report_path}")

        # Basic assertions (cold-start per subprocess ~7s)
        assert len(all_latencies) >= 10, "Not enough successful samples"
        overall_p50 = report["overall"]["percentiles"]["p50"]
        assert overall_p50 < 10.0, f"Overall P50 too high: {overall_p50:.1f}s (includes cold start)"


@pytest.mark.integration
class TestLatencyTargets:
    """
    Tests against specific latency targets for quality gates.

    These tests verify that the system meets performance requirements.

    IMPORTANT: These tests spawn subprocesses, so each request includes model
    loading overhead (~7s). For true warm latency measurement (50-230ms), use
    the perf_snapshot C++ tool which keeps the model loaded in memory.
    """

    # Latency targets account for subprocess cold-start overhead
    SUBPROCESS_P50_TARGET = 10.0  # seconds (includes ~7s model loading)
    SUBPROCESS_P99_TARGET = 15.0  # seconds
    COLD_START_TARGET = 60.0  # seconds (includes model loading)

    def test_meets_p50_target(self, tts_binary, english_config, tmp_path):
        """
        Verify subprocess P50 latency meets target.

        Note: Each subprocess cold-starts the model (~7s overhead).
        """
        # First request (warmup - but still cold start for subprocess)
        run_tts_benchmark(tts_binary, "Warmup.", english_config, tmp_path / "warmup.wav")

        latencies = []
        for i in range(5):
            success, latency, _ = run_tts_benchmark(
                tts_binary,
                "Testing latency target.",
                english_config,
                tmp_path / f"p50_{i}.wav"
            )
            if success:
                latencies.append(latency)

        assert len(latencies) >= 4, "Too many failures for P50 test"

        p50 = sorted(latencies)[len(latencies) // 2]
        print(f"\nP50 latency: {p50:.3f}s (target: <{self.SUBPROCESS_P50_TARGET}s, includes cold start)")

        assert p50 < self.SUBPROCESS_P50_TARGET, \
            f"P50 latency {p50:.1f}s exceeds target {self.SUBPROCESS_P50_TARGET}s (includes cold start)"


@pytest.mark.integration
class TestPerfSnapshotGate:
    """
    Warm latency regression gate using perf_snapshot C++ tool.

    Unlike the subprocess-based tests above (which measure cold-start latency ~7s),
    perf_snapshot keeps the model loaded in memory and measures true warm latency
    (50-230ms range).

    Phase 9.4: Periodic performance gate to catch latency regressions.

    The baseline is stored in reports/main/perf_snapshot_baseline.json and should
    be updated when intentional performance changes are made.
    """

    @pytest.fixture(scope="class")
    def perf_binary(self):
        """Path to perf_snapshot binary."""
        binary = BUILD_DIR / "perf_snapshot"
        if not binary.exists():
            pytest.skip(f"perf_snapshot not built: {binary}")
        return binary

    @pytest.fixture(scope="class")
    def baseline_path(self):
        """Path to performance baseline JSON."""
        # Primary: look for explicit baseline file
        baseline = PROJECT_ROOT / "reports" / "main" / "perf_snapshot_baseline.json"
        if baseline.exists():
            return baseline

        # Fallback: use most recent snapshot
        snapshot_dir = PROJECT_ROOT / "reports" / "main"
        snapshots = sorted(snapshot_dir.glob("perf_snapshot_*.json"), reverse=True)
        if snapshots:
            return snapshots[0]

        pytest.skip("No performance baseline found")

    def test_perf_gate_no_regression(self, perf_binary, baseline_path, tmp_path):
        """
        Run perf_snapshot --compare against baseline and fail on regression.

        Threshold: 20% slowdown = REGRESSION (defined in perf_snapshot.cpp)

        This test:
        1. Loads the Kokoro model
        2. Runs 3 warmup iterations
        3. Runs 20 benchmark iterations per sentence (short/medium/long)
        4. Compares P95 latencies against baseline
        5. Fails if any P95 exceeds baseline by >20%
        """
        print(f"\nUsing baseline: {baseline_path}")
        print(f"Binary: {perf_binary}")

        env = os.environ.copy()
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        result = subprocess.run(
            [str(perf_binary), "--compare", str(baseline_path)],
            capture_output=True,
            timeout=300,  # 5 minutes for warmup + benchmark
            env=env
        )

        stdout = result.stdout.decode("utf-8", errors="replace")
        stderr = result.stderr.decode("utf-8", errors="replace")

        print("\n=== perf_snapshot output ===")
        print(stdout)
        if stderr:
            print("\n=== stderr ===")
            print(stderr)

        # perf_snapshot returns 0 for no regression, 1 for regression
        if result.returncode != 0:
            # Extract regression details from output
            if "REGRESSION DETECTED" in stdout:
                pytest.fail(
                    f"Performance regression detected!\n"
                    f"Baseline: {baseline_path}\n"
                    f"Output:\n{stdout}\n"
                    f"To update baseline after intentional changes:\n"
                    f"  {perf_binary} --capture {baseline_path}"
                )
            else:
                pytest.fail(
                    f"perf_snapshot failed with exit code {result.returncode}\n"
                    f"stdout: {stdout[:500]}\n"
                    f"stderr: {stderr[:500]}"
                )

        assert "NO REGRESSION" in stdout, \
            f"Expected 'NO REGRESSION' in output but got:\n{stdout}"

    def test_perf_snapshot_capture_works(self, perf_binary, tmp_path):
        """
        Verify perf_snapshot --capture produces valid JSON.

        This is a sanity check that the capture functionality works,
        useful for setting up new baselines.
        """
        output_file = tmp_path / "test_snapshot.json"

        env = os.environ.copy()
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        result = subprocess.run(
            [str(perf_binary), "--capture", str(output_file)],
            capture_output=True,
            timeout=300,
            env=env
        )

        stdout = result.stdout.decode("utf-8", errors="replace")
        print("\n=== Capture output ===")
        print(stdout[-2000:] if len(stdout) > 2000 else stdout)

        assert result.returncode == 0, \
            f"Capture failed: {result.stderr.decode()[:500]}"

        assert output_file.exists(), "Output file not created"

        # Verify JSON is valid and has expected structure
        import json
        with open(output_file) as f:
            snapshot = json.load(f)

        assert "synthesis" in snapshot, "Missing synthesis stats"
        assert "end_to_end" in snapshot, "Missing end_to_end stats"
        assert "short" in snapshot["synthesis"], "Missing short sentence stats"
        assert "p50_ms" in snapshot["end_to_end"], "Missing P50 metric"

        print(f"\nCaptured snapshot P50: {snapshot['end_to_end']['p50_ms']:.1f}ms")
        print(f"Captured snapshot P95: {snapshot['end_to_end']['p95_ms']:.1f}ms")
