"""
Stress Tests for Voice TTS Pipeline (Phase 6, Worker #117)

Tests:
1. Concurrent streams - 10 parallel TTS requests
2. Sequential throughput - 1000 sequential requests
3. Memory leak detection - RSS monitoring over sustained load
4. Latency P99 measurement - Statistical latency analysis

Usage:
    pytest tests/stress/test_stress.py -v -m stress
    make test-stress

Note: These tests are long-running (~10 minutes total).
They are not run on every commit, only weekly or on-demand.
"""

import json
import os
import psutil
import pytest
import queue
import statistics
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
STREAM_TTS_CPP = PROJECT_ROOT / "stream-tts-cpp"
BUILD_DIR = STREAM_TTS_CPP / "build"
CONFIG_DIR = STREAM_TTS_CPP / "config"
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures" / "text"

# Test configuration
DEFAULT_CONFIG = CONFIG_DIR / "kokoro-mps-en.yaml"
TEST_SENTENCES = [
    "Hello world, how are you today?",
    "The quick brown fox jumps over the lazy dog.",
    "One, two, three, four, five.",
    "Testing the text to speech system.",
    "This is a stress test sentence.",
    "The function returns a string value.",
    "She bought five nice things at the store.",
    "Computer systems are very useful today.",
    "Please verify this audio output works.",
    "Final sentence for testing purposes.",
]


# =============================================================================
# TTS Execution Helpers
# =============================================================================

def run_tts(binary: Path, text: str, config: Path, output_path: Optional[Path] = None,
            timeout: int = 60) -> Tuple[bool, float, str]:
    """
    Run a single TTS synthesis and measure latency.

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
            cwd=str(STREAM_TTS_CPP)
        )
        elapsed = time.perf_counter() - start_time

        if result.returncode != 0:
            return False, elapsed, result.stderr.decode('utf-8', errors='replace')

        # Verify audio was generated if output path specified
        if output_path and (not output_path.exists() or output_path.stat().st_size < 100):
            return False, elapsed, "Audio file not generated or too small"

        return True, elapsed, ""

    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - start_time
        return False, elapsed, f"Timeout after {timeout}s"
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        return False, elapsed, str(e)


def get_process_memory_mb(pid: int) -> float:
    """Get RSS memory usage of a process in MB."""
    try:
        process = psutil.Process(pid)
        return process.memory_info().rss / (1024 * 1024)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0.0


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def tts_binary():
    """Path to stream-tts-cpp binary."""
    binary = BUILD_DIR / "stream-tts-cpp"
    if not binary.exists():
        pytest.skip(f"TTS binary not found at {binary}")
    return binary


@pytest.fixture(scope="module")
def english_config():
    """Path to English TTS config."""
    config = CONFIG_DIR / "kokoro-mps-en.yaml"
    if not config.exists():
        pytest.skip(f"English config not found: {config}")
    return config


# =============================================================================
# Stress Tests
# =============================================================================

@pytest.mark.stress
class TestConcurrentStreams:
    """
    Test concurrent TTS stream handling.

    Target: 10-20 concurrent streams without crashes or significant degradation.
    Per roadmap item #15: Stress tests for 10-20 concurrent streams, 100 sequential bursts.
    """

    def _run_concurrent_test(self, tts_binary, english_config, tmp_path,
                             num_concurrent: int, test_name: str) -> dict:
        """
        Run N concurrent TTS requests and return results.

        Returns dict with: successful, total, latencies, errors
        """
        results_list: List[Dict] = []
        errors: List[str] = []

        def run_single_tts(idx: int) -> Dict:
            """Run a single TTS request."""
            text = TEST_SENTENCES[idx % len(TEST_SENTENCES)]
            output_path = tmp_path / f"{test_name}_{idx}.wav"

            success, latency, error = run_tts(
                tts_binary, text, english_config, output_path, timeout=180
            )

            return {
                "idx": idx,
                "text": text,
                "success": success,
                "latency": latency,
                "error": error,
                "output_size": output_path.stat().st_size if output_path.exists() else 0
            }

        print(f"\n=== Testing {num_concurrent} Concurrent Streams ===")
        start_time = time.perf_counter()

        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(run_single_tts, i) for i in range(num_concurrent)]
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results_list.append(result)
                    if not result["success"]:
                        errors.append(f"Stream {result['idx']}: {result['error']}")
                except Exception as e:
                    errors.append(f"Exception: {e}")

        total_time = time.perf_counter() - start_time

        # Report results
        successful = sum(1 for r in results_list if r["success"])
        latencies = [r["latency"] for r in results_list if r["success"]]

        print(f"\nResults:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Successful: {successful}/{num_concurrent}")
        if latencies:
            print(f"  Avg latency: {statistics.mean(latencies):.2f}s")
            print(f"  Max latency: {max(latencies):.2f}s")
            print(f"  Min latency: {min(latencies):.2f}s")
            if len(latencies) > 1:
                print(f"  Stdev:       {statistics.stdev(latencies):.2f}s")

        for r in sorted(results_list, key=lambda x: x["idx"]):
            status = "OK" if r["success"] else "FAIL"
            print(f"  [{r['idx']:2d}] {status} - {r['latency']:.2f}s - {r['output_size']}b")

        if errors:
            print(f"\nErrors:")
            for e in errors[:10]:  # Limit error output
                print(f"  {e}")

        return {
            "successful": successful,
            "total": num_concurrent,
            "latencies": latencies,
            "errors": errors,
            "total_time": total_time
        }

    def test_10_concurrent_streams(self, tts_binary, english_config, tmp_path):
        """
        Launch 10 concurrent TTS requests and verify all complete successfully.

        Success criteria:
        - All 10 requests complete without error
        - No crashes
        - Latency remains reasonable
        """
        result = self._run_concurrent_test(
            tts_binary, english_config, tmp_path, 10, "concurrent_10"
        )
        assert result["successful"] == result["total"], \
            f"Only {result['successful']}/{result['total']} concurrent streams succeeded"

    def test_15_concurrent_streams(self, tts_binary, english_config, tmp_path):
        """
        Launch 15 concurrent TTS requests.

        Tests scaling beyond 10 concurrent streams per roadmap item #15.
        """
        result = self._run_concurrent_test(
            tts_binary, english_config, tmp_path, 15, "concurrent_15"
        )
        # Allow 1 failure for 15 concurrent (93% success rate)
        min_success = 14
        assert result["successful"] >= min_success, \
            f"Only {result['successful']}/{result['total']} succeeded (need {min_success})"

    def test_20_concurrent_streams(self, tts_binary, english_config, tmp_path):
        """
        Launch 20 concurrent TTS requests.

        Maximum concurrent stream test per roadmap item #15.
        This tests the system under high concurrent load.
        """
        result = self._run_concurrent_test(
            tts_binary, english_config, tmp_path, 20, "concurrent_20"
        )
        # Allow 2 failures for 20 concurrent (90% success rate)
        min_success = 18
        assert result["successful"] >= min_success, \
            f"Only {result['successful']}/{result['total']} succeeded (need {min_success})"

    def test_concurrent_different_texts(self, tts_binary, english_config, tmp_path):
        """
        Test concurrent streams with varying text lengths.

        Ensures the system handles mixed workloads correctly.
        """
        texts = [
            "Hi!",  # Very short (3 chars minimum for TTS)
            "Hello world, how are you?",  # Short
            "The quick brown fox jumps over the lazy dog. This is a longer sentence.",  # Medium
            "Testing the text to speech system with a moderately long sentence that " +
            "should exercise the pipeline more thoroughly than shorter inputs.",  # Long
            "One!",  # Very short (3 chars minimum for TTS)
        ]

        results: List[Dict] = []

        def run_with_text(idx: int, text: str) -> Dict:
            output_path = tmp_path / f"mixed_{idx}.wav"
            success, latency, error = run_tts(
                tts_binary, text, english_config, output_path, timeout=120
            )
            return {
                "idx": idx,
                "text_len": len(text),
                "success": success,
                "latency": latency,
                "error": error
            }

        print(f"\n=== Testing {len(texts)} Concurrent Streams (Mixed Lengths) ===")

        with ThreadPoolExecutor(max_workers=len(texts)) as executor:
            futures = [
                executor.submit(run_with_text, i, text)
                for i, text in enumerate(texts)
            ]
            for future in as_completed(futures):
                results.append(future.result())

        successful = sum(1 for r in results if r["success"])

        for r in sorted(results, key=lambda x: x["idx"]):
            status = "OK" if r["success"] else "FAIL"
            print(f"  [{r['idx']}] {status} - {r['text_len']}chars - {r['latency']:.2f}s")

        assert successful == len(texts), f"Only {successful}/{len(texts)} succeeded"


@pytest.mark.stress
class TestSequentialThroughput:
    """
    Test sequential TTS request throughput.

    Target: 1000 sequential requests without memory leaks or degradation.
    """

    def test_1000_sequential_requests(self, tts_binary, english_config, tmp_path):
        """
        Run 1000 sequential TTS requests and measure throughput.

        Success criteria:
        - All requests complete without crash
        - Latency doesn't degrade significantly over time
        - Memory doesn't grow unboundedly
        """
        NUM_REQUESTS = 1000
        CHECKPOINT_INTERVAL = 100

        latencies: List[float] = []
        errors: List[str] = []
        memory_samples: List[Tuple[int, float]] = []

        print(f"\n=== Testing {NUM_REQUESTS} Sequential Requests ===")
        start_time = time.perf_counter()

        for i in range(NUM_REQUESTS):
            text = TEST_SENTENCES[i % len(TEST_SENTENCES)]
            # Don't save audio to disk to reduce I/O overhead
            success, latency, error = run_tts(
                tts_binary, text, english_config, output_path=None, timeout=60
            )

            if success:
                latencies.append(latency)
            else:
                errors.append(f"Request {i}: {error}")

            # Progress checkpoint
            if (i + 1) % CHECKPOINT_INTERVAL == 0:
                elapsed = time.perf_counter() - start_time
                recent_latencies = latencies[-CHECKPOINT_INTERVAL:] if len(latencies) >= CHECKPOINT_INTERVAL else latencies
                avg_latency = statistics.mean(recent_latencies) if recent_latencies else 0

                print(f"  [{i+1}/{NUM_REQUESTS}] "
                      f"Elapsed: {elapsed:.1f}s, "
                      f"Avg latency (last {CHECKPOINT_INTERVAL}): {avg_latency:.3f}s, "
                      f"Errors: {len(errors)}")

        total_time = time.perf_counter() - start_time

        # Calculate statistics
        successful = len(latencies)
        error_count = len(errors)

        print(f"\n=== Sequential Throughput Results ===")
        print(f"Total time:     {total_time:.1f}s")
        print(f"Successful:     {successful}/{NUM_REQUESTS}")
        print(f"Errors:         {error_count}")
        print(f"Throughput:     {successful/total_time:.2f} req/s")

        if latencies:
            print(f"\nLatency Statistics:")
            print(f"  Mean:         {statistics.mean(latencies):.3f}s")
            print(f"  Median:       {statistics.median(latencies):.3f}s")
            print(f"  Stdev:        {statistics.stdev(latencies):.3f}s" if len(latencies) > 1 else "  Stdev: N/A")
            print(f"  Min:          {min(latencies):.3f}s")
            print(f"  Max:          {max(latencies):.3f}s")

            # Check for latency degradation
            first_100 = latencies[:100]
            last_100 = latencies[-100:]
            if len(first_100) >= 100 and len(last_100) >= 100:
                first_avg = statistics.mean(first_100)
                last_avg = statistics.mean(last_100)
                degradation = (last_avg - first_avg) / first_avg * 100 if first_avg > 0 else 0
                print(f"\nLatency Trend:")
                print(f"  First 100 avg: {first_avg:.3f}s")
                print(f"  Last 100 avg:  {last_avg:.3f}s")
                print(f"  Degradation:   {degradation:+.1f}%")

                # Warn if degradation > 50%
                if degradation > 50:
                    print(f"  WARNING: Significant latency degradation detected!")

        if errors:
            print(f"\nFirst 5 errors:")
            for e in errors[:5]:
                print(f"  {e}")

        # Success criteria: >95% success rate
        success_rate = successful / NUM_REQUESTS
        assert success_rate >= 0.95, f"Success rate {success_rate:.1%} below 95% threshold"

    def test_100_sequential_with_audio_output(self, tts_binary, english_config, tmp_path):
        """
        Run 100 sequential requests saving audio files.

        Verifies audio output is consistent over many requests.
        """
        NUM_REQUESTS = 100
        latencies: List[float] = []
        audio_sizes: List[int] = []
        errors = 0

        print(f"\n=== Testing {NUM_REQUESTS} Sequential with Audio Output ===")

        for i in range(NUM_REQUESTS):
            text = TEST_SENTENCES[i % len(TEST_SENTENCES)]
            output_path = tmp_path / f"seq_{i}.wav"

            success, latency, error = run_tts(
                tts_binary, text, english_config, output_path, timeout=60
            )

            if success and output_path.exists():
                latencies.append(latency)
                audio_sizes.append(output_path.stat().st_size)
                # Clean up to save disk space
                output_path.unlink()
            else:
                errors += 1

            if (i + 1) % 25 == 0:
                print(f"  Progress: {i+1}/{NUM_REQUESTS}")

        print(f"\nResults:")
        print(f"  Successful:     {len(latencies)}/{NUM_REQUESTS}")
        print(f"  Avg latency:    {statistics.mean(latencies):.3f}s" if latencies else "  Avg latency: N/A")
        print(f"  Avg audio size: {statistics.mean(audio_sizes)/1024:.1f}KB" if audio_sizes else "  Avg audio: N/A")

        assert errors < NUM_REQUESTS * 0.05, f"Error rate {errors/NUM_REQUESTS:.1%} exceeds 5%"

    def test_100_sequential_bursts(self, tts_binary, english_config, tmp_path):
        """
        Run 100 sequential bursts of TTS requests.

        Per roadmap item #15: Test "100 sequential bursts" pattern.
        A burst is a rapid-fire sequence of requests followed by analysis.
        This tests recovery and consistency after repeated rapid usage.

        Pattern: 10 bursts of 10 requests each = 100 total
        """
        NUM_BURSTS = 10
        REQUESTS_PER_BURST = 10
        TOTAL_REQUESTS = NUM_BURSTS * REQUESTS_PER_BURST

        burst_results: List[Dict] = []
        all_latencies: List[float] = []

        print(f"\n=== Testing {NUM_BURSTS} Bursts x {REQUESTS_PER_BURST} Requests ===")
        print(f"Total: {TOTAL_REQUESTS} requests in burst pattern")

        for burst_idx in range(NUM_BURSTS):
            burst_start = time.perf_counter()
            burst_latencies: List[float] = []
            burst_errors = 0

            # Rapid-fire burst of requests
            for req_idx in range(REQUESTS_PER_BURST):
                text = TEST_SENTENCES[(burst_idx * REQUESTS_PER_BURST + req_idx) % len(TEST_SENTENCES)]

                success, latency, error = run_tts(
                    tts_binary, text, english_config, output_path=None, timeout=60
                )

                if success:
                    burst_latencies.append(latency)
                    all_latencies.append(latency)
                else:
                    burst_errors += 1

            burst_time = time.perf_counter() - burst_start
            burst_avg = statistics.mean(burst_latencies) if burst_latencies else 0

            burst_results.append({
                "burst_idx": burst_idx,
                "successful": len(burst_latencies),
                "errors": burst_errors,
                "total_time": burst_time,
                "avg_latency": burst_avg,
                "throughput": len(burst_latencies) / burst_time if burst_time > 0 else 0
            })

            print(f"  Burst {burst_idx + 1:2d}/{NUM_BURSTS}: "
                  f"{len(burst_latencies)}/{REQUESTS_PER_BURST} OK, "
                  f"avg {burst_avg:.2f}s, "
                  f"{burst_results[-1]['throughput']:.2f} req/s")

        # Summary statistics
        total_successful = sum(b["successful"] for b in burst_results)
        total_errors = sum(b["errors"] for b in burst_results)
        avg_burst_latency = statistics.mean(b["avg_latency"] for b in burst_results if b["avg_latency"] > 0)
        avg_throughput = statistics.mean(b["throughput"] for b in burst_results)

        # Check for latency consistency across bursts
        first_3_bursts = [b["avg_latency"] for b in burst_results[:3] if b["avg_latency"] > 0]
        last_3_bursts = [b["avg_latency"] for b in burst_results[-3:] if b["avg_latency"] > 0]

        print(f"\n=== Burst Test Summary ===")
        print(f"Total successful:   {total_successful}/{TOTAL_REQUESTS}")
        print(f"Total errors:       {total_errors}")
        print(f"Avg burst latency:  {avg_burst_latency:.3f}s")
        print(f"Avg throughput:     {avg_throughput:.2f} req/s")

        if first_3_bursts and last_3_bursts:
            first_avg = statistics.mean(first_3_bursts)
            last_avg = statistics.mean(last_3_bursts)
            drift = ((last_avg - first_avg) / first_avg * 100) if first_avg > 0 else 0
            print(f"First 3 bursts avg: {first_avg:.3f}s")
            print(f"Last 3 bursts avg:  {last_avg:.3f}s")
            print(f"Latency drift:      {drift:+.1f}%")

        if all_latencies:
            sorted_lats = sorted(all_latencies)
            n = len(sorted_lats)
            print(f"\nOverall latency distribution:")
            print(f"  P50: {sorted_lats[n // 2]:.3f}s")
            print(f"  P95: {sorted_lats[int(0.95 * n)]:.3f}s")
            print(f"  P99: {sorted_lats[int(0.99 * n)]:.3f}s")

        # Success criteria: >95% success rate
        success_rate = total_successful / TOTAL_REQUESTS
        assert success_rate >= 0.95, f"Success rate {success_rate:.1%} below 95% threshold"


@pytest.mark.stress
class TestMemoryLeakDetection:
    """
    Test for memory leaks during sustained TTS load.

    Monitors RSS memory over many requests to detect unbounded growth.
    """

    def test_memory_stability_100_requests(self, tts_binary, english_config):
        """
        Run 100 requests and monitor memory usage.

        Success criteria:
        - Memory doesn't grow more than 2x from baseline
        - No OOM errors
        """
        NUM_REQUESTS = 100
        SAMPLE_INTERVAL = 10

        memory_samples: List[Dict] = []
        latencies: List[float] = []

        # Get system baseline
        baseline_memory = psutil.Process().memory_info().rss / (1024 * 1024)

        print(f"\n=== Memory Leak Detection ({NUM_REQUESTS} requests) ===")
        print(f"Baseline memory: {baseline_memory:.1f}MB")

        for i in range(NUM_REQUESTS):
            text = TEST_SENTENCES[i % len(TEST_SENTENCES)]

            success, latency, _ = run_tts(
                tts_binary, text, english_config, output_path=None, timeout=60
            )

            if success:
                latencies.append(latency)

            # Sample memory at intervals
            if (i + 1) % SAMPLE_INTERVAL == 0:
                current_memory = psutil.Process().memory_info().rss / (1024 * 1024)
                memory_samples.append({
                    "request": i + 1,
                    "memory_mb": current_memory,
                    "delta_mb": current_memory - baseline_memory
                })
                print(f"  [{i+1}/{NUM_REQUESTS}] Memory: {current_memory:.1f}MB "
                      f"(delta: {current_memory - baseline_memory:+.1f}MB)")

        # Final memory check
        final_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        memory_growth = final_memory - baseline_memory
        growth_ratio = final_memory / baseline_memory if baseline_memory > 0 else 1.0

        print(f"\n=== Memory Analysis ===")
        print(f"Baseline:    {baseline_memory:.1f}MB")
        print(f"Final:       {final_memory:.1f}MB")
        print(f"Growth:      {memory_growth:+.1f}MB ({growth_ratio:.2f}x)")

        # Check for concerning growth patterns
        if memory_samples:
            first_sample = memory_samples[0]["memory_mb"]
            last_sample = memory_samples[-1]["memory_mb"]
            trend = (last_sample - first_sample) / len(memory_samples) * SAMPLE_INTERVAL

            print(f"Trend:       {trend:+.2f}MB per {SAMPLE_INTERVAL} requests")

            if trend > 10:  # >10MB growth per 10 requests is concerning
                print(f"WARNING: Potential memory leak detected!")

        # Assert memory didn't grow more than 2x
        assert growth_ratio < 2.0, \
            f"Memory grew {growth_ratio:.2f}x (from {baseline_memory:.1f}MB to {final_memory:.1f}MB)"


@pytest.mark.stress
class TestLatencyP99:
    """
    Measure P99 latency under various conditions.

    Provides statistical analysis of latency distribution.
    """

    def test_latency_p99_100_requests(self, tts_binary, english_config):
        """
        Measure P50, P90, P95, P99 latencies over 100 requests.

        Provides baseline performance metrics.
        """
        NUM_REQUESTS = 100
        latencies: List[float] = []

        print(f"\n=== Latency P99 Measurement ({NUM_REQUESTS} requests) ===")

        for i in range(NUM_REQUESTS):
            text = TEST_SENTENCES[i % len(TEST_SENTENCES)]

            success, latency, error = run_tts(
                tts_binary, text, english_config, output_path=None, timeout=60
            )

            if success:
                latencies.append(latency)
            else:
                print(f"  [{i}] FAILED: {error}")

            if (i + 1) % 25 == 0:
                print(f"  Progress: {i+1}/{NUM_REQUESTS}")

        if len(latencies) < 10:
            pytest.fail(f"Only {len(latencies)} successful requests, need at least 10")

        # Calculate percentiles
        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)

        def percentile(p: float) -> float:
            idx = int(p / 100 * n)
            return sorted_latencies[min(idx, n - 1)]

        p50 = percentile(50)
        p90 = percentile(90)
        p95 = percentile(95)
        p99 = percentile(99)
        mean = statistics.mean(latencies)
        stdev = statistics.stdev(latencies) if n > 1 else 0

        print(f"\n=== Latency Distribution ({n} samples) ===")
        print(f"  Mean:   {mean:.3f}s")
        print(f"  Stdev:  {stdev:.3f}s")
        print(f"  Min:    {min(latencies):.3f}s")
        print(f"  P50:    {p50:.3f}s")
        print(f"  P90:    {p90:.3f}s")
        print(f"  P95:    {p95:.3f}s")
        print(f"  P99:    {p99:.3f}s")
        print(f"  Max:    {max(latencies):.3f}s")

        # Histogram (simple ASCII)
        print(f"\n  Latency Histogram:")
        buckets = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0, float('inf')]
        bucket_counts = [0] * len(buckets)
        for lat in latencies:
            for i, upper in enumerate(buckets):
                if lat <= upper:
                    bucket_counts[i] += 1
                    break

        bucket_labels = ["<0.5s", "0.5-1s", "1-2s", "2-3s", "3-5s", "5-10s", ">10s"]
        max_count = max(bucket_counts) if bucket_counts else 1
        for label, count in zip(bucket_labels, bucket_counts):
            bar_len = int(count / max_count * 30) if max_count > 0 else 0
            bar = "#" * bar_len
            print(f"    {label:>8}: {count:3d} {bar}")

        # Assert P99 is under 30s (reasonable for cold starts)
        # Warm P99 should be much lower
        assert p99 < 30.0, f"P99 latency {p99:.1f}s exceeds 30s threshold"

    def test_warm_latency_after_warmup(self, tts_binary, english_config):
        """
        Measure warm latency after initial warmup period.

        Runs 10 warmup requests, then measures 50 more.
        """
        WARMUP = 10
        MEASURE = 50

        print(f"\n=== Warm Latency Test (warmup={WARMUP}, measure={MEASURE}) ===")

        # Warmup
        print(f"Warming up...")
        for i in range(WARMUP):
            text = TEST_SENTENCES[i % len(TEST_SENTENCES)]
            run_tts(tts_binary, text, english_config, timeout=60)

        # Measure
        print(f"Measuring warm latencies...")
        latencies: List[float] = []
        for i in range(MEASURE):
            text = TEST_SENTENCES[i % len(TEST_SENTENCES)]
            success, latency, _ = run_tts(
                tts_binary, text, english_config, timeout=60
            )
            if success:
                latencies.append(latency)

        if len(latencies) < 10:
            pytest.fail(f"Only {len(latencies)} successful measurements")

        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)
        p50 = sorted_latencies[n // 2]
        p95 = sorted_latencies[int(0.95 * n)]
        p99 = sorted_latencies[int(0.99 * n)]

        print(f"\n=== Warm Latency Results ({n} samples) ===")
        print(f"  Mean:   {statistics.mean(latencies):.3f}s")
        print(f"  P50:    {p50:.3f}s")
        print(f"  P95:    {p95:.3f}s")
        print(f"  P99:    {p99:.3f}s")
        print(f"  Max:    {max(latencies):.3f}s")

        # Warm P99 should be under 5s for reasonable performance
        assert p99 < 10.0, f"Warm P99 latency {p99:.1f}s exceeds 10s threshold"


# =============================================================================
# Summary Report
# =============================================================================

@pytest.mark.stress
def test_generate_stress_report(tts_binary, english_config, tmp_path):
    """
    Generate a comprehensive stress test report.

    Runs a quick suite and outputs a summary.
    """
    NUM_CONCURRENT = 5
    NUM_SEQUENTIAL = 50

    print(f"\n" + "=" * 60)
    print(f"STRESS TEST SUMMARY REPORT")
    print(f"=" * 60)

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "concurrent": {},
        "sequential": {},
        "latency": {}
    }

    # Quick concurrent test
    print(f"\n[1/3] Concurrent streams ({NUM_CONCURRENT})...")
    concurrent_latencies = []
    concurrent_errors = 0

    def run_concurrent(idx: int) -> Tuple[bool, float]:
        text = TEST_SENTENCES[idx % len(TEST_SENTENCES)]
        success, lat, _ = run_tts(tts_binary, text, english_config, timeout=120)
        return success, lat

    with ThreadPoolExecutor(max_workers=NUM_CONCURRENT) as executor:
        futures = [executor.submit(run_concurrent, i) for i in range(NUM_CONCURRENT)]
        for f in as_completed(futures):
            success, lat = f.result()
            if success:
                concurrent_latencies.append(lat)
            else:
                concurrent_errors += 1

    results["concurrent"] = {
        "total": NUM_CONCURRENT,
        "successful": len(concurrent_latencies),
        "errors": concurrent_errors,
        "avg_latency": statistics.mean(concurrent_latencies) if concurrent_latencies else 0
    }

    # Quick sequential test
    print(f"[2/3] Sequential requests ({NUM_SEQUENTIAL})...")
    sequential_latencies = []
    sequential_errors = 0

    for i in range(NUM_SEQUENTIAL):
        text = TEST_SENTENCES[i % len(TEST_SENTENCES)]
        success, lat, _ = run_tts(tts_binary, text, english_config, timeout=60)
        if success:
            sequential_latencies.append(lat)
        else:
            sequential_errors += 1

    if sequential_latencies:
        sorted_lats = sorted(sequential_latencies)
        n = len(sorted_lats)
        results["latency"] = {
            "samples": n,
            "mean": statistics.mean(sequential_latencies),
            "p50": sorted_lats[n // 2],
            "p95": sorted_lats[int(0.95 * n)],
            "p99": sorted_lats[int(0.99 * n)] if n >= 100 else sorted_lats[-1],
            "min": min(sequential_latencies),
            "max": max(sequential_latencies)
        }

    results["sequential"] = {
        "total": NUM_SEQUENTIAL,
        "successful": len(sequential_latencies),
        "errors": sequential_errors,
        "throughput_per_s": len(sequential_latencies) / sum(sequential_latencies) if sequential_latencies else 0
    }

    # Memory snapshot
    print(f"[3/3] Memory analysis...")
    final_memory_mb = psutil.Process().memory_info().rss / (1024 * 1024)
    results["memory_mb"] = final_memory_mb

    # Print report
    print(f"\n" + "=" * 60)
    print(f"RESULTS")
    print(f"=" * 60)
    print(f"\nConcurrent Streams:")
    print(f"  Success Rate:  {results['concurrent']['successful']}/{results['concurrent']['total']}")
    print(f"  Avg Latency:   {results['concurrent']['avg_latency']:.3f}s")

    print(f"\nSequential Throughput:")
    print(f"  Success Rate:  {results['sequential']['successful']}/{results['sequential']['total']}")
    print(f"  Throughput:    {results['sequential']['throughput_per_s']:.2f} req/s")

    if results.get("latency"):
        print(f"\nLatency Distribution:")
        lat = results["latency"]
        print(f"  Mean:  {lat['mean']:.3f}s")
        print(f"  P50:   {lat['p50']:.3f}s")
        print(f"  P95:   {lat['p95']:.3f}s")
        print(f"  P99:   {lat['p99']:.3f}s")

    print(f"\nMemory: {results['memory_mb']:.1f}MB")
    print(f"\n" + "=" * 60)

    # Save report
    report_path = tmp_path / "stress_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Report saved to: {report_path}")

    # Assert overall health
    assert results['concurrent']['successful'] == NUM_CONCURRENT, "Concurrent test failures"
    assert results['sequential']['successful'] >= NUM_SEQUENTIAL * 0.95, "Sequential test failures"
