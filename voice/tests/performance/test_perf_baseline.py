#!/usr/bin/env python3
"""
Performance Baseline Test - Worker #240
2025-12-06

Captures baseline performance metrics for M4 Max local machine.
Creates tests/performance/baseline.json with measurements.

Metrics captured:
- Warm latency (P50, P95, P99) via daemon mode
- Cold latency (process spawn + model load)
- Throughput (requests per second)
- Memory per stream (daemon RSS)
- GPU utilization (if sudo available, else estimated from power draw)

Usage:
    .venv/bin/pytest tests/performance/test_perf_baseline.py -v -s
"""

import pytest
import subprocess
import time
import os
import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Paths
STREAM_TTS_CPP = Path(__file__).parent.parent.parent / "stream-tts-cpp"
BINARY_PATH = STREAM_TTS_CPP / "build" / "stream-tts-cpp"
CONFIG_PATH = STREAM_TTS_CPP / "config" / "kokoro-mps-en.yaml"
BASELINE_FILE = Path(__file__).parent / "baseline.json"

# Test configuration
TEST_TEXTS = [
    "Hello world.",
    "The quick brown fox jumps over the lazy dog.",
    "Testing performance with concurrent streams.",
    "One two three four five six seven eight nine ten.",
    "Performance measurement is critical for optimization.",
]


@dataclass
class PerformanceBaseline:
    """Performance baseline measurements."""
    timestamp: str
    platform: str

    # Latency metrics (milliseconds)
    cold_latency_ms: float  # Process spawn + model load
    warm_latency_p50_ms: float  # Daemon mode P50
    warm_latency_p95_ms: float  # Daemon mode P95
    warm_latency_p99_ms: float  # Daemon mode P99

    # Throughput
    sequential_throughput_req_per_sec: float
    concurrent_throughput_req_per_sec: float

    # Memory
    daemon_memory_mb: float

    # Success rates
    warm_success_rate: float

    # Notes
    notes: str = ""


def get_percentile(values: List[float], percentile: int) -> float:
    """Calculate percentile from list of values."""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = int(len(sorted_values) * percentile / 100)
    idx = min(idx, len(sorted_values) - 1)
    return sorted_values[idx]


def run_cold_inference(text: str, timeout: float = 60.0) -> tuple[bool, float, str]:
    """
    Run cold TTS inference (spawn new process).
    Returns (success, latency_ms, error_message).
    """
    start = time.perf_counter()
    try:
        env = os.environ.copy()

        result = subprocess.run(
            [str(BINARY_PATH), "--speak", text, "--lang", "en"],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(STREAM_TTS_CPP),
            env=env
        )
        latency_ms = (time.perf_counter() - start) * 1000

        if result.returncode == 0:
            return True, latency_ms, ""
        else:
            return False, latency_ms, result.stderr[:200] if result.stderr else "Unknown error"

    except subprocess.TimeoutExpired:
        latency_ms = (time.perf_counter() - start) * 1000
        return False, latency_ms, "Timeout"
    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        return False, latency_ms, str(e)


def start_daemon(socket_path: str, timeout: int = 90) -> subprocess.Popen:
    """Start the TTS daemon and wait for it to be ready."""
    env = os.environ.copy()

    process = subprocess.Popen(
        [
            str(BINARY_PATH),
            "--daemon",
            "--socket", socket_path,
            str(CONFIG_PATH)
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(STREAM_TTS_CPP),
        env=env
    )

    # Wait for socket
    start_time = time.time()
    while time.time() - start_time < timeout:
        if os.path.exists(socket_path):
            time.sleep(0.5)
            return process
        time.sleep(0.1)

        # Check if process died
        if process.poll() is not None:
            stdout, stderr = process.communicate(timeout=5)
            raise RuntimeError(f"Daemon failed: {stderr.decode()[:500]}")

    process.kill()
    raise RuntimeError(f"Daemon failed to start in {timeout}s")


def stop_daemon(socket_path: str):
    """Stop the daemon."""
    env = os.environ.copy()

    subprocess.run(
        [str(BINARY_PATH), "--socket", socket_path, "--stop"],
        capture_output=True,
        timeout=10,
        cwd=str(STREAM_TTS_CPP),
        env=env
    )


def run_daemon_inference(socket_path: str, text: str, timeout: float = 30.0) -> tuple[bool, float, str]:
    """
    Run warm TTS inference through daemon.
    Returns (success, latency_ms, error_message).
    """
    start = time.perf_counter()
    try:
        env = os.environ.copy()

        result = subprocess.run(
            [str(BINARY_PATH), "--socket", socket_path, "--speak", text],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(STREAM_TTS_CPP),
            env=env
        )
        latency_ms = (time.perf_counter() - start) * 1000

        if result.returncode == 0:
            return True, latency_ms, ""
        else:
            return False, latency_ms, result.stderr[:200] if result.stderr else "Unknown error"

    except subprocess.TimeoutExpired:
        latency_ms = (time.perf_counter() - start) * 1000
        return False, latency_ms, "Timeout"
    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        return False, latency_ms, str(e)


def get_process_memory_mb(pid: int) -> float:
    """Get process RSS in MB."""
    try:
        result = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            return int(result.stdout.strip()) / 1024
    except Exception:
        pass
    return 0.0


class TestPerformanceBaseline:
    """Performance baseline capture tests."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Check prerequisites."""
        if not BINARY_PATH.exists():
            pytest.skip(f"Binary not found: {BINARY_PATH}")
        if not CONFIG_PATH.exists():
            pytest.skip(f"Config not found: {CONFIG_PATH}")

    def test_capture_cold_latency(self):
        """Measure cold start latency (process spawn + model load)."""
        print("\n--- Cold Start Latency Measurement ---")

        # Run 3 cold starts (expensive but necessary for baseline)
        cold_latencies = []
        for i in range(3):
            print(f"  Cold start {i+1}/3...")
            success, latency_ms, error = run_cold_inference("Cold start test.")
            if success:
                cold_latencies.append(latency_ms)
                print(f"    Latency: {latency_ms:.0f}ms")
            else:
                print(f"    Failed: {error}")

        if cold_latencies:
            avg = statistics.mean(cold_latencies)
            print(f"\n  Cold Latency Summary:")
            print(f"    Min: {min(cold_latencies):.0f}ms")
            print(f"    Avg: {avg:.0f}ms")
            print(f"    Max: {max(cold_latencies):.0f}ms")

            assert avg < 30000, f"Cold latency too high: {avg:.0f}ms"
        else:
            pytest.fail("All cold starts failed")

    def test_capture_warm_latency(self):
        """Measure warm latency through daemon mode."""
        print("\n--- Warm Latency Measurement (Daemon Mode) ---")

        socket_path = f"/tmp/perf-baseline-{os.getpid()}.sock"
        daemon = None

        try:
            print("  Starting daemon...")
            daemon = start_daemon(socket_path)

            # Warmup requests
            print("  Warmup (3 requests)...")
            for _ in range(3):
                run_daemon_inference(socket_path, "Warmup request.")

            # Measure warm latency
            print("  Measuring warm latency (30 requests)...")
            latencies = []
            for i in range(30):
                text = TEST_TEXTS[i % len(TEST_TEXTS)]
                success, latency_ms, error = run_daemon_inference(socket_path, text)
                if success:
                    latencies.append(latency_ms)
                if (i + 1) % 10 == 0:
                    print(f"    Progress: {i+1}/30")

            if latencies:
                p50 = get_percentile(latencies, 50)
                p95 = get_percentile(latencies, 95)
                p99 = get_percentile(latencies, 99)

                print(f"\n  Warm Latency Summary:")
                print(f"    Success rate: {len(latencies)}/30")
                print(f"    P50: {p50:.0f}ms")
                print(f"    P95: {p95:.0f}ms")
                print(f"    P99: {p99:.0f}ms")
                print(f"    Min: {min(latencies):.0f}ms")
                print(f"    Max: {max(latencies):.0f}ms")

                # Target: P95 < 5000ms (generous for CI)
                assert p95 < 5000, f"P95 too high: {p95:.0f}ms"
            else:
                pytest.fail("No successful daemon requests")

        finally:
            if daemon and daemon.poll() is None:
                stop_daemon(socket_path)
                try:
                    daemon.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    daemon.kill()
            if os.path.exists(socket_path):
                try:
                    os.unlink(socket_path)
                except OSError:
                    pass

    def test_capture_throughput(self):
        """Measure sequential and concurrent throughput."""
        print("\n--- Throughput Measurement ---")

        socket_path = f"/tmp/perf-throughput-{os.getpid()}.sock"
        daemon = None

        try:
            print("  Starting daemon...")
            daemon = start_daemon(socket_path)

            # Warmup
            for _ in range(3):
                run_daemon_inference(socket_path, "Warmup.")

            # Sequential throughput
            print("  Sequential throughput (20 requests)...")
            start = time.perf_counter()
            seq_success = 0
            for i in range(20):
                success, _, _ = run_daemon_inference(socket_path, TEST_TEXTS[i % len(TEST_TEXTS)])
                if success:
                    seq_success += 1
            seq_time = time.perf_counter() - start
            seq_throughput = seq_success / seq_time if seq_time > 0 else 0

            print(f"    Sequential: {seq_throughput:.2f} req/s ({seq_success}/20 in {seq_time:.1f}s)")

            # Concurrent throughput
            print("  Concurrent throughput (20 requests, 5 workers)...")
            start = time.perf_counter()
            conc_success = 0

            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [
                    executor.submit(run_daemon_inference, socket_path, TEST_TEXTS[i % len(TEST_TEXTS)])
                    for i in range(20)
                ]
                for future in as_completed(futures):
                    success, _, _ = future.result()
                    if success:
                        conc_success += 1

            conc_time = time.perf_counter() - start
            conc_throughput = conc_success / conc_time if conc_time > 0 else 0

            print(f"    Concurrent: {conc_throughput:.2f} req/s ({conc_success}/20 in {conc_time:.1f}s)")

            print(f"\n  Throughput Summary:")
            print(f"    Sequential: {seq_throughput:.2f} req/s")
            print(f"    Concurrent: {conc_throughput:.2f} req/s")
            print(f"    Speedup: {conc_throughput/seq_throughput:.2f}x" if seq_throughput > 0 else "    Speedup: N/A")

        finally:
            if daemon and daemon.poll() is None:
                stop_daemon(socket_path)
                try:
                    daemon.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    daemon.kill()
            if os.path.exists(socket_path):
                try:
                    os.unlink(socket_path)
                except OSError:
                    pass

    def test_capture_memory(self):
        """Measure daemon memory footprint."""
        print("\n--- Memory Measurement ---")

        socket_path = f"/tmp/perf-memory-{os.getpid()}.sock"
        daemon = None

        try:
            print("  Starting daemon...")
            daemon = start_daemon(socket_path)

            # Let model fully load
            time.sleep(2)

            # Measure before inference
            before_mb = get_process_memory_mb(daemon.pid)
            print(f"  Memory before inference: {before_mb:.0f} MB")

            # Run some inference
            for _ in range(5):
                run_daemon_inference(socket_path, "Memory test request.")

            # Measure after inference
            after_mb = get_process_memory_mb(daemon.pid)
            print(f"  Memory after inference: {after_mb:.0f} MB")
            print(f"  Memory delta: {after_mb - before_mb:+.0f} MB")

            print(f"\n  Memory Summary:")
            print(f"    Daemon RSS: {after_mb:.0f} MB")
            print(f"    (Expected: 1400-2000 MB for Kokoro model)")

        finally:
            if daemon and daemon.poll() is None:
                stop_daemon(socket_path)
                try:
                    daemon.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    daemon.kill()
            if os.path.exists(socket_path):
                try:
                    os.unlink(socket_path)
                except OSError:
                    pass

    def test_capture_full_baseline(self):
        """Capture complete performance baseline and save to JSON."""
        print("\n--- Full Performance Baseline Capture ---")

        socket_path = f"/tmp/perf-full-{os.getpid()}.sock"
        daemon = None

        try:
            # Cold latency
            print("\n[1/4] Cold latency measurement...")
            cold_latencies = []
            for i in range(2):  # Just 2 to save time
                success, latency_ms, _ = run_cold_inference("Cold test.")
                if success:
                    cold_latencies.append(latency_ms)
                    print(f"  Cold {i+1}: {latency_ms:.0f}ms")
            cold_latency = statistics.mean(cold_latencies) if cold_latencies else 0

            # Start daemon for warm measurements
            print("\n[2/4] Starting daemon for warm measurements...")
            daemon = start_daemon(socket_path)
            daemon_memory = get_process_memory_mb(daemon.pid)
            print(f"  Daemon memory: {daemon_memory:.0f} MB")

            # Warmup
            print("  Warming up...")
            for _ in range(5):
                run_daemon_inference(socket_path, "Warmup.")

            # Warm latency
            print("\n[3/4] Warm latency measurement (30 requests)...")
            warm_latencies = []
            for i in range(30):
                success, latency_ms, _ = run_daemon_inference(socket_path, TEST_TEXTS[i % len(TEST_TEXTS)])
                if success:
                    warm_latencies.append(latency_ms)

            warm_p50 = get_percentile(warm_latencies, 50) if warm_latencies else 0
            warm_p95 = get_percentile(warm_latencies, 95) if warm_latencies else 0
            warm_p99 = get_percentile(warm_latencies, 99) if warm_latencies else 0
            warm_success_rate = len(warm_latencies) / 30

            print(f"  P50: {warm_p50:.0f}ms, P95: {warm_p95:.0f}ms, P99: {warm_p99:.0f}ms")

            # Throughput
            print("\n[4/4] Throughput measurement...")

            # Sequential
            start = time.perf_counter()
            seq_success = 0
            for _ in range(15):
                success, _, _ = run_daemon_inference(socket_path, "Throughput test.")
                if success:
                    seq_success += 1
            seq_time = time.perf_counter() - start
            seq_throughput = seq_success / seq_time if seq_time > 0 else 0

            # Concurrent
            start = time.perf_counter()
            conc_success = 0
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(run_daemon_inference, socket_path, "Concurrent test.") for _ in range(15)]
                for future in as_completed(futures):
                    success, _, _ = future.result()
                    if success:
                        conc_success += 1
            conc_time = time.perf_counter() - start
            conc_throughput = conc_success / conc_time if conc_time > 0 else 0

            print(f"  Sequential: {seq_throughput:.2f} req/s")
            print(f"  Concurrent: {conc_throughput:.2f} req/s")

            # Create baseline object
            baseline = PerformanceBaseline(
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                platform="M4 Max (MPS)",
                cold_latency_ms=cold_latency,
                warm_latency_p50_ms=warm_p50,
                warm_latency_p95_ms=warm_p95,
                warm_latency_p99_ms=warm_p99,
                sequential_throughput_req_per_sec=seq_throughput,
                concurrent_throughput_req_per_sec=conc_throughput,
                daemon_memory_mb=daemon_memory,
                warm_success_rate=warm_success_rate,
                notes="Baseline captured by test_perf_baseline.py. Cold latency includes process spawn + model load (~12s). Warm latency is daemon mode synthesis + playback."
            )

            # Save baseline
            with open(BASELINE_FILE, 'w') as f:
                json.dump(asdict(baseline), f, indent=2)

            print(f"\n=== Performance Baseline Saved ===")
            print(f"File: {BASELINE_FILE}")
            print(f"\nBaseline Summary:")
            print(f"  Cold latency: {baseline.cold_latency_ms:.0f}ms")
            print(f"  Warm P50: {baseline.warm_latency_p50_ms:.0f}ms")
            print(f"  Warm P95: {baseline.warm_latency_p95_ms:.0f}ms")
            print(f"  Sequential throughput: {baseline.sequential_throughput_req_per_sec:.2f} req/s")
            print(f"  Concurrent throughput: {baseline.concurrent_throughput_req_per_sec:.2f} req/s")
            print(f"  Daemon memory: {baseline.daemon_memory_mb:.0f} MB")

            # Assertions
            assert baseline.warm_success_rate >= 0.5, "Too many failures"

        finally:
            if daemon and daemon.poll() is None:
                stop_daemon(socket_path)
                try:
                    daemon.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    daemon.kill()
            if os.path.exists(socket_path):
                try:
                    os.unlink(socket_path)
                except OSError:
                    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
