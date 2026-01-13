#!/usr/bin/env python3
"""
GPU Utilization Test - Worker #240
2025-12-06

Measures actual GPU (MPS) utilization during TTS inference on M4 Max.
Uses macOS powermetrics to capture GPU power metrics.

Requirements:
- Must run with sudo for powermetrics access
- M4 Max with Metal GPU
- stream-tts-cpp binary built

Usage:
    sudo .venv/bin/pytest tests/performance/test_gpu_utilization.py -v -s
"""

import pytest
import subprocess
import time
import os
import json
import threading
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

# Paths
STREAM_TTS_CPP = Path(__file__).parent.parent.parent / "stream-tts-cpp"
BINARY_PATH = STREAM_TTS_CPP / "build" / "stream-tts-cpp"
CONFIG_PATH = STREAM_TTS_CPP / "config" / "kokoro-mps-en.yaml"

# Test configuration
TEST_TEXTS = [
    "Hello world, this is a test.",
    "The quick brown fox jumps over the lazy dog.",
    "Testing GPU utilization during speech synthesis.",
    "One two three four five six seven eight nine ten.",
    "Performance measurement is critical for optimization.",
]


@dataclass
class GPUMetrics:
    """GPU metrics captured during inference."""
    gpu_power_mw: float = 0.0
    gpu_active_residency_pct: float = 0.0
    gpu_freq_mhz: float = 0.0
    ane_power_mw: float = 0.0
    samples: int = 0
    raw_output: str = ""


@dataclass
class InferenceResult:
    """Result from a TTS inference request."""
    success: bool
    latency_ms: float
    error: Optional[str] = None


def check_sudo() -> bool:
    """Check if running with sudo."""
    return os.geteuid() == 0


def capture_gpu_metrics(duration_seconds: int, interval_ms: int = 500) -> GPUMetrics:
    """
    Capture GPU metrics using powermetrics.

    Note: Requires sudo to run powermetrics.
    """
    if not check_sudo():
        return GPUMetrics(raw_output="ERROR: Requires sudo")

    samples = max(1, duration_seconds * 1000 // interval_ms)

    try:
        result = subprocess.run(
            [
                "powermetrics",
                "--samplers", "gpu_power,ane_power",
                "-i", str(interval_ms),
                "-n", str(samples),
                "--format", "text"
            ],
            capture_output=True,
            text=True,
            timeout=duration_seconds + 10
        )

        output = result.stdout + result.stderr
        metrics = GPUMetrics(raw_output=output, samples=samples)

        # Parse GPU power from output
        # Example line: "GPU Power: 1234 mW"
        gpu_powers = []
        gpu_residencies = []
        gpu_freqs = []
        ane_powers = []

        for line in output.split('\n'):
            line_lower = line.lower()

            # GPU Power parsing
            if 'gpu power' in line_lower or 'gpu_power' in line_lower:
                try:
                    # Extract number before "mw"
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if 'mw' in part.lower() and i > 0:
                            val = float(parts[i-1].replace(',', ''))
                            gpu_powers.append(val)
                            break
                        try:
                            val = float(part.replace(',', ''))
                            if val > 0 and val < 100000:  # Reasonable mW range
                                gpu_powers.append(val)
                                break
                        except ValueError:
                            continue
                except (ValueError, IndexError):
                    pass

            # GPU Active Residency
            if 'active residency' in line_lower or 'gpu_active' in line_lower:
                try:
                    parts = line.split()
                    for part in parts:
                        if '%' in part:
                            val = float(part.replace('%', '').replace(',', ''))
                            gpu_residencies.append(val)
                            break
                        try:
                            val = float(part.replace(',', ''))
                            if 0 <= val <= 100:
                                gpu_residencies.append(val)
                                break
                        except ValueError:
                            continue
                except (ValueError, IndexError):
                    pass

            # GPU Frequency
            if 'gpu freq' in line_lower or 'frequency' in line_lower and 'gpu' in line_lower:
                try:
                    parts = line.split()
                    for part in parts:
                        if 'mhz' in part.lower():
                            val = float(part.replace('MHz', '').replace('mhz', '').replace(',', ''))
                            gpu_freqs.append(val)
                            break
                except (ValueError, IndexError):
                    pass

            # ANE Power
            if 'ane power' in line_lower or 'ane_power' in line_lower:
                try:
                    parts = line.split()
                    for part in parts:
                        try:
                            val = float(part.replace(',', ''))
                            if val > 0 and val < 100000:
                                ane_powers.append(val)
                                break
                        except ValueError:
                            continue
                except (ValueError, IndexError):
                    pass

        if gpu_powers:
            metrics.gpu_power_mw = statistics.mean(gpu_powers)
        if gpu_residencies:
            metrics.gpu_active_residency_pct = statistics.mean(gpu_residencies)
        if gpu_freqs:
            metrics.gpu_freq_mhz = statistics.mean(gpu_freqs)
        if ane_powers:
            metrics.ane_power_mw = statistics.mean(ane_powers)

        return metrics

    except subprocess.TimeoutExpired:
        return GPUMetrics(raw_output="ERROR: powermetrics timeout")
    except Exception as e:
        return GPUMetrics(raw_output=f"ERROR: {e}")


def run_tts_inference(text: str, timeout: float = 30.0) -> InferenceResult:
    """Run a single TTS inference."""
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
            return InferenceResult(success=True, latency_ms=latency_ms)
        else:
            return InferenceResult(
                success=False,
                latency_ms=latency_ms,
                error=result.stderr[:200] if result.stderr else "Unknown error"
            )

    except subprocess.TimeoutExpired:
        latency_ms = (time.perf_counter() - start) * 1000
        return InferenceResult(success=False, latency_ms=latency_ms, error="Timeout")
    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        return InferenceResult(success=False, latency_ms=latency_ms, error=str(e))


def get_process_memory_mb(pid: int) -> float:
    """Get process memory (RSS) in MB."""
    try:
        result = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            rss_kb = int(result.stdout.strip())
            return rss_kb / 1024
    except Exception:
        pass
    return 0.0


class TestGPUUtilization:
    """GPU utilization tests for M4 Max MPS backend."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Check prerequisites."""
        if not BINARY_PATH.exists():
            pytest.skip(f"Binary not found: {BINARY_PATH}")
        if not CONFIG_PATH.exists():
            pytest.skip(f"Config not found: {CONFIG_PATH}")

    def test_idle_gpu_baseline(self):
        """Measure GPU usage at idle (no inference)."""
        print("\n--- Idle GPU Baseline ---")

        if not check_sudo():
            pytest.skip("Requires sudo for powermetrics")

        print("Measuring idle GPU metrics (3 seconds)...")
        metrics = capture_gpu_metrics(duration_seconds=3, interval_ms=500)

        print(f"\nIdle GPU Metrics:")
        print(f"  GPU Power: {metrics.gpu_power_mw:.0f} mW")
        print(f"  GPU Active Residency: {metrics.gpu_active_residency_pct:.1f}%")
        print(f"  GPU Frequency: {metrics.gpu_freq_mhz:.0f} MHz")
        print(f"  ANE Power: {metrics.ane_power_mw:.0f} mW")
        print(f"  Samples: {metrics.samples}")

        if "ERROR" in metrics.raw_output:
            print(f"\nWarning: {metrics.raw_output[:200]}")

        # Store for comparison
        return metrics

    def test_gpu_during_single_inference(self):
        """Measure GPU usage during a single TTS inference."""
        print("\n--- GPU During Single Inference ---")

        if not check_sudo():
            pytest.skip("Requires sudo for powermetrics")

        # Start GPU monitoring in background
        gpu_results = []
        monitoring = True

        def monitor_gpu():
            """Background GPU monitoring."""
            while monitoring:
                metrics = capture_gpu_metrics(duration_seconds=2, interval_ms=250)
                gpu_results.append(metrics)
                if not monitoring:
                    break

        monitor_thread = threading.Thread(target=monitor_gpu, daemon=True)
        monitor_thread.start()

        # Run inference
        time.sleep(1)  # Let monitoring start
        print("Running TTS inference...")
        result = run_tts_inference("This is a test of GPU utilization measurement.")

        monitoring = False
        monitor_thread.join(timeout=5)

        print(f"\nInference Result:")
        print(f"  Success: {result.success}")
        print(f"  Latency: {result.latency_ms:.0f} ms")
        if result.error:
            print(f"  Error: {result.error}")

        if gpu_results:
            avg_power = statistics.mean([m.gpu_power_mw for m in gpu_results if m.gpu_power_mw > 0] or [0])
            avg_residency = statistics.mean([m.gpu_active_residency_pct for m in gpu_results if m.gpu_active_residency_pct > 0] or [0])

            print(f"\nGPU Metrics During Inference:")
            print(f"  Avg GPU Power: {avg_power:.0f} mW")
            print(f"  Avg GPU Active Residency: {avg_residency:.1f}%")
            print(f"  Samples collected: {len(gpu_results)}")

        assert result.success, f"Inference failed: {result.error}"

    def test_gpu_during_concurrent_inference(self):
        """Measure GPU usage during concurrent TTS requests."""
        print("\n--- GPU During Concurrent Inference (5 requests) ---")

        if not check_sudo():
            pytest.skip("Requires sudo for powermetrics")

        # Start long-running GPU monitoring
        monitoring = True
        gpu_samples = []

        def monitor_gpu_continuous():
            """Continuous GPU monitoring."""
            while monitoring:
                metrics = capture_gpu_metrics(duration_seconds=1, interval_ms=200)
                if metrics.gpu_power_mw > 0 or metrics.gpu_active_residency_pct > 0:
                    gpu_samples.append(metrics)

        monitor_thread = threading.Thread(target=monitor_gpu_continuous, daemon=True)
        monitor_thread.start()

        time.sleep(1)  # Let monitoring stabilize

        # Run concurrent inference
        print("Starting 5 concurrent TTS requests...")
        results: List[InferenceResult] = []
        start_time = time.perf_counter()

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(run_tts_inference, TEST_TEXTS[i])
                for i in range(5)
            ]
            for future in as_completed(futures):
                results.append(future.result())

        total_time = time.perf_counter() - start_time

        monitoring = False
        monitor_thread.join(timeout=5)

        successful = [r for r in results if r.success]
        latencies = [r.latency_ms for r in successful]

        print(f"\nConcurrent Inference Results:")
        print(f"  Total requests: 5")
        print(f"  Successful: {len(successful)}")
        print(f"  Wall time: {total_time:.1f}s")
        if latencies:
            print(f"  Avg latency: {statistics.mean(latencies):.0f} ms")
            print(f"  Max latency: {max(latencies):.0f} ms")

        if gpu_samples:
            powers = [m.gpu_power_mw for m in gpu_samples if m.gpu_power_mw > 0]
            residencies = [m.gpu_active_residency_pct for m in gpu_samples if m.gpu_active_residency_pct > 0]

            print(f"\nGPU Metrics During Concurrent Load:")
            if powers:
                print(f"  Avg GPU Power: {statistics.mean(powers):.0f} mW")
                print(f"  Max GPU Power: {max(powers):.0f} mW")
            if residencies:
                print(f"  Avg GPU Active Residency: {statistics.mean(residencies):.1f}%")
                print(f"  Max GPU Active Residency: {max(residencies):.1f}%")
            print(f"  Samples collected: {len(gpu_samples)}")

        assert len(successful) >= 3, f"Too many failures: {5 - len(successful)}/5"

    def test_memory_per_model_instance(self):
        """Measure memory footprint of the TTS model."""
        print("\n--- Memory Per Model Instance ---")

        # Measure baseline memory
        baseline_result = subprocess.run(
            ["vm_stat"],
            capture_output=True,
            text=True
        )
        print(f"System memory before inference:")
        for line in baseline_result.stdout.split('\n')[:5]:
            print(f"  {line}")

        # Start daemon and measure its memory
        socket_path = f"/tmp/gpu-test-{os.getpid()}.sock"

        env = os.environ.copy()

        print("\nStarting daemon to measure model memory...")
        daemon = subprocess.Popen(
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

        try:
            # Wait for daemon to load model
            start_wait = time.time()
            while time.time() - start_wait < 90:
                if os.path.exists(socket_path):
                    time.sleep(2)  # Let model fully load
                    break
                time.sleep(0.5)

            if daemon.poll() is not None:
                stdout, stderr = daemon.communicate(timeout=5)
                pytest.fail(f"Daemon failed to start: {stderr.decode()[:500]}")

            # Measure daemon memory
            daemon_memory_mb = get_process_memory_mb(daemon.pid)

            print(f"\nDaemon Memory (PID {daemon.pid}):")
            print(f"  RSS: {daemon_memory_mb:.0f} MB")

            # Get more detailed memory info
            ps_result = subprocess.run(
                ["ps", "-o", "pid,rss,vsz,comm", "-p", str(daemon.pid)],
                capture_output=True,
                text=True
            )
            print(f"\nDetailed process info:")
            print(ps_result.stdout)

            # Expected: ~2-4 GB for Kokoro model on MPS
            print(f"\nModel Memory Estimate: {daemon_memory_mb:.0f} MB")
            print("(Expected: 2000-4000 MB for Kokoro TTS model)")

            # Run a request to ensure model is fully loaded
            speak_result = subprocess.run(
                [str(BINARY_PATH), "--socket", socket_path, "--speak", "Memory test"],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(STREAM_TTS_CPP),
                env=env
            )

            if speak_result.returncode == 0:
                # Measure again after inference
                post_inference_memory = get_process_memory_mb(daemon.pid)
                print(f"\nMemory after first inference: {post_inference_memory:.0f} MB")
                memory_delta = post_inference_memory - daemon_memory_mb
                print(f"Memory delta: {memory_delta:+.0f} MB")

        finally:
            # Stop daemon
            subprocess.run(
                [str(BINARY_PATH), "--socket", socket_path, "--stop"],
                capture_output=True,
                timeout=10,
                cwd=str(STREAM_TTS_CPP),
                env=env
            )
            try:
                daemon.wait(timeout=10)
            except subprocess.TimeoutExpired:
                daemon.kill()
                daemon.wait()
            if os.path.exists(socket_path):
                try:
                    os.unlink(socket_path)
                except OSError:
                    pass

    def test_generate_gpu_report(self):
        """Generate comprehensive GPU utilization report."""
        print("\n--- Generating GPU Utilization Report ---")

        if not check_sudo():
            pytest.skip("Requires sudo for powermetrics")

        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "platform": "M4 Max (MPS)",
            "idle_metrics": {},
            "single_inference_metrics": {},
            "concurrent_inference_metrics": {},
        }

        # Idle baseline
        print("Measuring idle baseline...")
        idle_metrics = capture_gpu_metrics(duration_seconds=3)
        report["idle_metrics"] = {
            "gpu_power_mw": idle_metrics.gpu_power_mw,
            "gpu_active_residency_pct": idle_metrics.gpu_active_residency_pct,
        }

        # Single inference
        print("Measuring single inference...")
        monitoring = True
        single_samples = []

        def monitor_single():
            while monitoring:
                m = capture_gpu_metrics(duration_seconds=1, interval_ms=200)
                if m.gpu_power_mw > 0:
                    single_samples.append(m)

        t = threading.Thread(target=monitor_single, daemon=True)
        t.start()
        time.sleep(0.5)

        result = run_tts_inference("Single inference test for GPU measurement.")

        monitoring = False
        t.join(timeout=5)

        if single_samples:
            powers = [m.gpu_power_mw for m in single_samples if m.gpu_power_mw > 0]
            report["single_inference_metrics"] = {
                "inference_latency_ms": result.latency_ms,
                "inference_success": result.success,
                "avg_gpu_power_mw": statistics.mean(powers) if powers else 0,
                "max_gpu_power_mw": max(powers) if powers else 0,
            }

        # Save report
        report_dir = Path(__file__).parent.parent.parent / "reports" / "main"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_file = report_dir / f"gpu_utilization_baseline_{time.strftime('%Y-%m-%d')}.json"

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\nReport saved to: {report_file}")
        print(f"\nSummary:")
        print(f"  Idle GPU Power: {report['idle_metrics'].get('gpu_power_mw', 0):.0f} mW")
        if report['single_inference_metrics']:
            print(f"  Single Inference GPU Power: {report['single_inference_metrics'].get('avg_gpu_power_mw', 0):.0f} mW")

        # Basic assertion - test should complete
        assert True


class TestSimplifiedGPUMeasurement:
    """
    Simplified GPU measurement that doesn't require sudo.
    Uses subprocess output parsing instead of powermetrics.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """Check prerequisites."""
        if not BINARY_PATH.exists():
            pytest.skip(f"Binary not found: {BINARY_PATH}")

    def test_inference_timing_breakdown(self):
        """
        Measure inference timing without powermetrics.
        This provides useful data even without sudo access.
        """
        print("\n--- Inference Timing Breakdown (no sudo required) ---")

        results = []

        # Warmup
        print("Warmup (3 requests)...")
        for _ in range(3):
            run_tts_inference("Warmup request.")

        # Measure
        print("Measuring 10 requests...")
        for i, text in enumerate(TEST_TEXTS * 2):
            result = run_tts_inference(text)
            results.append(result)
            print(f"  Request {i+1}: {result.latency_ms:.0f}ms {'OK' if result.success else 'FAIL'}")

        successful = [r for r in results if r.success]
        latencies = [r.latency_ms for r in successful]

        if latencies:
            print(f"\nTiming Summary:")
            print(f"  Success rate: {len(successful)}/{len(results)}")
            print(f"  Min latency: {min(latencies):.0f}ms")
            print(f"  Max latency: {max(latencies):.0f}ms")
            print(f"  Avg latency: {statistics.mean(latencies):.0f}ms")
            print(f"  Median latency: {statistics.median(latencies):.0f}ms")
            if len(latencies) > 1:
                print(f"  Std dev: {statistics.stdev(latencies):.0f}ms")

        assert len(successful) >= 5, "Too many failures"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
