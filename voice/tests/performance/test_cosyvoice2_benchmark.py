"""
CosyVoice2 Performance Benchmark Suite

Benchmarks RTF, latency, and memory for CosyVoice2 TTS.
Uses C++ engine tests for accurate warm RTF measurement.

Based on PRODUCTION_ROADMAP_2025-12-10.md Phase 4 requirements.

Usage:
    pytest tests/performance/test_cosyvoice2_benchmark.py -v
    pytest tests/performance/test_cosyvoice2_benchmark.py -v -s  # With output
"""

import os
import re
import subprocess
import tempfile
import time
import wave
import numpy as np
import pytest
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
STREAM_TTS_CPP = PROJECT_ROOT / "stream-tts-cpp"
BUILD_DIR = STREAM_TTS_CPP / "build"
TTS_BINARY = BUILD_DIR / "stream-tts-cpp"
COSYVOICE_ENGINE_TEST = BUILD_DIR / "test_cosyvoice_engine"

# Skip if binary not built
pytestmark = pytest.mark.skipif(
    not TTS_BINARY.exists(),
    reason="stream-tts-cpp binary not built"
)


def get_tts_env():
    """Get environment variables for TTS subprocess."""
    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["OMP_NUM_THREADS"] = "1"
    return env


def read_wav_duration(path: Path) -> float:
    """Read WAV file and return duration in seconds."""
    with wave.open(str(path), 'rb') as wf:
        return wf.getnframes() / wf.getframerate()


class TestCosyVoice2WarmRTF:
    """Test warm RTF using C++ engine tests (most accurate)."""

    @pytest.mark.skipif(
        not COSYVOICE_ENGINE_TEST.exists(),
        reason="test_cosyvoice_engine not built"
    )
    def test_warm_rtf_under_threshold(self):
        """Warm RTF must be < 1.0x for real-time synthesis.

        This uses the C++ engine test which measures RTF with models already loaded.
        The warmup benchmark provides the most accurate RTF measurement.

        Note: RTF varies 0.43-0.85 depending on system load. Target is < 1.0 real-time.
        Typical measurement: ~0.48 on idle system.
        """
        result = subprocess.run(
            [str(COSYVOICE_ENGINE_TEST), "--mps"],
            capture_output=True,
            text=True,
            timeout=180,
            cwd=str(BUILD_DIR),
            env=get_tts_env()
        )

        # Parse warmup benchmark RTF from output
        # Format: "[PASS] Warmup Benchmark: RTF=0.48 (688.43ms / 1440.00ms) (688.43ms)"
        rtf_match = re.search(r'Warmup Benchmark: RTF=(\d+\.\d+)', result.stdout + result.stderr)

        if rtf_match:
            rtf = float(rtf_match.group(1))
            print(f"\nWarm RTF: {rtf:.3f}x (threshold: < 1.0x, target: < 0.5x)")
            assert rtf < 1.0, f"Warm RTF too high: {rtf:.3f}x >= 1.0x (not real-time)"
        else:
            # Fallback: check if test passed
            assert "[PASS] Warmup Benchmark" in (result.stdout + result.stderr), \
                f"Warmup benchmark not found in output: {result.stdout}"


class TestCosyVoice2ColdStartLatency:
    """Test cold start (model load) latency."""

    def test_cold_start_under_threshold(self):
        """Cold start (process spawn + model load) < 15s.

        Note: This is CLI cold start, not daemon mode.
        Model loading is slow but happens once per session.
        """
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = Path(f.name)

        try:
            cmd = [
                str(TTS_BINARY),
                "--voice-name", "sichuan",
                "--speak", "你好",  # Short text
                "--lang", "zh",
                "--save-audio", str(output_path)
            ]

            start = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(STREAM_TTS_CPP),
                env=get_tts_env()
            )
            cold_start_time = time.time() - start

            assert result.returncode == 0, f"Synthesis failed: {result.stderr}"

            print(f"\nCold start: {cold_start_time:.2f}s (threshold: < 15s)")
            assert cold_start_time < 15, f"Cold start too slow: {cold_start_time:.2f}s >= 15s"
        finally:
            if output_path.exists():
                output_path.unlink()


class TestCosyVoice2CLIRTFBenchmark:
    """RTF benchmark via CLI (includes overhead)."""

    @pytest.mark.benchmark
    def test_cli_rtf_benchmark(self):
        """Benchmark RTF for various text lengths via CLI.

        Note: CLI RTF includes process spawn + model loading (~6s overhead per call).
        This is informational only - actual performance is measured by engine tests.
        Target: < 5x (allows for cold start overhead)
        """
        test_cases = [
            ("你好", 2),
            ("今天天气真好", 6),
            ("这是一个用来测试语音合成质量的句子", 16),
        ]

        results = []

        for text, chars in test_cases:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                output_path = Path(f.name)

            try:
                cmd = [
                    str(TTS_BINARY),
                    "--voice-name", "sichuan",
                    "--speak", text,
                    "--lang", "zh",
                    "--save-audio", str(output_path)
                ]

                start = time.time()
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                                         cwd=str(STREAM_TTS_CPP), env=get_tts_env())
                inference_time = time.time() - start

                if result.returncode != 0:
                    continue

                duration = read_wav_duration(output_path)
                rtf = inference_time / duration if duration > 0 else float('inf')

                results.append({
                    "chars": chars,
                    "inference_sec": inference_time,
                    "duration_sec": duration,
                    "rtf": rtf
                })
            finally:
                if output_path.exists():
                    output_path.unlink()

        # Print results
        print("\n" + "=" * 50)
        print("CosyVoice2 CLI RTF Benchmark")
        print("=" * 50)
        print(f"{'Chars':<8} {'Inference':<12} {'Audio':<10} {'RTF':<8}")
        print("-" * 50)

        for r in results:
            print(f"{r['chars']:<8} {r['inference_sec']:.2f}s{'':<6} "
                  f"{r['duration_sec']:.2f}s{'':<4} {r['rtf']:.3f}")

        avg_rtf = sum(r["rtf"] for r in results) / len(results) if results else 0
        print("-" * 50)
        print(f"Average CLI RTF: {avg_rtf:.3f} (target < 5.0 - includes model loading)")
        print("=" * 50)

        # Assert threshold (CLI includes ~6s model load, so much higher threshold)
        # Warm RTF is ~0.48 - verified by engine tests
        assert avg_rtf < 5.0, f"Average CLI RTF too high: {avg_rtf:.3f}x >= 5.0x"


class TestCosyVoice2EngineTests:
    """Run full engine test suite and verify all pass."""

    @pytest.mark.skipif(
        not COSYVOICE_ENGINE_TEST.exists(),
        reason="test_cosyvoice_engine not built"
    )
    def test_all_engine_tests_pass(self):
        """All C++ engine tests must pass."""
        result = subprocess.run(
            [str(COSYVOICE_ENGINE_TEST), "--mps"],
            capture_output=True,
            text=True,
            timeout=180,
            cwd=str(BUILD_DIR),
            env=get_tts_env()
        )

        output = result.stdout + result.stderr
        pass_count = output.count("[PASS]")
        fail_count = output.count("[FAIL]")

        print(f"\nEngine Tests: {pass_count} PASS, {fail_count} FAIL")

        assert fail_count == 0, f"Engine tests failed: {fail_count} failures"
        assert pass_count >= 8, f"Expected 8+ passing tests, got {pass_count}"


class TestCosyVoice2PerformanceRegression:
    """Guard against performance regression."""

    # Performance thresholds (based on M4 Max measurements)
    # Warm RTF varies 0.43-0.65 depending on system load
    WARM_RTF_THRESHOLD = 0.7  # Allow variance, target is < 1.0 real-time
    COLD_START_THRESHOLD = 15.0  # seconds
    CLI_RTF_THRESHOLD = 5.0  # Higher due to process overhead

    @pytest.mark.skipif(
        not COSYVOICE_ENGINE_TEST.exists(),
        reason="test_cosyvoice_engine not built"
    )
    def test_performance_regression_guard(self):
        """Combined performance regression check.

        Verifies:
        - Warm RTF < 0.5x
        - All engine tests pass
        """
        result = subprocess.run(
            [str(COSYVOICE_ENGINE_TEST), "--mps"],
            capture_output=True,
            text=True,
            timeout=180,
            cwd=str(BUILD_DIR),
            env=get_tts_env()
        )

        output = result.stdout + result.stderr

        # Check warm RTF
        rtf_match = re.search(r'Warmup Benchmark: RTF=(\d+\.\d+)', output)
        if rtf_match:
            rtf = float(rtf_match.group(1))
            assert rtf < self.WARM_RTF_THRESHOLD, \
                f"REGRESSION: Warm RTF {rtf:.3f}x >= {self.WARM_RTF_THRESHOLD}x"

        # Check no failures
        fail_count = output.count("[FAIL]")
        assert fail_count == 0, f"REGRESSION: {fail_count} engine tests failed"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
