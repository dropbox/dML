#!/usr/bin/env python3
# Copyright 2024-2025 Andrew Yates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
DashVoice Performance Benchmark Framework

Unified framework for benchmarking all DashVoice models:
- Latency (time to first output)
- Throughput (samples per second)
- Real-Time Factor (RTF) for audio models
- Memory usage

Usage:
    python tests/benchmarks/benchmark_framework.py
    python tests/benchmarks/benchmark_framework.py --models kokoro echo_cancel
    python tests/benchmarks/benchmark_framework.py --warmup-rounds 3 --test-rounds 10
"""

import argparse
import gc
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    model_name: str
    test_name: str
    latency_ms: float  # Time to complete
    latency_std_ms: float  # Standard deviation
    throughput: float  # Items per second
    rtf: float | None  # Real-time factor (audio only)
    memory_mb: float  # Peak memory usage
    passed: bool  # Met performance target
    notes: str = ""


@dataclass
class BenchmarkTarget:
    """Performance target for a model."""

    model_name: str
    max_latency_ms: float = float("inf")
    min_throughput: float = 0.0
    min_rtf: float | None = None  # For audio: higher is better (e.g., 30x RTF)


class Benchmark:
    """Base class for benchmarks."""

    def __init__(
        self,
        name: str,
        target: BenchmarkTarget,
        warmup_rounds: int = 3,
        test_rounds: int = 10,
    ):
        self.name = name
        self.target = target
        self.warmup_rounds = warmup_rounds
        self.test_rounds = test_rounds

    def setup(self):
        """Set up the benchmark (load models, prepare data)."""

    def run_single(self) -> tuple[float, float | None]:
        """Run a single benchmark iteration.

        Returns:
            Tuple of (processing_time_seconds, audio_duration_seconds_or_none)
        """
        raise NotImplementedError

    def teardown(self):
        """Clean up after benchmark."""

    def run(self) -> BenchmarkResult:
        """Run the full benchmark with warmup and timing."""
        print(f"\n{'='*60}")
        print(f"Benchmark: {self.name}")
        print(f"{'='*60}")

        # Setup
        print("Setting up...")
        self.setup()

        # Warmup
        print(f"Warming up ({self.warmup_rounds} rounds)...")
        for _ in range(self.warmup_rounds):
            self.run_single()

        # Measure memory baseline
        gc.collect()
        import tracemalloc

        tracemalloc.start()

        # Test runs
        print(f"Running tests ({self.test_rounds} rounds)...")
        times = []
        audio_durations = []

        for _i in range(self.test_rounds):
            proc_time, audio_dur = self.run_single()
            times.append(proc_time)
            if audio_dur is not None:
                audio_durations.append(audio_dur)

        # Get memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_mb = peak / 1024 / 1024

        # Compute metrics
        avg_time_s = np.mean(times)
        std_time_s = np.std(times)
        avg_time_ms = avg_time_s * 1000
        std_time_ms = std_time_s * 1000
        throughput = 1.0 / avg_time_s if avg_time_s > 0 else 0

        # RTF for audio models
        rtf = None
        if audio_durations:
            avg_audio_dur = np.mean(audio_durations)
            if avg_time_s > 0:
                rtf = avg_audio_dur / avg_time_s

        # Check if passed targets
        passed = True
        notes = []

        if avg_time_ms > self.target.max_latency_ms:
            passed = False
            notes.append(
                f"Latency {avg_time_ms:.2f}ms > target {self.target.max_latency_ms}ms",
            )

        if throughput < self.target.min_throughput:
            passed = False
            notes.append(
                f"Throughput {throughput:.2f}/s < target {self.target.min_throughput}/s",
            )

        if self.target.min_rtf is not None and rtf is not None:
            if rtf < self.target.min_rtf:
                passed = False
                notes.append(f"RTF {rtf:.1f}x < target {self.target.min_rtf}x")

        # Cleanup
        self.teardown()

        result = BenchmarkResult(
            model_name=self.target.model_name,
            test_name=self.name,
            latency_ms=avg_time_ms,
            latency_std_ms=std_time_ms,
            throughput=throughput,
            rtf=rtf,
            memory_mb=memory_mb,
            passed=passed,
            notes="; ".join(notes) if notes else "All targets met",
        )

        # Print results
        print("\nResults:")
        print(f"  Latency: {result.latency_ms:.2f}ms +/- {result.latency_std_ms:.2f}ms")
        print(f"  Throughput: {result.throughput:.2f}/s")
        if rtf is not None:
            print(f"  RTF: {rtf:.1f}x real-time")
        print(f"  Memory: {result.memory_mb:.1f}MB")
        print(f"  Status: {'PASS' if passed else 'FAIL'} - {result.notes}")

        return result


class EchoCancellationBenchmark(Benchmark):
    """Benchmark for echo cancellation."""

    def __init__(self, warmup_rounds: int = 3, test_rounds: int = 10):
        super().__init__(
            name="Echo Cancellation (1s audio)",
            target=BenchmarkTarget(
                model_name="echo_cancel",
                max_latency_ms=5.0,
            ),
            warmup_rounds=warmup_rounds,
            test_rounds=test_rounds,
        )
        self.canceller = None
        self.mic_input = None
        self.reference = None

    def setup(self):
        from tools.dashvoice.echo_cancel import EchoCanceller

        sample_rate = 24000
        duration_s = 1.0

        # Create test signals
        t = np.linspace(0, duration_s, int(duration_s * sample_rate))
        self.reference = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        delay_samples = int(50 / 1000 * sample_rate)
        echo = np.zeros_like(self.reference)
        echo[delay_samples:] = 0.5 * self.reference[:-delay_samples]
        user_voice = 0.3 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
        self.mic_input = echo + user_voice

        # Create and warm up canceller
        self.canceller = EchoCanceller(sample_rate=sample_rate)
        self.canceller.warmup()

    def run_single(self) -> tuple[float, float | None]:
        self.canceller.add_reference(self.reference)
        start = time.perf_counter()
        self.canceller.process(self.mic_input)  # Result unused, only timing matters
        elapsed = time.perf_counter() - start
        audio_duration = len(self.mic_input) / 24000
        return elapsed, audio_duration


class VoiceFingerprintBenchmark(Benchmark):
    """Benchmark for voice fingerprint extraction."""

    def __init__(self, warmup_rounds: int = 3, test_rounds: int = 10):
        super().__init__(
            name="Voice Fingerprint Extraction",
            target=BenchmarkTarget(
                model_name="voice_fingerprint",
                max_latency_ms=100.0,  # Should be fast for 1s audio
            ),
            warmup_rounds=warmup_rounds,
            test_rounds=test_rounds,
        )
        self.db = None
        self.test_audio = None

    def setup(self):
        from tools.dashvoice.voice_database import VoiceDatabase

        self.db = VoiceDatabase()

        # Create test audio (1 second)
        sample_rate = 16000  # Resemblyzer expects 16kHz
        duration_s = 1.0
        t = np.linspace(0, duration_s, int(duration_s * sample_rate))
        # Simulate speech-like signal
        self.test_audio = np.sin(2 * np.pi * 200 * t).astype(np.float32)
        self.test_audio += 0.5 * np.sin(2 * np.pi * 400 * t)
        self.test_audio *= 0.5

    def run_single(self) -> tuple[float, float | None]:
        start = time.perf_counter()
        self.db.extract_embedding(self.test_audio, sample_rate=16000)  # Result unused, only timing matters
        elapsed = time.perf_counter() - start
        return elapsed, len(self.test_audio) / 16000


class VoiceMatchingBenchmark(Benchmark):
    """Benchmark for voice matching against database."""

    def __init__(self, warmup_rounds: int = 3, test_rounds: int = 10):
        super().__init__(
            name="Voice Matching (54 voices)",
            target=BenchmarkTarget(
                model_name="voice_matching",
                max_latency_ms=200.0,  # Includes embedding extraction
            ),
            warmup_rounds=warmup_rounds,
            test_rounds=test_rounds,
        )
        self.db = None
        self.test_audio = None

    def setup(self):
        from tools.dashvoice.voice_database import VoiceDatabase

        self.db = VoiceDatabase()

        # Create test audio
        sample_rate = 16000
        duration_s = 1.0
        t = np.linspace(0, duration_s, int(duration_s * sample_rate))
        self.test_audio = np.sin(2 * np.pi * 200 * t).astype(np.float32)
        self.test_audio += 0.5 * np.sin(2 * np.pi * 400 * t)
        self.test_audio *= 0.5

    def run_single(self) -> tuple[float, float | None]:
        start = time.perf_counter()
        is_match, name, score = self.db.is_dashvoice(self.test_audio, sample_rate=16000)
        elapsed = time.perf_counter() - start
        return elapsed, len(self.test_audio) / 16000


class PipelineBenchmark(Benchmark):
    """Benchmark for the complete DashVoice pipeline."""

    def __init__(self, warmup_rounds: int = 3, test_rounds: int = 10):
        super().__init__(
            name="DashVoice Pipeline (2s audio)",
            target=BenchmarkTarget(
                model_name="dashvoice_pipeline",
                max_latency_ms=200.0,  # Target: <200ms for 2s audio
            ),
            warmup_rounds=warmup_rounds,
            test_rounds=test_rounds,
        )
        self.pipeline = None
        self.test_audio = None

    def setup(self):
        from tools.dashvoice.pipeline import DashVoicePipeline

        self.pipeline = DashVoicePipeline()

        # Create speech-like test audio (2 seconds)
        sample_rate = 16000
        duration_s = 2.0
        t = np.linspace(0, duration_s, int(duration_s * sample_rate))
        # Mix of frequencies to simulate speech
        self.test_audio = np.sin(2 * np.pi * 200 * t) + 0.5 * np.sin(2 * np.pi * 400 * t)
        self.test_audio = (self.test_audio * 0.5).astype(np.float32)

    def run_single(self) -> tuple[float, float | None]:
        start = time.perf_counter()
        self.pipeline.process(self.test_audio, sample_rate=16000)  # Result unused, only timing matters
        elapsed = time.perf_counter() - start
        return elapsed, 2.0  # 2 seconds of audio


class KokoroTTSBenchmark(Benchmark):
    """Benchmark for Kokoro TTS generation using direct Model API.

    Uses direct model.generate() for accurate performance measurement.
    Avoids generate_audio() overhead (model reload, file I/O).
    """

    def __init__(self, warmup_rounds: int = 3, test_rounds: int = 5):
        super().__init__(
            name="Kokoro TTS (short text)",
            target=BenchmarkTarget(
                model_name="kokoro_tts",
                min_rtf=10.0,  # Target: >10x RTF (conservative for CI)
                # Typical performance: 13-15x RTF on Apple Silicon
            ),
            warmup_rounds=warmup_rounds,
            test_rounds=test_rounds,
        )
        self.model = None
        self.test_text = "Hello, how are you doing today?"
        self.voice = "af_bella"
        self.last_result = None

    def setup(self):
        from mlx_audio.tts.utils import load_model

        print("Loading Kokoro model...")
        self.model = load_model("prince-canuma/Kokoro-82M")

    def run_single(self) -> tuple[float, float | None]:
        start = time.perf_counter()
        for result in self.model.generate(
            text=self.test_text,
            voice=self.voice,
            speed=1.0,
            verbose=False,
        ):
            self.last_result = result
        elapsed = time.perf_counter() - start

        # Get audio duration
        audio = np.array(self.last_result.audio)
        audio_duration = len(audio) / self.last_result.sample_rate

        return elapsed, audio_duration

    def teardown(self):
        self.model = None


def run_all_benchmarks(
    models: list[str] | None = None,
    warmup_rounds: int = 3,
    test_rounds: int = 10,
) -> list[BenchmarkResult]:
    """Run all benchmarks or a subset."""

    all_benchmarks = {
        "echo_cancel": EchoCancellationBenchmark,
        "voice_fingerprint": VoiceFingerprintBenchmark,
        "voice_matching": VoiceMatchingBenchmark,
        "pipeline": PipelineBenchmark,
        "kokoro_tts": KokoroTTSBenchmark,
    }

    if models:
        benchmarks = {k: v for k, v in all_benchmarks.items() if k in models}
    else:
        benchmarks = all_benchmarks

    results = []
    for name, benchmark_cls in benchmarks.items():
        try:
            benchmark = benchmark_cls(
                warmup_rounds=warmup_rounds, test_rounds=test_rounds,
            )
            result = benchmark.run()
            results.append(result)
        except Exception as e:
            print(f"\nBenchmark {name} failed: {e}")
            import traceback

            traceback.print_exc()

    return results


def print_summary(results: list[BenchmarkResult]):
    """Print a summary of all benchmark results."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    print(f"\n{'Model':<20} {'Test':<30} {'Latency':<15} {'Status':<10}")
    print("-" * 80)

    passed_count = 0
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        latency = f"{r.latency_ms:.2f}ms"
        if r.rtf is not None:
            latency += f" ({r.rtf:.1f}x RTF)"
        print(f"{r.model_name:<20} {r.test_name:<30} {latency:<15} {status:<10}")
        if r.passed:
            passed_count += 1

    print("-" * 80)
    print(f"Total: {passed_count}/{len(results)} passed")


def main():
    parser = argparse.ArgumentParser(description="DashVoice Performance Benchmarks")
    parser.add_argument(
        "--models",
        nargs="*",
        help="Models to benchmark (default: all)",
    )
    parser.add_argument(
        "--warmup-rounds",
        type=int,
        default=3,
        help="Number of warmup rounds",
    )
    parser.add_argument(
        "--test-rounds",
        type=int,
        default=10,
        help="Number of test rounds",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file for results",
    )

    args = parser.parse_args()

    print("DashVoice Performance Benchmark Suite")
    print("=" * 60)

    results = run_all_benchmarks(
        models=args.models,
        warmup_rounds=args.warmup_rounds,
        test_rounds=args.test_rounds,
    )

    print_summary(results)

    if args.output:
        output_data = [asdict(r) for r in results]
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
