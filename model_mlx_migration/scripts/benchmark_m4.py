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
M4 Baseline Benchmark for Whisper MLX

Measures:
- Cold start latency
- Inference throughput
- Memory usage
- First partial latency (CTC)
- GPU utilization

Usage:
    python scripts/benchmark_m4.py
    python scripts/benchmark_m4.py --quick  # Fast sanity check
    python scripts/benchmark_m4.py --full   # Complete benchmark
"""

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class BenchmarkResult:
    """Single benchmark measurement."""
    name: str
    value: float
    unit: str
    target: Optional[float] = None
    passed: Optional[bool] = None

    def __post_init__(self):
        if self.target is not None:
            # Lower is better for latency/memory, higher for throughput
            if self.unit in ['ms', 's', 'MB', 'GB']:
                self.passed = self.value <= self.target
            else:
                self.passed = self.value >= self.target


@dataclass
class BenchmarkSuite:
    """Complete benchmark results."""
    device: str
    timestamp: str
    results: list
    summary: dict


def get_system_info() -> dict:
    """Get M4 system information."""
    info = {
        'chip': 'Unknown',
        'cores': os.cpu_count(),
        'memory_gb': 0,
    }

    try:
        # Get chip info
        result = subprocess.run(
            ['sysctl', '-n', 'machdep.cpu.brand_string'],
            capture_output=True, text=True
        )
        info['chip'] = result.stdout.strip()

        # Get memory
        result = subprocess.run(
            ['sysctl', '-n', 'hw.memsize'],
            capture_output=True, text=True
        )
        info['memory_gb'] = int(result.stdout.strip()) / (1024**3)

        # Get GPU cores (Apple Silicon specific)
        result = subprocess.run(
            ['system_profiler', 'SPDisplaysDataType', '-json'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            import json as json_mod
            data = json_mod.loads(result.stdout)
            # Parse GPU info from system_profiler

    except Exception as e:
        print(f"Warning: Could not get system info: {e}")

    return info


def get_memory_mb() -> float:
    """Get current process memory usage in MB."""
    try:
        result = subprocess.run(
            ['ps', '-o', 'rss=', '-p', str(os.getpid())],
            capture_output=True, text=True
        )
        return int(result.stdout.strip()) / 1024  # KB to MB
    except:
        return 0


def benchmark_cold_start() -> BenchmarkResult:
    """Measure time to load model from disk."""
    print("\n[1/6] Cold Start Benchmark...")

    # Force garbage collection
    gc.collect()

    import mlx.core as mx

    start = time.perf_counter()

    # Import and load model
    from tools.whisper_mlx import WhisperMLX
    model = WhisperMLX.from_pretrained('mlx-community/whisper-large-v3-mlx')

    # Force evaluation (actually load weights)
    mx.eval(model.encoder.conv1.weight)

    end = time.perf_counter()
    elapsed = end - start

    print(f"   Cold start: {elapsed:.2f}s")

    return BenchmarkResult(
        name="Cold Start",
        value=elapsed,
        unit="s",
        target=2.0  # Target: <2s
    )


def benchmark_memory_idle() -> BenchmarkResult:
    """Measure memory usage with no model loaded."""
    print("\n[2/6] Memory (Idle) Benchmark...")

    gc.collect()
    time.sleep(0.5)

    memory_mb = get_memory_mb()
    print(f"   Idle memory: {memory_mb:.0f} MB")

    return BenchmarkResult(
        name="Memory (Idle)",
        value=memory_mb,
        unit="MB",
        target=200  # Target: <200MB baseline
    )


def benchmark_memory_loaded() -> BenchmarkResult:
    """Measure memory usage with Whisper loaded."""
    print("\n[3/6] Memory (Model Loaded) Benchmark...")

    import mlx.core as mx
    from tools.whisper_mlx import WhisperMLX

    # Load model
    model = WhisperMLX.from_pretrained('mlx-community/whisper-large-v3-mlx')
    mx.eval(model.encoder.conv1.weight)

    gc.collect()
    time.sleep(0.5)

    memory_mb = get_memory_mb()
    print(f"   Loaded memory: {memory_mb:.0f} MB")

    return BenchmarkResult(
        name="Memory (Loaded)",
        value=memory_mb,
        unit="MB",
        target=4000  # Target: <4GB
    )


def benchmark_inference(num_runs: int = 5) -> list:
    """Measure inference latency and throughput."""
    print(f"\n[4/6] Inference Benchmark ({num_runs} runs)...")

    import mlx.core as mx
    from tools.whisper_mlx import WhisperMLX
    from tools.whisper_mlx.audio import load_audio

    # Find a test file
    test_files = list(Path("data/librispeech/test-clean").rglob("*.flac"))[:1]
    if not test_files:
        # Create synthetic audio
        print("   No test files found, using synthetic audio...")
        audio = mx.zeros((16000 * 10,))  # 10 seconds of silence
        audio_duration = 10.0
    else:
        audio = load_audio(str(test_files[0]))
        audio_duration = len(audio) / 16000
        print(f"   Test file: {test_files[0].name} ({audio_duration:.1f}s)")

    # Load model
    model = WhisperMLX.from_pretrained('mlx-community/whisper-large-v3-mlx')

    # Warmup
    print("   Warming up...")
    _ = model.transcribe(audio[:16000])  # 1 second
    mx.eval()

    # Benchmark runs
    latencies = []
    for i in range(num_runs):
        gc.collect()

        start = time.perf_counter()
        result = model.transcribe(audio)
        mx.eval()
        end = time.perf_counter()

        latency = (end - start) * 1000  # ms
        latencies.append(latency)
        print(f"   Run {i+1}: {latency:.0f}ms")

    avg_latency = sum(latencies) / len(latencies)
    throughput = audio_duration / (avg_latency / 1000)  # x realtime

    print(f"   Average: {avg_latency:.0f}ms ({throughput:.1f}x realtime)")

    return [
        BenchmarkResult(
            name="Inference Latency (avg)",
            value=avg_latency,
            unit="ms",
            target=500  # Target: <500ms for 10s audio
        ),
        BenchmarkResult(
            name="Throughput",
            value=throughput,
            unit="x realtime",
            target=10.0  # Target: >10x realtime
        ),
    ]


def benchmark_ctc_first_partial() -> BenchmarkResult:
    """Measure time to first CTC draft."""
    print("\n[5/6] CTC First Partial Benchmark...")

    try:
        import mlx.core as mx
        from tools.whisper_mlx import WhisperMLX
        from tools.whisper_mlx.ctc_head import CTCDraftHead

        # Check if CTC checkpoint exists
        ctc_checkpoint = Path("checkpoints/ctc_head_large_v3/best.npz")
        if not ctc_checkpoint.exists():
            ctc_checkpoint = Path("checkpoints/ctc_head_large_v3/step_7000.npz")

        if not ctc_checkpoint.exists():
            print("   CTC checkpoint not found, skipping...")
            return BenchmarkResult(
                name="First Partial (CTC)",
                value=float('inf'),
                unit="ms",
                target=200
            )

        # Load models
        model = WhisperMLX.from_pretrained('mlx-community/whisper-large-v3-mlx')
        ctc_head = CTCDraftHead(d_model=1280, vocab_size=51865)

        # Load CTC weights
        weights = mx.load(str(ctc_checkpoint))
        ctc_weights = {k.replace('ctc_head.', ''): v for k, v in weights.items()
                       if k.startswith('ctc_head.')}
        if ctc_weights:
            ctc_head.load_weights(list(ctc_weights.items()))

        # Create test audio (1 second)
        audio = mx.zeros((16000,))

        # Warmup
        mel = model.preprocessor(audio)
        encoder_out = model.encoder(mel)
        _ = ctc_head(encoder_out)
        mx.eval()

        # Benchmark
        start = time.perf_counter()
        mel = model.preprocessor(audio)
        encoder_out = model.encoder(mel)
        ctc_output = ctc_head(encoder_out)
        mx.eval()
        end = time.perf_counter()

        latency = (end - start) * 1000
        print(f"   First partial: {latency:.0f}ms")

        return BenchmarkResult(
            name="First Partial (CTC)",
            value=latency,
            unit="ms",
            target=200  # Target: <200ms
        )

    except Exception as e:
        print(f"   Error: {e}")
        return BenchmarkResult(
            name="First Partial (CTC)",
            value=float('inf'),
            unit="ms",
            target=200
        )


def benchmark_multi_head() -> BenchmarkResult:
    """Measure multi-head inference overhead."""
    print("\n[6/6] Multi-Head Overhead Benchmark...")

    try:
        import mlx.core as mx
        from tools.whisper_mlx import WhisperMLX
        from tools.whisper_mlx.multi_head import WhisperMultiHead, MultiHeadConfig

        # Create multi-head model with default config
        # MultiHeadConfig uses d_model=1280 by default for large-v3
        config = MultiHeadConfig()

        print("   Loading Whisper + multi-head model...")
        whisper = WhisperMLX.from_pretrained('mlx-community/whisper-large-v3-mlx')
        multi_head = WhisperMultiHead(config)

        # Create test encoder output (simulated 5s audio)
        # Whisper large-v3: 1280 dim, ~150 frames per 30s = ~25 frames per 5s
        encoder_output = mx.zeros((1, 25, 1280))

        # Warmup
        _ = multi_head(encoder_output)
        mx.eval()

        # Benchmark multi-head overhead only (not encoder)
        start = time.perf_counter()
        outputs = multi_head(encoder_output)
        mx.eval()
        end = time.perf_counter()

        latency = (end - start) * 1000
        print(f"   Multi-head overhead: {latency:.1f}ms")

        return BenchmarkResult(
            name="Multi-Head Overhead",
            value=latency,
            unit="ms",
            target=50  # Target: <50ms overhead for heads only
        )

    except Exception as e:
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return BenchmarkResult(
            name="Multi-Head Overhead",
            value=float('inf'),
            unit="ms",
            target=50
        )


def run_quick_benchmark() -> BenchmarkSuite:
    """Run fast sanity check."""
    print("=" * 60)
    print("M4 QUICK BENCHMARK")
    print("=" * 60)

    results = []

    # Memory only
    results.append(benchmark_memory_idle())

    # Single inference
    import mlx.core as mx
    from tools.whisper_mlx import WhisperMLX

    print("\n[Quick] Single Inference...")
    model = WhisperMLX.from_pretrained('mlx-community/whisper-large-v3-mlx')
    audio = mx.zeros((16000 * 3,))  # 3 seconds

    start = time.perf_counter()
    _ = model.transcribe(audio)
    mx.eval()
    end = time.perf_counter()

    results.append(BenchmarkResult(
        name="Quick Inference (3s audio)",
        value=(end - start) * 1000,
        unit="ms",
        target=300
    ))

    return create_suite(results)


def run_full_benchmark() -> BenchmarkSuite:
    """Run complete benchmark suite."""
    print("=" * 60)
    print("M4 FULL BENCHMARK")
    print("=" * 60)

    results = []

    results.append(benchmark_memory_idle())
    results.append(benchmark_cold_start())
    results.append(benchmark_memory_loaded())
    results.extend(benchmark_inference(num_runs=5))
    results.append(benchmark_ctc_first_partial())
    results.append(benchmark_multi_head())

    return create_suite(results)


def create_suite(results: list) -> BenchmarkSuite:
    """Create benchmark suite with summary."""
    from datetime import datetime

    system_info = get_system_info()

    passed = sum(1 for r in results if r.passed is True)
    failed = sum(1 for r in results if r.passed is False)
    skipped = sum(1 for r in results if r.passed is None)

    summary = {
        'passed': passed,
        'failed': failed,
        'skipped': skipped,
        'total': len(results),
        'pass_rate': passed / max(1, passed + failed) * 100,
    }

    return BenchmarkSuite(
        device=system_info.get('chip', 'Unknown'),
        timestamp=datetime.now().isoformat(),
        results=[asdict(r) for r in results],
        summary=summary,
    )


def print_results(suite: BenchmarkSuite):
    """Print formatted results."""
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Device: {suite.device}")
    print(f"Time: {suite.timestamp}")
    print()

    for r in suite.results:
        status = ""
        if r['passed'] is True:
            status = "PASS"
        elif r['passed'] is False:
            status = "FAIL"
        else:
            status = "SKIP"

        target_str = ""
        if r['target'] is not None:
            target_str = f" (target: {r['target']}{r['unit']})"

        print(f"  [{status}] {r['name']}: {r['value']:.1f}{r['unit']}{target_str}")

    print()
    print(f"Summary: {suite.summary['passed']}/{suite.summary['total']} passed "
          f"({suite.summary['pass_rate']:.0f}%)")


def main():
    parser = argparse.ArgumentParser(description="M4 Benchmark for Whisper MLX")
    parser.add_argument('--quick', action='store_true', help='Quick sanity check')
    parser.add_argument('--full', action='store_true', help='Full benchmark (default)')
    parser.add_argument('--output', type=str, help='Save results to JSON file')

    args = parser.parse_args()

    # Default to full if neither specified
    if not args.quick and not args.full:
        args.full = True

    try:
        if args.quick:
            suite = run_quick_benchmark()
        else:
            suite = run_full_benchmark()

        print_results(suite)

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(asdict(suite), f, indent=2)
            print(f"\nResults saved to: {output_path}")

    except KeyboardInterrupt:
        print("\nBenchmark interrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"\nBenchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
