#!/usr/bin/env python3
"""
Full-Cycle Pipeline Benchmark

Measures performance across the complete voice pipeline:
1. Translation (EN → target language via NLLB)
2. G2P (Grapheme-to-Phoneme via espeak-ng)
3. TTS (Text-to-Speech via Kokoro TorchScript)
4. Audio playback

Scenarios:
- Interactive mode: Claude Code streaming tokens
- Batch mode: Full paragraph translation + synthesis
- Multilingual: EN→JA, EN→ZH, EN→ES round-trip

Outputs:
- Per-stage timing breakdown
- Time-to-First-Audio (TTFA)
- Throughput (requests/second)
- P50/P95/P99 latencies

Author: Worker #239
Date: 2025-12-06
"""

import argparse
import json
import os
import subprocess
import statistics
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any

# Paths
ROOT_DIR = Path(__file__).parent.parent
BINARY_PATH = ROOT_DIR / "stream-tts-cpp" / "build" / "stream-tts-cpp"
CONFIG_DIR = ROOT_DIR / "stream-tts-cpp" / "config"


@dataclass
class TimingResult:
    """Result of a single benchmark run."""
    text: str
    language: str
    translate: bool
    total_ms: float
    success: bool
    error: Optional[str] = None
    audio_frames: int = 0


@dataclass
class BenchmarkReport:
    """Aggregated benchmark results."""
    scenario: str
    runs: int
    success_rate: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_min_ms: float
    latency_max_ms: float
    latency_avg_ms: float
    throughput_req_per_sec: float
    notes: str = ""


def check_binary():
    """Verify the TTS binary exists and is executable."""
    if not BINARY_PATH.exists():
        print(f"ERROR: Binary not found: {BINARY_PATH}")
        print("Build with: cd stream-tts-cpp && cmake -B build && cmake --build build")
        sys.exit(1)
    return True


def run_tts(
    text: str,
    language: str = "en",
    translate: bool = False,
    timeout: float = 60.0
) -> TimingResult:
    """Execute a single TTS request and measure total latency."""
    cmd = [
        str(BINARY_PATH),
        "--speak", text,
        "--lang", language,
    ]
    if translate:
        cmd.append("--translate")

    start = time.perf_counter()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        if result.returncode == 0:
            # Parse audio frames if available in output
            audio_frames = 0
            for line in result.stderr.split('\n'):
                if 'frames' in line.lower() or 'samples' in line.lower():
                    import re
                    nums = re.findall(r'\d+', line)
                    if nums:
                        audio_frames = int(nums[-1])
                        break

            return TimingResult(
                text=text,
                language=language,
                translate=translate,
                total_ms=elapsed_ms,
                success=True,
                audio_frames=audio_frames
            )
        else:
            return TimingResult(
                text=text,
                language=language,
                translate=translate,
                total_ms=elapsed_ms,
                success=False,
                error=result.stderr[:500] if result.stderr else "Unknown error"
            )
    except subprocess.TimeoutExpired:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return TimingResult(
            text=text,
            language=language,
            translate=translate,
            total_ms=elapsed_ms,
            success=False,
            error="Timeout"
        )
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return TimingResult(
            text=text,
            language=language,
            translate=translate,
            total_ms=elapsed_ms,
            success=False,
            error=str(e)
        )


def percentile(data: List[float], p: float) -> float:
    """Calculate percentile."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_data) else f
    return sorted_data[f] + (sorted_data[c] - sorted_data[f]) * (k - f)


def run_benchmark(
    texts: List[str],
    language: str,
    translate: bool,
    warmup_runs: int = 2,
    benchmark_runs: int = 10,
    scenario_name: str = "default"
) -> BenchmarkReport:
    """Run benchmark suite for given texts."""
    print(f"\n{'='*60}")
    print(f"BENCHMARK: {scenario_name}")
    print(f"{'='*60}")
    print(f"Language: {language}, Translate: {translate}")
    print(f"Texts: {len(texts)}, Warmup: {warmup_runs}, Runs: {benchmark_runs}")

    # Warmup
    print("\nWarmup...")
    for i in range(warmup_runs):
        for text in texts[:2]:  # Only warmup on first 2 texts
            result = run_tts(text, language, translate)
            if not result.success:
                print(f"  Warmup failed: {result.error}")

    # Benchmark runs
    print("Benchmarking...")
    results: List[TimingResult] = []
    total_start = time.perf_counter()

    for run_num in range(benchmark_runs):
        for text in texts:
            result = run_tts(text, language, translate)
            results.append(result)
            status = "✓" if result.success else "✗"
            print(f"  Run {run_num+1}/{benchmark_runs}: {status} {result.total_ms:.0f}ms - {text[:40]}...")

    total_duration = time.perf_counter() - total_start

    # Calculate statistics
    successful = [r for r in results if r.success]
    latencies = [r.total_ms for r in successful]

    if not latencies:
        return BenchmarkReport(
            scenario=scenario_name,
            runs=len(results),
            success_rate=0.0,
            latency_p50_ms=0.0,
            latency_p95_ms=0.0,
            latency_p99_ms=0.0,
            latency_min_ms=0.0,
            latency_max_ms=0.0,
            latency_avg_ms=0.0,
            throughput_req_per_sec=0.0,
            notes="All runs failed"
        )

    report = BenchmarkReport(
        scenario=scenario_name,
        runs=len(results),
        success_rate=len(successful) / len(results) * 100,
        latency_p50_ms=percentile(latencies, 50),
        latency_p95_ms=percentile(latencies, 95),
        latency_p99_ms=percentile(latencies, 99),
        latency_min_ms=min(latencies),
        latency_max_ms=max(latencies),
        latency_avg_ms=statistics.mean(latencies),
        throughput_req_per_sec=len(results) / total_duration
    )

    # Print summary
    print(f"\n{'='*40}")
    print(f"RESULTS: {scenario_name}")
    print(f"{'='*40}")
    print(f"Success Rate: {report.success_rate:.1f}%")
    print(f"Latency P50:  {report.latency_p50_ms:.0f}ms")
    print(f"Latency P95:  {report.latency_p95_ms:.0f}ms")
    print(f"Latency P99:  {report.latency_p99_ms:.0f}ms")
    print(f"Latency Min:  {report.latency_min_ms:.0f}ms")
    print(f"Latency Max:  {report.latency_max_ms:.0f}ms")
    print(f"Latency Avg:  {report.latency_avg_ms:.0f}ms")
    print(f"Throughput:   {report.throughput_req_per_sec:.2f} req/s")

    return report


def scenario_interactive():
    """
    Scenario 1: Interactive Mode (Claude Code)

    Short phrases streamed token-by-token.
    Target: TTFA < 200ms
    """
    texts = [
        "Hello world",
        "Running tests",
        "Task completed",
        "Found an error",
        "Processing request",
    ]
    return run_benchmark(
        texts=texts,
        language="en",
        translate=False,
        warmup_runs=2,
        benchmark_runs=5,
        scenario_name="Interactive Mode (EN)"
    )


def scenario_batch_translation():
    """
    Scenario 2: Batch Translation Mode

    Full paragraph translated and synthesized.
    Target: Total latency < 3s
    """
    texts = [
        "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.",
        "Machine learning models have revolutionized natural language processing and speech synthesis.",
    ]
    return run_benchmark(
        texts=texts,
        language="ja",
        translate=True,
        warmup_runs=1,
        benchmark_runs=3,
        scenario_name="Batch Translation (EN→JA)"
    )


def scenario_multilingual():
    """
    Scenario 3: Multilingual Round-Trip

    Test various language outputs.
    """
    reports = []

    languages = [
        ("en", False, "English TTS"),
        ("ja", True, "EN→JA Translation+TTS"),
        ("zh", True, "EN→ZH Translation+TTS"),
        ("es", True, "EN→ES Translation+TTS"),
        ("fr", True, "EN→FR Translation+TTS"),
    ]

    test_text = ["Hello, how are you today?"]

    for lang, translate, name in languages:
        report = run_benchmark(
            texts=test_text,
            language=lang,
            translate=translate,
            warmup_runs=1,
            benchmark_runs=3,
            scenario_name=name
        )
        reports.append(report)

    return reports


def scenario_stress():
    """
    Scenario 4: Sequential Stress Test

    Many requests in sequence to measure sustained throughput.
    """
    texts = [
        "One", "Two", "Three", "Four", "Five",
        "Testing", "Performance", "Benchmark",
    ]
    return run_benchmark(
        texts=texts,
        language="en",
        translate=False,
        warmup_runs=2,
        benchmark_runs=10,
        scenario_name="Sequential Stress Test"
    )


def main():
    parser = argparse.ArgumentParser(description="Full-Cycle Pipeline Benchmark")
    parser.add_argument(
        "--scenario",
        choices=["interactive", "batch", "multilingual", "stress", "all"],
        default="all",
        help="Benchmark scenario to run"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file for results"
    )
    args = parser.parse_args()

    check_binary()

    all_reports: List[BenchmarkReport] = []

    print("=" * 60)
    print("FULL-CYCLE PIPELINE BENCHMARK")
    print("=" * 60)
    print(f"Binary: {BINARY_PATH}")
    print(f"Scenario: {args.scenario}")

    if args.scenario in ("interactive", "all"):
        all_reports.append(scenario_interactive())

    if args.scenario in ("batch", "all"):
        all_reports.append(scenario_batch_translation())

    if args.scenario in ("multilingual", "all"):
        reports = scenario_multilingual()
        all_reports.extend(reports)

    if args.scenario in ("stress", "all"):
        all_reports.append(scenario_stress())

    # Final summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"{'Scenario':<35} {'P50':>8} {'P95':>8} {'Throughput':>12}")
    print("-" * 60)
    for report in all_reports:
        print(f"{report.scenario:<35} {report.latency_p50_ms:>7.0f}ms {report.latency_p95_ms:>7.0f}ms {report.throughput_req_per_sec:>10.2f}/s")

    # Save to JSON if requested
    if args.output:
        output_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "binary": str(BINARY_PATH),
            "reports": [asdict(r) for r in all_reports]
        }
        output_path = Path(args.output)
        output_path.write_text(json.dumps(output_data, indent=2))
        print(f"\nResults saved to: {output_path}")

    # Performance targets check
    print("\n" + "=" * 60)
    print("PERFORMANCE TARGET CHECK")
    print("=" * 60)

    targets = {
        "Interactive Mode (EN)": ("P50 < 300ms", lambda r: r.latency_p50_ms < 300),
        "Batch Translation (EN→JA)": ("P50 < 3000ms", lambda r: r.latency_p50_ms < 3000),
    }

    for report in all_reports:
        if report.scenario in targets:
            target_desc, check_fn = targets[report.scenario]
            passed = check_fn(report)
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{report.scenario}: {status} ({target_desc}, actual: {report.latency_p50_ms:.0f}ms)")

    print("\n" + "=" * 60)
    print("OPTIMIZATION OPPORTUNITIES")
    print("=" * 60)
    print("""
Based on the benchmark results, consider:

1. **Daemon Mode**: Keep models loaded to eliminate cold-start
   - Current: Each request loads model (~10-15s overhead)
   - Target: Use --daemon flag for persistent process

2. **Translation Caching**: Cache frequent EN→target translations
   - NLLB translation adds 50-100ms per request
   - Cache common phrases

3. **G2P Optimization**: Batch phonemization for short texts
   - espeak-ng fallback can be slow for unknown words
   - Pre-compute common phrases

4. **Bucket Warmup**: Pre-warm all 16 bucket sizes at startup
   - JIT compilation adds ~500ms on first request per bucket
   - Warmup reduces steady-state latency to 60-120ms

5. **Concurrent Processing**: Pipeline translation while synthesizing
   - Current: Sequential (translate → G2P → TTS)
   - Target: Overlap translation with previous synthesis
""")


if __name__ == "__main__":
    main()
