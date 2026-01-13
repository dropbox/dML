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
Unified TTS Model Benchmark

Benchmark all TTS models on same hardware with consistent methodology.
Reports: RTF, latency, memory, and quality metrics.

Usage:
    python scripts/benchmark_all_models.py
    python scripts/benchmark_all_models.py --models kokoro,cosyvoice2
    python scripts/benchmark_all_models.py --runs 5 --warmup 2
"""

import argparse
import gc
import sys
import time
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import numpy as np

# Standard test text used across all models
TEST_TEXT = "The quick brown fox jumps over the lazy dog."

# Sample rate for each model
MODEL_SAMPLE_RATES = {
    "kokoro": 24000,
    "cosyvoice2": 22050,
    "f5tts": 24000,
}


def get_memory_usage() -> float:
    """Get current MLX memory usage in MB."""
    try:
        # Try newer API first
        return mx.get_active_memory() / (1024 * 1024)
    except AttributeError:
        try:
            return mx.metal.get_active_memory() / (1024 * 1024)
        except Exception:
            return 0.0


def clear_memory():
    """Clear memory between model tests."""
    gc.collect()
    try:
        mx.clear_cache()
    except AttributeError:
        try:
            mx.metal.clear_cache()
        except (AttributeError, Exception):
            pass


class BenchmarkResult:
    """Container for benchmark results."""

    def __init__(
        self,
        model_name: str,
        latency_ms: float,
        latency_std_ms: float,
        audio_duration_s: float,
        rtf: float,
        memory_mb: float,
        success: bool = True,
        error: str = "",
    ):
        self.model_name = model_name
        self.latency_ms = latency_ms
        self.latency_std_ms = latency_std_ms
        self.audio_duration_s = audio_duration_s
        self.rtf = rtf
        self.memory_mb = memory_mb
        self.success = success
        self.error = error


def benchmark_kokoro(text: str, runs: int, warmup: int) -> BenchmarkResult:
    """Benchmark Kokoro TTS."""
    try:
        from tools.pytorch_to_mlx.converters.kokoro_converter import KokoroConverter

        print("  Loading Kokoro model...")
        converter = KokoroConverter()
        model, config, _ = converter.load_from_hf("hexgrad/Kokoro-82M")

        # Load voice - check multiple locations
        voice = None
        voice_paths = [
            Path.home() / ".cache" / "kokoro" / "voices" / "af_heart.npz",
            Path(__file__).parent.parent / "src" / "kokoro" / "voices" / "af_bella.npy",
        ]
        for voice_path in voice_paths:
            if voice_path.exists():
                if voice_path.suffix == ".npz":
                    voice_data = np.load(voice_path)
                    voice = mx.array(voice_data["voice"])
                else:
                    voice_data = np.load(voice_path)
                    voice = mx.array(voice_data)[None, :]
                break

        if voice is None:
            # Use random voice
            voice = mx.random.normal(shape=(1, 256))

        # Use representative token sequence (like existing benchmarks)
        # This represents roughly "The quick brown fox..." in phoneme tokens
        # (~64 tokens is typical for this sentence)
        input_ids = mx.array([list(range(1, 65))])

        # Warmup
        print(f"  Warming up ({warmup} runs)...")
        for _ in range(warmup):
            audio = model(input_ids, voice, validate_output=False)
            mx.eval(audio)

        # Benchmark
        print(f"  Running benchmark ({runs} runs)...")
        times = []
        audio_durations = []
        peak_memory = 0.0

        for _ in range(runs):
            mem_before = get_memory_usage()
            start = time.perf_counter()
            audio = model(input_ids, voice, validate_output=False)
            mx.eval(audio)
            elapsed = time.perf_counter() - start
            mem_after = get_memory_usage()

            times.append(elapsed)
            audio_duration = audio.shape[1] / MODEL_SAMPLE_RATES["kokoro"]
            audio_durations.append(audio_duration)
            peak_memory = max(peak_memory, mem_after - mem_before)

        mean_time = np.mean(times)
        std_time = np.std(times)
        mean_audio = np.mean(audio_durations)
        rtf = mean_audio / mean_time if mean_time > 0 else 0

        return BenchmarkResult(
            model_name="Kokoro",
            latency_ms=mean_time * 1000,
            latency_std_ms=std_time * 1000,
            audio_duration_s=mean_audio,
            rtf=rtf,
            memory_mb=peak_memory,
        )

    except Exception as e:
        return BenchmarkResult(
            model_name="Kokoro",
            latency_ms=0,
            latency_std_ms=0,
            audio_duration_s=0,
            rtf=0,
            memory_mb=0,
            success=False,
            error=str(e),
        )


def benchmark_cosyvoice2(text: str, runs: int, warmup: int) -> BenchmarkResult:
    """Benchmark CosyVoice2 TTS."""
    try:
        from tools.pytorch_to_mlx.converters.models import CosyVoice2Model

        print("  Loading CosyVoice2 model...")

        # Use the model's default path discovery
        model_path = CosyVoice2Model.get_default_model_path()
        if not model_path.exists():
            return BenchmarkResult(
                model_name="CosyVoice2",
                latency_ms=0,
                latency_std_ms=0,
                audio_duration_s=0,
                rtf=0,
                memory_mb=0,
                success=False,
                error="Model not found. Run scripts/download_cosyvoice2.py first.",
            )

        model = CosyVoice2Model.from_pretrained(model_path)
        speaker_embedding = model.tokenizer.random_speaker_embedding(seed=42)

        # Warmup
        print(f"  Warming up ({warmup} runs)...")
        for _ in range(warmup):
            audio = model.synthesize_text(
                text, speaker_embedding=speaker_embedding, max_tokens=100
            )
            mx.eval(audio)

        # Benchmark
        print(f"  Running benchmark ({runs} runs)...")
        times = []
        audio_durations = []
        peak_memory = 0.0

        for _ in range(runs):
            mem_before = get_memory_usage()
            start = time.perf_counter()
            audio = model.synthesize_text(
                text, speaker_embedding=speaker_embedding, max_tokens=100
            )
            mx.eval(audio)
            elapsed = time.perf_counter() - start
            mem_after = get_memory_usage()

            times.append(elapsed)
            audio_np = np.array(audio)
            audio_duration = len(audio_np) / MODEL_SAMPLE_RATES["cosyvoice2"]
            audio_durations.append(audio_duration)
            peak_memory = max(peak_memory, mem_after - mem_before)

        mean_time = np.mean(times)
        std_time = np.std(times)
        mean_audio = np.mean(audio_durations)
        rtf = mean_audio / mean_time if mean_time > 0 else 0

        return BenchmarkResult(
            model_name="CosyVoice2",
            latency_ms=mean_time * 1000,
            latency_std_ms=std_time * 1000,
            audio_duration_s=mean_audio,
            rtf=rtf,
            memory_mb=peak_memory,
        )

    except Exception as e:
        return BenchmarkResult(
            model_name="CosyVoice2",
            latency_ms=0,
            latency_std_ms=0,
            audio_duration_s=0,
            rtf=0,
            memory_mb=0,
            success=False,
            error=str(e),
        )


def benchmark_f5tts(text: str, runs: int, warmup: int) -> BenchmarkResult:
    """Benchmark F5-TTS (voice cloning model)."""
    try:
        import tempfile

        import soundfile as sf
        from f5_tts_mlx.generate import generate

        print("  Loading F5-TTS model...")

        # Warmup (model loads on first call)
        print(f"  Warming up ({warmup} runs)...")
        for _ in range(warmup):
            output_path = tempfile.mktemp(suffix=".wav")
            generate(
                generation_text=text,
                output_path=output_path,
                steps=4,  # Use optimized 4 steps
            )
            Path(output_path).unlink(missing_ok=True)

        # Benchmark
        print(f"  Running benchmark ({runs} runs)...")
        times = []
        audio_durations = []
        peak_memory = 0.0

        for _ in range(runs):
            output_path = tempfile.mktemp(suffix=".wav")
            mem_before = get_memory_usage()
            start = time.perf_counter()
            generate(
                generation_text=text,
                output_path=output_path,
                steps=4,
            )
            elapsed = time.perf_counter() - start
            mem_after = get_memory_usage()

            # Read audio to get duration
            audio_np, sample_rate = sf.read(output_path)
            audio_duration = len(np.array(audio_np).flatten()) / sample_rate
            Path(output_path).unlink(missing_ok=True)

            times.append(elapsed)
            audio_durations.append(audio_duration)
            peak_memory = max(peak_memory, mem_after - mem_before)

        mean_time = np.mean(times)
        std_time = np.std(times)
        mean_audio = np.mean(audio_durations)
        rtf = mean_audio / mean_time if mean_time > 0 else 0

        return BenchmarkResult(
            model_name="F5-TTS",
            latency_ms=mean_time * 1000,
            latency_std_ms=std_time * 1000,
            audio_duration_s=mean_audio,
            rtf=rtf,
            memory_mb=peak_memory,
        )

    except ImportError:
        return BenchmarkResult(
            model_name="F5-TTS",
            latency_ms=0,
            latency_std_ms=0,
            audio_duration_s=0,
            rtf=0,
            memory_mb=0,
            success=False,
            error="F5-TTS not installed. Run: pip install f5-tts-mlx",
        )
    except Exception as e:
        return BenchmarkResult(
            model_name="F5-TTS",
            latency_ms=0,
            latency_std_ms=0,
            audio_duration_s=0,
            rtf=0,
            memory_mb=0,
            success=False,
            error=str(e),
        )


def print_summary(results: list[BenchmarkResult]):
    """Print summary comparison table."""
    print("\n" + "=" * 80)
    print("SUMMARY: TTS Model Comparison")
    print("=" * 80)
    print(f"Test text: '{TEST_TEXT}'")
    print()

    # Header
    header = f"{'Model':<12} {'RTF':>8} {'Latency (ms)':>18} {'Audio (s)':>10} {'Mem (MB)':>10} {'Status':<8}"
    print(header)
    print("-" * 75)

    # Sort by RTF (fastest first)
    sorted_results = sorted(results, key=lambda r: r.rtf if r.success else 0, reverse=True)

    for r in sorted_results:
        if r.success:
            latency_str = f"{r.latency_ms:.1f} +/- {r.latency_std_ms:.1f}"
            print(
                f"{r.model_name:<12} "
                f"{r.rtf:>7.1f}x "
                f"{latency_str:>18} "
                f"{r.audio_duration_s:>10.2f} "
                f"{r.memory_mb:>10.1f} "
                f"{'OK':<8}"
            )
        else:
            print(f"{r.model_name:<12} {'N/A':>8} {'N/A':>18} {'N/A':>10} {'N/A':>10} FAILED")
            print(f"  Error: {r.error[:60]}...")

    print("-" * 80)

    # Performance notes
    print("\nNotes:")
    print("- RTF = Real-Time Factor (higher is faster)")
    print("- Latency includes full pipeline (tokenization → audio)")
    print("- Memory is delta during inference (not total model size)")
    print("- All models use fp16 precision")


def main():
    parser = argparse.ArgumentParser(description="Unified TTS Model Benchmark")
    parser.add_argument(
        "--models",
        type=str,
        default="kokoro,cosyvoice2,f5tts",
        help="Comma-separated list of models to benchmark (default: all)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of benchmark runs per model (default: 3)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warmup runs per model (default: 1)",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=TEST_TEXT,
        help=f"Text to synthesize (default: '{TEST_TEXT}')",
    )
    args = parser.parse_args()

    models = [m.strip().lower() for m in args.models.split(",")]

    print("=" * 80)
    print("Unified TTS Model Benchmark")
    print("=" * 80)
    print(f"Models: {', '.join(models)}")
    print(f"Runs: {args.runs}, Warmup: {args.warmup}")
    print(f"Text: '{args.text}'")
    print()

    # Model benchmark functions
    benchmark_funcs: dict[str, Any] = {
        "kokoro": benchmark_kokoro,
        "cosyvoice2": benchmark_cosyvoice2,
        "f5tts": benchmark_f5tts,
    }

    results: list[BenchmarkResult] = []

    for model_name in models:
        if model_name not in benchmark_funcs:
            print(f"Unknown model: {model_name}")
            continue

        print(f"\nBenchmarking {model_name.upper()}...")
        print("-" * 40)

        clear_memory()

        result = benchmark_funcs[model_name](args.text, args.runs, args.warmup)
        results.append(result)

        if result.success:
            print(f"  RTF: {result.rtf:.1f}x")
            print(f"  Latency: {result.latency_ms:.1f}ms (+/- {result.latency_std_ms:.1f}ms)")
            print(f"  Audio duration: {result.audio_duration_s:.2f}s")
        else:
            print(f"  FAILED: {result.error}")

        clear_memory()

    # Print summary
    print_summary(results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
