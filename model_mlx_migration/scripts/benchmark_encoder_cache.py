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
Benchmark encoder caching for WhisperMLX (OPT-W4).

Tests the speedup for repeated queries on the same audio with encoder caching.

Expected speedup:
- 2x for repeated queries (encoder is ~50% of standard mode time)
- Variable-length mode may show different characteristics

Usage:
    python scripts/benchmark_encoder_cache.py [--audio PATH] [--model MODEL]
"""

import argparse
import sys
import time

import numpy as np


def generate_test_audio(duration: float = 6.37, sample_rate: int = 16000) -> np.ndarray:
    """Generate synthetic test audio (sine waves with noise)."""
    t = np.linspace(0, duration, int(duration * sample_rate), dtype=np.float32)
    # Mix of frequencies to simulate speech-like audio
    audio = (
        0.5 * np.sin(2 * np.pi * 440 * t) +
        0.3 * np.sin(2 * np.pi * 880 * t) +
        0.2 * np.sin(2 * np.pi * 220 * t) +
        0.1 * np.random.randn(len(t))
    )
    return audio.astype(np.float32)


def load_audio_file(audio_path: str, sample_rate: int = 16000) -> np.ndarray:
    """Load audio file using ffmpeg."""
    try:
        import io
        import subprocess
        import wave

        # Use ffmpeg to convert to wav
        cmd = [
            "ffmpeg", "-i", audio_path,
            "-ac", "1",  # mono
            "-ar", str(sample_rate),
            "-f", "wav",
            "-"
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()}")

        # Parse wav from stdout
        wav_file = io.BytesIO(result.stdout)
        with wave.open(wav_file, 'rb') as wf:
            frames = wf.readframes(wf.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            return audio

    except Exception as e:
        print(f"Warning: Could not load audio file: {e}")
        print("Using synthetic audio instead")
        return generate_test_audio()


def benchmark_encoder_cache(
    audio: np.ndarray,
    model_name: str = "large-v3",
    num_iterations: int = 3,
    variable_length: bool = False,
) -> dict:
    """
    Benchmark encoder caching performance.

    Args:
        audio: Audio waveform (16kHz mono)
        model_name: Whisper model name
        num_iterations: Number of timing iterations
        variable_length: Use variable-length mode

    Returns:
        Dictionary with benchmark results
    """
    try:
        import mlx.core as mx

        from tools.whisper_mlx import WhisperMLX
    except ImportError as e:
        print(f"Error importing WhisperMLX: {e}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("Encoder Cache Benchmark (OPT-W4)")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Audio duration: {len(audio) / 16000:.2f}s")
    print(f"Variable-length mode: {variable_length}")
    print(f"Iterations: {num_iterations}")
    print()

    # Load model
    print("Loading model...")
    t0 = time.perf_counter()
    model = WhisperMLX.from_pretrained(model_name)
    load_time = time.perf_counter() - t0
    print(f"Model loaded in {load_time:.2f}s")

    results = {}

    # =========================================================================
    # Benchmark WITHOUT cache
    # =========================================================================
    print("\n--- Without encoder cache ---")
    model.disable_encoder_cache()

    times_no_cache = []
    for i in range(num_iterations):
        t0 = time.perf_counter()
        result = model.transcribe(
            audio.copy(),  # Copy to avoid any caching effects
            variable_length=variable_length,
        )
        mx.eval(result)  # Ensure computation complete
        elapsed = time.perf_counter() - t0
        times_no_cache.append(elapsed)
        print(f"  Iteration {i+1}: {elapsed*1000:.1f}ms")

    avg_no_cache = np.mean(times_no_cache)
    std_no_cache = np.std(times_no_cache)
    print(f"  Average: {avg_no_cache*1000:.1f}ms (±{std_no_cache*1000:.1f}ms)")

    results["no_cache"] = {
        "times": times_no_cache,
        "avg_ms": avg_no_cache * 1000,
        "std_ms": std_no_cache * 1000,
    }

    # =========================================================================
    # Benchmark WITH cache - First call (cache miss)
    # =========================================================================
    print("\n--- With encoder cache (first call = cache miss) ---")
    model.enable_encoder_cache(max_entries=16)

    times_first_call = []
    for i in range(num_iterations):
        model.clear_encoder_cache()  # Clear for fresh start
        t0 = time.perf_counter()
        result = model.transcribe(
            audio.copy(),
            variable_length=variable_length,
        )
        mx.eval(result)
        elapsed = time.perf_counter() - t0
        times_first_call.append(elapsed)
        print(f"  Iteration {i+1}: {elapsed*1000:.1f}ms (cache entries: {len(model._encoder_cache)})")

    avg_first = np.mean(times_first_call)
    std_first = np.std(times_first_call)
    print(f"  Average: {avg_first*1000:.1f}ms (±{std_first*1000:.1f}ms)")

    results["cache_miss"] = {
        "times": times_first_call,
        "avg_ms": avg_first * 1000,
        "std_ms": std_first * 1000,
    }

    # =========================================================================
    # Benchmark WITH cache - Second call (cache hit)
    # =========================================================================
    print("\n--- With encoder cache (second call = cache hit) ---")

    # Prime the cache with one call
    model.clear_encoder_cache()
    _ = model.transcribe(audio.copy(), variable_length=variable_length)

    times_cached = []
    for i in range(num_iterations):
        t0 = time.perf_counter()
        result = model.transcribe(
            audio.copy(),
            variable_length=variable_length,
        )
        mx.eval(result)
        elapsed = time.perf_counter() - t0
        times_cached.append(elapsed)

        stats = model.get_encoder_cache_stats()
        print(f"  Iteration {i+1}: {elapsed*1000:.1f}ms (hits: {stats['hits']}, rate: {stats['hit_rate']:.0%})")

    avg_cached = np.mean(times_cached)
    std_cached = np.std(times_cached)
    print(f"  Average: {avg_cached*1000:.1f}ms (±{std_cached*1000:.1f}ms)")

    results["cache_hit"] = {
        "times": times_cached,
        "avg_ms": avg_cached * 1000,
        "std_ms": std_cached * 1000,
    }

    # =========================================================================
    # Calculate speedup
    # =========================================================================
    speedup = avg_no_cache / avg_cached
    encoder_time_saved = avg_no_cache - avg_cached

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"No cache:    {avg_no_cache*1000:.1f}ms")
    print(f"Cache hit:   {avg_cached*1000:.1f}ms")
    print(f"Speedup:     {speedup:.2f}x")
    print(f"Time saved:  {encoder_time_saved*1000:.1f}ms ({encoder_time_saved/avg_no_cache*100:.1f}%)")

    # Cache stats
    stats = model.get_encoder_cache_stats()
    print("\nCache stats:")
    print(f"  Entries: {stats['entries']}")
    print(f"  Memory: {stats['memory_mb']:.2f} MB")
    print(f"  Hit rate: {stats['hit_rate']:.1%}")

    results["speedup"] = speedup
    results["time_saved_ms"] = encoder_time_saved * 1000
    results["time_saved_pct"] = encoder_time_saved / avg_no_cache * 100

    # =========================================================================
    # Language detection + transcription use case
    # =========================================================================
    print("\n--- Language detection + transcription (real-world use case) ---")
    model.clear_encoder_cache()

    # Without cache: two separate transcriptions
    t0 = time.perf_counter()
    _ = model.transcribe(audio.copy(), language=None, variable_length=variable_length)  # Auto-detect
    _ = model.transcribe(audio.copy(), language="en", variable_length=variable_length)  # Transcribe
    time_two_calls_no_cache = time.perf_counter() - t0

    # With cache: second call uses cached encoder
    model.clear_encoder_cache()
    t0 = time.perf_counter()
    _ = model.transcribe(audio.copy(), language=None, variable_length=variable_length)  # Auto-detect (cache miss)
    _ = model.transcribe(audio.copy(), language="en", variable_length=variable_length)  # Transcribe (cache hit)
    time_two_calls_with_cache = time.perf_counter() - t0

    print(f"  Two calls without cache: {time_two_calls_no_cache*1000:.1f}ms")
    print(f"  Two calls with cache:    {time_two_calls_with_cache*1000:.1f}ms")
    print(f"  Speedup:                 {time_two_calls_no_cache/time_two_calls_with_cache:.2f}x")

    results["two_call_speedup"] = time_two_calls_no_cache / time_two_calls_with_cache

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark encoder caching for WhisperMLX")
    parser.add_argument("--audio", type=str, help="Path to audio file (uses synthetic if not provided)")
    parser.add_argument("--model", type=str, default="large-v3", help="Whisper model name")
    parser.add_argument("--iterations", type=int, default=3, help="Number of timing iterations")
    parser.add_argument("--variable-length", action="store_true", help="Use variable-length mode")
    args = parser.parse_args()

    # Load or generate audio
    if args.audio:
        audio = load_audio_file(args.audio)
    else:
        print("Using synthetic test audio (6.37 seconds)")
        audio = generate_test_audio(duration=6.37)

    # Run benchmark
    results = benchmark_encoder_cache(
        audio=audio,
        model_name=args.model,
        num_iterations=args.iterations,
        variable_length=args.variable_length,
    )

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Single-call speedup: {results['speedup']:.2f}x")
    print(f"Two-call speedup: {results['two_call_speedup']:.2f}x")
    print(f"Encoder time saved: {results['time_saved_pct']:.1f}%")

    if results['speedup'] >= 1.3:
        print("\n✓ OPT-W4 encoder caching provides meaningful speedup")
    else:
        print("\n⚠ Speedup lower than expected - may need investigation")


if __name__ == "__main__":
    main()
