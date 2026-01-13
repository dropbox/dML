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
Benchmark C2: Batched Decoder Throughput

Compares sequential transcription vs batched transcription for
multiple audio files.

Expected improvement: 1.5-2x throughput
"""

import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import numpy as np

from tools.whisper_mlx import WhisperMLX


def generate_test_audio(duration_s: float = 5.0, sample_rate: int = 16000) -> np.ndarray:
    """Generate synthetic audio (white noise) for testing."""
    n_samples = int(duration_s * sample_rate)
    # Generate white noise
    audio = np.random.randn(n_samples).astype(np.float32) * 0.1
    return audio


def benchmark_sequential(model: WhisperMLX, audio_list: list, n_runs: int = 3) -> dict:
    """Benchmark sequential transcription."""
    times = []

    for run in range(n_runs):
        start = time.perf_counter()
        results = []
        for audio in audio_list:
            result = model.transcribe(audio, language="en")
            results.append(result)
        mx.eval()  # Ensure all computation is done
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Sequential run {run + 1}: {elapsed:.3f}s")

    return {
        "times": times,
        "avg_time": np.mean(times),
        "std_time": np.std(times),
        "results": results,
    }


def benchmark_batched(model: WhisperMLX, audio_list: list, n_runs: int = 3) -> dict:
    """Benchmark batched transcription."""
    times = []

    for run in range(n_runs):
        start = time.perf_counter()
        results = model.transcribe_batch(audio_list, language="en", verbose=False)
        mx.eval()  # Ensure all computation is done
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Batched run {run + 1}: {elapsed:.3f}s")

    return {
        "times": times,
        "avg_time": np.mean(times),
        "std_time": np.std(times),
        "results": results,
    }


def main():
    print("=" * 60)
    print("C2: Batched Decoder Throughput Benchmark")
    print("=" * 60)

    # Load model
    print("\nLoading WhisperMLX model (base)...")
    model = WhisperMLX.from_pretrained("base")
    print(f"Model loaded: {model.config.n_text_layer} decoder layers, {model.config.n_text_state} dim")

    # Warm up
    print("\nWarming up...")
    dummy_audio = generate_test_audio(5.0)
    _ = model.transcribe(dummy_audio, language="en")
    mx.eval()

    # Test different batch sizes
    batch_sizes = [2, 4, 8]
    audio_duration = 5.0  # seconds

    print("\n" + "=" * 60)
    print(f"Test audio: {audio_duration}s white noise")
    print("=" * 60)

    for batch_size in batch_sizes:
        print(f"\n--- Batch size: {batch_size} ---")

        # Generate test audio
        audio_list = [generate_test_audio(audio_duration) for _ in range(batch_size)]
        print(f"Generated {batch_size} audio clips of {audio_duration}s each")

        # Sequential benchmark
        print("\nSequential transcription:")
        seq_result = benchmark_sequential(model, audio_list, n_runs=3)

        # Batched benchmark
        print("\nBatched transcription:")
        batch_result = benchmark_batched(model, audio_list, n_runs=3)

        # Calculate speedup
        speedup = seq_result["avg_time"] / batch_result["avg_time"]
        throughput_seq = batch_size / seq_result["avg_time"]
        throughput_batch = batch_size / batch_result["avg_time"]

        print(f"\nResults (batch_size={batch_size}):")
        print(f"  Sequential: {seq_result['avg_time']:.3f}s +/- {seq_result['std_time']:.3f}s")
        print(f"  Batched:    {batch_result['avg_time']:.3f}s +/- {batch_result['std_time']:.3f}s")
        print(f"  Speedup:    {speedup:.2f}x")
        print(f"  Throughput: {throughput_seq:.2f} -> {throughput_batch:.2f} audio/s")

    # Try with real LibriSpeech audio if available
    librispeech_path = Path("/Users/ayates/model_mlx_migration/data/benchmarks/librispeech/LibriSpeech/test-clean")
    if librispeech_path.exists():
        print("\n" + "=" * 60)
        print("Testing with real LibriSpeech audio")
        print("=" * 60)

        # Find some audio files
        flac_files = list(librispeech_path.glob("*/*/*.flac"))[:8]
        if len(flac_files) >= 4:
            audio_paths = [str(f) for f in flac_files[:4]]
            print(f"Using {len(audio_paths)} LibriSpeech files")

            # Sequential
            print("\nSequential transcription:")
            seq_result = benchmark_sequential(model, audio_paths, n_runs=3)

            # Batched
            print("\nBatched transcription:")
            batch_result = benchmark_batched(model, audio_paths, n_runs=3)

            speedup = seq_result["avg_time"] / batch_result["avg_time"]
            print("\nResults (LibriSpeech, batch_size=4):")
            print(f"  Sequential: {seq_result['avg_time']:.3f}s")
            print(f"  Batched:    {batch_result['avg_time']:.3f}s")
            print(f"  Speedup:    {speedup:.2f}x")

            # Compare transcriptions
            print("\nTranscription comparison:")
            for i, (seq_res, batch_res) in enumerate(zip(seq_result["results"], batch_result["results"])):
                seq_text = seq_res["text"][:50] + "..." if len(seq_res["text"]) > 50 else seq_res["text"]
                batch_text = batch_res["text"][:50] + "..." if len(batch_res["text"]) > 50 else batch_res["text"]
                match = "MATCH" if seq_res["text"] == batch_res["text"] else "DIFFER"
                print(f"  [{i}] {match}")
                print(f"      Seq:   {seq_text}")
                print(f"      Batch: {batch_text}")

    print("\n" + "=" * 60)
    print("Benchmark complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
