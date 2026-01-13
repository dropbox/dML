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
Benchmark J10: Multi-User Batch Streaming Server

Compares throughput of:
1. Sequential single-user processing (baseline)
2. BatchingStreamServer with multiple concurrent users

Expected: 2-4x throughput improvement with 4-8 concurrent users
"""

import asyncio
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.whisper_mlx import WhisperMLX
from tools.whisper_mlx.streaming import (
    BatchingStreamServer,
    BatchServerConfig,
    StreamingConfig,
)


def generate_test_audio(duration_seconds: float, sample_rate: int = 16000) -> np.ndarray:
    """Generate synthetic audio for testing."""
    # Generate a simple speech-like pattern (sine wave with harmonics)
    t = np.linspace(0, duration_seconds, int(duration_seconds * sample_rate))
    audio = (
        0.3 * np.sin(2 * np.pi * 150 * t) +
        0.2 * np.sin(2 * np.pi * 300 * t) +
        0.1 * np.sin(2 * np.pi * 450 * t) +
        0.05 * np.random.randn(len(t))  # Add noise
    ).astype(np.float32)
    return audio


async def benchmark_sequential(
    model: WhisperMLX,
    num_users: int,
    audio_duration: float,
    num_iterations: int = 3,
) -> dict:
    """Benchmark sequential processing (one user at a time)."""
    StreamingConfig(
        use_vad=False,
        min_chunk_duration=0.5,
        max_chunk_duration=audio_duration,
        emit_partials=False,
        use_local_agreement=False,
    )

    # Generate audio for each user
    audios = [generate_test_audio(audio_duration) for _ in range(num_users)]

    times = []
    for iteration in range(num_iterations):
        start_time = time.perf_counter()

        for i, audio in enumerate(audios):
            # Process each user sequentially
            model.transcribe(audio, language="en")

        elapsed = time.perf_counter() - start_time
        times.append(elapsed)
        print(f"  Sequential iteration {iteration + 1}: {elapsed:.3f}s")

    avg_time = sum(times) / len(times)
    total_audio = num_users * audio_duration

    return {
        "method": "sequential",
        "num_users": num_users,
        "audio_per_user": audio_duration,
        "total_audio": total_audio,
        "avg_time": avg_time,
        "throughput_rtf": total_audio / avg_time,  # Real-time factor
        "times": times,
    }


async def benchmark_batch_server(
    model: WhisperMLX,
    num_users: int,
    audio_duration: float,
    num_iterations: int = 3,
) -> dict:
    """Benchmark BatchingStreamServer with concurrent users."""
    config = BatchServerConfig(
        max_batch_size=num_users,
        batch_timeout_ms=50.0,
        min_audio_duration=0.5,
        language="en",
    )
    server = BatchingStreamServer(model, config)

    # Generate audio for each user
    audios = [generate_test_audio(audio_duration) for _ in range(num_users)]

    times = []
    for iteration in range(num_iterations):
        start_time = time.perf_counter()

        # Create sessions and add audio concurrently
        for i in range(num_users):
            await server.create_session(f"user_{iteration}_{i}", use_local_agreement=False)
            await server.add_audio(f"user_{iteration}_{i}", audios[i])

        # Process batch
        session_ids = [f"user_{iteration}_{i}" for i in range(num_users)]
        await server._process_batch(session_ids)

        # Collect results
        for sid in session_ids:
            await server._get_results(sid)
            await server.finalize_session(sid)

        elapsed = time.perf_counter() - start_time
        times.append(elapsed)
        print(f"  Batch server iteration {iteration + 1}: {elapsed:.3f}s (batch_size={num_users})")

    avg_time = sum(times) / len(times)
    total_audio = num_users * audio_duration

    return {
        "method": "batch_server",
        "num_users": num_users,
        "audio_per_user": audio_duration,
        "total_audio": total_audio,
        "avg_time": avg_time,
        "throughput_rtf": total_audio / avg_time,
        "times": times,
    }


async def benchmark_transcribe_batch(
    model: WhisperMLX,
    num_users: int,
    audio_duration: float,
    num_iterations: int = 3,
) -> dict:
    """Benchmark direct transcribe_batch (C2 baseline for comparison)."""
    # Generate audio for each user
    audios = [generate_test_audio(audio_duration) for _ in range(num_users)]

    times = []
    for iteration in range(num_iterations):
        start_time = time.perf_counter()

        model.transcribe_batch(audios, language="en")

        elapsed = time.perf_counter() - start_time
        times.append(elapsed)
        print(f"  transcribe_batch iteration {iteration + 1}: {elapsed:.3f}s")

    avg_time = sum(times) / len(times)
    total_audio = num_users * audio_duration

    return {
        "method": "transcribe_batch",
        "num_users": num_users,
        "audio_per_user": audio_duration,
        "total_audio": total_audio,
        "avg_time": avg_time,
        "throughput_rtf": total_audio / avg_time,
        "times": times,
    }


async def main():
    print("=" * 60)
    print("J10: Multi-User Batch Streaming Server Benchmark")
    print("=" * 60)

    # Load model
    print("\nLoading model (base for fast benchmarks)...")
    model = WhisperMLX.from_pretrained("base")
    print(f"Model loaded: {model.config.n_mels} mels, {model.config.n_vocab} vocab")

    # Warmup
    print("\nWarming up model...")
    warmup_audio = generate_test_audio(2.0)
    _ = model.transcribe(warmup_audio, language="en")
    print("Warmup complete")

    # Test configurations
    user_counts = [1, 2, 4, 8]
    audio_duration = 5.0  # 5 seconds per user
    num_iterations = 3

    results = []

    for num_users in user_counts:
        print(f"\n{'=' * 60}")
        print(f"Testing with {num_users} concurrent users ({audio_duration}s audio each)")
        print("=" * 60)

        # Sequential baseline
        print("\n1. Sequential processing (baseline):")
        seq_result = await benchmark_sequential(
            model, num_users, audio_duration, num_iterations
        )
        results.append(seq_result)

        # Direct transcribe_batch (C2)
        print("\n2. transcribe_batch (C2 baseline):")
        batch_result = await benchmark_transcribe_batch(
            model, num_users, audio_duration, num_iterations
        )
        results.append(batch_result)

        # BatchingStreamServer (J10)
        print("\n3. BatchingStreamServer (J10):")
        server_result = await benchmark_batch_server(
            model, num_users, audio_duration, num_iterations
        )
        results.append(server_result)

        # Calculate speedups
        seq_time = seq_result["avg_time"]
        batch_time = batch_result["avg_time"]
        server_time = server_result["avg_time"]

        print(f"\n--- Results for {num_users} users ---")
        print(f"Sequential:        {seq_time:.3f}s (1.00x)")
        print(f"transcribe_batch:  {batch_time:.3f}s ({seq_time/batch_time:.2f}x speedup)")
        print(f"BatchingServer:    {server_time:.3f}s ({seq_time/server_time:.2f}x speedup)")
        print(f"Server overhead vs C2: {(server_time/batch_time - 1)*100:.1f}%")

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Users':>6} | {'Sequential':>12} | {'C2 Batch':>12} | {'J10 Server':>12} | {'Speedup':>8}")
    print("-" * 60)

    for num_users in user_counts:
        seq = next(r for r in results if r["method"] == "sequential" and r["num_users"] == num_users)
        batch = next(r for r in results if r["method"] == "transcribe_batch" and r["num_users"] == num_users)
        server = next(r for r in results if r["method"] == "batch_server" and r["num_users"] == num_users)

        speedup = seq["avg_time"] / server["avg_time"]
        print(f"{num_users:>6} | {seq['avg_time']:>10.3f}s | {batch['avg_time']:>10.3f}s | {server['avg_time']:>10.3f}s | {speedup:>6.2f}x")

    print("\nConclusion:")
    eight_user_seq = next(r for r in results if r["method"] == "sequential" and r["num_users"] == 8)
    eight_user_server = next(r for r in results if r["method"] == "batch_server" and r["num_users"] == 8)
    final_speedup = eight_user_seq["avg_time"] / eight_user_server["avg_time"]
    print(f"J10 BatchingStreamServer achieves {final_speedup:.2f}x throughput with 8 concurrent users")


if __name__ == "__main__":
    asyncio.run(main())
