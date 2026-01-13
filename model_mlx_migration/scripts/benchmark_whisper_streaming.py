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
Whisper Streaming Latency Benchmark
===================================

Measures key streaming latency metrics:
- Time-to-first-output (TTFO): Time from audio start to first transcription
- Per-chunk latency: Processing time for each audio chunk
- End-to-end latency: Total time including all pipeline stages
- Latency distribution: P50, P95, P99 percentiles

Usage:
    python scripts/benchmark_whisper_streaming.py
    python scripts/benchmark_whisper_streaming.py --model large-v3 --audio path/to/audio.wav
"""

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class LatencyMetrics:
    """Streaming latency metrics."""
    # Time-to-first-output (TTFO)
    ttfo_ms: float = 0.0

    # Per-chunk latencies
    chunk_latencies_ms: List[float] = field(default_factory=list)

    # Total processing time
    total_processing_ms: float = 0.0

    # Audio duration
    audio_duration_s: float = 0.0

    # Real-time factor
    rtf: float = 0.0

    @property
    def p50_latency_ms(self) -> float:
        if not self.chunk_latencies_ms:
            return 0.0
        return float(np.percentile(self.chunk_latencies_ms, 50))

    @property
    def p95_latency_ms(self) -> float:
        if not self.chunk_latencies_ms:
            return 0.0
        return float(np.percentile(self.chunk_latencies_ms, 95))

    @property
    def p99_latency_ms(self) -> float:
        if not self.chunk_latencies_ms:
            return 0.0
        return float(np.percentile(self.chunk_latencies_ms, 99))

    @property
    def avg_latency_ms(self) -> float:
        if not self.chunk_latencies_ms:
            return 0.0
        return float(np.mean(self.chunk_latencies_ms))

    @property
    def max_latency_ms(self) -> float:
        if not self.chunk_latencies_ms:
            return 0.0
        return float(np.max(self.chunk_latencies_ms))

    def __str__(self) -> str:
        return f"""Streaming Latency Metrics:
  Time-to-First-Output: {self.ttfo_ms:.1f} ms
  Chunk Latency (avg):  {self.avg_latency_ms:.1f} ms
  Chunk Latency (P50):  {self.p50_latency_ms:.1f} ms
  Chunk Latency (P95):  {self.p95_latency_ms:.1f} ms
  Chunk Latency (P99):  {self.p99_latency_ms:.1f} ms
  Chunk Latency (max):  {self.max_latency_ms:.1f} ms
  Total Processing:     {self.total_processing_ms:.1f} ms
  Audio Duration:       {self.audio_duration_s:.2f} s
  RTF (Real-Time):      {self.rtf:.3f}x"""


def load_test_audio(audio_path: Optional[str] = None, duration: float = 10.0) -> np.ndarray:
    """Load or generate test audio.

    Args:
        audio_path: Path to audio file (optional)
        duration: Duration for synthetic audio if no path provided

    Returns:
        Audio as float32 numpy array at 16kHz
    """
    import mlx.core as mx

    if audio_path:
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Use WhisperMLX's audio loading
        from tools.whisper_mlx.audio import load_audio
        audio = load_audio(str(path))
        return np.array(audio) if isinstance(audio, mx.array) else audio
    else:
        # Use a test audio file if available
        test_audio_path = Path(__file__).parent.parent / "data" / "test" / "jfk.wav"
        if test_audio_path.exists():
            from tools.whisper_mlx.audio import load_audio
            audio = load_audio(str(test_audio_path))
            return np.array(audio) if isinstance(audio, mx.array) else audio

        # Generate synthetic audio (simple sine wave for testing)
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        # 440 Hz tone with some variation
        audio = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        return audio


def benchmark_streaming_latency(
    model_name: str = "large-v3",
    audio: Optional[np.ndarray] = None,
    chunk_duration: float = 1.0,
    warmup_runs: int = 1,
) -> LatencyMetrics:
    """
    Benchmark streaming transcription latency.

    Args:
        model_name: Whisper model name
        audio: Audio array to transcribe (uses test audio if None)
        chunk_duration: Chunk size in seconds
        warmup_runs: Number of warmup runs before measuring

    Returns:
        LatencyMetrics with all measured values
    """
    from tools.whisper_mlx import WhisperMLX
    from tools.whisper_mlx.streaming import (
        SyncStreamingWhisper,
        StreamingConfig,
    )

    # Load model
    print(f"Loading model: {model_name}")
    model = WhisperMLX.from_pretrained(model_name)

    # Load audio
    if audio is None:
        audio = load_test_audio()

    audio_duration = len(audio) / 16000
    print(f"Audio duration: {audio_duration:.2f}s ({len(audio)} samples)")

    # Configure streaming
    config = StreamingConfig(
        min_chunk_duration=chunk_duration,
        max_chunk_duration=chunk_duration * 2,
        use_local_agreement=True,
        agreement_n=2,
    )

    # Warmup runs
    for i in range(warmup_runs):
        print(f"Warmup run {i+1}/{warmup_runs}")
        streaming = SyncStreamingWhisper(model, config)
        short_audio = audio[:int(16000 * 2)]  # 2 seconds
        chunks = [short_audio[j:j+int(16000*chunk_duration)]
                  for j in range(0, len(short_audio), int(16000*chunk_duration))]
        for chunk in chunks:
            streaming.process_audio(chunk)
        streaming.finalize()

    # Benchmark run
    print("\nBenchmarking...")
    metrics = LatencyMetrics()
    metrics.audio_duration_s = audio_duration

    streaming = SyncStreamingWhisper(model, config)

    # Split audio into chunks
    chunk_samples = int(16000 * chunk_duration)
    chunks = []
    for i in range(0, len(audio), chunk_samples):
        chunks.append(audio[i:i+chunk_samples])

    print(f"Processing {len(chunks)} chunks of {chunk_duration}s each")

    first_output_received = False
    total_start = time.perf_counter()
    audio_start = total_start  # When audio "starts playing"

    for i, chunk in enumerate(chunks):
        chunk_start = time.perf_counter()

        # Process chunk
        results = streaming.process_audio(chunk)

        chunk_end = time.perf_counter()
        chunk_latency_ms = (chunk_end - chunk_start) * 1000
        metrics.chunk_latencies_ms.append(chunk_latency_ms)

        # Track time-to-first-output
        if not first_output_received and results:
            for result in results:
                if result.text.strip():
                    metrics.ttfo_ms = (chunk_end - audio_start) * 1000
                    first_output_received = True
                    print(f"  First output at chunk {i+1}: '{result.text[:50]}...' "
                          f"(TTFO: {metrics.ttfo_ms:.1f}ms)")
                    break

        if (i + 1) % 5 == 0:
            print(f"  Processed {i+1}/{len(chunks)} chunks "
                  f"(avg latency: {np.mean(metrics.chunk_latencies_ms):.1f}ms)")

    # Finalize
    final_start = time.perf_counter()
    streaming.finalize()
    final_latency_ms = (time.perf_counter() - final_start) * 1000
    metrics.chunk_latencies_ms.append(final_latency_ms)

    total_end = time.perf_counter()
    metrics.total_processing_ms = (total_end - total_start) * 1000
    metrics.rtf = (metrics.total_processing_ms / 1000) / audio_duration

    # If no output was ever received
    if not first_output_received:
        metrics.ttfo_ms = metrics.total_processing_ms

    return metrics


def benchmark_batch_latency(
    model_name: str = "large-v3",
    audio: Optional[np.ndarray] = None,
) -> float:
    """
    Benchmark batch (non-streaming) transcription for comparison.

    Returns:
        Total latency in ms
    """
    from tools.whisper_mlx import WhisperMLX
    import mlx.core as mx

    print("\nBenchmarking batch transcription for comparison...")

    model = WhisperMLX.from_pretrained(model_name)

    if audio is None:
        audio = load_test_audio()

    audio_mx = mx.array(audio)

    start = time.perf_counter()
    result = model.transcribe(audio_mx)
    end = time.perf_counter()

    latency_ms = (end - start) * 1000
    audio_duration = len(audio) / 16000
    rtf = (latency_ms / 1000) / audio_duration

    print(f"Batch transcription: {latency_ms:.1f}ms (RTF: {rtf:.3f}x)")
    print(f"Result: {result['text'][:100]}...")

    return latency_ms


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Whisper streaming latency"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="large-v3-turbo",
        help="Model name (default: large-v3-turbo for P3 compliance)"
    )
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Path to audio file (uses test audio if not specified)"
    )
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=1.0,
        help="Chunk duration in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warmup runs (default: 1)"
    )
    parser.add_argument(
        "--skip-batch",
        action="store_true",
        help="Skip batch comparison"
    )

    args = parser.parse_args()

    # Load audio once
    audio = load_test_audio(args.audio)

    print("=" * 60)
    print("Whisper Streaming Latency Benchmark")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Chunk duration: {args.chunk_duration}s")
    print(f"Audio duration: {len(audio)/16000:.2f}s")
    print("=" * 60)

    # Benchmark streaming
    metrics = benchmark_streaming_latency(
        model_name=args.model,
        audio=audio,
        chunk_duration=args.chunk_duration,
        warmup_runs=args.warmup,
    )

    print("\n" + "=" * 60)
    print("STREAMING RESULTS")
    print("=" * 60)
    print(metrics)

    # Benchmark batch for comparison
    if not args.skip_batch:
        batch_latency = benchmark_batch_latency(
            model_name=args.model,
            audio=audio,
        )

        print("\n" + "=" * 60)
        print("COMPARISON: Streaming vs Batch")
        print("=" * 60)
        print(f"Streaming TTFO:       {metrics.ttfo_ms:.1f} ms")
        print(f"Batch total latency:  {batch_latency:.1f} ms")
        print(f"TTFO advantage:       {batch_latency - metrics.ttfo_ms:.1f} ms faster first output")

    # Gate check: P3 requires <500ms latency
    print("\n" + "=" * 60)
    print("P3 GATE CHECK: Latency < 500ms")
    print("=" * 60)
    gate_passed = metrics.ttfo_ms < 500
    print(f"TTFO: {metrics.ttfo_ms:.1f}ms {'PASS' if gate_passed else 'FAIL'}")
    gate_passed = metrics.p95_latency_ms < 500
    print(f"P95 Chunk Latency: {metrics.p95_latency_ms:.1f}ms {'PASS' if gate_passed else 'FAIL'}")

    return metrics


if __name__ == "__main__":
    main()
