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
CTC Streaming Evaluation - Test first partial latency with trained CTC head.

This script integrates CTCStreamingWhisper with the streaming eval framework
to verify <200ms first partial latency for Gate 1.

Usage:
    python scripts/test_ctc_streaming_eval.py
    python scripts/test_ctc_streaming_eval.py --samples 5
    python scripts/test_ctc_streaming_eval.py --checkpoint checkpoints/ctc_head_large_v3/best.npz
"""

import argparse
import asyncio
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Suppress warnings before imports
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def load_ctc_head(checkpoint_path: str, d_model: int = 1280):
    """Load trained CTC head from checkpoint."""
    import mlx.core as mx
    from tools.whisper_mlx.ctc_head import CTCDraftHead

    # Load flat params
    flat_params = dict(mx.load(checkpoint_path))

    # Unflatten
    def unflatten_params(flat_params):
        result = {}
        for key, value in flat_params.items():
            parts = key.split(".")
            current = result
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        return result

    nested_params = unflatten_params(flat_params)

    # Infer vocab size from proj.weight shape
    vocab_size = nested_params["proj"]["weight"].shape[0]

    # Create and load head
    ctc_head = CTCDraftHead(d_model=d_model, vocab_size=vocab_size)
    ctc_head.update(nested_params)

    return ctc_head


async def evaluate_ctc_streaming_latency(
    model,
    ctc_head,
    audio: np.ndarray,
    reference: str,
    sample_id: str,
    target_latency_ms: int = 200,
) -> dict:
    """
    Evaluate CTC streaming latency for a single sample.

    Returns dict with:
        - first_partial_latency_ms: Time to first CTC output
        - ctc_text: CTC transcription
        - reference: Ground truth
        - passed: Whether first partial < target
    """
    from tools.whisper_mlx.streaming import (
        CTCStreamingWhisper,
        CTCStreamingConfig,
    )

    # Configure for ultra-low latency
    config = CTCStreamingConfig.ultra_low_latency()

    # Create CTC streamer
    streamer = CTCStreamingWhisper(model, ctc_head, config)

    # Setup timing
    sample_rate = 16000
    chunk_duration_ms = 100  # 100ms chunks
    chunk_samples = int(chunk_duration_ms * sample_rate / 1000)
    num_chunks = (len(audio) + chunk_samples - 1) // chunk_samples

    # Timing
    start_time = time.perf_counter()
    first_partial_time = None
    first_ctc_text = ""
    final_ctc_text = ""

    # Create async audio generator
    async def audio_generator():
        for i in range(num_chunks):
            chunk_start = i * chunk_samples
            chunk_end = min((i + 1) * chunk_samples, len(audio))
            chunk = audio[chunk_start:chunk_end]
            yield chunk

    # Run streaming transcription
    async for result in streamer.transcribe_stream(audio_generator()):
        now = time.perf_counter()
        wall_time_ms = (now - start_time) * 1000

        # Track first CTC output
        if first_partial_time is None and result.ctc_draft.strip():
            first_partial_time = wall_time_ms
            first_ctc_text = result.ctc_draft

        # Track latest CTC output
        if result.ctc_draft.strip():
            final_ctc_text = result.ctc_draft

    # Calculate results
    if first_partial_time is None:
        first_partial_time = float('inf')
        first_ctc_text = ""

    passed = first_partial_time < target_latency_ms

    return {
        "sample_id": sample_id,
        "first_partial_latency_ms": first_partial_time,
        "first_ctc_text": first_ctc_text,
        "final_ctc_text": final_ctc_text,
        "reference": reference,
        "audio_duration_ms": len(audio) / sample_rate * 1000,
        "passed": passed,
        "target_ms": target_latency_ms,
    }


def load_test_samples(
    data_dir: str = "data",
    max_samples: int = 10,
) -> List[Tuple[np.ndarray, str, str]]:
    """Load test samples from LibriSpeech."""
    from tools.whisper_mlx.audio import load_audio

    samples = []

    # Try dev-clean first (smaller, faster to iterate)
    libri_path = Path(data_dir) / "benchmarks" / "librispeech" / "LibriSpeech" / "dev-clean"
    if not libri_path.exists():
        libri_path = Path(data_dir) / "librispeech" / "dev-clean"
    if not libri_path.exists():
        libri_path = Path(data_dir) / "multilingual" / "english" / "dev-clean"

    if libri_path.exists():
        # Find audio files with transcripts
        for trans_file in sorted(libri_path.rglob("*.trans.txt")):
            # Parse transcript file
            with open(trans_file, "r") as f:
                for line in f:
                    parts = line.strip().split(" ", 1)
                    if len(parts) == 2:
                        file_id, text = parts
                        audio_file = trans_file.parent / f"{file_id}.flac"
                        if audio_file.exists():
                            audio = load_audio(str(audio_file))
                            samples.append((audio, text, file_id))
                            if len(samples) >= max_samples:
                                return samples

    if not samples:
        # Fallback to benchmarks
        benchmark_path = Path(data_dir) / "benchmarks" / "librispeech" / "LibriSpeech" / "test-clean"
        if benchmark_path.exists():
            for trans_file in sorted(benchmark_path.rglob("*.trans.txt")):
                with open(trans_file, "r") as f:
                    for line in f:
                        parts = line.strip().split(" ", 1)
                        if len(parts) == 2:
                            file_id, text = parts
                            audio_file = trans_file.parent / f"{file_id}.flac"
                            if audio_file.exists():
                                audio = load_audio(str(audio_file))
                                samples.append((audio, text, file_id))
                                if len(samples) >= max_samples:
                                    return samples

    return samples


async def main():
    parser = argparse.ArgumentParser(description="CTC Streaming Latency Evaluation")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/ctc_head_large_v3/best.npz",
                        help="CTC head checkpoint path")
    parser.add_argument("--model", type=str,
                        default="mlx-community/whisper-large-v3-mlx",
                        help="WhisperMLX model")
    parser.add_argument("--samples", type=int, default=10,
                        help="Number of samples to evaluate")
    parser.add_argument("--target-ms", type=int, default=200,
                        help="Target first partial latency (ms)")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Data directory")

    args = parser.parse_args()

    print("=" * 70)
    print("CTC Streaming First Partial Latency Evaluation")
    print("=" * 70)
    print(f"Target: <{args.target_ms}ms first partial (Gate 1 requirement)")
    print()

    # Load model
    print(f"1. Loading WhisperMLX ({args.model})...")
    from tools.whisper_mlx import WhisperMLX
    model = WhisperMLX.from_pretrained(args.model)
    d_model = model.config.n_audio_state
    print(f"   d_model={d_model}")

    # Load CTC head
    print(f"2. Loading CTC head ({args.checkpoint})...")
    ctc_head = load_ctc_head(args.checkpoint, d_model=d_model)
    print(f"   Loaded. vocab_size={ctc_head.vocab_size}")

    # Load test samples
    print(f"3. Loading test samples (max {args.samples})...")
    samples = load_test_samples(args.data_dir, args.samples)
    print(f"   Loaded {len(samples)} samples")

    if not samples:
        print("ERROR: No test samples found!")
        return

    print()
    print("=" * 70)
    print("CTC Streaming Evaluation Results")
    print("=" * 70)
    print(f"{'Sample':<30} | {'Latency':<10} | {'Status':<6} | CTC Output (first 50 chars)")
    print("-" * 70)

    # Evaluate each sample
    results = []
    for audio, reference, sample_id in samples:
        result = await evaluate_ctc_streaming_latency(
            model, ctc_head, audio, reference, sample_id, args.target_ms
        )
        results.append(result)

        status = "PASS" if result["passed"] else "FAIL"
        ctc_preview = result["first_ctc_text"][:50] if result["first_ctc_text"] else "(empty)"
        print(f"{sample_id:<30} | {result['first_partial_latency_ms']:>7.1f}ms | {status:<6} | {ctc_preview}")

    # Summary
    print("-" * 70)
    latencies = [r["first_partial_latency_ms"] for r in results if r["first_partial_latency_ms"] != float('inf')]
    passed = sum(1 for r in results if r["passed"])

    if latencies:
        median_latency = np.median(latencies)
        p95_latency = np.percentile(latencies, 95)
        min_latency = min(latencies)
        max_latency = max(latencies)
    else:
        median_latency = p95_latency = min_latency = max_latency = float('inf')

    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Samples:       {len(results)}")
    print(f"Passed:        {passed}/{len(results)} ({100*passed/len(results):.0f}%)")
    print(f"Median:        {median_latency:.1f}ms")
    print(f"P95:           {p95_latency:.1f}ms")
    print(f"Min:           {min_latency:.1f}ms")
    print(f"Max:           {max_latency:.1f}ms")
    print()

    gate1_status = "PASS" if median_latency < args.target_ms else "FAIL"
    print(f"Gate 1 First Partial (<{args.target_ms}ms): {gate1_status}")

    # Quality note
    print()
    print("=" * 70)
    print("CTC Quality Assessment")
    print("=" * 70)
    print("Sample CTC outputs vs references:")
    print()
    for r in results[:3]:
        print(f"Reference: {r['reference'][:80]}...")
        print(f"CTC:       {r['final_ctc_text'][:80] if r['final_ctc_text'] else '(empty)'}...")
        print()

    print("Note: CTC quality depends on training progress.")
    print("      Current training: check checkpoints/ctc_head_large_v3/training.log")


if __name__ == "__main__":
    asyncio.run(main())
