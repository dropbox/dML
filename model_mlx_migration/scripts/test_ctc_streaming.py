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
Test script for CTC-accelerated streaming.

This script tests the CTCStreamingWhisper class with an untrained CTC head
to verify the pipeline works. The CTC output will be garbage (random tokens)
until the head is trained, but this validates:
1. Encoder -> CTC head inference works
2. Latency is in expected range (~60ms)
3. Pipeline integration is correct

Usage:
    python scripts/test_ctc_streaming.py [--audio path/to/audio.wav]

Expected output (untrained CTC):
    [CTC ~60ms] <garbage tokens>
    [DECODER] <correct transcription>
"""

import argparse
import asyncio
import time
from pathlib import Path

import numpy as np


async def test_ctc_streaming(audio_path: str = None, model_name: str = "small"):
    """Test CTC streaming pipeline."""
    print("=" * 60)
    print("CTC Streaming Pipeline Test")
    print("=" * 60)

    # Import modules
    print("\n1. Loading modules...")
    from tools.whisper_mlx import WhisperMLX
    from tools.whisper_mlx.streaming import (
        CTCStreamingWhisper,
        CTCStreamingConfig,
        HAS_CTC,
    )
    from tools.whisper_mlx.ctc_head import create_ctc_draft_head
    from tools.whisper_mlx.audio import load_audio

    if not HAS_CTC:
        print("ERROR: CTC module not available")
        return

    print("   CTC module available: YES")

    # Load model
    print(f"\n2. Loading WhisperMLX model ({model_name})...")
    start = time.perf_counter()
    model = WhisperMLX.from_pretrained(
        f"mlx-community/whisper-{model_name}-mlx",
        warmup=True,
    )
    print(f"   Model loaded in {time.perf_counter() - start:.2f}s")
    print(f"   Encoder d_model: {model.config.n_audio_state}")

    # Create CTC head (untrained - random weights)
    print("\n3. Creating CTC head (untrained)...")
    d_model = model.config.n_audio_state
    ctc_head = create_ctc_draft_head(model_name)
    print(f"   CTC head created: d_model={ctc_head.d_model}, vocab_size={ctc_head.vocab_size}")
    print("   WARNING: CTC head is UNTRAINED - output will be garbage tokens")

    # Create streaming config
    print("\n4. Creating streaming config...")
    config = CTCStreamingConfig(
        min_ctc_duration=0.5,  # 500ms minimum for CTC
        ctc_interval=0.3,      # CTC every 300ms
        decoder_interval=3.0,  # Decoder every 3s (slow path)
        use_vad=False,         # Disable VAD for testing
    )
    print(f"   min_ctc_duration: {config.min_ctc_duration}s")
    print(f"   ctc_interval: {config.ctc_interval}s")
    print(f"   decoder_interval: {config.decoder_interval}s")

    # Create streamer
    print("\n5. Creating CTCStreamingWhisper...")
    streamer = CTCStreamingWhisper(model, ctc_head, config)
    print("   Streamer created successfully")

    # Load or generate test audio
    if audio_path and Path(audio_path).exists():
        print(f"\n6. Loading audio: {audio_path}")
        audio = load_audio(audio_path)
        print(f"   Duration: {len(audio) / 16000:.2f}s")
    else:
        print("\n6. Generating test audio (2s sine wave at 440Hz)...")
        duration = 2.0
        t = np.linspace(0, duration, int(16000 * duration))
        audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
        print(f"   Generated {duration}s test tone")

    # Simulate streaming by chunking audio
    chunk_duration = 0.1  # 100ms chunks
    chunk_size = int(16000 * chunk_duration)
    n_chunks = len(audio) // chunk_size

    print(f"\n7. Streaming test ({n_chunks} chunks of {chunk_duration*1000:.0f}ms)...")
    print("-" * 60)

    async def audio_generator():
        for i in range(n_chunks):
            chunk = audio[i * chunk_size:(i + 1) * chunk_size]
            yield chunk
            # Simulate real-time streaming
            await asyncio.sleep(0.01)  # Small delay between chunks

    ctc_latencies = []
    results_count = 0

    async for result in streamer.transcribe_stream(audio_generator()):
        results_count += 1

        if result.ctc_is_new:
            ctc_latencies.append(result.ctc_latency_ms)
            # Truncate CTC output for display (will be garbage with untrained head)
            ctc_text = result.ctc_draft[:50] + "..." if len(result.ctc_draft) > 50 else result.ctc_draft
            print(f"[CTC ~{result.ctc_latency_ms:.0f}ms] {repr(ctc_text)}")

        if result.decoder_is_new:
            print(f"[DECODER] confirmed: {repr(result.confirmed_text)}")
            print(f"          speculative: {repr(result.speculative_text)}")

    print("-" * 60)

    # Summary
    print("\n8. Summary:")
    print(f"   Total results: {results_count}")
    print(f"   CTC outputs: {len(ctc_latencies)}")

    if ctc_latencies:
        avg_latency = sum(ctc_latencies) / len(ctc_latencies)
        min_latency = min(ctc_latencies)
        max_latency = max(ctc_latencies)
        print(f"   CTC latency (avg): {avg_latency:.1f}ms")
        print(f"   CTC latency (min): {min_latency:.1f}ms")
        print(f"   CTC latency (max): {max_latency:.1f}ms")

        # Evaluate against targets
        print("\n9. Target evaluation:")
        if avg_latency < 100:
            print(f"   ✅ CTC latency {avg_latency:.0f}ms < 100ms target")
        else:
            print(f"   ❌ CTC latency {avg_latency:.0f}ms > 100ms target")

        # Total first partial = audio accumulation + CTC latency
        # With 500ms min_ctc_duration + 60ms CTC = 560ms
        total_first_partial = config.min_ctc_duration * 1000 + avg_latency
        print(f"   First partial estimate: {total_first_partial:.0f}ms")
        print(f"   (audio_accumulation={config.min_ctc_duration*1000:.0f}ms + ctc={avg_latency:.0f}ms)")

        if total_first_partial < 200:
            print(f"   ✅ First partial {total_first_partial:.0f}ms < 200ms target!")
        elif total_first_partial < 400:
            print(f"   ⚠️  First partial {total_first_partial:.0f}ms < 400ms (good, not <200ms)")
        else:
            print(f"   ❌ First partial {total_first_partial:.0f}ms > 400ms (needs optimization)")

    print("\n" + "=" * 60)
    print("CTC streaming pipeline test complete!")
    print("NOTE: CTC output is garbage because head is untrained.")
    print("Train CTC head with train_ctc.py for meaningful output.")
    print("=" * 60)


async def benchmark_ctc_latency(model_name: str = "small"):
    """Benchmark CTC inference latency in isolation."""
    print("=" * 60)
    print("CTC Latency Benchmark")
    print("=" * 60)

    import mlx.core as mx
    from tools.whisper_mlx import WhisperMLX
    from tools.whisper_mlx.ctc_head import create_ctc_draft_head
    from tools.whisper_mlx.audio import log_mel_spectrogram

    # Load model
    print(f"\n1. Loading model ({model_name})...")
    model = WhisperMLX.from_pretrained(
        f"mlx-community/whisper-{model_name}-mlx",
        warmup=True,
    )

    # Create CTC head
    ctc_head = create_ctc_draft_head(model_name)

    # Test different audio durations
    durations = [0.3, 0.5, 1.0, 2.0, 3.0]
    n_iterations = 5

    print("\n2. Benchmarking CTC latency for different audio durations:")
    print("-" * 60)
    print(f"{'Duration':>10} | {'Mel':>8} | {'Encoder':>10} | {'CTC':>8} | {'Total':>8}")
    print("-" * 60)

    for duration in durations:
        # Generate test audio
        audio = np.random.randn(int(16000 * duration)).astype(np.float32) * 0.1

        mel_times = []
        encoder_times = []
        ctc_times = []

        for _ in range(n_iterations):
            # Mel spectrogram (use model's n_mels config)
            start = time.perf_counter()
            n_mels = model.config.n_mels
            mel = log_mel_spectrogram(audio, n_mels=n_mels)
            mel = mx.expand_dims(mx.array(mel), axis=0)
            mx.eval(mel)
            mel_time = (time.perf_counter() - start) * 1000
            mel_times.append(mel_time)

            # Encoder
            start = time.perf_counter()
            encoder_out = model.encoder(mel, variable_length=True)
            mx.eval(encoder_out)
            encoder_time = (time.perf_counter() - start) * 1000
            encoder_times.append(encoder_time)

            # CTC head
            start = time.perf_counter()
            ctc_logits = ctc_head(encoder_out)
            mx.eval(ctc_logits)
            tokens = ctc_head.decode_greedy(ctc_logits)
            ctc_time = (time.perf_counter() - start) * 1000
            ctc_times.append(ctc_time)

        # Average (skip first iteration - warmup)
        avg_mel = sum(mel_times[1:]) / (n_iterations - 1) if n_iterations > 1 else mel_times[0]
        avg_encoder = sum(encoder_times[1:]) / (n_iterations - 1) if n_iterations > 1 else encoder_times[0]
        avg_ctc = sum(ctc_times[1:]) / (n_iterations - 1) if n_iterations > 1 else ctc_times[0]
        avg_total = avg_mel + avg_encoder + avg_ctc

        print(f"{duration:>10.1f}s | {avg_mel:>7.1f}ms | {avg_encoder:>9.1f}ms | {avg_ctc:>7.1f}ms | {avg_total:>7.1f}ms")

    print("-" * 60)
    print("\nKey insight: CTC decode is ~1-3ms regardless of audio length.")
    print("Encoder time scales with audio length.")
    print("For sub-200ms: need 100-150ms audio + ~30-50ms encoder + ~2ms CTC = ~150-200ms")


def main():
    parser = argparse.ArgumentParser(description="Test CTC streaming pipeline")
    parser.add_argument("--audio", type=str, help="Path to audio file (optional)")
    parser.add_argument("--model", type=str, default="small", help="Model size (tiny/small/medium/large-v3)")
    parser.add_argument("--benchmark", action="store_true", help="Run latency benchmark only")

    args = parser.parse_args()

    if args.benchmark:
        asyncio.run(benchmark_ctc_latency(args.model))
    else:
        asyncio.run(test_ctc_streaming(args.audio, args.model))


if __name__ == "__main__":
    main()
