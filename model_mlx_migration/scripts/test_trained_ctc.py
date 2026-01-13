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
Test CTC streaming with trained checkpoint.

Measures first partial latency with the latest trained CTC head.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import mlx.core as mx


def load_ctc_checkpoint(checkpoint_path: str, d_model: int = 1280):
    """Load trained CTC head from checkpoint."""
    from tools.whisper_mlx.ctc_head import CTCDraftHead

    # Load weights
    weights = mx.load(checkpoint_path)

    # Create model
    ctc_head = CTCDraftHead(d_model=d_model)

    # Unflatten keys if needed (checkpoints use dot notation)
    nested_weights = {}
    for key, value in weights.items():
        parts = key.split(".")
        current = nested_weights
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value

    ctc_head.update(nested_weights)
    return ctc_head


def test_ctc_inference(ctc_head, model, audio_path: str):
    """Test CTC inference on real audio."""
    from tools.whisper_mlx.audio import load_audio, log_mel_spectrogram
    from tools.whisper_mlx.tokenizer import get_whisper_tokenizer

    print(f"\nLoading audio: {audio_path}")
    audio = load_audio(audio_path)
    duration = len(audio) / 16000
    print(f"Duration: {duration:.2f}s")

    # Get tokenizer
    tokenizer = get_whisper_tokenizer(multilingual=model.is_multilingual)

    # Test different audio chunk sizes
    chunk_durations = [0.5, 1.0, 2.0, 3.0]

    print("\n" + "=" * 70)
    print("CTC Inference Test - Trained Checkpoint")
    print("=" * 70)
    print(f"{'Audio':>10} | {'Mel':>7} | {'Encoder':>9} | {'CTC':>6} | {'Total':>7} | {'Text'}")
    print("-" * 70)

    for chunk_dur in chunk_durations:
        if chunk_dur > duration:
            chunk_dur = duration

        # Extract chunk
        chunk_samples = int(16000 * chunk_dur)
        chunk = audio[:chunk_samples]

        # Time mel spectrogram
        start = time.perf_counter()
        mel = log_mel_spectrogram(chunk, n_mels=model.config.n_mels)
        mel = mx.expand_dims(mx.array(mel), axis=0)
        mx.eval(mel)
        mel_time = (time.perf_counter() - start) * 1000

        # Time encoder
        start = time.perf_counter()
        encoder_out = model.encoder(mel, variable_length=True)
        mx.eval(encoder_out)
        encoder_time = (time.perf_counter() - start) * 1000

        # Time CTC head
        start = time.perf_counter()
        logits = ctc_head(encoder_out)
        mx.eval(logits)
        tokens = ctc_head.decode_greedy(logits)
        ctc_time = (time.perf_counter() - start) * 1000

        # Decode tokens
        text = tokenizer.decode(tokens)
        text_preview = text[:40] + "..." if len(text) > 40 else text
        text_preview = text_preview.replace("\n", " ")

        total = mel_time + encoder_time + ctc_time
        print(f"{chunk_dur:>9.1f}s | {mel_time:>6.1f}ms | {encoder_time:>8.1f}ms | {ctc_time:>5.1f}ms | {total:>6.1f}ms | {text_preview}")

    # Full audio transcription
    print("\n" + "=" * 70)
    print("Full Audio CTC Transcription")
    print("=" * 70)

    mel = log_mel_spectrogram(audio, n_mels=model.config.n_mels)
    mel = mx.expand_dims(mx.array(mel), axis=0)
    mx.eval(mel)

    encoder_out = model.encoder(mel, variable_length=True)
    mx.eval(encoder_out)

    logits = ctc_head(encoder_out)
    mx.eval(logits)
    tokens = ctc_head.decode_greedy(logits)

    ctc_text = tokenizer.decode(tokens)
    print(f"CTC Output ({len(tokens)} tokens):")
    print(ctc_text)

    # Compare with decoder
    print("\n" + "=" * 70)
    print("Decoder Reference")
    print("=" * 70)
    result = model.transcribe(audio)
    print("Decoder Output:")
    print(result["text"])

    return ctc_text, result["text"]


def benchmark_first_partial_latency(ctc_head, model, n_iterations: int = 10):
    """Benchmark first partial latency with trained CTC head."""
    from tools.whisper_mlx.audio import log_mel_spectrogram

    print("\n" + "=" * 70)
    print("First Partial Latency Benchmark (trained CTC)")
    print("=" * 70)

    # Target: <200ms first partial
    # Components: audio_accumulation + mel + encoder + ctc_decode
    # For <200ms, need ~100ms audio + ~100ms inference

    test_durations = [0.1, 0.2, 0.3, 0.5, 1.0]

    print(f"\nBenchmarking {n_iterations} iterations per duration...")
    print(f"{'Duration':>10} | {'Inference':>10} | {'First Partial':>15} | {'Status'}")
    print("-" * 60)

    for audio_dur in test_durations:
        # Generate random audio (simulates real speech)
        audio = np.random.randn(int(16000 * audio_dur)).astype(np.float32) * 0.1

        inference_times = []

        for i in range(n_iterations):
            start = time.perf_counter()

            # Mel
            mel = log_mel_spectrogram(audio, n_mels=model.config.n_mels)
            mel = mx.expand_dims(mx.array(mel), axis=0)
            mx.eval(mel)

            # Encoder
            encoder_out = model.encoder(mel, variable_length=True)
            mx.eval(encoder_out)

            # CTC
            logits = ctc_head(encoder_out)
            mx.eval(logits)
            _ = ctc_head.decode_greedy(logits)

            inference_time = (time.perf_counter() - start) * 1000
            inference_times.append(inference_time)

        # Skip first (warmup)
        avg_inference = sum(inference_times[1:]) / (n_iterations - 1) if n_iterations > 1 else inference_times[0]

        # First partial = audio accumulation + inference
        first_partial = audio_dur * 1000 + avg_inference

        status = "PASS" if first_partial < 200 else ("CLOSE" if first_partial < 300 else "FAIL")
        print(f"{audio_dur*1000:>9.0f}ms | {avg_inference:>9.1f}ms | {first_partial:>14.1f}ms | {status}")

    print("-" * 60)
    print("\nNote: First partial = audio_accumulation + inference_time")
    print("Target: <200ms for Gate 1")


def main():
    parser = argparse.ArgumentParser(description="Test trained CTC head")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/ctc_head_large_v3/best.npz",
                        help="Path to CTC checkpoint")
    parser.add_argument("--audio", type=str, help="Path to test audio file")
    parser.add_argument("--model", type=str, default="large-v3", help="Model size")
    parser.add_argument("--benchmark-only", action="store_true", help="Only run latency benchmark")

    args = parser.parse_args()

    print("=" * 70)
    print("Trained CTC Head Test")
    print("=" * 70)

    # Load Whisper model
    print(f"\n1. Loading WhisperMLX ({args.model})...")
    from tools.whisper_mlx import WhisperMLX
    model = WhisperMLX.from_pretrained(f"mlx-community/whisper-{args.model}-mlx", warmup=True)
    print(f"   Loaded. d_model={model.config.n_audio_state}")

    # Load trained CTC head
    print(f"\n2. Loading CTC checkpoint: {args.checkpoint}")
    ctc_head = load_ctc_checkpoint(args.checkpoint, d_model=model.config.n_audio_state)
    print(f"   Loaded. vocab_size={ctc_head.vocab_size}")

    # Benchmark latency
    benchmark_first_partial_latency(ctc_head, model, n_iterations=5)

    if args.benchmark_only:
        return

    # Test on real audio
    if args.audio:
        test_ctc_inference(ctc_head, model, args.audio)
    else:
        # Use a LibriSpeech test sample
        test_paths = [
            "data/benchmarks/librispeech/LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac",
            "data/librispeech/test-clean/1089/134686/1089-134686-0000.flac",
        ]
        for path in test_paths:
            if Path(path).exists():
                test_ctc_inference(ctc_head, model, path)
                break
        else:
            print("\nNo test audio found. Use --audio to specify a file.")


if __name__ == "__main__":
    main()
