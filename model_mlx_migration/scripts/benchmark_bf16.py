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
Benchmark BF16 vs FP16 compute on Whisper.

P0.3: Test if mx.bfloat16 provides speedup over mx.float16.

Expected outcomes:
- BF16 compute should be ~1.0-1.3x speed of FP16 (no slower)
- WER should be identical or within noise (within 0.05%)
- Memory usage should be similar

M4 Max has native BF16 support via Neural Engine, but MLX uses GPU.
BF16 has larger dynamic range (exponent) but less precision (mantissa).
For Whisper this should be fine since audio features are well-scaled.
"""

import argparse
import time
from pathlib import Path
from typing import Dict

import mlx.core as mx

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_bf16_support() -> bool:
    """Check if BF16 is supported on this device."""
    try:
        # Create a simple BF16 tensor and perform computation
        x = mx.array([1.0, 2.0, 3.0], dtype=mx.bfloat16)
        y = mx.array([4.0, 5.0, 6.0], dtype=mx.bfloat16)
        z = mx.matmul(x.reshape(1, 3), y.reshape(3, 1))
        mx.eval(z)
        return True
    except Exception as e:
        print(f"BF16 not supported: {e}")
        return False


def benchmark_model(
    model_name: str = "large-v3",
    dtype: mx.Dtype = mx.float16,
    n_runs: int = 5,
    audio_path: str = None,
) -> Dict:
    """
    Benchmark Whisper model with specified dtype.

    Returns dict with:
    - encode_time_ms: Average encoder time
    - decode_time_ms: Average decoder time
    - total_time_ms: Average total time
    - text: Transcribed text
    - n_tokens: Number of output tokens
    """
    from tools.whisper_mlx import WhisperMLX
    from tools.whisper_mlx.audio import load_audio, SAMPLE_RATE

    # Load audio
    if audio_path is None:
        audio_path = Path(__file__).parent.parent / "tests/fixtures/audio/test_speech.wav"
    audio = load_audio(str(audio_path), sample_rate=SAMPLE_RATE)

    # Create model with specified dtype
    print(f"Loading model with dtype={dtype}...")
    model = WhisperMLX.from_pretrained(
        model_name,
        dtype=dtype,
        warmup=True,
    )

    # Warmup run (not counted)
    _ = model.transcribe(audio)

    # Benchmark runs
    encode_times = []
    decode_times = []
    total_times = []
    result = None

    for i in range(n_runs):
        # Clear any pending ops
        mx.synchronize()

        t0 = time.perf_counter()

        # Time encoding separately
        from tools.whisper_mlx.audio import log_mel_spectrogram
        mel = log_mel_spectrogram(audio, n_mels=model.config.n_mels)
        target_len = model.config.n_audio_ctx * 2
        if mel.shape[0] < target_len:
            mel = mx.pad(mel, [(0, target_len - mel.shape[0]), (0, 0)])
        mel = mel[None]

        t_enc_start = time.perf_counter()
        audio_features = model.embed_audio(mel, variable_length=False)
        mx.eval(audio_features)
        t_enc = time.perf_counter() - t_enc_start

        # Full transcription (includes decoding)
        result = model.transcribe(audio)
        mx.synchronize()
        t_total = time.perf_counter() - t0

        # Estimate decode time (total - encode - mel)
        t_decode = t_total - t_enc

        encode_times.append(t_enc * 1000)
        decode_times.append(t_decode * 1000)
        total_times.append(t_total * 1000)

        print(f"  Run {i+1}: encode={t_enc*1000:.1f}ms, decode={t_decode*1000:.1f}ms, total={t_total*1000:.1f}ms")

    return {
        "dtype": str(dtype),
        "encode_time_ms": sum(encode_times) / len(encode_times),
        "decode_time_ms": sum(decode_times) / len(decode_times),
        "total_time_ms": sum(total_times) / len(total_times),
        "text": result["text"] if result else "",
        "n_tokens": len(result.get("segments", [])),
    }


def compare_wer(text1: str, text2: str) -> float:
    """
    Compute word error rate between two texts.

    Returns WER as a fraction (0.0 = identical).
    """
    words1 = text1.lower().split()
    words2 = text2.lower().split()

    if not words1 and not words2:
        return 0.0
    if not words1 or not words2:
        return 1.0

    # Simple WER using edit distance
    import numpy as np

    m, n = len(words1), len(words2)
    dp = np.zeros((m + 1, n + 1), dtype=int)

    for i in range(m + 1):
        dp[i, 0] = i
    for j in range(n + 1):
        dp[0, j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if words1[i-1] == words2[j-1]:
                dp[i, j] = dp[i-1, j-1]
            else:
                dp[i, j] = 1 + min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1])

    return dp[m, n] / max(m, n)


def main():
    parser = argparse.ArgumentParser(description="Benchmark BF16 vs FP16 on Whisper")
    parser.add_argument("--model", default="large-v3", help="Model name")
    parser.add_argument("--runs", type=int, default=5, help="Number of benchmark runs")
    parser.add_argument("--audio", help="Path to test audio file")
    args = parser.parse_args()

    print("=" * 60)
    print("P0.3: BF16 vs FP16 Benchmark")
    print("=" * 60)

    # Check BF16 support
    print("\nChecking BF16 support...")
    if not check_bf16_support():
        print("ERROR: BF16 not supported on this device")
        return
    print("BF16 supported!")

    # Benchmark FP16 (baseline)
    print("\n" + "-" * 60)
    print("Benchmarking FP16 (baseline)...")
    print("-" * 60)
    fp16_results = benchmark_model(
        model_name=args.model,
        dtype=mx.float16,
        n_runs=args.runs,
        audio_path=args.audio,
    )

    # Benchmark BF16
    print("\n" + "-" * 60)
    print("Benchmarking BF16...")
    print("-" * 60)
    bf16_results = benchmark_model(
        model_name=args.model,
        dtype=mx.bfloat16,
        n_runs=args.runs,
        audio_path=args.audio,
    )

    # Compare results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print("\nFP16 (baseline):")
    print(f"  Encode: {fp16_results['encode_time_ms']:.1f}ms")
    print(f"  Decode: {fp16_results['decode_time_ms']:.1f}ms")
    print(f"  Total:  {fp16_results['total_time_ms']:.1f}ms")
    print(f"  Text: {fp16_results['text'][:100]}...")

    print("\nBF16:")
    print(f"  Encode: {bf16_results['encode_time_ms']:.1f}ms")
    print(f"  Decode: {bf16_results['decode_time_ms']:.1f}ms")
    print(f"  Total:  {bf16_results['total_time_ms']:.1f}ms")
    print(f"  Text: {bf16_results['text'][:100]}...")

    # Compute speedups
    encode_speedup = fp16_results['encode_time_ms'] / bf16_results['encode_time_ms']
    decode_speedup = fp16_results['decode_time_ms'] / bf16_results['decode_time_ms']
    total_speedup = fp16_results['total_time_ms'] / bf16_results['total_time_ms']

    print("\nSpeedup (BF16 vs FP16):")
    print(f"  Encode: {encode_speedup:.2f}x")
    print(f"  Decode: {decode_speedup:.2f}x")
    print(f"  Total:  {total_speedup:.2f}x")

    # Compare accuracy
    wer = compare_wer(fp16_results['text'], bf16_results['text'])
    print("\nAccuracy:")
    print(f"  WER (BF16 vs FP16): {wer*100:.2f}%")
    print(f"  Texts match: {fp16_results['text'] == bf16_results['text']}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if total_speedup >= 1.1:
        print(f"RESULT: BF16 provides {total_speedup:.2f}x speedup - VIABLE")
    elif total_speedup >= 0.95:
        print(f"RESULT: BF16 similar speed ({total_speedup:.2f}x) - NO BENEFIT")
    else:
        print(f"RESULT: BF16 slower ({total_speedup:.2f}x) - NOT RECOMMENDED")

    if wer == 0:
        print("QUALITY: Identical output")
    elif wer < 0.01:
        print(f"QUALITY: Near-identical ({wer*100:.2f}% WER)")
    elif wer < 0.05:
        print(f"QUALITY: Acceptable ({wer*100:.2f}% WER, within noise)")
    else:
        print(f"QUALITY: WARNING - WER {wer*100:.2f}% exceeds threshold")


if __name__ == "__main__":
    main()
