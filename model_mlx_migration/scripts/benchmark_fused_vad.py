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
Benchmark FFT computation overhead to evaluate Phase 1.2 (Fused VAD-Whisper).

Measures:
1. Whisper mel spectrogram computation time
2. Silero VAD STFT computation time
3. Fused mel+STFT computation time (single FFT)
4. Potential savings from sharing FFT

Expected result: ~1.1x speedup on audio processing when sharing FFT.
"""

import sys
import time
from pathlib import Path

import numpy as np

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools/pytorch_to_mlx/converters/models"))

import mlx.core as mx


def benchmark_whisper_mel(audio: np.ndarray, iterations: int = 50) -> dict:
    """Benchmark Whisper mel spectrogram computation."""
    from tools.whisper_mlx.audio import log_mel_spectrogram

    times = []
    for _ in range(iterations):
        mx.synchronize()
        start = time.perf_counter()
        mel = log_mel_spectrogram(audio, n_mels=128)
        mx.eval(mel)
        mx.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "mean_ms": np.mean(times) * 1000,
        "std_ms": np.std(times) * 1000,
        "min_ms": np.min(times) * 1000,
    }


def benchmark_fused_mel_stft(audio: np.ndarray, iterations: int = 50) -> dict:
    """Benchmark fused mel + STFT computation (single FFT)."""
    from tools.whisper_mlx.audio import compute_stft_and_mel

    times = []
    for _ in range(iterations):
        mx.synchronize()
        start = time.perf_counter()
        mel, stft_mag = compute_stft_and_mel(audio, return_stft=True)
        mx.eval(mel)
        mx.eval(stft_mag)
        mx.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "mean_ms": np.mean(times) * 1000,
        "std_ms": np.std(times) * 1000,
        "min_ms": np.min(times) * 1000,
    }


def benchmark_silero_stft(audio: np.ndarray, iterations: int = 50) -> dict:
    """Benchmark Silero VAD STFT computation."""
    try:
        from silero_vad_mlx import STFT

        stft = STFT(filter_length=256, hop_length=128)
        audio_mx = mx.array(audio[None, :])

        times = []
        for _ in range(iterations):
            mx.synchronize()
            start = time.perf_counter()
            spec = stft(audio_mx)
            mx.eval(spec)
            mx.synchronize()
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        return {
            "mean_ms": np.mean(times) * 1000,
            "std_ms": np.std(times) * 1000,
            "min_ms": np.min(times) * 1000,
        }
    except ImportError as e:
        return {"error": str(e)}


def benchmark_silero_full(audio: np.ndarray, iterations: int = 20) -> dict:
    """Benchmark full Silero VAD inference."""
    try:
        from silero_vad_mlx import load_silero_vad

        vad = load_silero_vad(sample_rate=16000)
        audio_mx = mx.array(audio)

        times = []
        for _ in range(iterations):
            vad.reset_state()
            mx.synchronize()
            start = time.perf_counter()
            result = vad.detect(audio_mx, threshold=0.5)
            mx.eval(result)
            mx.synchronize()
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        return {
            "mean_ms": np.mean(times) * 1000,
            "std_ms": np.std(times) * 1000,
            "min_ms": np.min(times) * 1000,
        }
    except Exception as e:
        return {"error": str(e)}


def main():
    print("=" * 70)
    print("Phase 1.2: Fused VAD-Whisper FFT Benchmark")
    print("=" * 70)

    # Test with 30s audio (standard Whisper chunk)
    sample_rate = 16000
    duration = 30.0
    n_samples = int(duration * sample_rate)

    # Generate test audio (random noise)
    np.random.seed(42)
    audio = np.random.randn(n_samples).astype(np.float32) * 0.1

    print(f"\nAudio: {duration}s @ {sample_rate}Hz ({n_samples} samples)")
    print("-" * 70)

    # Benchmark Whisper mel
    print("\n1. Whisper mel spectrogram (n_fft=400, hop=160):")
    mel_result = benchmark_whisper_mel(audio)
    print(f"   Mean: {mel_result['mean_ms']:.2f}ms ± {mel_result['std_ms']:.2f}ms")

    # Benchmark fused mel + STFT
    print("\n2. Fused mel + STFT (single FFT, n_fft=400, hop=160):")
    fused_result = benchmark_fused_mel_stft(audio)
    print(f"   Mean: {fused_result['mean_ms']:.2f}ms ± {fused_result['std_ms']:.2f}ms")

    # Benchmark Silero STFT
    print("\n3. Silero STFT only (n_fft=256, hop=128):")
    silero_stft_result = benchmark_silero_stft(audio)
    if "error" not in silero_stft_result:
        print(f"   Mean: {silero_stft_result['mean_ms']:.2f}ms ± {silero_stft_result['std_ms']:.2f}ms")
    else:
        print(f"   Error: {silero_stft_result['error']}")

    # Benchmark full Silero VAD
    print("\n4. Full Silero VAD inference:")
    silero_full_result = benchmark_silero_full(audio)
    if "error" not in silero_full_result:
        print(f"   Mean: {silero_full_result['mean_ms']:.2f}ms ± {silero_full_result['std_ms']:.2f}ms")
    else:
        print(f"   Error: {silero_full_result['error']}")

    # Analysis
    print("\n" + "=" * 70)
    print("Analysis")
    print("=" * 70)

    if "error" not in silero_stft_result:
        separate_fft_time = mel_result['mean_ms'] + silero_stft_result['mean_ms']
        fused_time = fused_result['mean_ms']
        savings = separate_fft_time - fused_time
        speedup = separate_fft_time / fused_time

        print(f"\nSeparate FFT (mel + Silero STFT): {separate_fft_time:.2f}ms")
        print(f"Fused FFT (mel + STFT from same):  {fused_time:.2f}ms")
        print(f"Savings: {savings:.2f}ms ({speedup:.2f}x)")

        # Note: This doesn't account for Silero using different FFT params
        print("\nNOTE: Silero VAD uses n_fft=256, Whisper uses n_fft=400.")
        print("Direct reuse not possible without retraining VAD model.")

    # Compare to typical Whisper decode time
    print("\n" + "-" * 70)
    print("Context: Typical decode time for 30s audio")
    print("-" * 70)
    print("  Encoder:  ~200-400ms")
    print("  Decoder:  ~800-2000ms (depends on token count)")
    print("  Total:    ~1000-2400ms")
    print(f"\n  Mel computation: {mel_result['mean_ms']:.2f}ms")
    print(f"  Mel as % of total: {mel_result['mean_ms']/1000*100:.1f}% - {mel_result['mean_ms']/2400*100:.1f}%")

    # Conclusion
    print("\n" + "=" * 70)
    print("Conclusion")
    print("=" * 70)

    if "error" not in silero_stft_result:
        if savings > 0:
            overall_speedup = 1 + (savings / 1000)  # Assuming 1000ms total
            print(f"\nFused FFT would save ~{savings:.1f}ms per 30s chunk")
            print(f"Overall speedup: ~{overall_speedup:.3f}x ({(overall_speedup-1)*100:.1f}%)")
            print("\nHowever, Silero's trained weights expect n_fft=256, not 400.")
            print("Options:")
            print("  1. Train new VAD model on Whisper's FFT params (~20 commits)")
            print("  2. Use mel-energy-based VAD (no training, ~5 commits)")
            print("  3. Keep separate FFT, optimize elsewhere")
        else:
            print("\nNo savings from fused FFT - overhead negligible")
    else:
        print("\nCould not load Silero VAD for comparison")
        print(f"Mel computation alone: {mel_result['mean_ms']:.2f}ms")


if __name__ == "__main__":
    main()
