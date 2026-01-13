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
Kokoro MLX Audio Quality Analysis

Generates audio samples with the MLX model and analyzes quality metrics.
Saves audio files and spectrograms for visual inspection.

Usage:
    python scripts/compare_audio_quality.py

Output:
    - audio files in reports/audio/
    - spectrograms in reports/audio/
    - quality report printed to console
"""

import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx

from tools.pytorch_to_mlx.converters import KokoroConverter

# Constants
SAMPLE_RATE = 24000
MODEL_PATH = Path.home() / "models" / "kokoro"
OUTPUT_DIR = Path(__file__).parent.parent / "reports" / "audio"
VOICE_NAME = "af_heart"


def setup_output_dir():
    """Create output directory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def load_mlx_model():
    """Load the MLX Kokoro model."""
    print("Loading MLX Kokoro model...")
    start = time.time()

    converter = KokoroConverter()
    model, config, _ = converter.load_from_hf()

    load_time = time.time() - start
    print(f"Model loaded in {load_time:.2f}s")

    return model, converter


def generate_test_inputs():
    """Generate test phoneme sequences.

    Returns list of (name, input_ids) tuples.
    """
    # These are IPA phoneme token IDs from the Kokoro vocabulary
    # Based on the model's vocab of 178 tokens

    test_cases = [
        # Short sequence (7 tokens)
        ("short_7tok", [16, 43, 44, 45, 46, 47, 16]),
        # Medium sequence (15 tokens)
        ("medium_15tok", [16, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 16]),
        # Longer sequence (30 tokens)
        ("long_30tok", [16] + list(range(43, 72)) + [16]),
        # Repeated pattern (stress test)
        ("pattern_20tok", [16] + [43, 44, 45, 46] * 4 + [16, 17, 16]),
    ]

    return test_cases


def synthesize_audio(model, input_ids, voice_style):
    """Synthesize audio from phoneme tokens.

    Returns:
        audio_np: numpy array of shape [samples]
        latency_ms: synthesis time in milliseconds
    """
    input_array = mx.array([input_ids])

    start = time.time()
    audio = model.synthesize(input_array, voice_style)
    mx.eval(audio)  # Force evaluation
    latency_ms = (time.time() - start) * 1000

    audio_np = np.array(audio).squeeze()  # Remove batch dimension

    return audio_np, latency_ms


def save_audio_wav(audio_np, filepath, sample_rate=24000):
    """Save audio as WAV file."""
    try:
        import soundfile as sf

        sf.write(filepath, audio_np, sample_rate)
        return True
    except ImportError:
        # Fallback to scipy
        try:
            from scipy.io import wavfile

            # Scale to int16 range
            audio_int16 = (audio_np * 32767).astype(np.int16)
            wavfile.write(filepath, sample_rate, audio_int16)
            return True
        except ImportError:
            print("Warning: Could not save WAV (install soundfile or scipy)")
            return False


def compute_spectrogram(audio_np, sample_rate=24000, n_fft=1024, hop_length=256):
    """Compute mel spectrogram for visualization."""
    try:
        # Simple STFT-based spectrogram
        from scipy import signal

        # Compute spectrogram
        f, t, Sxx = signal.spectrogram(
            audio_np, fs=sample_rate, nperseg=n_fft, noverlap=n_fft - hop_length
        )

        # Convert to dB scale
        Sxx_db = 10 * np.log10(Sxx + 1e-10)

        return f, t, Sxx_db
    except ImportError:
        return None, None, None


def save_spectrogram_plot(audio_np, filepath, sample_rate=24000):
    """Save spectrogram as PNG image."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt

        f, t, Sxx_db = compute_spectrogram(audio_np, sample_rate)

        if Sxx_db is None:
            return False

        plt.figure(figsize=(10, 4))
        plt.pcolormesh(t, f, Sxx_db, shading="gouraud", cmap="magma")
        plt.ylabel("Frequency [Hz]")
        plt.xlabel("Time [sec]")
        plt.colorbar(label="Power [dB]")
        plt.title("Spectrogram")
        plt.tight_layout()
        plt.savefig(filepath, dpi=150)
        plt.close()

        return True
    except ImportError:
        print("Warning: matplotlib not available for spectrogram plots")
        return False


def analyze_audio_quality(audio_np, sample_rate=24000):
    """Compute audio quality metrics."""
    metrics = {}

    # Basic statistics
    metrics["length_samples"] = len(audio_np)
    metrics["duration_s"] = len(audio_np) / sample_rate
    metrics["min"] = float(np.min(audio_np))
    metrics["max"] = float(np.max(audio_np))
    metrics["mean"] = float(np.mean(audio_np))
    metrics["std"] = float(np.std(audio_np))
    metrics["rms"] = float(np.sqrt(np.mean(audio_np**2)))

    # Peak-to-peak range (indicates dynamic range)
    metrics["peak_to_peak"] = metrics["max"] - metrics["min"]

    # Zero crossing rate (rough measure of frequency content)
    zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_np)))) / 2
    metrics["zero_crossing_rate"] = zero_crossings / len(audio_np)

    # Energy
    metrics["energy"] = float(np.sum(audio_np**2))

    # Silence detection (samples below threshold)
    silence_threshold = 0.01 * metrics["rms"]
    silence_ratio = np.sum(np.abs(audio_np) < silence_threshold) / len(audio_np)
    metrics["silence_ratio"] = float(silence_ratio)

    return metrics


def benchmark_synthesis(model, voice_style, num_iterations=10):
    """Benchmark synthesis latency across multiple runs."""
    test_inputs = generate_test_inputs()
    results = {}

    for name, input_ids in test_inputs:
        latencies = []
        for _ in range(num_iterations):
            _, latency_ms = synthesize_audio(model, input_ids, voice_style)
            latencies.append(latency_ms)

        # Compute stats (skip first run for warmup)
        warmup_latencies = latencies[1:] if len(latencies) > 1 else latencies

        results[name] = {
            "num_tokens": len(input_ids),
            "mean_ms": np.mean(warmup_latencies),
            "std_ms": np.std(warmup_latencies),
            "min_ms": np.min(warmup_latencies),
            "max_ms": np.max(warmup_latencies),
        }

    return results


def run_analysis():
    """Run full audio quality analysis."""
    print("=" * 60)
    print("Kokoro MLX Audio Quality Analysis")
    print("=" * 60)

    # Setup
    output_dir = setup_output_dir()
    model, converter = load_mlx_model()

    # Load voice
    voice_path = MODEL_PATH / "voices" / f"{VOICE_NAME}.pt"
    print(f"\nLoading voice: {VOICE_NAME}")
    voice_style = model.load_voice(str(voice_path))
    mx.eval(voice_style)
    print(f"Voice style shape: {voice_style.shape}")

    # Generate test inputs
    test_cases = generate_test_inputs()

    # Results storage
    all_results = []

    print("\n" + "=" * 60)
    print("Generating Audio Samples")
    print("=" * 60)

    for name, input_ids in test_cases:
        print(f"\n--- {name} ({len(input_ids)} tokens) ---")

        # Synthesize
        audio_np, latency_ms = synthesize_audio(model, input_ids, voice_style)

        # Analyze
        metrics = analyze_audio_quality(audio_np)
        metrics["name"] = name
        metrics["num_tokens"] = len(input_ids)
        metrics["latency_ms"] = latency_ms

        # Compute real-time factor
        rtf = (latency_ms / 1000) / metrics["duration_s"]
        metrics["real_time_factor"] = rtf

        print(f"  Audio duration: {metrics['duration_s']:.3f}s")
        print(f"  Synthesis time: {latency_ms:.2f}ms")
        print(f"  Real-time factor: {rtf:.2f}x (< 1.0 is faster than real-time)")
        print(f"  Audio range: [{metrics['min']:.4f}, {metrics['max']:.4f}]")
        print(f"  RMS: {metrics['rms']:.6f}")

        # Save audio
        wav_path = output_dir / f"{name}.wav"
        if save_audio_wav(audio_np, wav_path):
            print(f"  Saved: {wav_path}")

        # Save spectrogram
        spec_path = output_dir / f"{name}_spectrogram.png"
        if save_spectrogram_plot(audio_np, spec_path):
            print(f"  Saved: {spec_path}")

        all_results.append(metrics)

    # Benchmark
    print("\n" + "=" * 60)
    print("Benchmarking Synthesis Latency (10 iterations)")
    print("=" * 60)

    benchmark_results = benchmark_synthesis(model, voice_style)

    print("\n| Test | Tokens | Mean (ms) | Std (ms) | Min (ms) | Max (ms) |")
    print("|------|--------|-----------|----------|----------|----------|")
    for name, stats in benchmark_results.items():
        print(
            f"| {name} | {stats['num_tokens']} | {stats['mean_ms']:.2f} | "
            f"{stats['std_ms']:.2f} | {stats['min_ms']:.2f} | {stats['max_ms']:.2f} |"
        )

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    avg_rtf = np.mean([r["real_time_factor"] for r in all_results])
    print(f"\nAverage real-time factor: {avg_rtf:.2f}x")
    print("  (Values < 1.0 mean synthesis is faster than real-time)")

    # Check audio quality
    avg_rms = np.mean([r["rms"] for r in all_results])
    avg_silence = np.mean([r["silence_ratio"] for r in all_results])

    print("\nAudio Quality Indicators:")
    print(f"  Average RMS: {avg_rms:.6f}")
    print(f"  Average silence ratio: {avg_silence:.2%}")

    quality_issues = []
    for r in all_results:
        if r["rms"] < 0.001:
            quality_issues.append(f"{r['name']}: Very low RMS ({r['rms']:.6f})")
        if r["silence_ratio"] > 0.8:
            quality_issues.append(
                f"{r['name']}: High silence ratio ({r['silence_ratio']:.2%})"
            )

    if quality_issues:
        print("\nPotential issues:")
        for issue in quality_issues:
            print(f"  - {issue}")
    else:
        print("\nNo obvious quality issues detected.")

    print(f"\nOutput files saved to: {output_dir}")

    return all_results, benchmark_results


if __name__ == "__main__":
    results, benchmarks = run_analysis()
