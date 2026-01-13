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
Benchmark audio loading methods for OPT-W3.

Compares:
1. ffmpeg subprocess (current)
2. soundfile + scipy resampling (proposed)
"""

import os
import subprocess
import tempfile
import time

# Suppress warnings
import warnings

import numpy as np

warnings.filterwarnings("ignore")


def load_audio_ffmpeg(file_path: str, sample_rate: int = 16000) -> np.ndarray:
    """Current implementation using ffmpeg subprocess."""
    cmd = [
        "ffmpeg", "-nostdin", "-i", file_path,
        "-f", "f32le", "-acodec", "pcm_f32le",
        "-ac", "1", "-ar", str(sample_rate), "-"
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return np.frombuffer(result.stdout, np.float32)


def load_audio_soundfile(file_path: str, sample_rate: int = 16000) -> np.ndarray:
    """Optimized implementation using soundfile + scipy resampling."""
    import soundfile as sf
    from scipy import signal

    # Load audio with soundfile (much faster than ffmpeg subprocess)
    audio, file_sr = sf.read(file_path, dtype='float32')

    # Convert stereo to mono if needed
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Resample if needed
    if file_sr != sample_rate:
        # Calculate number of samples in output
        num_samples = int(len(audio) * sample_rate / file_sr)
        audio = signal.resample(audio, num_samples).astype(np.float32)

    return audio


def create_test_audio(duration_seconds: float, sample_rate: int = 16000) -> str:
    """Create a test WAV file."""
    import soundfile as sf

    # Generate a simple sine wave
    t = np.linspace(0, duration_seconds, int(duration_seconds * sample_rate), dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    # Save to temp file
    fd, path = tempfile.mkstemp(suffix='.wav')
    os.close(fd)
    sf.write(path, audio, sample_rate)

    return path


def benchmark(func, file_path: str, iterations: int = 10) -> dict:
    """Benchmark a function."""
    times = []
    audio = None

    for i in range(iterations):
        start = time.perf_counter()
        audio = func(file_path)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "mean_ms": np.mean(times) * 1000,
        "std_ms": np.std(times) * 1000,
        "min_ms": np.min(times) * 1000,
        "samples": len(audio) if audio is not None else 0,
    }


def main():
    print("=" * 60)
    print("OPT-W3: Audio Format Optimization Benchmark")
    print("=" * 60)

    # Test durations
    durations = [2, 5, 10, 30, 60]

    print("\n## Test 1: Native 16kHz WAV files (no resampling needed)")
    print("| Duration | ffmpeg (ms) | soundfile (ms) | Speedup |")
    print("|----------|-------------|----------------|---------|")

    for dur in durations:
        # Create 16kHz test file
        test_file = create_test_audio(dur, 16000)

        try:
            ffmpeg_result = benchmark(load_audio_ffmpeg, test_file)
            sf_result = benchmark(load_audio_soundfile, test_file)

            speedup = ffmpeg_result["mean_ms"] / sf_result["mean_ms"]
            print(f"| {dur}s | {ffmpeg_result['mean_ms']:.1f} | {sf_result['mean_ms']:.1f} | **{speedup:.2f}x** |")
        finally:
            os.unlink(test_file)

    print("\n## Test 2: 44.1kHz WAV files (resampling needed)")
    print("| Duration | ffmpeg (ms) | soundfile (ms) | Speedup |")
    print("|----------|-------------|----------------|---------|")

    for dur in durations:
        # Create 44.1kHz test file
        test_file = create_test_audio(dur, 44100)

        try:
            ffmpeg_result = benchmark(load_audio_ffmpeg, test_file)
            sf_result = benchmark(load_audio_soundfile, test_file)

            speedup = ffmpeg_result["mean_ms"] / sf_result["mean_ms"]
            print(f"| {dur}s | {ffmpeg_result['mean_ms']:.1f} | {sf_result['mean_ms']:.1f} | **{speedup:.2f}x** |")
        finally:
            os.unlink(test_file)

    print("\n## Test 3: Real audio file (if available)")
    real_audio = "reports/audio/jfk.wav"
    if os.path.exists(real_audio):
        import soundfile as sf
        info = sf.info(real_audio)
        print(f"\nFile: {real_audio}")
        print(f"Duration: {info.duration:.2f}s, Sample Rate: {info.samplerate}Hz")

        ffmpeg_result = benchmark(load_audio_ffmpeg, real_audio, iterations=20)
        sf_result = benchmark(load_audio_soundfile, real_audio, iterations=20)

        speedup = ffmpeg_result["mean_ms"] / sf_result["mean_ms"]
        print("\n| Method | Mean (ms) | Std (ms) |")
        print("|--------|-----------|----------|")
        print(f"| ffmpeg | {ffmpeg_result['mean_ms']:.2f} | {ffmpeg_result['std_ms']:.2f} |")
        print(f"| soundfile | {sf_result['mean_ms']:.2f} | {sf_result['std_ms']:.2f} |")
        print(f"\n**Speedup: {speedup:.2f}x**")

        # Verify outputs match
        audio1 = load_audio_ffmpeg(real_audio)
        audio2 = load_audio_soundfile(real_audio)

        # Trim to same length
        min_len = min(len(audio1), len(audio2))
        audio1 = audio1[:min_len]
        audio2 = audio2[:min_len]

        max_diff = np.max(np.abs(audio1 - audio2))
        correlation = np.corrcoef(audio1, audio2)[0, 1]

        print("\n## Numerical Verification")
        print(f"- Max difference: {max_diff:.6f}")
        print(f"- Correlation: {correlation:.6f}")
        print(f"- Match: {'PASS' if max_diff < 0.01 else 'FAIL'}")
    else:
        print(f"\nSkipped: {real_audio} not found")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
