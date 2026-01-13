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
Test script for OPT-W2: Dynamic Chunk Sizing

This script validates whether processing shorter audio sequences
provides speedup compared to always padding to 30 seconds.

The hypothesis: Whisper's encoder uses O(n²) attention, so shorter
sequences should be proportionally faster.
"""

import os
import tempfile
import time

import numpy as np


def create_test_audio(duration_sec: float, output_path: str) -> str:
    """Create a test audio file with speech-like content."""
    # Generate a simple sine wave at 440 Hz (speech-like frequency)
    sample_rate = 16000
    samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, samples)

    # Create a more speech-like signal with varying frequencies
    audio = np.sin(2 * np.pi * 200 * t) * 0.3  # Base frequency
    audio += np.sin(2 * np.pi * 300 * t) * 0.2  # Harmonic
    audio += np.sin(2 * np.pi * 400 * t) * 0.1  # Harmonic

    # Add some noise
    audio += np.random.randn(samples) * 0.05

    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8

    # Save as wav
    import wave
    audio_int16 = (audio * 32767).astype(np.int16)

    with wave.open(output_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())

    return output_path


def benchmark_standard_transcription(audio_path: str, model_path: str) -> tuple[float, str]:
    """Benchmark standard mlx_whisper transcription."""
    import mlx_whisper

    start = time.perf_counter()
    result = mlx_whisper.transcribe(audio_path, path_or_hf_repo=model_path)
    elapsed = time.perf_counter() - start

    return elapsed, result['text']


def benchmark_encoder_only(audio_path: str, model_path: str, pad_to_30s: bool = True) -> tuple[float, int]:
    """
    Benchmark just the encoder with or without full padding.

    Returns: (elapsed_time, actual_frames_processed)
    """
    import mlx.core as mx
    from mlx_whisper.audio import N_FRAMES, load_audio, log_mel_spectrogram, pad_or_trim
    from mlx_whisper.load_models import load_model

    # Load model
    model = load_model(model_path, dtype=mx.float16)

    # Load audio
    audio = load_audio(audio_path)
    mel = log_mel_spectrogram(audio, n_mels=model.dims.n_mels)
    actual_frames = mel.shape[0]

    if pad_to_30s:
        mel = pad_or_trim(mel, N_FRAMES, axis=-2)
        process_frames = N_FRAMES
    else:
        # Round up to nearest multiple of 2 for conv2 stride
        process_frames = ((actual_frames + 1) // 2) * 2
        if process_frames > actual_frames:
            mel = pad_or_trim(mel, process_frames, axis=-2)

    mel = mel.astype(mx.float16)

    # Create dynamic encoder if not padding
    if not pad_to_30s:
        # Patch encoder to handle variable length
        original_pos_emb = model.encoder._positional_embedding
        import mlx.nn as nn

        def dynamic_encode(x):
            x = nn.gelu(model.encoder.conv1(x))
            x = nn.gelu(model.encoder.conv2(x))
            seq_len = x.shape[1]
            pos_emb = original_pos_emb[:seq_len]
            x = x + pos_emb
            for block in model.encoder.blocks:
                x, _, _ = block(x)
            x = model.encoder.ln_post(x)
            return x

        # Time the encoding
        mx.eval(mel)  # Ensure mel is evaluated
        start = time.perf_counter()
        encoded = dynamic_encode(mel[None])
        mx.eval(encoded)
        elapsed = time.perf_counter() - start
    else:
        # Standard encoding
        mx.eval(mel)
        start = time.perf_counter()
        encoded = model.encoder(mel[None])
        mx.eval(encoded)
        elapsed = time.perf_counter() - start

    return elapsed, process_frames


def main():
    print("=" * 60)
    print("OPT-W2: Dynamic Chunk Sizing - Benchmark")
    print("=" * 60)
    print()

    # Use turbo model for faster testing
    model_path = "mlx-community/whisper-large-v3-turbo"

    # Test various audio durations
    durations = [2, 5, 10, 15, 20, 25, 30]

    results = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for duration in durations:
            print(f"\n--- Testing {duration}s audio ---")
            audio_path = os.path.join(tmpdir, f"test_{duration}s.wav")
            create_test_audio(duration, audio_path)

            # Benchmark encoder with padding to 30s
            time_padded, frames_padded = benchmark_encoder_only(
                audio_path, model_path, pad_to_30s=True
            )
            print(f"  Padded to 30s: {time_padded*1000:.1f}ms ({frames_padded} frames)")

            # Benchmark encoder without padding
            time_dynamic, frames_dynamic = benchmark_encoder_only(
                audio_path, model_path, pad_to_30s=False
            )
            print(f"  Dynamic: {time_dynamic*1000:.1f}ms ({frames_dynamic} frames)")

            speedup = time_padded / time_dynamic if time_dynamic > 0 else 0
            print(f"  Speedup: {speedup:.2f}x")

            results.append({
                'duration': duration,
                'time_padded': time_padded,
                'time_dynamic': time_dynamic,
                'frames_padded': frames_padded,
                'frames_dynamic': frames_dynamic,
                'speedup': speedup,
            })

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print()
    print(f"{'Duration':>8} | {'Padded':>10} | {'Dynamic':>10} | {'Speedup':>8} | {'Savings':>8}")
    print("-" * 60)

    for r in results:
        savings_pct = (1 - 1/r['speedup']) * 100 if r['speedup'] > 0 else 0
        print(f"{r['duration']:>7}s | {r['time_padded']*1000:>8.1f}ms | "
              f"{r['time_dynamic']*1000:>8.1f}ms | {r['speedup']:>7.2f}x | {savings_pct:>6.1f}%")

    # Conclusion
    print()
    avg_speedup = np.mean([r['speedup'] for r in results if r['duration'] < 30])
    print(f"Average speedup for audio <30s: {avg_speedup:.2f}x")

    if avg_speedup > 1.2:
        print("\nConclusion: Dynamic chunk sizing IS beneficial")
        print("Recommendation: Implement full OPT-W2 integration")
    else:
        print("\nConclusion: Dynamic chunk sizing provides minimal benefit")
        print("Recommendation: Focus on other optimizations")


if __name__ == "__main__":
    main()
