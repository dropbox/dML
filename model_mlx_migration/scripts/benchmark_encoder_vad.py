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
Benchmark Encoder VAD head speedup potential.

Measures:
1. VAD head inference time
2. Decoder time per position
3. Potential speedup from skipping silence frames
"""

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import mlx.core as mx


def main():
    print("=" * 60)
    print("Encoder VAD Speedup Benchmark")
    print("=" * 60)

    # Load model
    print("\n[1/4] Loading model...")
    from tools.whisper_mlx.model import WhisperMLX
    from tools.whisper_mlx.encoder_vad import load_encoder_vad_head
    from tools.whisper_mlx.audio import log_mel_spectrogram

    model = WhisperMLX.from_pretrained("large-v3", warmup=True)
    n_state = model.config.n_audio_state
    n_mels = model.config.n_mels

    # Load VAD head
    vad_head = load_encoder_vad_head(
        "checkpoints/encoder_vad/encoder_vad_best.npz",
        n_state=n_state,
        hidden_dim=256,
        dtype=mx.float32,
    )
    print("  Model and VAD head loaded")

    # Load test audio
    print("\n[2/4] Preparing test data...")
    test_file = "tests/fixtures/audio/test_speech.wav"
    audio, sr = sf.read(test_file)
    if sr != 16000:
        ratio = 16000 / sr
        new_len = int(len(audio) * ratio)
        audio = np.interp(
            np.linspace(0, len(audio), new_len),
            np.arange(len(audio)),
            audio
        )
    audio = audio.astype(np.float32)

    # Compute mel and encode
    mel = log_mel_spectrogram(audio, n_mels=n_mels)
    target_len = 3000
    if mel.shape[0] < target_len:
        mel = mx.pad(mel, [(0, target_len - mel.shape[0]), (0, 0)])
    elif mel.shape[0] > target_len:
        mel = mel[:target_len, :]

    mel_batch = mel[None].astype(mx.float32)
    encoder_output = model.embed_audio(mel_batch, variable_length=False)
    mx.eval(encoder_output)
    print(f"  Encoder output shape: {encoder_output.shape}")

    # Benchmark VAD head
    print("\n[3/4] Benchmarking VAD head...")
    n_warmup = 3
    n_iter = 20

    # Warmup
    for _ in range(n_warmup):
        probs = vad_head(encoder_output, training=False)
        mx.eval(probs)

    # Benchmark
    vad_times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        probs = vad_head(encoder_output, training=False)
        mx.eval(probs)
        t1 = time.perf_counter()
        vad_times.append(t1 - t0)

    vad_mean = np.mean(vad_times) * 1000
    vad_std = np.std(vad_times) * 1000
    print(f"  VAD head time: {vad_mean:.3f} +/- {vad_std:.3f} ms")

    # Get VAD predictions
    probs = vad_head(encoder_output, training=False)
    mx.eval(probs)
    probs_np = np.array(probs[0])

    speech_ratio = (probs_np >= 0.5).mean()
    silence_ratio = 1 - speech_ratio
    print(f"  Speech ratio: {speech_ratio:.1%}")
    print(f"  Silence ratio: {silence_ratio:.1%}")

    # Benchmark a simple decode step (to get decoder time)
    print("\n[4/4] Estimating decoder time...")

    # Run transcription and time it
    n_transcribe_iter = 3
    transcribe_times = []

    for _ in range(n_transcribe_iter):
        t0 = time.perf_counter()
        result = model.transcribe(audio)
        t1 = time.perf_counter()
        transcribe_times.append(t1 - t0)

    transcribe_mean = np.mean(transcribe_times)
    n_tokens = len(result.get("text", "").split())  # Rough estimate
    print(f"  Transcription time: {transcribe_mean*1000:.1f} ms")
    print(f"  Output: '{result.get('text', '')[:50]}...'")
    print(f"  Approx tokens: {n_tokens}")

    # Calculate speedup
    print("\n" + "=" * 60)
    print("Speedup Analysis")
    print("=" * 60)

    print(f"\nVAD head overhead: {vad_mean:.3f} ms per transcription")

    # Typical decoder time is 60-80% of total time
    # If we skip silence frames, we save decoder time proportional to silence ratio
    decoder_fraction = 0.7  # Assume decoder is 70% of time
    decoder_time = transcribe_mean * decoder_fraction

    # Potential savings = silence_ratio * decoder_time
    potential_savings = silence_ratio * decoder_time

    # Net speedup = (original_time - potential_savings + vad_overhead) / original_time
    net_speedup = transcribe_mean / (transcribe_mean - potential_savings + vad_mean/1000)

    print(f"\nAssuming decoder is {decoder_fraction*100:.0f}% of total time:")
    print(f"  Decoder time: {decoder_time*1000:.1f} ms")
    print(f"  Potential savings from skipping {silence_ratio:.1%} silence: {potential_savings*1000:.1f} ms")
    print(f"  VAD overhead: {vad_mean:.3f} ms")
    print(f"  Net speedup: {net_speedup:.2f}x")

    # More realistic scenario with 20% silence
    print("\nWith typical 20% silence ratio:")
    typical_silence = 0.20
    typical_savings = typical_silence * decoder_time
    typical_speedup = transcribe_mean / (transcribe_mean - typical_savings + vad_mean/1000)
    print(f"  Expected speedup: {typical_speedup:.2f}x")

    return 0


if __name__ == "__main__":
    sys.exit(main())
