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
Benchmark Encoder VAD Speedup on Mixed Speech/Silence Audio.

Tests the encoder VAD optimization on audio with varying amounts of silence
to demonstrate the speedup potential.

Usage:
    python scripts/benchmark_encoder_vad_speedup.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import time
from typing import Dict

import numpy as np


def create_mixed_audio(
    speech_audio: np.ndarray,
    silence_ratio: float,
    sample_rate: int = 16000,
) -> np.ndarray:
    """
    Create audio with speech followed by silence.

    Args:
        speech_audio: Speech waveform
        silence_ratio: Fraction of total duration that should be silence
        sample_rate: Sample rate

    Returns:
        Audio with speech + silence
    """
    speech_duration = len(speech_audio)

    if silence_ratio <= 0:
        return speech_audio

    # Calculate silence duration to achieve target ratio
    total_duration = speech_duration / (1 - silence_ratio)
    silence_duration = int(total_duration - speech_duration)

    # Create silence
    silence = np.zeros(silence_duration, dtype=np.float32)

    # Combine: speech then silence
    return np.concatenate([speech_audio, silence])


def benchmark_single(
    model,
    audio: np.ndarray,
    label: str,
) -> Dict:
    """Run single transcription and return timing/quality info."""
    t0 = time.perf_counter()
    result = model.transcribe(audio, language="en", verbose=False)
    elapsed = time.perf_counter() - t0

    return {
        "label": label,
        "time_ms": elapsed * 1000,
        "text": result.get("text", ""),
        "vad_speech_ratio": result.get("vad_speech_ratio"),
    }


def main():
    print("=" * 70)
    print("Encoder VAD Speedup Benchmark")
    print("=" * 70)

    # Load a speech sample
    print("\n[1/4] Loading speech sample from LibriSpeech...")
    from datasets import load_dataset
    ds = load_dataset("librispeech_asr", "clean", split="test", streaming=True)
    sample = next(iter(ds))
    speech_audio = sample["audio"]["array"].astype(np.float32)
    speech_duration_s = len(speech_audio) / 16000
    print(f"  Speech sample: {speech_duration_s:.2f}s")

    # Load model WITHOUT encoder VAD
    print("\n[2/4] Loading WhisperMLX model...")
    from tools.whisper_mlx.model import WhisperMLX
    model = WhisperMLX.from_pretrained("large-v3", warmup=True)
    print("  Model loaded (no encoder VAD)")

    # Benchmark different silence ratios WITHOUT encoder VAD
    silence_ratios = [0.0, 0.2, 0.4, 0.6, 0.8]

    print("\n[3/4] Benchmarking WITHOUT encoder VAD...")
    baseline_results = []
    for ratio in silence_ratios:
        audio = create_mixed_audio(speech_audio, ratio)
        duration_s = len(audio) / 16000
        result = benchmark_single(model, audio, f"silence={ratio:.0%}")
        result["duration_s"] = duration_s
        result["silence_ratio"] = ratio
        baseline_results.append(result)
        print(f"  {ratio:>4.0%} silence ({duration_s:>5.1f}s): {result['time_ms']:>7.1f}ms")

    # Load encoder VAD head
    print("\n[4/4] Benchmarking WITH encoder VAD...")
    model.load_encoder_vad_head(
        "checkpoints/encoder_vad/encoder_vad_best.npz",
        threshold=0.15,
    )
    print("  Encoder VAD loaded (threshold=0.15)")

    vad_results = []
    for ratio in silence_ratios:
        audio = create_mixed_audio(speech_audio, ratio)
        duration_s = len(audio) / 16000
        result = benchmark_single(model, audio, f"silence={ratio:.0%}")
        result["duration_s"] = duration_s
        result["silence_ratio"] = ratio
        vad_results.append(result)
        vad_info = f" (speech_ratio={result['vad_speech_ratio']:.1%})" if result.get('vad_speech_ratio') else ""
        print(f"  {ratio:>4.0%} silence ({duration_s:>5.1f}s): {result['time_ms']:>7.1f}ms{vad_info}")

    # Results summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'Silence %':<12} {'Duration':>10} {'Baseline':>12} {'With VAD':>12} {'Speedup':>10}")
    print("-" * 60)

    for b, v in zip(baseline_results, vad_results):
        speedup = b["time_ms"] / v["time_ms"] if v["time_ms"] > 0 else float('inf')
        print(f"{b['silence_ratio']:>10.0%} {b['duration_s']:>9.1f}s {b['time_ms']:>10.1f}ms {v['time_ms']:>10.1f}ms {speedup:>9.2f}x")

    print("\n" + "-" * 70)
    print("NOTE: Encoder VAD provides speedup primarily by:")
    print("  1. Skipping decoder entirely for mostly-silent audio (<5% speech)")
    print("  2. Skipping silent chunks in transcribe_long()")
    print("\nFor audio with speech, the VAD overhead is ~0.3ms (negligible).")
    print("For audio with >95% silence, decoder is skipped entirely.")

    # Check if transcriptions match
    print("\n" + "=" * 70)
    print("QUALITY CHECK")
    print("=" * 70)
    all_match = True
    for b, v in zip(baseline_results, vad_results):
        match = b["text"].strip() == v["text"].strip()
        if not match:
            all_match = False
            print(f"\nMISMATCH at {b['silence_ratio']:.0%} silence:")
            print(f"  Baseline: {b['text'][:50]}...")
            print(f"  With VAD: {v['text'][:50]}...")
        else:
            print(f"{b['silence_ratio']:>4.0%} silence: ✓ transcriptions match")

    if all_match:
        print("\nPASS: All transcriptions match between baseline and VAD")
        return 0
    else:
        print("\nFAIL: Some transcriptions differ")
        return 1


if __name__ == "__main__":
    sys.exit(main())
