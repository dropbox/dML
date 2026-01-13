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
Phase 9.6: Audio Cleaning Benchmark

Evaluates audio cleaning pipeline on LibriSpeech + DEMAND noisy mix:
- SI-SNR improvement
- WER improvement (using Whisper)
- Latency measurements

Usage:
    python scripts/benchmark_audio_cleaning.py
    python scripts/benchmark_audio_cleaning.py --snr 10 --num-samples 50
"""

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class BenchmarkResult:
    """Results for a single sample."""
    sample_id: str
    snr_db: float
    si_snr_input: float
    si_snr_output: float
    si_snr_improvement: float
    wer_noisy: float
    wer_cleaned: float
    wer_improvement: float
    cleaning_time_ms: float
    enhancement_type: str


def compute_si_snr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """
    Compute Scale-Invariant Signal-to-Noise Ratio.

    SI-SNR = 10 * log10(||s_target||^2 / ||e_noise||^2)
    where s_target = <s, s_hat> * s / ||s||^2
    and e_noise = s_hat - s_target

    Args:
        reference: Clean reference signal
        estimate: Processed signal

    Returns:
        SI-SNR in dB
    """
    # Ensure same length
    min_len = min(len(reference), len(estimate))
    reference = reference[:min_len].astype(np.float64)
    estimate = estimate[:min_len].astype(np.float64)

    # Remove mean
    reference = reference - np.mean(reference)
    estimate = estimate - np.mean(estimate)

    # Compute SI-SNR
    dot = np.dot(reference, estimate)
    s_target = dot * reference / (np.dot(reference, reference) + 1e-8)
    e_noise = estimate - s_target

    si_snr = 10 * np.log10(
        np.dot(s_target, s_target) / (np.dot(e_noise, e_noise) + 1e-8) + 1e-8
    )

    return float(si_snr)


def mix_audio_at_snr(
    clean: np.ndarray,
    noise: np.ndarray,
    target_snr_db: float
) -> np.ndarray:
    """
    Mix clean audio with noise at target SNR level.

    Args:
        clean: Clean audio signal
        noise: Noise signal (will be repeated/cropped to match)
        target_snr_db: Target SNR in dB

    Returns:
        Mixed noisy audio
    """
    # Match noise length to clean
    if len(noise) < len(clean):
        # Repeat noise
        repeats = int(np.ceil(len(clean) / len(noise)))
        noise = np.tile(noise, repeats)
    noise = noise[:len(clean)]

    # Compute current energies
    clean_energy = np.sum(clean ** 2) + 1e-8
    noise_energy = np.sum(noise ** 2) + 1e-8

    # Compute scaling factor for target SNR
    # SNR = 10 * log10(P_signal / P_noise)
    # => P_noise = P_signal / 10^(SNR/10)
    target_noise_energy = clean_energy / (10 ** (target_snr_db / 10))
    scale = np.sqrt(target_noise_energy / noise_energy)

    # Mix
    noisy = clean + scale * noise

    # Normalize to prevent clipping
    max_val = np.max(np.abs(noisy))
    if max_val > 1.0:
        noisy = noisy / max_val * 0.95

    return noisy.astype(np.float32)


def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Compute Word Error Rate.

    WER = (S + D + I) / N
    where S=substitutions, D=deletions, I=insertions, N=reference words
    """
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    if len(ref_words) == 0:
        return 1.0 if len(hyp_words) > 0 else 0.0

    # Dynamic programming for edit distance
    m, n = len(ref_words), len(hyp_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n] / len(ref_words)


def load_audio(path: str, sample_rate: int = 16000) -> np.ndarray:
    """Load audio file and resample to target rate."""
    import soundfile as sf
    from scipy import signal as scipy_signal

    audio, file_sr = sf.read(path, dtype='float32')

    # Convert stereo to mono
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Resample if needed
    if file_sr != sample_rate:
        num_samples = int(len(audio) * sample_rate / file_sr)
        audio = scipy_signal.resample(audio, num_samples).astype(np.float32)

    return audio


def get_transcript_for_sample(sample_path: str) -> Optional[str]:
    """Get transcript for a LibriSpeech sample."""
    sample_path = Path(sample_path)

    # LibriSpeech transcript format: speaker-chapter-id.trans.txt
    # in the same directory as the audio files
    parts = sample_path.stem.split('-')
    if len(parts) != 3:
        return None

    trans_file = sample_path.parent / f"{parts[0]}-{parts[1]}.trans.txt"

    if not trans_file.exists():
        return None

    # Find the line matching our sample
    sample_id = sample_path.stem
    with open(trans_file, 'r') as f:
        for line in f:
            if line.startswith(sample_id):
                # Format: "id transcript"
                return line.split(' ', 1)[1].strip()

    return None


def run_benchmark(
    librispeech_dir: str,
    demand_dir: str,
    num_samples: int = 50,
    snr_levels: list = None,
    output_dir: str = "reports/audio_cleaning_benchmark",
) -> dict:
    """
    Run the audio cleaning benchmark.

    Args:
        librispeech_dir: Path to LibriSpeech test-clean
        demand_dir: Path to DEMAND noise directory
        num_samples: Number of samples to evaluate
        snr_levels: List of SNR levels to test (default: [0, 5, 10, 15, 20])
        output_dir: Directory to save results

    Returns:
        Summary statistics
    """
    from tools.audio_cleaning import AdaptiveRouter
    from tools.whisper_mlx import WhisperMLX

    if snr_levels is None:
        snr_levels = [5, 10, 15, 20]

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize router
    print("Initializing AdaptiveRouter...")
    router = AdaptiveRouter(sample_rate=16000)
    router.warmup()

    # Initialize Whisper
    print("Initializing WhisperMLX...")
    whisper = WhisperMLX.from_pretrained("large-v3")

    # Find test samples
    print(f"Finding LibriSpeech samples from {librispeech_dir}...")
    sample_files = list(Path(librispeech_dir).rglob("*.flac"))
    if len(sample_files) == 0:
        print(f"ERROR: No .flac files found in {librispeech_dir}")
        return {}

    random.shuffle(sample_files)
    sample_files = sample_files[:num_samples]
    print(f"Selected {len(sample_files)} samples")

    # Find noise files
    print(f"Loading DEMAND noise from {demand_dir}...")
    noise_files = list(Path(demand_dir).rglob("*.wav"))
    if len(noise_files) == 0:
        print(f"ERROR: No .wav files found in {demand_dir}")
        return {}

    # Load all noise files (use ch01 from each environment)
    noises = []
    for nf in noise_files:
        if 'ch01.wav' in str(nf):
            try:
                noise = load_audio(str(nf))
                noises.append((nf.parent.name, noise))
                print(f"  Loaded noise: {nf.parent.name}")
            except Exception as e:
                print(f"  Warning: Could not load {nf}: {e}")

    if len(noises) == 0:
        print("ERROR: No noise files could be loaded")
        return {}

    print(f"Loaded {len(noises)} noise environments")

    # Run benchmark
    all_results = []

    for snr_db in snr_levels:
        print(f"\n=== Testing SNR = {snr_db} dB ===")

        snr_results = []

        for i, sample_file in enumerate(sample_files):
            sample_id = sample_file.stem

            # Get transcript
            transcript = get_transcript_for_sample(sample_file)
            if transcript is None:
                continue

            try:
                # Load clean audio
                clean = load_audio(str(sample_file))

                # Select random noise
                noise_name, noise = random.choice(noises)

                # Mix at target SNR
                noisy = mix_audio_at_snr(clean, noise, snr_db)

                # Compute input SI-SNR
                si_snr_input = compute_si_snr(clean, noisy)

                # Run audio cleaning
                start_time = time.perf_counter()
                result = router.route(noisy)
                cleaning_time_ms = (time.perf_counter() - start_time) * 1000

                cleaned = result.audio

                # Compute output SI-SNR
                si_snr_output = compute_si_snr(clean, cleaned)

                # Transcribe noisy (without cleaning)
                noisy_result = whisper.transcribe(
                    noisy,
                    audio_cleaning=False
                )
                hyp_noisy = noisy_result.get("text", "").strip()

                # Transcribe cleaned
                clean_result = whisper.transcribe(
                    cleaned,
                    audio_cleaning=False  # Already cleaned
                )
                hyp_cleaned = clean_result.get("text", "").strip()

                # Compute WER
                wer_noisy = compute_wer(transcript, hyp_noisy)
                wer_cleaned = compute_wer(transcript, hyp_cleaned)

                bench_result = BenchmarkResult(
                    sample_id=sample_id,
                    snr_db=snr_db,
                    si_snr_input=si_snr_input,
                    si_snr_output=si_snr_output,
                    si_snr_improvement=si_snr_output - si_snr_input,
                    wer_noisy=wer_noisy,
                    wer_cleaned=wer_cleaned,
                    wer_improvement=wer_noisy - wer_cleaned,
                    cleaning_time_ms=cleaning_time_ms,
                    enhancement_type=result.enhancement_type,
                )

                snr_results.append(bench_result)
                all_results.append(bench_result)

                if (i + 1) % 10 == 0:
                    avg_si_snr_imp = np.mean([r.si_snr_improvement for r in snr_results])
                    avg_wer_imp = np.mean([r.wer_improvement for r in snr_results])
                    print(f"  [{i+1}/{len(sample_files)}] "
                          f"SI-SNR: +{avg_si_snr_imp:.1f}dB, "
                          f"WER: {avg_wer_imp*100:+.1f}%")

            except Exception as e:
                print(f"  Warning: Error processing {sample_id}: {e}")
                continue

        # Summary for this SNR level
        if snr_results:
            avg_si_snr_imp = np.mean([r.si_snr_improvement for r in snr_results])
            avg_wer_noisy = np.mean([r.wer_noisy for r in snr_results])
            avg_wer_cleaned = np.mean([r.wer_cleaned for r in snr_results])
            avg_time = np.mean([r.cleaning_time_ms for r in snr_results])

            print(f"\nSNR={snr_db}dB Summary:")
            print(f"  SI-SNR improvement: +{avg_si_snr_imp:.2f} dB")
            print(f"  WER noisy: {avg_wer_noisy*100:.1f}%")
            print(f"  WER cleaned: {avg_wer_cleaned*100:.1f}%")
            print(f"  WER improvement: {(avg_wer_noisy - avg_wer_cleaned)*100:+.1f}%")
            print(f"  Avg cleaning time: {avg_time:.1f}ms")

    # Overall summary
    print("\n" + "=" * 60)
    print("OVERALL BENCHMARK RESULTS")
    print("=" * 60)

    summary = {}
    for snr_db in snr_levels:
        snr_results = [r for r in all_results if r.snr_db == snr_db]
        if snr_results:
            summary[f"snr_{snr_db}db"] = {
                "n_samples": len(snr_results),
                "si_snr_improvement_db": float(np.mean([r.si_snr_improvement for r in snr_results])),
                "wer_noisy_pct": float(np.mean([r.wer_noisy for r in snr_results]) * 100),
                "wer_cleaned_pct": float(np.mean([r.wer_cleaned for r in snr_results]) * 100),
                "wer_relative_improvement_pct": float(
                    (np.mean([r.wer_noisy for r in snr_results]) -
                     np.mean([r.wer_cleaned for r in snr_results])) /
                    (np.mean([r.wer_noisy for r in snr_results]) + 1e-8) * 100
                ),
                "avg_cleaning_time_ms": float(np.mean([r.cleaning_time_ms for r in snr_results])),
            }

    # Print summary table
    print(f"\n{'SNR (dB)':<10} {'SI-SNR Imp':<12} {'WER Noisy':<12} {'WER Clean':<12} {'WER Rel Imp':<12} {'Time (ms)':<10}")
    print("-" * 68)
    for snr_db in snr_levels:
        key = f"snr_{snr_db}db"
        if key in summary:
            s = summary[key]
            print(f"{snr_db:<10} "
                  f"+{s['si_snr_improvement_db']:.1f} dB{'':<5}"
                  f"{s['wer_noisy_pct']:.1f}%{'':<7}"
                  f"{s['wer_cleaned_pct']:.1f}%{'':<7}"
                  f"{s['wer_relative_improvement_pct']:+.1f}%{'':<6}"
                  f"{s['avg_cleaning_time_ms']:.1f}")

    # Save detailed results
    results_file = os.path.join(output_dir, "benchmark_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            "summary": summary,
            "results": [
                {
                    "sample_id": r.sample_id,
                    "snr_db": r.snr_db,
                    "si_snr_input": r.si_snr_input,
                    "si_snr_output": r.si_snr_output,
                    "si_snr_improvement": r.si_snr_improvement,
                    "wer_noisy": r.wer_noisy,
                    "wer_cleaned": r.wer_cleaned,
                    "wer_improvement": r.wer_improvement,
                    "cleaning_time_ms": r.cleaning_time_ms,
                    "enhancement_type": r.enhancement_type,
                }
                for r in all_results
            ]
        }, f, indent=2)

    print(f"\nDetailed results saved to: {results_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Audio Cleaning Benchmark")
    parser.add_argument(
        "--librispeech-dir",
        default="data/LibriSpeech/test-clean",
        help="Path to LibriSpeech test-clean directory"
    )
    parser.add_argument(
        "--demand-dir",
        default="data/demand_noise",
        help="Path to DEMAND noise directory"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of samples to evaluate per SNR level"
    )
    parser.add_argument(
        "--snr",
        type=float,
        nargs='+',
        default=[5, 10, 15, 20],
        help="SNR levels to test (dB)"
    )
    parser.add_argument(
        "--output-dir",
        default="reports/audio_cleaning_benchmark",
        help="Output directory for results"
    )

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.librispeech_dir):
        print(f"ERROR: LibriSpeech directory not found: {args.librispeech_dir}")
        print("Try: --librispeech-dir data/LibriSpeech/dev-clean")
        sys.exit(1)

    if not os.path.exists(args.demand_dir):
        print(f"ERROR: DEMAND directory not found: {args.demand_dir}")
        sys.exit(1)

    run_benchmark(
        librispeech_dir=args.librispeech_dir,
        demand_dir=args.demand_dir,
        num_samples=args.num_samples,
        snr_levels=args.snr,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
