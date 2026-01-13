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
Phase 9 Benchmark: Whisper Baseline Performance Evaluation.

Establishes baseline WER and latency metrics for Whisper large-v3 with
beam search decoding on LibriSpeech dev-clean. This provides context for
comparing CTC path performance.

Usage:
    # Quick benchmark (100 samples)
    python -m tools.whisper_mlx.benchmark_whisper_baseline --num-samples 100

    # Full benchmark (all dev-clean samples)
    python -m tools.whisper_mlx.benchmark_whisper_baseline --dataset dev-clean

    # Output to JSON
    python -m tools.whisper_mlx.benchmark_whisper_baseline --output reports/benchmark_whisper.json

Outputs:
    - WER: Word Error Rate (lower is better)
    - Latency: Time from audio to text (ms)
    - RTF: Real-Time Factor (< 1.0 = faster than real-time)
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# =============================================================================
# WER Computation (shared with benchmark_ctc_path.py)
# =============================================================================

def levenshtein_distance(ref: list[str], hyp: list[str]) -> tuple[int, int, int, int]:
    """
    Compute Levenshtein distance with substitutions, insertions, deletions.

    Returns:
        (distance, substitutions, insertions, deletions)
    """
    m, n = len(ref), len(hyp)

    # dp[i][j] = (cost, subs, ins, dels)
    dp = [[(0, 0, 0, 0) for _ in range(n + 1)] for _ in range(m + 1)]

    # Initialize first row (insertions)
    for j in range(1, n + 1):
        dp[0][j] = (j, 0, j, 0)

    # Initialize first column (deletions)
    for i in range(1, m + 1):
        dp[i][0] = (i, 0, 0, i)

    # Fill dp table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                # Substitution
                sub_cost = dp[i - 1][j - 1][0] + 1
                sub_ops = (sub_cost, dp[i - 1][j - 1][1] + 1, dp[i - 1][j - 1][2], dp[i - 1][j - 1][3])

                # Insertion
                ins_cost = dp[i][j - 1][0] + 1
                ins_ops = (ins_cost, dp[i][j - 1][1], dp[i][j - 1][2] + 1, dp[i][j - 1][3])

                # Deletion
                del_cost = dp[i - 1][j][0] + 1
                del_ops = (del_cost, dp[i - 1][j][1], dp[i - 1][j][2], dp[i - 1][j][3] + 1)

                dp[i][j] = min(sub_ops, ins_ops, del_ops, key=lambda x: x[0])

    return dp[m][n]


def compute_wer(reference: str, hypothesis: str) -> tuple[float, int, int, int, int]:
    """
    Compute Word Error Rate.

    Returns:
        (wer, substitutions, insertions, deletions, ref_length)
    """
    # Normalize text
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    if len(ref_words) == 0:
        if len(hyp_words) == 0:
            return 0.0, 0, 0, 0, 0
        return 1.0, 0, len(hyp_words), 0, 0

    distance, subs, ins, dels = levenshtein_distance(ref_words, hyp_words)
    wer = distance / len(ref_words)

    return wer, subs, ins, dels, len(ref_words)


# =============================================================================
# Data Loading (shared with benchmark_ctc_path.py)
# =============================================================================

@dataclass
class AudioSample:
    """A single audio sample with transcript."""
    audio_path: str
    transcript: str
    speaker_id: str
    chapter_id: str
    utterance_id: str


def load_librispeech_samples(
    data_dir: str,
    max_samples: int = 0,
) -> list[AudioSample]:
    """
    Load LibriSpeech samples from directory.

    Args:
        data_dir: Path to LibriSpeech split (e.g., data/LibriSpeech/dev-clean)
        max_samples: Maximum samples to load (0 = all)

    Returns:
        List of AudioSample objects
    """
    samples = []
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    # Walk through speaker/chapter directories
    for speaker_dir in sorted(data_path.iterdir()):
        if not speaker_dir.is_dir():
            continue

        speaker_id = speaker_dir.name

        for chapter_dir in sorted(speaker_dir.iterdir()):
            if not chapter_dir.is_dir():
                continue

            chapter_id = chapter_dir.name

            # Find transcript file
            transcript_file = chapter_dir / f"{speaker_id}-{chapter_id}.trans.txt"
            if not transcript_file.exists():
                continue

            # Parse transcripts
            with open(transcript_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split(" ", 1)
                    if len(parts) != 2:
                        continue

                    utterance_id, transcript = parts

                    # Find audio file
                    audio_path = chapter_dir / f"{utterance_id}.flac"
                    if not audio_path.exists():
                        continue

                    samples.append(AudioSample(
                        audio_path=str(audio_path),
                        transcript=transcript,
                        speaker_id=speaker_id,
                        chapter_id=chapter_id,
                        utterance_id=utterance_id,
                    ))

                    if max_samples > 0 and len(samples) >= max_samples:
                        return samples

    return samples


# =============================================================================
# Benchmark Results
# =============================================================================

@dataclass
class BenchmarkResult:
    """Results from a single sample benchmark."""
    sample_id: str
    audio_duration_s: float

    # WER metrics
    wer: float
    substitutions: int
    insertions: int
    deletions: int
    ref_words: int
    hyp_words: int

    # Latency metrics (ms)
    total_latency_ms: float

    # Whisper specific
    beam_size: int
    language_detected: str
    no_speech_prob: float

    # Timing
    rtf: float  # Real-Time Factor


@dataclass
class AggregatedResults:
    """Aggregated benchmark results."""

    # Dataset info
    dataset: str
    model: str
    num_samples: int
    total_audio_duration_s: float

    # WER summary
    total_wer: float
    total_substitutions: int
    total_insertions: int
    total_deletions: int
    total_ref_words: int
    total_hyp_words: int

    # Latency summary (ms)
    mean_total_latency_ms: float

    # Latency percentiles (ms)
    p50_total_latency_ms: float
    p95_total_latency_ms: float
    p99_total_latency_ms: float

    # RTF summary
    mean_rtf: float
    median_rtf: float

    # Throughput
    samples_per_second: float
    audio_hours_per_hour: float  # How many hours of audio processed per wall-clock hour

    # Whisper specific
    beam_size: int
    avg_no_speech_prob: float

    # Per-sample results (optional)
    per_sample: list[dict] = field(default_factory=list)


# =============================================================================
# Benchmark Runner
# =============================================================================

def run_benchmark(
    samples: list[AudioSample],
    whisper_model,
    beam_size: int = 5,
    verbose: bool = False,
) -> tuple[AggregatedResults, list[BenchmarkResult]]:
    """
    Run benchmark on samples using Whisper.

    Bypasses VAD to avoid Silero initialization delays. Uses direct
    encoder + beam search decoder approach.

    Args:
        samples: List of audio samples
        whisper_model: Loaded WhisperMLX model
        beam_size: Beam size for decoding
        verbose: Print per-sample results

    Returns:
        Aggregated results and per-sample results
    """
    import mlx.core as mx
    import soundfile as sf
    from scipy import signal

    from .audio import log_mel_spectrogram
    from .beam_search import decode_with_beam_search
    from .tokenizer import get_whisper_tokenizer

    results = []
    start_time = time.time()

    # Get tokenizer
    tokenizer = get_whisper_tokenizer(multilingual=True, task="transcribe", language="en")

    # Warmup run
    if samples:
        print("Warming up...")
        audio, sr = sf.read(samples[0].audio_path)
        audio = audio.astype(np.float32)
        if sr != 16000:
            audio = signal.resample(audio, int(len(audio) * 16000 / sr))

        # Compute mel and pad to 30s (Whisper expects 30s)
        mel = log_mel_spectrogram(audio, n_mels=whisper_model.config.n_mels)

        # Pad to 3000 frames (30 seconds at 100Hz mel rate)
        target_frames = 3000
        if mel.shape[0] < target_frames:
            mel = np.pad(mel, ((0, target_frames - mel.shape[0]), (0, 0)), mode='constant')
        else:
            mel = mel[:target_frames]

        mel_tensor = mx.expand_dims(mx.array(mel), axis=0)
        encoder_out = whisper_model.encoder(mel_tensor)
        mx.eval(encoder_out)
        tokens, segments, avg_logprob, no_speech_prob = decode_with_beam_search(
            whisper_model,
            encoder_out,
            tokenizer,
            beam_size=beam_size,
        )
        print("Warmup complete\n")

    for i, sample in enumerate(samples):
        try:
            # Load audio
            audio, sr = sf.read(sample.audio_path)
            audio = audio.astype(np.float32)
            if sr != 16000:
                audio = signal.resample(audio, int(len(audio) * 16000 / sr))
                sr = 16000

            audio_duration_s = len(audio) / sr

            # Transcribe using direct encoding + beam search (bypass VAD)
            t0 = time.time()

            # Compute mel spectrogram
            mel = log_mel_spectrogram(audio, n_mels=whisper_model.config.n_mels)

            # Pad to 3000 frames (30 seconds at 100Hz mel rate)
            target_frames = 3000
            if mel.shape[0] < target_frames:
                mel = np.pad(mel, ((0, target_frames - mel.shape[0]), (0, 0)), mode='constant')
            else:
                mel = mel[:target_frames]

            mel_tensor = mx.expand_dims(mx.array(mel), axis=0)

            # Encode
            encoder_out = whisper_model.encoder(mel_tensor)
            mx.eval(encoder_out)

            # Decode using beam search
            tokens, segments, avg_logprob, no_speech_prob = decode_with_beam_search(
                whisper_model,
                encoder_out,
                tokenizer,
                beam_size=beam_size,
            )
            total_latency_ms = (time.time() - t0) * 1000

            hypothesis = tokenizer.decode(tokens).strip()
            language_detected = "en"

            # Compute WER
            wer, subs, ins, dels, ref_len = compute_wer(sample.transcript, hypothesis)

            rtf = (total_latency_ms / 1000) / audio_duration_s

            bench_result = BenchmarkResult(
                sample_id=sample.utterance_id,
                audio_duration_s=audio_duration_s,
                wer=wer,
                substitutions=subs,
                insertions=ins,
                deletions=dels,
                ref_words=ref_len,
                hyp_words=len(hypothesis.split()),
                total_latency_ms=total_latency_ms,
                beam_size=beam_size,
                language_detected=language_detected,
                no_speech_prob=no_speech_prob if not isinstance(no_speech_prob, float) or not np.isnan(no_speech_prob) else 0.0,
                rtf=rtf,
            )
            results.append(bench_result)

            if verbose:
                print(f"[{i+1}/{len(samples)}] {sample.utterance_id}")
                print(f"  Duration: {audio_duration_s:.2f}s")
                print(f"  WER: {wer:.2%} (S:{subs} I:{ins} D:{dels})")
                print(f"  Latency: {total_latency_ms:.1f}ms (RTF: {rtf:.3f})")
                print(f"  Ref: {sample.transcript[:60]}...")
                print(f"  Hyp: {hypothesis[:60]}...")
                print()

            # Progress
            if not verbose and (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(samples)} samples...")

        except Exception as e:
            print(f"Error processing {sample.utterance_id}: {e}")
            continue

    # Aggregate results
    total_time = time.time() - start_time

    if not results:
        raise ValueError("No successful benchmark results")

    # WER aggregation (weighted by ref words)
    total_errors = sum(r.substitutions + r.insertions + r.deletions for r in results)
    total_ref_words = sum(r.ref_words for r in results)
    total_wer = total_errors / total_ref_words if total_ref_words > 0 else 0.0

    # Latency aggregation
    total_latencies = [r.total_latency_ms for r in results]

    total_audio = sum(r.audio_duration_s for r in results)

    aggregated = AggregatedResults(
        dataset="librispeech",
        model="whisper-large-v3",
        num_samples=len(results),
        total_audio_duration_s=total_audio,

        total_wer=total_wer,
        total_substitutions=sum(r.substitutions for r in results),
        total_insertions=sum(r.insertions for r in results),
        total_deletions=sum(r.deletions for r in results),
        total_ref_words=total_ref_words,
        total_hyp_words=sum(r.hyp_words for r in results),

        mean_total_latency_ms=np.mean(total_latencies),

        p50_total_latency_ms=np.percentile(total_latencies, 50),
        p95_total_latency_ms=np.percentile(total_latencies, 95),
        p99_total_latency_ms=np.percentile(total_latencies, 99),

        mean_rtf=np.mean([r.rtf for r in results]),
        median_rtf=np.median([r.rtf for r in results]),

        samples_per_second=len(results) / total_time,
        audio_hours_per_hour=(total_audio / 3600) / (total_time / 3600),

        beam_size=beam_size,
        avg_no_speech_prob=np.mean([r.no_speech_prob for r in results]),

        per_sample=[asdict(r) for r in results],
    )

    return aggregated, results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 9 Benchmark: Whisper Baseline Performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="dev-clean",
        help="LibriSpeech split to benchmark (default: dev-clean)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/LibriSpeech",
        help="LibriSpeech root directory",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=0,
        help="Number of samples to benchmark (0 = all)",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size for decoding (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-sample results",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 9 Benchmark: Whisper Baseline Performance")
    print("=" * 70)
    print()

    # Load models
    print("Loading models...")

    from .model import WhisperMLX

    print("  Loading Whisper large-v3...")
    whisper_model = WhisperMLX.from_pretrained("large-v3")

    print()

    # Load samples
    dataset_path = Path(args.data_dir) / args.dataset
    print(f"Loading samples from {dataset_path}...")
    samples = load_librispeech_samples(
        str(dataset_path),
        max_samples=args.num_samples,
    )
    print(f"  Loaded {len(samples)} samples")
    print()

    # Run benchmark
    print(f"Running benchmark (beam_size={args.beam_size})...")
    print()

    aggregated, per_sample = run_benchmark(
        samples=samples,
        whisper_model=whisper_model,
        beam_size=args.beam_size,
        verbose=args.verbose,
    )

    # Print results
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()

    print(f"Dataset: {args.dataset}")
    print(f"Model: whisper-large-v3 (beam_size={args.beam_size})")
    print(f"Samples: {aggregated.num_samples}")
    print(f"Total audio: {aggregated.total_audio_duration_s / 60:.1f} minutes")
    print()

    print("WER Metrics:")
    print(f"  Word Error Rate: {aggregated.total_wer:.2%}")
    print(f"  Substitutions: {aggregated.total_substitutions}")
    print(f"  Insertions: {aggregated.total_insertions}")
    print(f"  Deletions: {aggregated.total_deletions}")
    print(f"  Reference words: {aggregated.total_ref_words}")
    print()

    print("Latency Metrics:")
    print(f"  Total (mean): {aggregated.mean_total_latency_ms:.1f}ms")
    print(f"  Total (p50): {aggregated.p50_total_latency_ms:.1f}ms")
    print(f"  Total (p95): {aggregated.p95_total_latency_ms:.1f}ms")
    print(f"  Total (p99): {aggregated.p99_total_latency_ms:.1f}ms")
    print()

    print("Throughput:")
    print(f"  Real-Time Factor (mean): {aggregated.mean_rtf:.3f}")
    print(f"  Real-Time Factor (median): {aggregated.median_rtf:.3f}")
    print(f"  Samples/second: {aggregated.samples_per_second:.2f}")
    print(f"  Audio hours/hour: {aggregated.audio_hours_per_hour:.1f}x")
    print()

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Remove per-sample from main output to keep file small
        output_data = asdict(aggregated)
        output_data["per_sample"] = []  # Clear for summary file

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to {output_path}")

        # Save per-sample results separately
        per_sample_path = output_path.with_suffix(".per_sample.json")
        with open(per_sample_path, "w") as f:
            json.dump(aggregated.per_sample, f, indent=2)
        print(f"Per-sample results saved to {per_sample_path}")


if __name__ == "__main__":
    main()
