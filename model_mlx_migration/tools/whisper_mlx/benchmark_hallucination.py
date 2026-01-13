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
Phase 9 Benchmark: Hallucination Detection Rate.

Measures the phoneme verification system's ability to detect Whisper hallucinations:
1. Run Whisper transcription on LibriSpeech audio
2. Use phoneme head to compare audio phonemes with Whisper output
3. Compare with ground truth to identify actual hallucinations
4. Compute detection rate and false positive rate

Key Metrics:
- Detection Rate: % of hallucinations (WER > threshold) correctly flagged
- False Positive Rate: % of correct outputs incorrectly flagged
- Precision: % of flagged outputs that are actual hallucinations
- Recall: % of hallucinations that were flagged

Usage:
    # Quick benchmark (50 samples)
    python -m tools.whisper_mlx.benchmark_hallucination --num-samples 50

    # Full benchmark on dev-clean
    python -m tools.whisper_mlx.benchmark_hallucination --dataset dev-clean

    # Output to JSON
    python -m tools.whisper_mlx.benchmark_hallucination --output reports/hallucination_benchmark.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None


# =============================================================================
# WER Computation
# =============================================================================


def levenshtein_distance(ref: list[str], hyp: list[str]) -> int:
    """Compute Levenshtein distance between word lists."""
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n]


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate."""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    if len(ref_words) == 0:
        return 1.0 if len(hyp_words) > 0 else 0.0

    distance = levenshtein_distance(ref_words, hyp_words)
    return distance / len(ref_words)


# =============================================================================
# Data Loading
# =============================================================================


@dataclass
class AudioSample:
    """A single audio sample with transcript."""
    audio_path: str
    transcript: str
    speaker_id: str


def load_librispeech_samples(
    data_dir: str,
    max_samples: int = 0,
) -> list[AudioSample]:
    """Load LibriSpeech samples from directory."""
    import random

    data_path = Path(data_dir)
    samples = []

    # Find all transcript files
    transcript_files = sorted(data_path.rglob("*.trans.txt"))

    for trans_file in transcript_files:
        parts = trans_file.parent.relative_to(data_path).parts
        speaker_id = parts[0] if len(parts) >= 2 else "unknown"

        with open(trans_file) as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) < 2:
                    continue

                file_id, transcript = parts
                audio_path = trans_file.parent / f"{file_id}.flac"

                if audio_path.exists():
                    samples.append(AudioSample(
                        audio_path=str(audio_path),
                        transcript=transcript.lower(),
                        speaker_id=speaker_id,
                    ))

    # Shuffle with fixed seed for reproducibility
    random.seed(42)
    random.shuffle(samples)

    if max_samples > 0:
        samples = samples[:max_samples]

    return samples


# =============================================================================
# Results Data Classes
# =============================================================================


@dataclass
class SampleResult:
    """Result for a single sample."""
    audio_path: str
    ground_truth: str
    whisper_hypothesis: str
    wer: float
    phoneme_similarity: float
    is_hallucination: bool  # WER > threshold
    is_flagged: bool  # similarity < threshold


@dataclass
class BenchmarkResult:
    """Aggregated benchmark results."""
    # Sample counts
    total_samples: int = 0
    num_hallucinations: int = 0  # WER > wer_threshold
    num_flagged: int = 0  # similarity < similarity_threshold

    # Detection metrics
    true_positives: int = 0  # Hallucination AND flagged
    false_positives: int = 0  # Correct AND flagged
    true_negatives: int = 0  # Correct AND not flagged
    false_negatives: int = 0  # Hallucination AND not flagged

    # Rates
    detection_rate: float = 0.0  # TP / (TP + FN)
    false_positive_rate: float = 0.0  # FP / (FP + TN)
    precision: float = 0.0  # TP / (TP + FP)
    recall: float = 0.0  # Same as detection_rate

    # WER statistics
    mean_wer: float = 0.0
    wer_threshold: float = 0.0

    # Similarity statistics
    mean_similarity: float = 0.0
    similarity_threshold: float = 0.0
    similarity_correct_mean: float = 0.0
    similarity_hallucination_mean: float = 0.0

    # Timing
    total_time_s: float = 0.0
    avg_latency_ms: float = 0.0

    # Individual results
    samples: list[dict] = field(default_factory=list)


# =============================================================================
# Benchmark Function
# =============================================================================


def run_hallucination_benchmark(
    data_dir: str,
    phoneme_checkpoint: str,
    whisper_model: str = "large-v3",
    num_samples: int = 0,
    wer_threshold: float = 0.3,
    similarity_threshold: float = 0.7,
) -> BenchmarkResult:
    """
    Run hallucination detection benchmark.

    Args:
        data_dir: Path to LibriSpeech split
        phoneme_checkpoint: Path to phoneme head checkpoint
        whisper_model: Whisper model size
        num_samples: Max samples (0 = all)
        wer_threshold: WER above this is considered hallucination
        similarity_threshold: Similarity below this triggers flag

    Returns:
        BenchmarkResult with detection metrics
    """
    if not HAS_MLX:
        raise RuntimeError("MLX not available")

    print(f"Loading Whisper model: {whisper_model}")
    from .model import WhisperMLX
    whisper = WhisperMLX.from_pretrained(whisper_model)
    print("  Model loaded")

    print(f"Loading phoneme head: {phoneme_checkpoint}")
    from .kokoro_phoneme_head import KokoroPhonemeHead
    phoneme_head = KokoroPhonemeHead.from_pretrained(phoneme_checkpoint)
    print(f"  Head loaded: d_model={phoneme_head.d_model}")

    print(f"\nLoading samples from: {data_dir}")
    samples = load_librispeech_samples(data_dir, num_samples)
    print(f"  Loaded {len(samples)} samples")

    if len(samples) == 0:
        print("ERROR: No samples found")
        return BenchmarkResult()

    # Import audio utilities
    from .audio import load_audio, log_mel_spectrogram

    result = BenchmarkResult(
        wer_threshold=wer_threshold,
        similarity_threshold=similarity_threshold,
    )

    start_time = time.time()
    wers = []
    similarities_correct = []
    similarities_hallucination = []

    for i, sample in enumerate(samples):
        try:
            # Load and process audio
            audio = load_audio(sample.audio_path, 16000)
            mel = log_mel_spectrogram(audio)
            mel = mel[None, :]  # Add batch dim
            mx.eval(mel)

            # Get encoder output
            encoder_out = whisper.encoder(mel, variable_length=True)
            mx.eval(encoder_out)

            # Run Whisper transcription
            transcription = whisper.transcribe(audio)
            hypothesis = transcription.get("text", "").lower().strip()

            # Compute WER against ground truth
            wer = compute_wer(sample.transcript, hypothesis)
            wers.append(wer)

            # Compute phoneme similarity between audio and hypothesis
            similarity, _ = phoneme_head.compare_with_text(encoder_out, hypothesis)

            # Classify
            is_hallucination = wer > wer_threshold
            is_flagged = similarity < similarity_threshold

            # Update counts
            result.total_samples += 1
            if is_hallucination:
                result.num_hallucinations += 1
                similarities_hallucination.append(similarity)
            else:
                similarities_correct.append(similarity)

            if is_flagged:
                result.num_flagged += 1

            # Confusion matrix
            if is_hallucination and is_flagged:
                result.true_positives += 1
            elif not is_hallucination and is_flagged:
                result.false_positives += 1
            elif not is_hallucination and not is_flagged:
                result.true_negatives += 1
            else:  # is_hallucination and not is_flagged
                result.false_negatives += 1

            # Store sample result
            result.samples.append({
                "audio_path": sample.audio_path,
                "ground_truth": sample.transcript,
                "hypothesis": hypothesis,
                "wer": wer,
                "similarity": similarity,
                "is_hallucination": is_hallucination,
                "is_flagged": is_flagged,
            })

            # Progress
            if (i + 1) % 10 == 0:
                curr_detection = result.true_positives / max(result.num_hallucinations, 1)
                curr_fpr = result.false_positives / max(result.total_samples - result.num_hallucinations, 1)
                print(f"  [{i + 1}/{len(samples)}] Detection: {curr_detection:.1%}, FPR: {curr_fpr:.1%}")

        except Exception as e:
            print(f"  Error on sample {i}: {e}")
            continue

    result.total_time_s = time.time() - start_time

    # Compute final metrics
    if result.total_samples > 0:
        result.avg_latency_ms = (result.total_time_s / result.total_samples) * 1000
        result.mean_wer = np.mean(wers) if wers else 0.0
        result.mean_similarity = np.mean(similarities_correct + similarities_hallucination) if (similarities_correct or similarities_hallucination) else 0.0

    if similarities_correct:
        result.similarity_correct_mean = np.mean(similarities_correct)
    if similarities_hallucination:
        result.similarity_hallucination_mean = np.mean(similarities_hallucination)

    # Detection rate = TP / (TP + FN)
    if result.num_hallucinations > 0:
        result.detection_rate = result.true_positives / result.num_hallucinations
        result.recall = result.detection_rate

    # False positive rate = FP / (FP + TN)
    num_correct = result.total_samples - result.num_hallucinations
    if num_correct > 0:
        result.false_positive_rate = result.false_positives / num_correct

    # Precision = TP / (TP + FP)
    if result.num_flagged > 0:
        result.precision = result.true_positives / result.num_flagged

    return result


def print_results(result: BenchmarkResult) -> None:
    """Print benchmark results."""
    print("\n" + "=" * 70)
    print("HALLUCINATION DETECTION BENCHMARK RESULTS")
    print("=" * 70)

    print("\nConfiguration:")
    print(f"  WER threshold (hallucination): > {result.wer_threshold:.1%}")
    print(f"  Similarity threshold (flag): < {result.similarity_threshold}")

    print("\nSample Statistics:")
    print(f"  Total samples: {result.total_samples}")
    print(f"  Hallucinations (WER > {result.wer_threshold:.0%}): {result.num_hallucinations} ({100*result.num_hallucinations/max(result.total_samples,1):.1f}%)")
    print(f"  Flagged (similarity < {result.similarity_threshold}): {result.num_flagged} ({100*result.num_flagged/max(result.total_samples,1):.1f}%)")

    print("\nConfusion Matrix:")
    print("                    Actual Hallucination")
    print("                    Yes         No")
    print(f"  Flagged Yes       {result.true_positives:4d} (TP)   {result.false_positives:4d} (FP)")
    print(f"  Flagged No        {result.false_negatives:4d} (FN)   {result.true_negatives:4d} (TN)")

    print("\nDetection Metrics:")
    print(f"  Detection Rate (Recall): {result.detection_rate:.1%}")
    print(f"  False Positive Rate: {result.false_positive_rate:.1%}")
    print(f"  Precision: {result.precision:.1%}")

    print("\nSimilarity Statistics:")
    print(f"  Mean similarity (correct outputs): {result.similarity_correct_mean:.3f}")
    print(f"  Mean similarity (hallucinations): {result.similarity_hallucination_mean:.3f}")
    print(f"  Separation: {result.similarity_correct_mean - result.similarity_hallucination_mean:.3f}")

    print("\nWER Statistics:")
    print(f"  Mean WER: {result.mean_wer:.1%}")

    print("\nTiming:")
    print(f"  Total time: {result.total_time_s:.1f}s")
    print(f"  Avg latency: {result.avg_latency_ms:.0f}ms per sample")

    print("=" * 70)


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark hallucination detection rate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="dev-clean",
        help="LibriSpeech split name (default: dev-clean)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Full path to data directory (overrides --dataset)",
    )
    parser.add_argument(
        "--phoneme-checkpoint",
        type=str,
        default="checkpoints/kokoro_phoneme_head_v1/kokoro_phoneme_head_1400_best.npz",
        help="Path to phoneme head checkpoint",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="large-v3",
        help="Whisper model size",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=0,
        help="Max samples (0 = all)",
    )
    parser.add_argument(
        "--wer-threshold",
        type=float,
        default=0.3,
        help="WER above this is hallucination (default: 0.3)",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.7,
        help="Similarity below this triggers flag (default: 0.7)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path",
    )

    args = parser.parse_args()

    # Determine data directory
    if args.data_dir:
        data_dir = args.data_dir
    else:
        data_dir = f"data/LibriSpeech/{args.dataset}"

    # Run benchmark
    result = run_hallucination_benchmark(
        data_dir=data_dir,
        phoneme_checkpoint=args.phoneme_checkpoint,
        whisper_model=args.whisper_model,
        num_samples=args.num_samples,
        wer_threshold=args.wer_threshold,
        similarity_threshold=args.similarity_threshold,
    )

    # Print results
    print_results(result)

    # Save to JSON if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict for JSON serialization
        result_dict = asdict(result)
        with open(output_path, "w") as f:
            json.dump(result_dict, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
