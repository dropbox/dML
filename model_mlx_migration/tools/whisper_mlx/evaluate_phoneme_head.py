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
Kokoro Phoneme Head Evaluation Suite.

Evaluates trained phoneme head checkpoints on LibriSpeech test sets:
- PER (Phoneme Error Rate) on test-clean and test-other
- Discrimination rate (correct vs wrong transcripts)
- Cohen's d effect size for verification signal quality
- Per-speaker analysis for robustness

Usage:
    # Quick evaluation on dev-clean
    python -m tools.whisper_mlx.evaluate_phoneme_head \
        --checkpoint checkpoints/kokoro_phoneme_head_v1/kokoro_phoneme_head_300_best.npz \
        --data-dir data/LibriSpeech/dev-clean \
        --num-samples 100

    # Full evaluation on test-clean
    python -m tools.whisper_mlx.evaluate_phoneme_head \
        --checkpoint checkpoints/kokoro_phoneme_head_v1/kokoro_phoneme_head_300_best.npz \
        --data-dir data/LibriSpeech/test-clean

    # Discrimination test
    python -m tools.whisper_mlx.evaluate_phoneme_head \
        --checkpoint checkpoints/kokoro_phoneme_head_v1/kokoro_phoneme_head_300_best.npz \
        --data-dir data/LibriSpeech/test-clean \
        --discrimination
"""

import argparse
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None

from .audio import load_audio, log_mel_spectrogram
from .kokoro_phoneme_head import KokoroPhonemeHead

# =============================================================================
# Configuration
# =============================================================================


@dataclass
class EvalConfig:
    """Configuration for phoneme head evaluation."""

    # Required
    checkpoint: str = ""
    data_dir: str = "data/LibriSpeech/test-clean"

    # Model
    whisper_model: str = "large-v3"

    # Evaluation options
    num_samples: int = 0  # 0 = all samples
    discrimination: bool = False  # Run discrimination test
    per_speaker: bool = False  # Per-speaker breakdown

    # Discrimination test settings
    num_wrong_transcripts: int = 5  # Wrong transcripts to compare per sample

    # Caching
    encoder_cache_dir: str | None = "/tmp/phoneme_eval_cache"


@dataclass
class EvalResult:
    """Results from phoneme head evaluation."""

    # PER metrics
    total_phoneme_errors: int = 0
    total_phoneme_length: int = 0
    per: float = 0.0  # Phoneme Error Rate

    # Sample counts
    num_samples: int = 0
    num_errors: int = 0  # Samples with errors

    # Timing
    total_time: float = 0.0
    avg_latency_ms: float = 0.0

    # Per-speaker breakdown (if enabled)
    per_speaker: dict[str, float] = field(default_factory=dict)


@dataclass
class DiscriminationResult:
    """Results from discrimination test."""

    # Discrimination metrics
    correct_count: int = 0
    total_count: int = 0
    accuracy: float = 0.0

    # Confidence distributions
    correct_similarities: list[float] = field(default_factory=list)
    wrong_similarities: list[float] = field(default_factory=list)

    # Effect size
    cohens_d: float = 0.0

    # Breakdown
    num_samples: int = 0


# =============================================================================
# Dataset Loading
# =============================================================================


@dataclass
class AudioSample:
    """Single audio sample for evaluation."""
    audio_path: str
    transcript: str
    speaker_id: str = ""


def load_librispeech_samples(data_dir: str, num_samples: int = 0) -> list[AudioSample]:
    """
    Load LibriSpeech samples from directory.

    Args:
        data_dir: Path to LibriSpeech split (e.g., test-clean)
        num_samples: Max samples to load (0 = all)

    Returns:
        List of AudioSample
    """
    data_path = Path(data_dir)
    samples = []

    # Find all transcript files
    transcript_files = sorted(data_path.rglob("*.trans.txt"))

    for trans_file in transcript_files:
        # Parse speaker/chapter from path: speaker/chapter/speaker-chapter.trans.txt
        parts = trans_file.parent.relative_to(data_path).parts
        if len(parts) >= 2:
            speaker_id = parts[0]
        else:
            speaker_id = "unknown"

        # Read transcripts
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
                        transcript=transcript.lower(),  # Normalize to lowercase
                        speaker_id=speaker_id,
                    ))

    # Shuffle and limit
    random.seed(42)  # Reproducible
    random.shuffle(samples)

    if num_samples > 0:
        samples = samples[:num_samples]

    return samples


def get_wrong_transcripts(
    all_transcripts: list[str],
    correct_transcript: str,
    num_wrong: int = 5,
) -> list[str]:
    """
    Get wrong transcripts for discrimination test.

    Selects transcripts that are different from the correct one,
    preferring similar-length transcripts for harder comparison.

    Args:
        all_transcripts: Pool of all transcripts
        correct_transcript: The correct transcript to exclude
        num_wrong: Number of wrong transcripts to return

    Returns:
        List of wrong transcripts
    """
    correct_len = len(correct_transcript)

    # Filter out the correct transcript
    candidates = [t for t in all_transcripts if t != correct_transcript]

    # Sort by length similarity
    candidates.sort(key=lambda t: abs(len(t) - correct_len))

    # Take from front (similar length) with some randomness
    if len(candidates) > num_wrong * 3:
        # Take from similar-length pool
        pool = candidates[:num_wrong * 3]
        return random.sample(pool, min(num_wrong, len(pool)))
    return candidates[:num_wrong]


# =============================================================================
# Phonemizer Interface
# =============================================================================


def phonemize_text(text: str, language: str = "en") -> tuple[str, list[int]]:
    """
    Phonemize text using Kokoro phonemizer.

    Args:
        text: Input text
        language: Language code

    Returns:
        Tuple of (phoneme string, token IDs)
    """
    try:
        from tools.pytorch_to_mlx.converters.models.kokoro_phonemizer import (
            phonemize_text as _phonemize,
        )
        return _phonemize(text, language=language)
    except Exception as e:
        print(f"Phonemization error: {e}")
        return "", []


# =============================================================================
# Evaluation Functions
# =============================================================================


def compute_per(predicted: list[int], reference: list[int]) -> tuple[int, int]:
    """
    Compute Phoneme Error Rate components.

    Args:
        predicted: Predicted phoneme token IDs
        reference: Reference phoneme token IDs

    Returns:
        Tuple of (edit_distance, reference_length)
    """
    # Levenshtein distance
    m, n = len(predicted), len(reference)
    if n == 0:
        return m, 1  # Avoid division by zero

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if predicted[i - 1] == reference[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # Deletion
                    dp[i][j - 1],      # Insertion
                    dp[i - 1][j - 1],  # Substitution
                )

    return dp[m][n], n


def cohens_d(group1: list[float], group2: list[float]) -> float:
    """
    Compute Cohen's d effect size.

    Args:
        group1: First group values
        group2: Second group values

    Returns:
        Cohen's d (positive = group1 > group2)
    """
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0

    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std < 1e-10:
        return 0.0

    return (mean1 - mean2) / pooled_std


def evaluate_per(
    head: KokoroPhonemeHead,
    whisper,
    samples: list[AudioSample],
    config: EvalConfig,
) -> EvalResult:
    """
    Evaluate Phoneme Error Rate on samples.

    Args:
        head: Trained phoneme head
        whisper: Whisper model (for encoder)
        samples: Samples to evaluate
        config: Evaluation config

    Returns:
        EvalResult with PER metrics
    """
    result = EvalResult()
    start_time = time.time()

    # Per-speaker tracking
    speaker_errors: dict[str, tuple[int, int]] = {}  # speaker -> (errors, length)

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

            # Predict phonemes
            predicted_tokens = head.predict(encoder_out)

            # Get reference phonemes from transcript
            _, reference_tokens = phonemize_text(sample.transcript)

            if not reference_tokens:
                continue

            # Compute PER
            edit_dist, ref_len = compute_per(predicted_tokens, reference_tokens)

            result.total_phoneme_errors += edit_dist
            result.total_phoneme_length += ref_len
            result.num_samples += 1

            if edit_dist > 0:
                result.num_errors += 1

            # Per-speaker tracking
            if config.per_speaker:
                speaker = sample.speaker_id
                if speaker not in speaker_errors:
                    speaker_errors[speaker] = (0, 0)
                curr = speaker_errors[speaker]
                speaker_errors[speaker] = (curr[0] + edit_dist, curr[1] + ref_len)

            # Progress
            if (i + 1) % 50 == 0:
                curr_per = result.total_phoneme_errors / max(result.total_phoneme_length, 1)
                print(f"  Evaluated {i + 1}/{len(samples)}: PER={curr_per:.1%}")

        except Exception as e:
            print(f"  Error on sample {i}: {e}")
            continue

    result.total_time = time.time() - start_time
    result.per = result.total_phoneme_errors / max(result.total_phoneme_length, 1)

    if result.num_samples > 0:
        result.avg_latency_ms = (result.total_time / result.num_samples) * 1000

    # Per-speaker PER
    if config.per_speaker:
        for speaker, (errors, length) in speaker_errors.items():
            result.per_speaker[speaker] = errors / max(length, 1)

    return result


def evaluate_discrimination(
    head: KokoroPhonemeHead,
    whisper,
    samples: list[AudioSample],
    config: EvalConfig,
) -> DiscriminationResult:
    """
    Evaluate discrimination ability: can the model distinguish correct vs wrong transcripts?

    Args:
        head: Trained phoneme head
        whisper: Whisper model (for encoder)
        samples: Samples to evaluate
        config: Evaluation config

    Returns:
        DiscriminationResult with discrimination metrics
    """
    result = DiscriminationResult()

    # Collect all transcripts for wrong transcript pool
    all_transcripts = [s.transcript for s in samples]

    for i, sample in enumerate(samples):
        try:
            # Load and process audio
            audio = load_audio(sample.audio_path, 16000)
            mel = log_mel_spectrogram(audio)
            mel = mel[None, :]
            mx.eval(mel)

            # Get encoder output
            encoder_out = whisper.encoder(mel, variable_length=True)
            mx.eval(encoder_out)

            # Compare with correct transcript
            correct_sim, _ = head.compare_with_text(encoder_out, sample.transcript)
            result.correct_similarities.append(correct_sim)

            # Compare with wrong transcripts
            wrong_transcripts = get_wrong_transcripts(
                all_transcripts,
                sample.transcript,
                config.num_wrong_transcripts,
            )

            for wrong_transcript in wrong_transcripts:
                wrong_sim, _ = head.compare_with_text(encoder_out, wrong_transcript)
                result.wrong_similarities.append(wrong_sim)

                result.total_count += 1
                if correct_sim > wrong_sim:
                    result.correct_count += 1

            result.num_samples += 1

            # Progress
            if (i + 1) % 50 == 0:
                curr_acc = result.correct_count / max(result.total_count, 1)
                print(f"  Evaluated {i + 1}/{len(samples)}: Discrimination={curr_acc:.1%}")

        except Exception as e:
            print(f"  Error on sample {i}: {e}")
            continue

    # Final metrics
    result.accuracy = result.correct_count / max(result.total_count, 1)
    result.cohens_d = cohens_d(result.correct_similarities, result.wrong_similarities)

    return result


# =============================================================================
# Main Evaluation
# =============================================================================


def run_evaluation(config: EvalConfig) -> tuple[EvalResult | None, DiscriminationResult | None]:
    """
    Run full evaluation.

    Args:
        config: Evaluation configuration

    Returns:
        Tuple of (EvalResult, DiscriminationResult)
    """
    if not HAS_MLX:
        print("ERROR: MLX not available")
        return None, None

    print(f"Loading checkpoint: {config.checkpoint}")
    head = KokoroPhonemeHead.from_pretrained(config.checkpoint)
    print(f"  Head: d_model={head.d_model}, vocab={head.phoneme_vocab}")

    print(f"Loading Whisper encoder: {config.whisper_model}")
    from .model import WhisperMLX
    whisper = WhisperMLX.from_pretrained(config.whisper_model)
    print("  Encoder loaded")

    print(f"\nLoading samples from: {config.data_dir}")
    samples = load_librispeech_samples(config.data_dir, config.num_samples)
    print(f"  Loaded {len(samples)} samples")

    if len(samples) == 0:
        print("ERROR: No samples found")
        return None, None

    per_result = None
    disc_result = None

    # PER evaluation
    if not config.discrimination:
        print("\n=== Phoneme Error Rate Evaluation ===")
        per_result = evaluate_per(head, whisper, samples, config)

        print("\nPER Results:")
        print(f"  Samples: {per_result.num_samples}")
        print(f"  Total errors: {per_result.total_phoneme_errors}")
        print(f"  Total length: {per_result.total_phoneme_length}")
        print(f"  PER: {per_result.per:.1%}")
        print(f"  Avg latency: {per_result.avg_latency_ms:.1f}ms")

        if config.per_speaker and per_result.per_speaker:
            print("\nPer-Speaker PER:")
            for speaker, per in sorted(per_result.per_speaker.items(), key=lambda x: x[1]):
                print(f"  {speaker}: {per:.1%}")

    # Discrimination evaluation
    if config.discrimination:
        print("\n=== Discrimination Test ===")
        disc_result = evaluate_discrimination(head, whisper, samples, config)

        print("\nDiscrimination Results:")
        print(f"  Samples: {disc_result.num_samples}")
        print(f"  Comparisons: {disc_result.total_count}")
        print(f"  Correct: {disc_result.correct_count}")
        print(f"  Accuracy: {disc_result.accuracy:.1%}")
        print(f"  Cohen's d: {disc_result.cohens_d:.2f}")

        # Distribution summary
        if disc_result.correct_similarities:
            print(f"\n  Correct similarity: mean={np.mean(disc_result.correct_similarities):.3f}, "
                  f"std={np.std(disc_result.correct_similarities):.3f}")
        if disc_result.wrong_similarities:
            print(f"  Wrong similarity: mean={np.mean(disc_result.wrong_similarities):.3f}, "
                  f"std={np.std(disc_result.wrong_similarities):.3f}")

    return per_result, disc_result


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Kokoro Phoneme Head",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to phoneme head checkpoint (.npz)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/LibriSpeech/test-clean",
        help="Path to LibriSpeech split",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=0,
        help="Max samples to evaluate (0=all)",
    )
    parser.add_argument(
        "--discrimination",
        action="store_true",
        help="Run discrimination test instead of PER",
    )
    parser.add_argument(
        "--per-speaker",
        action="store_true",
        help="Show per-speaker breakdown",
    )
    parser.add_argument(
        "--num-wrong",
        type=int,
        default=5,
        help="Wrong transcripts per sample for discrimination test",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="large-v3",
        help="Whisper model size",
    )

    args = parser.parse_args()

    config = EvalConfig(
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        num_samples=args.num_samples,
        discrimination=args.discrimination,
        per_speaker=args.per_speaker,
        num_wrong_transcripts=args.num_wrong,
        whisper_model=args.whisper_model,
    )

    run_evaluation(config)


if __name__ == "__main__":
    main()
