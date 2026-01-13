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
Benchmark Fast Embedding-Space Verification

Tests whether comparing Whisper encoder output with Kokoro text encoder output
can discriminate between correct and incorrect transcripts.

Key metrics:
1. Correlation between confidence and transcript correctness
2. Speed: verification time should be <200ms
3. Comparison with TTS-based verification (accuracy vs speed tradeoff)

Usage:
    python scripts/benchmark_fast_verification.py --max-samples 50
"""

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class FastVerificationSample:
    """Single benchmark sample."""
    audio_path: str
    reference: str
    hypothesis: str
    confidence: float
    wer: float
    is_correct: bool
    verification_time_ms: float


@dataclass
class FastVerificationResults:
    """Aggregated benchmark results."""
    n_samples: int
    # Correlation metrics
    pearson_r: float
    spearman_r: float
    # Classification metrics
    auc_roc: float
    optimal_threshold: float
    optimal_f1: float
    # Speed metrics
    mean_verification_ms: float
    min_verification_ms: float
    max_verification_ms: float
    # Comparison
    baseline_accuracy: float  # Always correct
    threshold_accuracy: float  # With optimal threshold


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate."""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0

    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))
    for i in range(len(ref_words) + 1):
        d[i, 0] = i
    for j in range(len(hyp_words) + 1):
        d[0, j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i, j] = d[i - 1, j - 1]
            else:
                d[i, j] = min(
                    d[i - 1, j] + 1,
                    d[i, j - 1] + 1,
                    d[i - 1, j - 1] + 1,
                )

    return d[len(ref_words), len(hyp_words)] / len(ref_words)


def load_librispeech_samples(split: str = "test-clean", max_samples: int = 50) -> List[Dict]:
    """Load LibriSpeech samples."""
    base_dir = Path("/Users/ayates/model_mlx_migration/data/benchmarks/librispeech/LibriSpeech")
    split_dir = base_dir / split

    if not split_dir.exists():
        raise FileNotFoundError(f"LibriSpeech {split} not found at {split_dir}")

    samples = []
    for trans_file in split_dir.rglob("*.trans.txt"):
        with open(trans_file) as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    audio_id, transcript = parts
                    audio_path = trans_file.parent / f"{audio_id}.flac"
                    if audio_path.exists():
                        samples.append({
                            "audio_path": str(audio_path),
                            "reference": transcript.lower(),
                        })
                        if len(samples) >= max_samples:
                            return samples
    return samples


def load_audio(path: str, sample_rate: int = 16000) -> np.ndarray:
    """Load and resample audio."""
    import soundfile as sf

    audio, sr = sf.read(path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    if sr != sample_rate:
        from scipy import signal
        audio = signal.resample(audio, int(len(audio) * sample_rate / sr))

    return audio.astype(np.float32)


def run_benchmark(
    max_samples: int = 50,
    inject_errors: bool = True,
    error_rate: float = 0.3,
) -> FastVerificationResults:
    """
    Run the fast verification benchmark.

    Args:
        max_samples: Maximum samples to test
        inject_errors: Whether to inject errors in some transcripts
        error_rate: Fraction of samples to inject errors
    """
    import mlx.core as mx
    from tools.whisper_mlx import WhisperMLX
    from tools.whisper_mlx.fast_verification import FastVerifier

    print("=" * 60)
    print("Fast Embedding-Space Verification Benchmark")
    print("=" * 60)

    # Load models
    print("\nLoading Whisper large-v3...")
    t0 = time.perf_counter()
    whisper = WhisperMLX.from_pretrained("large-v3")
    print(f"  Loaded in {(time.perf_counter() - t0) * 1000:.0f}ms")

    print("Loading Kokoro (text encoder only)...")
    t0 = time.perf_counter()
    verifier = FastVerifier.from_kokoro()
    print(f"  Loaded in {(time.perf_counter() - t0) * 1000:.0f}ms")

    # Warmup
    print("\nWarmup...")
    from tools.whisper_mlx.audio import log_mel_spectrogram
    dummy_audio = np.random.randn(16000).astype(np.float32)
    mel = log_mel_spectrogram(dummy_audio)
    if isinstance(mel, np.ndarray):
        mel = mx.array(mel)
    # Use variable_length=True for non-30s audio
    encoder_out = whisper.encoder(mel[None, :, :], variable_length=True)
    mx.eval(encoder_out)
    _ = verifier.compute_confidence(encoder_out, "warmup")
    print("  Done")

    # Load samples
    print("\nLoading LibriSpeech samples...")
    raw_samples = load_librispeech_samples("test-clean", max_samples)
    print(f"  Loaded {len(raw_samples)} samples")

    # Process
    samples = []
    print("\nProcessing...")

    for i, sample in enumerate(raw_samples):
        print(f"\r  {i + 1}/{len(raw_samples)}...", end="", flush=True)

        try:
            # Load audio
            audio = load_audio(sample["audio_path"])

            # Get Whisper encoder output
            mel = log_mel_spectrogram(audio)
            if isinstance(mel, np.ndarray):
                mel = mx.array(mel)
            encoder_out = whisper.encoder(mel[None, :, :], variable_length=True)
            mx.eval(encoder_out)

            # Get hypothesis (use reference or inject errors)
            hypothesis = sample["reference"]
            if inject_errors and np.random.rand() < error_rate:
                words = hypothesis.split()
                if len(words) > 1:
                    idx = np.random.randint(0, len(words) - 1)
                    words[idx], words[idx + 1] = words[idx + 1], words[idx]
                    hypothesis = " ".join(words)

            # Compute WER
            wer = compute_wer(sample["reference"], hypothesis)

            # Fast verification
            result = verifier.compute_confidence(encoder_out, hypothesis)

            samples.append(FastVerificationSample(
                audio_path=sample["audio_path"],
                reference=sample["reference"],
                hypothesis=hypothesis,
                confidence=result.confidence,
                wer=wer,
                is_correct=(wer == 0),
                verification_time_ms=result.verification_time_ms,
            ))

        except Exception as e:
            print(f"\n  Error on sample {i}: {e}")
            continue

    print(f"\n  Processed {len(samples)} samples")

    # Compute metrics
    print("\nComputing metrics...")
    from scipy import stats
    from sklearn.metrics import roc_auc_score, precision_recall_curve

    confidences = np.array([s.confidence for s in samples])
    wers = np.array([s.wer for s in samples])
    is_correct = np.array([s.is_correct for s in samples])
    times = np.array([s.verification_time_ms for s in samples])

    # Correlation
    pearson_r, _ = stats.pearsonr(confidences, 1 - wers)
    spearman_r, _ = stats.spearmanr(confidences, 1 - wers)

    # AUC-ROC
    try:
        auc_roc = roc_auc_score(is_correct, confidences)
    except ValueError:
        auc_roc = 0.5

    # Optimal threshold
    precision, recall, thresholds = precision_recall_curve(is_correct, confidences)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    optimal_idx = np.argmax(f1_scores[:-1])
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    optimal_f1 = f1_scores[optimal_idx]

    # Accuracy metrics
    baseline_accuracy = np.mean(is_correct)  # If we always commit
    threshold_predictions = confidences >= optimal_threshold
    threshold_accuracy = np.mean(threshold_predictions == is_correct)

    results = FastVerificationResults(
        n_samples=len(samples),
        pearson_r=float(pearson_r),
        spearman_r=float(spearman_r),
        auc_roc=float(auc_roc),
        optimal_threshold=float(optimal_threshold),
        optimal_f1=float(optimal_f1),
        mean_verification_ms=float(np.mean(times)),
        min_verification_ms=float(np.min(times)),
        max_verification_ms=float(np.max(times)),
        baseline_accuracy=float(baseline_accuracy),
        threshold_accuracy=float(threshold_accuracy),
    )

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Samples: {results.n_samples}")
    print()
    print("Correlation Metrics:")
    print(f"  Pearson r(confidence, 1-WER):  {results.pearson_r:.4f}")
    print(f"  Spearman r:                    {results.spearman_r:.4f}")
    print()
    print("Classification Metrics:")
    print(f"  AUC-ROC:                       {results.auc_roc:.4f}")
    print(f"  Optimal threshold:             {results.optimal_threshold:.4f}")
    print(f"  Optimal F1:                    {results.optimal_f1:.4f}")
    print()
    print("Speed Metrics:")
    print(f"  Mean verification time:        {results.mean_verification_ms:.1f}ms")
    print(f"  Min:                           {results.min_verification_ms:.1f}ms")
    print(f"  Max:                           {results.max_verification_ms:.1f}ms")
    print()
    print("Accuracy:")
    print(f"  Baseline (always commit):      {results.baseline_accuracy:.2%}")
    print(f"  With threshold:                {results.threshold_accuracy:.2%}")
    print("=" * 60)

    # Interpretation
    print("\nINTERPRETATION:")
    if results.mean_verification_ms < 200:
        print(f"  SPEED: {results.mean_verification_ms:.0f}ms meets <200ms target")
    else:
        print(f"  SPEED: {results.mean_verification_ms:.0f}ms exceeds 200ms target")

    if results.pearson_r > 0.3:
        print("  CORRELATION: Strong positive - higher confidence = lower WER")
    elif results.pearson_r > 0.1:
        print("  CORRELATION: Weak positive - some signal present")
    else:
        print("  CORRELATION: No signal - confidence doesn't predict correctness")

    if results.auc_roc > 0.7:
        print("  CLASSIFIER: Good - can distinguish correct vs wrong")
    elif results.auc_roc > 0.55:
        print("  CLASSIFIER: Weak - slightly better than random")
    else:
        print("  CLASSIFIER: Poor - not useful for decisions")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark Fast Verification")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--no-errors", action="store_true")
    parser.add_argument("--output", type=str, help="Save results to JSON")

    args = parser.parse_args()

    results = run_benchmark(
        max_samples=args.max_samples,
        inject_errors=not args.no_errors,
    )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(asdict(results), f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
