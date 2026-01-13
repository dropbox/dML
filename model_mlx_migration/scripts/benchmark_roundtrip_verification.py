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
Rigorous Benchmark for Round-Trip Spectrogram Verification

For academic publication, this script evaluates:
1. Correlation between round-trip confidence and transcript correctness
2. ROC curves for commit/don't-commit decisions
3. Comparison with baselines (always commit, confidence thresholds)

Benchmarks used:
- LibriSpeech test-clean (standard ASR benchmark)
- LibriSpeech test-other (harder, noisy)

Metrics reported:
- Pearson correlation: confidence vs. 1-WER
- AUC-ROC: confidence as classifier for "correct transcript"
- Precision@90% recall: threshold that catches 90% of correct transcripts
- Retraction rate reduction vs. baseline

Usage:
    python scripts/benchmark_roundtrip_verification.py --dataset librispeech-test-clean --max-samples 100
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

import numpy as np

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class BenchmarkSample:
    """Single benchmark sample."""
    audio_path: str
    reference: str                    # Ground truth transcript
    hypothesis: str                   # STT hypothesis (may be wrong)
    roundtrip_confidence: float       # Round-trip verification score
    wer: float                        # Word Error Rate
    is_correct: bool                  # WER == 0
    audio_duration: float             # Duration in seconds


@dataclass
class BenchmarkResults:
    """Aggregated benchmark results."""
    dataset: str
    n_samples: int
    # Correlation metrics
    pearson_r: float                  # Correlation(confidence, 1-WER)
    spearman_r: float                 # Rank correlation
    # Classification metrics (is_correct ~ confidence)
    auc_roc: float                    # Area under ROC curve
    precision_at_90_recall: float     # Precision when recall = 90%
    optimal_threshold: float          # Threshold maximizing F1
    optimal_f1: float                 # F1 at optimal threshold
    # Retraction metrics
    baseline_retraction_rate: float   # Always-commit retraction rate
    optimal_retraction_rate: float    # Retraction rate at optimal threshold
    retraction_reduction: float       # % reduction in retractions
    # Timing
    avg_verification_time_ms: float   # Average time per verification


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate."""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0

    # Dynamic programming for edit distance
    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))
    for i in range(len(ref_words) + 1):
        d[i, 0] = i
    for j in range(len(hyp_words) + 1):
        d[0, j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i, j] = d[i-1, j-1]
            else:
                d[i, j] = min(
                    d[i-1, j] + 1,      # deletion
                    d[i, j-1] + 1,      # insertion
                    d[i-1, j-1] + 1,    # substitution
                )

    return d[len(ref_words), len(hyp_words)] / len(ref_words)


def load_librispeech_samples(
    split: str = "test-clean",
    max_samples: int = 100,
) -> List[Dict]:
    """Load LibriSpeech samples with transcripts."""
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
    """Load audio file and resample."""
    import soundfile as sf

    audio, sr = sf.read(path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    if sr != sample_rate:
        import scipy.signal
        audio = scipy.signal.resample(audio, int(len(audio) * sample_rate / sr))

    return audio.astype(np.float32)


def transcribe_with_whisper(audio: np.ndarray, model=None) -> str:
    """Transcribe audio with Whisper."""
    if model is None:
        # Import here to avoid loading model if not needed
        from tools.whisper_mlx import WhisperMLX
        model = WhisperMLX.from_pretrained("large-v3")

    result = model.transcribe(audio)
    return result.get("text", "").lower().strip()


def compute_metrics(samples: List[BenchmarkSample]) -> BenchmarkResults:
    """Compute all metrics from benchmark samples."""
    from scipy import stats
    from sklearn.metrics import roc_auc_score, precision_recall_curve

    confidences = np.array([s.roundtrip_confidence for s in samples])
    wers = np.array([s.wer for s in samples])
    is_correct = np.array([s.is_correct for s in samples])

    # Correlation metrics
    pearson_r, _ = stats.pearsonr(confidences, 1 - wers)
    spearman_r, _ = stats.spearmanr(confidences, 1 - wers)

    # Classification metrics
    try:
        auc_roc = roc_auc_score(is_correct, confidences)
    except ValueError:
        auc_roc = 0.5  # All same class

    # Precision at 90% recall
    precision, recall, thresholds = precision_recall_curve(is_correct, confidences)
    idx_90 = np.argmin(np.abs(recall - 0.9))
    precision_at_90 = precision[idx_90] if idx_90 < len(precision) else precision[-1]

    # Optimal threshold (maximize F1)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    optimal_idx = np.argmax(f1_scores[:-1])  # Exclude last (undefined)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    optimal_f1 = f1_scores[optimal_idx]

    # Retraction metrics
    # Baseline: always commit (retract when wrong)
    baseline_retraction = 1 - np.mean(is_correct)

    # Optimal: commit only above threshold
    committed_mask = confidences >= optimal_threshold
    if committed_mask.sum() > 0:
        committed_wrong = np.sum(~is_correct[committed_mask])
        optimal_retraction = committed_wrong / len(samples)
    else:
        optimal_retraction = baseline_retraction

    retraction_reduction = (baseline_retraction - optimal_retraction) / baseline_retraction * 100

    return BenchmarkResults(
        dataset="",  # Set by caller
        n_samples=len(samples),
        pearson_r=float(pearson_r),
        spearman_r=float(spearman_r),
        auc_roc=float(auc_roc),
        precision_at_90_recall=float(precision_at_90),
        optimal_threshold=float(optimal_threshold),
        optimal_f1=float(optimal_f1),
        baseline_retraction_rate=float(baseline_retraction),
        optimal_retraction_rate=float(optimal_retraction),
        retraction_reduction=float(retraction_reduction),
        avg_verification_time_ms=0.0,  # Set by caller
    )


def run_benchmark(
    dataset: str = "test-clean",
    max_samples: int = 100,
    use_whisper_hypothesis: bool = True,
    inject_errors: bool = True,
    error_rate: float = 0.3,
) -> BenchmarkResults:
    """
    Run the full benchmark.

    Args:
        dataset: LibriSpeech split to use
        max_samples: Maximum samples to evaluate
        use_whisper_hypothesis: If True, use actual Whisper transcripts
                                If False, use ground truth (for error injection)
        inject_errors: If True, artificially inject errors in some transcripts
        error_rate: Fraction of samples to inject errors
    """
    from tools.whisper_mlx.roundtrip_verification import RoundTripVerifier

    print(f"\n{'='*60}")
    print("Round-Trip Verification Benchmark")
    print(f"Dataset: {dataset}, Max samples: {max_samples}")
    print(f"{'='*60}\n")

    # Load verifier
    print("Loading Kokoro verifier...")
    try:
        verifier = RoundTripVerifier.from_kokoro()
        print("✓ Kokoro verifier loaded")
    except Exception as e:
        print(f"✗ Kokoro failed: {e}")
        print("Using mock verifier (results will not be meaningful)")
        verifier = RoundTripVerifier.from_mock()

    # Load Whisper if needed
    whisper_model = None
    if use_whisper_hypothesis:
        print("Loading Whisper model...")
        from tools.whisper_mlx import WhisperMLX
        whisper_model = WhisperMLX.from_pretrained("large-v3")
        print("✓ Whisper loaded")

    # Load samples
    print(f"\nLoading {dataset} samples...")
    raw_samples = load_librispeech_samples(dataset, max_samples)
    print(f"✓ Loaded {len(raw_samples)} samples")

    # Process samples
    benchmark_samples = []
    verification_times = []

    for i, sample in enumerate(raw_samples):
        print(f"\rProcessing {i+1}/{len(raw_samples)}...", end="", flush=True)

        try:
            # Load audio
            audio = load_audio(sample["audio_path"])
            duration = len(audio) / 16000

            # Get hypothesis
            if use_whisper_hypothesis:
                hypothesis = transcribe_with_whisper(audio, whisper_model)
            else:
                hypothesis = sample["reference"]

            # Optionally inject errors
            if inject_errors and np.random.rand() < error_rate:
                words = hypothesis.split()
                if len(words) > 1:
                    # Swap two random words
                    idx = np.random.randint(0, len(words) - 1)
                    words[idx], words[idx + 1] = words[idx + 1], words[idx]
                    hypothesis = " ".join(words)

            # Compute WER
            wer = compute_wer(sample["reference"], hypothesis)

            # Compute round-trip confidence
            t0 = time.time()
            result = verifier.compute_confidence(audio, hypothesis)
            verification_times.append((time.time() - t0) * 1000)

            benchmark_samples.append(BenchmarkSample(
                audio_path=sample["audio_path"],
                reference=sample["reference"],
                hypothesis=hypothesis,
                roundtrip_confidence=result.confidence,
                wer=wer,
                is_correct=(wer == 0),
                audio_duration=duration,
            ))

        except Exception as e:
            print(f"\n  Error on sample {i}: {e}")
            continue

    print(f"\n✓ Processed {len(benchmark_samples)} samples")

    # Compute metrics
    print("\nComputing metrics...")
    results = compute_metrics(benchmark_samples)
    results.dataset = dataset
    results.avg_verification_time_ms = np.mean(verification_times)

    # Print results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Dataset: {results.dataset}")
    print(f"Samples: {results.n_samples}")
    print()
    print("Correlation Metrics:")
    print(f"  Pearson r(confidence, 1-WER):  {results.pearson_r:.4f}")
    print(f"  Spearman r:                    {results.spearman_r:.4f}")
    print()
    print("Classification Metrics (is_correct ~ confidence):")
    print(f"  AUC-ROC:                       {results.auc_roc:.4f}")
    print(f"  Precision @ 90% recall:        {results.precision_at_90_recall:.4f}")
    print(f"  Optimal threshold:             {results.optimal_threshold:.4f}")
    print(f"  Optimal F1:                    {results.optimal_f1:.4f}")
    print()
    print("Retraction Metrics:")
    print(f"  Baseline retraction rate:      {results.baseline_retraction_rate:.2%}")
    print(f"  Optimal retraction rate:       {results.optimal_retraction_rate:.2%}")
    print(f"  Retraction reduction:          {results.retraction_reduction:.1f}%")
    print()
    print(f"Avg verification time:           {results.avg_verification_time_ms:.1f} ms")
    print(f"{'='*60}\n")

    # Interpretation
    print("INTERPRETATION:")
    if results.pearson_r > 0.3:
        print("✓ STRONG POSITIVE CORRELATION: Higher confidence = lower WER")
    elif results.pearson_r > 0.1:
        print("~ WEAK POSITIVE CORRELATION: Some signal, needs tuning")
    else:
        print("✗ NO CORRELATION: Round-trip confidence doesn't predict correctness")

    if results.auc_roc > 0.7:
        print("✓ GOOD CLASSIFIER: Can distinguish correct vs wrong transcripts")
    elif results.auc_roc > 0.55:
        print("~ WEAK CLASSIFIER: Slightly better than random")
    else:
        print("✗ POOR CLASSIFIER: Not useful for commit decisions")

    if results.retraction_reduction > 20:
        print("✓ SIGNIFICANT IMPROVEMENT: Can reduce retractions by 20%+")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark Round-Trip Verification")
    parser.add_argument("--dataset", default="test-clean", choices=["test-clean", "test-other"])
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--use-ground-truth", action="store_true",
                        help="Use ground truth as hypothesis (faster, for testing)")
    parser.add_argument("--no-errors", action="store_true",
                        help="Don't inject artificial errors")
    parser.add_argument("--output", type=str, help="Save results to JSON")

    args = parser.parse_args()

    results = run_benchmark(
        dataset=args.dataset,
        max_samples=args.max_samples,
        use_whisper_hypothesis=not args.use_ground_truth,
        inject_errors=not args.no_errors,
    )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(asdict(results), f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
