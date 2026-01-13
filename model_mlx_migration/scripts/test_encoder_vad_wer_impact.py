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
Test E2E WER Impact of Encoder VAD Integration.

Compares transcription quality with and without encoder VAD:
1. Baseline: WhisperMLX without encoder VAD
2. With VAD: WhisperMLX with encoder VAD head

Expected result: WER should be within 0.05% of baseline (unchanged).

Usage:
    python scripts/test_encoder_vad_wer_impact.py --num-samples 100
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import time
from typing import Dict, List, Tuple

import jiwer
import numpy as np


def load_librispeech_samples(num_samples: int = 100) -> List[Tuple[np.ndarray, str, str]]:
    """Load LibriSpeech test-clean samples with ground truth."""
    from datasets import load_dataset

    print(f"Loading {num_samples} LibriSpeech test-clean samples...")
    ds = load_dataset("librispeech_asr", "clean", split="test")

    samples = []
    for i, sample in enumerate(ds):
        if i >= num_samples:
            break
        audio = sample["audio"]["array"].astype(np.float32)
        text = sample["text"].lower()  # Ground truth
        sample_id = f"{sample['chapter_id']}-{sample['id']}"
        samples.append((audio, text, sample_id))

    print(f"  Loaded {len(samples)} samples")
    return samples


def transcribe_samples(
    model,
    samples: List[Tuple[np.ndarray, str, str]],
    verbose: bool = False,
) -> Tuple[List[str], float]:
    """
    Transcribe all samples and return predictions with timing.

    Returns:
        Tuple of (predictions, total_time_seconds)
    """
    predictions = []
    total_time = 0.0

    for i, (audio, _, sample_id) in enumerate(samples):
        t0 = time.perf_counter()
        result = model.transcribe(audio, language="en", verbose=False)
        elapsed = time.perf_counter() - t0
        total_time += elapsed

        pred_text = result.get("text", "").lower().strip()
        predictions.append(pred_text)

        if verbose and (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{len(samples)}")

    return predictions, total_time


def compute_wer(references: List[str], predictions: List[str]) -> Dict[str, float]:
    """Compute WER and related metrics."""
    # Normalize text
    transform = jiwer.Compose([
        jiwer.RemoveWhiteSpace(replace_by_space=True),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ToLowerCase(),
    ])

    refs = [transform(r) for r in references]
    preds = [transform(p) for p in predictions]

    # Compute WER
    wer = jiwer.wer(refs, preds)

    # Compute exact match
    exact_matches = sum(1 for r, p in zip(refs, preds) if r == p)
    exact_match_rate = exact_matches / len(refs)

    return {
        "wer": wer,
        "exact_match_rate": exact_match_rate,
        "exact_matches": exact_matches,
        "total_samples": len(refs),
    }


def main():
    parser = argparse.ArgumentParser(description="Test encoder VAD WER impact")
    parser.add_argument("--num-samples", type=int, default=100,
                        help="Number of LibriSpeech test samples")
    parser.add_argument("--verbose", action="store_true",
                        help="Print progress")
    args = parser.parse_args()

    print("=" * 70)
    print("Encoder VAD E2E WER Impact Test")
    print("=" * 70)

    # Load test samples
    samples = load_librispeech_samples(args.num_samples)
    references = [text for _, text, _ in samples]

    # Calculate total audio duration
    total_audio_seconds = sum(len(audio) / 16000 for audio, _, _ in samples)
    print(f"Total audio duration: {total_audio_seconds:.1f}s ({total_audio_seconds/60:.1f}m)")

    # Load model
    print("\n[1/4] Loading WhisperMLX model...")
    from tools.whisper_mlx.model import WhisperMLX
    model = WhisperMLX.from_pretrained("large-v3", warmup=True)
    print("  Model loaded.")

    # Test 1: Baseline (without encoder VAD)
    print("\n[2/4] Running baseline transcription (no encoder VAD)...")
    time.perf_counter()
    baseline_preds, baseline_time = transcribe_samples(model, samples, args.verbose)
    baseline_metrics = compute_wer(references, baseline_preds)
    print(f"  Baseline WER: {baseline_metrics['wer']:.2%}")
    print(f"  Baseline time: {baseline_time:.2f}s (RTF: {baseline_time/total_audio_seconds:.2f}x)")

    # Load encoder VAD head
    print("\n[3/4] Loading encoder VAD head...")
    vad_weights_path = "checkpoints/encoder_vad/encoder_vad_best.npz"
    model.load_encoder_vad_head(vad_weights_path, threshold=0.15)
    print("  VAD head loaded (threshold=0.15)")
    print(f"  Encoder VAD enabled: {model.encoder_vad_enabled}")

    # Test 2: With encoder VAD
    print("\n[4/4] Running transcription with encoder VAD...")
    vad_preds, vad_time = transcribe_samples(model, samples, args.verbose)
    vad_metrics = compute_wer(references, vad_preds)
    print(f"  VAD WER: {vad_metrics['wer']:.2%}")
    print(f"  VAD time: {vad_time:.2f}s (RTF: {vad_time/total_audio_seconds:.2f}x)")

    # Compare results
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)

    wer_diff = vad_metrics['wer'] - baseline_metrics['wer']
    speedup = baseline_time / vad_time if vad_time > 0 else float('inf')

    print(f"\n{'Metric':<25} {'Baseline':>15} {'With VAD':>15} {'Diff':>12}")
    print("-" * 70)
    print(f"{'WER':<25} {baseline_metrics['wer']:>14.2%} {vad_metrics['wer']:>14.2%} {wer_diff:>+11.2%}")
    print(f"{'Exact Match':<25} {baseline_metrics['exact_match_rate']:>14.1%} {vad_metrics['exact_match_rate']:>14.1%}")
    print(f"{'Time (s)':<25} {baseline_time:>14.2f} {vad_time:>14.2f} {speedup:>10.2f}x")
    print(f"{'RTF':<25} {baseline_time/total_audio_seconds:>14.2f} {vad_time/total_audio_seconds:>14.2f}")

    # Check pass/fail
    print("\n" + "-" * 70)
    MAX_WER_DEGRADATION = 0.0005  # 0.05% max degradation allowed

    if wer_diff <= MAX_WER_DEGRADATION:
        print(f"PASS: WER change ({wer_diff:+.3%}) within tolerance (±{MAX_WER_DEGRADATION:.2%})")
        return_code = 0
    else:
        print(f"FAIL: WER increased by {wer_diff:.3%} (max allowed: {MAX_WER_DEGRADATION:.2%})")
        return_code = 1

    # Print summary for commit message
    print("\n" + "=" * 70)
    print("SUMMARY (for commit message)")
    print("=" * 70)
    print(f"LibriSpeech test-clean ({args.num_samples} samples)")
    print(f"  Baseline WER: {baseline_metrics['wer']:.2%}")
    print(f"  With Encoder VAD (threshold=0.15): {vad_metrics['wer']:.2%}")
    print(f"  WER Change: {wer_diff:+.3%}")
    print(f"  Speedup: {speedup:.2f}x")

    return return_code


if __name__ == "__main__":
    sys.exit(main())
