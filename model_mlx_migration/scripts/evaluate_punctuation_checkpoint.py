#!/usr/bin/env python3
"""
Evaluate punctuation head checkpoint on MELD test set.

Usage:
    python scripts/evaluate_punctuation_checkpoint.py \
        --checkpoint checkpoints/punct_meld_full/step_500.npz \
        --meld-dir data/emotion_punctuation/MELD.Raw \
        --max-samples 100
"""

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.whisper_mlx.audio import load_audio, log_mel_spectrogram
from tools.whisper_mlx.multi_head import MultiHeadConfig, PunctuationHead
from tools.whisper_mlx.train_punctuation import load_meld_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--meld-dir", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=100)
    args = parser.parse_args()

    print("=" * 60, flush=True)
    print("Punctuation Checkpoint Evaluation", flush=True)
    print("=" * 60, flush=True)

    # Load Whisper encoder
    print("Loading Whisper encoder...", flush=True)
    start = time.time()
    from mlx_whisper import load_models
    whisper = load_models.load_model("mlx-community/whisper-large-v3-mlx")
    encoder = whisper.encoder
    print(f"Encoder loaded in {time.time() - start:.1f}s", flush=True)

    # Load punctuation head
    print(f"Loading checkpoint: {args.checkpoint}", flush=True)
    config = MultiHeadConfig(
        d_model=1280,
        num_punctuation_classes=6,
        punctuation_hidden_dim=256,
        punctuation_use_emotion=False,
        punctuation_use_pitch=False,
    )
    head = PunctuationHead(config)
    head.load_weights(args.checkpoint)
    print("Punctuation head loaded", flush=True)

    # Load test samples
    print(f"Loading MELD test samples from {args.meld_dir}...", flush=True)
    meld_dir = Path(args.meld_dir)
    test_samples = load_meld_samples(meld_dir, split="test", max_samples=args.max_samples)
    print(f"Loaded {len(test_samples)} test samples", flush=True)

    # Labels
    LABELS = ["PERIOD", "COMMA", "QUESTION", "EXCLAIM", "ELLIPSIS", "NONE"]

    # Evaluate
    print("\nRunning evaluation...", flush=True)
    predictions = []
    ground_truths = []
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)

    for i, sample in enumerate(test_samples):
        if (i + 1) % 10 == 0:
            print(f"  Processing {i + 1}/{len(test_samples)}...", flush=True)

        # Load and process audio
        audio = load_audio(str(sample.audio_path))
        mel = log_mel_spectrogram(audio, n_mels=128)
        mel = mx.array(mel)

        # Pad to 3000 frames
        encoder_frames_total = 3000
        if mel.shape[0] < encoder_frames_total:
            pad_amount = encoder_frames_total - mel.shape[0]
            mel = mx.pad(mel, [(0, pad_amount), (0, 0)])
        else:
            mel = mel[:encoder_frames_total]

        mel = mel[None, :, :]  # Add batch dim

        # Get encoder output
        encoder_out = encoder(mel)

        # Get punctuation prediction
        logits = head(encoder_out, None, None)
        mx.eval(logits)

        # Mean pool and get prediction
        logits_mean = mx.mean(logits, axis=1)
        pred = int(mx.argmax(logits_mean, axis=-1))

        # Get ground truth from sample
        # The text should end with punctuation
        text = sample.text.strip()
        if text.endswith("?"):
            gt = 2  # QUESTION
        elif text.endswith("!"):
            gt = 3  # EXCLAMATION
        elif text.endswith("..."):
            gt = 4  # ELLIPSIS
        elif text.endswith("."):
            gt = 0  # PERIOD
        elif text.endswith(","):
            gt = 1  # COMMA
        else:
            gt = 5  # NONE

        predictions.append(pred)
        ground_truths.append(gt)

        per_class_total[gt] += 1
        if pred == gt:
            per_class_correct[gt] += 1

    # Calculate metrics
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)

    overall_accuracy = np.mean(predictions == ground_truths)

    print("\n" + "=" * 60, flush=True)
    print("RESULTS", flush=True)
    print("=" * 60, flush=True)
    print(f"Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy * 100:.1f}%)", flush=True)

    print("\nPer-class Recall:", flush=True)
    for i, label in enumerate(LABELS):
        total = per_class_total[i]
        correct = per_class_correct[i]
        recall = correct / total if total > 0 else 0
        print(f"  {label:10s}: {recall:.4f} ({correct}/{total})", flush=True)

    # Special focus on QUESTION recall
    question_total = per_class_total[2]
    question_correct = per_class_correct[2]
    question_recall = question_correct / question_total if question_total > 0 else 0
    print(f"\n** Question Recall: {question_recall:.4f} ({question_correct}/{question_total})", flush=True)

    print("\nConfusion Matrix (rows=GT, cols=Pred):", flush=True)
    confusion = np.zeros((6, 6), dtype=int)
    for gt, pred in zip(ground_truths, predictions):
        confusion[gt, pred] += 1
    print("         " + " ".join([f"{l:8s}" for l in LABELS]), flush=True)
    for i, label in enumerate(LABELS):
        row = " ".join([f"{confusion[i, j]:8d}" for j in range(6)])
        print(f"{label:8s} {row}", flush=True)


if __name__ == "__main__":
    main()
