#!/usr/bin/env python3
"""Diagnose punctuation head predictions to understand why it underperforms."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import numpy as np
from pathlib import Path
from collections import Counter


def diagnose_predictions():
    """Analyze what the punctuation head actually predicts."""

    print("=" * 60)
    print("Punctuation Head Diagnosis")
    print("=" * 60)

    # Load checkpoint and understand weights
    checkpoint_path = Path("checkpoints/punct_meld_full/best.npz")
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    weights = mx.load(str(checkpoint_path))

    print("\n1. Checkpoint weights analysis:")
    for key, value in weights.items():
        print(f"   {key}: {value.shape}")

    # Load model and make predictions on MELD test
    from tools.whisper_mlx import WhisperMLX
    from tools.whisper_mlx.multi_head import MultiHeadConfig, PunctuationHead
    from tools.whisper_mlx.audio import load_audio, compute_stft_and_mel
    import csv

    print("\n2. Loading models...")
    whisper = WhisperMLX.from_pretrained("mlx-community/whisper-large-v3-mlx")

    # Create punctuation head
    head_config = MultiHeadConfig(
        d_model=1280,
        punctuation_hidden_dim=256,
        num_punctuation_classes=6,
        punctuation_use_emotion=False,
        punctuation_use_pitch=False,
        use_layer_norm=True,
        dropout_rate=0.1,
    )
    punct_head = PunctuationHead(head_config)
    punct_head.load_weights(list(weights.items()))
    mx.eval(punct_head.parameters())

    # Class names
    PUNCT_CLASSES = ["PERIOD", "COMMA", "QUESTION", "EXCLAMATION", "ELLIPSIS", "NONE"]

    # Load MELD test samples
    print("\n3. Loading MELD test samples...")
    meld_dir = Path("data/emotion_punctuation/MELD.Raw")
    audio_dir = meld_dir / "audio_test"
    csv_path = meld_dir / "test_sent_emo.csv"

    samples = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dia_id = row["Dialogue_ID"]
            utt_id = row["Utterance_ID"]
            text = row.get("Utterance", "")
            audio_file = audio_dir / f"dia{dia_id}_utt{utt_id}.wav"
            if audio_file.exists():
                samples.append({
                    "audio_path": str(audio_file),
                    "text": text,
                })

    print(f"   Found {len(samples)} test samples")

    # Analyze predictions on subset
    n_samples = min(50, len(samples))
    print(f"\n4. Analyzing predictions on {n_samples} samples...")

    all_predictions = []
    all_ground_truth = []
    per_frame_stats = Counter()
    per_sample_stats = Counter()

    for i, sample in enumerate(samples[:n_samples]):
        # Load and process audio
        audio = load_audio(sample["audio_path"], sample_rate=16000)
        audio_mx = mx.array(audio, dtype=mx.float32)

        # Compute mel spectrogram (returns mel, stft)
        mel, _ = compute_stft_and_mel(audio_mx)
        mel = mel[None, ...]  # Add batch dim [1, T, 128]

        # Get encoder output (variable_length=True for short audio)
        encoder_out = whisper.encoder(mel, variable_length=True)
        mx.eval(encoder_out)

        # Get punctuation predictions
        logits = punct_head(encoder_out)
        probs = mx.softmax(logits, axis=-1)
        predictions = mx.argmax(logits, axis=-1)
        mx.eval(predictions, probs)

        # Convert to numpy for analysis
        preds = np.array(predictions[0])  # [T]
        prob_vals = np.array(probs[0])  # [T, 6]

        # Count per-frame predictions
        for pred in preds:
            per_frame_stats[PUNCT_CLASSES[pred]] += 1

        # Get dominant prediction for this sample
        pred_counts = Counter(preds.tolist())
        dominant = pred_counts.most_common(1)[0][0]
        per_sample_stats[PUNCT_CLASSES[dominant]] += 1

        # Get ground truth from text
        text = sample["text"].strip()
        if text.endswith("?"):
            gt = "QUESTION"
        elif text.endswith("!"):
            gt = "EXCLAMATION"
        elif text.endswith("."):
            gt = "PERIOD"
        elif text.endswith("..."):
            gt = "ELLIPSIS"
        elif text.endswith(","):
            gt = "COMMA"
        else:
            gt = "NONE"

        all_predictions.append(PUNCT_CLASSES[dominant])
        all_ground_truth.append(gt)

        if i < 5:  # Print detailed analysis for first 5
            print(f"\n   Sample {i+1}: '{text[:50]}...'")
            print(f"   Ground truth: {gt}")
            print("   Frame predictions distribution:")
            for cls, count in sorted(pred_counts.items(), key=lambda x: -x[1]):
                pct = 100 * count / len(preds)
                print(f"      {PUNCT_CLASSES[cls]}: {count} ({pct:.1f}%)")
            print("   Mean probabilities:")
            mean_probs = prob_vals.mean(axis=0)
            for j, cls in enumerate(PUNCT_CLASSES):
                print(f"      {cls}: {mean_probs[j]:.3f}")

    # Overall statistics
    print("\n" + "=" * 60)
    print("5. Overall Statistics")
    print("=" * 60)

    total_frames = sum(per_frame_stats.values())
    print("\n   Per-frame prediction distribution:")
    for cls in PUNCT_CLASSES:
        count = per_frame_stats[cls]
        pct = 100 * count / total_frames if total_frames > 0 else 0
        print(f"      {cls}: {count} ({pct:.1f}%)")

    print("\n   Per-sample dominant prediction distribution:")
    for cls in PUNCT_CLASSES:
        count = per_sample_stats[cls]
        pct = 100 * count / n_samples
        print(f"      {cls}: {count} ({pct:.1f}%)")

    print("\n   Ground truth distribution:")
    gt_counts = Counter(all_ground_truth)
    for cls in PUNCT_CLASSES:
        count = gt_counts.get(cls, 0)
        pct = 100 * count / n_samples
        print(f"      {cls}: {count} ({pct:.1f}%)")

    # Confusion matrix
    print("\n   Simple accuracy:")
    correct = sum(1 for p, g in zip(all_predictions, all_ground_truth) if p == g)
    print(f"      {correct}/{n_samples} = {100*correct/n_samples:.1f}%")

    # Per-class recall
    print("\n   Per-class recall:")
    for cls in PUNCT_CLASSES:
        gt_cls = [i for i, g in enumerate(all_ground_truth) if g == cls]
        if gt_cls:
            correct_cls = sum(1 for i in gt_cls if all_predictions[i] == cls)
            recall = correct_cls / len(gt_cls)
            print(f"      {cls}: {correct_cls}/{len(gt_cls)} = {100*recall:.1f}%")

    print("\n" + "=" * 60)
    print("6. ROOT CAUSE ANALYSIS")
    print("=" * 60)

    none_pct = 100 * per_frame_stats["NONE"] / total_frames if total_frames > 0 else 0

    print(f"""
    The punctuation head predicts NONE for {none_pct:.1f}% of frames.

    ROOT CAUSE: Frame-level vs Utterance-level mismatch

    - Training data has per-frame labels where most frames within an
      utterance are labeled NONE (only the last frame has punctuation)
    - Model learns to predict NONE for ~74% of frames (matching training dist)
    - When we aggregate to per-utterance predictions, NONE dominates

    SOLUTIONS:

    1. **Use Whisper's native punctuation** (RECOMMENDED)
       - Whisper already produces punctuation in its output text
       - Baseline achieves 0.53 macro F1 vs 0.19 for trained head
       - No additional training or inference cost

    2. **Utterance-level classification instead of frame-level**
       - Pool encoder outputs to single vector per utterance
       - Train a simple classifier on pooled representation
       - This matches the annotation granularity

    3. **CTC-style punctuation prediction**
       - Similar to CTC for ASR, allow blank predictions
       - Only predict punctuation at word boundaries
       - Requires word-aligned training data
    """)


if __name__ == "__main__":
    diagnose_predictions()
