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
Validate Encoder VAD Head accuracy against Silero VAD.

This script comprehensively tests the encoder VAD head by:
1. Comparing frame-level predictions with Silero VAD labels
2. Computing precision, recall, F1 at various thresholds
3. Testing on audio files with natural speech/silence patterns
4. Measuring inference overhead

Target: >95% accuracy vs Silero (per MANAGER directive)
"""

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import mlx.core as mx


def get_silero_vad_labels(audio: np.ndarray, n_encoder_positions: int = 1500) -> np.ndarray:
    """
    Get per-position VAD labels from Silero VAD.

    Returns binary labels aligned to encoder positions.
    """
    import torch

    # Load Silero VAD with utils
    vad_model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        trust_repo=True,
        verbose=False
    )

    # Get the get_speech_timestamps function from utils
    get_speech_timestamps = utils[0]

    # Convert to torch tensor
    audio_tensor = torch.from_numpy(audio).float()

    # Get speech timestamps using utility function
    speech_timestamps = get_speech_timestamps(
        audio_tensor,
        vad_model,
        sampling_rate=16000,
        threshold=0.5,
        min_speech_duration_ms=100,
        min_silence_duration_ms=100,
        return_seconds=True,
    )

    # Convert to frame-level labels
    duration_s = len(audio) / 16000
    frame_duration = duration_s / n_encoder_positions

    labels = np.zeros(n_encoder_positions, dtype=np.float32)

    for segment in speech_timestamps:
        start_frame = int(segment['start'] / frame_duration)
        end_frame = int(segment['end'] / frame_duration)
        labels[start_frame:min(end_frame + 1, n_encoder_positions)] = 1.0

    return labels


def evaluate_vad_accuracy(
    vad_probs: np.ndarray,
    silero_labels: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Compute accuracy metrics at given threshold.
    """
    preds = (vad_probs >= threshold).astype(np.float32)

    tp = np.sum((preds == 1) & (silero_labels == 1))
    fp = np.sum((preds == 1) & (silero_labels == 0))
    fn = np.sum((preds == 0) & (silero_labels == 1))
    tn = np.sum((preds == 0) & (silero_labels == 0))

    precision = tp / max(tp + fp, 1e-6)
    recall = tp / max(tp + fn, 1e-6)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    accuracy = (tp + tn) / len(preds)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn),
    }


def generate_test_audio_with_pauses(
    duration_s: float = 10.0,
    speech_prob: float = 0.6,
    min_segment_s: float = 0.5,
    max_segment_s: float = 2.0,
) -> tuple:
    """
    Generate synthetic audio with random speech/silence segments.

    Returns (audio, speech_mask) where speech_mask is binary at 16kHz.
    """
    sample_rate = 16000
    n_samples = int(duration_s * sample_rate)

    audio = np.zeros(n_samples, dtype=np.float32)
    mask = np.zeros(n_samples, dtype=np.float32)

    pos = 0
    while pos < n_samples:
        # Random segment length
        seg_len = int(np.random.uniform(min_segment_s, max_segment_s) * sample_rate)
        seg_len = min(seg_len, n_samples - pos)

        # Randomly decide speech or silence
        is_speech = np.random.random() < speech_prob

        if is_speech:
            # Generate speech-like signal (modulated noise)
            freq = np.random.uniform(100, 400)  # Pitch-like
            t = np.arange(seg_len) / sample_rate
            carrier = np.sin(2 * np.pi * freq * t)
            envelope = np.exp(-t * 2) * 0.5 + 0.5  # Attack/decay
            audio[pos:pos + seg_len] = carrier * envelope * 0.8
            mask[pos:pos + seg_len] = 1.0
        else:
            # Silence (small noise)
            audio[pos:pos + seg_len] = np.random.randn(seg_len) * 0.001

        pos += seg_len

    return audio, mask


def run_validation(
    audio_files: list,
    model,
    vad_head,
    n_mels: int = 128,
    verbose: bool = True,
):
    """
    Run validation on a list of audio files.

    Returns aggregate metrics.
    """
    from tools.whisper_mlx.audio import log_mel_spectrogram

    all_metrics = []
    total_vad_time = 0.0
    total_audio_duration = 0.0

    for audio_path in audio_files:
        if verbose:
            print(f"\nProcessing: {audio_path}")

        # Load audio
        if isinstance(audio_path, tuple):
            audio, mask = audio_path
            audio_name = "synthetic"
        else:
            audio, sr = sf.read(str(audio_path))
            audio_name = Path(audio_path).name

            # Resample if needed
            if sr != 16000:
                ratio = 16000 / sr
                new_len = int(len(audio) * ratio)
                audio = np.interp(
                    np.linspace(0, len(audio), new_len),
                    np.arange(len(audio)),
                    audio
                )

            audio = audio.astype(np.float32)

        duration_s = len(audio) / 16000
        total_audio_duration += duration_s

        # Compute mel spectrogram
        mel = log_mel_spectrogram(audio, n_mels=n_mels)

        # Pad to 30s
        target_len = 3000
        if mel.shape[0] < target_len:
            mel = mx.pad(mel, [(0, target_len - mel.shape[0]), (0, 0)])
        elif mel.shape[0] > target_len:
            mel = mel[:target_len, :]

        # Encode
        mel_batch = mel[None].astype(mx.float32)
        encoder_output = model.embed_audio(mel_batch, variable_length=False)
        mx.eval(encoder_output)

        # Run VAD head (timed)
        t0 = time.perf_counter()
        probs = vad_head(encoder_output, training=False)
        mx.eval(probs)
        t1 = time.perf_counter()
        total_vad_time += (t1 - t0)

        probs_np = np.array(probs[0])

        # Get actual frame count
        n_actual_frames = min(int(duration_s / 30.0 * 1500), 1500)
        actual_probs = probs_np[:n_actual_frames]

        # Get Silero labels for comparison
        silero_labels = get_silero_vad_labels(audio, n_actual_frames)

        # Evaluate at multiple thresholds
        for thresh in [0.3, 0.5, 0.7]:
            metrics = evaluate_vad_accuracy(actual_probs, silero_labels, thresh)
            metrics['threshold'] = thresh
            metrics['audio'] = audio_name
            metrics['duration_s'] = duration_s
            all_metrics.append(metrics)

        if verbose:
            best = evaluate_vad_accuracy(actual_probs, silero_labels, 0.5)
            print(f"  Duration: {duration_s:.2f}s")
            print(f"  Encoder VAD speech: {(actual_probs >= 0.5).mean():.1%}")
            print(f"  Silero VAD speech: {silero_labels.mean():.1%}")
            print(f"  @ 0.5 threshold: P={best['precision']:.3f}, R={best['recall']:.3f}, F1={best['f1']:.3f}")

    # Aggregate metrics at 0.5 threshold
    metrics_05 = [m for m in all_metrics if m['threshold'] == 0.5]

    total_tp = sum(m['tp'] for m in metrics_05)
    total_fp = sum(m['fp'] for m in metrics_05)
    total_fn = sum(m['fn'] for m in metrics_05)
    total_tn = sum(m['tn'] for m in metrics_05)

    agg_precision = total_tp / max(total_tp + total_fp, 1e-6)
    agg_recall = total_tp / max(total_tp + total_fn, 1e-6)
    agg_f1 = 2 * agg_precision * agg_recall / max(agg_precision + agg_recall, 1e-6)
    agg_accuracy = (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn)

    return {
        'precision': agg_precision,
        'recall': agg_recall,
        'f1': agg_f1,
        'accuracy': agg_accuracy,
        'vad_time_ms': total_vad_time * 1000,
        'audio_duration_s': total_audio_duration,
        'vad_overhead_pct': (total_vad_time / total_audio_duration) * 100 if total_audio_duration > 0 else 0,
        'all_metrics': all_metrics,
    }


def main():
    print("=" * 70)
    print("Encoder VAD vs Silero Validation")
    print("=" * 70)
    print("\nTarget: >95% accuracy (per MANAGER directive)")

    # Load model and VAD head
    print("\n[1/4] Loading models...")
    from tools.whisper_mlx.model import WhisperMLX
    from tools.whisper_mlx.encoder_vad import load_encoder_vad_head

    model = WhisperMLX.from_pretrained("large-v3", warmup=True)

    vad_head = load_encoder_vad_head(
        "checkpoints/encoder_vad/encoder_vad_best.npz",
        n_state=model.config.n_audio_state,
        hidden_dim=256,
        dtype=mx.float32,
    )

    print("  Models loaded.")

    # Find test audio files
    print("\n[2/4] Finding test audio files...")
    audio_dirs = [
        Path("tests/fixtures/audio"),
        Path("tests/prosody/prosody_audio_output"),
    ]

    audio_files = []
    for d in audio_dirs:
        if d.exists():
            audio_files.extend(list(d.glob("*.wav")))

    print(f"  Found {len(audio_files)} audio files")

    # Also generate synthetic test cases
    print("\n[3/4] Generating synthetic test audio...")
    synthetic_tests = []
    for i in range(5):
        audio, mask = generate_test_audio_with_pauses(
            duration_s=10.0,
            speech_prob=0.3 + i * 0.15,  # 30% to 90% speech
        )
        synthetic_tests.append((audio, mask))
    print(f"  Generated {len(synthetic_tests)} synthetic samples")

    # Run validation
    print("\n[4/4] Running validation...")

    all_tests = audio_files + synthetic_tests

    results = run_validation(
        all_tests,
        model,
        vad_head,
        n_mels=model.config.n_mels,
        verbose=True,
    )

    # Report results
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)

    print("\nAccuracy vs Silero VAD @ threshold=0.5:")
    print(f"  Precision: {results['precision']:.1%}")
    print(f"  Recall:    {results['recall']:.1%}")
    print(f"  F1 Score:  {results['f1']:.1%}")
    print(f"  Accuracy:  {results['accuracy']:.1%}")

    print("\nVAD Head Overhead:")
    print(f"  Total VAD time: {results['vad_time_ms']:.2f}ms")
    print(f"  Total audio:    {results['audio_duration_s']:.2f}s")
    print(f"  Overhead:       {results['vad_overhead_pct']:.4f}%")

    # Check against target
    print("\n" + "-" * 70)
    target_accuracy = 0.95
    if results['accuracy'] >= target_accuracy:
        print(f"✓ PASS: Accuracy {results['accuracy']:.1%} >= {target_accuracy:.0%} target")
        return 0
    else:
        print(f"✗ FAIL: Accuracy {results['accuracy']:.1%} < {target_accuracy:.0%} target")
        print(f"  Gap: {(target_accuracy - results['accuracy']) * 100:.1f}pp")
        return 1


if __name__ == "__main__":
    sys.exit(main())
