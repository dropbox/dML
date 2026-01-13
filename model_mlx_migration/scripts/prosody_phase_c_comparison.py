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
A/B Comparison: Phase C (F0 predictor conditioning) vs Phase B (prosody embeddings)

Phase C directly conditions the F0 predictor's AdaIN blocks with prosody embeddings,
bypassing the Instance Normalization bottleneck that limited Phase B.

This script:
1. Loads Kokoro with Phase B prosody embeddings
2. Enables Phase C F0 predictor conditioning
3. Loads trained fc_prosody weights
4. Compares F0 output with different emotion embeddings

Usage:
    python scripts/prosody_phase_c_comparison.py

Output: tests/prosody/phase_c_comparison_output/
"""

import json
import os
import sys
import time
import wave
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx

# Prosody type IDs from prosody_types.h
# These must match the training data!
PROSODY_TYPES = {
    "NEUTRAL": 0,
    "ANGRY": 40,       # EMOTION_ANGRY
    "SAD": 41,         # EMOTION_SAD
    "EXCITED": 42,     # EMOTION_EXCITED
    "CALM": 45,        # EMOTION_CALM
    "FRUSTRATED": 48,  # EMOTION_FRUSTRATED
    "NERVOUS": 49,     # EMOTION_NERVOUS
    "SURPRISED": 50,   # EMOTION_SURPRISED
}

# RAVDESS emotion to prosody type mapping
# RAVDESS codes: neutral(1), calm(2), happy(3), sad(4),
# angry(5), fearful(6), disgust(7), surprised(8)
RAVDESS_TO_PROSODY = {
    "neutral": 0,
    "calm": 24,
    "happy": 27,      # Map to EXCITED
    "sad": 22,
    "angry": 21,
    "fearful": 29,    # Map to NERVOUS
    "disgust": 28,    # Map to FRUSTRATED
    "surprised": 23,
}


def load_kokoro_model():
    """Load Kokoro model."""
    from tools.pytorch_to_mlx.converters import KokoroConverter

    converter = KokoroConverter()
    model, config, _ = converter.load_from_hf()
    return model, converter


def load_prosody_embeddings(path: str) -> mx.array:
    """Load Phase B prosody embedding table."""
    weights = mx.load(str(path))

    if "embedding.weight" in weights:
        return weights["embedding.weight"]
    elif "prosody_embedding.embedding.weight" in weights:
        return weights["prosody_embedding.embedding.weight"]
    else:
        for key, value in weights.items():
            if "embedding" in key.lower() and "weight" in key.lower():
                print(f"Using embedding weights from key: {key}")
                return value

    keys = list(weights.keys())
    raise ValueError(f"No embedding weights found in {path}. Keys: {keys}")


def load_fc_prosody_weights(predictor, path: str) -> None:
    """Load fc_prosody weights into prosody-conditioned blocks."""
    weights = mx.load(str(path))

    # Map saved weights to predictor layers
    # Weight keys are like: "F0_0_prosody.norm1.fc_prosody.weight"
    for key, value in weights.items():
        # Parse key to find target layer
        parts = key.split(".")
        if len(parts) < 4:
            continue

        block_name = parts[0]  # F0_0_prosody, F0_1_prosody, F0_2_prosody

        if not hasattr(predictor, block_name):
            print(f"Warning: Predictor has no attribute {block_name}")
            continue

        block = getattr(predictor, block_name)

        # Navigate to target parameter
        target = block
        for part in parts[1:-1]:
            if hasattr(target, part):
                target = getattr(target, part)
            else:
                print(f"Warning: {block_name} has no attribute {part}")
                target = None
                break

        if target is not None:
            param_name = parts[-1]
            if hasattr(target, param_name):
                setattr(target, param_name, value)
            else:
                print(f"Warning: {'.'.join(parts[:-1])} has no attribute {param_name}")


def text_to_tokens(text: str) -> Tuple[mx.array, int]:
    """Convert text to Kokoro tokens."""
    from tools.pytorch_to_mlx.converters.models.kokoro_phonemizer import phonemize_text

    phonemes, token_ids = phonemize_text(text)
    return mx.array([token_ids]), len(phonemes)


def save_wav(audio: np.ndarray, path: str, sample_rate: int = 24000):
    """Save audio to WAV file."""
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())

    return path


def compute_f0(audio: np.ndarray, sample_rate: int = 24000) -> Tuple[float, float]:
    """Compute mean and std F0 using librosa YIN (fast)."""
    try:
        import librosa
        # Use YIN instead of PYIN for speed (10-100x faster)
        f0 = librosa.yin(
            audio, fmin=50, fmax=500, sr=sample_rate
        )
        # Filter out unvoiced frames (f0 at fmin or fmax boundaries)
        f0_valid = f0[(f0 > 60) & (f0 < 480)]
        if len(f0_valid) > 0:
            return float(np.mean(f0_valid)), float(np.std(f0_valid))
    except Exception as e:
        print(f"F0 computation failed: {e}")
    return 0.0, 0.0


def synthesize_with_phase_c(
    model,
    converter,
    predictor_with_prosody,
    embedding_table: mx.array,
    text: str,
    prosody_type: int = 0,
) -> Tuple[np.ndarray, float, float]:
    """
    Synthesize with Phase C prosody conditioning.

    Returns audio, predicted F0 mean, predicted F0 std.
    """
    input_ids, phoneme_length = text_to_tokens(text)
    voice = converter.load_voice("af_heart", phoneme_length=phoneme_length)
    mx.eval(voice)

    batch_size, seq_length = input_ids.shape

    # Split voice embedding
    if voice.shape[-1] == 256:
        style = voice[:, :128]
        speaker = voice[:, 128:]
    else:
        style = voice
        speaker = voice

    # Get prosody embedding for this type
    if prosody_type < embedding_table.shape[0]:
        prosody_emb = embedding_table[prosody_type:prosody_type+1]  # [1, 768]
    else:
        prosody_emb = embedding_table[0:1]

    # Run BERT encoder
    bert_out = model.bert(input_ids, None)
    bert_enc = model.bert_encoder(bert_out)

    # Run text encoder for duration
    duration_feats = model.predictor.text_encoder(bert_enc, speaker)

    # Duration prediction
    dur_enc = model.predictor.lstm(duration_feats)
    duration_logits = model.predictor.duration_proj(dur_enc)

    # Compute alignment
    indices, total_frames, _ = model._compute_alignment(duration_logits, speed=1.0)

    # Expand features
    en_expanded_640 = model._expand_features(duration_feats, indices, total_frames)

    # Run shared BiLSTM
    x_shared = model.predictor.shared(en_expanded_640)

    # Phase C: F0 prediction with prosody-conditioned blocks
    x = x_shared
    x = predictor_with_prosody.F0_0_prosody(x, speaker, prosody_emb)
    x = predictor_with_prosody.F0_1_prosody(x, speaker, prosody_emb)
    x = predictor_with_prosody.F0_2_prosody(x, speaker, prosody_emb)
    f0 = predictor_with_prosody.F0_proj(x).squeeze(-1)

    # Get F0 statistics (model output is normalized, not Hz)
    mx.eval(f0)
    f0_np = np.array(f0).flatten()
    voiced = f0_np[f0_np > 0.1]
    f0_mean = float(np.mean(voiced)) if np.any(voiced) else float(np.mean(f0_np))
    f0_std = float(np.std(f0_np))

    # Noise prediction (standard path)
    x = x_shared
    x = model.predictor.N_0(x, speaker)
    x = model.predictor.N_1(x, speaker)
    x = model.predictor.N_2(x, speaker)
    noise = model.predictor.N_proj(x).squeeze(-1)

    # ASR features
    text_enc = model.text_encoder(input_ids, None)
    asr_expanded = model._expand_features(text_enc, indices, total_frames)

    # Decode
    audio = model.decoder(asr_expanded, f0, noise, style)
    mx.eval(audio)

    return np.array(audio).flatten(), f0_mean, f0_std


def synthesize_baseline(
    model,
    converter,
    text: str,
) -> Tuple[np.ndarray, float, float]:
    """Synthesize without prosody (baseline)."""
    input_ids, phoneme_length = text_to_tokens(text)
    voice = converter.load_voice("af_heart", phoneme_length=phoneme_length)
    mx.eval(voice)

    # Use standard synthesis
    audio = model.synthesize(input_ids, voice)
    mx.eval(audio)

    return np.array(audio).flatten(), 0.0, 0.0  # F0 stats computed from audio


def run_phase_c_comparison():
    """Run A/B comparison between Phase B and Phase C."""
    output_dir = Path("tests/prosody/phase_c_comparison_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Paths (try v5 first, then v4, v3, v2)
    # v5: Data-driven F0 multipliers from multilingual dataset
    emb_path = "models/prosody_embeddings_orthogonal/final.safetensors"
    prosody_embeddings_path = emb_path

    # Try model versions in order
    fc_versions = [
        "models/prosody_f0_conditioning_v5/fc_prosody_weights.safetensors",
        "models/prosody_f0_conditioning_v4/fc_prosody_weights.safetensors",
        "models/prosody_f0_conditioning_v3/fc_prosody_weights.safetensors",
        "models/prosody_f0_conditioning_v2/fc_prosody_weights.safetensors",
        "models/prosody_f0_conditioning_ravdess/fc_prosody_weights.safetensors",
    ]
    fc_prosody_weights_path = fc_versions[0]
    for path in fc_versions:
        if os.path.exists(path):
            fc_prosody_weights_path = path
            break

    # Check paths
    if not os.path.exists(prosody_embeddings_path):
        print(f"ERROR: Prosody embeddings not found at {prosody_embeddings_path}")
        return
    if not os.path.exists(fc_prosody_weights_path):
        print(f"ERROR: FC prosody weights not found at {fc_prosody_weights_path}")
        return

    # Test sentences
    test_sentences = [
        "Hello, this is a test of the prosody system.",
        "I am very happy to see you today!",
        "This is absolutely terrible news.",
        "Please stay calm, everything will be fine.",
        "I cannot believe this is happening right now!",
    ]

    # Emotions to test (matching Phase B comparison)
    emotions = [
        ("NEUTRAL", PROSODY_TYPES["NEUTRAL"]),
        ("ANGRY", PROSODY_TYPES["ANGRY"]),
        ("SAD", PROSODY_TYPES["SAD"]),
        ("EXCITED", PROSODY_TYPES["EXCITED"]),
        ("CALM", PROSODY_TYPES["CALM"]),
    ]

    results = []

    # Load model
    print("Loading Kokoro model...")
    model, converter = load_kokoro_model()

    # Load Phase B prosody embeddings
    print(f"Loading prosody embeddings from {prosody_embeddings_path}...")
    embedding_table = load_prosody_embeddings(prosody_embeddings_path)
    print(f"Embedding table shape: {embedding_table.shape}")

    # Enable Phase C prosody conditioning on predictor
    print("Enabling Phase C prosody conditioning...")
    model.predictor.enable_prosody_conditioning(prosody_dim=768, prosody_scale=0.1)

    # Copy original F0 weights to prosody blocks
    print("Copying F0 weights to prosody-conditioned blocks...")
    model.predictor.copy_f0_weights_to_prosody_blocks()

    # Load trained fc_prosody weights
    print(f"Loading fc_prosody weights from {fc_prosody_weights_path}...")
    load_fc_prosody_weights(model.predictor, fc_prosody_weights_path)

    print("\nGenerating audio for Phase C A/B comparison...")
    print("=" * 60)

    for sent_idx, sentence in enumerate(test_sentences):
        print(f"\n[Sentence {sent_idx + 1}] {sentence[:50]}...")

        sent_results = {"sentence": sentence, "comparisons": []}

        # Generate baseline (neutral, no prosody)
        baseline_audio, _, _ = synthesize_baseline(model, converter, sentence)
        baseline_path = str(output_dir / f"sent{sent_idx}_baseline.wav")
        save_wav(baseline_audio, baseline_path)
        baseline_f0_audio, baseline_f0_std = compute_f0(baseline_audio)
        baseline_duration = len(baseline_audio) / 24000

        print(f"  Baseline: F0={baseline_f0_audio:.1f}Hz, dur={baseline_duration:.2f}s")

        for emotion_name, emotion_id in emotions:
            # Generate with Phase C prosody conditioning
            prosody_audio, pred_f0_mean, pred_f0_std = synthesize_with_phase_c(
                model, converter, model.predictor, embedding_table, sentence, emotion_id
            )
            fname = f"sent{sent_idx}_{emotion_name.lower()}_phase_c.wav"
            prosody_path = str(output_dir / fname)
            save_wav(prosody_audio, prosody_path)

            # Compute F0 from audio
            prosody_f0_audio, prosody_f0_std = compute_f0(prosody_audio)
            prosody_duration = len(prosody_audio) / 24000

            # Compute differences
            base_f0 = baseline_f0_audio
            base_dur = baseline_duration
            f0_diff = prosody_f0_audio - base_f0 if base_f0 > 0 else 0
            f0_diff_pct = (f0_diff / base_f0 * 100) if base_f0 > 0 else 0
            dur_diff = prosody_duration - base_dur
            dur_diff_pct = (dur_diff / base_dur * 100) if base_dur > 0 else 0

            result = {
                "emotion": emotion_name,
                "prosody_id": emotion_id,
                "baseline_f0": baseline_f0_audio,
                "prosody_f0": prosody_f0_audio,
                "f0_diff_hz": f0_diff,
                "f0_diff_pct": f0_diff_pct,
                "pred_f0_mean": pred_f0_mean,  # Model's internal F0 prediction
                "baseline_duration": baseline_duration,
                "prosody_duration": prosody_duration,
                "dur_diff_s": dur_diff,
                "dur_diff_pct": dur_diff_pct,
                "baseline_path": baseline_path,
                "prosody_path": prosody_path,
            }
            sent_results["comparisons"].append(result)

            # Print
            f0_dir = "↑" if f0_diff > 0 else "↓" if f0_diff < 0 else "="
            dur_dir = "+" if dur_diff > 0 else "-" if dur_diff < 0 else "="
            f0_hz = prosody_f0_audio
            print(f"  {emotion_name:12}: F0={f0_hz:.1f}Hz "
                  f"({f0_dir}{abs(f0_diff_pct):.1f}%), "
                  f"dur={prosody_duration:.2f}s ({dur_dir}{abs(dur_diff_pct):.1f}%), "
                  f"pred_f0={pred_f0_mean:.3f}")

        results.append(sent_results)

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY: Average F0 changes by emotion (Phase C)")
    print("=" * 60)

    emotion_stats: Dict[str, List[float]] = {e[0]: [] for e in emotions}

    for sent_results in results:
        for comp in sent_results["comparisons"]:
            emotion_stats[comp["emotion"]].append(comp["f0_diff_pct"])

    # Expected changes based on v5 data-driven analysis (multilingual dataset)
    # Key difference from v4: SAD is -4% (lower F0), not +12%!
    expected = {
        "NEUTRAL": 0,
        "ANGRY": +7,    # F0 +7% (data-driven, was +22% in RAVDESS)
        "SAD": -4,      # F0 -4% LOWER (KEY FIX! was +12% in RAVDESS - wrong!)
        "EXCITED": +15, # F0 +15% (data-driven, was +20%)
        "CALM": 0,      # Same as neutral (data-driven, was +3%)
    }

    print(f"{'Emotion':<15} {'Avg F0 Change':>15} {'Expected':>15} {'Match':>10}")
    print("-" * 55)

    for emotion, changes in emotion_stats.items():
        avg = np.mean(changes) if changes else 0
        exp = expected.get(emotion, 0)
        # Match criteria:
        # - If expected is 0: actual should be small (< 5%)
        # - If non-zero: direction correct AND within 50% of target (or >=5%)
        if exp == 0:
            match = "✓" if abs(avg) < 5 else "✗"
        else:
            direction_correct = (avg * exp > 0)  # Same sign
            close_to_target = abs(avg - exp) < abs(exp) * 0.5  # <=50% off
            strong_signal = abs(avg) >= 5  # At least 5% change
            is_match = direction_correct and (close_to_target or strong_signal)
            match = "✓" if is_match else "✗"
        print(f"{emotion:<15} {avg:>+14.1f}% {exp:>+14d}% {match:>10}")

    # Save results
    results_path = str(output_dir / "phase_c_comparison_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "prosody_embeddings": prosody_embeddings_path,
            "fc_prosody_weights": fc_prosody_weights_path,
            "num_sentences": len(test_sentences),
            "emotions_tested": [e[0] for e in emotions],
            "results": results,
            "summary": {
                e: {"mean_f0_diff_pct": float(np.mean(v))}
                for e, v in emotion_stats.items()
            },
        }, f, indent=2)

    print(f"\nResults saved to: {results_path}")
    print(f"Audio files in: {output_dir}/")

    return results


if __name__ == "__main__":
    run_phase_c_comparison()
