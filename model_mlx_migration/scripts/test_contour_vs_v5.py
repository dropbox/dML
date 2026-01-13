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
Test Contour Model vs v5 Static Multipliers

Compares:
1. Baseline (no prosody)
2. v5 (static F0 multipliers via fc_prosody)
3. Contour model (learned F0 contours)

Usage:
    python scripts/test_contour_vs_v5.py
"""

import json
import os
import sys
import wave
from pathlib import Path
from typing import Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx

# Prosody type IDs
PROSODY_TYPES = {
    "NEUTRAL": 0,
    "ANGRY": 40,
    "SAD": 41,
    "EXCITED": 42,
    "CALM": 45,
    "FRUSTRATED": 48,
    "NERVOUS": 49,
    "SURPRISED": 50,
}

# Emotion subset for comparison
TEST_EMOTIONS = ["NEUTRAL", "ANGRY", "SAD", "EXCITED", "CALM"]


def load_kokoro_model():
    """Load Kokoro model using converter."""
    from tools.pytorch_to_mlx.converters import KokoroConverter

    converter = KokoroConverter()
    model, config, _ = converter.load_from_hf()
    return model, converter


def load_contour_model():
    """Load trained contour model."""
    from tools.pytorch_to_mlx.converters.models.kokoro import ProsodyContourPredictor

    model_path = Path("models/prosody_contour_v1")
    config_path = model_path / "config.json"
    weights_path = model_path / "best_model.npz"
    embeddings_path = Path("models/prosody_embeddings_orthogonal/final.safetensors")

    # Load config
    with open(config_path) as f:
        config = json.load(f)

    # Create model
    predictor = ProsodyContourPredictor(
        prosody_dim=config["prosody_dim"],
        hidden_dim=config["hidden_dim"],
        contour_len=config["contour_len"],
    )

    # Load contour weights
    contour_weights = dict(mx.load(str(weights_path)))

    # Load prosody embeddings and merge with contour weights
    emb_weights = mx.load(str(embeddings_path))
    if "embedding.weight" in emb_weights:
        contour_weights["embedding.weight"] = emb_weights["embedding.weight"]

    # Load combined weights
    predictor.load_weights(list(contour_weights.items()))

    # Also return embeddings for external use
    prosody_embeddings = contour_weights.get("embedding.weight", emb_weights.get("embedding.weight"))

    return predictor, prosody_embeddings


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
    """Compute mean and std F0 using librosa YIN."""
    try:
        import librosa
        f0 = librosa.yin(
            audio.astype(np.float32), fmin=50, fmax=500, sr=sample_rate
        )
        f0_valid = f0[(f0 > 60) & (f0 < 480)]
        if len(f0_valid) > 0:
            return float(np.mean(f0_valid)), float(np.std(f0_valid))
    except Exception as e:
        print(f"F0 computation failed: {e}")
    return 0.0, 0.0


def synthesize_baseline(model, converter, text: str) -> np.ndarray:
    """Synthesize without prosody (baseline)."""
    input_ids, phoneme_length = text_to_tokens(text)
    voice = converter.load_voice("af_heart", phoneme_length=phoneme_length)
    mx.eval(voice)

    audio = model.synthesize(input_ids, voice)
    mx.eval(audio)

    return np.array(audio).flatten()


def synthesize_with_contour(
    model,
    converter,
    contour_predictor,
    prosody_embeddings,
    text: str,
    prosody_type: int,
) -> np.ndarray:
    """Synthesize with contour-based F0 modification."""
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

    # Run BERT encoder
    bert_out = model.bert(input_ids, None)
    bert_enc = model.bert_encoder(bert_out)

    # Run text encoder
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

    # Standard F0 prediction (no prosody conditioning)
    x = x_shared
    x = model.predictor.F0_0(x, speaker)
    x = model.predictor.F0_1(x, speaker)
    x = model.predictor.F0_2(x, speaker)
    f0_baseline = model.predictor.F0_proj(x).squeeze(-1)
    mx.eval(f0_baseline)

    # Predict F0 modifiers from contour model
    f0_len = f0_baseline.shape[-1]
    # Create prosody_mask: [batch, seq_len] with prosody type ID for all tokens
    prosody_mask = mx.full((1, seq_length), prosody_type, dtype=mx.int32)
    # Use baseline_mean=0.32 based on trained neutral contour statistics
    # (from test 2: neutral mean=0.321)
    f0_modifiers = contour_predictor.predict_f0_modifiers(
        prosody_mask,
        f0_len,
        baseline_mean=0.321,  # Trained neutral contour mean
        modifier_range=0.5,   # Wider range for more noticeable effect
    )
    mx.eval(f0_modifiers)

    # Apply F0 modification - only to positive (voiced) F0 values
    # Negative F0 values represent unvoiced frames and should not be modified
    f0 = mx.where(f0_baseline > 0, f0_baseline * f0_modifiers, f0_baseline)

    # Noise prediction (standard)
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

    return np.array(audio).flatten()


def main():
    output_dir = Path("tests/prosody/contour_vs_v5_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Kokoro model...")
    model, converter = load_kokoro_model()

    print("Loading contour model...")
    try:
        contour_predictor, prosody_embeddings = load_contour_model()
    except Exception as e:
        print(f"Failed to load contour model: {e}")
        return

    # Test sentences
    test_sentences = [
        ("neutral_context", "The weather today is partly cloudy with mild temperatures."),
        ("angry_context", "I cannot believe you would do something so reckless!"),
        ("sad_context", "I miss the old days when things were simpler."),
        ("excited_context", "This is the most amazing news I've heard all year!"),
        ("calm_context", "Take a deep breath and let yourself relax."),
    ]

    results = {}

    for context_name, sentence in test_sentences:
        print(f"\n{'='*60}")
        print(f"Testing: {context_name}")
        print(f"Text: {sentence}")
        print(f"{'='*60}")

        # Generate baseline
        print("  Generating baseline...")
        audio_baseline = synthesize_baseline(model, converter, sentence)
        f0_baseline, _ = compute_f0(audio_baseline)

        save_wav(audio_baseline, str(output_dir / f"{context_name}_baseline.wav"))
        print(f"  Baseline F0: {f0_baseline:.1f} Hz")

        results[context_name] = {
            "baseline_f0": f0_baseline,
            "contour": {},
        }

        # Generate with each emotion using contour model
        for emotion in TEST_EMOTIONS:
            prosody_type = PROSODY_TYPES[emotion]
            print(f"  Generating {emotion}...")

            audio_contour = synthesize_with_contour(
                model, converter, contour_predictor, prosody_embeddings,
                sentence, prosody_type
            )
            f0_contour, _ = compute_f0(audio_contour)

            save_wav(audio_contour, str(output_dir / f"{context_name}_{emotion.lower()}.wav"))

            f0_change = (f0_contour - f0_baseline) / f0_baseline * 100 if f0_baseline > 0 else 0
            print(f"    {emotion}: {f0_contour:.1f} Hz ({f0_change:+.1f}%)")

            results[context_name]["contour"][emotion] = {
                "f0": f0_contour,
                "f0_change_pct": f0_change,
            }

    # Save results
    results_path = output_dir / "comparison_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY: Contour Model Results")
    print("="*60)

    # Expected F0 targets (from v5 data-driven multipliers)
    targets = {
        "NEUTRAL": 0,
        "ANGRY": 7,      # +7%
        "SAD": -4,       # -4%
        "EXCITED": 15,   # +15%
        "CALM": 0,       # neutral
    }

    avg_changes = {e: [] for e in TEST_EMOTIONS}

    for context_name, data in results.items():
        for emotion, emdata in data["contour"].items():
            avg_changes[emotion].append(emdata["f0_change_pct"])

    print(f"\n{'Emotion':<12} {'Avg F0 Change':<15} {'Target':<10} {'Status'}")
    print("-" * 50)

    for emotion in TEST_EMOTIONS:
        avg = np.mean(avg_changes[emotion]) if avg_changes[emotion] else 0
        target = targets[emotion]

        # Check direction
        if target > 0:
            status = "PASS" if avg > 0 else "FAIL"
        elif target < 0:
            status = "PASS" if avg < 0 else "FAIL"
        else:
            status = "PASS" if abs(avg) < 3 else "WARN"

        print(f"{emotion:<12} {avg:+.1f}%{'':>8} {target:+.0f}%{'':>5} {status}")

    print(f"\nResults saved to: {output_dir}")
    print(f"Audio files: {len(list(output_dir.glob('*.wav')))}")


if __name__ == "__main__":
    main()
