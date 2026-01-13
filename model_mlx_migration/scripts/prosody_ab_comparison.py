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
A/B Comparison: Phase A (rule-based) vs Phase B (learned prosody embeddings)

Compares:
- Phase A: Rule-based multipliers (duration/F0 adjustments)
- Phase B: Learned prosody embeddings from RAVDESS training

Metrics:
- F0 difference between neutral and emotional
- Duration difference
- Whisper transcription accuracy
- Subjective: listen to outputs

Usage:
    python scripts/prosody_ab_comparison.py

Output: tests/prosody/ab_comparison_output/
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

# Prosody type IDs (from train_prosody_embeddings.py)
PROSODY_TYPES = {
    "NEUTRAL": 0,
    "EMOTION_ANGRY": 40,
    "EMOTION_SAD": 41,
    "EMOTION_EXCITED": 42,
    "EMOTION_CALM": 45,
    "EMOTION_FRUSTRATED": 48,
    "EMOTION_NERVOUS": 49,
    "EMOTION_SURPRISED": 50,
}


def load_kokoro_model(enable_prosody: bool = False, prosody_weights_path: str = None):
    """Load Kokoro model with optional prosody embeddings."""
    from tools.pytorch_to_mlx.converters import KokoroConverter

    converter = KokoroConverter()
    model, config, _ = converter.load_from_hf()

    if enable_prosody and prosody_weights_path:
        model.enable_prosody_embedding()
        model.load_prosody_weights(prosody_weights_path)
        print(f"Loaded prosody embeddings from {prosody_weights_path}")

    return model, converter


def get_phonemes(text: str) -> List[str]:
    """Convert text to phonemes using Kokoro's tokenizer."""
    # Use espeak for phonemization
    try:
        import subprocess
        result = subprocess.run(
            ['espeak-ng', '-q', '--ipa', '-v', 'en-us', text],
            capture_output=True, text=True
        )
        phonemes = result.stdout.strip()
        return phonemes
    except Exception as e:
        print(f"Phonemization failed: {e}")
        return text


def text_to_tokens(text: str) -> Tuple[mx.array, int]:
    """Convert text to Kokoro tokens.

    Returns:
        tokens: mx.array of token IDs [1, seq_len]
        phoneme_length: number of phonemes (for voice selection)
    """
    from tools.pytorch_to_mlx.converters.models.kokoro_phonemizer import phonemize_text

    phonemes, token_ids = phonemize_text(text)
    return mx.array([token_ids]), len(phonemes)


def get_voice_embedding(converter, voice_name: str = "af_heart", phoneme_length: int = 10) -> mx.array:
    """Get voice embedding for synthesis."""
    voice = converter.load_voice(voice_name, phoneme_length=phoneme_length)
    mx.eval(voice)
    return voice


def save_wav(audio: np.ndarray, path: str, sample_rate: int = 24000):
    """Save audio to WAV file."""
    # Normalize to int16
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
    """Compute mean and std F0 using librosa."""
    try:
        import librosa
        f0, voiced_flag, _ = librosa.pyin(
            audio, fmin=50, fmax=500, sr=sample_rate
        )
        f0_voiced = f0[voiced_flag]
        if len(f0_voiced) > 0:
            return float(np.mean(f0_voiced)), float(np.std(f0_voiced))
    except Exception as e:
        print(f"F0 computation failed: {e}")
    return 0.0, 0.0


def synthesize_with_prosody(
    model,
    converter,
    text: str,
    prosody_type: int = 0,  # 0 = NEUTRAL
) -> np.ndarray:
    """Synthesize with Phase B prosody embedding."""
    input_ids, phoneme_length = text_to_tokens(text)
    voice = converter.load_voice("af_heart", phoneme_length=phoneme_length)
    mx.eval(voice)

    # Create prosody mask (same type for all tokens)
    prosody_mask = mx.full(input_ids.shape, prosody_type, dtype=mx.int32)

    # Synthesize with prosody - call model() directly to pass prosody_mask
    audio = model(input_ids, voice, prosody_mask=prosody_mask)
    mx.eval(audio)

    return np.array(audio).flatten()


def synthesize_baseline(
    model,
    converter,
    text: str,
) -> np.ndarray:
    """Synthesize without prosody (baseline)."""
    input_ids, phoneme_length = text_to_tokens(text)
    voice = converter.load_voice("af_heart", phoneme_length=phoneme_length)
    mx.eval(voice)

    # Synthesize without prosody
    audio = model.synthesize(input_ids, voice)
    mx.eval(audio)

    return np.array(audio).flatten()


def run_ab_comparison():
    """Run A/B comparison between Phase A and Phase B."""
    output_dir = Path("tests/prosody/ab_comparison_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Test sentences
    test_sentences = [
        "Hello, this is a test of the prosody system.",
        "I am very happy to see you today!",
        "This is absolutely terrible news.",
        "Please stay calm, everything will be fine.",
        "I cannot believe this is happening right now!",
    ]

    # Emotions to test
    emotions = [
        ("NEUTRAL", PROSODY_TYPES["NEUTRAL"]),
        ("ANGRY", PROSODY_TYPES["EMOTION_ANGRY"]),
        ("SAD", PROSODY_TYPES["EMOTION_SAD"]),
        ("EXCITED", PROSODY_TYPES["EMOTION_EXCITED"]),
        ("CALM", PROSODY_TYPES["EMOTION_CALM"]),
    ]

    results = []

    # Load model with prosody embeddings
    print("Loading Kokoro with trained prosody embeddings...")
    prosody_weights = "models/prosody_embeddings_ravdess_768/final.safetensors"

    if not os.path.exists(prosody_weights):
        print(f"ERROR: Prosody weights not found at {prosody_weights}")
        print("Run train_prosody_embeddings.py first")
        return

    model, converter = load_kokoro_model(
        enable_prosody=True,
        prosody_weights_path=prosody_weights
    )

    print("\nGenerating audio for A/B comparison...")
    print("=" * 60)

    for sent_idx, sentence in enumerate(test_sentences):
        print(f"\n[Sentence {sent_idx + 1}] {sentence[:50]}...")

        sent_results = {"sentence": sentence, "comparisons": []}

        # Generate baseline (neutral, no prosody embedding)
        baseline_audio = synthesize_baseline(model, converter, sentence)
        baseline_path = str(output_dir / f"sent{sent_idx}_baseline.wav")
        save_wav(baseline_audio, baseline_path)
        baseline_f0, baseline_f0_std = compute_f0(baseline_audio)
        baseline_duration = len(baseline_audio) / 24000

        print(f"  Baseline: F0={baseline_f0:.1f}Hz, dur={baseline_duration:.2f}s")

        for emotion_name, emotion_id in emotions:
            # Generate with prosody embedding
            prosody_audio = synthesize_with_prosody(model, converter, sentence, emotion_id)
            prosody_path = str(output_dir / f"sent{sent_idx}_{emotion_name.lower()}.wav")
            save_wav(prosody_audio, prosody_path)

            prosody_f0, prosody_f0_std = compute_f0(prosody_audio)
            prosody_duration = len(prosody_audio) / 24000

            # Compute differences
            f0_diff = prosody_f0 - baseline_f0 if baseline_f0 > 0 else 0
            f0_diff_pct = (f0_diff / baseline_f0 * 100) if baseline_f0 > 0 else 0
            dur_diff = prosody_duration - baseline_duration
            dur_diff_pct = (dur_diff / baseline_duration * 100) if baseline_duration > 0 else 0

            result = {
                "emotion": emotion_name,
                "prosody_id": emotion_id,
                "baseline_f0": baseline_f0,
                "prosody_f0": prosody_f0,
                "f0_diff_hz": f0_diff,
                "f0_diff_pct": f0_diff_pct,
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
            print(f"  {emotion_name:12}: F0={prosody_f0:.1f}Hz ({f0_dir}{abs(f0_diff_pct):.1f}%), "
                  f"dur={prosody_duration:.2f}s ({dur_dir}{abs(dur_diff_pct):.1f}%)")

        results.append(sent_results)

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY: Average F0 changes by emotion")
    print("=" * 60)

    emotion_stats: Dict[str, List[float]] = {e[0]: [] for e in emotions}

    for sent_results in results:
        for comp in sent_results["comparisons"]:
            emotion_stats[comp["emotion"]].append(comp["f0_diff_pct"])

    print(f"{'Emotion':<15} {'Avg F0 Change':>15} {'Expected':>15}")
    print("-" * 45)

    # Expected changes based on RAVDESS training data
    expected = {
        "NEUTRAL": 0,
        "ANGRY": +22,  # F0 should increase
        "SAD": +12,    # F0 slightly elevated (from RAVDESS data)
        "EXCITED": +20,
        "CALM": +3,    # Minimal change
    }

    for emotion, changes in emotion_stats.items():
        avg = np.mean(changes) if changes else 0
        exp = expected.get(emotion, 0)
        match = "✓" if (exp == 0 and abs(avg) < 5) or (exp != 0 and avg * exp > 0) else "✗"
        print(f"{emotion:<15} {avg:>+14.1f}% {exp:>+14d}% {match}")

    # Save results
    results_path = str(output_dir / "ab_comparison_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "prosody_weights": prosody_weights,
            "num_sentences": len(test_sentences),
            "emotions_tested": [e[0] for e in emotions],
            "results": results,
            "summary": {e: {"mean_f0_diff_pct": float(np.mean(v))} for e, v in emotion_stats.items()},
        }, f, indent=2)

    print(f"\nResults saved to: {results_path}")
    print(f"Audio files in: {output_dir}/")

    return results


if __name__ == "__main__":
    run_ab_comparison()
