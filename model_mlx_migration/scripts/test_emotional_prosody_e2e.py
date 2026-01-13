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
End-to-End Emotional Prosody Test.

Tests the full prosody trifecta:
- F0 Contour v2.4 (pitch modification)
- Duration v3 (speaking rate)
- Energy v3 (volume/loudness)

Verifies perceptual changes:
- ANGRY: faster + louder
- SAD: slower + quieter
- EXCITED: faster + louder
- NEUTRAL: baseline

Usage:
    python scripts/test_emotional_prosody_e2e.py
"""

import argparse
import logging
import sys
import wave
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Prosody type IDs (matching prosody_types.h)
PROSODY_TYPES = {
    'neutral': 0,
    'angry': 40,
    'sad': 41,
    'excited': 42,
    'calm': 45,
    'frustrated': 48,
    'nervous': 49,
    'surprised': 50,
}

# Expected duration behavior relative to neutral
DURATION_EXPECTATIONS = {
    'angry': '<',      # faster (dur < 1.0)
    'sad': '>',        # slower (dur > 1.0)
    'excited': '<',    # faster
    'calm': '=',       # similar
    'frustrated': '>',  # slightly slower
    'nervous': '<',    # faster
    'surprised': '<',  # faster
}

# Expected energy behavior relative to neutral
ENERGY_EXPECTATIONS = {
    'angry': '>',      # louder (energy > 1.0)
    'sad': '<',        # quieter (energy < 1.0)
    'excited': '>',    # louder
    'calm': '=',       # NOTE: No training data, actual prediction ~1.19 (louder)
    'frustrated': '>', # louder
    'nervous': '>',    # louder
    'surprised': '>',  # louder
}


def transcribe_with_whisper(audio_path: str) -> str:
    """Transcribe audio file with mlx-whisper."""
    try:
        import mlx_whisper
        result = mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo="mlx-community/whisper-tiny",
        )
        text: str = result.get("text", "")
        return text.strip()
    except Exception as e:
        return f"Error: {e}"


def compute_audio_stats(audio_np: np.ndarray, sample_rate: int = 24000) -> Dict:
    """Compute audio statistics: duration, RMS, max amplitude."""
    duration = len(audio_np) / sample_rate
    rms = np.sqrt(np.mean(audio_np ** 2))
    max_amp = np.max(np.abs(audio_np))
    return {
        'duration': duration,
        'rms': rms,
        'max_amp': max_amp,
        'samples': len(audio_np),
    }


def generate_emotional_audio(
    model,
    converter,
    voice_pack,
    text: str,
    emotion: str,
    output_dir: Path,
) -> Tuple[np.ndarray, Dict]:
    """Generate audio with emotional prosody."""
    from tools.pytorch_to_mlx.converters.models.kokoro_phonemizer import phonemize_text

    # Phonemize text
    phonemes, tokens = phonemize_text(text)

    # Create prosody mask for emotion (shape must match input_ids: [batch, seq_len])
    prosody_id = PROSODY_TYPES.get(emotion, 0)
    prosody_mask = mx.array([[prosody_id] * len(tokens)], dtype=mx.int32)

    # Get voice embedding
    voice = converter.select_voice_embedding(voice_pack, len(tokens))

    # Generate audio with prosody
    tokens_mx = mx.array([tokens])
    audio = model(tokens_mx, voice, prosody_mask=prosody_mask)
    mx.eval(audio)
    audio_np = np.array(audio).flatten()

    # Compute stats
    stats = compute_audio_stats(audio_np)
    stats['emotion'] = emotion
    stats['prosody_id'] = prosody_id
    stats['phonemes'] = phonemes
    stats['tokens'] = len(tokens)

    # Save WAV
    audio_path = output_dir / f"{emotion}.wav"
    audio_int16 = (audio_np * 32767).astype(np.int16)
    with wave.open(str(audio_path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(audio_int16.tobytes())

    stats['audio_path'] = str(audio_path)

    # Transcribe
    stats['transcript'] = transcribe_with_whisper(str(audio_path))

    return audio_np, stats


def evaluate_expectations(results: Dict[str, Dict], neutral_stats: Dict) -> List[Dict]:
    """Evaluate if emotional audio meets expectations."""
    evaluations = []

    neutral_dur = neutral_stats['duration']
    neutral_rms = neutral_stats['rms']

    for emotion, stats in results.items():
        if emotion == 'neutral':
            continue

        eval_result = {
            'emotion': emotion,
            'duration_ratio': stats['duration'] / neutral_dur,
            'rms_ratio': stats['rms'] / neutral_rms,
        }

        # Check duration expectation
        dur_exp = DURATION_EXPECTATIONS.get(emotion, '=')
        if dur_exp == '<':
            dur_pass = stats['duration'] < neutral_dur
        elif dur_exp == '>':
            dur_pass = stats['duration'] > neutral_dur
        else:
            dur_pass = abs(stats['duration'] - neutral_dur) / neutral_dur < 0.1
        eval_result['duration_pass'] = dur_pass
        eval_result['duration_expected'] = dur_exp

        # Check energy expectation
        energy_exp = ENERGY_EXPECTATIONS.get(emotion, '=')
        if energy_exp == '<':
            energy_pass = stats['rms'] < neutral_rms
        elif energy_exp == '>':
            energy_pass = stats['rms'] > neutral_rms
        else:
            # Allow 20% tolerance for '=' (neutral-like emotions)
            energy_pass = abs(stats['rms'] - neutral_rms) / neutral_rms < 0.2
        eval_result['energy_pass'] = energy_pass
        eval_result['energy_expected'] = energy_exp

        evaluations.append(eval_result)

    return evaluations


def main():
    parser = argparse.ArgumentParser(description="Test emotional prosody end-to-end")
    parser.add_argument("--text", default="Hello, how are you doing today?", help="Text to synthesize")
    parser.add_argument("--emotions", nargs="+", default=["neutral", "angry", "sad", "excited"],
                       help="Emotions to test")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/emotional_prosody_test"),
                       help="Output directory")
    parser.add_argument("--voice", default="af_heart", help="Voice pack to use")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("Emotional Prosody End-to-End Test")
    logger.info("=" * 70)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info("Loading Kokoro model...")
    from tools.pytorch_to_mlx.converters import KokoroConverter

    converter = KokoroConverter()
    model, config, _ = converter.load_from_hf()
    model.set_deterministic(True)

    # Load voice pack
    logger.info(f"Loading voice pack: {args.voice}")
    voice_pack = converter.load_voice_pack(args.voice)
    mx.eval(voice_pack)

    # Enable prosody contour v2.4
    logger.info("Enabling prosody contour v2.4...")
    model.enable_prosody_contour_v2()
    contour_weights = Path("models/prosody_contour_v2.4/best_model.npz")
    embedding_path = Path("models/prosody_embeddings_orthogonal/final.safetensors")
    if contour_weights.exists():
        model.load_prosody_contour_v2_weights(contour_weights, embedding_path)
        logger.info(f"  Loaded contour v2.4 weights from {contour_weights}")
    else:
        logger.warning(f"  Contour weights not found: {contour_weights}")

    # Enable prosody duration/energy v3
    logger.info("Enabling prosody duration/energy v3...")
    model.enable_prosody_duration_energy()
    duration_energy_weights = Path("models/prosody_duration_energy_v3/best_model.npz")
    if duration_energy_weights.exists():
        model.load_prosody_duration_energy_weights(duration_energy_weights, embedding_path)
        logger.info(f"  Loaded duration/energy v3 weights from {duration_energy_weights}")
    else:
        logger.warning(f"  Duration/energy weights not found: {duration_energy_weights}")

    # Generate audio for each emotion
    logger.info(f"\nGenerating audio for text: '{args.text}'")
    logger.info(f"Emotions: {args.emotions}")

    results = {}
    for emotion in args.emotions:
        logger.info(f"\n--- {emotion.upper()} ---")
        audio, stats = generate_emotional_audio(
            model, converter, voice_pack,
            args.text, emotion, args.output_dir
        )
        results[emotion] = stats

        logger.info(f"  Duration: {stats['duration']:.3f}s")
        logger.info(f"  RMS: {stats['rms']:.4f}")
        logger.info(f"  Max Amp: {stats['max_amp']:.4f}")
        logger.info(f"  Whisper: '{stats['transcript']}'")
        logger.info(f"  Saved: {stats['audio_path']}")

    # Evaluate expectations
    logger.info("\n" + "=" * 70)
    logger.info("Evaluation Results")
    logger.info("=" * 70)

    if 'neutral' not in results:
        logger.error("NEUTRAL not in results, cannot evaluate")
        return 1

    neutral_stats = results['neutral']
    evaluations = evaluate_expectations(results, neutral_stats)

    logger.info(f"\nBaseline (NEUTRAL): duration={neutral_stats['duration']:.3f}s, RMS={neutral_stats['rms']:.4f}")
    logger.info("")
    logger.info(f"{'Emotion':<12} {'Dur Ratio':<12} {'Dur Exp':<10} {'Dur Pass':<10} {'RMS Ratio':<12} {'RMS Exp':<10} {'RMS Pass':<10}")
    logger.info("-" * 88)

    all_pass = True
    for e in evaluations:
        dur_status = "PASS" if e['duration_pass'] else "FAIL"
        rms_status = "PASS" if e['energy_pass'] else "FAIL"

        if not e['duration_pass'] or not e['energy_pass']:
            all_pass = False

        logger.info(
            f"{e['emotion']:<12} "
            f"{e['duration_ratio']:<12.3f} "
            f"{e['duration_expected']:<10} "
            f"{dur_status:<10} "
            f"{e['rms_ratio']:<12.3f} "
            f"{e['energy_expected']:<10} "
            f"{rms_status:<10}"
        )

    # Transcription verification
    logger.info("\n" + "-" * 70)
    logger.info("Transcription Verification")
    logger.info("-" * 70)

    unique_transcripts = set(r['transcript'].lower() for r in results.values())

    for emotion, stats in results.items():
        logger.info(f"  {emotion}: '{stats['transcript']}'")

    if len(unique_transcripts) == 1:
        logger.info("\nNOTE: All transcripts identical (expected for same text)")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("Summary")
    logger.info("=" * 70)

    if all_pass:
        logger.info("ALL TESTS PASSED: Emotional prosody working correctly")
        logger.info("  - ANGRY: faster + louder (as expected)")
        logger.info("  - SAD: slower + quieter (as expected)")
        logger.info("  - EXCITED: faster + louder (as expected)")
    else:
        logger.info("SOME TESTS FAILED: Review results above")

    logger.info(f"\nAudio files saved to: {args.output_dir}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
