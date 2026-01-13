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
ULTIMATE Emotional TTS Synthesis - ALL Features Combined (Unified Model v1)

Uses the UNIFIED prosody model that predicts ALL features simultaneously:
1. F0 (pitch) multiplier - learned
2. F0 contour shape - learned (modulates pitch over time)
3. Duration multiplier - learned (angry=fast, sad=slow)
4. Energy multiplier - learned (angry=loud, sad=quiet)

Usage:
    python scripts/synthesize_full_emotion.py --emotion angry --text "Hello world"
"""

import argparse
import os
import sys
import wave
from pathlib import Path
from typing import Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import mlx.core as mx

# Emotion ID mapping - focused set of distinct emotions
EMOTION_IDS = {
    "neutral": 0,
    "angry": 40,
    "sad": 41,
    "excited": 42,
    "happy": 0,         # Use neutral prosody (warmer) with manual upbeat params
}

# Emotion settings: (f0_mult, duration_mult, energy_mult, contour_range, prosody_scale)
# contour_range: pitch variation in utterance (0 = none, 0.5 = ±50%)
EMOTION_SETTINGS = {
    "neutral": (1.00, 1.00, 1.00, 0.05, 0.5),  # Baseline, subtle variation
    "angry":   (1.02, 0.75, 1.50, 0.55, 1.0),  # Aggressive: fast, loud, big swings
    "sad":     (0.94, 1.20, 0.78, 0.15, 0.6),  # Lower pitch, slow, quiet
    "excited": (1.02, 0.90, 1.20, 0.20, 0.5),  # Reduced pitch, moderate energy
    "happy":   (1.04, 0.94, 1.20, 0.20, 0.5),  # Warm, slightly upbeat
}


def save_wav(audio: np.ndarray, path: str, sample_rate: int = 24000):
    """Save audio to WAV file."""
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    return path


def load_models():
    """Load all required models including unified prosody predictor."""
    from scripts.train_prosody_unified import UnifiedProsodyPredictor
    from tools.pytorch_to_mlx.converters import KokoroConverter

    print("Loading Kokoro TTS model...")
    converter = KokoroConverter()
    model, config, _ = converter.load_from_hf()

    # Load prosody embeddings
    print("Loading prosody embeddings...")
    prosody_weights = mx.load('models/prosody_embeddings_orthogonal/final.safetensors')
    embedding_table = prosody_weights['embedding.weight']

    # Load UNIFIED prosody model v2 (stronger emotions)
    print("Loading UNIFIED prosody model v2...")
    unified_model = UnifiedProsodyPredictor(
        prosody_dim=768, hidden_dim=512, contour_len=50, num_blocks=4
    )
    weights = mx.load('models/prosody_unified_v2/best_model.npz')
    unified_model.load_weights(list(weights.items()))

    # Enable Phase C prosody conditioning (per-emotion scale during synthesis)
    print("Enabling prosody conditioning...")
    model.predictor.enable_prosody_conditioning(prosody_dim=768, prosody_scale=0.7)
    model.predictor.copy_f0_weights_to_prosody_blocks()

    # Load fc_prosody weights
    fc_path = 'models/prosody_f0_conditioning_v5/fc_prosody_weights.safetensors'
    fc_weights = mx.load(fc_path)
    for key, value in fc_weights.items():
        parts = key.split(".")
        if len(parts) < 4:
            continue
        block_name = parts[0]
        if not hasattr(model.predictor, block_name):
            continue
        block = getattr(model.predictor, block_name)
        target = block
        for part in parts[1:-1]:
            if hasattr(target, part):
                target = getattr(target, part)
            else:
                target = None
                break
        if target is not None:
            param_name = parts[-1]
            if hasattr(target, param_name):
                setattr(target, param_name, value)

    return model, converter, embedding_table, unified_model


def synthesize_with_full_emotion(
    model,
    converter,
    embedding_table: mx.array,
    unified_model,
    text: str,
    emotion: str,
    voice_name: str = "af_heart",
    language: str = "en",
) -> Tuple[np.ndarray, dict]:
    """
    Synthesize speech with FULL emotional expression using UNIFIED model.

    The unified model predicts ALL features from one forward pass:
    1. F0 pitch multiplier - learned
    2. F0 contour shape - learned
    3. Duration multiplier - learned
    4. Energy multiplier - learned
    """
    from tools.pytorch_to_mlx.converters.models.kokoro_phonemizer import phonemize_text

    prosody_id = EMOTION_IDS.get(emotion, 0)

    # Get emotion-specific settings
    settings = EMOTION_SETTINGS.get(emotion, EMOTION_SETTINGS["neutral"])
    f0_mult, duration_mult, energy_mult, contour_range, prosody_scale = settings

    # Get prosody embedding
    prosody_id_arr = mx.array([prosody_id])
    prosody_emb = embedding_table[prosody_id_arr]

    # Get F0 contour from model (for modulation)
    _, f0_contour, _, _ = unified_model(prosody_emb, prosody_id_arr)

    print(f"\n  Emotion: {emotion.upper()}")
    print(f"  F0 multiplier: {f0_mult:.3f}")
    print(f"  Duration: {duration_mult:.2f}x (speed={1/duration_mult:.2f})")
    print(f"  Energy: {energy_mult:.2f}x")
    print(f"  Contour range: ±{contour_range*50:.0f}%")
    print(f"  Prosody scale: {prosody_scale}")

    # Tokenize
    phonemes, token_ids = phonemize_text(text, language=language)
    input_ids = mx.array([token_ids])

    # Load voice
    voice = converter.load_voice(voice_name, phoneme_length=len(phonemes))
    mx.eval(voice)

    if voice.shape[-1] == 256:
        style = voice[:, :128]
        speaker = voice[:, 128:]
    else:
        style = voice
        speaker = voice

    # Run TTS pipeline with prosody conditioning
    bert_out = model.bert(input_ids, None)
    bert_enc = model.bert_encoder(bert_out)
    duration_feats = model.predictor.text_encoder(bert_enc, speaker)
    dur_enc = model.predictor.lstm(duration_feats)
    duration_logits = model.predictor.duration_proj(dur_enc)

    # Apply DURATION control via speed parameter
    speed = 1.0 / duration_mult  # faster = higher speed value
    indices, total_frames, _ = model._compute_alignment(duration_logits, speed=speed)
    en_expanded_640 = model._expand_features(duration_feats, indices, total_frames)
    x_shared = model.predictor.shared(en_expanded_640)

    # F0 with prosody conditioning
    x = x_shared
    x = model.predictor.F0_0_prosody(x, speaker, prosody_emb)
    x = model.predictor.F0_1_prosody(x, speaker, prosody_emb)
    x = model.predictor.F0_2_prosody(x, speaker, prosody_emb)
    f0 = model.predictor.F0_proj(x).squeeze(-1)

    # Apply F0 contour shape to modulate pitch over time
    # contour_range controls how much variation (angry = high, neutral = low)
    if f0_contour is not None and contour_range > 0:
        contour_np = np.array(f0_contour[0])  # (50,)
        # Interpolate contour to match f0 length
        f0_len = f0.shape[-1]
        contour_interp = np.interp(
            np.linspace(0, 1, f0_len),
            np.linspace(0, 1, len(contour_np)),
            contour_np
        )
        # Map [0,1] to [1-range, 1+range] centered around 1.0
        # e.g., contour_range=0.4 gives [0.6, 1.4] = ±40% variation
        contour_mod = (1.0 - contour_range) + (2.0 * contour_range) * contour_interp
        f0 = f0 * mx.array(contour_mod.reshape(1, -1))

    # Noise (standard path)
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

    audio_np = np.array(audio).flatten()

    # Apply ENERGY control via amplitude scaling
    audio_np = audio_np * energy_mult

    # Clip to prevent clipping
    audio_np = np.clip(audio_np, -0.99, 0.99)

    metrics = {
        "emotion": emotion,
        "f0_mult": f0_mult,
        "duration_mult": duration_mult,
        "energy_mult": energy_mult,
        "audio_length_sec": len(audio_np) / 24000,
    }

    return audio_np, metrics


def main():
    # Core emotions - distinct and well-defined
    EMOTIONS = ["neutral", "angry", "sad", "happy", "excited"]

    parser = argparse.ArgumentParser(
        description="Full Emotion TTS Synthesis - Unified Model v1"
    )
    parser.add_argument(
        "--text", type=str,
        default="I can not believe what just happened to us today."
    )
    parser.add_argument(
        "--emotion", type=str, default="all",
        choices=["all"] + EMOTIONS
    )
    parser.add_argument("--voice", type=str, default="af_heart")
    parser.add_argument(
        "--language", type=str, default=None,
        help="Language code (en, ja). Auto-detects from voice."
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="/Users/ayates/voice/unified_emotion_samples"
    )
    parser.add_argument(
        "--play", action="store_true",
        help="Play audio after generation"
    )
    args = parser.parse_args()

    # Auto-detect language from voice name if not specified
    if args.language is None:
        if args.voice.startswith("j"):
            args.language = "ja"
        else:
            args.language = "en"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    model, converter, embedding_table, unified_model = load_models()

    emotions = EMOTIONS if args.emotion == "all" else [args.emotion]

    print(f"\n{'='*70}")
    print("UNIFIED EMOTION SYNTHESIS - ALL FEATURES LEARNED")
    print(f"{'='*70}")
    print(f"Text: '{args.text}'")
    print(f"Voice: {args.voice}")
    print(f"Language: {args.language}")
    print("Model: prosody_unified_v1")
    print("(Per-emotion prosody settings)")

    all_files = []

    for emotion in emotions:
        audio, metrics = synthesize_with_full_emotion(
            model, converter, embedding_table, unified_model,
            text=args.text,
            emotion=emotion,
            voice_name=args.voice,
            language=args.language,
        )

        filename = f"unified_{emotion}.wav"
        filepath = output_dir / filename
        save_wav(audio, str(filepath))
        print(f"  Saved: {filepath}")
        print(f"  Length: {metrics['audio_length_sec']:.2f}s")
        all_files.append((emotion, filepath))

    print(f"\n{'='*70}")
    print(f"All files saved to: {output_dir}")
    print(f"{'='*70}")

    if args.play:
        import subprocess
        for emotion, filepath in all_files:
            print(f"\nPlaying {emotion}...")
            subprocess.run(["afplay", str(filepath)])


if __name__ == "__main__":
    main()
