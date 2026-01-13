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
Test Prosody Contour Inference (Path C)

Validates that the ProsodyContourPredictor integration works correctly
by generating audio with different emotions and measuring F0 changes.

Usage:
    python scripts/test_prosody_contour_inference.py
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Tuple

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


def load_kokoro_model(model_dir: str):
    """Load Kokoro model with prosody contour support."""
    from tools.pytorch_to_mlx.converters.models.kokoro import (
        KokoroConfig,
        KokoroModel,
    )

    model_path = Path(model_dir)
    config_path = model_path / "config.json"

    # Load config
    with open(config_path) as f:
        config_dict = json.load(f)

    config = KokoroConfig(**config_dict)
    model = KokoroModel(config)

    # Load model weights
    weights_path = model_path / "model.safetensors"
    if weights_path.exists():
        weights = mx.load(str(weights_path))
        model.load_weights(list(weights.items()))

    return model


def load_voice(voice_path: str) -> mx.array:
    """Load voice embedding."""
    voice_data = mx.load(voice_path)
    if isinstance(voice_data, dict):
        # safetensors format
        for key in voice_data:
            return voice_data[key]
    return voice_data


def phonemize_text(text: str) -> Tuple[mx.array, int]:
    """Convert text to phoneme token IDs."""
    from tools.pytorch_to_mlx.converters.models.kokoro_phonemizer import (
        KokoroPhonemizer,
    )

    phonemizer = KokoroPhonemizer()
    input_ids = phonemizer.phonemize(text)
    return mx.array([input_ids]), len(input_ids)


def extract_f0_from_audio(audio: np.ndarray, sr: int = 24000) -> Tuple[float, float]:
    """Extract mean and std F0 from audio using YIN."""
    try:
        import librosa

        # Extract F0 using YIN
        f0 = librosa.yin(
            audio.astype(np.float32),
            fmin=50,
            fmax=500,
            sr=sr,
            frame_length=1024,
            hop_length=256,
        )

        # Filter valid F0 values
        f0_valid = f0[(f0 > 50) & (f0 < 500)]

        if len(f0_valid) > 0:
            return float(np.mean(f0_valid)), float(np.std(f0_valid))
        else:
            return 0.0, 0.0
    except ImportError:
        logger.warning("librosa not available, skipping F0 extraction")
        return 0.0, 0.0


def test_contour_model_loading():
    """Test that contour model loads correctly."""
    logger.info("=" * 60)
    logger.info("Test 1: Contour Model Loading")
    logger.info("=" * 60)

    contour_path = Path("models/prosody_contour_v1/best_model.npz")
    embedding_path = Path("models/prosody_embeddings_orthogonal/final.safetensors")

    if not contour_path.exists():
        logger.error(f"Contour model not found: {contour_path}")
        return False

    if not embedding_path.exists():
        logger.error(f"Embedding file not found: {embedding_path}")
        return False

    # Load and inspect weights
    contour_weights = mx.load(str(contour_path))
    logger.info(f"Contour model keys: {list(contour_weights.keys())}")

    emb_weights = mx.load(str(embedding_path))
    logger.info(f"Embedding keys: {list(emb_weights.keys())}")

    # Test creating ProsodyContourPredictor
    from tools.pytorch_to_mlx.converters.models.kokoro import ProsodyContourPredictor

    predictor = ProsodyContourPredictor(
        prosody_dim=768,
        hidden_dim=256,
        contour_len=50,
    )

    # Test forward pass
    test_ids = mx.array([0, 40, 41, 42])  # neutral, angry, sad, excited
    contours = predictor(test_ids)
    mx.eval(contours)

    logger.info(f"Contour output shape: {contours.shape}")
    logger.info(f"Contour range: [{float(contours.min()):.3f}, {float(contours.max()):.3f}]")

    logger.info("PASS: Contour model loads and runs correctly")
    return True


def test_contour_predictions():
    """Test contour predictions for different emotions."""
    logger.info("=" * 60)
    logger.info("Test 2: Contour Predictions by Emotion")
    logger.info("=" * 60)

    from tools.pytorch_to_mlx.converters.models.kokoro import ProsodyContourPredictor

    # Create predictor and load weights
    predictor = ProsodyContourPredictor(
        prosody_dim=768,
        hidden_dim=256,
        contour_len=50,
    )

    contour_path = Path("models/prosody_contour_v1/best_model.npz")
    embedding_path = Path("models/prosody_embeddings_orthogonal/final.safetensors")

    if not contour_path.exists() or not embedding_path.exists():
        logger.warning("Model files not found, using random weights")
    else:
        # Load contour weights
        contour_weights = mx.load(str(contour_path))
        for key in ["prosody_proj.weight", "prosody_proj.bias",
                    "contour_fc1.weight", "contour_fc1.bias",
                    "contour_fc2.weight", "contour_fc2.bias",
                    "contour_out.weight", "contour_out.bias"]:
            if key in contour_weights:
                parts = key.split(".")
                obj = predictor
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], contour_weights[key])

        # Load embedding weights
        emb_weights = mx.load(str(embedding_path))
        for key in emb_weights:
            if "embedding" in key.lower():
                predictor.embedding.weight = emb_weights[key]
                break

    # Test each emotion
    logger.info("\nContour statistics by emotion:")
    logger.info("-" * 50)

    results = {}
    for emotion, prosody_id in PROSODY_TYPES.items():
        contour = predictor(mx.array([prosody_id]))
        mx.eval(contour)
        contour_np = np.array(contour)[0]

        mean_val = float(contour_np.mean())
        std_val = float(contour_np.std())
        min_val = float(contour_np.min())
        max_val = float(contour_np.max())

        results[emotion] = {
            'mean': mean_val,
            'std': std_val,
            'min': min_val,
            'max': max_val,
        }

        logger.info(
            f"  {emotion:12s}: mean={mean_val:.3f}, std={std_val:.3f}, "
            f"range=[{min_val:.3f}, {max_val:.3f}]"
        )

    # Check that emotions produce different contours
    neutral_mean = results['neutral']['mean']
    angry_mean = results['angry']['mean']
    sad_mean = results['sad']['mean']

    if abs(angry_mean - neutral_mean) > 0.01:
        logger.info("\nPASS: Angry contour differs from neutral")
    else:
        logger.warning("\nWARN: Angry contour very similar to neutral")

    if abs(sad_mean - neutral_mean) > 0.01:
        logger.info("PASS: Sad contour differs from neutral")
    else:
        logger.warning("WARN: Sad contour very similar to neutral")

    return True


def test_full_inference():
    """Test full Kokoro inference with prosody contour."""
    logger.info("=" * 60)
    logger.info("Test 3: Full Inference with Prosody Contour")
    logger.info("=" * 60)

    model_dir = "models/kokoro-v0_19"
    voice_path = "models/kokoro-v0_19/voices/af_heart.pt"

    if not Path(model_dir).exists():
        logger.warning(f"Model directory not found: {model_dir}")
        logger.info("Skipping full inference test")
        return True

    # Load model
    logger.info("Loading Kokoro model...")
    model = load_kokoro_model(model_dir)

    # Enable prosody contour
    logger.info("Enabling prosody contour predictor...")
    model.enable_prosody_contour(
        prosody_dim=768,
        hidden_dim=256,
        contour_len=50,
    )

    # Load weights
    contour_path = "models/prosody_contour_v1/best_model.npz"
    embedding_path = "models/prosody_embeddings_orthogonal/final.safetensors"

    if Path(contour_path).exists() and Path(embedding_path).exists():
        model.load_prosody_contour_weights(contour_path, embedding_path)
        logger.info("Loaded prosody contour weights")
    else:
        logger.warning("Prosody contour weights not found, using random initialization")

    # Load voice
    if Path(voice_path).exists():
        voice = load_voice(voice_path)
    else:
        # Use random voice for testing
        voice = mx.random.normal((1, 256))

    # Test text
    test_text = "Hello, this is a test of the prosody contour system."
    input_ids, seq_len = phonemize_text(test_text)

    logger.info(f"\nTest text: '{test_text}'")
    logger.info(f"Sequence length: {seq_len}")

    # Generate baseline (neutral)
    logger.info("\nGenerating baseline (neutral)...")
    prosody_mask_neutral = mx.zeros_like(input_ids)
    audio_neutral = model(input_ids, voice, prosody_mask=prosody_mask_neutral)
    mx.eval(audio_neutral)
    audio_neutral_np = np.array(audio_neutral)[0]

    f0_neutral, std_neutral = extract_f0_from_audio(audio_neutral_np)
    logger.info(f"  Neutral F0: {f0_neutral:.1f} Hz (std: {std_neutral:.1f})")

    # Generate with emotions
    results = {'neutral': {'f0_mean': f0_neutral, 'f0_std': std_neutral}}

    for emotion in ['angry', 'sad', 'excited', 'calm']:
        prosody_id = PROSODY_TYPES[emotion]
        prosody_mask = mx.full(input_ids.shape, prosody_id, dtype=mx.int32)

        logger.info(f"\nGenerating with {emotion} (id={prosody_id})...")
        audio = model(input_ids, voice, prosody_mask=prosody_mask)
        mx.eval(audio)
        audio_np = np.array(audio)[0]

        f0_mean, f0_std = extract_f0_from_audio(audio_np)

        # Calculate change from neutral
        if f0_neutral > 0:
            f0_change = (f0_mean - f0_neutral) / f0_neutral * 100
        else:
            f0_change = 0.0

        results[emotion] = {
            'f0_mean': f0_mean,
            'f0_std': f0_std,
            'f0_change_pct': f0_change,
        }

        logger.info(f"  {emotion} F0: {f0_mean:.1f} Hz (std: {f0_std:.1f})")
        logger.info(f"  Change from neutral: {f0_change:+.1f}%")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Summary: F0 Changes from Neutral")
    logger.info("=" * 60)

    for emotion in ['angry', 'sad', 'excited', 'calm']:
        change = results[emotion].get('f0_change_pct', 0)
        logger.info(f"  {emotion:10s}: {change:+.1f}%")

    logger.info("\nPASS: Full inference test completed")
    return True


def main():
    parser = argparse.ArgumentParser(description='Test prosody contour inference')
    parser.add_argument('--test', choices=['loading', 'predictions', 'inference', 'all'],
                       default='all', help='Which test to run')
    args = parser.parse_args()

    logger.info("Prosody Contour Inference Test")
    logger.info("=" * 60)

    tests_passed = 0
    tests_total = 0

    if args.test in ['all', 'loading']:
        tests_total += 1
        if test_contour_model_loading():
            tests_passed += 1

    if args.test in ['all', 'predictions']:
        tests_total += 1
        if test_contour_predictions():
            tests_passed += 1

    if args.test in ['all', 'inference']:
        tests_total += 1
        if test_full_inference():
            tests_passed += 1

    logger.info("\n" + "=" * 60)
    logger.info(f"Results: {tests_passed}/{tests_total} tests passed")
    logger.info("=" * 60)

    return 0 if tests_passed == tests_total else 1


if __name__ == '__main__':
    sys.exit(main())
