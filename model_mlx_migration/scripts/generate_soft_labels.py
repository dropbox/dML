#!/usr/bin/env python3
"""
Generate Soft Labels from SOTA Models for Knowledge Distillation

This script processes training data through SOTA models (emotion2vec, wav2vec2-xlsr-ser)
and generates soft labels (probability distributions) for knowledge distillation training.

The soft labels allow our Whisper-based heads to learn from SOTA model knowledge,
improving performance beyond what hard labels alone can achieve.

Usage:
    python scripts/generate_soft_labels.py --manifest data/v4_expanded/train_manifest.json
    python scripts/generate_soft_labels.py --manifest data/v4_expanded/val_manifest.json --output-dir data/soft_labels_val
    python scripts/generate_soft_labels.py --model emotion2vec --limit 1000  # Test run
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx

# Audio processing
try:
    import librosa
except ImportError:
    print("Warning: librosa not available, using soundfile only")
    librosa = None

import soundfile as sf


# Model configurations for soft label generation
SOFT_LABEL_MODELS = {
    "emotion2vec": {
        "name": "Emotion2vec (Emotion Recognition)",
        "path": "models/sota/emotion2vec-mlx",
        "task": "emotion",
        "output_type": "features",  # needs classifier head
        "target_sr": 16000,
    },
    "wav2vec2-xlsr-ser": {
        "name": "Wav2Vec2-XLSR-SER (Speech Emotion)",
        "path": "models/sota/wav2vec2-xlsr-ser-mlx",
        "task": "emotion",
        "output_type": "logits",
        "num_classes": 8,
        "labels": ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"],
        "target_sr": 16000,
    },
    "ast": {
        "name": "AST (Audio Classification)",
        "path": "models/sota/ast-mlx",
        "task": "paralinguistics",
        "output_type": "logits",
        "num_classes": 527,  # AudioSet classes
        "target_sr": 16000,
    },
    "beats": {
        "name": "BEATs (Audio Pre-Training)",
        "path": "models/sota/beats-mlx",
        "task": "paralinguistics",
        "output_type": "features",
        "target_sr": 16000,
    },
}


def load_audio(audio_path: str, target_sr: int = 16000, max_duration: float = 30.0) -> Optional[np.ndarray]:
    """Load audio file and resample if needed."""
    try:
        audio, sr = sf.read(audio_path)

        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # Resample if needed
        if sr != target_sr:
            if librosa is not None:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            else:
                # Basic resampling via numpy (less accurate)
                ratio = target_sr / sr
                new_length = int(len(audio) * ratio)
                audio = np.interp(
                    np.linspace(0, len(audio) - 1, new_length),
                    np.arange(len(audio)),
                    audio
                )

        # Truncate to max duration
        max_samples = int(max_duration * target_sr)
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        return audio.astype(np.float32)
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return None


def load_model(model_name: str):
    """Load SOTA model for soft label generation."""
    config = SOFT_LABEL_MODELS[model_name]
    model_path = config["path"]

    if model_name == "emotion2vec":
        from tools.whisper_mlx.sota.emotion2vec_mlx import Emotion2vecModel
        model = Emotion2vecModel.from_pretrained(model_path)
    elif model_name == "wav2vec2-xlsr-ser":
        from tools.whisper_mlx.sota.wav2vec2_xlsr_mlx import Wav2Vec2ForSequenceClassification
        model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
    elif model_name == "ast":
        from tools.whisper_mlx.sota.ast_mlx import ASTForAudioClassification
        model = ASTForAudioClassification.from_pretrained(model_path)
    elif model_name == "beats":
        from tools.whisper_mlx.sota.beats_mlx import BEATsModel
        model = BEATsModel.from_pretrained(model_path)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    mx.eval(model.parameters())
    return model


def compute_mel_spectrogram(audio: np.ndarray, n_mels: int = 128, n_fft: int = 1024, hop_length: int = 160) -> np.ndarray:
    """Compute mel spectrogram for AST model."""
    if librosa is None:
        raise RuntimeError("librosa required for mel spectrogram computation")

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=16000,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.astype(np.float32)


def compute_fbank(audio: np.ndarray, n_mels: int = 128, n_fft: int = 512, hop_length: int = 160) -> np.ndarray:
    """Compute filterbank features for BEATs model."""
    if librosa is None:
        raise RuntimeError("librosa required for filterbank computation")

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=16000,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    fbank = librosa.power_to_db(mel, ref=np.max)
    # BEATs expects (batch, time, freq)
    return fbank.T.astype(np.float32)


def get_soft_labels(model, model_name: str, audio: np.ndarray, temperature: float = 1.0) -> dict:
    """Get soft labels (probabilities) from model."""
    config = SOFT_LABEL_MODELS[model_name]

    # Prepare input based on model type
    if model_name == "ast":
        mel = compute_mel_spectrogram(audio)
        # AST expects (batch, n_mels, time)
        # Pad or truncate to fixed length
        target_len = 1024
        if mel.shape[1] < target_len:
            mel = np.pad(mel, ((0, 0), (0, target_len - mel.shape[1])))
        else:
            mel = mel[:, :target_len]
        input_tensor = mx.array(mel[np.newaxis, :, :])
    elif model_name == "beats":
        fbank = compute_fbank(audio)
        # Pad to minimum length
        if fbank.shape[0] < 16:
            fbank = np.pad(fbank, ((0, 16 - fbank.shape[0]), (0, 0)))
        input_tensor = mx.array(fbank[np.newaxis, :, :])
    else:
        # Audio models (emotion2vec, wav2vec2)
        # Ensure minimum length (0.5 second)
        min_samples = 8000
        if len(audio) < min_samples:
            audio = np.pad(audio, (0, min_samples - len(audio)))
        input_tensor = mx.array(audio[np.newaxis, :])

    # Forward pass
    output = model(input_tensor)
    mx.eval(output)

    # Clean up input tensor to free memory
    del input_tensor

    result = {
        "model": model_name,
        "model_name": config["name"],
    }

    if config["output_type"] == "logits":
        # Get logits and compute softmax with temperature
        if isinstance(output, tuple):
            logits = np.array(output[0])
        else:
            logits = np.array(output)

        # Free MLX array after conversion to numpy
        del output

        # Apply temperature scaling
        logits = logits / temperature

        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        result["logits"] = logits.squeeze().tolist()
        result["probabilities"] = probs.squeeze().tolist()

        if "labels" in config:
            result["labels"] = config["labels"]
            result["predicted_label"] = config["labels"][np.argmax(probs)]
            result["confidence"] = float(np.max(probs))
    else:
        # Features - save pooled representation
        if isinstance(output, tuple):
            features = np.array(output[0])
        else:
            features = np.array(output)

        # Free MLX array after conversion to numpy
        del output

        # Global average pooling if needed
        if len(features.shape) == 3:
            features = features.mean(axis=1)

        result["features"] = features.squeeze().tolist()
        result["feature_dim"] = features.shape[-1]

    return result


def process_manifest(
    manifest_path: str,
    output_dir: str,
    models: list,
    temperature: float = 2.0,
    batch_save_size: int = 1000,
    limit: Optional[int] = None,
    skip_existing: bool = True,
):
    """Process manifest and generate soft labels for all samples."""
    os.makedirs(output_dir, exist_ok=True)

    # Load manifest
    print(f"Loading manifest: {manifest_path}")
    with open(manifest_path) as f:
        manifest = json.load(f)

    if limit:
        manifest = manifest[:limit]

    print(f"Processing {len(manifest)} samples")
    print(f"Models: {models}")
    print(f"Temperature: {temperature}")

    # Load models
    loaded_models = {}
    for model_name in models:
        print(f"Loading {model_name}...")
        loaded_models[model_name] = load_model(model_name)
        print(f"  {model_name} loaded")

    # Process samples
    results = []
    errors = []

    for i, sample in enumerate(tqdm(manifest, desc="Generating soft labels")):
        audio_path = sample["audio_path"]

        # Check if already processed
        sample_id = Path(audio_path).stem
        output_file = os.path.join(output_dir, f"{sample_id}.json")

        if skip_existing and os.path.exists(output_file):
            continue

        # Load audio
        audio = load_audio(audio_path)
        if audio is None:
            errors.append({"path": audio_path, "error": "Failed to load audio"})
            continue

        # Generate soft labels from each model
        sample_result = {
            "audio_path": audio_path,
            "original_labels": {
                "emotion": sample.get("emotion"),
                "para": sample.get("para"),
                "language": sample.get("language"),
            },
            "soft_labels": {},
        }

        for model_name, model in loaded_models.items():
            try:
                soft_labels = get_soft_labels(model, model_name, audio, temperature)
                sample_result["soft_labels"][model_name] = soft_labels
            except Exception as e:
                sample_result["soft_labels"][model_name] = {"error": str(e)}

        results.append(sample_result)

        # Save individual file
        with open(output_file, "w") as f:
            json.dump(sample_result, f, indent=2)

        # Periodic batch save of index and memory cleanup
        if (i + 1) % batch_save_size == 0:
            save_index(output_dir, results, errors)
            print(f"  Saved checkpoint at {i+1} samples")

        # Aggressive memory cleanup every 100 samples to prevent OOM
        if (i + 1) % 100 == 0:
            if hasattr(mx, 'clear_cache'):
                mx.clear_cache()
            gc.collect()

    # Final save
    save_index(output_dir, results, errors)

    print("\nComplete!")
    print(f"  Processed: {len(results)}")
    print(f"  Errors: {len(errors)}")
    print(f"  Output: {output_dir}")


def save_index(output_dir: str, results: list, errors: list):
    """Save index file with metadata."""
    index = {
        "total_processed": len(results),
        "total_errors": len(errors),
        "errors": errors[:100],  # Save first 100 errors
    }

    with open(os.path.join(output_dir, "index.json"), "w") as f:
        json.dump(index, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Generate soft labels from SOTA models")
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/v4_expanded/train_manifest.json",
        help="Path to training manifest",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/soft_labels",
        help="Output directory for soft labels",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="wav2vec2-xlsr-ser",
        help="Comma-separated list of models to use",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=2.0,
        help="Temperature for softmax (higher = softer labels)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples to process (for testing)",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Reprocess samples even if output exists",
    )

    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")]

    # Validate models
    for model in models:
        if model not in SOFT_LABEL_MODELS:
            print(f"Error: Unknown model '{model}'")
            print(f"Available models: {list(SOFT_LABEL_MODELS.keys())}")
            sys.exit(1)

    process_manifest(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        models=models,
        temperature=args.temperature,
        limit=args.limit,
        skip_existing=not args.no_skip_existing,
    )


if __name__ == "__main__":
    main()
