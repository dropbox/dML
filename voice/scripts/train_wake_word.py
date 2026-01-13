#!/usr/bin/env python3
"""
Wake Word Model Training Script

Trains custom wake word models using openWakeWord and Kokoro TTS for sample generation.

Usage:
    # Generate training data
    python scripts/train_wake_word.py --generate-samples --wake-word "hey voice"

    # Train model
    python scripts/train_wake_word.py --train --wake-word "hey voice"

    # Full pipeline
    python scripts/train_wake_word.py --full --wake-word "hey voice"

Requirements:
    - openWakeWord (pip install openwakeword)
    - Kokoro TTS (already in project)
    - audiomentations, speechbrain
"""

import argparse
import os
import sys
import uuid
import json
import logging
import random
from pathlib import Path
from typing import List, Optional
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import scipy.io.wavfile as wavfile
import torch
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models" / "wakeword"
DATA_DIR = PROJECT_ROOT / "data" / "wakeword"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Training parameters
SAMPLE_RATE = 16000  # openWakeWord requires 16kHz
KOKORO_SAMPLE_RATE = 24000  # Kokoro outputs 24kHz

# Voice variations for diversity
KOKORO_VOICES = [
    # American English voices
    "af_heart", "af_bella", "af_alloy", "af_aoede", "af_jessica",
    "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx", "am_puck",
    # British English voices
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
]


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to target sample rate."""
    import soxr
    return soxr.resample(audio, orig_sr, target_sr)


def generate_with_kokoro(text: str, output_path: Path, voice: str = None) -> bool:
    """Generate speech using Kokoro TTS and save as 16kHz WAV."""
    try:
        from kokoro import KPipeline
        import soundfile as sf

        # Get pipeline
        lang_code = 'a'  # American English
        pipe = KPipeline(lang_code=lang_code, repo_id='hexgrad/Kokoro-82M')

        if voice is None:
            voice = random.choice(KOKORO_VOICES)

        # Generate audio
        audio_chunks = []
        for result in pipe(text, voice=voice):
            audio_chunks.append(result.output.audio.numpy())

        if not audio_chunks:
            return False

        audio = np.concatenate(audio_chunks)

        # Resample from 24kHz to 16kHz
        audio_16k = resample_audio(audio, KOKORO_SAMPLE_RATE, SAMPLE_RATE)

        # Normalize and convert to int16
        audio_16k = audio_16k / np.max(np.abs(audio_16k) + 1e-8)
        audio_int16 = (audio_16k * 32767).astype(np.int16)

        # Save
        wavfile.write(str(output_path), SAMPLE_RATE, audio_int16)
        return True

    except Exception as e:
        logger.error(f"Failed to generate '{text}': {e}")
        return False


def generate_adversarial_texts(target_phrase: str, n_samples: int) -> List[str]:
    """Generate adversarial/similar phrases for negative samples."""
    words = target_phrase.lower().split()
    adversarial = []

    # Word substitutions
    substitutions = {
        "hey": ["hay", "haze", "hate", "eight", "hey there", "okay", "hey man", "say"],
        "voice": ["void", "choice", "vice", "boys", "noise", "invoice", "joyce", "vows"],
        "agent": ["agent", "asian", "ancient", "aging", "urgent", "aegean", "patient"],
    }

    # Generate variations
    for _ in range(n_samples // 4):
        # Single word substitution
        new_words = words.copy()
        idx = random.randint(0, len(words) - 1)
        word = words[idx]
        if word in substitutions:
            new_words[idx] = random.choice(substitutions[word])
        adversarial.append(" ".join(new_words))

    # Partial phrases
    for _ in range(n_samples // 4):
        if len(words) > 1:
            adversarial.append(words[0])  # Just "hey"
            adversarial.append(words[-1])  # Just "voice" or "agent"

    # Similar sounding phrases
    similar_phrases = [
        "hey there", "hey boy", "hey boys", "okay voice", "say voice",
        "hey choice", "hey toys", "hey noise", "hay voice", "hey joyce",
        "today voice", "my voice", "your voice", "the voice", "a voice",
        "hey ancient", "hey patient", "hey urgent", "okay agent",
        "page it", "wage it", "agent smith", "hey agent",
    ]

    for _ in range(n_samples // 4):
        adversarial.append(random.choice(similar_phrases))

    # Random common phrases
    common_phrases = [
        "hello", "hi there", "good morning", "how are you", "what time is it",
        "okay google", "alexa", "siri", "hey google", "hey siri",
        "computer", "assistant", "help me", "turn on", "turn off",
        "play music", "stop", "start", "pause", "resume", "next", "previous",
        "volume up", "volume down", "what's the weather", "set a timer",
    ]

    for _ in range(n_samples // 4):
        adversarial.append(random.choice(common_phrases))

    return adversarial[:n_samples]


def generate_samples(
    wake_word: str,
    output_dir: Path,
    n_positive: int = 500,
    n_negative: int = 500,
    n_test: int = None,
):
    """Generate positive and negative training samples."""
    # Default test samples to 10% of training size
    if n_test is None:
        n_test = max(20, n_positive // 10)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    pos_train_dir = output_dir / "positive_train"
    pos_test_dir = output_dir / "positive_test"
    neg_train_dir = output_dir / "negative_train"
    neg_test_dir = output_dir / "negative_test"

    for d in [pos_train_dir, pos_test_dir, neg_train_dir, neg_test_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating samples for wake word: '{wake_word}'")

    # Initialize Kokoro once
    from kokoro import KPipeline
    pipe = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')

    def generate_sample(text: str, output_path: Path, voice: str = None) -> bool:
        """Generate a single sample."""
        try:
            if voice is None:
                voice = random.choice(KOKORO_VOICES)

            audio_chunks = []
            for result in pipe(text, voice=voice):
                audio_chunks.append(result.output.audio.numpy())

            if not audio_chunks:
                return False

            audio = np.concatenate(audio_chunks)
            audio_16k = resample_audio(audio, KOKORO_SAMPLE_RATE, SAMPLE_RATE)
            audio_16k = audio_16k / np.max(np.abs(audio_16k) + 1e-8)
            audio_int16 = (audio_16k * 32767).astype(np.int16)

            wavfile.write(str(output_path), SAMPLE_RATE, audio_int16)
            return True
        except Exception as e:
            logger.warning(f"Sample generation failed: {e}")
            return False

    # Generate positive training samples
    logger.info(f"Generating {n_positive} positive training samples...")
    generated = 0
    attempts = 0
    pbar = tqdm(total=n_positive, desc="Positive train")
    while generated < n_positive and attempts < n_positive * 3:
        voice = KOKORO_VOICES[generated % len(KOKORO_VOICES)]
        output_path = pos_train_dir / f"{uuid.uuid4().hex}.wav"
        if generate_sample(wake_word, output_path, voice):
            generated += 1
            pbar.update(1)
        attempts += 1
    pbar.close()
    logger.info(f"Generated {generated} positive training samples")

    # Generate positive test samples
    logger.info(f"Generating {n_test} positive test samples...")
    generated = 0
    pbar = tqdm(total=n_test, desc="Positive test")
    for i in range(n_test):
        voice = KOKORO_VOICES[i % len(KOKORO_VOICES)]
        output_path = pos_test_dir / f"{uuid.uuid4().hex}.wav"
        if generate_sample(wake_word, output_path, voice):
            generated += 1
            pbar.update(1)
    pbar.close()

    # Generate negative/adversarial training samples
    logger.info(f"Generating {n_negative} negative training samples...")
    adversarial_texts = generate_adversarial_texts(wake_word, n_negative)
    generated = 0
    pbar = tqdm(total=n_negative, desc="Negative train")
    for i, text in enumerate(adversarial_texts):
        if generated >= n_negative:
            break
        voice = KOKORO_VOICES[i % len(KOKORO_VOICES)]
        output_path = neg_train_dir / f"{uuid.uuid4().hex}.wav"
        if generate_sample(text, output_path, voice):
            generated += 1
            pbar.update(1)
    pbar.close()

    # Generate negative test samples
    logger.info(f"Generating {n_test} negative test samples...")
    adversarial_texts = generate_adversarial_texts(wake_word, n_test)
    generated = 0
    pbar = tqdm(total=n_test, desc="Negative test")
    for i, text in enumerate(adversarial_texts):
        if generated >= n_test:
            break
        voice = KOKORO_VOICES[i % len(KOKORO_VOICES)]
        output_path = neg_test_dir / f"{uuid.uuid4().hex}.wav"
        if generate_sample(text, output_path, voice):
            generated += 1
            pbar.update(1)
    pbar.close()

    logger.info("Sample generation complete!")
    return True


def download_false_positive_data(output_dir: Path) -> Path:
    """Download LibriSpeech test-clean for false positive validation."""
    import urllib.request
    import tarfile

    output_dir.mkdir(parents=True, exist_ok=True)
    fp_data_dir = output_dir / "false_positive_data"

    if (fp_data_dir / "LibriSpeech").exists():
        logger.info("False positive data already exists")
        return fp_data_dir / "LibriSpeech" / "test-clean"

    fp_data_dir.mkdir(parents=True, exist_ok=True)

    # Download LibriSpeech test-clean (346MB)
    url = "https://www.openslr.org/resources/12/test-clean.tar.gz"
    tar_path = fp_data_dir / "test-clean.tar.gz"

    logger.info("Downloading LibriSpeech test-clean (~346MB)...")
    urllib.request.urlretrieve(url, tar_path)

    logger.info("Extracting...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(fp_data_dir)

    tar_path.unlink()  # Remove tar file

    return fp_data_dir / "LibriSpeech" / "test-clean"


def prepare_features(
    data_dir: Path,
    output_dir: Path,
    clip_duration_samples: int = 32000,
):
    """Compute openWakeWord features for training samples."""
    from openwakeword.utils import AudioFeatures
    import openwakeword

    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize feature extractor
    F = AudioFeatures(device='cpu', ncpu=4)

    # Process positive training samples
    pos_train_dir = data_dir / "positive_train"
    pos_clips = list(pos_train_dir.glob("*.wav"))

    if pos_clips:
        logger.info(f"Computing features for {len(pos_clips)} positive training samples...")
        pos_features = []
        for clip_path in tqdm(pos_clips, desc="Positive features"):
            try:
                sr, audio = wavfile.read(clip_path)
                if sr != SAMPLE_RATE:
                    audio = resample_audio(audio.astype(np.float32), sr, SAMPLE_RATE)

                # Pad or trim to fixed length
                if len(audio) < clip_duration_samples:
                    audio = np.pad(audio, (0, clip_duration_samples - len(audio)))
                else:
                    audio = audio[:clip_duration_samples]

                # Get features
                features = F.embed_clips(audio[np.newaxis, :].astype(np.int16))
                pos_features.append(features[0])
            except Exception as e:
                logger.warning(f"Failed to process {clip_path}: {e}")

        pos_features = np.array(pos_features)
        np.save(output_dir / "positive_features_train.npy", pos_features)
        logger.info(f"Saved positive features: {pos_features.shape}")

    # Process positive test samples
    pos_test_dir = data_dir / "positive_test"
    pos_test_clips = list(pos_test_dir.glob("*.wav"))

    if pos_test_clips:
        logger.info(f"Computing features for {len(pos_test_clips)} positive test samples...")
        pos_test_features = []
        for clip_path in tqdm(pos_test_clips, desc="Positive test features"):
            try:
                sr, audio = wavfile.read(clip_path)
                if sr != SAMPLE_RATE:
                    audio = resample_audio(audio.astype(np.float32), sr, SAMPLE_RATE)

                if len(audio) < clip_duration_samples:
                    audio = np.pad(audio, (0, clip_duration_samples - len(audio)))
                else:
                    audio = audio[:clip_duration_samples]

                features = F.embed_clips(audio[np.newaxis, :].astype(np.int16))
                pos_test_features.append(features[0])
            except Exception as e:
                logger.warning(f"Failed to process {clip_path}: {e}")

        pos_test_features = np.array(pos_test_features)
        np.save(output_dir / "positive_features_test.npy", pos_test_features)
        logger.info(f"Saved positive test features: {pos_test_features.shape}")

    # Process negative training samples
    neg_train_dir = data_dir / "negative_train"
    neg_clips = list(neg_train_dir.glob("*.wav"))

    if neg_clips:
        logger.info(f"Computing features for {len(neg_clips)} negative training samples...")
        neg_features = []
        for clip_path in tqdm(neg_clips, desc="Negative features"):
            try:
                sr, audio = wavfile.read(clip_path)
                if sr != SAMPLE_RATE:
                    audio = resample_audio(audio.astype(np.float32), sr, SAMPLE_RATE)

                if len(audio) < clip_duration_samples:
                    audio = np.pad(audio, (0, clip_duration_samples - len(audio)))
                else:
                    audio = audio[:clip_duration_samples]

                features = F.embed_clips(audio[np.newaxis, :].astype(np.int16))
                neg_features.append(features[0])
            except Exception as e:
                logger.warning(f"Failed to process {clip_path}: {e}")

        neg_features = np.array(neg_features)
        np.save(output_dir / "negative_features_train.npy", neg_features)
        logger.info(f"Saved negative features: {neg_features.shape}")

    # Process negative test samples
    neg_test_dir = data_dir / "negative_test"
    neg_test_clips = list(neg_test_dir.glob("*.wav"))

    if neg_test_clips:
        logger.info(f"Computing features for {len(neg_test_clips)} negative test samples...")
        neg_test_features = []
        for clip_path in tqdm(neg_test_clips, desc="Negative test features"):
            try:
                sr, audio = wavfile.read(clip_path)
                if sr != SAMPLE_RATE:
                    audio = resample_audio(audio.astype(np.float32), sr, SAMPLE_RATE)

                if len(audio) < clip_duration_samples:
                    audio = np.pad(audio, (0, clip_duration_samples - len(audio)))
                else:
                    audio = audio[:clip_duration_samples]

                features = F.embed_clips(audio[np.newaxis, :].astype(np.int16))
                neg_test_features.append(features[0])
            except Exception as e:
                logger.warning(f"Failed to process {clip_path}: {e}")

        neg_test_features = np.array(neg_test_features)
        np.save(output_dir / "negative_features_test.npy", neg_test_features)
        logger.info(f"Saved negative test features: {neg_test_features.shape}")

    return True


class WakeWordModel(torch.nn.Module):
    """Simple DNN model for wake word detection matching openWakeWord architecture."""

    def __init__(self, input_shape: tuple, layer_dim: int = 128, n_blocks: int = 1):
        super().__init__()

        self.input_shape = input_shape
        flat_size = input_shape[0] * input_shape[1]

        layers = [
            torch.nn.Flatten(),
            torch.nn.Linear(flat_size, layer_dim),
            torch.nn.LayerNorm(layer_dim),
            torch.nn.ReLU(),
        ]

        for _ in range(n_blocks):
            layers.extend([
                torch.nn.Linear(layer_dim, layer_dim),
                torch.nn.LayerNorm(layer_dim),
                torch.nn.ReLU(),
            ])

        layers.extend([
            torch.nn.Linear(layer_dim, 1),
            torch.nn.Sigmoid(),
        ])

        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def train_model(
    model_name: str,
    features_dir: Path,
    output_dir: Path,
    steps: int = 50000,
):
    """Train the wake word model."""
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Training model: {model_name}")

    # Load features
    pos_features_train = np.load(features_dir / "positive_features_train.npy")
    neg_features_train = np.load(features_dir / "negative_features_train.npy")
    pos_features_test = np.load(features_dir / "positive_features_test.npy")
    neg_features_test = np.load(features_dir / "negative_features_test.npy")

    logger.info(f"Positive train: {pos_features_train.shape}")
    logger.info(f"Negative train: {neg_features_train.shape}")
    logger.info(f"Positive test: {pos_features_test.shape}")
    logger.info(f"Negative test: {neg_features_test.shape}")

    # Determine input shape from features
    input_shape = pos_features_train.shape[1:]
    logger.info(f"Input shape: {input_shape}")

    # Create model
    model = WakeWordModel(
        input_shape=input_shape,
        layer_dim=128,
        n_blocks=1,
    )

    logger.info("Model architecture:")
    logger.info(model)

    # Prepare data loaders
    # Combine and create labels
    X_train = np.vstack([pos_features_train, neg_features_train])
    y_train = np.hstack([
        np.ones(len(pos_features_train)),
        np.zeros(len(neg_features_train))
    ]).astype(np.float32)

    X_test = np.vstack([pos_features_test, neg_features_test])
    y_test = np.hstack([
        np.ones(len(pos_features_test)),
        np.zeros(len(neg_features_test))
    ]).astype(np.float32)

    # Shuffle training data
    perm = np.random.permutation(len(X_train))
    X_train = X_train[perm]
    y_train = y_train[perm]

    # Create PyTorch datasets
    train_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train)
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0
    )

    val_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(y_test)
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=len(X_test),
        shuffle=False
    )

    # Training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    criterion = torch.nn.BCELoss()

    best_val_loss = float('inf')
    best_model_state = None

    logger.info(f"Training for {steps} steps...")

    step = 0
    epoch = 0
    pbar = tqdm(total=steps, desc="Training")

    while step < steps:
        epoch += 1
        for batch_x, batch_y in train_loader:
            if step >= steps:
                break

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            scheduler.step()

            step += 1
            pbar.update(1)

            # Validate periodically
            if step % 500 == 0:
                model.eval()
                with torch.no_grad():
                    for val_x, val_y in val_loader:
                        val_x = val_x.to(device)
                        val_y = val_y.to(device)
                        val_outputs = model(val_x)
                        val_loss = criterion(val_outputs.squeeze(), val_y).item()

                        # Compute accuracy
                        preds = (val_outputs.squeeze() > 0.5).float()
                        acc = (preds == val_y).float().mean().item()

                        # Compute recall (true positives / actual positives)
                        tp = ((preds == 1) & (val_y == 1)).sum().float()
                        fn = ((preds == 0) & (val_y == 1)).sum().float()
                        recall = (tp / (tp + fn + 1e-8)).item()

                        # Compute false positive rate
                        fp = ((preds == 1) & (val_y == 0)).sum().float()
                        tn = ((preds == 0) & (val_y == 0)).sum().float()
                        fpr = (fp / (fp + tn + 1e-8)).item()

                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'val_loss': f'{val_loss:.4f}',
                    'acc': f'{acc:.3f}',
                    'recall': f'{recall:.3f}',
                    'fpr': f'{fpr:.3f}'
                })

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

                model.train()

    pbar.close()

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Final evaluation
    model.eval()
    with torch.no_grad():
        for val_x, val_y in val_loader:
            val_x = val_x.to(device)
            val_y = val_y.to(device)
            val_outputs = model(val_x)

            preds = (val_outputs.squeeze() > 0.5).float()
            acc = (preds == val_y).float().mean().item()

            tp = ((preds == 1) & (val_y == 1)).sum().float()
            fn = ((preds == 0) & (val_y == 1)).sum().float()
            recall = (tp / (tp + fn + 1e-8)).item()

            fp = ((preds == 1) & (val_y == 0)).sum().float()
            tn = ((preds == 0) & (val_y == 0)).sum().float()
            fpr = (fp / (fp + tn + 1e-8)).item()

    logger.info(f"\nFinal Results:")
    logger.info(f"  Accuracy: {acc:.3f}")
    logger.info(f"  Recall: {recall:.3f}")
    logger.info(f"  False Positive Rate: {fpr:.3f}")

    # Export to ONNX
    onnx_path = output_dir / f"{model_name}.onnx"
    logger.info(f"Exporting model to {onnx_path}")

    model.to("cpu")
    dummy_input = torch.rand(1, *input_shape)
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        opset_version=13,
        input_names=['input'],
        output_names=[model_name],
        dynamic_axes={'input': {0: 'batch_size'}}
    )

    logger.info(f"Model exported to {onnx_path}")

    # Save training info
    info = {
        "model_name": model_name,
        "input_shape": list(input_shape),
        "accuracy": acc,
        "recall": recall,
        "false_positive_rate": fpr,
        "steps": steps,
    }
    with open(output_dir / f"{model_name}_info.json", "w") as f:
        json.dump(info, f, indent=2)

    return onnx_path


def main():
    parser = argparse.ArgumentParser(description="Train custom wake word models")
    parser.add_argument("--wake-word", type=str, required=True,
                       help="Wake word phrase (e.g., 'hey voice')")
    parser.add_argument("--generate-samples", action="store_true",
                       help="Generate training samples")
    parser.add_argument("--prepare-features", action="store_true",
                       help="Compute features from samples")
    parser.add_argument("--train", action="store_true",
                       help="Train the model")
    parser.add_argument("--full", action="store_true",
                       help="Run full pipeline (generate + features + train)")
    parser.add_argument("--n-positive", type=int, default=500,
                       help="Number of positive samples (default: 500)")
    parser.add_argument("--n-negative", type=int, default=500,
                       help="Number of negative samples (default: 500)")
    parser.add_argument("--steps", type=int, default=5000,
                       help="Training steps (default: 5000)")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory")

    args = parser.parse_args()

    # Create model name from wake word
    model_name = args.wake_word.lower().replace(" ", "_")

    # Setup directories
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = DATA_DIR / model_name

    samples_dir = output_dir / "samples"
    features_dir = output_dir / "features"

    # Run requested operations
    if args.full:
        args.generate_samples = True
        args.prepare_features = True
        args.train = True

    if args.generate_samples:
        generate_samples(
            wake_word=args.wake_word,
            output_dir=samples_dir,
            n_positive=args.n_positive,
            n_negative=args.n_negative,
        )

    if args.prepare_features:
        prepare_features(
            data_dir=samples_dir,
            output_dir=features_dir,
        )

    if args.train:
        onnx_path = train_model(
            model_name=model_name,
            features_dir=features_dir,
            output_dir=MODELS_DIR,
            steps=args.steps,
        )

        logger.info(f"\nTraining complete!")
        logger.info(f"Model saved to: {onnx_path}")
        logger.info(f"\nTo use the model, copy it to models/wakeword/ and update config.")


if __name__ == "__main__":
    main()
