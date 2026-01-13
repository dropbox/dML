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
Custom Wake Word Model Training Script

Trains a commercial wake word model for "Hey Agent" using:
1. Kokoro TTS for synthetic data generation (Apache 2.0)
2. OpenWakeWord embedding model architecture
3. MLX for training on Apple Silicon

This produces an ONNX model that we fully own and can use commercially.

License: Apache 2.0 (all components commercially licensed)
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Audio processing
try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    import scipy.signal as signal
except ImportError:
    signal = None

# Check if Piper is available
PIPER_AVAILABLE = False
try:
    from piper import PiperVoice
    PIPER_AVAILABLE = True
except ImportError:
    pass

# Check for MLX
MLX_AVAILABLE = False
try:
    import mlx.core as mx
    import mlx.nn as nn
    MLX_AVAILABLE = True
except ImportError:
    pass


# Wake word phrases and variations
WAKE_PHRASES = [
    "hey agent",
    "hey agent!",
    "Hey agent",
    "Hey Agent",
    "Hey Agent!",
]

# Adversarial phrases (phonetically similar but NOT wake word)
ADVERSARIAL_PHRASES = [
    "hey agency",
    "hey ancient",
    "hay agent",
    "hey patient",
    "hey urgent",
    "hey accent",
    "hey argent",
    "they agent",
    "day agent",
    "say agent",
    "pay agent",
    "hey sergeant",
    "hey regent",
]

# Negative phrases (random speech)
NEGATIVE_PHRASES = [
    "Hello world",
    "How are you today",
    "What time is it",
    "Nice to meet you",
    "Thank you very much",
    "Good morning",
    "Good evening",
    "See you later",
    "Have a nice day",
    "The weather is nice",
    "I need help with this",
    "Can you assist me",
    "Please wait a moment",
    "That sounds great",
    "I understand",
    "Let me think about it",
    "What do you mean",
    "Could you repeat that",
    "I don't understand",
    "Yes please",
    "No thank you",
    "Maybe later",
    "Absolutely",
    "Definitely not",
]


def download_piper_voice(voice_name: str = "en_US-lessac-medium") -> Path:
    """Download a Piper voice model if not already cached."""
    from huggingface_hub import hf_hub_download

    # Map voice names to HuggingFace paths
    voice_map = {
        "en_US-lessac-medium": ("rhasspy/piper-voices", "en/en_US/lessac/medium/en_US-lessac-medium.onnx"),
        "en_US-amy-medium": ("rhasspy/piper-voices", "en/en_US/amy/medium/en_US-amy-medium.onnx"),
        "en_US-ryan-medium": ("rhasspy/piper-voices", "en/en_US/ryan/medium/en_US-ryan-medium.onnx"),
        "en_GB-alan-medium": ("rhasspy/piper-voices", "en/en_GB/alan/medium/en_GB-alan-medium.onnx"),
        "en_GB-alba-medium": ("rhasspy/piper-voices", "en/en_GB/alba/medium/en_GB-alba-medium.onnx"),
    }

    if voice_name not in voice_map:
        voice_name = "en_US-lessac-medium"

    repo_id, file_path = voice_map[voice_name]
    json_path = file_path.replace(".onnx", ".onnx.json")

    # Download model and config
    model_path = hf_hub_download(repo_id=repo_id, filename=file_path)
    _config_path = hf_hub_download(repo_id=repo_id, filename=json_path)  # Config downloaded alongside model

    return Path(model_path)


def generate_synthetic_data(
    output_dir: Path,
    num_positive: int = 1000,
    num_adversarial: int = 500,
    num_negative: int = 2000,
    sample_rate: int = 16000,
    voices: Optional[List[str]] = None,
) -> dict:
    """
    Generate synthetic audio data using Piper TTS.

    Args:
        output_dir: Directory to save generated audio
        num_positive: Number of positive wake word samples
        num_adversarial: Number of adversarial (similar but wrong) samples
        num_negative: Number of negative (random speech) samples
        sample_rate: Target sample rate (16kHz for wake word)
        voices: List of voice names to use (cycles through)

    Returns:
        Dict with paths to generated data
    """
    if not PIPER_AVAILABLE:
        raise ImportError("Piper TTS not available. Install with: pip install piper-tts")

    # Create output directories
    positive_dir = output_dir / "positive"
    adversarial_dir = output_dir / "adversarial"
    negative_dir = output_dir / "negative"

    for d in [positive_dir, adversarial_dir, negative_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Default Piper voices
    DEFAULT_VOICES = [
        "en_US-lessac-medium",
        "en_US-amy-medium",
        "en_US-ryan-medium",
        "en_GB-alan-medium",
        "en_GB-alba-medium",
    ]

    if voices is None:
        voices = DEFAULT_VOICES

    # Load Piper voices
    print("Loading Piper TTS voices...")
    voice_models = {}
    for voice_name in voices:
        try:
            model_path = download_piper_voice(voice_name)
            voice_models[voice_name] = PiperVoice.load(str(model_path))
            print(f"  Loaded: {voice_name}")
        except Exception as e:
            print(f"  Failed to load {voice_name}: {e}")

    if not voice_models:
        raise RuntimeError("No voices loaded")

    voice_list = list(voice_models.values())
    _voice_names = list(voice_models.keys())  # Names available for logging if needed
    print(f"Using {len(voice_list)} voices")

    stats = {
        "positive": 0,
        "adversarial": 0,
        "negative": 0,
        "errors": 0,
    }

    def synthesize_with_piper(text: str, voice_idx: int) -> Optional[np.ndarray]:
        """Synthesize audio using Piper."""
        voice = voice_list[voice_idx % len(voice_list)]
        try:
            # Piper returns AudioChunk objects
            chunks = list(voice.synthesize(text))
            if not chunks:
                return None

            # Concatenate all audio chunks
            audio_arrays = [chunk.audio_float_array for chunk in chunks]
            audio = np.concatenate(audio_arrays) if len(audio_arrays) > 1 else audio_arrays[0]

            # Get sample rate from chunk
            piper_sr = chunks[0].sample_rate

            # Resample to 16kHz if needed
            if piper_sr != sample_rate:
                audio = resample_audio(audio, piper_sr, sample_rate)

            return audio.astype(np.float32)
        except Exception as e:
            print(f"    Synthesis error: {e}")
            return None

    # Generate positive samples
    print(f"\nGenerating {num_positive} positive samples...")
    for i in range(num_positive):
        try:
            phrase = random.choice(WAKE_PHRASES)
            audio = synthesize_with_piper(phrase, i)

            if audio is not None and len(audio) > 0:
                # Add slight augmentation (volume variation)
                volume = random.uniform(0.7, 1.0)
                audio = audio * volume

                # Save
                out_path = positive_dir / f"positive_{i:05d}.wav"
                sf.write(str(out_path), audio, sample_rate)
                stats["positive"] += 1

            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{num_positive} positive samples")

        except Exception as e:
            stats["errors"] += 1
            if stats["errors"] < 5:
                print(f"  Error generating positive sample {i}: {e}")

    # Generate adversarial samples
    print(f"\nGenerating {num_adversarial} adversarial samples...")
    for i in range(num_adversarial):
        try:
            phrase = random.choice(ADVERSARIAL_PHRASES)
            audio = synthesize_with_piper(phrase, i)

            if audio is not None and len(audio) > 0:
                volume = random.uniform(0.7, 1.0)
                audio = audio * volume

                out_path = adversarial_dir / f"adversarial_{i:05d}.wav"
                sf.write(str(out_path), audio, sample_rate)
                stats["adversarial"] += 1

            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{num_adversarial} adversarial samples")

        except Exception:
            stats["errors"] += 1

    # Generate negative samples
    print(f"\nGenerating {num_negative} negative samples...")
    for i in range(num_negative):
        try:
            phrase = random.choice(NEGATIVE_PHRASES)
            audio = synthesize_with_piper(phrase, i)

            if audio is not None and len(audio) > 0:
                volume = random.uniform(0.7, 1.0)
                audio = audio * volume

                out_path = negative_dir / f"negative_{i:05d}.wav"
                sf.write(str(out_path), audio, sample_rate)
                stats["negative"] += 1

            if (i + 1) % 200 == 0:
                print(f"  Generated {i + 1}/{num_negative} negative samples")

        except Exception:
            stats["errors"] += 1

    print("\nGeneration complete:")
    print(f"  Positive: {stats['positive']}")
    print(f"  Adversarial: {stats['adversarial']}")
    print(f"  Negative: {stats['negative']}")
    print(f"  Errors: {stats['errors']}")

    return {
        "positive_dir": str(positive_dir),
        "adversarial_dir": str(adversarial_dir),
        "negative_dir": str(negative_dir),
        "stats": stats,
    }


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to target sample rate."""
    if orig_sr == target_sr:
        return audio

    if signal is not None:
        # Use scipy for resampling
        num_samples = int(len(audio) * target_sr / orig_sr)
        return signal.resample(audio, num_samples).astype(np.float32)
    else:
        # Simple linear interpolation
        ratio = target_sr / orig_sr
        new_length = int(len(audio) * ratio)
        x_old = np.arange(len(audio))
        x_new = np.linspace(0, len(audio) - 1, new_length)
        return np.interp(x_new, x_old, audio).astype(np.float32)


# ============== AUDIO AUGMENTATION ==============

def augment_audio(
    audio: np.ndarray,
    sample_rate: int = 16000,
    augment_prob: float = 0.5,
) -> np.ndarray:
    """
    Apply random audio augmentations.

    Augmentations:
    - Volume variation
    - Time stretch (slight speed changes)
    - Pitch shift
    - Add noise
    - Time shift
    """
    if random.random() > augment_prob:
        return audio

    audio = audio.copy()

    # Volume variation (always applied)
    volume = random.uniform(0.6, 1.0)
    audio = audio * volume

    # Time stretch (10% of the time)
    if random.random() < 0.1:
        try:
            import librosa
            rate = random.uniform(0.9, 1.1)
            audio = librosa.effects.time_stretch(audio, rate=rate)
        except Exception:
            pass

    # Pitch shift (10% of the time)
    if random.random() < 0.1:
        try:
            import librosa
            n_steps = random.uniform(-2, 2)
            audio = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=n_steps)
        except Exception:
            pass

    # Add noise (30% of the time)
    if random.random() < 0.3:
        noise_level = random.uniform(0.002, 0.01)
        noise = np.random.randn(len(audio)) * noise_level
        audio = audio + noise

    # Time shift (20% of the time) - shift signal by small amount
    if random.random() < 0.2:
        shift = int(sample_rate * random.uniform(-0.1, 0.1))  # Up to 100ms shift
        if shift > 0:
            audio = np.concatenate([np.zeros(shift), audio[:-shift]])
        elif shift < 0:
            audio = np.concatenate([audio[-shift:], np.zeros(-shift)])

    return audio.astype(np.float32)


def load_audio_files(directory: Path, max_files: Optional[int] = None) -> List[np.ndarray]:
    """Load all WAV files from a directory."""
    audio_files = sorted(directory.glob("*.wav"))
    if max_files:
        audio_files = audio_files[:max_files]

    audio_data = []
    for f in audio_files:
        try:
            audio, sr = sf.read(str(f))
            audio_data.append(audio.astype(np.float32))
        except Exception:
            pass

    return audio_data


def compute_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int = 16000,
    n_fft: int = 512,
    hop_length: int = 160,
    n_mels: int = 32,
    use_librosa: bool = True,
) -> np.ndarray:
    """
    Compute mel spectrogram features.

    Uses librosa if available for better quality mel filterbanks.
    Falls back to scipy-based computation.
    """
    if use_librosa:
        try:
            import librosa
            # Librosa mel spectrogram - better quality filterbanks
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                fmin=20,
                fmax=8000,  # Focus on speech frequencies
            )
            # Convert to log scale (dB)
            mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            # Normalize to [-1, 1] range
            mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-10) * 2 - 1
            return mel_spec.T  # Shape: [time, n_mels]
        except ImportError:
            pass

    # Fallback: Simple mel spectrogram using scipy
    if signal is None:
        raise ImportError("scipy or librosa required for mel spectrogram computation")

    # Compute STFT
    f, t, stft_result = signal.stft(audio, sample_rate, nperseg=n_fft, noverlap=n_fft - hop_length)
    power = np.abs(stft_result) ** 2

    # Create mel filterbank (simplified)
    mel_freqs = 700 * (10 ** (np.linspace(0, 1, n_mels + 2) * 2.595) - 1)
    mel_freqs = np.clip(mel_freqs, 0, sample_rate / 2)

    mel_filterbank = np.zeros((n_mels, len(f)))
    for i in range(n_mels):
        low = mel_freqs[i]
        center = mel_freqs[i + 1]
        high = mel_freqs[i + 2]

        for j, freq in enumerate(f):
            if low <= freq < center:
                mel_filterbank[i, j] = (freq - low) / (center - low + 1e-10)
            elif center <= freq < high:
                mel_filterbank[i, j] = (high - freq) / (high - center + 1e-10)

    # Apply filterbank
    mel_spec = np.dot(mel_filterbank, power)

    # Log compression
    mel_spec = np.log(mel_spec + 1e-10)

    return mel_spec.T  # Shape: [time, n_mels]


def create_training_data(
    positive_dir: Path,
    adversarial_dir: Path,
    negative_dir: Path,
    max_frames: int = 76,  # Match OpenWakeWord embedding input
    n_mels: int = 32,
    augment: bool = False,
    augment_factor: int = 3,  # How many augmented copies of positive samples
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create training dataset from audio files.

    Args:
        positive_dir: Directory with positive (wake word) samples
        adversarial_dir: Directory with adversarial (similar but wrong) samples
        negative_dir: Directory with negative (random speech) samples
        max_frames: Number of time frames to use
        n_mels: Number of mel bands
        augment: Whether to apply data augmentation
        augment_factor: Number of augmented copies per positive sample

    Returns:
        X: Features array [N, max_frames, n_mels]
        y: Labels array [N] (1 = wake word, 0 = not wake word)
    """
    print("Loading audio files...")

    positive_audio = load_audio_files(positive_dir)
    adversarial_audio = load_audio_files(adversarial_dir)
    negative_audio = load_audio_files(negative_dir)

    print(f"  Positive: {len(positive_audio)} files")
    print(f"  Adversarial: {len(adversarial_audio)} files")
    print(f"  Negative: {len(negative_audio)} files")

    X_list = []
    y_list = []

    def process_audio(audio, label, apply_augment=False):
        """Process a single audio sample."""
        try:
            if apply_augment:
                audio = augment_audio(audio, augment_prob=0.8)
            mel = compute_mel_spectrogram(audio, n_mels=n_mels)
            # Pad or truncate to max_frames
            if mel.shape[0] < max_frames:
                mel = np.pad(mel, ((0, max_frames - mel.shape[0]), (0, 0)))
            else:
                mel = mel[:max_frames]
            return mel, label
        except Exception:
            return None, None

    # Process positive samples with augmentation
    print("Computing features for positive samples...")
    for audio in positive_audio:
        # Original sample
        mel, label = process_audio(audio, 1, apply_augment=False)
        if mel is not None:
            X_list.append(mel)
            y_list.append(label)

        # Augmented samples (if enabled)
        if augment:
            for _ in range(augment_factor):
                mel, label = process_audio(audio, 1, apply_augment=True)
                if mel is not None:
                    X_list.append(mel)
                    y_list.append(label)

    # Process adversarial as negative (with light augmentation)
    print("Computing features for adversarial samples...")
    for audio in adversarial_audio:
        mel, label = process_audio(audio, 0, apply_augment=False)
        if mel is not None:
            X_list.append(mel)
            y_list.append(label)

        if augment:
            # Only 1 augmented copy for adversarial
            mel, label = process_audio(audio, 0, apply_augment=True)
            if mel is not None:
                X_list.append(mel)
                y_list.append(label)

    # Process negative samples (with light augmentation)
    print("Computing features for negative samples...")
    for audio in negative_audio:
        mel, label = process_audio(audio, 0, apply_augment=False)
        if mel is not None:
            X_list.append(mel)
            y_list.append(label)

        if augment:
            # Only 1 augmented copy for negatives
            mel, label = process_audio(audio, 0, apply_augment=True)
            if mel is not None:
                X_list.append(mel)
                y_list.append(label)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    print(f"Total samples: {len(X)}, Positive: {sum(y)}, Negative: {len(y) - sum(y)}")

    return X, y


class WakeWordClassifier(nn.Module):
    """Simple wake word classifier in MLX (FC baseline)."""

    def __init__(self, input_frames: int = 76, input_mels: int = 32, hidden_dim: int = 128):
        super().__init__()

        input_dim = input_frames * input_mels

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(0.3)

    def __call__(self, x):
        # Flatten input: [batch, frames, mels] -> [batch, frames * mels]
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)

        x = mx.maximum(self.fc1(x), 0)  # ReLU
        x = self.dropout(x)
        x = mx.maximum(self.fc2(x), 0)  # ReLU
        x = self.dropout(x)
        x = self.fc3(x)

        return mx.sigmoid(x)


class WakeWordCNN(nn.Module):
    """
    CNN-based wake word classifier.

    Architecture inspired by keyword spotting literature:
    - Conv layers extract temporal/spectral patterns
    - Global pooling handles variable-length input
    - Dense layer for classification
    """

    def __init__(self, input_frames: int = 76, input_mels: int = 32):
        super().__init__()

        # Store dimensions
        self.input_frames = input_frames
        self.input_mels = input_mels

        # Conv layers: [batch, time, mels] (MLX format)
        # MLX Conv1d signature: Conv1d(in_channels, out_channels, kernel_size, ...)
        self.conv1 = nn.Conv1d(in_channels=input_mels, out_channels=64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm(64)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm(64)

        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm(128)

        self.conv4 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm(128)

        # Dense layers after global pooling
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.4)

    def __call__(self, x):
        # Input: [batch, frames, mels]
        # MLX Conv1d expects: [batch, length, in_channels]
        # So input is already in correct format: [batch, frames, mels]
        # where frames is length and mels is in_channels

        # Conv block 1
        x = self.conv1(x)  # [batch, frames, 64]
        x = self.bn1(x)
        x = mx.maximum(x, 0)  # ReLU
        # Simple max pooling by factor of 2 over the length dimension
        x = x[:, ::2, :]  # Downsample by 2

        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = mx.maximum(x, 0)
        x = x[:, ::2, :]  # Downsample by 2

        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = mx.maximum(x, 0)
        x = x[:, ::2, :]  # Downsample by 2

        # Conv block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = mx.maximum(x, 0)

        # Global average pooling: [batch, time, channels] -> [batch, channels]
        x = mx.mean(x, axis=1)

        # Dense layers
        x = self.dropout(x)
        x = mx.maximum(self.fc1(x), 0)
        x = self.dropout(x)
        x = self.fc2(x)

        return mx.sigmoid(x)


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    model_type: str = "cnn",  # "cnn" or "fc"
) -> nn.Module:
    """
    Train wake word classifier using MLX.

    Args:
        model_type: "cnn" for CNN model (recommended), "fc" for baseline FC model
    """

    if not MLX_AVAILABLE:
        raise ImportError("MLX required for training")

    print(f"\nTraining {model_type.upper()} model on {len(X_train)} samples, validating on {len(X_val)} samples")

    # Select model architecture
    if model_type.lower() == "cnn":
        model = WakeWordCNN()
        print("  Architecture: CNN (4 conv layers + global pooling + FC)")
    else:
        model = WakeWordClassifier()
        print("  Architecture: FC (3 linear layers)")

    # Initialize optimizer
    import mlx.optimizers as optim
    optimizer = optim.Adam(learning_rate=learning_rate)

    def loss_fn(model, x, y):
        pred = model(x)
        # Binary cross entropy
        loss = -mx.mean(y * mx.log(pred + 1e-7) + (1 - y) * mx.log(1 - pred + 1e-7))
        return loss

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Convert validation set to MLX array
    X_val_mx = mx.array(X_val)

    best_val_acc = 0

    for epoch in range(epochs):
        # Shuffle training data
        indices = np.random.permutation(len(X_train))

        epoch_loss = 0
        num_batches = 0

        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i + batch_size].tolist()
            x_batch = mx.array(X_train[batch_idx])
            y_batch = mx.array(y_train[batch_idx].reshape(-1, 1).astype(np.float32))

            loss, grads = loss_and_grad(model, x_batch, y_batch)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            epoch_loss += float(loss)
            num_batches += 1

        # Validation (in eval mode for proper BatchNorm/Dropout behavior)
        model.eval()
        val_pred = model(X_val_mx)
        mx.eval(val_pred)
        val_pred_np = np.array(val_pred).flatten()
        val_acc = np.mean((val_pred_np > 0.5) == y_val)
        model.train()  # Back to training mode

        avg_loss = epoch_loss / num_batches

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}: Loss={avg_loss:.4f}, Val Acc={val_acc:.4f}")

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")

    # Return model in eval mode for inference
    model.eval()
    return model


def export_cnn_to_onnx(
    model: WakeWordCNN,
    output_path: Path,
    input_frames: int = 76,
    input_mels: int = 32,
):
    """
    Export CNN model to ONNX format.

    ONNX Conv1d expects input shape: [batch, channels, length]
    MLX Conv1d uses shape: [batch, length, channels]

    We handle this by transposing at input/output of conv layers.
    """
    try:
        import onnx
        from onnx import TensorProto, helper
    except ImportError:
        print("ONNX not available, skipping export")
        return

    print(f"\nExporting CNN model to {output_path}")

    # Get weights from MLX model - flatten nested dicts
    def flatten_params(params, prefix=""):
        flat = {}
        for k, v in params.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                flat.update(flatten_params(v, key))
            else:
                flat[key] = v
        return flat

    weights = flatten_params(dict(model.parameters()))

    # Create ONNX graph
    # Input: [batch, frames, mels] to match MLX format
    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, input_frames, input_mels]
    )
    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 1]
    )

    nodes = []
    initializers = []

    # Transpose input from [B, L, C] to [B, C, L] for ONNX Conv1d
    nodes.append(helper.make_node(
        "Transpose",
        inputs=["input"],
        outputs=["conv_input"],
        perm=[0, 2, 1]  # [B, L, C] -> [B, C, L]
    ))

    current_input = "conv_input"
    current_length = input_frames

    # Helper to add a conv block: Conv1d + BatchNorm + ReLU + MaxPool
    def add_conv_block(
        block_idx: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        input_name: str,
        pool: bool = True,
    ) -> tuple:
        """Add Conv1d + BatchNorm + ReLU + optional MaxPool block."""
        nonlocal nodes, initializers, current_length

        prefix = f"conv{block_idx}"
        bn_prefix = f"bn{block_idx}"

        # Conv1d weight: MLX stores [out, kernel, in], ONNX wants [out, in, kernel]
        conv_weight = np.array(weights[f"{prefix}.weight"])
        # MLX Conv1d weight shape is [out_channels, kernel_size, in_channels]
        # ONNX Conv expects [out_channels, in_channels, kernel_size]
        conv_weight = np.transpose(conv_weight, (0, 2, 1))

        initializers.append(helper.make_tensor(
            f"{prefix}_weight", TensorProto.FLOAT,
            conv_weight.shape, conv_weight.flatten().tolist()
        ))

        # Conv1d bias (if exists) - MLX doesn't have bias by default
        has_bias = f"{prefix}.bias" in weights
        if has_bias:
            conv_bias = np.array(weights[f"{prefix}.bias"])
            initializers.append(helper.make_tensor(
                f"{prefix}_bias", TensorProto.FLOAT,
                conv_bias.shape, conv_bias.flatten().tolist()
            ))

        # Padding for 'same' output (kernel_size // 2)
        pad = kernel_size // 2

        # Conv node
        conv_inputs = [input_name, f"{prefix}_weight"]
        if has_bias:
            conv_inputs.append(f"{prefix}_bias")

        nodes.append(helper.make_node(
            "Conv",
            inputs=conv_inputs,
            outputs=[f"{prefix}_out"],
            kernel_shape=[kernel_size],
            pads=[pad, pad],  # Same padding
            strides=[1],
        ))

        # BatchNorm: Y = (X - mean) / sqrt(var + eps) * scale + bias
        # MLX BatchNorm stores: weight (scale), bias, running_mean, running_var
        bn_scale = np.array(weights[f"{bn_prefix}.weight"])
        bn_bias = np.array(weights[f"{bn_prefix}.bias"])
        bn_mean = np.array(weights[f"{bn_prefix}.running_mean"])
        bn_var = np.array(weights[f"{bn_prefix}.running_var"])

        initializers.append(helper.make_tensor(
            f"{bn_prefix}_scale", TensorProto.FLOAT,
            bn_scale.shape, bn_scale.flatten().tolist()
        ))
        initializers.append(helper.make_tensor(
            f"{bn_prefix}_bias", TensorProto.FLOAT,
            bn_bias.shape, bn_bias.flatten().tolist()
        ))
        initializers.append(helper.make_tensor(
            f"{bn_prefix}_mean", TensorProto.FLOAT,
            bn_mean.shape, bn_mean.flatten().tolist()
        ))
        initializers.append(helper.make_tensor(
            f"{bn_prefix}_var", TensorProto.FLOAT,
            bn_var.shape, bn_var.flatten().tolist()
        ))

        nodes.append(helper.make_node(
            "BatchNormalization",
            inputs=[f"{prefix}_out", f"{bn_prefix}_scale", f"{bn_prefix}_bias",
                    f"{bn_prefix}_mean", f"{bn_prefix}_var"],
            outputs=[f"{bn_prefix}_out"],
            epsilon=1e-5,
        ))

        # ReLU
        nodes.append(helper.make_node(
            "Relu",
            inputs=[f"{bn_prefix}_out"],
            outputs=[f"relu{block_idx}_out"],
        ))

        output_name = f"relu{block_idx}_out"

        # Strided slice with step 2 (equivalent to x[:, ::2, :] in MLX)
        # NOTE: MLX uses x[:, ::2, :] which is SUBSAMPLING, not MaxPool!
        # ONNX Slice with steps=[2] achieves the same effect.
        # Data is in ONNX format [B, C, L], so we slice axis 2 (length)
        if pool:
            # Create constants for Slice: starts, ends, axes, steps
            slice_starts = np.array([0], dtype=np.int64)
            slice_ends = np.array([2147483647], dtype=np.int64)  # INT_MAX for "to end"
            slice_axes = np.array([2], dtype=np.int64)  # Slice along length axis (axis 2)
            slice_steps = np.array([2], dtype=np.int64)  # Step of 2

            initializers.append(helper.make_tensor(
                f"slice{block_idx}_starts", TensorProto.INT64,
                slice_starts.shape, slice_starts.tolist()
            ))
            initializers.append(helper.make_tensor(
                f"slice{block_idx}_ends", TensorProto.INT64,
                slice_ends.shape, slice_ends.tolist()
            ))
            initializers.append(helper.make_tensor(
                f"slice{block_idx}_axes", TensorProto.INT64,
                slice_axes.shape, slice_axes.tolist()
            ))
            initializers.append(helper.make_tensor(
                f"slice{block_idx}_steps", TensorProto.INT64,
                slice_steps.shape, slice_steps.tolist()
            ))

            nodes.append(helper.make_node(
                "Slice",
                inputs=[output_name, f"slice{block_idx}_starts", f"slice{block_idx}_ends",
                        f"slice{block_idx}_axes", f"slice{block_idx}_steps"],
                outputs=[f"pool{block_idx}_out"],
            ))
            output_name = f"pool{block_idx}_out"
            current_length = current_length // 2

        return output_name, out_channels

    # Conv block 1: 32 -> 64, k=5
    current_input, channels = add_conv_block(
        1, input_mels, 64, 5, current_input, pool=True
    )

    # Conv block 2: 64 -> 64, k=5
    current_input, channels = add_conv_block(
        2, 64, 64, 5, current_input, pool=True
    )

    # Conv block 3: 64 -> 128, k=3
    current_input, channels = add_conv_block(
        3, 64, 128, 3, current_input, pool=True
    )

    # Conv block 4: 128 -> 128, k=3 (no pool)
    current_input, channels = add_conv_block(
        4, 128, 128, 3, current_input, pool=False
    )

    # Global Average Pooling: [B, C, L] -> [B, C, 1] -> [B, C]
    nodes.append(helper.make_node(
        "GlobalAveragePool",
        inputs=[current_input],
        outputs=["gap_out"],
    ))

    # Squeeze to remove the last dimension: [B, C, 1] -> [B, C]
    # In ONNX opset 13+, axes is an input tensor, not an attribute
    squeeze_axes = np.array([2], dtype=np.int64)
    initializers.append(helper.make_tensor(
        "squeeze_axes", TensorProto.INT64,
        squeeze_axes.shape, squeeze_axes.tolist()
    ))
    nodes.append(helper.make_node(
        "Squeeze",
        inputs=["gap_out", "squeeze_axes"],
        outputs=["gap_squeezed"],
    ))

    # FC1: 128 -> 64 + ReLU
    fc1_weight = np.array(weights["fc1.weight"]).T  # [128, 64] -> [64, 128] for ONNX
    fc1_bias = np.array(weights["fc1.bias"])

    initializers.append(helper.make_tensor(
        "fc1_weight", TensorProto.FLOAT, fc1_weight.shape, fc1_weight.flatten().tolist()
    ))
    initializers.append(helper.make_tensor(
        "fc1_bias", TensorProto.FLOAT, fc1_bias.shape, fc1_bias.flatten().tolist()
    ))

    nodes.append(helper.make_node("MatMul", ["gap_squeezed", "fc1_weight"], ["fc1_mm"]))
    nodes.append(helper.make_node("Add", ["fc1_mm", "fc1_bias"], ["fc1_out"]))
    nodes.append(helper.make_node("Relu", ["fc1_out"], ["fc1_relu"]))

    # FC2: 64 -> 1 + Sigmoid
    fc2_weight = np.array(weights["fc2.weight"]).T
    fc2_bias = np.array(weights["fc2.bias"])

    initializers.append(helper.make_tensor(
        "fc2_weight", TensorProto.FLOAT, fc2_weight.shape, fc2_weight.flatten().tolist()
    ))
    initializers.append(helper.make_tensor(
        "fc2_bias", TensorProto.FLOAT, fc2_bias.shape, fc2_bias.flatten().tolist()
    ))

    nodes.append(helper.make_node("MatMul", ["fc1_relu", "fc2_weight"], ["fc2_mm"]))
    nodes.append(helper.make_node("Add", ["fc2_mm", "fc2_bias"], ["fc2_out"]))
    nodes.append(helper.make_node("Sigmoid", ["fc2_out"], ["output"]))

    # Create graph
    graph = helper.make_graph(
        nodes,
        "hey_agent_cnn_classifier",
        [input_tensor],
        [output_tensor],
        initializers
    )

    # Create model
    onnx_model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx_model.ir_version = 7

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(onnx_model, str(output_path))

    print(f"CNN model exported to {output_path}")

    # Verify the model
    try:
        onnx.checker.check_model(onnx_model)
        print("ONNX model validation: PASSED")
    except Exception as e:
        print(f"ONNX model validation: FAILED - {e}")


def export_to_onnx(
    model: nn.Module,
    output_path: Path,
    input_frames: int = 76,
    input_mels: int = 32,
    model_type: str = "fc",
):
    """Export trained model to ONNX format."""
    try:
        import onnx
        from onnx import TensorProto, helper
    except ImportError:
        print("ONNX not available, skipping export")
        return

    # Route to appropriate export function based on model type
    if model_type.lower() == "cnn" or isinstance(model, WakeWordCNN):
        export_cnn_to_onnx(model, output_path, input_frames, input_mels)
        return

    print(f"\nExporting FC model to {output_path}")

    # Get weights from MLX model - flatten nested dicts
    def flatten_params(params, prefix=""):
        flat = {}
        for k, v in params.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                flat.update(flatten_params(v, key))
            else:
                flat[key] = v
        return flat

    weights = flatten_params(dict(model.parameters()))

    # Create ONNX graph
    # Input: [batch, frames, mels]
    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, input_frames, input_mels]
    )
    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 1]
    )

    # Build nodes
    nodes = []
    initializers = []

    # Flatten
    nodes.append(helper.make_node(
        "Flatten",
        inputs=["input"],
        outputs=["flat"],
        axis=1
    ))

    # FC1: MatMul + Add + ReLU
    fc1_weight = np.array(weights["fc1.weight"]).T  # Transpose for ONNX format
    fc1_bias = np.array(weights["fc1.bias"])

    initializers.append(helper.make_tensor(
        "fc1_weight", TensorProto.FLOAT, fc1_weight.shape, fc1_weight.flatten().tolist()
    ))
    initializers.append(helper.make_tensor(
        "fc1_bias", TensorProto.FLOAT, fc1_bias.shape, fc1_bias.flatten().tolist()
    ))

    nodes.append(helper.make_node("MatMul", ["flat", "fc1_weight"], ["fc1_mm"]))
    nodes.append(helper.make_node("Add", ["fc1_mm", "fc1_bias"], ["fc1_out"]))
    nodes.append(helper.make_node("Relu", ["fc1_out"], ["relu1"]))

    # FC2
    fc2_weight = np.array(weights["fc2.weight"]).T
    fc2_bias = np.array(weights["fc2.bias"])

    initializers.append(helper.make_tensor(
        "fc2_weight", TensorProto.FLOAT, fc2_weight.shape, fc2_weight.flatten().tolist()
    ))
    initializers.append(helper.make_tensor(
        "fc2_bias", TensorProto.FLOAT, fc2_bias.shape, fc2_bias.flatten().tolist()
    ))

    nodes.append(helper.make_node("MatMul", ["relu1", "fc2_weight"], ["fc2_mm"]))
    nodes.append(helper.make_node("Add", ["fc2_mm", "fc2_bias"], ["fc2_out"]))
    nodes.append(helper.make_node("Relu", ["fc2_out"], ["relu2"]))

    # FC3 + Sigmoid
    fc3_weight = np.array(weights["fc3.weight"]).T
    fc3_bias = np.array(weights["fc3.bias"])

    initializers.append(helper.make_tensor(
        "fc3_weight", TensorProto.FLOAT, fc3_weight.shape, fc3_weight.flatten().tolist()
    ))
    initializers.append(helper.make_tensor(
        "fc3_bias", TensorProto.FLOAT, fc3_bias.shape, fc3_bias.flatten().tolist()
    ))

    nodes.append(helper.make_node("MatMul", ["relu2", "fc3_weight"], ["fc3_mm"]))
    nodes.append(helper.make_node("Add", ["fc3_mm", "fc3_bias"], ["fc3_out"]))
    nodes.append(helper.make_node("Sigmoid", ["fc3_out"], ["output"]))

    # Create graph
    graph = helper.make_graph(
        nodes,
        "hey_agent_classifier",
        [input_tensor],
        [output_tensor],
        initializers
    )

    # Create model
    onnx_model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx_model.ir_version = 7

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(onnx_model, str(output_path))

    print(f"Model exported to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train custom wake word model")
    parser.add_argument("--output-dir", type=Path, default=Path("models/wakeword/hey_agent"),
                       help="Output directory for model and data")
    parser.add_argument("--generate-data", action="store_true",
                       help="Generate synthetic training data")
    parser.add_argument("--num-positive", type=int, default=1000,
                       help="Number of positive samples to generate")
    parser.add_argument("--num-adversarial", type=int, default=500,
                       help="Number of adversarial samples to generate")
    parser.add_argument("--num-negative", type=int, default=2000,
                       help="Number of negative samples to generate")
    parser.add_argument("--train", action="store_true",
                       help="Train the model")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--model-type", type=str, default="cnn", choices=["cnn", "fc"],
                       help="Model architecture: cnn (recommended) or fc (baseline)")
    parser.add_argument("--augment", action="store_true",
                       help="Enable data augmentation (noise, pitch shift, time stretch)")
    parser.add_argument("--augment-factor", type=int, default=3,
                       help="Number of augmented copies per positive sample")
    parser.add_argument("--export-onnx", action="store_true",
                       help="Export model to ONNX format")

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = args.output_dir / "training_data"

    if args.generate_data:
        print("=" * 60)
        print("GENERATING SYNTHETIC TRAINING DATA")
        print("=" * 60)

        result = generate_synthetic_data(
            output_dir=data_dir,
            num_positive=args.num_positive,
            num_adversarial=args.num_adversarial,
            num_negative=args.num_negative,
        )

        # Save generation info
        with open(args.output_dir / "data_info.json", "w") as f:
            json.dump(result, f, indent=2)

    if args.train:
        print("=" * 60)
        print("TRAINING WAKE WORD MODEL")
        print("=" * 60)
        print(f"  Model type: {args.model_type.upper()}")
        print(f"  Augmentation: {'ENABLED' if args.augment else 'DISABLED'}")
        if args.augment:
            print(f"  Augment factor: {args.augment_factor}x positive samples")

        # Load or check data directories
        positive_dir = data_dir / "positive"
        adversarial_dir = data_dir / "adversarial"
        negative_dir = data_dir / "negative"

        if not positive_dir.exists():
            print("Error: Training data not found. Run with --generate-data first.")
            return

        # Create training data with optional augmentation
        X, y = create_training_data(
            positive_dir,
            adversarial_dir,
            negative_dir,
            augment=args.augment,
            augment_factor=args.augment_factor,
        )

        # Split into train/val
        indices = np.random.permutation(len(X))
        split = int(0.9 * len(X))
        train_idx = indices[:split]
        val_idx = indices[split:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Train with selected model type
        model = train_model(
            X_train, y_train, X_val, y_val,
            epochs=args.epochs,
            model_type=args.model_type,
        )

        # Save model weights using MLX save - flatten nested dicts
        weights_path = args.output_dir / "hey_agent_mlx.safetensors"

        def flatten_params(params, prefix=""):
            """Flatten nested parameter dict to single level."""
            flat = {}
            for k, v in params.items():
                key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    flat.update(flatten_params(v, key))
                else:
                    flat[key] = v
            return flat

        flat_weights = flatten_params(dict(model.parameters()))
        mx.save_safetensors(str(weights_path), flat_weights)
        print(f"Model weights saved to {weights_path}")

        if args.export_onnx:
            onnx_path = args.output_dir / "hey_agent.onnx"
            export_to_onnx(model, onnx_path, model_type=args.model_type)


if __name__ == "__main__":
    main()
