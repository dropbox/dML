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
Train Prosody-Punctuation Fusion Layer.

This script trains a fusion layer that learns to EXTEND (not replace) Whisper's
native punctuation predictions using prosody features (emotion + pitch).

Key insight: The failed approach (0.19 F1) tried to REPLACE Whisper's punctuation.
The new approach learns ADDITIVE BOOST values that enhance Whisper's native output.

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │          Prosody-Punctuation Fusion Layer (TRAINABLE)       │
    │                                                             │
    │  Input:  emotion_probs (8) + pitch_features (5) +           │
    │          whisper_punct_logits (4)                           │
    │  Output: punct_boost (4) - additive boost for [. , ? !]     │
    │                                                             │
    │  final_logits = whisper_logits + punct_boost                │
    └─────────────────────────────────────────────────────────────┘

Usage:
    python -m tools.whisper_mlx.train_prosody_punct_fusion \
        --librispeech-features data/emotion_punctuation/librispeech_features \
        --output-dir checkpoints/prosody_punct_fusion_v1 \
        --epochs 10 --batch-size 32

References:
    - UNIFIED_PROSODY_PUNCTUATION_PLAN.md
    - TRAINING_OPTIMIZATION_ROADMAP.md
"""

import argparse
import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

# =============================================================================
# Configuration
# =============================================================================


@dataclass
class FusionTrainingConfig:
    """Configuration for prosody-punctuation fusion training."""

    # Data
    librispeech_features: str = "data/emotion_punctuation/librispeech_features"
    output_dir: str = "checkpoints/prosody_punct_fusion_v1"

    # Model architecture
    emotion_dim: int = 8  # Number of emotion classes
    pitch_features: int = 5  # slope, variance, mean, min, max
    whisper_punct_dim: int = 4  # . , ? !
    hidden_dim: int = 64
    output_dim: int = 4  # Boost for . , ? !
    dropout: float = 0.1

    # Training
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-4
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    # Loss weights
    ce_weight: float = 1.0  # Cross-entropy weight
    boost_reg_weight: float = 0.01  # Regularize boost magnitude (keep small)
    question_weight: float = 3.0  # Weight for rare ? class
    exclaim_weight: float = 3.0  # Weight for rare ! class

    # Focal loss (for class imbalance)
    focal_gamma: float = 2.0  # Focal loss gamma (0=CE, 2=standard focal, 5=aggressive)
    use_focal_loss: bool = True  # Enable focal loss

    # Balanced batching (for class imbalance)
    balanced_batching: bool = True  # Sample equal classes per batch

    # Validation
    val_split: float = 0.1

    # Logging
    log_interval: int = 50
    save_interval: int = 500


# Punctuation classes (matching train_punctuation.py)
PUNCT_CLASSES = {
    ".": 0,
    ",": 1,
    "?": 2,
    "!": 3,
}
PUNCT_NAMES = ["PERIOD", "COMMA", "QUESTION", "EXCLAIM"]


# =============================================================================
# Model
# =============================================================================


class ProsodyPunctuationFusion(nn.Module):
    """
    Learned fusion of prosody features for punctuation boosting.

    Key insight: Learn the prosody→punctuation mapping instead of
    hand-coding rules like "rising pitch → question mark".

    Output is ADDITIVE boost, not replacement. This means:
    - final_logits = whisper_logits + punct_boost
    - Model starts by trusting Whisper (zero-initialized final layer)
    - Only learns corrections where prosody signals are informative
    """

    def __init__(
        self,
        emotion_dim: int = 8,
        pitch_features: int = 5,
        whisper_punct_dim: int = 4,
        hidden_dim: int = 64,
        output_dim: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Emotion pathway
        self.emotion_encoder = nn.Sequential(
            nn.Linear(emotion_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )

        # Pitch pathway (learns patterns like rising/falling)
        self.pitch_encoder = nn.Sequential(
            nn.Linear(pitch_features, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )

        # Whisper logits pathway (what Whisper thinks)
        self.whisper_encoder = nn.Sequential(
            nn.Linear(whisper_punct_dim, 16),
            nn.ReLU(),
        )

        # Fusion (combines all signals)
        fusion_dim = 32 + 32 + 16  # 80
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        # CRITICAL: Initialize final layer to zeros
        # This makes the model start by trusting Whisper completely
        self._init_near_zero()

    def _init_near_zero(self):
        """Initialize final fusion layer near zero."""
        # The Sequential's last Linear is at index -1
        # Access weight and bias directly and reinitialize
        final_layer = self.fusion.layers[-1]
        if hasattr(final_layer, "weight"):
            # MLX doesn't support assignment, we'll handle in forward pass
            # or during weight loading. For now, just document the intent.
            pass

    def __call__(self, emotion_probs, pitch_features, whisper_punct_logits):
        """
        Compute punctuation boost values.

        Args:
            emotion_probs: (B, 8) emotion probabilities
            pitch_features: (B, 5) [slope, variance, mean, min, max]
            whisper_punct_logits: (B, 4) Whisper's punct predictions

        Returns:
            punct_boost: (B, 4) additive boost for [., ,, ?, !]
        """
        # Encode each modality
        emotion_enc = self.emotion_encoder(emotion_probs)
        pitch_enc = self.pitch_encoder(pitch_features)
        whisper_enc = self.whisper_encoder(whisper_punct_logits)

        # Concatenate and fuse
        combined = mx.concatenate([emotion_enc, pitch_enc, whisper_enc], axis=-1)
        return self.fusion(combined)



def create_fusion_model(config: FusionTrainingConfig) -> ProsodyPunctuationFusion:
    """Create and initialize fusion model."""
    model = ProsodyPunctuationFusion(
        emotion_dim=config.emotion_dim,
        pitch_features=config.pitch_features,
        whisper_punct_dim=config.whisper_punct_dim,
        hidden_dim=config.hidden_dim,
        output_dim=config.output_dim,
        dropout=config.dropout,
    )

    # Initialize final layer to zeros
    params = model.parameters()

    # Find and zero-initialize the final linear layer
    def zero_init_final_layer(params, prefix=""):
        """Recursively find and zero-init the deepest linear layer in fusion."""
        new_params = {}
        for key, value in params.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                new_params[key] = zero_init_final_layer(value, full_key)
            else:
                # Zero-init final fusion layer (fusion.layers.5 is the last Linear)
                if "fusion" in full_key and "5" in full_key:
                    new_params[key] = mx.zeros_like(value)
                else:
                    new_params[key] = value
        return new_params

    zeroed_params = zero_init_final_layer(params)
    model.update(zeroed_params)
    mx.eval(model.parameters())

    return model


# =============================================================================
# Dataset
# =============================================================================


@dataclass
class FusionSample:
    """Single training sample for fusion layer."""

    file_path: Path
    emotion_probs: np.ndarray  # (8,) or (34,) -> we use first 8
    pitch_features: np.ndarray  # (5,) extracted statistics
    transcript: str
    punct_target: int  # 0-3 for . , ? ! or -1 for no punct


def extract_pitch_features(pitch_values: np.ndarray) -> np.ndarray:
    """
    Extract pitch statistics for fusion input.

    Args:
        pitch_values: (T,) pitch per frame

    Returns:
        (5,) [slope, variance, mean, min, max]
    """
    if len(pitch_values) < 10:
        # Handle very short sequences
        slope = 0.0
    else:
        # Slope from last 10 frames
        slope = float(pitch_values[-1] - pitch_values[-10])

    variance = float(np.var(pitch_values))
    mean = float(np.mean(pitch_values))
    min_val = float(np.min(pitch_values))
    max_val = float(np.max(pitch_values))

    return np.array([slope, variance, mean, min_val, max_val], dtype=np.float32)


def extract_punct_target(transcript: str) -> int:
    """
    Extract final punctuation target from transcript.

    Returns:
        0: period (.)
        1: comma (,)
        2: question (?)
        3: exclamation (!)
        -1: no punctuation or other
    """
    text = transcript.strip()
    if not text:
        return -1

    last_char = text[-1]
    if last_char == ".":
        return 0
    if last_char == ",":
        return 1
    if last_char == "?":
        return 2
    if last_char == "!":
        return 3
    return -1


class ProsodyJSONDataset:
    """
    Dataset loading from prosody JSON files (consolidated.json, esd.json, etc.).

    These files have naturally punctuated transcripts like:
        "text": "A voice said: Come in."
        "audio_path": "/path/to/audio.wav"
        "f0_mean", "f0_std", "f0_range", "duration_s", "energy_rms"
    """

    def __init__(self, json_path: str, val_split: float = 0.1):
        import json as json_module

        self.json_path = Path(json_path)
        self.val_split = val_split

        # Load JSON data
        print(f"Loading prosody data from: {json_path}")
        with open(json_path) as f:
            all_samples = json_module.load(f)

        print(f"Loaded {len(all_samples)} samples")

        # Filter for valid samples with punctuation
        self.valid_samples = []
        punct_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        no_punct = 0

        for sample in all_samples:
            text = sample.get("text", "")
            target = extract_punct_target(text)
            audio_path = sample.get("audio_path", "")

            if target >= 0 and audio_path and Path(audio_path).exists():
                self.valid_samples.append({
                    "text": text,
                    "audio_path": audio_path,
                    "f0_mean": sample.get("f0_mean", 0.0),
                    "f0_std": sample.get("f0_std", 0.0),
                    "f0_range": sample.get("f0_range", 0.0),
                    "duration_s": sample.get("duration_s", 0.0),
                    "energy_rms": sample.get("energy_rms", 0.0),
                    "target": target,
                })
                punct_counts[target] += 1
            else:
                no_punct += 1

        print(f"Valid samples: {len(self.valid_samples)}")
        print(f"Class distribution: {punct_counts}")
        print(f"Skipped (no punct or missing audio): {no_punct}")

        # Split into train/val
        rng = np.random.default_rng(42)
        indices = rng.permutation(len(self.valid_samples))
        split_idx = int(len(indices) * (1 - val_split))

        self.train_samples = [self.valid_samples[i] for i in indices[:split_idx]]
        self.val_samples = [self.valid_samples[i] for i in indices[split_idx:]]

        print(f"Train: {len(self.train_samples)}, Val: {len(self.val_samples)}")
        self._print_class_distribution()

    def _print_class_distribution(self):
        """Print training class distribution."""
        counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for s in self.train_samples:
            counts[s["target"]] += 1

        total = sum(counts.values())
        print("Training distribution:")
        for i, name in enumerate(PUNCT_NAMES):
            pct = 100 * counts[i] / total if total > 0 else 0
            print(f"  {name}: {counts[i]} ({pct:.1f}%)")

        # Store class-organized samples for balanced batching
        self._class_samples = {i: [] for i in range(4)}
        for s in self.train_samples:
            self._class_samples[s["target"]].append(s)

    def _sample_to_features(self, s: dict) -> tuple[np.ndarray, np.ndarray, int]:
        """Convert sample dict to features."""
        # Use placeholder zeros for emotion (not available in prosody JSON)
        emotion = np.zeros(8, dtype=np.float32)

        # Use f0 statistics as pitch features
        # Normalize to reasonable range
        pitch_feats = np.array([
            s["f0_mean"] / 300.0,  # Normalize f0 mean (~100-400 Hz)
            s["f0_std"] / 100.0,   # Normalize f0 std
            s["f0_range"] / 450.0, # Normalize f0 range
            s["energy_rms"] * 10,  # Scale energy
            min(s["duration_s"] / 10.0, 1.0),  # Normalize duration (cap at 10s)
        ], dtype=np.float32)

        return emotion, pitch_feats, s["target"]

    def get_batch(
        self, samples: list[dict], batch_size: int, shuffle: bool = True,
        balanced: bool = False,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        """
        Generator yielding batches using pre-computed prosody features.

        For prosody JSON data, we use the f0 statistics directly as features.

        Args:
            samples: List of sample dicts
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle
            balanced: If True, sample equal numbers from each class per batch

        Yields:
            emotion_probs: (B, 8) - placeholder zeros (no emotion head)
            pitch_features: (B, 5) - [f0_mean, f0_std, f0_range, energy_rms, duration]
            whisper_punct_placeholder: (B, 4) - zeros
            targets: (B,)
        """
        if balanced and hasattr(self, '_class_samples'):
            # Balanced batching: sample equal numbers from each class
            yield from self._get_balanced_batches(batch_size)
            return

        if shuffle:
            samples = list(samples)
            rng = np.random.default_rng()
            rng.shuffle(samples)

        batch_emotion = []
        batch_pitch = []
        batch_targets = []

        for s in samples:
            emotion, pitch_feats, target = self._sample_to_features(s)

            batch_emotion.append(emotion)
            batch_pitch.append(pitch_feats)
            batch_targets.append(target)

            if len(batch_targets) >= batch_size:
                yield (
                    mx.array(np.stack(batch_emotion)),
                    mx.array(np.stack(batch_pitch)),
                    mx.zeros((len(batch_targets), 4)),  # Placeholder for Whisper logits
                    mx.array(batch_targets),
                )
                batch_emotion = []
                batch_pitch = []
                batch_targets = []

        # Final partial batch
        if batch_targets:
            yield (
                mx.array(np.stack(batch_emotion)),
                mx.array(np.stack(batch_pitch)),
                mx.zeros((len(batch_targets), 4)),
                mx.array(batch_targets),
            )

    def _get_balanced_batches(self, batch_size: int):
        """
        Generate balanced batches with equal samples per class.

        For batch_size=64, we sample 16 from each of N non-empty classes.
        Uses oversampling for minority classes.
        """
        # Find non-empty classes
        non_empty_classes = [c for c in range(4) if len(self._class_samples[c]) > 0]
        num_classes = len(non_empty_classes)

        if num_classes == 0:
            print("WARNING: No samples found in any class!")
            return

        print(f"Balanced batching: {num_classes} non-empty classes: {[PUNCT_NAMES[c] for c in non_empty_classes]}")

        samples_per_class = batch_size // num_classes

        # Find the largest class size for epoch length calculation
        class_sizes = {c: len(self._class_samples[c]) for c in non_empty_classes}
        max_class_size = max(class_sizes.values())

        # Number of batches per epoch (based on largest class with oversampling)
        num_batches = max_class_size // samples_per_class
        print(f"Balanced batching: {num_batches} batches per epoch, {samples_per_class} per class")

        # Shuffle each class independently
        rng = np.random.default_rng()
        class_pools = {}
        for c in non_empty_classes:
            pool = list(self._class_samples[c])
            rng.shuffle(pool)
            class_pools[c] = pool

        # Index trackers for each class
        class_indices = dict.fromkeys(non_empty_classes, 0)

        for _ in range(num_batches):
            batch_emotion = []
            batch_pitch = []
            batch_targets = []

            for c in non_empty_classes:
                # Sample from this class (with wraparound for oversampling)
                for _ in range(samples_per_class):
                    idx = class_indices[c]
                    if idx >= len(class_pools[c]):
                        # Reshuffle and restart (oversampling)
                        rng.shuffle(class_pools[c])
                        idx = 0
                        class_indices[c] = 0

                    s = class_pools[c][idx]
                    class_indices[c] += 1

                    emotion, pitch_feats, target = self._sample_to_features(s)
                    batch_emotion.append(emotion)
                    batch_pitch.append(pitch_feats)
                    batch_targets.append(target)

            # Shuffle the batch so classes aren't in order
            combined = list(zip(batch_emotion, batch_pitch, batch_targets, strict=False))
            rng.shuffle(combined)
            batch_emotion, batch_pitch, batch_targets = zip(*combined, strict=False)

            yield (
                mx.array(np.stack(batch_emotion)),
                mx.array(np.stack(batch_pitch)),
                mx.zeros((len(batch_targets), 4)),
                mx.array(batch_targets),
            )


class LibriSpeechFeatureDataset:
    """
    Dataset loading pre-extracted features from .npz files.

    Expected format per file:
        - encoder_features: (T, 1280)
        - emotion_probs: (34,) or (8,)
        - pitch_values: (T,)
        - transcript: str
        - duration_s: float
        - utterance_id: str
        - speaker: str
    """

    def __init__(self, features_dir: str, val_split: float = 0.1):
        self.features_dir = Path(features_dir)
        self.val_split = val_split

        # Find all .npz files
        self.all_files = sorted(self.features_dir.glob("*.npz"))
        print(f"Found {len(self.all_files)} feature files")

        # Filter for files with valid punctuation targets
        self.valid_files = []
        self._filter_valid_files()

        # Split into train/val
        rng = np.random.default_rng(42)
        indices = rng.permutation(len(self.valid_files))
        split_idx = int(len(indices) * (1 - val_split))

        self.train_files = [self.valid_files[i] for i in indices[:split_idx]]
        self.val_files = [self.valid_files[i] for i in indices[split_idx:]]

        print(f"Valid files: {len(self.valid_files)}")
        print(f"Train: {len(self.train_files)}, Val: {len(self.val_files)}")

        # Class distribution
        self._print_class_distribution()

    def _filter_valid_files(self):
        """Filter for files with valid punctuation targets."""
        punct_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        no_punct = 0

        for file_path in self.all_files:
            try:
                # Quick load just transcript
                data = np.load(file_path, allow_pickle=True)
                transcript = str(data["transcript"])
                target = extract_punct_target(transcript)

                if target >= 0:
                    self.valid_files.append(file_path)
                    punct_counts[target] += 1
                else:
                    no_punct += 1
            except Exception:
                continue

        print(f"Class distribution: {punct_counts}")
        print(f"Skipped (no punct): {no_punct}")

    def _print_class_distribution(self):
        """Print training class distribution."""
        counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for f in self.train_files:
            try:
                data = np.load(f, allow_pickle=True)
                target = extract_punct_target(str(data["transcript"]))
                if target >= 0:
                    counts[target] += 1
            except Exception:
                continue

        total = sum(counts.values())
        print("Training distribution:")
        for i, name in enumerate(PUNCT_NAMES):
            pct = 100 * counts[i] / total if total > 0 else 0
            print(f"  {name}: {counts[i]} ({pct:.1f}%)")

    def load_sample(self, file_path: Path) -> dict | None:
        """Load a single sample from .npz file."""
        try:
            data = np.load(file_path, allow_pickle=True)

            # Extract transcript and target
            transcript = str(data["transcript"])
            target = extract_punct_target(transcript)

            if target < 0:
                return None

            # Extract emotion probs (take first 8 if more)
            emotion_probs = data["emotion_probs"].astype(np.float32)
            if len(emotion_probs) > 8:
                emotion_probs = emotion_probs[:8]
            elif len(emotion_probs) < 8:
                # Pad if needed
                padded = np.zeros(8, dtype=np.float32)
                padded[: len(emotion_probs)] = emotion_probs
                emotion_probs = padded

            # Extract pitch features
            pitch_values = data["pitch_values"].astype(np.float32)
            pitch_features = extract_pitch_features(pitch_values)

            return {
                "emotion_probs": emotion_probs,
                "pitch_features": pitch_features,
                "transcript": transcript,
                "target": target,
            }
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

    def get_batch(
        self, files: list[Path], batch_size: int, shuffle: bool = True,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        """
        Generator yielding batches.

        Yields:
            emotion_probs: (B, 8)
            pitch_features: (B, 5)
            whisper_punct_placeholder: (B, 4) - zeros for now
            targets: (B,)
        """
        if shuffle:
            files = list(files)
            rng = np.random.default_rng()
            rng.shuffle(files)

        batch_emotion = []
        batch_pitch = []
        batch_targets = []

        for f in files:
            sample = self.load_sample(f)
            if sample is None:
                continue

            batch_emotion.append(sample["emotion_probs"])
            batch_pitch.append(sample["pitch_features"])
            batch_targets.append(sample["target"])

            if len(batch_targets) >= batch_size:
                yield (
                    mx.array(np.stack(batch_emotion)),
                    mx.array(np.stack(batch_pitch)),
                    mx.zeros((len(batch_targets), 4)),  # Placeholder for Whisper logits
                    mx.array(batch_targets),
                )
                batch_emotion = []
                batch_pitch = []
                batch_targets = []

        # Final partial batch
        if batch_targets:
            yield (
                mx.array(np.stack(batch_emotion)),
                mx.array(np.stack(batch_pitch)),
                mx.zeros((len(batch_targets), 4)),
                mx.array(batch_targets),
            )


# =============================================================================
# Training
# =============================================================================


def compute_loss(
    model: ProsodyPunctuationFusion,
    emotion_probs: mx.array,
    pitch_features: mx.array,
    whisper_punct: mx.array,
    targets: mx.array,
    config: FusionTrainingConfig,
) -> tuple[mx.array, dict]:
    """
    Compute training loss with focal loss for class imbalance.

    Focal loss: FL(p) = -(1-p)^gamma * log(p)
    This focuses learning on hard examples and down-weights easy ones.

    When integrated with Whisper, the boost will be added to Whisper's logits.
    """
    # Forward pass
    boost = model(emotion_probs, pitch_features, whisper_punct)

    # For training without Whisper logits, treat boost as direct logits
    # In deployment, this will be: final = whisper_logits + boost
    logits = boost + whisper_punct  # whisper_punct is zeros during training

    # Class weights for rare classes (? and !)
    weights = mx.array(
        [1.0, 1.0, config.question_weight, config.exclaim_weight], dtype=mx.float32,
    )

    # Compute softmax probabilities
    probs = mx.softmax(logits, axis=-1)

    # Gather target probabilities
    batch_size = targets.shape[0]
    target_probs = probs[mx.arange(batch_size), targets]

    if config.use_focal_loss:
        # Focal loss: FL(p) = -(1-p)^gamma * log(p)
        # This down-weights easy examples and focuses on hard ones
        gamma = config.focal_gamma
        focal_weight = (1.0 - target_probs) ** gamma
        log_target_probs = mx.log(target_probs + 1e-10)
        target_weights = weights[targets]
        ce_loss = -mx.mean(focal_weight * log_target_probs * target_weights)
    else:
        # Standard cross-entropy with class weights
        log_target_probs = mx.log(target_probs + 1e-10)
        target_weights = weights[targets]
        ce_loss = -mx.mean(log_target_probs * target_weights)

    # Boost magnitude regularization (keep boosts small)
    boost_reg = mx.mean(boost**2)

    # Total loss
    total_loss = config.ce_weight * ce_loss + config.boost_reg_weight * boost_reg

    # Metrics
    preds = mx.argmax(logits, axis=-1)
    accuracy = mx.mean(preds == targets)

    metrics = {
        "loss": float(total_loss),
        "ce_loss": float(ce_loss),
        "boost_reg": float(boost_reg),
        "accuracy": float(accuracy),
        "mean_boost": float(mx.mean(mx.abs(boost))),
    }

    return total_loss, metrics


def train_step(
    model: ProsodyPunctuationFusion,
    optimizer: optim.Optimizer,
    emotion_probs: mx.array,
    pitch_features: mx.array,
    whisper_punct: mx.array,
    targets: mx.array,
    config: FusionTrainingConfig,
) -> dict:
    """Single training step."""

    def loss_fn(model):
        loss, metrics = compute_loss(
            model, emotion_probs, pitch_features, whisper_punct, targets, config,
        )
        return loss, metrics

    # Compute gradients
    grad_fn = nn.value_and_grad(model, loss_fn)
    (loss, metrics), grads = grad_fn(model)

    # Clip gradients
    grads, grad_norm = optim.clip_grad_norm(grads, config.grad_clip)
    metrics["grad_norm"] = float(grad_norm)

    # Update weights
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)

    return metrics


def evaluate(
    model: ProsodyPunctuationFusion,
    dataset,  # LibriSpeechFeatureDataset or ProsodyJSONDataset
    config: FusionTrainingConfig,
) -> dict:
    """Evaluate on validation set."""
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    class_correct = dict.fromkeys(range(4), 0)
    class_total = dict.fromkeys(range(4), 0)

    # Support both dataset types
    val_data = getattr(dataset, 'val_files', None) or getattr(dataset, 'val_samples', [])

    for emotion, pitch, whisper, targets in dataset.get_batch(
        val_data, config.batch_size, shuffle=False,
    ):
        loss, metrics = compute_loss(
            model, emotion, pitch, whisper, targets, config,
        )
        mx.eval(loss)

        batch_size = targets.shape[0]
        total_loss += float(loss) * batch_size
        total_samples += batch_size

        # Per-class accuracy
        boost = model(emotion, pitch, whisper)
        logits = boost + whisper
        preds = mx.argmax(logits, axis=-1)
        mx.eval(preds)

        preds_np = np.array(preds)
        targets_np = np.array(targets)

        total_correct += np.sum(preds_np == targets_np)

        for i in range(4):
            mask = targets_np == i
            class_total[i] += np.sum(mask)
            class_correct[i] += np.sum((preds_np == i) & mask)

    # Compute per-class F1 scores
    f1_scores = {}
    for i, name in enumerate(PUNCT_NAMES):
        if class_total[i] > 0:
            recall = class_correct[i] / class_total[i]
            # For F1, we'd need precision too, but use recall as proxy
            f1_scores[name] = recall
        else:
            f1_scores[name] = 0.0

    return {
        "val_loss": total_loss / max(total_samples, 1),
        "val_accuracy": total_correct / max(total_samples, 1),
        "val_samples": total_samples,
        **{f"val_{k}_recall": v for k, v in f1_scores.items()},
    }


def train(config: FusionTrainingConfig, prosody_json: str = None):
    """Main training loop."""
    print("=" * 60)
    print("Prosody-Punctuation Fusion Training")
    print("=" * 60)
    if prosody_json:
        print(f"Prosody JSON: {prosody_json}")
    else:
        print(f"Features dir: {config.librispeech_features}")
    print(f"Output dir: {config.output_dir}")
    print(f"Epochs: {config.epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Focal loss: {config.use_focal_loss} (gamma={config.focal_gamma})")
    print(f"Balanced batching: {config.balanced_batching}")
    print(f"Class weights: ?={config.question_weight}, !={config.exclaim_weight}")
    print("=" * 60)

    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = output_dir / "config.json"
    config_dict = vars(config).copy()
    config_dict["prosody_json"] = prosody_json
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    # Load dataset - choose based on data source
    print("\nLoading dataset...")
    if prosody_json:
        dataset = ProsodyJSONDataset(prosody_json, val_split=config.val_split)
        train_data = dataset.train_samples
        num_train = len(train_data)
    else:
        dataset = LibriSpeechFeatureDataset(
            config.librispeech_features, val_split=config.val_split,
        )
        train_data = dataset.train_files
        num_train = len(train_data)

    if num_train == 0:
        print("ERROR: No valid training samples found!")
        return

    # Create model
    print("\nCreating model...")
    model = create_fusion_model(config)

    # Count parameters using tree_flatten
    flat_params = tree_flatten(model.parameters())
    param_count = sum(p.size for _, p in flat_params)
    print(f"Parameters: {param_count:,}")

    # Create optimizer with warmup
    steps_per_epoch = num_train // config.batch_size
    total_steps = config.epochs * steps_per_epoch

    warmup_scheduler = optim.schedulers.linear_schedule(
        init=1e-8, end=config.learning_rate, steps=config.warmup_steps,
    )
    decay_scheduler = optim.schedulers.cosine_decay(
        init=config.learning_rate, decay_steps=total_steps - config.warmup_steps,
    )
    scheduler = optim.schedulers.join_schedules(
        schedules=[warmup_scheduler, decay_scheduler],
        boundaries=[config.warmup_steps],
    )

    optimizer = optim.AdamW(
        learning_rate=scheduler, weight_decay=config.weight_decay,
    )

    # Training loop
    print("\nStarting training...")
    global_step = 0
    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(config.epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_batches = 0

        for emotion, pitch, whisper, targets in dataset.get_batch(
            train_data, config.batch_size, shuffle=True,
            balanced=config.balanced_batching if isinstance(dataset, ProsodyJSONDataset) else False,
        ):
            metrics = train_step(
                model, optimizer, emotion, pitch, whisper, targets, config,
            )

            epoch_loss += metrics["loss"]
            epoch_acc += metrics["accuracy"]
            epoch_batches += 1
            global_step += 1

            # Logging
            if global_step % config.log_interval == 0:
                avg_loss = epoch_loss / epoch_batches
                avg_acc = epoch_acc / epoch_batches
                lr = scheduler(global_step)
                elapsed = time.time() - start_time
                print(
                    f"Step {global_step:5d} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Acc: {avg_acc:.3f} | "
                    f"Boost: {metrics['mean_boost']:.4f} | "
                    f"LR: {lr:.2e} | "
                    f"Time: {elapsed:.0f}s",
                )

            # Save checkpoint
            if global_step % config.save_interval == 0:
                ckpt_path = output_dir / f"step_{global_step}.npz"
                mx.savez(str(ckpt_path), **dict(tree_flatten(model.parameters())))
                print(f"Saved checkpoint: {ckpt_path}")

        # End of epoch
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / max(epoch_batches, 1)
        avg_acc = epoch_acc / max(epoch_batches, 1)

        print(f"\nEpoch {epoch + 1}/{config.epochs} complete ({epoch_time:.0f}s)")
        print(f"  Train Loss: {avg_loss:.4f}, Train Acc: {avg_acc:.3f}")

        # Validation
        val_metrics = evaluate(model, dataset, config)
        print(f"  Val Loss: {val_metrics['val_loss']:.4f}, Val Acc: {val_metrics['val_accuracy']:.3f}")
        print("  Per-class recall: ", end="")
        for name in PUNCT_NAMES:
            print(f"{name}={val_metrics[f'val_{name}_recall']:.3f} ", end="")
        print()

        # Save best model
        if val_metrics["val_accuracy"] > best_val_acc:
            best_val_acc = val_metrics["val_accuracy"]
            best_path = output_dir / "best.npz"
            mx.savez(str(best_path), **dict(tree_flatten(model.parameters())))
            print(f"  New best! Saved to {best_path}")

        # Garbage collection
        gc.collect()

    # Save final model
    final_path = output_dir / "final.npz"
    mx.savez(str(final_path), **dict(tree_flatten(model.parameters())))

    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Total time: {total_time / 60:.1f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.3f}")
    print(f"Checkpoints saved to: {output_dir}")
    print("=" * 60)


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Train Prosody-Punctuation Fusion Layer",
    )

    # Data
    parser.add_argument(
        "--librispeech-features",
        type=str,
        default="data/emotion_punctuation/librispeech_features",
        help="Directory with pre-extracted LibriSpeech features (.npz files)",
    )
    parser.add_argument(
        "--prosody-json",
        type=str,
        default=None,
        help="Path to prosody JSON file (e.g., data/prosody/consolidated.json). "
             "Overrides --librispeech-features if provided.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/prosody_punct_fusion_v1",
        help="Output directory for checkpoints",
    )

    # Model
    parser.add_argument(
        "--hidden-dim", type=int, default=64, help="Hidden dimension",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="Dropout rate",
    )

    # Training
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="Learning rate",
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=1000, help="Warmup steps",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.01, help="Weight decay",
    )

    # Loss weights
    parser.add_argument(
        "--question-weight", type=float, default=3.0, help="Weight for ? class",
    )
    parser.add_argument(
        "--exclaim-weight", type=float, default=3.0, help="Weight for ! class",
    )
    parser.add_argument(
        "--boost-reg-weight",
        type=float,
        default=0.01,
        help="Regularization weight for boost magnitude",
    )

    # Focal loss and balanced batching
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Focal loss gamma (0=CE, 2=standard focal, 5=aggressive). Default 2.0",
    )
    parser.add_argument(
        "--no-focal-loss",
        action="store_true",
        help="Disable focal loss (use standard cross-entropy)",
    )
    parser.add_argument(
        "--no-balanced-batching",
        action="store_true",
        help="Disable balanced batching (use random sampling)",
    )

    # Logging
    parser.add_argument(
        "--log-interval", type=int, default=50, help="Log every N steps",
    )
    parser.add_argument(
        "--save-interval", type=int, default=500, help="Save every N steps",
    )

    args = parser.parse_args()

    config = FusionTrainingConfig(
        librispeech_features=args.librispeech_features,
        output_dir=args.output_dir,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        question_weight=args.question_weight,
        exclaim_weight=args.exclaim_weight,
        boost_reg_weight=args.boost_reg_weight,
        focal_gamma=args.focal_gamma,
        use_focal_loss=not args.no_focal_loss,
        balanced_batching=not args.no_balanced_batching,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
    )

    train(config, prosody_json=args.prosody_json)


if __name__ == "__main__":
    main()
