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
RichDecoder v4 Training - Optimized for Speed.

Key optimizations over v3:
1. Encoder feature caching (3-4x speedup)
2. Gradient accumulation (effective batch=16)
3. bfloat16 mixed precision (1.5-2x speedup)
4. Prefetch data loading (eliminates I/O wait)
5. Cosine LR with warmup (faster convergence)
6. Early stopping (avoid plateau waste)

Expected: 10h → 1-2h training time

Usage:
    # First, pre-extract encoder features:
    python scripts/preextract_encoder_features.py \
        --manifest data/v3_multitask/train_manifest.json \
        --output-dir data/v3_multitask/encoder_cache

    # Then train with cached features:
    python -m tools.whisper_mlx.train_rich_decoder_v4 \
        --output-dir checkpoints/rich_decoder_v4 \
        --encoder-cache data/v3_multitask/encoder_cache \
        --init-from checkpoints/rich_decoder_v1/best.npz
"""

import argparse
import gc
import json
import math
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_map

# Project root for resolving relative paths
# Can be overridden via PROJECT_ROOT environment variable
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).parent.parent.parent))

from .encoder_lora import EncoderBlockLoRA

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """Training configuration with speed optimizations."""

    # Data
    train_manifest: str = "data/v3_multitask/train_manifest.json"
    val_manifest: str = "data/v3_multitask/val_manifest.json"
    encoder_cache_dir: str | None = None  # Pre-extracted features

    # Model
    whisper_model: str = "mlx-community/whisper-large-v3-mlx"
    init_from: str | None = None  # Initialize from checkpoint

    # LoRA (decoder)
    lora_rank: int = 32
    lora_alpha: int = 64

    # Encoder LoRA (adapts top encoder layers)
    use_encoder_lora: bool = True
    encoder_lora_rank: int = 16
    encoder_lora_alpha: int = 32
    encoder_lora_start_layer: int = 20  # Adapt layers 20-31 (top 12)

    # Prosody features
    use_prosody: bool = True
    prosody_dim: int = 4  # [f0, energy, f0_delta, energy_delta]

    # Output
    output_dir: str = "checkpoints/rich_decoder_v4"

    # Training - Optimized defaults
    epochs: int = 10
    batch_size: int = 4
    accumulation_steps: int = 4  # Effective batch = 16
    learning_rate: float = 5e-5
    min_learning_rate: float = 1e-6
    warmup_steps: int = 500
    weight_decay: float = 0.01

    # Speed optimizations
    use_bfloat16: bool = True
    prefetch_batches: int = 4
    use_early_stopping: bool = True
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 0.001

    # Optimizer choice: "adamw" or "sgd"
    # SGD has no optimizer state, avoiding Metal memory accumulation
    optimizer: str = "adamw"
    sgd_momentum: float = 0.9  # Momentum for SGD

    # Logging
    log_interval: int = 50
    save_interval: int = 500
    eval_interval: int = 1000


# =============================================================================
# Learning Rate Schedule
# =============================================================================

def get_cosine_lr(
    step: int,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """Cosine learning rate with linear warmup."""
    if step < warmup_steps:
        # Linear warmup
        return max_lr * step / warmup_steps
    # Cosine decay
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progress = min(1.0, progress)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


# =============================================================================
# Early Stopping
# =============================================================================

class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 3, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
            return True
        return False


# =============================================================================
# Prefetch Data Loader
# =============================================================================

class PrefetchDataLoader:
    """
    Data loader with background prefetching.

    Eliminates I/O wait by loading next batches in background thread.
    """

    def __init__(
        self,
        manifest: list[dict],
        encoder_cache_dir: str | None,
        batch_size: int,
        prefetch_count: int = 4,
        shuffle: bool = True,
        fallback_encoder=None,  # Whisper encoder for cache misses
        use_prosody: bool = True,  # Load prosody features if available
        cache_only: bool = True,  # Only use samples with cached features
    ):
        self.encoder_cache_dir = Path(encoder_cache_dir) if encoder_cache_dir else None
        self.batch_size = batch_size
        self.prefetch_count = prefetch_count
        self.shuffle = shuffle
        self.fallback_encoder = fallback_encoder
        self.use_prosody = use_prosody
        self.cache_only = cache_only

        # Filter manifest to cached samples only if requested
        if cache_only and self.encoder_cache_dir:
            original_count = len(manifest)
            self.manifest = self._filter_to_cached(manifest)
            filtered_count = len(self.manifest)
            cache_rate = filtered_count / original_count * 100 if original_count > 0 else 0
            print(f"Cache-only mode: {filtered_count}/{original_count} samples ({cache_rate:.1f}% cached)")
            if filtered_count == 0:
                raise ValueError("No cached samples found! Run encoder pre-extraction first.")
        else:
            self.manifest = manifest

        self.queue = queue.Queue(maxsize=prefetch_count)
        self.stop_event = threading.Event()
        self.thread = None

        # Emotion mapping - MUST match label_taxonomy.py EMOTION_CLASSES_9
        # ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised", "contempt"]
        self.emotion_to_id = {
            'neutral': 0,
            'calm': 1,
            'happy': 2,
            'sad': 3,
            'angry': 4,
            'fearful': 5, 'fear': 5,  # alias for manifest compatibility
            'disgust': 6,
            'surprised': 7, 'surprise': 7,  # alias for manifest compatibility
            'contempt': 8,  # NEW in v2.0 taxonomy
            'other': 0,  # map unknown to neutral
        }

        # Stats
        self.cache_hits = 0
        self.cache_misses = 0
        self.prosody_found = 0
        self.prosody_missing = 0

    def _filter_to_cached(self, manifest: list[dict]) -> list[dict]:
        """Filter manifest to only include samples with cached encoder features or features_path."""
        import hashlib
        cached = []
        for sample in manifest:
            # First check if sample has pre-extracted features (pseudo-labels)
            features_path = sample.get("features_path")
            if features_path:
                # Resolve path relative to repo root if needed
                if not features_path.startswith("/"):
                    full_path = PROJECT_ROOT / features_path
                else:
                    full_path = Path(features_path)
                if full_path.exists():
                    cached.append(sample)
                    continue

            # Fall back to encoder cache
            audio_path = sample.get("audio_path", sample.get("path", ""))
            if audio_path and self.encoder_cache_dir:
                cache_key = hashlib.sha256(audio_path.encode()).hexdigest()[:16]
                cache_path = self.encoder_cache_dir / cache_key[:2] / f"{cache_key}.npz"
                if cache_path.exists():
                    cached.append(sample)
        return cached

    def _get_cache_path(self, audio_path: str) -> Path:
        """Get cache file path for audio."""
        import hashlib
        cache_key = hashlib.sha256(audio_path.encode()).hexdigest()[:16]
        return self.encoder_cache_dir / cache_key[:2] / f"{cache_key}.npz"

    def _load_sample(self, sample: dict) -> tuple[np.ndarray, int, np.ndarray | None] | None:
        """Load a single sample from cache or features_path.

        Returns:
            Tuple of (encoder_out, emotion_id, prosody) where prosody may be None
        """
        # Get emotion label
        emotion = sample.get("emotion", "neutral")
        emotion_id = self.emotion_to_id.get(emotion, 0)

        # First, try features_path (pre-extracted features for pseudo-labels)
        features_path = sample.get("features_path")
        if features_path:
            # Resolve path relative to repo root if needed
            if not features_path.startswith("/"):
                full_path = PROJECT_ROOT / features_path
            else:
                full_path = Path(features_path)

            if full_path.exists():
                try:
                    data = np.load(str(full_path))
                    # LibriSpeech features use 'encoder_features' key
                    features_key = 'encoder_features' if 'encoder_features' in data else 'encoder_output'
                    encoder_out = data[features_key].astype(np.float32)

                    # Load prosody if available and requested
                    prosody = None
                    if self.use_prosody and "prosody" in data:
                        prosody = data["prosody"].astype(np.float32)
                        self.prosody_found += 1
                    elif self.use_prosody:
                        self.prosody_missing += 1

                    self.cache_hits += 1
                    return encoder_out, emotion_id, prosody
                except Exception:
                    pass

        # Fall back to encoder cache
        audio_path = sample.get("audio_path", sample.get("path", ""))
        if self.encoder_cache_dir and audio_path:
            cache_path = self._get_cache_path(audio_path)
            if cache_path.exists():
                try:
                    data = np.load(cache_path)
                    encoder_out = data["encoder_output"].astype(np.float32)

                    # Load prosody if available and requested
                    prosody = None
                    if self.use_prosody and "prosody" in data:
                        prosody = data["prosody"].astype(np.float32)
                        self.prosody_found += 1
                    elif self.use_prosody:
                        self.prosody_missing += 1

                    self.cache_hits += 1
                    return encoder_out, emotion_id, prosody
                except Exception:
                    pass

        # NOTE: Fallback encoder computation removed - MLX Metal ops are NOT
        # thread-safe and crash when called from background prefetch thread.
        # Training uses cached samples only. As pre-extraction runs in parallel,
        # more samples become available in subsequent epochs.
        self.cache_misses += 1
        return None

    def _create_batch(self, samples: list[dict]) -> dict | None:
        """Create a batch from samples."""
        encoder_outs = []
        emotion_ids = []
        prosody_list = []

        for sample in samples:
            result = self._load_sample(sample)
            if result is not None:
                encoder_out, emotion_id, prosody = result
                encoder_outs.append(encoder_out)
                emotion_ids.append(emotion_id)
                prosody_list.append(prosody)

        if not encoder_outs:
            return None

        # Pad encoder outputs to same length
        max_len = max(e.shape[0] for e in encoder_outs)
        padded = []
        for e in encoder_outs:
            if e.shape[0] < max_len:
                pad = np.zeros((max_len - e.shape[0], e.shape[1]), dtype=np.float32)
                e = np.concatenate([e, pad], axis=0)
            padded.append(e)

        batch = {
            "encoder_out": mx.array(np.stack(padded)),
            "emotion_ids": mx.array(emotion_ids),
        }

        # Add prosody if any sample has it
        if self.use_prosody and any(p is not None for p in prosody_list):
            # Pad prosody to same length (4 features)
            padded_prosody = []
            for p in prosody_list:
                if p is None:
                    # No prosody - use zeros
                    p = np.zeros((max_len, 4), dtype=np.float32)
                elif p.shape[0] < max_len:
                    pad = np.zeros((max_len - p.shape[0], 4), dtype=np.float32)
                    p = np.concatenate([p, pad], axis=0)
                elif p.shape[0] > max_len:
                    p = p[:max_len]
                padded_prosody.append(p)

            batch["prosody"] = mx.array(np.stack(padded_prosody))

        return batch

    def _prefetch_loop(self, indices: list[int]):
        """Background thread that prefetches batches."""
        for i in range(0, len(indices), self.batch_size):
            if self.stop_event.is_set():
                break

            batch_indices = indices[i:i + self.batch_size]
            samples = [self.manifest[j] for j in batch_indices]
            batch = self._create_batch(samples)

            if batch is not None:
                self.queue.put(batch)

        # Signal end
        self.queue.put(None)

    def __iter__(self):
        """Iterate over batches with prefetching."""
        indices = list(range(len(self.manifest)))
        if self.shuffle:
            rng = np.random.default_rng()
            rng.shuffle(indices)

        # Start prefetch thread
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._prefetch_loop, args=(indices,))
        self.thread.start()

        # Yield batches from queue with timeout to avoid infinite blocking
        timeout_count = 0
        max_timeouts = 10  # Allow some timeouts before giving up
        timeout_seconds = 30  # Wait up to 30s for each batch

        while True:
            try:
                batch = self.queue.get(timeout=timeout_seconds)
                if batch is None:
                    break
                timeout_count = 0  # Reset on successful batch
                yield batch
            except queue.Empty:
                timeout_count += 1
                # Check if prefetch thread is still running
                if not self.thread.is_alive():
                    # Thread finished, no more data
                    break
                if timeout_count >= max_timeouts:
                    print(f"WARNING: Queue timeout {timeout_count}x - cache may be too sparse")
                    print("Consider waiting for more encoder features to be cached")
                    break
                print(f"Queue timeout ({timeout_count}/{max_timeouts}) - waiting for prefetch...")

        # Wait for thread to finish
        self.thread.join(timeout=5.0)

    def stop(self):
        """Stop prefetching."""
        self.stop_event.set()
        if self.thread:
            self.thread.join()

    def __len__(self):
        return (len(self.manifest) + self.batch_size - 1) // self.batch_size


# =============================================================================
# Model Components
# =============================================================================

class EmotionHead(nn.Module):
    """Emotion classification head."""

    def __init__(self, input_dim: int = 1280, num_classes: int = 8):
        super().__init__()
        self.pool = nn.Linear(input_dim, input_dim)
        self.classifier = nn.Linear(input_dim, num_classes)

    def __call__(self, x: mx.array) -> mx.array:
        # Global average pooling
        x = mx.mean(x, axis=1)  # [B, T, D] -> [B, D]
        x = nn.gelu(self.pool(x))
        return self.classifier(x)


class EncoderLoRAModule(nn.Module):
    """
    Encoder LoRA module for adapting cached encoder outputs.

    Since we use pre-cached encoder features, we can't inject LoRA into
    the encoder forward pass. Instead, we apply LoRA as a post-processing
    step on the cached outputs.

    This creates LoRA adapters for each layer and applies them to produce
    a modified encoder output that can learn task-specific representations.
    """

    def __init__(
        self,
        n_state: int = 1280,
        rank: int = 16,
        alpha: int = 32,
        start_layer: int = 20,
        end_layer: int = 32,
    ):
        super().__init__()

        self.n_state = n_state
        self.rank = rank
        self.alpha = alpha
        self.start_layer = start_layer
        self.end_layer = end_layer

        # Create adapters for each layer
        # We store them as a list for proper nn.Module tracking
        self.adapters = []
        for i in range(end_layer - start_layer):
            adapter = EncoderBlockLoRA(
                n_state=n_state,
                rank=rank,
                alpha=alpha,
                adapt_query=True,
                adapt_key=True,
                adapt_value=True,
            )
            self.adapters.append(adapter)
            # Register as attribute for parameter tracking
            setattr(self, f'adapter_{i}', adapter)

    def __call__(self, encoder_out: mx.array) -> mx.array:
        """
        Apply encoder LoRA adaptations to cached encoder output.

        Args:
            encoder_out: Cached encoder output [B, T, D]

        Returns:
            Adapted encoder output [B, T, D]
        """
        # Apply each adapter's LoRA as additive delta
        # This approximates what would happen if LoRA were in the encoder
        delta = mx.zeros_like(encoder_out)

        for adapter in self.adapters:
            # Each adapter adds its Q, K, V LoRA deltas
            if adapter.q_lora is not None:
                delta = delta + adapter.q_lora(encoder_out)
            if adapter.k_lora is not None:
                delta = delta + adapter.k_lora(encoder_out)
            if adapter.v_lora is not None:
                delta = delta + adapter.v_lora(encoder_out)

        # Scale by number of adapted layers for stability
        n_adapters = len(self.adapters)
        if n_adapters > 0:
            delta = delta / n_adapters

        return encoder_out + delta

    def param_count(self) -> int:
        """Total trainable parameters."""
        return sum(a.param_count() for a in self.adapters)


class RichDecoderV4(nn.Module):
    """
    RichDecoder v4 - Optimized for speed.

    Uses cached encoder outputs, so only trains classification heads + LoRA.
    Optionally includes encoder LoRA for adapting encoder representations.
    Optionally includes prosody features for improved emotion recognition.
    """

    def __init__(
        self,
        input_dim: int = 1280,
        num_emotions: int = 8,
        lora_rank: int = 32,
        lora_alpha: int = 64,
        use_encoder_lora: bool = True,
        encoder_lora_rank: int = 16,
        encoder_lora_alpha: int = 32,
        encoder_lora_start_layer: int = 20,
        use_prosody: bool = True,
        prosody_dim: int = 4,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.use_encoder_lora = use_encoder_lora
        self.use_prosody = use_prosody
        self.prosody_dim = prosody_dim

        # Encoder LoRA for adapting cached encoder outputs
        if use_encoder_lora:
            self.encoder_lora = EncoderLoRAModule(
                n_state=input_dim,
                rank=encoder_lora_rank,
                alpha=encoder_lora_alpha,
                start_layer=encoder_lora_start_layer,
                end_layer=32,  # Whisper large-v3 has 32 encoder layers
            )
        else:
            self.encoder_lora = None

        # Prosody projection layer: combines encoder (1280) + prosody (4) -> 1280
        if use_prosody:
            self.prosody_projection = nn.Linear(input_dim + prosody_dim, input_dim)
        else:
            self.prosody_projection = None

        # Classification heads
        self.emotion_head = EmotionHead(input_dim, num_emotions)

    def __call__(
        self,
        encoder_out: mx.array,
        prosody: mx.array | None = None,
    ) -> dict[str, mx.array]:
        """Forward pass on cached encoder outputs.

        Args:
            encoder_out: Encoder output [B, T, 1280]
            prosody: Optional prosody features [B, T, 4]

        Returns:
            Dict with 'emotion' logits
        """
        # Apply encoder LoRA if enabled
        if self.encoder_lora is not None:
            encoder_out = self.encoder_lora(encoder_out)

        # Concatenate prosody features if available
        if self.use_prosody and prosody is not None and self.prosody_projection is not None:
            # Concatenate: [B, T, 1280] + [B, T, 4] -> [B, T, 1284]
            combined = mx.concatenate([encoder_out, prosody], axis=-1)
            # Project back to input_dim: [B, T, 1284] -> [B, T, 1280]
            encoder_out = self.prosody_projection(combined)

        emotion_logits = self.emotion_head(encoder_out)
        return {"emotion": emotion_logits}

    def total_params(self) -> int:
        """Total trainable parameters."""
        def count_params(params):
            total = 0
            for v in params.values():
                if isinstance(v, dict):
                    total += count_params(v)
                elif hasattr(v, 'size'):
                    total += v.size
            return total
        return count_params(self.parameters())

    def encoder_lora_params(self) -> int:
        """Encoder LoRA parameters only."""
        if self.encoder_lora is not None:
            return self.encoder_lora.param_count()
        return 0


# =============================================================================
# Trainer
# =============================================================================

class RichDecoderV4Trainer:
    """Optimized trainer with all speedups."""

    def __init__(self, config: TrainingConfig):
        self.config = config

        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

        # Initialize model
        self.model = RichDecoderV4(
            input_dim=1280,
            num_emotions=8,
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            use_encoder_lora=config.use_encoder_lora,
            encoder_lora_rank=config.encoder_lora_rank,
            encoder_lora_alpha=config.encoder_lora_alpha,
            encoder_lora_start_layer=config.encoder_lora_start_layer,
            use_prosody=config.use_prosody,
            prosody_dim=config.prosody_dim,
        )

        # Print parameter counts
        total_params = self.model.total_params()
        encoder_lora_params = self.model.encoder_lora_params()
        print("Model parameters:")
        print(f"  Total: {total_params:,}")
        print(f"  Encoder LoRA: {encoder_lora_params:,}")
        print(f"  Classification head: {total_params - encoder_lora_params:,}")

        # Convert to bfloat16 if enabled
        if config.use_bfloat16:
            def convert_to_bfloat16(x):
                if isinstance(x, mx.array) and x.dtype in [mx.float32, mx.float16]:
                    return x.astype(mx.bfloat16)
                return x
            self.model.update(tree_map(convert_to_bfloat16, self.model.parameters()))
            print("  Converted model parameters to bfloat16")

        # Load initial weights if specified
        if config.init_from and Path(config.init_from).exists():
            print(f"Loading weights from {config.init_from}")
            # Load compatible weights
            self._load_compatible_weights(config.init_from)

        # Optimizer - SGD avoids Metal memory accumulation from AdamW state
        if config.optimizer.lower() == "sgd":
            self.optimizer = optim.SGD(
                learning_rate=config.learning_rate,
                momentum=config.sgd_momentum,
                weight_decay=config.weight_decay,
            )
            print(f"  Using SGD optimizer (momentum={config.sgd_momentum}, no state accumulation)")
        else:
            self.optimizer = optim.AdamW(
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
            )
            print("  Using AdamW optimizer (WARNING: may cause Metal memory crash)")

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta,
        ) if config.use_early_stopping else None

        # Stats
        self.step = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0

    def _load_compatible_weights(self, path: str):
        """Load compatible weights from checkpoint."""
        try:
            weights = dict(mx.load(path))

            # Flatten model weights for comparison (only arrays)
            def flatten_dict(d, prefix=''):
                flat = {}
                for k, v in d.items():
                    key = f'{prefix}.{k}' if prefix else k
                    if isinstance(v, dict):
                        flat.update(flatten_dict(v, key))
                    elif hasattr(v, 'shape'):  # mx.array check
                        flat[key] = v
                return flat

            model_flat = flatten_dict(dict(self.model.parameters()))

            loaded = 0
            for key, value in weights.items():
                if key in model_flat and model_flat[key].shape == value.shape:
                    model_flat[key] = value
                    loaded += 1

            # Unflatten back to nested dict structure
            def unflatten_dict(flat):
                nested = {}
                for key, value in flat.items():
                    parts = key.split('.')
                    d = nested
                    for part in parts[:-1]:
                        if part not in d:
                            d[part] = {}
                        d = d[part]
                    d[parts[-1]] = value
                return nested

            self.model.update(unflatten_dict(model_flat))
            print(f"  Loaded {loaded} weight tensors")
        except Exception as e:
            print(f"  Warning: Could not load weights: {e}")

    def loss_fn(self, model: nn.Module, batch: dict) -> tuple[mx.array, dict]:
        """Compute loss with bfloat16 optimization."""
        encoder_out = batch["encoder_out"]
        emotion_ids = batch["emotion_ids"]
        prosody = batch.get("prosody", None)

        # Forward pass (bfloat16 if enabled)
        outputs = model(encoder_out, prosody=prosody)

        # Loss in float32 for stability
        emotion_logits = outputs["emotion"].astype(mx.float32)
        emotion_loss = mx.mean(
            nn.losses.cross_entropy(emotion_logits, emotion_ids),
        )

        # Accuracy
        preds = mx.argmax(emotion_logits, axis=-1)
        accuracy = mx.mean(preds == emotion_ids)

        return emotion_loss, {"accuracy": accuracy}

    def train_step(self, batch: dict) -> tuple[float, dict]:
        """Single training step with gradient accumulation."""
        loss_and_grad_fn = nn.value_and_grad(self.model, self.loss_fn)
        (loss, metrics), grads = loss_and_grad_fn(self.model, batch)

        # Scale gradients for accumulation
        scale = 1.0 / self.config.accumulation_steps
        grads = tree_map(lambda g: g * scale, grads)

        # Force evaluation to release intermediate computation graph memory
        mx.eval(loss)
        return float(loss), metrics, grads

    def train(
        self,
        train_loader: PrefetchDataLoader,
        val_loader: PrefetchDataLoader | None,
    ):
        """Main training loop with all optimizations."""
        config = self.config

        total_steps = len(train_loader) * config.epochs
        print(f"\n{'='*60}")
        print("Training RichDecoder v4 (Optimized)")
        print(f"{'='*60}")
        print(f"  Epochs: {config.epochs}")
        print(f"  Batch size: {config.batch_size} x {config.accumulation_steps} = {config.batch_size * config.accumulation_steps}")
        print(f"  Total steps: {total_steps}")
        print(f"  bfloat16: {config.use_bfloat16}")
        print(f"  Prefetch: {config.prefetch_batches} batches")
        print(f"  Early stopping: {config.use_early_stopping} (patience={config.early_stopping_patience})")
        print()

        # Training log
        log_path = Path(config.output_dir) / "training.log"
        log_file = open(log_path, "a")

        accumulated_grads = None
        accumulation_count = 0
        epoch_losses = []

        start_time = time.time()

        for epoch in range(config.epochs):
            print(f"Epoch {epoch + 1}/{config.epochs}")
            epoch_start = time.time()
            epoch_losses = []

            for batch in train_loader:
                self.step += 1

                # Update learning rate
                lr = get_cosine_lr(
                    self.step,
                    config.warmup_steps,
                    total_steps,
                    config.learning_rate,
                    config.min_learning_rate,
                )
                self.optimizer.learning_rate = lr

                # Training step
                loss, metrics, grads = self.train_step(batch)
                epoch_losses.append(loss)

                # Accumulate gradients with explicit evaluation to prevent memory buildup
                if accumulated_grads is None:
                    accumulated_grads = grads
                else:
                    accumulated_grads = tree_map(
                        lambda a, g: a + g, accumulated_grads, grads,
                    )
                # Force evaluation to free intermediate computation memory
                mx.eval(tree_map(lambda x: x, accumulated_grads))
                accumulation_count += 1

                # Apply gradients when accumulated enough
                if accumulation_count >= config.accumulation_steps:
                    self.optimizer.update(self.model, accumulated_grads)
                    mx.eval(self.model.parameters())
                    accumulated_grads = None
                    accumulation_count = 0
                    # Clear any accumulated Metal memory
                    gc.collect()
                    mx.clear_cache()

                # Logging
                if self.step % config.log_interval == 0:
                    acc = float(metrics["accuracy"])
                    msg = f"  Step {self.step}: loss={loss:.4f}, acc={acc:.2%}, lr={lr:.2e}"
                    print(msg)
                    log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}\n")
                    log_file.flush()
                    # Periodic memory cleanup to prevent Metal resource limit errors
                    gc.collect()
                    mx.clear_cache()

                # Save checkpoint
                if self.step % config.save_interval == 0:
                    self._save_checkpoint(f"step_{self.step}.npz")
                    gc.collect()
                    mx.clear_cache()

                # Evaluation (skip if no validation loader)
                if self.step % config.eval_interval == 0 and val_loader is not None:
                    val_loss, val_acc = self.evaluate(val_loader)
                    # Clear evaluation memory
                    gc.collect()
                    mx.clear_cache()

                    msg = f"  Validation: loss={val_loss:.4f}, acc={val_acc:.2%}"
                    print(msg)
                    log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}\n")

                    # Save best model
                    if val_acc > self.best_val_acc:
                        self.best_val_acc = val_acc
                        self._save_checkpoint("best.npz")
                        print(f"  New best model! acc={val_acc:.2%}")

                    # Early stopping
                    if self.early_stopping and self.early_stopping(val_loss):
                        print(f"  Early stopping triggered (patience={config.early_stopping_patience})")
                        log_file.close()
                        return

            # Epoch summary
            epoch_time = time.time() - epoch_start
            avg_loss = np.mean(epoch_losses)
            print(f"  Epoch {epoch + 1} complete: avg_loss={avg_loss:.4f}, time={epoch_time:.1f}s")

            # End of epoch evaluation (skip if no validation loader)
            if val_loader is not None:
                val_loss, val_acc = self.evaluate(val_loader)
                # Clear epoch-end evaluation memory
                gc.collect()
                mx.clear_cache()

                print(f"  Validation: loss={val_loss:.4f}, acc={val_acc:.2%}")

                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self._save_checkpoint("best.npz")
                    print(f"  New best model! acc={val_acc:.2%}")
            else:
                # Save checkpoint based on training loss when no validation
                self._save_checkpoint(f"epoch_{epoch + 1}.npz")

            # End of epoch memory cleanup
            gc.collect()
            mx.clear_cache()

        total_time = time.time() - start_time
        print("\nTraining complete!")
        print(f"  Total time: {total_time/3600:.2f} hours")
        print(f"  Best accuracy: {self.best_val_acc:.2%}")

        log_file.close()

    def evaluate(self, val_loader: PrefetchDataLoader) -> tuple[float, float]:
        """Evaluate on validation set."""
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in val_loader:
            prosody = batch.get("prosody", None)
            outputs = self.model(batch["encoder_out"], prosody=prosody)
            emotion_logits = outputs["emotion"]

            # Loss
            loss = mx.mean(nn.losses.cross_entropy(
                emotion_logits, batch["emotion_ids"],
            ))
            total_loss += float(loss) * len(batch["emotion_ids"])

            # Accuracy
            preds = mx.argmax(emotion_logits, axis=-1)
            correct = mx.sum(preds == batch["emotion_ids"])
            total_correct += int(correct)
            total_samples += len(batch["emotion_ids"])

        avg_loss = total_loss / max(total_samples, 1)
        accuracy = total_correct / max(total_samples, 1)

        return avg_loss, accuracy

    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = Path(self.config.output_dir) / filename

        # Flatten nested parameter dict for mx.savez
        # Only include mx.array values, skip lists and other non-array types
        def flatten_dict(d, prefix=''):
            flat = {}
            for k, v in d.items():
                key = f'{prefix}.{k}' if prefix else k
                if isinstance(v, dict):
                    flat.update(flatten_dict(v, key))
                elif hasattr(v, 'shape'):  # mx.array check
                    flat[key] = v
                # Skip lists and other non-array types
            return flat

        flat_params = flatten_dict(dict(self.model.parameters()))
        mx.savez(str(path), **flat_params)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train RichDecoder v4 (Optimized)")
    parser.add_argument("--output-dir", default="checkpoints/rich_decoder_v4")
    parser.add_argument("--encoder-cache", required=True, help="Pre-extracted encoder features")
    parser.add_argument("--train-manifest", default="data/v3_multitask/train_manifest.json")
    parser.add_argument("--val-manifest", default="data/v3_multitask/val_manifest.json")
    parser.add_argument("--init-from", help="Initialize from checkpoint")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--no-early-stopping", action="store_true")

    # Optimizer options
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw",
                       help="Optimizer: adamw (default, may crash) or sgd (no state, stable)")
    parser.add_argument("--sgd-momentum", type=float, default=0.9,
                       help="Momentum for SGD optimizer (default: 0.9)")

    # Encoder LoRA options
    parser.add_argument("--no-encoder-lora", action="store_true",
                       help="Disable encoder LoRA (only train classification head)")
    parser.add_argument("--encoder-lora-rank", type=int, default=16)
    parser.add_argument("--encoder-lora-alpha", type=int, default=32)
    parser.add_argument("--encoder-lora-start-layer", type=int, default=20,
                       help="First encoder layer to adapt (default: 20, adapts layers 20-31)")

    # Prosody options
    parser.add_argument("--no-prosody", action="store_true",
                       help="Disable prosody features")

    args = parser.parse_args()

    # Create config
    config = TrainingConfig(
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        encoder_cache_dir=args.encoder_cache,
        output_dir=args.output_dir,
        init_from=args.init_from,
        epochs=args.epochs,
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        learning_rate=args.lr,
        use_early_stopping=not args.no_early_stopping,
        # Optimizer settings
        optimizer=args.optimizer,
        sgd_momentum=args.sgd_momentum,
        # Encoder LoRA settings
        use_encoder_lora=not args.no_encoder_lora,
        encoder_lora_rank=args.encoder_lora_rank,
        encoder_lora_alpha=args.encoder_lora_alpha,
        encoder_lora_start_layer=args.encoder_lora_start_layer,
        # Prosody settings
        use_prosody=not args.no_prosody,
    )

    # Load manifests
    with open(config.train_manifest) as f:
        train_manifest = json.load(f)
    with open(config.val_manifest) as f:
        val_manifest = json.load(f)

    print(f"Train: {len(train_manifest)} samples")
    print(f"Val: {len(val_manifest)} samples")

    # NOTE: No fallback encoder - MLX Metal ops crash in background thread.
    # Training uses pre-cached samples only. Run pre-extraction in parallel
    # to grow the cache while training.

    # Create data loaders with prefetching
    train_loader = PrefetchDataLoader(
        manifest=train_manifest,
        encoder_cache_dir=config.encoder_cache_dir,
        batch_size=config.batch_size,
        prefetch_count=config.prefetch_batches,
        shuffle=True,
        fallback_encoder=None,  # Removed - crashes in bg thread
        use_prosody=config.use_prosody,
    )
    # Validation: Try cache-only first, fall back to no-cache if not available
    # Note: validation with no cache will skip samples without cached features
    try:
        val_loader = PrefetchDataLoader(
            manifest=val_manifest,
            encoder_cache_dir=config.encoder_cache_dir,
            batch_size=config.batch_size,
            prefetch_count=2,
            shuffle=False,
            fallback_encoder=None,
            use_prosody=config.use_prosody,
            cache_only=True,  # Try cache-only first
        )
    except ValueError:
        # No cached validation samples - create loader without cache-only
        print("WARNING: No cached validation samples. Validation will be skipped.")
        val_loader = None

    # Print cache status
    print(f"Using cached encoder features from: {config.encoder_cache_dir}")
    if config.use_prosody:
        print("Prosody features: ENABLED (from cache)")

    # Create trainer
    trainer = RichDecoderV4Trainer(config)

    # Train
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
