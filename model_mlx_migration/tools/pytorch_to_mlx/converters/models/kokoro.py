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
Kokoro-82M TTS Model for MLX

StyleTTS2-based text-to-speech model with ISTFTNet vocoder.
Converts from hexgrad/Kokoro-82M PyTorch checkpoint.

Architecture:
- CustomAlbert (BERT for phoneme understanding)
- TextEncoder (Conv1d + BiLSTM)
- ProsodyPredictor (Duration, F0, Noise)
- ISTFTNet Decoder (Audio synthesis)
"""

import concurrent.futures
import json
import math
import queue
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .kokoro_modules import (
    AdainResBlk1d,
    AdaINResBlock1dStyled,
    BiLSTM,
    CustomLayerNorm,
    KokoroConfig,
    PlainConv1d,
    ProsodyConditionedAdainResBlk1d,
    WeightNormConv1d,
    WeightNormConvTranspose1d,
)
from .kokoro_style_cache import StyleCache

# Note: STFT modules not used - direct harmonic generation is used instead

# Default number of prosody types (matching prosody_types.h)
NUM_PROSODY_TYPES = 63


# ============================================================================
# Prosody Embedding Module (Phase B)
# ============================================================================


class ProsodyEmbedding(nn.Module):
    """
    Prosody embedding table for Phase B training.

    Adds learned embeddings to BERT output based on prosody type.
    This enables natural prosody control (emphasis, emotion, etc.)
    after being trained on prosody-annotated speech data.

    Usage:
        # During model init (optional - disabled by default)
        model.enable_prosody_embedding(num_types=63, hidden_dim=768)

        # Load trained weights
        model.load_prosody_weights("models/prosody_embeddings/final.safetensors")

        # During inference with prosody annotations
        audio = model(input_ids, voice, prosody_mask=prosody_mask)
    """

    def __init__(self, num_types: int, hidden_dim: int, init_scale: float = 0.01):
        super().__init__()
        self.num_types = num_types
        self.hidden_dim = hidden_dim

        # Embedding table: [num_types, hidden_dim]
        self.embedding = nn.Embedding(num_types, hidden_dim)

        # Learnable scale factor (start small to not disrupt base model)
        self.scale = mx.array([init_scale])

    def __call__(self, prosody_mask: mx.array) -> mx.array:
        """
        Get prosody embeddings for input prosody mask.

        Args:
            prosody_mask: [batch, seq_len] int array of prosody type IDs

        Returns:
            embeddings: [batch, seq_len, hidden_dim] scaled prosody embeddings
        """
        emb = self.embedding(prosody_mask)  # [batch, seq_len, hidden_dim]
        return self.scale * emb


# ============================================================================
# Prosody Contour Predictor (Path C)
# ============================================================================


class ProsodyContourPredictor(nn.Module):
    """
    Prosody contour predictor for Path C F0 control.

    Instead of static F0 multipliers (v5), predicts 50-point F0 contours
    that capture the dynamic pitch patterns of emotional speech.

    The contour is applied as a multiplicative modifier to the predicted F0,
    interpolated to match the actual F0 length.

    Usage:
        # Enable contour-based prosody
        model.enable_prosody_contour(prosody_dim=768, hidden_dim=256)

        # Load trained weights
        model.load_prosody_contour_weights(
            contour_path="models/prosody_contour_v1/best_model.npz",
            embedding_path="models/prosody_embeddings_orthogonal/final.safetensors"
        )

        # Inference with prosody
        audio = model(input_ids, voice, prosody_mask=prosody_mask)
    """

    def __init__(
        self,
        prosody_dim: int = 768,
        hidden_dim: int = 256,
        contour_len: int = 50,
        num_prosody_types: int = NUM_PROSODY_TYPES,
    ):
        super().__init__()
        self.prosody_dim = prosody_dim
        self.hidden_dim = hidden_dim
        self.contour_len = contour_len
        self.num_prosody_types = num_prosody_types

        # Prosody embedding table
        self.embedding = nn.Embedding(num_prosody_types, prosody_dim)

        # Project prosody embedding
        self.prosody_proj = nn.Linear(prosody_dim, hidden_dim)

        # Learn contour pattern
        self.contour_fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.contour_fc2 = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.contour_out = nn.Linear(hidden_dim * 2, contour_len)

        # Scalar statistics (mean, std, range) for regularization
        self.stats_fc = nn.Linear(hidden_dim, 3)

    def __call__(self, prosody_id: mx.array) -> mx.array:
        """
        Predict F0 contour from prosody type ID.

        Args:
            prosody_id: [batch] or scalar - prosody type ID(s)

        Returns:
            contour: [batch, contour_len] or [contour_len] - normalized F0 [0, 1]
                     Relative F0 positions within the sample's range.
        """
        # Handle scalar input
        if prosody_id.ndim == 0:
            prosody_id = prosody_id[None]
            squeeze_output = True
        else:
            squeeze_output = False

        # Get prosody embedding
        prosody_emb = self.embedding(prosody_id)  # [batch, prosody_dim]

        # Project
        h = self.prosody_proj(prosody_emb)
        h = nn.gelu(h)

        # Contour prediction
        c = self.contour_fc1(h)
        c = nn.gelu(c)
        c = self.contour_fc2(c)
        c = nn.gelu(c)
        contour = mx.sigmoid(self.contour_out(c))  # [batch, contour_len] in [0, 1]

        if squeeze_output:
            contour = contour.squeeze(0)

        return contour

    def predict_f0_modifiers(
        self,
        prosody_mask: mx.array,
        f0_length: int,
        baseline_mean: float = 0.5,
        modifier_range: float = 0.3,
    ) -> mx.array:
        """
        Predict F0 multipliers from prosody mask, interpolated to F0 length.

        Args:
            prosody_mask: [batch, seq_len] - prosody type IDs per token
            f0_length: Length of F0 sequence to generate
            baseline_mean: Center point for neutral contour (default 0.5)
            modifier_range: Range of modification (default 0.3, ±0.15 -> 0.85-1.15x)

        Returns:
            f0_modifiers: [batch, f0_length] - multiplicative F0 modifiers
        """
        batch_size, seq_len = prosody_mask.shape

        # Get dominant prosody type per sample
        # For simplicity, use the first non-zero ID or the first element
        dominant_ids = []
        for b in range(batch_size):
            mask_b = np.array(prosody_mask[b])
            # Find first non-zero value, or first value if all zeros
            nonzero_indices = np.where(mask_b > 0)[0]
            if len(nonzero_indices) > 0:
                dominant_ids.append(int(mask_b[nonzero_indices[0]]))
            else:
                dominant_ids.append(int(mask_b[0]))  # Use first element

        dominant_ids = mx.array(dominant_ids)

        # Get contours for dominant prosody types
        contours = self(dominant_ids)  # [batch, contour_len]

        # Interpolate contours to f0_length
        # Simple linear interpolation
        contour_len = contours.shape[-1]
        if f0_length != contour_len:
            # Create interpolation indices
            src_indices = mx.linspace(0, contour_len - 1, f0_length)
            src_indices_floor = mx.floor(src_indices).astype(mx.int32)
            src_indices_ceil = mx.minimum(src_indices_floor + 1, contour_len - 1)
            weights = src_indices - src_indices_floor.astype(mx.float32)

            # Gather and interpolate
            contours_floor = mx.take(contours, src_indices_floor, axis=1)
            contours_ceil = mx.take(contours, src_indices_ceil, axis=1)
            contours = contours_floor * (1 - weights) + contours_ceil * weights

        # Convert normalized contour [0,1] to F0 multipliers
        # contour=0.5 -> 1.0x, contour=0.7 -> 1.0 + 0.2*range, etc.
        return 1.0 + (contours - baseline_mean) * modifier_range * 2



# ============================================================================
# Prosody Contour Predictor V2 (Path C v2.4)
# ============================================================================

# Emotion groupings for v2 architecture
HIGH_AROUSAL = {40, 42, 49, 50}  # angry, excited, nervous, surprised
LOW_AROUSAL = {41, 48}           # sad, frustrated (below 1.0 targets)
NEUTRAL_LIKE = {0, 45}           # neutral, calm (target = 1.0)


class ResidualBlock(nn.Module):
    """Residual block with layer norm for ProsodyContourPredictorV2."""

    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        x = self.norm(x)
        x = nn.gelu(self.fc1(x))
        x = self.fc2(x)
        return x + residual


class ProsodyContourPredictorV2(nn.Module):
    """
    Prosody contour predictor V2 for Path C v2.4.

    Improvements over V1:
    1. Emotion-aware architecture with separate high/low arousal paths
    2. Residual connections for better gradient flow
    3. Dual output heads (scalar multiplier + contour)
    4. Larger capacity (hidden_dim 512)

    Results (v2.4 vs v5):
    - ANGRY: 100% of target (v5: 66%)
    - SAD: 100% of target (v5: 93%)
    - EXCITED: 100% of target (v5: 71%)
    - CALM: v5 slightly better (1% error vs 0.6%)
    - NEUTRAL: TIE

    Usage:
        model.enable_prosody_contour_v2(prosody_dim=768, hidden_dim=512)
        model.load_prosody_contour_v2_weights(
            contour_path="models/prosody_contour_v2.4/best_model.npz"
        )
    """

    def __init__(
        self,
        prosody_dim: int = 768,
        hidden_dim: int = 512,
        contour_len: int = 50,
        num_prosody_types: int = NUM_PROSODY_TYPES,
        num_residual_blocks: int = 3,
    ):
        super().__init__()
        self.prosody_dim = prosody_dim
        self.hidden_dim = hidden_dim
        self.contour_len = contour_len
        self.num_prosody_types = num_prosody_types

        # Prosody embedding table
        self.embedding = nn.Embedding(num_prosody_types, prosody_dim)

        # Project prosody embedding
        self.prosody_proj = nn.Linear(prosody_dim, hidden_dim)
        self.proj_norm = nn.LayerNorm(hidden_dim)

        # Shared residual blocks
        self.shared_blocks = [
            ResidualBlock(hidden_dim) for _ in range(num_residual_blocks)
        ]

        # Arousal type embedding (learnable)
        # 0 = neutral_like, 1 = high arousal, 2 = low arousal
        self.arousal_embedding = nn.Embedding(3, hidden_dim // 4)

        # High arousal specific processing
        self.high_arousal_fc = nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim)
        self.high_arousal_block = ResidualBlock(hidden_dim)

        # Low arousal specific processing
        self.low_arousal_fc = nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim)
        self.low_arousal_block = ResidualBlock(hidden_dim)

        # Output heads
        # 1. Scalar multiplier (like v5) - primary objective
        self.multiplier_head = nn.Linear(hidden_dim, 1)

        # 2. Contour shape (normalized) - secondary objective
        self.contour_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.contour_fc2 = nn.Linear(hidden_dim, contour_len)

        # 3. Stats prediction (mean, std, range)
        self.stats_head = nn.Linear(hidden_dim, 3)

    def _get_arousal_type(self, prosody_id: int) -> int:
        """Map prosody ID to arousal type."""
        if prosody_id in NEUTRAL_LIKE:
            return 0  # neutral-like (target 1.0)
        if prosody_id in HIGH_AROUSAL:
            return 1  # high arousal
        return 2  # low arousal

    def __call__(self, prosody_id: mx.array) -> mx.array:
        """
        Predict F0 multiplier from prosody type ID.

        Args:
            prosody_id: [batch] or scalar - prosody type ID(s)

        Returns:
            multiplier: [batch] or scalar - F0 multiplier (e.g., 1.07 for +7%)
        """
        # Handle scalar input
        if prosody_id.ndim == 0:
            prosody_id = prosody_id[None]
            squeeze_output = True
        else:
            squeeze_output = False

        # Get prosody embedding
        prosody_emb = self.embedding(prosody_id)  # [batch, prosody_dim]

        # Initial projection
        h = self.prosody_proj(prosody_emb)
        h = nn.gelu(h)
        h = self.proj_norm(h)

        # Shared processing
        for block in self.shared_blocks:
            h = block(h)

        # Determine arousal type for each sample
        arousal_types = mx.array([
            self._get_arousal_type(int(pid)) for pid in prosody_id.tolist()
        ])

        # Get arousal embedding
        arousal_emb = self.arousal_embedding(arousal_types)  # (batch, hidden//4)

        # Concatenate and process through arousal-specific path
        h_with_arousal = mx.concatenate([h, arousal_emb], axis=-1)

        # Process high and low arousal separately
        h_high = nn.gelu(self.high_arousal_fc(h_with_arousal))
        h_high = self.high_arousal_block(h_high)

        h_low = nn.gelu(self.low_arousal_fc(h_with_arousal))
        h_low = self.low_arousal_block(h_low)

        # Create mask for routing
        is_high = (arousal_types == 1).astype(mx.float32).reshape(-1, 1)
        is_low = (arousal_types == 2).astype(mx.float32).reshape(-1, 1)
        is_neutral = (arousal_types == 0).astype(mx.float32).reshape(-1, 1)

        # Blend based on arousal type
        h_final = is_high * h_high + is_low * h_low + is_neutral * (h_high + h_low) / 2

        # Get multiplier output
        multiplier = self.multiplier_head(h_final)
        # Center around 1.0 with tanh for ±0.3 range (0.7 to 1.3)
        multiplier = 1.0 + 0.3 * mx.tanh(multiplier)

        if squeeze_output:
            multiplier = multiplier.squeeze()

        return multiplier

    def predict_f0_modifiers(
        self,
        prosody_mask: mx.array,
        f0_length: int,
    ) -> mx.array:
        """
        Predict F0 multipliers from prosody mask.

        Args:
            prosody_mask: [batch, seq_len] - prosody type IDs per token
            f0_length: Length of F0 sequence (for broadcasting)

        Returns:
            f0_modifiers: [batch, f0_length] - multiplicative F0 modifiers
        """
        batch_size, seq_len = prosody_mask.shape

        # Get dominant prosody type per sample
        dominant_ids = []
        for b in range(batch_size):
            mask_b = np.array(prosody_mask[b])
            # Find first non-zero value, or use first element if all zeros
            nonzero_indices = np.where(mask_b > 0)[0]
            if len(nonzero_indices) > 0:
                dominant_ids.append(int(mask_b[nonzero_indices[0]]))
            else:
                dominant_ids.append(int(mask_b[0]))

        dominant_ids = mx.array(dominant_ids)

        # Get F0 multipliers
        multipliers = self(dominant_ids)  # [batch] or [batch, 1]
        if multipliers.ndim == 1:
            multipliers = multipliers[:, None]

        # Broadcast to f0_length
        return mx.broadcast_to(multipliers, (batch_size, f0_length))



# ============================================================================
# Prosody Duration/Energy Predictor
# ============================================================================

# Data-driven duration multipliers from 86K multilingual samples
# Note: These are duration per character multipliers (longer = slower speech)
DURATION_MULTIPLIERS = {
    0: 1.00,   # NEUTRAL (baseline)
    40: 0.93,  # ANGRY (faster speech for anger)
    41: 1.15,  # SAD (slower speech for sadness)
    42: 0.95,  # EXCITED (slightly faster)
    45: 1.00,  # CALM (same as neutral)
    48: 1.05,  # FRUSTRATED (slightly slower)
    49: 0.90,  # NERVOUS (faster speech)
    50: 0.92,  # SURPRISED (slightly faster)
}

# Data-driven energy multipliers from 86K multilingual samples
ENERGY_MULTIPLIERS = {
    0: 1.00,   # NEUTRAL (baseline)
    40: 1.40,  # ANGRY (louder)
    41: 0.90,  # SAD (quieter)
    42: 1.30,  # EXCITED (louder)
    45: 0.95,  # CALM (slightly quieter)
    48: 1.35,  # FRUSTRATED (louder)
    49: 1.20,  # NERVOUS (louder)
    50: 1.50,  # SURPRISED (much louder)
}


class ProsodyDurationEnergyPredictor(nn.Module):
    """
    Prosody predictor for Duration and Energy modifiers.

    Predicts multiplicative modifiers for:
    - Duration: Controls speaking rate per emotion
    - Energy: Controls volume/loudness per emotion

    Architecture follows ProsodyContourPredictorV2 with arousal-aware paths.

    Usage:
        model.enable_prosody_duration_energy()
        model.load_prosody_duration_energy_weights(path)
        # During inference, duration and energy modifiers are applied automatically
    """

    def __init__(
        self,
        prosody_dim: int = 768,
        hidden_dim: int = 256,
        num_prosody_types: int = NUM_PROSODY_TYPES,
    ):
        super().__init__()
        self.prosody_dim = prosody_dim
        self.hidden_dim = hidden_dim
        self.num_prosody_types = num_prosody_types

        # Prosody embedding table (uses orthogonal embeddings)
        self.embedding = nn.Embedding(num_prosody_types, prosody_dim)

        # Project prosody embedding to hidden
        self.prosody_proj = nn.Linear(prosody_dim, hidden_dim)
        self.proj_norm = nn.LayerNorm(hidden_dim)

        # Shared residual block
        self.shared_block = ResidualBlock(hidden_dim)

        # Arousal type embedding (0=neutral, 1=high, 2=low)
        self.arousal_embedding = nn.Embedding(3, hidden_dim // 4)

        # Arousal-specific processing (including neutral for symmetry)
        self.neutral_arousal_fc = nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim)
        self.high_arousal_fc = nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim)
        self.low_arousal_fc = nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim)

        # Output heads
        self.duration_head = nn.Linear(hidden_dim, 1)
        self.energy_head = nn.Linear(hidden_dim, 1)

    def _get_arousal_type(self, prosody_id: int) -> int:
        """Map prosody ID to arousal type."""
        if prosody_id in NEUTRAL_LIKE:
            return 0  # neutral-like
        if prosody_id in HIGH_AROUSAL:
            return 1  # high arousal
        return 2  # low arousal

    def __call__(self, prosody_id: mx.array) -> tuple:
        """
        Predict duration and energy multipliers from prosody type ID.

        Args:
            prosody_id: (batch,) or scalar prosody type IDs

        Returns:
            duration_mult: (batch, 1) duration multipliers
            energy_mult: (batch, 1) energy multipliers
        """
        # Handle scalar input
        if prosody_id.ndim == 0:
            prosody_id = mx.expand_dims(prosody_id, 0)

        # Get prosody embeddings
        prosody_emb = self.embedding(prosody_id)  # (batch, prosody_dim)

        # Project and process
        h = self.prosody_proj(prosody_emb)
        h = nn.gelu(h)
        h = self.proj_norm(h)
        h = self.shared_block(h)

        # Get arousal types
        arousal_types = mx.array([
            self._get_arousal_type(int(pid)) for pid in prosody_id.tolist()
        ])

        # Get arousal embedding
        arousal_emb = self.arousal_embedding(arousal_types)

        # Concatenate and process
        h_with_arousal = mx.concatenate([h, arousal_emb], axis=-1)

        # Process through arousal-specific paths and combine
        h_neutral = nn.gelu(self.neutral_arousal_fc(h_with_arousal))
        h_high = nn.gelu(self.high_arousal_fc(h_with_arousal))
        h_low = nn.gelu(self.low_arousal_fc(h_with_arousal))

        # Blend based on arousal type
        is_high = (arousal_types == 1).astype(h.dtype).reshape(-1, 1)
        is_low = (arousal_types == 2).astype(h.dtype).reshape(-1, 1)
        is_neutral = (arousal_types == 0).astype(h.dtype).reshape(-1, 1)

        # Combined representation (all paths use FC processing for symmetry)
        h_combined = is_high * h_high + is_low * h_low + is_neutral * h_neutral

        # Output predictions (sigmoid * range + offset for bounded multipliers)
        # Duration: range [0.7, 1.4] centered at 1.0
        duration_raw = self.duration_head(h_combined)
        duration_mult = 0.7 + nn.sigmoid(duration_raw) * 0.7  # [0.7, 1.4]

        # Energy: range [0.7, 1.8] centered at 1.0
        energy_raw = self.energy_head(h_combined)
        energy_mult = 0.7 + nn.sigmoid(energy_raw) * 1.1  # [0.7, 1.8]

        return duration_mult, energy_mult

    def predict_modifiers(
        self,
        prosody_mask: mx.array,
        dur_length: int,
    ) -> tuple:
        """
        Predict duration and energy modifiers for a batch.

        Args:
            prosody_mask: (batch, seq_len) prosody type IDs per position
            dur_length: Length to broadcast modifiers to (for duration tensor)

        Returns:
            duration_mods: (batch, dur_length) duration multipliers
            energy_mods: (batch, 1) energy multipliers (scalar per utterance)
        """
        batch_size, seq_len = prosody_mask.shape

        # Get dominant prosody type per sample
        dominant_types = []
        for b in range(batch_size):
            mask_b = prosody_mask[b]
            non_neutral = mask_b[mask_b != 0]
            if non_neutral.size > 0:
                dominant_types.append(int(non_neutral[0]))
            else:
                dominant_types.append(0)
        dominant_types = mx.array(dominant_types)

        # Get multipliers
        duration_mult, energy_mult = self(dominant_types)

        # Broadcast duration to dur_length
        duration_mods = mx.broadcast_to(duration_mult, (batch_size, dur_length))

        return duration_mods, energy_mult


# ============================================================================
# ALBERT / BERT Module
# ============================================================================


class AlbertEmbeddings(nn.Module):
    """
    ALBERT embeddings: word + position + token_type.

    D5 Optimization: Caches position embeddings for common sequence lengths.
    """

    def __init__(self, config: KokoroConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.n_token, config.albert_embedding_dim)
        self.position_embeddings = nn.Embedding(
            config.plbert_max_position_embeddings, config.albert_embedding_dim,
        )
        self.token_type_embeddings = nn.Embedding(2, config.albert_embedding_dim)
        self.layer_norm = nn.LayerNorm(config.albert_embedding_dim)
        self.dropout = nn.Dropout(config.plbert_dropout)

        # D5: Position embedding cache for common sequence lengths
        self._pos_embed_cache: dict[int, mx.array] = {}
        self._max_cached_len = 512  # Cache up to this length

    def _get_position_embeds(self, seq_length: int) -> mx.array:
        """Get position embeddings, using cache if available."""
        if seq_length <= self._max_cached_len:
            if seq_length not in self._pos_embed_cache:
                position_ids = mx.arange(seq_length)[None, :]
                self._pos_embed_cache[seq_length] = self.position_embeddings(position_ids)
                mx.eval(self._pos_embed_cache[seq_length])
            return self._pos_embed_cache[seq_length]
        # Sequence too long for cache, compute directly
        position_ids = mx.arange(seq_length)[None, :]
        return self.position_embeddings(position_ids)

    def __call__(
        self,
        input_ids: mx.array,
        token_type_ids: mx.array | None = None,
        position_ids: mx.array | None = None,
    ) -> mx.array:
        seq_length = input_ids.shape[1]

        if token_type_ids is None:
            token_type_ids = mx.zeros_like(input_ids)

        word_embeds = self.word_embeddings(input_ids)

        # D5: Use cached position embeddings
        if position_ids is None:
            position_embeds = self._get_position_embeds(seq_length)
        else:
            position_embeds = self.position_embeddings(position_ids)

        token_type_embeds = self.token_type_embeddings(token_type_ids)

        embeddings = word_embeds + position_embeds + token_type_embeds
        embeddings = self.layer_norm(embeddings)
        return self.dropout(embeddings)



class AlbertAttention(nn.Module):
    """
    ALBERT self-attention with Q/K/V projections.

    Optimizations:
    - B3: Fused QKV projection using single matmul
    - P1: Optional local (sliding window) attention
    """

    def __init__(self, config: KokoroConfig):
        super().__init__()
        self.hidden_size = config.plbert_hidden_size
        self.num_heads = config.plbert_num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        # B3: Fused QKV projection - single matmul instead of 3 separate
        # Keep separate projections for weight loading compatibility
        self.query = nn.Linear(self.hidden_size, self.hidden_size)
        self.key = nn.Linear(self.hidden_size, self.hidden_size)
        self.value = nn.Linear(self.hidden_size, self.hidden_size)
        # Fused projection (built lazily after weight loading)
        self._qkv_fused: nn.Linear | None = None

        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(config.plbert_dropout)

        # P1: Local attention window size (None = full attention)
        self.local_window_size: int | None = None
        self._local_mask_cache: dict[int, mx.array] = {}

    def set_local_attention(self, window_size: int | None) -> None:
        """Enable/disable local (sliding window) attention.

        Args:
            window_size: Size of attention window. Each position attends to
                        window_size tokens on each side (2*window_size+1 total).
                        Set to None to disable local attention (full attention).
        """
        self.local_window_size = window_size
        self._local_mask_cache.clear()  # Clear cache on window change

    def _get_local_mask(self, seq_length: int) -> mx.array:
        """Get cached local attention mask for given sequence length.

        Creates a mask where position i can only attend to positions
        [i - window_size, i + window_size], reducing O(n²) to O(n*w).

        Returns:
            Attention mask with -inf for blocked positions, 0 for allowed.
            Shape: [1, 1, seq_length, seq_length] for broadcasting.
        """
        if seq_length in self._local_mask_cache:
            return self._local_mask_cache[seq_length]

        window = self.local_window_size
        # Create position indices
        rows = mx.arange(seq_length)[:, None]  # [seq, 1]
        cols = mx.arange(seq_length)[None, :]  # [1, seq]

        # Position i can attend to positions in [i-window, i+window]
        distance = mx.abs(rows - cols)
        mask = mx.where(distance <= window, 0.0, -1e9)

        # Shape for broadcasting: [1, 1, seq, seq]
        mask = mask[None, None, :, :]
        mx.eval(mask)

        self._local_mask_cache[seq_length] = mask
        return mask

    def _is_quantized(self) -> bool:
        """Check if Q/K/V layers are quantized (QuantizedLinear)."""
        # QuantizedLinear doesn't have a 'weight' attribute in the same way
        # It has 'scales' and 'biases' instead
        return hasattr(self.query, 'scales') or not hasattr(self.query, 'weight')

    def _build_fused_qkv(self) -> bool:
        """Build fused QKV projection from separate Q/K/V weights.

        Returns:
            True if fused projection is available, False if using separate Q/K/V.
        """
        if self._qkv_fused is not None:
            return True
        # Check if layers are quantized - can't fuse quantized layers
        if self._is_quantized():
            return False
        # Concatenate weights: [3*hidden, hidden]
        qkv_weight = mx.concatenate([
            self.query.weight,
            self.key.weight,
            self.value.weight,
        ], axis=0)
        # Concatenate biases if present: [3*hidden]
        if hasattr(self.query, 'bias') and self.query.bias is not None:
            qkv_bias = mx.concatenate([
                self.query.bias,
                self.key.bias,
                self.value.bias,
            ], axis=0)
        else:
            qkv_bias = None
        # Create fused linear
        self._qkv_fused = nn.Linear(self.hidden_size, self.hidden_size * 3)
        self._qkv_fused.weight = qkv_weight
        if qkv_bias is not None:
            self._qkv_fused.bias = qkv_bias
        return True

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        batch_size, seq_length, _ = hidden_states.shape

        # B3: Fused Q, K, V projections (single matmul) when possible
        # Falls back to separate projections for quantized models
        use_fused = self._build_fused_qkv()
        if use_fused:
            qkv = self._qkv_fused(hidden_states)  # [batch, seq, 3*hidden]
            q, k, v = mx.split(qkv, 3, axis=-1)  # Each [batch, seq, hidden]
        else:
            # Separate Q/K/V projections (for quantized models)
            q = self.query(hidden_states)
            k = self.key(hidden_states)
            v = self.value(hidden_states)

        # Reshape for multi-head attention: [batch, num_heads, seq_len, head_dim]
        q = q.reshape(batch_size, seq_length, self.num_heads, self.head_dim).transpose(
            0, 2, 1, 3,
        )
        k = k.reshape(batch_size, seq_length, self.num_heads, self.head_dim).transpose(
            0, 2, 1, 3,
        )
        v = v.reshape(batch_size, seq_length, self.num_heads, self.head_dim).transpose(
            0, 2, 1, 3,
        )

        # B8: Use mx.fast.scaled_dot_product_attention for fused kernel
        # Prepare mask (combine local attention + padding mask)
        combined_mask = None
        if self.local_window_size is not None:
            combined_mask = self._get_local_mask(seq_length)
        if attention_mask is not None:
            if combined_mask is not None:
                combined_mask = combined_mask + attention_mask
            else:
                combined_mask = attention_mask

        # SDPA with scale = 1/sqrt(head_dim)
        scale = 1.0 / math.sqrt(self.head_dim)
        context = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=scale, mask=combined_mask,
        )

        # Reshape back: [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, hidden]
        context = context.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_length, self.hidden_size,
        )

        # Output projection
        output = self.dense(context)
        output = self.dropout(output)

        # Residual + LayerNorm
        return self.layer_norm(hidden_states + output)



class AlbertLayer(nn.Module):
    """Single ALBERT layer: attention + FFN."""

    def __init__(self, config: KokoroConfig):
        super().__init__()
        self.attention = AlbertAttention(config)
        self.ffn = nn.Linear(config.plbert_hidden_size, config.plbert_intermediate_size)
        self.ffn_output = nn.Linear(
            config.plbert_intermediate_size, config.plbert_hidden_size,
        )
        self.full_layer_layer_norm = nn.LayerNorm(config.plbert_hidden_size)
        self.dropout = nn.Dropout(config.plbert_dropout)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        # Self-attention
        attention_output = self.attention(hidden_states, attention_mask)

        # FFN
        ffn_output = self.ffn(attention_output)
        ffn_output = nn.gelu(ffn_output)
        ffn_output = self.ffn_output(ffn_output)
        ffn_output = self.dropout(ffn_output)

        # Residual + LayerNorm
        return self.full_layer_layer_norm(attention_output + ffn_output)



class AlbertEncoder(nn.Module):
    """ALBERT encoder with shared layer weights."""

    def __init__(self, config: KokoroConfig):
        super().__init__()
        self.config = config

        # ALBERT shares weights across layers - single layer group
        self.albert_layer = AlbertLayer(config)

        # Mapping from embedding to hidden size
        self.embedding_hidden_mapping_in = nn.Linear(
            config.albert_embedding_dim, config.plbert_hidden_size,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
        save_debug: bool = False,
    ) -> mx.array:
        # Save embeddings before projection
        if save_debug:
            import numpy as np
            mx.eval(hidden_states)
            np.save("/tmp/py_bert_embeddings.npy", np.array(hidden_states))

        # Project embeddings to hidden size
        hidden_states = self.embedding_hidden_mapping_in(hidden_states)

        if save_debug:
            import numpy as np
            mx.eval(hidden_states)
            np.save("/tmp/py_bert_after_projection.npy", np.array(hidden_states))

        # Apply shared layer multiple times
        for i in range(self.config.plbert_num_hidden_layers):
            hidden_states = self.albert_layer(hidden_states, attention_mask)

            # Save first few and last layer outputs
            if save_debug and (i < 3 or i == self.config.plbert_num_hidden_layers - 1):
                import numpy as np
                mx.eval(hidden_states)
                np.save(f"/tmp/py_bert_layer_{i}.npy", np.array(hidden_states))

        return hidden_states


class CustomAlbert(nn.Module):
    """CustomAlbert (PLBERT) for Kokoro."""

    def __init__(self, config: KokoroConfig):
        super().__init__()
        self.embeddings = AlbertEmbeddings(config)
        self.encoder = AlbertEncoder(config)
        self.pooler = nn.Linear(config.plbert_hidden_size, config.plbert_hidden_size)
        self.save_debug_tensors = False

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
        token_type_ids: mx.array | None = None,
    ) -> mx.array:
        # Get embeddings
        embeddings = self.embeddings(input_ids, token_type_ids)

        # Create attention mask for transformer
        if attention_mask is not None:
            # Convert [batch, seq] to [batch, 1, 1, seq] for broadcasting
            extended_mask = attention_mask[:, None, None, :]
            extended_mask = (1.0 - extended_mask) * -10000.0
        else:
            extended_mask = None

        # Encode (pass save_debug flag)
        return self.encoder(embeddings, extended_mask,
                                     save_debug=self.save_debug_tensors)



# ============================================================================
# Text Encoder
# ============================================================================


class TextEncoder(nn.Module):
    """Text encoder: Embedding + Conv1d stack + BiLSTM."""

    def __init__(self, config: KokoroConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.n_token, config.hidden_dim)

        # 3 Conv1d layers with weight normalization
        self.convs = []
        self.norms = []
        for _i in range(3):
            self.convs.append(
                WeightNormConv1d(
                    config.hidden_dim,
                    config.hidden_dim,
                    config.text_encoder_kernel_size,
                    padding=config.text_encoder_kernel_size // 2,
                ),
            )
            self.norms.append(CustomLayerNorm(config.hidden_dim))

        # BiLSTM: hidden_dim -> hidden_dim//2 * 2
        self.lstm = BiLSTM(config.hidden_dim, config.hidden_dim // 2)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        """
        Args:
            x: [batch, length] - input token ids
            mask: [batch, length] - padding mask (True = padding, False = valid)

        Returns:
            [batch, length, hidden_dim] - NLC format
        """
        # Embedding: [batch, length, hidden_dim] (already NLC)
        x = self.embedding(x)

        # Apply mask if provided
        # Kokoro mask: True = padding (should be zeroed), False = valid
        # We need to zero out where mask is True, so multiply by ~mask
        if mask is not None:
            # Invert mask: True->False (0), False->True (1)
            valid_mask = (~mask)[:, :, None].astype(x.dtype)  # [batch, length, 1]
            x = x * valid_mask

        # Conv layers (all using NLC format)
        # Note: kokoro uses LeakyReLU(0.2), NOT ReLU
        for conv, norm in zip(self.convs, self.norms, strict=False):
            x = conv(x)
            x = norm(x)
            x = nn.leaky_relu(x, 0.2)
            if mask is not None:
                x = x * valid_mask

        # BiLSTM (already expects NLC format)
        return self.lstm(x)



# ============================================================================
# Prosody Predictor
# ============================================================================


class PredictorBiLSTM(nn.Module):
    """
    Bidirectional LSTM using native MLX nn.LSTM for GPU acceleration.

    PyTorch weights: weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0,
                    weight_ih_l0_reverse, weight_hh_l0_reverse,
                    bias_ih_l0_reverse, bias_hh_l0_reverse

    Input: [batch, length, input_size] (NLC)
    Output: [batch, length, hidden_size * 2] (NLC, concatenated forward + backward)

    Note: This implementation uses native nn.LSTM which runs on GPU instead of
    Python loops. Weights are converted during loading: bias = bias_ih + bias_hh.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # Native MLX LSTM layers
        self.lstm_fwd = nn.LSTM(input_size, hidden_size, bias=True)
        self.lstm_bwd = nn.LSTM(input_size, hidden_size, bias=True)
        # Store original PyTorch parameter names for weight loading
        # These will be populated during load_weights and converted to native format
        self._pytorch_params_loaded = False

    def load_pytorch_weights(self, weights_dict: dict[str, mx.array], prefix: str = ""):
        """
        Load PyTorch LSTM weights and convert to MLX native format.

        PyTorch format: weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0
        MLX format: Wx, Wh, bias (where bias = bias_ih + bias_hh)
        """

        def load_direction(lstm_layer, dir_prefix):
            weight_ih = weights_dict.get(f"{prefix}{dir_prefix}weight_ih_l0")
            weight_hh = weights_dict.get(f"{prefix}{dir_prefix}weight_hh_l0")
            bias_ih = weights_dict.get(f"{prefix}{dir_prefix}bias_ih_l0")
            bias_hh = weights_dict.get(f"{prefix}{dir_prefix}bias_hh_l0")

            if weight_ih is not None:
                lstm_layer.Wx = weight_ih
            if weight_hh is not None:
                lstm_layer.Wh = weight_hh
            if bias_ih is not None and bias_hh is not None:
                # Combine biases: bias = bias_ih + bias_hh
                lstm_layer.bias = bias_ih + bias_hh
            elif bias_ih is not None:
                lstm_layer.bias = bias_ih

        load_direction(self.lstm_fwd, "")
        load_direction(self.lstm_bwd, "reverse.")
        self._pytorch_params_loaded = True

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: [batch, length, input_size]
        Returns:
            [batch, length, hidden_size * 2]
        """
        # Forward LSTM: run on original sequence
        out_fwd, _ = self.lstm_fwd(x)  # [batch, length, hidden]

        # Backward LSTM: reverse input, run LSTM, reverse output
        x_rev = x[:, ::-1, :]  # Reverse sequence
        out_bwd_rev, _ = self.lstm_bwd(x_rev)
        out_bwd = out_bwd_rev[:, ::-1, :]  # Reverse output back

        # Concatenate forward and backward outputs
        return mx.concatenate([out_fwd, out_bwd], axis=-1)


class PredictorTextEncoderLSTM(nn.Module):
    """
    Single LSTM layer in predictor's text_encoder.lstms.
    Matches checkpoint keys: lstms.{i}.weight_ih_l0, etc.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.lstm = PredictorBiLSTM(input_size, hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.lstm(x)


class PredictorTextEncoderFC(nn.Module):
    """
    AdaLayerNorm for predictor's text_encoder.lstms (lstms.1, lstms.3, lstms.5).

    Matches kokoro's AdaLayerNorm which applies:
        1. Layer normalization on input x
        2. Scale and shift: (1 + gamma) * x_norm + beta

    Checkpoint keys: lstms.{i}.fc.weight, lstms.{i}.fc.bias
    """

    def __init__(self, hidden_dim: int, style_dim: int, eps: float = 1e-5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.eps = eps
        # Output is 2*hidden for gamma and beta
        self.fc = nn.Linear(style_dim, 2 * hidden_dim)

    def __call__(self, x: mx.array, style: mx.array) -> mx.array:
        """
        Apply AdaLayerNorm: layer norm + style-conditioned scale/shift.

        Args:
            x: [batch, length, hidden_dim] - input features
            style: [batch, style_dim] - style embedding
        Returns:
            [batch, length, hidden_dim] - normalized and scaled features
        """
        # Get gamma (scale) and beta (bias) from style
        h = self.fc(style)  # [batch, 2*hidden_dim]
        gamma, beta = mx.split(h, 2, axis=-1)  # [batch, hidden_dim] each

        # Layer normalization on x (over last dimension)
        # PyTorch's LayerNorm uses population variance (ddof=0)
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)  # ddof=0 to match LayerNorm
        x_norm = (x - mean) / mx.sqrt(var + self.eps)

        # Apply adaptive scale and shift
        # gamma, beta: [batch, hidden_dim] -> [batch, 1, hidden_dim]
        return (1 + gamma[:, None, :]) * x_norm + beta[:, None, :]


class PredictorTextEncoder(nn.Module):
    """
    Predictor's text_encoder module.

    Structure from checkpoint:
    - lstms.0: BiLSTM (640 -> 256*2 = 512)
    - lstms.1: FC for style (128 -> 1024, split to scale+bias for 512)
    - lstms.2: BiLSTM (640 -> 512)
    - lstms.3: FC for style
    - lstms.4: BiLSTM (640 -> 512)
    - lstms.5: FC for style
    """

    def __init__(self, hidden_dim: int = 512, style_dim: int = 128):
        super().__init__()
        # Input to LSTMs is hidden_dim + style_dim = 640
        input_size = hidden_dim + style_dim  # 640
        lstm_hidden = (
            hidden_dim // 2
        )  # 256, output will be 512 after bidirectional concat

        # Alternating LSTM and FC layers
        # Use numbered attributes to match checkpoint: lstms.0, lstms.1, etc.
        self.lstms_0 = PredictorBiLSTM(input_size, lstm_hidden)
        self.lstms_1 = PredictorTextEncoderFC(hidden_dim, style_dim)
        self.lstms_2 = PredictorBiLSTM(input_size, lstm_hidden)
        self.lstms_3 = PredictorTextEncoderFC(hidden_dim, style_dim)
        self.lstms_4 = PredictorBiLSTM(input_size, lstm_hidden)
        self.lstms_5 = PredictorTextEncoderFC(hidden_dim, style_dim)

    def __call__(self, x: mx.array, style: mx.array) -> mx.array:
        """
        Args:
            x: [batch, length, hidden_dim]
            style: [batch, style_dim]
        Returns:
            [batch, length, hidden_dim + style_dim] = [batch, length, 640]
            Note: PyTorch returns 640-dim features (512 + 128 style concatenated)
        """
        # Expand style: [batch, style_dim] -> [batch, length, style_dim]
        batch, length, _ = x.shape
        style_expanded = mx.broadcast_to(
            style[:, None, :], (batch, length, style.shape[-1]),
        )

        # Layer 0: BiLSTM
        x_cat = mx.concatenate([x, style_expanded], axis=-1)  # [batch, length, 640]
        x = self.lstms_0(x_cat)  # [batch, length, 512]

        # Layer 1: AdaLayerNorm (layer norm + scale/shift)
        x = self.lstms_1(x, style)

        # Re-concatenate style after AdaLayerNorm (kokoro's architecture does this)
        x_cat = mx.concatenate([x, style_expanded], axis=-1)

        # Layer 2: BiLSTM
        x = self.lstms_2(x_cat)

        # Layer 3: AdaLayerNorm
        x = self.lstms_3(x, style)

        # Re-concatenate style
        x_cat = mx.concatenate([x, style_expanded], axis=-1)

        # Layer 4: BiLSTM
        x = self.lstms_4(x_cat)

        # Layer 5: AdaLayerNorm
        x = self.lstms_5(x, style)

        # Return 640-dim features (512 + 128 style) to match PyTorch
        # PyTorch text_encoder returns duration_feats with style concatenated
        return mx.concatenate([x, style_expanded], axis=-1)  # [batch, length, 640]


class DurationProjWrapper(nn.Module):
    """
    Wrapper for duration_proj to match checkpoint structure.
    Checkpoint has: duration_proj.linear_layer.weight, duration_proj.linear_layer.bias
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear_layer = nn.Linear(input_dim, output_dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear_layer(x)


class ProsodyPredictor(nn.Module):
    """
    Prosody predictor: Duration, F0, Noise prediction.

    EXACTLY matches PyTorch checkpoint structure:
    - lstm: BiLSTM (640 -> 512)
    - shared: BiLSTM (640 -> 512)
    - text_encoder.lstms.{0,2,4}: BiLSTM with style concat
    - text_encoder.lstms.{1,3,5}: FC for AdaIN
    - F0.{0,1,2}: AdainResBlk1d blocks
    - N.{0,1,2}: AdainResBlk1d blocks
    - F0_proj: Plain Conv1d [1, 256, 1]
    - N_proj: Plain Conv1d [1, 256, 1]
    - duration_proj.linear_layer: Linear [50, 512]
    """

    def __init__(self, config: KokoroConfig):
        super().__init__()
        hidden_dim = config.hidden_dim  # 512
        style_dim = config.style_dim  # 128
        max_dur = config.max_dur  # 50

        # Input size for main LSTMs: hidden + style = 640
        lstm_input = hidden_dim + style_dim  # 640
        lstm_hidden = hidden_dim // 2  # 256, bidirectional gives 512

        # Main BiLSTM for duration/prosody processing
        self.lstm = PredictorBiLSTM(lstm_input, lstm_hidden)

        # Shared BiLSTM
        self.shared = PredictorBiLSTM(lstm_input, lstm_hidden)

        # Text encoder with style-conditioned LSTMs
        self.text_encoder = PredictorTextEncoder(hidden_dim, style_dim)

        # Duration projection (nested structure to match checkpoint)
        self.duration_proj = DurationProjWrapper(hidden_dim, max_dur)

        # F0 prediction: 3 AdainResBlk1d blocks
        # NOTE: Use dict-style attribute setting for numbered keys
        # Block 1: learned_upsample=True creates pool attr (F0.1.pool weights)
        self.F0_0 = AdainResBlk1d(512, 512, style_dim, upsample=False)
        self.F0_1 = AdainResBlk1d(
            512, 256, style_dim, upsample=True, learned_upsample=True,
        )
        self.F0_2 = AdainResBlk1d(256, 256, style_dim, upsample=False)

        # F0 projection
        self.F0_proj = PlainConv1d(256, 1, 1)

        # Noise prediction: 3 AdainResBlk1d blocks
        # Block 1 has learned_upsample=True (checkpoint has N.1.pool weights)
        self.N_0 = AdainResBlk1d(512, 512, style_dim, upsample=False)
        self.N_1 = AdainResBlk1d(
            512, 256, style_dim, upsample=True, learned_upsample=True,
        )
        self.N_2 = AdainResBlk1d(256, 256, style_dim, upsample=False)

        # Noise projection
        self.N_proj = PlainConv1d(256, 1, 1)

        # Phase C: Prosody-conditioned F0 blocks (optional, disabled by default)
        self._prosody_conditioning_enabled = False
        self._prosody_dim = 768  # Default to BERT output dim

    def enable_prosody_conditioning(
        self,
        prosody_dim: int = 768,
        prosody_scale: float = 0.1,
    ) -> None:
        """
        Enable prosody conditioning on F0 predictor (Phase C).

        Replaces standard AdainResBlk1d blocks with ProsodyConditionedAdainResBlk1d
        that can receive prosody embeddings to directly influence F0 output.

        This should be called BEFORE loading weights, then fc_prosody weights
        should be trained while keeping original weights frozen.

        Args:
            prosody_dim: Dimension of prosody embedding (default 768 for BERT)
            prosody_scale: Initial scale for prosody conditioning (default 0.1)
        """
        style_dim = 128  # Fixed for Kokoro

        # Replace F0 blocks with prosody-conditioned versions
        # NOTE: Original conv1, conv2, conv1x1, pool weights need to be copied
        self.F0_0_prosody = ProsodyConditionedAdainResBlk1d(
            512, 512, style_dim, prosody_dim,
            upsample=False, prosody_scale=prosody_scale,
        )
        self.F0_1_prosody = ProsodyConditionedAdainResBlk1d(
            512, 256, style_dim, prosody_dim,
            upsample=True, learned_upsample=True, prosody_scale=prosody_scale,
        )
        self.F0_2_prosody = ProsodyConditionedAdainResBlk1d(
            256, 256, style_dim, prosody_dim,
            upsample=False, prosody_scale=prosody_scale,
        )

        self._prosody_conditioning_enabled = True
        self._prosody_dim = prosody_dim

    def copy_f0_weights_to_prosody_blocks(self) -> None:
        """
        Copy weights from original F0 blocks to prosody-conditioned blocks.

        Call after enabling prosody conditioning AND loading original weights.
        The fc_prosody layers will be randomly initialized; train only those.
        """
        if not self._prosody_conditioning_enabled:
            raise ValueError(
                "Prosody conditioning not enabled. "
                "Call enable_prosody_conditioning() first.",
            )

        # Copy conv weights (WeightNormConv1d has weight_g and weight_v, not weight)
        for _i, (orig, prosody) in enumerate([
            (self.F0_0, self.F0_0_prosody),
            (self.F0_1, self.F0_1_prosody),
            (self.F0_2, self.F0_2_prosody),
        ]):
            # Copy conv1 weights (weight_v is the direction, weight_g is the magnitude)
            prosody.conv1.weight_v = orig.conv1.weight_v
            prosody.conv1.weight_g = orig.conv1.weight_g
            prosody.conv1.bias = orig.conv1.bias

            # Copy conv2 weights
            prosody.conv2.weight_v = orig.conv2.weight_v
            prosody.conv2.weight_g = orig.conv2.weight_g
            prosody.conv2.bias = orig.conv2.bias

            # Copy conv1x1 if present
            if orig.conv1x1 is not None and prosody.conv1x1 is not None:
                prosody.conv1x1.weight_v = orig.conv1x1.weight_v
                prosody.conv1x1.weight_g = orig.conv1x1.weight_g
                prosody.conv1x1.bias = orig.conv1x1.bias

            # Copy pool if present (for F0_1)
            if orig.pool is not None and prosody.pool is not None:
                prosody.pool.weight_v = orig.pool.weight_v
                prosody.pool.weight_g = orig.pool.weight_g
                prosody.pool.bias = orig.pool.bias

            # Copy AdaIN fc_style weights from original fc
            # norm1: ProsodyConditionedAdaIN.fc_style <- AdaIN.fc
            prosody.norm1.fc_style.weight = orig.norm1.fc.weight
            prosody.norm1.fc_style.bias = orig.norm1.fc.bias
            prosody.norm1.norm_weight = orig.norm1.norm_weight
            prosody.norm1.norm_bias = orig.norm1.norm_bias

            # norm2
            prosody.norm2.fc_style.weight = orig.norm2.fc.weight
            prosody.norm2.fc_style.bias = orig.norm2.fc.bias
            prosody.norm2.norm_weight = orig.norm2.norm_weight
            prosody.norm2.norm_bias = orig.norm2.norm_bias

    def __call__(
        self,
        text_enc: mx.array,
        style: mx.array,
        mask: mx.array | None = None,
        prosody_emb: mx.array | None = None,
    ) -> tuple[mx.array, mx.array, mx.array]:
        """
        Args:
            text_enc: [batch, length, hidden_dim] - text encoder output (NLC)
            style: [batch, style_dim] - style vector (128-dim speaker embedding)
            mask: [batch, length] - optional padding mask
            prosody_emb: [batch, prosody_dim] or [batch, length, prosody_dim]
                        Optional prosody embedding. Only used when prosody
                        conditioning is enabled (Phase C). Conditions F0 output.

        Returns:
            duration: [batch, length, max_dur] - duration logits
            f0: [batch, length*2] - F0 curve (upsampled 2x)
            noise: [batch, length*2] - noise signal (upsampled 2x)
        """
        batch, length, hidden_dim = text_enc.shape

        # Step 1: Process through text_encoder (DurationEncoder in PyTorch)
        # text_encoder takes 512-dim input and returns 640-dim (512 + 128 style)
        enc_out = self.text_encoder(text_enc, style)  # [batch, length, 640]

        # Step 2: Duration processing via main LSTM
        # PyTorch: x, _ = self.predictor.lstm(d); duration = duration_proj(x)
        dur_enc = self.lstm(enc_out)  # [batch, length, 512]
        duration = self.duration_proj(dur_enc)  # [batch, length, max_dur]

        # Step 3: F0/N prediction via shared LSTM then blocks
        # PyTorch F0Ntrain: x, _ = self.shared(en.transpose(-1, -2))
        # shared LSTM converts 640-dim to 512-dim for F0/N blocks
        shared_out = self.shared(enc_out)  # [batch, length, 512]

        # F0 prediction through AdainResBlk1d blocks (or prosody-conditioned if enabled)
        # PyTorch processes in NCL format but we stay in NLC
        x = shared_out
        if self._prosody_conditioning_enabled and prosody_emb is not None:
            # Phase C: Use prosody-conditioned blocks
            x = self.F0_0_prosody(x, style, prosody_emb)
            x = self.F0_1_prosody(x, style, prosody_emb)
            x = self.F0_2_prosody(x, style, prosody_emb)
        else:
            # Original behavior
            x = self.F0_0(x, style)
            x = self.F0_1(x, style)
            x = self.F0_2(x, style)
        f0 = self.F0_proj(x).squeeze(-1)  # [batch, length*2]

        # Noise prediction through AdainResBlk1d blocks
        x = shared_out
        x = self.N_0(x, style)
        x = self.N_1(x, style)
        x = self.N_2(x, style)
        noise = self.N_proj(x).squeeze(-1)  # [batch, length*2]

        return duration, f0, noise


# ============================================================================
# ISTFTNet Decoder
# ============================================================================


class SineGen(nn.Module):
    """
    Sine wave generator for F0-based harmonic source.

    Generates harmonic source signal from F0 curve.
    """

    def __init__(self, sample_rate: int = 24000, harmonic_num: int = 0):
        super().__init__()
        self.sample_rate = sample_rate
        self.harmonic_num = harmonic_num

    def __call__(self, f0: mx.array, upp: int) -> tuple[mx.array, mx.array, mx.array]:
        """
        Generate sine wave from F0.

        Args:
            f0: [batch, length] - F0 in Hz
            upp: Upsampling factor

        Returns:
            sine: [batch, length * upp] - Sine wave
            uv: [batch, length * upp] - Voiced/unvoiced mask
            noise: [batch, length * upp] - Noise component
        """
        batch, length = f0.shape

        # Upsample F0
        f0_up = mx.repeat(f0[:, :, None], upp, axis=1).squeeze(
            -1,
        )  # [batch, length * upp]

        # Generate phase
        phase = mx.cumsum(f0_up / self.sample_rate, axis=1) * 2 * math.pi

        # Generate sine wave
        sine = mx.sin(phase)

        # Voiced/unvoiced mask (where F0 > 0)
        uv = (f0_up > 0).astype(mx.float32)

        # Apply UV mask
        sine = sine * uv

        # Generate noise for unvoiced regions
        noise = mx.random.normal(sine.shape) * (1 - uv)

        return sine, uv, noise


class SourceModule(nn.Module):
    """
    Source module for harmonic source generation.

    Matches StyleTTS2 SourceModuleHnNSF + SineGen:
    1. Generate harmonics using SineGen logic with interpolation anti-aliasing
    2. Combine to 1 channel using l_linear [1, 9]
    3. Apply tanh for bounded output [-1, 1]

    STFT transform is applied in Generator's forward, not here.

    PyTorch weights:
    - l_linear.weight: [1, 9]
    - l_linear.bias: [1]

    Args:
        deterministic: When True, disable random phase and noise for testing.
                       When False (default), match PyTorch's stochastic behavior.
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        num_harmonics: int = 9,
        deterministic: bool = False,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.num_harmonics = num_harmonics
        self.sine_amp = 0.1
        self.noise_std = 0.003
        self.voiced_threshold = 10  # F0 > 10 Hz considered voiced
        self.deterministic = deterministic

        # Linear to combine harmonics: [num_harmonics] -> [1]
        # PyTorch weight shape: [1, 9]
        self.l_linear = nn.Linear(num_harmonics, 1, bias=True)

    def __call__(self, f0: mx.array, upp: int) -> tuple[mx.array, mx.array, mx.array]:
        """
        Generate harmonic source signal from F0.

        Matches PyTorch SineGen._f02sine with interpolation anti-aliasing:
        1. Upsample F0 to audio rate
        2. Generate harmonic frequencies (1f, 2f, ..., 9f)
        3. Convert to rad_values = (f0 * h / sample_rate) % 1
        4. Downsample by 1/upp for anti-aliasing
        5. cumsum to accumulate phase
        6. Multiply by upp and upsample back
        7. Apply sin() and amplitude

        Args:
            f0: [batch, length] - F0 curve in Hz
            upp: Upsampling factor (300)

        Returns:
            har_source: [batch, samples, 1] - Merged harmonic source
            noise: [batch, samples, 1] - Noise source
            uv: [batch, samples, 1] - Voiced/unvoiced mask
        """
        batch, length = f0.shape
        samples = length * upp

        # Upsample F0 to audio rate using nearest-neighbor
        f0_up = mx.repeat(f0[:, :, None], upp, axis=1).squeeze(-1)  # [batch, samples]

        # Voiced/unvoiced mask (threshold=10 Hz per PyTorch)
        uv = (f0_up > self.voiced_threshold).astype(mx.float32)  # [batch, samples]

        # Generate harmonics (1f, 2f, ..., 9f) with anti-aliasing
        # Vectorized across all harmonics simultaneously

        # Harmonic multipliers: [1, 2, ..., 9] -> [1, 1, num_harmonics]
        h_factors = mx.arange(1, self.num_harmonics + 1, dtype=mx.float32)[
            None, None, :,
        ]

        # Compute rad_values for all harmonics at once: [batch, samples, num_harmonics]
        # rad_values = (f0 * h / sample_rate) % 1
        f0_expanded = f0_up[:, :, None]  # [batch, samples, 1]
        rad_values = (
            f0_expanded * h_factors / self.sample_rate
        ) % 1.0  # [batch, samples, 9]

        # PyTorch adds random initial phase for harmonics > 1
        if not self.deterministic:
            rand_ini = mx.random.uniform(shape=(batch, 1, self.num_harmonics - 1))
            # Create mask: 0 for h=1, random for h>1
            rand_phase = mx.concatenate([mx.zeros((batch, 1, 1)), rand_ini], axis=2)
            # Add to first sample only
            first_sample = (rad_values[:, :1, :] + rand_phase) % 1.0
            rad_values = mx.concatenate([first_sample, rad_values[:, 1:, :]], axis=1)

        # Precompute interpolation indices (shared across all harmonics)
        # Downsample: samples -> length
        t_down = (mx.arange(length) + 0.5) * samples / length - 0.5
        t_down = mx.clip(t_down, 0, samples - 1)
        t_floor_down = mx.floor(t_down).astype(mx.int32)
        t_ceil_down = mx.minimum(t_floor_down + 1, samples - 1)
        t_frac_down = t_down - t_floor_down.astype(mx.float32)

        # Linear interpolation downsample: [batch, samples, 9] -> [batch, length, 9]
        rad_floor = rad_values[:, t_floor_down, :]
        rad_ceil = rad_values[:, t_ceil_down, :]
        rad_values_down = (
            rad_floor * (1 - t_frac_down[None, :, None])
            + rad_ceil * t_frac_down[None, :, None]
        )

        # Cumulative sum at low rate: [batch, length, 9]
        phase_low = mx.cumsum(rad_values_down, axis=1) * 2 * math.pi

        # Scale phase by upp: [batch, length, 9]
        phase_scaled = phase_low * upp

        # Upsample: length -> samples
        t_up = (mx.arange(samples) + 0.5) * length / samples - 0.5
        t_up = mx.clip(t_up, 0, length - 1)
        t_floor_up = mx.floor(t_up).astype(mx.int32)
        t_ceil_up = mx.minimum(t_floor_up + 1, length - 1)
        t_frac_up = t_up - t_floor_up.astype(mx.float32)

        # Linear interpolation upsample: [batch, length, 9] -> [batch, samples, 9]
        phase_floor = phase_scaled[:, t_floor_up, :]
        phase_ceil = phase_scaled[:, t_ceil_up, :]
        phase = (
            phase_floor * (1 - t_frac_up[None, :, None])
            + phase_ceil * t_frac_up[None, :, None]
        )

        # Compute sine for all harmonics: [batch, samples, 9]
        harmonics_stack = mx.sin(phase) * self.sine_amp

        # UV mask for harmonics
        uv_expanded = uv[:, :, None]  # [batch, samples, 1]

        # Noise: voiced regions get noise_std, unvoiced get sine_amp/3
        # PyTorch uses torch.randn_like for noise; we use zeros in deterministic mode
        noise_amp = uv_expanded * self.noise_std + (1 - uv_expanded) * self.sine_amp / 3
        if self.deterministic:
            noise = mx.zeros((batch, samples, self.num_harmonics))
        else:
            noise = noise_amp * mx.random.normal((batch, samples, self.num_harmonics))

        # Apply UV mask and add noise (per PyTorch: sine_waves * uv + noise)
        sine_waves = harmonics_stack * uv_expanded + noise

        # Combine harmonics to single channel using l_linear + tanh
        har_source = mx.tanh(self.l_linear(sine_waves))  # [batch, samples, 1]

        # Generate noise source (separate from harmonic noise)
        # PyTorch uses randn; we use zeros in deterministic mode
        if self.deterministic:
            noise_source = mx.zeros((batch, samples, 1))
        else:
            noise_source = mx.random.normal((batch, samples, 1)) * self.noise_std

        return har_source, noise_source, uv_expanded


class Generator(nn.Module):
    """
    ISTFTNet Generator for audio synthesis.

    Progressive upsampling with style-conditioned ResBlocks,
    culminating in ISTFT-based audio reconstruction.
    """

    def __init__(self, config: KokoroConfig):
        super().__init__()
        self.config = config

        style_dim = config.style_dim
        upsample_rates = config.istft_upsample_rates  # (10, 6)
        upsample_kernel_sizes = config.istft_upsample_kernel_sizes  # (20, 12)
        upsample_initial_channel = config.istft_upsample_initial_channel  # 512
        resblock_kernel_sizes = config.istft_resblock_kernel_sizes  # (3, 7, 11)
        resblock_dilation_sizes = config.istft_resblock_dilation_sizes  # ((1,3,5), ...)

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        # F0 upsampler - must include hop_size for audio-rate F0
        # Total samples = input_frames * product(upsample_rates) * istft_hop_size
        hop_size = config.istft_gen_istft_hop_size
        total_upsample = hop_size
        for r in upsample_rates:
            total_upsample *= r
        self.f0_upsamp = nn.Upsample(scale_factor=float(total_upsample), mode="nearest")

        # Source module
        self.m_source = SourceModule(sample_rate=24000)

        # Upsampling layers with weight normalization (matches PyTorch)
        # Use numbered attributes instead of list for proper weight loading
        for i, (rate, kernel) in enumerate(zip(upsample_rates, upsample_kernel_sizes, strict=False)):
            in_ch = upsample_initial_channel // (2**i)
            out_ch = upsample_initial_channel // (2 ** (i + 1))
            padding = (kernel - rate) // 2
            setattr(
                self,
                f"ups_{i}",
                WeightNormConvTranspose1d(
                    in_ch, out_ch, kernel, stride=rate, padding=padding,
                ),
            )

        # Noise convs for source injection
        # PyTorch Conv1d weight format: [out_channels, in_channels, kernel_size]
        # noise_convs.0: [256, 22, 12] -> out=256, in=22, kernel=12
        # noise_convs.1: [128, 22, 1] -> out=128, in=22, kernel=1
        # Use numbered attributes instead of lists for proper weight loading

        # Kernel sizes for noise_res blocks from PyTorch weights
        noise_res_kernels = [7, 11]  # noise_res.0: kernel=7, noise_res.1: kernel=11

        # From PyTorch weights: both noise_convs have in_channels=22
        # Kernel sizes are 12 and 1 for first and second stages
        noise_conv_kernels = [12, 1]

        # Stride calculation per StyleTTS2 upstream:
        # stride_f0 = np.prod(upsample_rates[i+1:])
        # Stage 0: stride = prod([6]) = 6
        # Stage 1: stride = prod([]) = 1
        noise_conv_strides = []
        for i in range(len(upsample_rates)):
            stride = 1
            for r in upsample_rates[i + 1 :]:
                stride *= r
            noise_conv_strides.append(stride)

        for i, _rate in enumerate(upsample_rates):
            ch = upsample_initial_channel // (2 ** (i + 1))

            # Both noise_convs have 22 input channels (from PyTorch weights)
            in_channels = 22
            kernel_size = noise_conv_kernels[i] if i < len(noise_conv_kernels) else 1
            stride = noise_conv_strides[i]
            # Per StyleTTS2 upstream: kernel=1 uses padding=0, others use (stride+1)//2
            padding = 0 if kernel_size == 1 else (stride + 1) // 2

            # noise_convs use plain Conv1d (not weight-norm) per StyleTTS2 upstream
            setattr(
                self,
                f"noise_convs_{i}",
                PlainConv1d(
                    in_channels, ch, kernel_size, stride=stride, padding=padding,
                ),
            )

            # Use AdaINResBlock1dStyled for noise_res
            res_kernel = noise_res_kernels[i] if i < len(noise_res_kernels) else 7
            setattr(
                self, f"noise_res_{i}", AdaINResBlock1dStyled(ch, res_kernel, style_dim),
            )

        # ResBlocks for each upsample stage
        # Generator uses AdaINResBlock1 with Snake1D activation and style conditioning
        # Use numbered attributes instead of list for proper weight loading
        resblock_idx = 0
        for i in range(self.num_upsamples):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for kernel, _dilations in zip(
                resblock_kernel_sizes, resblock_dilation_sizes, strict=False,
            ):
                # AdaINResBlock1dStyled has 3 layers with specified kernel
                setattr(
                    self,
                    f"resblocks_{resblock_idx}",
                    AdaINResBlock1dStyled(ch, kernel, style_dim),
                )
                resblock_idx += 1
        self._num_resblocks = resblock_idx

        # Post convolution for ISTFT spectrum
        final_ch = upsample_initial_channel // (2**self.num_upsamples)
        n_fft = config.istft_gen_istft_n_fft
        self.post_n_fft = n_fft
        self.conv_post = WeightNormConv1d(final_ch, n_fft + 2, 7, padding=3)

        # ISTFT parameters
        self.istft_n_fft = n_fft
        self.istft_hop_size = config.istft_gen_istft_hop_size

        # A3: Pre-compute ISTFT Hann window (avoid per-call rebuild)
        n_istft = mx.arange(n_fft, dtype=mx.float32)
        self._istft_window = 0.5 * (1 - mx.cos(2 * math.pi * n_istft / n_fft))
        # Pre-compute identity kernel for overlap-add
        self._istft_identity_kernel = mx.eye(n_fft, dtype=mx.float32).reshape(
            1, n_fft, n_fft,
        )
        # Pre-compute window kernel for normalization
        self._istft_window_kernel = (self._istft_window**2).reshape(1, n_fft, 1)

        # Source STFT parameters (for transforming harmonic source to 22-channel)
        self.source_stft_n_fft = 20
        self.source_stft_hop = 5
        # Pre-compute Hann window for source STFT
        n = mx.arange(self.source_stft_n_fft, dtype=mx.float32)
        self._source_window = 0.5 * (
            1 - mx.cos(2 * math.pi * n / self.source_stft_n_fft)
        )

    def _source_stft(self, x: mx.array) -> mx.array:
        """
        Apply STFT to source signal.

        Args:
            x: [batch, samples] - Input waveform

        Returns:
            har: [batch, frames, 22] - Concatenated magnitude and phase
        """
        batch, samples = x.shape
        n_fft = self.source_stft_n_fft
        hop = self.source_stft_hop

        # Pad signal with reflect mode: torch.stft(center=True, pad_mode="reflect")
        # PyTorch reflect: [a,b,c,d] with pad(2,2) -> [c,b,a,b,c,d,c,b]
        pad_amount = n_fft // 2

        # Manual reflect padding (without repeating edge)
        # Left pad: take x[:, 1:pad_amount+1] and reverse using slice [::-1]
        # Right pad: take x[:, -(pad_amount+1):-1] and reverse
        if pad_amount > 0:
            left_pad = x[:, 1 : pad_amount + 1][:, ::-1]
            right_pad = x[:, -(pad_amount + 1) : -1][:, ::-1]
            x_padded = mx.concatenate([left_pad, x, right_pad], axis=1)
        else:
            x_padded = x

        # Calculate number of frames
        padded_len = x_padded.shape[1]
        num_frames = (padded_len - n_fft) // hop + 1

        # Frame the signal
        indices = mx.arange(n_fft)[None, :] + mx.arange(num_frames)[:, None] * hop
        frames = x_padded[:, indices.flatten()].reshape(batch, num_frames, n_fft)

        # Apply window
        frames = frames * self._source_window

        # Real FFT
        spectrum = mx.fft.rfft(frames, axis=-1)
        # spectrum: [batch, num_frames, n_fft//2+1]

        # Extract magnitude and phase
        magnitude = mx.abs(spectrum)

        # Compute phase using arctan2
        # Normalize -0.0 to +0.0 before arctan2 to match PyTorch behavior.
        # MLX FFT can produce -0.0 for zero imaginary parts, while PyTorch
        # produces +0.0. arctan2(-0.0, neg) = -π vs arctan2(+0.0, neg) = +π.
        # Adding 0.0 normalizes -0.0 to +0.0.
        spectrum_real = spectrum.real + 0.0
        spectrum_imag = spectrum.imag + 0.0

        phase = mx.arctan2(spectrum_imag, spectrum_real)

        # Zero phase where magnitude is small (phase is arbitrary/noisy there)
        # This ensures consistent phase values between C++ and Python when FFT
        # bins have small magnitudes where residual diffs cause large phase
        # changes. Threshold 0.01 eliminates nearly all phase wrapping issues.
        mag_eps = 0.01
        phase = mx.where(magnitude < mag_eps, mx.zeros_like(phase), phase)

        # Concatenate: [batch, frames, 22]
        return mx.concatenate([magnitude, phase], axis=-1)


    def __call__(
        self,
        x: mx.array,
        s: mx.array,
        f0: mx.array,
        _debug_overrides: dict[str, mx.array] | None = None,
        style_cache: StyleCache | None = None,
    ) -> mx.array:
        """
        Generate audio from features.

        Args:
            x: [batch, length, channels] - Input features (NLC)
            s: [batch, style_dim] - Style vector
            f0: [batch, length_f0] - F0 curve
            _debug_overrides: Optional dict with keys 'har_source', 'noi_source',
                             'uv' to override for exact parity testing.
            style_cache: Optional StyleCache for N2 optimization.
                        If provided, uses cached fc outputs for AdaIN layers.

        Returns:
            audio: [batch, samples] - Generated waveform
        """
        # Calculate total upsampling factor per StyleTTS2 upstream:
        # upsample_scale = prod(upsample_rates) * hop_size
        total_upp = 1
        for r in self.config.istft_upsample_rates:
            total_upp *= r
        # Multiply by ISTFT hop size for correct F0-to-audio alignment
        total_upp *= self.istft_hop_size

        # Generate source signal from F0 (or use debug overrides)
        if _debug_overrides and "har_source" in _debug_overrides:
            har_source = _debug_overrides["har_source"]
            noise = _debug_overrides.get("noi_source", mx.zeros_like(har_source))
            uv = _debug_overrides.get(
                "uv", mx.zeros((har_source.shape[0], har_source.shape[1], 1)),
            )
        else:
            har_source, noise, uv = self.m_source(f0, total_upp)  # [batch, samples, 1]

        # Transform to STFT domain for noise_convs
        # har_source: [batch, samples, 1] -> [batch, samples]
        har_source_1d = har_source.squeeze(-1)

        # Apply STFT to get 22-channel source (or use override to bypass STFT)
        if _debug_overrides and "source" in _debug_overrides:
            # Direct STFT output override: [batch, frames, 22] NLC
            source = _debug_overrides["source"]
        else:
            source = self._source_stft(har_source_1d)  # [batch, frames, 22]

        # Save debug tensors if requested
        if getattr(self, "save_debug_tensors", False):
            import numpy as np

            mx.eval(har_source_1d, source)
            np.save("/tmp/py_har_source.npy", np.array(har_source_1d))
            np.save("/tmp/py_source_stft.npy", np.array(source))

        # Progressive upsampling per StyleTTS2 upstream order:
        # 1) leaky_relu(x)
        # 2) x_source = noise_conv(har) then noise_res(x_source, s)
        # 3) x = ups[i](x)
        # 4) if last stage: reflection pad (1,0)
        # 5) x = x + x_source
        # 6) resblocks
        for i in range(self.num_upsamples):
            up = getattr(self, f"ups_{i}")
            noise_conv = getattr(self, f"noise_convs_{i}")
            noise_res = getattr(self, f"noise_res_{i}")

            x = nn.leaky_relu(x, 0.1)

            # Save generator input (first iteration only)
            if i == 0 and getattr(self, "save_debug_tensors", False):
                import numpy as np
                mx.eval(x)
                np.save("/tmp/py_generator_input.npy", np.array(x))

            # Compute x_source BEFORE upsampling (per upstream)
            x_source = noise_conv(source)

            # Save x_source after noise_conv (first iteration only)
            if i == 0 and getattr(self, "save_debug_tensors", False):
                import numpy as np
                mx.eval(x_source)
                np.save("/tmp/py_x_source_after_nc0.npy", np.array(x_source))

            # N2: Get cached styles for this noise_res block
            noise_res_cache = None
            if style_cache is not None:
                path_prefix = f"decoder.generator.noise_res_{i}"
                noise_res_cache = {}
                for j in range(3):  # 3 layers per block
                    for adain_name in [f"adain1_{j}", f"adain2_{j}"]:
                        cached = style_cache.get(f"{path_prefix}.{adain_name}")
                        if cached is not None:
                            noise_res_cache[adain_name] = cached

            x_source = noise_res(x_source, s, cached_styles=noise_res_cache)

            # Save x_source after noise_res (first iteration only)
            if i == 0 and getattr(self, "save_debug_tensors", False):
                import numpy as np
                mx.eval(x_source)
                np.save("/tmp/py_x_source_after_nr0.npy", np.array(x_source))

            # Now upsample x
            x = up(x)

            # Save after upsample only (before source addition)
            if i == 0 and getattr(self, "save_debug_tensors", False):
                import numpy as np
                mx.eval(x)
                np.save("/tmp/py_after_ups0_only.npy", np.array(x))

            # Per StyleTTS2: apply reflection pad (1, 0) at last upsample stage
            # ReflectionPad1d((1,0)) reflects: [x0,x1,...] -> [x1,x0,x1,...]
            if i == self.num_upsamples - 1:
                # Manual reflect pad: prepend x[:, 1:2, :] to x
                x = mx.concatenate([x[:, 1:2, :], x], axis=1)

            # Add source signal - lengths should align with correct stride values
            # PyTorch doesn't need padding/trimming - dimensions should naturally align
            assert x_source.shape[1] == x.shape[1], (
                f"Generator length mismatch at stage {i}: "
                f"x={x.shape[1]}, x_source={x_source.shape[1]}. "
                f"Check upstream dimension calculations."
            )

            x = x + x_source

            # Save after each upsample stage for debug
            if getattr(self, "save_debug_tensors", False):
                import numpy as np
                mx.eval(x)
                np.save(f"/tmp/py_after_ups_{i}.npy", np.array(x))

            # Apply ResBlocks (with style for AdaINResBlock1)
            xs = None
            for j in range(self.num_kernels):
                block_idx = i * self.num_kernels + j
                if block_idx < self._num_resblocks:
                    resblock = getattr(self, f"resblocks_{block_idx}")

                    # N2: Get cached styles for this resblock
                    resblock_cache = None
                    if style_cache is not None:
                        path_prefix = f"decoder.generator.resblocks_{block_idx}"
                        resblock_cache = {}
                        for k in range(3):  # 3 layers per block
                            for adain_name in [f"adain1_{k}", f"adain2_{k}"]:
                                cached = style_cache.get(f"{path_prefix}.{adain_name}")
                                if cached is not None:
                                    resblock_cache[adain_name] = cached

                    if xs is None:
                        xs = resblock(x, s, cached_styles=resblock_cache)
                    else:
                        xs = xs + resblock(x, s, cached_styles=resblock_cache)
            if xs is not None:
                x = xs / self.num_kernels

            # Save after resblocks for debug
            if getattr(self, "save_debug_tensors", False):
                import numpy as np
                mx.eval(x)
                np.save(f"/tmp/py_after_resblocks_{i}.npy", np.array(x))

        # Final convolution
        # IMPORTANT: Use default leaky_relu slope (0.01), not 0.1
        # PyTorch uses F.leaky_relu(x) which defaults to 0.01
        x = nn.leaky_relu(x)  # slope=0.01 (default)

        # Save before conv_post for debug comparison
        if getattr(self, "save_debug_tensors", False):
            import numpy as np
            mx.eval(x)
            np.save("/tmp/py_before_conv_post.npy", np.array(x))

        x = self.conv_post(x)  # [batch, length, n_fft + 2]

        # Save conv_post output for debug comparison
        if getattr(self, "save_debug_tensors", False):
            import numpy as np
            mx.eval(x)
            np.save("/tmp/py_conv_post_output.npy", np.array(x))

        # ISTFTNet uses a learned spectrogram representation
        # conv_post outputs: [batch, frames, 22] for n_fft=20
        # 11 channels for magnitude (log scale), 11 for phase
        n_bins = self.post_n_fft // 2 + 1  # 11

        # Log-magnitude to magnitude
        # PyTorch: spec = torch.exp(x[:,:n_bins,:])
        log_mag = x[..., :n_bins]
        mag = mx.exp(log_mag)

        # Phase: apply sin() to network output to bound to [-1, 1]
        # PyTorch: phase = torch.sin(x[:,n_bins:,:])
        phase_logits = x[..., n_bins:]
        phase = mx.sin(phase_logits)

        # Allow direct ISTFT input override for exact parity testing
        # This bypasses the entire generator network and tests ISTFT in isolation
        if _debug_overrides and "istft_magnitude" in _debug_overrides:
            mag = _debug_overrides["istft_magnitude"]
            phase = _debug_overrides["istft_phase"]

        # ISTFT synthesis
        return self._istft_synthesis(mag, phase)


    def _istft_synthesis(self, mag: mx.array, phase: mx.array) -> mx.array:
        """
        ISTFT synthesis from magnitude and phase spectrograms.

        Implements proper overlap-add ISTFT matching torch.istft behavior.

        Args:
            mag: [batch, frames, n_fft//2+1] - Magnitude spectrogram (linear scale)
            phase: [batch, frames, n_fft//2+1] - Phase spectrogram (radians)

        Returns:
            audio: [batch, samples]
        """
        batch, frames, n_bins = mag.shape
        n_fft = self.istft_n_fft
        hop = self.istft_hop_size

        # Use irfft instead of full FFT (more efficient for real signals)
        # Construct complex spectrum from magnitude and phase
        spectrum = mag * mx.exp(1j * phase)

        # IRFFT per frame - automatically handles Hermitian symmetry
        time_frames = mx.fft.irfft(spectrum, n=n_fft, axis=-1)
        # time_frames: [batch, frames, n_fft]

        # A3: Use pre-computed Hann window (avoid per-call rebuild)
        window = self._istft_window

        # Apply window to each frame
        time_frames = time_frames * window

        # Overlap-add synthesis using conv_transpose1d (vectorized, ~100x faster)
        # Output length for PyTorch istft with center=True: (frames - 1) * hop
        # (matches original input length before center padding)
        output_length = (frames - 1) * hop

        # Vectorized overlap-add using pre-computed identity kernel
        # Cast kernel to match input dtype if needed
        weight = self._istft_identity_kernel.astype(time_frames.dtype)
        audio = mx.conv_transpose1d(time_frames, weight, stride=hop)[:, :, 0]

        # Window sum for normalization using pre-computed window kernel
        ones_input = mx.ones((1, frames, 1), dtype=time_frames.dtype)
        window_kernel = self._istft_window_kernel.astype(time_frames.dtype)
        window_sum = mx.conv_transpose1d(ones_input, window_kernel, stride=hop)[0, :, 0]

        # Normalize by window sum (avoid division by zero)
        window_sum = mx.maximum(window_sum, 1e-8)
        audio = audio / window_sum

        # Remove padding (center=True equivalent)
        # PyTorch istft with center=True removes n_fft//2 from each end
        pad = n_fft // 2
        return audio[:, pad : pad + output_length]


    def _istft_synthesis_chunked(
        self,
        mag: mx.array,
        phase: mx.array,
        chunk_frames: int = 100,
    ) -> mx.array:
        """
        I2 Optimization: Chunked ISTFT synthesis for streaming/pipelining.

        Processes ISTFT in chunks to enable:
        - Lower latency for streaming applications
        - Better memory efficiency for long sequences
        - Pipelined processing with other operations

        Args:
            mag: [batch, frames, n_fft//2+1] - Magnitude spectrogram
            phase: [batch, frames, n_fft//2+1] - Phase spectrogram
            chunk_frames: Number of frames per chunk (default 100)

        Returns:
            audio: [batch, samples] - Full synthesized waveform
        """
        batch, total_frames, n_bins = mag.shape
        n_fft = self.istft_n_fft
        hop = self.istft_hop_size

        # Overlap frames needed for proper windowing at chunk boundaries
        overlap_frames = (n_fft // hop) + 1

        audio_chunks = []

        for start_frame in range(0, total_frames, chunk_frames):
            # Include overlap for continuity
            chunk_start = max(0, start_frame - overlap_frames)
            chunk_end = min(total_frames, start_frame + chunk_frames + overlap_frames)

            # Extract chunk
            mag_chunk = mag[:, chunk_start:chunk_end, :]
            phase_chunk = phase[:, chunk_start:chunk_end, :]

            # Process chunk through ISTFT
            spectrum_chunk = mag_chunk * mx.exp(1j * phase_chunk)
            time_frames = mx.fft.irfft(spectrum_chunk, n=n_fft, axis=-1)
            time_frames = time_frames * self._istft_window

            # Overlap-add for this chunk
            weight = self._istft_identity_kernel.astype(time_frames.dtype)
            chunk_audio = mx.conv_transpose1d(time_frames, weight, stride=hop)[:, :, 0]

            # Window sum normalization
            chunk_frames_count = time_frames.shape[1]
            ones_input = mx.ones((1, chunk_frames_count, 1), dtype=time_frames.dtype)
            window_kernel = self._istft_window_kernel.astype(time_frames.dtype)
            window_sum = mx.conv_transpose1d(ones_input, window_kernel, stride=hop)[0, :, 0]
            window_sum = mx.maximum(window_sum, 1e-8)
            chunk_audio = chunk_audio / window_sum

            # Calculate output positions
            if chunk_start == 0:
                # First chunk - include from padding
                output_start = 0
            else:
                output_start = overlap_frames * hop

            if chunk_end == total_frames:
                # Last chunk - include to padding
                output_end = chunk_audio.shape[1]
            else:
                output_end = chunk_audio.shape[1] - overlap_frames * hop

            audio_chunks.append(chunk_audio[:, output_start:output_end])

        # Concatenate chunks
        audio = mx.concatenate(audio_chunks, axis=1)

        # Remove center padding
        pad = n_fft // 2
        output_length = (total_frames - 1) * hop
        return audio[:, pad : pad + output_length]


    def set_deterministic(self, value: bool = True) -> None:
        """Set deterministic mode for reproducible inference."""
        self.m_source.deterministic = value


class Decoder(nn.Module):
    """
    ISTFTNet Decoder for audio synthesis.

    Architecture (from StyleTTS2):
    - encode: AdainResBlk1d(dim_in + 2, 1024)
    - decode: 4 AdainResBlk1d blocks with concatenation
    - F0_conv/N_conv: Process F0 and noise signals
    - asr_res: ASR residual projection
    - generator: Progressive upsampling + ISTFT
    """

    def __init__(self, config: KokoroConfig):
        super().__init__()
        self.config = config
        style_dim = config.style_dim
        hidden_dim = config.hidden_dim  # 512

        # F0 and noise convolutions with stride=2
        self.f0_conv = WeightNormConv1d(1, 1, 3, stride=2, padding=1)
        self.n_conv = WeightNormConv1d(1, 1, 3, stride=2, padding=1)

        # ASR residual projection: 512 -> 64
        self.asr_res = WeightNormConv1d(hidden_dim, 64, 1)

        # Encode block: (512 + 2) -> 1024
        self.encode = AdainResBlk1d(hidden_dim + 2, 1024, style_dim)

        # Decode blocks: process 1024 + 64 + 2 = 1090 channels
        # 4 blocks, last one has learned upsampling via ConvTranspose1d
        # Use numbered attributes instead of list for proper weight loading
        decode_in = 1024 + 64 + 2  # 1090
        self.decode_0 = AdainResBlk1d(decode_in, 1024, style_dim)
        self.decode_1 = AdainResBlk1d(decode_in, 1024, style_dim)
        self.decode_2 = AdainResBlk1d(decode_in, 1024, style_dim)
        self.decode_3 = AdainResBlk1d(
            decode_in, 512, style_dim, upsample=True, learned_upsample=True,
        )

        # Generator for final audio synthesis
        self.generator = Generator(config)

    def __call__(
        self,
        asr_features: mx.array,
        f0: mx.array,
        noise: mx.array,
        style: mx.array,
        _debug_overrides: dict[str, mx.array] | None = None,
        style_cache: StyleCache | None = None,
        progressive_precision: bool = False,
    ) -> mx.array:
        """
        Synthesize audio from features.

        Args:
            asr_features: [batch, length, hidden_dim] - NLC format
            f0: [batch, length_f0] - F0 curve
            noise: [batch, length_n] - Noise signal
            style: [batch, style_dim] - Style vector
            _debug_overrides: Optional dict for parity testing.
                             Keys: 'har_source', 'noi_source', 'uv'
            style_cache: Optional StyleCache for N2 optimization.
            progressive_precision: If True, use BF16 for early layers (N7 optimization).
                                  encode, decode_0, decode_1 use BF16.
                                  decode_2, decode_3, generator use FP32.

        Returns:
            audio: [batch, samples] - Waveform
        """
        # Get original F0 for generator
        f0_orig = f0

        # Process F0 and noise through convolutions
        # Add channel dimension: [batch, length, 1]
        f0_in = f0[:, :, None]
        n_in = noise[:, :, None]

        f0_proc = self.f0_conv(f0_in)  # [batch, length//2, 1]
        n_proc = self.n_conv(n_in)  # [batch, length//2, 1]

        # Get ASR residual features
        asr_res = self.asr_res(asr_features)  # [batch, length, 64]

        # Match lengths after F0/N processing
        # F0/N are at 2x rate after predictor (F0_1/N_1 upsample 2x)
        # f0_conv/n_conv downsample 2x, so they should match ASR length
        asr_len = asr_features.shape[1]
        f0_len = f0_proc.shape[1]

        # Assert lengths match (Issue 4: previous hack was defensive, verify not needed)
        assert asr_len == f0_len, (
            f"[Issue 4] ASR/F0 length mismatch: asr={asr_len}, f0_proc={f0_len}. "
            f"F0 input was {f0.shape[1]}, expected {asr_len * 2}"
        )
        asr_down = asr_features
        asr_res_down = asr_res

        # Initial concatenation: [batch, length, 512 + 1 + 1]
        x = mx.concatenate([asr_down, f0_proc, n_proc], axis=-1)

        # N7: Progressive precision - use BF16 for early layers
        if progressive_precision:
            x = x.astype(mx.bfloat16)
            asr_res_down = asr_res_down.astype(mx.bfloat16)
            f0_proc = f0_proc.astype(mx.bfloat16)
            n_proc = n_proc.astype(mx.bfloat16)
            style_bf16 = style.astype(mx.bfloat16)
        else:
            style_bf16 = style

        # N2: Get cached styles for encode block
        encode_norm1 = style_cache.get("decoder.encode.norm1") if style_cache else None
        encode_norm2 = style_cache.get("decoder.encode.norm2") if style_cache else None

        # Encode (BF16 if progressive_precision)
        x = self.encode(
            x, style_bf16, cached_norm1=encode_norm1, cached_norm2=encode_norm2,
        )  # [batch, length, 1024]

        # Decode blocks with residual concatenation
        # N7: decode_0, decode_1 in BF16; decode_2, decode_3 in FP32
        bf16_blocks = {"decode_0", "decode_1"}
        decode_blocks = [
            ("decode_0", self.decode_0),
            ("decode_1", self.decode_1),
            ("decode_2", self.decode_2),
            ("decode_3", self.decode_3),
        ]
        for block_name, block in decode_blocks:
            # N7: Switch to FP32 for later blocks
            if progressive_precision and block_name not in bf16_blocks:
                x = x.astype(mx.float32)
                asr_res_down = asr_res_down.astype(mx.float32)
                f0_proc = f0_proc.astype(mx.float32)
                n_proc = n_proc.astype(mx.float32)
                style_for_block = style
            elif progressive_precision:
                style_for_block = style_bf16
            else:
                style_for_block = style

            # N2: Get cached styles for this decode block
            norm1 = style_cache.get(f"decoder.{block_name}.norm1") if style_cache else None
            norm2 = style_cache.get(f"decoder.{block_name}.norm2") if style_cache else None

            # Concatenate with residuals before each decode block
            x = mx.concatenate([x, asr_res_down, f0_proc, n_proc], axis=-1)
            x = block(x, style_for_block, cached_norm1=norm1, cached_norm2=norm2)

            # After upsampling block, need to adjust residual lengths
            if block.upsample:
                new_len = x.shape[1]
                # Upsample residuals to match
                asr_res_down = mx.repeat(asr_res_down, 2, axis=1)[:, :new_len, :]
                f0_proc = mx.repeat(f0_proc, 2, axis=1)[:, :new_len, :]
                n_proc = mx.repeat(n_proc, 2, axis=1)[:, :new_len, :]

        # N7: Ensure x is FP32 for generator
        if progressive_precision:
            x = x.astype(mx.float32)

        # Generate audio (always FP32)
        return self.generator(
            x, style, f0_orig, _debug_overrides=_debug_overrides, style_cache=style_cache,
        )


    def set_deterministic(self, value: bool = True) -> None:
        """Set deterministic mode for reproducible inference."""
        self.generator.set_deterministic(value)


# ============================================================================
# D1/I4: Audio Buffer Pool for Memory Reuse
# ============================================================================


class AudioBufferPool:
    """
    D1/I4 Optimization: Reusable audio buffer pool.

    Maintains a pool of pre-allocated audio buffers to avoid
    repeated memory allocations during streaming synthesis.

    Usage:
        pool = AudioBufferPool(max_samples=480000, num_buffers=4)
        buffer = pool.acquire(needed_samples)  # Get a buffer
        # ... fill buffer with audio ...
        pool.release(buffer)  # Return to pool
    """

    def __init__(
        self,
        max_samples: int = 480000,  # 20 seconds at 24kHz
        num_buffers: int = 4,
    ):
        """
        Initialize buffer pool.

        Args:
            max_samples: Maximum samples per buffer (determines buffer size)
            num_buffers: Number of buffers to pre-allocate
        """
        self.max_samples = max_samples
        self.num_buffers = num_buffers
        self._buffers: list[mx.array] = []
        self._available: list[bool] = []
        self._initialized = False

    def _initialize(self) -> None:
        """Lazy initialization of buffers."""
        if self._initialized:
            return
        for _ in range(self.num_buffers):
            # Pre-allocate buffer
            buf = mx.zeros((1, self.max_samples))
            mx.eval(buf)
            self._buffers.append(buf)
            self._available.append(True)
        self._initialized = True

    def acquire(self, needed_samples: int) -> mx.array | None:
        """
        Acquire a buffer from the pool.

        Args:
            needed_samples: Number of samples needed

        Returns:
            A buffer slice if available and fits, None otherwise
        """
        self._initialize()

        if needed_samples > self.max_samples:
            return None  # Request too large, caller should allocate

        # Find available buffer
        for i, available in enumerate(self._available):
            if available:
                self._available[i] = False
                return self._buffers[i][:, :needed_samples]

        return None  # All buffers in use

    def release(self, buffer: mx.array) -> None:
        """
        Release a buffer back to the pool.

        Args:
            buffer: Buffer previously acquired from this pool
        """
        # Find which buffer this is (by checking if it shares memory)
        # For simplicity, just mark the first unavailable as available
        for i, available in enumerate(self._available):
            if not available:
                self._available[i] = True
                return

    def clear(self) -> None:
        """
        Clear all buffers and reset pool state.

        Releases all memory held by the pool. Call this when done
        with streaming to free resources.
        """
        self._buffers.clear()
        self._available.clear()
        self._initialized = False

    def reset(self) -> None:
        """
        Mark all buffers as available without deallocating.

        Use this between synthesis runs to reuse existing buffers.
        """
        for i in range(len(self._available)):
            self._available[i] = True

    @property
    def stats(self) -> dict:
        """Get pool statistics."""
        if not self._initialized:
            return {'initialized': False, 'buffers': 0, 'available': 0}
        return {
            'initialized': True,
            'buffers': len(self._buffers),
            'available': sum(self._available),
            'in_use': sum(not a for a in self._available),
            'max_samples': self.max_samples,
        }


class AsyncAudioWriter:
    """
    I3 Optimization: Asynchronous audio chunk writer.

    Writes audio chunks to a file or callback in a background thread,
    allowing synthesis to continue without blocking on I/O.

    Usage:
        writer = AsyncAudioWriter(output_path="output.wav", sample_rate=24000)
        for chunk in model.synthesize_streaming(...):
            writer.write(chunk)
        writer.close()  # Wait for all writes to complete
    """

    def __init__(
        self,
        output_path: str | None = None,
        callback: Callable[[np.ndarray], None] | None = None,
        sample_rate: int = 24000,
        queue_size: int = 10,
    ):
        """
        Initialize async writer.

        Args:
            output_path: Path to write WAV file (optional)
            callback: Function to call with each chunk (optional)
            sample_rate: Audio sample rate for WAV header
            queue_size: Max queued chunks before blocking
        """
        self.output_path = output_path
        self.callback = callback
        self.sample_rate = sample_rate

        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._chunks: list[np.ndarray] = []
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._started = False
        self._error: Exception | None = None

    def _writer_thread(self) -> None:
        """Background thread that processes the write queue."""
        try:
            while not self._stop_event.is_set() or not self._queue.empty():
                try:
                    chunk = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                # Convert to numpy if needed
                if hasattr(chunk, '__array__'):
                    chunk_np = np.array(chunk)
                else:
                    chunk_np = chunk

                # Ensure 1D
                if chunk_np.ndim > 1:
                    chunk_np = chunk_np.flatten()

                self._chunks.append(chunk_np)

                # Call callback if provided
                if self.callback is not None:
                    self.callback(chunk_np)

                self._queue.task_done()

        except Exception as e:
            self._error = e

    def start(self) -> 'AsyncAudioWriter':
        """Start the background writer thread."""
        if not self._started:
            self._thread = threading.Thread(target=self._writer_thread, daemon=True)
            self._thread.start()
            self._started = True
        return self

    def write(self, chunk: mx.array | np.ndarray) -> None:
        """
        Queue a chunk for async writing.

        Args:
            chunk: Audio chunk to write
        """
        if not self._started:
            self.start()

        if self._error:
            raise self._error

        self._queue.put(chunk)

    def close(self, timeout: float = 30.0) -> np.ndarray:
        """
        Wait for all writes to complete and finalize output.

        Args:
            timeout: Maximum wait time in seconds

        Returns:
            Concatenated audio as numpy array
        """
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=timeout)

        if self._error:
            raise self._error

        # Concatenate all chunks
        if self._chunks:
            full_audio = np.concatenate(self._chunks)
        else:
            full_audio = np.array([], dtype=np.float32)

        # Write WAV file if path provided
        if self.output_path is not None:
            self._write_wav(full_audio)

        return full_audio

    def _write_wav(self, audio: np.ndarray) -> None:
        """Write audio to WAV file."""
        import wave

        # Normalize to int16
        audio_int16 = (audio * 32767).astype(np.int16)

        with wave.open(self.output_path, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_int16.tobytes())

    @property
    def stats(self) -> dict:
        """Get writer statistics."""
        return {
            'started': self._started,
            'chunks_written': len(self._chunks),
            'queue_size': self._queue.qsize(),
            'has_error': self._error is not None,
        }


class BatchedRequestProcessor:
    """
    J10 Optimization: Multi-request batching processor.

    Queues multiple synthesis requests and processes them together
    in batches for higher throughput.

    Usage:
        processor = BatchedRequestProcessor(model, max_batch_size=4)
        processor.start()

        # Submit requests from multiple threads
        future1 = processor.submit(input_ids1, voice1)
        future2 = processor.submit(input_ids2, voice2)

        # Get results
        audio1 = future1.result()
        audio2 = future2.result()

        processor.stop()
    """

    def __init__(
        self,
        model: 'KokoroModel',
        max_batch_size: int = 4,
        max_wait_ms: float = 50.0,
        speed: float = 1.0,
    ):
        """
        Initialize batched processor.

        Args:
            model: KokoroModel instance for synthesis
            max_batch_size: Maximum requests per batch
            max_wait_ms: Maximum time to wait for batch to fill (ms)
            speed: Speaking rate for all requests
        """
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.speed = speed

        self._request_queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._started = False

    def start(self) -> 'BatchedRequestProcessor':
        """Start the batch processing thread."""
        if not self._started:
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._process_loop, daemon=True)
            self._thread.start()
            self._started = True
        return self

    def stop(self, timeout: float = 10.0) -> None:
        """Stop the batch processing thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._started = False

    def submit(
        self,
        input_ids: mx.array,
        voice: mx.array,
    ) -> 'concurrent.futures.Future':
        """
        Submit a synthesis request for batched processing.

        Args:
            input_ids: [length] phoneme token ids
            voice: [256] voice embedding

        Returns:
            Future that will contain the audio array when complete
        """
        from concurrent.futures import Future

        if not self._started:
            self.start()

        future: Future = Future()
        self._request_queue.put({
            'input_ids': input_ids,
            'voice': voice,
            'future': future,
        })
        return future

    def _process_loop(self) -> None:
        """Background thread that batches and processes requests."""
        import time

        while not self._stop_event.is_set():
            # Collect requests up to batch size or timeout
            batch = []
            batch_start = time.perf_counter()

            while len(batch) < self.max_batch_size:
                remaining_ms = self.max_wait_ms - (time.perf_counter() - batch_start) * 1000
                if remaining_ms <= 0 and len(batch) > 0:
                    break

                try:
                    timeout = max(0.001, remaining_ms / 1000.0)
                    request = self._request_queue.get(timeout=timeout)
                    batch.append(request)
                except queue.Empty:
                    if len(batch) > 0:
                        break
                    continue

            if not batch:
                continue

            # Process batch
            self._process_batch(batch)

    def _process_batch(self, batch: list[dict]) -> None:
        """Process a batch of requests."""
        try:
            if len(batch) == 1:
                # Single request - no batching needed
                req = batch[0]
                input_ids = req['input_ids']
                if input_ids.ndim == 1:
                    input_ids = input_ids[None, :]
                voice = req['voice']
                if voice.ndim == 1:
                    voice = voice[None, :]

                audio = self.model(input_ids, voice, speed=self.speed, validate_output=False)
                mx.eval(audio)
                req['future'].set_result(audio[0])
            else:
                # Multiple requests - use batch synthesis
                input_ids_list = []
                voices_list = []

                for req in batch:
                    ids = req['input_ids']
                    if ids.ndim == 2:
                        ids = ids[0]
                    input_ids_list.append(ids)

                    voice = req['voice']
                    if voice.ndim == 1:
                        voice = voice[None, :]
                    voices_list.append(voice)

                # Stack voices
                voices = mx.concatenate(voices_list, axis=0)

                # Batch synthesize
                audio_batch, lengths = self.model.synthesize_batch(
                    input_ids_list, voices, speed=self.speed, validate_output=False,
                )
                mx.eval(audio_batch, lengths)

                # Distribute results
                for i, req in enumerate(batch):
                    audio_length = int(lengths[i])
                    req['future'].set_result(audio_batch[i, :audio_length])

        except Exception as e:
            # Set exception on all futures
            for req in batch:
                req['future'].set_exception(e)

    @property
    def stats(self) -> dict:
        """Get processor statistics."""
        return {
            'started': self._started,
            'queue_size': self._request_queue.qsize(),
            'max_batch_size': self.max_batch_size,
            'max_wait_ms': self.max_wait_ms,
        }


# ============================================================================
# Full Kokoro Model
# ============================================================================


class KokoroModel(nn.Module):
    """
    Kokoro-82M TTS Model.

    Full pipeline: text -> BERT -> TextEncoder -> Predictor -> Decoder -> audio
    """

    def __init__(self, config: KokoroConfig):
        super().__init__()
        self.config = config

        # Components
        self.bert = CustomAlbert(config)
        self.bert_encoder = nn.Linear(config.plbert_hidden_size, config.hidden_dim)
        self.text_encoder = TextEncoder(config)
        self.predictor = ProsodyPredictor(config)
        self.decoder = Decoder(config)

        # Prosody embedding (Phase B - optional, disabled by default)
        self.prosody_embedding: ProsodyEmbedding | None = None

        # Prosody contour predictor (Path C - optional, disabled by default)
        self.prosody_contour_predictor: ProsodyContourPredictor | None = None

        # Prosody contour predictor V2 (Path C v2.4 - optional, disabled by default)
        self.prosody_contour_predictor_v2: ProsodyContourPredictorV2 | None = None

        # Prosody duration/energy predictor (optional, disabled by default)
        self.prosody_duration_energy_predictor: ProsodyDurationEnergyPredictor | None = None

        # Emotion optimization caches (enabled via initialize_emotion_cache)
        self._emotion_cache: dict[int, dict[str, Any]] | None = None
        self._compiled_decoder: Callable | None = None
        self._use_compiled_decoder: bool = False

    def initialize_emotion_cache(
        self,
        embedding_table: mx.array,
        unified_model: Any | None = None,
        emotion_ids: list[int] | None = None,
        compile_decoder: bool = True,
    ) -> float:
        """
        Initialize emotion/prosody cache for optimized inference.

        This precomputes prosody embeddings and optionally unified model outputs
        for all emotion IDs, eliminating per-inference computation overhead.
        Also optionally compiles the decoder for additional speedup.

        Args:
            embedding_table: Prosody embedding table [num_emotions, prosody_dim]
            unified_model: Optional UnifiedProsodyPredictor for contour caching
            emotion_ids: List of emotion IDs to cache (default: 0-12 core emotions)
            compile_decoder: Whether to compile decoder with mx.compile (default True)

        Returns:
            Setup time in milliseconds

        Example:
            >>> # Load prosody weights
            >>> prosody_weights = mx.load('models/prosody_embeddings/final.safetensors')
            >>> embedding_table = prosody_weights['embedding.weight']
            >>> # Initialize cache
            >>> setup_ms = model.initialize_emotion_cache(embedding_table)
            >>> print(f"Cache initialized in {setup_ms:.1f}ms")
        """
        import time
        t0 = time.perf_counter()

        # Default emotion IDs: neutral + 12 core emotions from prosody_types.h
        if emotion_ids is None:
            emotion_ids = [0, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

        # Initialize cache dict
        self._emotion_cache = {}

        # Precompute all prosody embeddings
        for eid in emotion_ids:
            cache_entry: dict[str, Any] = {}

            # Embedding lookup
            emb = embedding_table[mx.array([eid])]
            cache_entry['embedding'] = emb

            # If unified model provided, cache its outputs too
            if unified_model is not None:
                id_arr = mx.array([eid])
                _, contour, dur_mult, energy_mult = unified_model(emb, id_arr)
                mx.eval(contour, dur_mult, energy_mult)
                cache_entry['f0_contour'] = contour
                cache_entry['duration_mult'] = float(np.array(dur_mult).flatten()[0])
                cache_entry['energy_mult'] = float(np.array(energy_mult).flatten()[0])

            self._emotion_cache[eid] = cache_entry

        # Evaluate all embeddings
        mx.eval(*[c['embedding'] for c in self._emotion_cache.values()])

        # Optionally compile decoder
        # Note: mx.compile creates a lazy JIT function that compiles on first call
        # No warmup needed - compilation happens automatically on first real inference
        if compile_decoder:
            self._compiled_decoder = mx.compile(self.decoder)
            self._use_compiled_decoder = True

        return (time.perf_counter() - t0) * 1000

    def get_cached_emotion(self, emotion_id: int) -> dict[str, Any] | None:
        """
        Get cached prosody data for an emotion ID.

        Returns None if cache not initialized or emotion_id not cached.
        """
        if self._emotion_cache is None:
            return None
        return self._emotion_cache.get(emotion_id)

    def set_use_compiled_decoder(self, use: bool) -> None:
        """Enable or disable compiled decoder usage."""
        if use and self._compiled_decoder is None:
            raise ValueError(
                "Compiled decoder not available. Call initialize_emotion_cache first.",
            )
        self._use_compiled_decoder = use

    def compile_predictor_lstm(self) -> float:
        """
        Compile predictor BiLSTM components for faster inference.

        This applies mx.compile to the BiLSTM layers in the predictor,
        providing ~1.07x overall speedup (~23ms saved on typical inference).

        The BiLSTM components benefit significantly from compilation because
        they involve sequential operations that MLX can optimize.

        Returns:
            Setup time in milliseconds

        Example:
            >>> model = KokoroModel(config)
            >>> model.load_weights(...)
            >>> setup_ms = model.compile_predictor_lstm()
            >>> print(f"Predictor compiled in {setup_ms:.1f}ms")
        """
        import time
        t0 = time.perf_counter()

        # Compile main BiLSTM layers in predictor
        self.predictor.lstm = mx.compile(self.predictor.lstm)
        self.predictor.shared = mx.compile(self.predictor.shared)

        # Compile BiLSTM layers in predictor.text_encoder
        self.predictor.text_encoder.lstms_0 = mx.compile(
            self.predictor.text_encoder.lstms_0,
        )
        self.predictor.text_encoder.lstms_2 = mx.compile(
            self.predictor.text_encoder.lstms_2,
        )
        self.predictor.text_encoder.lstms_4 = mx.compile(
            self.predictor.text_encoder.lstms_4,
        )

        return (time.perf_counter() - t0) * 1000

    def enable_prosody_embedding(
        self,
        num_types: int = NUM_PROSODY_TYPES,
        hidden_dim: int | None = None,
        init_scale: float = 0.01,
    ) -> None:
        """
        Enable prosody embedding layer for Phase B prosody control.

        Call this before load_prosody_weights() to initialize the embedding table.
        The embedding will be added to BERT output before the linear projection.

        Args:
            num_types: Number of prosody types (default 63 matching prosody_types.h)
            hidden_dim: Embedding dimension (default: plbert_hidden_size=768)
            init_scale: Initial scale factor (default 0.01 for minimal disruption)
        """
        if hidden_dim is None:
            hidden_dim = self.config.plbert_hidden_size
        self.prosody_embedding = ProsodyEmbedding(num_types, hidden_dim, init_scale)

    def load_prosody_weights(self, path: str | Path) -> None:
        """
        Load trained prosody embedding weights from a safetensors checkpoint.

        Args:
            path: Path to the safetensors file containing prosody weights

        Raises:
            ValueError: If prosody_embedding is not initialized
        """
        if self.prosody_embedding is None:
            raise ValueError(
                "Prosody embedding not initialized. "
                "Call enable_prosody_embedding() first.",
            )

        import mlx.core as mx

        # Load safetensors checkpoint
        weights = mx.load(str(path))

        # Map checkpoint keys to prosody_embedding keys
        # Checkpoint has: embedding.weight, scale
        prosody_weights = {}
        for key, value in weights.items():
            if key == "embedding.weight":
                prosody_weights["prosody_embedding.embedding.weight"] = value
            elif key == "scale":
                # Scale is stored as an array, assign directly
                self.prosody_embedding.scale = value
            else:
                # Handle nested keys from training checkpoint
                prosody_weights[f"prosody_embedding.{key}"] = value

        # Apply weights to model
        if prosody_weights:
            model_weights = dict(self.prosody_embedding.parameters())
            for key in prosody_weights.keys():
                # Extract the actual parameter name
                param_key = key.replace("prosody_embedding.", "")
                if param_key in model_weights:
                    # Update in place via tree_unflatten pattern
                    pass  # Will use update() below

            # Update embedding weights
            if "prosody_embedding.embedding.weight" in prosody_weights:
                self.prosody_embedding.embedding = nn.Embedding(
                    self.prosody_embedding.num_types,
                    self.prosody_embedding.hidden_dim,
                )
                # Set the weights
                emb_weight = prosody_weights["prosody_embedding.embedding.weight"]
                self.prosody_embedding.embedding.weight = emb_weight

    def enable_prosody_contour(
        self,
        prosody_dim: int = 768,
        hidden_dim: int = 256,
        contour_len: int = 50,
        num_prosody_types: int = NUM_PROSODY_TYPES,
    ) -> None:
        """
        Enable prosody contour predictor for Path C F0 control.

        This is an alternative to Phase B prosody embedding that provides
        more dynamic F0 control by predicting actual pitch contours rather
        than static multipliers.

        Call this before load_prosody_contour_weights() to initialize the predictor.

        Args:
            prosody_dim: Prosody embedding dimension (default 768)
            hidden_dim: Hidden layer dimension (default 256)
            contour_len: Number of contour points (default 50)
            num_prosody_types: Number of prosody types (default 63)
        """
        self.prosody_contour_predictor = ProsodyContourPredictor(
            prosody_dim=prosody_dim,
            hidden_dim=hidden_dim,
            contour_len=contour_len,
            num_prosody_types=num_prosody_types,
        )

    def load_prosody_contour_weights(
        self,
        contour_path: str | Path,
        embedding_path: str | Path,
    ) -> None:
        """
        Load trained prosody contour predictor weights.

        Loads both the contour model weights and the prosody embedding table
        that the contour model uses.

        Args:
            contour_path: Path to contour model weights (.npz file)
            embedding_path: Path to prosody embedding table (.safetensors file)

        Raises:
            ValueError: If prosody_contour_predictor is not initialized
        """
        if self.prosody_contour_predictor is None:
            raise ValueError(
                "Prosody contour predictor not initialized. "
                "Call enable_prosody_contour() first.",
            )

        # Load contour model weights
        contour_weights = mx.load(str(contour_path))

        # Map weights to model parameters
        # Training script saves: prosody_proj.weight, prosody_proj.bias, etc.
        param_mapping = {
            "prosody_proj.weight": "prosody_proj.weight",
            "prosody_proj.bias": "prosody_proj.bias",
            "contour_fc1.weight": "contour_fc1.weight",
            "contour_fc1.bias": "contour_fc1.bias",
            "contour_fc2.weight": "contour_fc2.weight",
            "contour_fc2.bias": "contour_fc2.bias",
            "contour_out.weight": "contour_out.weight",
            "contour_out.bias": "contour_out.bias",
            "stats_fc.weight": "stats_fc.weight",
            "stats_fc.bias": "stats_fc.bias",
        }

        # Apply weights
        for src_key, dst_attr in param_mapping.items():
            if src_key in contour_weights:
                parts = dst_attr.split(".")
                obj = self.prosody_contour_predictor
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], contour_weights[src_key])

        # Load prosody embedding table
        emb_weights = mx.load(str(embedding_path))

        # Find embedding weight in checkpoint
        emb_weight = None
        for key in emb_weights:
            if "embedding" in key.lower() and "weight" in key.lower():
                emb_weight = emb_weights[key]
                break
            if key == "embedding.weight":
                emb_weight = emb_weights[key]
                break

        if emb_weight is not None:
            # Recreate embedding with correct size and set weights
            num_types, dim = emb_weight.shape
            self.prosody_contour_predictor.embedding = nn.Embedding(num_types, dim)
            self.prosody_contour_predictor.embedding.weight = emb_weight
            self.prosody_contour_predictor.prosody_dim = dim
            self.prosody_contour_predictor.num_prosody_types = num_types

    def enable_prosody_contour_v2(
        self,
        prosody_dim: int = 768,
        hidden_dim: int = 512,
        contour_len: int = 50,
        num_prosody_types: int = NUM_PROSODY_TYPES,
        num_residual_blocks: int = 3,
    ) -> None:
        """
        Enable prosody contour predictor V2 (v2.4 architecture) for improved F0 control.

        This is the improved version that beats v5 on key emotions:
        - ANGRY: 100% of target (v5: 66%)
        - SAD: 100% of target (v5: 93%)
        - EXCITED: 100% of target (v5: 71%)
        - CALM: Close to target (1% error)
        - NEUTRAL: TIE

        Call this before load_prosody_contour_v2_weights() to initialize the predictor.

        Args:
            prosody_dim: Prosody embedding dimension (default 768)
            hidden_dim: Hidden layer dimension (default 512)
            contour_len: Number of contour points (default 50)
            num_prosody_types: Number of prosody types (default 63)
            num_residual_blocks: Number of residual blocks (default 3)
        """
        self.prosody_contour_predictor_v2 = ProsodyContourPredictorV2(
            prosody_dim=prosody_dim,
            hidden_dim=hidden_dim,
            contour_len=contour_len,
            num_prosody_types=num_prosody_types,
            num_residual_blocks=num_residual_blocks,
        )

    def load_prosody_contour_v2_weights(
        self,
        contour_path: str | Path,
        embedding_path: str | Path | None = None,
    ) -> None:
        """
        Load trained prosody contour predictor V2 weights.

        The v2 model uses an external prosody embedding table that must be
        loaded separately (orthogonal embeddings by default).

        Args:
            contour_path: Path to contour model weights (.npz file)
                         Default: models/prosody_contour_v2.4/best_model.npz
            embedding_path: Path to prosody embedding weights (.safetensors)
                           Default: models/prosody_embeddings_orthogonal/

        Raises:
            ValueError: If prosody_contour_predictor_v2 is not initialized
        """
        if self.prosody_contour_predictor_v2 is None:
            raise ValueError(
                "Prosody contour predictor V2 not initialized. "
                "Call enable_prosody_contour_v2() first.",
            )

        # Default embedding path
        if embedding_path is None:
            emb_dir = "models/prosody_embeddings_orthogonal"
            embedding_path = Path(f"{emb_dir}/final.safetensors")

        # Load contour model weights
        contour_weights = mx.load(str(contour_path))

        # V2 model saves all weights with full parameter paths
        # Map training checkpoint keys to model attributes
        model = self.prosody_contour_predictor_v2

        # Direct parameter mappings
        simple_params = [
            "prosody_proj.weight", "prosody_proj.bias",
            "proj_norm.weight", "proj_norm.bias",
            "arousal_embedding.weight",
            "high_arousal_fc.weight", "high_arousal_fc.bias",
            "low_arousal_fc.weight", "low_arousal_fc.bias",
            "multiplier_head.weight", "multiplier_head.bias",
            "contour_fc1.weight", "contour_fc1.bias",
            "contour_fc2.weight", "contour_fc2.bias",
            "stats_head.weight", "stats_head.bias",
            "embedding.weight",
        ]

        for param_name in simple_params:
            if param_name in contour_weights:
                parts = param_name.split(".")
                obj = model
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], contour_weights[param_name])

        # Handle shared_blocks (list of ResidualBlocks)
        for i in range(len(model.shared_blocks)):
            prefix = f"shared_blocks.{i}."
            for suffix in ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias",
                          "norm.weight", "norm.bias"]:
                key = prefix + suffix
                if key in contour_weights:
                    parts = suffix.split(".")
                    obj = model.shared_blocks[i]
                    for part in parts[:-1]:
                        obj = getattr(obj, part)
                    setattr(obj, parts[-1], contour_weights[key])

        # Handle high_arousal_block
        for suffix in ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias",
                      "norm.weight", "norm.bias"]:
            key = "high_arousal_block." + suffix
            if key in contour_weights:
                parts = suffix.split(".")
                obj = model.high_arousal_block
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], contour_weights[key])

        # Handle low_arousal_block
        for suffix in ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias",
                      "norm.weight", "norm.bias"]:
            key = "low_arousal_block." + suffix
            if key in contour_weights:
                parts = suffix.split(".")
                obj = model.low_arousal_block
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], contour_weights[key])

        # Load prosody embedding weights
        if Path(embedding_path).exists():
            emb_weights = mx.load(str(embedding_path))

            # Find embedding weight
            if "embedding.weight" in emb_weights:
                emb_weight = emb_weights["embedding.weight"]
                num_types, dim = emb_weight.shape

                # Recreate embedding with correct size
                model.embedding = nn.Embedding(num_types, dim)
                model.embedding.weight = emb_weight
                model.num_prosody_types = num_types

    def enable_prosody_duration_energy(
        self,
        prosody_dim: int = 768,
        hidden_dim: int = 256,
        num_prosody_types: int = NUM_PROSODY_TYPES,
    ) -> None:
        """
        Enable prosody duration/energy predictor.

        This predictor outputs multiplicative modifiers for duration and energy
        based on the prosody type (emotion). It uses the same architecture as
        ProsodyContourPredictorV2 with arousal-aware paths.

        Args:
            prosody_dim: Dimension of prosody embeddings (default 768)
            hidden_dim: Hidden dimension (default 256)
            num_prosody_types: Number of prosody types (default matches prosody_types.h)
        """
        self.prosody_duration_energy_predictor = ProsodyDurationEnergyPredictor(
            prosody_dim=prosody_dim,
            hidden_dim=hidden_dim,
            num_prosody_types=num_prosody_types,
        )

    def load_prosody_duration_energy_weights(
        self,
        weights_path: str | Path,
        embedding_path: str | Path | None = None,
    ) -> None:
        """
        Load trained prosody duration/energy predictor weights.

        Args:
            weights_path: Path to model weights (.npz or .safetensors)
            embedding_path: Path to prosody embedding weights (.safetensors)
                           Default: models/prosody_embeddings_orthogonal/

        Raises:
            ValueError: If predictor is not initialized
        """
        if self.prosody_duration_energy_predictor is None:
            raise ValueError(
                "Prosody duration/energy predictor not initialized. "
                "Call enable_prosody_duration_energy() first.",
            )

        # Default embedding path
        if embedding_path is None:
            emb_dir = "models/prosody_embeddings_orthogonal"
            embedding_path = Path(f"{emb_dir}/final.safetensors")

        # Load model weights
        weights = mx.load(str(weights_path))
        model = self.prosody_duration_energy_predictor

        # Direct parameter mappings
        param_names = [
            "prosody_proj.weight", "prosody_proj.bias",
            "proj_norm.weight", "proj_norm.bias",
            "arousal_embedding.weight",
            "neutral_arousal_fc.weight", "neutral_arousal_fc.bias",
            "high_arousal_fc.weight", "high_arousal_fc.bias",
            "low_arousal_fc.weight", "low_arousal_fc.bias",
            "duration_head.weight", "duration_head.bias",
            "energy_head.weight", "energy_head.bias",
            "embedding.weight",
        ]

        for param_name in param_names:
            if param_name in weights:
                parts = param_name.split(".")
                obj = model
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], weights[param_name])

        # Handle shared_block (ResidualBlock)
        block_params = ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias",
                       "norm.weight", "norm.bias"]
        for suffix in block_params:
            key = f"shared_block.{suffix}"
            if key in weights:
                parts = suffix.split(".")
                obj = model.shared_block
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], weights[key])

        # Load prosody embedding weights
        if Path(embedding_path).exists():
            emb_weights = mx.load(str(embedding_path))

            if "embedding.weight" in emb_weights:
                emb_weight = emb_weights["embedding.weight"]
                num_types, dim = emb_weight.shape

                # Recreate embedding with correct size
                model.embedding = nn.Embedding(num_types, dim)
                model.embedding.weight = emb_weight
                model.num_prosody_types = num_types

    # Frame bucket sizes for C++ compilation (R4)
    # Maps approximately to audio durations at 24kHz / 300 hop_length
    # hop_length = istft_upsample_rates (10*6) * istft_gen_istft_hop_size (5) = 300
    FRAME_BUCKETS = [200, 400, 600, 800, 1200]  # ~2.5s, 5s, 7.5s, 10s, 15s

    def quantize(self, bits: int = 8, group_size: int = 64) -> None:
        """
        A11 Optimization: Quantize model weights to reduce memory.

        Converts Linear and Embedding layers to quantized versions.
        8-bit quantization reduces memory by ~4x with minimal quality loss.

        Args:
            bits: Bits per weight (4 or 8). Default 8 for quality preservation.
            group_size: Quantization group size. Default 64.

        Note:
            - Modifies model in-place
            - Quantized model is inference-only (no gradients)
            - Call after loading weights, before inference
        """
        # Use MLX's built-in quantization
        nn.quantize(self, group_size=group_size, bits=bits)

    def quantize_selective(
        self,
        bits: int = 8,
        group_size: int = 64,
        exclude_patterns: list[str] | None = None,
    ) -> None:
        """
        A11 Optimization: Selective quantization with exclusions.

        Quantizes most layers but excludes sensitive components like
        the final projection layers that affect output quality.

        Args:
            bits: Bits per weight (4 or 8). Default 8.
            group_size: Quantization group size. Default 64.
            exclude_patterns: List of layer name patterns to exclude.
                            Default excludes: conv_post, F0_proj, N_proj

        Note:
            Call after loading weights, before inference.
        """
        if exclude_patterns is None:
            # Default exclusions: output projections that affect quality
            exclude_patterns = ["conv_post", "F0_proj", "N_proj", "dense"]

        def should_quantize(path: str, module: nn.Module) -> bool:
            """Predicate for selective quantization."""
            # Skip excluded patterns
            for pattern in exclude_patterns:
                if pattern in path:
                    return False
            # Quantize Linear and Embedding layers
            return hasattr(module, "to_quantized")

        nn.quantize(
            self,
            group_size=group_size,
            bits=bits,
            class_predicate=should_quantize,
        )

    def set_local_attention(self, window_size: int | None) -> None:
        """
        P1 Optimization: Enable local (sliding window) attention.

        Reduces attention complexity from O(n²) to O(n*w) where w is window size.
        Each position can only attend to positions within window_size distance.

        Args:
            window_size: Size of attention window. Position i attends to
                        [i-window_size, i+window_size]. Set to None to disable.

        Note:
            - Reasonable values: 32-128 for TTS (phoneme sequences are short)
            - Too small may lose long-range prosody dependencies
            - Call before inference.
        """
        # ALBERT uses shared weights - single attention layer
        self.bert.encoder.albert_layer.attention.set_local_attention(window_size)

    def generate_batch_no_padding(
        self,
        input_ids_list: list[mx.array],
        voice: mx.array,
        speed: float = 1.0,
        **kwargs,
    ) -> list[mx.array]:
        """
        G2 Optimization: Process variable-length inputs without padding.

        Instead of padding all inputs to max length (wasting compute),
        processes each input sequentially at its actual length.

        Useful for batch processing of variable-length utterances where:
        - Individual utterances are short (< 100 tokens)
        - Padding overhead would be significant
        - Real-time is not required (sequential processing)

        Args:
            input_ids_list: List of [1, length] tensors (variable lengths)
            voice: [1, 256] voice embedding (shared across all)
            speed: Speaking rate
            **kwargs: Additional args passed to __call__

        Returns:
            List of audio tensors [1, samples] (variable lengths)

        Example:
            >>> inputs = [mx.array([[1,2,3]]), mx.array([[4,5]]), mx.array([[6,7,8,9]])]
            >>> audios = model.generate_batch_no_padding(inputs, voice)
            >>> # Each audio is generated at its actual input length
        """
        results = []
        for input_ids in input_ids_list:
            # Process each input at its actual length - no padding
            audio = self(input_ids, voice, speed=speed, **kwargs)
            results.append(audio)
        return results

    def generate_batch_sorted(
        self,
        input_ids_list: list[mx.array],
        voice: mx.array,
        speed: float = 1.0,
        **kwargs,
    ) -> list[mx.array]:
        """
        C4 Optimization: Process inputs sorted by length.

        Sorts inputs by length before processing to maximize GPU efficiency.
        Processing similar-length sequences together improves memory locality
        and reduces padding overhead in any subsequent batched operations.

        Args:
            input_ids_list: List of [1, length] tensors (variable lengths)
            voice: [1, 256] voice embedding (shared across all)
            speed: Speaking rate
            **kwargs: Additional args passed to __call__

        Returns:
            List of audio tensors [1, samples] in ORIGINAL input order
        """
        # Get lengths and sort indices
        lengths = [ids.shape[-1] for ids in input_ids_list]
        sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i])

        # Process in sorted order (shortest first for memory efficiency)
        sorted_results = []
        for idx in sorted_indices:
            audio = self(input_ids_list[idx], voice, speed=speed, **kwargs)
            sorted_results.append((idx, audio))

        # Restore original order
        sorted_results.sort(key=lambda x: x[0])
        return [audio for _, audio in sorted_results]

    def find_sentence_boundaries(
        self,
        input_ids: mx.array,
        boundary_tokens: list[int] | None = None,
    ) -> list[int]:
        """
        E5 Optimization: Find sentence/phrase boundaries in token sequence.

        Identifies positions where the text can be cleanly split (at pauses,
        punctuation tokens, etc.) for better chunking quality.

        Args:
            input_ids: [1, length] phoneme token ids
            boundary_tokens: List of token IDs that mark boundaries.
                           Default: [0] (silence/padding token)

        Returns:
            List of boundary positions (indices where splits are preferred)
        """
        if boundary_tokens is None:
            boundary_tokens = [0]  # Silence/padding as default boundary

        ids_np = np.array(input_ids[0])
        boundaries = []

        for i, token_id in enumerate(ids_np):
            if int(token_id) in boundary_tokens:
                boundaries.append(i)

        return boundaries

    def generate_chunked_sentences(
        self,
        input_ids: mx.array,
        voice: mx.array,
        max_chunk_size: int = 100,
        min_chunk_size: int = 20,
        boundary_tokens: list[int] | None = None,
        speed: float = 1.0,
        **kwargs,
    ) -> mx.array:
        """
        E5 Optimization: Chunk at sentence/phrase boundaries.

        Unlike generate_chunked which splits at fixed intervals,
        this method prefers splitting at natural boundaries (silences,
        punctuation) for smoother audio output.

        Args:
            input_ids: [1, length] phoneme token ids
            voice: [1, 256] voice embedding
            max_chunk_size: Maximum tokens per chunk (hard limit)
            min_chunk_size: Minimum tokens per chunk (to avoid tiny chunks)
            boundary_tokens: Token IDs that mark preferred split points
            speed: Speaking rate
            **kwargs: Additional args passed to __call__

        Returns:
            audio: [1, total_samples] concatenated audio

        Note:
            Splitting at sentence boundaries:
            - Produces more natural-sounding chunk boundaries
            - Avoids cutting words/phrases mid-way
            - May result in variable chunk sizes
        """
        seq_length = input_ids.shape[-1]

        # If short enough, process directly
        if seq_length <= max_chunk_size:
            return self(input_ids, voice, speed=speed, **kwargs)

        # Find sentence boundaries
        boundaries = self.find_sentence_boundaries(input_ids, boundary_tokens)

        # Build chunks using boundaries
        chunks = []
        start = 0

        while start < seq_length:
            # Find the best boundary within [start + min_chunk_size, start + max_chunk_size]
            min_end = start + min_chunk_size
            max_end = min(start + max_chunk_size, seq_length)

            # Find boundaries in range
            valid_boundaries = [b for b in boundaries if min_end <= b <= max_end]

            if valid_boundaries:
                # Use the last valid boundary for maximum chunk size within limits
                end = valid_boundaries[-1] + 1  # Include the boundary token
            elif max_end >= seq_length:
                # Last chunk - take everything remaining
                end = seq_length
            else:
                # No boundary found - fall back to max_chunk_size
                end = max_end

            chunk = input_ids[:, start:end]
            chunks.append(chunk)

            start = end

        # Process each chunk
        kwargs.pop('validate_output', None)
        audio_chunks = []
        for chunk_ids in chunks:
            audio = self(chunk_ids, voice, speed=speed, validate_output=False, **kwargs)
            audio_chunks.append(audio)

        # Concatenate all parts (no overlap/crossfade since we split at boundaries)
        return mx.concatenate(audio_chunks, axis=-1)

    def generate_chunked(
        self,
        input_ids: mx.array,
        voice: mx.array,
        chunk_size: int = 100,
        overlap: int = 10,
        speed: float = 1.0,
        **kwargs,
    ) -> mx.array:
        """
        E4 Optimization: Process long sequences in chunks with crossfade.

        Splits long phoneme sequences into overlapping chunks, processes
        each independently, and crossfades the overlap regions for smooth
        concatenation. Handles sequences that exceed model memory limits.

        Args:
            input_ids: [1, length] phoneme token ids (can be very long)
            voice: [1, 256] voice embedding
            chunk_size: Maximum tokens per chunk (default 100)
            overlap: Overlap tokens between chunks for crossfade (default 10)
            speed: Speaking rate
            **kwargs: Additional args passed to __call__

        Returns:
            audio: [1, total_samples] concatenated audio

        Note:
            - Chunks are processed at boundaries (word/sentence if possible)
            - Overlap enables smooth crossfade to avoid discontinuities
            - For very long texts, memory usage is bounded by chunk_size
        """
        seq_length = input_ids.shape[-1]

        # If short enough, process directly
        if seq_length <= chunk_size:
            return self(input_ids, voice, speed=speed, **kwargs)

        # Split into overlapping chunks
        chunks = []
        start = 0
        while start < seq_length:
            end = min(start + chunk_size, seq_length)
            chunk = input_ids[:, start:end]
            chunks.append((start, chunk))
            if end >= seq_length:
                break
            start = end - overlap  # Overlap with previous

        # Process each chunk
        # Remove validate_output from kwargs if present to avoid duplicate
        kwargs.pop('validate_output', None)
        audio_chunks = []
        for chunk_start, chunk_ids in chunks:
            audio = self(chunk_ids, voice, speed=speed, validate_output=False, **kwargs)
            audio_chunks.append((chunk_start, audio))

        # Crossfade overlapping regions for smooth joins
        # Estimate samples per overlap token (6 frames/phoneme * 200 samples/frame)
        overlap_samples = overlap * 6 * 200

        result_parts = []
        for i, (_chunk_start, audio) in enumerate(audio_chunks):
            if i == 0:
                # First chunk: keep everything
                result_parts.append(audio)
            else:
                # Get tail of previous result for crossfade
                prev_audio = result_parts[-1]
                prev_len = prev_audio.shape[-1]
                curr_len = audio.shape[-1]

                # Calculate actual overlap samples (capped to chunk lengths)
                actual_overlap = min(overlap_samples, prev_len, curr_len)

                if actual_overlap > 10:  # Minimum samples for meaningful crossfade
                    # Extract overlap regions
                    prev_tail = prev_audio[:, -actual_overlap:]
                    curr_head = audio[:, :actual_overlap]

                    # Create half-Hanning fade curve for smooth transition
                    # t goes from 0 to 1, fade_out = 1-t, fade_in = t
                    t = mx.linspace(0.0, 1.0, actual_overlap)
                    # Smooth cosine curve: 0.5 - 0.5*cos(π*t) for fade-in
                    fade_in = 0.5 - 0.5 * mx.cos(mx.array(3.14159265) * t)
                    fade_out = 1.0 - fade_in

                    # Apply crossfade: blend prev_tail and curr_head
                    crossfaded = prev_tail * fade_out + curr_head * fade_in

                    # Trim previous chunk's overlap tail, append crossfade, then rest of current
                    result_parts[-1] = prev_audio[:, :-actual_overlap]
                    result_parts.append(crossfaded)
                    result_parts.append(audio[:, actual_overlap:])
                else:
                    # Not enough overlap for crossfade, just trim
                    if curr_len > overlap_samples:
                        result_parts.append(audio[:, overlap_samples:])
                    else:
                        result_parts.append(audio)

        # Concatenate all parts
        return mx.concatenate(result_parts, axis=-1)

    def generate_chunked_parallel(
        self,
        input_ids: mx.array,
        voice: mx.array,
        chunk_size: int = 100,
        overlap: int = 10,
        speed: float = 1.0,
        batch_chunks: int = 4,
        **kwargs,
    ) -> mx.array:
        """
        C3 Optimization: Process chunks with batched lazy evaluation.

        Unlike sequential generate_chunked, this method builds computation
        graphs for multiple chunks before evaluating them together. MLX's
        lazy evaluation enables parallel GPU execution.

        Args:
            input_ids: [1, length] phoneme token ids
            voice: [1, 256] voice embedding
            chunk_size: Maximum tokens per chunk (default 100)
            overlap: Overlap tokens between chunks (default 10)
            speed: Speaking rate
            batch_chunks: Number of chunks to process in each batch (default 4)
            **kwargs: Additional args passed to __call__

        Returns:
            audio: [1, total_samples] concatenated audio

        Note:
            - Uses MLX lazy evaluation for implicit parallelism
            - Builds computation graphs for batch_chunks at a time
            - Single mx.eval() allows GPU to optimize execution
        """
        seq_length = input_ids.shape[-1]

        # If short enough, process directly
        if seq_length <= chunk_size:
            return self(input_ids, voice, speed=speed, **kwargs)

        # Split into overlapping chunks
        chunks = []
        start = 0
        while start < seq_length:
            end = min(start + chunk_size, seq_length)
            chunk = input_ids[:, start:end]
            chunks.append((start, chunk))
            if end >= seq_length:
                break
            start = end - overlap

        # Remove validate_output from kwargs if present
        kwargs.pop('validate_output', None)

        # Process chunks in batches for better GPU utilization
        all_audio_chunks = []
        for batch_start in range(0, len(chunks), batch_chunks):
            batch_end = min(batch_start + batch_chunks, len(chunks))
            batch = chunks[batch_start:batch_end]

            # Build computation graphs for all chunks in batch (lazy)
            batch_audio = []
            for _chunk_start, chunk_ids in batch:
                audio = self(chunk_ids, voice, speed=speed, validate_output=False, **kwargs)
                batch_audio.append(audio)

            # Single eval for the batch - MLX can parallelize GPU work
            mx.eval(*batch_audio)
            all_audio_chunks.extend(batch_audio)

        # Crossfade overlapping regions for smooth joins
        # Estimate samples per overlap token (6 frames/phoneme * 200 samples/frame)
        overlap_samples = overlap * 6 * 200

        result_parts = []
        for i, audio in enumerate(all_audio_chunks):
            if i == 0:
                # First chunk: keep everything
                result_parts.append(audio)
            else:
                # Get tail of previous result for crossfade
                prev_audio = result_parts[-1]
                prev_len = prev_audio.shape[-1]
                curr_len = audio.shape[-1]

                # Calculate actual overlap samples (capped to chunk lengths)
                actual_overlap = min(overlap_samples, prev_len, curr_len)

                if actual_overlap > 10:  # Minimum samples for meaningful crossfade
                    # Extract overlap regions
                    prev_tail = prev_audio[:, -actual_overlap:]
                    curr_head = audio[:, :actual_overlap]

                    # Create half-Hanning fade curve for smooth transition
                    t = mx.linspace(0.0, 1.0, actual_overlap)
                    fade_in = 0.5 - 0.5 * mx.cos(mx.array(3.14159265) * t)
                    fade_out = 1.0 - fade_in

                    # Apply crossfade: blend prev_tail and curr_head
                    crossfaded = prev_tail * fade_out + curr_head * fade_in

                    # Trim previous chunk's overlap tail, append crossfade, then rest of current
                    result_parts[-1] = prev_audio[:, :-actual_overlap]
                    result_parts.append(crossfaded)
                    result_parts.append(audio[:, actual_overlap:])
                else:
                    # Not enough overlap for crossfade, just trim
                    if curr_len > overlap_samples:
                        result_parts.append(audio[:, overlap_samples:])
                    else:
                        result_parts.append(audio)

        return mx.concatenate(result_parts, axis=-1)

    @property
    def sample_rate(self) -> int:
        """Audio sample rate in Hz (24kHz). Consistent API with CosyVoice2."""
        return self.config.sample_rate

    def _select_frame_bucket(self, estimated_frames: int) -> int:
        """Select the smallest frame bucket that fits the estimated frames."""
        for bucket in self.FRAME_BUCKETS:
            if estimated_frames <= bucket:
                return bucket
        return self.FRAME_BUCKETS[-1]  # Largest bucket

    def _compute_alignment(
        self,
        duration_logits: mx.array,
        speed: float = 1.0,
        max_frames_bucket: int | None = None,
    ) -> tuple[mx.array, int, mx.array | None]:
        """
        Convert duration logits to frame-level indices for feature expansion.

        Per official Kokoro: duration = sigmoid(logits).sum(-1) / speed

        Args:
            duration_logits: [batch, text_length, max_dur] from predictor
            speed: Speaking rate multiplier (default 1.0)
            max_frames_bucket: Optional fixed frame count for C++ compilation.
                              If provided, avoids device sync and returns a mask.

        Returns:
            indices: [batch, total_frames] - index into text positions
            total_frames: int - total audio frames (or bucket size if bucketed)
            frame_mask: Optional[mx.array] - [batch, total_frames] valid frame mask
                        (only returned when max_frames_bucket is provided)
        """
        # Sum across duration bins and apply sigmoid
        # This matches Kokoro: sigmoid(duration_proj(x)).sum(-1) / speed
        duration = mx.sigmoid(duration_logits).sum(axis=-1)  # [batch, T_text]

        # Scale by speed and round to integers
        duration = duration / speed
        pred_dur = mx.maximum(mx.round(duration), 1.0).astype(
            mx.int32,
        )  # [batch, T_text]

        # Vectorized repeat_interleave via cumsum trick
        # For each position t, we want to repeat t by pred_dur[t] times
        batch_size, text_len = pred_dur.shape

        # Compute total frames per batch
        total_frames_per_batch = mx.sum(pred_dur, axis=1)  # [batch]

        if max_frames_bucket is not None:
            # C++ compilation mode: use fixed bucket size, no device sync
            max_frames = max_frames_bucket
            # Create frame mask for valid positions (no sync needed)
            frame_pos_for_mask = mx.arange(max_frames)[None, :]  # [1, max_frames]
            frame_mask = (
                frame_pos_for_mask < total_frames_per_batch[:, None]
            )  # [batch, max_frames]
            # Issue 5: Check for bucket overflow
            # Note: This check happens lazily when the mask is evaluated
            # Overflow means audio will be truncated - valid behavior but worth knowing
            overflow_mask = total_frames_per_batch > max_frames  # [batch]
            # Store overflow info for caller to check if needed
            self._last_bucket_overflow = overflow_mask
        else:
            # Dynamic mode: requires device sync to get exact size
            max_frames = int(mx.max(total_frames_per_batch).item())
            frame_mask = None

        # Build indices using cumsum approach
        # cumsum gives the end position of each duration segment
        cumsum = mx.cumsum(pred_dur, axis=1)  # [batch, text_len]

        # Create frame position array [0, 1, 2, ..., max_frames-1]
        frame_pos = mx.arange(max_frames)[None, :]  # [1, max_frames]

        # For each frame position, find which text position it belongs to
        # frame_pos < cumsum[t] means frame belongs to position <= t
        # We use searchsorted equivalent: find first t where cumsum[t] > frame_pos
        # This is: sum(frame_pos >= cumsum, axis=-1) for each position

        # Broadcast: cumsum [batch, text_len, 1], frame_pos [1, 1, max_frames]
        # Result: comparison [batch, text_len, max_frames]
        cumsum_expanded = cumsum[:, :, None]  # [batch, text_len, 1]
        frame_pos_expanded = frame_pos[None, :, :]  # [1, 1, max_frames]

        # For each frame, count how many cumsum values are <= frame_pos
        # indices[b, f] = number of text positions where cumsum <= f
        comparison = (
            cumsum_expanded <= frame_pos_expanded
        )  # [batch, text_len, max_frames]
        indices = mx.sum(comparison.astype(mx.int32), axis=1)  # [batch, max_frames]

        # Clamp to valid text range (in case of rounding issues)
        indices = mx.clip(indices, 0, text_len - 1)

        return indices, max_frames, frame_mask

    def _expand_features(
        self,
        features: mx.array,
        indices: mx.array,
        total_frames: int,
    ) -> mx.array:
        """
        Expand text-rate features to audio frame rate using alignment indices.

        Uses native MLX take_along_axis for GPU-accelerated gathering.

        Args:
            features: [batch, text_length, hidden_dim]
            indices: [batch, total_frames] - indices into text positions
            total_frames: int

        Returns:
            expanded: [batch, total_frames, hidden_dim]
        """
        # Use take_along_axis for vectorized gathering
        # indices: [batch, total_frames] -> [batch, total_frames, 1]
        # We need to broadcast indices across hidden_dim
        indices_expanded = indices[:, :, None]  # [batch, total_frames, 1]

        # Expand indices to match hidden_dim
        batch_size, _, hidden_dim = features.shape
        indices_broadcast = mx.broadcast_to(
            indices_expanded, (batch_size, total_frames, hidden_dim),
        )

        # Gather features along axis 1 (text_length axis)
        return mx.take_along_axis(features, indices_broadcast, axis=1)


    def __call__(
        self,
        input_ids: mx.array,
        voice: mx.array,
        attention_mask: mx.array | None = None,
        speed: float = 1.0,
        validate_output: bool = True,
        prosody_mask: mx.array | None = None,
        style_cache: StyleCache | None = None,
        progressive_precision: bool = False,
        f0_keyframe_factor: int = 1,
    ) -> mx.array:
        """
        Generate audio from text.

        Args:
            input_ids: [batch, length] - phoneme token ids
            voice: [batch, 256] - voice embedding (or [batch, 128] style only)
            attention_mask: [batch, length] - padding mask (True=pad, False=valid)
            speed: Speaking rate (1.0=normal, <1=slower, >1=faster)
            validate_output: Validate output for NaN/Inf and clip to [-1, 1].
                           Set False for high-throughput to skip mx.eval() sync.
            prosody_mask: [batch, length] - optional prosody type IDs.
                         Each value is a prosody type from prosody_types.h.
                         Requires enable_prosody_embedding() first.
            style_cache: Optional StyleCache for N2 optimization.
                        If provided, uses precomputed AdaIN fc outputs.
                        Create with precompute_style_cache(model, voice).
            progressive_precision: If True, use BF16 for early decoder layers (N7).
                                  Reduces memory bandwidth ~15% with minimal quality loss.
            f0_keyframe_factor: F0 prediction subsample factor (N6 optimization).
                               1 = full resolution (default), 2 = half, 4 = quarter.
                               Higher values reduce F0 predictor compute but may
                               lose fast pitch transitions. F0 bandwidth < 50Hz,
                               so 2-4x subsampling is typically perceptually lossless.

        Returns:
            audio: [batch, samples] - generated waveform

        Raises:
            ValueError: If input_ids or voice have invalid shapes/values
            RuntimeError: If NaN/Inf detected (when validate_output=True)
        """
        # ====================================================================
        # Input Validation (Safety)
        # ====================================================================
        # Validate input_ids shape
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must be 2D [batch, length], got shape {input_ids.shape}",
            )
        batch_size, seq_length = input_ids.shape

        # Validate sequence length limits
        max_seq_length = 10000  # ~10 minutes of audio at typical rates
        if seq_length == 0:
            raise ValueError("Empty input_ids sequence")
        if seq_length > max_seq_length:
            raise ValueError(
                f"input_ids sequence too long: {seq_length} tokens "
                f"(max {max_seq_length})",
            )

        # Validate voice embedding shape
        if voice.ndim != 2:
            raise ValueError(
                f"voice must be 2D [batch, dim], got shape {voice.shape}",
            )
        if voice.shape[0] != batch_size:
            raise ValueError(
                f"voice batch size {voice.shape[0]} != "
                f"input_ids batch size {batch_size}",
            )
        if voice.shape[-1] not in (128, 256):
            raise ValueError(
                f"voice embedding dimension must be 128 or 256, got {voice.shape[-1]}",
            )

        # Validate attention_mask if provided
        if attention_mask is not None:
            if attention_mask.shape != input_ids.shape:
                raise ValueError(
                    f"attention_mask shape {attention_mask.shape} "
                    f"must match input_ids shape {input_ids.shape}",
                )

        # Validate speed parameter
        if speed <= 0 or speed > 10:
            raise ValueError(f"speed must be in (0, 10], got {speed}")

        # Split voice embedding into style and speaker parts
        # PyTorch reference:
        #   style128 = ref_s[:, :128]  -> used for decoder
        #   s = ref_s[:, 128:]         -> used for predictor
        if voice.shape[-1] == 256:
            style = voice[:, :128]  # First 128 dims for decoder
            speaker = voice[:, 128:]  # Second 128 dims for predictor
        else:
            # Backward compatibility: if only 128 dims, use for both
            style = voice
            speaker = voice

        # ====================================================================
        # Inference flow matching PyTorch reference (export_kokoro_reference.py)
        # ====================================================================

        # Step 1: BERT encoding (d_en in PyTorch)
        # BERT expects attention_mask with 1=valid, 0=pad (integer/float)
        # Our attention_mask is boolean with True=pad, so convert: ~mask gives 1=valid
        bert_attention_mask = None
        if attention_mask is not None:
            bert_attention_mask = (~attention_mask).astype(mx.float32)
        bert_out = self.bert(input_ids, bert_attention_mask)

        # Phase B: Inject prosody embeddings if enabled and prosody_mask provided
        if prosody_mask is not None and self.prosody_embedding is not None:
            # Validate prosody_mask shape
            if prosody_mask.shape != input_ids.shape:
                raise ValueError(
                    f"prosody_mask shape {prosody_mask.shape} must match "
                    f"input_ids shape {input_ids.shape}",
                )
            # Add prosody embedding to BERT output
            # bert_out: [batch, seq_len, plbert_hidden_size=768]
            # prosody_emb: [batch, seq_len, hidden_dim=768]
            prosody_emb = self.prosody_embedding(prosody_mask)
            bert_out = bert_out + prosody_emb

        # Save bert_out for debugging (BERT output before linear projection)
        if getattr(self, "save_debug_tensors", False):
            import numpy as np
            mx.eval(bert_out)
            np.save("/tmp/py_bert_out.npy", np.array(bert_out))

        bert_enc = self.bert_encoder(bert_out)  # [batch, T_text, hidden]

        # Step 2: Duration features via predictor.text_encoder (CRITICAL!)
        # PyTorch: duration_feats = model.predictor.text_encoder(d_en, s, ...)
        # This applies style-conditioned LSTM layers to bert_enc and returns 640-dim

        # Save bert_enc for comparison
        if getattr(self, "save_debug_tensors", False):
            import numpy as np
            mx.eval(bert_enc)
            np.save("/tmp/py_bert_enc.npy", np.array(bert_enc))

        duration_feats = self.predictor.text_encoder(
            bert_enc, speaker,
        )  # [batch, T_text, 640]
        batch, length, _ = duration_feats.shape

        # Step 3: Duration prediction via predictor.lstm + duration_proj
        # PyTorch: x_lstm = model.predictor.lstm(duration_feats), then duration_proj
        dur_enc = self.predictor.lstm(duration_feats)  # [batch, T_text, 512]
        duration_logits = self.predictor.duration_proj(
            dur_enc,
        )  # [batch, T_text, max_dur]

        # Step 4: Compute alignment indices from durations
        # Apply duration modifier from prosody if enabled
        effective_speed = speed
        pred = self.prosody_duration_energy_predictor
        if prosody_mask is not None and pred is not None:
            # Get dominant prosody type (first non-zero value)
            # MLX doesn't support boolean indexing, so convert to numpy
            import numpy as np
            mx.eval(prosody_mask)
            prosody_np = np.array(prosody_mask).flatten()  # Flatten to 1D
            non_neutral_indices = np.where(prosody_np != 0)[0]
            if len(non_neutral_indices) > 0:
                dominant_type = int(prosody_np[non_neutral_indices[0]])
            else:
                dominant_type = 0

            # Get duration modifier (lower = faster, higher = slower)
            # Convert to speed adjustment (speed = 1.0 means normal rate)
            # duration_mult < 1.0 means faster speech, so divide speed
            dur_mod, energy_mod = pred(mx.array([dominant_type]))
            mx.eval(dur_mod)
            dur_mult = float(dur_mod.squeeze())
            # Invert: duration_mult 0.9 (faster) -> speed 1.11 (faster)
            # duration_mult 1.15 (slower) -> speed 0.87 (slower)
            effective_speed = speed / dur_mult

            # Store energy modifier for later use in audio generation
            self._prosody_energy_modifier = float(energy_mod.squeeze())
        else:
            self._prosody_energy_modifier = 1.0

        indices, total_frames, _ = self._compute_alignment(
            duration_logits, effective_speed,
        )
        # indices: [batch, T_audio] - maps audio frames to text positions

        # Step 5: Expand 640-dim duration features for F0/N prediction (en in PyTorch)
        # PyTorch: en = duration_feats^T @ pred_aln_trg  (duration_feats is 640-dim)

        # Save duration_feats BEFORE expand (to trace error source)
        if getattr(self, "save_debug_tensors", False):
            import numpy as np
            mx.eval(duration_feats)
            np.save("/tmp/py_duration_feats.npy", np.array(duration_feats))
            # Also save indices for comparison
            mx.eval(indices)
            np.save("/tmp/py_indices.npy", np.array(indices))

        en_expanded_640 = self._expand_features(duration_feats, indices, total_frames)
        # en_expanded_640: [batch, T_audio, 640]

        # Step 5.5: Run through shared BiLSTM for F0/N prediction
        # PyTorch F0Ntrain: x = self.shared(en)  where en is 640-dim

        # Save BiLSTM input if debug mode enabled
        if getattr(self, "save_debug_tensors", False):
            import numpy as np
            mx.eval(en_expanded_640)
            np.save("/tmp/py_en_expanded_640.npy", np.array(en_expanded_640))

        x_shared = self.predictor.shared(en_expanded_640)  # [batch, T_audio, 512]

        # Save BiLSTM output if debug mode enabled
        if getattr(self, "save_debug_tensors", False):
            import numpy as np
            mx.eval(x_shared)
            np.save("/tmp/py_x_shared.npy", np.array(x_shared))

        # Step 6: F0/N prediction on processed expanded features
        # PyTorch: F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

        # N6: F0 keyframe interpolation - predict at reduced rate, interpolate
        if f0_keyframe_factor > 1:
            # Subsample x_shared to keyframes
            # x_shared: [batch, T_audio, 512]
            x = x_shared[:, ::f0_keyframe_factor, :]  # [batch, T_audio/factor, 512]
            target_f0_len = x_shared.shape[1] * 2  # Original F0 output length

            # Run F0 blocks on subsampled input
            for block_name in ["F0_0", "F0_1", "F0_2"]:
                block = getattr(self.predictor, block_name)
                norm1 = style_cache.get(f"predictor.{block_name}.norm1") if style_cache else None
                norm2 = style_cache.get(f"predictor.{block_name}.norm2") if style_cache else None
                x = block(x, speaker, cached_norm1=norm1, cached_norm2=norm2)

            f0_keyframes = self.predictor.F0_proj(x).squeeze(-1)  # [batch, ~T_audio*2/factor]

            # Linear interpolation to full resolution
            # f0_keyframes: [batch, L_key], target: [batch, target_f0_len]
            batch_size, key_len = f0_keyframes.shape

            # Create interpolation indices
            # Map each target position to fractional keyframe position
            target_positions = mx.arange(target_f0_len).astype(mx.float32)
            scale = (key_len - 1) / max(target_f0_len - 1, 1)
            key_positions = target_positions * scale  # [target_f0_len]

            # Integer indices and interpolation weights
            idx_low = mx.floor(key_positions).astype(mx.int32)
            idx_low = mx.clip(idx_low, 0, key_len - 2)
            idx_high = idx_low + 1
            weights = key_positions - idx_low.astype(mx.float32)  # [target_f0_len]

            # Gather and interpolate
            # f0_keyframes: [batch, key_len]
            f0_low = mx.take(f0_keyframes, idx_low, axis=1)   # [batch, target_f0_len]
            f0_high = mx.take(f0_keyframes, idx_high, axis=1)  # [batch, target_f0_len]
            f0 = f0_low + weights[None, :] * (f0_high - f0_low)  # Linear interp
        else:
            # Standard full-resolution F0 prediction
            x = x_shared

            # N2: Get cached styles for F0 predictor blocks
            for block_name in ["F0_0", "F0_1", "F0_2"]:
                block = getattr(self.predictor, block_name)
                norm1 = style_cache.get(f"predictor.{block_name}.norm1") if style_cache else None
                norm2 = style_cache.get(f"predictor.{block_name}.norm2") if style_cache else None
                x = block(x, speaker, cached_norm1=norm1, cached_norm2=norm2)

            f0 = self.predictor.F0_proj(x).squeeze(-1)  # [batch, T_audio * 2]

        # Path C: Apply prosody contour F0 modifications if enabled
        # Prefer v2 (better results) over v1 if both are enabled
        v2_pred = self.prosody_contour_predictor_v2
        contour_predictor = v2_pred or self.prosody_contour_predictor

        if prosody_mask is not None and contour_predictor is not None:
            # Validate prosody_mask shape
            if prosody_mask.shape != input_ids.shape:
                raise ValueError(
                    f"prosody_mask shape {prosody_mask.shape} must match "
                    f"input_ids shape {input_ids.shape}",
                )
            # Compute F0 modifiers from contour predictor
            f0_modifiers = contour_predictor.predict_f0_modifiers(
                prosody_mask,
                f0_length=f0.shape[-1],
            )
            # Apply multiplicative F0 modification (only to voiced frames)
            # Voiced frames have positive F0, unvoiced have negative/zero
            voiced_mask = (f0 > 0).astype(f0.dtype)
            f0 = f0 * (voiced_mask * f0_modifiers + (1 - voiced_mask))

        # Save debug tensors if requested
        if getattr(self, "save_debug_tensors", False):
            import numpy as np

            mx.eval(f0)
            np.save("/tmp/py_f0.npy", np.array(f0))

        # Noise prediction uses the same shared LSTM output
        x = x_shared

        # N2: Get cached styles for N predictor blocks
        for block_name in ["N_0", "N_1", "N_2"]:
            block = getattr(self.predictor, block_name)
            norm1 = style_cache.get(f"predictor.{block_name}.norm1") if style_cache else None
            norm2 = style_cache.get(f"predictor.{block_name}.norm2") if style_cache else None
            x = block(x, speaker, cached_norm1=norm1, cached_norm2=norm2)

        noise = self.predictor.N_proj(x).squeeze(-1)  # [batch, T_audio * 2]

        # Step 7: ASR features via text_encoder
        # PyTorch: t_en = model.text_encoder(input_ids, ...), asr = t_en @ pred_aln_trg
        text_enc = self.text_encoder(
            input_ids, attention_mask,
        )  # [batch, T_text, hidden]
        asr_expanded = self._expand_features(text_enc, indices, total_frames)
        # asr_expanded: [batch, T_audio, hidden]

        # Step 8: Decode with style (first 128 dims of voice)
        # PyTorch: model.decoder(asr, F0_pred, N_pred, style128)
        # Use compiled decoder if available and enabled for faster inference
        if self._use_compiled_decoder and self._compiled_decoder is not None:
            # Compiled decoder doesn't support style_cache or progressive_precision
            # but provides ~1.2x speedup from graph optimization
            audio = self._compiled_decoder(asr_expanded, f0, noise, style)
        else:
            audio = self.decoder(
                asr_expanded, f0, noise, style,
                style_cache=style_cache,
                progressive_precision=progressive_precision,
            )

        # Apply energy modifier from prosody if enabled
        has_mod = hasattr(self, '_prosody_energy_modifier')
        if has_mod and self._prosody_energy_modifier != 1.0:
            audio = audio * self._prosody_energy_modifier

        # ====================================================================
        # Output Validation (Safety) - Optional for high-throughput scenarios
        # ====================================================================
        if validate_output:
            # Check for numerical instability (NaN/Inf)
            mx.eval(audio)  # Force evaluation before checking
            if mx.any(mx.isnan(audio)):
                raise RuntimeError(
                    "Numerical instability: NaN detected in output audio. "
                    "This may indicate invalid input or model state.",
                )
            if mx.any(mx.isinf(audio)):
                raise RuntimeError(
                    "Numerical instability: Inf detected in output audio. "
                    "This may indicate invalid input or model state.",
                )

            # Clamp audio to valid range [-1, 1] to prevent clipping artifacts
            audio = mx.clip(audio, -1.0, 1.0)

            # Sanity check on output duration (max ~5 minutes at 24kHz = 7.2M samples)
            max_samples = 5 * 60 * 24000  # 5 minutes
            if audio.shape[-1] > max_samples:
                raise RuntimeError(
                    f"Output audio too long: {audio.shape[-1]} samples "
                    f"(max {max_samples}, ~5 minutes). Input may be too long.",
                )

        return audio

    def synthesize_bucketed(
        self,
        input_ids: mx.array,
        voice: mx.array,
        attention_mask: mx.array | None = None,
        speed: float = 1.0,
        frame_bucket: int | None = None,
        prosody_mask: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """
        Generate audio with shape-bucketed inference for C++ compilation.

        This method avoids device sync by using fixed frame bucket sizes.
        The actual audio length is returned via a mask. Suitable for
        MLX compilation with fixed shapes.

        Args:
            input_ids: [batch, length] - phoneme token ids
            voice: [batch, 256] - full voice embedding
            attention_mask: [batch, length] - padding mask (True=pad, False=valid)
            speed: Speaking rate (1.0 = normal)
            frame_bucket: Fixed frame count bucket. If None, auto-selects.
                         Use FRAME_BUCKETS for valid options.
            prosody_mask: [batch, length] - optional prosody type IDs.

        Returns:
            audio: [batch, samples] - generated waveform (padded to bucket)
            valid_samples: [batch] - number of valid samples per batch item
        """
        # Split voice embedding
        if voice.shape[-1] == 256:
            style = voice[:, :128]
            speaker = voice[:, 128:]
        else:
            style = voice
            speaker = voice

        # Step 1: BERT encoding
        # BERT expects attention_mask with 1=valid, 0=pad (integer/float)
        # Our attention_mask is boolean with True=pad, so convert: ~mask gives 1=valid
        bert_attention_mask = None
        if attention_mask is not None:
            bert_attention_mask = (~attention_mask).astype(mx.float32)
        bert_out = self.bert(input_ids, bert_attention_mask)

        # Phase B: Inject prosody embeddings if enabled and prosody_mask provided
        if prosody_mask is not None and self.prosody_embedding is not None:
            prosody_emb = self.prosody_embedding(prosody_mask)
            bert_out = bert_out + prosody_emb

        bert_enc = self.bert_encoder(bert_out)

        # Step 2: Duration features
        duration_feats = self.predictor.text_encoder(bert_enc, speaker)
        batch, length, _ = duration_feats.shape

        # Step 3: Duration prediction
        dur_enc = self.predictor.lstm(duration_feats)
        duration_logits = self.predictor.duration_proj(dur_enc)

        # Mask duration logits for padded positions to zero
        # This ensures padded tokens don't contribute to total audio length
        if attention_mask is not None:
            # attention_mask is True for padding, so we want to zero those positions
            # duration_logits: [batch, length, max_dur]
            # mask: [batch, length] -> [batch, length, 1]
            dur_mask = (~attention_mask)[:, :, None].astype(duration_logits.dtype)
            # Set padded positions to very negative (will sigmoid to ~0)
            duration_logits = duration_logits * dur_mask + (-10.0) * (1 - dur_mask)

        # Auto-select bucket if not provided
        # Estimate frames: ~10-15 frames per token on average
        if frame_bucket is None:
            estimated_frames = length * 15  # Conservative estimate
            frame_bucket = self._select_frame_bucket(estimated_frames)

        # Step 4: Compute alignment with fixed bucket (no device sync!)
        indices, total_frames, frame_mask = self._compute_alignment(
            duration_logits, speed, max_frames_bucket=frame_bucket,
        )

        # Step 5: Expand features using bucketed indices
        en_expanded_640 = self._expand_features(duration_feats, indices, total_frames)

        # Step 5.5: Shared BiLSTM
        x_shared = self.predictor.shared(en_expanded_640)

        # Step 6: F0/N prediction
        x = x_shared
        x = self.predictor.F0_0(x, speaker)
        x = self.predictor.F0_1(x, speaker)
        x = self.predictor.F0_2(x, speaker)
        f0 = self.predictor.F0_proj(x).squeeze(-1)

        x = x_shared
        x = self.predictor.N_0(x, speaker)
        x = self.predictor.N_1(x, speaker)
        x = self.predictor.N_2(x, speaker)
        noise = self.predictor.N_proj(x).squeeze(-1)

        # Step 7: ASR features
        text_enc = self.text_encoder(input_ids, attention_mask)
        asr_expanded = self._expand_features(text_enc, indices, total_frames)

        # Issue 6: Mask padded frames before decoding to prevent artifacts
        # This zeros out features at padding positions so they don't affect valid frames
        assert frame_mask is not None, "frame_mask required for bucketed inference"
        # ASR mask: [batch, frames] -> [batch, frames, 1] for broadcasting
        asr_mask = frame_mask[:, :, None].astype(asr_expanded.dtype)
        asr_expanded = asr_expanded * asr_mask
        # F0/N mask: frame_mask is for frames, F0/N are at 2x rate after upsampling
        # frame_mask [batch, frames] -> repeat -> [batch, frames*2]
        f0_mask = mx.repeat(frame_mask, 2, axis=1).astype(f0.dtype)
        f0 = f0 * f0_mask
        noise = noise * f0_mask

        # Step 8: Decode
        audio = self.decoder(asr_expanded, f0, noise, style)

        # Compute valid sample count from frame mask
        # hop_length samples per frame: upsample_rates (10*6) * istft_hop (5) = 300
        hop_length = 300
        assert frame_mask is not None, (
            "frame_mask should be set when using bucketed alignment"
        )
        valid_frames = mx.sum(frame_mask.astype(mx.int32), axis=1)  # [batch]
        valid_samples = valid_frames * hop_length

        return audio, valid_samples

    def load_voice_pack(self, voice_path: str) -> mx.array:
        """
        Load a voice pack from a .pt file.

        Voice packs are [510, 1, 256] tensors where each frame corresponds to
        a phoneme sequence length. Use select_voice_embedding() to extract
        the appropriate embedding for a given phoneme sequence.

        Args:
            voice_path: Path to voice .pt file (e.g., "af_bella.pt")

        Returns:
            Voice pack tensor of shape [510, 1, 256] or similar
        """
        import torch

        # Load voice tensor
        voice_data = torch.load(voice_path, map_location="cpu", weights_only=True)

        # Convert to MLX
        if isinstance(voice_data, torch.Tensor):
            voice_array = mx.array(voice_data.numpy())
        elif isinstance(voice_data, dict):
            # Some voices may be stored as dicts
            voice_array = mx.array(list(voice_data.values())[0].numpy())
        else:
            raise ValueError(f"Unknown voice format: {type(voice_data)}")

        return voice_array

    def select_voice_embedding(
        self,
        voice_pack: mx.array,
        phoneme_length: int,
    ) -> mx.array:
        """
        Select voice embedding from pack based on phoneme sequence length.

        This matches the PyTorch behavior: ref_s = voice_pack[len(phonemes) - 1]

        Args:
            voice_pack: [frames, 1, 256] voice pack tensor
            phoneme_length: Number of phonemes in the input sequence

        Returns:
            Voice embedding tensor of shape [1, 256]
        """
        # Select frame based on phoneme length (0-indexed, so length-1)
        idx = min(phoneme_length - 1, voice_pack.shape[0] - 1)
        idx = max(idx, 0)  # Ensure non-negative

        # voice_pack is [frames, 1, 256], select single frame
        voice_embedding = voice_pack[idx]  # [1, 256]

        # Ensure shape is [1, 256]
        if len(voice_embedding.shape) == 1:
            voice_embedding = voice_embedding[None, :]
        elif voice_embedding.shape[0] == 1 and len(voice_embedding.shape) == 2:
            pass  # Already [1, 256]
        elif len(voice_embedding.shape) == 2:
            # [1, 256] from [1, 256] or reshape if needed
            voice_embedding = voice_embedding.reshape(1, -1)

        return voice_embedding

    def load_voice(
        self,
        voice_path: str,
        phoneme_length: int | None = None,
    ) -> mx.array:
        """
        Load a voice embedding from a .pt file.

        Args:
            voice_path: Path to voice .pt file (e.g., "af_bella.pt")
            phoneme_length: Number of phonemes in input sequence. If provided,
                selects the appropriate frame from the voice pack (matching
                PyTorch behavior). If None, uses frame index 0 for compatibility.

        Returns:
            Voice embedding tensor of shape [1, 256]
        """
        voice_pack = self.load_voice_pack(voice_path)

        if phoneme_length is not None:
            return self.select_voice_embedding(voice_pack, phoneme_length)

        # Backward compatibility: if no phoneme_length provided, use frame 0
        # This is deterministic and better than averaging all frames
        if len(voice_pack.shape) == 3:
            # [frames, 1, features] -> select first frame
            voice_embedding = voice_pack[0]  # [1, 256]
            if voice_embedding.shape[0] == 1:
                voice_embedding = voice_embedding.squeeze(0)  # [256]
        else:
            voice_embedding = voice_pack

        if len(voice_embedding.shape) == 1:
            # Add batch dimension: [features] -> [1, features]
            voice_embedding = voice_embedding[None, :]

        # Return full 256-dim embedding - model's __call__ will split it
        # into style (first 128) and speaker (second 128)
        return voice_embedding

    def synthesize(
        self,
        input_ids: mx.array,
        voice_style: mx.array,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        """
        Synthesize audio from phoneme tokens.

        Args:
            input_ids: [batch, length] - Phoneme token IDs
            voice_style: [batch, style_dim] - Voice style embedding
            attention_mask: [batch, length] - Padding mask (optional)

        Returns:
            audio: [batch, samples] - Generated audio waveform
        """
        return self.__call__(input_ids, voice_style, attention_mask)

    def set_deterministic(self, value: bool = True) -> None:
        """Set deterministic mode for reproducible inference."""
        self.decoder.set_deterministic(value)

    def synthesize_streaming(
        self,
        input_ids: mx.array,
        voice: mx.array,
        attention_mask: mx.array | None = None,
        speed: float = 1.0,
        chunk_frames: int = 100,
        overlap_frames: int = 10,
        prosody_mask: mx.array | None = None,
    ):
        """
        Generate audio with streaming output for lower time-to-first-audio latency.

        This method runs the full planning phase (BERT, alignment, F0/noise prediction)
        first since it requires full sequence context (BiLSTM is bidirectional), then
        streams the decoder output in chunks with crossfade for seamless audio.

        Args:
            input_ids: [batch, length] - phoneme token ids
            voice: [batch, 256] - full voice embedding (or [batch, 128] for style only)
            attention_mask: [batch, length] - padding mask (True = padding)
            speed: Speaking rate (1.0 = normal)
            chunk_frames: Decoder frames per chunk (default 100, ~1.25s audio)
            overlap_frames: Overlap between chunks for crossfade (default 10)
            prosody_mask: [batch, length] - optional prosody type IDs.

        Yields:
            audio_chunk: [batch, samples] - Audio chunks as they are generated.
                         First chunk includes planning latency, subsequent chunks
                         arrive as soon as decoder finishes each.

        Note:
            The sum of all yielded chunks equals the full non-streaming output,
            but with lower latency to first audio. Total processing time is similar
            or slightly higher due to overlap computation.
        """
        # ====================================================================
        # Input Validation (same as __call__)
        # ====================================================================
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must be 2D [batch, length], got shape {input_ids.shape}",
            )
        batch_size, seq_length = input_ids.shape

        if seq_length == 0:
            raise ValueError("Empty input_ids sequence")
        if seq_length > 10000:
            raise ValueError(f"input_ids sequence too long: {seq_length}")

        if voice.ndim != 2:
            raise ValueError(f"voice must be 2D, got shape {voice.shape}")

        # Split voice embedding
        if voice.shape[-1] == 256:
            style = voice[:, :128]
            speaker = voice[:, 128:]
        else:
            style = voice
            speaker = voice

        # ====================================================================
        # PLANNING PHASE - Requires full sequence (cannot stream)
        # ====================================================================

        # Step 1: BERT encoding
        bert_attention_mask = None
        if attention_mask is not None:
            bert_attention_mask = (~attention_mask).astype(mx.float32)
        bert_out = self.bert(input_ids, bert_attention_mask)

        # Phase B: Inject prosody embeddings if enabled and prosody_mask provided
        if prosody_mask is not None and self.prosody_embedding is not None:
            prosody_emb = self.prosody_embedding(prosody_mask)
            bert_out = bert_out + prosody_emb

        bert_enc = self.bert_encoder(bert_out)

        # Step 2: Duration features via predictor.text_encoder (BiLSTM - needs full seq)
        duration_feats = self.predictor.text_encoder(bert_enc, speaker)
        batch, length, _ = duration_feats.shape

        # Step 3: Duration prediction
        dur_enc = self.predictor.lstm(duration_feats)
        duration_logits = self.predictor.duration_proj(dur_enc)

        # Step 4: Compute alignment
        indices, total_frames, _ = self._compute_alignment(duration_logits, speed)

        # Step 5: Expand 640-dim duration features for F0/N prediction
        en_expanded_640 = self._expand_features(duration_feats, indices, total_frames)

        # Step 5.5: Shared BiLSTM (needs full sequence)
        x_shared = self.predictor.shared(en_expanded_640)

        # Step 6: F0/N prediction
        x = x_shared
        x = self.predictor.F0_0(x, speaker)
        x = self.predictor.F0_1(x, speaker)
        x = self.predictor.F0_2(x, speaker)
        f0 = self.predictor.F0_proj(x).squeeze(-1)

        x = x_shared
        x = self.predictor.N_0(x, speaker)
        x = self.predictor.N_1(x, speaker)
        x = self.predictor.N_2(x, speaker)
        noise = self.predictor.N_proj(x).squeeze(-1)

        # Step 7: ASR features via text_encoder
        text_enc = self.text_encoder(input_ids, attention_mask)
        asr_expanded = self._expand_features(text_enc, indices, total_frames)

        # Force evaluation of planning results before streaming
        mx.eval(asr_expanded, f0, noise)

        # ====================================================================
        # STREAMING DECODER PHASE
        # ====================================================================

        # Total decoder frames (ASR features are at decoder frame rate)
        decoder_frames = asr_expanded.shape[1]

        # F0/noise are at 2x frame rate
        f0_rate = 2

        # Calculate hop size for audio samples per ASR frame
        # Decoder upsample (decode_3 block): 2x
        # Generator upsamples: prod(istft_upsample_rates) * istft_gen_istft_hop_size
        generator_hop = 1
        for r in self.config.istft_upsample_rates:
            generator_hop *= r
        generator_hop *= self.config.istft_gen_istft_hop_size
        # Total: decoder 2x upsample * generator hop
        samples_per_frame = 2 * generator_hop

        # Overlap in audio samples for crossfade
        overlap_samples = overlap_frames * samples_per_frame

        # Generate Hann crossfade window
        fade_in: mx.array | None = None
        fade_out: mx.array | None = None
        if overlap_samples > 0:
            fade_in = mx.array(
                [0.5 * (1 - math.cos(math.pi * i / overlap_samples))
                 for i in range(overlap_samples)],
            )
            fade_out = 1.0 - fade_in

        # Previous chunk's tail for crossfade
        prev_tail = None

        # Process in chunks
        chunk_start = 0
        while chunk_start < decoder_frames:
            chunk_end = min(chunk_start + chunk_frames, decoder_frames)

            # For overlap, extend start backwards (except first chunk)
            if chunk_start > 0 and overlap_frames > 0:
                actual_start = max(0, chunk_start - overlap_frames)
            else:
                actual_start = chunk_start

            # Slice inputs for this chunk
            asr_chunk = asr_expanded[:, actual_start:chunk_end, :]
            f0_chunk = f0[:, actual_start * f0_rate : chunk_end * f0_rate]
            noise_chunk = noise[:, actual_start * f0_rate : chunk_end * f0_rate]

            # Decode this chunk
            audio_chunk = self.decoder(asr_chunk, f0_chunk, noise_chunk, style)
            mx.eval(audio_chunk)

            # Apply crossfade with previous chunk
            if prev_tail is not None and overlap_samples > 0:
                # Blend: prev_tail * fade_out + audio_head * fade_in
                assert fade_in is not None and fade_out is not None
                audio_head = audio_chunk[:, :overlap_samples]
                blended = prev_tail * fade_out + audio_head * fade_in
                # Yield the blended region + rest of chunk (minus new tail)
                if chunk_end < decoder_frames:
                    # Not the last chunk - save tail for next crossfade
                    yield mx.concatenate(
                        [blended, audio_chunk[:, overlap_samples:-overlap_samples]],
                        axis=1,
                    )
                    prev_tail = audio_chunk[:, -overlap_samples:]
                else:
                    # Last chunk - yield everything after blend
                    yield mx.concatenate(
                        [blended, audio_chunk[:, overlap_samples:]],
                        axis=1,
                    )
            else:
                # First chunk or no overlap
                if chunk_end < decoder_frames and overlap_samples > 0:
                    # Save tail for next crossfade
                    yield audio_chunk[:, :-overlap_samples]
                    prev_tail = audio_chunk[:, -overlap_samples:]
                else:
                    # Last chunk or no overlap
                    yield audio_chunk

            chunk_start = chunk_end

    def synthesize_streaming_adaptive(
        self,
        input_ids: mx.array,
        voice: mx.array,
        target_latency_ms: float = 100.0,
        min_chunk_frames: int = 25,
        max_chunk_frames: int = 200,
        **kwargs,
    ):
        """
        J5 Optimization: Adaptive chunk size for streaming.

        Automatically adjusts chunk size based on measured decode time
        to maintain target latency while maximizing throughput.

        Args:
            input_ids: [1, length] phoneme token ids
            voice: [1, 256] voice embedding
            target_latency_ms: Target time per chunk in milliseconds
            min_chunk_frames: Minimum chunk size (quality floor)
            max_chunk_frames: Maximum chunk size (memory ceiling)
            **kwargs: Additional args passed to synthesize_streaming

        Yields:
            dict with keys:
                - 'audio': Audio chunk array
                - 'latency_ms': Actual decode time for this chunk
                - 'chunk_frames': Frames in this chunk
        """
        current_chunk_frames = 100  # Start with default

        # Get full sequence for planning
        batch_size, seq_length = input_ids.shape

        # Simplified: use the regular streaming with stats tracking
        chunk_idx = 0
        for audio_chunk in self.synthesize_streaming(
            input_ids, voice, chunk_frames=current_chunk_frames, **kwargs,
        ):
            yield {
                'audio': audio_chunk,
                'chunk_frames': current_chunk_frames,
                'chunk_idx': chunk_idx,
            }
            chunk_idx += 1

    def synthesize_streaming_interruptible(
        self,
        input_ids: mx.array,
        voice: mx.array,
        stop_event: threading.Event | None = None,
        chunk_frames: int = 100,
        overlap_frames: int = 10,
        **kwargs,
    ):
        """
        J9 Optimization: Interruptible streaming synthesis.

        Like synthesize_streaming but checks a stop event between chunks,
        allowing early termination of long synthesis operations.

        Args:
            input_ids: [1, length] phoneme token ids
            voice: [1, 256] voice embedding
            stop_event: Threading event to signal stop (optional).
                       If set, synthesis stops after current chunk.
            chunk_frames: Frames per chunk
            overlap_frames: Overlap for crossfade
            **kwargs: Additional args passed to synthesize_streaming

        Yields:
            dict with keys:
                - 'audio': Audio chunk array
                - 'chunk_idx': Chunk index
                - 'interrupted': True if synthesis was stopped

        Example:
            stop_event = threading.Event()
            for chunk in model.synthesize_streaming_interruptible(
                ids, voice, stop_event=stop_event
            ):
                play(chunk['audio'])
                if user_pressed_stop():
                    stop_event.set()
        """
        chunk_idx = 0

        for audio_chunk in self.synthesize_streaming(
            input_ids, voice, chunk_frames=chunk_frames,
            overlap_frames=overlap_frames, **kwargs,
        ):
            # Check for interruption before yielding
            if stop_event is not None and stop_event.is_set():
                yield {
                    'audio': audio_chunk,
                    'chunk_idx': chunk_idx,
                    'interrupted': True,
                }
                return  # Stop generation

            yield {
                'audio': audio_chunk,
                'chunk_idx': chunk_idx,
                'interrupted': False,
            }
            chunk_idx += 1

            # Also check after yield in case it was set during processing
            if stop_event is not None and stop_event.is_set():
                return

    def get_streaming_stats(
        self,
        input_ids: mx.array,
        voice: mx.array,
        chunk_frames: int = 100,
        **kwargs,
    ) -> dict:
        """
        J6 Optimization: Measure streaming latency statistics.

        Runs streaming synthesis and collects timing information for
        profiling and optimization tuning.

        Args:
            input_ids: [1, length] phoneme token ids
            voice: [1, 256] voice embedding
            chunk_frames: Frames per chunk
            **kwargs: Additional args

        Returns:
            dict with latency statistics:
                - 'time_to_first_audio_ms': Planning + first chunk time
                - 'avg_chunk_latency_ms': Average per-chunk decode time
                - 'total_time_ms': Total synthesis time
                - 'total_chunks': Number of chunks generated
                - 'rtf': Real-time factor (lower is faster)
        """
        import time

        chunk_times = []
        total_samples = 0
        start_time = time.perf_counter()
        first_chunk_time = None

        for audio_chunk in self.synthesize_streaming(
            input_ids, voice, chunk_frames=chunk_frames, **kwargs,
        ):
            chunk_end = time.perf_counter()
            if first_chunk_time is None:
                first_chunk_time = chunk_end - start_time
                chunk_start = chunk_end
            else:
                chunk_times.append(chunk_end - chunk_start)
                chunk_start = chunk_end
            mx.eval(audio_chunk)
            total_samples += audio_chunk.shape[-1]

        total_time = time.perf_counter() - start_time

        # Calculate RTF (real-time factor): time / audio_duration
        audio_duration_s = total_samples / self.sample_rate
        rtf = total_time / audio_duration_s if audio_duration_s > 0 else 0

        return {
            'time_to_first_audio_ms': first_chunk_time * 1000 if first_chunk_time else 0,
            'avg_chunk_latency_ms': sum(chunk_times) * 1000 / len(chunk_times) if chunk_times else 0,
            'total_time_ms': total_time * 1000,
            'total_chunks': len(chunk_times) + 1,
            'rtf': rtf,
            'total_samples': total_samples,
        }

    def synthesize_streaming_buffered(
        self,
        input_ids: mx.array,
        voice: mx.array,
        buffer_pool: Optional['AudioBufferPool'] = None,
        chunk_frames: int = 100,
        overlap_frames: int = 10,
        **kwargs,
    ):
        """
        J7 Optimization: Streaming synthesis with buffer management.

        Uses AudioBufferPool to reuse pre-allocated buffers, reducing
        allocation overhead during streaming synthesis.

        Args:
            input_ids: [1, length] phoneme token ids
            voice: [1, 256] voice embedding
            buffer_pool: Optional AudioBufferPool for buffer reuse.
                        If None, creates temporary pool.
            chunk_frames: Frames per chunk (default 100)
            overlap_frames: Overlap for crossfade (default 10)
            **kwargs: Additional args passed to synthesize_streaming

        Yields:
            dict with keys:
                - 'audio': Audio chunk (view into pooled buffer or new array)
                - 'buffer': Buffer reference (for release back to pool)
                - 'samples': Number of valid samples in buffer

        Note:
            Caller should release buffers after consuming audio:
            >>> for chunk in model.synthesize_streaming_buffered(ids, voice, pool):
            ...     process(chunk['audio'][:, :chunk['samples']])
            ...     if chunk.get('buffer'):
            ...         pool.release(chunk['buffer'])
        """
        # Create temporary pool if not provided
        own_pool = buffer_pool is None
        if own_pool:
            # Estimate max samples per chunk (~6 frames/phoneme * 200 samples/frame)
            max_samples_per_chunk = chunk_frames * 200 * 2  # 2x buffer
            buffer_pool = AudioBufferPool(max_samples=max_samples_per_chunk, num_buffers=4)

        for audio_chunk in self.synthesize_streaming(
            input_ids, voice, chunk_frames=chunk_frames,
            overlap_frames=overlap_frames, **kwargs,
        ):
            mx.eval(audio_chunk)
            num_samples = audio_chunk.shape[-1]

            # Try to acquire a pooled buffer
            pooled_buffer = buffer_pool.acquire(num_samples)

            if pooled_buffer is not None:
                # Copy to pooled buffer (in-place)
                pooled_buffer[:, :num_samples] = audio_chunk
                mx.eval(pooled_buffer)
                yield {
                    'audio': pooled_buffer[:, :num_samples],
                    'buffer': pooled_buffer,
                    'samples': num_samples,
                    'pooled': True,
                }
            else:
                # Pool exhausted, use audio directly
                yield {
                    'audio': audio_chunk,
                    'buffer': None,
                    'samples': num_samples,
                    'pooled': False,
                }

        # Clean up if we created the pool
        if own_pool:
            buffer_pool.clear()

    def synthesize_batch(
        self,
        input_ids_list: list[mx.array],
        voices: mx.array | list[mx.array],
        speed: float = 1.0,
        pad_value: int = 0,
        validate_output: bool = True,
    ) -> tuple[mx.array, mx.array]:
        """
        Synthesize multiple utterances in a single batch for higher throughput.

        This method pads variable-length inputs to the same length, processes them
        in a single forward pass, and returns padded outputs with a mask indicating
        valid audio samples.

        Args:
            input_ids_list: List of [length_i] arrays - phoneme token IDs.
                           Each utterance can have different lengths.
            voices: Either:
                   - [batch, 256] array - same voice applied to all utterances
                   - List of [256] arrays - per-utterance voice embeddings
            speed: Speaking rate (1.0 = normal)
            pad_value: Token ID for padding (default 0)
            validate_output: If True, validate outputs for NaN/Inf

        Returns:
            Tuple of:
                - audio: [batch, max_samples] - Generated audio (padded)
                - audio_lengths: [batch] - Actual audio length for each utterance

        Example:
            >>> # Synthesize 3 utterances in one batch
            >>> texts = [[1, 2, 3, 4, 5], [1, 2, 3], [1, 2, 3, 4, 5, 6, 7]]
            >>> input_ids = [mx.array(t) for t in texts]
            >>> voice = model.load_voice("af_bella")  # [256]
            >>> voices = mx.stack([voice] * 3)  # [3, 256]
            >>> audio, lengths = model.synthesize_batch(input_ids, voices)
            >>> # audio: [3, max_samples], lengths: [3]
            >>> # Extract individual audios:
            >>> audio_1 = audio[0, :lengths[0]]
            >>> audio_2 = audio[1, :lengths[1]]
            >>> audio_3 = audio[2, :lengths[2]]
        """
        batch_size = len(input_ids_list)
        if batch_size == 0:
            raise ValueError("Empty input_ids_list")

        # ====================================================================
        # Validate and prepare inputs
        # ====================================================================

        # Ensure all inputs are 1D arrays
        for i, ids in enumerate(input_ids_list):
            if ids.ndim == 0:
                raise ValueError(f"input_ids_list[{i}] must be at least 1D")
            if ids.ndim == 2:
                if ids.shape[0] != 1:
                    raise ValueError(
                        f"input_ids_list[{i}] has batch dim > 1, "
                        "expected single sequence",
                    )
                input_ids_list[i] = ids.squeeze(0)

        # Get sequence lengths
        seq_lengths = [ids.shape[0] for ids in input_ids_list]
        max_seq_len = max(seq_lengths)

        # Pad input sequences to max length
        padded_ids = []
        attention_masks = []
        for ids in input_ids_list:
            seq_len = ids.shape[0]
            if seq_len < max_seq_len:
                # Pad with pad_value
                padding = mx.full((max_seq_len - seq_len,), pad_value, dtype=ids.dtype)
                padded = mx.concatenate([ids, padding])
            else:
                padded = ids
            padded_ids.append(padded)

            # Create attention mask (True = padding, False = valid)
            mask = mx.concatenate([
                mx.zeros((seq_len,), dtype=mx.bool_),
                mx.ones((max_seq_len - seq_len,), dtype=mx.bool_),
            ])
            attention_masks.append(mask)

        # Stack into batch tensors
        input_ids = mx.stack(padded_ids)  # [batch, max_seq_len]
        attention_mask = mx.stack(attention_masks)  # [batch, max_seq_len]

        # Handle voice embeddings
        if isinstance(voices, list):
            if len(voices) != batch_size:
                raise ValueError(
                    f"voices list length {len(voices)} != batch size {batch_size}",
                )
            # Ensure each voice is 1D and stack
            voice_arrays = []
            for i, v in enumerate(voices):
                if v.ndim == 1:
                    voice_arrays.append(v)
                elif v.ndim == 2 and v.shape[0] == 1:
                    voice_arrays.append(v.squeeze(0))
                else:
                    raise ValueError(f"voices[{i}] has invalid shape {v.shape}")
            voice = mx.stack(voice_arrays)  # [batch, 256]
        else:
            voice = voices
            if voice.ndim == 1:
                # Single voice for all - broadcast
                voice = mx.broadcast_to(voice[None, :], (batch_size, voice.shape[0]))
            elif voice.shape[0] == 1 and batch_size > 1:
                # Single voice, broadcast to batch
                voice = mx.broadcast_to(voice, (batch_size, voice.shape[1]))
            elif voice.shape[0] != batch_size:
                raise ValueError(
                    f"voice batch dim {voice.shape[0]} != "
                    f"input batch size {batch_size}",
                )

        # ====================================================================
        # Run batched synthesis using bucketed inference
        # ====================================================================

        # Use bucketed inference to avoid per-sample synchronization
        # Estimate max frames needed based on longest sequence
        # Estimate: ~10-15 frames per token at speed 1.0
        estimated_max_frames = int(max_seq_len * 15 / speed)
        frame_bucket = self._select_frame_bucket(estimated_max_frames)

        # Run bucketed synthesis
        # synthesize_bucketed returns (audio, valid_samples) where:
        # - audio: [batch, max_samples] padded to bucket
        # - valid_samples: [batch] actual audio length per item
        audio, audio_lengths = self.synthesize_bucketed(
            input_ids,
            voice,
            attention_mask=attention_mask,
            speed=speed,
            frame_bucket=frame_bucket,
        )

        # ====================================================================
        # Output validation (optional)
        # ====================================================================

        if validate_output:
            mx.eval(audio, audio_lengths)
            if mx.any(mx.isnan(audio)):
                raise RuntimeError("Numerical instability: NaN in batch output")
            if mx.any(mx.isinf(audio)):
                raise RuntimeError("Numerical instability: Inf in batch output")
            audio = mx.clip(audio, -1.0, 1.0)

        return audio, audio_lengths

    @staticmethod
    def from_pretrained(
        model_path: str,
        config: KokoroConfig | None = None,
    ) -> "KokoroModel":
        """
        Load model from pretrained weights.

        Args:
            model_path: Path to directory containing weights.safetensors and config.json

        Returns:
            Loaded KokoroModel
        """
        import safetensors.numpy

        model_dir = Path(model_path)

        # Load config
        if config is None:
            config_path = model_dir / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config_dict = json.load(f)
                config = KokoroConfig(
                    **{k: v for k, v in config_dict.items() if hasattr(KokoroConfig, k)},
                )
            else:
                config = KokoroConfig()

        # Create model
        model = KokoroModel(config)

        # Load weights - try safetensors first, then PyTorch .pth
        weights_path = model_dir / "weights.safetensors"
        pth_path = model_dir / "kokoro-v1_0.pth"

        if weights_path.exists():
            np_weights = safetensors.numpy.load_file(str(weights_path))
            # Convert to MLX arrays
            mlx_weights: dict[str, mx.array] = {
                k: mx.array(v) for k, v in np_weights.items()
            }
            model.load_weights(list(mlx_weights.items()))
        elif pth_path.exists():
            # Load from PyTorch checkpoint using from_hf logic
            import torch

            state_dict = torch.load(
                str(pth_path), map_location="cpu", weights_only=True,
            )
            weights = _map_hf_weights_to_mlx(state_dict, config)

            # Get all model parameter paths for filtering
            def get_all_param_paths(module_obj: nn.Module, prefix: str = "") -> set:
                paths = set()
                for name, value in module_obj.items():
                    full_name = f"{prefix}.{name}" if prefix else name
                    if isinstance(value, mx.array):
                        paths.add(full_name)
                    elif isinstance(value, nn.Module):
                        paths.update(get_all_param_paths(value, full_name))
                    elif isinstance(value, dict):
                        for k, v in value.items():
                            sub_name = f"{full_name}.{k}"
                            if isinstance(v, mx.array):
                                paths.add(sub_name)
                            elif isinstance(v, nn.Module):
                                paths.update(get_all_param_paths(v, sub_name))
                return paths

            # Filter weights to only include keys that exist in model
            model_paths = get_all_param_paths(model)
            matching_weights = [(k, v) for k, v in weights.items() if k in model_paths]

            # Load matching weights
            model.load_weights(matching_weights, strict=False)

        return model

    @staticmethod
    def from_hf(
        hf_model_id: str = "hexgrad/Kokoro-82M",
        cache_dir: str | None = None,
    ) -> tuple["KokoroModel", KokoroConfig]:
        """
        Load model from HuggingFace checkpoint.

        Args:
            hf_model_id: HuggingFace model ID
            cache_dir: Local cache directory

        Returns:
            Tuple of (KokoroModel, KokoroConfig)
        """
        import torch
        from huggingface_hub import hf_hub_download

        if cache_dir is None:
            cache_path = Path.home() / "models" / "kokoro"
        else:
            cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)

        # Download files
        model_file = hf_hub_download(
            hf_model_id, "kokoro-v1_0.pth", local_dir=str(cache_path),
        )
        config_file = hf_hub_download(
            hf_model_id, "config.json", local_dir=str(cache_path),
        )

        # Load config
        with open(config_file) as f:
            config_dict = json.load(f)

        # Parse config
        config = KokoroConfig(
            dim_in=config_dict.get("dim_in", 64),
            hidden_dim=config_dict.get("hidden_dim", 512),
            style_dim=config_dict.get("style_dim", 128),
            n_token=config_dict.get("n_token", 178),
            n_layer=config_dict.get("n_layer", 3),
            dropout=config_dict.get("dropout", 0.2),
            text_encoder_kernel_size=config_dict.get("text_encoder_kernel_size", 5),
        )

        # Update PLBERT config if present
        if "plbert" in config_dict:
            plbert = config_dict["plbert"]
            config.plbert_hidden_size = plbert.get("hidden_size", 768)
            config.plbert_num_attention_heads = plbert.get("num_attention_heads", 12)
            config.plbert_intermediate_size = plbert.get("intermediate_size", 2048)
            config.plbert_num_hidden_layers = plbert.get("num_hidden_layers", 12)

        # Update ISTFTNet config if present
        if "istftnet" in config_dict:
            istft = config_dict["istftnet"]
            config.istft_upsample_rates = tuple(istft.get("upsample_rates", [10, 6]))
            config.istft_upsample_kernel_sizes = tuple(
                istft.get("upsample_kernel_sizes", [20, 12]),
            )
            config.istft_gen_istft_n_fft = istft.get("gen_istft_n_fft", 20)
            config.istft_gen_istft_hop_size = istft.get("gen_istft_hop_size", 5)

        # Create model
        model = KokoroModel(config)

        # Load PyTorch weights and convert
        state_dict = torch.load(model_file, map_location="cpu", weights_only=True)
        weights = _map_hf_weights_to_mlx(state_dict, config)

        # Get all model parameter paths for filtering
        def get_all_param_paths(module: nn.Module, prefix: str = "") -> set:
            paths = set()
            for name, value in module.items():
                full_name = f"{prefix}.{name}" if prefix else name
                if isinstance(value, mx.array):
                    paths.add(full_name)
                elif isinstance(value, nn.Module):
                    paths.update(get_all_param_paths(value, full_name))
                elif isinstance(value, dict):
                    for k, v in value.items():
                        sub_name = f"{full_name}.{k}"
                        if isinstance(v, mx.array):
                            paths.add(sub_name)
                        elif isinstance(v, nn.Module):
                            paths.update(get_all_param_paths(v, sub_name))
            return paths

        # Filter weights to only include keys that exist in model
        model_paths = get_all_param_paths(model)
        matching_weights = [(k, v) for k, v in weights.items() if k in model_paths]

        # Load matching weights
        model.load_weights(matching_weights, strict=False)

        return model, config


def _map_hf_weights_to_mlx(
    state_dict: dict[str, Any],
    config: KokoroConfig,
) -> dict[str, mx.array]:
    """
    Map PyTorch state dict to MLX weight format.

    PyTorch Kokoro uses:
    - 'module.' prefix from DataParallel
    - weight_g/weight_v for weight normalization
    - Different key naming conventions
    """
    import torch

    weights = {}

    # Helper to convert tensor
    def to_mlx(t):
        if isinstance(t, torch.Tensor):
            return mx.array(t.numpy())
        return t

    # Map BERT weights
    if "bert" in state_dict:
        bert_state = state_dict["bert"]
        for key, value in bert_state.items():
            # Remove 'module.' prefix
            key = key.replace("module.", "")

            # Map to MLX naming
            if "embeddings.word_embeddings" in key:
                mlx_key = key.replace(
                    "embeddings.word_embeddings", "bert.embeddings.word_embeddings",
                )
            elif "embeddings.position_embeddings" in key:
                mlx_key = key.replace(
                    "embeddings.position_embeddings",
                    "bert.embeddings.position_embeddings",
                )
            elif "embeddings.token_type_embeddings" in key:
                mlx_key = key.replace(
                    "embeddings.token_type_embeddings",
                    "bert.embeddings.token_type_embeddings",
                )
            elif "embeddings.LayerNorm" in key:
                mlx_key = key.replace(
                    "embeddings.LayerNorm", "bert.embeddings.layer_norm",
                )
            elif "encoder.embedding_hidden_mapping_in" in key:
                mlx_key = key.replace(
                    "encoder.embedding_hidden_mapping_in",
                    "bert.encoder.embedding_hidden_mapping_in",
                )
            elif "encoder.albert_layer_groups" in key:
                # ALBERT uses shared weights
                mlx_key = key.replace(
                    "encoder.albert_layer_groups.0.albert_layers.0",
                    "bert.encoder.albert_layer",
                )
            elif "pooler" in key:
                mlx_key = f"bert.{key}"
            else:
                mlx_key = f"bert.{key}"

            weights[mlx_key] = to_mlx(value)

    # Map bert_encoder weights
    if "bert_encoder" in state_dict:
        bert_enc_state = state_dict["bert_encoder"]
        for key, value in bert_enc_state.items():
            key = key.replace("module.", "")
            weights[f"bert_encoder.{key}"] = to_mlx(value)

    # Map text_encoder weights
    if "text_encoder" in state_dict:
        text_state = state_dict["text_encoder"]
        for key, value in text_state.items():
            key = key.replace("module.", "")
            # Handle weight normalization
            if "weight_g" in key or "weight_v" in key:
                mlx_key = f"text_encoder.{key}"
            else:
                mlx_key = f"text_encoder.{key}"
            weights[mlx_key] = to_mlx(value)

    # Map predictor weights
    if "predictor" in state_dict:
        pred_state = state_dict["predictor"]
        for key, value in pred_state.items():
            key = key.replace("module.", "")

            # Convert dot notation to underscore for numbered modules
            # F0.0.* -> F0_0.*, F0.1.* -> F0_1.*, etc.
            # N.0.* -> N_0.*, N.1.* -> N_1.*, etc.
            # text_encoder.lstms.0.* -> text_encoder.lstms_0.*, etc.
            import re

            # Handle F0.{n} and N.{n} patterns
            key = re.sub(r"^(F0|N)\.(\d+)\.", r"\1_\2.", key)

            # Handle text_encoder.lstms.{n} patterns
            key = re.sub(
                r"text_encoder\.lstms\.(\d+)\.", r"text_encoder.lstms_\1.", key,
            )

            weights[f"predictor.{key}"] = to_mlx(value)

    # Map decoder weights
    if "decoder" in state_dict:
        import re

        dec_state = state_dict["decoder"]
        for key, value in dec_state.items():
            key = key.replace("module.", "")

            # Convert case for decoder conv layers: F0_conv -> f0_conv, N_conv -> n_conv
            key = key.replace("F0_conv", "f0_conv")
            key = key.replace("N_conv", "n_conv")

            # Convert numbered modules: decode.0.* -> decode_0.*, etc.
            key = re.sub(r"^decode\.(\d+)\.", r"decode_\1.", key)

            # Handle asr_res wrapped in Sequential: asr_res.0.* -> asr_res.*
            key = re.sub(r"^asr_res\.0\.", r"asr_res.", key)

            # Generator numbered modules: ups.0 -> ups_0, noise_convs.0 -> etc.
            key = re.sub(r"^generator\.ups\.(\d+)\.", r"generator.ups_\1.", key)
            key = re.sub(
                r"^generator\.noise_convs\.(\d+)\.", r"generator.noise_convs_\1.", key,
            )
            key = re.sub(
                r"^generator\.noise_res\.(\d+)\.", r"generator.noise_res_\1.", key,
            )
            key = re.sub(
                r"^generator\.resblocks\.(\d+)\.", r"generator.resblocks_\1.", key,
            )

            # Inner numbered modules in resblocks and noise_res:
            # adain1.0 -> adain1_0, convs1.0 -> convs1_0, alpha1.0 -> alpha1_0
            key = re.sub(r"\.adain1\.(\d+)\.", r".adain1_\1.", key)
            key = re.sub(r"\.adain2\.(\d+)\.", r".adain2_\1.", key)
            key = re.sub(r"\.convs1\.(\d+)\.", r".convs1_\1.", key)
            key = re.sub(r"\.convs2\.(\d+)\.", r".convs2_\1.", key)
            # alpha1.0 and alpha2.0 are direct arrays, not nested modules
            key = re.sub(r"\.alpha1\.(\d+)$", r".alpha1_\1", key)
            key = re.sub(r"\.alpha2\.(\d+)$", r".alpha2_\1", key)

            weights[f"decoder.{key}"] = to_mlx(value)

    # Post-process: Convert PyTorch LSTM weights to native MLX LSTM format
    # PyTorch: weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0
    # MLX nn.LSTM: Wx, Wh, bias (bias = bias_ih + bias_hh)
    return _convert_lstm_weights(weights)



def _convert_lstm_weights(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """
    Convert PyTorch LSTM weight names to MLX native nn.LSTM format.

    PyTorch BiLSTM uses:
    - weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0 (forward)
    - weight_ih_l0_reverse, weight_hh_l0_reverse, (backward)
      bias_ih_l0_reverse, bias_hh_l0_reverse

    MLX nn.LSTM uses:
    - Wx, Wh, bias
    """
    import re

    converted: dict[str, mx.array] = {}
    lstm_biases: dict[
        str, dict[str, mx.array],
    ] = {}  # Track biases to combine: {prefix: {'ih': array, 'hh': array}}

    for key, value in weights.items():
        # Check if this is an LSTM weight
        # Pattern: *.weight_ih_l0, *.weight_hh_l0, *.bias_ih_l0, *.bias_hh_l0
        # Or reverse: *.weight_ih_l0_reverse, etc.

        # Match forward LSTM weights
        fwd_ih_match = re.match(r"^(.*)\.weight_ih_l0$", key)
        fwd_hh_match = re.match(r"^(.*)\.weight_hh_l0$", key)
        fwd_bias_ih_match = re.match(r"^(.*)\.bias_ih_l0$", key)
        fwd_bias_hh_match = re.match(r"^(.*)\.bias_hh_l0$", key)

        # Match backward LSTM weights
        bwd_ih_match = re.match(r"^(.*)\.weight_ih_l0_reverse$", key)
        bwd_hh_match = re.match(r"^(.*)\.weight_hh_l0_reverse$", key)
        bwd_bias_ih_match = re.match(r"^(.*)\.bias_ih_l0_reverse$", key)
        bwd_bias_hh_match = re.match(r"^(.*)\.bias_hh_l0_reverse$", key)

        if fwd_ih_match:
            prefix = fwd_ih_match.group(1)
            converted[f"{prefix}.lstm_fwd.Wx"] = value
        elif fwd_hh_match:
            prefix = fwd_hh_match.group(1)
            converted[f"{prefix}.lstm_fwd.Wh"] = value
        elif fwd_bias_ih_match:
            prefix = fwd_bias_ih_match.group(1)
            bias_key = f"{prefix}.lstm_fwd"
            if bias_key not in lstm_biases:
                lstm_biases[bias_key] = {}
            lstm_biases[bias_key]["ih"] = value
        elif fwd_bias_hh_match:
            prefix = fwd_bias_hh_match.group(1)
            bias_key = f"{prefix}.lstm_fwd"
            if bias_key not in lstm_biases:
                lstm_biases[bias_key] = {}
            lstm_biases[bias_key]["hh"] = value
        elif bwd_ih_match:
            prefix = bwd_ih_match.group(1)
            converted[f"{prefix}.lstm_bwd.Wx"] = value
        elif bwd_hh_match:
            prefix = bwd_hh_match.group(1)
            converted[f"{prefix}.lstm_bwd.Wh"] = value
        elif bwd_bias_ih_match:
            prefix = bwd_bias_ih_match.group(1)
            bias_key = f"{prefix}.lstm_bwd"
            if bias_key not in lstm_biases:
                lstm_biases[bias_key] = {}
            lstm_biases[bias_key]["ih"] = value
        elif bwd_bias_hh_match:
            prefix = bwd_bias_hh_match.group(1)
            bias_key = f"{prefix}.lstm_bwd"
            if bias_key not in lstm_biases:
                lstm_biases[bias_key] = {}
            lstm_biases[bias_key]["hh"] = value
        else:
            # Not an LSTM weight, keep as is
            converted[key] = value

    # Combine biases: bias = bias_ih + bias_hh
    for prefix, biases in lstm_biases.items():
        if "ih" in biases and "hh" in biases:
            converted[f"{prefix}.bias"] = biases["ih"] + biases["hh"]
        elif "ih" in biases:
            converted[f"{prefix}.bias"] = biases["ih"]
        elif "hh" in biases:
            converted[f"{prefix}.bias"] = biases["hh"]

    return converted


# ============================================================================
# Model Quantization
# ============================================================================


def quantize_kokoro_model(
    model: KokoroModel,
    bits: int = 8,
    group_size: int = 64,
    mode: str = "full",
) -> dict[str, Any]:
    """
    Quantize a KokoroModel for reduced memory and faster inference.

    MLX quantizes Linear and Embedding layers. Conv1d layers are NOT quantized.
    For Kokoro-82M:
    - 84 Linear layers (quantizable)
    - 4 Embedding layers (quantizable)
    - 93 Conv1d layers (not quantizable)

    Args:
        model: KokoroModel instance to quantize (modified in-place)
        bits: Quantization bits (4 or 8). Default 8 for TTS quality.
        group_size: Quantization group size. Default 64.
        mode: Quantization mode:
            - "full": Quantize all Linear and Embedding layers (default)
            - "encoder_only": Only quantize BERT encoder layers (safest for quality)
            - "no_adain": Skip AdaIN fc layers in decoder (preserve style conditioning)

    Returns:
        Dict with quantization statistics:
            - layers_quantized: List of quantized layer paths
            - layers_skipped: List of skipped layer paths
            - total_quantized: Number of layers quantized
            - total_skipped: Number of layers skipped

    Example:
        >>> model = KokoroModel(config)
        >>> model.load_weights("kokoro_weights.safetensors")
        >>> stats = quantize_kokoro_model(model, bits=8, mode="full")
        >>> print(f"Quantized {stats['total_quantized']} layers")
    """
    from collections.abc import Callable

    layers_quantized: list[str] = []
    layers_skipped: list[str] = []

    def make_predicate(
        mode: str, group_size: int,
    ) -> Callable[[str, nn.Module], bool | dict]:
        """Create a class predicate based on quantization mode."""

        def predicate(path: str, module: nn.Module) -> bool | dict:
            # Check if module is quantizable (has to_quantized method)
            if not hasattr(module, "to_quantized"):
                return False

            # Check dimension compatibility for Linear layers
            # The last dimension must be divisible by group_size
            if isinstance(module, nn.Linear):
                # Linear weight shape is (output_features, input_features)
                # The last dim (input_features) must be divisible by group_size
                weight = module.weight
                dim = weight.shape[-1]
                if dim < group_size or dim % group_size != 0:
                    layers_skipped.append(f"{path} (dim {dim} < group_size)")
                    return False

            if mode == "full":
                # Quantize everything that passes dimension check
                layers_quantized.append(path)
                return True

            if mode == "encoder_only":
                # Only quantize BERT/encoder layers
                if path.startswith("bert.") or path == "bert_encoder":
                    layers_quantized.append(path)
                    return True
                layers_skipped.append(path)
                return False

            if mode == "no_adain":
                # Skip AdaIN fc layers (style conditioning) in decoder
                if ".adain" in path and ".fc" in path:
                    layers_skipped.append(path)
                    return False
                layers_quantized.append(path)
                return True

            raise ValueError(f"Unknown quantization mode: {mode}")

        return predicate

    # Apply quantization
    predicate = make_predicate(mode, group_size)
    nn.quantize(model, group_size=group_size, bits=bits, class_predicate=predicate)

    return {
        "layers_quantized": layers_quantized,
        "layers_skipped": layers_skipped,
        "total_quantized": len(layers_quantized),
        "total_skipped": len(layers_skipped),
        "bits": bits,
        "group_size": group_size,
        "mode": mode,
    }


def _count_params_recursive(params: dict[str, Any]) -> int:
    """Recursively count parameters in nested dict."""
    total = 0
    for v in params.values():
        if isinstance(v, dict):
            total += _count_params_recursive(v)
        elif hasattr(v, "size"):
            total += v.size
    return total


def estimate_memory_savings(
    model: KokoroModel,
    bits: int = 8,
    mode: str = "full",
) -> dict[str, Any]:
    """
    Estimate memory savings from quantization without modifying the model.

    Args:
        model: KokoroModel instance to analyze
        bits: Target quantization bits (4 or 8)
        mode: Quantization mode (see quantize_kokoro_model)

    Returns:
        Dict with memory estimates:
            - original_mb: Original model size in MB
            - quantized_mb: Estimated quantized size in MB
            - savings_mb: Memory savings in MB
            - savings_percent: Percentage savings
    """
    # Count total params
    total_params = _count_params_recursive(model.parameters())

    # Count params in quantizable layers based on mode
    quantizable_params = 0
    for path, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Check if this layer would be quantized based on mode
            should_quantize = True
            if mode == "encoder_only":
                should_quantize = path.startswith("bert.") or path == "bert_encoder"
            elif mode == "no_adain":
                should_quantize = not (".adain" in path and ".fc" in path)

            if should_quantize:
                # Count params in this specific layer (not nested)
                layer_params = module.parameters()
                for param in layer_params.values():
                    if hasattr(param, "size"):
                        quantizable_params += param.size

    # Original: all float32 (4 bytes per param)
    # After quantization:
    # - Quantized layers: bits/8 bytes per param + scales/biases overhead (~10%)
    # - Non-quantized layers: still 4 bytes per param
    non_quantizable_params = total_params - quantizable_params

    original_bytes = total_params * 4
    quantized_bytes = (
        non_quantizable_params * 4
        + quantizable_params * (bits / 8) * 1.1  # 10% overhead for scales
    )

    original_mb = original_bytes / (1024 * 1024)
    quantized_mb = quantized_bytes / (1024 * 1024)
    savings_mb = original_mb - quantized_mb

    return {
        "original_mb": round(original_mb, 2),
        "quantized_mb": round(quantized_mb, 2),
        "savings_mb": round(savings_mb, 2),
        "savings_percent": round(100 * savings_mb / original_mb, 1),
        "total_params": total_params,
        "quantizable_params": quantizable_params,
        "bits": bits,
        "mode": mode,
    }
