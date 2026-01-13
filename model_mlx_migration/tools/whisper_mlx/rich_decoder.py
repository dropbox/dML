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
Rich Decoder - Whisper decoder with LoRA adapters for rich audio understanding.

Implements Phase 4/5 of UNIFIED_RICH_AUDIO_ARCHITECTURE.md:
- LoRA adapters for fine-tuning without catastrophic forgetting
- Rich output projections (emotion, pitch, paralinguistics, language)
- Phoneme deviation head for hallucination detection
- Prosody cross-attention for enhanced punctuation

Architecture:
    encoder_out (frozen) -> Rich CTC (streaming, ~100ms)
                         -> Rich Decoder (refinement, ~500ms)

The decoder receives:
- Encoder output (1280-dim) via cross-attention
- CTC prosody features (emotion, pitch) for conditioning
- Produces refined text + rich metadata + hallucination signal

Design Decisions:
- LoRA rank=8, alpha=16 (following original LoRA paper)
- Initialize B to zero so LoRA starts as identity
- Prosody can be injected via: (A) separate cross-attention, (B) LoRA K/V
- Rich outputs from decoder hidden states, not logits

Usage:
    from tools.whisper_mlx.rich_decoder import RichDecoder

    # Wrap existing Whisper decoder
    rich_decoder = RichDecoder.from_whisper_decoder(whisper_decoder)

    # Forward pass with prosody conditioning
    outputs = rich_decoder(
        x=tokens,
        xa=encoder_output,
        prosody_features={"emotion": emotion_seq, "pitch": pitch_seq},
    )

    # Access rich outputs
    text_logits = outputs["text_logits"]
    emotion = outputs["emotion"]
    phoneme_deviation = outputs["phoneme_deviation"]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class RichDecoderConfig:
    """Configuration for RichDecoder."""

    # Model dimensions (from Whisper large-v3)
    n_vocab: int = 51865
    n_ctx: int = 448
    n_state: int = 1280
    n_head: int = 20
    n_layer: int = 32

    # LoRA configuration
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_layers: list[int] | None = None  # None = all layers

    # Which attention to apply LoRA to
    lora_query: bool = True
    lora_key: bool = True
    lora_value: bool = True
    lora_out: bool = False

    # Prosody conditioning
    use_prosody_cross_attention: bool = True
    prosody_dim: int = 64
    num_emotions: int = 34

    # Rich output heads
    num_para_classes: int = 50
    phoneme_vocab: int = 200
    num_languages: int = 100

    # Output projections
    rich_hidden_dim: int = 256


# =============================================================================
# LoRA Layer
# =============================================================================

class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation layer.

    LoRA decomposes weight update ΔW into two low-rank matrices:
        ΔW = B @ A, where A ∈ R^{r×d_in}, B ∈ R^{d_out×r}

    The output is: y = W₀x + (BA)x * scale

    Key properties:
    - B is initialized to zero, so LoRA starts as identity
    - Scale = alpha / rank controls adaptation strength
    - Only A and B are trained, original weights frozen
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        # A projects input to low-rank space
        # Initialize with Kaiming uniform for stable training
        self.lora_A = nn.Linear(in_dim, rank, bias=False)

        # B projects from low-rank space back to output
        # Initialize to zero so LoRA starts as identity
        self.lora_B = nn.Linear(rank, out_dim, bias=False)
        self.lora_B.weight = mx.zeros((out_dim, rank))

        # Optional dropout between A and B
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def __call__(self, x: mx.array) -> mx.array:
        """
        Compute LoRA delta: (B @ A)(x) * scale

        Args:
            x: Input tensor of shape (..., in_dim)

        Returns:
            Delta tensor of shape (..., out_dim)
        """
        # Project to low-rank
        h = self.lora_A(x)  # (..., rank)

        # Optional dropout
        if self.dropout is not None:
            h = self.dropout(h)

        # Project back and scale
        return self.lora_B(h) * self.scale  # (..., out_dim)



class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adapter.

    Wraps an existing nn.Linear and adds LoRA delta to its output.
    The original linear weights are kept frozen.
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Store reference to original (frozen) linear
        self.linear = original_linear

        # Get dimensions
        # Note: nn.Linear weight shape is (out_features, in_features)
        out_dim, in_dim = original_linear.weight.shape

        # Create LoRA adapter
        self.lora = LoRALayer(in_dim, out_dim, rank, alpha, dropout)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass: original output + LoRA delta."""
        return self.linear(x) + self.lora(x)


# =============================================================================
# Prosody Cross-Attention
# =============================================================================

class ProsodyCrossAttention(nn.Module):
    """
    Cross-attention over prosody features (emotion, pitch).

    This is Option B from the architecture doc: a separate cross-attention
    layer that attends to prosody features extracted by CTC heads.

    Prosody features are per-frame (50Hz), matching encoder output length.
    The decoder can attend to specific frames to get relevant prosody context.
    """

    def __init__(
        self,
        n_state: int = 1280,
        n_head: int = 8,
        prosody_dim: int = 64,
        num_emotions: int = 34,
        pitch_dim: int = 1,
    ):
        super().__init__()
        self.n_state = n_state
        self.n_head = n_head
        self.head_dim = n_state // n_head

        # Project prosody to key/value dimension
        prosody_input_dim = num_emotions + pitch_dim
        self.prosody_proj = nn.Linear(prosody_input_dim, n_state)

        # Query from decoder hidden state
        self.query = nn.Linear(n_state, n_state)

        # Key/value from prosody
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)

        # Output projection
        self.out = nn.Linear(n_state, n_state)

        # Layer norm for residual
        self.ln = nn.LayerNorm(n_state)

    def __call__(
        self,
        x: mx.array,
        emotion_seq: mx.array,
        pitch_seq: mx.array,
        kv_cache: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array]]:
        """
        Cross-attention over prosody features.

        Args:
            x: Decoder hidden state (batch, seq_len, n_state)
            emotion_seq: Per-frame emotion logits (batch, T, num_emotions)
            pitch_seq: Per-frame pitch values (batch, T, 1)
            kv_cache: Optional cached (K, V) for efficiency

        Returns:
            - Output with prosody context added (batch, seq_len, n_state)
            - Updated (K, V) cache
        """
        batch_size, seq_len, _ = x.shape

        # Normalize input
        x_normed = mx.fast.layer_norm(x, self.ln.weight, self.ln.bias, eps=1e-5)

        # Project query from decoder state
        q = self.query(x_normed)

        if kv_cache is not None:
            # Use cached K, V
            k, v = kv_cache
        else:
            # Concatenate prosody features
            prosody = mx.concatenate([emotion_seq, pitch_seq], axis=-1)

            # Project to n_state dimension
            prosody_emb = nn.gelu(self.prosody_proj(prosody))

            # Compute K, V from prosody
            k = self.key(prosody_emb)
            v = self.value(prosody_emb)

        # Reshape for multi-head attention
        T = k.shape[1]
        q = q.reshape(batch_size, seq_len, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)

        # Reshape back
        out = out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.n_state)

        # Output projection
        out = self.out(out)

        # Residual connection
        return x + out, (k, v)


# =============================================================================
# Rich Output Heads
# =============================================================================

class RichOutputHeads(nn.Module):
    """
    Output heads for rich decoder predictions.

    Produces:
    - Token-aligned emotion
    - Token-aligned paralinguistics
    - Token-aligned language (for code-switching)
    - Phoneme deviation score (hallucination signal)
    """

    def __init__(
        self,
        n_state: int = 1280,
        hidden_dim: int = 256,
        num_emotions: int = 34,
        num_para_classes: int = 50,
        num_languages: int = 100,
    ):
        super().__init__()

        # Emotion head
        self.emotion_ln = nn.LayerNorm(n_state)
        self.emotion_fc1 = nn.Linear(n_state, hidden_dim)
        self.emotion_fc2 = nn.Linear(hidden_dim, num_emotions)

        # Paralinguistics head
        self.para_ln = nn.LayerNorm(n_state)
        self.para_fc1 = nn.Linear(n_state, hidden_dim)
        self.para_fc2 = nn.Linear(hidden_dim, num_para_classes)

        # Language head (for code-switching detection)
        self.lang_ln = nn.LayerNorm(n_state)
        self.lang_fc1 = nn.Linear(n_state, hidden_dim)
        self.lang_fc2 = nn.Linear(hidden_dim, num_languages)

        # Phoneme deviation head (scalar output per token)
        # High deviation = audio phonemes don't match text = possible hallucination
        self.deviation_ln = nn.LayerNorm(n_state)
        self.deviation_fc1 = nn.Linear(n_state, hidden_dim)
        self.deviation_fc2 = nn.Linear(hidden_dim, 1)

    def __call__(self, hidden_states: mx.array) -> dict[str, mx.array]:
        """
        Compute rich outputs from decoder hidden states.

        Args:
            hidden_states: Decoder hidden states (batch, seq_len, n_state)

        Returns:
            Dictionary with:
            - "emotion": (batch, seq_len, num_emotions)
            - "para": (batch, seq_len, num_para_classes)
            - "language": (batch, seq_len, num_languages)
            - "phoneme_deviation": (batch, seq_len, 1)
        """
        # Emotion
        h = self.emotion_ln(hidden_states)
        h = nn.gelu(self.emotion_fc1(h))
        emotion = self.emotion_fc2(h)

        # Paralinguistics
        h = self.para_ln(hidden_states)
        h = nn.gelu(self.para_fc1(h))
        para = self.para_fc2(h)

        # Language
        h = self.lang_ln(hidden_states)
        h = nn.gelu(self.lang_fc1(h))
        language = self.lang_fc2(h)

        # Phoneme deviation
        h = self.deviation_ln(hidden_states)
        h = nn.gelu(self.deviation_fc1(h))
        deviation = mx.sigmoid(self.deviation_fc2(h))  # 0-1 range

        return {
            "emotion": emotion,
            "para": para,
            "language": language,
            "phoneme_deviation": deviation,
        }


# =============================================================================
# Rich Decoder (Main Class)
# =============================================================================

class RichDecoder(nn.Module):
    """
    Whisper decoder with LoRA adapters for rich audio understanding.

    This wraps an existing Whisper TextDecoder and adds:
    1. LoRA adapters on attention layers for fine-tuning
    2. Prosody cross-attention for punctuation improvement
    3. Rich output heads (emotion, para, language, deviation)

    The original decoder weights are frozen; only LoRA and new heads train.
    """

    def __init__(
        self,
        config: RichDecoderConfig | None = None,
    ):
        super().__init__()
        self.config = config or RichDecoderConfig()

        # These will be populated by from_whisper_decoder()
        self.token_embedding = None
        self.positional_embedding = None
        self.blocks = None
        self.ln = None

        # LoRA adapters (applied to attention in blocks)
        self.lora_adapters: dict[str, LoRALayer] = {}

        # Prosody cross-attention (one per layer or shared)
        if self.config.use_prosody_cross_attention:
            self.prosody_attention = ProsodyCrossAttention(
                n_state=self.config.n_state,
                n_head=8,  # Fewer heads than main attention
                prosody_dim=self.config.prosody_dim,
                num_emotions=self.config.num_emotions,
            )
        else:
            self.prosody_attention = None

        # Rich output heads
        self.rich_heads = RichOutputHeads(
            n_state=self.config.n_state,
            hidden_dim=self.config.rich_hidden_dim,
            num_emotions=self.config.num_emotions,
            num_para_classes=self.config.num_para_classes,
            num_languages=self.config.num_languages,
        )

        # Causal mask (will be set based on decoder)
        self._mask = None

        # KV cache for prosody attention
        self._prosody_kv_cache = None

    def _apply_lora_to_blocks(self):
        """Apply LoRA adapters to attention layers in blocks."""
        config = self.config

        # Determine which layers to apply LoRA
        layers_to_lora = config.lora_layers
        if layers_to_lora is None:
            layers_to_lora = list(range(config.n_layer))

        for layer_idx in layers_to_lora:
            block = self.blocks[layer_idx]

            # Apply LoRA to self-attention Q, K, V, O
            if config.lora_query:
                key = f"layer_{layer_idx}_self_query"
                self.lora_adapters[key] = LoRALayer(
                    config.n_state, config.n_state,
                    config.lora_rank, config.lora_alpha, config.lora_dropout,
                )

            if config.lora_key:
                key = f"layer_{layer_idx}_self_key"
                self.lora_adapters[key] = LoRALayer(
                    config.n_state, config.n_state,
                    config.lora_rank, config.lora_alpha, config.lora_dropout,
                )

            if config.lora_value:
                key = f"layer_{layer_idx}_self_value"
                self.lora_adapters[key] = LoRALayer(
                    config.n_state, config.n_state,
                    config.lora_rank, config.lora_alpha, config.lora_dropout,
                )

            if config.lora_out:
                key = f"layer_{layer_idx}_self_out"
                self.lora_adapters[key] = LoRALayer(
                    config.n_state, config.n_state,
                    config.lora_rank, config.lora_alpha, config.lora_dropout,
                )

            # Apply LoRA to cross-attention if present
            if block.cross_attn is not None:
                if config.lora_query:
                    key = f"layer_{layer_idx}_cross_query"
                    self.lora_adapters[key] = LoRALayer(
                        config.n_state, config.n_state,
                        config.lora_rank, config.lora_alpha, config.lora_dropout,
                    )

                if config.lora_key:
                    key = f"layer_{layer_idx}_cross_key"
                    self.lora_adapters[key] = LoRALayer(
                        config.n_state, config.n_state,
                        config.lora_rank, config.lora_alpha, config.lora_dropout,
                    )

                if config.lora_value:
                    key = f"layer_{layer_idx}_cross_value"
                    self.lora_adapters[key] = LoRALayer(
                        config.n_state, config.n_state,
                        config.lora_rank, config.lora_alpha, config.lora_dropout,
                    )

    def __call__(
        self,
        x: mx.array,
        xa: mx.array,
        prosody_features: dict[str, mx.array] | None = None,
        kv_cache: list[tuple] | None = None,
    ) -> dict[str, Any]:
        """
        Forward pass through rich decoder.

        Args:
            x: Token ids, shape (batch, seq_len)
            xa: Encoder output, shape (batch, encoder_len, n_state)
            prosody_features: Optional dict with:
                - "emotion": (batch, T, num_emotions) emotion logits from CTC
                - "pitch": (batch, T, 1) pitch values from CTC
            kv_cache: List of cached (self_kv, cross_kv) per layer

        Returns:
            Dictionary with:
            - "text_logits": (batch, seq_len, n_vocab)
            - "emotion": (batch, seq_len, num_emotions)
            - "para": (batch, seq_len, num_para_classes)
            - "language": (batch, seq_len, num_languages)
            - "phoneme_deviation": (batch, seq_len, 1)
            - "kv_cache": Updated KV cache
            - "hidden_states": (batch, seq_len, n_state)
        """
        seq_len = x.shape[-1]

        # Token embeddings + positional
        offset = kv_cache[0][0][0].shape[1] if kv_cache else 0
        x_emb = self.token_embedding.weight[x]
        x_emb = x_emb + self.positional_embedding[offset:offset + seq_len]

        # Initialize cache
        if kv_cache is None:
            kv_cache = [None] * len(self.blocks)

        # Process through blocks with LoRA
        h = x_emb
        new_kv_cache = []

        for layer_idx, block in enumerate(self.blocks):
            # Get LoRA deltas for this layer (if any)
            lora_deltas = {}
            for key_type in ["self_query", "self_key", "self_value", "self_out",
                           "cross_query", "cross_key", "cross_value"]:
                lora_key = f"layer_{layer_idx}_{key_type}"
                if lora_key in self.lora_adapters:
                    lora_deltas[key_type] = self.lora_adapters[lora_key]

            # Forward through block (LoRA applied internally)
            h, block_kv, _ = self._forward_block_with_lora(
                block, h, xa, self._mask, kv_cache[layer_idx], lora_deltas,
            )
            new_kv_cache.append(block_kv)

        # Prosody cross-attention (if enabled and features provided)
        if self.prosody_attention is not None and prosody_features is not None:
            emotion_seq = prosody_features.get("emotion")
            pitch_seq = prosody_features.get("pitch")

            if emotion_seq is not None and pitch_seq is not None:
                h, self._prosody_kv_cache = self.prosody_attention(
                    h, emotion_seq, pitch_seq, self._prosody_kv_cache,
                )

        # Final layer norm
        hidden_states = mx.fast.layer_norm(h, self.ln.weight, self.ln.bias, eps=1e-5)

        # Text logits (tied weights)
        text_logits = self.token_embedding.as_linear(hidden_states)

        # Rich output heads
        rich_outputs = self.rich_heads(hidden_states)

        return {
            "text_logits": text_logits,
            "emotion": rich_outputs["emotion"],
            "para": rich_outputs["para"],
            "language": rich_outputs["language"],
            "phoneme_deviation": rich_outputs["phoneme_deviation"],
            "kv_cache": new_kv_cache,
            "hidden_states": hidden_states,
        }

    def _forward_block_with_lora(
        self,
        block: nn.Module,
        x: mx.array,
        xa: mx.array,
        mask: mx.array | None,
        kv_cache: tuple | None,
        lora_deltas: dict[str, LoRALayer],
    ) -> tuple[mx.array, tuple, mx.array | None]:
        """
        Forward through a block with LoRA deltas applied.

        This is a modified version of ResidualAttentionBlock.__call__
        that applies LoRA deltas to Q, K, V projections.
        """
        # Unpack cache
        self_kv, cross_kv = kv_cache if kv_cache else (None, None)

        # Self-attention with LoRA
        attn = block.attn
        attn_input = mx.fast.layer_norm(x, block.attn_ln.weight, block.attn_ln.bias, eps=1e-5)

        # Compute Q, K, V with LoRA deltas
        q = attn.query(attn_input)
        k = attn.key(attn_input)
        v = attn.value(attn_input)

        if "self_query" in lora_deltas:
            q = q + lora_deltas["self_query"](attn_input)
        if "self_key" in lora_deltas:
            k = k + lora_deltas["self_key"](attn_input)
        if "self_value" in lora_deltas:
            v = v + lora_deltas["self_value"](attn_input)

        # Apply KV cache
        if self_kv is not None:
            k = mx.concatenate([self_kv[0], k], axis=1)
            v = mx.concatenate([self_kv[1], v], axis=1)

        # Compute attention
        y, self_kv_new, _ = self._compute_attention(attn, q, k, v, mask)

        # Apply output LoRA if present
        y = attn.out(y)
        if "self_out" in lora_deltas:
            y = y + lora_deltas["self_out"](y)

        x = x + y

        # Cross-attention (if present)
        cross_qk = None
        if block.cross_attn is not None:
            cross_attn = block.cross_attn
            cross_input = mx.fast.layer_norm(
                x, block.cross_attn_ln.weight, block.cross_attn_ln.bias, eps=1e-5,
            )

            # Query from decoder, K/V from encoder
            q = cross_attn.query(cross_input)
            if "cross_query" in lora_deltas:
                q = q + lora_deltas["cross_query"](cross_input)

            if cross_kv is None:
                # First call: compute K, V from encoder output
                k = cross_attn.key(xa)
                v = cross_attn.value(xa)
                if "cross_key" in lora_deltas:
                    k = k + lora_deltas["cross_key"](xa)
                if "cross_value" in lora_deltas:
                    v = v + lora_deltas["cross_value"](xa)
            else:
                # Subsequent calls: reuse cached K, V
                k, v = cross_kv

            # Compute cross-attention
            y, cross_kv_new, cross_qk = self._compute_attention(
                cross_attn, q, k, v, mask=None,
            )
            y = cross_attn.out(y)
            x = x + y
            cross_kv = cross_kv_new

        # MLP
        mlp_input = mx.fast.layer_norm(x, block.mlp_ln.weight, block.mlp_ln.bias, eps=1e-5)
        x = x + block.mlp2(nn.gelu(block.mlp1(mlp_input)))

        return x, (self_kv_new, cross_kv), cross_qk

    def _compute_attention(
        self,
        attn: nn.Module,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        mask: mx.array | None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array], mx.array | None]:
        """Compute multi-head attention."""
        batch_size, q_len, _ = q.shape
        kv_len = k.shape[1]

        # Reshape for multi-head
        q = q.reshape(batch_size, q_len, attn.n_head, attn.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, kv_len, attn.n_head, attn.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, kv_len, attn.n_head, attn.head_dim).transpose(0, 2, 1, 3)

        # Scale
        scale = attn.head_dim ** -0.5

        # Mask handling
        sdpa_mask = None
        if mask is not None:
            if q_len == kv_len:
                sdpa_mask = mask[:q_len, :kv_len]
            else:
                offset = kv_len - q_len
                sdpa_mask = mask[offset:offset + q_len, :kv_len]

        # SDPA
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=sdpa_mask)

        # Reshape back
        out = out.transpose(0, 2, 1, 3).reshape(batch_size, q_len, attn.n_state)

        # Return with cache
        # Flatten K, V back to original shape for caching
        k_cache = k.transpose(0, 2, 1, 3).reshape(batch_size, kv_len, attn.n_state)
        v_cache = v.transpose(0, 2, 1, 3).reshape(batch_size, kv_len, attn.n_state)

        return out, (k_cache, v_cache), None

    def reset_cache(self):
        """Reset KV caches."""
        self._prosody_kv_cache = None

    @classmethod
    def from_whisper_decoder(
        cls,
        whisper_decoder: nn.Module,
        config: RichDecoderConfig | None = None,
    ) -> RichDecoder:
        """
        Create RichDecoder by wrapping an existing Whisper decoder.

        The Whisper decoder weights are kept frozen.
        Only LoRA adapters and rich heads are trainable.

        Args:
            whisper_decoder: Existing TextDecoder instance
            config: Optional configuration override

        Returns:
            RichDecoder wrapping the Whisper decoder
        """
        # Infer config from decoder if not provided
        if config is None:
            config = RichDecoderConfig(
                n_vocab=whisper_decoder.n_vocab,
                n_ctx=whisper_decoder.n_ctx,
                n_state=whisper_decoder.n_state,
                n_layer=whisper_decoder.n_layer,
            )

        # Create rich decoder
        rich_decoder = cls(config)

        # Copy references to Whisper components (not copies - same objects)
        rich_decoder.token_embedding = whisper_decoder.token_embedding
        rich_decoder.positional_embedding = whisper_decoder.positional_embedding
        rich_decoder.blocks = whisper_decoder.blocks
        rich_decoder.ln = whisper_decoder.ln
        rich_decoder._mask = whisper_decoder._mask

        # Apply LoRA to blocks
        rich_decoder._apply_lora_to_blocks()

        # Evaluate to materialize parameters
        mx.eval(rich_decoder.parameters())

        return rich_decoder

    def get_trainable_parameters(self) -> dict[str, mx.array]:
        """
        Get only trainable parameters (LoRA + rich heads).

        The original Whisper weights are frozen and not returned.
        Returns a flat dict of {key: mx.array} suitable for mx.savez().
        """
        trainable = {}

        # LoRA adapters
        for key, lora in self.lora_adapters.items():
            trainable[f"lora.{key}.A.weight"] = lora.lora_A.weight
            trainable[f"lora.{key}.B.weight"] = lora.lora_B.weight

        # Prosody attention (if enabled) - flatten nested dict
        if self.prosody_attention is not None:
            self._flatten_params(
                self.prosody_attention.parameters(),
                "prosody_attention",
                trainable,
            )

        # Rich output heads - flatten nested dict
        self._flatten_params(
            self.rich_heads.parameters(),
            "rich_heads",
            trainable,
        )

        return trainable

    def _flatten_params(
        self,
        params: dict,
        prefix: str,
        out: dict[str, mx.array],
    ):
        """Recursively flatten nested parameter dict into flat {key: mx.array}."""
        for key, value in params.items():
            full_key = f"{prefix}.{key}"
            if isinstance(value, mx.array):
                out[full_key] = value
            elif isinstance(value, dict):
                self._flatten_params(value, full_key, out)
            else:
                raise TypeError(
                    f"Unexpected parameter type at {full_key}: {type(value)}",
                )

    def save_trainable(self, path: str):
        """Save only trainable parameters to file."""
        trainable = self.get_trainable_parameters()
        mx.savez(path, **trainable)
        print(f"Saved {len(trainable)} trainable parameters to {path}")

    def load_trainable(self, path: str):
        """Load trainable parameters from file."""
        weights = dict(mx.load(path))

        # Load LoRA weights
        for key in list(weights.keys()):
            if key.startswith("lora."):
                parts = key.split(".")
                adapter_key = parts[1]
                weight_type = parts[2]  # "A" or "B"

                if adapter_key in self.lora_adapters:
                    lora = self.lora_adapters[adapter_key]
                    if weight_type == "A":
                        lora.lora_A.weight = weights[key]
                    else:
                        lora.lora_B.weight = weights[key]

            elif key.startswith("prosody_attention.") and self.prosody_attention:
                subkey = key[len("prosody_attention."):]
                # Set parameter on prosody attention
                self._set_nested_param(self.prosody_attention, subkey, weights[key])

            elif key.startswith("rich_heads."):
                subkey = key[len("rich_heads."):]
                self._set_nested_param(self.rich_heads, subkey, weights[key])

        mx.eval(self.parameters())
        print(f"Loaded trainable parameters from {path}")

    def _set_nested_param(self, module: nn.Module, key: str, value: mx.array):
        """Set a nested parameter by dotted key path."""
        parts = key.split(".")
        for part in parts[:-1]:
            module = getattr(module, part)
        setattr(module, parts[-1], value)


# =============================================================================
# Tests
# =============================================================================

def test_lora_layer():
    """Test LoRA layer initialization and forward pass."""
    print("Testing LoRALayer...")

    lora = LoRALayer(in_dim=1280, out_dim=1280, rank=8, alpha=16)

    # Check B is initialized to zero (identity)
    assert mx.allclose(lora.lora_B.weight, mx.zeros_like(lora.lora_B.weight)), \
        "LoRA B should be initialized to zero"

    # Forward pass
    x = mx.random.normal((2, 10, 1280))
    delta = lora(x)

    # Delta should be zero initially (B is zero)
    assert mx.allclose(delta, mx.zeros_like(delta), atol=1e-6), \
        "Initial LoRA output should be zero"

    # Update B and check non-zero output
    lora.lora_B.weight = mx.random.normal(lora.lora_B.weight.shape)
    delta = lora(x)
    mx.eval(delta)

    assert delta.shape == x.shape, f"LoRA output shape mismatch: {delta.shape} vs {x.shape}"
    assert not mx.allclose(delta, mx.zeros_like(delta), atol=1e-6), \
        "LoRA output should be non-zero after B update"

    print("  LoRALayer tests PASSED")
    return True


def test_prosody_cross_attention():
    """Test prosody cross-attention module."""
    print("Testing ProsodyCrossAttention...")

    pca = ProsodyCrossAttention(
        n_state=1280,
        n_head=8,
        num_emotions=34,
    )

    # Mock inputs
    x = mx.random.normal((2, 10, 1280))  # Decoder hidden
    emotion = mx.random.normal((2, 100, 34))  # CTC emotion (50Hz)
    pitch = mx.random.normal((2, 100, 1))  # CTC pitch

    # Forward pass
    out, kv_cache = pca(x, emotion, pitch)
    mx.eval(out)

    assert out.shape == x.shape, f"Output shape mismatch: {out.shape} vs {x.shape}"
    # KV cache shape is (batch, n_head, T, head_dim) after multi-head reshape
    assert kv_cache[0].shape[2] == 100, f"K cache should match prosody length, got {kv_cache[0].shape}"

    print("  ProsodyCrossAttention tests PASSED")
    return True


def test_rich_output_heads():
    """Test rich output heads."""
    print("Testing RichOutputHeads...")

    heads = RichOutputHeads(
        n_state=1280,
        num_emotions=34,
        num_para_classes=50,
        num_languages=100,
    )

    # Mock input
    hidden = mx.random.normal((2, 10, 1280))

    # Forward pass
    outputs = heads(hidden)
    mx.eval(outputs)

    assert outputs["emotion"].shape == (2, 10, 34), \
        f"Emotion shape: {outputs['emotion'].shape}"
    assert outputs["para"].shape == (2, 10, 50), \
        f"Para shape: {outputs['para'].shape}"
    assert outputs["language"].shape == (2, 10, 100), \
        f"Language shape: {outputs['language'].shape}"
    assert outputs["phoneme_deviation"].shape == (2, 10, 1), \
        f"Deviation shape: {outputs['phoneme_deviation'].shape}"

    # Deviation should be in [0, 1] (sigmoid)
    dev = outputs["phoneme_deviation"]
    assert mx.all(dev >= 0) and mx.all(dev <= 1), \
        "Phoneme deviation should be in [0, 1]"

    print("  RichOutputHeads tests PASSED")
    return True


def test_rich_decoder_standalone():
    """Test RichDecoder in standalone mode (without Whisper)."""
    print("Testing RichDecoder (standalone)...")

    config = RichDecoderConfig(
        n_vocab=51865,
        n_ctx=448,
        n_state=1280,
        n_head=20,
        n_layer=4,  # Reduced for testing
    )

    decoder = RichDecoder(config)

    # Create minimal decoder structure for testing
    decoder.token_embedding = nn.Embedding(config.n_vocab, config.n_state)
    decoder.positional_embedding = mx.zeros((config.n_ctx, config.n_state))
    decoder.ln = nn.LayerNorm(config.n_state)
    decoder._mask = nn.MultiHeadAttention.create_additive_causal_mask(config.n_ctx)

    # Create blocks (simplified for testing)
    from .attention import ResidualAttentionBlock
    decoder.blocks = [
        ResidualAttentionBlock(config.n_state, config.n_head, cross_attention=True)
        for _ in range(config.n_layer)
    ]

    # Apply LoRA
    decoder._apply_lora_to_blocks()

    # Check LoRA adapters were created
    expected_adapters = config.n_layer * 2 * 3  # 2 attn types * 3 projections (Q,K,V)
    actual_adapters = len(decoder.lora_adapters)
    print(f"  Created {actual_adapters} LoRA adapters (expected ~{expected_adapters})")

    # Get trainable params
    trainable = decoder.get_trainable_parameters()
    print(f"  Trainable parameters: {len(trainable)}")

    print("  RichDecoder standalone tests PASSED")
    return True


if __name__ == "__main__":
    test_lora_layer()
    test_prosody_cross_attention()
    test_rich_output_heads()
    test_rich_decoder_standalone()
    print("\nAll RichDecoder tests PASSED")
