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
MLX Implementation of NLLB-200 (M2M-100 architecture)

Encoder-decoder transformer for multilingual translation.
Based on HuggingFace transformers M2M100 implementation.
"""

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class NLLBConfig:
    """Configuration for NLLB model."""

    vocab_size: int = 256206
    d_model: int = 1024
    encoder_layers: int = 12
    decoder_layers: int = 12
    encoder_attention_heads: int = 16
    decoder_attention_heads: int = 16
    encoder_ffn_dim: int = 4096
    decoder_ffn_dim: int = 4096
    activation_function: str = "relu"
    dropout: float = 0.1
    attention_dropout: float = 0.0
    activation_dropout: float = 0.0
    max_position_embeddings: int = 1024
    scale_embedding: bool = True
    pad_token_id: int = 1
    bos_token_id: int = 0
    eos_token_id: int = 2
    decoder_start_token_id: int = 2
    layer_norm_eps: float = 1e-5

    @classmethod
    def from_hf(cls, hf_config) -> "NLLBConfig":
        """Create config from HuggingFace M2M100Config."""
        return cls(
            vocab_size=hf_config.vocab_size,
            d_model=hf_config.d_model,
            encoder_layers=hf_config.encoder_layers,
            decoder_layers=hf_config.decoder_layers,
            encoder_attention_heads=hf_config.encoder_attention_heads,
            decoder_attention_heads=hf_config.decoder_attention_heads,
            encoder_ffn_dim=hf_config.encoder_ffn_dim,
            decoder_ffn_dim=hf_config.decoder_ffn_dim,
            activation_function=hf_config.activation_function,
            dropout=hf_config.dropout,
            attention_dropout=hf_config.attention_dropout,
            activation_dropout=hf_config.activation_dropout,
            max_position_embeddings=hf_config.max_position_embeddings,
            scale_embedding=hf_config.scale_embedding,
            pad_token_id=hf_config.pad_token_id,
            bos_token_id=hf_config.bos_token_id,
            eos_token_id=hf_config.eos_token_id,
            decoder_start_token_id=hf_config.decoder_start_token_id,
        )


class NLLBAttention(nn.Module):
    """
    Multi-head attention for NLLB.

    Can be used for:
    - Self-attention (encoder and decoder)
    - Cross-attention (decoder attending to encoder)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,  # noqa: ARG002 - API parameter for consistency
    ):
        super().__init__()
        # Note: is_decoder kept for API consistency; causal masking handled via mask arg
        del is_decoder  # Explicitly mark as unused
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def __call__(
        self,
        hidden_states: mx.array,
        key_value_states: mx.array | None = None,
        attention_mask: mx.array | None = None,
        cache: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array] | None]:
        """
        Forward pass for attention.

        Args:
            hidden_states: [batch, seq_len, embed_dim]
            key_value_states: For cross-attention, encoder hidden states
            attention_mask: Attention mask
            cache: KV cache for generation

        Returns:
            output: [batch, seq_len, embed_dim]
            cache: Updated KV cache
        """
        batch_size, seq_len, _ = hidden_states.shape

        is_cross_attention = key_value_states is not None

        # Project queries
        queries = self.q_proj(hidden_states)

        # For cross-attention, use encoder states for K and V
        if is_cross_attention:
            assert key_value_states is not None
            keys = self.k_proj(key_value_states)
            values = self.v_proj(key_value_states)
        else:
            keys = self.k_proj(hidden_states)
            values = self.v_proj(hidden_states)

        # Handle KV cache
        if cache is not None:
            past_key, past_value = cache
            if is_cross_attention:
                # Cross-attention cache doesn't grow
                keys = past_key
                values = past_value
            else:
                # Self-attention cache grows
                keys = mx.concatenate([past_key, keys], axis=1)
                values = mx.concatenate([past_value, values], axis=1)

        # Always return cache for generation (not just when cache was provided)
        new_cache = (keys, values)

        # Reshape for multi-head attention
        # [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
        queries = queries.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        queries = queries.transpose(0, 2, 1, 3)

        kv_seq_len = keys.shape[1]
        keys = keys.reshape(batch_size, kv_seq_len, self.num_heads, self.head_dim)
        keys = keys.transpose(0, 2, 1, 3)

        values = values.reshape(batch_size, kv_seq_len, self.num_heads, self.head_dim)
        values = values.transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        scores = (queries @ keys.transpose(0, 1, 3, 2)) * self.scale

        if attention_mask is not None:
            scores = scores + attention_mask

        weights = mx.softmax(scores, axis=-1)

        # Apply attention to values
        output = weights @ values

        # Reshape back
        output = output.transpose(0, 2, 1, 3)
        output = output.reshape(batch_size, seq_len, self.embed_dim)

        # Output projection
        output = self.out_proj(output)

        return output, new_cache


class NLLBEncoderLayer(nn.Module):
    """Single encoder layer with self-attention and FFN."""

    def __init__(self, config: NLLBConfig):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = NLLBAttention(
            embed_dim=config.d_model,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(
            config.d_model, eps=config.layer_norm_eps,
        )

        self.fc1 = nn.Linear(config.d_model, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, config.d_model)
        self.final_layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

        self.activation = nn.relu if config.activation_function == "relu" else nn.gelu

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        """
        Forward pass for encoder layer.

        Args:
            hidden_states: [batch, seq_len, embed_dim]
            attention_mask: Optional attention mask

        Returns:
            hidden_states: [batch, seq_len, embed_dim]
        """
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = residual + hidden_states

        # FFN with residual
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return residual + hidden_states



class NLLBDecoderLayer(nn.Module):
    """Single decoder layer with self-attention, cross-attention, and FFN."""

    def __init__(self, config: NLLBConfig):
        super().__init__()
        self.embed_dim = config.d_model

        # Self-attention
        self.self_attn = NLLBAttention(
            embed_dim=config.d_model,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.self_attn_layer_norm = nn.LayerNorm(
            config.d_model, eps=config.layer_norm_eps,
        )

        # Cross-attention (encoder-decoder attention)
        self.encoder_attn = NLLBAttention(
            embed_dim=config.d_model,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(
            config.d_model, eps=config.layer_norm_eps,
        )

        # FFN
        self.fc1 = nn.Linear(config.d_model, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, config.d_model)
        self.final_layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

        self.activation = nn.relu if config.activation_function == "relu" else nn.gelu

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        attention_mask: mx.array | None = None,
        encoder_attention_mask: mx.array | None = None,
        cache: tuple[tuple[mx.array, mx.array], tuple[mx.array, mx.array]] | None = None,
    ) -> tuple[mx.array, tuple | None]:
        """
        Forward pass for decoder layer.

        Args:
            hidden_states: [batch, seq_len, embed_dim]
            encoder_hidden_states: [batch, src_len, embed_dim]
            attention_mask: Causal mask for self-attention
            encoder_attention_mask: Mask for cross-attention
            cache: (self_attn_cache, cross_attn_cache)

        Returns:
            hidden_states: [batch, seq_len, embed_dim]
            cache: Updated caches
        """
        self_attn_cache = cache[0] if cache is not None else None
        cross_attn_cache = cache[1] if cache is not None else None

        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, self_attn_cache = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            cache=self_attn_cache,
        )
        hidden_states = residual + hidden_states

        # Cross-attention with residual
        residual = hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)
        hidden_states, cross_attn_cache = self.encoder_attn(
            hidden_states,
            key_value_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            cache=cross_attn_cache,
        )
        hidden_states = residual + hidden_states

        # FFN with residual
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        # Always return cache for generation
        new_cache = (self_attn_cache, cross_attn_cache)

        return hidden_states, new_cache


def create_sinusoidal_positions(
    num_positions: int, d_model: int, padding_idx: int = 1,
) -> mx.array:
    """
    Create sinusoidal position embeddings matching HuggingFace M2M100.

    Uses the tensor2tensor formula:
    - First half of embedding is sin, second half is cos
    - Positions 0 and 1 (padding) are zeroed out
    - Position IDs start at padding_idx + 1 (usually 2)
    """
    half_dim = d_model // 2
    log_scale = math.log(10000) / (half_dim - 1)
    inv_freq = mx.exp(mx.arange(half_dim) * -log_scale)
    emb = mx.arange(num_positions)[:, None] * inv_freq[None, :]
    emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=1)

    # Zero out padding positions (0 and 1 for padding_idx=1)
    # The first padding_idx + 1 positions are zeroed
    if padding_idx is not None:
        # Create a mask for non-padding positions
        mask = mx.arange(num_positions) > padding_idx
        emb = emb * mask[:, None]

    return emb  # type: ignore[no-any-return]


class NLLBEncoder(nn.Module):
    """NLLB Encoder stack."""

    def __init__(self, config: NLLBConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id

        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        # Sinusoidal position embeddings (not learned)
        # Pre-compute for max_position_embeddings + offset
        self._position_embeddings: mx.array | None = None  # Lazy init

        self.layers = [NLLBEncoderLayer(config) for _ in range(config.encoder_layers)]
        self.layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

    def _get_position_embeddings(self, max_pos: int) -> mx.array:
        """Get or create sinusoidal position embeddings."""
        # Offset=2 in HF implementation, so we need max_pos + 2
        required_size = max_pos + 2
        if (
            self._position_embeddings is None
            or self._position_embeddings.shape[0] < required_size
        ):
            self._position_embeddings = create_sinusoidal_positions(
                max(required_size, self.config.max_position_embeddings + 2),
                self.config.d_model,
                padding_idx=self.padding_idx,
            )
        assert self._position_embeddings is not None
        return self._position_embeddings

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        """
        Forward pass for encoder.

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]

        Returns:
            hidden_states: [batch, seq_len, embed_dim]
        """
        batch_size, seq_len = input_ids.shape

        # Token embeddings
        hidden_states = self.embed_tokens(input_ids) * self.embed_scale

        # Create position IDs matching HF: start at padding_idx + 1 for non-padding tokens
        # Simplified: assume all tokens are non-padding, use sequential IDs starting at 2
        position_ids = mx.arange(self.padding_idx + 1, self.padding_idx + 1 + seq_len)

        # Get position embeddings and index
        pos_emb = self._get_position_embeddings(self.padding_idx + seq_len + 1)
        position_embeddings = pos_emb[position_ids]

        hidden_states = hidden_states + position_embeddings

        # Prepare attention mask (convert to additive mask)
        if attention_mask is not None:
            # [batch, seq_len] -> [batch, 1, 1, seq_len]
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * -1e9

        # Encoder layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        return self.layer_norm(hidden_states)



class NLLBDecoder(nn.Module):
    """NLLB Decoder stack with KV-cache support."""

    def __init__(self, config: NLLBConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id

        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        # Sinusoidal position embeddings (not learned)
        self._position_embeddings: mx.array | None = None  # Lazy init

        self.layers = [NLLBDecoderLayer(config) for _ in range(config.decoder_layers)]
        self.layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

    def _get_position_embeddings(self, max_pos: int) -> mx.array:
        """Get or create sinusoidal position embeddings."""
        required_size = max_pos + 2
        if (
            self._position_embeddings is None
            or self._position_embeddings.shape[0] < required_size
        ):
            self._position_embeddings = create_sinusoidal_positions(
                max(required_size, self.config.max_position_embeddings + 2),
                self.config.d_model,
                padding_idx=self.padding_idx,
            )
        assert self._position_embeddings is not None
        return self._position_embeddings

    def __call__(
        self,
        input_ids: mx.array,
        encoder_hidden_states: mx.array,
        attention_mask: mx.array | None = None,
        encoder_attention_mask: mx.array | None = None,
        cache: list[tuple | None] | None = None,
    ) -> tuple[mx.array, list[tuple | None]]:
        """
        Forward pass for decoder.

        Args:
            input_ids: [batch, tgt_len]
            encoder_hidden_states: [batch, src_len, embed_dim]
            attention_mask: Causal mask
            encoder_attention_mask: Source attention mask
            cache: List of layer caches

        Returns:
            hidden_states: [batch, tgt_len, embed_dim]
            cache: Updated caches
        """
        batch_size, seq_len = input_ids.shape

        # Determine position offset from cache
        past_length = 0
        if cache is not None and cache[0] is not None:
            past_key, _ = cache[0][0]  # First layer, self-attention cache
            past_length = past_key.shape[1]

        # Token embeddings
        hidden_states = self.embed_tokens(input_ids) * self.embed_scale

        # Create position IDs matching HF: start at padding_idx + 1 + past_length
        start_pos = self.padding_idx + 1 + past_length
        position_ids = mx.arange(start_pos, start_pos + seq_len)

        # Get position embeddings and index
        pos_emb = self._get_position_embeddings(start_pos + seq_len)
        position_embeddings = pos_emb[position_ids]

        hidden_states = hidden_states + position_embeddings

        # Create causal mask
        if attention_mask is None:
            # Causal mask: upper triangular
            mask = mx.triu(
                mx.full((seq_len, seq_len + past_length), -1e9), k=past_length + 1,
            )
            attention_mask = mask[None, None, :, :]

        # Prepare encoder attention mask
        if encoder_attention_mask is not None:
            encoder_attention_mask = encoder_attention_mask[:, None, None, :]
            encoder_attention_mask = (1.0 - encoder_attention_mask) * -1e9

        # Initialize cache if needed
        if cache is None:
            cache = [None] * len(self.layers)

        new_cache = []

        # Decoder layers
        for i, layer in enumerate(self.layers):
            hidden_states, layer_cache = layer(
                hidden_states,
                encoder_hidden_states,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                cache=cache[i],
            )
            new_cache.append(layer_cache)

        hidden_states = self.layer_norm(hidden_states)

        return hidden_states, new_cache


class NLLBModel(nn.Module):
    """
    Complete NLLB encoder-decoder model.

    Example:
        config = NLLBConfig()
        model = NLLBModel(config)

        # Encode source
        encoder_output = model.encode(input_ids)

        # Decode (with KV-cache for generation)
        logits, cache = model.decode(decoder_input_ids, encoder_output)
    """

    def __init__(self, config: NLLBConfig):
        super().__init__()
        self.config = config

        self.encoder = NLLBEncoder(config)
        self.decoder = NLLBDecoder(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def encode(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        """
        Encode source sequence.

        Args:
            input_ids: [batch, src_len]
            attention_mask: [batch, src_len]

        Returns:
            encoder_hidden_states: [batch, src_len, embed_dim]
        """
        return self.encoder(input_ids, attention_mask)

    def decode(
        self,
        decoder_input_ids: mx.array,
        encoder_hidden_states: mx.array,
        encoder_attention_mask: mx.array | None = None,
        cache: list[tuple | None] | None = None,
    ) -> tuple[mx.array, list[tuple | None]]:
        """
        Decode target sequence.

        Args:
            decoder_input_ids: [batch, tgt_len]
            encoder_hidden_states: [batch, src_len, embed_dim]
            encoder_attention_mask: [batch, src_len]
            cache: KV cache for efficient generation

        Returns:
            logits: [batch, tgt_len, vocab_size]
            cache: Updated KV cache
        """
        decoder_output, cache = self.decoder(
            decoder_input_ids,
            encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            cache=cache,
        )

        logits = self.lm_head(decoder_output)

        return logits, cache

    def __call__(
        self,
        input_ids: mx.array,
        decoder_input_ids: mx.array,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        """
        Full forward pass (for training).

        Args:
            input_ids: Source tokens [batch, src_len]
            decoder_input_ids: Target tokens [batch, tgt_len]
            attention_mask: Source mask [batch, src_len]

        Returns:
            logits: [batch, tgt_len, vocab_size]
        """
        encoder_hidden_states = self.encode(input_ids, attention_mask)
        logits, _ = self.decode(
            decoder_input_ids, encoder_hidden_states, attention_mask,
        )
        return logits

    @staticmethod
    def from_pretrained(path: str) -> "NLLBModel":
        """
        Load model from converted weights.

        Args:
            path: Path to MLX weights directory

        Returns:
            Loaded model
        """
        import json
        from pathlib import Path

        model_path = Path(path)

        # Load config
        with open(model_path / "config.json") as f:
            config_dict = json.load(f)

        config = NLLBConfig(
            vocab_size=config_dict.get("vocab_size", 256206),
            d_model=config_dict.get("d_model", 1024),
            encoder_layers=config_dict.get("encoder_layers", 12),
            decoder_layers=config_dict.get("decoder_layers", 12),
            encoder_attention_heads=config_dict.get("encoder_attention_heads", 16),
            decoder_attention_heads=config_dict.get("decoder_attention_heads", 16),
            encoder_ffn_dim=config_dict.get("encoder_ffn_dim", 4096),
            decoder_ffn_dim=config_dict.get("decoder_ffn_dim", 4096),
        )

        model = NLLBModel(config)

        # Load weights
        weights_path = model_path / "weights.safetensors"
        if weights_path.exists():
            weights = mx.load(str(weights_path))
        else:
            weights = mx.load(str(model_path / "weights.npz"))

        assert isinstance(weights, dict)
        model.load_weights(list(weights.items()))

        return model

    @staticmethod
    def from_hf(hf_path: str) -> "NLLBModel":
        """
        Load model with weights from HuggingFace.

        Args:
            hf_path: HuggingFace model path (e.g., "facebook/nllb-200-distilled-600M")

        Returns:
            NLLBModel with loaded weights
        """
        try:
            import torch  # noqa: F401
            from transformers import AutoConfig, AutoModelForSeq2SeqLM
        except ImportError:
            raise ImportError("transformers and torch required for from_hf()") from None

        # Load HuggingFace config
        hf_config = AutoConfig.from_pretrained(hf_path)

        # Create MLX config from HF config
        config = NLLBConfig(
            vocab_size=hf_config.vocab_size,
            d_model=hf_config.d_model,
            encoder_layers=hf_config.encoder_layers,
            decoder_layers=hf_config.decoder_layers,
            encoder_attention_heads=hf_config.encoder_attention_heads,
            decoder_attention_heads=hf_config.decoder_attention_heads,
            encoder_ffn_dim=hf_config.encoder_ffn_dim,
            decoder_ffn_dim=hf_config.decoder_ffn_dim,
            activation_function=hf_config.activation_function,
            dropout=hf_config.dropout,
            attention_dropout=hf_config.attention_dropout,
            activation_dropout=hf_config.activation_dropout,
            max_position_embeddings=hf_config.max_position_embeddings,
            scale_embedding=hf_config.scale_embedding,
            pad_token_id=hf_config.pad_token_id,
            bos_token_id=hf_config.bos_token_id,
            eos_token_id=hf_config.eos_token_id,
            decoder_start_token_id=hf_config.decoder_start_token_id,
        )

        # Create MLX model
        model = NLLBModel(config)

        # Load HuggingFace model
        hf_model = AutoModelForSeq2SeqLM.from_pretrained(hf_path)

        # Map weights from HuggingFace to MLX
        mlx_weights = _map_hf_weights_to_mlx(hf_model, config)

        # Load weights into MLX model
        model.load_weights(list(mlx_weights.items()))

        return model


def _map_hf_weights_to_mlx(hf_model, config: NLLBConfig) -> dict[str, mx.array]:
    """
    Map HuggingFace NLLB weights to MLX format.

    HuggingFace structure:
    - model.shared.weight -> shared embedding
    - model.encoder.layers.N.self_attn.{q,k,v,out}_proj.{weight,bias}
    - model.encoder.layers.N.self_attn_layer_norm.{weight,bias}
    - model.encoder.layers.N.fc1.{weight,bias}
    - model.encoder.layers.N.fc2.{weight,bias}
    - model.encoder.layers.N.final_layer_norm.{weight,bias}
    - model.encoder.layer_norm.{weight,bias}
    - model.decoder.layers.N.self_attn.* (same as encoder)
    - model.decoder.layers.N.encoder_attn.{q,k,v,out}_proj.{weight,bias}
    - model.decoder.layers.N.encoder_attn_layer_norm.{weight,bias}
    - model.decoder.layers.N.fc1.{weight,bias}
    - model.decoder.layers.N.fc2.{weight,bias}
    - model.decoder.layers.N.final_layer_norm.{weight,bias}
    - model.decoder.layer_norm.{weight,bias}

    MLX structure:
    - encoder.embed_tokens.weight
    - encoder.layers.N.self_attn.{q,k,v,out}_proj.{weight,bias}
    - encoder.layers.N.self_attn_layer_norm.{weight,bias}
    - encoder.layers.N.fc1.{weight,bias}
    - encoder.layers.N.fc2.{weight,bias}
    - encoder.layers.N.final_layer_norm.{weight,bias}
    - encoder.layer_norm.{weight,bias}
    - decoder.embed_tokens.weight
    - decoder.layers.N.* (similar to encoder + encoder_attn)
    - decoder.layer_norm.{weight,bias}
    - lm_head.weight (tied to shared.weight)
    """
    import torch

    mlx_weights = {}

    # Get HuggingFace state dict
    hf_state = hf_model.state_dict()

    def to_mlx(tensor: torch.Tensor) -> mx.array:
        """Convert PyTorch tensor to MLX array."""
        return mx.array(tensor.detach().cpu().numpy())

    # Shared embedding -> encoder and decoder embed_tokens
    shared_weight = to_mlx(hf_state["model.shared.weight"])
    mlx_weights["encoder.embed_tokens.weight"] = shared_weight
    mlx_weights["decoder.embed_tokens.weight"] = shared_weight

    # lm_head is tied to shared weight (but needs transpose for Linear)
    mlx_weights["lm_head.weight"] = shared_weight

    # Encoder layers
    for i in range(config.encoder_layers):
        prefix_hf = f"model.encoder.layers.{i}"
        prefix_mlx = f"encoder.layers.{i}"

        # Self-attention projections
        for proj in ["q_proj", "k_proj", "v_proj", "out_proj"]:
            mlx_weights[f"{prefix_mlx}.self_attn.{proj}.weight"] = to_mlx(
                hf_state[f"{prefix_hf}.self_attn.{proj}.weight"],
            )
            mlx_weights[f"{prefix_mlx}.self_attn.{proj}.bias"] = to_mlx(
                hf_state[f"{prefix_hf}.self_attn.{proj}.bias"],
            )

        # Self-attention layer norm
        mlx_weights[f"{prefix_mlx}.self_attn_layer_norm.weight"] = to_mlx(
            hf_state[f"{prefix_hf}.self_attn_layer_norm.weight"],
        )
        mlx_weights[f"{prefix_mlx}.self_attn_layer_norm.bias"] = to_mlx(
            hf_state[f"{prefix_hf}.self_attn_layer_norm.bias"],
        )

        # FFN
        mlx_weights[f"{prefix_mlx}.fc1.weight"] = to_mlx(
            hf_state[f"{prefix_hf}.fc1.weight"],
        )
        mlx_weights[f"{prefix_mlx}.fc1.bias"] = to_mlx(
            hf_state[f"{prefix_hf}.fc1.bias"],
        )
        mlx_weights[f"{prefix_mlx}.fc2.weight"] = to_mlx(
            hf_state[f"{prefix_hf}.fc2.weight"],
        )
        mlx_weights[f"{prefix_mlx}.fc2.bias"] = to_mlx(
            hf_state[f"{prefix_hf}.fc2.bias"],
        )

        # Final layer norm
        mlx_weights[f"{prefix_mlx}.final_layer_norm.weight"] = to_mlx(
            hf_state[f"{prefix_hf}.final_layer_norm.weight"],
        )
        mlx_weights[f"{prefix_mlx}.final_layer_norm.bias"] = to_mlx(
            hf_state[f"{prefix_hf}.final_layer_norm.bias"],
        )

    # Encoder final layer norm
    mlx_weights["encoder.layer_norm.weight"] = to_mlx(
        hf_state["model.encoder.layer_norm.weight"],
    )
    mlx_weights["encoder.layer_norm.bias"] = to_mlx(
        hf_state["model.encoder.layer_norm.bias"],
    )

    # Decoder layers
    for i in range(config.decoder_layers):
        prefix_hf = f"model.decoder.layers.{i}"
        prefix_mlx = f"decoder.layers.{i}"

        # Self-attention projections
        for proj in ["q_proj", "k_proj", "v_proj", "out_proj"]:
            mlx_weights[f"{prefix_mlx}.self_attn.{proj}.weight"] = to_mlx(
                hf_state[f"{prefix_hf}.self_attn.{proj}.weight"],
            )
            mlx_weights[f"{prefix_mlx}.self_attn.{proj}.bias"] = to_mlx(
                hf_state[f"{prefix_hf}.self_attn.{proj}.bias"],
            )

        # Self-attention layer norm
        mlx_weights[f"{prefix_mlx}.self_attn_layer_norm.weight"] = to_mlx(
            hf_state[f"{prefix_hf}.self_attn_layer_norm.weight"],
        )
        mlx_weights[f"{prefix_mlx}.self_attn_layer_norm.bias"] = to_mlx(
            hf_state[f"{prefix_hf}.self_attn_layer_norm.bias"],
        )

        # Cross-attention (encoder_attn) projections
        for proj in ["q_proj", "k_proj", "v_proj", "out_proj"]:
            mlx_weights[f"{prefix_mlx}.encoder_attn.{proj}.weight"] = to_mlx(
                hf_state[f"{prefix_hf}.encoder_attn.{proj}.weight"],
            )
            mlx_weights[f"{prefix_mlx}.encoder_attn.{proj}.bias"] = to_mlx(
                hf_state[f"{prefix_hf}.encoder_attn.{proj}.bias"],
            )

        # Cross-attention layer norm
        mlx_weights[f"{prefix_mlx}.encoder_attn_layer_norm.weight"] = to_mlx(
            hf_state[f"{prefix_hf}.encoder_attn_layer_norm.weight"],
        )
        mlx_weights[f"{prefix_mlx}.encoder_attn_layer_norm.bias"] = to_mlx(
            hf_state[f"{prefix_hf}.encoder_attn_layer_norm.bias"],
        )

        # FFN
        mlx_weights[f"{prefix_mlx}.fc1.weight"] = to_mlx(
            hf_state[f"{prefix_hf}.fc1.weight"],
        )
        mlx_weights[f"{prefix_mlx}.fc1.bias"] = to_mlx(
            hf_state[f"{prefix_hf}.fc1.bias"],
        )
        mlx_weights[f"{prefix_mlx}.fc2.weight"] = to_mlx(
            hf_state[f"{prefix_hf}.fc2.weight"],
        )
        mlx_weights[f"{prefix_mlx}.fc2.bias"] = to_mlx(
            hf_state[f"{prefix_hf}.fc2.bias"],
        )

        # Final layer norm
        mlx_weights[f"{prefix_mlx}.final_layer_norm.weight"] = to_mlx(
            hf_state[f"{prefix_hf}.final_layer_norm.weight"],
        )
        mlx_weights[f"{prefix_mlx}.final_layer_norm.bias"] = to_mlx(
            hf_state[f"{prefix_hf}.final_layer_norm.bias"],
        )

    # Decoder final layer norm
    mlx_weights["decoder.layer_norm.weight"] = to_mlx(
        hf_state["model.decoder.layer_norm.weight"],
    )
    mlx_weights["decoder.layer_norm.bias"] = to_mlx(
        hf_state["model.decoder.layer_norm.bias"],
    )

    return mlx_weights
