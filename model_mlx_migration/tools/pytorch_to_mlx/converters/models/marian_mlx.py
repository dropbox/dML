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
MLX Implementation of Marian (OPUS-MT) Translation Model

Based on HuggingFace MarianMTModel.
Architecture is similar to BART/NLLB with learned position embeddings.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


@dataclass
class MarianConfig:
    """Configuration for Marian model."""

    vocab_size: int = 65001
    d_model: int = 512
    encoder_layers: int = 6
    decoder_layers: int = 6
    encoder_attention_heads: int = 8
    decoder_attention_heads: int = 8
    encoder_ffn_dim: int = 2048
    decoder_ffn_dim: int = 2048
    activation_function: str = "swish"
    max_position_embeddings: int = 512
    pad_token_id: int = 65000
    eos_token_id: int = 0
    decoder_start_token_id: int = 65000


class MarianAttention(nn.Module):
    """Multi-head attention for Marian."""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
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
        batch_size, seq_len, _ = hidden_states.shape

        is_cross_attention = key_value_states is not None

        queries = self.q_proj(hidden_states)

        if is_cross_attention:
            keys = self.k_proj(key_value_states)
            values = self.v_proj(key_value_states)
        else:
            keys = self.k_proj(hidden_states)
            values = self.v_proj(hidden_states)

        if cache is not None:
            past_key, past_value = cache
            if is_cross_attention:
                keys = past_key
                values = past_value
            else:
                keys = mx.concatenate([past_key, keys], axis=1)
                values = mx.concatenate([past_value, values], axis=1)

        new_cache = (keys, values)

        queries = queries.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        queries = queries.transpose(0, 2, 1, 3)

        kv_seq_len = keys.shape[1]
        keys = keys.reshape(batch_size, kv_seq_len, self.num_heads, self.head_dim)
        keys = keys.transpose(0, 2, 1, 3)

        values = values.reshape(batch_size, kv_seq_len, self.num_heads, self.head_dim)
        values = values.transpose(0, 2, 1, 3)

        scores = (queries @ keys.transpose(0, 1, 3, 2)) * self.scale

        if attention_mask is not None:
            scores = scores + attention_mask

        weights = mx.softmax(scores, axis=-1)
        output = weights @ values

        output = output.transpose(0, 2, 1, 3)
        output = output.reshape(batch_size, seq_len, self.embed_dim)

        return self.out_proj(output), new_cache


class MarianEncoderLayer(nn.Module):
    """Single encoder layer."""

    def __init__(self, config: MarianConfig):
        super().__init__()
        self.self_attn = MarianAttention(config.d_model, config.encoder_attention_heads)
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.fc1 = nn.Linear(config.d_model, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, config.d_model)
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        self.activation = nn.silu  # swish = silu

    def __call__(
        self, hidden_states: mx.array, attention_mask: mx.array | None = None,
    ) -> mx.array:
        residual = hidden_states
        hidden_states, _ = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states
        return self.final_layer_norm(hidden_states)



class MarianDecoderLayer(nn.Module):
    """Single decoder layer."""

    def __init__(self, config: MarianConfig):
        super().__init__()
        self.self_attn = MarianAttention(config.d_model, config.decoder_attention_heads)
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.encoder_attn = MarianAttention(
            config.d_model, config.decoder_attention_heads,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.fc1 = nn.Linear(config.d_model, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, config.d_model)
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        self.activation = nn.silu

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        attention_mask: mx.array | None = None,
        encoder_attention_mask: mx.array | None = None,
        cache: tuple | None = None,
    ) -> tuple[mx.array, tuple | None]:
        self_attn_cache = cache[0] if cache is not None else None
        cross_attn_cache = cache[1] if cache is not None else None

        residual = hidden_states
        hidden_states, self_attn_cache = self.self_attn(
            hidden_states, attention_mask=attention_mask, cache=self_attn_cache,
        )
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states, cross_attn_cache = self.encoder_attn(
            hidden_states,
            key_value_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            cache=cross_attn_cache,
        )
        hidden_states = residual + hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states, (self_attn_cache, cross_attn_cache)


class MarianEncoder(nn.Module):
    """Marian encoder stack."""

    def __init__(self, config: MarianConfig):
        super().__init__()
        self.embed_scale = config.d_model**0.5  # sqrt(d_model) scaling
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.embed_positions = nn.Embedding(
            config.max_position_embeddings, config.d_model,
        )
        self.layers = [MarianEncoderLayer(config) for _ in range(config.encoder_layers)]

    def __call__(
        self, input_ids: mx.array, attention_mask: mx.array | None = None,
    ) -> mx.array:
        seq_len = input_ids.shape[1]

        hidden_states = self.embed_tokens(input_ids) * self.embed_scale  # Apply scaling
        position_ids = mx.arange(seq_len)
        hidden_states = hidden_states + self.embed_positions(position_ids)

        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * -1e9

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        return hidden_states


class MarianDecoder(nn.Module):
    """Marian decoder stack."""

    def __init__(self, config: MarianConfig):
        super().__init__()
        self.config = config
        self.embed_scale = config.d_model**0.5  # sqrt(d_model) scaling
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.embed_positions = nn.Embedding(
            config.max_position_embeddings, config.d_model,
        )
        self.layers = [MarianDecoderLayer(config) for _ in range(config.decoder_layers)]

    def __call__(
        self,
        input_ids: mx.array,
        encoder_hidden_states: mx.array,
        encoder_attention_mask: mx.array | None = None,
        cache: list | None = None,
    ) -> tuple[mx.array, list]:
        seq_len = input_ids.shape[1]

        past_length = 0
        if cache is not None and cache[0] is not None:
            past_key, _ = cache[0][0]
            past_length = past_key.shape[1]

        hidden_states = self.embed_tokens(input_ids) * self.embed_scale  # Apply scaling
        position_ids = mx.arange(past_length, past_length + seq_len)
        hidden_states = hidden_states + self.embed_positions(position_ids)

        # Causal mask
        mask = mx.triu(mx.full((seq_len, seq_len + past_length), -1e9), k=past_length + 1)
        attention_mask = mask[None, None, :, :]

        if encoder_attention_mask is not None:
            encoder_attention_mask = encoder_attention_mask[:, None, None, :]
            encoder_attention_mask = (1.0 - encoder_attention_mask) * -1e9

        if cache is None:
            cache = [None] * len(self.layers)

        new_cache = []
        for i, layer in enumerate(self.layers):
            hidden_states, layer_cache = layer(
                hidden_states,
                encoder_hidden_states,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                cache=cache[i],
            )
            new_cache.append(layer_cache)

        return hidden_states, new_cache


class MarianModel(nn.Module):
    """Complete Marian encoder-decoder model."""

    def __init__(self, config: MarianConfig):
        super().__init__()
        self.config = config
        self.encoder = MarianEncoder(config)
        self.decoder = MarianDecoder(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # final_logits_bias is crucial for correct output distribution
        self.final_logits_bias = mx.zeros((1, config.vocab_size))

    def encode(
        self, input_ids: mx.array, attention_mask: mx.array | None = None,
    ) -> mx.array:
        return self.encoder(input_ids, attention_mask)

    def decode(
        self,
        decoder_input_ids: mx.array,
        encoder_hidden_states: mx.array,
        encoder_attention_mask: mx.array | None = None,
        cache: list | None = None,
    ) -> tuple[mx.array, list]:
        decoder_output, cache = self.decoder(
            decoder_input_ids,
            encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            cache=cache,
        )
        logits = self.lm_head(decoder_output) + self.final_logits_bias
        return logits, cache

    @staticmethod
    def from_pretrained(path: str, dtype: mx.Dtype = mx.float16) -> "MarianModel":
        """Load model from converted weights or HuggingFace."""
        from huggingface_hub import snapshot_download

        path = Path(path)
        if not path.exists():
            path = Path(
                snapshot_download(
                    repo_id=str(path),
                    allow_patterns=["*.json", "*.safetensors", "*.bin"],
                ),
            )

        with open(path / "config.json") as f:
            config_dict = json.load(f)

        config = MarianConfig(
            vocab_size=config_dict.get("vocab_size", 65001),
            d_model=config_dict.get("d_model", 512),
            encoder_layers=config_dict.get("encoder_layers", 6),
            decoder_layers=config_dict.get("decoder_layers", 6),
            encoder_attention_heads=config_dict.get("encoder_attention_heads", 8),
            decoder_attention_heads=config_dict.get("decoder_attention_heads", 8),
            encoder_ffn_dim=config_dict.get("encoder_ffn_dim", 2048),
            decoder_ffn_dim=config_dict.get("decoder_ffn_dim", 2048),
            max_position_embeddings=config_dict.get("max_position_embeddings", 512),
            pad_token_id=config_dict.get("pad_token_id", 65000),
            eos_token_id=config_dict.get("eos_token_id", 0),
            decoder_start_token_id=config_dict.get("decoder_start_token_id", 65000),
        )

        model = MarianModel(config)

        # Try safetensors first, then pytorch bin
        weights_path = path / "model.safetensors"
        if not weights_path.exists():
            weights_path = path / "pytorch_model.bin"

        if weights_path.suffix == ".safetensors":
            weights = mx.load(str(weights_path))
        else:
            import torch

            pt_weights = torch.load(str(weights_path), map_location="cpu")
            weights = {k: mx.array(v.numpy()) for k, v in pt_weights.items()}

        weights = _sanitize_marian_weights(weights)
        weights = {k: v.astype(dtype) for k, v in weights.items()}
        model.load_weights(list(weights.items()))

        return model


def _sanitize_marian_weights(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """Map HuggingFace Marian weights to MLX format."""
    replacements = [
        ("model.encoder.", "encoder."),
        ("model.decoder.", "decoder."),
    ]

    # Skip explicit embed_tokens (shared.weight handles them)
    ignored = [
        "model.encoder.embed_tokens.weight",  # Handled by model.shared.weight
        "model.decoder.embed_tokens.weight",  # Handled by model.shared.weight
    ]

    sanitized = {}
    for k, v in weights.items():
        if any(ig == k for ig in ignored):
            continue

        # Handle shared embedding - copy to encoder, decoder, and lm_head (tied weights)
        if k == "model.shared.weight":
            sanitized["encoder.embed_tokens.weight"] = v
            sanitized["decoder.embed_tokens.weight"] = v
            sanitized["lm_head.weight"] = v  # lm_head is tied to shared embedding
            continue

        new_key = k
        for old, new in replacements:
            new_key = new_key.replace(old, new)
        sanitized[new_key] = v

    return sanitized
