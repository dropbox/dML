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
T5 Model Implementation for MLX

Adapted from mlx-examples/t5 for use with MADLAD-400 translation.

Architecture:
- Encoder-decoder transformer with relative position biases
- Supports T5 and FLAN-T5 variants
- Compatible with HuggingFace T5ForConditionalGeneration weights

OPT-5: KV-Cache Preallocation
=============================
Uses preallocated cache buffers with slice assignment instead of concatenation.
This avoids O(n) copy operations per decode step, providing ~1.1x latency improvement.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class T5KVCache:
    """
    Preallocated KV-cache for T5 decoder attention (OPT-5).

    Uses chunk-based preallocation and slice assignment instead of
    concatenation for O(1) per-token updates vs O(n) with concatenation.

    T5 uses transposed key layout:
    - Keys: [B, H, D, S] (sequence dim last for efficient matmul)
    - Values: [B, H, S, D] (standard layout)
    """
    step = 256  # Allocation chunk size

    def __init__(self):
        self.keys = None
        self.values = None
        self.offset = 0

    def update_and_fetch(
        self,
        keys: mx.array,
        values: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """
        Update cache with new keys/values and return full cached sequences.

        Args:
            keys: New keys [B, H, D, S_new] (T5 transposed layout)
            values: New values [B, H, S_new, D]

        Returns:
            Tuple of (cached_keys, cached_values) with all positions up to offset
        """
        prev = self.offset
        num_new = keys.shape[3]  # S dimension is last for keys

        # Allocate or expand cache if needed
        if self.keys is None or (prev + num_new) > self.keys.shape[3]:
            B, H, D, _ = keys.shape
            V_D = values.shape[3]

            # Allocate in chunks for efficiency
            n_steps = (self.step + num_new - 1) // self.step
            alloc_size = n_steps * self.step

            # Keys: [B, H, D, S] - sequence dim last
            k_shape = (B, H, D, alloc_size)
            # Values: [B, H, S, D] - standard layout
            v_shape = (B, H, alloc_size, V_D)

            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)

            if self.keys is not None:
                # Trim existing cache to actual used size if needed
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev]
                    self.values = self.values[:, :, :prev, :]
                # Concatenate with new allocation
                self.keys = mx.concatenate([self.keys, new_k], axis=3)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v

        # Update offset
        self.offset += num_new

        # Slice assignment - O(1) instead of O(n) concatenation
        self.keys[..., prev:self.offset] = keys
        self.values[:, :, prev:self.offset, :] = values

        # Return views of valid cache portion
        return self.keys[..., :self.offset], self.values[:, :, :self.offset, :]

    def get_state(self) -> tuple[mx.array, mx.array]:
        """Get current cache state as tuple for compatibility with existing code."""
        if self.keys is None:
            return None
        return (
            self.keys[..., :self.offset],
            self.values[:, :, :self.offset, :],
        )

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        """Trim n tokens from the end of the cache."""
        n = min(self.offset, n)
        self.offset -= n
        return n

    @classmethod
    def from_tuple(cls, cache_tuple: tuple[mx.array, mx.array]) -> "T5KVCache":
        """Create a cache from existing (keys, values) tuple for backwards compat."""
        cache = cls()
        if cache_tuple is not None:
            keys, values = cache_tuple
            cache.keys = keys
            cache.values = values
            cache.offset = keys.shape[3]  # S dim is last for keys
        return cache

    def copy(self) -> "T5KVCache":
        """Create a deep copy of this cache (for speculative decoding)."""
        new_cache = T5KVCache()
        if self.keys is not None:
            # Copy arrays to avoid sharing state
            new_cache.keys = self.keys[..., :self.offset]
            new_cache.values = self.values[:, :, :self.offset, :]
            new_cache.offset = self.offset
        return new_cache


def copy_cache_list(cache_list: list[T5KVCache]) -> list[T5KVCache]:
    """Create a deep copy of a cache list (for speculative decoding)."""
    if cache_list is None:
        return None
    return [c.copy() if c is not None else None for c in cache_list]


def _relative_position_bucket(
    relative_position, bidirectional=True, num_buckets=32, max_distance=128,
):
    """
    Translate relative position to a bucket number for relative attention.

    Adapted from HuggingFace T5 implementation.
    """
    relative_buckets = 0
    if bidirectional:
        num_buckets //= 2
        relative_buckets += (relative_position > 0).astype(mx.int16) * num_buckets
        relative_position = mx.abs(relative_position)
    else:
        relative_position = -mx.minimum(
            relative_position, mx.zeros_like(relative_position),
        )

    max_exact = num_buckets // 2
    is_small = relative_position < max_exact

    scale = (num_buckets - max_exact) / np.log(max_distance / max_exact)
    relative_position_if_large = max_exact + (
        mx.log(relative_position.astype(mx.float32) / max_exact) * scale
    ).astype(mx.int16)
    relative_position_if_large = mx.minimum(relative_position_if_large, num_buckets - 1)
    relative_buckets += mx.where(
        is_small, relative_position, relative_position_if_large,
    )
    return relative_buckets


class RelativePositionBias(nn.Module):
    def __init__(self, config, bidirectional: bool):
        self.bidirectional = bidirectional
        self.num_buckets = config.relative_attention_num_buckets
        self.max_distance = getattr(config, "relative_attention_max_distance", 128)
        self.n_heads = config.num_heads
        self.embeddings = nn.Embedding(
            config.relative_attention_num_buckets, config.num_heads,
        )

    def __call__(self, query_length: int, key_length: int, offset: int = 0):
        """Compute binned relative position bias"""
        context_position = mx.arange(offset, query_length)[:, None]
        memory_position = mx.arange(key_length)[None, :]

        relative_position = memory_position - context_position
        relative_position_bucket = _relative_position_bucket(
            relative_position,
            bidirectional=self.bidirectional,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )

        values = self.embeddings(relative_position_bucket)
        return values.transpose(2, 0, 1)


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        inner_dim = config.d_kv * config.num_heads
        self.num_heads = config.num_heads
        self.query_proj = nn.Linear(config.d_model, inner_dim, bias=False)
        self.key_proj = nn.Linear(config.d_model, inner_dim, bias=False)
        self.value_proj = nn.Linear(config.d_model, inner_dim, bias=False)
        self.out_proj = nn.Linear(inner_dim, config.d_model, bias=False)

    def __call__(
        self,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        mask: mx.array | None,
        cache: T5KVCache | None = None,
    ) -> tuple[mx.array, T5KVCache]:
        """
        Multi-head attention with optional KV cache.

        Args:
            queries: Query tensor [B, L, D]
            keys: Key tensor [B, S, D]
            values: Value tensor [B, S, D]
            mask: Attention mask
            cache: Optional T5KVCache for incremental decoding

        Returns:
            Tuple of (output, cache) where cache is T5KVCache
        """
        queries = self.query_proj(queries)
        keys = self.key_proj(keys)
        values = self.value_proj(values)

        num_heads = self.num_heads
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        queries = queries.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        # Keys: [B, H, D, S] - transposed for efficient matmul
        keys = keys.reshape(B, S, num_heads, -1).transpose(0, 2, 3, 1)
        # Values: [B, H, S, D]
        values = values.reshape(B, S, num_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            # OPT-5: Use preallocated cache with slice assignment
            keys, values = cache.update_and_fetch(keys, values)

        scores = queries @ keys
        if mask is not None:
            scores = scores + mask.astype(scores.dtype)

        scores = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        values_hat = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.out_proj(values_hat), cache


class DenseActivation(nn.Module):
    def __init__(self, config):
        super().__init__()
        mlp_dims = config.d_ff or config.d_model * 4
        self.gated = hasattr(config, "feed_forward_proj")
        activation = (
            "relu"
            if not self.gated
            else config.feed_forward_proj.removeprefix("gated-")
        )
        if self.gated:
            self.wi_0 = nn.Linear(config.d_model, mlp_dims, bias=False)
            self.wi_1 = nn.Linear(config.d_model, mlp_dims, bias=False)
        else:
            self.wi = nn.Linear(config.d_model, mlp_dims, bias=False)
        self.wo = nn.Linear(mlp_dims, config.d_model, bias=False)
        if activation == "relu":
            self.act = nn.relu
        elif activation == "gelu":
            self.act = nn.gelu
        elif activation == "silu":
            self.act = nn.silu
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def __call__(self, x):
        if self.gated:
            hidden_act = self.act(self.wi_0(x))
            hidden_linear = self.wi_1(x)
            x = hidden_act * hidden_linear
        else:
            x = self.act(self.wi(x))
        return self.wo(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.ln1 = nn.RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.ln2 = nn.RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dense = DenseActivation(config)

    def __call__(self, x, mask):
        y = self.ln1(x)
        y, _ = self.attention(y, y, y, mask=mask)
        x = x + y

        y = self.ln2(x)
        y = self.dense(y)
        return x + y


class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = [
            TransformerEncoderLayer(config) for i in range(config.num_layers)
        ]
        self.ln = nn.RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.relative_attention_bias = RelativePositionBias(config, bidirectional=True)

    def __call__(self, x: mx.array):
        pos_bias = self.relative_attention_bias(x.shape[1], x.shape[1])
        for layer in self.layers:
            x = layer(x, mask=pos_bias)
        return self.ln(x)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attention = MultiHeadAttention(config)
        self.cross_attention = MultiHeadAttention(config)
        self.ln1 = nn.RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.ln2 = nn.RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.ln3 = nn.RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dense = DenseActivation(config)

    def __call__(
        self,
        x: mx.array,
        memory: mx.array,
        mask: mx.array,
        memory_mask: mx.array,
        cache: T5KVCache | None = None,
    ):
        """
        Decoder layer with KV cache for self-attention.

        Args:
            x: Input tensor [B, L, D]
            memory: Encoder output [B, S_enc, D]
            mask: Causal attention mask
            memory_mask: Cross-attention mask (typically None)
            cache: Optional T5KVCache for self-attention

        Returns:
            Tuple of (output, cache)
        """
        y = self.ln1(x)
        y, cache = self.self_attention(y, y, y, mask, cache)
        x = x + y

        y = self.ln2(x)
        y, _ = self.cross_attention(y, memory, memory, memory_mask)
        x = x + y

        y = self.ln3(x)
        y = self.dense(y)
        x = x + y

        return x, cache


class TransformerDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_layers = getattr(config, "num_decoder_layers", config.num_layers)
        self.layers = [TransformerDecoderLayer(config) for i in range(n_layers)]
        self.ln = nn.RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.relative_attention_bias = RelativePositionBias(config, bidirectional=False)
        self.n_layers = n_layers

    def make_cache(self) -> list[T5KVCache]:
        """Create empty KV caches for all decoder layers (OPT-5)."""
        return [T5KVCache() for _ in range(self.n_layers)]

    def __call__(self, x, memory, mask, memory_mask, cache=None):
        """
        Decoder forward pass with KV cache.

        Args:
            x: Input tensor [B, L, D]
            memory: Encoder output [B, S_enc, D]
            mask: Optional causal mask
            memory_mask: Optional cross-attention mask
            cache: List of T5KVCache objects (one per layer)

        Returns:
            Tuple of (output, cache)
        """
        if cache is not None and cache[0] is not None:
            # Get offset from first layer's cache
            offset = cache[0].offset
        else:
            offset = 0
            cache = self.make_cache()

        T = offset + x.shape[1]
        pos_bias = self.relative_attention_bias(T, T, offset=offset)
        if mask is not None:
            mask = mask + pos_bias
        else:
            mask = pos_bias

        for e, layer in enumerate(self.layers):
            x, cache[e] = layer(x, memory, mask, memory_mask, cache=cache[e])
        x = self.ln(x)

        return x, cache


class OutputHead(nn.Module):
    def __init__(self, config):
        self.linear = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def __call__(self, inputs):
        return self.linear(inputs)


class T5Model(nn.Module):
    """
    T5 encoder-decoder model for MLX.

    Supports T5, FLAN-T5, and MADLAD-400 variants.
    """

    def __init__(self, config):
        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(config)
        self.tie_word_embeddings = getattr(config, "tie_word_embeddings", True)
        if not self.tie_word_embeddings:
            self.lm_head = OutputHead(config)
        self.model_dim = config.d_model
        self.config = config

    def encode(self, inputs: mx.array):
        """Encode input tokens to hidden states."""
        return self.encoder(self.wte(inputs))

    def decode(
        self,
        inputs: mx.array,
        memory: mx.array,
        cache=None,
    ):
        """
        Decode with encoder memory and optional KV cache.

        Args:
            inputs: Decoder input token IDs [B, T]
            memory: Encoder hidden states [B, S_enc, D]
            cache: Optional list of T5KVCache (one per decoder layer)

        Returns:
            Tuple of (logits, cache) where cache is list of T5KVCache
        """
        inputs = self.wte(inputs)
        T = inputs.shape[1]

        # Calculate offset from cache if present (OPT-5 uses T5KVCache objects)
        if cache is not None and cache[0] is not None:
            offset = cache[0].offset
        else:
            offset = 0

        # Create causal mask that accounts for cached positions
        # New queries can attend to all cached positions plus current positions
        if T > 1:
            # Total key length including cache
            total_kv_len = offset + T
            # Create mask: (T_query, T_total_kv)
            # Query position i (at offset+i) can attend to key positions [0, offset+i]
            # Use broadcasting: row i should have -inf for columns > offset+i

            # Create query positions [0, 1, ..., T-1] + offset = [offset, offset+1, ..., offset+T-1]
            query_pos = mx.arange(T) + offset  # [T]
            # Create key positions [0, 1, ..., total_kv_len-1]
            key_pos = mx.arange(total_kv_len)  # [total_kv_len]

            # Mask where key_pos > query_pos (causal: can't attend to future)
            # query_pos[:, None] has shape [T, 1], key_pos has shape [total_kv_len]
            # Broadcasting gives [T, total_kv_len]
            causal_mask = key_pos[None, :] > query_pos[:, None]
            mask = mx.where(causal_mask, float("-inf"), 0.0).astype(inputs.dtype)
        else:
            mask = None

        y, cache = self.decoder(
            inputs, memory=memory, mask=mask, memory_mask=None, cache=cache,
        )
        if not self.tie_word_embeddings:
            y = self.lm_head(y)
        else:
            y *= self.model_dim**-0.5
            y = y @ self.wte.weight.T
        return y, cache

    def __call__(
        self,
        inputs: mx.array,
        decoder_inputs: mx.array,
    ):
        return self.decode(decoder_inputs, self.encode(inputs))[0]

    @classmethod
    def sanitize(cls, weights):
        """Sanitize HuggingFace weight names to MLX format."""
        shared_replacement_patterns = [
            (".block.", ".layers."),
            (".k.", ".key_proj."),
            (".o.", ".out_proj."),
            (".q.", ".query_proj."),
            (".v.", ".value_proj."),
            ("shared.", "wte."),
            # MADLAD uses decoder.embed_tokens instead of shared
            ("decoder.embed_tokens.", "wte."),
            ("lm_head.", "lm_head.linear."),
            (".layer.0.layer_norm.", ".ln1."),
            (".layer.1.layer_norm.", ".ln2."),
            (".layer.2.layer_norm.", ".ln3."),
            (".final_layer_norm.", ".ln."),
            (
                "layers.0.layer.0.SelfAttention.relative_attention_bias.",
                "relative_attention_bias.embeddings.",
            ),
        ]

        encoder_replacement_patterns = [
            (".layer.0.SelfAttention.", ".attention."),
            (".layer.1.DenseReluDense.", ".dense."),
        ]

        decoder_replacement_patterns = [
            (".layer.0.SelfAttention.", ".self_attention."),
            (".layer.1.EncDecAttention.", ".cross_attention."),
            (".layer.2.DenseReluDense.", ".dense."),
        ]

        ignored_keys = [
            "decoder.layers.0.cross_attention.relative_attention_bias.weight",
        ]

        def replace_key(key: str) -> str:
            for old, new in shared_replacement_patterns:
                key = key.replace(old, new)
            if key.startswith("encoder."):
                for old, new in encoder_replacement_patterns:
                    key = key.replace(old, new)
            elif key.startswith("decoder."):
                for old, new in decoder_replacement_patterns:
                    key = key.replace(old, new)
            return key

        weights = {replace_key(k): v for k, v in weights.items()}
        for key in ignored_keys:
            if key in weights:
                del weights[key]
        return weights

    def decode_early_exit(
        self,
        inputs: mx.array,
        memory: mx.array,
        cache=None,
        num_layers: int = 4,
        early_exit_head=None,
    ):
        """
        Decode using only first N decoder layers (for speculative decoding).

        This produces faster but less accurate predictions for drafting.
        Can use a trained early_exit_head for better token prediction.

        Args:
            inputs: Decoder input token IDs
            memory: Encoder hidden states
            cache: Optional list of T5KVCache (only first num_layers will be used)
            num_layers: Number of decoder layers to use (default: 4)
            early_exit_head: Optional trained head for token prediction.
                             If None, uses default lm_head projection.

        Returns:
            Tuple of (logits, cache) where cache only has num_layers entries
        """
        inputs = self.wte(inputs)
        T = inputs.shape[1]

        # Use decoder's relative position bias
        if cache is not None and cache[0] is not None:
            offset = cache[0].offset
        else:
            offset = 0
            # Only allocate cache for early layers (OPT-5)
            cache = [T5KVCache() for _ in range(num_layers)]

        # Create causal mask that accounts for cached positions
        if T > 1:
            total_kv_len = offset + T
            query_pos = mx.arange(T) + offset
            key_pos = mx.arange(total_kv_len)
            causal_mask = key_pos[None, :] > query_pos[:, None]
            mask = mx.where(causal_mask, float("-inf"), 0.0).astype(inputs.dtype)
        else:
            mask = None

        T_full = offset + T
        pos_bias = self.decoder.relative_attention_bias(T_full, T_full, offset=offset)
        if mask is not None:
            mask = mask + pos_bias
        else:
            mask = pos_bias

        # Only run first num_layers
        x = inputs
        for i in range(min(num_layers, len(self.decoder.layers))):
            x, cache[i] = self.decoder.layers[i](
                x, memory, mask, memory_mask=None, cache=cache[i],
            )
        x = self.decoder.ln(x)

        # Project to logits using early_exit_head if provided
        if early_exit_head is not None:
            y = early_exit_head(x)
        elif not self.tie_word_embeddings:
            y = self.lm_head(x)
        else:
            y = x * self.model_dim**-0.5
            y = y @ self.wte.weight.T
        return y, cache

    @staticmethod
    def trim_decoder_cache(cache, num_to_remove: int):
        """
        Trim the decoder cache by removing the last N entries.

        Used in speculative decoding when draft tokens are rejected.

        Args:
            cache: List of T5KVCache objects per layer
            num_to_remove: Number of tokens to remove from end of cache

        Returns:
            Same cache list (modified in-place via trim)
        """
        if cache is None or num_to_remove <= 0:
            return cache

        for layer_cache in cache:
            if layer_cache is not None:
                layer_cache.trim(num_to_remove)
        return cache

    @classmethod
    def from_pretrained(
        cls, path_or_repo: str, dtype: mx.Dtype = mx.bfloat16,
    ) -> "T5Model":
        """
        Load T5 model from HuggingFace or local path.

        Args:
            path_or_repo: HuggingFace repo ID or local path
            dtype: Model dtype (default: bfloat16)

        Returns:
            Loaded T5Model
        """
        from huggingface_hub import snapshot_download

        path = Path(path_or_repo)
        if not path.exists():
            path = Path(
                snapshot_download(
                    repo_id=path_or_repo,
                    allow_patterns=["*.json", "*.safetensors", "*.model"],
                ),
            )

        with open(path / "config.json") as f:
            config = SimpleNamespace(**json.load(f))

        model = cls(config)

        # Load weights - support both single-file and sharded safetensors
        single_file = path / "model.safetensors"
        index_file = path / "model.safetensors.index.json"

        if single_file.exists():
            # Single safetensors file
            weights = mx.load(str(single_file))
        elif index_file.exists():
            # Sharded safetensors - load all shards and merge

            with open(index_file) as f:
                index = json.load(f)

            # Find unique shard files
            shard_files = sorted(set(index["weight_map"].values()))
            print(f"Loading {len(shard_files)} weight shards...")

            weights = {}
            for shard_file in shard_files:
                shard_path = path / shard_file
                if shard_path.exists():
                    shard_weights = mx.load(str(shard_path))
                    weights.update(shard_weights)
                else:
                    raise FileNotFoundError(f"Shard file not found: {shard_path}")
        else:
            raise FileNotFoundError(
                f"Could not find model weights at {path}. "
                f"Expected model.safetensors or model.safetensors.index.json",
            )

        weights = cls.sanitize(weights)
        weights = {k: v.astype(dtype) for k, v in weights.items()}
        model.load_weights(list(weights.items()))
        return model
