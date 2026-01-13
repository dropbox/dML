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
MLX implementation of Zipformer2 encoder components.

This module ports the Zipformer2 encoder architecture from icefall/k2-fsa
to MLX for efficient inference on Apple Silicon.

Reference: tools/third_party/icefall/egs/librispeech/ASR/zipformer/zipformer.py
"""

import math

import mlx.core as mx
import mlx.nn as nn

from .scaling import (
    ActivationDropoutAndLinear,
    Balancer,
    BiasNorm,
    ScaledLinear,
    Whiten,
)


class ChunkCausalDepthwiseConv1d(nn.Module):
    """
    MLX implementation of ChunkCausalDepthwiseConv1d from icefall.

    Causal depthwise 1D convolution that operates in a chunkwise manner.
    Uses two convolutions: one causal (right-padded) and one within-chunk.

    Args:
        channels: Number of input/output channels.
        kernel_size: Kernel size (must be odd).
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"

        self.channels = channels
        self.kernel_size = kernel_size
        half_kernel_size = (kernel_size + 1) // 2

        # Causal convolution (will pad manually on left)
        # MLX conv1d weight shape: (C_out, K, C_in)
        # For depthwise with groups=channels: (channels, K, 1)
        self.causal_conv_weight = mx.random.normal(
            shape=(channels, half_kernel_size, 1),
        ) * 0.1
        self.causal_conv_bias = mx.random.uniform(
            low=-0.1, high=0.1, shape=(channels,),
        )

        # Chunkwise convolution with symmetric padding
        self.chunkwise_conv_weight = mx.random.normal(
            shape=(channels, kernel_size, 1),
        ) * 0.1
        self.chunkwise_conv_bias = mx.random.uniform(
            low=-0.1, high=0.1, shape=(channels,),
        )

        # Scale factors for chunk edges
        # Shape: (2, channels, kernel_size) - first row for left edge, second for right
        self.chunkwise_conv_scale = mx.zeros((2, channels, kernel_size))

    def _depthwise_conv1d(
        self,
        x: mx.array,
        weight: mx.array,
        bias: mx.array,
        padding: int = 0,
    ) -> mx.array:
        """
        Depthwise 1D convolution.

        Args:
            x: Input of shape (batch, channels, time)
            weight: Weight of shape (channels, kernel_size, 1) for depthwise
            bias: Bias of shape (channels,)
            padding: Symmetric padding amount

        Returns:
            Output of shape (batch, channels, time)
        """
        batch, channels, time = x.shape

        # Transpose to MLX format: (batch, time, channels)
        x = mx.transpose(x, (0, 2, 1))

        if padding > 0:
            x = mx.pad(x, [(0, 0), (padding, padding), (0, 0)])

        # Use groups=channels for depthwise convolution
        out = mx.conv1d(x, weight, groups=channels)

        # Transpose back to (batch, channels, time)
        out = mx.transpose(out, (0, 2, 1))

        # Add bias
        out = out + mx.reshape(bias, (1, channels, 1))

        return out

    def _get_chunk_scale(self, chunk_size: int) -> mx.array:
        """
        Get scaling factors for chunk positions.

        Returns tensor of shape (channels, chunk_size).
        """
        left_edge = self.chunkwise_conv_scale[0]  # (channels, kernel_size)
        right_edge = self.chunkwise_conv_scale[1]  # (channels, kernel_size)

        if chunk_size < self.kernel_size:
            left_edge = left_edge[:, :chunk_size]
            right_edge = right_edge[:, -chunk_size:]
        else:
            t = chunk_size - self.kernel_size
            channels = left_edge.shape[0]
            pad = mx.zeros((channels, t))
            left_edge = mx.concatenate([left_edge, pad], axis=-1)
            right_edge = mx.concatenate([pad, right_edge], axis=-1)

        return 1.0 + (left_edge + right_edge)

    def __call__(self, x: mx.array, chunk_size: int = -1) -> mx.array:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, channels, seq_len)
            chunk_size: Chunk size in frames; -1 means no chunking

        Returns:
            Output tensor of shape (batch_size, channels, seq_len)
        """
        batch_size, num_channels, seq_len = x.shape

        left_pad = self.kernel_size // 2
        (self.kernel_size + 1) // 2

        if chunk_size < 0 or chunk_size > seq_len:
            chunk_size = seq_len

        # Compute right padding to make seq_len divisible by chunk_size
        right_pad = (-seq_len) % chunk_size

        # Pad input: left for causal conv, right for chunk alignment
        x_padded = mx.pad(x, [(0, 0), (0, 0), (left_pad, right_pad)])

        # Causal convolution
        x_for_causal = x_padded[:, :, : left_pad + seq_len]
        x_causal = self._depthwise_conv1d(
            x_for_causal,
            self.causal_conv_weight,
            self.causal_conv_bias,
            padding=0,
        )

        # Chunkwise convolution
        x_for_chunk = x_padded[:, :, left_pad:]  # (batch, channels, seq_len + right_pad)
        num_chunks = x_for_chunk.shape[2] // chunk_size

        # Reshape to process chunks: (batch * num_chunks, channels, chunk_size)
        x_chunk = mx.reshape(
            x_for_chunk, (batch_size, num_channels, num_chunks, chunk_size),
        )
        x_chunk = mx.transpose(x_chunk, (0, 2, 1, 3))
        x_chunk = mx.reshape(
            x_chunk, (batch_size * num_chunks, num_channels, chunk_size),
        )

        # Apply chunkwise conv with symmetric padding
        x_chunk = self._depthwise_conv1d(
            x_chunk,
            self.chunkwise_conv_weight,
            self.chunkwise_conv_bias,
            padding=self.kernel_size // 2,
        )

        # Apply chunk-position-dependent scaling
        chunk_scale = self._get_chunk_scale(chunk_size)  # (channels, chunk_size)
        x_chunk = x_chunk * mx.expand_dims(chunk_scale, axis=0)

        # Reshape back: (batch, channels, num_chunks * chunk_size)
        x_chunk = mx.reshape(
            x_chunk, (batch_size, num_chunks, num_channels, chunk_size),
        )
        x_chunk = mx.transpose(x_chunk, (0, 2, 1, 3))
        x_chunk = mx.reshape(
            x_chunk, (batch_size, num_channels, num_chunks * chunk_size),
        )

        # Trim to original length
        x_chunk = x_chunk[:, :, :seq_len]

        return x_chunk + x_causal

    def streaming_forward(
        self,
        x: mx.array,
        cache: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """
        Streaming forward pass.

        Args:
            x: Input tensor of shape (batch_size, channels, seq_len)
            cache: Cached left context of shape (batch_size, channels, left_pad)

        Returns:
            - Output tensor of shape (batch_size, channels, seq_len)
            - Updated cache
        """
        batch_size, num_channels, seq_len = x.shape
        left_pad = self.kernel_size // 2

        # Concatenate cache with input
        x = mx.concatenate([cache, x], axis=2)

        # Update cache
        new_cache = x[:, :, -left_pad:]

        # Causal convolution
        x_causal = self._depthwise_conv1d(
            x,
            self.causal_conv_weight,
            self.causal_conv_bias,
            padding=0,
        )

        # Chunkwise convolution (using seq_len as chunk size for streaming)
        x_chunk = x[:, :, left_pad:]
        x_chunk = self._depthwise_conv1d(
            x_chunk,
            self.chunkwise_conv_weight,
            self.chunkwise_conv_bias,
            padding=self.kernel_size // 2,
        )

        # Apply scaling
        chunk_scale = self._get_chunk_scale(seq_len)
        x_chunk = x_chunk * mx.expand_dims(chunk_scale, axis=0)

        return x_chunk + x_causal, new_cache


class CompactRelPositionalEncoding(nn.Module):
    """
    MLX implementation of compact relative positional encoding.

    Uses compressed coordinates via atan to handle very long sequences
    efficiently while maintaining resolution for nearby positions.

    Args:
        embed_dim: Embedding dimension (must be even).
        length_factor: Scale factor for position resolution (>= 1.0).
    """

    def __init__(
        self,
        embed_dim: int,
        length_factor: float = 1.0,
    ):
        super().__init__()
        assert embed_dim % 2 == 0, "embed_dim must be even"
        assert length_factor >= 1.0, "length_factor must be >= 1.0"

        self.embed_dim = embed_dim
        self.length_factor = length_factor
        self._pe_cache = None
        self._pe_cache_len = 0

    def _compute_pe(self, max_len: int) -> mx.array:
        """Compute positional encoding for positions [-max_len+1, max_len-1]."""
        # Position indices: -(T-1) to (T-1)
        positions = mx.arange(-(max_len - 1), max_len, dtype=mx.float32)
        positions = mx.expand_dims(positions, axis=1)  # (2T-1, 1)

        # Frequencies
        freqs = 1 + mx.arange(self.embed_dim // 2, dtype=mx.float32)

        # Compression for large positions
        compression_length = self.embed_dim**0.5
        x_sign = mx.sign(positions)
        x_abs = mx.abs(positions)
        x_compressed = (
            compression_length
            * x_sign
            * (mx.log(x_abs + compression_length) - math.log(compression_length))
        )

        # Length scale for FFT resolution
        length_scale = self.length_factor * self.embed_dim / (2.0 * math.pi)

        # Apply atan for bounded range
        x_atan = mx.arctan(x_compressed / length_scale)

        # Compute sinusoidal embeddings
        angles = x_atan * freqs  # (2T-1, embed_dim//2)
        cosines = mx.cos(angles)
        sines = mx.sin(angles)

        # Interleave cosines and sines
        pe = mx.zeros((positions.shape[0], self.embed_dim))
        pe = pe.at[:, 0::2].add(cosines)
        pe = pe.at[:, 1::2].add(sines)

        # Add bias term in last position
        pe = pe.at[:, -1].add(1.0)

        return pe

    def __call__(self, x: mx.array, left_context_len: int = 0) -> mx.array:
        """
        Compute positional encoding.

        Args:
            x: Input tensor of shape (seq_len, batch_size, embed_dim)
            left_context_len: Length of cached left context (for streaming)

        Returns:
            Positional encoding of shape (1, left_context_len + 2*seq_len - 1, embed_dim)
        """
        seq_len = x.shape[0]
        total_len = seq_len + left_context_len

        # Recompute PE if needed
        if self._pe_cache is None or self._pe_cache_len < total_len:
            self._pe_cache = self._compute_pe(max(total_len, 1000))
            self._pe_cache_len = self._pe_cache.shape[0] // 2 + 1

        # Extract relevant portion
        # PE array has positions [-(T-1), T-1] for length T
        # We need positions [-(seq_len-1), total_len-1]
        center = self._pe_cache.shape[0] // 2
        start = center - (total_len - 1)
        end = center + seq_len

        pos_emb = self._pe_cache[start:end]  # (left_context_len + 2*seq_len - 1, embed_dim)
        pos_emb = mx.expand_dims(pos_emb, axis=0)  # (1, ..., embed_dim)

        return pos_emb


class RelPositionMultiheadAttentionWeights(nn.Module):
    """
    MLX implementation of attention weight computation with relative position encoding.

    Computes attention weights that can be consumed by SelfAttention and NonlinAttention.

    Args:
        embed_dim: Input embedding dimension.
        pos_dim: Positional encoding dimension.
        num_heads: Number of attention heads.
        query_head_dim: Query/key dimension per head.
        pos_head_dim: Positional encoding dimension per head.
    """

    def __init__(
        self,
        embed_dim: int,
        pos_dim: int,
        num_heads: int,
        query_head_dim: int,
        pos_head_dim: int,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.query_head_dim = query_head_dim
        self.pos_head_dim = pos_head_dim

        key_head_dim = query_head_dim
        in_proj_dim = (query_head_dim + key_head_dim + pos_head_dim) * num_heads

        # Input projection for Q, K, P
        self.in_proj = ScaledLinear(
            embed_dim, in_proj_dim, bias=True, initial_scale=query_head_dim**-0.25,
        )

        # Positional encoding projection
        self.linear_pos = ScaledLinear(
            pos_dim, num_heads * pos_head_dim, bias=False, initial_scale=0.05,
        )

        # Training regularization (identity during inference)
        self.whiten_keys = Whiten(num_groups=num_heads)
        self.balance_keys = Balancer(
            key_head_dim * num_heads,
            channel_dim=-1,
            min_positive=0.4,
            max_positive=0.6,
        )

    def __call__(
        self,
        x: mx.array,
        pos_emb: mx.array,
        key_padding_mask: mx.array | None = None,
        attn_mask: mx.array | None = None,
    ) -> mx.array:
        """
        Compute attention weights.

        Args:
            x: Input of shape (seq_len, batch_size, embed_dim)
            pos_emb: Positional encoding of shape (1, 2*seq_len-1, pos_dim)
            key_padding_mask: Padding mask of shape (batch_size, seq_len), True = masked
            attn_mask: Attention mask of shape (seq_len, seq_len), True = masked

        Returns:
            Attention weights of shape (num_heads, batch_size, seq_len, seq_len)
        """
        x = self.in_proj(x)

        seq_len, batch_size, _ = x.shape
        query_dim = self.query_head_dim * self.num_heads

        # Split into Q, K, P
        q = x[:, :, :query_dim]
        k = x[:, :, query_dim : 2 * query_dim]
        p = x[:, :, 2 * query_dim :]

        # Apply regularization (identity during inference)
        k = self.whiten_keys(self.balance_keys(k))

        # Reshape for multi-head attention
        q = mx.reshape(q, (seq_len, batch_size, self.num_heads, self.query_head_dim))
        k = mx.reshape(k, (seq_len, batch_size, self.num_heads, self.query_head_dim))
        p = mx.reshape(p, (seq_len, batch_size, self.num_heads, self.pos_head_dim))

        # Permute: (seq, batch, heads, dim) -> (heads, batch, seq, dim)
        q = mx.transpose(q, (2, 1, 0, 3))  # (heads, batch, seq, query_dim)
        p = mx.transpose(p, (2, 1, 0, 3))  # (heads, batch, seq, pos_dim)
        k = mx.transpose(k, (2, 1, 3, 0))  # (heads, batch, query_dim, seq)

        # Content-based attention scores
        attn_scores = mx.matmul(q, k)  # (heads, batch, seq, seq)

        # Position-based attention scores
        pos_emb = self.linear_pos(pos_emb)  # (1, 2*seq-1, heads*pos_dim)
        seq_len2 = 2 * seq_len - 1
        pos_emb = mx.reshape(
            pos_emb, (-1, seq_len2, self.num_heads, self.pos_head_dim),
        )
        pos_emb = mx.transpose(pos_emb, (2, 0, 3, 1))  # (heads, 1, pos_dim, 2*seq-1)

        # (heads, batch, seq, pos_dim) @ (heads, 1, pos_dim, 2*seq-1) -> (heads, batch, seq, 2*seq-1)
        pos_scores = mx.matmul(p, pos_emb)

        # Convert relative to absolute positions using strided indexing
        # This extracts the correct relative position for each (query, key) pair
        # We need to implement the as_strided behavior
        pos_scores_abs = self._rel_to_abs_pos(pos_scores, seq_len)

        attn_scores = attn_scores + pos_scores_abs

        # Apply masks
        if attn_mask is not None:
            # attn_mask: (seq, seq), True = masked
            attn_scores = mx.where(
                attn_mask,
                mx.full(attn_scores.shape, -1000.0),
                attn_scores,
            )

        if key_padding_mask is not None:
            # key_padding_mask: (batch, seq), True = masked
            mask = mx.expand_dims(mx.expand_dims(key_padding_mask, 0), 0)
            attn_scores = mx.where(
                mask,
                mx.full(attn_scores.shape, -1000.0),
                attn_scores,
            )

        # Softmax
        attn_weights = mx.softmax(attn_scores, axis=-1)

        return attn_weights

    def _rel_to_abs_pos(self, pos_scores: mx.array, seq_len: int) -> mx.array:
        """
        Convert relative position scores to absolute position indexing.

        Args:
            pos_scores: Shape (heads, batch, seq, 2*seq-1)
            seq_len: Sequence length

        Returns:
            Scores of shape (heads, batch, seq, seq)
        """
        num_heads, batch_size, time1, rel_len = pos_scores.shape

        # For query position i (0 to seq_len-1), key position j (0 to seq_len-1):
        # Relative position = j - i ranges from -(seq_len-1) to (seq_len-1)
        # In the relative array of length 2*seq_len-1:
        # - Index 0 corresponds to relative position -(seq_len-1)
        # - Index seq_len-1 corresponds to relative position 0
        # - Index 2*seq_len-2 corresponds to relative position (seq_len-1)
        # So: array_index = (j - i) + (seq_len - 1) = j - i + seq_len - 1

        # Create index array where indices[i, j] = j - i + seq_len - 1
        query_indices = mx.arange(seq_len)  # [0, 1, ..., seq_len-1]
        key_indices = mx.arange(seq_len)  # [0, 1, ..., seq_len-1]

        # indices[i, j] = key_indices[j] - query_indices[i] + seq_len - 1
        indices = (
            mx.expand_dims(key_indices, axis=0)
            - mx.expand_dims(query_indices, axis=1)
            + (seq_len - 1)
        )
        # indices shape: (seq_len, seq_len)

        # Use mx.take_along_axis to gather along the last dimension
        # We need to expand indices to match pos_scores shape
        indices_expanded = mx.broadcast_to(
            mx.reshape(indices, (1, 1, seq_len, seq_len)),
            (num_heads, batch_size, seq_len, seq_len),
        )

        # Gather along last axis
        result = mx.take_along_axis(pos_scores, indices_expanded, axis=-1)

        return result

    def streaming_forward(
        self,
        x: mx.array,
        pos_emb: mx.array,
        cached_key: mx.array,
        left_context_len: int,
        key_padding_mask: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """
        Streaming forward pass.

        Args:
            x: Input of shape (seq_len, batch_size, embed_dim)
            pos_emb: Positional encoding
            cached_key: Cached keys of shape (left_context_len, batch_size, key_dim)
            left_context_len: Number of cached frames
            key_padding_mask: Padding mask

        Returns:
            - Attention weights of shape (num_heads, batch_size, seq_len, seq_len+left_context)
            - Updated cached_key
        """
        x = self.in_proj(x)

        seq_len, batch_size, _ = x.shape
        query_dim = self.query_head_dim * self.num_heads

        # Split into Q, K, P
        q = x[:, :, :query_dim]
        k = x[:, :, query_dim : 2 * query_dim]
        p = x[:, :, 2 * query_dim :]

        # Concatenate cached keys
        k = mx.concatenate([cached_key, k], axis=0)
        new_cached_key = k[-left_context_len:]

        k_len = k.shape[0]

        # Reshape
        q = mx.reshape(q, (seq_len, batch_size, self.num_heads, self.query_head_dim))
        k = mx.reshape(k, (k_len, batch_size, self.num_heads, self.query_head_dim))
        p = mx.reshape(p, (seq_len, batch_size, self.num_heads, self.pos_head_dim))

        # Permute
        q = mx.transpose(q, (2, 1, 0, 3))
        p = mx.transpose(p, (2, 1, 0, 3))
        k = mx.transpose(k, (2, 1, 3, 0))

        # Content attention
        attn_scores = mx.matmul(q, k)

        # Position attention
        pos_emb = self.linear_pos(pos_emb)
        seq_len2 = 2 * seq_len - 1 + left_context_len
        pos_emb = mx.reshape(
            pos_emb, (-1, seq_len2, self.num_heads, self.pos_head_dim),
        )
        pos_emb = mx.transpose(pos_emb, (2, 0, 3, 1))

        pos_scores = mx.matmul(p, pos_emb)
        pos_scores_abs = self._rel_to_abs_pos_streaming(pos_scores, seq_len, k_len)

        attn_scores = attn_scores + pos_scores_abs

        # Apply mask
        if key_padding_mask is not None:
            mask = mx.expand_dims(mx.expand_dims(key_padding_mask, 0), 0)
            attn_scores = mx.where(
                mask,
                mx.full(attn_scores.shape, -1000.0),
                attn_scores,
            )

        attn_weights = mx.softmax(attn_scores, axis=-1)

        return attn_weights, new_cached_key

    def _rel_to_abs_pos_streaming(
        self, pos_scores: mx.array, seq_len: int, k_len: int,
    ) -> mx.array:
        """Convert relative to absolute positions for streaming case."""
        num_heads, batch_size, time1, rel_len = pos_scores.shape

        # For streaming: query indices are [0, seq_len-1], key indices are [0, k_len-1]
        # Relative position = key_pos - query_pos
        # In streaming, rel_len = 2*seq_len - 1 + left_context_len
        # The center (query_pos=0, key_pos=0) is at index seq_len - 1

        query_indices = mx.arange(seq_len)
        key_indices = mx.arange(k_len)

        # indices[i, j] = key_indices[j] - query_indices[i] + (seq_len - 1)
        indices = (
            mx.expand_dims(key_indices, axis=0)
            - mx.expand_dims(query_indices, axis=1)
            + (seq_len - 1)
        )
        # indices shape: (seq_len, k_len)

        indices_expanded = mx.broadcast_to(
            mx.reshape(indices, (1, 1, seq_len, k_len)),
            (num_heads, batch_size, seq_len, k_len),
        )

        result = mx.take_along_axis(pos_scores, indices_expanded, axis=-1)

        return result


class SelfAttention(nn.Module):
    """
    MLX implementation of SelfAttention from Zipformer2.

    Uses precomputed attention weights to compute weighted values.

    Args:
        embed_dim: Input/output embedding dimension.
        num_heads: Number of attention heads.
        value_head_dim: Value dimension per head.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        value_head_dim: int,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.value_head_dim = value_head_dim

        self.in_proj = nn.Linear(embed_dim, num_heads * value_head_dim, bias=True)
        self.out_proj = ScaledLinear(
            num_heads * value_head_dim, embed_dim, bias=True, initial_scale=0.05,
        )
        self.whiten = Whiten(num_groups=1)

    def __call__(
        self,
        x: mx.array,
        attn_weights: mx.array,
    ) -> mx.array:
        """
        Forward pass.

        Args:
            x: Input of shape (seq_len, batch_size, embed_dim)
            attn_weights: Attention weights of shape (num_heads, batch_size, seq_len, seq_len)

        Returns:
            Output of shape (seq_len, batch_size, embed_dim)
        """
        seq_len, batch_size, embed_dim = x.shape

        x = self.in_proj(x)  # (seq_len, batch_size, num_heads * value_head_dim)

        # Reshape for attention
        x = mx.reshape(x, (seq_len, batch_size, self.num_heads, self.value_head_dim))
        x = mx.transpose(x, (2, 1, 0, 3))  # (num_heads, batch_size, seq_len, value_head_dim)

        # Apply attention weights
        x = mx.matmul(attn_weights, x)  # (num_heads, batch_size, seq_len, value_head_dim)

        # Reshape back
        x = mx.transpose(x, (2, 1, 0, 3))  # (seq_len, batch_size, num_heads, value_head_dim)
        x = mx.reshape(x, (seq_len, batch_size, self.num_heads * self.value_head_dim))

        x = self.out_proj(x)
        x = self.whiten(x)

        return x

    def streaming_forward(
        self,
        x: mx.array,
        attn_weights: mx.array,
        cached_val: mx.array,
        left_context_len: int,
    ) -> tuple[mx.array, mx.array]:
        """
        Streaming forward pass.

        Args:
            x: Input of shape (seq_len, batch_size, embed_dim)
            attn_weights: Shape (num_heads, batch_size, seq_len, seq_len + left_context_len)
            cached_val: Cached values of shape (left_context_len, batch_size, value_dim)
            left_context_len: Number of cached frames

        Returns:
            - Output of shape (seq_len, batch_size, embed_dim)
            - Updated cached values
        """
        seq_len, batch_size, embed_dim = x.shape

        x = self.in_proj(x)

        # Concatenate with cache
        x = mx.concatenate([cached_val, x], axis=0)
        new_cached_val = x[-left_context_len:]

        seq_len2 = x.shape[0]

        # Reshape
        x = mx.reshape(x, (seq_len2, batch_size, self.num_heads, self.value_head_dim))
        x = mx.transpose(x, (2, 1, 0, 3))

        # Apply attention
        x = mx.matmul(attn_weights, x)

        # Reshape back
        x = mx.transpose(x, (2, 1, 0, 3))
        x = mx.reshape(x, (seq_len, batch_size, self.num_heads * self.value_head_dim))

        x = self.out_proj(x)

        return x, new_cached_val


class FeedforwardModule(nn.Module):
    """
    MLX implementation of FeedforwardModule from Zipformer2.

    Uses SwooshL activation between two linear layers.

    Args:
        embed_dim: Input/output dimension.
        feedforward_dim: Hidden dimension.
    """

    def __init__(
        self,
        embed_dim: int,
        feedforward_dim: int,
    ):
        super().__init__()
        self.in_proj = nn.Linear(embed_dim, feedforward_dim)
        self.hidden_balancer = Balancer(
            feedforward_dim,
            channel_dim=-1,
            min_positive=0.3,
            max_positive=1.0,
        )
        self.out_proj = ActivationDropoutAndLinear(
            feedforward_dim,
            embed_dim,
            activation="SwooshL",
        )
        self.out_whiten = Whiten(num_groups=1)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass."""
        x = self.in_proj(x)
        x = self.hidden_balancer(x)
        x = self.out_proj(x)
        x = self.out_whiten(x)
        return x


class NonlinAttention(nn.Module):
    """
    MLX implementation of NonlinAttention from Zipformer2.

    Uses attention weights to perform a convolution-like operation
    with nonlinear gating.

    Args:
        channels: Number of input/output channels.
        hidden_channels: Number of hidden channels.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels

        self.in_proj = nn.Linear(channels, hidden_channels * 3, bias=True)
        self.balancer = Balancer(hidden_channels, channel_dim=-1)
        self.out_proj = ScaledLinear(
            hidden_channels, channels, bias=True, initial_scale=0.05,
        )
        self.whiten1 = Whiten(num_groups=1)
        self.whiten2 = Whiten(num_groups=1)

    def __call__(
        self,
        x: mx.array,
        attn_weights: mx.array,
    ) -> mx.array:
        """
        Forward pass.

        Args:
            x: Input of shape (seq_len, batch_size, channels)
            attn_weights: Shape (num_heads, batch_size, seq_len, seq_len)

        Returns:
            Output of shape (seq_len, batch_size, channels)
        """
        x = self.in_proj(x)

        seq_len, batch_size, _ = x.shape
        num_heads = attn_weights.shape[0]

        # Split into s (gate), x (main), y (output gate)
        s, x, y = mx.split(x, 3, axis=2)

        # Apply tanh to s
        s = self.balancer(s)
        s = mx.tanh(s)

        # Gate x with s
        x = self.whiten1(x)
        x = x * s

        # Reshape for attention
        head_dim = self.hidden_channels // num_heads
        x = mx.reshape(x, (seq_len, batch_size, num_heads, head_dim))
        x = mx.transpose(x, (2, 1, 0, 3))  # (num_heads, batch, seq, head_dim)

        # Apply attention
        x = mx.matmul(attn_weights, x)

        # Reshape back
        x = mx.transpose(x, (2, 1, 0, 3))
        x = mx.reshape(x, (seq_len, batch_size, self.hidden_channels))

        # Final gating with y
        x = x * y

        x = self.out_proj(x)
        x = self.whiten2(x)

        return x

    def streaming_forward(
        self,
        x: mx.array,
        attn_weights: mx.array,
        cached_x: mx.array,
        left_context_len: int,
    ) -> tuple[mx.array, mx.array]:
        """Streaming forward pass."""
        x = self.in_proj(x)

        seq_len, batch_size, _ = x.shape
        num_heads = attn_weights.shape[0]

        s, x, y = mx.split(x, 3, axis=2)
        s = mx.tanh(s)
        x = x * s

        head_dim = self.hidden_channels // num_heads
        x = mx.reshape(x, (seq_len, batch_size, num_heads, head_dim))
        x = mx.transpose(x, (2, 1, 0, 3))

        # Concatenate with cache
        x = mx.concatenate([cached_x, x], axis=2)
        new_cached_x = x[:, :, -left_context_len:, :]

        # Apply attention
        x = mx.matmul(attn_weights, x)

        x = mx.transpose(x, (2, 1, 0, 3))
        x = mx.reshape(x, (seq_len, batch_size, self.hidden_channels))

        x = x * y
        x = self.out_proj(x)

        return x, new_cached_x


class ConvolutionModule(nn.Module):
    """
    MLX implementation of ConvolutionModule from Zipformer2.

    Depthwise convolution with gating and optional causal mode.

    Args:
        channels: Number of input/output channels.
        kernel_size: Convolution kernel size (must be odd).
        causal: Whether to use causal convolution.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        causal: bool = False,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"

        self.channels = channels
        self.kernel_size = kernel_size
        self.causal = causal

        self.in_proj = nn.Linear(channels, 2 * channels)
        self.balancer1 = Balancer(channels, channel_dim=-1)

        if causal:
            self.depthwise_conv = ChunkCausalDepthwiseConv1d(
                channels=channels, kernel_size=kernel_size,
            )
        else:
            # Standard depthwise conv
            # MLX conv1d weight shape: (C_out, K, C_in) with C_in=1 for depthwise
            self.depthwise_conv_weight = mx.random.normal(
                shape=(channels, kernel_size, 1),
            ) * 0.1
            self.depthwise_conv_bias = mx.zeros((channels,))

        self.balancer2 = Balancer(channels, channel_dim=1)
        self.whiten = Whiten(num_groups=1)
        self.out_proj = ActivationDropoutAndLinear(
            channels, channels, activation="SwooshR",
        )

    def _standard_depthwise_conv(self, x: mx.array) -> mx.array:
        """Standard depthwise conv with symmetric padding."""
        batch, channels, time = x.shape
        padding = self.kernel_size // 2

        # Transpose to MLX format: (batch, time, channels)
        x = mx.transpose(x, (0, 2, 1))

        if padding > 0:
            x = mx.pad(x, [(0, 0), (padding, padding), (0, 0)])

        # Use groups=channels for depthwise convolution
        out = mx.conv1d(x, self.depthwise_conv_weight, groups=channels)

        # Transpose back to (batch, channels, time)
        out = mx.transpose(out, (0, 2, 1))

        # Add bias
        out = out + mx.reshape(self.depthwise_conv_bias, (1, channels, 1))

        return out

    def __call__(
        self,
        x: mx.array,
        src_key_padding_mask: mx.array | None = None,
        chunk_size: int = -1,
    ) -> mx.array:
        """
        Forward pass.

        Args:
            x: Input of shape (seq_len, batch_size, channels)
            src_key_padding_mask: Padding mask of shape (batch_size, seq_len)
            chunk_size: Chunk size for causal mode

        Returns:
            Output of shape (seq_len, batch_size, channels)
        """
        x = self.in_proj(x)  # (seq_len, batch, 2*channels)

        # Split into main path and gate
        x, s = mx.split(x, 2, axis=2)
        s = self.balancer1(s)
        s = mx.sigmoid(s)
        x = x * s

        # Permute to (batch, channels, time)
        x = mx.transpose(x, (1, 2, 0))

        # Apply padding mask
        if src_key_padding_mask is not None:
            mask = mx.expand_dims(src_key_padding_mask, 1)  # (batch, 1, seq)
            x = mx.where(mask, mx.zeros_like(x), x)

        # Depthwise convolution
        if self.causal:
            x = self.depthwise_conv(x, chunk_size=chunk_size)
        else:
            x = self._standard_depthwise_conv(x)

        x = self.balancer2(x)

        # Permute back to (time, batch, channels)
        x = mx.transpose(x, (2, 0, 1))

        x = self.whiten(x)
        x = self.out_proj(x)

        return x

    def streaming_forward(
        self,
        x: mx.array,
        cache: mx.array,
        src_key_padding_mask: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Streaming forward pass."""
        x = self.in_proj(x)

        x, s = mx.split(x, 2, axis=2)
        s = mx.sigmoid(s)
        x = x * s

        x = mx.transpose(x, (1, 2, 0))

        if src_key_padding_mask is not None:
            mask = mx.expand_dims(src_key_padding_mask, 1)
            x = mx.where(mask, mx.zeros_like(x), x)

        x, cache = self.depthwise_conv.streaming_forward(x, cache)

        x = mx.transpose(x, (2, 0, 1))
        x = self.out_proj(x)

        return x, cache


class BypassModule(nn.Module):
    """
    MLX implementation of BypassModule from Zipformer2.

    Implements learnable residual scaling for layer skip connections.

    Args:
        embed_dim: Embedding dimension.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.bypass_scale = mx.full((embed_dim,), 0.5)

    def __call__(self, src_orig: mx.array, src: mx.array) -> mx.array:
        """
        Apply bypass.

        Args:
            src_orig: Original input (seq_len, batch_size, embed_dim)
            src: Transformed input (seq_len, batch_size, embed_dim)

        Returns:
            Blended output
        """
        return src_orig + (src - src_orig) * self.bypass_scale


class Zipformer2EncoderLayer(nn.Module):
    """
    MLX implementation of Zipformer2EncoderLayer.

    A single encoder layer consisting of:
    - Self-attention with relative positional encoding
    - NonlinAttention
    - Two convolution modules
    - Three feedforward modules
    - Bypass connections

    Args:
        embed_dim: Embedding dimension.
        pos_dim: Positional encoding dimension.
        num_heads: Number of attention heads.
        query_head_dim: Query/key dimension per head.
        pos_head_dim: Positional encoding dimension per head.
        value_head_dim: Value dimension per head.
        feedforward_dim: Feedforward hidden dimension.
        cnn_module_kernel: Convolution kernel size.
        causal: Whether to use causal convolutions.
    """

    def __init__(
        self,
        embed_dim: int,
        pos_dim: int,
        num_heads: int,
        query_head_dim: int,
        pos_head_dim: int,
        value_head_dim: int,
        feedforward_dim: int,
        cnn_module_kernel: int = 31,
        causal: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Bypass modules
        self.bypass = BypassModule(embed_dim)
        self.bypass_mid = BypassModule(embed_dim)

        # Attention weight computation
        self.self_attn_weights = RelPositionMultiheadAttentionWeights(
            embed_dim=embed_dim,
            pos_dim=pos_dim,
            num_heads=num_heads,
            query_head_dim=query_head_dim,
            pos_head_dim=pos_head_dim,
        )

        # Two self-attention modules
        self.self_attn1 = SelfAttention(embed_dim, num_heads, value_head_dim)
        self.self_attn2 = SelfAttention(embed_dim, num_heads, value_head_dim)

        # Three feedforward modules
        self.feed_forward1 = FeedforwardModule(
            embed_dim, (feedforward_dim * 3) // 4,
        )
        self.feed_forward2 = FeedforwardModule(embed_dim, feedforward_dim)
        self.feed_forward3 = FeedforwardModule(
            embed_dim, (feedforward_dim * 5) // 4,
        )

        # NonlinAttention
        self.nonlin_attention = NonlinAttention(
            embed_dim, hidden_channels=3 * embed_dim // 4,
        )

        # Two convolution modules
        self.conv_module1 = ConvolutionModule(
            embed_dim, cnn_module_kernel, causal=causal,
        )
        self.conv_module2 = ConvolutionModule(
            embed_dim, cnn_module_kernel, causal=causal,
        )

        # Normalization and regularization
        self.norm = BiasNorm(embed_dim)
        self.balancer1 = Balancer(embed_dim, channel_dim=-1)
        self.balancer2 = Balancer(embed_dim, channel_dim=-1)
        self.balancer_na = Balancer(embed_dim, channel_dim=-1)
        self.balancer_ff2 = Balancer(embed_dim, channel_dim=-1)
        self.balancer_ff3 = Balancer(embed_dim, channel_dim=-1)
        self.whiten = Whiten(num_groups=1)

    def __call__(
        self,
        src: mx.array,
        pos_emb: mx.array,
        chunk_size: int = -1,
        attn_mask: mx.array | None = None,
        src_key_padding_mask: mx.array | None = None,
    ) -> mx.array:
        """
        Forward pass through encoder layer.

        Args:
            src: Input of shape (seq_len, batch_size, embed_dim)
            pos_emb: Positional encoding
            chunk_size: Chunk size for causal mode
            attn_mask: Attention mask (seq_len, seq_len)
            src_key_padding_mask: Padding mask (batch_size, seq_len)

        Returns:
            Output of shape (seq_len, batch_size, embed_dim)
        """
        src_orig = src

        # Compute attention weights (shared across attention modules)
        attn_weights = self.self_attn_weights(
            src,
            pos_emb=pos_emb,
            attn_mask=attn_mask,
            key_padding_mask=src_key_padding_mask,
        )

        # First feedforward
        src = src + self.feed_forward1(src)

        # NonlinAttention (uses first head of attention weights)
        na = self.balancer_na(
            self.nonlin_attention(src, attn_weights[0:1]),
        )
        src = src + na

        # First self-attention
        self_attn = self.self_attn1(src, attn_weights)
        src = src + self_attn

        # First convolution
        src = src + self.conv_module1(
            src, src_key_padding_mask=src_key_padding_mask, chunk_size=chunk_size,
        )

        # Second feedforward
        src = src + self.balancer_ff2(self.feed_forward2(src))

        # Mid-layer bypass
        src = self.bypass_mid(src_orig, src)

        # Second self-attention
        self_attn = self.self_attn2(src, attn_weights)
        src = src + self_attn

        # Second convolution
        src = src + self.conv_module2(
            src, src_key_padding_mask=src_key_padding_mask, chunk_size=chunk_size,
        )

        # Third feedforward
        src = src + self.balancer_ff3(self.feed_forward3(src))

        # Normalize and apply bypass
        src = self.balancer1(src)
        src = self.norm(src)
        src = self.bypass(src_orig, src)
        src = self.balancer2(src)
        src = self.whiten(src)

        return src

    def streaming_forward(
        self,
        src: mx.array,
        pos_emb: mx.array,
        cached_key: mx.array,
        cached_nonlin_attn: mx.array,
        cached_val1: mx.array,
        cached_val2: mx.array,
        cached_conv1: mx.array,
        cached_conv2: mx.array,
        left_context_len: int,
        src_key_padding_mask: mx.array | None = None,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
        """
        Streaming forward pass.

        Returns:
            - Output
            - Updated cached_key
            - Updated cached_nonlin_attn
            - Updated cached_val1
            - Updated cached_val2
            - Updated cached_conv1
            - Updated cached_conv2
        """
        src_orig = src

        # Compute attention weights with cached keys
        attn_weights, cached_key = self.self_attn_weights.streaming_forward(
            src,
            pos_emb=pos_emb,
            cached_key=cached_key,
            left_context_len=left_context_len,
            key_padding_mask=src_key_padding_mask,
        )

        # First feedforward
        src = src + self.feed_forward1(src)

        # NonlinAttention with cache
        na, cached_nonlin_attn = self.nonlin_attention.streaming_forward(
            src,
            attn_weights[0:1],
            cached_x=cached_nonlin_attn,
            left_context_len=left_context_len,
        )
        src = src + na

        # First self-attention with cache
        self_attn, cached_val1 = self.self_attn1.streaming_forward(
            src,
            attn_weights=attn_weights,
            cached_val=cached_val1,
            left_context_len=left_context_len,
        )
        src = src + self_attn

        # First convolution with cache
        src_conv, cached_conv1 = self.conv_module1.streaming_forward(
            src,
            cache=cached_conv1,
            src_key_padding_mask=(
                src_key_padding_mask[:, left_context_len:]
                if src_key_padding_mask is not None
                else None
            ),
        )
        src = src + src_conv

        # Second feedforward
        src = src + self.feed_forward2(src)

        # Mid-layer bypass
        src = self.bypass_mid(src_orig, src)

        # Second self-attention with cache
        self_attn, cached_val2 = self.self_attn2.streaming_forward(
            src,
            attn_weights=attn_weights,
            cached_val=cached_val2,
            left_context_len=left_context_len,
        )
        src = src + self_attn

        # Second convolution with cache
        src_conv, cached_conv2 = self.conv_module2.streaming_forward(
            src,
            cache=cached_conv2,
            src_key_padding_mask=(
                src_key_padding_mask[:, left_context_len:]
                if src_key_padding_mask is not None
                else None
            ),
        )
        src = src + src_conv

        # Third feedforward
        src = src + self.feed_forward3(src)

        # Normalize and bypass
        src = self.norm(src)
        src = self.bypass(src_orig, src)

        return (
            src,
            cached_key,
            cached_nonlin_attn,
            cached_val1,
            cached_val2,
            cached_conv1,
            cached_conv2,
        )


class Zipformer2Encoder(nn.Module):
    """
    MLX implementation of Zipformer2Encoder.

    A stack of Zipformer2EncoderLayers with positional encoding.

    Args:
        encoder_layer: Template encoder layer to clone.
        num_layers: Number of layers.
        pos_dim: Positional encoding dimension.
    """

    def __init__(
        self,
        encoder_layer: Zipformer2EncoderLayer,
        num_layers: int,
        pos_dim: int,
    ):
        super().__init__()
        self.encoder_pos = CompactRelPositionalEncoding(pos_dim)
        self.num_layers = num_layers

        # Create layers (note: in MLX we need to create them individually)
        # For now, store the template and create copies
        self.layers = [encoder_layer]
        for _ in range(num_layers - 1):
            # Create new layer with same config
            layer = Zipformer2EncoderLayer(
                embed_dim=encoder_layer.embed_dim,
                pos_dim=pos_dim,
                num_heads=encoder_layer.self_attn_weights.num_heads,
                query_head_dim=encoder_layer.self_attn_weights.query_head_dim,
                pos_head_dim=encoder_layer.self_attn_weights.pos_head_dim,
                value_head_dim=encoder_layer.self_attn1.value_head_dim,
                feedforward_dim=encoder_layer.feed_forward2.in_proj.weight.shape[1],
                cnn_module_kernel=encoder_layer.conv_module1.kernel_size,
                causal=encoder_layer.conv_module1.causal,
            )
            self.layers.append(layer)

    def __call__(
        self,
        src: mx.array,
        chunk_size: int = -1,
        feature_mask: mx.array | float = 1.0,
        attn_mask: mx.array | None = None,
        src_key_padding_mask: mx.array | None = None,
    ) -> mx.array:
        """
        Forward pass through encoder.

        Args:
            src: Input of shape (seq_len, batch_size, embed_dim)
            chunk_size: Chunk size for causal mode
            feature_mask: Feature dropout mask
            attn_mask: Attention mask
            src_key_padding_mask: Padding mask

        Returns:
            Output of shape (seq_len, batch_size, embed_dim)
        """
        pos_emb = self.encoder_pos(src)
        output = src

        # Apply feature mask
        if isinstance(feature_mask, mx.array):
            output = output * feature_mask

        for layer in self.layers:
            output = layer(
                output,
                pos_emb,
                chunk_size=chunk_size,
                attn_mask=attn_mask,
                src_key_padding_mask=src_key_padding_mask,
            )
            if isinstance(feature_mask, mx.array):
                output = output * feature_mask

        return output

    def streaming_forward(
        self,
        src: mx.array,
        states: list[mx.array],
        left_context_len: int,
        src_key_padding_mask: mx.array | None = None,
    ) -> tuple[mx.array, list[mx.array]]:
        """
        Streaming forward pass.

        Args:
            src: Input of shape (seq_len, batch_size, embed_dim)
            states: List of cached tensors (6 per layer)
            left_context_len: Number of cached frames
            src_key_padding_mask: Padding mask

        Returns:
            - Output
            - Updated states
        """
        pos_emb = self.encoder_pos(src, left_context_len)
        output = src

        new_states = []
        for i, layer in enumerate(self.layers):
            (
                cached_key,
                cached_nonlin_attn,
                cached_val1,
                cached_val2,
                cached_conv1,
                cached_conv2,
            ) = states[i * 6 : (i + 1) * 6]

            (
                output,
                new_cached_key,
                new_cached_nonlin_attn,
                new_cached_val1,
                new_cached_val2,
                new_cached_conv1,
                new_cached_conv2,
            ) = layer.streaming_forward(
                output,
                pos_emb,
                cached_key=cached_key,
                cached_nonlin_attn=cached_nonlin_attn,
                cached_val1=cached_val1,
                cached_val2=cached_val2,
                cached_conv1=cached_conv1,
                cached_conv2=cached_conv2,
                left_context_len=left_context_len,
                src_key_padding_mask=src_key_padding_mask,
            )

            new_states.extend([
                new_cached_key,
                new_cached_nonlin_attn,
                new_cached_val1,
                new_cached_val2,
                new_cached_conv1,
                new_cached_conv2,
            ])

        return output, new_states


class SimpleDownsample(nn.Module):
    """
    MLX implementation of SimpleDownsample.

    Downsamples by weighted sum with learned attention weights.

    Args:
        channels: Number of channels.
        downsample: Downsampling factor.
    """

    def __init__(self, channels: int, downsample: int):
        super().__init__()
        self.downsample = downsample
        self.bias = mx.zeros((downsample,))

    def __call__(self, src: mx.array) -> mx.array:
        """
        Forward pass.

        Args:
            src: Input of shape (seq_len, batch_size, channels)

        Returns:
            Output of shape ((seq_len + downsample - 1) // downsample, batch_size, channels)
        """
        seq_len, batch_size, channels = src.shape
        ds = self.downsample
        d_seq_len = (seq_len + ds - 1) // ds

        # Pad to multiple of downsample
        pad = d_seq_len * ds - seq_len
        if pad > 0:
            src_extra = mx.broadcast_to(
                src[-1:], (pad, batch_size, channels),
            )
            src = mx.concatenate([src, src_extra], axis=0)

        # Reshape: (d_seq_len, ds, batch, channels)
        src = mx.reshape(src, (d_seq_len, ds, batch_size, channels))

        # Compute attention weights
        weights = mx.softmax(self.bias, axis=0)  # (ds,)
        weights = mx.reshape(weights, (1, ds, 1, 1))

        # Weighted sum
        output = mx.sum(src * weights, axis=1)  # (d_seq_len, batch, channels)

        return output


class SimpleUpsample(nn.Module):
    """
    MLX implementation of SimpleUpsample.

    Upsamples by repeating frames.

    Args:
        num_channels: Number of channels.
        upsample: Upsampling factor.
    """

    def __init__(self, num_channels: int, upsample: int):
        super().__init__()
        self.upsample = upsample

    def __call__(self, src: mx.array) -> mx.array:
        """
        Forward pass.

        Args:
            src: Input of shape (seq_len, batch_size, channels)

        Returns:
            Output of shape (seq_len * upsample, batch_size, channels)
        """
        seq_len, batch_size, num_channels = src.shape

        # Expand: (seq_len, 1, batch, channels) -> (seq_len, upsample, batch, channels)
        src = mx.expand_dims(src, axis=1)
        src = mx.broadcast_to(
            src, (seq_len, self.upsample, batch_size, num_channels),
        )

        # Reshape: (seq_len * upsample, batch, channels)
        src = mx.reshape(src, (seq_len * self.upsample, batch_size, num_channels))

        return src


class DownsampledZipformer2Encoder(nn.Module):
    """
    MLX implementation of DownsampledZipformer2Encoder.

    Wraps a Zipformer2Encoder with downsampling and upsampling.

    Args:
        encoder: The wrapped encoder.
        dim: Embedding dimension.
        downsample: Downsampling factor.
        causal: Whether using causal mode.
    """

    def __init__(
        self,
        encoder: Zipformer2Encoder,
        dim: int,
        downsample: int,
        causal: bool = False,
    ):
        super().__init__()
        self.downsample_factor = downsample
        self.downsample = SimpleDownsample(dim, downsample)
        self.num_layers = encoder.num_layers
        self.encoder = encoder
        self.upsample = SimpleUpsample(dim, downsample)
        self.out_combiner = BypassModule(dim)

    def __call__(
        self,
        src: mx.array,
        chunk_size: int = -1,
        feature_mask: mx.array | float = 1.0,
        attn_mask: mx.array | None = None,
        src_key_padding_mask: mx.array | None = None,
    ) -> mx.array:
        """Forward pass."""
        src_orig = src

        src = self.downsample(src)
        ds = self.downsample_factor

        if attn_mask is not None:
            attn_mask = attn_mask[::ds, ::ds]

        src = self.encoder(
            src,
            chunk_size=chunk_size // ds if chunk_size > 0 else chunk_size,
            feature_mask=feature_mask,
            attn_mask=attn_mask,
            src_key_padding_mask=src_key_padding_mask,
        )

        src = self.upsample(src)

        # Trim to original length
        src = src[: src_orig.shape[0]]

        return self.out_combiner(src_orig, src)

    def streaming_forward(
        self,
        src: mx.array,
        states: list[mx.array],
        left_context_len: int,
        src_key_padding_mask: mx.array | None = None,
    ) -> tuple[mx.array, list[mx.array]]:
        """Streaming forward pass."""
        src_orig = src
        src = self.downsample(src)

        src, new_states = self.encoder.streaming_forward(
            src,
            states=states,
            left_context_len=left_context_len,
            src_key_padding_mask=src_key_padding_mask,
        )

        src = self.upsample(src)
        src = src[: src_orig.shape[0]]

        return self.out_combiner(src_orig, src), new_states
