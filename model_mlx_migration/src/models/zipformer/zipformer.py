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
Full Zipformer model implementation for MLX.

Implements the complete multi-stage Zipformer encoder architecture including:
- Conv2dSubsampling (encoder_embed)
- Multi-resolution encoder stages with U-Net style downsampling
- Full pretrained weight loading support

Reference: k2-fsa/icefall streaming Zipformer
"""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .scaling import BiasNorm


@dataclass
class ZipformerConfig:
    """Configuration for Zipformer model.

    Default values match the icefall streaming checkpoint:
    k2-fsa/icefall/zipformer-streaming-librispeech
    """

    # Input features
    num_features: int = 80  # Mel spectrogram features

    # Encoder stages configuration
    # num_encoder_layers: tuple of layers per stage
    num_encoder_layers: tuple[int, ...] = (2, 2, 3, 4, 3, 2)

    # d_model per stage (encoder hidden dimension)
    encoder_dims: tuple[int, ...] = (192, 256, 384, 512, 384, 256)

    # Attention dimensions per stage (determines query/key head dimension)
    # Note: This is separate from d_model and typically smaller
    # query_head_dim = attention_dim // num_heads
    # From checkpoint: in_proj_dim = num_heads * (2 * query_head_dim + pos_dim)
    # Stage 0,1,2,4,5: in_proj=272 = 4*(2*32+4) -> attention_dim=128
    # Stage 3: in_proj=544 = 8*(2*32+4) -> attention_dim=256
    attention_dims: tuple[int, ...] = (128, 128, 128, 256, 128, 128)

    # Feedforward dimensions per stage
    # NOTE: Icefall Zipformer2 uses three feedforward modules per layer
    # (feed_forward1/2/3) and they may have different hidden dims in the
    # checkpoint (ff1/ff2/ff3). Historically this code used a single
    # `feedforward_dims` value per stage and reused it for all three modules.
    #
    # For backward compatibility:
    # - If ff{1,2,3}_dims are not provided, they default to feedforward_dims.
    feedforward_dims: tuple[int, ...] = (384, 576, 768, 1152, 768, 576)
    ff1_dims: tuple[int, ...] | None = None
    ff2_dims: tuple[int, ...] | None = None
    ff3_dims: tuple[int, ...] | None = None

    # Number of attention heads per stage
    num_heads: tuple[int, ...] = (4, 4, 4, 8, 4, 4)

    # Downsampling factors per stage
    downsampling_factors: tuple[int, ...] = (1, 2, 4, 8, 4, 2)

    # CNN module kernel sizes per stage (varies by stage in icefall checkpoint)
    cnn_module_kernels: tuple[int, ...] = (31, 31, 15, 15, 15, 31)

    # Positional encoding dimensions
    # pos_dim: dimension of the positional encoding vectors (before projection)
    # pos_head_dim: dimension per head after projection
    # linear_pos: (pos_dim, num_heads * pos_head_dim)
    pos_dim: int = 48  # Dimension of RelPositionalEncoding output
    pos_head_dim: int = 4  # Per-head positional dimension after projection

    # Value head dimension - fixed at 12 across all stages in icefall checkpoint
    value_head_dim: int = 12

    # Encoder embed configuration
    encoder_embed_dim: int = 128  # Output of conv stack before projection

    # Causal mode for streaming
    causal: bool = True

    def __post_init__(self) -> None:
        # Keep the legacy single-dim config working by mapping to ff1/ff2/ff3.
        if self.ff1_dims is None:
            self.ff1_dims = self.feedforward_dims
        if self.ff2_dims is None:
            self.ff2_dims = self.feedforward_dims
        if self.ff3_dims is None:
            self.ff3_dims = self.feedforward_dims


class ConvolutionModule(nn.Module):
    """
    Convolution module for Zipformer encoder layer.

    Uses depthwise separable convolution with gating.
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int = 31,
        causal: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.causal = causal

        # Input projection (doubles channels for gating)
        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=True)

        # Depthwise conv - for causal mode uses chunk-causal structure
        # We'll store weights directly for causal conv
        if causal:
            half_kernel = (kernel_size + 1) // 2
            # Causal conv (looks at past)
            self.causal_conv_weight = mx.zeros((d_model, half_kernel, 1))
            self.causal_conv_bias = mx.zeros((d_model,))
            # Chunkwise conv (within chunk)
            self.chunkwise_conv_weight = mx.zeros((d_model, kernel_size, 1))
            self.chunkwise_conv_bias = mx.zeros((d_model,))
            self.chunkwise_conv_scale = mx.zeros((2, d_model, kernel_size))
        else:
            self.depthwise_conv_weight = mx.zeros((d_model, kernel_size, 1))
            self.depthwise_conv_bias = mx.zeros((d_model,))

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def _get_chunk_scale(self, chunk_size: int) -> mx.array:
        """Get chunk edge scaling factors.

        The chunkwise convolution output needs to be scaled at chunk boundaries
        to account for edge effects. The scale is learned during training.

        Args:
            chunk_size: Size of the current chunk (seq_len)

        Returns:
            Scale tensor of shape (1, chunk_size, d_model)
        """
        left_edge = self.chunkwise_conv_scale[0]   # (d_model, kernel_size)
        right_edge = self.chunkwise_conv_scale[1]  # (d_model, kernel_size)

        if chunk_size < self.kernel_size:
            left_edge = left_edge[:, :chunk_size]
            right_edge = right_edge[:, -chunk_size:]
        else:
            t = chunk_size - self.kernel_size
            channels = left_edge.shape[0]
            pad = mx.zeros((channels, t))
            left_edge = mx.concatenate([left_edge, pad], axis=-1)
            right_edge = mx.concatenate([pad, right_edge], axis=-1)

        # scale shape: (d_model, chunk_size) -> need (1, chunk_size, d_model)
        scale = 1.0 + (left_edge + right_edge)
        scale = mx.transpose(scale, (1, 0))  # (chunk_size, d_model)
        scale = mx.expand_dims(scale, 0)  # (1, chunk_size, d_model)
        return scale

    def __call__(
        self,
        x: mx.array,
        src_key_padding_mask: mx.array | None = None,
    ) -> mx.array:
        """
        Args:
            x: (seq_len, batch_size, d_model)
            src_key_padding_mask: Optional mask

        Returns:
            Output of shape (seq_len, batch_size, d_model)
        """
        # Input projection and gating
        x = self.in_proj(x)  # (seq, batch, 2*d_model)
        x, gate = mx.split(x, 2, axis=-1)
        x = x * mx.sigmoid(gate)

        # Transpose to (batch, seq, d_model) for conv
        seq_len, batch_size, d_model = x.shape
        x = mx.transpose(x, (1, 0, 2))  # (batch, seq, d_model)

        # Apply padding mask
        if src_key_padding_mask is not None:
            mask = mx.expand_dims(src_key_padding_mask, -1)
            x = mx.where(mask, mx.zeros_like(x), x)

        # Depthwise convolution
        if self.causal:
            x = self._causal_depthwise_conv(x)
        else:
            x = self._standard_depthwise_conv(x)

        # Transpose back
        x = mx.transpose(x, (1, 0, 2))  # (seq, batch, d_model)

        # SwooshR activation and output projection
        x = swoosh_r(x)  # SwooshR, not SiLU
        x = self.out_proj(x)

        return x

    def _standard_depthwise_conv(self, x: mx.array) -> mx.array:
        """Standard depthwise conv with symmetric padding."""
        padding = self.kernel_size // 2
        if padding > 0:
            x = mx.pad(x, [(0, 0), (padding, padding), (0, 0)])
        x = mx.conv1d(x, self.depthwise_conv_weight, groups=self.d_model)
        x = x + mx.reshape(self.depthwise_conv_bias, (1, 1, -1))
        return x

    def _causal_depthwise_conv(self, x: mx.array) -> mx.array:
        """Chunk-causal depthwise conv.

        Matches icefall ChunkCausalDepthwiseConv1d with chunk_size = seq_len.
        """
        batch_size, seq_len, d_model = x.shape
        left_pad = self.kernel_size // 2  # 15 for kernel_size=31
        (self.kernel_size + 1) // 2  # 16 for kernel_size=31

        # Pad input once (left only, no right pad when chunk_size = seq_len)
        x_padded = mx.pad(x, [(0, 0), (left_pad, 0), (0, 0)])
        # x_padded shape: (batch, seq_len + left_pad, d_model)

        # Causal component: conv on x_padded[..., : left_pad + seq_len]
        # Since we padded left_pad on left, this is the full padded tensor
        x_causal = mx.conv1d(x_padded, self.causal_conv_weight, groups=d_model)
        # Output shape: (batch, seq_len + left_pad - half_kernel + 1, d_model)
        # = (batch, seq_len, d_model) since left_pad = half_kernel - 1
        x_causal = x_causal + mx.reshape(self.causal_conv_bias, (1, 1, -1))

        # Chunkwise component: conv on original x with symmetric padding
        # PyTorch Conv1d has padding=kernel_size//2 built-in
        padding = self.kernel_size // 2
        x_chunk = mx.pad(x, [(0, 0), (padding, padding), (0, 0)])
        x_chunk = mx.conv1d(x_chunk, self.chunkwise_conv_weight, groups=d_model)
        x_chunk = x_chunk + mx.reshape(self.chunkwise_conv_bias, (1, 1, -1))
        # Apply chunk edge scaling
        chunk_scale = self._get_chunk_scale(seq_len)
        x_chunk = x_chunk * chunk_scale
        # Output shape: (batch, seq_len, d_model)

        return x_causal + x_chunk

    def streaming_forward(
        self,
        x: mx.array,
        cached_conv: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """
        Streaming forward pass for ConvolutionModule.

        Processes input chunk using cached left context for causal convolution.

        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model)
            cached_conv: Cached left context of shape (batch, d_model, cache_len)
                         where cache_len = kernel_size // 2

        Returns:
            Tuple of:
              - output: (seq_len, batch_size, d_model)
              - new_cached_conv: (batch, d_model, cache_len)
        """
        seq_len, batch_size, d_model = x.shape
        cache_len = self.kernel_size // 2  # 15 for kernel_size=31

        if cached_conv.shape != (batch_size, d_model, cache_len):
            raise ValueError(
                f"Expected cached_conv shape {(batch_size, d_model, cache_len)}, "
                f"got {cached_conv.shape}",
            )

        # Input projection and gating
        x = self.in_proj(x)  # (seq, batch, 2*d_model)
        x, gate = mx.split(x, 2, axis=-1)
        x = x * mx.sigmoid(gate)

        # Transpose to (batch, seq, d_model) for conv
        x = mx.transpose(x, (1, 0, 2))  # (batch, seq, d_model)

        # For streaming, concatenate cached context with current input
        # Cache is (batch, d_model, cache_len), need (batch, cache_len, d_model)
        cache_t = mx.transpose(cached_conv, (0, 2, 1))  # (batch, cache_len, d_model)
        x_extended = mx.concatenate([cache_t, x], axis=1)  # (batch, cache_len + seq, d_model)

        # Update cache: take the last cache_len frames from input (not extended)
        # This will be used for the next chunk
        if seq_len >= cache_len:
            new_cache_t = x[:, -cache_len:, :]  # (batch, cache_len, d_model)
        else:
            # If seq_len < cache_len, need to keep some old cache
            keep_len = cache_len - seq_len
            new_cache_t = mx.concatenate([cache_t[:, -keep_len:, :], x], axis=1)
        new_cached_conv = mx.transpose(new_cache_t, (0, 2, 1))  # (batch, d_model, cache_len)

        # Apply causal convolution on extended input
        (self.kernel_size + 1) // 2  # 16 for kernel_size=31
        x_causal = mx.conv1d(x_extended, self.causal_conv_weight, groups=d_model)
        x_causal = x_causal + mx.reshape(self.causal_conv_bias, (1, 1, -1))
        # Output: (batch, cache_len + seq - half_kernel + 1, d_model)
        # = (batch, seq, d_model) since cache_len = half_kernel - 1

        # Chunkwise component: operates on current chunk only with symmetric padding
        padding = self.kernel_size // 2
        x_chunk = mx.pad(x, [(0, 0), (padding, padding), (0, 0)])
        x_chunk = mx.conv1d(x_chunk, self.chunkwise_conv_weight, groups=d_model)
        x_chunk = x_chunk + mx.reshape(self.chunkwise_conv_bias, (1, 1, -1))
        # Apply chunk edge scaling
        chunk_scale = self._get_chunk_scale(seq_len)
        x_chunk = x_chunk * chunk_scale
        # Output: (batch, seq, d_model)

        x = x_causal + x_chunk

        # Transpose back
        x = mx.transpose(x, (1, 0, 2))  # (seq, batch, d_model)

        # SwooshR activation and output projection
        x = swoosh_r(x)  # SwooshR, not SiLU
        x = self.out_proj(x)

        return x, new_cached_conv


class NonlinAttention(nn.Module):
    """
    Non-linear attention module.

    Uses sigmoid gating with attention-weighted values.
    """

    def __init__(self, d_model: int, hidden_channels: int):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.in_proj = nn.Linear(d_model, 3 * hidden_channels, bias=True)
        self.out_proj = nn.Linear(hidden_channels, d_model, bias=True)

    def __call__(
        self,
        x: mx.array,
        attn_weights: mx.array,
    ) -> mx.array:
        """
        Args:
            x: (seq_len, batch_size, d_model)
            attn_weights: (batch * heads, seq, seq)

        Returns:
            Output of shape (seq_len, batch_size, d_model)
        """
        seq_len, batch_size, _ = x.shape

        # Project and split
        proj = self.in_proj(x)  # (seq, batch, 3*hidden)
        s, v, y = mx.split(proj, 3, axis=-1)

        # Tanh gate
        s = mx.tanh(s)
        v = v * s

        # Apply attention using all heads
        # attn_weights: (batch * heads, seq, seq)
        total_heads = attn_weights.shape[0]
        num_heads = total_heads // batch_size
        head_dim = self.hidden_channels // num_heads

        # Reshape v for multi-head attention
        v = mx.reshape(v, (seq_len, batch_size, num_heads, head_dim))
        v = mx.transpose(v, (1, 2, 0, 3))  # (batch, heads, seq, head_dim)
        v = mx.reshape(v, (batch_size * num_heads, seq_len, head_dim))

        # Apply attention with all heads
        out = mx.matmul(attn_weights, v)  # (batch*heads, seq, head_dim)

        # Reshape back
        out = mx.reshape(out, (batch_size, num_heads, seq_len, head_dim))
        out = mx.transpose(out, (2, 0, 1, 3))  # (seq, batch, heads, head_dim)
        out = mx.reshape(out, (seq_len, batch_size, self.hidden_channels))

        # Output gate and projection
        out = out * y
        out = self.out_proj(out)

        return out

    def streaming_forward(
        self,
        x: mx.array,
        attn_weights: mx.array,
        cached_v: mx.array,
        left_context_len: int,
    ) -> tuple[mx.array, mx.array]:
        """
        Streaming forward pass for NonlinAttention.

        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model)
            attn_weights: Attention weights of shape (batch*heads, seq_len, kv_len)
            cached_v: Cached v values of shape (left_ctx, batch_size, hidden_channels)
            left_context_len: Number of cached left context frames

        Returns:
            Tuple of:
              - output: (seq_len, batch_size, d_model)
              - new_cached_v: (left_ctx, batch_size, hidden_channels)
        """
        seq_len, batch_size, _ = x.shape

        # Project and split
        proj = self.in_proj(x)  # (seq, batch, 3*hidden)
        s, v, y = mx.split(proj, 3, axis=-1)

        # Tanh gate
        s = mx.tanh(s)
        v = v * s

        # Concatenate cached v with new v
        v_extended = mx.concatenate([cached_v, v], axis=0)  # (left_ctx + seq, batch, hidden)

        # Update cache: keep last left_context_len frames
        new_cached_v = v_extended[-left_context_len:]  # (left_ctx, batch, hidden)

        # Apply attention using all heads
        total_heads = attn_weights.shape[0]
        num_heads = total_heads // batch_size
        head_dim = self.hidden_channels // num_heads
        kv_len = left_context_len + seq_len

        # Reshape v_extended for multi-head attention
        v_extended = mx.reshape(v_extended, (kv_len, batch_size, num_heads, head_dim))
        v_extended = mx.transpose(v_extended, (1, 2, 0, 3))  # (batch, heads, kv_len, head_dim)
        v_extended = mx.reshape(v_extended, (batch_size * num_heads, kv_len, head_dim))

        # Apply attention: (batch*heads, seq_len, kv_len) @ (batch*heads, kv_len, head_dim)
        out = mx.matmul(attn_weights, v_extended)  # (batch*heads, seq_len, head_dim)

        # Reshape back
        out = mx.reshape(out, (batch_size, num_heads, seq_len, head_dim))
        out = mx.transpose(out, (2, 0, 1, 3))  # (seq, batch, heads, head_dim)
        out = mx.reshape(out, (seq_len, batch_size, self.hidden_channels))

        # Output gate and projection
        out = out * y
        out = self.out_proj(out)

        return out, new_cached_v


class SelfAttention(nn.Module):
    """
    Self-attention module with value projection.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        value_head_dim: int,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.value_head_dim = value_head_dim
        self.attention_dim = num_heads * value_head_dim

        self.in_proj = nn.Linear(d_model, self.attention_dim, bias=True)
        self.out_proj = nn.Linear(self.attention_dim, d_model, bias=True)

    def __call__(
        self,
        x: mx.array,
        attn_weights: mx.array,
    ) -> mx.array:
        """
        Args:
            x: (seq_len, batch_size, d_model)
            attn_weights: (batch * heads, seq, seq)

        Returns:
            Output of shape (seq_len, batch_size, d_model)
        """
        seq_len, batch_size, _ = x.shape

        # Project values
        v = self.in_proj(x)  # (seq, batch, attention_dim)
        v = mx.reshape(v, (seq_len, batch_size, self.num_heads, self.value_head_dim))
        v = mx.transpose(v, (1, 2, 0, 3))  # (batch, heads, seq, head_dim)
        v = mx.reshape(v, (batch_size * self.num_heads, seq_len, self.value_head_dim))

        # Apply attention
        out = mx.matmul(attn_weights, v)  # (batch*heads, seq, head_dim)
        out = mx.reshape(out, (batch_size, self.num_heads, seq_len, self.value_head_dim))
        out = mx.transpose(out, (2, 0, 1, 3))  # (seq, batch, heads, head_dim)
        out = mx.reshape(out, (seq_len, batch_size, self.attention_dim))

        out = self.out_proj(out)
        return out

    def streaming_forward(
        self,
        x: mx.array,
        attn_weights: mx.array,
        cached_val: mx.array,
        left_context_len: int,
    ) -> tuple[mx.array, mx.array]:
        """
        Streaming forward pass for SelfAttention.

        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model)
            attn_weights: Attention weights of shape (batch*heads, seq_len, left_ctx+seq_len)
            cached_val: Cached values of shape (left_ctx, batch_size, attention_dim)
            left_context_len: Number of cached left context frames

        Returns:
            Tuple of:
              - output: (seq_len, batch_size, d_model)
              - new_cached_val: (left_ctx, batch_size, attention_dim)
        """
        seq_len, batch_size, _ = x.shape

        # Project current chunk values
        v = self.in_proj(x)  # (seq, batch, attention_dim)

        # Concatenate cached values with new values
        v_extended = mx.concatenate([cached_val, v], axis=0)  # (left_ctx + seq, batch, attention_dim)

        # Update cache: keep last left_context_len frames
        new_cached_val = v_extended[-left_context_len:]  # (left_ctx, batch, attention_dim)

        # Reshape for multi-head attention
        kv_len = left_context_len + seq_len
        v_extended = mx.reshape(v_extended, (kv_len, batch_size, self.num_heads, self.value_head_dim))
        v_extended = mx.transpose(v_extended, (1, 2, 0, 3))  # (batch, heads, kv_len, head_dim)
        v_extended = mx.reshape(v_extended, (batch_size * self.num_heads, kv_len, self.value_head_dim))

        # Apply attention: (batch*heads, seq_len, kv_len) @ (batch*heads, kv_len, head_dim)
        out = mx.matmul(attn_weights, v_extended)  # (batch*heads, seq_len, head_dim)
        out = mx.reshape(out, (batch_size, self.num_heads, seq_len, self.value_head_dim))
        out = mx.transpose(out, (2, 0, 1, 3))  # (seq, batch, heads, head_dim)
        out = mx.reshape(out, (seq_len, batch_size, self.attention_dim))

        out = self.out_proj(out)
        return out, new_cached_val


class RelPositionMultiheadAttentionWeights(nn.Module):
    """
    Computes attention weights with relative positional encoding.

    Args:
        d_model: model hidden dimension
        num_heads: number of attention heads
        query_head_dim: dimension per head for query/key
        pos_head_dim: dimension per head for positional attention (default 4)
        pos_emb_dim: dimension of input positional embeddings (default 48)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        query_head_dim: int,
        pos_head_dim: int = 4,
        pos_emb_dim: int = 48,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.query_head_dim = query_head_dim
        self.pos_head_dim = pos_head_dim

        # Q, K, P projection - P uses pos_head_dim
        in_proj_dim = 2 * num_heads * query_head_dim + num_heads * pos_head_dim
        self.in_proj = nn.Linear(d_model, in_proj_dim, bias=True)

        # Position projection: from pos_emb_dim to num_heads * pos_head_dim
        self.linear_pos = nn.Linear(pos_emb_dim, num_heads * pos_head_dim, bias=False)

    def __call__(
        self,
        x: mx.array,
        pos_emb: mx.array,
        attn_mask: mx.array | None = None,
        key_padding_mask: mx.array | None = None,
    ) -> mx.array:
        """
        Args:
            x: (seq_len, batch_size, d_model)
            pos_emb: (batch, 2*seq-1, pos_emb_dim) - positional embeddings

        Returns:
            Attention weights of shape (batch * heads, seq, seq)
        """
        seq_len, batch_size, _ = x.shape

        # Project input
        proj = self.in_proj(x)  # (seq, batch, in_proj_dim)

        # Split into Q, K, P
        query_dim = self.num_heads * self.query_head_dim
        q = proj[:, :, :query_dim]
        k = proj[:, :, query_dim:2*query_dim]
        p = proj[:, :, 2*query_dim:]

        # Reshape for multi-head
        q = mx.reshape(q, (seq_len, batch_size, self.num_heads, self.query_head_dim))
        k = mx.reshape(k, (seq_len, batch_size, self.num_heads, self.query_head_dim))
        p = mx.reshape(p, (seq_len, batch_size, self.num_heads, self.pos_head_dim))

        # Transpose
        q = mx.transpose(q, (1, 2, 0, 3))  # (batch, heads, seq, head_dim)
        k = mx.transpose(k, (1, 2, 0, 3))
        p = mx.transpose(p, (1, 2, 0, 3))  # (batch, heads, seq, pos_head_dim)

        # Project position embeddings
        pos_proj = self.linear_pos(pos_emb)  # (batch, 2*seq-1, heads*pos_head_dim)
        pos_proj = mx.reshape(pos_proj, (batch_size, -1, self.num_heads, self.pos_head_dim))
        pos_proj = mx.transpose(pos_proj, (0, 2, 1, 3))  # (batch, heads, 2*seq-1, pos_head_dim)

        # Content attention (no scaling - matches icefall)
        content_score = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2)))

        # Position attention
        pos_score = mx.matmul(p, mx.transpose(pos_proj, (0, 1, 3, 2)))
        pos_score = mx.reshape(pos_score, (batch_size * self.num_heads, seq_len, -1))
        pos_score = self._rel_shift(pos_score, seq_len)
        pos_score = mx.reshape(pos_score, (batch_size, self.num_heads, seq_len, seq_len))

        attn_score = content_score + pos_score

        # Apply masks
        if attn_mask is not None:
            attn_score = attn_score + attn_mask

        if key_padding_mask is not None:
            mask = mx.expand_dims(mx.expand_dims(key_padding_mask, 1), 1)
            attn_score = mx.where(mask, mx.full(attn_score.shape, -1e9), attn_score)

        attn_weights = mx.softmax(attn_score, axis=-1)

        # Reshape to (batch * heads, seq, seq)
        attn_weights = mx.reshape(
            attn_weights, (batch_size * self.num_heads, seq_len, seq_len),
        )

        return attn_weights

    def _rel_shift(self, x: mx.array, seq_len: int) -> mx.array:
        """Convert relative position scores to absolute position scores.

        For query position q attending to key position k, we need to extract
        the relative position score at index (seq_len - 1 - q) + k from the
        input, which represents relative position (k - q).

        Args:
            x: Relative position scores of shape (batch_heads, seq_len, 2*seq_len-1)
            seq_len: Sequence length for both query and key

        Returns:
            Absolute position scores of shape (batch_heads, seq_len, seq_len)
        """
        # Create index matrix: index[q, k] = (seq_len - 1 - q) + k
        rows = mx.arange(seq_len - 1, -1, -1)  # [seq_len-1, seq_len-2, ..., 0]
        cols = mx.arange(seq_len)  # [0, 1, ..., seq_len-1]
        indexes = rows[:, None] + cols[None, :]  # (seq_len, seq_len)
        indexes = mx.expand_dims(indexes, 0)  # (1, seq_len, seq_len)

        # Gather along last axis using the index matrix
        return mx.take_along_axis(x, indexes, axis=-1)

    def _rel_shift_streaming(
        self, x: mx.array, seq_len: int, kv_len: int,
    ) -> mx.array:
        """Convert relative position scores to absolute for streaming.

        In streaming mode, query has seq_len positions (current chunk) but
        key/value has kv_len positions (left_context + current chunk).
        The index formula is the same: index[q, k] = (seq_len - 1 - q) + k,
        but k ranges from 0 to kv_len-1 instead of seq_len-1.

        Args:
            x: Relative position scores of shape (batch_heads, seq_len, rel_len)
               where rel_len = seq_len + kv_len - 1
            seq_len: Query sequence length (current chunk)
            kv_len: Key/Value sequence length (left_ctx + seq_len)

        Returns:
            Absolute position scores of shape (batch_heads, seq_len, kv_len)
        """
        # Create index matrix: index[q, k] = (seq_len - 1 - q) + k
        rows = mx.arange(seq_len - 1, -1, -1)  # [seq_len-1, seq_len-2, ..., 0]
        cols = mx.arange(kv_len)  # [0, 1, ..., kv_len-1]
        indexes = rows[:, None] + cols[None, :]  # (seq_len, kv_len)
        indexes = mx.expand_dims(indexes, 0)  # (1, seq_len, kv_len)

        # Gather along last axis using the index matrix
        return mx.take_along_axis(x, indexes, axis=-1)

    def streaming_forward(
        self,
        x: mx.array,
        pos_emb: mx.array,
        cached_key: mx.array,
        left_context_len: int,
        valid_cache_len: int = 0,
    ) -> tuple[mx.array, mx.array]:
        """
        Streaming forward pass for attention weight computation.

        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model)
            pos_emb: Positional embedding of shape (batch, seq_len+kv_len-1, pos_emb_dim)
                     Extended for left context
            cached_key: Cached keys of shape (left_ctx, batch_size, num_heads*query_head_dim)
            left_context_len: Number of cached left context frames
            valid_cache_len: Number of valid (non-empty) cached frames. When processing
                           the first chunk, this is 0. Cache positions beyond valid_cache_len
                           are masked to prevent attending to uninitialized values.

        Returns:
            Tuple of:
              - attn_weights: (batch*heads, seq_len, left_ctx+seq_len)
              - new_cached_key: (left_ctx, batch_size, num_heads*query_head_dim)
        """
        seq_len, batch_size, _ = x.shape
        kv_len = left_context_len + seq_len

        # Project input
        proj = self.in_proj(x)  # (seq, batch, in_proj_dim)

        # Split into Q, K, P
        query_dim = self.num_heads * self.query_head_dim
        q = proj[:, :, :query_dim]
        k = proj[:, :, query_dim:2*query_dim]
        p = proj[:, :, 2*query_dim:]

        # Concatenate cached keys with new keys
        # cached_key: (left_ctx, batch, query_dim)
        k_extended = mx.concatenate([cached_key, k], axis=0)  # (kv_len, batch, query_dim)

        # Update key cache: keep last left_context_len frames
        new_cached_key = k_extended[-left_context_len:]  # (left_ctx, batch, query_dim)

        # Reshape for multi-head
        q = mx.reshape(q, (seq_len, batch_size, self.num_heads, self.query_head_dim))
        k_extended = mx.reshape(k_extended, (kv_len, batch_size, self.num_heads, self.query_head_dim))
        p = mx.reshape(p, (seq_len, batch_size, self.num_heads, self.pos_head_dim))

        # Transpose
        q = mx.transpose(q, (1, 2, 0, 3))  # (batch, heads, seq, head_dim)
        k_extended = mx.transpose(k_extended, (1, 2, 0, 3))  # (batch, heads, kv_len, head_dim)
        p = mx.transpose(p, (1, 2, 0, 3))  # (batch, heads, seq, pos_head_dim)

        # Project position embeddings
        # pos_emb shape: (batch, seq_len+kv_len-1, pos_emb_dim)
        pos_proj = self.linear_pos(pos_emb)  # (batch, seq_len+kv_len-1, heads*pos_head_dim)
        pos_proj = mx.reshape(pos_proj, (batch_size, -1, self.num_heads, self.pos_head_dim))
        pos_proj = mx.transpose(pos_proj, (0, 2, 1, 3))  # (batch, heads, seq_len+kv_len-1, pos_head_dim)

        # Content attention: q @ k^T
        scale = self.query_head_dim ** -0.5
        content_score = mx.matmul(q, mx.transpose(k_extended, (0, 1, 3, 2))) * scale
        # content_score: (batch, heads, seq_len, kv_len)

        # Position attention
        pos_score = mx.matmul(p, mx.transpose(pos_proj, (0, 1, 3, 2)))
        # pos_score: (batch, heads, seq_len, seq_len+kv_len-1)
        pos_score = mx.reshape(pos_score, (batch_size * self.num_heads, seq_len, -1))
        pos_score = self._rel_shift_streaming(pos_score, seq_len, kv_len)
        pos_score = mx.reshape(pos_score, (batch_size, self.num_heads, seq_len, kv_len))

        attn_score = content_score + pos_score

        # Apply mask to prevent attending to invalid (empty) cache positions
        # The mask sets invalid positions to -1000 so they become ~0 after softmax
        if valid_cache_len < left_context_len:
            # Create mask: True for positions that should be masked (invalid cache)
            # Cache positions are [0, left_context_len), current chunk is [left_context_len, kv_len)
            # Invalid cache positions are [0, left_context_len - valid_cache_len)
            invalid_cache_len = left_context_len - valid_cache_len
            # Create mask of shape (1, 1, 1, kv_len)
            mask = mx.concatenate([
                mx.ones((1, 1, 1, invalid_cache_len)),  # Invalid cache: mask=1
                mx.zeros((1, 1, 1, valid_cache_len + seq_len)),  # Valid cache + current: mask=0
            ], axis=-1)
            # Apply mask: where mask==1, set score to -1000
            attn_score = mx.where(mask > 0.5, mx.array(-1000.0), attn_score)

        attn_weights = mx.softmax(attn_score, axis=-1)

        # Reshape to (batch * heads, seq, kv_len)
        attn_weights = mx.reshape(
            attn_weights, (batch_size * self.num_heads, seq_len, kv_len),
        )

        return attn_weights, new_cached_key


def swoosh_l(x: mx.array) -> mx.array:
    """SwooshL activation function.

    SwooshL(x) = log(1 + exp(x - 4)) - 0.08*x - 0.035

    This is a custom activation designed for Zipformer that provides
    better gradient flow than standard activations.
    """
    return mx.log(1 + mx.exp(x - 4)) - 0.08 * x - 0.035


def swoosh_r(x: mx.array) -> mx.array:
    """SwooshR activation function.

    SwooshR(x) = log(1 + exp(x - 1)) - 0.08*x - 0.313261687

    This is a custom activation designed for Zipformer.
    """
    return mx.log(1 + mx.exp(x - 1)) - 0.08 * x - 0.313261687


class FeedforwardModule(nn.Module):
    """Feedforward module with SwooshL activation.

    Uses SwooshL activation (not SiLU) as in the original Zipformer.
    """

    def __init__(self, d_model: int, feedforward_dim: int):
        super().__init__()
        self.in_proj = nn.Linear(d_model, feedforward_dim, bias=True)
        self.out_proj = nn.Linear(feedforward_dim, d_model, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.in_proj(x)
        x = swoosh_l(x)  # SwooshL, not SiLU
        x = self.out_proj(x)
        return x


class BypassModule(nn.Module):
    """Learnable skip connection with per-channel scale.

    The bypass_scale is a learned vector of shape (d_model,) that controls
    how much of the processed signal vs original signal to use per channel.

    Output = src_orig + scale * (src - src_orig)
           = (1 - scale) * src_orig + scale * src
    """

    def __init__(self, d_model: int):
        super().__init__()
        # Initialize with 0.5 (equal mix) - will be overwritten by checkpoint
        self.bypass_scale = mx.ones((d_model,)) * 0.5

    def __call__(self, src_orig: mx.array, src: mx.array) -> mx.array:
        # Scale is per-channel, shape (d_model,)
        scale = mx.clip(self.bypass_scale, 0.0, 1.0)
        return src_orig + scale * (src - src_orig)


class Zipformer2EncoderLayer(nn.Module):
    """
    Single Zipformer2 encoder layer.

    Contains two self-attention modules, three feedforward modules,
    two convolution modules, and nonlinear attention.
    """

    def __init__(
        self,
        d_model: int,
        attention_dim: int,
        num_heads: int,
        ff1_dim: int,
        ff2_dim: int,
        ff3_dim: int,
        kernel_size: int = 31,
        pos_head_dim: int = 4,
        pos_emb_dim: int = 48,
        value_head_dim: int = 12,
        causal: bool = True,
    ):
        super().__init__()
        self.d_model = d_model

        query_head_dim = attention_dim // num_heads

        # Attention weight computation
        self.self_attn_weights = RelPositionMultiheadAttentionWeights(
            d_model=d_model,
            num_heads=num_heads,
            query_head_dim=query_head_dim,
            pos_head_dim=pos_head_dim,
            pos_emb_dim=pos_emb_dim,
        )

        # Two self-attention modules
        self.self_attn1 = SelfAttention(d_model, num_heads, value_head_dim)
        self.self_attn2 = SelfAttention(d_model, num_heads, value_head_dim)

        # Non-linear attention
        nonlin_hidden = 3 * d_model // 4
        self.nonlin_attention = NonlinAttention(d_model, nonlin_hidden)

        # Three feedforward modules
        self.feed_forward1 = FeedforwardModule(d_model, ff1_dim)
        self.feed_forward2 = FeedforwardModule(d_model, ff2_dim)
        self.feed_forward3 = FeedforwardModule(d_model, ff3_dim)

        # Two convolution modules
        self.conv_module1 = ConvolutionModule(d_model, kernel_size, causal)
        self.conv_module2 = ConvolutionModule(d_model, kernel_size, causal)

        # Normalization
        self.norm = BiasNorm(d_model)

        # Bypass modules (with learned per-channel scales)
        self.bypass = BypassModule(d_model)
        self.bypass_mid = BypassModule(d_model)

        # Overall bypass scale - vector of shape (d_model,) for checkpoint compatibility
        # Note: This is separate from the bypass module scales and may be used
        # for additional scaling. In practice, values are typically 0.5.
        self.bypass_scale = mx.ones((d_model,)) * 0.5

    def __call__(
        self,
        src: mx.array,
        pos_emb: mx.array,
        attn_mask: mx.array | None = None,
        src_key_padding_mask: mx.array | None = None,
    ) -> mx.array:
        """
        Args:
            src: (seq_len, batch_size, d_model)
            pos_emb: (batch, 2*seq-1, d_model)

        Returns:
            Output of shape (seq_len, batch_size, d_model)
        """
        src_orig = src

        # Compute attention weights (shared)
        attn_weights = self.self_attn_weights(
            src, pos_emb,
            attn_mask=attn_mask,
            key_padding_mask=src_key_padding_mask,
        )

        # First feedforward
        src = src + self.feed_forward1(src)

        # Non-linear attention (uses all heads, matching streaming path)
        src = src + self.nonlin_attention(src, attn_weights)

        # First self-attention
        src = src + self.self_attn1(src, attn_weights)

        # First convolution
        src = src + self.conv_module1(src, src_key_padding_mask)

        # Second feedforward
        src = src + self.feed_forward2(src)

        # Mid-layer bypass
        src = self.bypass_mid(src_orig, src)

        # Second self-attention
        src = src + self.self_attn2(src, attn_weights)

        # Second convolution
        src = src + self.conv_module2(src, src_key_padding_mask)

        # Third feedforward
        src = src + self.feed_forward3(src)

        # Normalize
        src = self.norm(src)

        # Final bypass
        src = self.bypass(src_orig, src)

        return src

    def streaming_forward(
        self,
        src: mx.array,
        pos_emb: mx.array,
        cached_key: mx.array,
        cached_val1: mx.array,
        cached_val2: mx.array,
        cached_nonlin_attn: mx.array,
        cached_conv1: mx.array,
        cached_conv2: mx.array,
        left_context_len: int,
        valid_cache_len: int = 0,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
        """
        Streaming forward pass for encoder layer.

        Args:
            src: Input tensor of shape (seq_len, batch_size, d_model)
            pos_emb: Positional embedding of shape (batch, seq_len+kv_len-1, pos_emb_dim)
            cached_key: Cached attention keys (left_ctx, batch, query_dim)
            cached_val1: Cached values for self_attn1 (left_ctx, batch, attention_dim)
            cached_val2: Cached values for self_attn2 (left_ctx, batch, attention_dim)
            cached_nonlin_attn: Cached values for nonlin_attention (left_ctx, batch, hidden_channels)
            cached_conv1: Cached conv1 context (batch, d_model, kernel//2)
            cached_conv2: Cached conv2 context (batch, d_model, kernel//2)
            left_context_len: Number of left context frames
            valid_cache_len: Number of valid (populated) cache frames

        Returns:
            Tuple of (output, new_cached_key, new_cached_val1, new_cached_val2,
                      new_cached_nonlin_attn, new_cached_conv1, new_cached_conv2)
        """
        src_orig = src

        # Compute streaming attention weights with cache masking
        attn_weights, new_cached_key = self.self_attn_weights.streaming_forward(
            src, pos_emb, cached_key, left_context_len, valid_cache_len,
        )

        # First feedforward
        src = src + self.feed_forward1(src)

        # First self-attention (streaming)
        attn1_out, new_cached_val1 = self.self_attn1.streaming_forward(
            src, attn_weights, cached_val1, left_context_len,
        )
        src = src + attn1_out

        # Non-linear attention (streaming)
        nonlin_out, new_cached_nonlin_attn = self.nonlin_attention.streaming_forward(
            src, attn_weights, cached_nonlin_attn, left_context_len,
        )
        src = src + nonlin_out

        # First convolution (streaming)
        conv1_out, new_cached_conv1 = self.conv_module1.streaming_forward(src, cached_conv1)
        src = src + conv1_out

        # Second feedforward
        src = src + self.feed_forward2(src)

        # Mid-layer bypass
        src = self.bypass_mid(src_orig, src)

        # Second self-attention (streaming)
        attn2_out, new_cached_val2 = self.self_attn2.streaming_forward(
            src, attn_weights, cached_val2, left_context_len,
        )
        src = src + attn2_out

        # Second convolution (streaming)
        conv2_out, new_cached_conv2 = self.conv_module2.streaming_forward(src, cached_conv2)
        src = src + conv2_out

        # Third feedforward
        src = src + self.feed_forward3(src)

        # Normalize
        src = self.norm(src)

        # Final bypass
        src = self.bypass(src_orig, src)

        return (
            src,
            new_cached_key,
            new_cached_val1,
            new_cached_val2,
            new_cached_nonlin_attn,
            new_cached_conv1,
            new_cached_conv2,
        )


class RelPositionalEncoding(nn.Module):
    """Relative positional encoding.

    Uses icefall's log-atan encoding scheme for zipformer.
    """

    def __init__(self, pos_dim: int, max_len: int = 5000):
        super().__init__()
        self.pos_dim = pos_dim
        self.max_len = max_len
        self._pe = None
        self.length_factor = 1.0  # Matches icefall default

    def _compute_pe(self, length: int) -> mx.array:
        """Compute positional encoding using icefall's log-atan scheme."""
        import math

        # Positions from -(length-1) to (length-1)
        x = mx.arange(-(length - 1), length, dtype=mx.float32)[:, None]

        # freqs = [1, 2, 3, ..., pos_dim/2]
        freqs = mx.arange(1, self.pos_dim // 2 + 1, dtype=mx.float32)

        # compression_length = sqrt(embed_dim)
        compression_length = math.sqrt(self.pos_dim)

        # Compressed positions using log
        # x_compressed = sign(x) * compression_length * (log(|x| + compression_length) - log(compression_length))
        x_compressed = mx.sign(x) * compression_length * (
            mx.log(mx.abs(x) + compression_length) - math.log(compression_length)
        )

        # length_scale = length_factor * embed_dim / (2 * pi)
        length_scale = self.length_factor * self.pos_dim / (2 * math.pi)

        # x_atan = atan(x_compressed / length_scale)
        x_atan = mx.arctan(x_compressed / length_scale)

        # cosines = cos(x_atan * freqs), sines = sin(x_atan * freqs)
        cosines = mx.cos(x_atan * freqs)
        sines = mx.sin(x_atan * freqs)

        # Build PE: pe[:, 0::2] = cosines, pe[:, 1::2] = sines, pe[:, -1] = 1
        pe = mx.zeros((2 * length - 1, self.pos_dim))
        pe = pe.at[:, 0::2].add(cosines)
        pe = pe.at[:, 1::2].add(sines)
        # Last column is always 1.0 (overwrites any previous value)
        pe = pe.at[:, -1].add(1.0 - pe[:, -1])

        return pe

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: (seq_len, batch_size, d_model) - only shape is used

        Returns:
            Positional encoding (batch, 2*seq-1, pos_dim)
        """
        seq_len = x.shape[0]
        batch_size = x.shape[1]

        if self._pe is None or self._pe.shape[0] < 2 * seq_len - 1:
            self._pe = self._compute_pe(max(seq_len, self.max_len))

        center = self._pe.shape[0] // 2
        pe = self._pe[center - seq_len + 1:center + seq_len]
        return mx.broadcast_to(pe[None, :, :], (batch_size, 2 * seq_len - 1, self.pos_dim))

    def streaming(self, seq_len: int, kv_len: int, batch_size: int) -> mx.array:
        """
        Generate positional encoding for streaming mode.

        In streaming mode, queries (current chunk) attend to both cached keys
        (past context) and current keys. The relative position encoding must
        represent the temporal relationship correctly:

        - Cache keys (positions 0 to left_ctx-1) represent PAST frames
        - Current keys (positions left_ctx to kv_len-1) represent CURRENT frames

        For query q (0 to seq_len-1) attending to key k (0 to kv_len-1):
        - If k is a cache key: relative_pos = (k - left_ctx) - q (negative, in the past)
        - If k is a current key: relative_pos = (k - left_ctx) - q (0 to seq_len-1)

        The relative positions range from:
        - Most negative: q=seq_len-1, k=0 → -(kv_len - 1)
        - Most positive: q=0, k=kv_len-1 → seq_len - 1

        Args:
            seq_len: Current chunk sequence length
            kv_len: Total key/value length (left_context + seq_len)
            batch_size: Batch size

        Returns:
            Positional encoding (batch, seq_len + kv_len - 1, pos_dim)
        """
        total_len = max(seq_len, kv_len)

        if self._pe is None or self._pe.shape[0] < 2 * total_len - 1:
            self._pe = self._compute_pe(max(total_len, self.max_len))

        # For streaming: relative positions range from -(kv_len-1) to (seq_len-1)
        # This requires seq_len + kv_len - 1 positions
        # PE[center] = relative position 0
        # PE[center - i] = relative position -i
        # PE[center + i] = relative position +i
        center = self._pe.shape[0] // 2
        start = center - (kv_len - 1)
        end = center + seq_len
        pe = self._pe[start:end]
        return mx.broadcast_to(pe[None, :, :], (batch_size, seq_len + kv_len - 1, self.pos_dim))


class Zipformer2Encoder(nn.Module):
    """
    Single-resolution Zipformer2 encoder (stack of layers).
    """

    def __init__(
        self,
        d_model: int,
        attention_dim: int,
        num_heads: int,
        ff1_dim: int,
        ff2_dim: int,
        ff3_dim: int,
        num_layers: int,
        kernel_size: int = 31,
        pos_dim: int = 48,
        pos_head_dim: int = 4,
        value_head_dim: int = 12,
        causal: bool = True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model

        # Positional encoder uses pos_dim (48) for encoding dimension
        self.pos_encoder = RelPositionalEncoding(pos_dim)

        self.layers = [
            Zipformer2EncoderLayer(
                d_model=d_model,
                attention_dim=attention_dim,
                num_heads=num_heads,
                ff1_dim=ff1_dim,
                ff2_dim=ff2_dim,
                ff3_dim=ff3_dim,
                kernel_size=kernel_size,
                pos_head_dim=pos_head_dim,
                pos_emb_dim=pos_dim,
                value_head_dim=value_head_dim,
                causal=causal,
            )
            for _ in range(num_layers)
        ]

    def __call__(
        self,
        src: mx.array,
        src_key_padding_mask: mx.array | None = None,
    ) -> mx.array:
        """
        Args:
            src: (seq_len, batch_size, d_model)

        Returns:
            Output of shape (seq_len, batch_size, d_model)
        """
        pos_emb = self.pos_encoder(src)

        for layer in self.layers:
            src = layer(src, pos_emb, src_key_padding_mask=src_key_padding_mask)

        return src

    def streaming_forward(
        self,
        src: mx.array,
        states: list[mx.array],
        left_context_len: int,
        src_key_padding_mask: mx.array | None = None,
        valid_cache_len: int = 0,
    ) -> tuple[mx.array, list[mx.array]]:
        """
        Streaming forward pass through the encoder.

        Args:
            src: Input tensor of shape (seq_len, batch_size, d_model)
            states: List of cached tensors, 6 per layer:
                    [cached_key, cached_val1, cached_val2, cached_nonlin_attn,
                     cached_conv1, cached_conv2] for each layer
            left_context_len: Number of left context frames for attention cache
            src_key_padding_mask: Optional padding mask
            valid_cache_len: Number of valid (populated) cache frames

        Returns:
            - Output tensor of shape (seq_len, batch_size, d_model)
            - Updated states list
        """
        seq_len, batch_size, _ = src.shape
        kv_len = left_context_len + seq_len

        # Compute positional encoding for streaming
        pos_emb = self.pos_encoder.streaming(seq_len, kv_len, batch_size)

        new_states = []
        for i, layer in enumerate(self.layers):
            # Extract states for this layer (6 states per layer)
            layer_states = states[i * 6 : (i + 1) * 6]
            (cached_key, cached_val1, cached_val2,
             cached_nonlin_attn, cached_conv1, cached_conv2) = layer_states

            # Run streaming forward for this layer with cache masking
            (src, new_cached_key, new_cached_val1, new_cached_val2,
             new_cached_nonlin_attn, new_cached_conv1, new_cached_conv2) = layer.streaming_forward(
                src, pos_emb,
                cached_key, cached_val1, cached_val2,
                cached_nonlin_attn, cached_conv1, cached_conv2,
                left_context_len,
                valid_cache_len,
            )

            # Collect updated states
            new_states.extend([
                new_cached_key, new_cached_val1, new_cached_val2,
                new_cached_nonlin_attn, new_cached_conv1, new_cached_conv2,
            ])

        return src, new_states

    def init_states(
        self,
        batch_size: int,
        left_context_len: int,
    ) -> list[mx.array]:
        """
        Initialize streaming states for the encoder.

        Args:
            batch_size: Batch size
            left_context_len: Number of left context frames

        Returns:
            List of initialized state tensors (6 per layer)
        """
        states = []
        for layer in self.layers:
            # Get dimensions from the layer
            num_heads = layer.self_attn_weights.num_heads
            query_head_dim = layer.self_attn_weights.query_head_dim
            value_head_dim = layer.self_attn1.value_head_dim
            kernel_size = layer.conv_module1.kernel_size
            d_model = layer.d_model
            nonlin_hidden = 3 * d_model // 4

            query_dim = num_heads * query_head_dim
            attention_value_dim = num_heads * value_head_dim
            conv_cache_len = kernel_size // 2

            # Initialize caches to zeros
            cached_key = mx.zeros((left_context_len, batch_size, query_dim))
            cached_val1 = mx.zeros((left_context_len, batch_size, attention_value_dim))
            cached_val2 = mx.zeros((left_context_len, batch_size, attention_value_dim))
            cached_nonlin_attn = mx.zeros((left_context_len, batch_size, nonlin_hidden))
            cached_conv1 = mx.zeros((batch_size, d_model, conv_cache_len))
            cached_conv2 = mx.zeros((batch_size, d_model, conv_cache_len))

            states.extend([
                cached_key, cached_val1, cached_val2,
                cached_nonlin_attn, cached_conv1, cached_conv2,
            ])

        return states


class SimpleDownsample(nn.Module):
    """Downsampling via learned weighted sum."""

    def __init__(self, d_model: int, downsample: int):
        super().__init__()
        self.downsample = downsample
        self.bias = mx.zeros((downsample,))

    def __call__(self, src: mx.array) -> mx.array:
        """
        Args:
            src: (seq_len, batch_size, d_model)

        Returns:
            Output of shape (ceil(seq_len/downsample), batch_size, d_model)
        """
        seq_len, batch_size, d_model = src.shape
        ds = self.downsample
        d_seq_len = (seq_len + ds - 1) // ds

        # Pad to multiple of downsample
        pad = d_seq_len * ds - seq_len
        if pad > 0:
            src = mx.concatenate([src, mx.broadcast_to(src[-1:], (pad, batch_size, d_model))], axis=0)

        # Reshape and apply weighted sum
        src = mx.reshape(src, (d_seq_len, ds, batch_size, d_model))
        weights = mx.softmax(self.bias)[:, None, None]
        src = mx.sum(src * weights, axis=1)

        return src


class SimpleUpsample(nn.Module):
    """Upsampling via repetition."""

    def __init__(self, d_model: int, upsample: int):
        super().__init__()
        self.upsample = upsample

    def __call__(self, src: mx.array) -> mx.array:
        """
        Args:
            src: (seq_len, batch_size, d_model)

        Returns:
            Output of shape (seq_len * upsample, batch_size, d_model)
        """
        seq_len, batch_size, d_model = src.shape
        src = mx.expand_dims(src, 1)  # (seq, 1, batch, d_model)
        src = mx.broadcast_to(src, (seq_len, self.upsample, batch_size, d_model))
        src = mx.reshape(src, (seq_len * self.upsample, batch_size, d_model))
        return src


class DownsampledZipformer2Encoder(nn.Module):
    """
    Zipformer2 encoder with downsampling and upsampling.

    Downsamples input, processes through encoder, upsamples back.
    """

    def __init__(
        self,
        d_model: int,
        attention_dim: int,
        num_heads: int,
        ff1_dim: int,
        ff2_dim: int,
        ff3_dim: int,
        num_layers: int,
        downsample: int,
        kernel_size: int = 31,
        pos_dim: int = 48,
        pos_head_dim: int = 4,
        value_head_dim: int = 12,
        causal: bool = True,
    ):
        super().__init__()
        self.downsample_factor = downsample

        self.downsample = SimpleDownsample(d_model, downsample)
        self.encoder = Zipformer2Encoder(
            d_model=d_model,
            attention_dim=attention_dim,
            num_heads=num_heads,
            ff1_dim=ff1_dim,
            ff2_dim=ff2_dim,
            ff3_dim=ff3_dim,
            num_layers=num_layers,
            kernel_size=kernel_size,
            pos_dim=pos_dim,
            pos_head_dim=pos_head_dim,
            value_head_dim=value_head_dim,
            causal=causal,
        )
        self.upsample = SimpleUpsample(d_model, downsample)
        self.out_combiner = BypassModule(d_model)

    def __call__(
        self,
        src: mx.array,
        src_key_padding_mask: mx.array | None = None,
    ) -> mx.array:
        """
        Args:
            src: (seq_len, batch_size, d_model)

        Returns:
            Output of shape (seq_len, batch_size, d_model)
        """
        src_orig = src
        original_len = src.shape[0]

        src = self.downsample(src)
        src = self.encoder(src, src_key_padding_mask)
        src = self.upsample(src)

        # Trim to original length
        src = src[:original_len]

        return self.out_combiner(src_orig, src)

    def streaming_forward(
        self,
        src: mx.array,
        states: list[mx.array],
        left_context_len: int,
        src_key_padding_mask: mx.array | None = None,
        valid_cache_len: int = 0,
    ) -> tuple[mx.array, list[mx.array]]:
        """
        Streaming forward pass with downsampling.

        Args:
            src: Input tensor of shape (seq_len, batch_size, d_model)
            states: Cached states from encoder
            left_context_len: Number of left context frames (in downsampled space)
            src_key_padding_mask: Optional padding mask
            valid_cache_len: Number of valid (populated) cache frames (in downsampled space)

        Returns:
            - Output tensor of shape (seq_len, batch_size, d_model)
            - Updated states list
        """
        src_orig = src
        original_len = src.shape[0]

        # Downsample
        src = self.downsample(src)

        # Process through encoder (streaming)
        src, new_states = self.encoder.streaming_forward(
            src, states, left_context_len, src_key_padding_mask, valid_cache_len,
        )

        # Upsample
        src = self.upsample(src)

        # Trim to original length
        src = src[:original_len]

        return self.out_combiner(src_orig, src), new_states

    def init_states(
        self,
        batch_size: int,
        left_context_len: int,
    ) -> list[mx.array]:
        """
        Initialize streaming states for this encoder.

        Args:
            batch_size: Batch size
            left_context_len: Number of left context frames (in downsampled space)

        Returns:
            List of initialized state tensors
        """
        return self.encoder.init_states(batch_size, left_context_len)


class Conv2dSubsampling(nn.Module):
    """
    2D convolutional subsampling for input features (encoder_embed).

    Converts mel spectrogram to encoder input through:
    1. Three 2D conv layers with SwooshR activation
    2. ConvNext block
    3. Linear projection with normalization

    Architecture matches icefall/k2-fsa subsampling.py:
    - conv0: kernel=3, padding=(0,1), stride=1 -> T-2, F
    - conv4: kernel=3, padding=0, stride=2 -> (T-5)//2+1, 39
    - conv7: kernel=3, padding=0, stride=(1,2) -> (T-7)//2, 19

    Final output: T' = (T-7)//2, F' = 19, linear_in = 128 * 19 = 2432

    Note: MLX conv2d uses NHWC format:
    - Input: (N, H, W, C) where H=time, W=freq
    - Weight: (C_out, H_filter, W_filter, C_in)
    """

    def __init__(
        self,
        num_features: int = 80,
        output_dim: int = 192,
        intermediate_dim: int = 128,
    ):
        super().__init__()
        self.num_features = num_features
        self.output_dim = output_dim

        # Compute output frequency dimension: (((in_channels - 1) // 2) - 1) // 2
        self.out_width = (((num_features - 1) // 2) - 1) // 2  # 19 for 80 features

        # Conv stack: 1 -> 8 -> 32 -> 128 channels
        # MLX conv2d weight format: (C_out, H, W, C_in)
        self.conv0_weight = mx.zeros((8, 3, 3, 1))
        self.conv0_bias = mx.zeros((8,))

        self.conv4_weight = mx.zeros((32, 3, 3, 8))
        self.conv4_bias = mx.zeros((32,))

        self.conv7_weight = mx.zeros((128, 3, 3, 32))
        self.conv7_bias = mx.zeros((128,))

        # ConvNext block
        self.convnext_dw_weight = mx.zeros((128, 7, 7, 1))
        self.convnext_dw_bias = mx.zeros((128,))
        self.convnext_pw1_weight = mx.zeros((384, 1, 1, 128))
        self.convnext_pw1_bias = mx.zeros((384,))
        self.convnext_pw2_weight = mx.zeros((128, 1, 1, 384))
        self.convnext_pw2_bias = mx.zeros((128,))

        # Output projection: 128 * out_width = 128 * 19 = 2432
        linear_in = 128 * self.out_width
        self.out = nn.Linear(linear_in, output_dim, bias=True)
        self.out_norm_bias = mx.zeros((output_dim,))
        self.out_norm_log_scale = mx.array(0.0)

    def _swoosh_r(self, x: mx.array) -> mx.array:
        """SwooshR activation matching icefall implementation.

        SwooshR(x) = log(1 + exp(x - 1)) - 0.08*x - 0.31326168
                   = softplus(x - 1) - 0.08*x - 0.31326168

        This is the exact formula used in k2-fsa/icefall checkpoints.
        """
        # Numerically stable softplus(x-1)
        z = x - 1.0
        softplus_z = mx.where(z > 20, z, mx.log1p(mx.exp(z)))
        return softplus_z - 0.08 * x - 0.31326168

    def _swoosh_l(self, x: mx.array) -> mx.array:
        """SwooshL activation for ConvNext (icefall Conv2dSubsampling).

        activation(x) = log(1 + exp(x - 4)) - 0.08*x - 0.035
                      = softplus(x - 4) - 0.08*x - 0.035

        This matches the icefall ConvNext activation module.
        """
        # Numerically stable softplus(x - 4)
        z = x - 4.0
        softplus_z = mx.where(z > 20, z, mx.log1p(mx.exp(z)))
        return softplus_z - 0.08 * x - 0.035

    def streaming_forward(
        self,
        x: mx.array,
        state: tuple[mx.array, mx.array] | None,
        right_context: mx.array | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array]]:
        """
        Streaming forward pass for Conv2dSubsampling.

        Uses fbank-level caching to produce similar frame counts as non-streaming,
        with optional right context for improved quality at chunk boundaries.

        The approach:
        1. Cache 7 fbank frames between chunks for conv stack left context
        2. Process (7 cache + chunk_frames) through conv stack
        3. Use cached left context and provided right context for ConvNext
        4. Output ALL conv stack frames (no reservation)

        For 45 fbank + 7 cache = 52 total:
        - Conv stack: (52-7)//2 = 22 frames
        - Output: 22 frames (all)

        Args:
            x: (batch, chunk_frames, num_features) - typically 45 fbank frames
            state: Tuple of:
                   - convnext_state: (batch, 128, 3, out_width) cached conv frames in NCHW
                   - fbank_cache: (batch, 7, num_features) cached fbank frames
                   If None, initializes both to zeros.
            right_context: Optional right context frames from conv stack output of NEXT chunk.
                          Shape (batch, 3, out_width, 128) in NHWC format.
                          If None, uses zeros (causes quality degradation on last ~3 frames).

        Returns:
            Tuple of:
              - output: (out_frames, batch, output_dim)
              - new_state: Tuple of (new_convnext_state, new_fbank_cache)
        """
        batch_size, time, features = x.shape
        FBANK_CACHE_SIZE = 7  # Conv stack receptive field
        CONVNEXT_CONTEXT = 3  # ConvNext 7x7 kernel needs 3 frames each side

        # Initialize or unpack state
        if state is None:
            convnext_state = mx.zeros((batch_size, 128, CONVNEXT_CONTEXT, self.out_width), dtype=x.dtype)
            fbank_cache = mx.zeros((batch_size, FBANK_CACHE_SIZE, features), dtype=x.dtype)
        else:
            convnext_state, fbank_cache = state

        # Validate shapes
        if convnext_state.shape != (batch_size, 128, CONVNEXT_CONTEXT, self.out_width):
            raise ValueError(
                f"Expected convnext_state shape {(batch_size, 128, CONVNEXT_CONTEXT, self.out_width)}, "
                f"got {convnext_state.shape}",
            )
        if fbank_cache.shape != (batch_size, FBANK_CACHE_SIZE, features):
            raise ValueError(
                f"Expected fbank_cache shape {(batch_size, FBANK_CACHE_SIZE, features)}, "
                f"got {fbank_cache.shape}",
            )

        # Prepend cached fbank frames for conv stack context
        x_with_cache = mx.concatenate([fbank_cache, x], axis=1)  # (batch, 7+time, features)

        # Update fbank cache for next chunk
        new_fbank_cache = x[:, -FBANK_CACHE_SIZE:, :]

        # Process through conv stack
        x_conv = mx.expand_dims(x_with_cache, -1)

        # Conv0: pad freq only
        x_conv = mx.pad(x_conv, [(0, 0), (0, 0), (1, 1), (0, 0)])
        x_conv = mx.conv2d(x_conv, self.conv0_weight, stride=(1, 1))
        x_conv = x_conv + mx.reshape(self.conv0_bias, (1, 1, 1, -1))
        x_conv = self._swoosh_r(x_conv)

        # Conv4: stride=2, no padding
        x_conv = mx.conv2d(x_conv, self.conv4_weight, stride=(2, 2))
        x_conv = x_conv + mx.reshape(self.conv4_bias, (1, 1, 1, -1))
        x_conv = self._swoosh_r(x_conv)

        # Conv7: stride=(1,2), no padding
        x_conv = mx.conv2d(x_conv, self.conv7_weight, stride=(1, 2))
        x_conv = x_conv + mx.reshape(self.conv7_bias, (1, 1, 1, -1))
        x_conv = self._swoosh_r(x_conv)  # (batch, t_sub, out_width, 128)

        out_frames = x_conv.shape[1]
        x_out = x_conv  # Use all conv stack output

        # ConvNext with cached left context and optional right context
        convnext_left = mx.transpose(convnext_state, (0, 2, 3, 1))  # NCHW -> NHWC

        if right_context is not None:
            convnext_right = right_context
        else:
            # Zero padding for right context
            convnext_right = mx.zeros((batch_size, CONVNEXT_CONTEXT, self.out_width, 128), dtype=x_out.dtype)

        # Build ConvNext input: [3 left] + [out_frames content] + [3 right]
        concat_in = mx.concatenate([convnext_left, x_out, convnext_right], axis=1)

        # Residual
        residual = x_out

        # Update ConvNext state: last 3 frames of conv stack output
        new_convnext_state = mx.transpose(x_out[:, -CONVNEXT_CONTEXT:, :, :], (0, 3, 1, 2))

        # ConvNext block
        y = mx.pad(concat_in, [(0, 0), (0, 0), (3, 3), (0, 0)])
        y = mx.conv2d(y, self.convnext_dw_weight, groups=128)
        y = y + mx.reshape(self.convnext_dw_bias, (1, 1, 1, -1))

        y = mx.conv2d(y, self.convnext_pw1_weight)
        y = y + mx.reshape(self.convnext_pw1_bias, (1, 1, 1, -1))
        y = self._swoosh_l(y)

        y = mx.conv2d(y, self.convnext_pw2_weight)
        y = y + mx.reshape(self.convnext_pw2_bias, (1, 1, 1, -1))
        y = y + residual

        # Flatten
        y = mx.transpose(y, (0, 1, 3, 2))
        y = mx.reshape(y, (batch_size, out_frames, 128 * self.out_width))

        # Output projection + BiasNorm
        y = self.out(y)
        diff = y - self.out_norm_bias
        var = mx.mean(diff ** 2, axis=-1, keepdims=True)
        scales = mx.rsqrt(var + 1e-8) * mx.exp(self.out_norm_log_scale)
        y = y * scales

        # Transpose to encoder format
        y = mx.transpose(y, (1, 0, 2))

        new_state = (new_convnext_state, new_fbank_cache)
        return y, new_state

    def get_conv_stack_output(self, x: mx.array) -> mx.array:
        """
        Run only the conv stack (without ConvNext) for lookahead computation.

        Used to get right context frames from the next chunk.

        Args:
            x: (batch, chunk_frames, num_features)

        Returns:
            Conv stack output: (batch, (chunk_frames-7)//2, out_width, 128)
        """
        x_conv = mx.expand_dims(x, -1)

        x_conv = mx.pad(x_conv, [(0, 0), (0, 0), (1, 1), (0, 0)])
        x_conv = mx.conv2d(x_conv, self.conv0_weight, stride=(1, 1))
        x_conv = x_conv + mx.reshape(self.conv0_bias, (1, 1, 1, -1))
        x_conv = self._swoosh_r(x_conv)

        x_conv = mx.conv2d(x_conv, self.conv4_weight, stride=(2, 2))
        x_conv = x_conv + mx.reshape(self.conv4_bias, (1, 1, 1, -1))
        x_conv = self._swoosh_r(x_conv)

        x_conv = mx.conv2d(x_conv, self.conv7_weight, stride=(1, 2))
        x_conv = x_conv + mx.reshape(self.conv7_bias, (1, 1, 1, -1))
        x_conv = self._swoosh_r(x_conv)

        return x_conv  # (batch, t_sub, out_width, 128)

    def get_conv_stack_output_with_cache(
        self,
        x: mx.array,
        fbank_cache: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """
        Run conv stack with fbank caching for lookahead computation.

        Args:
            x: (batch, chunk_frames, num_features)
            fbank_cache: (batch, 7, num_features)

        Returns:
            Tuple of:
            - Conv stack output: (batch, out_frames, out_width, 128)
            - New fbank cache: (batch, 7, num_features)
        """
        FBANK_CACHE_SIZE = 7
        x_with_cache = mx.concatenate([fbank_cache, x], axis=1)
        new_fbank_cache = x[:, -FBANK_CACHE_SIZE:, :]

        x_conv = mx.expand_dims(x_with_cache, -1)

        x_conv = mx.pad(x_conv, [(0, 0), (0, 0), (1, 1), (0, 0)])
        x_conv = mx.conv2d(x_conv, self.conv0_weight, stride=(1, 1))
        x_conv = x_conv + mx.reshape(self.conv0_bias, (1, 1, 1, -1))
        x_conv = self._swoosh_r(x_conv)

        x_conv = mx.conv2d(x_conv, self.conv4_weight, stride=(2, 2))
        x_conv = x_conv + mx.reshape(self.conv4_bias, (1, 1, 1, -1))
        x_conv = self._swoosh_r(x_conv)

        x_conv = mx.conv2d(x_conv, self.conv7_weight, stride=(1, 2))
        x_conv = x_conv + mx.reshape(self.conv7_bias, (1, 1, 1, -1))
        x_conv = self._swoosh_r(x_conv)

        return x_conv, new_fbank_cache

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: (batch_size, time, num_features) - mel spectrogram

        Returns:
            Output of shape ((time-7)//2, batch_size, output_dim)
        """
        batch_size, time, features = x.shape

        # Input is (batch, time, features), add channel dim to get NHWC
        # (batch, time, features) -> (batch, time, features, 1)
        x = mx.expand_dims(x, -1)

        # Conv0: padding=(0,1) means time_pad=0, freq_pad=1, stride=1
        # (batch, T, 80, 1) -> (batch, T-2, 80, 8)
        x = mx.pad(x, [(0, 0), (0, 0), (1, 1), (0, 0)])  # Only pad freq
        x = mx.conv2d(x, self.conv0_weight, stride=(1, 1))
        x = x + mx.reshape(self.conv0_bias, (1, 1, 1, -1))
        x = self._swoosh_r(x)

        # Conv4: padding=0, stride=2
        # (batch, T-2, 80, 8) -> (batch, (T-5)//2+1, 39, 32)
        x = mx.conv2d(x, self.conv4_weight, stride=(2, 2))
        x = x + mx.reshape(self.conv4_bias, (1, 1, 1, -1))
        x = self._swoosh_r(x)

        # Conv7: padding=0, stride=(1,2)
        # (batch, (T-5)//2+1, 39, 32) -> (batch, (T-7)//2, 19, 128)
        x = mx.conv2d(x, self.conv7_weight, stride=(1, 2))
        x = x + mx.reshape(self.conv7_bias, (1, 1, 1, -1))
        x = self._swoosh_r(x)

        # ConvNext block (preserves shape)
        residual = x
        # Depthwise conv (groups=128) with padding (3,3) for 7x7 kernel
        x = mx.pad(x, [(0, 0), (3, 3), (3, 3), (0, 0)])
        x = mx.conv2d(x, self.convnext_dw_weight, groups=128)
        x = x + mx.reshape(self.convnext_dw_bias, (1, 1, 1, -1))
        # Pointwise 1
        x = mx.conv2d(x, self.convnext_pw1_weight)
        x = x + mx.reshape(self.convnext_pw1_bias, (1, 1, 1, -1))
        # SwooshL activation for ConvNext (numerically stable)
        x = self._swoosh_l(x)
        # Pointwise 2
        x = mx.conv2d(x, self.convnext_pw2_weight)
        x = x + mx.reshape(self.convnext_pw2_bias, (1, 1, 1, -1))
        x = x + residual

        # Flatten: (batch, T', F', 128) -> (batch, T', 128 * F')
        # PyTorch does transpose(1,2).reshape() which gives (batch, T', C*F')
        # MLX NHWC format needs transpose to get same order
        x = mx.transpose(x, (0, 1, 3, 2))  # (batch, T', C, F')
        batch_size, t_out, channels, f_out = x.shape
        x = mx.reshape(x, (batch_size, t_out, channels * f_out))

        # Output projection
        x = self.out(x)  # (batch, T', output_dim)

        # Output norm (BiasNorm style)
        # BiasNorm: scales = (mean((x - bias)^2))^(-0.5) * exp(log_scale)
        #           output = x * scales
        diff = x - self.out_norm_bias
        var = mx.mean(diff ** 2, axis=-1, keepdims=True)
        scales = mx.rsqrt(var + 1e-8) * mx.exp(self.out_norm_log_scale)
        x = x * scales

        # Transpose to encoder format: (T', batch, output_dim)
        x = mx.transpose(x, (1, 0, 2))

        return x


class Zipformer(nn.Module):
    """
    Full Zipformer model with multi-resolution encoder stages.

    Architecture:
    - Conv2dSubsampling (encoder_embed)
    - 6 encoder stages with varying resolutions
    - U-Net style structure (resolutions decrease then increase)
    - Multi-stage output combination for joiner (max(encoder_dims))
    """

    def __init__(self, config: ZipformerConfig):
        super().__init__()
        self.config = config
        self.encoder_dims = config.encoder_dims
        self.output_dim = max(config.encoder_dims)

        # Encoder embed (2D subsampling)
        self.encoder_embed = Conv2dSubsampling(
            num_features=config.num_features,
            output_dim=config.encoder_dims[0],
            intermediate_dim=config.encoder_embed_dim,
        )

        # First stage (no downsampling)
        self.encoders = []

        # Stage 0 - base resolution
        self.encoders.append(
            Zipformer2Encoder(
                d_model=config.encoder_dims[0],
                attention_dim=config.attention_dims[0],
                num_heads=config.num_heads[0],
                ff1_dim=config.ff1_dims[0],
                ff2_dim=config.ff2_dims[0],
                ff3_dim=config.ff3_dims[0],
                num_layers=config.num_encoder_layers[0],
                kernel_size=config.cnn_module_kernels[0],
                pos_dim=config.pos_dim,
                pos_head_dim=config.pos_head_dim,
                value_head_dim=config.value_head_dim,
                causal=config.causal,
            ),
        )

        # Stages 1-5 - with downsampling
        for i in range(1, len(config.encoder_dims)):
            self.encoders.append(
                DownsampledZipformer2Encoder(
                    d_model=config.encoder_dims[i],
                    attention_dim=config.attention_dims[i],
                    num_heads=config.num_heads[i],
                    ff1_dim=config.ff1_dims[i],
                    ff2_dim=config.ff2_dims[i],
                    ff3_dim=config.ff3_dims[i],
                    num_layers=config.num_encoder_layers[i],
                    downsample=config.downsampling_factors[i],
                    kernel_size=config.cnn_module_kernels[i],
                    pos_dim=config.pos_dim,
                    pos_head_dim=config.pos_head_dim,
                    value_head_dim=config.value_head_dim,
                    causal=config.causal,
                ),
            )

        # Output downsampling
        self.downsample_output_bias = mx.zeros((2,))

    def __call__(
        self,
        x: mx.array,
        x_lens: mx.array | None = None,
        unfreeze_layers: int | None = None,
    ) -> tuple[mx.array, mx.array | None]:
        """
        Args:
            x: Input mel spectrogram (batch_size, time, num_features)
            x_lens: Input lengths (batch_size,)
            unfreeze_layers: Optional number of top encoder stages to allow gradients through.
                - None (default): no gradient freezing (full encoder trainable)
                - 0: fully frozen encoder (no gradients through any stage)
                - N>0: unfreeze the top N stages (stages 6-N .. 5 frozen)

        Returns:
            - Encoder output (batch_size, time', output_dim) where output_dim = max(encoder_dims)
            - Output lengths (batch_size,)
        """
        # Apply encoder embed
        x = self.encoder_embed(x)  # (time', batch, d_model)

        # Store outputs from each encoder stage for multi-scale combination
        outputs: list[mx.array] = []

        freeze_before_stage = 0
        if unfreeze_layers is not None:
            if unfreeze_layers <= 0:
                freeze_before_stage = len(self.encoders)
            else:
                freeze_before_stage = max(0, len(self.encoders) - unfreeze_layers)

        # Process through encoder stages
        for i, encoder in enumerate(self.encoders):
            # Convert channels to match this stage's dimension (truncate or zero-pad)
            x = self._convert_num_channels(x, self.encoder_dims[i])

            x = encoder(x)
            if i < freeze_before_stage:
                x = mx.stop_gradient(x)
            outputs.append(x)

        # Combine multi-scale outputs to get full dimension (max(encoder_dims))
        x = self._get_full_dim_output(outputs)

        # Final downsampling
        seq_len, batch_size, d_model = x.shape
        ds = 2
        d_seq_len = (seq_len + ds - 1) // ds

        pad = d_seq_len * ds - seq_len
        if pad > 0:
            x = mx.concatenate([x, mx.broadcast_to(x[-1:], (pad, batch_size, d_model))], axis=0)

        x = mx.reshape(x, (d_seq_len, ds, batch_size, d_model))
        weights = mx.softmax(self.downsample_output_bias)[:, None, None]
        x = mx.sum(x * weights, axis=1)

        # Transpose to (batch, time, d_model)
        x = mx.transpose(x, (1, 0, 2))

        # Compute output lengths if provided
        # Conv2dSubsampling: T -> (T-7)//2
        # Final downsampling: ceil(T'/2)
        out_lens = None
        if x_lens is not None:
            embed_lens = (x_lens - 7) // 2  # Conv2dSubsampling
            out_lens = (embed_lens + 1) // 2  # Final ceil(T'/2) downsampling

        return x, out_lens

    def _convert_num_channels(self, x: mx.array, num_channels: int) -> mx.array:
        """
        Convert tensor to have specified number of channels.

        Matches icefall's convert_num_channels:
        - If num_channels <= current: truncate
        - If num_channels > current: zero-pad

        Args:
            x: Input tensor of shape (..., current_channels)
            num_channels: Desired number of channels

        Returns:
            Tensor of shape (..., num_channels)
        """
        current_channels = x.shape[-1]
        if num_channels <= current_channels:
            return x[..., :num_channels]
        # Pad with zeros
        pad_shape = list(x.shape)
        pad_shape[-1] = num_channels - current_channels
        zeros = mx.zeros(pad_shape, dtype=x.dtype)
        return mx.concatenate([x, zeros], axis=-1)

    def _get_full_dim_output(self, outputs: list[mx.array]) -> mx.array:
        """
        Combine outputs from multiple encoder stages to get full dimensional output.

        This matches icefall's _get_full_dim_output() method which creates an output
        with dimension = max(encoder_dims) by concatenating partial outputs from
        stages with larger dimensions.

        For encoder_dims = (192, 256, 384, 512, 384, 256):
        - Start with outputs[-1] (stage 5, 256-dim) as base
        - Add outputs[4][..., 256:384] (128 additional channels from stage 4)
        - Add outputs[3][..., 384:512] (128 additional channels from stage 3)
        - Final output: 512-dim (256 + 128 + 128)

        Args:
            outputs: List of encoder stage outputs, each (seq_len, batch, stage_dim)

        Returns:
            Combined output of shape (seq_len, batch, max(encoder_dims))
        """
        num_encoders = len(self.encoder_dims)
        assert len(outputs) == num_encoders

        output_pieces = [outputs[-1]]  # Start with last stage output
        cur_dim = self.encoder_dims[-1]

        # Iterate backwards, collecting additional channels from larger stages
        for i in range(num_encoders - 2, -1, -1):
            d = self.encoder_dims[i]
            if d > cur_dim:
                # This stage has more channels, extract the additional ones
                this_output = outputs[i]
                output_pieces.append(this_output[..., cur_dim:d])
                cur_dim = d

        assert cur_dim == self.output_dim, f"Expected {self.output_dim}, got {cur_dim}"

        # Concatenate all pieces along the last dimension
        return mx.concatenate(output_pieces, axis=-1)

    def streaming_forward(
        self,
        x: mx.array,
        states: list[mx.array],
        chunk_size: int = 16,
        left_context_len: int = 128,
        processed_frames: int = 0,
        right_context: mx.array | None = None,
    ) -> tuple[mx.array, list[mx.array]]:
        """
        Streaming forward pass through the full Zipformer encoder.

        Processes audio chunks through:
        1. Conv2dSubsampling (encoder_embed) with ConvNext caching
        2. All encoder stages (Zipformer2Encoder + DownsampledZipformer2Encoder)
        3. Final downsampling

        Args:
            x: Input mel spectrogram chunk (batch_size, chunk_frames, num_features)
               Typically 45 fbank frames per chunk.
            states: List of cached states:
                    - states[0]: convnext_state for Conv2dSubsampling (batch, 128, 3, out_width)
                    - states[1:]: encoder states (6 per layer per encoder stage)
            chunk_size: Unused (kept for API compatibility)
            left_context_len: Left context frames at base resolution (default 128)
            processed_frames: Total frames processed so far at the encoder "base" time
                resolution (post-Conv2dSubsampling). Used to compute valid cache length
                for attention masking; for the first chunk this should be 0.
            right_context: Optional right context from next chunk's conv stack output.
                          Shape (batch, 3, out_width, 128) in NHWC format.
                          If provided, improves quality of current chunk's output.

        Returns:
            - Encoder output (batch_size, out_frames, output_dim) where output_dim = max(encoder_dims)
            - Updated states list
        """
        # Extract embed state and encoder states
        embed_state = states[0]
        encoder_states = states[1:]

        # 1. Apply encoder embed (streaming)
        x, new_embed_state = self.encoder_embed.streaming_forward(x, embed_state, right_context)
        # x is now (seq_len, batch_size, encoder_dims[0])

        # Store outputs from each encoder stage for multi-scale combination
        outputs: list[mx.array] = []
        new_encoder_states: list[mx.array] = []

        # Track state index for each encoder
        state_idx = 0

        # 2. Process through encoder stages
        for i, encoder in enumerate(self.encoders):
            # Convert channels to match this stage's dimension
            x = self._convert_num_channels(x, self.encoder_dims[i])

            # Compute left context for this stage (scaled by downsampling factor)
            ds_factor = self.config.downsampling_factors[i]
            stage_left_context = left_context_len // ds_factor

            # Compute valid cache length for this stage
            # This is the number of cached frames that contain valid data (not zeros)
            # Scaled by downsampling factor and capped at stage_left_context
            stage_valid_cache = min(processed_frames // ds_factor, stage_left_context)

            # Get number of states for this encoder
            if i == 0:
                # Zipformer2Encoder - direct access to num_layers
                num_layers = self.config.num_encoder_layers[i]
            else:
                # DownsampledZipformer2Encoder - get from inner encoder
                num_layers = self.config.num_encoder_layers[i]

            num_states = num_layers * 6

            # Extract states for this encoder
            encoder_stage_states = encoder_states[state_idx : state_idx + num_states]
            state_idx += num_states

            # Run streaming forward with cache masking
            if i == 0:
                # Zipformer2Encoder
                x, new_stage_states = encoder.streaming_forward(
                    x, encoder_stage_states, stage_left_context,
                    src_key_padding_mask=None, valid_cache_len=stage_valid_cache,
                )
            else:
                # DownsampledZipformer2Encoder
                x, new_stage_states = encoder.streaming_forward(
                    x, encoder_stage_states, stage_left_context,
                    src_key_padding_mask=None, valid_cache_len=stage_valid_cache,
                )
            new_encoder_states.extend(new_stage_states)
            outputs.append(x)

        # 3. Combine multi-scale outputs to get full dimension (max(encoder_dims))
        x = self._get_full_dim_output(outputs)

        # 4. Final downsampling (same as non-streaming)
        seq_len, batch_size, d_model = x.shape
        ds = 2
        d_seq_len = (seq_len + ds - 1) // ds

        pad = d_seq_len * ds - seq_len
        if pad > 0:
            x = mx.concatenate([x, mx.broadcast_to(x[-1:], (pad, batch_size, d_model))], axis=0)

        x = mx.reshape(x, (d_seq_len, ds, batch_size, d_model))
        weights = mx.softmax(self.downsample_output_bias)[:, None, None]
        x = mx.sum(x * weights, axis=1)

        # Transpose to (batch, time, d_model)
        x = mx.transpose(x, (1, 0, 2))

        # Combine all states
        new_states = [new_embed_state] + new_encoder_states

        return x, new_states

    def init_states(
        self,
        batch_size: int,
        left_context_len: int = 128,
    ) -> list:
        """
        Initialize streaming states for the full Zipformer encoder.

        Args:
            batch_size: Batch size
            left_context_len: Left context frames at base resolution (default 128)

        Returns:
            List of initialized state tensors:
            - states[0]: embed_state tuple (convnext_state, fbank_cache)
            - states[1:]: encoder states (6 per layer for each encoder stage)
        """
        states = []

        # 1. Initialize embed state (Conv2dSubsampling)
        # Tuple of (convnext_state, fbank_cache)
        convnext_state = mx.zeros(
            (batch_size, 128, 3, self.encoder_embed.out_width),
            dtype=mx.float32,
        )
        fbank_cache = mx.zeros(
            (batch_size, 7, self.encoder_embed.num_features),
            dtype=mx.float32,
        )
        embed_state = (convnext_state, fbank_cache)
        states.append(embed_state)

        # 2. Initialize encoder states for each stage
        for i, encoder in enumerate(self.encoders):
            # Compute left context for this stage (scaled by downsampling factor)
            ds_factor = self.config.downsampling_factors[i]
            stage_left_context = left_context_len // ds_factor

            # Get states from encoder
            encoder_stage_states = encoder.init_states(batch_size, stage_left_context)
            states.extend(encoder_stage_states)

        return states

    def get_num_states(self) -> int:
        """Return the total number of state tensors for streaming."""
        # 1 embed state + 6 states per layer for each encoder
        total_layers = sum(self.config.num_encoder_layers)
        return 1 + total_layers * 6


def load_zipformer_weights(
    model: Zipformer,
    weights: dict[str, mx.array],
) -> None:
    """
    Load pretrained weights into Zipformer model.

    Args:
        model: The Zipformer model instance.
        weights: Dictionary of weights from converted checkpoint.
    """
    def get_weight(key: str) -> mx.array | None:
        return weights.get(key)

    # Load encoder embed weights
    # Note: PyTorch conv2d weights are (out, in, H, W), MLX expects (out, H, W, in)
    if (w := get_weight('encoder_embed.conv.0.weight')) is not None:
        model.encoder_embed.conv0_weight = mx.transpose(w, (0, 2, 3, 1))
    if (w := get_weight('encoder_embed.conv.0.bias')) is not None:
        model.encoder_embed.conv0_bias = w
    if (w := get_weight('encoder_embed.conv.4.weight')) is not None:
        model.encoder_embed.conv4_weight = mx.transpose(w, (0, 2, 3, 1))
    if (w := get_weight('encoder_embed.conv.4.bias')) is not None:
        model.encoder_embed.conv4_bias = w
    if (w := get_weight('encoder_embed.conv.7.weight')) is not None:
        model.encoder_embed.conv7_weight = mx.transpose(w, (0, 2, 3, 1))
    if (w := get_weight('encoder_embed.conv.7.bias')) is not None:
        model.encoder_embed.conv7_bias = w

    # ConvNext weights - also need transpose for conv2d weights
    if (w := get_weight('encoder_embed.convnext.depthwise_conv.weight')) is not None:
        model.encoder_embed.convnext_dw_weight = mx.transpose(w, (0, 2, 3, 1))
    if (w := get_weight('encoder_embed.convnext.depthwise_conv.bias')) is not None:
        model.encoder_embed.convnext_dw_bias = w
    if (w := get_weight('encoder_embed.convnext.pointwise_conv1.weight')) is not None:
        model.encoder_embed.convnext_pw1_weight = mx.transpose(w, (0, 2, 3, 1))
    if (w := get_weight('encoder_embed.convnext.pointwise_conv1.bias')) is not None:
        model.encoder_embed.convnext_pw1_bias = w
    if (w := get_weight('encoder_embed.convnext.pointwise_conv2.weight')) is not None:
        model.encoder_embed.convnext_pw2_weight = mx.transpose(w, (0, 2, 3, 1))
    if (w := get_weight('encoder_embed.convnext.pointwise_conv2.bias')) is not None:
        model.encoder_embed.convnext_pw2_bias = w

    # Output projection
    if (w := get_weight('encoder_embed.out.weight')) is not None:
        model.encoder_embed.out.weight = w
    if (w := get_weight('encoder_embed.out.bias')) is not None:
        model.encoder_embed.out.bias = w
    if (w := get_weight('encoder_embed.out_norm.bias')) is not None:
        model.encoder_embed.out_norm_bias = w
    if (w := get_weight('encoder_embed.out_norm.log_scale')) is not None:
        model.encoder_embed.out_norm_log_scale = w

    # Output downsampling
    if (w := get_weight('encoder.downsample_output.bias')) is not None:
        model.downsample_output_bias = w

    # Load encoder stage weights
    _load_encoder_weights(model, weights)


def _load_encoder_weights(model: Zipformer, weights: dict[str, mx.array]) -> None:
    """Load weights for all encoder stages."""

    def get_weight(key: str) -> mx.array | None:
        return weights.get(key)

    # Stage 0 - regular encoder (checkpoint uses 'encoders.0.layers...')
    _load_encoder_stage_weights(model.encoders[0], weights, 'encoders.0')

    # Stages 1-5 - downsampled encoders (checkpoint uses 'encoders.N.encoder.layers...')
    for i in range(1, len(model.encoders)):
        encoder = model.encoders[i]
        prefix = f'encoders.{i}'

        # Downsample bias
        if (w := get_weight(f'{prefix}.downsample.bias')) is not None:
            encoder.downsample.bias = w

        # Out combiner
        if (w := get_weight(f'{prefix}.out_combiner.bypass_scale')) is not None:
            encoder.out_combiner.bypass_scale = w

        # Inner encoder
        _load_encoder_stage_weights(encoder.encoder, weights, f'{prefix}.encoder')


def _load_encoder_stage_weights(
    encoder: Zipformer2Encoder,
    weights: dict[str, mx.array],
    prefix: str,
) -> None:
    """Load weights for a single encoder stage."""

    def get_weight(key: str) -> mx.array | None:
        return weights.get(key)

    for layer_idx, layer in enumerate(encoder.layers):
        layer_prefix = f'{prefix}.layers.{layer_idx}'
        _load_encoder_layer_weights(layer, weights, layer_prefix)


def _load_encoder_layer_weights(
    layer: Zipformer2EncoderLayer,
    weights: dict[str, mx.array],
    prefix: str,
) -> None:
    """Load weights for a single encoder layer."""

    def get_weight(suffix: str) -> mx.array | None:
        return weights.get(f'{prefix}.{suffix}')

    # Bypass scales
    if (w := get_weight('bypass_scale')) is not None:
        layer.bypass_scale = w
    if (w := get_weight('bypass.bypass_scale')) is not None:
        layer.bypass.bypass_scale = w
    if (w := get_weight('bypass_mid.bypass_scale')) is not None:
        layer.bypass_mid.bypass_scale = w

    # Norm
    if (w := get_weight('norm.bias')) is not None:
        layer.norm.bias = w
    if (w := get_weight('norm.log_scale')) is not None:
        layer.norm.log_scale = w

    # Self-attention weights (attention weight computation)
    if (w := get_weight('self_attn_weights.in_proj.weight')) is not None:
        layer.self_attn_weights.in_proj.weight = w
    if (w := get_weight('self_attn_weights.in_proj.bias')) is not None:
        layer.self_attn_weights.in_proj.bias = w
    if (w := get_weight('self_attn_weights.linear_pos.weight')) is not None:
        layer.self_attn_weights.linear_pos.weight = w

    # Self-attention 1 & 2
    for attn_name in ['self_attn1', 'self_attn2']:
        attn = getattr(layer, attn_name)
        if (w := get_weight(f'{attn_name}.in_proj.weight')) is not None:
            attn.in_proj.weight = w
        if (w := get_weight(f'{attn_name}.in_proj.bias')) is not None:
            attn.in_proj.bias = w
        if (w := get_weight(f'{attn_name}.out_proj.weight')) is not None:
            attn.out_proj.weight = w
        if (w := get_weight(f'{attn_name}.out_proj.bias')) is not None:
            attn.out_proj.bias = w

    # Non-linear attention
    if (w := get_weight('nonlin_attention.in_proj.weight')) is not None:
        layer.nonlin_attention.in_proj.weight = w
    if (w := get_weight('nonlin_attention.in_proj.bias')) is not None:
        layer.nonlin_attention.in_proj.bias = w
    if (w := get_weight('nonlin_attention.out_proj.weight')) is not None:
        layer.nonlin_attention.out_proj.weight = w
    if (w := get_weight('nonlin_attention.out_proj.bias')) is not None:
        layer.nonlin_attention.out_proj.bias = w

    # Feedforward modules
    for ff_name in ['feed_forward1', 'feed_forward2', 'feed_forward3']:
        ff = getattr(layer, ff_name)
        if (w := get_weight(f'{ff_name}.in_proj.weight')) is not None:
            ff.in_proj.weight = w
        if (w := get_weight(f'{ff_name}.in_proj.bias')) is not None:
            ff.in_proj.bias = w
        if (w := get_weight(f'{ff_name}.out_proj.weight')) is not None:
            ff.out_proj.weight = w
        if (w := get_weight(f'{ff_name}.out_proj.bias')) is not None:
            ff.out_proj.bias = w

    # Convolution modules
    for conv_name in ['conv_module1', 'conv_module2']:
        conv = getattr(layer, conv_name)

        # In/out projections
        if (w := get_weight(f'{conv_name}.in_proj.weight')) is not None:
            conv.in_proj.weight = w
        if (w := get_weight(f'{conv_name}.in_proj.bias')) is not None:
            conv.in_proj.bias = w
        if (w := get_weight(f'{conv_name}.out_proj.weight')) is not None:
            conv.out_proj.weight = w
        if (w := get_weight(f'{conv_name}.out_proj.bias')) is not None:
            conv.out_proj.bias = w

        # Depthwise conv (causal structure)
        # Note: PyTorch weights are (out, in_per_group, kernel), MLX expects (out, kernel, in_per_group)
        if (w := get_weight(f'{conv_name}.depthwise_conv.causal_conv.weight')) is not None:
            conv.causal_conv_weight = mx.transpose(w, (0, 2, 1))  # (out, 1, k) -> (out, k, 1)
        if (w := get_weight(f'{conv_name}.depthwise_conv.causal_conv.bias')) is not None:
            conv.causal_conv_bias = w
        if (w := get_weight(f'{conv_name}.depthwise_conv.chunkwise_conv.weight')) is not None:
            conv.chunkwise_conv_weight = mx.transpose(w, (0, 2, 1))  # (out, 1, k) -> (out, k, 1)
        if (w := get_weight(f'{conv_name}.depthwise_conv.chunkwise_conv.bias')) is not None:
            conv.chunkwise_conv_bias = w
        if (w := get_weight(f'{conv_name}.depthwise_conv.chunkwise_conv_scale')) is not None:
            conv.chunkwise_conv_scale = w
