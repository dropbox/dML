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
MLX implementation of Zipformer encoder matching icefall pretrained weights.

This module implements the exact architecture used in the pretrained model to
enable loading pretrained weights directly without modification.
"""

import math

import mlx.core as mx
import mlx.nn as nn


class RelPositionMultiheadAttention(nn.Module):
    """
    Multi-head attention with relative positional encoding.

    This matches the icefall RelPositionMultiheadAttention architecture.

    Weight mapping from pretrained:
    - in_proj.weight: (in_proj_dim, embed_dim)
    - in_proj.bias: (in_proj_dim,)
    - linear_pos.weight: (num_heads * pos_dim, embed_dim)
    - in_proj2.weight: (attention_dim // 2, embed_dim)
    - out_proj.weight: (embed_dim, attention_dim // 2)
    - out_proj.bias: (embed_dim,)
    - out_proj2.weight: (embed_dim, attention_dim // 2)
    - out_proj2.bias: (embed_dim,)
    """

    def __init__(
        self,
        embed_dim: int,
        attention_dim: int,
        num_heads: int,
        pos_dim: int = 4,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.pos_dim = pos_dim
        self.head_dim = attention_dim // num_heads

        # in_proj projects to: Q (attention_dim) + K (attention_dim) + V1 (attention_dim//2) + P (pos_dim * num_heads)
        in_proj_dim = 2 * attention_dim + attention_dim // 2 + pos_dim * num_heads

        self.in_proj = nn.Linear(embed_dim, in_proj_dim, bias=True)
        self.linear_pos = nn.Linear(embed_dim, num_heads * pos_dim, bias=False)

        # Second value projection for the second attention
        self.in_proj2 = nn.Linear(embed_dim, attention_dim // 2, bias=False)

        # Output projections
        self.out_proj = nn.Linear(attention_dim // 2, embed_dim, bias=True)
        self.out_proj2 = nn.Linear(attention_dim // 2, embed_dim, bias=True)

    def _compute_positional_encoding(self, seq_len: int, batch_size: int) -> mx.array:
        """Compute relative positional encoding."""
        # Create position indices: -(seq_len-1) to (seq_len-1)
        positions = mx.arange(-(seq_len - 1), seq_len, dtype=mx.float32)
        # Simple positional encoding using sin/cos
        pe = mx.zeros((2 * seq_len - 1, self.embed_dim))
        div_term = mx.exp(
            mx.arange(0, self.embed_dim, 2, dtype=mx.float32) *
            (-math.log(10000.0) / self.embed_dim),
        )
        pe = pe.at[:, 0::2].add(mx.sin(positions[:, None] * div_term))
        pe = pe.at[:, 1::2].add(mx.cos(positions[:, None] * div_term))
        return mx.broadcast_to(pe[None, :, :], (batch_size, 2 * seq_len - 1, self.embed_dim))

    def _rel_shift(self, x: mx.array) -> mx.array:
        """Convert relative position scores to absolute positions."""
        # x: (batch * heads, seq, 2*seq - 1)
        batch_heads, seq_len, rel_len = x.shape

        # Pad and reshape to extract the correct diagonal
        x = mx.pad(x, [(0, 0), (0, 0), (0, 1)])
        x = mx.reshape(x, (batch_heads, -1, seq_len))
        x = x[:, 1:, :]
        x = mx.reshape(x, (batch_heads, seq_len, rel_len))
        x = x[:, :, :seq_len]

        return x

    def __call__(
        self,
        x: mx.array,
        pos_emb: mx.array,
        attn_mask: mx.array | None = None,
        key_padding_mask: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """
        Forward pass.

        Args:
            x: Input of shape (seq_len, batch_size, embed_dim)
            pos_emb: Positional embedding of shape (batch_size, 2*seq_len-1, embed_dim)
            attn_mask: Optional attention mask
            key_padding_mask: Optional key padding mask

        Returns:
            - First attention output (seq_len, batch_size, embed_dim)
            - Attention weights (batch_size * num_heads, seq_len, seq_len)
        """
        seq_len, batch_size, _ = x.shape

        # Project input
        proj = self.in_proj(x)  # (seq_len, batch, in_proj_dim)

        # Split into Q, K, V1, P
        q = proj[:, :, : self.attention_dim]
        k = proj[:, :, self.attention_dim : 2 * self.attention_dim]
        v1 = proj[:, :, 2 * self.attention_dim : 2 * self.attention_dim + self.attention_dim // 2]
        p = proj[:, :, 2 * self.attention_dim + self.attention_dim // 2 :]

        # Project position
        pos_proj = self.linear_pos(pos_emb)  # (batch, 2*seq-1, num_heads * pos_dim)

        # Reshape for multi-head attention
        # Q, K: (seq, batch, heads, head_dim)
        q = mx.reshape(q, (seq_len, batch_size, self.num_heads, self.head_dim))
        k = mx.reshape(k, (seq_len, batch_size, self.num_heads, self.head_dim))
        v1 = mx.reshape(v1, (seq_len, batch_size, self.num_heads, self.head_dim // 2))
        p = mx.reshape(p, (seq_len, batch_size, self.num_heads, self.pos_dim))

        # Transpose to (batch, heads, seq, dim) -> then merge batch and heads
        q = mx.transpose(q, (1, 2, 0, 3))  # (batch, heads, seq, head_dim)
        k = mx.transpose(k, (1, 2, 0, 3))
        v1 = mx.transpose(v1, (1, 2, 0, 3))
        p = mx.transpose(p, (1, 2, 0, 3))  # (batch, heads, seq, pos_dim)

        # Position encoding: (batch, 2*seq-1, heads, pos_dim)
        pos_proj = mx.reshape(pos_proj, (batch_size, -1, self.num_heads, self.pos_dim))
        pos_proj = mx.transpose(pos_proj, (0, 2, 1, 3))  # (batch, heads, 2*seq-1, pos_dim)

        # Content attention: Q @ K^T
        scale = self.head_dim ** -0.5
        content_score = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) * scale

        # Position attention: P @ Pos^T
        pos_score = mx.matmul(p, mx.transpose(pos_proj, (0, 1, 3, 2)))

        # Reshape for rel_shift
        pos_score = mx.reshape(pos_score, (batch_size * self.num_heads, seq_len, -1))
        pos_score = self._rel_shift(pos_score)
        pos_score = mx.reshape(pos_score, (batch_size, self.num_heads, seq_len, seq_len))

        # Combined attention scores
        attn_score = content_score + pos_score

        # Apply masks
        if attn_mask is not None:
            attn_score = attn_score + attn_mask

        if key_padding_mask is not None:
            # key_padding_mask: (batch, seq) with True for padded positions
            mask = mx.expand_dims(mx.expand_dims(key_padding_mask, 1), 1)
            attn_score = mx.where(mask, mx.full(attn_score.shape, -1e9), attn_score)

        attn_weights = mx.softmax(attn_score, axis=-1)

        # Apply attention to first value
        out1 = mx.matmul(attn_weights, v1)  # (batch, heads, seq, head_dim//2)
        out1 = mx.transpose(out1, (2, 0, 1, 3))  # (seq, batch, heads, head_dim//2)
        out1 = mx.reshape(out1, (seq_len, batch_size, self.attention_dim // 2))
        out1 = self.out_proj(out1)

        # Return attention weights for use by forward2 and pooling
        attn_weights_flat = mx.reshape(
            attn_weights, (batch_size * self.num_heads, seq_len, seq_len),
        )

        return out1, attn_weights_flat

    def forward2(
        self,
        x: mx.array,
        attn_weights: mx.array,
    ) -> mx.array:
        """
        Second attention forward using precomputed weights.

        Args:
            x: Input of shape (seq_len, batch_size, embed_dim)
            attn_weights: Attention weights from forward() (batch*heads, seq, seq)

        Returns:
            Output of shape (seq_len, batch_size, embed_dim)
        """
        seq_len, batch_size, _ = x.shape

        # Project to second values
        v2 = self.in_proj2(x)  # (seq, batch, attention_dim // 2)
        v2 = mx.reshape(v2, (seq_len, batch_size, self.num_heads, self.head_dim // 2))
        v2 = mx.transpose(v2, (1, 2, 0, 3))  # (batch, heads, seq, head_dim//2)

        # Reshape attention weights
        attn_weights = mx.reshape(
            attn_weights, (batch_size, self.num_heads, seq_len, seq_len),
        )

        # Apply attention
        out2 = mx.matmul(attn_weights, v2)
        out2 = mx.transpose(out2, (2, 0, 1, 3))
        out2 = mx.reshape(out2, (seq_len, batch_size, self.attention_dim // 2))
        out2 = self.out_proj2(out2)

        return out2


class PoolingModule(nn.Module):
    """
    Pooling module that computes running average and projects.

    Weight mapping from pretrained:
    - proj.weight: (d_model, d_model)
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def __call__(
        self,
        x: mx.array,
        src_key_padding_mask: mx.array | None = None,
    ) -> mx.array:
        """
        Forward pass.

        Args:
            x: Input of shape (seq_len, batch_size, d_model)
            src_key_padding_mask: Optional mask (batch, seq) with True for padded

        Returns:
            Output of shape (seq_len, batch_size, d_model)
        """
        seq_len = x.shape[0]

        if src_key_padding_mask is not None:
            # Apply padding mask
            mask = mx.logical_not(src_key_padding_mask).astype(x.dtype)  # (batch, seq)
            cum_mask = mx.cumsum(mask, axis=1)  # (batch, seq)
            x = mx.cumsum(x, axis=0)  # (seq, batch, d_model)
            pooling_mask = mask / (cum_mask + 1e-8)
            pooling_mask = mx.expand_dims(mx.transpose(pooling_mask, (1, 0)), -1)  # (seq, batch, 1)
            x = x * pooling_mask
        else:
            cum_mask = mx.arange(1, seq_len + 1, dtype=x.dtype)  # (seq,)
            x = mx.cumsum(x, axis=0)
            pooling_mask = (1.0 / cum_mask)[:, None, None]  # (seq, 1, 1)
            x = x * pooling_mask

        x = self.proj(x)
        return x


class FeedforwardModule(nn.Module):
    """
    Feedforward module with SiLU activation.

    Weight mapping from pretrained:
    - in_proj.weight: (feedforward_dim, d_model)
    - in_proj.bias: (feedforward_dim,)
    - out_proj.weight: (d_model, feedforward_dim)
    - out_proj.bias: (d_model,)
    """

    def __init__(self, d_model: int, feedforward_dim: int):
        super().__init__()
        self.in_proj = nn.Linear(d_model, feedforward_dim, bias=True)
        self.out_proj = nn.Linear(feedforward_dim, d_model, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass with SiLU activation."""
        x = self.in_proj(x)
        x = nn.silu(x)
        x = self.out_proj(x)
        return x


class ConvolutionModule(nn.Module):
    """
    Convolution module with pointwise and depthwise convolutions.

    Weight mapping from pretrained:
    - pointwise_conv1.weight: (2*d_model, 1, d_model) - MLX format
    - pointwise_conv1.bias: (2*d_model,)
    - depthwise_conv.weight: (d_model, kernel_size, 1) - MLX format
    - depthwise_conv.bias: (d_model,)
    - depthwise_conv.chunkwise_conv_scale: (2, d_model, kernel_size) - chunk edge scaling
    - pointwise_conv2.weight: (d_model, 1, d_model) - MLX format
    - pointwise_conv2.bias: (d_model,)
    """

    def __init__(self, d_model: int, kernel_size: int = 31):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size

        # Pointwise convolutions as 1D conv with kernel_size=1
        self.pointwise_conv1_weight = mx.zeros((2 * d_model, 1, d_model))
        self.pointwise_conv1_bias = mx.zeros((2 * d_model,))

        # Depthwise convolution
        self.depthwise_conv_weight = mx.zeros((d_model, kernel_size, 1))
        self.depthwise_conv_bias = mx.zeros((d_model,))
        # Chunk edge scaling for streaming (left and right edge corrections)
        self.depthwise_conv_chunkwise_conv_scale = mx.zeros((2, d_model, kernel_size))

        # Second pointwise conv
        self.pointwise_conv2_weight = mx.zeros((d_model, 1, d_model))
        self.pointwise_conv2_bias = mx.zeros((d_model,))

    def _get_chunk_scale(self, chunk_size: int) -> mx.array:
        """Get chunk edge scaling factors.

        The convolution output needs to be scaled at chunk boundaries
        to account for edge effects. The scale is learned during training.

        Args:
            chunk_size: Size of the current chunk (seq_len)

        Returns:
            Scale tensor of shape (1, chunk_size, d_model)
        """
        left_edge = self.depthwise_conv_chunkwise_conv_scale[0]   # (d_model, kernel_size)
        right_edge = self.depthwise_conv_chunkwise_conv_scale[1]  # (d_model, kernel_size)

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
        Forward pass.

        Args:
            x: Input of shape (seq_len, batch_size, d_model)
            src_key_padding_mask: Optional mask

        Returns:
            Output of shape (seq_len, batch_size, d_model)
        """
        seq_len, batch_size, d_model = x.shape

        # Apply padding mask
        if src_key_padding_mask is not None:
            mask = mx.expand_dims(mx.transpose(src_key_padding_mask, (1, 0)), -1)
            x = mx.where(mask, mx.zeros_like(x), x)

        # Transpose to (batch, seq, d_model) for conv1d
        x = mx.transpose(x, (1, 0, 2))

        # First pointwise conv: (batch, seq, d_model) -> (batch, seq, 2*d_model)
        x = mx.conv1d(x, self.pointwise_conv1_weight)
        x = x + mx.reshape(self.pointwise_conv1_bias, (1, 1, -1))

        # Split and apply GLU
        x, gate = mx.split(x, 2, axis=-1)
        x = x * mx.sigmoid(gate)

        # Depthwise conv with padding
        pad_len = (self.kernel_size - 1) // 2
        x = mx.pad(x, [(0, 0), (pad_len, pad_len), (0, 0)])
        x = mx.conv1d(x, self.depthwise_conv_weight, groups=self.d_model)
        x = x + mx.reshape(self.depthwise_conv_bias, (1, 1, -1))
        # Apply chunk edge scaling (for streaming compatibility)
        chunk_scale = self._get_chunk_scale(seq_len)
        x = x * chunk_scale

        # SiLU activation
        x = nn.silu(x)

        # Second pointwise conv
        x = mx.conv1d(x, self.pointwise_conv2_weight)
        x = x + mx.reshape(self.pointwise_conv2_bias, (1, 1, -1))

        # Transpose back to (seq, batch, d_model)
        x = mx.transpose(x, (1, 0, 2))

        return x


class BasicNorm(nn.Module):
    """
    Simple normalization (RMS norm variant).

    No weights from pretrained (just eps parameter which we skip).
    """

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        """Apply normalization."""
        variance = mx.mean(x * x, axis=-1, keepdims=True)
        return x * mx.rsqrt(variance + self.eps)


class ZipformerEncoderLayer(nn.Module):
    """
    Single Zipformer encoder layer matching icefall architecture.

    Weight mapping from pretrained:
    - bypass_scale: scalar
    - self_attn.* (RelPositionMultiheadAttention)
    - pooling.* (PoolingModule)
    - feed_forward1.*, feed_forward2.*, feed_forward3.* (FeedforwardModule)
    - conv_module1.*, conv_module2.* (ConvolutionModule)
    """

    def __init__(
        self,
        d_model: int = 384,
        attention_dim: int = 192,
        num_heads: int = 8,
        feedforward_dim: int = 1024,
        kernel_size: int = 31,
        pos_dim: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.attention_dim = attention_dim

        # Attention module
        self.self_attn = RelPositionMultiheadAttention(
            embed_dim=d_model,
            attention_dim=attention_dim,
            num_heads=num_heads,
            pos_dim=pos_dim,
        )

        # Pooling module
        self.pooling = PoolingModule(d_model)

        # Feedforward modules
        self.feed_forward1 = FeedforwardModule(d_model, feedforward_dim)
        self.feed_forward2 = FeedforwardModule(d_model, feedforward_dim)
        self.feed_forward3 = FeedforwardModule(d_model, feedforward_dim)

        # Convolution modules
        self.conv_module1 = ConvolutionModule(d_model, kernel_size)
        self.conv_module2 = ConvolutionModule(d_model, kernel_size)

        # Normalization
        self.norm_final = BasicNorm(d_model)

        # Bypass scale (scalar parameter)
        self.bypass_scale = mx.array(0.5)

    def __call__(
        self,
        src: mx.array,
        pos_emb: mx.array,
        attn_mask: mx.array | None = None,
        src_key_padding_mask: mx.array | None = None,
    ) -> mx.array:
        """
        Forward pass through encoder layer.

        Args:
            src: Input of shape (seq_len, batch_size, d_model)
            pos_emb: Positional embedding (batch, 2*seq-1, d_model)
            attn_mask: Optional attention mask
            src_key_padding_mask: Optional padding mask

        Returns:
            Output of shape (seq_len, batch_size, d_model)
        """
        src_orig = src

        # First feedforward (macaron style)
        src = src + self.feed_forward1(src)

        # Self-attention (first output)
        attn_out, attn_weights = self.self_attn(
            src, pos_emb, attn_mask=attn_mask, key_padding_mask=src_key_padding_mask,
        )
        src = src + attn_out

        # Pooling (uses first head of attention weights)
        pooling_out = self.pooling(src, src_key_padding_mask)
        src = src + pooling_out

        # First convolution
        src = src + self.conv_module1(src, src_key_padding_mask)

        # Second feedforward
        src = src + self.feed_forward2(src)

        # Second self-attention using same attention weights
        attn_out2 = self.self_attn.forward2(src, attn_weights)
        src = src + attn_out2

        # Second convolution
        src = src + self.conv_module2(src, src_key_padding_mask)

        # Third feedforward
        src = src + self.feed_forward3(src)

        # Final norm
        src = self.norm_final(src)

        # Bypass connection
        bypass_scale = mx.clip(self.bypass_scale, 0.0, 1.0)
        src = src_orig + bypass_scale * (src - src_orig)

        return src


def load_encoder_layer_weights(
    layer: ZipformerEncoderLayer,
    weights: dict[str, mx.array],
    prefix: str,
) -> None:
    """
    Load pretrained weights into an encoder layer.

    Args:
        layer: The encoder layer to load weights into.
        weights: Dictionary of weights from converted checkpoint.
        prefix: Key prefix (e.g., 'encoder.encoders.0.layers.0.')
    """
    def get_weight(suffix: str) -> mx.array | None:
        key = prefix + suffix
        return weights.get(key)

    # Bypass scale
    if (w := get_weight('bypass_scale')) is not None:
        layer.bypass_scale = w

    # Self-attention
    if (w := get_weight('self_attn.in_proj.weight')) is not None:
        layer.self_attn.in_proj.weight = w
    if (w := get_weight('self_attn.in_proj.bias')) is not None:
        layer.self_attn.in_proj.bias = w
    if (w := get_weight('self_attn.linear_pos.weight')) is not None:
        layer.self_attn.linear_pos.weight = w
    if (w := get_weight('self_attn.in_proj2.weight')) is not None:
        layer.self_attn.in_proj2.weight = w
    if (w := get_weight('self_attn.out_proj.weight')) is not None:
        layer.self_attn.out_proj.weight = w
    if (w := get_weight('self_attn.out_proj.bias')) is not None:
        layer.self_attn.out_proj.bias = w
    if (w := get_weight('self_attn.out_proj2.weight')) is not None:
        layer.self_attn.out_proj2.weight = w
    if (w := get_weight('self_attn.out_proj2.bias')) is not None:
        layer.self_attn.out_proj2.bias = w

    # Pooling
    if (w := get_weight('pooling.proj.weight')) is not None:
        layer.pooling.proj.weight = w

    # Feedforward modules
    for ff_name in ['feed_forward1', 'feed_forward2', 'feed_forward3']:
        ff_module = getattr(layer, ff_name)
        if (w := get_weight(f'{ff_name}.in_proj.weight')) is not None:
            ff_module.in_proj.weight = w
        if (w := get_weight(f'{ff_name}.in_proj.bias')) is not None:
            ff_module.in_proj.bias = w
        if (w := get_weight(f'{ff_name}.out_proj.weight')) is not None:
            ff_module.out_proj.weight = w
        if (w := get_weight(f'{ff_name}.out_proj.bias')) is not None:
            ff_module.out_proj.bias = w

    # Convolution modules
    for conv_name in ['conv_module1', 'conv_module2']:
        conv_module = getattr(layer, conv_name)
        if (w := get_weight(f'{conv_name}.pointwise_conv1.weight')) is not None:
            conv_module.pointwise_conv1_weight = w
        if (w := get_weight(f'{conv_name}.pointwise_conv1.bias')) is not None:
            conv_module.pointwise_conv1_bias = w
        if (w := get_weight(f'{conv_name}.depthwise_conv.weight')) is not None:
            conv_module.depthwise_conv_weight = w
        if (w := get_weight(f'{conv_name}.depthwise_conv.bias')) is not None:
            conv_module.depthwise_conv_bias = w
        if (w := get_weight(f'{conv_name}.pointwise_conv2.weight')) is not None:
            conv_module.pointwise_conv2_weight = w
        if (w := get_weight(f'{conv_name}.pointwise_conv2.bias')) is not None:
            conv_module.pointwise_conv2_bias = w
