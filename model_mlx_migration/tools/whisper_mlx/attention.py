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
Attention modules for WhisperMLX.

Implements:
- Multi-head attention with optional fused operations
- Cross-attention for encoder-decoder
- Support for preallocated KV-cache
- Sliding window attention for encoder (OPT-1.1-SW)
"""


import mlx.core as mx
import mlx.nn as nn


def create_sliding_window_mask(
    seq_len: int,
    window_size: int,
    dtype: mx.Dtype = mx.float16,
) -> mx.array:
    """
    Create a sliding window attention mask for encoder self-attention.

    Each position can attend to positions within `window_size // 2` distance.
    Positions outside the window are masked with -inf.

    OPT-1.1-SW: Sliding Window Encoder Attention
    - Expected 1.5-2x encoder speedup for long sequences
    - Quality impact: minimal for local context models like Whisper

    Args:
        seq_len: Sequence length
        window_size: Total window size (each position attends to window_size tokens)
        dtype: Data type for the mask

    Returns:
        Mask of shape (seq_len, seq_len), where:
        - 0.0 = attend (within window)
        - -inf = mask out (outside window)

    Example:
        With seq_len=5, window_size=3:
        [[  0,   0,-inf,-inf,-inf],
         [  0,   0,   0,-inf,-inf],
         [-inf,  0,   0,   0,-inf],
         [-inf,-inf,  0,   0,   0],
         [-inf,-inf,-inf,  0,   0]]
    """
    # Create position indices
    positions = mx.arange(seq_len)

    # Distance matrix: distance[i, j] = |i - j|
    # Shape: (seq_len, seq_len)
    distance = mx.abs(positions[:, None] - positions[None, :])

    # Within window: distance <= window_size // 2
    half_window = window_size // 2
    within_window = distance <= half_window

    # Create mask: 0 for within window, -inf for outside
    return mx.where(within_window, mx.array(0.0, dtype=dtype), mx.array(float("-inf"), dtype=dtype))



class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with support for:
    - Self-attention (decoder)
    - Cross-attention (encoder-decoder)
    - Preallocated KV-cache
    - Optional fused attention (mx.fast.scaled_dot_product_attention)
    - Optional attention weight extraction for AlignAtt (J2)
    """

    def __init__(
        self,
        n_state: int,
        n_head: int,
        use_fused: bool = True,
        return_attention_weights: bool = False,
    ):
        """
        Args:
            n_state: Hidden dimension
            n_head: Number of attention heads
            use_fused: Use MLX fused attention if available
            return_attention_weights: Return attention weights for AlignAtt (J2).
                                     When True, uses slower non-SDPA path.
        """
        super().__init__()
        self.n_head = n_head
        self.n_state = n_state
        self.head_dim = n_state // n_head
        self.use_fused = use_fused
        self.return_attention_weights = return_attention_weights

        # Projections
        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)

    @property
    def _is_quantized(self) -> bool:
        """
        Check if this layer has been quantized.

        QuantizedLinear replaces Linear after quantization and has a
        different internal structure (no .weight attribute with same shape).
        This property detects quantization to disable fused QKV projection.
        """
        from mlx.nn import QuantizedLinear
        return isinstance(self.query, QuantizedLinear)

    def _get_fused_qkv_weight(self) -> mx.array:
        """
        Get cached fused QKV weight for self-attention (OPT-NEW-7).
        Lazily creates and caches the concatenated weight matrix.
        """
        if not hasattr(self, '_cached_qkv_weight') or self._cached_qkv_weight is None:
            # Concatenate Q, K, V weights into single matrix for fused matmul
            self._cached_qkv_weight = mx.concatenate(
                [self.query.weight, self.key.weight, self.value.weight], axis=0,
            )
        return self._cached_qkv_weight

    def _get_fused_kv_weight(self) -> mx.array:
        """
        Get cached fused KV weight for cross-attention (OPT-NEW-7).
        Lazily creates and caches the concatenated weight matrix.
        """
        if not hasattr(self, '_cached_kv_weight') or self._cached_kv_weight is None:
            # Concatenate K, V weights into single matrix for fused matmul
            self._cached_kv_weight = mx.concatenate(
                [self.key.weight, self.value.weight], axis=0,
            )
        return self._cached_kv_weight

    def __call__(
        self,
        x: mx.array,
        xa: mx.array | None = None,
        mask: mx.array | None = None,
        kv_cache: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array], mx.array | None]:
        """
        Forward pass.

        Args:
            x: Query input, shape (batch, seq_len, n_state)
            xa: Key/value input for cross-attention (None for self-attention)
            mask: Attention mask (causal mask for decoder)
            kv_cache: Cached (K, V) tensors from previous steps

        Returns:
            Tuple of:
            - Output tensor, shape (batch, seq_len, n_state)
            - Updated (K, V) cache
            - Attention weights (optional, for visualization)
        """
        if xa is None:
            # Self-attention
            if self._is_quantized:
                # Quantized mode: use separate projections
                # (QuantizedLinear doesn't support weight concatenation)
                q = self.query(x)
                k = self.key(x)
                v = self.value(x)
            else:
                # Non-quantized: OPT-NEW-7 fused QKV projection (single matmul)
                qkv_weight = self._get_fused_qkv_weight()
                qkv = x @ qkv_weight.T
                # Split and add biases (query has bias, key has no bias, value has bias)
                q = qkv[..., :self.n_state] + self.query.bias
                k = qkv[..., self.n_state:2*self.n_state]
                v = qkv[..., 2*self.n_state:] + self.value.bias

            if kv_cache is not None:
                # Append to cache
                k = mx.concatenate([kv_cache[0], k], axis=1)
                v = mx.concatenate([kv_cache[1], v], axis=1)
        elif kv_cache is None:
            # Cross-attention, first call
            q = self.query(x)
            if self._is_quantized:
                # Quantized mode: use separate projections
                k = self.key(xa)
                v = self.value(xa)
            else:
                # Non-quantized: OPT-NEW-7 fused KV projection
                kv_weight = self._get_fused_kv_weight()
                kv = xa @ kv_weight.T
                # Split and add bias (key has no bias, value has bias)
                k = kv[..., :self.n_state]
                v = kv[..., self.n_state:] + self.value.bias
        else:
            # Cross-attention, subsequent calls - reuse cached K, V
            q = self.query(x)
            k, v = kv_cache

        # Compute attention
        out, attn_weights = self._attention(q, k, v, mask)

        return self.out(out), (k, v), attn_weights

    def _attention(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        mask: mx.array | None = None,
    ) -> tuple[mx.array, mx.array | None]:
        """
        Compute scaled dot-product attention.

        Supports both standard autoregressive decoding and speculative decoding
        with KV cache where we process multiple tokens at once.

        Uses mx.fast.scaled_dot_product_attention (OPT-NEW-2) for fused Metal kernel
        unless return_attention_weights is True (for AlignAtt J2).

        Args:
            q: Query tensor, shape (batch, q_len, n_state)
            k: Key tensor, shape (batch, kv_len, n_state)
            v: Value tensor, shape (batch, kv_len, n_state)
            mask: Optional attention mask (shape: n_ctx, n_ctx)

        Returns:
            Tuple of (output, attention_weights)
            - attention_weights: (batch, n_head, q_len, kv_len) if return_attention_weights
        """
        batch_size, q_len, _ = q.shape
        kv_len = k.shape[1]

        # Reshape to (batch, n_head, seq_len, head_dim) for SDPA
        q = q.reshape(batch_size, q_len, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, kv_len, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, kv_len, self.n_head, self.head_dim).transpose(0, 2, 1, 3)

        # Scale factor
        scale = self.head_dim ** -0.5

        # Prepare mask
        sdpa_mask = None
        if mask is not None:
            # Check if mask is already the correct shape (custom mask case)
            if mask.shape == (q_len, kv_len):
                # Custom mask already in correct shape, use directly
                sdpa_mask = mask
            elif q_len == kv_len:
                # Standard case: no KV cache, use full causal mask
                sdpa_mask = mask[:q_len, :kv_len]
            else:
                # KV cache case: queries are at positions [offset, offset+q_len)
                offset = kv_len - q_len
                sdpa_mask = mask[offset:offset + q_len, :kv_len]

        # J2 AlignAtt: Non-SDPA path to extract attention weights
        if self.return_attention_weights:
            # Manual attention computation (slower but returns weights)
            # (batch, n_head, q_len, head_dim) @ (batch, n_head, head_dim, kv_len)
            # = (batch, n_head, q_len, kv_len)
            scores = (q @ k.transpose(0, 1, 3, 2)) * scale

            if sdpa_mask is not None:
                scores = scores + sdpa_mask

            # Softmax over key dimension
            attn_weights = mx.softmax(scores, axis=-1)

            # Weighted sum of values
            # (batch, n_head, q_len, kv_len) @ (batch, n_head, kv_len, head_dim)
            # = (batch, n_head, q_len, head_dim)
            out = attn_weights @ v

            # Reshape back to (batch, seq_len, n_state)
            out = out.transpose(0, 2, 1, 3).reshape(batch_size, q_len, self.n_state)

            return out, attn_weights

        # OPT-NEW-2: Use mx.fast.scaled_dot_product_attention for fused Metal kernel
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=sdpa_mask)

        # Reshape back to (batch, seq_len, n_state)
        out = out.transpose(0, 2, 1, 3).reshape(batch_size, q_len, self.n_state)

        # Note: SDPA doesn't return attention weights, return None
        return out, None


class ResidualAttentionBlock(nn.Module):
    """
    Transformer block with:
    - Self-attention + residual
    - Optional cross-attention + residual
    - MLP + residual
    """

    def __init__(
        self,
        n_state: int,
        n_head: int,
        cross_attention: bool = False,
        use_fused: bool = True,
        return_cross_attention_weights: bool = False,
    ):
        """
        Args:
            n_state: Hidden dimension
            n_head: Number of attention heads
            cross_attention: Include cross-attention (for decoder)
            use_fused: Use fused attention operations
            return_cross_attention_weights: Return cross-attention weights for AlignAtt (J2)
        """
        super().__init__()

        # Self-attention (never returns weights - self-attention not useful for AlignAtt)
        self.attn = MultiHeadAttention(n_state, n_head, use_fused=use_fused)
        self.attn_ln = nn.LayerNorm(n_state)

        # Cross-attention (decoder only) - optionally returns weights for AlignAtt
        self.cross_attn = (
            MultiHeadAttention(
                n_state, n_head, use_fused=use_fused,
                return_attention_weights=return_cross_attention_weights,
            )
            if cross_attention
            else None
        )
        self.cross_attn_ln = nn.LayerNorm(n_state) if cross_attention else None

        # MLP
        n_mlp = n_state * 4
        self.mlp1 = nn.Linear(n_state, n_mlp)
        self.mlp2 = nn.Linear(n_mlp, n_state)
        self.mlp_ln = nn.LayerNorm(n_state)

    def __call__(
        self,
        x: mx.array,
        xa: mx.array | None = None,
        mask: mx.array | None = None,
        kv_cache: tuple[tuple[mx.array, mx.array], tuple[mx.array, mx.array]] | None = None,
    ) -> tuple[mx.array, tuple, mx.array | None]:
        """
        Forward pass.

        Args:
            x: Input tensor, shape (batch, seq_len, n_state)
            xa: Encoder output for cross-attention
            mask: Causal mask for self-attention
            kv_cache: Tuple of ((self_k, self_v), (cross_k, cross_v))

        Returns:
            Tuple of:
            - Output tensor
            - Updated KV cache tuple
            - Cross-attention weights (for alignment)
        """
        # Unpack cache
        self_kv, cross_kv = kv_cache if kv_cache else (None, None)

        # Self-attention with residual
        # OPT-NEW-4: Use mx.fast.layer_norm for fused Metal kernel
        attn_input = mx.fast.layer_norm(x, self.attn_ln.weight, self.attn_ln.bias, eps=1e-5)
        y, self_kv, _ = self.attn(attn_input, mask=mask, kv_cache=self_kv)
        x = x + y

        # Cross-attention with residual (if present)
        cross_qk = None
        if self.cross_attn is not None:
            # OPT-NEW-4: Use mx.fast.layer_norm for fused Metal kernel
            cross_input = mx.fast.layer_norm(x, self.cross_attn_ln.weight, self.cross_attn_ln.bias, eps=1e-5)
            y, cross_kv, cross_qk = self.cross_attn(
                cross_input, xa, kv_cache=cross_kv,
            )
            x = x + y

        # MLP with residual
        # OPT-NEW-4: Use mx.fast.layer_norm for fused Metal kernel
        mlp_input = mx.fast.layer_norm(x, self.mlp_ln.weight, self.mlp_ln.bias, eps=1e-5)
        x = x + self.mlp2(nn.gelu(self.mlp1(mlp_input)))

        return x, (self_kv, cross_kv), cross_qk
