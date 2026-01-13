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
KV-Cache implementations for WhisperMLX decoder.

Provides:
- Standard dynamic cache (like mlx-whisper)
- Preallocated cache (zero-allocation during decoding)
- INT8 quantized cache (50% memory reduction on cross-attention)

OPT-2.3: INT8 KV Cache Quantization
Stores cross-attention K/V as INT8 with per-tensor scaling.
This provides 50% memory reduction for the cross-attention cache
(encoder K/V), which is the largest memory consumer during decoding.
"""


import mlx.core as mx


def quantize_to_int8(x: mx.array) -> tuple[mx.array, mx.array]:
    """
    Quantize tensor to INT8 with per-tensor scaling.

    OPT-2.3: INT8 quantization for KV cache.

    Args:
        x: Input tensor (typically float16 or float32)

    Returns:
        Tuple of (quantized_int8, scale)
        - quantized_int8: Values in [-127, 127] as int8
        - scale: Scale factor for dequantization (x_max / 127)
    """
    x_max = mx.max(mx.abs(x))
    # Avoid division by zero for all-zero tensors
    scale = mx.maximum(x_max, 1e-10) / 127.0
    quantized = mx.round(x / scale).astype(mx.int8)
    return quantized, scale


def dequantize_from_int8(quantized: mx.array, scale: mx.array, dtype: mx.Dtype = mx.float16) -> mx.array:
    """
    Dequantize INT8 tensor back to floating point.

    Args:
        quantized: INT8 tensor
        scale: Scale factor from quantization
        dtype: Output dtype

    Returns:
        Dequantized tensor
    """
    return quantized.astype(dtype) * scale


class DynamicKVCache:
    """
    Dynamic KV cache that grows with each token.

    This is the standard approach used by mlx-whisper.
    New memory is allocated on each token generation.
    """

    def __init__(self, n_layers: int):
        """
        Args:
            n_layers: Number of decoder layers
        """
        self.n_layers = n_layers
        # Each layer has: ((self_k, self_v), (cross_k, cross_v))
        self._cache: list[tuple | None] = [None] * n_layers

    def get(self, layer_idx: int) -> tuple | None:
        """Get cache for a layer."""
        return self._cache[layer_idx]

    def update(
        self,
        layer_idx: int,
        self_kv: tuple[mx.array, mx.array],
        cross_kv: tuple[mx.array, mx.array],
    ):
        """Update cache for a layer."""
        self._cache[layer_idx] = (self_kv, cross_kv)

    def reset(self):
        """Clear all cached values."""
        self._cache = [None] * self.n_layers

    @property
    def position(self) -> int:
        """Current position in the cache (number of tokens generated)."""
        if self._cache[0] is None:
            return 0
        return self._cache[0][0][0].shape[1]  # self_k sequence length


class PreallocatedKVCache:
    """
    Preallocated KV cache with in-place updates.

    Allocates all memory upfront and writes to fixed positions.
    This eliminates memory allocation during decoding, providing
    ~1.2-1.3x speedup over dynamic allocation.

    The cross-attention K/V is computed once from encoder output
    and reused for all decoder steps.
    """

    def __init__(
        self,
        max_seq_len: int,
        n_layers: int,
        n_heads: int,
        head_dim: int,
        dtype: mx.Dtype = mx.float16,
    ):
        """
        Args:
            max_seq_len: Maximum sequence length (n_text_ctx, typically 448)
            n_layers: Number of decoder layers
            n_heads: Number of attention heads
            head_dim: Dimension per attention head
            dtype: Data type for cache tensors
        """
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.dtype = dtype

        # Preallocate self-attention caches
        # Shape: (n_layers, max_seq_len, n_heads * head_dim)
        n_state = n_heads * head_dim
        self._self_k = mx.zeros((n_layers, max_seq_len, n_state), dtype=dtype)
        self._self_v = mx.zeros((n_layers, max_seq_len, n_state), dtype=dtype)

        # Cross-attention cache (set once from encoder output)
        self._cross_k: mx.array | None = None  # (n_layers, encoder_len, n_state)
        self._cross_v: mx.array | None = None

        # Current position
        self._position = 0

    def set_cross_attention(
        self,
        layer_idx: int,
        cross_k: mx.array,
        cross_v: mx.array,
    ):
        """
        Set cross-attention K/V from encoder output.

        Called once per sequence, before decoding starts.
        """
        if self._cross_k is None:
            # Initialize on first call
            encoder_len = cross_k.shape[1]
            n_state = cross_k.shape[2]
            self._cross_k = mx.zeros(
                (self.n_layers, encoder_len, n_state), dtype=self.dtype,
            )
            self._cross_v = mx.zeros(
                (self.n_layers, encoder_len, n_state), dtype=self.dtype,
            )

        # Store cross-attention for this layer
        self._cross_k[layer_idx] = cross_k.squeeze(0)
        self._cross_v[layer_idx] = cross_v.squeeze(0)

    def update_self_attention(
        self,
        layer_idx: int,
        k: mx.array,
        v: mx.array,
    ):
        """
        Update self-attention cache with new K/V at current position.

        Args:
            layer_idx: Layer index
            k: New key, shape (batch, 1, n_state)
            v: New value, shape (batch, 1, n_state)
        """
        # Write to current position (in-place update)
        self._self_k[layer_idx, self._position] = k.squeeze(0).squeeze(0)
        self._self_v[layer_idx, self._position] = v.squeeze(0).squeeze(0)

    def get_self_attention(
        self,
        layer_idx: int,
    ) -> tuple[mx.array, mx.array]:
        """
        Get self-attention K/V up to current position.

        Returns:
            Tuple of (K, V) with shape (1, position+1, n_state)
        """
        pos = self._position + 1
        k = self._self_k[layer_idx, :pos][None]
        v = self._self_v[layer_idx, :pos][None]
        return k, v

    def get_cross_attention(
        self,
        layer_idx: int,
    ) -> tuple[mx.array, mx.array]:
        """
        Get cross-attention K/V for a layer.

        Returns:
            Tuple of (K, V) with shape (1, encoder_len, n_state)
        """
        return self._cross_k[layer_idx][None], self._cross_v[layer_idx][None]

    def advance(self):
        """Advance position after generating a token."""
        self._position += 1

    def reset(self):
        """Reset cache for new sequence."""
        self._position = 0
        self._cross_k = None
        self._cross_v = None
        # Note: We don't zero out self_k/self_v since they'll be overwritten

    @property
    def position(self) -> int:
        """Current position in the cache."""
        return self._position


class QuantizedKVCache:
    """
    INT8 quantized KV cache for memory efficiency (OPT-2.3).

    Stores cross-attention K/V as INT8 with per-tensor scaling,
    reducing memory usage by 50% for the cross-attention cache.
    Self-attention K/V remains in float16 for accuracy during
    incremental updates.

    Memory savings:
    - Cross-attention (encoder K/V): 50% reduction
    - Self-attention: No change (accuracy-critical)

    Expected WER impact: <0.1% degradation
    """

    def __init__(
        self,
        max_seq_len: int,
        n_layers: int,
        n_heads: int,
        head_dim: int,
        dtype: mx.Dtype = mx.float16,
    ):
        """
        Args:
            max_seq_len: Maximum sequence length (n_text_ctx, typically 448)
            n_layers: Number of decoder layers
            n_heads: Number of attention heads
            head_dim: Dimension per attention head
            dtype: Data type for self-attention cache (cross-attn is INT8)
        """
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.dtype = dtype

        # Self-attention: Use float16 for accuracy (preallocated)
        n_state = n_heads * head_dim
        self._self_k = mx.zeros((n_layers, max_seq_len, n_state), dtype=dtype)
        self._self_v = mx.zeros((n_layers, max_seq_len, n_state), dtype=dtype)

        # Cross-attention: Use INT8 for memory savings
        # Stores (quantized_int8, scale) per layer
        self._cross_k_int8: mx.array | None = None  # (n_layers, encoder_len, n_state) int8
        self._cross_v_int8: mx.array | None = None
        self._cross_k_scale: mx.array | None = None  # (n_layers,) float16
        self._cross_v_scale: mx.array | None = None

        # Current position
        self._position = 0

    def set_cross_attention(
        self,
        layer_idx: int,
        cross_k: mx.array,
        cross_v: mx.array,
    ):
        """
        Set cross-attention K/V from encoder output with INT8 quantization.

        Called once per sequence, before decoding starts.
        """
        # Squeeze batch dimension
        cross_k = cross_k.squeeze(0)  # (encoder_len, n_state)
        cross_v = cross_v.squeeze(0)

        if self._cross_k_int8 is None:
            # Initialize on first call
            encoder_len = cross_k.shape[0]
            n_state = cross_k.shape[1]
            self._cross_k_int8 = mx.zeros(
                (self.n_layers, encoder_len, n_state), dtype=mx.int8,
            )
            self._cross_v_int8 = mx.zeros(
                (self.n_layers, encoder_len, n_state), dtype=mx.int8,
            )
            self._cross_k_scale = mx.zeros((self.n_layers,), dtype=self.dtype)
            self._cross_v_scale = mx.zeros((self.n_layers,), dtype=self.dtype)

        # Quantize and store
        k_int8, k_scale = quantize_to_int8(cross_k)
        v_int8, v_scale = quantize_to_int8(cross_v)

        self._cross_k_int8[layer_idx] = k_int8
        self._cross_v_int8[layer_idx] = v_int8
        self._cross_k_scale[layer_idx] = k_scale
        self._cross_v_scale[layer_idx] = v_scale

    def update_self_attention(
        self,
        layer_idx: int,
        k: mx.array,
        v: mx.array,
    ):
        """
        Update self-attention cache with new K/V at current position.

        Self-attention uses full precision for accuracy during incremental updates.
        """
        self._self_k[layer_idx, self._position] = k.squeeze(0).squeeze(0)
        self._self_v[layer_idx, self._position] = v.squeeze(0).squeeze(0)

    def get_self_attention(
        self,
        layer_idx: int,
    ) -> tuple[mx.array, mx.array]:
        """
        Get self-attention K/V up to current position.

        Returns full precision values.
        """
        pos = self._position + 1
        k = self._self_k[layer_idx, :pos][None]
        v = self._self_v[layer_idx, :pos][None]
        return k, v

    def get_cross_attention(
        self,
        layer_idx: int,
    ) -> tuple[mx.array, mx.array]:
        """
        Get cross-attention K/V for a layer with dequantization.

        Dequantizes INT8 values back to float16 for attention computation.
        """
        # Dequantize on access
        k = dequantize_from_int8(
            self._cross_k_int8[layer_idx],
            self._cross_k_scale[layer_idx],
            self.dtype,
        )[None]
        v = dequantize_from_int8(
            self._cross_v_int8[layer_idx],
            self._cross_v_scale[layer_idx],
            self.dtype,
        )[None]
        return k, v

    def advance(self):
        """Advance position after generating a token."""
        self._position += 1

    def reset(self):
        """Reset cache for new sequence."""
        self._position = 0
        self._cross_k_int8 = None
        self._cross_v_int8 = None
        self._cross_k_scale = None
        self._cross_v_scale = None

    @property
    def position(self) -> int:
        """Current position in the cache."""
        return self._position

    @property
    def memory_usage_mb(self) -> float:
        """
        Estimate memory usage of the cache in MB.

        Useful for comparing with non-quantized cache.
        """
        n_state = self.n_heads * self.head_dim
        bytes_per_element = 2  # float16

        # Self-attention: full precision
        self_attn_mb = (
            self.n_layers * self.max_seq_len * n_state * bytes_per_element * 2
        ) / (1024 * 1024)

        # Cross-attention: INT8 (1 byte) + scales
        if self._cross_k_int8 is not None:
            encoder_len = self._cross_k_int8.shape[1]
            # INT8 storage (1 byte per element)
            cross_attn_mb = (
                self.n_layers * encoder_len * n_state * 1 * 2
            ) / (1024 * 1024)
            # Scales (negligible)
            cross_attn_mb += (self.n_layers * 2 * bytes_per_element) / (1024 * 1024)
        else:
            cross_attn_mb = 0.0

        return self_attn_mb + cross_attn_mb


class KVCacheManager:
    """
    Unified interface for KV cache management.

    Wraps either dynamic, preallocated, or quantized cache with consistent API.
    """

    def __init__(
        self,
        n_layers: int,
        max_seq_len: int = 448,
        n_heads: int = 20,
        head_dim: int = 64,
        dtype: mx.Dtype = mx.float16,
        preallocate: bool = False,
        quantize: bool = False,
    ):
        """
        Args:
            n_layers: Number of decoder layers
            max_seq_len: Maximum sequence length
            n_heads: Number of attention heads
            head_dim: Dimension per head
            dtype: Data type for cache
            preallocate: Use preallocated cache (faster but more memory)
            quantize: Use INT8 quantized cache for cross-attention (OPT-2.3).
                      Provides 50% memory reduction with <0.1% WER impact.
                      Only effective when preallocate=True.
        """
        self.preallocate = preallocate
        self.quantize = quantize

        if quantize and preallocate:
            # OPT-2.3: Quantized cache for memory efficiency
            self._cache = QuantizedKVCache(
                max_seq_len, n_layers, n_heads, head_dim, dtype,
            )
        elif preallocate:
            self._cache = PreallocatedKVCache(
                max_seq_len, n_layers, n_heads, head_dim, dtype,
            )
        else:
            self._cache = DynamicKVCache(n_layers)

    def get(self, layer_idx: int) -> tuple | None:
        """Get cache for a layer (dynamic mode)."""
        if self.preallocate:
            raise ValueError("Use get_self_attention/get_cross_attention for preallocated cache")
        return self._cache.get(layer_idx)

    def update(
        self,
        layer_idx: int,
        self_kv: tuple[mx.array, mx.array],
        cross_kv: tuple[mx.array, mx.array],
    ):
        """Update cache for a layer (dynamic mode)."""
        if self.preallocate:
            raise ValueError("Use update_self_attention for preallocated cache")
        self._cache.update(layer_idx, self_kv, cross_kv)

    def reset(self):
        """Reset cache."""
        self._cache.reset()

    @property
    def position(self) -> int:
        """Current cache position."""
        return self._cache.position
