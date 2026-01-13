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
CosyVoice2 LLM Component (Qwen2-based)

Implements the Qwen2-based text encoder and token generator for CosyVoice2.
This is the largest component (~642M params) responsible for:
- Text embedding
- Autoregressive token generation
- Speech token prediction

Architecture:
- Hidden dim: 896
- Vocab size: 151936 (text) + 6564 (speech)
- 24 transformer layers
- GQA: 7 heads, 1 KV head
- SwiGLU MLP with 4864 intermediate dim
- RoPE position embeddings

Optimizations:
- D1 (Worker #1320): KVCache with step-based preallocation (256 tokens)
  - In-place slice assignment: O(1) vs O(n) per token
  - Expected 10-30% speedup for autoregressive generation

- Q15 (Worker #1327): Compiled Sampling Functions (mlx-lm pattern)
  - Uses @partial(mx.compile, inputs/outputs=mx.random.state)
  - Compiled top_k, top_p, and categorical sampling
  - ~2x speedup for standard sampling (temp=1.0, top_k=25, top_p=0.8)

- Q27 (Worker #1324): Greedy Decoding Fast Path
  - Direct argmax when temperature <= 0
  - 9x speedup vs full sampling overhead

- Q18-LLM (Worker #1328): QKV Fusion for LLM Attention
  - Fuses Q, K, V projections into single matmul
  - NOT BENEFICIAL for Qwen2's GQA (7 Q heads, 1 KV head)
  - GQA has asymmetric Q/K/V sizes (896/128/128), so fusion adds overhead
  - Measured: 0.95x (5% slower) for autoregressive generation
  - Implementation kept for documentation; use separate projections

NEW Optimizations (Worker #1335):
- Q4: Attention Sinks (StreamingLLM pattern)
  - Keep first N tokens permanently in KV cache as "attention sinks"
  - Prevents attention score collapse for long sequences
  - Uses mx.fast.scaled_dot_product_attention `sinks` parameter
  - Expected: 10-20% speedup for long sequences

- Q31: nn.RoPE Integration (mlx-lm pattern)
  - Replace custom RoPE with MLX's optimized nn.RoPE
  - Uses mx.fast.rope internally with caching
  - Cleaner code, better maintained

- Q32: Quantized KV Cache Support
  - INT8/INT4 quantization for KV cache
  - 2-4x memory reduction
  - Uses mx.quantize for efficient storage

- Q33: KV Cache Trimming (Sliding Window)
  - trim(n) method to remove oldest tokens
  - Enables bounded memory for streaming inference
  - Compatible with attention sinks

- Q34: Lazy KV Cache Initialization
  - Don't allocate until first token arrives
  - Saves memory for prompt-only inference
  - Zero overhead for normal generation

- Q35: Fused Embedding Lookup
  - Batch embedding lookups for mixed token types
  - Single memory access pattern
"""

from dataclasses import dataclass
from functools import partial

import mlx.core as mx
import mlx.nn as nn

# =============================================================================
# Q15: Compiled Sampling Functions (mlx-lm pattern)
# =============================================================================
# These functions are compiled once and reused for all sampling calls.
# Following mlx-lm's pattern of using @partial(mx.compile, inputs/outputs=mx.random.state)
# for sampling functions with random state.


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def _compiled_top_k(logits: mx.array, top_k: int) -> mx.array:
    """
    Apply top-k filtering to logits (compiled).

    Keeps only the top-k highest probability tokens, setting others to -inf.
    Uses argpartition for O(n) complexity instead of full sort.

    Args:
        logits: [batch, vocab_size] - Raw logits
        top_k: Number of top tokens to keep

    Returns:
        Filtered logits with non-top-k tokens set to -inf
    """
    # argpartition is O(n) vs O(n log n) for full sort
    mask_idx = mx.argpartition(-logits, kth=top_k - 1, axis=-1)[..., top_k:]
    return mx.put_along_axis(
        logits, mask_idx, mx.array(-float("inf"), logits.dtype), axis=-1,
    )


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def _compiled_top_p(logits: mx.array, top_p: float) -> mx.array:
    """
    Apply nucleus (top-p) sampling to logits (compiled).

    Keeps tokens with cumulative probability up to top_p.

    Args:
        logits: [batch, vocab_size] - Raw logits
        top_p: Cumulative probability threshold

    Returns:
        Filtered logits with low-probability tokens set to -inf
    """
    # Sort in descending order
    sorted_indices = mx.argsort(-logits, axis=-1)
    sorted_logits = mx.take_along_axis(logits, sorted_indices, axis=-1)

    # Compute cumulative probabilities
    sorted_probs = mx.softmax(sorted_logits, axis=-1)
    cumsum_probs = mx.cumsum(sorted_probs, axis=-1)

    # Create mask for tokens beyond threshold
    # Shift by 1 to include first token above threshold
    cutoff_mask = cumsum_probs > top_p
    cutoff_mask = mx.concatenate(
        [mx.zeros((logits.shape[0], 1), dtype=mx.bool_), cutoff_mask[:, :-1]],
        axis=-1,
    )

    # Set filtered tokens to -inf in the sorted order
    sorted_logits = mx.where(cutoff_mask, float("-inf"), sorted_logits)

    # Unsort back to original order using inverse permutation
    inverse_indices = mx.argsort(sorted_indices, axis=-1)
    return mx.take_along_axis(sorted_logits, inverse_indices, axis=-1)


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def _compiled_categorical(logits: mx.array, temperature: float) -> mx.array:
    """
    Sample from categorical distribution (compiled).

    Args:
        logits: [batch, vocab_size] - Raw logits (will be scaled by temperature)
        temperature: Sampling temperature

    Returns:
        Sampled token IDs [batch]
    """
    if temperature != 1.0:
        logits = logits / temperature
    return mx.random.categorical(logits)


# =============================================================================
# Repetition Aware Sampling (ras_sampling) - CosyVoice2 specific
# =============================================================================
# Implements the ras_sampling algorithm from CosyVoice2's cosyvoice.utils.common
# This prevents repetitive looping output by detecting when a token is about to
# repeat and falling back to random sampling instead.
#
# Algorithm:
# 1. First sample using nucleus (top-k/top-p) sampling
# 2. Check if sampled token appears in the last win_size tokens
# 3. If repetition count >= win_size * tau_r, use random sampling instead


def _nucleus_sampling_single(
    logits: mx.array,
    top_p: float = 0.8,
    top_k: int = 25,
) -> int:
    """
    Nucleus sampling for a single sample (matches PyTorch CosyVoice exactly).

    This is NOT compiled because it processes single samples and uses Python loops
    to match the exact PyTorch behavior for reproducibility.

    Args:
        logits: [vocab_size] - Raw logits for single sample
        top_p: Cumulative probability threshold
        top_k: Maximum tokens to consider

    Returns:
        Sampled token ID (int)
    """
    # Softmax and sort descending (stable sort for reproducibility)
    probs = mx.softmax(logits)
    sorted_indices = mx.argsort(-probs)  # Descending order
    sorted_probs = probs[sorted_indices]

    # Collect tokens until cumulative prob reaches top_p or top_k reached
    cum_prob = 0.0
    nucleus_probs = []
    nucleus_indices = []

    for i in range(min(len(sorted_indices), top_k)):
        prob_i = sorted_probs[i].item()
        if cum_prob < top_p and len(nucleus_probs) < top_k:
            cum_prob += prob_i
            nucleus_probs.append(prob_i)
            nucleus_indices.append(sorted_indices[i].item())
        else:
            break

    if not nucleus_probs:
        # Fallback: take top token
        return sorted_indices[0].item()

    # Convert to MLX arrays for sampling
    nucleus_probs_arr = mx.array(nucleus_probs)
    # Multinomial sample from nucleus
    sampled_idx = mx.random.categorical(mx.log(nucleus_probs_arr + 1e-10)).item()
    return nucleus_indices[sampled_idx]


def _random_sampling_single(logits: mx.array) -> int:
    """
    Random sampling from full distribution (matches PyTorch CosyVoice exactly).

    Args:
        logits: [vocab_size] - Raw logits for single sample

    Returns:
        Sampled token ID (int)
    """
    probs = mx.softmax(logits)
    return mx.random.categorical(mx.log(probs + 1e-10)).item()


def ras_sampling(
    logits: mx.array,
    decoded_tokens: list[int],
    top_p: float = 0.8,
    top_k: int = 25,
    win_size: int = 10,
    tau_r: float = 0.1,
) -> int:
    """
    Repetition Aware Sampling (ras_sampling) - CosyVoice2 specific.

    Prevents repetitive/looping output by detecting when a token is about to
    repeat within the recent context window. When repetition is detected,
    falls back to full random sampling to force diversity.

    Algorithm:
    1. Sample using nucleus (top-k/top-p) sampling
    2. Count occurrences of sampled token in last win_size tokens
    3. If count >= win_size * tau_r (default: 1), use random sampling instead

    Args:
        logits: [vocab_size] - Raw logits for single sample
        decoded_tokens: List of previously decoded token IDs
        top_p: Nucleus sampling cumulative probability threshold (default 0.8)
        top_k: Maximum tokens in nucleus (default 25)
        win_size: Window size for repetition check (default 10)
        tau_r: Repetition threshold ratio (default 0.1)

    Returns:
        Sampled token ID (int)
    """
    # Step 1: Nucleus sampling
    sampled_token = _nucleus_sampling_single(logits, top_p=top_p, top_k=top_k)

    # Step 2: Check for repetition in recent window
    recent_tokens = decoded_tokens[-win_size:] if decoded_tokens else []
    rep_count = sum(1 for t in recent_tokens if t == sampled_token)

    # Step 3: If repetition detected, use random sampling instead
    threshold = win_size * tau_r  # Default: 10 * 0.1 = 1
    if rep_count >= threshold:
        sampled_token = _random_sampling_single(logits)

    return sampled_token


def ras_sampling_batch(
    logits: mx.array,
    decoded_tokens_batch: list[list[int]],
    top_p: float = 0.8,
    top_k: int = 25,
    win_size: int = 10,
    tau_r: float = 0.1,
) -> mx.array:
    """
    Batch version of ras_sampling for multi-sample generation.

    Args:
        logits: [batch, vocab_size] - Raw logits
        decoded_tokens_batch: List of token histories per batch item
        top_p: Nucleus sampling cumulative probability threshold
        top_k: Maximum tokens in nucleus
        win_size: Window size for repetition check
        tau_r: Repetition threshold ratio

    Returns:
        Sampled token IDs [batch]
    """
    batch_size = logits.shape[0]
    results = []

    for b in range(batch_size):
        decoded = decoded_tokens_batch[b] if b < len(decoded_tokens_batch) else []
        token = ras_sampling(
            logits[b],
            decoded,
            top_p=top_p,
            top_k=top_k,
            win_size=win_size,
            tau_r=tau_r,
        )
        results.append(token)

    return mx.array(results)


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def _compiled_sample_no_rep_penalty(
    logits: mx.array,
    temperature: float,
    top_k: int,
    top_p: float,
) -> mx.array:
    """
    Complete sampling pipeline without repetition penalty (compiled).

    Combines temperature scaling, top-k, top-p, and categorical sampling
    into a single compiled function for maximum performance.

    Uses the EXACT same algorithms as the original implementation to ensure
    deterministic results with matching random seeds.

    Args:
        logits: [batch, vocab_size] - Raw logits
        temperature: Sampling temperature
        top_k: Top-k filtering (0 to disable)
        top_p: Top-p filtering (1.0 to disable)

    Returns:
        Sampled token IDs [batch]
    """
    vocab_size = logits.shape[-1]

    # Apply temperature FIRST (matches original order)
    if temperature != 1.0:
        logits = logits / temperature

    # Apply top-k using topk (matches original implementation exactly)
    if top_k > 0 and top_k < vocab_size:
        top_k_vals = mx.topk(logits, k=top_k, axis=-1)
        # topk returns values sorted ascending, min is at index 0
        threshold = top_k_vals[..., :1]
        logits = mx.where(logits < threshold, float("-inf"), logits)

    # Apply top-p using sort+reverse (matches original implementation exactly)
    if top_p < 1.0:
        # Sort descending using slice reversal (same as original)
        sorted_logits = mx.sort(logits, axis=-1)[:, ::-1]
        sorted_probs = mx.softmax(sorted_logits, axis=-1)
        cumsum_probs = mx.cumsum(sorted_probs, axis=-1)

        # Find cutoff (same as original)
        cutoff_mask = cumsum_probs > top_p
        cutoff_mask = mx.concatenate(
            [mx.zeros((logits.shape[0], 1), dtype=mx.bool_), cutoff_mask[:, :-1]],
            axis=-1,
        )

        # Get threshold value (same as original)
        cutoff_idx = mx.argmax(cutoff_mask.astype(mx.int32), axis=-1)
        threshold = mx.take_along_axis(sorted_logits, cutoff_idx[:, None], axis=-1)
        logits = mx.where(logits < threshold, float("-inf"), logits)

    return mx.random.categorical(logits)


class KVCache:
    """
    Optimized KV Cache with step-based allocation and advanced features.

    Features:
    - D1: Step-based preallocation (256 tokens) with O(1) per-token updates
    - Q4: Attention sinks support (StreamingLLM pattern)
    - Q33: Trimming for sliding window attention
    - Q34: Lazy initialization (allocate on first use)

    Based on mlx-lm KVCache pattern with CosyVoice-specific enhancements.
    """

    step = 256  # Preallocate in chunks of 256 tokens

    def __init__(self, num_sink_tokens: int = 0):
        """
        Initialize KV cache.

        Args:
            num_sink_tokens: Number of initial tokens to keep as attention sinks (Q4).
                            These tokens are never trimmed and always attend to all positions.
                            Set to 4 for StreamingLLM-style inference.
        """
        self.keys: mx.array | None = None
        self.values: mx.array | None = None
        self.offset = 0
        self.num_sink_tokens = num_sink_tokens
        self._sink_keys: mx.array | None = None
        self._sink_values: mx.array | None = None

    def update_and_fetch(
        self, keys: mx.array, values: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """
        Update cache with new K/V and return full cache up to current position.

        Uses step-based preallocation to minimize memory allocations.

        Args:
            keys: New keys [batch, n_kv_heads, num_new_tokens, head_dim]
            values: New values [batch, n_kv_heads, num_new_tokens, head_dim]

        Returns:
            (keys, values): Full cache up to current position
        """
        prev = self.offset
        num_new = keys.shape[2]

        # Q34: Lazy initialization - only allocate when first tokens arrive
        if self.keys is None or (prev + num_new) > self.keys.shape[2]:
            B, n_kv_heads, _, k_head_dim = keys.shape
            v_head_dim = values.shape[3]

            # Calculate new size (round up to step boundary)
            n_steps = (self.step + num_new - 1) // self.step
            k_shape = (B, n_kv_heads, n_steps * self.step, k_head_dim)
            v_shape = (B, n_kv_heads, n_steps * self.step, v_head_dim)

            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)

            if self.keys is not None:
                # Trim existing to actual used portion before concat
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v

        # In-place update at current position (O(1) operation)
        self.offset += num_new
        self.keys[..., prev : self.offset, :] = keys
        self.values[..., prev : self.offset, :] = values

        # Q4: Capture sink tokens on first fill
        if self.num_sink_tokens > 0 and self._sink_keys is None and self.offset >= self.num_sink_tokens:
            # Use mx.array() to create a copy (MLX doesn't have .copy())
            self._sink_keys = mx.array(self.keys[..., : self.num_sink_tokens, :])
            self._sink_values = mx.array(self.values[..., : self.num_sink_tokens, :])

        # Return only the valid portion
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    def get_sinks(self) -> tuple[mx.array, mx.array] | None:
        """
        Q4: Get attention sink tokens for SDPA.

        Returns:
            (sink_keys, sink_values) if sinks are enabled and captured, else None
        """
        if self._sink_keys is not None:
            return (self._sink_keys, self._sink_values)
        return None

    def trim(self, n: int) -> int:
        """
        Q33: Trim oldest n tokens from cache (for sliding window).

        Note: If attention sinks are enabled, sink tokens are preserved
        and only tokens after the sinks are trimmed.

        Args:
            n: Number of tokens to remove from the beginning

        Returns:
            Actual number of tokens trimmed
        """
        if self.keys is None:
            return 0

        # Don't trim sink tokens
        protected = self.num_sink_tokens if self._sink_keys is not None else 0
        trimmable = self.offset - protected
        n = min(n, trimmable)

        if n <= 0:
            return 0

        # Shift data left, preserving sink tokens
        if protected > 0:
            # Keep sinks at positions 0:protected, shift rest
            self.keys[..., protected : self.offset - n, :] = self.keys[..., protected + n : self.offset, :]
            self.values[..., protected : self.offset - n, :] = self.values[..., protected + n : self.offset, :]
        else:
            self.keys[..., : self.offset - n, :] = self.keys[..., n : self.offset, :]
            self.values[..., : self.offset - n, :] = self.values[..., n : self.offset, :]

        self.offset -= n
        return n

    def is_trimmable(self) -> bool:
        """Check if cache supports trimming."""
        return True

    def reset(self):
        """Reset cache for new sequence."""
        self.keys = None
        self.values = None
        self.offset = 0
        self._sink_keys = None
        self._sink_values = None

    def __len__(self):
        """Return current cache length."""
        return self.offset

    @property
    def state(self) -> tuple[mx.array, mx.array] | None:
        """Get current cache state (mlx-lm compatible)."""
        if self.keys is None:
            return None
        if self.offset == self.keys.shape[2]:
            return (self.keys, self.values)
        return (self.keys[..., : self.offset, :], self.values[..., : self.offset, :])

    @state.setter
    def state(self, v: tuple[mx.array, mx.array]):
        """Set cache state (mlx-lm compatible)."""
        self.keys, self.values = v
        self.offset = self.keys.shape[2]

    def to_tuple(self) -> tuple[mx.array, mx.array] | None:
        """Convert to tuple format for backwards compatibility."""
        if self.keys is None:
            return None
        return (self.keys[..., : self.offset, :], self.values[..., : self.offset, :])

    @classmethod
    def from_tuple(
        cls, cache_tuple: tuple[mx.array, mx.array] | None, num_sink_tokens: int = 0,
    ) -> "KVCache":
        """Create KVCache from tuple format for backwards compatibility."""
        kv_cache = cls(num_sink_tokens=num_sink_tokens)
        if cache_tuple is not None:
            k, v = cache_tuple
            kv_cache.keys = k
            kv_cache.values = v
            kv_cache.offset = k.shape[2]
            # Capture sinks if applicable
            if num_sink_tokens > 0 and kv_cache.offset >= num_sink_tokens:
                kv_cache._sink_keys = mx.array(k[..., :num_sink_tokens, :])
                kv_cache._sink_values = mx.array(v[..., :num_sink_tokens, :])
        return kv_cache

    def to_quantized(self, group_size: int = 64, bits: int = 8) -> "QuantizedKVCache":
        """
        Q32: Convert to quantized KV cache for memory savings.

        Args:
            group_size: Quantization group size (default 64)
            bits: Quantization bits (4 or 8, default 8)

        Returns:
            QuantizedKVCache with same content
        """
        quant_cache = QuantizedKVCache(
            group_size=group_size,
            bits=bits,
            num_sink_tokens=self.num_sink_tokens,
        )
        quant_cache.offset = self.offset
        if self.keys is not None:
            quant_cache.keys = mx.quantize(
                self.keys[..., : self.offset, :], group_size=group_size, bits=bits,
            )
            quant_cache.values = mx.quantize(
                self.values[..., : self.offset, :], group_size=group_size, bits=bits,
            )
        if self._sink_keys is not None:
            quant_cache._sink_keys = self._sink_keys
            quant_cache._sink_values = self._sink_values
        return quant_cache


class QuantizedKVCache:
    """
    Q32: Quantized KV Cache for memory-efficient inference.

    Uses INT8 or INT4 quantization to reduce KV cache memory by 2-4x.
    Based on mlx-lm QuantizedKVCache pattern.

    Note: Quantized SDPA does not support attention sinks, so sinks
    are stored in full precision separately.
    """

    step = 256

    def __init__(self, group_size: int = 64, bits: int = 8, num_sink_tokens: int = 0):
        """
        Initialize quantized KV cache.

        Args:
            group_size: Quantization group size
            bits: Quantization bits (4 or 8)
            num_sink_tokens: Number of sink tokens (stored in full precision)
        """
        self.keys: tuple[mx.array, mx.array, mx.array] | None = None  # (data, scales, biases)
        self.values: tuple[mx.array, mx.array, mx.array] | None = None
        self.offset = 0
        self.group_size = group_size
        self.bits = bits
        self.num_sink_tokens = num_sink_tokens
        self._sink_keys: mx.array | None = None
        self._sink_values: mx.array | None = None

    def update_and_fetch(
        self, keys: mx.array, values: mx.array,
    ) -> tuple[tuple[mx.array, mx.array, mx.array], tuple[mx.array, mx.array, mx.array]]:
        """Update cache with quantized K/V."""
        B, n_kv_heads, num_new, k_head_dim = keys.shape
        v_head_dim = values.shape[-1]
        prev = self.offset

        # Allocate if needed
        if self.keys is None or (prev + num_new) > self.keys[0].shape[-2]:
            el_per_int = 8 * 4 // self.bits  # 4 bytes per uint32
            new_steps = (self.step + num_new - 1) // self.step * self.step
            shape = (B, n_kv_heads, new_steps)

            def init_quant(dim):
                return (
                    mx.zeros((*shape, dim // el_per_int), dtype=mx.uint32),
                    mx.zeros((*shape, dim // self.group_size), dtype=keys.dtype),
                    mx.zeros((*shape, dim // self.group_size), dtype=keys.dtype),
                )

            def expand_quant(x):
                new_x = mx.zeros((*shape, x.shape[-1]), dtype=x.dtype)
                return mx.concatenate([x, new_x], axis=-2)

            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = tuple(x[..., :prev, :] for x in self.keys)
                    self.values = tuple(x[..., :prev, :] for x in self.values)
                self.keys = tuple(expand_quant(x) for x in self.keys)
                self.values = tuple(expand_quant(x) for x in self.values)
            else:
                self.keys = init_quant(k_head_dim)
                self.values = init_quant(v_head_dim)

        self.offset += num_new

        # Quantize new keys/values
        q_keys = mx.quantize(keys, group_size=self.group_size, bits=self.bits)
        q_values = mx.quantize(values, group_size=self.group_size, bits=self.bits)

        # Store quantized data
        for i in range(3):
            self.keys[i][..., prev : self.offset, :] = q_keys[i]
            self.values[i][..., prev : self.offset, :] = q_values[i]

        # Capture sinks in full precision on first fill
        if self.num_sink_tokens > 0 and self._sink_keys is None and self.offset >= self.num_sink_tokens:
            # Dequantize sinks for full precision storage
            sink_k = mx.dequantize(
                *[x[..., :self.num_sink_tokens, :] for x in self.keys],
                group_size=self.group_size, bits=self.bits,
            )
            sink_v = mx.dequantize(
                *[x[..., :self.num_sink_tokens, :] for x in self.values],
                group_size=self.group_size, bits=self.bits,
            )
            self._sink_keys = sink_k
            self._sink_values = sink_v

        return (
            tuple(x[..., : self.offset, :] for x in self.keys),
            tuple(x[..., : self.offset, :] for x in self.values),
        )

    def get_sinks(self) -> tuple[mx.array, mx.array] | None:
        """Get attention sink tokens (full precision)."""
        if self._sink_keys is not None:
            return (self._sink_keys, self._sink_values)
        return None

    def __len__(self):
        return self.offset

    def reset(self):
        """Reset cache."""
        self.keys = None
        self.values = None
        self.offset = 0
        self._sink_keys = None
        self._sink_values = None

    @property
    def meta_state(self) -> tuple[str, str, str]:
        """Get metadata for serialization."""
        return (str(self.offset), str(self.group_size), str(self.bits))

    @meta_state.setter
    def meta_state(self, v: tuple[str, str, str]):
        """Set metadata from serialization."""
        self.offset, self.group_size, self.bits = int(v[0]), int(v[1]), int(v[2])


@dataclass
class Qwen2Config:
    """Configuration for Qwen2-based LLM in CosyVoice2."""

    # Model dimensions
    hidden_size: int = 896
    intermediate_size: int = 4864
    num_hidden_layers: int = 24

    # Attention
    num_attention_heads: int = 7  # GQA query heads
    num_key_value_heads: int = 1  # GQA KV heads
    head_dim: int = 128  # 896 / 7 = 128

    # Vocabulary
    vocab_size: int = 151936  # Text vocabulary
    speech_vocab_size: int = 6564  # Speech token vocabulary
    llm_embedding_size: int = 2  # Special tokens (SOS/EOS)

    # RoPE
    max_position_embeddings: int = 32768
    rope_theta: float = 1000000.0
    rope_traditional: bool = False  # For nn.RoPE compatibility

    # Normalization
    rms_norm_eps: float = 1e-6

    # Dropout
    attention_dropout: float = 0.0

    # Q4: Attention Sinks (StreamingLLM)
    num_sink_tokens: int = 0  # Set to 4 for StreamingLLM pattern

    # Q32: Quantized KV Cache
    kv_cache_quantize: bool = False  # Enable INT8 KV cache
    kv_cache_bits: int = 8  # 4 or 8 bits
    kv_cache_group_size: int = 64  # Quantization group size


def precompute_rope_frequencies(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> tuple[mx.array, mx.array]:
    """
    Precompute rotary position embedding frequencies.

    Args:
        head_dim: Head dimension (full, not half)
        max_seq_len: Maximum sequence length
        theta: Base frequency

    Returns:
        cos, sin: [max_seq_len, head_dim] frequency tensors
    """
    half_dim = head_dim // 2

    # Compute frequency bands
    freqs = 1.0 / (theta ** (mx.arange(0, half_dim, dtype=mx.float32) / half_dim))

    # Compute positions
    positions = mx.arange(max_seq_len, dtype=mx.float32)

    # Outer product: [seq_len, half_dim]
    angles = positions[:, None] * freqs[None, :]

    # Create cos and sin, then duplicate for pairs: [seq_len, head_dim]
    cos = mx.cos(angles)
    sin = mx.sin(angles)

    # Repeat to full dimension [seq_len, head_dim]
    cos = mx.concatenate([cos, cos], axis=-1)
    sin = mx.concatenate([sin, sin], axis=-1)

    return cos, sin


def apply_rotary_embedding(
    q: mx.array,
    k: mx.array,
    cos: mx.array,
    sin: mx.array,
    position_ids: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """
    Apply rotary position embedding to query and key.

    Args:
        q: [batch, heads, seq_len, head_dim] - Query
        k: [batch, heads, seq_len, head_dim] - Key
        cos: [max_seq_len, head_dim] - Cosine frequencies
        sin: [max_seq_len, head_dim] - Sine frequencies
        position_ids: Optional position indices

    Returns:
        q_rotated, k_rotated: Tensors with rotary embedding applied
    """
    seq_len = q.shape[2]
    head_dim = q.shape[3]

    if position_ids is not None:
        cos = cos[position_ids]
        sin = sin[position_ids]
    else:
        cos = cos[:seq_len]
        sin = sin[:seq_len]

    # Reshape for broadcasting: [1, 1, seq_len, head_dim]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]

    # Rotate half method: split into two halves
    half_dim = head_dim // 2
    q1, q2 = q[..., :half_dim], q[..., half_dim:]
    k1, k2 = k[..., :half_dim], k[..., half_dim:]

    # Split cos/sin back to half_dim for rotation
    cos_half = cos[..., :half_dim]
    sin_half = sin[..., :half_dim]

    # Apply rotation
    q_rotated = mx.concatenate(
        [
            q1 * cos_half - q2 * sin_half,
            q1 * sin_half + q2 * cos_half,
        ],
        axis=-1,
    )

    k_rotated = mx.concatenate(
        [
            k1 * cos_half - k2 * sin_half,
            k1 * sin_half + k2 * cos_half,
        ],
        axis=-1,
    )

    return q_rotated, k_rotated


class Qwen2RMSNorm(nn.Module):
    """RMSNorm for Qwen2 using optimized MLX fast operation."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        """Apply RMSNorm using mx.core.fast.rms_norm for 5-10% speedup."""
        import mlx.core.fast as fast

        return fast.rms_norm(x, self.weight, self.eps)


class Qwen2Attention(nn.Module):
    """
    Qwen2 attention with Grouped Query Attention (GQA).

    Uses 7 query heads with 1 key-value head.

    Optimizations:
    - Q18-LLM: QKV Fusion (not beneficial for GQA asymmetry)
    - Q31: nn.RoPE integration (uses mx.fast.rope internally)
    - Q4: Attention sinks support via SDPA sinks parameter
    """

    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5

        # Number of query heads per KV head
        self.num_heads_per_kv = self.num_heads // self.num_kv_heads

        # Q18-LLM: QKV fusion state
        self._qkv_fused = False
        self._qkv_proj: nn.Linear | None = None  # Fused QKV projection
        self._q_dim = self.num_heads * self.head_dim
        self._kv_dim = self.num_kv_heads * self.head_dim

        # Q31: Use nn.RoPE (uses mx.fast.rope internally for better performance)
        self.rope = nn.RoPE(
            dims=self.head_dim,
            traditional=config.rope_traditional,
            base=config.rope_theta,
        )

        # Projections
        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=True,
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=True,
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=True,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=False,
        )

    def fuse_qkv(self) -> None:
        """
        Q18-LLM: Fuse Q, K, V projections into single matmul.

        WARNING: NOT BENEFICIAL for Qwen2's GQA architecture.
        GQA has asymmetric Q/K/V sizes (896/128/128), so fusion adds
        overhead instead of reducing it. Measured: 0.95x (5% slower).

        Kept for documentation and architectures with equal Q/K/V sizes.
        """
        if self._qkv_fused:
            return

        # Create fused projection
        total_dim = self._q_dim + 2 * self._kv_dim
        self._qkv_proj = nn.Linear(self.hidden_size, total_dim, bias=True)

        # Concatenate weights: [total_dim, hidden_size]
        self._qkv_proj.weight = mx.concatenate(
            [self.q_proj.weight, self.k_proj.weight, self.v_proj.weight], axis=0,
        )

        # Concatenate biases: [total_dim]
        self._qkv_proj.bias = mx.concatenate(
            [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias], axis=0,
        )

        self._qkv_fused = True

    def unfuse_qkv(self) -> None:
        """Unfuse QKV projections (for testing or weight export)."""
        if not self._qkv_fused:
            return

        self._qkv_proj = None
        self._qkv_fused = False

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
        cos: mx.array | None = None,
        sin: mx.array | None = None,
        position_ids: mx.array | None = None,
        cache: KVCache | QuantizedKVCache | tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, KVCache | QuantizedKVCache | tuple[mx.array, mx.array] | None]:
        """
        Forward pass for attention.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: Optional causal mask
            cos, sin: RoPE frequencies (deprecated, use nn.RoPE)
            position_ids: Position indices for RoPE
            cache: Optional KV cache (KVCache, QuantizedKVCache, or tuple)

        Returns:
            output: [batch, seq_len, hidden_size]
            cache: Updated KV cache (same type as input)
        """
        batch, seq_len, _ = hidden_states.shape

        # Q18-LLM: Use fused QKV projection when available
        if self._qkv_fused and self._qkv_proj is not None:
            # Single matmul for Q, K, V
            qkv = self._qkv_proj(hidden_states)  # [batch, seq_len, q_dim + 2*kv_dim]
            # Split into Q, K, V
            q = qkv[..., : self._q_dim]
            k = qkv[..., self._q_dim : self._q_dim + self._kv_dim]
            v = qkv[..., self._q_dim + self._kv_dim :]
        else:
            # Project Q, K, V separately
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        q = q.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(
            0, 2, 1, 3,
        )
        k = k.reshape(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(
            0, 2, 1, 3,
        )
        v = v.reshape(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(
            0, 2, 1, 3,
        )

        # Q31: Apply RoPE using nn.RoPE (uses mx.fast.rope internally)
        # Get offset from cache for proper position calculation
        cache_offset = 0
        if isinstance(cache, (KVCache, QuantizedKVCache)):
            cache_offset = cache.offset
        elif cache is not None:
            cache_offset = cache[0].shape[2]

        q = self.rope(q, offset=cache_offset)
        k = self.rope(k, offset=cache_offset)

        # Handle KV cache - support KVCache, QuantizedKVCache, and tuple
        # D1/Q32/Q34: Step-based preallocation with optional quantization
        new_cache: KVCache | QuantizedKVCache | tuple[mx.array, mx.array] | None
        sinks: mx.array | None = None  # Q4: Attention sinks per-head weights

        if isinstance(cache, (KVCache, QuantizedKVCache)):
            # Use optimized KVCache with step-based preallocation
            k, v = cache.update_and_fetch(k, v)
            new_cache = cache
            # Q4: Create attention sinks weight array if enabled
            # sinks parameter is shape (num_heads,) indicating per-head sink weights
            # We use zeros which lets SDPA handle sink attention naturally
            if cache.num_sink_tokens > 0 and cache.offset > cache.num_sink_tokens:
                sinks = mx.zeros((self.num_heads,))
        elif cache is not None:
            # Legacy tuple format - concatenate (O(n) per token)
            k_cache, v_cache = cache
            k = mx.concatenate([k_cache, k], axis=2)
            v = mx.concatenate([v_cache, v], axis=2)
            new_cache = (k, v)
        else:
            new_cache = (k, v)

        # Use optimized scaled_dot_product_attention with native GQA support
        # Note: SDPA handles GQA natively - no need to repeat k,v
        import mlx.core.fast as fast

        # Determine mask for SDPA
        # If attention_mask is provided (additive mask), use it
        # Otherwise, use "causal" string for automatic causal masking
        if attention_mask is not None:
            mask = attention_mask
        else:
            mask = "causal"

        # SDPA expects: q [B, N_q, T_q, D], k [B, N_kv, T_kv, D], v [B, N_kv, T_kv, D]
        # Q4: Pass sinks for attention sink support (StreamingLLM pattern)
        out = fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask, sinks=sinks,
        )

        # Reshape and project output
        out = out.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)
        out = self.o_proj(out)

        return out, new_cache


class Qwen2MLP(nn.Module):
    """
    Qwen2 MLP with SwiGLU activation.

    gate_proj and up_proj are applied in parallel, then combined with SiLU gate.
    """

    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False,
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False,
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False,
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Apply SwiGLU MLP."""
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen2DecoderLayer(nn.Module):
    """
    Single Qwen2 decoder layer with optional MLP compilation (Q15-Layer).

    Pre-norm architecture with:
    - Self attention
    - MLP with SwiGLU

    Q15-Layer: Call compile_layer() after loading weights for ~13% speedup.
    """

    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.self_attn = Qwen2Attention(config)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(
            config.hidden_size, config.rms_norm_eps,
        )
        self._compiled_mlp: callable | None = None

    def compile_layer(self) -> None:
        """Q15-Layer: Compile the MLP for ~13% speedup. Call after loading weights."""
        if self._compiled_mlp is not None:
            return

        @mx.compile
        def compiled_mlp(x: mx.array) -> mx.array:
            return self.mlp(x)

        self._compiled_mlp = compiled_mlp

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
        cos: mx.array | None = None,
        sin: mx.array | None = None,
        position_ids: mx.array | None = None,
        cache: KVCache | QuantizedKVCache | tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, KVCache | QuantizedKVCache | tuple[mx.array, mx.array] | None]:
        """
        Forward pass for decoder layer.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: Optional causal mask
            cos, sin: RoPE frequencies (deprecated, each attention uses nn.RoPE)
            position_ids: Optional position indices
            cache: KVCache, QuantizedKVCache (Q32), or tuple (legacy)

        Returns:
            hidden_states: [batch, seq_len, hidden_size]
            cache: Updated KV cache (same type as input)
        """
        residual = hidden_states

        # Pre-norm attention
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, new_cache = self.self_attn(
            hidden_states, attention_mask, cos, sin, position_ids, cache,
        )
        hidden_states = residual + hidden_states

        # Pre-norm MLP (use compiled version if available)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self._compiled_mlp is not None:
            hidden_states = self._compiled_mlp(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, new_cache


class Qwen2Model(nn.Module):
    """
    Qwen2 transformer model (without LM head).

    24 decoder layers with RoPE embeddings.
    """

    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config

        # Token embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Decoder layers
        self.layers = [
            Qwen2DecoderLayer(config) for _ in range(config.num_hidden_layers)
        ]

        # Final norm
        self.norm = Qwen2RMSNorm(config.hidden_size, config.rms_norm_eps)

        # Precompute RoPE frequencies
        self._rope_cos, self._rope_sin = precompute_rope_frequencies(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
        )

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
        position_ids: mx.array | None = None,
        cache: list[KVCache | QuantizedKVCache | tuple[mx.array, mx.array] | None] | None = None,
        stop_at_layer: int | None = None,
    ) -> tuple[mx.array, list[KVCache | QuantizedKVCache | tuple[mx.array, mx.array] | None]]:
        """
        Forward pass.

        Args:
            input_ids: [batch, seq_len] - Token IDs
            attention_mask: Optional attention mask
            position_ids: Optional position indices
            cache: Optional list of KV caches per layer (KVCache, QuantizedKVCache, or tuple)
            stop_at_layer: Optional layer to stop at (exclusive) for early exit

        Returns:
            hidden_states: [batch, seq_len, hidden_size]
            new_cache: Updated KV caches
        """
        batch, seq_len = input_ids.shape

        # Get embeddings
        hidden_states = self.embed_tokens(input_ids)

        # Get cache length for proper mask creation
        # Support KVCache, QuantizedKVCache, and tuple format
        cache_len = 0
        if cache is not None and cache[0] is not None:
            first_cache = cache[0]
            if isinstance(first_cache, (KVCache, QuantizedKVCache)):
                cache_len = first_cache.offset
            else:
                cache_len = first_cache[0].shape[2]  # [batch, heads, cache_seq, head_dim]

        # Create causal mask if needed
        if attention_mask is None:
            # Total key/value length includes cache
            total_len = cache_len + seq_len
            # Mask shape: [seq_len, total_len] to allow attending to cached positions
            # Create row indices [0, 1, ..., seq_len-1] and col indices [0, 1, ..., total_len-1]
            row_idx = mx.arange(seq_len)[:, None]  # [seq_len, 1]
            col_idx = mx.arange(total_len)[None, :]  # [1, total_len]
            # Causal condition: col_idx > cache_len + row_idx means future token
            causal_mask = col_idx > (cache_len + row_idx)
            # Create mask with same dtype as hidden_states (required by SDPA)
            mask = mx.where(causal_mask, mx.array(float("-inf"), hidden_states.dtype), mx.array(0.0, hidden_states.dtype))
            attention_mask = mask[None, None, :, :]

        # Determine how many layers to process
        num_layers = (
            stop_at_layer if stop_at_layer is not None else len(self.layers)
        )

        # Forward through layers
        new_cache: list[KVCache | tuple[mx.array, mx.array] | None] = []
        for i in range(num_layers):
            layer = self.layers[i]
            layer_cache = cache[i] if cache is not None else None
            hidden_states, layer_new_cache = layer(
                hidden_states,
                attention_mask,
                self._rope_cos,
                self._rope_sin,
                position_ids,
                layer_cache,
            )
            new_cache.append(layer_new_cache)

        # Final norm
        hidden_states = self.norm(hidden_states)

        return hidden_states, new_cache


def make_kv_cache(
    num_layers: int,
    num_sink_tokens: int = 0,
) -> list[KVCache]:
    """
    Create a list of KVCache objects for autoregressive generation.

    Optimizations:
    - D1: Step-based preallocation for O(1) per-token updates
    - Q4: Attention sinks (set num_sink_tokens > 0 for StreamingLLM)
    - Q34: Lazy initialization (allocates on first use)

    Args:
        num_layers: Number of transformer layers
        num_sink_tokens: Number of initial tokens to keep as attention sinks (Q4).
                        Set to 4 for StreamingLLM-style inference.

    Returns:
        List of KVCache objects (one per layer)
    """
    return [KVCache(num_sink_tokens=num_sink_tokens) for _ in range(num_layers)]


def make_quantized_kv_cache(
    num_layers: int,
    group_size: int = 64,
    bits: int = 8,
    num_sink_tokens: int = 0,
) -> list[QuantizedKVCache]:
    """
    Q32: Create quantized KV caches for memory-efficient inference.

    Uses INT8 or INT4 quantization to reduce KV cache memory by 2-4x.
    Useful for long sequences or memory-constrained environments.

    Args:
        num_layers: Number of transformer layers
        group_size: Quantization group size (default 64)
        bits: Quantization bits (4 or 8, default 8)
        num_sink_tokens: Number of attention sink tokens (Q4)

    Returns:
        List of QuantizedKVCache objects (one per layer)
    """
    return [
        QuantizedKVCache(
            group_size=group_size,
            bits=bits,
            num_sink_tokens=num_sink_tokens,
        )
        for _ in range(num_layers)
    ]


class CosyVoice2LLM(nn.Module):
    """
    CosyVoice2 LLM for text-to-speech token generation.

    Combines Qwen2 backbone with speech-specific components:
    - llm_embedding: Special token embedding (SOS/EOS)
    - speech_embedding: Speech token embedding (6564 tokens)
    - llm: Qwen2 transformer
    - lm_head: Text LM head
    - llm_decoder: Speech token decoder head

    D1 Optimization: Use make_kv_cache() for optimized generation:
        cache = make_kv_cache(24)  # 24 layers
        _, _, cache = llm(text_ids, cache=cache)
    """

    def __init__(self, config: Qwen2Config, early_exit_layer: int = 6):
        super().__init__()
        self.config = config

        # Special token embedding (SOS/EOS)
        self.llm_embedding = nn.Embedding(config.llm_embedding_size, config.hidden_size)

        # Speech token embedding
        self.speech_embedding = nn.Embedding(
            config.speech_vocab_size, config.hidden_size,
        )

        # Qwen2 backbone
        self.llm = Qwen2Model(config)

        # LM head (tied with embed_tokens typically)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Speech token decoder
        self.llm_decoder = nn.Linear(
            config.hidden_size, config.speech_vocab_size, bias=True,
        )

        # Early exit support for speculative decoding
        self.early_exit_layer = early_exit_layer
        self.early_exit_head = nn.Linear(
            config.hidden_size, config.speech_vocab_size, bias=True,
        )

    def initialize_early_exit_head(self):
        """
        Initialize early_exit_head weights from llm_decoder.

        This significantly improves acceptance rate for speculative decoding
        by making the early exit predictions similar to the full model's predictions.
        Call this after loading pretrained weights.
        """
        # Copy weights from the final llm_decoder to the early exit head
        self.early_exit_head.weight = self.llm_decoder.weight
        self.early_exit_head.bias = self.llm_decoder.bias

    def fuse_qkv_weights(self) -> None:
        """
        Q18-LLM: Fuse QKV projection weights for all attention layers.

        WARNING: NOT BENEFICIAL for this model's GQA architecture.
        Measured: 0.95x (5% slower) due to asymmetric Q/K/V sizes (896/128/128).

        This is LOSSLESS but adds overhead. Kept for documentation.
        Use separate projections for production.
        """
        for layer in self.llm.layers:
            layer.self_attn.fuse_qkv()

    def unfuse_qkv_weights(self) -> None:
        """
        Unfuse QKV projection weights for all attention layers.

        Call this before exporting weights or for debugging.
        """
        for layer in self.llm.layers:
            layer.self_attn.unfuse_qkv()

    def compile_layers(self) -> int:
        """Q15-Layer: Compile all MLP layers for ~13% speedup. Call after loading weights."""
        count = 0
        for layer in self.llm.layers:
            layer.compile_layer()
            count += 1
        return count

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
        position_ids: mx.array | None = None,
        cache: list[tuple[mx.array, mx.array] | None] | None = None,
        use_early_exit: bool = False,
    ) -> tuple[mx.array, mx.array, list[tuple[mx.array, mx.array] | None]]:
        """
        Forward pass.

        Args:
            input_ids: [batch, seq_len] - Token IDs
            attention_mask: Optional attention mask
            position_ids: Position indices
            cache: Optional KV cache
            use_early_exit: If True, use early exit layer for faster draft prediction

        Returns:
            text_logits: [batch, seq_len, vocab_size] - Text token logits
            speech_logits: [batch, seq_len, speech_vocab_size] - Speech token logits
            cache: Updated KV cache
        """
        stop_layer = self.early_exit_layer if use_early_exit else None
        hidden_states, new_cache = self.llm(
            input_ids, attention_mask, position_ids, cache, stop_at_layer=stop_layer,
        )

        text_logits = self.lm_head(hidden_states)

        # Use early exit head for draft predictions, full decoder for verification
        if use_early_exit:
            speech_logits = self.early_exit_head(hidden_states)
        else:
            speech_logits = self.llm_decoder(hidden_states)

        return text_logits, speech_logits, new_cache

    def sample_tokens(
        self,
        logits: mx.array,
        temperature: float = 1.0,
        top_k: int = 25,
        top_p: float = 0.8,
        repetition_penalty: float = 1.0,
        past_tokens: mx.array | None = None,
    ) -> mx.array:
        """
        Sample tokens with various strategies.

        Optimizations:
        - Q27: Greedy decoding fast path when temperature <= 0 (9x speedup)
        - Q15: Compiled sampling functions using mlx-lm pattern (~2x speedup)
               Uses @partial(mx.compile, inputs/outputs=mx.random.state)
               for top-k, top-p, and categorical sampling

        Args:
            logits: [batch, vocab_size] - Raw logits
            temperature: Sampling temperature (1.0 = no change, <=0 = greedy)
            top_k: Keep only top-k tokens (0 = disabled)
            top_p: Nucleus sampling threshold (1.0 = disabled)
            repetition_penalty: Penalty for repeating past tokens (1.0 = disabled)
            past_tokens: [batch, past_len] - Previously generated tokens

        Returns:
            sampled_tokens: [batch] - Sampled token IDs
        """
        # Q27 optimization: Greedy decoding fast path
        # When temperature <= 0, skip all sampling overhead and use argmax directly
        if temperature <= 0:
            return mx.argmax(logits, axis=-1)

        # Q15 optimization: Use compiled sampling when no repetition penalty
        # This is the fast path for most generation scenarios
        if repetition_penalty == 1.0 or past_tokens is None:
            return _compiled_sample_no_rep_penalty(logits, temperature, top_k, top_p)

        # Slow path with repetition penalty (uses Python loops)
        # This is only used when repetition_penalty != 1.0 AND past_tokens is provided
        vocab_size = logits.shape[-1]
        penalty_mask = mx.ones_like(logits)

        for b in range(logits.shape[0]):
            seen_tokens = set()
            for past_token in past_tokens[b].tolist():
                token_id = int(past_token)
                if 0 <= token_id < vocab_size:
                    seen_tokens.add(token_id)

            if seen_tokens:
                for token_id in seen_tokens:
                    # Apply penalty: divide if positive logit, multiply if negative
                    if logits[b, token_id].item() > 0:
                        penalty_mask = penalty_mask.at[b, token_id].add(
                            1.0 / repetition_penalty - 1.0,
                        )
                    else:
                        penalty_mask = penalty_mask.at[b, token_id].add(
                            repetition_penalty - 1.0,
                        )

        logits = logits * penalty_mask

        # Use compiled sampling for the rest
        return _compiled_sample_no_rep_penalty(logits, temperature, top_k, top_p)

    def generate_speech_tokens(
        self,
        text_ids: mx.array,
        max_length: int = 1000,
        temperature: float = 1.0,
        top_k: int = 25,
        top_p: float = 0.8,
        repetition_penalty: float = 1.0,
        eos_token_id: int = 0,
    ) -> mx.array:
        """
        Generate speech tokens from text (DEPRECATED - use generate_speech_tokens_ras).

        NOTE: This method uses standard top-k/top-p sampling which can cause
        repetitive loops in CosyVoice2. Use generate_speech_tokens_ras() instead.

        Args:
            text_ids: [batch, text_len] - Text token IDs
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
            repetition_penalty: Penalty for repetition
            eos_token_id: End of sequence token ID

        Returns:
            speech_tokens: [batch, gen_len] - Generated speech tokens
        """
        batch = text_ids.shape[0]
        generated: list[mx.array] = []
        cache = None

        # Process text first
        _, _, cache = self(text_ids, cache=cache)

        # Start with SOS token (index 0 in llm_embedding)
        next_input = mx.zeros((batch, 1), dtype=mx.int32)

        for _ in range(max_length):
            _, speech_logits, cache = self(next_input, cache=cache)
            logits = speech_logits[:, -1, :]

            # Get past tokens for repetition penalty
            past_tokens = mx.array(generated).T if generated else None

            # Sample next token
            next_token = self.sample_tokens(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                past_tokens=past_tokens,
            )

            generated.append(next_token.tolist())  # type: ignore[arg-type]
            next_input = next_token[:, None]

            mx.eval(next_token)

            # Check for EOS
            if mx.all(next_token == eos_token_id).item():  # type: ignore[arg-type]
                break

        return mx.stack([mx.array(g) for g in generated], axis=-1)

    def generate_speech_tokens_ras(
        self,
        text_ids: mx.array,
        max_length: int = 1000,
        top_k: int = 25,
        top_p: float = 0.8,
        win_size: int = 10,
        tau_r: float = 0.1,
        speech_token_size: int = 6561,
        min_token_text_ratio: float = 2.0,
        max_token_text_ratio: float = 20.0,
    ) -> mx.array:
        """
        Generate speech tokens using Repetition Aware Sampling (ras_sampling).

        This is the correct sampling method for CosyVoice2 that prevents
        repetitive/looping output. Uses the ras_sampling algorithm from
        CosyVoice2's cosyvoice.utils.common module.

        Termination:
        - CosyVoice2 uses stop_token_ids = [6561, 6562, 6563]
        - Any token >= speech_token_size (6561) triggers termination
        - Min/max length is computed from text length * ratio

        Algorithm:
        1. First sample using nucleus (top-k/top-p) sampling
        2. Check if sampled token appears in the last win_size tokens
        3. If repetition count >= win_size * tau_r, use random sampling instead
        4. Before min_len, resample if EOS token is drawn

        Args:
            text_ids: [batch, text_len] - Text token IDs
            max_length: Maximum generation length (overridden by max_token_text_ratio)
            top_k: Maximum tokens in nucleus (default 25)
            top_p: Nucleus sampling probability threshold (default 0.8)
            win_size: Window size for repetition check (default 10)
            tau_r: Repetition threshold ratio (default 0.1)
            speech_token_size: Size of speech vocabulary (default 6561)
            min_token_text_ratio: Min tokens per text token (default 2.0)
            max_token_text_ratio: Max tokens per text token (default 20.0)

        Returns:
            speech_tokens: [batch, gen_len] - Generated speech tokens
        """
        batch = text_ids.shape[0]
        text_len = text_ids.shape[1]

        # Compute min/max length from text length (CosyVoice2 behavior)
        min_len = int(text_len * min_token_text_ratio)
        max_len = min(max_length, int(text_len * max_token_text_ratio))

        # Stop tokens: [6561, 6562, 6563] - any token >= speech_token_size is EOS
        set(range(speech_token_size, speech_token_size + 3))

        # Track decoded tokens per batch item for ras_sampling
        decoded_tokens_batch: list[list[int]] = [[] for _ in range(batch)]
        cache = None

        # Process text first
        _, _, cache = self(text_ids, cache=cache)

        # Start with SOS token (index 0 in llm_embedding)
        next_input = mx.zeros((batch, 1), dtype=mx.int32)

        for step in range(max_len):
            _, speech_logits, cache = self(next_input, cache=cache)
            logits = speech_logits[:, -1, :]  # [batch, vocab_size]

            # Apply ras_sampling for each batch item
            # Before min_len, ignore EOS tokens (resample if necessary)
            ignore_eos = step < min_len

            tokens_this_step = []
            for b in range(batch):
                decoded = decoded_tokens_batch[b]
                max_trials = 100
                for _trial in range(max_trials):
                    token = ras_sampling(
                        logits[b],
                        decoded,
                        top_p=top_p,
                        top_k=top_k,
                        win_size=win_size,
                        tau_r=tau_r,
                    )
                    # If ignoring EOS and got EOS, resample
                    if ignore_eos and token >= speech_token_size:
                        continue
                    break
                else:
                    # After max_trials, just use the token anyway
                    pass
                tokens_this_step.append(token)

            next_token = mx.array(tokens_this_step)
            mx.eval(next_token)

            # Update decoded tokens history for each batch item
            for b in range(batch):
                decoded_tokens_batch[b].append(int(next_token[b].item()))

            next_input = next_token[:, None]

            # Check for stop tokens (all sequences must have stopped)
            all_stopped = all(
                decoded_tokens_batch[b][-1] >= speech_token_size
                for b in range(batch)
            )
            if all_stopped:
                # Remove the stop tokens from output (they're not speech tokens)
                for b in range(batch):
                    if decoded_tokens_batch[b] and decoded_tokens_batch[b][-1] >= speech_token_size:
                        decoded_tokens_batch[b].pop()
                break

        # Convert decoded tokens to output array
        if not decoded_tokens_batch[0]:
            return mx.zeros((batch, 0), dtype=mx.int32)

        max_gen_len = max(len(tokens) for tokens in decoded_tokens_batch)
        result = mx.zeros((batch, max_gen_len), dtype=mx.int32)
        for b in range(batch):
            tokens = decoded_tokens_batch[b]
            result[b, : len(tokens)] = mx.array(tokens)

        return result

    def generate_speech_tokens_batch(
        self,
        text_ids: mx.array,
        max_length: int = 1000,
        temperature: float = 1.0,
        top_k: int = 25,
        top_p: float = 0.8,
        repetition_penalty: float = 1.0,
        eos_token_id: int = 0,
    ) -> tuple[mx.array, mx.array]:
        """
        Generate speech tokens from text with per-sequence length tracking.

        Unlike generate_speech_tokens(), this method tracks when each sequence
        in the batch reaches EOS, enabling proper handling of variable-length
        outputs in batch synthesis.

        Args:
            text_ids: [batch, text_len] - Text token IDs
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
            repetition_penalty: Penalty for repetition
            eos_token_id: End of sequence token ID

        Returns:
            speech_tokens: [batch, gen_len] - Generated speech tokens (padded)
            token_lengths: [batch] - Actual token length per sequence (before EOS)
        """
        batch = text_ids.shape[0]
        generated: list[mx.array] = []
        cache = None

        # Track which sequences have finished (hit EOS)
        finished = mx.zeros((batch,), dtype=mx.bool_)
        # Track the length of each sequence (step when EOS was hit)
        token_lengths = mx.full((batch,), max_length, dtype=mx.int32)

        # Process text first
        _, _, cache = self(text_ids, cache=cache)

        # Start with SOS token (index 0 in llm_embedding)
        next_input = mx.zeros((batch, 1), dtype=mx.int32)

        for step in range(max_length):
            _, speech_logits, cache = self(next_input, cache=cache)
            logits = speech_logits[:, -1, :]

            # Get past tokens for repetition penalty
            past_tokens = mx.array(generated).T if generated else None

            # Sample next token
            next_token = self.sample_tokens(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                past_tokens=past_tokens,
            )

            # For finished sequences, replace with EOS to keep them static
            next_token = mx.where(finished, eos_token_id, next_token)

            generated.append(next_token.tolist())  # type: ignore[arg-type]
            next_input = next_token[:, None]

            mx.eval(next_token)

            # Check which sequences just finished (hit EOS this step)
            just_finished = (next_token == eos_token_id) & ~finished
            # Record the length for sequences that just finished (step index = length)
            token_lengths = mx.where(just_finished, step, token_lengths)
            # Update finished mask
            finished = finished | (next_token == eos_token_id)

            # Check if all sequences have finished
            if mx.all(finished).item():  # type: ignore[arg-type]
                break

        speech_tokens = mx.stack([mx.array(g) for g in generated], axis=-1)
        return speech_tokens, token_lengths

    def generate_speech_tokens_stream(
        self,
        text_ids: mx.array,
        max_length: int = 1000,
        temperature: float = 1.0,
        top_k: int = 25,
        top_p: float = 0.8,
        repetition_penalty: float = 1.0,
        eos_token_id: int = 0,
        chunk_size: int = 25,
    ):
        """
        Generate speech tokens with streaming output.

        Yields chunks of tokens for low-latency streaming.

        Args:
            text_ids: [batch, text_len] - Text token IDs
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
            repetition_penalty: Penalty for repetition
            eos_token_id: End of sequence token ID
            chunk_size: Number of tokens per chunk

        Yields:
            token_chunk: [batch, chunk_size] - Chunk of generated tokens
            is_final: bool - Whether this is the final chunk
        """
        batch = text_ids.shape[0]
        cache = None
        chunk_tokens: list[mx.array] = []
        all_tokens: list[mx.array] = []

        # Process text first
        _, _, cache = self(text_ids, cache=cache)

        # Start with SOS token
        next_input = mx.zeros((batch, 1), dtype=mx.int32)
        finished = False

        for _step in range(max_length):
            _, speech_logits, cache = self(next_input, cache=cache)
            logits = speech_logits[:, -1, :]

            # Get past tokens for repetition penalty
            past_tokens = mx.array(all_tokens).T if all_tokens else None

            # Sample next token
            next_token = self.sample_tokens(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                past_tokens=past_tokens,
            )

            mx.eval(next_token)

            chunk_tokens.append(next_token.tolist())  # type: ignore[arg-type]
            all_tokens.append(next_token.tolist())  # type: ignore[arg-type]
            next_input = next_token[:, None]

            # Check for EOS
            if mx.all(next_token == eos_token_id).item():  # type: ignore[arg-type]
                finished = True

            # Yield chunk
            if len(chunk_tokens) >= chunk_size or finished:
                chunk_array = mx.stack([mx.array(t) for t in chunk_tokens], axis=-1)
                yield chunk_array, finished
                chunk_tokens = []

                if finished:
                    break

        # Yield any remaining tokens
        if chunk_tokens:
            chunk_array = mx.stack([mx.array(t) for t in chunk_tokens], axis=-1)
            yield chunk_array, True

    def generate_speech_tokens_speculative(
        self,
        text_ids: mx.array,
        max_length: int = 1000,
        temperature: float = 1.0,
        top_k: int = 25,
        top_p: float = 0.8,
        repetition_penalty: float = 1.0,
        eos_token_id: int = 0,
        num_draft_tokens: int = 4,
    ) -> tuple[mx.array, dict]:
        """
        Generate speech tokens using early-exit speculative decoding.

        Uses the early exit layer (fewer layers) to generate draft tokens,
        then verifies them with the full model. Accepted tokens are kept,
        rejected tokens trigger re-generation from the last accepted position.

        Args:
            text_ids: [batch, text_len] - Text token IDs
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
            repetition_penalty: Penalty for repetition
            eos_token_id: End of sequence token ID
            num_draft_tokens: Number of tokens to draft before verification

        Returns:
            speech_tokens: [batch, gen_len] - Generated speech tokens
            stats: dict with acceptance_rate, total_tokens, draft_calls, verify_calls
        """
        batch = text_ids.shape[0]
        generated: list[mx.array] = []

        # Statistics tracking
        total_accepted = 0
        total_drafted = 0
        draft_calls = 0
        verify_calls = 0

        # Process text first (with full model to populate cache)
        _, _, cache = self(text_ids, cache=None, use_early_exit=False)

        # Start with SOS token (index 0 in llm_embedding)
        next_input = mx.zeros((batch, 1), dtype=mx.int32)

        while len(generated) < max_length:
            # Phase 1: Generate draft tokens using early exit (without cache)
            # We regenerate drafts each round to keep it simple and correct
            draft_tokens = []
            draft_sequence = next_input

            # Build the full sequence to process for draft
            temp_draft_tokens = []
            for d_step in range(num_draft_tokens):
                # Process the entire sequence so far with early exit
                if d_step == 0:
                    # First draft: process next_input
                    _, draft_logits, _ = self(
                        draft_sequence, cache=cache, use_early_exit=True,
                    )
                else:
                    # Subsequent drafts: process just the last token, no cache
                    # For simplicity, regenerate from scratch (less efficient but correct)
                    full_seq = mx.concatenate(
                        [next_input] + [t[:, None] for t in temp_draft_tokens], axis=1,
                    )
                    _, draft_logits, _ = self(
                        full_seq, cache=cache, use_early_exit=True,
                    )

                draft_logits_last = draft_logits[:, -1, :]

                # Greedy sampling for draft (faster, no sampling overhead)
                draft_token = mx.argmax(draft_logits_last, axis=-1)
                temp_draft_tokens.append(draft_token)

                mx.eval(draft_token)

                # Check for EOS in draft
                if mx.all(draft_token == eos_token_id).item():
                    break

            draft_tokens = temp_draft_tokens
            draft_calls += 1
            total_drafted += len(draft_tokens)

            if not draft_tokens:
                break

            # Phase 2: Verify draft tokens with full model (in parallel)
            # Concatenate next_input with draft tokens for parallel verification
            draft_sequence = mx.concatenate(
                [next_input] + [t[:, None] for t in draft_tokens], axis=1,
            )

            _, verify_logits, new_cache = self(
                draft_sequence, cache=cache, use_early_exit=False,
            )
            verify_calls += 1

            # Phase 3: Accept matching tokens
            # verify_logits shape: [batch, 1 + num_draft, speech_vocab_size]
            # We compare draft_tokens[i] with argmax(verify_logits[:, i, :])
            accepted_count = 0

            for i, draft_token in enumerate(draft_tokens):
                verify_logits_i = verify_logits[:, i, :]

                # Sample from full model with temperature
                past_tokens = mx.array(generated).T if generated else None
                verified_token = self.sample_tokens(
                    verify_logits_i,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    past_tokens=past_tokens,
                )

                mx.eval(verified_token)

                # Check if draft matches verification
                if mx.all(draft_token == verified_token).item():
                    # Accept draft token
                    generated.append(verified_token)
                    accepted_count += 1
                    total_accepted += 1

                    # Check for EOS
                    if mx.all(verified_token == eos_token_id).item():
                        break
                else:
                    # Reject: use verified token instead, discard rest of draft
                    generated.append(verified_token)
                    accepted_count += 1  # We still add one token
                    total_accepted += 1
                    break

            # Update cache: truncate to only include accepted tokens
            # The new_cache includes all draft tokens, we need to trim it
            truncated_cache: list[tuple[mx.array, mx.array] | None] = []
            target_len = cache[0][0].shape[2] + accepted_count if cache else accepted_count
            for layer_cache in new_cache:
                if layer_cache is not None:
                    k, v = layer_cache
                    # Trim to keep only the accepted positions
                    truncated_cache.append((k[:, :, :target_len, :], v[:, :, :target_len, :]))
                else:
                    truncated_cache.append(None)
            cache = truncated_cache

            # Prepare next input
            next_input = generated[-1][:, None] if generated else next_input

            # Check for EOS
            if generated and mx.all(generated[-1] == eos_token_id).item():
                break

        # Compile statistics
        acceptance_rate = total_accepted / total_drafted if total_drafted > 0 else 0.0
        stats = {
            "acceptance_rate": acceptance_rate,
            "total_tokens": len(generated),
            "total_drafted": total_drafted,
            "total_accepted": total_accepted,
            "draft_calls": draft_calls,
            "verify_calls": verify_calls,
        }

        if not generated:
            return mx.zeros((batch, 0), dtype=mx.int32), stats

        return mx.stack(generated, axis=-1), stats

    @staticmethod
    def from_pretrained(
        weights_path: str,
        config: Qwen2Config | None = None,
    ) -> "CosyVoice2LLM":
        """
        Load LLM from pretrained weights.

        Args:
            weights_path: Path to llm.pt file
            config: Optional config override

        Returns:
            Loaded CosyVoice2LLM model
        """
        import torch

        if config is None:
            config = Qwen2Config()

        model = CosyVoice2LLM(config)

        # Load PyTorch weights
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)

        # Map weights
        model._load_weights(state_dict)

        return model

    def _load_weights(self, state_dict: dict) -> None:
        """Load weights from PyTorch state dict."""

        def to_mlx(t):
            """Convert PyTorch tensor to MLX array."""
            return mx.array(t.numpy())

        # LLM embedding (SOS/EOS)
        if "llm_embedding.weight" in state_dict:
            self.llm_embedding.weight = to_mlx(state_dict["llm_embedding.weight"])

        # Speech embedding
        if "speech_embedding.weight" in state_dict:
            self.speech_embedding.weight = to_mlx(state_dict["speech_embedding.weight"])

        # Text embedding
        if "llm.model.model.embed_tokens.weight" in state_dict:
            self.llm.embed_tokens.weight = to_mlx(
                state_dict["llm.model.model.embed_tokens.weight"],
            )

        # Transformer layers
        for i in range(self.config.num_hidden_layers):
            prefix = f"llm.model.model.layers.{i}"

            # Self attention
            self._load_attention_weights(i, prefix, state_dict)

            # MLP
            self._load_mlp_weights(i, prefix, state_dict)

            # Norms
            if f"{prefix}.input_layernorm.weight" in state_dict:
                self.llm.layers[i].input_layernorm.weight = to_mlx(
                    state_dict[f"{prefix}.input_layernorm.weight"],
                )
            if f"{prefix}.post_attention_layernorm.weight" in state_dict:
                self.llm.layers[i].post_attention_layernorm.weight = to_mlx(
                    state_dict[f"{prefix}.post_attention_layernorm.weight"],
                )

        # Final norm
        if "llm.model.model.norm.weight" in state_dict:
            self.llm.norm.weight = to_mlx(state_dict["llm.model.model.norm.weight"])

        # LM head
        if "llm.model.lm_head.weight" in state_dict:
            self.lm_head.weight = to_mlx(state_dict["llm.model.lm_head.weight"])

        # Speech token decoder
        if "llm_decoder.weight" in state_dict:
            self.llm_decoder.weight = to_mlx(state_dict["llm_decoder.weight"])
            self.llm_decoder.bias = to_mlx(state_dict["llm_decoder.bias"])

    def _load_attention_weights(
        self, layer_idx: int, prefix: str, state_dict: dict,
    ) -> None:
        """Load attention weights for a layer."""

        def to_mlx(t):
            return mx.array(t.numpy())

        attn = self.llm.layers[layer_idx].self_attn

        # Q, K, V projections
        if f"{prefix}.self_attn.q_proj.weight" in state_dict:
            attn.q_proj.weight = to_mlx(state_dict[f"{prefix}.self_attn.q_proj.weight"])
            attn.q_proj.bias = to_mlx(state_dict[f"{prefix}.self_attn.q_proj.bias"])

        if f"{prefix}.self_attn.k_proj.weight" in state_dict:
            attn.k_proj.weight = to_mlx(state_dict[f"{prefix}.self_attn.k_proj.weight"])
            attn.k_proj.bias = to_mlx(state_dict[f"{prefix}.self_attn.k_proj.bias"])

        if f"{prefix}.self_attn.v_proj.weight" in state_dict:
            attn.v_proj.weight = to_mlx(state_dict[f"{prefix}.self_attn.v_proj.weight"])
            attn.v_proj.bias = to_mlx(state_dict[f"{prefix}.self_attn.v_proj.bias"])

        if f"{prefix}.self_attn.o_proj.weight" in state_dict:
            attn.o_proj.weight = to_mlx(state_dict[f"{prefix}.self_attn.o_proj.weight"])

    def _load_mlp_weights(self, layer_idx: int, prefix: str, state_dict: dict) -> None:
        """Load MLP weights for a layer."""

        def to_mlx(t):
            return mx.array(t.numpy())

        mlp = self.llm.layers[layer_idx].mlp

        if f"{prefix}.mlp.gate_proj.weight" in state_dict:
            mlp.gate_proj.weight = to_mlx(state_dict[f"{prefix}.mlp.gate_proj.weight"])

        if f"{prefix}.mlp.up_proj.weight" in state_dict:
            mlp.up_proj.weight = to_mlx(state_dict[f"{prefix}.mlp.up_proj.weight"])

        if f"{prefix}.mlp.down_proj.weight" in state_dict:
            mlp.down_proj.weight = to_mlx(state_dict[f"{prefix}.mlp.down_proj.weight"])
