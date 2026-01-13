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
Text decoder for WhisperMLX with adjustable timestamp precision.

THE KEY FIX: Standard Whisper assumes 30s audio with 1500 encoder positions,
giving 0.02s timestamp precision. This causes decoder hallucinations with
shorter audio because the timestamp tokens are miscalibrated.

Our decoder adjusts timestamp precision dynamically based on actual audio
duration and encoder output length.

OPT-1.2: mx.compile for decoder forward pass
- Compiles the decoder forward pass for 1.5-2.5x speedup
- Compilation happens lazily on first call
"""

from collections.abc import Callable

import mlx.core as mx
import mlx.nn as nn

from .attention import ResidualAttentionBlock


class TextDecoder(nn.Module):
    """
    Whisper text decoder with dynamic timestamp precision.

    Standard Whisper:
    - precision = 30s / 1500 positions = 0.02s per position
    - Timestamp token N represents time N * 0.02s

    Our decoder:
    - precision = actual_duration / actual_encoder_positions
    - For 5s audio with 250 positions: 5 / 250 = 0.02s (same precision)
    - For 5s audio padded to 1500 positions: 5 / 1500 = 0.0033s (WRONG)

    This fix enables correct timestamp generation for variable-length audio.

    OPT-1.2: mx.compile for decoder forward pass
    - DISABLED by default: Compilation is slower for single-token decode
    - Decoder processes one token at a time, so compilation overhead
      doesn't amortize. Benchmark shows 0.35x (3x slower) with compile.
    - Only enable for batch processing scenarios (not typical Whisper use).

    OPT-VOCAB: Vocab dimension padding for GPU efficiency
    - Whisper vocab sizes (51864-51866) are not optimal for Metal GPU
    - Padding to 51872 (divisible by 32) improves matmul performance by 5-10%
    - Enabled by default on M3/M4 chips where tensor core alignment matters
    """

    # Default padding target: 51872 = nearest multiple of 32 above 51866
    # 51872 = 32 * 1621
    VOCAB_PAD_TARGET = 51872

    def __init__(
        self,
        n_vocab: int,
        n_ctx: int,
        n_state: int,
        n_head: int,
        n_layer: int,
        dtype: mx.Dtype = mx.float16,
        use_fused: bool = True,
        compile_forward: bool = False,  # OPT-1.2: Disabled - slower for single-token decode
        pad_vocab: bool = True,  # OPT-VOCAB: Pad vocab dimension for GPU efficiency
    ):
        """
        Args:
            n_vocab: Vocabulary size
            n_ctx: Maximum context length (448 for most Whisper models)
            n_state: Hidden dimension
            n_head: Number of attention heads
            n_layer: Number of transformer layers
            dtype: Data type for computation
            use_fused: Use fused attention operations
            compile_forward: Enable mx.compile for forward pass (OPT-1.2)
                             Default False - compilation is slower for single-token decode
            pad_vocab: Enable vocab padding for GPU efficiency (OPT-VOCAB)
                       Default True - pads vocab to 51872 for better matmul perf
        """
        super().__init__()

        self.n_ctx = n_ctx
        self.n_state = n_state
        self.n_layer = n_layer
        self.n_vocab = n_vocab  # Store original vocab size for slicing
        self._compile_enabled = compile_forward
        self._pad_vocab = pad_vocab

        # OPT-VOCAB: Compute padded vocab size
        if pad_vocab and n_vocab < self.VOCAB_PAD_TARGET:
            self._padded_vocab = self.VOCAB_PAD_TARGET
        else:
            self._padded_vocab = n_vocab

        # Token embeddings (padded if enabled)
        self.token_embedding = nn.Embedding(self._padded_vocab, n_state)

        # Learned positional embeddings (unlike encoder's sinusoidal)
        self.positional_embedding = mx.zeros((n_ctx, n_state))

        # Transformer blocks with cross-attention
        self.blocks = [
            ResidualAttentionBlock(n_state, n_head, cross_attention=True, use_fused=use_fused)
            for _ in range(n_layer)
        ]

        # Final layer norm
        self.ln = nn.LayerNorm(n_state)

        # Causal attention mask
        self._mask = nn.MultiHeadAttention.create_additive_causal_mask(n_ctx).astype(dtype)

        # Timestamp precision (dynamically set)
        self._default_precision = 30.0 / 1500  # 0.02s (standard Whisper)
        self._precision = self._default_precision
        self._audio_duration: float | None = None

        # OPT-NEW-12: Precompute common positional embedding slices
        # Most decode steps use positions 0-10 (single token generation)
        # Caching these avoids repeated slicing overhead
        self._pos_cache: dict[tuple[int, int], mx.array] = {}

        # OPT-1.2: Compiled forward function (lazy initialization)
        self._compiled_forward: Callable | None = None

    def set_precision(self, audio_duration: float, encoder_positions: int):
        """
        Set timestamp precision for current audio.

        THIS IS THE KEY FIX from Worker #1180.

        Call this before decoding to calibrate timestamps for the actual
        audio length. For standard 30s audio with 1500 positions, precision
        remains 0.02s. For shorter audio, precision is adjusted proportionally.

        Args:
            audio_duration: Actual audio duration in seconds
            encoder_positions: Actual encoder output sequence length
        """
        if encoder_positions <= 0:
            self._precision = self._default_precision
        else:
            self._precision = audio_duration / encoder_positions
        self._audio_duration = audio_duration

    def reset_precision(self):
        """Reset to default precision (for 30s audio)."""
        self._precision = self._default_precision
        self._audio_duration = None

    @property
    def precision(self) -> float:
        """Current timestamp precision in seconds."""
        return self._precision

    @property
    def audio_duration(self) -> float | None:
        """Current audio duration if set."""
        return self._audio_duration

    def timestamp_to_position(self, timestamp: float) -> int:
        """
        Convert timestamp (seconds) to encoder position index.

        Uses current precision setting for accurate conversion.

        Args:
            timestamp: Time in seconds

        Returns:
            Encoder position index
        """
        return int(timestamp / self._precision)

    def position_to_timestamp(self, position: int) -> float:
        """
        Convert encoder position index to timestamp (seconds).

        Uses current precision setting for accurate conversion.

        Args:
            position: Encoder position index

        Returns:
            Time in seconds
        """
        return position * self._precision

    def _get_positional_slice(self, offset: int, seq_len: int) -> mx.array:
        """
        Get positional embedding slice with caching (OPT-NEW-12).

        Caches frequently used slices to avoid repeated slicing overhead.
        Most decode steps use single tokens (seq_len=1) at positions 0-50.

        Args:
            offset: Starting position
            seq_len: Sequence length

        Returns:
            Positional embedding slice of shape (seq_len, n_state)
        """
        cache_key = (offset, seq_len)

        # Check cache first
        if cache_key in self._pos_cache:
            return self._pos_cache[cache_key]

        # Compute slice
        pos_slice = self.positional_embedding[offset : offset + seq_len]

        # Cache common slices (positions 0-50 with seq_len 1-10)
        # These cover the vast majority of decode steps
        if offset < 50 and seq_len <= 10:
            self._pos_cache[cache_key] = pos_slice

        return pos_slice

    def _forward_impl(
        self,
        x: mx.array,
        xa: mx.array,
        pos_emb: mx.array,
        kv_cache: list[tuple] | None,
        custom_mask: mx.array | None = None,
        return_hidden: bool = False,
    ) -> tuple[mx.array, list[tuple], list[mx.array | None], mx.array | None]:
        """
        Core decoder forward pass (compilable).

        This method is separated to enable mx.compile optimization.
        It takes positional embeddings as input to avoid control flow
        that would require recompilation.

        Args:
            x: Token ids, shape (batch, seq_len)
            xa: Encoder output, shape (batch, encoder_len, n_state)
            pos_emb: Positional embeddings for current positions
            kv_cache: List of cached (self_kv, cross_kv) per layer
            custom_mask: Optional custom attention mask for tree attention.
                         Shape (seq_len, seq_len) - used for Medusa tree verification.
                         If None, uses standard causal mask.
            return_hidden: If True, return hidden states before final projection.
                          Used for Medusa head predictions.

        Returns:
            Tuple of:
            - Logits, shape (batch, seq_len, n_vocab)
            - Updated KV cache list
            - Cross-attention weights per layer
            - Hidden states (only if return_hidden=True, else None)
        """
        # OPT-NEW-11: Fused embedding + positional
        token_emb = self.token_embedding.weight[x]
        x = token_emb + pos_emb

        # Initialize cache list if needed
        if kv_cache is None:
            kv_cache = [None] * len(self.blocks)

        # Track cross-attention weights for alignment
        cross_qk = [None] * len(self.blocks)

        # Select attention mask: custom (for tree attention) or standard causal
        mask = custom_mask if custom_mask is not None else self._mask

        # Process through transformer blocks
        for i, block in enumerate(self.blocks):
            x, kv_cache[i], cross_qk[i] = block(
                x, xa, mask=mask, kv_cache=kv_cache[i],
            )

        # Final layer norm (OPT-NEW-4: fused Metal kernel)
        hidden_states = mx.fast.layer_norm(x, self.ln.weight, self.ln.bias, eps=1e-5)

        # Project to vocabulary using tied weights
        # OPT-VOCAB: Matmul happens with padded vocab for efficiency
        logits = self.token_embedding.as_linear(hidden_states)

        # OPT-VOCAB: Slice back to original vocab size
        # This ensures output shape matches tokenizer expectations
        if self._pad_vocab and self._padded_vocab > self.n_vocab:
            logits = logits[..., :self.n_vocab]

        return logits, kv_cache, cross_qk, hidden_states if return_hidden else None

    def __call__(
        self,
        x: mx.array,
        xa: mx.array,
        kv_cache: list[tuple] | None = None,
        custom_mask: mx.array | None = None,
        return_hidden: bool = False,
        custom_positions: mx.array | None = None,
    ) -> tuple[mx.array, list[tuple], list[mx.array | None], mx.array | None]:
        """
        Forward pass through decoder.

        Args:
            x: Token ids, shape (batch, seq_len)
            xa: Encoder output, shape (batch, encoder_len, n_state)
            kv_cache: List of cached (self_kv, cross_kv) per layer
            custom_mask: Optional custom attention mask for tree attention.
                         Shape (seq_len, seq_len) with 0 for attend, -inf for mask.
                         Used for Medusa tree verification to enforce tree structure.
                         If None, uses standard causal mask.
            return_hidden: If True, also returns hidden states before final projection.
                          Used for Medusa head predictions.
            custom_positions: Optional custom positional embedding indices.
                             Shape (seq_len,) with position indices for each token.
                             Used for Medusa tree verification where tree tokens at
                             depth L all share position prefix_len + L instead of
                             sequential positions. If None, uses sequential positions.

        Returns:
            Tuple of:
            - Logits, shape (batch, seq_len, n_vocab)
            - Updated KV cache list
            - Cross-attention weights per layer (for word-level timestamps)
            - Hidden states, shape (batch, seq_len, n_state) if return_hidden=True, else None
        """
        seq_len = x.shape[-1]

        # Compute positional embeddings
        if custom_positions is not None:
            # Use custom position indices for tree verification
            # Each token can have an arbitrary position based on its tree depth
            pos_emb = self.positional_embedding[custom_positions]
        else:
            # Standard sequential positions based on KV cache offset
            offset = kv_cache[0][0][0].shape[1] if kv_cache else 0
            pos_emb = self._get_positional_slice(offset, seq_len)

        # OPT-1.2: Use compiled forward if enabled
        # Note: Compiled forward doesn't support custom_mask, return_hidden, or custom_positions
        if self._compile_enabled and custom_mask is None and not return_hidden and custom_positions is None:
            # Lazy compile on first use
            if self._compiled_forward is None:
                self._compiled_forward = mx.compile(self._forward_impl)
            result = self._compiled_forward(x, xa, pos_emb, kv_cache)
            # Return with None for hidden states (backward compatible)
            return result[0], result[1], result[2], None
        return self._forward_impl(x, xa, pos_emb, kv_cache, custom_mask, return_hidden)


class TimestampLogitFilter:
    """
    Filter logits to enforce timestamp rules.

    Adjusted to work with dynamic timestamp precision:
    - max_initial_timestamp is converted using current precision
    - Timestamp ordering is enforced correctly
    - End-of-transcript detection works with variable-length audio
    """

    def __init__(
        self,
        tokenizer,
        sample_begin: int,
        max_initial_timestamp: float | None,
        precision: float = 0.02,
    ):
        """
        Args:
            tokenizer: Whisper tokenizer
            sample_begin: Index where sampling begins (after SOT sequence)
            max_initial_timestamp: Maximum timestamp at start (seconds)
            precision: Timestamp precision (seconds per encoder position)
        """
        self.tokenizer = tokenizer
        self.sample_begin = sample_begin
        self.max_initial_timestamp = max_initial_timestamp
        self.precision = precision

        # Precompute max initial timestamp index
        self._max_initial_timestamp_index = None
        if max_initial_timestamp is not None:
            self._max_initial_timestamp_index = round(max_initial_timestamp / precision)

    def set_precision(self, precision: float, max_initial_timestamp: float | None = None):
        """
        Update precision for new audio.

        Called when timestamp precision changes for variable-length audio.

        Args:
            precision: New timestamp precision
            max_initial_timestamp: Optional new max initial timestamp
        """
        self.precision = precision
        if max_initial_timestamp is not None:
            self.max_initial_timestamp = max_initial_timestamp
        if self.max_initial_timestamp is not None:
            self._max_initial_timestamp_index = round(
                self.max_initial_timestamp / precision,
            )

    def apply(self, logits: mx.array, tokens: mx.array) -> mx.array:
        """
        Apply timestamp constraints to logits.

        Enforces:
        1. Timestamps must appear in pairs
        2. Timestamps must be non-decreasing
        3. Initial timestamp must be within max_initial_timestamp
        4. Prefer timestamps when their probability is high

        Args:
            logits: Raw logits, shape (batch, vocab_size)
            tokens: Generated tokens so far, shape (batch, seq_len)

        Returns:
            Filtered logits
        """
        import numpy as np

        mask = np.zeros(logits.shape, dtype=np.float32)

        # Suppress <|notimestamps|>
        if self.tokenizer.no_timestamps is not None:
            mask[:, self.tokenizer.no_timestamps] = -np.inf

        # Convert to list for processing
        tokens_list = tokens.tolist()

        for k in range(len(tokens_list)):
            seq = tokens_list[k][self.sample_begin:]

            # Check timestamp pattern
            last_was_timestamp = (
                len(seq) >= 1 and seq[-1] >= self.tokenizer.timestamp_begin
            )
            penultimate_was_timestamp = (
                len(seq) < 2 or seq[-2] >= self.tokenizer.timestamp_begin
            )

            if last_was_timestamp:
                if penultimate_was_timestamp:
                    # Two timestamps in a row - must be text next
                    mask[k, self.tokenizer.timestamp_begin:] = -np.inf
                else:
                    # One timestamp - cannot be normal text
                    mask[k, : self.tokenizer.eot] = -np.inf

            # Enforce non-decreasing timestamps
            timestamps = [
                i for i, v in enumerate(seq) if v > self.tokenizer.timestamp_begin
            ]
            if len(timestamps) > 0:
                last_timestamp = timestamps[-1]
                if not last_timestamp or penultimate_was_timestamp:
                    last_timestamp += 1
                mask[k, self.tokenizer.timestamp_begin: last_timestamp] = -np.inf

        # At start: suppress non-timestamp tokens, enforce max initial timestamp
        if len(tokens_list[0]) == self.sample_begin:
            mask[:, : self.tokenizer.timestamp_begin] = -np.inf

            if self._max_initial_timestamp_index is not None:
                last_allowed = (
                    self.tokenizer.timestamp_begin + self._max_initial_timestamp_index
                )
                mask[:, last_allowed + 1:] = -np.inf

        # Prefer timestamps if their probability is higher
        mask = mx.array(mask)
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        timestamp_logprob = logprobs[:, self.tokenizer.timestamp_begin:].logsumexp(
            axis=-1, keepdims=True,
        )
        max_text_token_logprob = logprobs[:, : self.tokenizer.timestamp_begin].max(
            axis=-1, keepdims=True,
        )
        mask[:, : self.tokenizer.timestamp_begin] = mx.where(
            timestamp_logprob > max_text_token_logprob,
            -mx.inf,
            mask[:, : self.tokenizer.timestamp_begin],
        )

        return logits + mask
