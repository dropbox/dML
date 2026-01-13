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
Audio encoder for WhisperMLX.

Implements variable-length encoding:
- Accepts any mel spectrogram length (not just 30s)
- Slices positional embeddings for actual input length
- Proven by Worker #1180: sinusoidal embeddings are position-invariant

OPT-1.1: mx.compile for encoder forward pass
- Compiles the encoder forward pass for 1.5-2.5x speedup
- Compilation happens lazily on first call with each unique input signature

OPT-1.1-SW: Sliding window encoder attention
- Limits attention to local context window for speedup
- Expected 1.5-2x encoder speedup for long sequences
- Configurable window_size (default: None = full attention)
"""

import math

import mlx.core as mx
import mlx.nn as nn

from .attention import ResidualAttentionBlock, create_sliding_window_mask


def sinusoids(length: int, channels: int, max_timescale: int = 10000) -> mx.array:
    """
    Generate sinusoidal positional embeddings.

    These embeddings are position-invariant: any prefix of the embedding
    sequence works correctly. This enables variable-length encoding by
    simply slicing the embeddings to match input length.

    Args:
        length: Sequence length
        channels: Embedding dimension (must be even)
        max_timescale: Maximum timescale for sinusoids

    Returns:
        Positional embeddings, shape (length, channels)
    """
    assert channels % 2 == 0, "Channels must be even for sinusoidal embeddings"

    log_timescale_increment = math.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = mx.exp(-log_timescale_increment * mx.arange(channels // 2))

    scaled_time = mx.arange(length)[:, None] * inv_timescales[None, :]
    return mx.concatenate([mx.sin(scaled_time), mx.cos(scaled_time)], axis=1)


class AudioEncoder(nn.Module):
    """
    Whisper audio encoder with variable-length support.

    Key optimization: Instead of padding all audio to 30s, we:
    1. Compute mel spectrogram for actual audio length
    2. Slice positional embeddings to match actual input
    3. Process only the necessary frames

    This provides 2.6x average speedup for <30s audio (proven by Worker #1180).

    OPT-1.1: mx.compile for encoder forward pass
    - Call compile_forward() after initialization to enable JIT compilation
    - Provides 1.5-2.5x additional speedup after warmup
    """

    def __init__(
        self,
        n_mels: int,
        n_ctx: int,
        n_state: int,
        n_head: int,
        n_layer: int,
        dtype: mx.Dtype = mx.float16,
        use_fused: bool = True,
        compile_forward: bool = True,
        window_size: int | None = None,
    ):
        """
        Args:
            n_mels: Number of mel frequency bands (80 or 128)
            n_ctx: Maximum context length (1500 for 30s audio)
            n_state: Hidden dimension
            n_head: Number of attention heads
            n_layer: Number of transformer layers
            dtype: Data type for computation
            use_fused: Use fused attention operations
            compile_forward: Enable mx.compile for forward pass (OPT-1.1)
            window_size: Sliding window size for attention (OPT-1.1-SW).
                        None = full attention, 512 = recommended for speedup
        """
        super().__init__()

        self.n_ctx = n_ctx
        self.n_state = n_state
        self._compile_enabled = compile_forward
        self._dtype = dtype

        # OPT-1.1-SW: Sliding window attention
        self.window_size = window_size

        # Convolutional frontend
        self.conv1 = nn.Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)

        # Positional embeddings (precompute for max length)
        # These are position-invariant, so we can slice for shorter sequences
        self._positional_embedding = sinusoids(n_ctx, n_state).astype(dtype)

        # OPT-1.1-SW: Pre-compute sliding window mask for max length
        # Will be sliced for shorter sequences
        if window_size is not None:
            self._sliding_window_mask = create_sliding_window_mask(n_ctx, window_size, dtype)
        else:
            self._sliding_window_mask = None

        # Transformer blocks
        self.blocks = [
            ResidualAttentionBlock(n_state, n_head, use_fused=use_fused)
            for _ in range(n_layer)
        ]

        # Final layer norm
        self.ln_post = nn.LayerNorm(n_state)

        # OPT-1.1: Compiled forward functions (lazy initialization)
        # We compile separate versions for with/without mask
        self._compiled_forward_fixed: callable | None = None
        self._compiled_forward_variable: callable | None = None
        self._compiled_forward_fixed_masked: callable | None = None
        self._compiled_forward_variable_masked: callable | None = None

    def _forward_impl(
        self,
        x: mx.array,
        pos_embedding: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        """
        Core encoder forward pass (compilable).

        This method is separated to enable mx.compile optimization.
        It takes positional embeddings as input to avoid control flow
        that would require recompilation.

        Args:
            x: Mel spectrogram input
            pos_embedding: Positional embeddings (sliced for actual length)
            mask: Optional attention mask (sliding window mask for OPT-1.1-SW)
        """
        # Handle unbatched input: (n_frames, n_mels) -> (1, n_frames, n_mels)
        if x.ndim == 2:
            x = x[None]

        # Convolutional frontend
        x = nn.gelu(self.conv1(x))
        x = nn.gelu(self.conv2(x))

        # Add positional embedding
        x = x + pos_embedding

        # Transformer blocks
        # OPT-1.1-SW: Pass sliding window mask to self-attention
        for block in self.blocks:
            x, _, _ = block(x, mask=mask)

        # Final layer norm (OPT-NEW-4: fused Metal kernel)
        return mx.fast.layer_norm(x, self.ln_post.weight, self.ln_post.bias, eps=1e-5)


    def _forward_impl_no_mask(
        self,
        x: mx.array,
        pos_embedding: mx.array,
    ) -> mx.array:
        """
        Core encoder forward pass without mask (compilable).

        Separate function for mx.compile when no sliding window is used.
        This avoids recompilation when switching between masked/unmasked.
        """
        # Handle unbatched input: (n_frames, n_mels) -> (1, n_frames, n_mels)
        if x.ndim == 2:
            x = x[None]

        # Convolutional frontend
        x = nn.gelu(self.conv1(x))
        x = nn.gelu(self.conv2(x))

        # Add positional embedding
        x = x + pos_embedding

        # Transformer blocks (no mask = full attention)
        for block in self.blocks:
            x, _, _ = block(x)

        # Final layer norm (OPT-NEW-4: fused Metal kernel)
        return mx.fast.layer_norm(x, self.ln_post.weight, self.ln_post.bias, eps=1e-5)


    def __call__(
        self,
        x: mx.array,
        variable_length: bool = False,
    ) -> mx.array:
        """
        Encode audio mel spectrogram.

        Args:
            x: Mel spectrogram, shape (n_frames, n_mels) or (batch, n_frames, n_mels)
            variable_length: If True, use actual input length instead of padding

        Returns:
            Encoded audio features, shape (batch, seq_len, n_state)

        Note:
            MLX Conv1d expects input shape (batch, length, channels).
            Mel spectrograms should be (n_frames, n_mels) to match mlx-whisper.
        """
        # Determine sequence length after conv2 (stride=2)
        input_frames = x.shape[-2] if x.ndim == 3 else x.shape[0]
        seq_len = (input_frames + 1) // 2

        if variable_length:
            # VARIABLE LENGTH MODE: Slice positional embeddings to actual length
            if seq_len > self.n_ctx:
                raise ValueError(
                    f"Input sequence length {seq_len} exceeds maximum context {self.n_ctx}",
                )
            pos = self._positional_embedding[:seq_len]
        else:
            # STANDARD MODE: Assert exact shape match
            if seq_len != self._positional_embedding.shape[0]:
                raise ValueError(
                    f"Input shape mismatch: got {seq_len}, expected {self._positional_embedding.shape[0]}. "
                    f"Use variable_length=True for non-30s audio.",
                )
            pos = self._positional_embedding

        # OPT-1.1-SW: Get sliding window mask (sliced for variable length)
        use_sliding_window = self._sliding_window_mask is not None
        if use_sliding_window:
            if variable_length:
                mask = self._sliding_window_mask[:seq_len, :seq_len]
            else:
                mask = self._sliding_window_mask
        else:
            mask = None

        # OPT-1.1: Use compiled forward if enabled
        if self._compile_enabled:
            if use_sliding_window:
                # With sliding window mask
                if variable_length:
                    if self._compiled_forward_variable_masked is None:
                        self._compiled_forward_variable_masked = mx.compile(self._forward_impl)
                    return self._compiled_forward_variable_masked(x, pos, mask)
                if self._compiled_forward_fixed_masked is None:
                    self._compiled_forward_fixed_masked = mx.compile(self._forward_impl)
                return self._compiled_forward_fixed_masked(x, pos, mask)
            # Without sliding window mask (use no-mask version for efficiency)
            if variable_length:
                if self._compiled_forward_variable is None:
                    self._compiled_forward_variable = mx.compile(self._forward_impl_no_mask)
                return self._compiled_forward_variable(x, pos)
            if self._compiled_forward_fixed is None:
                self._compiled_forward_fixed = mx.compile(self._forward_impl_no_mask)
            return self._compiled_forward_fixed(x, pos)
        if use_sliding_window:
            return self._forward_impl(x, pos, mask)
        return self._forward_impl_no_mask(x, pos)

    def get_output_length(self, n_mel_frames: int) -> int:
        """
        Calculate encoder output length for given mel spectrogram length.

        The conv2 layer has stride=2, so output is roughly half the input.

        Args:
            n_mel_frames: Number of mel spectrogram frames

        Returns:
            Encoder output sequence length
        """
        # After conv1 (padding=1): same length
        # After conv2 (stride=2, padding=1): (length + 1) // 2
        return (n_mel_frames + 1) // 2

    def set_window_size(self, window_size: int | None) -> None:
        """
        Set sliding window size for encoder attention (OPT-1.1-SW).

        Args:
            window_size: Window size for sliding window attention.
                        None = full attention (no speedup, maximum quality)
                        512 = recommended for 1.5-2x speedup with minimal quality loss
                        256 = more aggressive, ~2x speedup with small quality impact

        Note:
            This invalidates any compiled forward functions, so there will be
            a recompilation cost on the next forward pass.
        """
        if window_size == self.window_size:
            return  # No change

        self.window_size = window_size

        # Regenerate mask
        if window_size is not None:
            self._sliding_window_mask = create_sliding_window_mask(
                self.n_ctx, window_size, self._dtype,
            )
        else:
            self._sliding_window_mask = None

        # Invalidate compiled functions (will be recompiled on next use)
        self._compiled_forward_fixed_masked = None
        self._compiled_forward_variable_masked = None
