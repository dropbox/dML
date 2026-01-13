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
RNN-T Joiner in MLX.

The joiner combines encoder and decoder outputs to produce
logits over the vocabulary.
"""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class JoinerConfig:
    """Configuration for RNN-T joiner."""
    encoder_dim: int = 512
    decoder_dim: int = 512
    joiner_dim: int = 512
    vocab_size: int = 500


class Joiner(nn.Module):
    """
    RNN-T Joiner.

    Combines encoder and decoder representations:
        logits = output_linear(tanh(encoder_proj(enc) + decoder_proj(dec)))

    The joiner produces logits over the vocabulary for each
    (encoder_time, decoder_time) pair.
    """

    def __init__(self, config: JoinerConfig):
        """
        Initialize joiner.

        Args:
            config: Joiner configuration.
        """
        super().__init__()
        self.config = config

        # Projections
        self.encoder_proj = nn.Linear(config.encoder_dim, config.joiner_dim)
        self.decoder_proj = nn.Linear(config.decoder_dim, config.joiner_dim)

        # Output layer
        self.output_linear = nn.Linear(config.joiner_dim, config.vocab_size)

    def __call__(
        self,
        encoder_out: mx.array,
        decoder_out: mx.array,
        project_input: bool = True,
    ) -> mx.array:
        """
        Forward pass.

        Args:
            encoder_out: Encoder output.
                - Shape (batch, T, encoder_dim) for full sequence, or
                - Shape (batch, T, s_range, encoder_dim) for pruned output
            decoder_out: Decoder output.
                - Shape (batch, U, decoder_dim) for full sequence, or
                - Shape (batch, T, s_range, decoder_dim) for pruned output
            project_input: If True, apply encoder_proj and decoder_proj.
                          Set False if already projected.

        Returns:
            Logits of shape (batch, T, U, vocab_size) or
            (batch, T, s_range, vocab_size) for pruned output.
        """
        if project_input:
            encoder_out = self.encoder_proj(encoder_out)
            decoder_out = self.decoder_proj(decoder_out)

        # Handle different input shapes
        if encoder_out.ndim == 3 and decoder_out.ndim == 3:
            # Full computation: (batch, T, dim) + (batch, U, dim)
            # Expand to (batch, T, U, dim)
            # encoder: (batch, T, 1, dim)
            # decoder: (batch, 1, U, dim)
            encoder_out = mx.expand_dims(encoder_out, axis=2)
            decoder_out = mx.expand_dims(decoder_out, axis=1)

        # Combine via addition
        logits = encoder_out + decoder_out

        # Apply tanh activation
        logits = mx.tanh(logits)

        # Project to vocabulary
        logits = self.output_linear(logits)

        return logits

    def forward_streaming(
        self,
        encoder_out: mx.array,
        decoder_out: mx.array,
    ) -> mx.array:
        """
        Forward pass for streaming inference (single time step).

        Args:
            encoder_out: Encoder output of shape (batch, 1, encoder_dim).
            decoder_out: Decoder output of shape (batch, 1, decoder_dim).

        Returns:
            Logits of shape (batch, 1, vocab_size).
        """
        # Project
        encoder_out = self.encoder_proj(encoder_out)
        decoder_out = self.decoder_proj(decoder_out)

        # Combine (both are batch, 1, dim)
        logits = encoder_out + decoder_out

        # Apply tanh
        logits = mx.tanh(logits)

        # Project to vocabulary
        logits = self.output_linear(logits)

        return logits
