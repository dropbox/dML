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
Stateless RNN-T Decoder (Predictor) in MLX.

This is a stateless decoder that uses convolution instead of recurrence.
Based on the icefall pruned_transducer_stateless architecture.
"""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class DecoderConfig:
    """Configuration for stateless RNN-T decoder."""
    vocab_size: int = 500
    decoder_dim: int = 512
    blank_id: int = 0
    context_size: int = 2


class Decoder(nn.Module):
    """
    Stateless RNN-T decoder (predictor).

    Uses embedding + grouped convolution instead of RNN.
    This design removes recurrent connections while maintaining
    context awareness through convolution.

    Architecture:
        embedding -> (pad + conv) -> relu -> output

    The conv layer uses groups to reduce parameters while
    maintaining per-channel processing.
    """

    def __init__(self, config: DecoderConfig):
        """
        Initialize decoder.

        Args:
            config: Decoder configuration.
        """
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.decoder_dim = config.decoder_dim
        self.blank_id = config.blank_id
        self.context_size = config.context_size

        # Embedding layer
        self.embedding = nn.Embedding(config.vocab_size, config.decoder_dim)

        # Grouped convolution for context
        # groups = decoder_dim / 4 (each group has 4 channels)
        # This gives (out_channels, kernel_size, 4) weight shape
        if config.context_size > 1:
            self.conv = nn.Conv1d(
                in_channels=config.decoder_dim,
                out_channels=config.decoder_dim,
                kernel_size=config.context_size,
                groups=config.decoder_dim // 4,
                padding=0,
                bias=False,
            )
        else:
            self.conv = None

    def __call__(
        self,
        y: mx.array,
        need_pad: bool = True,
    ) -> mx.array:
        """
        Forward pass.

        Args:
            y: Token IDs of shape (batch, seq_len).
            need_pad: If True, pad on the left with blank tokens.
                     Set to False during streaming inference when
                     context is already provided.

        Returns:
            Decoder output of shape (batch, seq_len, decoder_dim).
        """
        # Get embeddings
        # Handle negative token IDs during tracing (clamp and mask)
        token_ids = mx.maximum(y, 0)  # Clamp negatives to 0
        embedding = self.embedding(token_ids)  # (batch, seq, decoder_dim)

        # Zero out embeddings for negative token IDs
        mask = (y >= 0).astype(embedding.dtype)
        embedding = embedding * mask[:, :, None]

        if self.conv is not None:
            if need_pad:
                # Pad on left with blank token embedding
                blank_embed = self.embedding(
                    mx.full((embedding.shape[0], self.context_size - 1),
                            self.blank_id, dtype=mx.int32),
                )
                embedding = mx.concatenate([blank_embed, embedding], axis=1)

            # Apply convolution: (batch, seq, dim)
            # MLX conv1d expects (batch, seq, channels)
            embedding = self.conv(embedding)

        # Apply ReLU
        embedding = nn.relu(embedding)

        return embedding

    def forward_one_step(
        self,
        y: mx.array,
        context: mx.array,
    ) -> mx.array:
        """
        Process one step during streaming inference.

        Args:
            y: Current token ID of shape (batch, 1).
            context: Previous context embeddings of shape
                    (batch, context_size - 1, decoder_dim).

        Returns:
            Decoder output of shape (batch, 1, decoder_dim).
        """
        # Get embedding for current token
        embedding = self.embedding(y)  # (batch, 1, decoder_dim)

        if self.conv is not None:
            # Concatenate with context
            embedding = mx.concatenate([context, embedding], axis=1)
            # Apply conv (no padding needed)
            embedding = self.conv(embedding)

        # Apply ReLU
        embedding = nn.relu(embedding)

        return embedding
