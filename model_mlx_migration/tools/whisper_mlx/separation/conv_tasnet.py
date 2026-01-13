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
Conv-TasNet: Time-domain Audio Separation Network.

Reference:
    Luo & Mesgarani, "Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude
    Masking for Speech Separation" (2019)
    https://arxiv.org/abs/1809.07454

This implementation ports Conv-TasNet to Apple MLX for efficient inference
on Apple Silicon devices.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Union

if TYPE_CHECKING:
    import numpy

import mlx.core as mx
import mlx.nn as nn

from .decoder import Decoder
from .encoder import Encoder
from .tcn import TemporalConvNet


@dataclass
class ConvTasNetConfig:
    """
    Configuration for Conv-TasNet model.

    Default values match the Asteroid/Libri2Mix pretrained model.
    Parameter names follow Asteroid conventions.
    """

    # Encoder/Decoder parameters
    n_filters: int = 512  # N: Number of filters in encoder
    kernel_size: int = 32  # L: Encoder kernel size (2ms at 16kHz)
    stride: int = 16  # Encoder stride (L // 2)

    # TCN Separator parameters (Asteroid naming)
    bn_chan: int = 128  # B: Bottleneck channels (TCN input/output)
    hid_chan: int = 512  # H: Hidden channels inside TCN blocks
    skip_chan: int = 128  # Sc: Skip connection channels
    conv_kernel_size: int = 3  # P: TCN kernel size
    n_blocks: int = 8  # X: Number of blocks per repeat
    n_repeats: int = 3  # R: Number of repeats (stacks)

    # Output parameters
    n_src: int = 2  # C: Number of sources to separate

    # Streaming parameters
    causal: bool = True  # Whether to use causal convolutions
    sample_rate: int = 16000  # Expected sample rate

    @classmethod
    def default_2spk(cls) -> "ConvTasNetConfig":
        """Default configuration for 2-speaker separation (Asteroid Libri2Mix)."""
        return cls(n_src=2)

    @classmethod
    def default_3spk(cls) -> "ConvTasNetConfig":
        """Default configuration for 3-speaker separation."""
        return cls(n_src=3)

    @classmethod
    def streaming(cls, n_sources: int = 2) -> "ConvTasNetConfig":
        """Configuration optimized for streaming inference."""
        return cls(
            n_src=n_sources,
            causal=True,
            # Smaller model for lower latency
            bn_chan=64,
            hid_chan=256,
            skip_chan=64,
            n_blocks=6,
            n_repeats=2,
        )

    @classmethod
    def from_asteroid(cls, model_args: dict[str, Any]) -> "ConvTasNetConfig":
        """Create config from Asteroid model_args dictionary."""
        return cls(
            n_filters=model_args.get("n_filters", 512),
            kernel_size=model_args.get("kernel_size", 32),
            stride=model_args.get("stride", 16),
            bn_chan=model_args.get("bn_chan", 128),
            hid_chan=model_args.get("hid_chan", 512),
            skip_chan=model_args.get("skip_chan", 128),
            conv_kernel_size=model_args.get("conv_kernel_size", 3),
            n_blocks=model_args.get("n_blocks", 8),
            n_repeats=model_args.get("n_repeats", 3),
            n_src=model_args.get("n_src", 2),
            sample_rate=model_args.get("sample_rate", 16000),
        )


class ConvTasNet(nn.Module):
    """
    Conv-TasNet model for speech source separation.

    Architecture:
        Input waveform (B, T)
            │
            ▼
        Encoder: Conv1d(1, N, L, stride=L//2) + ReLU
            │ (B, N, T')
            ▼
        TCN Separator: Stacked dilated convolutions
            │ (B, C, N, T') masks
            ▼
        Decoder: ConvTranspose1d(N, 1, L, stride=L//2)
            │
            ▼
        Output waveforms (B, C, T)

    Args:
        config: Model configuration. If None, uses default 2-speaker config.
    """

    def __init__(self, config: ConvTasNetConfig | None = None):
        super().__init__()
        self.config = config or ConvTasNetConfig.default_2spk()

        # Encoder: waveform -> latent representation
        self.encoder = Encoder(
            n_filters=self.config.n_filters,
            kernel_size=self.config.kernel_size,
            stride=self.config.stride,
        )

        # TCN Separator: estimate separation masks
        self.separator = TemporalConvNet(
            n_filters=self.config.n_filters,
            bn_chan=self.config.bn_chan,
            hid_chan=self.config.hid_chan,
            skip_chan=self.config.skip_chan,
            kernel_size=self.config.conv_kernel_size,
            n_layers=self.config.n_blocks,
            n_stacks=self.config.n_repeats,
            n_sources=self.config.n_src,
            causal=self.config.causal,
        )

        # Decoder: masked representation -> waveforms
        self.decoder = Decoder(
            n_filters=self.config.n_filters,
            kernel_size=self.config.kernel_size,
            stride=self.config.stride,
        )

    def __call__(self, waveform: mx.array) -> mx.array:
        """
        Separate mixed audio into individual sources.

        Args:
            waveform: Mixed audio of shape (B, T) or (T,).

        Returns:
            Separated sources of shape (B, C, T) or (C, T) if input was 1D.
        """
        # Handle 1D input
        squeeze_output = False
        if waveform.ndim == 1:
            waveform = mx.expand_dims(waveform, axis=0)
            squeeze_output = True

        # Store original length for output trimming
        original_length = waveform.shape[1]

        # Encode: (B, T) -> (B, N, T')
        encoded = self.encoder(waveform)

        # Separate: (B, N, T') -> (B, C, N, T') masks
        masks = self.separator(encoded)

        # Decode: (B, N, T'), (B, C, N, T') -> (B, C, T)
        separated = self.decoder(encoded, masks)

        # Trim to original length (decoder may produce slightly longer output)
        if separated.shape[2] > original_length:
            separated = separated[:, :, :original_length]
        elif separated.shape[2] < original_length:
            # Pad if shorter (rare edge case)
            pad_len = original_length - separated.shape[2]
            separated = mx.pad(separated, [(0, 0), (0, 0), (0, pad_len)])

        # Squeeze if input was 1D
        if squeeze_output:
            separated = mx.squeeze(separated, axis=0)

        return separated

    def separate(
        self,
        audio: Union[mx.array, "numpy.ndarray"],
        normalize: bool = True,
    ) -> list[mx.array]:
        """
        Convenience method to separate audio and return list of sources.

        Args:
            audio: Mixed audio of shape (T,) or (B, T).
            normalize: Whether to normalize output to [-1, 1].

        Returns:
            List of separated source arrays.
        """
        # Convert to MLX if needed
        if isinstance(audio, mx.array):
            # Already MLX array, ensure float32
            if audio.dtype != mx.float32:
                audio = audio.astype(mx.float32)
        else:
            # Assume numpy array or similar
            import numpy as np

            if not isinstance(audio, np.ndarray):
                audio = np.array(audio)
            audio = mx.array(audio.astype(np.float32))

        # Run separation
        separated = self(audio)
        mx.eval(separated)

        # Handle batch dimension
        if separated.ndim == 2:
            # (C, T) -> list of (T,)
            sources = [separated[i] for i in range(separated.shape[0])]
        else:
            # (B, C, T) -> list of (B, T)
            sources = [separated[:, i, :] for i in range(separated.shape[1])]

        # Normalize if requested
        if normalize:
            normalized = []
            for src in sources:
                max_val = mx.max(mx.abs(src))
                if max_val > 0:
                    src = src / max_val
                normalized.append(src)
            sources = normalized

        return sources

    def separate_streaming(
        self,
        audio: mx.array,
        chunk_size: int = 16000,
        overlap: int = 8000,
    ) -> list[list[mx.array]]:
        """
        Separate audio in streaming chunks with overlap-add.

        Args:
            audio: Full audio of shape (T,).
            chunk_size: Chunk size in samples.
            overlap: Overlap between chunks in samples.

        Returns:
            List of chunks, each containing list of separated sources.
        """
        if audio.ndim != 1:
            raise ValueError("Streaming mode expects 1D audio input")

        total_samples = audio.shape[0]
        step_size = chunk_size - overlap

        all_chunks = []
        for start in range(0, total_samples, step_size):
            end = min(start + chunk_size, total_samples)
            chunk = audio[start:end]

            # Pad last chunk if needed
            if chunk.shape[0] < chunk_size:
                pad_size = chunk_size - chunk.shape[0]
                chunk = mx.pad(chunk, [(0, pad_size)])

            # Separate this chunk
            sources = self.separate(chunk, normalize=False)

            # Trim back if padded
            if end - start < chunk_size:
                sources = [src[: end - start] for src in sources]

            all_chunks.append(sources)

        return all_chunks

    def get_latency_samples(self) -> int:
        """
        Get the algorithmic latency in samples.

        This is the minimum delay before the first output sample can be produced.

        Returns:
            Latency in samples.
        """
        # Latency comes from encoder kernel and TCN receptive field
        encoder_latency = self.config.kernel_size
        tcn_rf = self.separator.get_receptive_field()
        decoder_latency = self.config.kernel_size

        return encoder_latency + tcn_rf + decoder_latency

    def get_latency_ms(self) -> float:
        """
        Get the algorithmic latency in milliseconds.

        Returns:
            Latency in milliseconds.
        """
        samples = self.get_latency_samples()
        return samples / self.config.sample_rate * 1000


def load_conv_tasnet(
    weights_path: str,
    config: ConvTasNetConfig | None = None,
) -> ConvTasNet:
    """
    Load a Conv-TasNet model from weights file.

    Args:
        weights_path: Path to weights file (safetensors format).
        config: Model configuration. If None, inferred from weights.

    Returns:
        Loaded ConvTasNet model.
    """
    # Load weights
    weights = mx.load(weights_path)

    # Create model
    model = ConvTasNet(config)

    # Load weights into model
    from mlx.utils import tree_unflatten

    model.update(tree_unflatten(list(weights.items())))

    return model
