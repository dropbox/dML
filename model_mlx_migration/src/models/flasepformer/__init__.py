"""
Source Separation module using MossFormer2 MLX.

This module provides multi-speaker source separation capabilities using the
MossFormer2 architecture ported to Apple MLX for efficient inference on Apple Silicon.

Key components:
- SourceSeparator: Main class for separating mixed audio into speaker streams
- SeparatorConfig: Configuration dataclass
- create_separator: Factory function for easy instantiation

Usage:
    from models.flasepformer import SourceSeparator, create_separator

    # Using class directly
    separator = SourceSeparator(num_speakers=2)
    sources = separator.separate(mixed_audio)

    # Using factory function
    separator = create_separator(num_speakers=2, sample_rate=16000)
    sources = separator.separate(mixed_audio)

Performance:
    - SI-SDRi: ~21 dB (state-of-the-art)
    - Speed: 3.65x real-time on Apple Silicon
    - Supports: 2-speaker (16kHz) and 3-speaker (8kHz) separation

Note:
    The roadmap targets FLASepformer (2.29x faster than MossFormer2), but
    MossFormer2 is used as the current implementation since it's already
    available with MLX support and provides excellent quality.
"""

from .separator import (
    SeparatorConfig,
    SourceSeparator,
    create_separator,
)

__all__ = [
    "SourceSeparator",
    "SeparatorConfig",
    "create_separator",
]
