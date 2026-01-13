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
Whisper model configurations for all model sizes.
"""

from dataclasses import dataclass


@dataclass
class WhisperConfig:
    """Configuration for Whisper model architecture."""

    # Audio encoder
    n_mels: int = 128  # Number of mel frequency bands (80 for v1-v2, 128 for v3)
    n_audio_ctx: int = 1500  # Max audio context length (30s / 0.02s = 1500 frames)
    n_audio_state: int = 1280  # Audio encoder hidden dimension
    n_audio_head: int = 20  # Number of attention heads in audio encoder
    n_audio_layer: int = 32  # Number of transformer layers in audio encoder

    # Text decoder
    n_vocab: int = 51866  # Vocabulary size (51865 for multilingual, 51864 for English)
    n_text_ctx: int = 448  # Max text context length
    n_text_state: int = 1280  # Text decoder hidden dimension
    n_text_head: int = 20  # Number of attention heads in text decoder
    n_text_layer: int = 32  # Number of transformer layers in text decoder

    # Audio processing
    sample_rate: int = 16000  # Audio sample rate
    n_fft: int = 400  # FFT window size
    hop_length: int = 160  # Hop length between frames
    chunk_length: int = 30  # Standard chunk length in seconds

    # Model metadata
    name: str = "large-v3"  # Model name for identification

    @property
    def frames_per_second(self) -> float:
        """Audio frames per second after mel spectrogram."""
        return self.sample_rate / self.hop_length  # 100 fps for 16kHz/160

    @property
    def time_precision(self) -> float:
        """Default timestamp precision in seconds."""
        return self.chunk_length / self.n_audio_ctx  # 0.02s for 30s/1500

    def compute_precision(self, audio_duration: float, encoder_positions: int) -> float:
        """
        Compute dynamic timestamp precision based on actual audio length.

        This is the KEY FIX that enables dynamic chunk sizing:
        - Standard Whisper assumes 30s audio with 1500 encoder positions
        - For shorter audio, we adjust precision proportionally
        - This prevents decoder hallucinations with variable-length input

        Args:
            audio_duration: Actual audio duration in seconds
            encoder_positions: Actual encoder output sequence length

        Returns:
            Timestamp precision in seconds per encoder position
        """
        if encoder_positions <= 0:
            return self.time_precision
        return audio_duration / encoder_positions


# Predefined configurations for all Whisper model sizes
WHISPER_CONFIGS = {
    # Whisper v1/v2 models (80 mels)
    # Note: Multilingual models (tiny, base, etc.) have n_vocab=51865
    # English-only models (.en suffix) have n_vocab=51864
    "tiny": WhisperConfig(
        n_mels=80,
        n_audio_state=384,
        n_audio_head=6,
        n_audio_layer=4,
        n_vocab=51865,  # Multilingual model
        n_text_state=384,
        n_text_head=6,
        n_text_layer=4,
        name="tiny",
    ),
    "tiny.en": WhisperConfig(
        n_mels=80,
        n_audio_state=384,
        n_audio_head=6,
        n_audio_layer=4,
        n_vocab=51864,
        n_text_state=384,
        n_text_head=6,
        n_text_layer=4,
        name="tiny.en",
    ),
    "base": WhisperConfig(
        n_mels=80,
        n_audio_state=512,
        n_audio_head=8,
        n_audio_layer=6,
        n_vocab=51865,  # Multilingual model
        n_text_state=512,
        n_text_head=8,
        n_text_layer=6,
        name="base",
    ),
    "base.en": WhisperConfig(
        n_mels=80,
        n_audio_state=512,
        n_audio_head=8,
        n_audio_layer=6,
        n_vocab=51864,
        n_text_state=512,
        n_text_head=8,
        n_text_layer=6,
        name="base.en",
    ),
    "small": WhisperConfig(
        n_mels=80,
        n_audio_state=768,
        n_audio_head=12,
        n_audio_layer=12,
        n_vocab=51865,  # Multilingual model
        n_text_state=768,
        n_text_head=12,
        n_text_layer=12,
        name="small",
    ),
    "small.en": WhisperConfig(
        n_mels=80,
        n_audio_state=768,
        n_audio_head=12,
        n_audio_layer=12,
        n_vocab=51864,
        n_text_state=768,
        n_text_head=12,
        n_text_layer=12,
        name="small.en",
    ),
    "medium": WhisperConfig(
        n_mels=80,
        n_audio_state=1024,
        n_audio_head=16,
        n_audio_layer=24,
        n_vocab=51865,  # Multilingual model
        n_text_state=1024,
        n_text_head=16,
        n_text_layer=24,
        name="medium",
    ),
    "medium.en": WhisperConfig(
        n_mels=80,
        n_audio_state=1024,
        n_audio_head=16,
        n_audio_layer=24,
        n_vocab=51864,
        n_text_state=1024,
        n_text_head=16,
        n_text_layer=24,
        name="medium.en",
    ),
    "large": WhisperConfig(
        n_mels=80,
        n_audio_state=1280,
        n_audio_head=20,
        n_audio_layer=32,
        n_vocab=51865,
        n_text_state=1280,
        n_text_head=20,
        n_text_layer=32,
        name="large",
    ),
    "large-v2": WhisperConfig(
        n_mels=80,
        n_audio_state=1280,
        n_audio_head=20,
        n_audio_layer=32,
        n_vocab=51865,
        n_text_state=1280,
        n_text_head=20,
        n_text_layer=32,
        name="large-v2",
    ),

    # Whisper v3 models (128 mels)
    "large-v3": WhisperConfig(
        n_mels=128,
        n_audio_state=1280,
        n_audio_head=20,
        n_audio_layer=32,
        n_vocab=51866,
        n_text_state=1280,
        n_text_head=20,
        n_text_layer=32,
        name="large-v3",
    ),
    "large-v3-turbo": WhisperConfig(
        n_mels=128,
        n_audio_state=1280,
        n_audio_head=20,
        n_audio_layer=32,
        n_vocab=51866,
        n_text_state=1280,
        n_text_head=20,
        n_text_layer=4,  # Turbo has only 4 decoder layers
        name="large-v3-turbo",
    ),

    # Distil-whisper models (distilled from large-v2/v3)
    # distil-large-v3 has same encoder as large-v3 but only 2 decoder layers
    "distil-large-v3": WhisperConfig(
        n_mels=128,
        n_audio_state=1280,
        n_audio_head=20,
        n_audio_layer=32,
        n_vocab=51866,
        n_text_state=1280,
        n_text_head=20,
        n_text_layer=2,  # Distilled to 2 decoder layers
        name="distil-large-v3",
    ),
    "distil-large-v2": WhisperConfig(
        n_mels=80,
        n_audio_state=1280,
        n_audio_head=20,
        n_audio_layer=32,
        n_vocab=51865,
        n_text_state=1280,
        n_text_head=20,
        n_text_layer=2,  # Distilled to 2 decoder layers
        name="distil-large-v2",
    ),
    "distil-medium.en": WhisperConfig(
        n_mels=80,
        n_audio_state=1024,
        n_audio_head=16,
        n_audio_layer=24,
        n_vocab=51864,
        n_text_state=1024,
        n_text_head=16,
        n_text_layer=2,  # Distilled to 2 decoder layers
        name="distil-medium.en",
    ),
}


def get_config(model_name: str) -> WhisperConfig:
    """
    Get configuration for a Whisper model.

    Args:
        model_name: Model name (e.g., "large-v3", "base.en")

    Returns:
        WhisperConfig for the requested model

    Raises:
        ValueError: If model name is not recognized
    """
    # Normalize model name
    name = model_name.lower().replace("whisper-", "").replace("_", "-")

    if name not in WHISPER_CONFIGS:
        available = ", ".join(sorted(WHISPER_CONFIGS.keys()))
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")

    return WHISPER_CONFIGS[name]
