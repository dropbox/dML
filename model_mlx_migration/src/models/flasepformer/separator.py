"""
Source Separator using MossFormer2 MLX.

This module provides source separation capabilities using the MossFormer2 architecture
ported to Apple MLX. It separates mixed audio into individual speaker streams.

Usage:
    from models.flasepformer import SourceSeparator

    separator = SourceSeparator(num_speakers=2)
    sources = separator.separate(mixed_audio)
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
from mlx.utils import tree_unflatten

# Add third-party MossFormer2 to path
_MOSSFORMER_PATH = Path(__file__).parent.parent.parent.parent / "tools/third_party/mossformer_ss_mlx/python"
if str(_MOSSFORMER_PATH) not in sys.path:
    sys.path.insert(0, str(_MOSSFORMER_PATH))

from huggingface_hub import hf_hub_download  # noqa: E402 - import after sys.path modification
from mossformer2_ss_16k import MossFormer2_SS_16K_MLX  # noqa: E402


@dataclass
class SeparatorConfig:
    """Configuration for source separator."""
    num_speakers: int = 2
    sample_rate: int = 16000
    encoder_embedding_dim: int = 512
    mossformer_sequence_dim: int = 512
    num_mossformer_layer: int = 24
    encoder_kernel_size: int = 16
    is_whamr: bool = False
    compile_model: bool = True


# Model repository mappings
MODEL_REPOS = {
    (2, 16000, False): "starkdmi/MossFormer2_SS_2SPK_16K_MLX",
    (2, 8000, True): "starkdmi/MossFormer2_SS_2SPK_WHAMR_8K_MLX",
    (3, 8000, False): "starkdmi/MossFormer2_SS_3SPK_8K_MLX",
}


class SourceSeparator:
    """
    Source separator using MossFormer2 MLX.

    Separates mixed audio into individual speaker streams using the MossFormer2
    architecture optimized for Apple Silicon via MLX.

    Args:
        num_speakers: Number of speakers to separate (2 or 3)
        config: Optional configuration override
        weights_path: Optional path to pre-downloaded weights

    Example:
        >>> separator = SourceSeparator(num_speakers=2)
        >>> mixed = mx.array(np.random.randn(16000).astype(np.float32))  # 1s audio
        >>> sources = separator.separate(mixed)
        >>> len(sources)
        2
    """

    def __init__(
        self,
        num_speakers: int = 2,
        config: SeparatorConfig | None = None,
        weights_path: str | None = None,
    ):
        if num_speakers not in [2, 3]:
            raise ValueError(f"num_speakers must be 2 or 3, got {num_speakers}")

        self.num_speakers = num_speakers
        self.config = config or self._get_default_config(num_speakers)
        self._model = None
        self._weights_path = weights_path

    def _get_default_config(self, num_speakers: int) -> SeparatorConfig:
        """Get default configuration for given number of speakers."""
        if num_speakers == 2:
            return SeparatorConfig(
                num_speakers=2,
                sample_rate=16000,
                is_whamr=False,
            )
        # 3 speakers
        return SeparatorConfig(
            num_speakers=3,
            sample_rate=8000,  # 3-speaker model uses 8kHz
            is_whamr=False,
        )

    def _download_weights(self) -> str:
        """Download model weights from Hugging Face Hub."""
        key = (self.config.num_speakers, self.config.sample_rate, self.config.is_whamr)
        if key not in MODEL_REPOS:
            raise ValueError(
                f"No pretrained model for {self.config.num_speakers} speakers "
                f"at {self.config.sample_rate}Hz (whamr={self.config.is_whamr})",
            )

        repo_id = MODEL_REPOS[key]
        weights_path = hf_hub_download(
            repo_id=repo_id,
            filename="model_fp32.safetensors",
        )
        return weights_path

    def _load_model(self) -> None:
        """Load and initialize the model."""
        # Get weights path
        weights_path = self._weights_path
        if weights_path is None:
            weights_path = self._download_weights()

        # Create model architecture
        args = SimpleNamespace(
            encoder_embedding_dim=self.config.encoder_embedding_dim,
            mossformer_sequence_dim=self.config.mossformer_sequence_dim,
            num_mossformer_layer=self.config.num_mossformer_layer,
            encoder_kernel_size=self.config.encoder_kernel_size,
            num_spks=self.config.num_speakers,
            skip_mask_multiplication=self.config.is_whamr,
        )

        model = MossFormer2_SS_16K_MLX(args)

        # Load weights
        weights = mx.load(weights_path)
        model.update(tree_unflatten(list(weights.items())))

        # Optionally compile for faster inference
        if self.config.compile_model:
            model = mx.compile(model)

        self._model = model

    @property
    def model(self):
        """Lazy-load model on first access."""
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def sample_rate(self) -> int:
        """Expected sample rate for input audio."""
        return self.config.sample_rate

    def separate(
        self,
        audio: mx.array | np.ndarray,
    ) -> list[mx.array]:
        """
        Separate mixed audio into individual speaker streams.

        Args:
            audio: Mixed audio input. Can be:
                - 1D array (T,) - single audio
                - 2D array (B, T) - batched audio

        Returns:
            List of separated sources, each with shape matching input.
            If input is (T,), returns [(T,), (T,), ...] for num_speakers sources.
            If input is (B, T), returns [(B, T), (B, T), ...] for num_speakers sources.
        """
        # Convert to MLX if needed
        if isinstance(audio, np.ndarray):
            audio = mx.array(audio.astype(np.float32))

        # Track original shape
        was_1d = audio.ndim == 1
        if was_1d:
            audio = mx.expand_dims(audio, axis=0)  # (T,) -> (1, T)

        # Run separation
        separated = self.model(audio)
        mx.eval(separated)

        # Process outputs
        sources = []
        for src in separated:
            if was_1d:
                src = mx.squeeze(src, axis=0)  # (1, T) -> (T,)
            sources.append(src)

        return sources

    def separate_to_numpy(
        self,
        audio: mx.array | np.ndarray,
    ) -> list[np.ndarray]:
        """
        Separate mixed audio and return as numpy arrays.

        Convenience method for when numpy output is needed.

        Args:
            audio: Mixed audio input (see separate() for shapes)

        Returns:
            List of separated sources as numpy arrays
        """
        sources = self.separate(audio)
        return [np.array(src) for src in sources]

    def separate_streaming(
        self,
        audio: mx.array | np.ndarray,
        chunk_size: int = 16000,  # 1 second at 16kHz
        overlap: int = 0,
    ) -> list[list[mx.array]]:
        """
        Separate audio in streaming chunks.

        Note: This is a simple chunked approach. For true streaming separation,
        a causal model variant would be needed.

        Args:
            audio: Full audio to process in chunks
            chunk_size: Size of each chunk in samples
            overlap: Number of overlapping samples between chunks

        Returns:
            List of lists: outer list is per chunk, inner is per speaker
        """
        if isinstance(audio, np.ndarray):
            audio = mx.array(audio.astype(np.float32))

        if audio.ndim == 2:
            raise ValueError("Streaming mode only supports 1D audio input")

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
            sources = self.separate(chunk)

            # Trim back to original chunk size if padded
            if end - start < chunk_size:
                sources = [src[:end - start] for src in sources]

            all_chunks.append(sources)

        return all_chunks


def create_separator(
    num_speakers: int = 2,
    sample_rate: int = 16000,
    whamr: bool = False,
) -> SourceSeparator:
    """
    Factory function to create a source separator.

    Args:
        num_speakers: Number of speakers to separate (2 or 3)
        sample_rate: Expected sample rate (16000 for 2spk, 8000 for 3spk)
        whamr: Use WHAMR model variant (for noisy/reverberant audio)

    Returns:
        Configured SourceSeparator instance
    """
    config = SeparatorConfig(
        num_speakers=num_speakers,
        sample_rate=sample_rate,
        is_whamr=whamr,
    )
    return SourceSeparator(num_speakers=num_speakers, config=config)
