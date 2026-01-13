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
CosyVoice2 Tokenizer - Text and Speaker Embedding Processing

This module handles:
1. Text tokenization using Qwen2 BPE tokenizer
2. Speaker embedding extraction (placeholder - requires onnxruntime)

The CosyVoice2 pipeline requires:
- Text → text_ids (Qwen2 BPE tokens)
- Reference audio → speaker_embedding (192-dim x-vector from CAM++)

Note: ONNX runtime is not available on Python 3.14. When using Python 3.13
or earlier, speaker embeddings can be extracted using campplus.onnx.
"""

import warnings
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

# Module-level ONNX availability check (warning emitted only once)
_onnx_checked = False
_onnx_available = False


def _check_onnx_available() -> bool:
    """Check if ONNX runtime is available, warn once if not."""
    global _onnx_checked, _onnx_available
    if not _onnx_checked:
        _onnx_checked = True
        try:
            import onnxruntime  # noqa: F401

            _onnx_available = True
        except ImportError:
            warnings.warn(
                "onnxruntime not available. Speaker embedding extraction disabled. "
                "Use random_speaker_embedding() for testing.",
                stacklevel=3,
            )
            _onnx_available = False
    return _onnx_available


@dataclass
class CosyVoice2TokenizerConfig:
    """Configuration for CosyVoice2 tokenizer."""

    # Text tokenizer
    tokenizer_path: str | None = None
    max_length: int = 2048

    # Speaker embedding
    speaker_dim: int = 192  # CAM++ output dimension
    sample_rate: int = 16000  # CAM++ expects 16kHz


class CosyVoice2Tokenizer:
    """
    CosyVoice2 Tokenizer for text processing and speaker embedding.

    Usage:
        tokenizer = CosyVoice2Tokenizer.from_pretrained(model_path)

        # Text to tokens
        text_ids = tokenizer.encode("Hello, world!")

        # Speaker embedding (requires onnxruntime and reference audio)
        speaker_emb = tokenizer.extract_speaker_embedding(audio)
    """

    def __init__(self, config: CosyVoice2TokenizerConfig):
        self.config = config
        self._text_tokenizer = None
        self._speaker_encoder = None
        # Check ONNX availability (warning emitted once per module)
        self._onnx_available = _check_onnx_available()

    def _load_text_tokenizer(self, tokenizer_path: str):
        """Load Qwen2 text tokenizer from path."""
        try:
            from transformers import AutoTokenizer

            self._text_tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=True,
            )
        except ImportError:
            raise ImportError(
                "transformers package required for text tokenization. "
                "Install with: pip install transformers",
            ) from None

    def encode(
        self,
        text: str | list[str],
        add_special_tokens: bool = True,
        return_tensors: str = "mlx",
    ) -> mx.array:
        """
        Encode text to token IDs.

        Args:
            text: Input text string or list of strings
            add_special_tokens: Whether to add special tokens
            return_tensors: "mlx" for mx.array, "np" for numpy

        Returns:
            Token IDs as mx.array [batch, seq_len] or [seq_len] for single string
        """
        if self._text_tokenizer is None:
            raise RuntimeError(
                "Text tokenizer not loaded. Call from_pretrained() first.",
            )

        is_single = isinstance(text, str)
        if is_single:
            text = [text]

        # Tokenize
        encoded = self._text_tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            max_length=self.config.max_length,
            truncation=True,
            padding=True,
            return_tensors="np",
        )

        input_ids = encoded["input_ids"]

        if return_tensors == "np":
            return input_ids[0] if is_single else input_ids
        # mlx
        result = mx.array(input_ids)
        return result[0] if is_single else result

    def decode(
        self,
        token_ids: mx.array | np.ndarray | list[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode token IDs back to text.

        Args:
            token_ids: Token IDs
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded text string
        """
        if self._text_tokenizer is None:
            raise RuntimeError(
                "Text tokenizer not loaded. Call from_pretrained() first.",
            )

        if isinstance(token_ids, mx.array):
            token_ids = np.array(token_ids).tolist()
        elif isinstance(token_ids, np.ndarray):
            token_ids = token_ids.tolist()

        return self._text_tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
        )

    def extract_speaker_embedding(
        self,
        audio: mx.array | np.ndarray,
        sample_rate: int = 16000,
    ) -> mx.array:
        """
        Extract speaker embedding from reference audio using CAM++.

        Args:
            audio: Audio waveform [samples] or [channels, samples]
            sample_rate: Audio sample rate (will resample to 16kHz if different)

        Returns:
            Speaker embedding [192] (normalized x-vector)

        Raises:
            RuntimeError: If onnxruntime not available
        """
        if not self._onnx_available:
            raise RuntimeError(
                "onnxruntime not available for speaker embedding extraction. "
                "Use random_speaker_embedding() for testing, or install onnxruntime "
                "with a compatible Python version (<=3.13).",
            )

        if self._speaker_encoder is None:
            raise RuntimeError(
                "Speaker encoder not loaded. Ensure campplus.onnx exists in model path.",
            )

        # Convert to numpy if needed
        if isinstance(audio, mx.array):
            audio = np.array(audio)

        # Ensure mono
        if audio.ndim > 1:
            audio = audio.mean(axis=0)

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            try:
                import librosa

                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            except ImportError:
                # Fall back to scipy if librosa not available
                from scipy import signal

                num_samples = int(len(audio) * 16000 / sample_rate)
                audio = signal.resample(audio, num_samples)

        # Run ONNX inference
        embedding = self._speaker_encoder.run(
            None, {"audio": audio.astype(np.float32)[None, :]},
        )[0]

        # Normalize
        embedding = embedding / np.linalg.norm(embedding, axis=-1, keepdims=True)

        return mx.array(embedding.squeeze())

    def random_speaker_embedding(self, seed: int = 42) -> mx.array:
        """
        Generate a random speaker embedding for testing.

        This creates a normalized random vector that can be used
        as a placeholder speaker embedding when reference audio
        is not available or onnxruntime is not installed.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Random normalized speaker embedding [192]
        """
        mx.random.seed(seed)
        embedding = mx.random.normal((self.config.speaker_dim,))
        # L2 normalize
        return embedding / mx.sqrt(mx.sum(embedding * embedding))

    def zero_speaker_embedding(self) -> mx.array:
        """
        Generate a zero speaker embedding.

        This can be used for "default" voice synthesis without
        voice cloning characteristics.

        Returns:
            Zero speaker embedding [192]
        """
        return mx.zeros((self.config.speaker_dim,))

    @property
    def vocab_size(self) -> int:
        """Get text vocabulary size."""
        if self._text_tokenizer is None:
            return 0
        return len(self._text_tokenizer)

    @property
    def pad_token_id(self) -> int:
        """Get padding token ID."""
        if self._text_tokenizer is None:
            return 0
        return self._text_tokenizer.pad_token_id or 0

    @property
    def eos_token_id(self) -> int:
        """Get end-of-sequence token ID."""
        if self._text_tokenizer is None:
            return 0
        return self._text_tokenizer.eos_token_id or 0

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        config: CosyVoice2TokenizerConfig | None = None,
    ) -> "CosyVoice2Tokenizer":
        """
        Load tokenizer from pretrained model path.

        Args:
            model_path: Path to CosyVoice2 model directory
            config: Optional config override

        Returns:
            Loaded CosyVoice2Tokenizer
        """
        model_path = Path(model_path)

        if config is None:
            config = CosyVoice2TokenizerConfig()

        tokenizer = cls(config)

        # Load text tokenizer from CosyVoice-BlankEN subdirectory
        tokenizer_path = model_path / "CosyVoice-BlankEN"
        if tokenizer_path.exists():
            tokenizer._load_text_tokenizer(str(tokenizer_path))
        else:
            # Try parent directory or other common locations
            alt_paths = [
                model_path / "tokenizer",
                model_path,
            ]
            for path in alt_paths:
                if (path / "vocab.json").exists() or (path / "tokenizer.json").exists():
                    tokenizer._load_text_tokenizer(str(path))
                    break
            else:
                warnings.warn(
                    f"Text tokenizer not found in {model_path}. encode() will not work.",
                    stacklevel=2,
                )

        # Load speaker encoder if onnxruntime available
        if tokenizer._onnx_available:
            speaker_encoder_path = model_path / "campplus.onnx"
            if speaker_encoder_path.exists():
                import onnxruntime

                tokenizer._speaker_encoder = onnxruntime.InferenceSession(
                    str(speaker_encoder_path),
                    providers=["CPUExecutionProvider"],
                )
            else:
                warnings.warn(
                    f"campplus.onnx not found in {model_path}. "
                    "extract_speaker_embedding() will not work.",
                    stacklevel=2,
                )

        return tokenizer

    @staticmethod
    def get_default_model_path() -> Path:
        """Get default cache path for CosyVoice2 model."""
        return Path.home() / ".cache" / "cosyvoice2" / "cosyvoice2-0.5b"
