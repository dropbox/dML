"""ECAPA-TDNN Model Implementation in MLX - Native Format with Compile.

This module implements the full ECAPA-TDNN model for spoken language identification.
Reference: https://arxiv.org/abs/2005.07143
Checkpoint: speechbrain/lang-id-voxlingua107-ecapa

OPTIMIZED: Uses MLX native (B, T, C) format - NO TRANSPOSITIONS.
OPTIMIZED: Uses mx.compile() for JIT compilation.

Architecture:
    Input: 60 mel filterbanks (B, T, 60)
    Block 0: TDNN (Conv1d + BN + ReLU)
    Blocks 1-3: SE-Res2Net blocks with increasing dilation
    MFA: Multi-Feature Aggregation (concatenate + project)
    ASP: Attentive Statistics Pooling
    FC: Final embedding projection
    Classifier: Language classification head (optional)
"""

import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from .ecapa_config import ECAPATDNNConfig
from .ecapa_layers import (
    AttentiveStatisticsPooling,
    BatchNorm1d,
    Conv1d,
    SERes2NetBlock,
    TDNNBlock,
)


class ECAPATDNN(nn.Module):
    """ECAPA-TDNN Embedding Model - Native (B, T, C) format.

    Extracts speaker/language embeddings from mel filterbank features.

    Args:
        config: Model configuration
    """

    def __init__(self, config: ECAPATDNNConfig):
        super().__init__()
        self.config = config

        # Initial TDNN layer
        self.blocks_0 = TDNNBlock(
            config.n_mels,
            config.channels[0],
            kernel_size=config.kernel_sizes[0],
            dilation=config.dilations[0],
        )

        # SE-Res2Net blocks (blocks 1-3)
        self.blocks_1 = SERes2NetBlock(
            config.channels[1],
            kernel_size=config.kernel_sizes[1],
            dilation=config.dilations[1],
            scale=config.res2net_scale,
            se_channels=config.se_channels,
        )

        self.blocks_2 = SERes2NetBlock(
            config.channels[2],
            kernel_size=config.kernel_sizes[2],
            dilation=config.dilations[2],
            scale=config.res2net_scale,
            se_channels=config.se_channels,
        )

        self.blocks_3 = SERes2NetBlock(
            config.channels[3],
            kernel_size=config.kernel_sizes[3],
            dilation=config.dilations[3],
            scale=config.res2net_scale,
            se_channels=config.se_channels,
        )

        # Multi-Feature Aggregation
        # Concatenates outputs from blocks 1, 2, 3 (3 * 1024 = 3072)
        self.mfa = TDNNBlock(
            config.channels[4],  # 3072
            config.channels[4],  # 3072
            kernel_size=1,
        )

        # Attentive Statistics Pooling
        self.asp = AttentiveStatisticsPooling(
            config.channels[4],  # 3072
            config.attention_channels,  # 128
        )

        # BatchNorm after ASP
        self.asp_bn = BatchNorm1d(config.channels[4] * 2)  # 6144

        # Final embedding layer
        self.fc = Conv1d(
            config.channels[4] * 2,  # 6144
            config.lin_neurons,  # 256
            kernel_size=1,
        )

    def __call__(
        self,
        x: mx.array,
        return_features: bool = False,
    ) -> mx.array:
        """Extract embeddings from mel filterbank features.

        Args:
            x: Input tensor (batch, time, n_mels) - NATIVE MLX FORMAT
               OR (batch, n_mels, time) - will be transposed once at input

        Returns:
            Embeddings of shape (batch, 1, lin_neurons)
        """
        # Ensure input is (B, T, C) format - native MLX
        # If input is (B, n_mels, T) (PyTorch format), transpose once at input
        if x.shape[-1] != self.config.n_mels and x.shape[1] == self.config.n_mels:
            # Input is (B, C, T), convert to (B, T, C) once at input boundary
            x = mx.transpose(x, (0, 2, 1))

        # Block 0: Initial TDNN
        x = self.blocks_0(x)

        # SE-Res2Net blocks
        out1 = self.blocks_1(x)
        out2 = self.blocks_2(out1)
        out3 = self.blocks_3(out2)

        # Multi-Feature Aggregation: concatenate and project (axis=2 for channels)
        mfa_in = mx.concatenate([out1, out2, out3], axis=2)
        mfa_out = self.mfa(mfa_in)

        # Attentive Statistics Pooling
        asp_out = self.asp(mfa_out)

        # BatchNorm
        asp_out = self.asp_bn(asp_out)

        # Final embedding
        return self.fc(asp_out)

        # Output: (B, 1, 256) - native MLX format

    def encode(self, x: mx.array) -> mx.array:
        """Alias for forward pass, returns (B, 256) embeddings."""
        emb = self(x)
        return emb.squeeze(1)


class ECAPAClassifier(nn.Module):
    """Language Classifier for ECAPA-TDNN - Native (B, T, C) format.

    Takes embeddings and predicts language probabilities.

    Args:
        config: Model configuration
    """

    def __init__(self, config: ECAPATDNNConfig):
        super().__init__()
        self.config = config

        # Input normalization
        self.norm = BatchNorm1d(config.lin_neurons)

        # DNN block
        self.linear = nn.Linear(config.lin_neurons, config.classifier_hidden)
        self.dnn_norm = BatchNorm1d(config.classifier_hidden)

        # Output layer
        self.out = nn.Linear(config.classifier_hidden, config.num_languages)

    def __call__(self, x: mx.array) -> mx.array:
        """Classify language from embeddings.

        Args:
            x: Embeddings (batch, 1, lin_neurons) or (batch, lin_neurons)

        Returns:
            Log probabilities (batch, num_languages)
        """
        # Handle different input shapes
        if x.ndim == 3:
            x = x.squeeze(1)  # (B, 1, D) -> (B, D)

        # Reshape for BatchNorm1d: (B, D) -> (B, 1, D) for native format
        x = x[:, None, :]
        x = self.norm(x)
        x = x.squeeze(1)  # (B, 1, D) -> (B, D)

        # DNN block
        x = self.linear(x)
        x = x[:, None, :]
        x = self.dnn_norm(x)
        x = x.squeeze(1)
        x = nn.leaky_relu(x)

        # Output
        return self.out(x)


class ECAPATDNNForLanguageID(nn.Module):
    """Complete ECAPA-TDNN model for Language Identification.

    Combines embedding model and classifier.
    OPTIMIZED: Uses mx.compile() for JIT compilation.

    Args:
        config: Model configuration
    """

    def __init__(self, config: ECAPATDNNConfig | None = None):
        super().__init__()
        self.config = config or ECAPATDNNConfig.voxlingua107()
        self.embedding_model = ECAPATDNN(self.config)
        self.classifier = ECAPAClassifier(self.config)

        # Label encoder (loaded from file)
        self.labels: dict[int, str] | None = None

        # Compiled forward pass (created on first call)
        self._compiled_forward = None

    def _forward_impl(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """Internal forward implementation for compilation."""
        # Get embeddings
        embeddings = self.embedding_model(x)

        # Classify
        logits = self.classifier(embeddings)

        # Get predictions
        predictions = mx.argmax(logits, axis=-1)

        return logits, predictions, embeddings

    def __call__(
        self,
        x: mx.array,
        return_embedding: bool = False,
        use_compile: bool = True,
    ) -> tuple[mx.array, mx.array]:
        """Predict language from mel features.

        Args:
            x: Mel filterbank features (batch, time, n_mels) or (batch, n_mels, time)
            return_embedding: If True, also return embeddings
            use_compile: If True, use compiled forward pass (default True)

        Returns:
            Tuple of (logits, predicted_indices) or
            (logits, predicted_indices, embeddings) if return_embedding
        """
        if use_compile:
            # Create compiled function on first call
            if self._compiled_forward is None:
                self._compiled_forward = mx.compile(self._forward_impl)

            logits, predictions, embeddings = self._compiled_forward(x)
        else:
            logits, predictions, embeddings = self._forward_impl(x)

        if return_embedding:
            return logits, predictions, embeddings

        return logits, predictions

    def classify_batch(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array, list]:
        """Classify a batch of audio, returning full SpeechBrain-style output.

        Args:
            x: Mel filterbank features (batch, time, n_mels) or (batch, n_mels, time)

        Returns:
            Tuple of (log_probs, confidence, predictions, language_codes)
        """
        logits, predictions = self(x)

        # Apply softmax for probabilities
        probs = mx.softmax(logits, axis=-1)

        # Get confidence (max probability)
        confidence = mx.max(probs, axis=-1)

        # Get language codes
        if self.labels is not None:
            lang_codes = [self.labels.get(int(p), "unknown") for p in predictions.tolist()]
        else:
            lang_codes = [str(int(p)) for p in predictions.tolist()]

        return logits, confidence, predictions, lang_codes

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        config: ECAPATDNNConfig | None = None,
    ) -> "ECAPATDNNForLanguageID":
        """Load model from converted MLX weights.

        Args:
            path: Path to directory containing:
                - weights.npz: MLX weights
                - config.json: Configuration (optional)
                - label_encoder.txt: Language labels (optional)

        Returns:
            Loaded model
        """
        path = Path(path)

        # Load config if available
        config_path = path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config_dict = json.load(f)
                config = ECAPATDNNConfig(**config_dict)

        # Create model
        model = cls(config)

        # Load weights using load_weights (handles list indices properly)
        weights_path = path / "weights.npz"
        if weights_path.exists():
            weights = mx.load(str(weights_path))
            model.load_weights(list(weights.items()))

        # Load labels (format: "'code: Language' => index")
        labels_path = path / "label_encoder.txt"
        if labels_path.exists():
            model.labels = {}
            with open(labels_path) as f:
                for line in f:
                    line = line.strip()
                    if "=>" in line:
                        # Format: "'ab: Abkhazian' => 0"
                        parts = line.split("=>")
                        if len(parts) == 2:
                            label_part = parts[0].strip().strip("'")
                            idx = int(parts[1].strip())
                            # Extract just the code (e.g., "ab" from "ab: Abkhazian")
                            lang_code = label_part.split(":")[0].strip()
                            model.labels[idx] = lang_code
                    elif len(line.split()) >= 2:
                        # Fallback: "code index" format
                        parts = line.split()
                        lang_code = parts[0]
                        idx = int(parts[1])
                        model.labels[idx] = lang_code

        return model

    def apply_weights(self, weights: dict[str, mx.array]) -> dict[str, Any]:
        """Apply loaded weights to model parameters.

        Args:
            weights: Dictionary of weight arrays

        Returns:
            Updated parameter dictionary
        """
        # This will be called by model.update()
        # Convert flat weight dict to nested structure
        params = {}

        for key, value in weights.items():
            parts = key.split(".")
            current = params
            for _i, part in enumerate(parts[:-1]):
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value

        return params
