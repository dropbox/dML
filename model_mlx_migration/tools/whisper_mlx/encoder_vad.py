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
Encoder VAD Head for WhisperMLX (Phase 3 Optimization).

Adds a lightweight Voice Activity Detection head to the encoder output
to identify speech vs silence frames. This enables skipping decoder
calls for silent positions, providing ~1.2x speedup.

Architecture:
- Input: Encoder output (batch, seq_len, n_state)
- Simple MLP: Linear → GELU → Linear → Sigmoid
- Output: Per-position speech probability (batch, seq_len)

Training:
- Distill from Silero VAD (pretrained, high-quality VAD)
- Run Silero on audio, align outputs to encoder positions
- Train with BCE loss

Inference:
- Run encoder as normal
- Get VAD probabilities for each position
- Skip decoder for positions below speech threshold

Expected speedup: 1.2x overall (skipping ~20% silent frames)
"""


import mlx.core as mx
import mlx.nn as nn
import numpy


class EncoderVADHead(nn.Module):
    """
    Lightweight VAD head for Whisper encoder outputs.

    Predicts speech probability for each encoder position.
    Uses a small MLP to minimize overhead (< 1ms inference time).

    Architecture:
        Linear(n_state, hidden_dim) → GELU → Linear(hidden_dim, 1) → Sigmoid

    Example:
        vad_head = EncoderVADHead(n_state=1280, hidden_dim=256)
        encoder_output = model.encoder(mel)  # (batch, seq_len, 1280)
        speech_probs = vad_head(encoder_output)  # (batch, seq_len)
        speech_mask = speech_probs > 0.5  # Boolean mask
    """

    def __init__(
        self,
        n_state: int = 1280,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        dtype: mx.Dtype = mx.float16,
    ):
        """
        Initialize EncoderVADHead.

        Args:
            n_state: Encoder hidden dimension (1280 for large-v3)
            hidden_dim: MLP hidden dimension (256 is sufficient)
            dropout: Dropout rate for training (default: 0.1)
            dtype: Data type for computation
        """
        super().__init__()

        self.n_state = n_state
        self.hidden_dim = hidden_dim
        self._dtype = dtype

        # Simple 2-layer MLP
        # Layer 1: Project to hidden dimension
        self.proj = nn.Linear(n_state, hidden_dim)

        # Layer 2: Project to single logit
        self.classifier = nn.Linear(hidden_dim, 1)

        # Dropout for training
        self.dropout = nn.Dropout(p=dropout)

        # Initialize weights with small values to start near 0.5 probability
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training start."""
        # Small initialization so sigmoid starts near 0.5
        # This prevents the model from being overconfident initially
        scale = 0.02
        self.proj.weight = mx.random.normal(
            shape=self.proj.weight.shape,
            scale=scale,
            dtype=self._dtype,
        )
        self.proj.bias = mx.zeros_like(self.proj.bias)

        self.classifier.weight = mx.random.normal(
            shape=self.classifier.weight.shape,
            scale=scale,
            dtype=self._dtype,
        )
        # Initialize bias slightly negative to start with slight silence preference
        # This is safer for speech recognition (better to include than exclude)
        self.classifier.bias = mx.zeros_like(self.classifier.bias)

    def __call__(
        self,
        encoder_output: mx.array,
        training: bool = False,
    ) -> mx.array:
        """
        Forward pass: predict speech probability per encoder position.

        Args:
            encoder_output: Encoder hidden states (batch, seq_len, n_state)
            training: If True, apply dropout

        Returns:
            Speech probabilities (batch, seq_len), values in [0, 1]
        """
        # Project to hidden dimension
        x = self.proj(encoder_output)  # (batch, seq_len, hidden_dim)
        x = nn.gelu(x)

        # Apply dropout during training
        if training:
            x = self.dropout(x)

        # Classify each position
        logits = self.classifier(x)  # (batch, seq_len, 1)
        logits = logits.squeeze(-1)  # (batch, seq_len)

        # Convert to probability
        return mx.sigmoid(logits)


    def get_logits(
        self,
        encoder_output: mx.array,
        training: bool = False,
    ) -> mx.array:
        """
        Get raw logits (before sigmoid) for BCE loss computation.

        Args:
            encoder_output: Encoder hidden states (batch, seq_len, n_state)
            training: If True, apply dropout

        Returns:
            Logits (batch, seq_len), unbounded real values
        """
        x = self.proj(encoder_output)
        x = nn.gelu(x)

        if training:
            x = self.dropout(x)

        logits = self.classifier(x)
        return logits.squeeze(-1)

    def get_speech_mask(
        self,
        encoder_output: mx.array,
        threshold: float = 0.15,
    ) -> mx.array:
        """
        Get boolean mask for speech positions.

        Args:
            encoder_output: Encoder hidden states (batch, seq_len, n_state)
            threshold: Probability threshold for speech (default: 0.15).
                      Lower values are more conservative (include more frames).
                      The model outputs lower probabilities than Silero VAD,
                      so 0.1-0.2 works better than 0.5.

        Returns:
            Boolean mask (batch, seq_len), True where speech detected
        """
        probs = self(encoder_output, training=False)
        return probs >= threshold


class SileroVADDistiller:
    """
    Distill VAD knowledge from Silero VAD to encoder VAD head.

    Silero VAD operates at sample level (32ms frames at 16kHz).
    Whisper encoder operates at ~20ms per position (30s → 1500 positions).
    This class handles the alignment between these resolutions.

    Usage:
        distiller = SileroVADDistiller()

        # Get Silero VAD labels for audio
        labels = distiller.get_vad_labels(audio)  # (n_encoder_positions,)

        # Use labels to train encoder VAD head
        encoder_output = model.encoder(mel)
        vad_probs = vad_head(encoder_output)
        loss = distiller.compute_loss(vad_probs, labels)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        silero_window_size_samples: int = 512,
        encoder_hop_length: int = 160,  # Whisper's hop length
        encoder_conv_stride: int = 2,   # Whisper encoder conv2 stride
    ):
        """
        Initialize Silero VAD distiller.

        Args:
            sample_rate: Audio sample rate (must be 16000 for Silero)
            silero_window_size_samples: Silero window size (512 = 32ms)
            encoder_hop_length: Whisper mel spectrogram hop length (160)
            encoder_conv_stride: Whisper encoder conv2 stride (2)
        """
        self.sample_rate = sample_rate
        self.silero_window_size = silero_window_size_samples
        self.encoder_hop_length = encoder_hop_length
        self.encoder_conv_stride = encoder_conv_stride

        # Silero VAD model (lazy load)
        self._silero_model = None
        self._silero_utils = None

        # Calculate time resolution
        # Silero: 512 samples / 16000 Hz = 32ms per frame
        self.silero_frame_ms = silero_window_size_samples / sample_rate * 1000

        # Whisper encoder: hop_length * conv_stride / sample_rate
        # = 160 * 2 / 16000 = 20ms per position
        self.encoder_frame_ms = (encoder_hop_length * encoder_conv_stride) / sample_rate * 1000

    def _load_silero(self):
        """Lazy-load Silero VAD model."""
        if self._silero_model is None:
            import torch

            # Load Silero VAD
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False,
            )
            self._silero_model = model
            self._silero_utils = utils

        return self._silero_model, self._silero_utils

    def get_vad_labels(
        self,
        audio: "numpy.ndarray",
        n_encoder_positions: int,
    ) -> mx.array:
        """
        Get VAD labels aligned to encoder positions.

        Args:
            audio: Audio waveform (n_samples,), float32
            n_encoder_positions: Number of encoder output positions

        Returns:
            VAD labels (n_encoder_positions,), values in [0, 1]
        """
        import numpy as np
        import torch

        model, utils = self._load_silero()

        # Convert audio to torch tensor
        audio_tensor = torch.from_numpy(audio).float()
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dim

        # Run Silero VAD to get frame-level probabilities
        # Silero expects (batch, samples) and returns per-frame probs
        with torch.no_grad():
            # Use Silero's streaming interface for consistent behavior
            vad_probs_list = []
            chunk_size = self.silero_window_size

            for i in range(0, audio_tensor.shape[1], chunk_size):
                chunk = audio_tensor[:, i:i+chunk_size]
                if chunk.shape[1] < chunk_size:
                    # Pad last chunk
                    chunk = torch.nn.functional.pad(
                        chunk, (0, chunk_size - chunk.shape[1]),
                    )

                prob = model(chunk, self.sample_rate)
                vad_probs_list.append(prob.item())

            vad_probs = np.array(vad_probs_list, dtype=np.float32)

        # Reset Silero's internal state
        model.reset_states()

        # Align Silero frames to encoder positions
        # Silero has n_silero_frames, encoder has n_encoder_positions
        # Use linear interpolation to align
        n_silero_frames = len(vad_probs)

        if n_silero_frames == 0:
            return mx.zeros((n_encoder_positions,))

        # Interpolate to encoder resolution
        encoder_labels = np.interp(
            np.linspace(0, n_silero_frames - 1, n_encoder_positions),
            np.arange(n_silero_frames),
            vad_probs,
        )

        return mx.array(encoder_labels.astype(np.float32))

    def compute_loss(
        self,
        vad_logits: mx.array,
        labels: mx.array,
        reduction: str = "mean",
    ) -> mx.array:
        """
        Compute binary cross-entropy loss for VAD training.

        Args:
            vad_logits: Predicted logits from VAD head (batch, seq_len)
            labels: Ground truth labels from Silero (batch, seq_len)
            reduction: "mean", "sum", or "none"

        Returns:
            BCE loss value
        """
        # Binary cross-entropy with logits
        # BCE = -[y * log(sigmoid(x)) + (1-y) * log(1-sigmoid(x))]
        # = max(x, 0) - x*y + log(1 + exp(-|x|))

        # Numerically stable BCE
        pos_term = mx.maximum(vad_logits, 0)
        neg_term = vad_logits * labels
        log_term = mx.log(1 + mx.exp(-mx.abs(vad_logits)))

        loss = pos_term - neg_term + log_term

        if reduction == "mean":
            return mx.mean(loss)
        if reduction == "sum":
            return mx.sum(loss)
        return loss


def create_encoder_vad_head(
    n_state: int = 1280,
    hidden_dim: int = 256,
    dtype: mx.Dtype = mx.float16,
) -> EncoderVADHead:
    """
    Create an EncoderVADHead with recommended settings.

    Args:
        n_state: Encoder hidden dimension (1280 for large-v3)
        hidden_dim: MLP hidden dimension (256 recommended)
        dtype: Data type

    Returns:
        Initialized EncoderVADHead
    """
    return EncoderVADHead(
        n_state=n_state,
        hidden_dim=hidden_dim,
        dtype=dtype,
    )


def load_encoder_vad_head(
    weights_path: str,
    n_state: int = 1280,
    hidden_dim: int = 256,
    dtype: mx.Dtype = mx.float16,
) -> EncoderVADHead:
    """
    Load a trained EncoderVADHead from weights file.

    Args:
        weights_path: Path to .npz or .safetensors weights file
        n_state: Encoder hidden dimension
        hidden_dim: MLP hidden dimension
        dtype: Data type

    Returns:
        EncoderVADHead with loaded weights
    """
    from mlx.utils import tree_unflatten

    # Create model
    model = EncoderVADHead(
        n_state=n_state,
        hidden_dim=hidden_dim,
        dtype=dtype,
    )

    # Load weights
    weights = mx.load(weights_path)
    weights_list = [(k, v.astype(dtype)) for k, v in weights.items()]
    nested_weights = tree_unflatten(weights_list)

    model.update(nested_weights)
    mx.eval(model.parameters())

    return model


def save_encoder_vad_head(
    model: EncoderVADHead,
    weights_path: str,
):
    """
    Save EncoderVADHead weights to file.

    Args:
        model: Trained EncoderVADHead
        weights_path: Output path (.npz or .safetensors)
    """
    from mlx.utils import tree_flatten

    # Flatten weights
    flat_weights = dict(tree_flatten(model.parameters()))

    # Save
    if weights_path.endswith('.safetensors'):
        mx.save_safetensors(weights_path, flat_weights)
    else:
        mx.savez(weights_path, **flat_weights)


__all__ = [
    "EncoderVADHead",
    "SileroVADDistiller",
    "create_encoder_vad_head",
    "load_encoder_vad_head",
    "save_encoder_vad_head",
]
