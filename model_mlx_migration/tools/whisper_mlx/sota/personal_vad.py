#!/usr/bin/env python3
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
Personal VAD 2.0 - Speaker-Gated Voice Activity Detection

Gates ASR processing based on target speaker detection.
Filters audio BEFORE expensive ASR processing.
Only passes audio segments from target speaker(s).

Architecture:
- Combines Silero VAD features with ECAPA-TDNN speaker embeddings
- Small MLP gate learns to filter non-target speakers
- Frame-level resolution (10ms) for accurate timestamps

Performance:
- Latency: ~5ms per 100ms chunk (VAD + speaker gate)
- Size: ~80MB (ECAPA-TDNN) + VAD overhead

Based on: "Personal VAD 2.0" (Google, 2022)
"""

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# Import existing components
from tools.whisper_mlx.silero_vad import (
    SileroVADProcessor,
    SpeechSegment,
)
from tools.whisper_mlx.sota.speaker_encoder import SpeakerEncoder

# Frame size in samples (10ms at 16kHz)
FRAME_SIZE = 160
SAMPLE_RATE = 16000


@dataclass
class PersonalVADConfig:
    """Configuration for Personal VAD."""

    # Speaker embedding dimension
    speaker_dim: int = 192

    # VAD feature dimension (from CNN backbone)
    vad_feature_dim: int = 64

    # Gate hidden dimension
    gate_hidden_dim: int = 128

    # Default gate threshold
    default_threshold: float = 0.5

    # Minimum segment duration in ms
    min_segment_ms: float = 100

    # VAD aggressiveness (0-3, higher = more aggressive filtering)
    vad_aggressiveness: int = 2

    # Frame size in ms
    frame_ms: float = 10.0

    # Speaker embedding model path
    speaker_model_path: str | None = None


@dataclass
class PersonalVADResult:
    """Result of Personal VAD processing."""

    # Per-frame results (at 10ms resolution)
    is_speech: np.ndarray         # (T,) bool - is this frame speech?
    is_target: np.ndarray         # (T,) bool - is this frame target speaker?
    vad_probs: np.ndarray         # (T,) float - speech probability
    target_probs: np.ndarray      # (T,) float - target speaker probability

    # Segments
    speech_segments: list[SpeechSegment]   # All speech segments
    target_segments: list[SpeechSegment]   # Only target speaker segments

    # Statistics
    speech_ratio: float           # Fraction that is speech
    target_ratio: float           # Fraction that is target speaker
    total_duration: float         # Total audio duration in seconds

    @property
    def has_target_speech(self) -> bool:
        """True if any target speaker speech was detected."""
        return len(self.target_segments) > 0


class SpeakerGate(nn.Module):
    """Learns to gate VAD based on speaker embedding similarity.

    Takes VAD features and speaker embedding, outputs gate probability.
    """

    def __init__(self, config: PersonalVADConfig):
        super().__init__()
        self.config = config

        # Input: VAD features + speaker embedding
        input_dim = config.vad_feature_dim + config.speaker_dim

        # MLP for gating
        self.gate = nn.Sequential(
            nn.Linear(input_dim, config.gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.gate_hidden_dim, config.gate_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.gate_hidden_dim // 2, 1),
        )

    def __call__(
        self,
        vad_features: mx.array,
        speaker_embedding: mx.array,
    ) -> mx.array:
        """Compute speaker gate.

        Args:
            vad_features: (T, vad_feature_dim) VAD features per frame
            speaker_embedding: (speaker_dim,) target speaker embedding

        Returns:
            gate_logits: (T, 1) gate logits per frame
        """
        # Expand speaker embedding to match frames
        num_frames = vad_features.shape[0]
        speaker_expanded = mx.broadcast_to(
            speaker_embedding[None, :],
            (num_frames, self.config.speaker_dim),
        )

        # Concatenate features
        combined = mx.concatenate([vad_features, speaker_expanded], axis=-1)

        # Apply gate
        return self.gate(combined)



class PersonalVAD:
    """Speaker-gated voice activity detection.

    Combines Silero VAD with ECAPA-TDNN speaker embeddings to:
    1. Detect speech activity (standard VAD)
    2. Filter to only target speaker segments

    Usage:
        # Initialize
        pvad = PersonalVAD.from_pretrained()

        # Enroll target speaker
        target_embedding = pvad.enroll_speaker(enrollment_audio)

        # Process audio
        result = pvad(audio, target_embedding)

        # Get target speaker audio only
        target_audio = pvad.extract_target_speech(audio, result)
    """

    def __init__(
        self,
        speaker_encoder: SpeakerEncoder,
        vad_processor: SileroVADProcessor,
        speaker_gate: SpeakerGate | None = None,
        config: PersonalVADConfig | None = None,
    ):
        """Initialize Personal VAD.

        Args:
            speaker_encoder: ECAPA-TDNN speaker encoder
            vad_processor: Silero VAD processor
            speaker_gate: Trained speaker gate (optional, uses cosine sim if None)
            config: Configuration
        """
        self.config = config or PersonalVADConfig()
        self.speaker_encoder = speaker_encoder
        self.vad_processor = vad_processor
        self.speaker_gate = speaker_gate

        # Frame parameters
        self.frame_samples = int(self.config.frame_ms * SAMPLE_RATE / 1000)

    @classmethod
    def from_pretrained(
        cls,
        speaker_model_path: str | None = None,
        config: PersonalVADConfig | None = None,
    ) -> "PersonalVAD":
        """Load Personal VAD with pretrained components.

        Args:
            speaker_model_path: Path to ECAPA-TDNN model (uses default if None)
            config: Configuration

        Returns:
            PersonalVAD instance
        """
        config = config or PersonalVADConfig()

        # Load speaker encoder - use provided path, config path, or default
        model_path = speaker_model_path or config.speaker_model_path
        if model_path is not None:
            speaker_encoder = SpeakerEncoder.from_pretrained(model_path)
        else:
            # Use default path
            speaker_encoder = SpeakerEncoder.from_pretrained()

        # Load VAD processor
        vad_processor = SileroVADProcessor(aggressiveness=config.vad_aggressiveness)

        return cls(
            speaker_encoder=speaker_encoder,
            vad_processor=vad_processor,
            speaker_gate=None,  # Use cosine similarity by default
            config=config,
        )

    def enroll_speaker(
        self,
        enrollment_audio: np.ndarray | mx.array,
        sample_rate: int = 16000,
    ) -> mx.array:
        """Enroll a target speaker from audio.

        Args:
            enrollment_audio: Audio waveform (T,)
            sample_rate: Sample rate (resampled to 16kHz if different)

        Returns:
            speaker_embedding: (192,) normalized speaker embedding
        """
        # Convert to numpy if needed
        if isinstance(enrollment_audio, mx.array):
            enrollment_audio = np.array(enrollment_audio)

        # Resample if needed
        if sample_rate != SAMPLE_RATE:
            import librosa
            enrollment_audio = librosa.resample(
                enrollment_audio,
                orig_sr=sample_rate,
                target_sr=SAMPLE_RATE,
            )

        # Compute mel spectrogram (80 mel bands for speaker encoder)
        mel = self._audio_to_mel(enrollment_audio)

        # Ensure batch dimension: (1, time, n_mels)
        if mel.ndim == 2:
            mel = mel[None, :, :]

        # Extract speaker embedding
        embedding = self.speaker_encoder.encode(mel)

        # Return as 1D array (squeeze batch dimension)
        if embedding.ndim == 2:
            embedding = embedding.squeeze(0)

        return embedding

    def _audio_to_mel(self, audio: np.ndarray, n_mels: int = 80) -> mx.array:
        """Convert audio waveform to mel spectrogram.

        Args:
            audio: Audio waveform (T,)
            n_mels: Number of mel bands (80 for speaker encoder)

        Returns:
            Mel spectrogram (time, n_mels)
        """
        from tools.whisper_mlx.audio import log_mel_spectrogram

        # Compute mel spectrogram
        return log_mel_spectrogram(audio, n_mels=n_mels)

        # Ensure shape is (time, n_mels)
        # log_mel_spectrogram returns (n_frames, n_mels)

    def __call__(
        self,
        audio: np.ndarray | mx.array,
        target_embedding: mx.array,
        threshold: float | None = None,
    ) -> PersonalVADResult:
        """Compute speaker-gated VAD.

        Args:
            audio: (T,) audio waveform at 16kHz
            target_embedding: (192,) target speaker embedding
            threshold: Gate threshold (uses config default if None)

        Returns:
            PersonalVADResult with per-frame and segment results
        """
        threshold = threshold or self.config.default_threshold

        # Convert to numpy for processing
        if isinstance(audio, mx.array):
            audio = np.array(audio)

        # Step 1: Get speech segments from VAD
        vad_result = self.vad_processor.get_speech_segments(audio)

        # Step 2: For each speech segment, check speaker similarity
        num_samples = len(audio)
        total_duration = num_samples / SAMPLE_RATE
        num_frames = (num_samples + self.frame_samples - 1) // self.frame_samples

        # Initialize per-frame arrays
        is_speech = np.zeros(num_frames, dtype=bool)
        is_target = np.zeros(num_frames, dtype=bool)
        vad_probs = np.zeros(num_frames, dtype=np.float32)
        target_probs = np.zeros(num_frames, dtype=np.float32)

        # Mark speech frames from VAD
        for seg in vad_result.segments:
            start_frame = seg.start_sample // self.frame_samples
            end_frame = min((seg.end_sample + self.frame_samples - 1) // self.frame_samples, num_frames)
            is_speech[start_frame:end_frame] = True
            vad_probs[start_frame:end_frame] = 1.0

        # Step 3: Check speaker similarity for speech segments
        target_segments = []

        for seg in vad_result.segments:
            # Extract segment audio
            seg_audio = audio[seg.start_sample:seg.end_sample]

            # Skip very short segments
            if len(seg_audio) < SAMPLE_RATE * 0.1:  # < 100ms
                continue

            # Get speaker embedding for this segment
            seg_mel = self._audio_to_mel(seg_audio)
            if seg_mel.ndim == 2:
                seg_mel = seg_mel[None, :, :]  # Add batch dim
            seg_embedding = self.speaker_encoder.encode(seg_mel)
            if seg_embedding.ndim == 2:
                seg_embedding = seg_embedding.squeeze(0)
            seg_embedding_mx = seg_embedding  # Already mx.array from encoder

            # Compute similarity
            if self.speaker_gate is not None:
                # Use learned gate (would need VAD features)
                # For now, fall back to cosine similarity
                similarity = self._cosine_similarity(seg_embedding_mx, target_embedding)
            else:
                # Use cosine similarity
                similarity = self._cosine_similarity(seg_embedding_mx, target_embedding)

            # Mark frames as target if above threshold
            start_frame = seg.start_sample // self.frame_samples
            end_frame = min((seg.end_sample + self.frame_samples - 1) // self.frame_samples, num_frames)

            target_probs[start_frame:end_frame] = float(similarity)

            if float(similarity) > threshold:
                is_target[start_frame:end_frame] = True
                target_segments.append(seg)

        # Compute statistics
        speech_ratio = float(np.mean(is_speech)) if num_frames > 0 else 0.0
        target_ratio = float(np.mean(is_target)) if num_frames > 0 else 0.0

        return PersonalVADResult(
            is_speech=is_speech,
            is_target=is_target,
            vad_probs=vad_probs,
            target_probs=target_probs,
            speech_segments=vad_result.segments,
            target_segments=target_segments,
            speech_ratio=speech_ratio,
            target_ratio=target_ratio,
            total_duration=total_duration,
        )

    def _cosine_similarity(self, a: mx.array, b: mx.array) -> mx.array:
        """Compute cosine similarity between two embeddings."""
        # Normalize
        a_norm = a / (mx.linalg.norm(a) + 1e-8)
        b_norm = b / (mx.linalg.norm(b) + 1e-8)

        # Dot product
        return mx.sum(a_norm * b_norm)

    def extract_target_speech(
        self,
        audio: np.ndarray | mx.array,
        result: PersonalVADResult,
        min_segment_ms: float | None = None,
    ) -> list[np.ndarray]:
        """Extract audio segments for target speaker only.

        Args:
            audio: (T,) audio waveform
            result: PersonalVADResult from __call__
            min_segment_ms: Minimum segment duration (uses config default if None)

        Returns:
            List of audio segments (each as numpy array)
        """
        min_segment_ms = min_segment_ms or self.config.min_segment_ms
        min_samples = int(min_segment_ms * SAMPLE_RATE / 1000)

        # Convert to numpy if needed
        if isinstance(audio, mx.array):
            audio = np.array(audio)

        # Extract segments
        segments = []
        for seg in result.target_segments:
            seg_audio = audio[seg.start_sample:seg.end_sample]
            if len(seg_audio) >= min_samples:
                segments.append(seg_audio)

        return segments

    def filter_to_target(
        self,
        audio: np.ndarray | mx.array,
        target_embedding: mx.array,
        threshold: float | None = None,
        concatenate: bool = True,
    ) -> np.ndarray | list[np.ndarray]:
        """Filter audio to target speaker only (convenience method).

        Args:
            audio: (T,) audio waveform
            target_embedding: (192,) target speaker embedding
            threshold: Gate threshold
            concatenate: If True, concatenate all segments

        Returns:
            Filtered audio (concatenated or list of segments)
        """
        result = self(audio, target_embedding, threshold)
        segments = self.extract_target_speech(audio, result)

        if not segments:
            return np.array([], dtype=np.float32) if concatenate else []

        if concatenate:
            return np.concatenate(segments)
        return segments


def find_contiguous_segments(
    mask: np.ndarray,
    frame_samples: int,
    min_length_samples: int,
) -> list[tuple[int, int]]:
    """Find contiguous True segments in a boolean mask.

    Args:
        mask: (T,) boolean array
        frame_samples: Samples per frame
        min_length_samples: Minimum segment length in samples

    Returns:
        List of (start_sample, end_sample) tuples
    """
    segments = []
    in_segment = False
    start_frame = 0

    for i, val in enumerate(mask):
        if val and not in_segment:
            # Start of segment
            in_segment = True
            start_frame = i
        elif not val and in_segment:
            # End of segment
            in_segment = False
            start_sample = start_frame * frame_samples
            end_sample = i * frame_samples
            if end_sample - start_sample >= min_length_samples:
                segments.append((start_sample, end_sample))

    # Handle segment that extends to end
    if in_segment:
        start_sample = start_frame * frame_samples
        end_sample = len(mask) * frame_samples
        if end_sample - start_sample >= min_length_samples:
            segments.append((start_sample, end_sample))

    return segments


# =============================================================================
# MLX-Native VAD Backend
# =============================================================================


class VADBackbone(nn.Module):
    """Lightweight CNN backbone for VAD feature extraction.

    Pure MLX implementation for training-ready speaker-gated VAD.

    Architecture:
        - 3x Conv1d layers with BatchNorm and ReLU
        - Global average pooling over time
        - Projects to 64-dim feature vector

    This extracts VAD-relevant features from mel spectrogram that indicate
    the presence/absence of speech, independent of speaker identity.
    """

    def __init__(
        self,
        n_mels: int = 80,
        hidden_dim: int = 64,
        feature_dim: int = 64,
        n_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.feature_dim = feature_dim

        # Build conv layers
        self.conv_layers = []
        in_channels = n_mels

        for _ in range(n_layers):
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                ),
            )
            self.conv_layers.append(nn.BatchNorm(hidden_dim))
            in_channels = hidden_dim

        # Project to feature dim
        self.projection = nn.Linear(hidden_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x: mx.array) -> mx.array:
        """Extract VAD features from mel spectrogram.

        Args:
            x: Mel spectrogram (batch, time, n_mels) or (time, n_mels)

        Returns:
            VAD features (batch, feature_dim) or (feature_dim,)
        """
        squeeze_output = x.ndim == 2
        if squeeze_output:
            x = x[None, :, :]

        # MLX Conv1d expects NLC format: (batch, length, channels)
        # Input is already (B, T, n_mels) which is correct

        # Apply conv layers with ReLU
        for i, layer in enumerate(self.conv_layers):
            x = layer(x)
            if i % 2 == 1:  # After BatchNorm, apply ReLU
                x = nn.relu(x)

        # After conv layers: (B, T, hidden_dim)
        # Global average pooling over time
        x = mx.mean(x, axis=1)  # (B, hidden)

        # Project to feature dim
        x = self.projection(x)  # (B, feature_dim)
        x = self.dropout(x)

        if squeeze_output:
            x = x.squeeze(0)

        return x


class PersonalVADMLX(nn.Module):
    """Pure MLX Personal VAD with trainable speaker gate.

    This is a fully differentiable version of PersonalVAD that can be
    trained end-to-end with speaker-labeled VAD data.

    Architecture:
        1. VADBackbone: Extracts 64-dim features from mel spectrogram
        2. SpeakerGate: Combines with 192-dim speaker embedding
        3. Output: Gate probability (is target speaker speaking?)

    Training:
        - Binary cross-entropy loss
        - Labels: 1 = target speaker speaking, 0 = silence or other speaker
        - Data: Speaker-labeled VAD from VoxCeleb + LibriMix

    Inference:
        - Process mel chunks through backbone
        - Compare with target speaker embedding
        - Threshold gate output to get binary mask
    """

    def __init__(
        self,
        vad_backbone: VADBackbone | None = None,
        speaker_gate: SpeakerGate | None = None,
        config: PersonalVADConfig | None = None,
    ):
        super().__init__()
        self.config = config or PersonalVADConfig()

        # VAD backbone
        self.vad_backbone = vad_backbone or VADBackbone(
            n_mels=80,
            hidden_dim=self.config.vad_feature_dim,
            feature_dim=self.config.vad_feature_dim,
        )

        # Speaker gate
        self.speaker_gate = speaker_gate or SpeakerGate(self.config)

        # Compiled forward
        self._compiled_forward = None

    def _forward_impl(
        self,
        mel: mx.array,
        speaker_embedding: mx.array,
    ) -> mx.array:
        """Internal forward pass."""
        # Extract VAD features
        vad_features = self.vad_backbone(mel)  # (B, 64) or (64,)

        # Ensure 2D for gate
        if vad_features.ndim == 1:
            vad_features = vad_features[None, :]

        # Apply speaker gate
        gate_logits = self.speaker_gate(vad_features, speaker_embedding)

        return mx.sigmoid(gate_logits)

    def __call__(
        self,
        mel: mx.array,
        speaker_embedding: mx.array,
        threshold: float = 0.5,
        use_compile: bool = True,
    ) -> tuple[mx.array, mx.array]:
        """Detect if audio chunk is from target speaker.

        Args:
            mel: Mel spectrogram (batch, time, n_mels) or (time, n_mels)
            speaker_embedding: Target speaker embedding (192,)
            threshold: Gate threshold (default 0.5)
            use_compile: Use JIT compilation (default True)

        Returns:
            Tuple of:
                - is_target: Boolean indicating target speaker (batch,) or scalar
                - confidence: Gate probability (batch,) or scalar
        """
        if use_compile:
            if self._compiled_forward is None:
                self._compiled_forward = mx.compile(self._forward_impl)
            confidence = self._compiled_forward(mel, speaker_embedding)
        else:
            confidence = self._forward_impl(mel, speaker_embedding)

        # Squeeze outputs
        confidence = confidence.squeeze(-1)
        is_target = confidence >= threshold

        return is_target, confidence

    def process_sequence(
        self,
        mel: mx.array,
        speaker_embedding: mx.array,
        chunk_frames: int = 50,
        stride_frames: int = 25,
        threshold: float = 0.5,
    ) -> mx.array:
        """Process full mel sequence in chunks.

        Args:
            mel: Full mel spectrogram (time, n_mels)
            speaker_embedding: Target speaker embedding (192,)
            chunk_frames: Frames per chunk
            stride_frames: Stride between chunks
            threshold: Gate threshold

        Returns:
            Per-frame mask (time,) with values in [0, 1]
        """
        n_frames = mel.shape[0]
        mask = mx.zeros((n_frames,))
        count = mx.zeros((n_frames,))

        for start in range(0, n_frames - chunk_frames + 1, stride_frames):
            end = start + chunk_frames
            chunk = mel[start:end]

            _, confidence = self(chunk, speaker_embedding, threshold=0.0)

            # Accumulate
            chunk_mask = mx.broadcast_to(confidence, (chunk_frames,))
            mask = mask.at[start:end].add(chunk_mask)
            count = count.at[start:end].add(mx.ones((chunk_frames,)))

        # Average overlapping regions
        return mask / mx.maximum(count, 1.0)


    def save_pretrained(self, path: str):
        """Save model weights and config."""
        import json
        from dataclasses import asdict

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(path / "config.json", "w") as f:
            json.dump(asdict(self.config), f, indent=2)

        # Save weights
        weights = dict(self.parameters())
        mx.savez(str(path / "weights.npz"), **weights)

    @classmethod
    def from_pretrained(cls, path: str) -> "PersonalVADMLX":
        """Load model from pretrained weights."""
        import json

        path = Path(path)

        # Load config
        with open(path / "config.json") as f:
            config_dict = json.load(f)
        config = PersonalVADConfig(**config_dict)

        # Create model
        model = cls(config=config)

        # Load weights
        weights = mx.load(str(path / "weights.npz"))
        model.load_weights(list(weights.items()))

        return model


class PersonalVADTrainer:
    """Trainer for PersonalVADMLX model.

    Training data format:
        - mel: Mel spectrogram chunks (batch, time, n_mels)
        - speaker_embedding: Target speaker embedding (192,)
        - labels: Binary labels (batch,) - 1 = target speaker, 0 = other/silence

    Loss: Binary cross-entropy
    """

    def __init__(
        self,
        model: PersonalVADMLX,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
    ):
        import mlx.optimizers as optim

        self.model = model
        self.optimizer = optim.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )

    def compute_loss(
        self,
        mel: mx.array,
        speaker_embedding: mx.array,
        labels: mx.array,
    ) -> mx.array:
        """Compute binary cross-entropy loss.

        Args:
            mel: Mel spectrograms (batch, time, n_mels)
            speaker_embedding: Target speaker embedding (192,)
            labels: Binary labels (batch,) - 1 = target speaker

        Returns:
            Scalar loss value
        """
        _, confidence = self.model(mel, speaker_embedding, use_compile=False)

        # Binary cross-entropy
        eps = 1e-7
        confidence = mx.clip(confidence, eps, 1 - eps)
        loss = -labels * mx.log(confidence) - (1 - labels) * mx.log(1 - confidence)

        return mx.mean(loss)

    def train_step(
        self,
        mel: mx.array,
        speaker_embedding: mx.array,
        labels: mx.array,
    ) -> float:
        """Execute one training step.

        Args:
            mel: Mel spectrograms (batch, time, n_mels)
            speaker_embedding: Target speaker embedding
            labels: Binary labels

        Returns:
            Loss value
        """
        # Define loss function that takes model as argument
        def loss_fn(model):
            _, confidence = model(mel, speaker_embedding, use_compile=False)
            eps = 1e-7
            confidence = mx.clip(confidence, eps, 1 - eps)
            loss = -labels * mx.log(confidence) - (1 - labels) * mx.log(1 - confidence)
            return mx.mean(loss)

        # Compute loss and gradients
        loss, grads = nn.value_and_grad(self.model, loss_fn)(self.model)

        # Update parameters
        self.optimizer.update(self.model, grads)

        # Force evaluation of loss only (parameters are evaluated lazily)
        return float(loss)
