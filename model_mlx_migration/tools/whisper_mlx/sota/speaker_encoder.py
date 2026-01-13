"""Speaker Encoder for Phase 10 Speaker Adaptation.

Provides speaker embedding extraction using ECAPA-TDNN trained on VoxCeleb.
Used for:
- Speaker Query Attention in encoder conditioning
- Personal VAD (speaker-gated voice activity detection)
- Speaker identification and tracking

Reference: ECAPA-TDNN arXiv:2005.07143
Checkpoint: speechbrain/spkrec-ecapa-voxceleb
"""

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .ecapa_config import ECAPATDNNConfig
from .ecapa_tdnn import ECAPATDNN


class SpeakerEncoder(nn.Module):
    """Speaker Encoder for extracting speaker embeddings.

    This is a wrapper around ECAPA-TDNN optimized for speaker verification.
    It extracts 192-dimensional speaker embeddings from audio.

    Key features:
    - Trained on VoxCeleb1+2 for speaker verification
    - 192-dim embeddings (L2-normalized for cosine similarity)
    - Can process variable-length audio
    - Supports batch processing

    Usage:
        encoder = SpeakerEncoder.from_pretrained("models/ecapa-spkver-mlx")
        embedding = encoder.encode(mel_features)  # (B, 192)

    For speaker similarity:
        sim = mx.sum(emb1 * emb2)  # Cosine similarity (embeddings are normalized)
    """

    def __init__(self, config: ECAPATDNNConfig | None = None):
        super().__init__()
        self.config = config or ECAPATDNNConfig.voxceleb_speaker()
        self.embedding_model = ECAPATDNN(self.config)
        self._compiled_encode = None

    def __call__(
        self,
        x: mx.array,
        normalize: bool = True,
    ) -> mx.array:
        """Extract speaker embedding from mel features.

        Args:
            x: Mel filterbank features (batch, time, n_mels) or (batch, n_mels, time)
            normalize: If True, L2-normalize the output (default True)

        Returns:
            Speaker embeddings of shape (batch, 192)
        """
        return self.encode(x, normalize=normalize)

    def encode(
        self,
        x: mx.array,
        normalize: bool = True,
    ) -> mx.array:
        """Extract speaker embedding from mel features.

        Args:
            x: Mel filterbank features (batch, time, n_mels) or (batch, n_mels, time)
            normalize: If True, L2-normalize the output (default True)

        Returns:
            Speaker embeddings of shape (batch, 192)
        """
        # Get embeddings from model
        emb = self.embedding_model(x)  # (B, 1, 192)
        emb = emb.squeeze(1)  # (B, 192)

        if normalize:
            # L2 normalize for cosine similarity
            emb = emb / (mx.linalg.norm(emb, axis=-1, keepdims=True) + 1e-8)

        return emb

    def encode_compiled(
        self,
        x: mx.array,
        normalize: bool = True,
    ) -> mx.array:
        """Compiled version of encode for faster inference."""
        if self._compiled_encode is None:
            def _encode_impl(x, normalize):
                emb = self.embedding_model(x).squeeze(1)
                if normalize:
                    emb = emb / (mx.linalg.norm(emb, axis=-1, keepdims=True) + 1e-8)
                return emb
            self._compiled_encode = mx.compile(_encode_impl)

        return self._compiled_encode(x, normalize)

    def similarity(
        self,
        emb1: mx.array,
        emb2: mx.array,
    ) -> mx.array:
        """Compute cosine similarity between speaker embeddings.

        Args:
            emb1: Speaker embedding(s) (B, 192) or (192,)
            emb2: Speaker embedding(s) (B, 192) or (192,)

        Returns:
            Similarity scores in [-1, 1]
        """
        # Ensure 2D
        if emb1.ndim == 1:
            emb1 = emb1[None, :]
        if emb2.ndim == 1:
            emb2 = emb2[None, :]

        # Cosine similarity (embeddings should already be normalized)
        return mx.sum(emb1 * emb2, axis=-1)

    def is_same_speaker(
        self,
        emb1: mx.array,
        emb2: mx.array,
        threshold: float = 0.7,
    ) -> mx.array:
        """Check if two embeddings are from the same speaker.

        Args:
            emb1: Speaker embedding(s)
            emb2: Speaker embedding(s)
            threshold: Similarity threshold (default 0.7)

        Returns:
            Boolean array indicating same speaker
        """
        sim = self.similarity(emb1, emb2)
        return sim > threshold

    @classmethod
    def from_pretrained(
        cls,
        path: str = "models/ecapa-spkver-mlx",
    ) -> "SpeakerEncoder":
        """Load speaker encoder from pretrained weights.

        Args:
            path: Path to directory containing weights.npz and config.json

        Returns:
            Loaded SpeakerEncoder model
        """
        path = Path(path)

        # Load config
        config_path = path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config_dict = json.load(f)
            config = ECAPATDNNConfig(
                n_mels=config_dict.get("n_mels", 80),
                lin_neurons=config_dict.get("lin_neurons", 192),
                num_languages=1,  # Not used
            )
        else:
            config = ECAPATDNNConfig.voxceleb_speaker()

        # Create model
        model = cls(config)

        # Load weights
        weights_path = path / "weights.npz"
        if weights_path.exists():
            weights = mx.load(str(weights_path))

            # Filter to embedding_model weights
            emb_weights = {
                k.replace("embedding_model.", ""): v
                for k, v in weights.items()
                if k.startswith("embedding_model.")
            }
            model.embedding_model.load_weights(list(emb_weights.items()))
        else:
            raise FileNotFoundError(f"Weights not found at {weights_path}")

        return model

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension (192)."""
        return self.config.lin_neurons


class SpeakerDatabase:
    """Simple speaker database for speaker tracking.

    Maintains a set of known speakers with their embeddings.
    Used for:
    - Speaker identification in multi-speaker scenarios
    - Voice focus (prioritizing specific speakers)
    - Personal VAD (detecting target speaker activity)
    """

    def __init__(
        self,
        similarity_threshold: float = 0.7,
        ema_decay: float = 0.9,
    ):
        """Initialize speaker database.

        Args:
            similarity_threshold: Threshold for same-speaker match
            ema_decay: Exponential moving average decay for updating embeddings
        """
        self.threshold = similarity_threshold
        self.ema_decay = ema_decay
        self.embeddings: dict[int, mx.array] = {}
        self.next_id = 0

    def add_speaker(
        self,
        embedding: mx.array,
        speaker_id: int | None = None,
    ) -> int:
        """Add a new speaker to the database.

        Args:
            embedding: Speaker embedding (192,)
            speaker_id: Optional specific ID (auto-assigned if None)

        Returns:
            Speaker ID
        """
        # Normalize
        embedding = embedding / (mx.linalg.norm(embedding) + 1e-8)

        if speaker_id is None:
            speaker_id = self.next_id
            self.next_id += 1

        self.embeddings[speaker_id] = embedding
        return speaker_id

    def identify(
        self,
        embedding: mx.array,
        update: bool = True,
    ) -> tuple[int, float]:
        """Identify speaker from embedding.

        Args:
            embedding: Query embedding (192,)
            update: If True, update the matched speaker's embedding with EMA

        Returns:
            Tuple of (speaker_id, similarity_score)
            Returns (-1, 0.0) if no match above threshold
        """
        if not self.embeddings:
            return -1, 0.0

        # Normalize query
        embedding = embedding / (mx.linalg.norm(embedding) + 1e-8)

        best_id = -1
        best_sim = 0.0

        for spk_id, spk_emb in self.embeddings.items():
            sim = float(mx.sum(embedding * spk_emb))
            if sim > best_sim:
                best_sim = sim
                best_id = spk_id

        if best_sim >= self.threshold:
            if update:
                # Update with EMA
                old_emb = self.embeddings[best_id]
                new_emb = self.ema_decay * old_emb + (1 - self.ema_decay) * embedding
                new_emb = new_emb / (mx.linalg.norm(new_emb) + 1e-8)
                self.embeddings[best_id] = new_emb
            return best_id, best_sim

        return -1, best_sim

    def identify_or_add(
        self,
        embedding: mx.array,
    ) -> tuple[int, float, bool]:
        """Identify speaker or add new one if not found.

        Args:
            embedding: Query embedding (192,)

        Returns:
            Tuple of (speaker_id, similarity, is_new)
        """
        spk_id, sim = self.identify(embedding)
        if spk_id >= 0:
            return spk_id, sim, False

        # Add as new speaker
        new_id = self.add_speaker(embedding)
        return new_id, 1.0, True

    def clear(self):
        """Clear all speakers."""
        self.embeddings.clear()
        self.next_id = 0

    def __len__(self) -> int:
        return len(self.embeddings)
