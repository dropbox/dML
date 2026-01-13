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
Kokoro Phoneme Head for Voice-Agnostic Verification.

Predicts Kokoro phoneme tokens directly from Whisper encoder output,
enabling fast verification of transcripts without running the full decoder.

Key insight: Both Whisper encoder and Kokoro bert_enc are voice-agnostic.
By training a small head to predict Kokoro phoneme tokens from Whisper
encoder output, we can compare transcripts in a common phoneme space.

Architecture:
    Audio → Whisper Encoder → Kokoro Phoneme Head → Phoneme Tokens
                                    ↓
    Text → Kokoro Phonemizer → Phoneme Tokens
                                    ↓
                        Compare with Edit Distance

Usage:
    from tools.whisper_mlx.kokoro_phoneme_head import KokoroPhonemeHead

    # Create and load trained head
    head = KokoroPhonemeHead.from_pretrained("checkpoints/kokoro_phoneme_head")

    # Predict phonemes from encoder output (already computed for STT)
    phoneme_tokens = head.predict(encoder_output)

    # Compare with candidate transcript
    confidence = head.compare_with_text(encoder_output, "hello world")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import mlx.core as mx
    import mlx.nn as nn
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None
    nn = None


# Kokoro phoneme vocabulary size (from misaki G2P / espeak-ng IPA)
# This matches N_TOKENS in tools/pytorch_to_mlx/converters/models/kokoro_phonemizer.py
KOKORO_PHONEME_VOCAB_SIZE = 178


@dataclass
class PhonemeHeadConfig:
    """Configuration for Kokoro Phoneme Head."""
    d_model: int = 1280  # Whisper large-v3 encoder dim
    phoneme_vocab: int = KOKORO_PHONEME_VOCAB_SIZE
    blank_id: int = 0  # CTC blank token
    use_layer_norm: bool = True
    hidden_dim: int = 512  # Optional hidden layer
    dropout: float = 0.0  # Dropout rate for regularization (0.1-0.2 recommended)


class KokoroPhonemeHead(nn.Module):
    """
    Predict Kokoro phoneme tokens from Whisper encoder output.

    This is a small CTC head that maps Whisper's 1280-dim encoder
    output to Kokoro's ~178 phoneme token vocabulary.

    Unlike the full CTC head (which predicts 51K Whisper tokens),
    this head has much fewer parameters and trains faster.
    """

    def __init__(
        self,
        d_model: int = 1280,
        phoneme_vocab: int = KOKORO_PHONEME_VOCAB_SIZE,
        blank_id: int = 0,
        use_layer_norm: bool = True,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ):
        """
        Initialize Kokoro Phoneme Head.

        Args:
            d_model: Whisper encoder hidden dimension (1280 for large-v3)
            phoneme_vocab: Kokoro phoneme vocabulary size (~178)
            blank_id: CTC blank token ID
            use_layer_norm: Apply layer norm before projection
            hidden_dim: Optional hidden layer dimension (for deeper head)
            dropout: Dropout rate after hidden layer (0.1-0.2 recommended for regularization)
        """
        super().__init__()

        self.d_model = d_model
        self.phoneme_vocab = phoneme_vocab
        self.blank_id = blank_id
        self._use_layer_norm = use_layer_norm
        self.hidden_dim = hidden_dim
        self._dropout_rate = dropout

        # Layer norm for stability
        if use_layer_norm:
            self.ln = nn.LayerNorm(d_model)

        # Projection layers
        if hidden_dim is not None:
            # Two-layer head for more capacity
            self.hidden = nn.Linear(d_model, hidden_dim)
            # Dropout after hidden layer for regularization
            if dropout > 0:
                self.dropout = nn.Dropout(dropout)
            self.proj = nn.Linear(hidden_dim, phoneme_vocab)
        else:
            # Simple single projection
            self.hidden = None
            self.proj = nn.Linear(d_model, phoneme_vocab)

    def __call__(self, encoder_output: mx.array, training: bool = False) -> mx.array:
        """
        Forward pass: encoder hidden states -> phoneme logits.

        Args:
            encoder_output: (batch, T, d_model) Whisper encoder output
            training: If True, apply dropout for regularization (default: False for inference)

        Returns:
            logits: (batch, T, phoneme_vocab) per-frame phoneme logits
        """
        x = encoder_output

        if self._use_layer_norm:
            x = self.ln(x)

        if self.hidden is not None:
            x = nn.gelu(self.hidden(x))
            # Apply dropout after activation (only during training)
            if training and self._dropout_rate > 0:
                x = self.dropout(x)

        return self.proj(x)

    def predict(
        self,
        encoder_output: mx.array,
        collapse: bool = True,
    ) -> list[int]:
        """
        Predict phoneme tokens from encoder output.

        Args:
            encoder_output: (batch, T, d_model) or (T, d_model)
            collapse: If True, collapse blanks and repeats (standard CTC decode)

        Returns:
            List of phoneme token IDs
        """
        # Add batch dimension if needed
        if encoder_output.ndim == 2:
            encoder_output = encoder_output[None, :]

        logits = self(encoder_output)  # Default training=False for inference
        mx.eval(logits)

        # Greedy decode
        preds = mx.argmax(logits, axis=-1)
        mx.eval(preds)
        preds_np = np.array(preds).squeeze()

        if not collapse:
            return preds_np.tolist()

        # Collapse blanks and repeats
        collapsed = []
        prev = -1
        for t in preds_np:
            if t != self.blank_id and t != prev:
                collapsed.append(int(t))
            prev = t

        return collapsed

    def compare_with_text(
        self,
        encoder_output: mx.array,
        text: str,
        language: str = "en",
    ) -> tuple[float, int]:
        """
        Compare encoder output with a text transcript.

        Args:
            encoder_output: Whisper encoder output
            text: Text transcript to compare
            language: Language for phonemization

        Returns:
            Tuple of (similarity [0,1], edit_distance)
        """
        # Get predicted phonemes from audio
        predicted_tokens = self.predict(encoder_output)

        # Get reference phonemes from text
        try:
            from tools.pytorch_to_mlx.converters.models.kokoro_phonemizer import (
                phonemize_text,
            )
            _, reference_tokens = phonemize_text(text, language=language)
        except Exception as e:
            import warnings
            warnings.warn(f"Phonemization failed for '{text[:50]}...': {e}", stacklevel=2)
            return 0.0, len(predicted_tokens)

        # Compute edit distance
        edit_dist = self._levenshtein(predicted_tokens, reference_tokens)
        max_len = max(len(predicted_tokens), len(reference_tokens), 1)
        similarity = 1.0 - edit_dist / max_len

        return float(similarity), edit_dist

    def _levenshtein(self, s1: list[int], s2: list[int]) -> int:
        """Compute Levenshtein edit distance between two token sequences."""
        if len(s1) < len(s2):
            return self._levenshtein(s2, s1)
        if len(s2) == 0:
            return len(s1)

        prev = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev[j + 1] + 1
                deletions = curr[j] + 1
                substitutions = prev[j] + (c1 != c2)
                curr.append(min(insertions, deletions, substitutions))
            prev = curr

        return prev[-1]

    @classmethod
    def from_pretrained(cls, path: str) -> KokoroPhonemeHead:
        """
        Load pretrained Kokoro Phoneme Head.

        Args:
            path: Path to checkpoint directory or .npz file

        Returns:
            Loaded KokoroPhonemeHead
        """
        path = Path(path)

        # Find weight file
        if path.is_dir():
            weight_files = sorted(path.glob("*.npz"))
            if not weight_files:
                raise FileNotFoundError(f"No .npz files in {path}")
            weight_path = weight_files[-1]  # Latest
        else:
            weight_path = path

        # Load weights and infer config
        weights = dict(mx.load(str(weight_path)))

        # Infer dimensions from weights
        # If hidden layer exists: hidden.weight = (hidden_dim, d_model), proj.weight = (vocab, hidden_dim)
        # If no hidden layer: proj.weight = (vocab, d_model)
        if "hidden.weight" in weights:
            hidden_dim, d_model = weights["hidden.weight"].shape
            phoneme_vocab, _ = weights["proj.weight"].shape
        elif "proj.weight" in weights:
            phoneme_vocab, d_model = weights["proj.weight"].shape
            hidden_dim = None
        else:
            raise ValueError("Cannot find proj.weight in checkpoint")

        # Create model (dropout=0 for inference, old checkpoints have no dropout)
        model = cls(
            d_model=d_model,
            phoneme_vocab=phoneme_vocab,
            hidden_dim=hidden_dim,
            use_layer_norm="ln.weight" in weights,
            dropout=0.0,  # No dropout at inference time
        )

        # Load weights
        if model._use_layer_norm and "ln.weight" in weights:
            model.ln.weight = weights["ln.weight"]
            model.ln.bias = weights["ln.bias"]

        if model.hidden is not None and "hidden.weight" in weights:
            model.hidden.weight = weights["hidden.weight"]
            model.hidden.bias = weights["hidden.bias"]

        model.proj.weight = weights["proj.weight"]
        model.proj.bias = weights["proj.bias"]

        mx.eval(model.parameters())
        return model

    def save(self, path: str) -> None:
        """
        Save model weights.

        Args:
            path: Path to save weights (.npz file)
        """
        weights = {}

        if self._use_layer_norm:
            weights["ln.weight"] = self.ln.weight
            weights["ln.bias"] = self.ln.bias

        if self.hidden is not None:
            weights["hidden.weight"] = self.hidden.weight
            weights["hidden.bias"] = self.hidden.bias

        weights["proj.weight"] = self.proj.weight
        weights["proj.bias"] = self.proj.bias

        mx.savez(path, **weights)


def create_kokoro_phoneme_head(
    model_size: str = "large-v3",
    hidden_dim: int | None = 512,
    dropout: float = 0.0,
) -> KokoroPhonemeHead:
    """
    Factory function to create Kokoro Phoneme Head.

    Args:
        model_size: Whisper model size
        hidden_dim: Hidden layer dimension (None for single-layer)
        dropout: Dropout rate for regularization (0.1-0.2 recommended)

    Returns:
        KokoroPhonemeHead configured for the model
    """
    d_model_map = {
        "tiny": 384,
        "base": 512,
        "small": 768,
        "medium": 1024,
        "large": 1280,
        "large-v2": 1280,
        "large-v3": 1280,
    }

    d_model = d_model_map.get(model_size, 1280)

    return KokoroPhonemeHead(
        d_model=d_model,
        phoneme_vocab=KOKORO_PHONEME_VOCAB_SIZE,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )


# =============================================================================
# Verification API
# =============================================================================


class PhonemeVerifier:
    """
    Fast transcript verification using Kokoro Phoneme Head.

    Compares Whisper encoder output with candidate transcripts
    in Kokoro phoneme space, achieving <10ms verification latency.
    """

    def __init__(
        self,
        head: KokoroPhonemeHead,
        threshold: float = 0.7,
    ):
        """
        Initialize verifier.

        Args:
            head: Trained Kokoro Phoneme Head
            threshold: Similarity threshold for commit decision
        """
        self.head = head
        self.threshold = threshold

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        threshold: float = 0.7,
    ) -> PhonemeVerifier:
        """Load verifier from pretrained head."""
        head = KokoroPhonemeHead.from_pretrained(path)
        return cls(head, threshold)

    def verify(
        self,
        encoder_output: mx.array,
        transcript: str,
        language: str = "en",
    ) -> tuple[bool, float, int]:
        """
        Verify a transcript against encoder output.

        Args:
            encoder_output: Whisper encoder output (already computed for STT)
            transcript: Candidate transcript to verify
            language: Language for phonemization

        Returns:
            Tuple of (should_commit, similarity, edit_distance)
        """
        similarity, edit_dist = self.head.compare_with_text(
            encoder_output, transcript, language,
        )
        should_commit = similarity >= self.threshold
        return should_commit, similarity, edit_dist

    def rank_candidates(
        self,
        encoder_output: mx.array,
        candidates: list[str],
        language: str = "en",
    ) -> list[tuple[str, float, int]]:
        """
        Rank multiple candidate transcripts by similarity.

        Args:
            encoder_output: Whisper encoder output
            candidates: List of candidate transcripts
            language: Language for phonemization

        Returns:
            List of (transcript, similarity, edit_distance) sorted by similarity
        """
        results = []
        for candidate in candidates:
            similarity, edit_dist = self.head.compare_with_text(
                encoder_output, candidate, language,
            )
            results.append((candidate, similarity, edit_dist))

        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results


# =============================================================================
# Quick Test
# =============================================================================


def test_kokoro_phoneme_head():
    """Test the Kokoro Phoneme Head (without trained weights)."""
    print("Testing KokoroPhonemeHead...")

    if not HAS_MLX:
        print("MLX not available, skipping")
        return

    # Create head
    head = create_kokoro_phoneme_head(model_size="large-v3", hidden_dim=512)
    print(f"  Created head: d_model={head.d_model}, vocab={head.phoneme_vocab}")

    # Mock encoder output
    encoder_output = mx.random.normal((1, 100, 1280))

    # Test forward pass
    logits = head(encoder_output)
    mx.eval(logits)
    print(f"  Logits shape: {logits.shape}")
    assert logits.shape == (1, 100, KOKORO_PHONEME_VOCAB_SIZE)

    # Test prediction
    tokens = head.predict(encoder_output)
    print(f"  Predicted tokens: {len(tokens)} (from 100 frames)")

    # Test comparison (with random predictions)
    similarity, edit_dist = head.compare_with_text(encoder_output, "hello world")
    print(f"  Comparison: similarity={similarity:.3f}, edit_dist={edit_dist}")

    print("KokoroPhonemeHead tests PASSED")


if __name__ == "__main__":
    test_kokoro_phoneme_head()
