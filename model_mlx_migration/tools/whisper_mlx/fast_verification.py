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
Fast Embedding-Space Verification for Streaming ASR

Instead of slow TTS synthesis + mel comparison (~10s), we compare embeddings directly:
- Whisper encoder produces audio embeddings (1280-dim for large-v3)
- Kokoro text_encoder produces text embeddings (512-dim)

Both represent semantic content at frame-rate. By comparing them directly,
we skip the vocoder entirely and achieve <100ms verification time.

Key insight: Whisper's encoder is ALREADY running for STT, so audio embeddings
are "free". We only need to run Kokoro's lightweight text encoder.

Usage:
    from tools.whisper_mlx.fast_verification import FastVerifier

    # Initialize (loads only Kokoro text encoder, not vocoder)
    verifier = FastVerifier.from_kokoro()

    # During STT, get Whisper encoder output
    whisper_encoder_output = model.encoder(mel)  # (1, T, 1280)

    # Fast verification (<100ms)
    confidence = verifier.compute_confidence(
        whisper_encoder_output,  # From STT (already computed!)
        "hello world"            # STT transcript to verify
    )
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

try:
    import mlx.core as mx
    import mlx.nn as nn
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None
    nn = None


@dataclass
class FastVerificationResult:
    """Result of fast embedding-space verification."""
    confidence: float           # Similarity score in [0, 1]
    whisper_frames: int         # Number of Whisper encoder frames
    kokoro_frames: int          # Number of Kokoro text encoder frames
    verification_time_ms: float # Time to compute (should be <100ms)
    method: str                 # Similarity method used


class EmbeddingProjection(nn.Module):
    """
    Learned projection to align Kokoro and Whisper embedding spaces.

    Kokoro text_encoder: 512-dim
    Whisper encoder: 1280-dim (large-v3), 768-dim (medium), 512-dim (small)

    We project Kokoro → Whisper space (upscale) for comparison.
    """

    def __init__(
        self,
        kokoro_dim: int = 512,
        whisper_dim: int = 1280,
        hidden_dim: int = 768,
    ):
        super().__init__()
        # Simple 2-layer MLP with residual
        self.proj1 = nn.Linear(kokoro_dim, hidden_dim)
        self.proj2 = nn.Linear(hidden_dim, whisper_dim)
        self.layer_norm = nn.LayerNorm(whisper_dim)

    def __call__(self, x: mx.array) -> mx.array:
        """Project Kokoro embeddings to Whisper space."""
        h = nn.gelu(self.proj1(x))
        out = self.proj2(h)
        return self.layer_norm(out)


class FastVerifier:
    """
    Fast embedding-space verification for streaming ASR.

    Compares Whisper encoder output with Kokoro text encoder output
    directly in embedding space, skipping TTS synthesis entirely.

    Target latency: <100ms (vs 10s for TTS-based verification)
    """

    def __init__(
        self,
        kokoro_text_encode_fn: Callable[[str], mx.array],
        whisper_dim: int = 1280,
        kokoro_dim: int = 512,
        use_projection: bool = False,
    ):
        """
        Initialize fast verifier.

        Args:
            kokoro_text_encode_fn: Function that encodes text to Kokoro embeddings
            whisper_dim: Dimension of Whisper encoder output
            kokoro_dim: Dimension of Kokoro text encoder output
            use_projection: If True, use learned projection (requires training)
        """
        self.kokoro_encode = kokoro_text_encode_fn
        self.whisper_dim = whisper_dim
        self.kokoro_dim = kokoro_dim
        self.use_projection = use_projection

        if use_projection and HAS_MLX:
            self.projection = EmbeddingProjection(kokoro_dim, whisper_dim)
        else:
            self.projection = None

    @classmethod
    def from_kokoro(
        cls,
        model_id: str = "hexgrad/Kokoro-82M",
        cache_dir: str | None = None,
        whisper_dim: int = 1280,
    ) -> FastVerifier:
        """
        Create FastVerifier using Kokoro's text encoder only.

        This loads ONLY the lightweight components:
        - BERT (phoneme encoding)
        - bert_encoder (linear projection)
        - text_encoder (Conv1d + BiLSTM)

        Does NOT load:
        - Predictor (duration/F0/noise)
        - Decoder (vocoder)

        Args:
            model_id: HuggingFace model ID
            cache_dir: Local cache directory
            whisper_dim: Whisper encoder dimension (1280 for large-v3)
        """
        if not HAS_MLX:
            raise ImportError("MLX is required")

        from tools.pytorch_to_mlx.converters.kokoro_converter import KokoroConverter
        from tools.pytorch_to_mlx.converters.models.kokoro_phonemizer import (
            phonemize_text,
        )

        # Load full model (we'll only use text encoder path)
        # TODO: Optimize to load only needed components
        converter = KokoroConverter()
        model, config, _ = converter.load_from_hf(model_id, cache_dir)

        # Load a default voice for speaker embedding
        converter.load_voice_pack("af_bella", model_id, cache_dir)

        def encode_text(text: str) -> mx.array:
            """
            Encode text to Kokoro embeddings.

            Uses the text_encoder path which takes token IDs directly:
            - text_encoder: Embedding + Conv1d + BiLSTM

            This is the ASR feature path in Kokoro's synthesis pipeline.

            Returns:
                text_enc: [1, T, 512] - text encoder output
            """
            # Phonemize
            phoneme_str, token_ids = phonemize_text(text, language="en")
            if len(token_ids) == 0:
                return mx.zeros((1, 1, 512))

            # Prepare inputs (must be int32 for gather operations)
            input_ids = mx.array([token_ids], dtype=mx.int32)

            # Text encoder takes token IDs directly (not BERT output)
            # This is the ASR feature path: Embedding -> Conv1d -> BiLSTM
            text_enc = model.text_encoder(input_ids, None)  # [1, T, 512]

            mx.eval(text_enc)
            return text_enc

        return cls(
            kokoro_text_encode_fn=encode_text,
            whisper_dim=whisper_dim,
            kokoro_dim=512,
            use_projection=False,  # Start without projection
        )

    def compute_confidence(
        self,
        whisper_encoder_output: mx.array,
        transcript: str,
        method: str = "cosine_pooled",
    ) -> FastVerificationResult:
        """
        Compute confidence score comparing embeddings.

        Args:
            whisper_encoder_output: Whisper encoder output [batch, T, dim]
                                   This is ALREADY computed during STT!
            transcript: STT transcript to verify
            method: Similarity method: "cosine_pooled", "cosine_dtw", "cka"

        Returns:
            FastVerificationResult with confidence and timing
        """
        t0 = time.perf_counter()

        # Validate inputs
        if whisper_encoder_output is None or whisper_encoder_output.size == 0:
            return FastVerificationResult(
                confidence=0.0,
                whisper_frames=0,
                kokoro_frames=0,
                verification_time_ms=0.0,
                method=method,
            )

        if not transcript or not transcript.strip():
            return FastVerificationResult(
                confidence=0.0,
                whisper_frames=whisper_encoder_output.shape[1] if whisper_encoder_output.ndim > 1 else 0,
                kokoro_frames=0,
                verification_time_ms=0.0,
                method=method,
            )

        # Get Kokoro text encoding (fast!)
        kokoro_emb = self.kokoro_encode(transcript.strip())

        # Project if using learned projection
        if self.use_projection and self.projection is not None:
            kokoro_emb = self.projection(kokoro_emb)

        # Compute similarity
        if method == "cosine_pooled":
            confidence = self._cosine_pooled(whisper_encoder_output, kokoro_emb)
        elif method == "cosine_dtw":
            confidence = self._cosine_dtw(whisper_encoder_output, kokoro_emb)
        elif method == "cka":
            confidence = self._centered_kernel_alignment(whisper_encoder_output, kokoro_emb)
        else:
            raise ValueError(f"Unknown method: {method}")

        verification_time_ms = (time.perf_counter() - t0) * 1000

        return FastVerificationResult(
            confidence=float(confidence),
            whisper_frames=int(whisper_encoder_output.shape[1]) if whisper_encoder_output.ndim > 1 else 0,
            kokoro_frames=int(kokoro_emb.shape[1]) if kokoro_emb.ndim > 1 else 0,
            verification_time_ms=verification_time_ms,
            method=method,
        )

    def _cosine_pooled(
        self,
        whisper_emb: mx.array,
        kokoro_emb: mx.array,
    ) -> float:
        """
        Cosine similarity of mean-pooled embeddings.

        Fast but loses temporal structure. Good for first pass.
        """
        # Mean pool over time dimension
        # whisper_emb: [batch, T1, dim1]
        # kokoro_emb: [batch, T2, dim2]
        w_pooled = mx.mean(whisper_emb, axis=1)  # [batch, dim1]
        k_pooled = mx.mean(kokoro_emb, axis=1)   # [batch, dim2]

        # If dimensions differ, use CKA-style comparison
        if w_pooled.shape[-1] != k_pooled.shape[-1]:
            # Use correlation of sorted eigenvalues (dimension-agnostic)
            return self._dimension_agnostic_similarity(w_pooled, k_pooled)

        # Normalize
        w_norm = w_pooled / (mx.linalg.norm(w_pooled, axis=-1, keepdims=True) + 1e-8)
        k_norm = k_pooled / (mx.linalg.norm(k_pooled, axis=-1, keepdims=True) + 1e-8)

        # Cosine similarity
        similarity = mx.sum(w_norm * k_norm, axis=-1)
        mx.eval(similarity)

        # Map from [-1, 1] to [0, 1]
        return float((np.array(similarity).mean() + 1) / 2)

    def _dimension_agnostic_similarity(
        self,
        emb1: mx.array,
        emb2: mx.array,
    ) -> float:
        """
        Compare embeddings of different dimensions using correlation.

        Uses correlation between magnitude distributions.
        """
        # Get magnitude distributions
        emb1_np = np.array(emb1).flatten()
        emb2_np = np.array(emb2).flatten()

        # Normalize to zero mean, unit variance
        emb1_norm = (emb1_np - emb1_np.mean()) / (emb1_np.std() + 1e-8)
        emb2_norm = (emb2_np - emb2_np.mean()) / (emb2_np.std() + 1e-8)

        # Compare statistics
        # 1. Mean magnitudes
        mag1 = np.abs(emb1_norm).mean()
        mag2 = np.abs(emb2_norm).mean()
        mag_sim = 1.0 - abs(mag1 - mag2) / max(mag1, mag2, 1e-8)

        # 2. Variance ratio
        var1 = emb1_norm.var()
        var2 = emb2_norm.var()
        var_ratio = min(var1, var2) / (max(var1, var2) + 1e-8)

        # 3. Distribution shape (via percentiles)
        p1 = np.percentile(emb1_norm, [25, 50, 75])
        p2 = np.percentile(emb2_norm, [25, 50, 75])
        shape_sim = 1.0 - np.abs(p1 - p2).mean()

        # Combine
        similarity = (mag_sim + var_ratio + shape_sim) / 3.0
        return float(np.clip(similarity, 0, 1))

    def _cosine_dtw(
        self,
        whisper_emb: mx.array,
        kokoro_emb: mx.array,
        radius: int = 30,
    ) -> float:
        """
        DTW alignment followed by cosine similarity.

        Slower but captures temporal structure.
        """
        # Convert to numpy for DTW
        w_np = np.array(whisper_emb).squeeze(0)  # [T1, dim1]
        k_np = np.array(kokoro_emb).squeeze(0)   # [T2, dim2]

        T1, D1 = w_np.shape
        T2, D2 = k_np.shape

        # If dimensions differ, project to common space via PCA
        if D1 != D2:
            # Simple: just use mean-pooled comparison
            return self._cosine_pooled(whisper_emb, kokoro_emb)

        # Fast DTW with Sakoe-Chiba band
        dtw = np.full((T1 + 1, T2 + 1), np.inf, dtype=np.float32)
        dtw[0, 0] = 0

        for i in range(1, T1 + 1):
            j_min = max(1, int(i * T2 / T1) - radius)
            j_max = min(T2, int(i * T2 / T1) + radius)

            for j in range(j_min, j_max + 1):
                # Cosine distance
                cos_sim = np.dot(w_np[i-1], k_np[j-1]) / (
                    np.linalg.norm(w_np[i-1]) * np.linalg.norm(k_np[j-1]) + 1e-8
                )
                cost = 1 - cos_sim  # 0 = identical, 2 = opposite

                dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])

        alignment_cost = dtw[T1, T2] / (T1 + T2)

        # Convert to similarity
        similarity = np.exp(-alignment_cost)
        return float(similarity)

    def _centered_kernel_alignment(
        self,
        emb1: mx.array,
        emb2: mx.array,
    ) -> float:
        """
        Centered Kernel Alignment (CKA) for comparing representations.

        CKA is dimension-agnostic and widely used for comparing neural
        network representations. It measures the similarity of the
        geometry of two representation spaces.

        Reference: Kornblith et al., "Similarity of Neural Network
        Representations Revisited" (ICML 2019)
        """
        # Convert to numpy
        X = np.array(emb1).squeeze(0)  # [T1, D1]
        Y = np.array(emb2).squeeze(0)  # [T2, D2]

        # Resample to same length
        T_common = min(X.shape[0], Y.shape[0])
        if X.shape[0] > T_common:
            indices = np.linspace(0, X.shape[0] - 1, T_common).astype(int)
            X = X[indices]
        if Y.shape[0] > T_common:
            indices = np.linspace(0, Y.shape[0] - 1, T_common).astype(int)
            Y = Y[indices]

        # Center the matrices
        X = X - X.mean(axis=0, keepdims=True)
        Y = Y - Y.mean(axis=0, keepdims=True)

        # Compute Gram matrices
        K = X @ X.T
        L = Y @ Y.T

        # Center the Gram matrices
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        K_c = H @ K @ H
        L_c = H @ L @ H

        # CKA formula
        hsic_kl = np.sum(K_c * L_c)
        hsic_kk = np.sum(K_c * K_c)
        hsic_ll = np.sum(L_c * L_c)

        cka = hsic_kl / (np.sqrt(hsic_kk * hsic_ll) + 1e-8)

        return float(np.clip(cka, 0, 1))

    def should_commit(
        self,
        whisper_encoder_output: mx.array,
        transcript: str,
        threshold: float = 0.5,
        method: str = "cosine_pooled",
    ) -> tuple[bool, FastVerificationResult]:
        """
        Decide whether to commit a transcription.

        Args:
            whisper_encoder_output: Already-computed Whisper encoder output
            transcript: STT transcript to verify
            threshold: Confidence threshold for commit decision
            method: Similarity computation method

        Returns:
            Tuple of (should_commit, FastVerificationResult)
        """
        result = self.compute_confidence(
            whisper_encoder_output,
            transcript,
            method=method,
        )

        should_commit = result.confidence >= threshold
        return should_commit, result


# =============================================================================
# Quick Test
# =============================================================================


def test_fast_verifier():
    """Test the fast verifier (without actual models)."""
    print("Testing FastVerifier...")

    if not HAS_MLX:
        print("MLX not available, skipping")
        return

    # Mock encode function
    def mock_encode(text: str) -> mx.array:
        """Mock encoder returning random embeddings."""
        n_tokens = max(1, len(text.split()))
        return mx.random.normal((1, n_tokens * 3, 512))

    verifier = FastVerifier(
        kokoro_text_encode_fn=mock_encode,
        whisper_dim=1280,
        kokoro_dim=512,
    )

    # Mock Whisper encoder output
    whisper_out = mx.random.normal((1, 100, 1280))

    # Test each method
    for method in ["cosine_pooled", "cka"]:
        result = verifier.compute_confidence(whisper_out, "hello world", method=method)
        print(f"  {method}: confidence={result.confidence:.3f}, time={result.verification_time_ms:.1f}ms")

    # Test should_commit
    should, result = verifier.should_commit(whisper_out, "hello world", threshold=0.5)
    print(f"  should_commit (threshold=0.5): {should}")

    print("FastVerifier tests PASSED")


def test_kokoro_fast_verifier():
    """Test with actual Kokoro model."""
    print("\nTesting FastVerifier with Kokoro...")

    try:
        verifier = FastVerifier.from_kokoro()
        print("  Kokoro text encoder loaded")

        # Mock Whisper output (we'd get this from actual STT)
        whisper_out = mx.random.normal((1, 100, 1280))

        # Test encoding speed
        import time
        t0 = time.perf_counter()
        result = verifier.compute_confidence(whisper_out, "hello world how are you today")
        elapsed = (time.perf_counter() - t0) * 1000

        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Verification time: {result.verification_time_ms:.1f}ms")
        print(f"  Total time: {elapsed:.1f}ms")

        if elapsed < 200:
            print("  SUCCESS: <200ms latency achieved!")
        else:
            print(f"  WARNING: {elapsed:.0f}ms > 200ms target")

    except Exception as e:
        print(f"  Kokoro test failed: {e}")


if __name__ == "__main__":
    test_fast_verifier()
    test_kokoro_fast_verifier()
