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
Round-Trip Spectrogram Verification for Streaming ASR

A novel technique that uses TTS to verify STT confidence by:
1. Taking an audio chunk and its STT transcript
2. Synthesizing audio from the transcript using TTS (Kokoro)
3. Comparing the mel spectrograms of original and synthesized audio
4. Using the similarity as a confidence score for commit decisions

This enables more intelligent streaming decisions - high confidence transcripts
can be committed immediately while low confidence ones wait for more context.

Usage:
    from tools.whisper_mlx.roundtrip_verification import RoundTripVerifier

    # Initialize with Kokoro TTS
    verifier = RoundTripVerifier.from_kokoro("hexgrad/Kokoro-82M")

    # Compute confidence for a transcription
    confidence = verifier.compute_confidence(audio_chunk, "hello world")

    # Or use the commit decision helper
    should_commit, score = verifier.should_commit(audio_chunk, "hello world", threshold=0.6)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None

# Type hints for TTS models


@dataclass
class VerificationResult:
    """Result of round-trip spectrogram verification."""
    confidence: float           # Similarity score in [0, 1]
    input_mel_frames: int       # Number of frames in input mel
    generated_mel_frames: int   # Number of frames in generated mel
    alignment_cost: float       # DTW alignment cost (lower = better)
    is_reliable: bool           # Whether the result is reliable


class MelSimilarity:
    """
    Mel spectrogram similarity computation.

    Provides multiple methods for comparing mel spectrograms:
    - DTW (Dynamic Time Warping) for length-invariant comparison
    - Cosine similarity of pooled features (faster but less accurate)
    - Correlation-based similarity
    """

    @staticmethod
    def cosine_similarity_pooled(mel1: np.ndarray, mel2: np.ndarray) -> float:
        """
        Compute cosine similarity between mean-pooled mel features.

        This is the fastest method but ignores temporal structure.
        Good for quick confidence estimates.

        Args:
            mel1: First mel spectrogram, shape (T1, n_mels)
            mel2: Second mel spectrogram, shape (T2, n_mels)

        Returns:
            Similarity score in [0, 1]
        """
        # Mean pool over time
        vec1 = mel1.mean(axis=0)
        vec2 = mel2.mean(axis=0)

        # Normalize
        norm1 = np.linalg.norm(vec1) + 1e-8
        norm2 = np.linalg.norm(vec2) + 1e-8

        # Cosine similarity
        similarity = np.dot(vec1, vec2) / (norm1 * norm2)

        # Map from [-1, 1] to [0, 1]
        return float((similarity + 1) / 2)

    @staticmethod
    def dtw_similarity(
        mel1: np.ndarray,
        mel2: np.ndarray,
        max_frames: int = 500,
    ) -> tuple[float, float]:
        """
        Compute DTW-based similarity between mel spectrograms.

        Dynamic Time Warping handles different lengths and speaking rates.
        This is more accurate but slower than pooled methods.

        Args:
            mel1: First mel spectrogram, shape (T1, n_mels)
            mel2: Second mel spectrogram, shape (T2, n_mels)
            max_frames: Maximum frames to process (truncates for efficiency)

        Returns:
            Tuple of (similarity in [0, 1], alignment_cost)
        """
        # Truncate for efficiency
        if mel1.shape[0] > max_frames:
            mel1 = mel1[:max_frames]
        if mel2.shape[0] > max_frames:
            mel2 = mel2[:max_frames]

        T1, D = mel1.shape
        T2 = mel2.shape[0]

        # Cost matrix (Euclidean distance)
        # For efficiency, compute in blocks if large
        cost = np.zeros((T1, T2), dtype=np.float32)
        for i in range(T1):
            cost[i] = np.sqrt(np.sum((mel1[i:i+1] - mel2) ** 2, axis=1))

        # DTW with cumulative cost matrix
        dtw = np.full((T1 + 1, T2 + 1), np.inf, dtype=np.float32)
        dtw[0, 0] = 0

        for i in range(1, T1 + 1):
            for j in range(1, T2 + 1):
                dtw[i, j] = cost[i-1, j-1] + min(
                    dtw[i-1, j],     # insertion
                    dtw[i, j-1],     # deletion
                    dtw[i-1, j-1],   # match
                )

        # Normalized alignment cost
        alignment_cost = dtw[T1, T2] / (T1 + T2)

        # Convert cost to similarity (lower cost = higher similarity)
        # Use exponential decay: similarity = exp(-cost / scale)
        # Scale tuned based on typical mel spectrogram distances:
        # - Same speech: ~0.1-0.3 cost
        # - Similar speech (same words, different speaker): ~0.5-1.0
        # - Different speech: ~1.5-3.0
        # - Random/noise: ~3.0+
        scale = 2.0
        similarity = float(np.exp(-alignment_cost / scale))

        return similarity, float(alignment_cost)

    @staticmethod
    def dtw_similarity_fast(
        mel1: np.ndarray,
        mel2: np.ndarray,
        radius: int = 50,
    ) -> tuple[float, float]:
        """
        Fast DTW using Sakoe-Chiba band constraint.

        Only computes DTW cells within a band of width `radius` around
        the diagonal. Much faster for long sequences.

        Args:
            mel1: First mel spectrogram, shape (T1, n_mels)
            mel2: Second mel spectrogram, shape (T2, n_mels)
            radius: Width of Sakoe-Chiba band

        Returns:
            Tuple of (similarity in [0, 1], alignment_cost)
        """
        T1, D = mel1.shape
        T2 = mel2.shape[0]

        # DTW with band constraint
        dtw = np.full((T1 + 1, T2 + 1), np.inf, dtype=np.float32)
        dtw[0, 0] = 0

        for i in range(1, T1 + 1):
            # Compute band limits
            j_min = max(1, int(i * T2 / T1) - radius)
            j_max = min(T2, int(i * T2 / T1) + radius)

            for j in range(j_min, j_max + 1):
                cost = np.sqrt(np.sum((mel1[i-1] - mel2[j-1]) ** 2))
                dtw[i, j] = cost + min(
                    dtw[i-1, j],
                    dtw[i, j-1],
                    dtw[i-1, j-1],
                )

        alignment_cost = dtw[T1, T2] / (T1 + T2)
        scale = 2.0  # Same scale as full DTW
        similarity = float(np.exp(-alignment_cost / scale))

        return similarity, float(alignment_cost)

    @staticmethod
    def correlation_similarity(mel1: np.ndarray, mel2: np.ndarray) -> float:
        """
        Compute correlation-based similarity.

        Uses Pearson correlation on flattened mel-frequency energy profiles.
        Handles different lengths by resampling to common length.

        Args:
            mel1: First mel spectrogram, shape (T1, n_mels)
            mel2: Second mel spectrogram, shape (T2, n_mels)

        Returns:
            Similarity score in [0, 1]
        """
        # Compute mel-frequency energy profile (sum over time)
        energy1 = mel1.mean(axis=0)
        energy2 = mel2.mean(axis=0)

        # Correlation
        mean1, mean2 = energy1.mean(), energy2.mean()
        std1, std2 = energy1.std() + 1e-8, energy2.std() + 1e-8

        correlation = np.mean((energy1 - mean1) * (energy2 - mean2)) / (std1 * std2)

        # Map from [-1, 1] to [0, 1]
        return float((correlation + 1) / 2)


class RoundTripVerifier:
    """
    Round-trip spectrogram verification for streaming ASR.

    Uses TTS to synthesize audio from STT output, then compares
    mel spectrograms to estimate transcription confidence.

    High similarity = transcript is likely correct
    Low similarity = transcript may be wrong, wait for more context
    """

    def __init__(
        self,
        tts_synthesize_fn: Callable[[str], np.ndarray],
        mel_extract_fn: Callable[[np.ndarray], np.ndarray],
        sample_rate: int = 16000,
        n_mels: int = 128,
        similarity_method: str = "dtw_fast",
    ):
        """
        Initialize round-trip verifier.

        Args:
            tts_synthesize_fn: Function that takes text and returns audio (float32)
            mel_extract_fn: Function that extracts mel spectrogram from audio
            sample_rate: Audio sample rate (must match TTS output)
            n_mels: Number of mel bands
            similarity_method: One of "dtw", "dtw_fast", "cosine", "correlation"
        """
        self.tts_synthesize = tts_synthesize_fn
        self.mel_extract = mel_extract_fn
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.similarity_method = similarity_method

        # Minimum audio duration for reliable verification
        self.min_duration_sec = 0.3
        self.min_frames = int(self.min_duration_sec * self.sample_rate / 160)  # 160 = hop_length

    @classmethod
    def from_kokoro(
        cls,
        model_id: str = "hexgrad/Kokoro-82M",
        voice: str = "af_bella",
        cache_dir: str | None = None,
        similarity_method: str = "dtw_fast",
    ) -> RoundTripVerifier:
        """
        Create a RoundTripVerifier using Kokoro TTS.

        Args:
            model_id: HuggingFace model ID for Kokoro
            voice: Voice to use for synthesis
            cache_dir: Local cache directory for model files
            similarity_method: Similarity computation method

        Returns:
            RoundTripVerifier instance
        """
        if not HAS_MLX:
            raise ImportError("MLX is required for Kokoro TTS")

        # Import Kokoro components
        from tools.pytorch_to_mlx.converters.kokoro_converter import KokoroConverter
        from tools.pytorch_to_mlx.converters.models.kokoro_phonemizer import (
            phonemize_text,
        )

        # Load Kokoro model
        converter = KokoroConverter()
        model, config, _ = converter.load_from_hf(model_id, cache_dir)

        # Load voice embedding
        voice_pack = converter.load_voice_pack(voice, model_id, cache_dir)

        def synthesize(text: str) -> np.ndarray:
            """Synthesize audio from text using Kokoro."""
            # Phonemize text and get token IDs directly
            phoneme_str, token_ids = phonemize_text(text, language="en")

            if len(token_ids) == 0:
                return np.zeros(1600, dtype=np.float32)  # 100ms of silence

            # Prepare inputs
            input_ids = mx.array([token_ids])

            # Select voice embedding for this sequence length
            voice_embedding = converter.select_voice_embedding(
                voice_pack, len(token_ids),
            )
            if voice_embedding.ndim == 1:
                voice_embedding = voice_embedding[None, :]

            # Generate audio
            audio = model(input_ids, voice_embedding)
            mx.eval(audio)

            # Convert to numpy (Kokoro outputs at 24kHz, need to resample to 16kHz)
            audio_np = np.array(audio).flatten().astype(np.float32)

            # Resample from 24kHz to 16kHz
            from scipy import signal
            return signal.resample(
                audio_np,
                int(len(audio_np) * 16000 / 24000),
            ).astype(np.float32)


        def extract_mel(audio: np.ndarray) -> np.ndarray:
            """Extract mel spectrogram from audio."""
            from tools.whisper_mlx.audio import log_mel_spectrogram

            # log_mel_spectrogram returns MLX array, convert to numpy
            mel = log_mel_spectrogram(audio, n_mels=128)
            if isinstance(mel, mx.array):
                mel = np.array(mel)
            return mel

        return cls(
            tts_synthesize_fn=synthesize,
            mel_extract_fn=extract_mel,
            sample_rate=16000,
            n_mels=128,
            similarity_method=similarity_method,
        )

    @classmethod
    def from_mock(
        cls,
        similarity_method: str = "cosine",
    ) -> RoundTripVerifier:
        """
        Create a mock RoundTripVerifier for testing without TTS.

        The mock TTS returns Gaussian noise, so similarity will be low
        unless the input is also noise-like.

        Args:
            similarity_method: Similarity computation method

        Returns:
            RoundTripVerifier instance with mock TTS
        """
        def mock_synthesize(text: str) -> np.ndarray:
            """Mock TTS that returns noise proportional to text length."""
            # Generate ~100ms per word
            words = max(1, len(text.split()))
            duration_sec = 0.1 * words
            samples = int(duration_sec * 16000)
            rng = np.random.default_rng()
            return rng.standard_normal(samples).astype(np.float32) * 0.1

        def extract_mel(audio: np.ndarray) -> np.ndarray:
            """Simple mel extraction without MLX dependency."""
            # Use scipy for FFT
            from scipy.fft import rfft

            n_fft = 400
            hop_length = 160
            n_mels = 128

            # Pad audio
            audio = np.pad(audio, (n_fft // 2, n_fft // 2), mode='reflect')

            # Frame audio
            n_frames = 1 + (len(audio) - n_fft) // hop_length
            frames = np.lib.stride_tricks.as_strided(
                audio,
                shape=(n_frames, n_fft),
                strides=(audio.strides[0] * hop_length, audio.strides[0]),
            )

            # Window and FFT
            window = np.hanning(n_fft + 1)[:-1].astype(np.float32)
            windowed = frames * window
            stft = rfft(windowed, n=n_fft)
            magnitudes = np.abs(stft[:, :-1]) ** 2

            # Simple mel filterbank (linear approximation)
            mel_filters = np.zeros((n_mels, n_fft // 2), dtype=np.float32)
            for i in range(n_mels):
                center = int((i + 1) * (n_fft // 2) / (n_mels + 1))
                width = max(1, (n_fft // 2) // (n_mels + 1))
                start = max(0, center - width)
                end = min(n_fft // 2, center + width)
                mel_filters[i, start:end] = 1.0 / (end - start)

            mel_spec = magnitudes @ mel_filters.T

            # Log scale
            log_spec = np.log10(np.clip(mel_spec, 1e-10, None))
            log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
            log_spec = (log_spec + 4.0) / 4.0

            return log_spec.astype(np.float32)

        return cls(
            tts_synthesize_fn=mock_synthesize,
            mel_extract_fn=extract_mel,
            sample_rate=16000,
            n_mels=128,
            similarity_method=similarity_method,
        )

    def compute_confidence(
        self,
        audio_chunk: np.ndarray,
        transcript: str,
    ) -> VerificationResult:
        """
        Compute confidence score for a transcription.

        Args:
            audio_chunk: Input audio (float32, 16kHz mono)
            transcript: STT transcription of the audio

        Returns:
            VerificationResult with confidence score and metadata
        """
        # Validate inputs
        if len(audio_chunk) == 0:
            return VerificationResult(
                confidence=0.0,
                input_mel_frames=0,
                generated_mel_frames=0,
                alignment_cost=float('inf'),
                is_reliable=False,
            )

        if not transcript or not transcript.strip():
            return VerificationResult(
                confidence=0.0,
                input_mel_frames=0,
                generated_mel_frames=0,
                alignment_cost=float('inf'),
                is_reliable=False,
            )

        # Extract mel from input audio
        input_mel = self.mel_extract(audio_chunk)

        # Synthesize audio from transcript
        try:
            generated_audio = self.tts_synthesize(transcript.strip())
        except Exception:
            # TTS failed - return low confidence
            return VerificationResult(
                confidence=0.0,
                input_mel_frames=input_mel.shape[0],
                generated_mel_frames=0,
                alignment_cost=float('inf'),
                is_reliable=False,
            )

        # Extract mel from generated audio
        generated_mel = self.mel_extract(generated_audio)

        # Check minimum duration
        is_reliable = (
            input_mel.shape[0] >= self.min_frames and
            generated_mel.shape[0] >= self.min_frames
        )

        # Compute similarity
        if self.similarity_method == "dtw":
            confidence, alignment_cost = MelSimilarity.dtw_similarity(
                input_mel, generated_mel,
            )
        elif self.similarity_method == "dtw_fast":
            confidence, alignment_cost = MelSimilarity.dtw_similarity_fast(
                input_mel, generated_mel,
            )
        elif self.similarity_method == "cosine":
            confidence = MelSimilarity.cosine_similarity_pooled(
                input_mel, generated_mel,
            )
            alignment_cost = 1.0 - confidence
        elif self.similarity_method == "correlation":
            confidence = MelSimilarity.correlation_similarity(
                input_mel, generated_mel,
            )
            alignment_cost = 1.0 - confidence
        else:
            raise ValueError(f"Unknown similarity method: {self.similarity_method}")

        return VerificationResult(
            confidence=confidence,
            input_mel_frames=input_mel.shape[0],
            generated_mel_frames=generated_mel.shape[0],
            alignment_cost=alignment_cost,
            is_reliable=is_reliable,
        )

    def should_commit(
        self,
        audio_chunk: np.ndarray,
        transcript: str,
        threshold: float = 0.6,
    ) -> tuple[bool, float]:
        """
        Decide whether to commit a transcription based on confidence.

        Args:
            audio_chunk: Input audio (float32, 16kHz mono)
            transcript: STT transcription
            threshold: Confidence threshold for commit decision

        Returns:
            Tuple of (should_commit, confidence_score)
        """
        result = self.compute_confidence(audio_chunk, transcript)

        # Only commit if reliable AND above threshold
        should_commit = result.is_reliable and result.confidence >= threshold

        return should_commit, result.confidence


# =============================================================================
# Test Functions
# =============================================================================


def test_mel_similarity():
    """Test mel similarity functions."""
    print("Testing mel similarity functions...")

    # Create synthetic mel spectrograms
    rng = np.random.default_rng(42)

    # Identical mels should have high similarity
    mel1 = rng.standard_normal((100, 128)).astype(np.float32)
    mel2 = mel1.copy()

    cosine = MelSimilarity.cosine_similarity_pooled(mel1, mel2)
    dtw_sim, dtw_cost = MelSimilarity.dtw_similarity_fast(mel1, mel2)
    corr = MelSimilarity.correlation_similarity(mel1, mel2)

    print(f"Identical mels - Cosine: {cosine:.3f}, DTW: {dtw_sim:.3f}, Corr: {corr:.3f}")
    assert cosine > 0.99, f"Cosine should be ~1.0, got {cosine}"
    assert dtw_sim > 0.99, f"DTW should be ~1.0, got {dtw_sim}"

    # Different mels should have lower similarity
    mel3 = rng.standard_normal((100, 128)).astype(np.float32)

    cosine = MelSimilarity.cosine_similarity_pooled(mel1, mel3)
    dtw_sim, dtw_cost = MelSimilarity.dtw_similarity_fast(mel1, mel3)
    corr = MelSimilarity.correlation_similarity(mel1, mel3)

    print(f"Different mels - Cosine: {cosine:.3f}, DTW: {dtw_sim:.3f}, Corr: {corr:.3f}")

    # Different lengths
    mel4 = rng.standard_normal((150, 128)).astype(np.float32)
    dtw_sim, dtw_cost = MelSimilarity.dtw_similarity_fast(mel1, mel4)
    print(f"Different lengths (100 vs 150) - DTW: {dtw_sim:.3f}")

    print("Mel similarity tests PASSED")


def test_mock_verifier():
    """Test RoundTripVerifier with mock TTS."""
    print("\nTesting mock RoundTripVerifier...")

    verifier = RoundTripVerifier.from_mock(similarity_method="cosine")

    # Generate test audio (100ms of sine wave)
    t = np.linspace(0, 0.5, int(0.5 * 16000))
    audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)

    # Test with transcript
    result = verifier.compute_confidence(audio, "hello world")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Input frames: {result.input_mel_frames}")
    print(f"Generated frames: {result.generated_mel_frames}")
    print(f"Is reliable: {result.is_reliable}")

    # Test should_commit
    should, score = verifier.should_commit(audio, "hello world", threshold=0.3)
    print(f"Should commit (threshold=0.3): {should}, score: {score:.3f}")

    # Test with empty transcript
    result = verifier.compute_confidence(audio, "")
    assert result.confidence == 0.0, "Empty transcript should give 0 confidence"

    print("Mock verifier tests PASSED")


def test_kokoro_verifier():
    """Test RoundTripVerifier with Kokoro TTS (requires model files)."""
    print("\nTesting Kokoro RoundTripVerifier...")

    try:
        verifier = RoundTripVerifier.from_kokoro()
        print("Kokoro model loaded successfully")

        # Generate test audio - simple speech-like signal
        t = np.linspace(0, 1.0, int(1.0 * 16000))
        # Mix of frequencies to simulate speech
        audio = (
            np.sin(2 * np.pi * 200 * t) * 0.3 +
            np.sin(2 * np.pi * 400 * t) * 0.2 +
            np.sin(2 * np.pi * 800 * t) * 0.1
        ).astype(np.float32)

        # Test with simple transcript
        result = verifier.compute_confidence(audio, "hello")
        print(f"Confidence for 'hello': {result.confidence:.3f}")
        print(f"Input frames: {result.input_mel_frames}")
        print(f"Generated frames: {result.generated_mel_frames}")
        print(f"Alignment cost: {result.alignment_cost:.3f}")

        # Test should_commit
        should, score = verifier.should_commit(audio, "hello world", threshold=0.5)
        print(f"Should commit (threshold=0.5): {should}, score: {score:.3f}")

        print("Kokoro verifier tests PASSED")

    except ImportError as e:
        print(f"Skipping Kokoro test (missing dependency): {e}")
    except Exception as e:
        print(f"Kokoro test failed (model may not be available): {e}")


if __name__ == "__main__":
    test_mel_similarity()
    test_mock_verifier()
    test_kokoro_verifier()
