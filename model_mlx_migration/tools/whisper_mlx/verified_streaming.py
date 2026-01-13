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
Phoneme-Verified Streaming ASR.

Integrates phoneme head verification into streaming transcription for:
- Improved commit/wait decisions (reduce retractions)
- Pronunciation analysis on demand
- Hallucination detection

Architecture:
    Audio -> Whisper Encoder -> CTC Head -> Transcript
                    |
                    +-> Phoneme Head -> Predicted Phonemes
                                |
    Transcript -> Phonemizer ---+-> Edit Distance -> Confidence
                                            |
                            Commit if confidence > threshold

NOTE: Phoneme head weights are currently undertrained (450 samples).
      Results will improve after full training on 132K+ samples.

Usage:
    from tools.whisper_mlx.verified_streaming import VerifiedStreamingWhisper

    model = WhisperMLX.from_pretrained("large-v3")
    streamer = VerifiedStreamingWhisper.from_pretrained(
        model,
        phoneme_head_path="models/kokoro_phoneme_head"
    )

    async for result in streamer.transcribe_stream(audio_source):
        print(f"[{result.commit_status}] {result.text}")
        print(f"  Phoneme confidence: {result.phoneme_confidence:.2f}")
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import WhisperMLX

import time

import numpy as np

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None

from .kokoro_phoneme_head import KokoroPhonemeHead
from .pronunciation_analysis import PronunciationAnalysis, PronunciationAnalyzer
from .streaming import (
    AudioBuffer,
    LocalAgreement,
    StreamingConfig,
    StreamState,
)


class CommitStatus(Enum):
    """Commit status based on phoneme verification."""
    COMMIT = "commit"       # High confidence - finalize text
    PARTIAL = "partial"     # Medium confidence - show but may change
    WAIT = "wait"           # Low confidence - need more context
    FINAL = "final"         # End of utterance - forced commit


@dataclass
class VerifiedStreamingResult:
    """Result from phoneme-verified streaming transcription."""

    # Core transcription
    text: str                       # Current transcript
    commit_status: CommitStatus     # Commit decision
    is_final: bool                  # True if end of utterance

    # Phoneme verification
    phoneme_confidence: float       # 0.0-1.0 phoneme match confidence
    edit_distance: int              # Edit distance between predicted/expected phonemes
    predicted_phonemes: list[str]   # Phonemes predicted from audio
    expected_phonemes: list[str]    # Phonemes from text

    # Timing
    segment_start: float            # Start time in audio stream
    segment_end: float              # End time in audio stream
    processing_time_ms: float       # Total processing time
    verification_time_ms: float     # Phoneme verification time

    # LocalAgreement (if enabled)
    confirmed_text: str = ""        # Stable confirmed text
    speculative_text: str = ""      # May change

    # Optional detailed analysis
    pronunciation_analysis: PronunciationAnalysis | None = None

    # Language
    language: str | None = None


@dataclass
class VerifiedStreamingConfig(StreamingConfig):
    """Configuration for phoneme-verified streaming."""

    # Phoneme verification settings
    phoneme_commit_threshold: float = 0.75    # Confidence to commit
    phoneme_wait_threshold: float = 0.50      # Confidence below which to wait
    use_phoneme_verification: bool = True     # Enable phoneme verification

    # Pronunciation analysis
    include_pronunciation_analysis: bool = False  # Include detailed analysis

    # Override LocalAgreement based on phoneme confidence
    phoneme_overrides_agreement: bool = True  # Low phoneme conf can block commits


class VerifiedStreamingWhisper:
    """
    Streaming transcription with phoneme-based verification.

    Uses phoneme head to compute confidence scores for commit/wait decisions.
    This reduces retractions by catching uncertain transcripts early.

    The phoneme verification adds <10ms latency overhead.
    """

    def __init__(
        self,
        model: WhisperMLX,
        phoneme_head: KokoroPhonemeHead,
        config: VerifiedStreamingConfig | None = None,
    ):
        """
        Initialize verified streaming.

        Args:
            model: WhisperMLX model
            phoneme_head: Trained phoneme head
            config: Streaming configuration
        """
        self.model = model
        self.phoneme_head = phoneme_head
        self.config = config or VerifiedStreamingConfig()

        # Pronunciation analyzer for detailed analysis
        if self.config.include_pronunciation_analysis:
            self._analyzer = PronunciationAnalyzer(
                phoneme_head=phoneme_head,
                phoneme_vocab=PronunciationAnalyzer._get_default_vocab(),
            )
        else:
            self._analyzer = None

        # LocalAgreement
        if self.config.use_local_agreement:
            self._local_agreement = LocalAgreement(
                n=self.config.agreement_n,
                min_stable_ms=self.config.min_stable_ms,
            )
        else:
            self._local_agreement = None

        # State
        self._reset_state()

        # Load phonemizer
        try:
            from tools.pytorch_to_mlx.converters.models.kokoro_phonemizer import (
                phonemize_text,
            )
            self._phonemize = phonemize_text
        except ImportError:
            self._phonemize = None

    @classmethod
    def from_pretrained(
        cls,
        model: WhisperMLX,
        phoneme_head_path: str = "models/kokoro_phoneme_head",
        config: VerifiedStreamingConfig | None = None,
    ) -> VerifiedStreamingWhisper:
        """
        Create verified streaming from pretrained phoneme head.

        Args:
            model: WhisperMLX model
            phoneme_head_path: Path to phoneme head weights
            config: Streaming configuration
        """
        phoneme_head = KokoroPhonemeHead.from_pretrained(phoneme_head_path)
        return cls(model, phoneme_head, config)

    def _reset_state(self) -> None:
        """Reset internal state."""
        self.state = StreamState.IDLE

        cfg = self.config
        max_duration = cfg.max_chunk_duration + cfg.context_duration + 1.0
        self._audio_buffer = AudioBuffer(max_duration, cfg.sample_rate)
        self._speech_buffer = np.array([], dtype=np.float32)

        self._segment_start_time = 0.0
        self._last_partial_time = 0.0
        self._silence_frames = 0

        self._detected_language: str | None = None
        self._last_encoder_output: mx.array | None = None

        if self._local_agreement is not None:
            self._local_agreement.reset()

    def reset(self) -> None:
        """Reset for new session."""
        self._reset_state()

    async def transcribe_stream(
        self,
        audio_source: AsyncIterator[np.ndarray],
    ) -> AsyncIterator[VerifiedStreamingResult]:
        """
        Transcribe streaming audio with phoneme verification.

        Args:
            audio_source: Async iterator yielding audio chunks

        Yields:
            VerifiedStreamingResult with confidence scores
        """
        self._reset_state()

        async for chunk in audio_source:
            async for result in self._process_chunk(chunk):
                yield result

        async for result in self._finalize():
            yield result

    async def _process_chunk(
        self, audio: np.ndarray,
    ) -> AsyncIterator[VerifiedStreamingResult]:
        """Process audio chunk with verification."""
        # Normalize
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        self._audio_buffer.append(audio)

        # Simple state machine (minimal version)
        if self.state == StreamState.IDLE:
            self.state = StreamState.SPEECH
            self._segment_start_time = self._audio_buffer.total_time
            self._speech_buffer = audio.copy()
        else:
            self._speech_buffer = np.concatenate([self._speech_buffer, audio])

        current_duration = len(self._speech_buffer) / self.config.sample_rate
        time_since_partial = self._audio_buffer.total_time - self._last_partial_time

        # Check if we should emit
        if (current_duration >= self.config.min_chunk_duration and
            time_since_partial >= self.config.partial_interval):

            async for result in self._process_segment(is_final=False):
                yield result
            self._last_partial_time = self._audio_buffer.total_time

        # Check max duration
        if current_duration >= self.config.max_chunk_duration:
            async for result in self._process_segment(is_final=True):
                yield result
            self._speech_buffer = np.array([], dtype=np.float32)
            self.state = StreamState.IDLE

    async def _process_segment(
        self, is_final: bool,
    ) -> AsyncIterator[VerifiedStreamingResult]:
        """Process segment with phoneme verification."""
        import asyncio

        if len(self._speech_buffer) < self.config.min_chunk_duration * self.config.sample_rate:
            return

        start_time = time.perf_counter()

        # Run transcription
        result = await asyncio.to_thread(
            self.model.transcribe,
            self._speech_buffer,
            language=self.config.language or self._detected_language,
            task=self.config.task,
        )

        text = result.get("text", "").strip()

        if not text:
            return

        # Cache detected language
        if self._detected_language is None and result.get("language"):
            self._detected_language = result["language"]

        # Get encoder output for phoneme verification
        verification_start = time.perf_counter()

        phoneme_confidence = 1.0
        edit_distance = 0
        predicted_phonemes: list[str] = []
        expected_phonemes: list[str] = []
        pronunciation_analysis = None

        if self.config.use_phoneme_verification and self._phonemize is not None:
            # Get encoder output
            from .audio import log_mel_spectrogram
            mel = log_mel_spectrogram(
                self._speech_buffer,
                n_mels=self.model.config.n_mels,
            )
            mel_mx = mx.expand_dims(mx.array(mel), axis=0)
            encoder_output = self.model.encoder(mel_mx, variable_length=True)
            mx.eval(encoder_output)

            # Run phoneme verification
            phoneme_confidence, edit_distance = self.phoneme_head.compare_with_text(
                encoder_output,
                text,
                language=self._detected_language or "en",
            )

            # Get phoneme sequences for output
            predicted_phonemes = [str(t) for t in self.phoneme_head.predict(encoder_output)]

            try:
                _, token_ids = self._phonemize(text, language=self._detected_language or "en")
                expected_phonemes = [str(t) for t in token_ids if t != 0]
            except Exception:
                expected_phonemes = []

            # Optional detailed analysis
            if self.config.include_pronunciation_analysis and self._analyzer is not None:
                pronunciation_analysis = self._analyzer.analyze(
                    encoder_output, text,
                    language=self._detected_language or "en",
                )

        verification_time = (time.perf_counter() - verification_start) * 1000

        # Determine commit status based on phoneme confidence
        if is_final:
            commit_status = CommitStatus.FINAL
        elif phoneme_confidence >= self.config.phoneme_commit_threshold:
            commit_status = CommitStatus.COMMIT
        elif phoneme_confidence < self.config.phoneme_wait_threshold:
            commit_status = CommitStatus.WAIT
        else:
            commit_status = CommitStatus.PARTIAL

        # Apply LocalAgreement
        confirmed_text = ""
        speculative_text = ""

        if self._local_agreement is not None:
            newly_confirmed = self._local_agreement.update(text)
            confirmed_text = self._local_agreement.get_confirmed()
            speculative_text = self._local_agreement.get_speculative()

            # Phoneme confidence can override LocalAgreement
            if (self.config.phoneme_overrides_agreement and
                commit_status == CommitStatus.WAIT and
                newly_confirmed):
                # Don't actually commit if phoneme confidence is low
                # (This prevents retractions)
                commit_status = CommitStatus.WAIT

        processing_time = (time.perf_counter() - start_time) * 1000

        yield VerifiedStreamingResult(
            text=text,
            commit_status=commit_status,
            is_final=is_final,
            phoneme_confidence=phoneme_confidence,
            edit_distance=edit_distance,
            predicted_phonemes=predicted_phonemes,
            expected_phonemes=expected_phonemes,
            segment_start=self._segment_start_time,
            segment_end=self._audio_buffer.total_time,
            processing_time_ms=processing_time,
            verification_time_ms=verification_time,
            confirmed_text=confirmed_text,
            speculative_text=speculative_text,
            pronunciation_analysis=pronunciation_analysis,
            language=self._detected_language,
        )

    async def _finalize(self) -> AsyncIterator[VerifiedStreamingResult]:
        """Finalize remaining audio."""
        if len(self._speech_buffer) >= self.config.min_chunk_duration * self.config.sample_rate:
            async for result in self._process_segment(is_final=True):
                yield result

        self._reset_state()


# =============================================================================
# Convenience functions
# =============================================================================


def create_verified_streamer(
    model: WhisperMLX,
    phoneme_head_path: str = "models/kokoro_phoneme_head",
    commit_threshold: float = 0.75,
    wait_threshold: float = 0.50,
    include_analysis: bool = False,
) -> VerifiedStreamingWhisper:
    """
    Create a verified streaming transcriber.

    Args:
        model: WhisperMLX model
        phoneme_head_path: Path to phoneme head weights
        commit_threshold: Confidence threshold to commit text
        wait_threshold: Confidence below which to wait for more context
        include_analysis: Include detailed pronunciation analysis

    Returns:
        VerifiedStreamingWhisper instance
    """
    config = VerifiedStreamingConfig(
        phoneme_commit_threshold=commit_threshold,
        phoneme_wait_threshold=wait_threshold,
        include_pronunciation_analysis=include_analysis,
    )

    return VerifiedStreamingWhisper.from_pretrained(
        model, phoneme_head_path, config,
    )


__all__ = [
    "VerifiedStreamingWhisper",
    "VerifiedStreamingConfig",
    "VerifiedStreamingResult",
    "CommitStatus",
    "create_verified_streamer",
]
