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
ROVER Integration Module for High-Accuracy ASR.

This module integrates multiple ASR sources with ROVER voting to achieve
<1.5% WER in high-accuracy mode. It combines:

1. **Transducer** (primary): Zipformer + Pruned RNN-T for streaming ASR
2. **CTC** (secondary): CTC head on encoder for fast draft tokens
3. **Whisper** (fallback): OpenAI Whisper for 100+ language support

The ROVER algorithm aligns hypotheses from multiple sources and votes
on the best word at each position using confidence-weighted voting.

Usage:
    from src.decoding.rover_integration import HighAccuracyDecoder

    decoder = HighAccuracyDecoder.from_pretrained(
        zipformer_checkpoint="checkpoints/zipformer/...",
        whisper_model="large-v3",
    )
    result = decoder.transcribe("audio.wav")
    print(result.text)  # High-accuracy transcription
    print(result.sources)  # Individual source outputs

Phase 8 Implementation:
- Phase 8.1: ROVER voting algorithm [DONE]
- Phase 8.2: Whisper integration [THIS FILE]
- Phase 8.3: CTC integration [THIS FILE]
- Phase 8.4: Phoneme weighting [PENDING - needs phoneme head]
- Phase 8.5: Validation [PENDING]
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import numpy as np

from .rover import ROVER, Hypothesis, ROVERConfig, ROVERResult

# =============================================================================
# Abstract Base Class for ROVER Sources
# =============================================================================


@dataclass
class SourceResult:
    """Result from a single ASR source."""
    words: list[str]
    confidences: list[float]
    source_name: str
    raw_text: str  # Original text before word splitting
    metadata: dict = field(default_factory=dict)


class ROVERSource(ABC):
    """Abstract base class for ASR sources that can feed into ROVER."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the source name (e.g., 'transducer', 'ctc', 'whisper')."""

    @abstractmethod
    def transcribe(
        self,
        audio: str | np.ndarray | mx.array,
        **kwargs,
    ) -> SourceResult:
        """
        Transcribe audio and return word-level result with confidences.

        Args:
            audio: Audio file path or waveform array

        Returns:
            SourceResult with words, confidences, and metadata
        """


# =============================================================================
# Whisper ROVER Source (Phase 8.2)
# =============================================================================


class WhisperROVERSource(ROVERSource):
    """
    Whisper ASR source for ROVER integration.

    Wraps WhisperMLX to produce word-level hypotheses with confidences.
    Uses segment-level avg_logprob as confidence proxy for each word.

    Attributes:
        model: WhisperMLX model instance
        language: Target language code (None for auto-detect)
    """

    def __init__(
        self,
        model_name: str = "large-v3",
        language: str | None = None,
        _device: str = "mps",
    ):
        """
        Initialize Whisper ROVER source.

        Args:
            model_name: Whisper model name (e.g., "large-v3", "medium", "small")
            language: Target language (None for auto-detect)
            _device: Compute device (ignored for MLX, always uses Metal)
        """
        self._model_name = model_name
        self._language = language
        self._model = None  # Lazy loading

    @property
    def name(self) -> str:
        return "whisper"

    def _ensure_model_loaded(self) -> None:
        """Load Whisper model on first use."""
        if self._model is not None:
            return

        # Import here to avoid circular dependencies
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tools"))

        try:
            from whisper_mlx.model import WhisperMLX
            self._model = WhisperMLX.from_pretrained(self._model_name)
        except ImportError as e:
            msg = f"WhisperMLX not found. Ensure tools/whisper_mlx is available: {e}"
            raise ImportError(msg) from e

    def transcribe(
        self,
        audio: str | np.ndarray | mx.array,
        **kwargs,
    ) -> SourceResult:
        """
        Transcribe audio using Whisper.

        Args:
            audio: Audio file path or waveform

        Returns:
            SourceResult with word-level breakdown and confidences
        """
        self._ensure_model_loaded()

        # Run Whisper transcription
        result = self._model.transcribe(
            audio,
            language=self._language,
            **kwargs,
        )

        # Extract text and segments
        text = result.get("text", "").strip()
        segments = result.get("segments", [])

        # Convert to word-level with confidences
        words, confidences = self._extract_word_confidences(text, segments)

        return SourceResult(
            words=words,
            confidences=confidences,
            source_name=self.name,
            raw_text=text,
            metadata={
                "language": result.get("language", "unknown"),
                "segments": segments,
            },
        )

    def _extract_word_confidences(
        self,
        text: str,
        segments: list[dict],
    ) -> tuple[list[str], list[float]]:
        """
        Extract word-level confidences from Whisper output.

        Whisper doesn't provide per-word confidences directly.
        We use segment-level avg_logprob as a proxy for all words in that segment.

        Args:
            text: Full transcription text
            segments: List of segment dictionaries

        Returns:
            Tuple of (words, confidences)
        """
        if not text:
            return [], []

        # Split text into words
        words = text.split()
        if not words:
            return [], []

        # If no segments, use uniform confidence
        if not segments:
            return words, [0.5] * len(words)

        # Build word-to-confidence mapping based on segment positions
        word_confidences = []
        word_idx = 0

        for segment in segments:
            seg_text = segment.get("text", "").strip()
            seg_words = seg_text.split()

            # Convert avg_logprob to probability (bounded to [0.01, 0.99])
            avg_logprob = segment.get("avg_logprob", -1.0)
            # logprob is log(p), so p = exp(logprob)
            # Typical range: -0.1 (very confident) to -2.0 (less confident)
            confidence = max(0.01, min(0.99, np.exp(avg_logprob)))

            # Assign confidence to each word in segment
            for _ in seg_words:
                if word_idx < len(words):
                    word_confidences.append(confidence)
                    word_idx += 1

        # Fill remaining words with average confidence
        if len(word_confidences) < len(words):
            avg_conf = np.mean(word_confidences) if word_confidences else 0.5
            word_confidences.extend([avg_conf] * (len(words) - len(word_confidences)))

        return words, word_confidences

    def to_hypothesis(self, result: SourceResult) -> Hypothesis:
        """Convert SourceResult to ROVER Hypothesis."""
        return Hypothesis(
            words=result.words,
            confidences=result.confidences,
            source=result.source_name,
        )


# =============================================================================
# CTC ROVER Source (Phase 8.3)
# =============================================================================


class CTCROVERSource(ROVERSource):
    """
    CTC ASR source for ROVER integration.

    Uses CTC head on encoder output for fast draft transcription.
    Provides quick, moderately-accurate hypotheses that complement
    the slower but more accurate transducer and Whisper.

    Attributes:
        ctc_head: CTCDraftHead model
        tokenizer: Tokenizer for decoding tokens to text
    """

    def __init__(
        self,
        ctc_head=None,
        tokenizer=None,
        encoder=None,
    ):
        """
        Initialize CTC ROVER source.

        Args:
            ctc_head: Pre-loaded CTCDraftHead (or will be loaded)
            tokenizer: Tokenizer for token-to-text decoding
            encoder: Audio encoder for feature extraction
        """
        self._ctc_head = ctc_head
        self._tokenizer = tokenizer
        self._encoder = encoder

    @property
    def name(self) -> str:
        return "ctc"

    def transcribe(
        self,
        _audio: str | np.ndarray | mx.array,
        encoder_output: mx.array | None = None,
        **_kwargs,
    ) -> SourceResult:
        """
        Transcribe audio using CTC decoding.

        Can accept pre-computed encoder output to avoid redundant encoding
        when used alongside transducer (which shares the encoder).

        Args:
            _audio: Audio file path or waveform (unused, requires encoder_output)
            encoder_output: Pre-computed encoder hidden states

        Returns:
            SourceResult with word-level breakdown and confidences
        """
        if encoder_output is None:
            if self._encoder is None:
                msg = "No encoder provided and no encoder_output given"
                raise ValueError(msg)
            # Would need to run encoder here - for now require pre-computed
            msg = (
                "CTC source requires pre-computed encoder_output. "
                "Use with TransducerROVERSource which shares the encoder."
            )
            raise NotImplementedError(msg)

        # Run CTC decoding
        logits = self._ctc_head(encoder_output)

        # Greedy decode with confidences
        tokens, confidences = self._decode_with_confidences(logits)

        # Decode tokens to text
        if self._tokenizer is not None:
            text = self._tokenizer.decode(tokens)
            words = text.split()
        else:
            # Without tokenizer, return token IDs as strings
            text = " ".join(str(t) for t in tokens)
            words = [str(t) for t in tokens]

        # Redistribute confidences to words
        word_confidences = self._token_to_word_confidences(
            tokens, confidences, words,
        )

        return SourceResult(
            words=words,
            confidences=word_confidences,
            source_name=self.name,
            raw_text=text,
            metadata={
                "num_tokens": len(tokens),
                "num_frames": logits.shape[1] if logits.ndim > 1 else logits.shape[0],
            },
        )

    def _decode_with_confidences(
        self,
        logits: mx.array,
    ) -> tuple[list[int], list[float]]:
        """
        CTC greedy decoding with per-token confidences.

        Args:
            logits: (batch, T, vocab_size) or (T, vocab_size)

        Returns:
            Tuple of (tokens, confidences)
        """
        if logits.ndim == 3:
            logits = logits[0]

        # Get predictions and probabilities
        probs = mx.softmax(logits, axis=-1)
        predictions = mx.argmax(logits, axis=-1)
        mx.eval(predictions, probs)

        # Get confidence (probability of selected token)
        predictions_np = np.array(predictions)
        probs_np = np.array(probs)

        # Collapse blanks and consecutive repeats with confidence tracking
        blank_id = 0  # Standard CTC blank
        tokens = []
        confidences = []
        prev_token = blank_id

        for frame_idx, token in enumerate(predictions_np):
            if token != blank_id and token != prev_token:
                tokens.append(int(token))
                # Confidence is the probability of this token at this frame
                confidences.append(float(probs_np[frame_idx, token]))
            prev_token = token

        return tokens, confidences

    def _token_to_word_confidences(
        self,
        _tokens: list[int],
        token_confidences: list[float],
        words: list[str],
    ) -> list[float]:
        """
        Map token-level confidences to word-level.

        Uses simple averaging across tokens that make up each word.
        _tokens: Currently unused (kept for future BPE boundary mapping).
        """
        if not words or not token_confidences:
            return [0.5] * len(words)

        # Simple approach: distribute equally
        # More sophisticated: use token boundaries from BPE
        avg_conf = np.mean(token_confidences) if token_confidences else 0.5
        return [avg_conf] * len(words)

    def to_hypothesis(self, result: SourceResult) -> Hypothesis:
        """Convert SourceResult to ROVER Hypothesis."""
        return Hypothesis(
            words=result.words,
            confidences=result.confidences,
            source=result.source_name,
        )


# =============================================================================
# Transducer ROVER Source (Primary ASR)
# =============================================================================


class TransducerROVERSource(ROVERSource):
    """
    Transducer (RNN-T) ASR source for ROVER integration.

    Uses Zipformer encoder + Pruned RNN-T decoder for streaming ASR.
    This is the primary ASR system with best streaming WER.

    Attributes:
        model: ASRModel (Zipformer + Transducer)
        tokenizer: SentencePiece tokenizer
    """

    def __init__(
        self,
        model=None,
        tokenizer=None,
        decoding_method: str = "greedy",
    ):
        """
        Initialize Transducer ROVER source.

        Args:
            model: Pre-loaded ASRModel
            tokenizer: SentencePiece tokenizer
            decoding_method: "greedy" or "beam"
        """
        self._model = model
        self._tokenizer = tokenizer
        self._decoding_method = decoding_method

    @property
    def name(self) -> str:
        return "transducer"

    def transcribe(
        self,
        audio: str | np.ndarray | mx.array,
        **_kwargs,
    ) -> SourceResult:
        """
        Transcribe audio using RNN-T transducer.

        Args:
            audio: Audio file path or waveform

        Returns:
            SourceResult with word-level breakdown and confidences
        """
        if self._model is None:
            msg = "Transducer model not loaded"
            raise ValueError(msg)

        # Import decoding functions
        from src.models.zipformer.decoding import DecodingResult, greedy_search
        from src.models.zipformer.features import FbankExtractor, load_audio

        # Load and preprocess audio
        if isinstance(audio, str):
            waveform, sr = load_audio(audio)
        else:
            waveform = audio

        # Extract features
        fbank = FbankExtractor()
        features = fbank(mx.array(waveform))

        # Run encoder
        encoder_out, encoder_out_len = self._model.encode(features)

        # Run decoder
        result: DecodingResult = greedy_search(
            self._model.decoder,
            self._model.joiner,
            encoder_out[0],  # Remove batch dim
            encoder_out_len,
        )

        # Decode tokens to text
        if self._tokenizer is not None:
            text = self._tokenizer.decode(result.tokens)
        else:
            text = " ".join(str(t) for t in result.tokens)

        words = text.split()

        # Compute per-word confidence from score
        # RNN-T score is total log probability
        # Distribute evenly across words as approximation
        avg_logprob = result.score / max(len(result.tokens), 1)
        confidence = max(0.01, min(0.99, np.exp(avg_logprob)))
        word_confidences = [confidence] * len(words)

        return SourceResult(
            words=words,
            confidences=word_confidences,
            source_name=self.name,
            raw_text=text,
            metadata={
                "tokens": result.tokens,
                "score": result.score,
                "encoder_output": encoder_out,  # For CTC sharing
            },
        )

    def to_hypothesis(self, result: SourceResult) -> Hypothesis:
        """Convert SourceResult to ROVER Hypothesis."""
        return Hypothesis(
            words=result.words,
            confidences=result.confidences,
            source=result.source_name,
        )


# =============================================================================
# High-Accuracy Decoder (ROVER Combination)
# =============================================================================


@dataclass
class HighAccuracyResult:
    """Result from high-accuracy ROVER decoding."""
    text: str  # Final combined transcription
    words: list[str]  # Word list
    confidences: list[float]  # Per-word confidences
    sources: dict[str, SourceResult]  # Individual source results
    rover_result: ROVERResult  # Full ROVER result with voting details


class HighAccuracyDecoder:
    """
    High-accuracy ASR decoder using ROVER voting.

    Combines multiple ASR sources (Transducer, CTC, Whisper) using
    ROVER voting algorithm to achieve <1.5% WER.

    The decoder runs all enabled sources in parallel and votes on
    the best word at each position.

    Usage:
        decoder = HighAccuracyDecoder(
            transducer_source=TransducerROVERSource(...),
            whisper_source=WhisperROVERSource(...),
        )
        result = decoder.transcribe("audio.wav")
        print(result.text)

    Attributes:
        sources: List of enabled ROVER sources
        rover: ROVER voting instance
    """

    def __init__(
        self,
        transducer_source: TransducerROVERSource | None = None,
        ctc_source: CTCROVERSource | None = None,
        whisper_source: WhisperROVERSource | None = None,
        rover_config: ROVERConfig | None = None,
    ):
        """
        Initialize high-accuracy decoder.

        Args:
            transducer_source: Transducer (RNN-T) source
            ctc_source: CTC draft source
            whisper_source: Whisper fallback source
            rover_config: Configuration for ROVER voting
        """
        self._transducer = transducer_source
        self._ctc = ctc_source
        self._whisper = whisper_source

        self._rover = ROVER(config=rover_config or ROVERConfig())

        # Build source list
        self._sources: list[ROVERSource] = []
        if transducer_source is not None:
            self._sources.append(transducer_source)
        if ctc_source is not None:
            self._sources.append(ctc_source)
        if whisper_source is not None:
            self._sources.append(whisper_source)

    @classmethod
    def from_whisper_only(
        cls,
        model_name: str = "large-v3",
        language: str | None = None,
    ) -> "HighAccuracyDecoder":
        """
        Create decoder with only Whisper (for testing/fallback).

        Args:
            model_name: Whisper model name
            language: Target language

        Returns:
            HighAccuracyDecoder with Whisper source only
        """
        whisper_source = WhisperROVERSource(
            model_name=model_name,
            language=language,
        )
        return cls(whisper_source=whisper_source)

    @classmethod
    def from_transducer_and_whisper(
        cls,
        transducer_source: TransducerROVERSource,
        whisper_model: str = "large-v3",
        language: str | None = None,
    ) -> "HighAccuracyDecoder":
        """
        Create decoder with Transducer + Whisper voting.

        Args:
            transducer_source: Pre-configured transducer source
            whisper_model: Whisper model name
            language: Target language

        Returns:
            HighAccuracyDecoder with two-source voting
        """
        whisper_source = WhisperROVERSource(
            model_name=whisper_model,
            language=language,
        )
        return cls(
            transducer_source=transducer_source,
            whisper_source=whisper_source,
        )

    def transcribe(
        self,
        audio: str | np.ndarray | mx.array,
        **kwargs,
    ) -> HighAccuracyResult:
        """
        Transcribe audio using all enabled sources and ROVER voting.

        Args:
            audio: Audio file path or waveform

        Returns:
            HighAccuracyResult with combined transcription
        """
        if not self._sources:
            msg = "No ASR sources configured"
            raise ValueError(msg)

        # Run all sources
        source_results: dict[str, SourceResult] = {}
        encoder_output = None

        for source in self._sources:
            if isinstance(source, CTCROVERSource) and encoder_output is not None:
                # CTC can reuse encoder output from transducer
                result = source.transcribe(audio, encoder_output=encoder_output)
            else:
                result = source.transcribe(audio, **kwargs)

            source_results[source.name] = result

            # Cache encoder output for CTC
            if isinstance(source, TransducerROVERSource):
                encoder_output = result.metadata.get("encoder_output")

        # Convert to ROVER hypotheses
        hypotheses = [
            Hypothesis(
                words=r.words,
                confidences=r.confidences,
                source=r.source_name,
            )
            for r in source_results.values()
        ]

        # Run ROVER voting
        rover_result = self._rover.vote(hypotheses)

        # Build final result
        text = " ".join(rover_result.words)

        return HighAccuracyResult(
            text=text,
            words=rover_result.words,
            confidences=rover_result.confidences,
            sources=source_results,
            rover_result=rover_result,
        )

    def transcribe_with_source(
        self,
        audio: str | np.ndarray | mx.array,
        source_name: str,
        **kwargs,
    ) -> SourceResult:
        """
        Transcribe using only a specific source (bypass ROVER).

        Useful for debugging or when only one source is needed.

        Args:
            audio: Audio file path or waveform
            source_name: Name of source to use

        Returns:
            SourceResult from the specified source
        """
        for source in self._sources:
            if source.name == source_name:
                return source.transcribe(audio, **kwargs)

        msg = f"Source '{source_name}' not found. Available: {[s.name for s in self._sources]}"
        raise ValueError(msg)


# =============================================================================
# Convenience Functions
# =============================================================================


def combine_whisper_transducer(
    whisper_text: str,
    whisper_logprob: float,
    transducer_text: str,
    transducer_score: float,
) -> str:
    """
    Simple two-source combination using ROVER.

    Args:
        whisper_text: Whisper transcription
        whisper_logprob: Whisper avg_logprob
        transducer_text: Transducer transcription
        transducer_score: Transducer total score

    Returns:
        Combined transcription
    """
    rover = ROVER()

    whisper_words = whisper_text.split()
    transducer_words = transducer_text.split()

    # Convert scores to per-word confidences
    whisper_conf = max(0.01, min(0.99, np.exp(whisper_logprob)))
    transducer_conf = max(0.01, min(0.99, np.exp(transducer_score / max(len(transducer_words), 1))))

    result = rover.vote([
        Hypothesis(
            words=transducer_words,
            confidences=[transducer_conf] * len(transducer_words),
            source="transducer",
        ),
        Hypothesis(
            words=whisper_words,
            confidences=[whisper_conf] * len(whisper_words),
            source="whisper",
        ),
    ])

    return " ".join(result.words)
