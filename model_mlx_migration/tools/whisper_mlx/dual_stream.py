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
Dual-Stream Rich Audio Understanding - Phase 7 Implementation.

Implements the dual-stream architecture from UNIFIED_RICH_AUDIO_ARCHITECTURE.md:
- CTC path: Immediate output (~100ms latency), may revise
- Decoder path: Confirmed output (~500ms latency), higher quality

The consumer receives both streams and merges them using alignment_id to:
1. Display CTC immediately (provisional)
2. Apply Decoder confirmations (mark as final)
3. Apply Decoder diffs (update displayed text)
4. Handle backtracks (remove/replace tokens)

Key Components:
- StreamEvent: Event emitted on either stream (token, confirm, diff, backtrack)
- RichStreamConsumer: Reference implementation for merging dual streams
- RichDualPathStreamer: Orchestrates both paths with RichToken output

Usage:
    from tools.whisper_mlx.dual_stream import RichDualPathStreamer, RichStreamConsumer

    # Create streamer with model and rich heads
    streamer = RichDualPathStreamer(
        model=whisper_mlx,
        rich_head=rich_ctc_head,
    )

    # Create consumer to merge streams
    consumer = RichStreamConsumer()

    # Process audio
    async for event in streamer.process_stream(audio_source):
        consumer.handle_event(event)

        # Get current state
        text = consumer.get_display_text()
        confirmed = consumer.get_confirmed_text()
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np

if TYPE_CHECKING:
    from .confidence_calibration import ConfidenceCalibrator
    from .model import WhisperMLX
    from .rich_ctc_head import RichCTCHead, RichToken

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None


# =============================================================================
# Stream Event Types
# =============================================================================

class EventType(Enum):
    """Type of stream event."""
    TOKEN = "token"           # New token from CTC or decoder
    CONFIRM = "confirm"       # Decoder confirms CTC token
    DIFF = "diff"             # Decoder differs from CTC
    BACKTRACK = "backtrack"   # CTC revises previous output
    FINAL = "final"           # End of utterance


@dataclass
class StreamEvent:
    """
    Event emitted on either stream (CTC or Decoder).

    This is the core data structure for dual-stream communication.
    The consumer uses these events to build the merged output.
    """
    # Event metadata
    event_type: EventType          # Type of event
    alignment_id: str              # Unique ID to correlate CTC <-> Decoder
    stream: str                    # "ctc" or "decoder"
    timestamp_ms: float            # When this event was emitted

    # For TOKEN events
    token: RichToken | None = None

    # For CONFIRM events
    confirmed: bool = False

    # For DIFF events (decoder disagrees with CTC)
    diff: dict[str, Any] | None = None  # {"field": "token", "ctc": "their", "decoder": "there"}

    # For BACKTRACK events
    backtrack_to_id: str | None = None  # Alignment ID to backtrack to
    reason: str | None = None           # "low_confidence", "phoneme_mismatch", etc.


@dataclass
class TokenState:
    """State of a token in the consumer."""
    token: RichToken
    is_confirmed: bool = False
    is_provisional: bool = True
    ctc_version: RichToken | None = None
    decoder_version: RichToken | None = None


# =============================================================================
# Rich Stream Consumer
# =============================================================================

class RichStreamConsumer:
    """
    Reference implementation for merging CTC and Decoder streams.

    Responsibilities:
    1. Display CTC tokens immediately (provisional)
    2. Mark tokens as confirmed when decoder agrees
    3. Update tokens when decoder differs
    4. Handle backtracks by removing/replacing tokens

    Example:
        consumer = RichStreamConsumer()

        # Process events from dual streamer
        for event in events:
            consumer.handle_event(event)

        # Get merged output
        text = consumer.get_display_text()
        confirmed = consumer.get_confirmed_text()

        # Register callbacks for UI updates
        consumer.on_token_added = lambda t: ui.add_token(t)
        consumer.on_token_confirmed = lambda id: ui.mark_confirmed(id)
        consumer.on_token_updated = lambda id, t: ui.update_token(id, t)
        consumer.on_backtrack = lambda id: ui.remove_after(id)
    """

    def __init__(self):
        """Initialize the consumer."""
        # Token storage: alignment_id -> TokenState
        self._tokens: dict[str, TokenState] = {}

        # Ordered list of alignment IDs (for display order)
        self._token_order: list[str] = []

        # Confirmed token IDs
        self._confirmed: set = set()

        # Event history (for debugging/replay)
        self._event_history: list[StreamEvent] = []

        # Callbacks for UI integration
        self.on_token_added: Callable[[TokenState], None] | None = None
        self.on_token_confirmed: Callable[[str], None] | None = None
        self.on_token_updated: Callable[[str, TokenState], None] | None = None
        self.on_backtrack: Callable[[str], None] | None = None
        self.on_diff: Callable[[str, dict], None] | None = None

    def handle_event(self, event: StreamEvent) -> None:
        """
        Process a stream event.

        Args:
            event: StreamEvent from either CTC or Decoder stream
        """
        self._event_history.append(event)

        if event.event_type == EventType.TOKEN:
            self._handle_token(event)
        elif event.event_type == EventType.CONFIRM:
            self._handle_confirm(event)
        elif event.event_type == EventType.DIFF:
            self._handle_diff(event)
        elif event.event_type == EventType.BACKTRACK:
            self._handle_backtrack(event)
        elif event.event_type == EventType.FINAL:
            self._handle_final(event)

    def _handle_token(self, event: StreamEvent) -> None:
        """Handle new token event."""
        if event.token is None:
            return

        alignment_id = event.alignment_id

        if alignment_id not in self._tokens:
            # New token
            state = TokenState(
                token=event.token,
                is_confirmed=(event.stream == "decoder"),
                is_provisional=(event.stream == "ctc"),
            )

            if event.stream == "ctc":
                state.ctc_version = event.token
            else:
                state.decoder_version = event.token
                state.is_confirmed = True
                self._confirmed.add(alignment_id)

            self._tokens[alignment_id] = state
            self._token_order.append(alignment_id)

            if self.on_token_added:
                self.on_token_added(state)
        else:
            # Update existing token
            state = self._tokens[alignment_id]

            if event.stream == "decoder":
                state.decoder_version = event.token
                state.token = event.token  # Decoder takes precedence
                state.is_confirmed = True
                state.is_provisional = False
                self._confirmed.add(alignment_id)

                if self.on_token_confirmed:
                    self.on_token_confirmed(alignment_id)
            else:
                # CTC update (revision)
                state.ctc_version = event.token
                if not state.is_confirmed:
                    state.token = event.token

                    if self.on_token_updated:
                        self.on_token_updated(alignment_id, state)

    def _handle_confirm(self, event: StreamEvent) -> None:
        """Handle confirm event (decoder agrees with CTC)."""
        alignment_id = event.alignment_id

        if alignment_id in self._tokens:
            state = self._tokens[alignment_id]
            state.is_confirmed = True
            state.is_provisional = False
            self._confirmed.add(alignment_id)

            if self.on_token_confirmed:
                self.on_token_confirmed(alignment_id)

    def _handle_diff(self, event: StreamEvent) -> None:
        """Handle diff event (decoder disagrees with CTC)."""
        alignment_id = event.alignment_id

        if alignment_id in self._tokens and event.diff:
            state = self._tokens[alignment_id]

            # Apply diff to token
            if event.token is not None:
                state.decoder_version = event.token
                state.token = event.token  # Decoder wins

            state.is_confirmed = True
            state.is_provisional = False
            self._confirmed.add(alignment_id)

            if self.on_diff:
                self.on_diff(alignment_id, event.diff)

            if self.on_token_updated:
                self.on_token_updated(alignment_id, state)

    def _handle_backtrack(self, event: StreamEvent) -> None:
        """Handle backtrack event (CTC revises previous output)."""
        backtrack_id = event.backtrack_to_id

        if backtrack_id is None:
            return

        # Find index of backtrack point
        try:
            backtrack_idx = self._token_order.index(backtrack_id)
        except ValueError:
            return

        # Remove all tokens after backtrack point
        tokens_to_remove = self._token_order[backtrack_idx:]

        for token_id in tokens_to_remove:
            # Only remove if not confirmed by decoder
            if token_id not in self._confirmed:
                del self._tokens[token_id]
                self._token_order.remove(token_id)

        if self.on_backtrack:
            self.on_backtrack(backtrack_id)

    def _handle_final(self, event: StreamEvent) -> None:
        """Handle final event (end of utterance)."""
        # Mark all remaining tokens as final
        for token_id in self._token_order:
            if token_id not in self._confirmed:
                state = self._tokens[token_id]
                state.is_confirmed = True
                state.is_provisional = False
                self._confirmed.add(token_id)

    def get_display_text(self) -> str:
        """Get current display text (all tokens)."""
        tokens = []
        for alignment_id in self._token_order:
            if alignment_id in self._tokens:
                state = self._tokens[alignment_id]
                tokens.append(state.token.token)
        return "".join(tokens)

    def get_confirmed_text(self) -> str:
        """Get confirmed text only."""
        tokens = []
        for alignment_id in self._token_order:
            if alignment_id in self._confirmed and alignment_id in self._tokens:
                state = self._tokens[alignment_id]
                tokens.append(state.token.token)
        return "".join(tokens)

    def get_provisional_text(self) -> str:
        """Get provisional (unconfirmed) text."""
        tokens = []
        for alignment_id in self._token_order:
            if alignment_id not in self._confirmed and alignment_id in self._tokens:
                state = self._tokens[alignment_id]
                tokens.append(state.token.token)
        return "".join(tokens)

    def get_token_states(self) -> list[TokenState]:
        """Get all token states in order."""
        return [
            self._tokens[aid]
            for aid in self._token_order
            if aid in self._tokens
        ]

    def get_token_by_id(self, alignment_id: str) -> TokenState | None:
        """Get token state by alignment ID."""
        return self._tokens.get(alignment_id)

    def get_event_history(self) -> list[StreamEvent]:
        """Get event history (for debugging/replay)."""
        return self._event_history.copy()

    def reset(self) -> None:
        """Reset consumer state for new utterance."""
        self._tokens.clear()
        self._token_order.clear()
        self._confirmed.clear()
        self._event_history.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get consumer statistics."""
        return {
            "total_tokens": len(self._tokens),
            "confirmed_tokens": len(self._confirmed),
            "provisional_tokens": len(self._tokens) - len(self._confirmed),
            "total_events": len(self._event_history),
            "events_by_type": {
                t.value: sum(1 for e in self._event_history if e.event_type == t)
                for t in EventType
            },
        }


# =============================================================================
# Dual Stream Configuration
# =============================================================================

@dataclass
class RichDualStreamConfig:
    """Configuration for RichDualPathStreamer."""
    # Audio settings
    sample_rate: int = 16000

    # CTC path settings (fast, ~100ms latency)
    ctc_chunk_duration: float = 1.0     # CTC processes 1s chunks
    ctc_interval: float = 0.3           # Update every 300ms
    ctc_confidence_threshold: float = 0.7  # Below this, may backtrack

    # Decoder path settings (quality, ~500ms latency)
    decoder_chunk_duration: float = 3.0  # Decoder processes 3s chunks
    decoder_interval: float = 1.0        # Update every 1s

    # Phoneme verification for backtracking
    use_phoneme_verification: bool = True
    phoneme_mismatch_threshold: float = 0.3  # Backtrack if mismatch > this

    # Diff detection
    diff_on_text_mismatch: bool = True   # Emit diff when text differs
    diff_on_emotion_mismatch: bool = False  # Emit diff when emotion differs

    # Model settings
    language: str | None = None
    task: str = "transcribe"

    # Confidence calibration settings
    use_calibration: bool = True         # Apply confidence calibration
    conservative_calibration: bool = True  # Shift confidence down for streaming
    calibration_path: str | None = None  # Path to load calibration params


# =============================================================================
# Rich Dual Path Streamer
# =============================================================================

class RichDualPathStreamer:
    """
    Dual-path streaming with rich token output.

    Orchestrates both CTC and Decoder paths, emitting StreamEvents
    that can be consumed by RichStreamConsumer for merged output.

    Architecture:
        Audio -> Encoder -> CTC Path (fast) -> StreamEvent (TOKEN)
                        -> Decoder Path (quality) -> StreamEvent (CONFIRM/DIFF)

    Example:
        streamer = RichDualPathStreamer(model, rich_head, config)
        consumer = RichStreamConsumer()

        async for event in streamer.process_stream(audio_source):
            consumer.handle_event(event)

            # Update UI
            if event.event_type == EventType.TOKEN:
                ui.show_provisional(consumer.get_display_text())
            elif event.event_type == EventType.CONFIRM:
                ui.mark_confirmed(event.alignment_id)
    """

    def __init__(
        self,
        model: WhisperMLX,
        rich_head: RichCTCHead | None = None,
        config: RichDualStreamConfig | None = None,
        calibrator: ConfidenceCalibrator | None = None,
    ):
        """
        Initialize dual-path streamer.

        Args:
            model: WhisperMLX model (encoder + decoder)
            rich_head: RichCTCHead for rich outputs (optional)
            config: Streamer configuration
            calibrator: Confidence calibrator (optional)
        """
        self.model = model
        self.rich_head = rich_head
        self.config = config or RichDualStreamConfig()

        # Confidence calibration
        self._calibrator = calibrator
        if calibrator is None and self.config.calibration_path:
            self._load_calibration(self.config.calibration_path)

        # Audio buffers
        self._audio_buffer = np.array([], dtype=np.float32)
        self._total_audio_time: float = 0.0

        # Timing
        self._last_ctc_time: float = 0.0
        self._last_decoder_time: float = 0.0

        # Token tracking
        self._ctc_tokens: dict[str, RichToken] = {}  # alignment_id -> token
        self._pending_confirmations: list[str] = []    # IDs awaiting decoder confirmation
        self._prev_ctc_text: str = ""

        # Detected language
        self._detected_language: str | None = None

        # Import audio utilities
        from .audio import log_mel_spectrogram
        self._log_mel_spectrogram = log_mel_spectrogram

        # Import tokenizer
        from .tokenizer import get_whisper_tokenizer
        self._tokenizer = get_whisper_tokenizer(
            multilingual=model.is_multilingual,
            task=self.config.task,
        )

    def reset(self) -> None:
        """Reset for new streaming session."""
        self._audio_buffer = np.array([], dtype=np.float32)
        self._total_audio_time = 0.0
        self._last_ctc_time = 0.0
        self._last_decoder_time = 0.0
        self._ctc_tokens.clear()
        self._pending_confirmations.clear()
        self._prev_ctc_text = ""
        self._detected_language = None

    def _load_calibration(self, path: str) -> None:
        """Load confidence calibration from file."""
        try:
            from .confidence_calibration import ConfidenceCalibrator
            self._calibrator = ConfidenceCalibrator()
            self._calibrator.load(path)
        except (ImportError, FileNotFoundError, Exception) as e:
            import warnings
            warnings.warn(f"Could not load calibration from {path}: {e}", stacklevel=2)
            self._calibrator = None

    def _calibrate_token(self, token: RichToken) -> RichToken:
        """
        Apply confidence calibration to a token.

        Args:
            token: RichToken with raw confidence scores

        Returns:
            RichToken with calibrated confidence scores
        """
        if not self.config.use_calibration or self._calibrator is None:
            return token

        # Import calibration function
        # Calibrate each confidence field
        from dataclasses import replace

        from .confidence_calibration import calibrate_for_streaming

        calibrated_conf = calibrate_for_streaming(
            token.confidence,
            calibrator=self._calibrator,
            output_type="text",
            conservative=self.config.conservative_calibration,
        )

        calibrated_emotion_conf = calibrate_for_streaming(
            token.emotion_confidence,
            calibrator=self._calibrator,
            output_type="emotion",
            conservative=self.config.conservative_calibration,
        )

        # Calibrate phoneme confidences
        calibrated_phoneme_conf = None
        if token.phoneme_confidence:
            calibrated_phoneme_conf = [
                calibrate_for_streaming(
                    c,
                    calibrator=self._calibrator,
                    output_type="phoneme",
                    conservative=self.config.conservative_calibration,
                )
                for c in token.phoneme_confidence
            ]

        # Calibrate para confidence
        calibrated_para_conf = None
        if token.para_confidence is not None:
            calibrated_para_conf = calibrate_for_streaming(
                token.para_confidence,
                calibrator=self._calibrator,
                output_type="para",
                conservative=self.config.conservative_calibration,
            )

        return replace(
            token,
            confidence=calibrated_conf,
            emotion_confidence=calibrated_emotion_conf,
            phoneme_confidence=calibrated_phoneme_conf,
            para_confidence=calibrated_para_conf,
        )

    def set_calibrator(self, calibrator: ConfidenceCalibrator) -> None:
        """Set the confidence calibrator."""
        self._calibrator = calibrator

    async def process_stream(
        self,
        audio_source: AsyncIterator[np.ndarray],
    ) -> AsyncIterator[StreamEvent]:
        """
        Process streaming audio and emit stream events.

        Args:
            audio_source: Async iterator yielding audio chunks

        Yields:
            StreamEvent objects for consumer to process
        """
        self.reset()

        async for chunk in audio_source:
            async for event in self._process_chunk(chunk):
                yield event

        # Final events
        async for event in self._finalize():
            yield event

    async def _process_chunk(
        self,
        audio: np.ndarray,
    ) -> AsyncIterator[StreamEvent]:
        """Process an audio chunk through both paths."""
        # Normalize audio
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Accumulate audio
        self._audio_buffer = np.concatenate([self._audio_buffer, audio])
        self._total_audio_time += len(audio) / self.config.sample_rate

        current_duration = len(self._audio_buffer) / self.config.sample_rate

        # Check CTC path
        time_since_ctc = self._total_audio_time - self._last_ctc_time
        if current_duration >= self.config.ctc_chunk_duration and \
           time_since_ctc >= self.config.ctc_interval:
            async for event in self._run_ctc_path():
                yield event
            self._last_ctc_time = self._total_audio_time

        # Check Decoder path
        time_since_decoder = self._total_audio_time - self._last_decoder_time
        if current_duration >= self.config.decoder_chunk_duration and \
           time_since_decoder >= self.config.decoder_interval:
            async for event in self._run_decoder_path():
                yield event
            self._last_decoder_time = self._total_audio_time

    async def _run_ctc_path(self) -> AsyncIterator[StreamEvent]:
        """Run CTC path and emit token events."""
        import asyncio

        if self.rich_head is None:
            return

        # Get audio chunk
        chunk_samples = int(self.config.ctc_chunk_duration * self.config.sample_rate)
        audio_chunk = self._audio_buffer[-chunk_samples:] if len(self._audio_buffer) > chunk_samples \
                      else self._audio_buffer

        # Convert to mel spectrogram
        mel = self._log_mel_spectrogram(audio_chunk, n_mels=self.model.config.n_mels)
        mel_tensor = mx.expand_dims(mx.array(mel), axis=0)

        # Run encoder
        encoder_output = await asyncio.to_thread(
            lambda: self.model.encoder(mel_tensor, variable_length=True),
        )
        mx.eval(encoder_output)

        # Run rich head
        outputs = self.rich_head(encoder_output)
        mx.eval(outputs)

        # Decode text
        tokens, tokens_with_timing = self.rich_head.decode_text_greedy(outputs)

        # Get text
        text = self._tokenizer.decode(tokens) if tokens else ""

        # Check for backtrack
        if self._should_backtrack(text, outputs):
            # Find backtrack point
            backtrack_id = self._find_backtrack_point(text)
            if backtrack_id:
                yield StreamEvent(
                    event_type=EventType.BACKTRACK,
                    alignment_id=str(uuid.uuid4())[:8],
                    stream="ctc",
                    timestamp_ms=time.time() * 1000,
                    backtrack_to_id=backtrack_id,
                    reason="phoneme_mismatch" if self.config.use_phoneme_verification else "low_confidence",
                )

        # Emit token events
        from .rich_ctc_head import outputs_to_rich_tokens

        rich_tokens = outputs_to_rich_tokens(
            outputs=outputs,
            token_ids=tokens,
            tokenizer=self._tokenizer,
            language=self._detected_language or "en",
            stream="ctc",
        )

        for rich_token in rich_tokens:
            # Skip if already emitted
            if rich_token.alignment_id in self._ctc_tokens:
                continue

            # Apply confidence calibration
            calibrated_token = self._calibrate_token(rich_token)

            self._ctc_tokens[calibrated_token.alignment_id] = calibrated_token
            self._pending_confirmations.append(calibrated_token.alignment_id)

            yield StreamEvent(
                event_type=EventType.TOKEN,
                alignment_id=calibrated_token.alignment_id,
                stream="ctc",
                timestamp_ms=time.time() * 1000,
                token=calibrated_token,
            )

        self._prev_ctc_text = text

    async def _run_decoder_path(self) -> AsyncIterator[StreamEvent]:
        """Run decoder path and emit confirm/diff events."""
        import asyncio

        # Get audio chunk
        chunk_samples = int(self.config.decoder_chunk_duration * self.config.sample_rate)
        audio_chunk = self._audio_buffer[-chunk_samples:] if len(self._audio_buffer) > chunk_samples \
                      else self._audio_buffer

        # Run full transcription
        result = await asyncio.to_thread(
            self.model.transcribe,
            audio_chunk,
            language=self.config.language or self._detected_language,
            task=self.config.task,
        )

        decoder_text = result.get("text", "").strip()

        # Cache detected language
        if self._detected_language is None and result.get("language"):
            self._detected_language = result["language"]

        # Compare with CTC tokens and emit confirm/diff events
        for alignment_id in list(self._pending_confirmations):
            if alignment_id not in self._ctc_tokens:
                continue

            ctc_token = self._ctc_tokens[alignment_id]

            # Simple text comparison (could be more sophisticated)
            if ctc_token.token in decoder_text:
                # Confirm
                yield StreamEvent(
                    event_type=EventType.CONFIRM,
                    alignment_id=alignment_id,
                    stream="decoder",
                    timestamp_ms=time.time() * 1000,
                    confirmed=True,
                )
            else:
                # Diff - decoder disagrees
                yield StreamEvent(
                    event_type=EventType.DIFF,
                    alignment_id=alignment_id,
                    stream="decoder",
                    timestamp_ms=time.time() * 1000,
                    diff={
                        "field": "token",
                        "ctc": ctc_token.token,
                        "decoder": decoder_text[:20],  # First 20 chars for context
                    },
                )

            self._pending_confirmations.remove(alignment_id)

    def _should_backtrack(self, text: str, outputs: dict[str, Any]) -> bool:
        """Check if CTC should backtrack based on confidence/phonemes."""
        if not self.config.use_phoneme_verification:
            return False

        # Get text confidence
        text_logits = outputs.get("text_logits")
        if text_logits is None:
            return False

        if text_logits.ndim == 3:
            text_logits = text_logits[0]

        probs = mx.softmax(text_logits, axis=-1)
        max_probs = mx.max(probs, axis=-1)
        mean_confidence = float(mx.mean(max_probs))

        return mean_confidence < self.config.ctc_confidence_threshold

    def _find_backtrack_point(self, new_text: str) -> str | None:
        """Find alignment ID to backtrack to."""
        # Simple implementation: find first differing token
        # Could be more sophisticated with alignment algorithms

        if not self._ctc_tokens:
            return None

        # Get ordered tokens
        token_ids = list(self._ctc_tokens.keys())

        # Find first token that might need revision
        for _i, token_id in enumerate(token_ids):
            token = self._ctc_tokens[token_id]
            if token.confidence < 0.5:
                return token_id

        return None

    async def _finalize(self) -> AsyncIterator[StreamEvent]:
        """Emit final events for remaining audio."""
        # Process remaining audio through decoder
        if len(self._audio_buffer) > 0:
            async for event in self._run_decoder_path():
                yield event

        # Emit final event
        yield StreamEvent(
            event_type=EventType.FINAL,
            alignment_id=str(uuid.uuid4())[:8],
            stream="decoder",
            timestamp_ms=time.time() * 1000,
        )


# =============================================================================
# Utility Functions
# =============================================================================

def create_dual_stream_pipeline(
    model: WhisperMLX,
    rich_head: RichCTCHead | None = None,
    config: RichDualStreamConfig | None = None,
) -> tuple[RichDualPathStreamer, RichStreamConsumer]:
    """
    Create a complete dual-stream pipeline.

    Returns streamer and consumer pair ready for use.

    Example:
        streamer, consumer = create_dual_stream_pipeline(model, rich_head)

        async for event in streamer.process_stream(audio):
            consumer.handle_event(event)
            print(consumer.get_display_text())
    """
    streamer = RichDualPathStreamer(model, rich_head, config)
    consumer = RichStreamConsumer()
    return streamer, consumer


def event_to_dict(event: StreamEvent) -> dict[str, Any]:
    """Convert StreamEvent to dictionary for serialization."""
    result = {
        "event_type": event.event_type.value,
        "alignment_id": event.alignment_id,
        "stream": event.stream,
        "timestamp_ms": event.timestamp_ms,
    }

    if event.token is not None:
        result["token"] = {
            "text": event.token.token,
            "token_id": event.token.token_id,
            "confidence": event.token.confidence,
            "emotion": event.token.emotion,
            "pitch_hz": event.token.pitch_hz,
            "start_time_ms": event.token.start_time_ms,
            "end_time_ms": event.token.end_time_ms,
        }

    if event.confirmed:
        result["confirmed"] = event.confirmed

    if event.diff:
        result["diff"] = event.diff

    if event.backtrack_to_id:
        result["backtrack_to_id"] = event.backtrack_to_id
        result["reason"] = event.reason

    return result


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Core types
    "EventType",
    "StreamEvent",
    "TokenState",

    # Consumer
    "RichStreamConsumer",

    # Streamer
    "RichDualPathStreamer",
    "RichDualStreamConfig",

    # Utilities
    "create_dual_stream_pipeline",
    "event_to_dict",
]
