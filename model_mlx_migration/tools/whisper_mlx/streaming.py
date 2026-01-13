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
WhisperMLX Streaming Transcription
==================================

Real-time streaming speech-to-text with:
- VAD-based speech endpoint detection
- Configurable chunk duration and overlap
- Partial (unstable) vs final (stable) results
- LocalAgreement algorithm for stable partial results
- Integration with WhisperMLX optimizations

Usage (async generator):
    async for result in streaming_whisper.transcribe_stream(audio_generator):
        if result.is_final:
            print(f"[FINAL] {result.text}")
        elif result.is_confirmed:
            # LocalAgreement confirmed this text won't change
            print(f"[CONFIRMED] {result.confirmed_text}")
        else:
            print(f"[PARTIAL] {result.text}")

Usage (callback-based):
    streaming_whisper.set_callback(on_transcription)
    await streaming_whisper.process_audio(audio_chunk)

LocalAgreement:
    The LocalAgreement-n algorithm only emits confirmed text when n consecutive
    transcriptions agree on a common prefix. This reduces "flickering" partial
    results while maintaining low latency for confirmed text.

    Example with n=2:
        - Transcription 1: "Hello wor" -> no output (only 1 sample)
        - Transcription 2: "Hello world" -> confirms "Hello wor"
        - Transcription 3: "Hello world, how" -> confirms "Hello world"

    Disable with: StreamingConfig(use_local_agreement=False)
"""
from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import WhisperMLX
import asyncio
import time

import numpy as np

try:
    import webrtcvad
    HAS_VAD = True
except ImportError:
    HAS_VAD = False

# CTC head for sub-200ms first partial latency
try:
    from .ctc_head import CTCDraftHead
    HAS_CTC = True
except ImportError:
    HAS_CTC = False
    CTCDraftHead = None  # type: ignore


class StreamState(Enum):
    """Current state of the streaming session."""
    IDLE = "idle"           # No speech detected, waiting
    SPEECH = "speech"       # Speech in progress
    PROCESSING = "processing"  # Processing accumulated audio
    FINALIZING = "finalizing"  # Processing remaining audio at end


class AlignAttPolicy:
    """
    J2: AlignAtt - Attention-guided streaming emission policy.

    Uses cross-attention weights from the decoder to determine when the model
    is confident enough to emit text. Instead of fixed chunking or agreement-only,
    AlignAtt looks at where the model is "paying attention" in the audio.

    Key insight:
    - If attention is concentrated on RECENT audio frames, the model is still
      "looking" for more context - don't emit yet
    - If attention is distributed across OLDER audio frames, the model is
      confident about the current output - safe to emit

    This provides word-aligned emission boundaries that naturally correspond
    to speech boundaries, often better than LocalAgreement for streaming.

    Example:
        policy = AlignAttPolicy(frame_threshold=0.5, recent_threshold=0.2)

        # During decoding, for each token:
        cross_attn_weights = decoder_output.cross_attention_weights
        if policy.should_emit(cross_attn_weights, token_idx=5):
            emit(tokens[:6])  # Emit up to and including token 5
        else:
            # Wait for more audio context
            pass

    Args:
        frame_threshold: Cumulative attention mass threshold (0-1).
            Default 0.5 means "where is 50% of attention mass located?"
        recent_threshold: What fraction of audio is considered "recent" (0-1).
            Default 0.2 means the last 20% of audio frames.
            If attention mass is concentrated in the recent region, don't emit.
        min_confidence: Minimum average attention confidence for emission.
            Default 0.3 - tokens with very diffuse attention are uncertain.
    """

    def __init__(
        self,
        frame_threshold: float = 0.5,
        recent_threshold: float = 0.2,
        min_confidence: float = 0.3,
    ):
        if not 0.0 < frame_threshold < 1.0:
            raise ValueError("frame_threshold must be in (0, 1)")
        if not 0.0 < recent_threshold < 1.0:
            raise ValueError("recent_threshold must be in (0, 1)")

        self.frame_threshold = frame_threshold
        self.recent_threshold = recent_threshold
        self.min_confidence = min_confidence

        # Track emitted token count for incremental emission
        self._emitted_count = 0

    def should_emit(
        self,
        cross_attention_weights: np.ndarray,
        current_token_idx: int,
    ) -> bool:
        """
        Determine if model is confident enough to emit current token.

        Args:
            cross_attention_weights: Shape (batch, n_heads, text_len, audio_len)
                or (n_heads, text_len, audio_len). Cross-attention from decoder.
            current_token_idx: Index of token being evaluated for emission.

        Returns:
            True if attention is stable enough to emit this token.
        """
        # Handle batch dimension
        if len(cross_attention_weights.shape) == 4:
            # Take first batch item
            weights = cross_attention_weights[0]
        else:
            weights = cross_attention_weights

        n_heads, text_len, audio_len = weights.shape

        if current_token_idx >= text_len:
            return False

        # Average attention across heads for current token
        # Shape: (audio_len,)
        attn = weights[:, current_token_idx, :].mean(axis=0)

        # Check minimum confidence - if attention is too diffuse, wait
        max_attn = float(np.max(attn))
        if max_attn < self.min_confidence:
            return False

        # Find where attention mass is concentrated
        # cumsum gives cumulative probability distribution
        attn_normalized = attn / (attn.sum() + 1e-8)
        cumsum = np.cumsum(attn_normalized)

        # Find the frame where cumulative mass exceeds threshold
        mass_frame_indices = np.where(cumsum >= self.frame_threshold)[0]
        if len(mass_frame_indices) == 0:
            # All attention on last frames - not confident
            return False

        mass_frame = mass_frame_indices[0]

        # Calculate boundary for "recent" audio
        recent_boundary = int(audio_len * (1.0 - self.recent_threshold))

        # If attention mass is concentrated before the recent region,
        # the model is confident (not still looking at recent audio)
        return bool(mass_frame < recent_boundary)

    def get_emit_boundary(
        self,
        cross_attention_weights: np.ndarray,
    ) -> int:
        """
        Find the token index where we should stop emitting.

        Scans through tokens and finds where the model becomes uncertain
        (attention shifts to recent audio frames).

        Args:
            cross_attention_weights: Shape (batch, n_heads, text_len, audio_len)
                or (n_heads, text_len, audio_len)

        Returns:
            Number of tokens that can be safely emitted.
            0 means nothing should be emitted yet.
        """
        # Handle batch dimension
        if len(cross_attention_weights.shape) == 4:
            weights = cross_attention_weights[0]
        else:
            weights = cross_attention_weights

        n_heads, text_len, audio_len = weights.shape

        emit_until = 0
        for i in range(text_len):
            if self.should_emit(weights, i):
                emit_until = i + 1
            else:
                # Stop at first uncertain token
                break

        return emit_until

    def update_with_weights(
        self,
        text: str,
        cross_attention_weights: np.ndarray,
        tokenizer=None,
    ) -> str:
        """
        Update with new transcription and attention weights, return emittable text.

        This is the main interface for streaming integration. Takes full text
        and attention weights, returns the portion that can be safely emitted.

        Args:
            text: Full transcription text (all tokens decoded so far)
            cross_attention_weights: Attention weights from decoder
            tokenizer: Optional tokenizer for word-level boundary detection

        Returns:
            Text that can be safely emitted (may be empty if uncertain)
        """
        emit_boundary = self.get_emit_boundary(cross_attention_weights)

        if emit_boundary <= self._emitted_count:
            return ""

        # For now, use character-level approximation
        # (proper implementation would use tokenizer for word boundaries)
        # Emit proportionally based on token boundary
        if len(cross_attention_weights.shape) == 4:
            text_len = cross_attention_weights.shape[2]
        else:
            text_len = cross_attention_weights.shape[1]

        if text_len == 0:
            return ""

        # Approximate character position from token position
        chars_per_token = len(text) / text_len if text_len > 0 else 0
        emit_chars = int(emit_boundary * chars_per_token)

        # Find word boundary
        emit_text = text[:emit_chars]
        space_idx = emit_text.rfind(" ")
        if space_idx > 0:
            emit_text = emit_text[:space_idx + 1]

        # Track what we've emitted
        prev_emitted_chars = int(self._emitted_count * chars_per_token)
        new_text = emit_text[prev_emitted_chars:]
        self._emitted_count = emit_boundary

        return new_text

    def reset(self) -> None:
        """Reset for new segment."""
        self._emitted_count = 0


class LocalAgreement:
    """
    LocalAgreement-n policy for stable streaming transcription.

    Only output text when n consecutive transcriptions produce the same prefix.
    This reduces "flickering" partial results while maintaining low latency
    for confirmed text.

    Based on research from whisper_streaming (IWSLT 2025).

    Example:
        agreement = LocalAgreement(n=2)

        # First transcription: "Hello wor" -> returns ""
        confirmed = agreement.update("Hello wor")

        # Second transcription: "Hello world" -> returns ""
        # (no common prefix agreement yet)
        confirmed = agreement.update("Hello world")

        # Third transcription: "Hello world, how" -> returns "Hello world"
        # (both recent transcriptions start with "Hello world")
        confirmed = agreement.update("Hello world, how")

    With min_stable_ms > 0 (timed commit):
        agreement = LocalAgreement(n=3, min_stable_ms=1500)

        # Prefix must have n-way agreement AND be stable for 1.5s before commit.
        # This trades latency for stability - reduces retractions on long audio.
    """

    def __init__(self, n: int = 2, min_stable_ms: float = 0.0):
        """
        Initialize LocalAgreement.

        Args:
            n: Number of consecutive agreements needed (default: 2)
            min_stable_ms: Minimum time prefix must be stable before commit (default: 0)
                          Set > 0 to trade latency for fewer retractions.
        """
        if n < 2:
            raise ValueError("n must be >= 2")
        self.n = n
        self.min_stable_ms = min_stable_ms
        self.history: list[str] = []
        self.committed: str = ""
        # Timed commit tracking: maps prefix -> first_seen_timestamp_ms
        self._pending_prefix: str = ""
        self._pending_since_ms: float = 0.0

    def update(self, new_transcript: str, current_time_ms: float | None = None) -> str:
        """
        Update with new transcription, return newly confirmed text.

        Args:
            new_transcript: Latest transcription result
            current_time_ms: Current wall clock time in ms (for timed commits).
                            If None and min_stable_ms > 0, uses time.time() * 1000.

        Returns:
            Newly confirmed text (empty string if no agreement yet)
        """
        # Normalize whitespace
        new_transcript = new_transcript.strip()
        self.history.append(new_transcript)

        # Keep only last n transcripts
        if len(self.history) > self.n:
            self.history = self.history[-self.n:]

        if len(self.history) < self.n:
            return ""

        # Find common prefix among last n transcripts
        recent = self.history[-self.n:]
        common_prefix = self._longest_common_prefix(recent)

        # Only commit what's new beyond previously committed
        if len(common_prefix) > len(self.committed):
            # Timed commit logic: wait for prefix to be stable
            if self.min_stable_ms > 0:
                if current_time_ms is None:
                    current_time_ms = time.time() * 1000

                # Check if pending prefix is still compatible
                # Compatible means: new prefix extends or equals the pending prefix
                # Incompatible means: new prefix doesn't start with pending (prefix shrank/changed)
                if self._pending_prefix:
                    if common_prefix.startswith(self._pending_prefix):
                        # New prefix extends pending - keep the original timer
                        # This allows commit of stable portion even as text grows
                        pass
                    else:
                        # Prefix changed incompatibly - reset timer
                        self._pending_prefix = common_prefix
                        self._pending_since_ms = current_time_ms
                        return ""
                else:
                    # No pending prefix yet - start tracking
                    self._pending_prefix = common_prefix
                    self._pending_since_ms = current_time_ms
                    return ""

                # Check if stable long enough
                elapsed_ms = current_time_ms - self._pending_since_ms
                if elapsed_ms < self.min_stable_ms:
                    # Not stable enough yet
                    return ""

                # Prefix has been stable long enough - commit it
                # Commit up to the pending prefix (not common_prefix which may be longer)
                commit_text = self._pending_prefix
                if len(commit_text) > len(self.committed):
                    new_text = commit_text[len(self.committed):]
                    self.committed = commit_text
                    # Update pending to track the new longer prefix if any
                    if len(common_prefix) > len(commit_text):
                        self._pending_prefix = common_prefix
                        self._pending_since_ms = current_time_ms
                    else:
                        self._pending_prefix = ""
                        self._pending_since_ms = 0.0
                    return new_text
                return ""
            # No timed commit - commit immediately (original behavior)
            new_text = common_prefix[len(self.committed):]
            self.committed = common_prefix
            return new_text

        # If common prefix shrank or stayed same, reset pending tracking
        if self.min_stable_ms > 0:
            if not common_prefix.startswith(self._pending_prefix) if self._pending_prefix else False:
                # Prefix changed incompatibly
                self._pending_prefix = common_prefix if len(common_prefix) > len(self.committed) else ""
                self._pending_since_ms = (time.time() * 1000) if self._pending_prefix else 0.0

        return ""

    def get_confirmed(self) -> str:
        """Get all confirmed text so far."""
        return self.committed

    def get_speculative(self) -> str:
        """Get speculative (unconfirmed) text from latest transcription."""
        if not self.history:
            return ""
        latest = self.history[-1]
        if len(latest) > len(self.committed):
            return latest[len(self.committed):]
        return ""

    def reset(self) -> None:
        """Reset for new segment."""
        self.history = []
        self.committed = ""
        self._pending_prefix = ""
        self._pending_since_ms = 0.0

    def _longest_common_prefix(self, strings: list[str]) -> str:
        """
        Find longest common prefix among strings, ending at word boundary.

        Only commits complete words - a word is complete when it appears
        in the same position across ALL transcripts and is followed by
        whitespace (or is the end of the shortest string).
        """
        if not strings:
            return ""

        # Find character-level common prefix first
        prefix = strings[0]
        for s in strings[1:]:
            while not s.startswith(prefix):
                prefix = prefix[:-1]
                if not prefix:
                    return ""

        # Now back off to last complete word boundary
        # A word is "complete" if the prefix ends at whitespace
        # or if all strings have whitespace/end after this prefix
        if not prefix:
            return ""

        # Check if prefix ends at a word boundary in ALL strings
        def is_word_boundary_in_all(prefix_len: int) -> bool:
            for s in strings:
                if prefix_len < len(s) and not s[prefix_len].isspace():
                    return False
            return True

        # If current prefix end is not a word boundary, back off
        if not is_word_boundary_in_all(len(prefix)):
            # Find last space in prefix
            space_idx = prefix.rfind(" ")
            if space_idx > 0:
                prefix = prefix[:space_idx]
            else:
                # No word boundary found, don't commit partial word
                return ""

        return prefix


@dataclass
class StreamingResult:
    """Result from streaming transcription."""
    text: str                    # Transcribed text
    is_final: bool              # True if this is a stable final result
    is_partial: bool            # True if this is an unstable partial result
    segment_start: float        # Start time in audio stream (seconds)
    segment_end: float          # End time in audio stream (seconds)
    language: str | None = None
    confidence: float = 1.0     # Confidence estimate (0-1)

    # Timing metadata
    processing_time: float = 0.0  # Time to transcribe this chunk
    audio_duration: float = 0.0   # Duration of audio transcribed

    # LocalAgreement fields (when use_local_agreement=True)
    confirmed_text: str = ""     # Text confirmed by LocalAgreement
    speculative_text: str = ""   # Text not yet confirmed (may change)
    is_confirmed: bool = False   # True if this result includes new confirmed text

    @property
    def rtf(self) -> float:
        """Real-time factor (processing_time / audio_duration)."""
        if self.audio_duration > 0:
            return self.processing_time / self.audio_duration
        return 0.0


# P3 Latency Mode Constants (TTFO targets based on profiling)
LATENCY_MODES = {
    "fast": {
        "model": "small",
        "ttfo_ms": 172,
        "description": "Fastest TTFO (~170ms), good quality for English",
    },
    "balanced": {
        "model": "large-v3-turbo",
        "ttfo_ms": 477,
        "description": "P3 compliant (<500ms), near large-v3 quality",
    },
    "quality": {
        "model": "large-v3",
        "ttfo_ms": 800,
        "description": "Best quality, higher latency (~800ms)",
    },
}


@dataclass
class StreamingConfig:
    """Configuration for streaming transcription."""
    # Audio settings
    sample_rate: int = 16000           # Expected sample rate

    # VAD settings
    use_vad: bool = True               # Use VAD for endpoint detection
    vad_aggressiveness: int = 2        # 0-3, higher = more aggressive filtering
    vad_frame_duration_ms: int = 30    # VAD frame size (10, 20, or 30 ms)

    # Chunking settings
    min_chunk_duration: float = 0.5    # Minimum audio to process (seconds)
    max_chunk_duration: float = 10.0   # Maximum chunk before forced processing
    silence_threshold_duration: float = 0.5  # Silence duration to trigger endpoint

    # Context/overlap settings
    context_duration: float = 0.5      # Audio context from previous chunk

    # Output settings
    emit_partials: bool = True         # Emit partial results during processing
    partial_interval: float = 2.0      # Minimum interval between partials (seconds)
                                       # Increased from 0.5s to 2.0s for RTF optimization
                                       # Each partial re-encodes entire audio buffer

    # LocalAgreement settings (reduces flickering partial results)
    use_local_agreement: bool = True   # Use LocalAgreement for stable partials
    agreement_n: int = 2               # Number of consecutive agreements needed
    min_stable_ms: float = 0.0         # Min time prefix must be stable before commit (ms)
                                       # Set > 0 to trade latency for fewer retractions
                                       # Recommended: 1500ms for stable balanced mode
    commit_partials: bool = True       # Emit confirmed_text during partials
                                       # Set False to only commit on finals (0 retractions)
                                       # User sees speculative text during speech

    # Model settings (passed to WhisperMLX)
    language: str | None = None     # Language code or None for auto-detect
    task: str = "transcribe"           # "transcribe" or "translate"

    # J9: Initial prompt for terminology injection
    initial_prompt: str | None = None  # Domain-specific vocabulary hints

    # P3 Latency Mode (see P3_TTFO_ANALYSIS_2025-12-20.md)
    # Controls which model to use for streaming based on TTFO requirements
    # - "fast": small model (~170ms TTFO), good for English
    # - "balanced": large-v3-turbo (~477ms TTFO), P3 compliant, recommended default
    # - "quality": large-v3 (~800ms TTFO), best quality, fails P3 gate
    latency_mode: str = "balanced"

    # RTF Optimization (commits #1458+)
    # Enable encoder caching to avoid re-encoding same audio
    use_encoder_cache: bool = True     # Enable encoder caching in model
    encoder_cache_entries: int = 4     # Number of encoder outputs to cache

    @property
    def recommended_model(self) -> str:
        """Get recommended model name based on latency mode."""
        mode_info = LATENCY_MODES.get(self.latency_mode, LATENCY_MODES["balanced"])
        return mode_info["model"]

    @property
    def expected_ttfo_ms(self) -> int:
        """Get expected TTFO in milliseconds for this latency mode."""
        mode_info = LATENCY_MODES.get(self.latency_mode, LATENCY_MODES["balanced"])
        return mode_info["ttfo_ms"]


# Streaming configuration presets for different RTF/latency tradeoffs
STREAMING_PRESETS = {
    "low_latency": {
        # Optimized for lowest first partial latency
        # Use with small model (latency_mode="fast") for ~1260ms first partial
        # Fundamental limit: Whisper needs ~1s of audio context for BPE output
        # To achieve <200ms, use CTC head for speculative early output
        "emit_partials": True,
        "partial_interval": 1.0,  # Fast first partial trigger
        "use_encoder_cache": True,
        "encoder_cache_entries": 4,
        "latency_mode": "fast",  # Recommend small model
        "agreement_n": 2,  # Lower n for faster confirmation in low-latency mode
        "description": "Low latency: ~1260ms first partial with small model",
    },
    "realtime": {
        # Optimized for RTF < 1.0 (real-time capable)
        # Sacrifices partial feedback for speed
        "emit_partials": False,
        "partial_interval": 10.0,  # No partials (only finals)
        "use_encoder_cache": True,
        "encoder_cache_entries": 2,
        "agreement_n": 2,  # Not critical since partials disabled
        "description": "Real-time optimized: RTF < 1.0, no partials",
    },
    "balanced": {
        # Balance between RTF and user feedback
        # Partials every 2s, should achieve RTF ~1.5-2.0
        # n=3 reduces retractions by 58% with no WER impact
        "emit_partials": True,
        "partial_interval": 2.0,
        "use_encoder_cache": True,
        "encoder_cache_entries": 4,
        "agreement_n": 3,  # Reduces retractions by 58% vs n=2
        "min_stable_ms": 0.0,  # No timed commit delay (default)
        "description": "Balanced: RTF ~1.5-2.0, partials every 2s, n=3 for stability",
    },
    "stable": {
        # Maximum stability: minimize retractions at cost of higher commit latency
        # Uses timed commits: prefix must be stable for 1.5s before commit
        # Best for long-form transcription where retractions break user trust
        "emit_partials": True,
        "partial_interval": 2.0,
        "use_encoder_cache": True,
        "encoder_cache_entries": 4,
        "agreement_n": 3,
        "min_stable_ms": 1500.0,  # 1.5s stability required before commit
        "description": "Stable: Maximum stability, 1.5s timed commit delay",
    },
    "no_retract": {
        # Zero retractions guaranteed: only commit on final results
        # All text during speech is speculative (may change)
        # Commits only after silence detection (endpoint)
        # Best for applications where retractions are unacceptable
        "emit_partials": True,
        "partial_interval": 2.0,
        "use_encoder_cache": True,
        "encoder_cache_entries": 4,
        "agreement_n": 2,  # Agreement still used internally, not exposed
        "commit_partials": False,  # Key: don't commit during partials
        "description": "No-retract: 0 retractions guaranteed, commits only on finals",
    },
    "responsive": {
        # Prioritize user feedback over RTF
        # More frequent partials, higher RTF
        # n=3 for stability without latency penalty
        "emit_partials": True,
        "partial_interval": 1.0,
        "use_encoder_cache": True,
        "encoder_cache_entries": 8,
        "agreement_n": 3,  # Reduces retractions without latency penalty
        "description": "Responsive: RTF ~3-4, partials every 1s, n=3 for stability",
    },
    "legacy": {
        # Original settings (before RTF optimization)
        # Very frequent partials, highest RTF
        "emit_partials": True,
        "partial_interval": 0.5,
        "use_encoder_cache": False,
        "encoder_cache_entries": 0,
        "agreement_n": 2,  # Original behavior
        "description": "Legacy: RTF ~5+, partials every 0.5s",
    },
}


def get_streaming_config(preset: str = "balanced", **overrides) -> StreamingConfig:
    """
    Get a StreamingConfig with preset settings.

    Args:
        preset: One of "realtime", "balanced", "responsive", "legacy"
        **overrides: Override specific config values

    Returns:
        StreamingConfig with preset values and any overrides applied

    Example:
        # Real-time mode
        config = get_streaming_config("realtime")

        # Balanced mode with custom language
        config = get_streaming_config("balanced", language="en")
    """
    if preset not in STREAMING_PRESETS:
        raise ValueError(f"Unknown preset '{preset}'. Available: {list(STREAMING_PRESETS.keys())}")

    preset_values = {k: v for k, v in STREAMING_PRESETS[preset].items() if k != "description"}
    preset_values.update(overrides)

    return StreamingConfig(**preset_values)


class AudioBuffer:
    """Circular buffer for streaming audio with efficient slicing."""

    def __init__(self, max_duration: float, sample_rate: int):
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)
        self.buffer = np.zeros(self.max_samples, dtype=np.float32)
        self.write_pos = 0
        self.total_samples_received = 0

    def append(self, audio: np.ndarray) -> None:
        """Append audio to buffer, wrapping if needed."""
        n_samples = len(audio)
        self.total_samples_received += n_samples

        if n_samples >= self.max_samples:
            # Audio larger than buffer - keep only the tail
            self.buffer = audio[-self.max_samples:].astype(np.float32)
            self.write_pos = 0
        else:
            # Check if we need to wrap
            end_pos = self.write_pos + n_samples
            if end_pos <= self.max_samples:
                # No wrap needed
                self.buffer[self.write_pos:end_pos] = audio.astype(np.float32)
                self.write_pos = end_pos
            else:
                # Wrap around
                first_part = self.max_samples - self.write_pos
                self.buffer[self.write_pos:] = audio[:first_part].astype(np.float32)
                remainder = audio[first_part:].astype(np.float32)
                self.buffer[:n_samples - first_part] = remainder
                self.write_pos = n_samples - first_part

    def get_audio(self, duration: float) -> np.ndarray:
        """Get the most recent audio of specified duration."""
        n_samples = min(int(duration * self.sample_rate), self.write_pos)
        if n_samples == 0:
            return np.array([], dtype=np.float32)
        return self.buffer[self.write_pos - n_samples:self.write_pos].copy()

    @property
    def duration(self) -> float:
        """Current buffer duration in seconds."""
        return self.write_pos / self.sample_rate

    def clear(self) -> None:
        """Clear the buffer."""
        self.write_pos = 0

    @property
    def total_time(self) -> float:
        """Total time of audio received since creation."""
        return self.total_samples_received / self.sample_rate


class StreamingWhisper:
    """
    Streaming speech-to-text using WhisperMLX.

    Features:
    - VAD-based endpoint detection for natural sentence boundaries
    - Configurable chunk duration limits
    - Partial results during long utterances
    - Context overlap for better accuracy at chunk boundaries
    - Async generator and callback interfaces

    Example:
        from tools.whisper_mlx import WhisperMLX
        from tools.whisper_mlx.streaming import StreamingWhisper, StreamingConfig

        model = WhisperMLX.from_pretrained("large-v3")
        config = StreamingConfig(emit_partials=True)
        streamer = StreamingWhisper(model, config)

        async for result in streamer.transcribe_stream(audio_generator):
            print(f"[{'FINAL' if result.is_final else 'PARTIAL'}] {result.text}")
    """

    def __init__(
        self,
        model: WhisperMLX,  # Forward reference - imported at runtime
        config: StreamingConfig | None = None,
    ):
        """
        Initialize streaming transcription.

        Args:
            model: Initialized WhisperMLX model
            config: Streaming configuration (uses defaults if None)
        """
        self.model = model
        self.config = config or StreamingConfig()

        # RTF Optimization: Enable encoder caching
        # This avoids re-encoding when transcribing identical audio
        if self.config.use_encoder_cache:
            if hasattr(model, 'enable_encoder_cache') and not model.encoder_cache_enabled:
                model.enable_encoder_cache(max_entries=self.config.encoder_cache_entries)

        # VAD setup
        if self.config.use_vad:
            if not HAS_VAD:
                raise ImportError(
                    "webrtcvad required for VAD. Install: pip install webrtcvad",
                )
            self.vad = webrtcvad.Vad(self.config.vad_aggressiveness)
        else:
            self.vad = None

        # LocalAgreement setup
        if self.config.use_local_agreement:
            self._local_agreement: LocalAgreement | None = LocalAgreement(
                n=self.config.agreement_n,
                min_stable_ms=self.config.min_stable_ms,
            )
        else:
            self._local_agreement = None

        # State
        self.state = StreamState.IDLE
        self._reset_state()

        # Callback (optional)
        self._callback: Callable[[StreamingResult], None] | None = None

    def _reset_state(self) -> None:
        """Reset internal state for new session."""
        self.state = StreamState.IDLE

        # Audio accumulation
        cfg = self.config
        max_duration = cfg.max_chunk_duration + cfg.context_duration + 1.0
        self._audio_buffer = AudioBuffer(max_duration, self.config.sample_rate)
        self._speech_buffer = np.array([], dtype=np.float32)  # Current speech segment

        # Timing
        self._segment_start_time = 0.0
        self._last_partial_time = 0.0
        self._silence_frames = 0
        self._speech_frames = 0

        # Context from previous chunk
        self._context_audio: np.ndarray | None = None

        # Detected language (cached after first detection)
        self._detected_language: str | None = None

        # Reset LocalAgreement for new segment
        if hasattr(self, "_local_agreement") and self._local_agreement is not None:
            self._local_agreement.reset()

    def set_callback(self, callback: Callable[[StreamingResult], None]) -> None:
        """Set callback for transcription results."""
        self._callback = callback

    def reset(self) -> None:
        """Reset for new streaming session."""
        self._reset_state()
        self._detected_language = None

    async def transcribe_stream(
        self,
        audio_source: AsyncIterator[np.ndarray],
    ) -> AsyncIterator[StreamingResult]:
        """
        Transcribe streaming audio source.

        Args:
            audio_source: Async iterator yielding audio chunks (float32, 16kHz mono)

        Yields:
            StreamingResult objects with transcription results
        """
        self._reset_state()

        async for chunk in audio_source:
            # Process chunk and yield any results
            async for result in self._process_chunk(chunk):
                yield result

        # Finalize - process any remaining audio
        async for result in self._finalize():
            yield result

    async def _process_chunk(self, audio: np.ndarray) -> AsyncIterator[StreamingResult]:
        """Process an audio chunk and yield results if ready."""
        # Normalize to float32 if needed
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Add to buffer
        self._audio_buffer.append(audio)

        # Run VAD if enabled
        if self.vad is not None:
            is_speech = self._check_vad(audio)
        else:
            # Without VAD, assume all audio is speech
            is_speech = True

        # State machine
        if self.state == StreamState.IDLE:
            if is_speech:
                # Speech started
                self.state = StreamState.SPEECH
                audio_len = len(audio) / self.config.sample_rate
                self._segment_start_time = self._audio_buffer.total_time - audio_len
                self._speech_buffer = audio.copy()
                self._silence_frames = 0
                self._speech_frames = 1

        elif self.state == StreamState.SPEECH:
            # Accumulate speech
            self._speech_buffer = np.concatenate([self._speech_buffer, audio])

            if is_speech:
                self._speech_frames += 1
                self._silence_frames = 0
            else:
                self._silence_frames += 1

            # Check for endpoint conditions
            current_duration = len(self._speech_buffer) / self.config.sample_rate
            silence_duration = (
                self._silence_frames * len(audio) / self.config.sample_rate
            )

            # Endpoint detected (silence threshold)
            if silence_duration >= self.config.silence_threshold_duration:
                async for result in self._process_segment(is_final=True):
                    yield result
                self.state = StreamState.IDLE

            # Max duration reached - forced segment break
            elif current_duration >= self.config.max_chunk_duration:
                # CRITICAL: Emit as FINAL to preserve the transcription!
                # Otherwise, this content is lost when buffer is cleared.
                # The evaluation harness only collects FINAL results.
                async for result in self._process_segment(is_final=True):
                    yield result

                # Reset for next segment, keeping context overlap
                ctx_dur = self.config.context_duration
                context_samples = int(ctx_dur * self.config.sample_rate)
                if len(self._speech_buffer) > context_samples:
                    self._context_audio = self._speech_buffer[-context_samples:].copy()
                    # Clear buffer and reset LocalAgreement for new segment
                    self._speech_buffer = np.array([], dtype=np.float32)
                else:
                    self._context_audio = self._speech_buffer.copy()
                    self._speech_buffer = np.array([], dtype=np.float32)

                # Update segment start time for next segment
                self._segment_start_time = self._audio_buffer.total_time

            # Emit partial result if configured and enough time passed
            elif self.config.emit_partials:
                buf = self._audio_buffer
                since_partial = buf.total_time - self._last_partial_time
                min_dur = self.config.min_chunk_duration
                interval = self.config.partial_interval
                if current_duration >= min_dur and since_partial >= interval:
                    async for result in self._process_segment(
                        is_final=False, is_partial=True,
                    ):
                        yield result
                    self._last_partial_time = buf.total_time

    async def _finalize(self) -> AsyncIterator[StreamingResult]:
        """Process any remaining audio at end of stream."""
        self.state = StreamState.FINALIZING

        min_samples = int(self.config.min_chunk_duration * self.config.sample_rate)
        if len(self._speech_buffer) >= min_samples:
            async for result in self._process_segment(is_final=True):
                yield result

        self._reset_state()

    async def _process_segment(
        self,
        is_final: bool,
        is_partial: bool = False,
    ) -> AsyncIterator[StreamingResult]:
        """Transcribe accumulated speech segment."""
        min_samples = int(self.config.min_chunk_duration * self.config.sample_rate)
        if len(self._speech_buffer) < min_samples:
            return

        audio_to_transcribe = self._speech_buffer

        # Add context from previous chunk if available
        if self._context_audio is not None and not is_partial:
            audio_to_transcribe = np.concatenate(
                [self._context_audio, self._speech_buffer],
            )

        audio_duration = len(audio_to_transcribe) / self.config.sample_rate

        # Transcribe
        start_time = time.perf_counter()

        # Use model's transcribe method
        result = self.model.transcribe(
            audio_to_transcribe,
            language=self.config.language or self._detected_language,
            task=self.config.task,
            variable_length=False,  # Use standard mode for accuracy
            initial_prompt=self.config.initial_prompt,  # J9: Terminology injection
        )

        processing_time = time.perf_counter() - start_time

        # Cache detected language
        if self._detected_language is None and result.get("language"):
            self._detected_language = result["language"]

        # Extract text, removing context portion if applicable
        text = result.get("text", "").strip()

        if text:
            # Apply LocalAgreement for partial results
            confirmed_text = ""
            speculative_text = ""
            is_confirmed = False

            if is_partial and self._local_agreement is not None:
                # Update LocalAgreement with new partial text
                newly_confirmed = self._local_agreement.update(text)

                if self.config.commit_partials:
                    # Normal behavior: show confirmed text during partials
                    confirmed_text = self._local_agreement.get_confirmed()
                    speculative_text = self._local_agreement.get_speculative()
                    is_confirmed = len(newly_confirmed) > 0
                else:
                    # No-commit mode: all text is speculative during partials
                    # This guarantees 0 retractions (confirmed never changes)
                    confirmed_text = ""
                    speculative_text = text
                    is_confirmed = False
            elif is_final and not is_partial:
                # Final results are fully confirmed
                confirmed_text = text
                speculative_text = ""
                is_confirmed = True
                # Reset LocalAgreement after final segment
                if self._local_agreement is not None:
                    self._local_agreement.reset()

            streaming_result = StreamingResult(
                text=text,
                is_final=is_final and not is_partial,
                is_partial=is_partial,
                segment_start=self._segment_start_time,
                segment_end=self._audio_buffer.total_time,
                language=result.get("language"),
                processing_time=processing_time,
                audio_duration=audio_duration,
                confirmed_text=confirmed_text,
                speculative_text=speculative_text,
                is_confirmed=is_confirmed,
            )

            if self._callback:
                self._callback(streaming_result)

            yield streaming_result

        # Clear speech buffer if final
        if is_final and not is_partial:
            self._speech_buffer = np.array([], dtype=np.float32)
            self._context_audio = None

    def _check_vad(self, audio: np.ndarray) -> bool:
        """Check if audio chunk contains speech using VAD."""
        if self.vad is None:
            return True

        # VAD requires 16-bit PCM
        pcm = (audio * 32767).astype(np.int16)

        # Process in VAD frames
        sr = self.config.sample_rate
        frame_size = int(sr * self.config.vad_frame_duration_ms / 1000)
        n_frames = len(pcm) // frame_size

        if n_frames == 0:
            return False

        speech_frames = 0
        for i in range(n_frames):
            frame = pcm[i * frame_size:(i + 1) * frame_size]
            if self.vad.is_speech(frame.tobytes(), self.config.sample_rate):
                speech_frames += 1

        # Consider speech if >50% of frames are speech
        return speech_frames > n_frames // 2


# Synchronous wrapper for non-async contexts
class SyncStreamingWhisper:
    """
    Synchronous streaming transcription wrapper.

    For use in non-async contexts (e.g., callbacks from audio libraries).
    """

    def __init__(
        self,
        model: WhisperMLX,
        config: StreamingConfig | None = None,
    ):
        self._async_streamer = StreamingWhisper(model, config)
        self._results_queue: list[StreamingResult] = []

    def process_audio(self, audio: np.ndarray) -> list[StreamingResult]:
        """
        Process audio chunk and return any results.

        Args:
            audio: Audio chunk (float32, 16kHz mono)

        Returns:
            List of StreamingResult objects (empty if no results ready)
        """
        # Run async processing in event loop
        async def process():
            return [result async for result in self._async_streamer._process_chunk(audio)]

        return asyncio.run(process())

    def finalize(self) -> list[StreamingResult]:
        """Finalize and return any remaining results."""
        async def process():
            return [result async for result in self._async_streamer._finalize()]

        return asyncio.run(process())

    def reset(self) -> None:
        """Reset for new session."""
        self._async_streamer.reset()


@dataclass
class DualPathConfig:
    """Configuration for dual-path streaming transcription."""
    # Audio settings
    sample_rate: int = 16000

    # Fast path settings (instant speculative output)
    fast_chunk_duration: float = 1.0      # Fast path chunk size (seconds)
    fast_latency_target_ms: float = 200   # Target latency for fast path

    # Quality path settings (confirmed output)
    quality_chunk_duration: float = 5.0   # Quality path chunk size (seconds)
    quality_overlap: float = 1.0          # Overlap for quality path chunks
    agreement_n: int = 2                  # LocalAgreement threshold

    # VAD settings
    use_vad: bool = True
    vad_aggressiveness: int = 2

    # Model settings
    language: str | None = None
    task: str = "transcribe"

    # J9: Initial prompt for terminology injection
    initial_prompt: str | None = None  # Domain-specific vocabulary hints


@dataclass
class DualPathResult:
    """Result from dual-path streaming transcription."""
    # Speculative (fast path) output - may change
    speculative_text: str
    speculative_is_new: bool

    # Confirmed (quality path) output - stable
    confirmed_text: str
    confirmed_is_new: bool

    # Full accumulated text
    full_speculative: str  # All speculative text so far
    full_confirmed: str    # All confirmed text so far

    # Timing
    fast_latency_ms: float = 0.0
    quality_latency_ms: float = 0.0
    audio_time: float = 0.0


class DualPathStreamer:
    """
    Dual-path streaming: instant speculative + quality-confirmed.

    Architecture:
    - Fast path: 1s chunks, greedy decode, ~200ms latency
    - Quality path: 5s chunks with LocalAgreement, ~1.5s latency

    The fast path provides immediate feedback that may change.
    The quality path provides stable, confirmed text.

    Example:
        model = WhisperMLX.from_pretrained("large-v3")
        config = DualPathConfig()
        streamer = DualPathStreamer(model, config)

        async for result in streamer.process_stream(audio_source):
            # Display speculative text (may change)
            print(f"[SPECULATIVE] {result.speculative_text}")

            # Display confirmed text (stable)
            if result.confirmed_is_new:
                print(f"[CONFIRMED] {result.confirmed_text}")
    """

    def __init__(
        self,
        model: WhisperMLX,
        config: DualPathConfig | None = None,
    ):
        """
        Initialize dual-path streaming.

        Args:
            model: Initialized WhisperMLX model
            config: Dual-path configuration (uses defaults if None)
        """
        self.model = model
        self.config = config or DualPathConfig()

        # Fast path buffer
        self._fast_buffer = AudioBuffer(
            max_duration=self.config.fast_chunk_duration * 3,
            sample_rate=self.config.sample_rate,
        )

        # Quality path buffer
        self._quality_buffer = AudioBuffer(
            max_duration=self.config.quality_chunk_duration * 3,
            sample_rate=self.config.sample_rate,
        )

        # LocalAgreement for quality path
        self._local_agreement = LocalAgreement(n=self.config.agreement_n)

        # VAD setup
        if self.config.use_vad:
            if not HAS_VAD:
                raise ImportError(
                    "webrtcvad required for VAD. Install: pip install webrtcvad",
                )
            self._vad = webrtcvad.Vad(self.config.vad_aggressiveness)
        else:
            self._vad = None

        # State tracking
        self._total_audio_time = 0.0
        self._last_fast_time = 0.0
        self._last_quality_time = 0.0

        # Accumulated text
        self._full_speculative = ""
        self._full_confirmed = ""

        # Previous speculative for delta
        self._prev_speculative = ""

        # Detected language (cached)
        self._detected_language: str | None = None

    def reset(self) -> None:
        """Reset for new streaming session."""
        self._fast_buffer.clear()
        self._quality_buffer.clear()
        self._local_agreement.reset()
        self._total_audio_time = 0.0
        self._last_fast_time = 0.0
        self._last_quality_time = 0.0
        self._full_speculative = ""
        self._full_confirmed = ""
        self._prev_speculative = ""
        self._detected_language = None

    async def process_stream(
        self,
        audio_source: AsyncIterator[np.ndarray],
    ) -> AsyncIterator[DualPathResult]:
        """
        Process streaming audio with dual-path transcription.

        Args:
            audio_source: Async iterator yielding audio chunks (float32, 16kHz mono)

        Yields:
            DualPathResult with speculative and confirmed text
        """
        self.reset()

        async for chunk in audio_source:
            async for result in self._process_chunk(chunk):
                yield result

        # Finalize
        async for result in self._finalize():
            yield result

    async def _process_chunk(
        self, audio: np.ndarray,
    ) -> AsyncIterator[DualPathResult]:
        """Process an audio chunk through both paths."""
        # Normalize to float32
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Add to both buffers
        self._fast_buffer.append(audio)
        self._quality_buffer.append(audio)
        self._total_audio_time += len(audio) / self.config.sample_rate

        # Check if fast path should run
        fast_result = None
        fast_new = False
        fast_latency = 0.0

        if self._fast_buffer.duration >= self.config.fast_chunk_duration:
            start = time.perf_counter()
            fast_text = await self._run_fast_path()
            fast_latency = (time.perf_counter() - start) * 1000

            # Calculate delta
            if fast_text != self._prev_speculative:
                fast_result = fast_text
                fast_new = True
                self._prev_speculative = fast_text
                self._full_speculative = self._full_confirmed + fast_text

        # Check if quality path should run
        quality_result = None
        quality_new = False
        quality_latency = 0.0

        if self._quality_buffer.duration >= self.config.quality_chunk_duration:
            start = time.perf_counter()
            newly_confirmed = await self._run_quality_path()
            quality_latency = (time.perf_counter() - start) * 1000

            if newly_confirmed:
                quality_result = newly_confirmed
                quality_new = True
                self._full_confirmed += newly_confirmed
                # Reset speculative since confirmed text advanced
                self._prev_speculative = ""
                self._full_speculative = self._full_confirmed

        # Yield result if either path produced output
        if fast_new or quality_new:
            yield DualPathResult(
                speculative_text=fast_result or "",
                speculative_is_new=fast_new,
                confirmed_text=quality_result or "",
                confirmed_is_new=quality_new,
                full_speculative=self._full_speculative,
                full_confirmed=self._full_confirmed,
                fast_latency_ms=fast_latency,
                quality_latency_ms=quality_latency,
                audio_time=self._total_audio_time,
            )

    async def _run_fast_path(self) -> str:
        """Run fast path transcription."""
        audio = self._fast_buffer.get_audio(self.config.fast_chunk_duration)

        # Run transcription (greedy, fast)
        # Use asyncio.to_thread for CPU-bound transcription
        result = await asyncio.to_thread(
            self.model.transcribe,
            audio,
            language=self.config.language or self._detected_language,
            task=self.config.task,
            variable_length=False,
            initial_prompt=self.config.initial_prompt,  # J9: Terminology injection
        )

        # Cache detected language
        if self._detected_language is None and result.get("language"):
            self._detected_language = result["language"]

        # Clear fast buffer (keep small overlap for context)
        overlap_samples = int(0.2 * self.config.sample_rate)
        if self._fast_buffer.write_pos > overlap_samples:
            # Keep last 0.2s as context
            overlap_audio = self._fast_buffer.get_audio(0.2)
            self._fast_buffer.clear()
            self._fast_buffer.append(overlap_audio)
        else:
            self._fast_buffer.clear()

        return result.get("text", "").strip()

    async def _run_quality_path(self) -> str:
        """Run quality path transcription with LocalAgreement."""
        audio = self._quality_buffer.get_audio(self.config.quality_chunk_duration)

        # Run transcription
        result = await asyncio.to_thread(
            self.model.transcribe,
            audio,
            language=self.config.language or self._detected_language,
            task=self.config.task,
            variable_length=False,
            initial_prompt=self.config.initial_prompt,  # J9: Terminology injection
        )

        text = result.get("text", "").strip()

        # Update LocalAgreement
        newly_confirmed = self._local_agreement.update(text)

        # Keep overlap for next chunk
        overlap_samples = int(self.config.quality_overlap * self.config.sample_rate)
        if self._quality_buffer.write_pos > overlap_samples:
            overlap_audio = self._quality_buffer.get_audio(self.config.quality_overlap)
            self._quality_buffer.clear()
            self._quality_buffer.append(overlap_audio)
        else:
            self._quality_buffer.clear()

        return newly_confirmed

    async def _finalize(self) -> AsyncIterator[DualPathResult]:
        """Finalize and emit remaining audio."""
        # Process remaining fast path audio
        fast_text = ""
        fast_latency = 0.0
        min_fast_samples = int(0.3 * self.config.sample_rate)

        if self._fast_buffer.write_pos >= min_fast_samples:
            start = time.perf_counter()
            audio = self._fast_buffer.buffer[:self._fast_buffer.write_pos]
            result = await asyncio.to_thread(
                self.model.transcribe,
                audio,
                language=self.config.language or self._detected_language,
                task=self.config.task,
                variable_length=False,
                initial_prompt=self.config.initial_prompt,  # J9: Terminology injection
            )
            fast_text = result.get("text", "").strip()
            fast_latency = (time.perf_counter() - start) * 1000

        # Process remaining quality path audio
        quality_text = ""
        quality_latency = 0.0
        min_quality_samples = int(0.5 * self.config.sample_rate)

        if self._quality_buffer.write_pos >= min_quality_samples:
            start = time.perf_counter()
            audio = self._quality_buffer.buffer[:self._quality_buffer.write_pos]
            result = await asyncio.to_thread(
                self.model.transcribe,
                audio,
                language=self.config.language or self._detected_language,
                task=self.config.task,
                variable_length=False,
                initial_prompt=self.config.initial_prompt,  # J9: Terminology injection
            )
            text = result.get("text", "").strip()

            # Final text is fully confirmed
            if text:
                # Get any remaining confirmed from LocalAgreement
                remaining = self._local_agreement.get_speculative()
                quality_text = remaining + " " + text if remaining else text
                quality_text = quality_text.strip()

            quality_latency = (time.perf_counter() - start) * 1000

        if fast_text or quality_text:
            self._full_speculative = self._full_confirmed + (quality_text or fast_text)
            self._full_confirmed += quality_text if quality_text else ""

            yield DualPathResult(
                speculative_text=fast_text,
                speculative_is_new=bool(fast_text),
                confirmed_text=quality_text,
                confirmed_is_new=bool(quality_text),
                full_speculative=self._full_speculative,
                full_confirmed=self._full_confirmed,
                fast_latency_ms=fast_latency,
                quality_latency_ms=quality_latency,
                audio_time=self._total_audio_time,
            )


# =============================================================================
# J10: Multi-User Batch Streaming Server
# =============================================================================


@dataclass
class BatchServerConfig:
    """Configuration for multi-user batching stream server (J10)."""

    # Batching settings
    max_batch_size: int = 8  # Maximum sessions to batch together
    batch_timeout_ms: float = 100.0  # Max wait time before processing batch
    min_audio_duration: float = 0.5  # Minimum audio per session before batching

    # Audio settings
    sample_rate: int = 16000

    # Model settings
    language: str | None = None  # Shared language (None = auto-detect)
    task: str = "transcribe"

    # Session settings
    session_timeout_seconds: float = 60.0  # Inactive sessions are cleaned up
    max_sessions: int = 100  # Maximum concurrent sessions


@dataclass
class BatchSessionState:
    """Per-session state for multi-user batching."""

    session_id: str
    audio_buffer: np.ndarray  # Accumulated audio
    total_audio_time: float  # Total audio received
    last_activity: float  # Timestamp of last chunk
    local_agreement: LocalAgreement | None
    confirmed_text: str  # All confirmed text so far
    detected_language: str | None
    is_active: bool


@dataclass
class BatchResult:
    """Result from batch server for a single session."""

    session_id: str
    text: str  # Latest transcription
    confirmed_text: str  # Confirmed text (LocalAgreement)
    speculative_text: str  # Unconfirmed speculative text
    is_confirmed: bool  # True if new text was confirmed
    processing_time_ms: float
    batch_size: int  # Number of sessions in this batch
    language: str | None


class BatchingStreamServer:
    """
    Multi-user batching stream server (J10 optimization).

    Collects audio chunks from multiple concurrent streaming sessions,
    batches them together for efficient GPU processing, and returns
    results to the correct sessions.

    This maximizes GPU utilization in server scenarios where multiple
    users are streaming audio simultaneously.

    Architecture:
    - Sessions: Per-user state (audio buffer, LocalAgreement)
    - Collector: Accumulates audio from all sessions
    - Batcher: Groups ready sessions into batches
    - Processor: Runs transcribe_batch() on batches
    - Distributor: Routes results back to sessions

    Benefits over single-user streaming:
    - 2-4x throughput with 4-8 concurrent users
    - Better GPU utilization (batch matmuls)
    - Shared encoder computation

    Example:
        from tools.whisper_mlx import WhisperMLX
        from tools.whisper_mlx.streaming import BatchingStreamServer, BatchServerConfig

        model = WhisperMLX.from_pretrained("large-v3")
        config = BatchServerConfig(max_batch_size=8)
        server = BatchingStreamServer(model, config)

        # Start the batch processing loop
        asyncio.create_task(server.run_batch_loop())

        # Handle incoming audio from users
        async def handle_user(session_id: str, audio_chunks: AsyncIterator):
            async for chunk in audio_chunks:
                results = await server.add_audio(session_id, chunk)
                for result in results:
                    yield result  # Send back to user

            # Finalize when user disconnects
            final = await server.finalize_session(session_id)
            if final:
                yield final
    """

    def __init__(
        self,
        model: WhisperMLX,
        config: BatchServerConfig | None = None,
    ):
        """
        Initialize the batching stream server.

        Args:
            model: Initialized WhisperMLX model
            config: Server configuration (uses defaults if None)
        """
        self.model = model
        self.config = config or BatchServerConfig()

        # Session management
        self._sessions: dict[str, BatchSessionState] = {}
        self._session_lock = asyncio.Lock()

        # Batch collection
        self._pending_sessions: set[str] = set()  # Sessions with pending audio
        self._pending_lock = asyncio.Lock()

        # Result queues (session_id -> list of results)
        self._results: dict[str, list[BatchResult]] = {}
        self._results_lock = asyncio.Lock()

        # Events
        self._new_audio_event = asyncio.Event()
        self._shutdown = False

        # Stats
        self._batches_processed = 0
        self._total_sessions_served = 0

    async def create_session(
        self,
        session_id: str,
        use_local_agreement: bool = True,
        agreement_n: int = 2,
    ) -> bool:
        """
        Create a new streaming session.

        Args:
            session_id: Unique identifier for this session
            use_local_agreement: Whether to use LocalAgreement for stable output
            agreement_n: LocalAgreement threshold

        Returns:
            True if session created, False if session already exists
        """
        async with self._session_lock:
            if session_id in self._sessions:
                return False

            if len(self._sessions) >= self.config.max_sessions:
                raise RuntimeError(
                    f"Maximum sessions ({self.config.max_sessions}) reached",
                )

            self._sessions[session_id] = BatchSessionState(
                session_id=session_id,
                audio_buffer=np.array([], dtype=np.float32),
                total_audio_time=0.0,
                last_activity=time.time(),
                local_agreement=LocalAgreement(agreement_n) if use_local_agreement else None,
                confirmed_text="",
                detected_language=None,
                is_active=True,
            )
            self._total_sessions_served += 1
            return True

    async def add_audio(
        self,
        session_id: str,
        audio: np.ndarray,
    ) -> list[BatchResult]:
        """
        Add audio chunk to a session.

        This method is non-blocking. Audio is accumulated and will be
        processed in the next batch.

        Args:
            session_id: Session identifier
            audio: Audio chunk (float32, 16kHz mono)

        Returns:
            List of any pending results for this session (may be empty)
        """
        # Normalize audio
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Auto-create session if needed (outside lock to avoid deadlock)
        async with self._session_lock:
            if session_id not in self._sessions:
                # Create session inline to avoid deadlock
                if len(self._sessions) >= self.config.max_sessions:
                    raise RuntimeError(
                        f"Maximum sessions ({self.config.max_sessions}) reached",
                    )
                self._sessions[session_id] = BatchSessionState(
                    session_id=session_id,
                    audio_buffer=np.array([], dtype=np.float32),
                    total_audio_time=0.0,
                    last_activity=time.time(),
                    local_agreement=LocalAgreement(2),  # Default agreement
                    confirmed_text="",
                    detected_language=None,
                    is_active=True,
                )
                self._total_sessions_served += 1

            session = self._sessions[session_id]
            session.audio_buffer = np.concatenate([session.audio_buffer, audio])
            session.total_audio_time += len(audio) / self.config.sample_rate
            session.last_activity = time.time()

        # Mark session as having pending audio
        async with self._pending_lock:
            if session.total_audio_time >= self.config.min_audio_duration:
                self._pending_sessions.add(session_id)
                self._new_audio_event.set()

        # Return any pending results
        return await self._get_results(session_id)

    async def finalize_session(self, session_id: str) -> BatchResult | None:
        """
        Finalize a session and get remaining transcription.

        Args:
            session_id: Session identifier

        Returns:
            Final result if there was remaining audio, None otherwise
        """
        async with self._session_lock:
            if session_id not in self._sessions:
                return None

            session = self._sessions[session_id]
            session.is_active = False

            # Process any remaining audio
            if len(session.audio_buffer) > 0:
                min_samples = int(0.3 * self.config.sample_rate)
                if len(session.audio_buffer) >= min_samples:
                    # Transcribe remaining audio
                    result = await asyncio.to_thread(
                        self.model.transcribe,
                        session.audio_buffer,
                        language=self.config.language or session.detected_language,
                        task=self.config.task,
                    )
                    text = result.get("text", "").strip()

                    if text:
                        # For final result, all text is confirmed
                        confirmed = session.confirmed_text + " " + text
                        confirmed = confirmed.strip()

                        final_result = BatchResult(
                            session_id=session_id,
                            text=text,
                            confirmed_text=confirmed,
                            speculative_text="",
                            is_confirmed=True,
                            processing_time_ms=0,  # Not measured for finalize
                            batch_size=1,
                            language=result.get("language"),
                        )

                        # Clean up session
                        del self._sessions[session_id]
                        return final_result

            # Clean up session
            del self._sessions[session_id]
            return None

    async def run_batch_loop(self) -> None:
        """
        Main batch processing loop.

        This runs continuously, collecting audio from sessions and
        processing them in batches. Should be run as a background task.
        """
        while not self._shutdown:
            # Wait for new audio or timeout
            try:
                await asyncio.wait_for(
                    self._new_audio_event.wait(),
                    timeout=self.config.batch_timeout_ms / 1000.0,
                )
            except TimeoutError:
                pass

            self._new_audio_event.clear()

            # Collect ready sessions
            async with self._pending_lock:
                ready_sessions = list(self._pending_sessions)
                self._pending_sessions.clear()

            if not ready_sessions:
                # Clean up inactive sessions
                await self._cleanup_inactive_sessions()
                continue

            # Limit batch size
            batch_sessions = ready_sessions[: self.config.max_batch_size]

            # Process batch
            await self._process_batch(batch_sessions)
            self._batches_processed += 1

            # Put remaining sessions back
            if len(ready_sessions) > self.config.max_batch_size:
                async with self._pending_lock:
                    for sid in ready_sessions[self.config.max_batch_size :]:
                        self._pending_sessions.add(sid)

    async def _process_batch(self, session_ids: list[str]) -> None:
        """Process a batch of sessions."""
        if not session_ids:
            return

        start_time = time.perf_counter()

        # Collect audio from sessions
        audio_list = []
        session_list = []

        async with self._session_lock:
            for sid in session_ids:
                if sid not in self._sessions:
                    continue
                session = self._sessions[sid]
                if len(session.audio_buffer) > 0:
                    audio_list.append(session.audio_buffer.copy())
                    session_list.append(session)
                    # Clear buffer (audio is now in batch)
                    session.audio_buffer = np.array([], dtype=np.float32)

        if not audio_list:
            return

        batch_size = len(audio_list)

        # Run batch transcription
        try:
            results = await asyncio.to_thread(
                self.model.transcribe_batch,
                audio_list,
                language=self.config.language,
                task=self.config.task,
            )
        except Exception as e:
            # On error, return empty results
            print(f"Batch transcription error: {e}")
            return

        processing_time = (time.perf_counter() - start_time) * 1000

        # Distribute results to sessions
        async with self._session_lock:
            async with self._results_lock:
                for session, result in zip(session_list, results, strict=False):
                    text = result.get("text", "").strip()
                    language = result.get("language")

                    # Cache detected language
                    if session.detected_language is None and language:
                        session.detected_language = language

                    # Apply LocalAgreement if enabled
                    confirmed_text = session.confirmed_text
                    speculative_text = ""
                    is_confirmed = False

                    if text and session.local_agreement is not None:
                        newly_confirmed = session.local_agreement.update(text)
                        if newly_confirmed:
                            session.confirmed_text += (
                                " " + newly_confirmed
                                if session.confirmed_text
                                else newly_confirmed
                            )
                            confirmed_text = session.confirmed_text
                            is_confirmed = True
                        speculative_text = session.local_agreement.get_speculative()
                    elif text:
                        # No LocalAgreement - all text is speculative
                        speculative_text = text

                    batch_result = BatchResult(
                        session_id=session.session_id,
                        text=text,
                        confirmed_text=confirmed_text,
                        speculative_text=speculative_text,
                        is_confirmed=is_confirmed,
                        processing_time_ms=processing_time / batch_size,
                        batch_size=batch_size,
                        language=language,
                    )

                    # Add to results queue
                    if session.session_id not in self._results:
                        self._results[session.session_id] = []
                    self._results[session.session_id].append(batch_result)

    async def _get_results(self, session_id: str) -> list[BatchResult]:
        """Get and clear pending results for a session."""
        async with self._results_lock:
            return self._results.pop(session_id, [])

    async def _cleanup_inactive_sessions(self) -> None:
        """Remove sessions that have been inactive too long."""
        now = time.time()
        timeout = self.config.session_timeout_seconds

        async with self._session_lock:
            inactive = [
                sid
                for sid, session in self._sessions.items()
                if (now - session.last_activity) > timeout and not session.is_active
            ]
            for sid in inactive:
                del self._sessions[sid]

    async def shutdown(self) -> None:
        """Shutdown the batch server."""
        self._shutdown = True
        self._new_audio_event.set()  # Wake up batch loop

    def get_stats(self) -> dict:
        """Get server statistics."""
        return {
            "active_sessions": len(self._sessions),
            "batches_processed": self._batches_processed,
            "total_sessions_served": self._total_sessions_served,
        }


# =============================================================================
# CTC Streaming for Sub-200ms First Partial Latency
# =============================================================================


@dataclass
class CTCStreamingConfig:
    """
    Configuration for CTC-accelerated streaming.

    Target: <200ms first partial latency via CTC head on encoder features.
    """
    # Audio settings
    sample_rate: int = 16000

    # CTC settings - optimized for sub-200ms latency
    # CTC can emit from shorter audio than decoder needs
    min_ctc_duration: float = 0.5  # Minimum audio for CTC output (500ms)
    ctc_interval: float = 0.3      # CTC output interval (300ms between updates)

    # Full decoder settings (for verified text)
    decoder_interval: float = 2.0   # Run decoder every 2s for confirmed text
    min_decoder_duration: float = 1.0  # Minimum audio for decoder

    # VAD settings
    use_vad: bool = True
    vad_aggressiveness: int = 2
    vad_frame_duration_ms: int = 30
    silence_threshold_duration: float = 0.5

    # Model settings
    language: str | None = None
    task: str = "transcribe"

    # LocalAgreement for decoder path
    use_local_agreement: bool = True
    agreement_n: int = 2

    @classmethod
    def for_latency_target(cls, target_ms: int = 200) -> CTCStreamingConfig:
        """
        Create config optimized for a specific first partial latency target.

        Args:
            target_ms: Target first partial latency in milliseconds
                - 200: Ultra-low latency (100ms audio + ~100ms inference)
                - 400: Low latency (300ms audio + ~100ms inference)
                - 600: Default (500ms audio + ~100ms inference)

        Returns:
            CTCStreamingConfig optimized for the target latency
        """
        # Inference time is ~60-100ms, budget rest for audio accumulation
        inference_budget_ms = 100

        # Calculate audio accumulation based on target
        audio_budget_ms = max(50, target_ms - inference_budget_ms)

        # CTC interval should be smaller for lower latency
        ctc_interval = max(0.05, audio_budget_ms / 2000)

        return cls(
            min_ctc_duration=audio_budget_ms / 1000,
            ctc_interval=ctc_interval,
            use_vad=False,  # Disable VAD for lowest latency
        )

    @classmethod
    def ultra_low_latency(cls) -> CTCStreamingConfig:
        """
        Config for <200ms first partial latency (Gate 1 target).

        Settings:
        - 100ms audio accumulation
        - 50ms CTC update interval
        - VAD disabled for lowest latency
        - ~100ms inference = ~200ms total first partial
        """
        return cls(
            min_ctc_duration=0.1,   # 100ms audio
            ctc_interval=0.05,      # 50ms updates
            decoder_interval=2.0,   # Decoder every 2s
            min_decoder_duration=0.5,
            use_vad=False,          # Disable for lowest latency
        )


@dataclass
class CTCStreamingResult:
    """
    Result from CTC-accelerated streaming transcription.

    Contains both CTC draft (fast, speculative) and decoder verified (slow, accurate).
    """
    # CTC draft output (fast path, ~60ms latency)
    ctc_draft: str = ""           # CTC speculative text (may be wrong)
    ctc_tokens: list[int] = None  # Raw CTC token IDs  # type: ignore
    ctc_is_new: bool = False      # True if CTC draft changed since last result
    ctc_latency_ms: float = 0.0   # CTC inference latency (encoder + CTC head)

    # Decoder verified output (slow path, ~500ms latency)
    decoder_text: str = ""        # Full decoder transcription
    confirmed_text: str = ""      # LocalAgreement confirmed (stable)
    speculative_text: str = ""    # LocalAgreement speculative (may change)
    decoder_is_new: bool = False  # True if decoder ran this iteration

    # Timing
    audio_duration: float = 0.0   # Duration of audio processed
    total_audio_time: float = 0.0  # Total audio received so far

    # For UI display
    @property
    def display_text(self) -> str:
        """Get best available text for display."""
        if self.confirmed_text:
            return self.confirmed_text + (f" {self.speculative_text}" if self.speculative_text else "")
        return self.ctc_draft

    def __post_init__(self):
        if self.ctc_tokens is None:
            self.ctc_tokens = []


class CTCStreamingWhisper:
    """
    CTC-accelerated streaming for sub-200ms first partial latency.

    Key insight: CTC head runs on encoder features (~60ms total) while
    full decoder runs periodically for verified text (~500ms).

    Architecture:
        Audio (500ms) -> Mel (~5ms) -> Encoder (~50ms) -> CTC Head (~3ms)
        = ~60ms first partial (vs 243ms with decoder)

        Periodically: Full decoder -> Verified text

    Latency breakdown (measured on M4):
        - Mel spectrogram: ~5ms for 500ms audio
        - Encoder: ~50ms for 500ms audio (variable_length=True)
        - CTC head decode: ~3ms
        - Total: ~60ms after audio accumulation

    With 500ms audio accumulation + 60ms inference = ~560ms first partial
    With 300ms audio accumulation + 60ms inference = ~360ms first partial

    To achieve <200ms, we need:
        - Trained CTC head (currently untrained)
        - 100-150ms audio chunks
        - This requires CTC to work on very short audio

    Usage:
        from tools.whisper_mlx import WhisperMLX
        from tools.whisper_mlx.streaming import CTCStreamingWhisper, CTCStreamingConfig
        from tools.whisper_mlx.ctc_head import CTCDraftHead

        model = WhisperMLX.from_pretrained("large-v3")
        ctc_head = CTCDraftHead.load_weights("ctc_head.safetensors", d_model=1280)
        config = CTCStreamingConfig()
        streamer = CTCStreamingWhisper(model, ctc_head, config)

        async for result in streamer.transcribe_stream(audio_source):
            if result.ctc_is_new:
                print(f"[CTC ~{result.ctc_latency_ms:.0f}ms] {result.ctc_draft}")
            if result.decoder_is_new:
                print(f"[VERIFIED] {result.confirmed_text}")
    """

    def __init__(
        self,
        model: WhisperMLX,
        ctc_head: CTCDraftHead,
        config: CTCStreamingConfig | None = None,
    ):
        """
        Initialize CTC-accelerated streaming.

        Args:
            model: WhisperMLX model (encoder + decoder)
            ctc_head: Trained CTC head (or untrained for pipeline testing)
            config: Streaming configuration
        """
        if not HAS_CTC:
            raise ImportError("CTC head module not available")

        self.model = model
        self.ctc_head = ctc_head
        self.config = config or CTCStreamingConfig()

        # VAD setup
        if self.config.use_vad:
            if not HAS_VAD:
                raise ImportError("webrtcvad required for VAD")
            self._vad = webrtcvad.Vad(self.config.vad_aggressiveness)
        else:
            self._vad = None

        # LocalAgreement for decoder path
        if self.config.use_local_agreement:
            self._local_agreement: LocalAgreement | None = LocalAgreement(
                n=self.config.agreement_n,
            )
        else:
            self._local_agreement = None

        # State
        self._reset_state()

        # Import audio utilities (lazy import to avoid circular deps)
        from .audio import log_mel_spectrogram
        self._log_mel_spectrogram = log_mel_spectrogram

        # Load tokenizer for CTC decoding
        from .tokenizer import get_whisper_tokenizer
        self._tokenizer = get_whisper_tokenizer(
            multilingual=model.is_multilingual,
            task=self.config.task,
        )

        # Warmup CTC path (mel + encoder + CTC head) with dummy audio
        # This compiles Metal kernels for the CTC-specific path
        self._warmup_ctc()

    def _warmup_ctc(self) -> None:
        """Warmup CTC inference path with dummy audio."""
        import mlx.core as mx

        # Generate 100ms of dummy audio (matches min_ctc_duration)
        warmup_duration = max(0.1, self.config.min_ctc_duration)
        warmup_samples = int(self.config.sample_rate * warmup_duration)
        dummy_audio = np.zeros(warmup_samples, dtype=np.float32)

        # Run mel spectrogram
        mel = self._log_mel_spectrogram(dummy_audio, n_mels=self.model.config.n_mels)
        mel = mx.expand_dims(mx.array(mel), axis=0)
        mx.eval(mel)

        # Run encoder (this warms up the encoder for short audio)
        encoder_output = self.model.encoder(mel, variable_length=True)
        mx.eval(encoder_output)

        # Run CTC head
        ctc_logits = self.ctc_head(encoder_output)
        mx.eval(ctc_logits)

        # Decode (compiles greedy decode path)
        _ = self.ctc_head.decode_greedy(ctc_logits)

    def _reset_state(self) -> None:
        """Reset internal state for new session."""
        # Audio buffer
        max_duration = 30.0  # Max audio to hold
        self._audio_buffer = AudioBuffer(max_duration, self.config.sample_rate)
        self._speech_buffer = np.array([], dtype=np.float32)

        # Timing
        self._total_audio_time = 0.0
        self._last_ctc_time = 0.0
        self._last_decoder_time = 0.0
        self._silence_frames = 0

        # CTC state
        self._prev_ctc_draft = ""
        self._ctc_history: list[str] = []

        # Decoder state
        self._confirmed_text = ""
        self._detected_language: str | None = None

        # Reset LocalAgreement
        if self._local_agreement is not None:
            self._local_agreement.reset()

    def reset(self) -> None:
        """Reset for new streaming session."""
        self._reset_state()

    async def transcribe_stream(
        self,
        audio_source: AsyncIterator[np.ndarray],
    ) -> AsyncIterator[CTCStreamingResult]:
        """
        Transcribe streaming audio with CTC-accelerated first partial.

        Args:
            audio_source: Async iterator yielding audio chunks (float32, 16kHz mono)

        Yields:
            CTCStreamingResult with CTC draft and decoder verified text
        """
        self._reset_state()

        async for chunk in audio_source:
            async for result in self._process_chunk(chunk):
                yield result

        # Finalize
        async for result in self._finalize():
            yield result

    async def _process_chunk(self, audio: np.ndarray) -> AsyncIterator[CTCStreamingResult]:
        """Process an audio chunk through CTC and optionally decoder."""
        # Normalize audio
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Add to buffers
        self._audio_buffer.append(audio)
        self._speech_buffer = np.concatenate([self._speech_buffer, audio])
        self._total_audio_time += len(audio) / self.config.sample_rate

        # VAD check
        is_speech = True
        if self._vad is not None:
            is_speech = self._check_vad(audio)

        if is_speech:
            self._silence_frames = 0
        else:
            self._silence_frames += 1

        current_duration = len(self._speech_buffer) / self.config.sample_rate
        silence_duration = self._silence_frames * len(audio) / self.config.sample_rate

        # Check if we should run CTC
        ctc_result = None
        time_since_ctc = self._total_audio_time - self._last_ctc_time
        if (current_duration >= self.config.min_ctc_duration and
            time_since_ctc >= self.config.ctc_interval):
            ctc_result = await self._run_ctc()
            self._last_ctc_time = self._total_audio_time

        # Check if we should run decoder
        decoder_result = None
        time_since_decoder = self._total_audio_time - self._last_decoder_time
        run_decoder = False

        # Run decoder if: enough time passed, or silence detected (endpoint)
        if (current_duration >= self.config.min_decoder_duration and
            time_since_decoder >= self.config.decoder_interval):
            run_decoder = True
        elif (silence_duration >= self.config.silence_threshold_duration and
              current_duration >= self.config.min_decoder_duration):
            run_decoder = True

        if run_decoder:
            decoder_result = await self._run_decoder()
            self._last_decoder_time = self._total_audio_time

            # If endpoint (silence), clear buffer
            if silence_duration >= self.config.silence_threshold_duration:
                self._speech_buffer = np.array([], dtype=np.float32)
                self._prev_ctc_draft = ""
                if self._local_agreement is not None:
                    self._local_agreement.reset()

        # Yield result if we have anything new
        if ctc_result is not None or decoder_result is not None:
            yield CTCStreamingResult(
                ctc_draft=ctc_result["text"] if ctc_result else self._prev_ctc_draft,
                ctc_tokens=ctc_result["tokens"] if ctc_result else [],
                ctc_is_new=ctc_result is not None and ctc_result["text"] != self._prev_ctc_draft,
                ctc_latency_ms=ctc_result["latency_ms"] if ctc_result else 0.0,
                decoder_text=decoder_result["text"] if decoder_result else "",
                confirmed_text=decoder_result["confirmed"] if decoder_result else self._confirmed_text,
                speculative_text=decoder_result["speculative"] if decoder_result else "",
                decoder_is_new=decoder_result is not None,
                audio_duration=current_duration,
                total_audio_time=self._total_audio_time,
            )

            # Update state
            if ctc_result is not None:
                self._prev_ctc_draft = ctc_result["text"]

    async def _run_ctc(self) -> dict:
        """
        Run CTC head on current audio buffer.

        Returns dict with: text, tokens, latency_ms
        """
        import mlx.core as mx

        start_time = time.perf_counter()

        # Convert to mel spectrogram (use model's n_mels config)
        audio = self._speech_buffer
        n_mels = self.model.config.n_mels
        mel = self._log_mel_spectrogram(audio, n_mels=n_mels)

        # Add batch dimension: (T, n_mels) -> (1, T, n_mels)
        mel = mx.expand_dims(mx.array(mel), axis=0)

        # Run encoder (variable_length for shorter audio)
        encoder_output = self.model.encoder(mel, variable_length=True)
        mx.eval(encoder_output)

        # Run CTC head
        ctc_logits = self.ctc_head(encoder_output)
        mx.eval(ctc_logits)

        # Decode
        tokens = self.ctc_head.decode_greedy(ctc_logits)

        # Convert tokens to text (use CTC tokenizer)
        text = self._tokenizer.decode(tokens) if tokens else ""

        latency_ms = (time.perf_counter() - start_time) * 1000

        return {
            "text": text.strip(),
            "tokens": tokens,
            "latency_ms": latency_ms,
        }

    async def _run_decoder(self) -> dict:
        """
        Run full decoder on current audio buffer.

        Returns dict with: text, confirmed, speculative
        """
        audio = self._speech_buffer
        if len(audio) < self.config.min_decoder_duration * self.config.sample_rate:
            return {"text": "", "confirmed": self._confirmed_text, "speculative": ""}

        # Run full transcription
        result = await asyncio.to_thread(
            self.model.transcribe,
            audio,
            language=self.config.language or self._detected_language,
            task=self.config.task,
            variable_length=False,
        )

        text = result.get("text", "").strip()

        # Cache detected language
        if self._detected_language is None and result.get("language"):
            self._detected_language = result["language"]

        # Apply LocalAgreement
        confirmed = self._confirmed_text
        speculative = ""

        if text and self._local_agreement is not None:
            newly_confirmed = self._local_agreement.update(text)
            if newly_confirmed:
                self._confirmed_text = (self._confirmed_text + " " + newly_confirmed).strip()
                confirmed = self._confirmed_text
            speculative = self._local_agreement.get_speculative()
        elif text:
            speculative = text

        return {
            "text": text,
            "confirmed": confirmed,
            "speculative": speculative,
        }

    async def _finalize(self) -> AsyncIterator[CTCStreamingResult]:
        """Process remaining audio at end of stream."""
        min_samples = int(0.3 * self.config.sample_rate)
        if len(self._speech_buffer) >= min_samples:
            # Final decoder run
            result = await self._run_decoder()

            # Final CTC run
            ctc_result = await self._run_ctc()

            yield CTCStreamingResult(
                ctc_draft=ctc_result["text"],
                ctc_tokens=ctc_result["tokens"],
                ctc_is_new=True,
                ctc_latency_ms=ctc_result["latency_ms"],
                decoder_text=result["text"],
                confirmed_text=result["confirmed"],
                speculative_text=result["speculative"],
                decoder_is_new=True,
                audio_duration=len(self._speech_buffer) / self.config.sample_rate,
                total_audio_time=self._total_audio_time,
            )

        self._reset_state()

    def _check_vad(self, audio: np.ndarray) -> bool:
        """Check if audio contains speech."""
        if self._vad is None:
            return True

        pcm = (audio * 32767).astype(np.int16)
        frame_size = int(self.config.sample_rate * self.config.vad_frame_duration_ms / 1000)
        n_frames = len(pcm) // frame_size

        if n_frames == 0:
            return False

        speech_frames = 0
        for i in range(n_frames):
            frame = pcm[i * frame_size:(i + 1) * frame_size]
            if self._vad.is_speech(frame.tobytes(), self.config.sample_rate):
                speech_frames += 1

        return speech_frames > n_frames // 2


__all__ = [
    "StreamingWhisper",
    "SyncStreamingWhisper",
    "StreamingConfig",
    "StreamingResult",
    "StreamState",
    "AudioBuffer",
    "LocalAgreement",
    # J2: AlignAtt word-aligned streaming
    "AlignAttPolicy",
    "DualPathStreamer",
    "DualPathConfig",
    "DualPathResult",
    "LATENCY_MODES",
    # RTF optimization presets (commit #1459)
    "STREAMING_PRESETS",
    "get_streaming_config",
    # J10: Multi-user batch server
    "BatchingStreamServer",
    "BatchServerConfig",
    "BatchSessionState",
    "BatchResult",
    # CTC streaming for sub-200ms first partial (commit #1461)
    "CTCStreamingWhisper",
    "CTCStreamingConfig",
    "CTCStreamingResult",
    "HAS_CTC",
]
