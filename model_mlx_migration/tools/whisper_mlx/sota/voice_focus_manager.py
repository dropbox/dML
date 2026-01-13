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
VoiceFocusManager - Runtime priority control for multi-speaker ASR.

Manages which speakers get transcription priority in multi-speaker scenarios.
Integrates with PersonalVAD and speaker embeddings to provide dynamic
speaker switching and focus modes.

Architecture:
```
                     ┌──────────────────────┐
                     │  VoiceFocusManager   │
                     └──────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
        ┌───────────┐  ┌───────────┐  ┌───────────┐
        │ Primary   │  │ Secondary │  │Background │
        │ Speaker   │  │ Speakers  │  │ Speakers  │
        │ (focused) │  │ (tracked) │  │ (ignored) │
        └───────────┘  └───────────┘  └───────────┘
              │               │               │
              ▼               ▼               │
        ┌───────────┐  ┌───────────┐         │
        │ Full ASR  │  │Selective  │         ▼
        │ Pipeline  │  │ Output    │    [dropped]
        └───────────┘  └───────────┘
```

Focus Modes:
- SINGLE: Only transcribe primary speaker, ignore all others
- PRIMARY_PLUS: Primary + secondary speakers (e.g., meeting host + participants)
- ALL: Transcribe all detected speakers
- DYNAMIC: Automatically switch focus based on activity

Usage:
    # Create manager
    manager = VoiceFocusManager()

    # Set primary speaker (the user we're focusing on)
    manager.set_primary_speaker(speaker_embedding)

    # Process audio with focus control
    result = manager.process_audio(
        audio=audio_chunk,
        encoder_output=encoder_output,
        detected_speakers=[emb1, emb2, emb3],
    )

    if result.should_transcribe:
        transcription = model.decode(...)
"""

from dataclasses import dataclass
from enum import Enum

import mlx.core as mx
import mlx.nn as nn


class FocusMode(Enum):
    """Speaker focus mode for transcription."""

    SINGLE = 0  # Only primary speaker
    PRIMARY_PLUS = 1  # Primary + registered secondary speakers
    ALL = 2  # All detected speakers
    DYNAMIC = 3  # Auto-switch based on activity


class SpeakerPriority(Enum):
    """Priority level for a speaker."""

    PRIMARY = 0  # Main speaker (highest priority)
    SECONDARY = 1  # Known/registered speakers
    BACKGROUND = 2  # Detected but not registered
    UNKNOWN = 3  # Cannot identify


@dataclass
class SpeakerState:
    """Track state for a single speaker."""

    speaker_id: int
    embedding: mx.array
    priority: SpeakerPriority = SpeakerPriority.UNKNOWN
    is_active: bool = False
    last_seen_time: float = 0.0
    total_active_time: float = 0.0
    utterance_count: int = 0
    confidence: float = 0.0


@dataclass
class FocusResult:
    """Result of focus decision for a frame/chunk."""

    should_transcribe: bool  # Whether to run decoder
    active_speaker_id: int | None  # Which speaker is active
    priority: SpeakerPriority  # Priority of active speaker
    confidence: float  # Confidence in speaker identification
    mode: FocusMode  # Current focus mode
    reason: str  # Explanation for decision


@dataclass
class VoiceFocusConfig:
    """Configuration for voice focus manager."""

    # Focus mode
    default_mode: FocusMode = FocusMode.PRIMARY_PLUS

    # Speaker matching threshold (cosine similarity)
    speaker_threshold: float = 0.7

    # Minimum VAD confidence to consider speech
    min_vad_confidence: float = 0.5

    # Time (seconds) before speaker is considered inactive
    speaker_timeout: float = 2.0

    # Whether to switch primary speaker based on activity
    allow_primary_switch: bool = False

    # Minimum time (seconds) speaking to become primary candidate
    min_speaking_time_for_primary: float = 10.0

    # Embedding dimension for speaker vectors
    embedding_dim: int = 192


class VoiceFocusManager:
    """
    Manages speaker focus and priority for multi-speaker ASR.

    Provides runtime control over which speakers get transcribed,
    enabling scenarios like:
    - Focus on single user (ignore background conversation)
    - Meeting transcription (track multiple registered speakers)
    - Interview mode (primary + interviewer)
    """

    def __init__(
        self,
        config: VoiceFocusConfig | None = None,
        speaker_encoder: nn.Module | None = None,
    ):
        """
        Initialize voice focus manager.

        Args:
            config: Configuration options
            speaker_encoder: Optional ECAPA-TDNN encoder for embeddings
        """
        self.config = config or VoiceFocusConfig()
        self.speaker_encoder = speaker_encoder

        # Speaker tracking
        self._primary_embedding: mx.array | None = None
        self._secondary_embeddings: dict[int, mx.array] = {}
        self._speaker_states: dict[int, SpeakerState] = {}
        self._next_speaker_id: int = 0

        # Current focus state
        self._current_mode: FocusMode = self.config.default_mode
        self._current_active_speaker: int | None = None

        # Statistics
        self._total_frames: int = 0
        self._transcribed_frames: int = 0
        self._dropped_frames: int = 0

    def set_primary_speaker(self, embedding: mx.array) -> int:
        """
        Set the primary speaker to focus on.

        Args:
            embedding: Speaker embedding (192-dim ECAPA vector)

        Returns:
            Speaker ID assigned to this speaker
        """
        self._primary_embedding = embedding
        speaker_id = self._assign_speaker_id()

        self._speaker_states[speaker_id] = SpeakerState(
            speaker_id=speaker_id,
            embedding=embedding,
            priority=SpeakerPriority.PRIMARY,
        )

        return speaker_id

    def add_secondary_speaker(self, embedding: mx.array) -> int:
        """
        Add a secondary speaker to track.

        Args:
            embedding: Speaker embedding

        Returns:
            Speaker ID assigned to this speaker
        """
        speaker_id = self._assign_speaker_id()
        self._secondary_embeddings[speaker_id] = embedding

        self._speaker_states[speaker_id] = SpeakerState(
            speaker_id=speaker_id,
            embedding=embedding,
            priority=SpeakerPriority.SECONDARY,
        )

        return speaker_id

    def set_mode(self, mode: FocusMode) -> None:
        """Set the focus mode."""
        self._current_mode = mode

    def get_mode(self) -> FocusMode:
        """Get current focus mode."""
        return self._current_mode

    def process_frame(
        self,
        vad_result: float,
        speaker_embedding: mx.array | None,
        timestamp: float,
    ) -> FocusResult:
        """
        Process a single frame and decide whether to transcribe.

        Args:
            vad_result: Voice activity detection confidence [0, 1]
            speaker_embedding: Detected speaker embedding (if available)
            timestamp: Current timestamp in seconds

        Returns:
            FocusResult with transcription decision
        """
        self._total_frames += 1

        # No speech detected
        if vad_result < self.config.min_vad_confidence:
            return FocusResult(
                should_transcribe=False,
                active_speaker_id=None,
                priority=SpeakerPriority.UNKNOWN,
                confidence=vad_result,
                mode=self._current_mode,
                reason="No speech detected",
            )

        # No speaker embedding available
        if speaker_embedding is None:
            # In ALL mode, transcribe anyway
            if self._current_mode == FocusMode.ALL:
                self._transcribed_frames += 1
                return FocusResult(
                    should_transcribe=True,
                    active_speaker_id=None,
                    priority=SpeakerPriority.UNKNOWN,
                    confidence=vad_result,
                    mode=self._current_mode,
                    reason="ALL mode - no speaker ID",
                )
            self._dropped_frames += 1
            return FocusResult(
                should_transcribe=False,
                active_speaker_id=None,
                priority=SpeakerPriority.UNKNOWN,
                confidence=vad_result,
                mode=self._current_mode,
                reason="No speaker embedding",
            )

        # Match speaker to known speakers
        speaker_id, priority, confidence = self._identify_speaker(speaker_embedding)

        # Update speaker state
        self._update_speaker_state(speaker_id, speaker_embedding, timestamp, priority)

        # Make focus decision based on mode
        should_transcribe, reason = self._make_focus_decision(priority, confidence)

        if should_transcribe:
            self._transcribed_frames += 1
            self._current_active_speaker = speaker_id
        else:
            self._dropped_frames += 1

        return FocusResult(
            should_transcribe=should_transcribe,
            active_speaker_id=speaker_id,
            priority=priority,
            confidence=confidence,
            mode=self._current_mode,
            reason=reason,
        )

    def _identify_speaker(
        self,
        embedding: mx.array,
    ) -> tuple[int, SpeakerPriority, float]:
        """
        Identify speaker from embedding.

        Args:
            embedding: Speaker embedding to match

        Returns:
            Tuple of (speaker_id, priority, confidence)
        """
        best_match_id = None
        best_similarity = -1.0
        best_priority = SpeakerPriority.UNKNOWN

        # Check primary speaker
        if self._primary_embedding is not None:
            sim = self._cosine_similarity(embedding, self._primary_embedding)
            if sim > self.config.speaker_threshold and sim > best_similarity:
                best_match_id = self._get_speaker_id_by_embedding(
                    self._primary_embedding,
                )
                best_similarity = sim
                best_priority = SpeakerPriority.PRIMARY

        # Check secondary speakers
        for speaker_id, sec_embedding in self._secondary_embeddings.items():
            sim = self._cosine_similarity(embedding, sec_embedding)
            if sim > self.config.speaker_threshold and sim > best_similarity:
                best_match_id = speaker_id
                best_similarity = sim
                best_priority = SpeakerPriority.SECONDARY

        # Check existing background speakers
        for speaker_id, state in self._speaker_states.items():
            if state.priority == SpeakerPriority.BACKGROUND:
                sim = self._cosine_similarity(embedding, state.embedding)
                if sim > self.config.speaker_threshold and sim > best_similarity:
                    best_match_id = speaker_id
                    best_similarity = sim
                    best_priority = SpeakerPriority.BACKGROUND

        # New speaker
        if best_match_id is None:
            best_match_id = self._assign_speaker_id()
            best_priority = SpeakerPriority.BACKGROUND
            best_similarity = 0.0

        return best_match_id, best_priority, best_similarity

    def _make_focus_decision(
        self,
        priority: SpeakerPriority,
        confidence: float,
    ) -> tuple[bool, str]:
        """
        Decide whether to transcribe based on mode and priority.

        Args:
            priority: Speaker priority level
            confidence: Confidence in speaker identification

        Returns:
            Tuple of (should_transcribe, reason)
        """
        if self._current_mode == FocusMode.SINGLE:
            if priority == SpeakerPriority.PRIMARY:
                return True, "SINGLE mode - primary speaker"
            return False, "SINGLE mode - non-primary speaker dropped"

        if self._current_mode == FocusMode.PRIMARY_PLUS:
            if priority in (SpeakerPriority.PRIMARY, SpeakerPriority.SECONDARY):
                return True, f"PRIMARY_PLUS mode - {priority.name}"
            return False, "PRIMARY_PLUS mode - background speaker dropped"

        if self._current_mode == FocusMode.ALL:
            return True, "ALL mode - all speakers transcribed"

        if self._current_mode == FocusMode.DYNAMIC:
            # In dynamic mode, follow activity
            # Transcribe any confidently identified speaker
            if confidence > self.config.speaker_threshold:
                return True, f"DYNAMIC mode - confident match ({confidence:.2f})"
            return False, f"DYNAMIC mode - low confidence ({confidence:.2f})"

        return False, "Unknown mode"

    def _update_speaker_state(
        self,
        speaker_id: int,
        embedding: mx.array,
        timestamp: float,
        priority: SpeakerPriority,
    ) -> None:
        """Update state for a speaker."""
        if speaker_id not in self._speaker_states:
            self._speaker_states[speaker_id] = SpeakerState(
                speaker_id=speaker_id,
                embedding=embedding,
                priority=priority,
            )

        state = self._speaker_states[speaker_id]
        state.is_active = True
        state.last_seen_time = timestamp
        state.utterance_count += 1

    def _assign_speaker_id(self) -> int:
        """Assign a new unique speaker ID."""
        speaker_id = self._next_speaker_id
        self._next_speaker_id += 1
        return speaker_id

    def _get_speaker_id_by_embedding(self, embedding: mx.array) -> int | None:
        """Find speaker ID by embedding reference."""
        for speaker_id, state in self._speaker_states.items():
            if state.embedding is embedding:
                return speaker_id
        return None

    def _cosine_similarity(self, a: mx.array, b: mx.array) -> float:
        """Compute cosine similarity between two embeddings."""
        a_flat = a.flatten()
        b_flat = b.flatten()
        dot = mx.sum(a_flat * b_flat)
        norm_a = mx.sqrt(mx.sum(a_flat * a_flat))
        norm_b = mx.sqrt(mx.sum(b_flat * b_flat))
        sim = dot / (norm_a * norm_b + 1e-8)
        mx.eval(sim)
        return float(sim)

    def get_stats(self) -> dict:
        """Get manager statistics."""
        return {
            "total_frames": self._total_frames,
            "transcribed_frames": self._transcribed_frames,
            "dropped_frames": self._dropped_frames,
            "transcription_rate": (
                self._transcribed_frames / max(self._total_frames, 1)
            ),
            "num_speakers": len(self._speaker_states),
            "current_mode": self._current_mode.name,
            "current_active_speaker": self._current_active_speaker,
        }

    def get_speaker_states(self) -> dict[int, SpeakerState]:
        """Get all speaker states."""
        return self._speaker_states.copy()

    def clear(self) -> None:
        """Clear all state."""
        self._primary_embedding = None
        self._secondary_embeddings.clear()
        self._speaker_states.clear()
        self._next_speaker_id = 0
        self._current_active_speaker = None
        self._total_frames = 0
        self._transcribed_frames = 0
        self._dropped_frames = 0


def create_voice_focus_manager(
    mode: FocusMode = FocusMode.PRIMARY_PLUS,
    speaker_threshold: float = 0.7,
    **kwargs,
) -> VoiceFocusManager:
    """
    Factory function to create a voice focus manager.

    Args:
        mode: Default focus mode
        speaker_threshold: Threshold for speaker matching
        **kwargs: Additional config options

    Returns:
        Configured VoiceFocusManager
    """
    config = VoiceFocusConfig(
        default_mode=mode,
        speaker_threshold=speaker_threshold,
        **kwargs,
    )
    return VoiceFocusManager(config=config)


# Module exports
__all__ = [
    "FocusMode",
    "FocusResult",
    "SpeakerPriority",
    "SpeakerState",
    "VoiceFocusConfig",
    "VoiceFocusManager",
    "create_voice_focus_manager",
]
