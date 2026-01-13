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
Tests for VoiceFocusManager.

Tests cover:
1. Focus modes (SINGLE, PRIMARY_PLUS, ALL, DYNAMIC)
2. Speaker priority levels
3. Speaker identification and matching
4. Statistics tracking
"""
# ruff: noqa: SLF001, RET504

import mlx.core as mx
import pytest

from tools.whisper_mlx.sota.voice_focus_manager import (
    FocusMode,
    FocusResult,
    SpeakerPriority,
    SpeakerState,
    VoiceFocusConfig,
    VoiceFocusManager,
    create_voice_focus_manager,
)


def make_embedding(seed: int = 0, dim: int = 192) -> mx.array:
    """Create a mock speaker embedding."""
    mx.random.seed(seed)
    emb = mx.random.normal((dim,))
    # Normalize
    emb = emb / mx.sqrt(mx.sum(emb * emb))
    return emb


class TestFocusMode:
    """Tests for FocusMode enum."""

    def test_mode_values(self):
        """Test all modes exist."""
        assert FocusMode.SINGLE is not None
        assert FocusMode.PRIMARY_PLUS is not None
        assert FocusMode.ALL is not None
        assert FocusMode.DYNAMIC is not None

    def test_mode_ordering(self):
        """Test mode values are distinct."""
        modes = [FocusMode.SINGLE, FocusMode.PRIMARY_PLUS, FocusMode.ALL, FocusMode.DYNAMIC]
        values = [m.value for m in modes]
        assert len(set(values)) == 4


class TestSpeakerPriority:
    """Tests for SpeakerPriority enum."""

    def test_priority_values(self):
        """Test all priorities exist."""
        assert SpeakerPriority.PRIMARY is not None
        assert SpeakerPriority.SECONDARY is not None
        assert SpeakerPriority.BACKGROUND is not None
        assert SpeakerPriority.UNKNOWN is not None


class TestVoiceFocusConfig:
    """Tests for configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = VoiceFocusConfig()
        assert config.default_mode == FocusMode.PRIMARY_PLUS
        assert config.speaker_threshold == 0.7
        assert config.min_vad_confidence == 0.5

    def test_custom_config(self):
        """Test custom configuration."""
        config = VoiceFocusConfig(
            default_mode=FocusMode.SINGLE,
            speaker_threshold=0.8,
        )
        assert config.default_mode == FocusMode.SINGLE
        assert config.speaker_threshold == 0.8


class TestSpeakerState:
    """Tests for SpeakerState dataclass."""

    def test_state_creation(self):
        """Test state creation with defaults."""
        emb = make_embedding()
        state = SpeakerState(speaker_id=0, embedding=emb)
        assert state.speaker_id == 0
        assert state.priority == SpeakerPriority.UNKNOWN
        assert state.is_active is False


class TestVoiceFocusManager:
    """Tests for VoiceFocusManager."""

    def test_manager_creation(self):
        """Test basic manager creation."""
        manager = VoiceFocusManager()
        assert manager.config is not None
        assert manager.get_mode() == FocusMode.PRIMARY_PLUS

    def test_set_primary_speaker(self):
        """Test setting primary speaker."""
        manager = VoiceFocusManager()
        emb = make_embedding(seed=42)
        speaker_id = manager.set_primary_speaker(emb)

        assert speaker_id == 0
        assert manager._primary_embedding is not None
        assert speaker_id in manager._speaker_states
        assert manager._speaker_states[speaker_id].priority == SpeakerPriority.PRIMARY

    def test_add_secondary_speaker(self):
        """Test adding secondary speakers."""
        manager = VoiceFocusManager()
        emb1 = make_embedding(seed=1)
        emb2 = make_embedding(seed=2)

        id1 = manager.add_secondary_speaker(emb1)
        id2 = manager.add_secondary_speaker(emb2)

        assert id1 != id2
        assert id1 in manager._secondary_embeddings
        assert id2 in manager._secondary_embeddings

    def test_set_mode(self):
        """Test setting focus mode."""
        manager = VoiceFocusManager()
        manager.set_mode(FocusMode.SINGLE)
        assert manager.get_mode() == FocusMode.SINGLE

        manager.set_mode(FocusMode.ALL)
        assert manager.get_mode() == FocusMode.ALL

    def test_process_frame_no_speech(self):
        """Test processing frame with no speech."""
        manager = VoiceFocusManager()
        emb = make_embedding()

        result = manager.process_frame(
            vad_result=0.1,  # Low VAD
            speaker_embedding=emb,
            timestamp=0.0,
        )

        assert result.should_transcribe is False
        assert "No speech" in result.reason

    def test_process_frame_single_mode_primary(self):
        """Test SINGLE mode with primary speaker."""
        manager = VoiceFocusManager()
        manager.set_mode(FocusMode.SINGLE)

        primary_emb = make_embedding(seed=42)
        manager.set_primary_speaker(primary_emb)

        # Process frame with primary speaker
        result = manager.process_frame(
            vad_result=0.9,
            speaker_embedding=primary_emb,
            timestamp=1.0,
        )

        assert result.should_transcribe is True
        assert result.priority == SpeakerPriority.PRIMARY

    def test_process_frame_single_mode_other(self):
        """Test SINGLE mode with non-primary speaker."""
        manager = VoiceFocusManager()
        manager.set_mode(FocusMode.SINGLE)

        primary_emb = make_embedding(seed=42)
        other_emb = make_embedding(seed=99)  # Different speaker
        manager.set_primary_speaker(primary_emb)

        result = manager.process_frame(
            vad_result=0.9,
            speaker_embedding=other_emb,
            timestamp=1.0,
        )

        assert result.should_transcribe is False
        assert "non-primary" in result.reason.lower()

    def test_process_frame_all_mode(self):
        """Test ALL mode transcribes everyone."""
        manager = VoiceFocusManager()
        manager.set_mode(FocusMode.ALL)

        # Any speaker should be transcribed
        emb = make_embedding(seed=123)
        result = manager.process_frame(
            vad_result=0.9,
            speaker_embedding=emb,
            timestamp=1.0,
        )

        assert result.should_transcribe is True
        assert result.mode == FocusMode.ALL

    def test_process_frame_primary_plus_mode(self):
        """Test PRIMARY_PLUS mode with secondary speaker."""
        manager = VoiceFocusManager()
        manager.set_mode(FocusMode.PRIMARY_PLUS)

        primary_emb = make_embedding(seed=1)
        secondary_emb = make_embedding(seed=2)
        background_emb = make_embedding(seed=3)

        manager.set_primary_speaker(primary_emb)
        manager.add_secondary_speaker(secondary_emb)

        # Primary should be transcribed
        result1 = manager.process_frame(
            vad_result=0.9,
            speaker_embedding=primary_emb,
            timestamp=1.0,
        )
        assert result1.should_transcribe is True

        # Secondary should be transcribed
        result2 = manager.process_frame(
            vad_result=0.9,
            speaker_embedding=secondary_emb,
            timestamp=2.0,
        )
        assert result2.should_transcribe is True

        # Background should NOT be transcribed
        result3 = manager.process_frame(
            vad_result=0.9,
            speaker_embedding=background_emb,
            timestamp=3.0,
        )
        assert result3.should_transcribe is False

    def test_get_stats(self):
        """Test statistics retrieval."""
        manager = VoiceFocusManager()
        stats = manager.get_stats()

        assert "total_frames" in stats
        assert "transcribed_frames" in stats
        assert "dropped_frames" in stats
        assert "num_speakers" in stats
        assert "current_mode" in stats

    def test_clear(self):
        """Test clearing manager state."""
        manager = VoiceFocusManager()
        manager.set_primary_speaker(make_embedding(seed=1))
        manager.add_secondary_speaker(make_embedding(seed=2))

        assert len(manager._speaker_states) > 0

        manager.clear()

        assert len(manager._speaker_states) == 0
        assert manager._primary_embedding is None

    def test_cosine_similarity(self):
        """Test cosine similarity computation."""
        manager = VoiceFocusManager()

        # Same vector should have similarity ~1
        emb = make_embedding(seed=1)
        sim_same = manager._cosine_similarity(emb, emb)
        assert sim_same > 0.99

        # Orthogonal vectors should have low similarity
        emb1 = mx.array([1.0, 0.0, 0.0])
        emb2 = mx.array([0.0, 1.0, 0.0])
        sim_orth = manager._cosine_similarity(emb1, emb2)
        assert abs(sim_orth) < 0.01


class TestFocusResult:
    """Tests for FocusResult dataclass."""

    def test_result_creation(self):
        """Test result creation."""
        result = FocusResult(
            should_transcribe=True,
            active_speaker_id=1,
            priority=SpeakerPriority.PRIMARY,
            confidence=0.95,
            mode=FocusMode.SINGLE,
            reason="Test reason",
        )
        assert result.should_transcribe is True
        assert result.active_speaker_id == 1
        assert result.priority == SpeakerPriority.PRIMARY


class TestCreateVoiceFocusManager:
    """Tests for factory function."""

    def test_create_default(self):
        """Test default factory creation."""
        manager = create_voice_focus_manager()
        assert isinstance(manager, VoiceFocusManager)
        assert manager.get_mode() == FocusMode.PRIMARY_PLUS

    def test_create_with_mode(self):
        """Test factory with custom mode."""
        manager = create_voice_focus_manager(mode=FocusMode.SINGLE)
        assert manager.get_mode() == FocusMode.SINGLE

    def test_create_with_threshold(self):
        """Test factory with custom threshold."""
        manager = create_voice_focus_manager(speaker_threshold=0.9)
        assert manager.config.speaker_threshold == 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
