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
Tests for dual-stream rich audio architecture (Phase 7).

Tests the StreamEvent/Consumer pattern without requiring full models.
"""

from dataclasses import dataclass

import pytest

# Import dual_stream module
from tools.whisper_mlx.dual_stream import (
    EventType,
    RichDualStreamConfig,
    RichStreamConsumer,
    StreamEvent,
    event_to_dict,
)

# =============================================================================
# Mock RichToken for testing
# =============================================================================

@dataclass
class MockRichToken:
    """Minimal RichToken mock for testing."""
    alignment_id: str
    stream: str
    timestamp_ms: float
    start_time_ms: float
    end_time_ms: float
    start_frame: int
    end_frame: int
    token: str
    token_id: int
    confidence: float
    language: str = "en"
    language_confidence: float = 1.0
    emotion: str = "neutral"
    emotion_confidence: float = 0.9
    pitch_hz: float = 150.0
    pitch_confidence: float = 0.8
    phonemes: list[str] | None = None
    phoneme_confidence: list[float] | None = None
    phoneme_deviation: float = 0.0
    para_class: int | None = None
    para_confidence: float | None = None
    speaker_embedding: list[float] | None = None


def create_mock_token(
    alignment_id: str,
    text: str,
    stream: str = "ctc",
    confidence: float = 0.9,
    start_ms: float = 0.0,
) -> MockRichToken:
    """Create a mock token for testing."""
    return MockRichToken(
        alignment_id=alignment_id,
        stream=stream,
        timestamp_ms=start_ms + 20,
        start_time_ms=start_ms,
        end_time_ms=start_ms + 20,
        start_frame=int(start_ms / 20),
        end_frame=int(start_ms / 20) + 1,
        token=text,
        token_id=hash(text) % 50000,
        confidence=confidence,
        phonemes=None,
        phoneme_confidence=None,
    )


# =============================================================================
# Test: Event Types
# =============================================================================

def test_event_types():
    """Test EventType enum has all expected values."""
    assert EventType.TOKEN.value == "token"
    assert EventType.CONFIRM.value == "confirm"
    assert EventType.DIFF.value == "diff"
    assert EventType.BACKTRACK.value == "backtrack"
    assert EventType.FINAL.value == "final"


def test_stream_event_creation():
    """Test creating StreamEvent objects."""
    event = StreamEvent(
        event_type=EventType.TOKEN,
        alignment_id="test123",
        stream="ctc",
        timestamp_ms=1000.0,
    )
    assert event.event_type == EventType.TOKEN
    assert event.alignment_id == "test123"
    assert event.stream == "ctc"
    assert event.timestamp_ms == 1000.0


def test_stream_event_with_token():
    """Test StreamEvent with attached token."""
    token = create_mock_token("t1", "Hello")
    event = StreamEvent(
        event_type=EventType.TOKEN,
        alignment_id="t1",
        stream="ctc",
        timestamp_ms=100.0,
        token=token,
    )
    assert event.token is not None
    assert event.token.token == "Hello"


# =============================================================================
# Test: Consumer Basic Operations
# =============================================================================

def test_consumer_initialization():
    """Test consumer initializes with empty state."""
    consumer = RichStreamConsumer()
    assert consumer.get_display_text() == ""
    assert consumer.get_confirmed_text() == ""
    stats = consumer.get_stats()
    assert stats["total_tokens"] == 0
    assert stats["confirmed_tokens"] == 0


def test_consumer_single_token():
    """Test consumer handles a single CTC token."""
    consumer = RichStreamConsumer()

    token = create_mock_token("t1", "Hello")
    event = StreamEvent(
        event_type=EventType.TOKEN,
        alignment_id="t1",
        stream="ctc",
        timestamp_ms=100.0,
        token=token,
    )

    consumer.handle_event(event)

    assert consumer.get_display_text() == "Hello"
    assert consumer.get_provisional_text() == "Hello"
    assert consumer.get_confirmed_text() == ""  # Not yet confirmed

    stats = consumer.get_stats()
    assert stats["total_tokens"] == 1
    assert stats["provisional_tokens"] == 1


def test_consumer_multiple_tokens():
    """Test consumer handles multiple tokens in sequence."""
    consumer = RichStreamConsumer()

    words = ["Hello", " ", "world"]
    for i, word in enumerate(words):
        token = create_mock_token(f"t{i}", word, start_ms=i * 100)
        event = StreamEvent(
            event_type=EventType.TOKEN,
            alignment_id=f"t{i}",
            stream="ctc",
            timestamp_ms=i * 100 + 20,
            token=token,
        )
        consumer.handle_event(event)

    assert consumer.get_display_text() == "Hello world"
    assert len(consumer.get_token_states()) == 3


# =============================================================================
# Test: Confirmation Flow
# =============================================================================

def test_consumer_confirm_token():
    """Test consumer confirms token from decoder."""
    consumer = RichStreamConsumer()

    # CTC emits token
    token = create_mock_token("t1", "Hello")
    ctc_event = StreamEvent(
        event_type=EventType.TOKEN,
        alignment_id="t1",
        stream="ctc",
        timestamp_ms=100.0,
        token=token,
    )
    consumer.handle_event(ctc_event)

    assert consumer.get_provisional_text() == "Hello"
    assert consumer.get_confirmed_text() == ""

    # Decoder confirms
    confirm_event = StreamEvent(
        event_type=EventType.CONFIRM,
        alignment_id="t1",
        stream="decoder",
        timestamp_ms=500.0,
        confirmed=True,
    )
    consumer.handle_event(confirm_event)

    assert consumer.get_provisional_text() == ""
    assert consumer.get_confirmed_text() == "Hello"

    stats = consumer.get_stats()
    assert stats["confirmed_tokens"] == 1


def test_consumer_decoder_token_auto_confirms():
    """Test that decoder tokens are auto-confirmed."""
    consumer = RichStreamConsumer()

    # Decoder emits token directly (no CTC)
    token = create_mock_token("t1", "Hello", stream="decoder")
    event = StreamEvent(
        event_type=EventType.TOKEN,
        alignment_id="t1",
        stream="decoder",
        timestamp_ms=100.0,
        token=token,
    )
    consumer.handle_event(event)

    # Decoder tokens are immediately confirmed
    assert consumer.get_confirmed_text() == "Hello"
    assert consumer.get_provisional_text() == ""


# =============================================================================
# Test: Diff Flow
# =============================================================================

def test_consumer_diff_updates_token():
    """Test consumer handles diff event (decoder disagrees)."""
    consumer = RichStreamConsumer()

    # CTC emits "their"
    ctc_token = create_mock_token("t1", "their")
    ctc_event = StreamEvent(
        event_type=EventType.TOKEN,
        alignment_id="t1",
        stream="ctc",
        timestamp_ms=100.0,
        token=ctc_token,
    )
    consumer.handle_event(ctc_event)

    assert consumer.get_display_text() == "their"

    # Decoder disagrees, emits "there"
    decoder_token = create_mock_token("t1", "there", stream="decoder")
    diff_event = StreamEvent(
        event_type=EventType.DIFF,
        alignment_id="t1",
        stream="decoder",
        timestamp_ms=500.0,
        token=decoder_token,
        diff={"field": "token", "ctc": "their", "decoder": "there"},
    )
    consumer.handle_event(diff_event)

    # Token should be updated to decoder version
    assert consumer.get_display_text() == "there"
    assert consumer.get_confirmed_text() == "there"


# =============================================================================
# Test: Backtrack Flow
# =============================================================================

def test_consumer_backtrack_removes_unconfirmed():
    """Test consumer handles backtrack by removing unconfirmed tokens."""
    consumer = RichStreamConsumer()

    # Emit tokens: "Hello", " ", "world"
    words = ["Hello", " ", "world"]
    for i, word in enumerate(words):
        token = create_mock_token(f"t{i}", word)
        event = StreamEvent(
            event_type=EventType.TOKEN,
            alignment_id=f"t{i}",
            stream="ctc",
            timestamp_ms=i * 100,
            token=token,
        )
        consumer.handle_event(event)

    assert consumer.get_display_text() == "Hello world"

    # Confirm first token
    confirm_event = StreamEvent(
        event_type=EventType.CONFIRM,
        alignment_id="t0",
        stream="decoder",
        timestamp_ms=400.0,
        confirmed=True,
    )
    consumer.handle_event(confirm_event)

    # Backtrack to t1 (should remove t1 and t2 if unconfirmed)
    backtrack_event = StreamEvent(
        event_type=EventType.BACKTRACK,
        alignment_id="back1",
        stream="ctc",
        timestamp_ms=500.0,
        backtrack_to_id="t1",
        reason="low_confidence",
    )
    consumer.handle_event(backtrack_event)

    # Only confirmed token should remain
    assert consumer.get_display_text() == "Hello"
    assert consumer.get_confirmed_text() == "Hello"


def test_consumer_backtrack_preserves_confirmed():
    """Test backtrack preserves confirmed tokens."""
    consumer = RichStreamConsumer()

    # Emit and confirm all tokens
    words = ["Hello", " ", "world"]
    for i, word in enumerate(words):
        token = create_mock_token(f"t{i}", word)
        event = StreamEvent(
            event_type=EventType.TOKEN,
            alignment_id=f"t{i}",
            stream="ctc",
            timestamp_ms=i * 100,
            token=token,
        )
        consumer.handle_event(event)

        # Confirm each
        confirm_event = StreamEvent(
            event_type=EventType.CONFIRM,
            alignment_id=f"t{i}",
            stream="decoder",
            timestamp_ms=i * 100 + 50,
            confirmed=True,
        )
        consumer.handle_event(confirm_event)

    assert consumer.get_confirmed_text() == "Hello world"

    # Backtrack - should not remove confirmed
    backtrack_event = StreamEvent(
        event_type=EventType.BACKTRACK,
        alignment_id="back1",
        stream="ctc",
        timestamp_ms=500.0,
        backtrack_to_id="t1",
        reason="low_confidence",
    )
    consumer.handle_event(backtrack_event)

    # All confirmed, none removed
    assert consumer.get_display_text() == "Hello world"


# =============================================================================
# Test: Final Event
# =============================================================================

def test_consumer_final_confirms_all():
    """Test final event confirms all remaining tokens."""
    consumer = RichStreamConsumer()

    # Emit tokens without confirming
    words = ["Hello", " ", "world"]
    for i, word in enumerate(words):
        token = create_mock_token(f"t{i}", word)
        event = StreamEvent(
            event_type=EventType.TOKEN,
            alignment_id=f"t{i}",
            stream="ctc",
            timestamp_ms=i * 100,
            token=token,
        )
        consumer.handle_event(event)

    assert consumer.get_provisional_text() == "Hello world"
    assert consumer.get_confirmed_text() == ""

    # Final event
    final_event = StreamEvent(
        event_type=EventType.FINAL,
        alignment_id="final1",
        stream="decoder",
        timestamp_ms=1000.0,
    )
    consumer.handle_event(final_event)

    # All should be confirmed
    assert consumer.get_provisional_text() == ""
    assert consumer.get_confirmed_text() == "Hello world"


# =============================================================================
# Test: Consumer Reset
# =============================================================================

def test_consumer_reset():
    """Test consumer reset clears all state."""
    consumer = RichStreamConsumer()

    # Add some tokens
    for i in range(3):
        token = create_mock_token(f"t{i}", f"word{i}")
        event = StreamEvent(
            event_type=EventType.TOKEN,
            alignment_id=f"t{i}",
            stream="ctc",
            timestamp_ms=i * 100,
            token=token,
        )
        consumer.handle_event(event)

    assert consumer.get_stats()["total_tokens"] == 3

    consumer.reset()

    assert consumer.get_display_text() == ""
    assert consumer.get_stats()["total_tokens"] == 0
    assert len(consumer.get_event_history()) == 0


# =============================================================================
# Test: Callbacks
# =============================================================================

def test_consumer_callbacks():
    """Test consumer callback registration."""
    consumer = RichStreamConsumer()

    added_tokens = []
    confirmed_ids = []

    consumer.on_token_added = lambda t: added_tokens.append(t)
    consumer.on_token_confirmed = lambda id: confirmed_ids.append(id)

    # Add token
    token = create_mock_token("t1", "Hello")
    event = StreamEvent(
        event_type=EventType.TOKEN,
        alignment_id="t1",
        stream="ctc",
        timestamp_ms=100.0,
        token=token,
    )
    consumer.handle_event(event)

    assert len(added_tokens) == 1
    assert added_tokens[0].token.token == "Hello"

    # Confirm token
    confirm_event = StreamEvent(
        event_type=EventType.CONFIRM,
        alignment_id="t1",
        stream="decoder",
        timestamp_ms=200.0,
        confirmed=True,
    )
    consumer.handle_event(confirm_event)

    assert "t1" in confirmed_ids


# =============================================================================
# Test: Serialization
# =============================================================================

def test_event_to_dict():
    """Test StreamEvent serialization to dict."""
    token = create_mock_token("t1", "Hello")
    event = StreamEvent(
        event_type=EventType.TOKEN,
        alignment_id="t1",
        stream="ctc",
        timestamp_ms=100.0,
        token=token,
    )

    d = event_to_dict(event)

    assert d["event_type"] == "token"
    assert d["alignment_id"] == "t1"
    assert d["stream"] == "ctc"
    assert d["timestamp_ms"] == 100.0
    assert "token" in d
    assert d["token"]["text"] == "Hello"


def test_event_to_dict_confirm():
    """Test confirm event serialization."""
    event = StreamEvent(
        event_type=EventType.CONFIRM,
        alignment_id="t1",
        stream="decoder",
        timestamp_ms=500.0,
        confirmed=True,
    )

    d = event_to_dict(event)

    assert d["event_type"] == "confirm"
    assert d["confirmed"] is True


def test_event_to_dict_diff():
    """Test diff event serialization."""
    event = StreamEvent(
        event_type=EventType.DIFF,
        alignment_id="t1",
        stream="decoder",
        timestamp_ms=500.0,
        diff={"field": "token", "ctc": "their", "decoder": "there"},
    )

    d = event_to_dict(event)

    assert d["event_type"] == "diff"
    assert d["diff"]["ctc"] == "their"
    assert d["diff"]["decoder"] == "there"


def test_event_to_dict_backtrack():
    """Test backtrack event serialization."""
    event = StreamEvent(
        event_type=EventType.BACKTRACK,
        alignment_id="back1",
        stream="ctc",
        timestamp_ms=500.0,
        backtrack_to_id="t1",
        reason="low_confidence",
    )

    d = event_to_dict(event)

    assert d["event_type"] == "backtrack"
    assert d["backtrack_to_id"] == "t1"
    assert d["reason"] == "low_confidence"


# =============================================================================
# Test: Configuration
# =============================================================================

def test_config_defaults():
    """Test RichDualStreamConfig has sensible defaults."""
    config = RichDualStreamConfig()

    assert config.sample_rate == 16000
    assert config.ctc_chunk_duration == 1.0
    assert config.ctc_interval == 0.3
    assert config.decoder_chunk_duration == 3.0
    assert config.decoder_interval == 1.0


# =============================================================================
# Test: Event History
# =============================================================================

def test_consumer_event_history():
    """Test consumer maintains event history."""
    consumer = RichStreamConsumer()

    events = [
        StreamEvent(EventType.TOKEN, "t1", "ctc", 100.0, token=create_mock_token("t1", "A")),
        StreamEvent(EventType.TOKEN, "t2", "ctc", 200.0, token=create_mock_token("t2", "B")),
        StreamEvent(EventType.CONFIRM, "t1", "decoder", 300.0, confirmed=True),
    ]

    for event in events:
        consumer.handle_event(event)

    history = consumer.get_event_history()
    assert len(history) == 3
    assert history[0].event_type == EventType.TOKEN
    assert history[2].event_type == EventType.CONFIRM


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
