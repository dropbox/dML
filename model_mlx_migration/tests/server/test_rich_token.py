# Copyright 2024-2026 Andrew Yates
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

"""Tests for RichToken output format."""

import json

from src.server.rich_token import (
    ASRMode,
    EmotionLabel,
    HallucinationInfo,
    LanguageInfo,
    ParalinguisticsInfo,
    PhonemeInfo,
    PitchInfo,
    RichToken,
    SingingInfo,
    SpeakerInfo,
    StreamingResponse,
    WordTimestamp,
    create_final_token,
    create_partial_token,
)


class TestWordTimestamp:
    """Tests for WordTimestamp dataclass."""

    def test_basic_creation(self):
        wt = WordTimestamp(word="hello", start_ms=0.0, end_ms=200.0)
        assert wt.word == "hello"
        assert wt.start_ms == 0.0
        assert wt.end_ms == 200.0
        assert wt.confidence == 1.0  # Default

    def test_with_confidence(self):
        wt = WordTimestamp(word="world", start_ms=200.0, end_ms=400.0, confidence=0.95)
        assert wt.confidence == 0.95


class TestPitchInfo:
    """Tests for PitchInfo dataclass."""

    def test_basic_creation(self):
        pitch = PitchInfo(mean_hz=150.0, std_hz=25.0, min_hz=100.0, max_hz=200.0)
        assert pitch.mean_hz == 150.0
        assert pitch.std_hz == 25.0
        assert pitch.min_hz == 100.0
        assert pitch.max_hz == 200.0
        assert pitch.voiced_ratio == 1.0  # Default

    def test_with_frame_f0(self):
        pitch = PitchInfo(
            mean_hz=150.0,
            std_hz=25.0,
            min_hz=100.0,
            max_hz=200.0,
            frame_f0_hz=[140.0, 150.0, 160.0],
            voiced_ratio=0.8,
        )
        assert pitch.frame_f0_hz == [140.0, 150.0, 160.0]
        assert pitch.voiced_ratio == 0.8


class TestPhonemeInfo:
    """Tests for PhonemeInfo dataclass."""

    def test_basic_creation(self):
        phonemes = PhonemeInfo(phonemes=["h", "ɛ", "l", "oʊ"])
        assert phonemes.phonemes == ["h", "ɛ", "l", "oʊ"]
        assert phonemes.confidences == []

    def test_with_confidences(self):
        phonemes = PhonemeInfo(
            phonemes=["h", "ɛ", "l", "oʊ"],
            confidences=[0.9, 0.85, 0.92, 0.88],
        )
        assert len(phonemes.confidences) == 4


class TestParalinguisticsInfo:
    """Tests for ParalinguisticsInfo dataclass."""

    def test_basic_creation(self):
        para = ParalinguisticsInfo(event="laughter", confidence=0.95)
        assert para.event == "laughter"
        assert para.confidence == 0.95

    def test_with_all_probs(self):
        para = ParalinguisticsInfo(
            event="laughter",
            confidence=0.95,
            all_probs={"laughter": 0.95, "speech": 0.03, "silence": 0.02},
        )
        assert para.all_probs["laughter"] == 0.95


class TestLanguageInfo:
    """Tests for LanguageInfo dataclass."""

    def test_basic_creation(self):
        lang = LanguageInfo(language="en", confidence=0.98)
        assert lang.language == "en"
        assert lang.confidence == 0.98

    def test_with_all_probs(self):
        lang = LanguageInfo(
            language="en",
            confidence=0.98,
            all_probs={"en": 0.98, "es": 0.01, "fr": 0.01},
        )
        assert len(lang.all_probs) == 3


class TestSingingInfo:
    """Tests for SingingInfo dataclass."""

    def test_not_singing(self):
        singing = SingingInfo(is_singing=False, singing_confidence=0.1)
        assert not singing.is_singing
        assert singing.techniques == []

    def test_singing_with_techniques(self):
        singing = SingingInfo(
            is_singing=True,
            singing_confidence=0.95,
            techniques=["vibrato", "belt"],
            technique_confidences=[0.9, 0.8],
        )
        assert singing.is_singing
        assert len(singing.techniques) == 2


class TestSpeakerInfo:
    """Tests for SpeakerInfo dataclass."""

    def test_basic_creation(self):
        speaker = SpeakerInfo(embedding=[0.1] * 256)
        assert len(speaker.embedding) == 256
        assert speaker.speaker_id is None

    def test_with_speaker_id(self):
        speaker = SpeakerInfo(
            embedding=[0.1] * 256,
            speaker_id="speaker_1",
            similarity=0.92,
        )
        assert speaker.speaker_id == "speaker_1"
        assert speaker.similarity == 0.92


class TestHallucinationInfo:
    """Tests for HallucinationInfo dataclass."""

    def test_no_hallucination(self):
        hall = HallucinationInfo(score=0.05, is_hallucinated=False)
        assert hall.score == 0.05
        assert not hall.is_hallucinated

    def test_hallucination_detected(self):
        hall = HallucinationInfo(
            score=0.85,
            is_hallucinated=True,
            phoneme_mismatch=0.6,
            energy_silence=0.2,
            repetition=0.05,
        )
        assert hall.is_hallucinated
        assert hall.phoneme_mismatch == 0.6


class TestRichToken:
    """Tests for RichToken dataclass."""

    def test_minimal_creation(self):
        token = RichToken(
            text="hello world",
            confidence=0.95,
            start_time_ms=0.0,
            end_time_ms=1000.0,
        )
        assert token.text == "hello world"
        assert token.confidence == 0.95
        assert token.mode == ASRMode.STREAMING
        assert not token.is_final
        assert not token.is_partial

    def test_full_creation(self):
        token = RichToken(
            text="hello world",
            confidence=0.95,
            start_time_ms=0.0,
            end_time_ms=1000.0,
            mode=ASRMode.HIGH_ACCURACY,
            is_final=True,
            word_timestamps=[
                WordTimestamp("hello", 0.0, 400.0),
                WordTimestamp("world", 500.0, 900.0),
            ],
            emotion=EmotionLabel.HAPPY,
            emotion_confidence=0.85,
            pitch=PitchInfo(150.0, 25.0, 100.0, 200.0),
            phonemes=PhonemeInfo(["h", "ɛ", "l", "oʊ"]),
            language=LanguageInfo("en", 0.98),
            hallucination=HallucinationInfo(0.05, False),
            speaker=SpeakerInfo([0.1] * 256),
            utterance_id="abc123",
        )
        assert token.mode == ASRMode.HIGH_ACCURACY
        assert token.is_final
        assert len(token.word_timestamps) == 2
        assert token.emotion == EmotionLabel.HAPPY
        assert token.pitch.mean_hz == 150.0


class TestRichTokenSerialization:
    """Tests for RichToken JSON serialization."""

    def test_to_dict_minimal(self):
        token = RichToken(
            text="hello",
            confidence=0.9,
            start_time_ms=0.0,
            end_time_ms=500.0,
        )
        d = token.to_dict()
        assert d["text"] == "hello"
        assert d["confidence"] == 0.9
        assert d["mode"] == "streaming"
        assert "emotion" not in d

    def test_to_dict_with_features(self):
        token = RichToken(
            text="hello world",
            confidence=0.95,
            start_time_ms=0.0,
            end_time_ms=1000.0,
            emotion=EmotionLabel.NEUTRAL,
            emotion_confidence=0.85,
            pitch=PitchInfo(150.0, 25.0, 100.0, 200.0),
        )
        d = token.to_dict()
        assert d["emotion"]["label"] == "neutral"
        assert d["emotion"]["confidence"] == 0.85
        assert d["pitch"]["mean_hz"] == 150.0

    def test_to_json(self):
        token = RichToken(
            text="test",
            confidence=0.9,
            start_time_ms=0.0,
            end_time_ms=500.0,
        )
        json_str = token.to_json()
        parsed = json.loads(json_str)
        assert parsed["text"] == "test"

    def test_roundtrip_minimal(self):
        original = RichToken(
            text="hello world",
            confidence=0.95,
            start_time_ms=0.0,
            end_time_ms=1000.0,
        )
        json_str = original.to_json()
        restored = RichToken.from_json(json_str)
        assert restored.text == original.text
        assert restored.confidence == original.confidence

    def test_roundtrip_with_timestamps(self):
        original = RichToken(
            text="hello world",
            confidence=0.95,
            start_time_ms=0.0,
            end_time_ms=1000.0,
            word_timestamps=[
                WordTimestamp("hello", 0.0, 400.0, 0.9),
                WordTimestamp("world", 500.0, 900.0, 0.95),
            ],
        )
        restored = RichToken.from_json(original.to_json())
        assert len(restored.word_timestamps) == 2
        assert restored.word_timestamps[0].word == "hello"
        assert restored.word_timestamps[1].confidence == 0.95

    def test_roundtrip_with_emotion(self):
        original = RichToken(
            text="test",
            confidence=0.9,
            start_time_ms=0.0,
            end_time_ms=500.0,
            emotion=EmotionLabel.HAPPY,
            emotion_confidence=0.85,
        )
        restored = RichToken.from_json(original.to_json())
        assert restored.emotion == EmotionLabel.HAPPY
        assert restored.emotion_confidence == 0.85

    def test_roundtrip_with_pitch(self):
        original = RichToken(
            text="test",
            confidence=0.9,
            start_time_ms=0.0,
            end_time_ms=500.0,
            pitch=PitchInfo(150.0, 25.0, 100.0, 200.0, voiced_ratio=0.8),
        )
        restored = RichToken.from_json(original.to_json())
        assert restored.pitch.mean_hz == 150.0
        assert restored.pitch.voiced_ratio == 0.8

    def test_roundtrip_with_phonemes(self):
        original = RichToken(
            text="test",
            confidence=0.9,
            start_time_ms=0.0,
            end_time_ms=500.0,
            phonemes=PhonemeInfo(["t", "ɛ", "s", "t"], [0.9, 0.8, 0.85, 0.9]),
        )
        restored = RichToken.from_json(original.to_json())
        assert restored.phonemes.phonemes == ["t", "ɛ", "s", "t"]
        assert len(restored.phonemes.confidences) == 4

    def test_roundtrip_with_language(self):
        original = RichToken(
            text="test",
            confidence=0.9,
            start_time_ms=0.0,
            end_time_ms=500.0,
            language=LanguageInfo("en", 0.98),
        )
        restored = RichToken.from_json(original.to_json())
        assert restored.language.language == "en"

    def test_roundtrip_with_hallucination(self):
        original = RichToken(
            text="test",
            confidence=0.9,
            start_time_ms=0.0,
            end_time_ms=500.0,
            hallucination=HallucinationInfo(
                0.05, False, phoneme_mismatch=0.03, energy_silence=0.02,
            ),
        )
        restored = RichToken.from_json(original.to_json())
        assert not restored.hallucination.is_hallucinated
        assert restored.hallucination.phoneme_mismatch == 0.03

    def test_roundtrip_with_speaker(self):
        original = RichToken(
            text="test",
            confidence=0.9,
            start_time_ms=0.0,
            end_time_ms=500.0,
            speaker=SpeakerInfo([0.1] * 256, "spk1", 0.92),
        )
        restored = RichToken.from_json(original.to_json())
        assert len(restored.speaker.embedding) == 256
        assert restored.speaker.speaker_id == "spk1"


class TestStreamingResponse:
    """Tests for StreamingResponse wrapper."""

    def test_partial_response(self):
        token = create_partial_token(
            text="hello",
            start_ms=0.0,
            end_ms=500.0,
            chunk_index=1,
        )
        response = StreamingResponse(
            type="partial",
            token=token,
            sequence=1,
        )
        assert response.type == "partial"
        assert response.token.text == "hello"

    def test_final_response(self):
        token = create_final_token(
            text="hello world",
            start_ms=0.0,
            end_ms=1000.0,
        )
        response = StreamingResponse(
            type="final",
            token=token,
            sequence=5,
        )
        d = response.to_dict()
        assert d["type"] == "final"
        assert d["token"]["text"] == "hello world"

    def test_error_response(self):
        response = StreamingResponse(
            type="error",
            error="Connection timeout",
            sequence=10,
        )
        d = response.to_dict()
        assert d["type"] == "error"
        assert d["error"] == "Connection timeout"

    def test_metadata_response(self):
        response = StreamingResponse(
            type="metadata",
            metadata={"status": "connected", "session_id": "abc123"},
            sequence=0,
        )
        d = response.to_dict()
        assert d["metadata"]["status"] == "connected"


class TestFactoryFunctions:
    """Tests for token factory functions."""

    def test_create_partial_token(self):
        token = create_partial_token(
            text="hello",
            start_ms=0.0,
            end_ms=500.0,
            chunk_index=1,
            utterance_id="test123",
        )
        assert token.text == "hello"
        assert token.is_partial
        assert not token.is_final
        assert token.mode == ASRMode.STREAMING
        assert token.chunk_index == 1

    def test_create_final_token(self):
        token = create_final_token(
            text="hello world",
            start_ms=0.0,
            end_ms=1000.0,
            utterance_id="test123",
            mode=ASRMode.HIGH_ACCURACY,
        )
        assert token.text == "hello world"
        assert token.is_final
        assert not token.is_partial
        assert token.mode == ASRMode.HIGH_ACCURACY

    def test_create_final_token_with_timestamps(self):
        timestamps = [
            WordTimestamp("hello", 0.0, 400.0),
            WordTimestamp("world", 500.0, 900.0),
        ]
        token = create_final_token(
            text="hello world",
            start_ms=0.0,
            end_ms=1000.0,
            word_timestamps=timestamps,
        )
        assert len(token.word_timestamps) == 2


class TestASRMode:
    """Tests for ASRMode enum."""

    def test_streaming_mode(self):
        assert ASRMode.STREAMING.value == "streaming"

    def test_high_accuracy_mode(self):
        assert ASRMode.HIGH_ACCURACY.value == "high_accuracy"

    def test_from_string(self):
        assert ASRMode("streaming") == ASRMode.STREAMING
        assert ASRMode("high_accuracy") == ASRMode.HIGH_ACCURACY


class TestEmotionLabel:
    """Tests for EmotionLabel enum."""

    def test_all_labels(self):
        labels = [e.value for e in EmotionLabel]
        assert "neutral" in labels
        assert "happy" in labels
        assert "sad" in labels
        assert "angry" in labels
        assert "fear" in labels
        assert "disgust" in labels
        assert "surprise" in labels
        assert "contempt" in labels

    def test_from_string(self):
        assert EmotionLabel("happy") == EmotionLabel.HAPPY
