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
Unit tests for Prosody-Conditioned Beam Search.

Tests the prosody beam search module including:
- ProsodyFeatures dataclass
- ProsodyBoostConfig
- ProsodyBeamSearch scoring logic
- Integration with multi-head architecture
"""

import mlx.core as mx
import mlx.nn as nn
import pytest


class TestProsodyFeatures:
    """Tests for ProsodyFeatures dataclass."""

    def test_default_initialization(self):
        """Test default feature values."""
        from tools.whisper_mlx.prosody_beam_search import ProsodyFeatures

        features = ProsodyFeatures()

        assert features.pitch_values is None
        assert features.pitch_slope == 0.0
        assert features.pitch_range == 0.0
        assert features.voicing_ratio == 0.0
        assert features.emotion_probs == {}
        assert features.dominant_emotion == "neutral"
        assert features.emotion_confidence == 0.0
        assert features.intensity == 0.0
        assert features.audio_duration == 0.0

    def test_custom_initialization(self):
        """Test custom feature values."""
        from tools.whisper_mlx.prosody_beam_search import ProsodyFeatures

        pitch_values = mx.array([100.0, 110.0, 120.0])
        emotion_probs = {"happy": 0.8, "neutral": 0.2}

        features = ProsodyFeatures(
            pitch_values=pitch_values,
            pitch_slope=0.15,
            pitch_range=50.0,
            voicing_ratio=0.9,
            emotion_probs=emotion_probs,
            dominant_emotion="happy",
            emotion_confidence=0.8,
            intensity=0.7,
            audio_duration=3.5,
        )

        assert features.pitch_slope == 0.15
        assert features.pitch_range == 50.0
        assert features.voicing_ratio == 0.9
        assert features.emotion_probs == emotion_probs
        assert features.dominant_emotion == "happy"
        assert features.emotion_confidence == 0.8
        assert features.intensity == 0.7
        assert features.audio_duration == 3.5


class TestProsodyBoostConfig:
    """Tests for ProsodyBoostConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from tools.whisper_mlx.prosody_beam_search import ProsodyBoostConfig

        config = ProsodyBoostConfig()

        # Question mark defaults
        assert config.question_rising_pitch_boost == 3.0
        assert config.question_pitch_threshold == 0.1

        # Period defaults
        assert config.period_falling_pitch_boost == 1.5
        assert config.period_pitch_threshold == -0.05

        # Exclamation defaults
        assert config.exclamation_surprise_boost == 2.5
        assert config.exclamation_joy_boost == 2.5
        assert config.exclamation_anger_boost == 2.0
        assert config.exclamation_emotion_threshold == 0.3

        # Intensity defaults
        assert config.high_intensity_punctuation_boost == 1.2
        assert config.high_intensity_threshold == 0.7

        # Flags
        assert config.enable_pitch_rules is True
        assert config.enable_emotion_rules is True
        assert config.enable_intensity_rules is True

    def test_custom_config(self):
        """Test custom configuration values."""
        from tools.whisper_mlx.prosody_beam_search import ProsodyBoostConfig

        config = ProsodyBoostConfig(
            question_rising_pitch_boost=4.0,
            exclamation_surprise_boost=3.0,
            enable_pitch_rules=False,
        )

        assert config.question_rising_pitch_boost == 4.0
        assert config.exclamation_surprise_boost == 3.0
        assert config.enable_pitch_rules is False


class TestProsodyBeamSearchScoring:
    """Tests for prosody score factor computation."""

    @pytest.fixture
    def prosody_search(self):
        """Create ProsodyBeamSearch without heads for scoring tests."""
        from tools.whisper_mlx.prosody_beam_search import ProsodyBeamSearch

        return ProsodyBeamSearch(pitch_head=None, emotion_head=None)

    @pytest.fixture
    def mock_tokenizer(self, prosody_search):
        """Create mock tokenizer with punctuation IDs."""
        class MockTokenizer:
            def encode(self, text):
                punctuation_map = {
                    "?": [100],
                    ".": [101],
                    "!": [102],
                    ",": [103],
                }
                return punctuation_map.get(text, [0])

        tokenizer = MockTokenizer()
        # Initialize punctuation IDs
        prosody_search._initialize_punctuation_ids(tokenizer)
        return tokenizer

    def test_no_boost_for_non_punctuation(self, prosody_search, mock_tokenizer):
        """Test that non-punctuation tokens get factor 1.0."""
        from tools.whisper_mlx.prosody_beam_search import ProsodyFeatures

        features = ProsodyFeatures(pitch_slope=0.2)

        # Token ID 50 is not punctuation
        factor = prosody_search.prosody_score_factor(
            token_id=50,
            prosody_features=features,
            tokenizer=mock_tokenizer,
        )

        assert factor == 1.0

    def test_question_mark_boost_with_rising_pitch(self, prosody_search, mock_tokenizer):
        """Test question mark boost when pitch is rising."""
        from tools.whisper_mlx.prosody_beam_search import ProsodyFeatures

        features = ProsodyFeatures(pitch_slope=0.15)  # Rising pitch

        # Token ID 100 = "?"
        factor = prosody_search.prosody_score_factor(
            token_id=100,
            prosody_features=features,
            tokenizer=mock_tokenizer,
        )

        # Should get 3x boost (default config)
        assert factor == 3.0

    def test_no_question_boost_with_falling_pitch(self, prosody_search, mock_tokenizer):
        """Test no question boost when pitch is falling."""
        from tools.whisper_mlx.prosody_beam_search import ProsodyFeatures

        features = ProsodyFeatures(pitch_slope=-0.1)  # Falling pitch

        factor = prosody_search.prosody_score_factor(
            token_id=100,  # "?"
            prosody_features=features,
            tokenizer=mock_tokenizer,
        )

        assert factor == 1.0

    def test_period_boost_with_falling_pitch(self, prosody_search, mock_tokenizer):
        """Test period boost when pitch is falling."""
        from tools.whisper_mlx.prosody_beam_search import ProsodyFeatures

        features = ProsodyFeatures(pitch_slope=-0.1)  # Falling pitch

        factor = prosody_search.prosody_score_factor(
            token_id=101,  # "."
            prosody_features=features,
            tokenizer=mock_tokenizer,
        )

        # Should get 1.5x boost (default config)
        assert factor == 1.5

    def test_exclamation_boost_with_surprise_emotion(self, prosody_search, mock_tokenizer):
        """Test exclamation boost with surprise emotion."""
        from tools.whisper_mlx.prosody_beam_search import ProsodyFeatures

        features = ProsodyFeatures(
            emotion_probs={"surprise": 0.5, "neutral": 0.3, "happy": 0.2},
        )

        factor = prosody_search.prosody_score_factor(
            token_id=102,  # "!"
            prosody_features=features,
            tokenizer=mock_tokenizer,
        )

        # Should get 2.5x boost (default config)
        assert factor == 2.5

    def test_exclamation_boost_with_joy_emotion(self, prosody_search, mock_tokenizer):
        """Test exclamation boost with joy/happy emotion."""
        from tools.whisper_mlx.prosody_beam_search import ProsodyFeatures

        features = ProsodyFeatures(
            emotion_probs={"joy": 0.5, "neutral": 0.3, "surprise": 0.2},
        )

        factor = prosody_search.prosody_score_factor(
            token_id=102,  # "!"
            prosody_features=features,
            tokenizer=mock_tokenizer,
        )

        assert factor == 2.5

    def test_exclamation_boost_with_anger_emotion(self, prosody_search, mock_tokenizer):
        """Test exclamation boost with anger emotion."""
        from tools.whisper_mlx.prosody_beam_search import ProsodyFeatures

        features = ProsodyFeatures(
            emotion_probs={"anger": 0.5, "neutral": 0.3},
        )

        factor = prosody_search.prosody_score_factor(
            token_id=102,  # "!"
            prosody_features=features,
            tokenizer=mock_tokenizer,
        )

        assert factor == 2.0

    def test_high_intensity_punctuation_boost(self, prosody_search, mock_tokenizer):
        """Test high intensity boost for any punctuation."""
        from tools.whisper_mlx.prosody_beam_search import ProsodyFeatures

        features = ProsodyFeatures(intensity=0.8)  # High intensity

        # Test with comma (should only get intensity boost)
        factor = prosody_search.prosody_score_factor(
            token_id=103,  # ","
            prosody_features=features,
            tokenizer=mock_tokenizer,
        )

        assert factor == 1.2

    def test_combined_boosts(self, prosody_search, mock_tokenizer):
        """Test multiple boosts combining multiplicatively."""
        from tools.whisper_mlx.prosody_beam_search import ProsodyFeatures

        features = ProsodyFeatures(
            emotion_probs={"surprise": 0.5, "neutral": 0.5},  # Surprise triggers boost
            intensity=0.8,  # Also triggers boost
        )

        factor = prosody_search.prosody_score_factor(
            token_id=102,  # "!"
            prosody_features=features,
            tokenizer=mock_tokenizer,
        )

        # surprise (2.5) * intensity (1.2) = 3.0
        # Note: emotion rules use elif logic, so only surprise is applied (not joy)
        assert factor == pytest.approx(3.0, rel=0.01)

    def test_disabled_pitch_rules(self, mock_tokenizer):
        """Test that disabled pitch rules skip pitch-based boosts."""
        from tools.whisper_mlx.prosody_beam_search import (
            ProsodyBeamSearch,
            ProsodyBoostConfig,
            ProsodyFeatures,
        )

        config = ProsodyBoostConfig(enable_pitch_rules=False)
        prosody_search = ProsodyBeamSearch(config=config)
        prosody_search._initialize_punctuation_ids(mock_tokenizer)

        features = ProsodyFeatures(pitch_slope=0.2)  # Rising pitch

        factor = prosody_search.prosody_score_factor(
            token_id=100,  # "?"
            prosody_features=features,
            tokenizer=mock_tokenizer,
        )

        # Should be 1.0 because pitch rules are disabled
        assert factor == 1.0


class TestProsodyFeatureExtraction:
    """Tests for prosody feature extraction from encoder output."""

    @pytest.fixture
    def mock_pitch_head(self):
        """Create a mock pitch head."""
        class MockPitchHead(nn.Module):
            def __call__(self, encoder_output):
                batch, T, d_model = encoder_output.shape
                # Return rising pitch pattern
                pitch_hz = mx.linspace(100, 200, T)[None, :]  # Rising from 100 to 200 Hz
                voicing_prob = mx.ones((batch, T)) * 0.9  # Mostly voiced
                return pitch_hz, voicing_prob

        return MockPitchHead()

    @pytest.fixture
    def mock_emotion_head(self):
        """Create a mock emotion head."""
        class MockEmotionHead(nn.Module):
            def __call__(self, encoder_output):
                batch = encoder_output.shape[0]
                # Return logits favoring "surprise" (index 7)
                logits = mx.zeros((batch, 8))
                return logits.at[:, 7].add(2.0)  # Boost surprise

        return MockEmotionHead()

    def test_feature_extraction_with_pitch(self, mock_pitch_head):
        """Test pitch feature extraction."""
        from tools.whisper_mlx.prosody_beam_search import ProsodyBeamSearch

        prosody_search = ProsodyBeamSearch(pitch_head=mock_pitch_head)

        # Create mock encoder output
        encoder_output = mx.zeros((1, 100, 1280))

        features = prosody_search.extract_prosody_features(encoder_output)

        # Check extracted features
        assert features.voicing_ratio > 0.5  # Mostly voiced
        assert features.pitch_slope > 0  # Rising pitch
        assert features.audio_duration == pytest.approx(2.0, rel=0.1)  # 100 frames / 50 Hz

    def test_feature_extraction_with_emotion(self, mock_emotion_head):
        """Test emotion feature extraction."""
        from tools.whisper_mlx.prosody_beam_search import ProsodyBeamSearch

        prosody_search = ProsodyBeamSearch(emotion_head=mock_emotion_head)

        encoder_output = mx.zeros((1, 100, 1280))

        features = prosody_search.extract_prosody_features(encoder_output)

        # Check extracted features
        assert features.dominant_emotion == "surprise"
        assert features.emotion_confidence > 0.5
        assert "surprise" in features.emotion_probs

    def test_feature_extraction_without_heads(self):
        """Test feature extraction with no heads returns defaults."""
        from tools.whisper_mlx.prosody_beam_search import ProsodyBeamSearch

        prosody_search = ProsodyBeamSearch()

        encoder_output = mx.zeros((1, 100, 1280))

        features = prosody_search.extract_prosody_features(encoder_output)

        assert features.pitch_values is None
        assert features.pitch_slope == 0.0
        assert features.dominant_emotion == "neutral"


class TestProsodyBeamSearchDecoder:
    """Tests for prosody-conditioned beam search decoder."""

    def test_decoder_initialization(self):
        """Test decoder can be initialized."""
        from tools.whisper_mlx.prosody_beam_search import (
            ProsodyBeamSearch,
            ProsodyBeamSearchDecoder,
            ProsodyFeatures,
        )

        # Create mock model
        class MockModel(nn.Module):
            pass

        model = MockModel()
        features = ProsodyFeatures()
        scorer = ProsodyBeamSearch()

        decoder = ProsodyBeamSearchDecoder(
            model=model,
            beam_size=5,
            prosody_features=features,
            prosody_scorer=scorer,
        )

        assert decoder.beam_size == 5
        assert decoder.prosody_features is features
        assert decoder.prosody_scorer is scorer


class TestStatistics:
    """Tests for prosody scoring statistics."""

    def test_stats_tracking(self):
        """Test statistics are tracked correctly."""
        from tools.whisper_mlx.prosody_beam_search import (
            ProsodyBeamSearch,
            ProsodyFeatures,
        )

        prosody_search = ProsodyBeamSearch()

        # Initialize punctuation IDs with mock tokenizer
        class MockTokenizer:
            def encode(self, text):
                return {"?": [100], ".": [101], "!": [102]}.get(text, [0])

        prosody_search._initialize_punctuation_ids(MockTokenizer())

        features = ProsodyFeatures(pitch_slope=0.2)

        # Score some tokens
        prosody_search.prosody_score_factor(100, features, MockTokenizer())  # ? with boost
        prosody_search.prosody_score_factor(50, features, MockTokenizer())   # Non-punct
        prosody_search.prosody_score_factor(50, features, MockTokenizer())   # Non-punct

        stats = prosody_search.get_stats()

        assert stats["total_tokens_scored"] == 3
        assert stats["boosts_applied"] == 1
        assert stats["boost_rate"] == pytest.approx(1/3)

    def test_stats_reset(self):
        """Test statistics can be reset."""
        from tools.whisper_mlx.prosody_beam_search import (
            ProsodyBeamSearch,
            ProsodyFeatures,
        )

        prosody_search = ProsodyBeamSearch()

        # Initialize and score
        class MockTokenizer:
            def encode(self, text):
                return {"?": [100]}.get(text, [0])

        prosody_search._initialize_punctuation_ids(MockTokenizer())
        features = ProsodyFeatures(pitch_slope=0.2)
        prosody_search.prosody_score_factor(100, features, MockTokenizer())

        assert prosody_search.total_tokens_scored > 0

        prosody_search.reset_stats()

        assert prosody_search.total_tokens_scored == 0
        assert prosody_search.boosts_applied == 0


class TestFactoryFunction:
    """Tests for create_prosody_beam_search factory function."""

    def test_create_without_checkpoints(self):
        """Test creation without checkpoint files."""
        from tools.whisper_mlx.prosody_beam_search import create_prosody_beam_search

        prosody_search = create_prosody_beam_search(model_size="large-v3")

        assert prosody_search.pitch_head is None
        assert prosody_search.emotion_head is None
        assert prosody_search.config is not None

    def test_create_with_custom_config(self):
        """Test creation with custom config."""
        from tools.whisper_mlx.prosody_beam_search import (
            ProsodyBoostConfig,
            create_prosody_beam_search,
        )

        config = ProsodyBoostConfig(question_rising_pitch_boost=5.0)
        prosody_search = create_prosody_beam_search(config=config)

        assert prosody_search.config.question_rising_pitch_boost == 5.0


class TestStreamingProsodyDecoder:
    """Tests for streaming prosody decoder integration."""

    def test_streaming_decoder_initialization(self):
        """Test streaming decoder can be initialized."""
        from tools.whisper_mlx.prosody_beam_search import StreamingProsodyDecoder

        # Create mock model
        class MockModel(nn.Module):
            pass

        model = MockModel()

        decoder = StreamingProsodyDecoder(model=model)

        assert decoder.model is model
        assert decoder.prosody_search is not None

    def test_streaming_decoder_stats(self):
        """Test streaming decoder exposes prosody stats."""
        from tools.whisper_mlx.prosody_beam_search import StreamingProsodyDecoder

        class MockModel(nn.Module):
            pass

        decoder = StreamingProsodyDecoder(model=MockModel())

        stats = decoder.get_prosody_stats()

        assert "boosts_applied" in stats
        assert "total_tokens_scored" in stats


class TestModuleImport:
    """Tests for module import and __all__ exports."""

    def test_module_import(self):
        """Test module can be imported without errors."""
        from tools.whisper_mlx import prosody_beam_search

        assert hasattr(prosody_beam_search, "ProsodyBeamSearch")
        assert hasattr(prosody_beam_search, "ProsodyFeatures")
        assert hasattr(prosody_beam_search, "ProsodyBoostConfig")

    def test_all_exports(self):
        """Test __all__ contains expected exports."""
        from tools.whisper_mlx.prosody_beam_search import __all__

        expected = [
            "ProsodyBeamSearch",
            "ProsodyBeamSearchDecoder",
            "ProsodyFeatures",
            "ProsodyBoostConfig",
            "StreamingProsodyDecoder",
            "create_prosody_beam_search",
        ]

        for name in expected:
            assert name in __all__


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_emotion_probs(self):
        """Test handling of empty emotion probabilities."""
        from tools.whisper_mlx.prosody_beam_search import (
            ProsodyBeamSearch,
            ProsodyFeatures,
        )

        prosody_search = ProsodyBeamSearch()

        class MockTokenizer:
            def encode(self, text):
                return {"!": [100]}.get(text, [0])

        prosody_search._initialize_punctuation_ids(MockTokenizer())

        # Empty emotion probs
        features = ProsodyFeatures(emotion_probs={})

        factor = prosody_search.prosody_score_factor(100, features, MockTokenizer())

        # Should not crash, should return 1.0 (no boost)
        assert factor == 1.0

    def test_none_emotion_probs(self):
        """Test handling of None emotion probabilities."""
        from tools.whisper_mlx.prosody_beam_search import (
            ProsodyBeamSearch,
            ProsodyFeatures,
        )

        prosody_search = ProsodyBeamSearch()

        class MockTokenizer:
            def encode(self, text):
                return {"!": [100]}.get(text, [0])

        prosody_search._initialize_punctuation_ids(MockTokenizer())

        # Create features but reset emotion_probs to None
        features = ProsodyFeatures()
        features.emotion_probs = None

        factor = prosody_search.prosody_score_factor(100, features, MockTokenizer())

        # Should not crash, should return 1.0
        assert factor == 1.0

    def test_very_short_audio(self):
        """Test feature extraction with very short audio."""
        from tools.whisper_mlx.prosody_beam_search import ProsodyBeamSearch

        class MockPitchHead(nn.Module):
            def __call__(self, encoder_output):
                batch, T, _ = encoder_output.shape
                return mx.zeros((batch, T)), mx.zeros((batch, T))

        prosody_search = ProsodyBeamSearch(pitch_head=MockPitchHead())

        # Very short audio (5 frames = 0.1 seconds)
        encoder_output = mx.zeros((1, 5, 1280))

        features = prosody_search.extract_prosody_features(encoder_output)

        # Should not crash
        assert features.audio_duration == pytest.approx(0.1, rel=0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
