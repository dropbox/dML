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
Tests for PhonemeEnhancedAdaptationEngine.

Tests cover:
1. Dataclass initialization
2. Adaptation tier transitions
3. Quality scoring
4. Speaker tracking and state management
5. Training data collection
6. Adapter registration
"""

# ruff: noqa: SLF001

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import pytest

from tools.whisper_mlx.sota.phoneme_adaptation import (
    AdaptationSample,
    AdaptationTier,
    PhonemeAdaptationConfig,
    PhonemeEnhancedAdaptationEngine,
    SpeakerAdaptationState,
    create_adaptation_engine,
)


class MockPhonemeHead(nn.Module):
    """Mock phoneme head for testing."""

    def __init__(self, vocab_size: int = 178):
        super().__init__()
        self.proj = nn.Linear(1280, vocab_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.proj(x)


@dataclass
class MockDecoderOutput:
    """Mock decoder output for testing."""

    text: str
    confidence: float


@dataclass
class MockCTCOutput:
    """Mock CTC output for testing."""

    phonemes: list[str]


class TestAdaptationTier:
    """Tests for AdaptationTier enum."""

    def test_tier_ordering(self):
        """Tiers should be orderable by value."""
        assert AdaptationTier.TIER_0_SUTA.value < AdaptationTier.TIER_1_RECOGNIZED.value
        assert AdaptationTier.TIER_1_RECOGNIZED.value < AdaptationTier.TIER_2_VOCAB.value
        assert AdaptationTier.TIER_2_VOCAB.value < AdaptationTier.TIER_3_FULL.value

    def test_tier_names(self):
        """Tiers should have descriptive names."""
        assert "SUTA" in AdaptationTier.TIER_0_SUTA.name
        assert "RECOGNIZED" in AdaptationTier.TIER_1_RECOGNIZED.name
        assert "VOCAB" in AdaptationTier.TIER_2_VOCAB.name
        assert "FULL" in AdaptationTier.TIER_3_FULL.name


class TestAdaptationSample:
    """Tests for AdaptationSample dataclass."""

    def test_sample_creation(self):
        """Test creating an adaptation sample."""
        sample = AdaptationSample(
            audio=mx.zeros((16000,)),
            transcript="Hello world",
            speaker_id=1,
            speaker_embedding=mx.random.normal((192,)),
            quality_score=0.8,
            phoneme_score=0.75,
            confidence=0.9,
            timestamp=1704067200.0,
        )

        assert sample.transcript == "Hello world"
        assert sample.speaker_id == 1
        assert sample.quality_score == 0.8
        assert sample.audio.shape == (16000,)


class TestPhonemeAdaptationConfig:
    """Tests for configuration dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PhonemeAdaptationConfig()

        assert config.quality_threshold == 0.7
        assert config.min_samples_for_lora == 100
        assert config.max_samples_per_speaker == 1000
        # Use approximate comparison for floating point
        weights_sum = config.phoneme_weight + config.confidence_weight + config.consistency_weight
        assert abs(weights_sum - 1.0) < 1e-6

    def test_custom_config(self):
        """Test custom configuration."""
        config = PhonemeAdaptationConfig(
            quality_threshold=0.8,
            min_samples_for_lora=50,
        )

        assert config.quality_threshold == 0.8
        assert config.min_samples_for_lora == 50


class TestSpeakerAdaptationState:
    """Tests for speaker state management."""

    def test_default_state(self):
        """Test default speaker state."""
        state = SpeakerAdaptationState(speaker_id=1)

        assert state.speaker_id == 1
        assert len(state.samples) == 0
        assert not state.vocab_trained
        assert not state.lora_trained
        assert state.adaptation_tier == AdaptationTier.TIER_0_SUTA

    def test_state_with_samples(self):
        """Test state with samples."""
        state = SpeakerAdaptationState(
            speaker_id=1,
            samples=[
                AdaptationSample(
                    audio=mx.zeros((16000,)),
                    transcript="test",
                    speaker_id=1,
                    speaker_embedding=mx.zeros((192,)),
                    quality_score=0.8,
                    phoneme_score=0.7,
                    confidence=0.9,
                    timestamp=0.0,
                ),
            ],
        )

        assert len(state.samples) == 1


class TestPhonemeEnhancedAdaptationEngine:
    """Tests for the main adaptation engine."""

    def test_engine_creation(self):
        """Test basic engine creation."""
        engine = PhonemeEnhancedAdaptationEngine()

        assert engine.phoneme_head is None
        assert engine.config.quality_threshold == 0.7
        assert engine._total_accepted == 0
        assert engine._total_rejected == 0

    def test_engine_with_phoneme_head(self):
        """Test engine with phoneme head."""
        head = MockPhonemeHead()
        engine = PhonemeEnhancedAdaptationEngine(phoneme_head=head)

        assert engine.phoneme_head is head

    def test_engine_with_custom_config(self):
        """Test engine with custom config."""
        config = PhonemeAdaptationConfig(quality_threshold=0.9)
        engine = PhonemeEnhancedAdaptationEngine(config=config)

        assert engine.config.quality_threshold == 0.9

    def test_process_utterance_no_embedding(self):
        """Test processing without speaker embedding."""
        engine = PhonemeEnhancedAdaptationEngine()

        audio = mx.zeros((16000,))
        encoder_out = mx.random.normal((1, 100, 1280))
        decoder_out = MockDecoderOutput(text="Hello", confidence=0.9)

        decision = engine.process_utterance(
            audio=audio,
            encoder_output=encoder_out,
            decoder_output=decoder_out,
        )

        assert not decision.accept
        assert "No speaker embedding" in decision.reason

    def test_process_utterance_accepts_quality(self):
        """Test processing accepts high-quality utterance."""
        engine = PhonemeEnhancedAdaptationEngine(quality_threshold=0.5)

        # Use audio long enough (1 second at 16kHz)
        audio = mx.zeros((16000,))
        encoder_out = mx.random.normal((1, 100, 1280))
        decoder_out = MockDecoderOutput(text="Hello world", confidence=0.9)
        speaker_emb = mx.random.normal((192,))

        decision = engine.process_utterance(
            audio=audio,
            encoder_output=encoder_out,
            decoder_output=decoder_out,
            speaker_embedding=speaker_emb,
        )

        assert decision.accept
        assert decision.quality_score > 0
        assert decision.is_new_speaker

    def test_process_utterance_rejects_short_audio(self):
        """Test processing rejects too-short audio."""
        config = PhonemeAdaptationConfig(min_utterance_length=1.0)
        engine = PhonemeEnhancedAdaptationEngine(config=config)

        # Audio too short (0.1 second)
        audio = mx.zeros((1600,))
        encoder_out = mx.random.normal((1, 10, 1280))
        decoder_out = MockDecoderOutput(text="Hi", confidence=0.9)
        speaker_emb = mx.random.normal((192,))

        decision = engine.process_utterance(
            audio=audio,
            encoder_output=encoder_out,
            decoder_output=decoder_out,
            speaker_embedding=speaker_emb,
        )

        assert not decision.accept
        assert "too short" in decision.reason.lower()

    def test_speaker_tier_progression(self):
        """Test tier progression with samples."""
        config = PhonemeAdaptationConfig(
            min_samples_for_vocab=5,
            min_samples_for_lora=10,
            quality_threshold=0.5,
        )
        engine = PhonemeEnhancedAdaptationEngine(config=config)

        speaker_emb = mx.random.normal((192,))
        speaker_emb = speaker_emb / mx.linalg.norm(speaker_emb)

        # Process multiple utterances
        speaker_id = None
        for i in range(12):
            audio = mx.zeros((16000,))
            encoder_out = mx.random.normal((1, 100, 1280))
            decoder_out = MockDecoderOutput(text=f"Test {i}", confidence=0.9)

            decision = engine.process_utterance(
                audio=audio,
                encoder_output=encoder_out,
                decoder_output=decoder_out,
                speaker_embedding=speaker_emb,
            )

            if speaker_id is None:
                speaker_id = decision.speaker_id

        # Check tier after 12 samples
        tier = engine.get_speaker_tier(speaker_id)
        assert tier == AdaptationTier.TIER_2_VOCAB

    def test_has_sufficient_data(self):
        """Test checking for sufficient training data."""
        config = PhonemeAdaptationConfig(
            min_samples_for_lora=5,
            quality_threshold=0.5,
        )
        engine = PhonemeEnhancedAdaptationEngine(config=config)

        speaker_emb = mx.random.normal((192,))
        speaker_emb = speaker_emb / mx.linalg.norm(speaker_emb)

        # Initially no data
        assert not engine.has_sufficient_data(0)

        # Add samples
        speaker_id = None
        for i in range(6):
            audio = mx.zeros((16000,))
            encoder_out = mx.random.normal((1, 100, 1280))
            decoder_out = MockDecoderOutput(text=f"Test {i}", confidence=0.9)

            decision = engine.process_utterance(
                audio=audio,
                encoder_output=encoder_out,
                decoder_output=decoder_out,
                speaker_embedding=speaker_emb,
            )
            if speaker_id is None:
                speaker_id = decision.speaker_id

        # Now should have sufficient data
        assert engine.has_sufficient_data(speaker_id)

    def test_get_training_data(self):
        """Test retrieving training data."""
        engine = PhonemeEnhancedAdaptationEngine(quality_threshold=0.5)

        speaker_emb = mx.random.normal((192,))
        audio = mx.zeros((16000,))
        encoder_out = mx.random.normal((1, 100, 1280))
        decoder_out = MockDecoderOutput(text="Hello", confidence=0.9)

        decision = engine.process_utterance(
            audio=audio,
            encoder_output=encoder_out,
            decoder_output=decoder_out,
            speaker_embedding=speaker_emb,
        )

        data = engine.get_training_data(decision.speaker_id)
        assert len(data) == 1
        assert data[0].transcript == "Hello"

    def test_get_custom_vocabulary(self):
        """Test vocabulary collection."""
        engine = PhonemeEnhancedAdaptationEngine(quality_threshold=0.5)

        speaker_emb = mx.random.normal((192,))
        audio = mx.zeros((16000,))
        encoder_out = mx.random.normal((1, 100, 1280))
        decoder_out = MockDecoderOutput(text="Hello world", confidence=0.9)

        decision = engine.process_utterance(
            audio=audio,
            encoder_output=encoder_out,
            decoder_output=decoder_out,
            speaker_embedding=speaker_emb,
        )

        vocab = engine.get_custom_vocabulary(decision.speaker_id)
        assert "Hello" in vocab
        assert "world" in vocab

    def test_register_adapter(self):
        """Test registering a trained adapter."""
        engine = PhonemeEnhancedAdaptationEngine()

        # Create mock adapter weights
        adapter_weights = {"lora.weight": mx.zeros((10, 10))}

        # Register adapter
        engine.register_adapter(speaker_id=1, adapter_weights=adapter_weights)

        # Check it's registered
        adapter = engine.get_adapter(speaker_id=1)
        assert adapter is not None
        assert "lora.weight" in adapter

    def test_get_stats(self):
        """Test statistics collection."""
        engine = PhonemeEnhancedAdaptationEngine(quality_threshold=0.5)

        speaker_emb = mx.random.normal((192,))
        audio = mx.zeros((16000,))
        encoder_out = mx.random.normal((1, 100, 1280))
        decoder_out = MockDecoderOutput(text="Test", confidence=0.9)

        # Process an utterance
        engine.process_utterance(
            audio=audio,
            encoder_output=encoder_out,
            decoder_output=decoder_out,
            speaker_embedding=speaker_emb,
        )

        stats = engine.get_stats()
        assert stats["total_speakers"] >= 1
        assert stats["total_accepted"] >= 0
        assert stats["total_rejected"] >= 0
        assert "acceptance_rate" in stats
        assert "tier_distribution" in stats

    def test_clear_speaker(self):
        """Test clearing speaker data."""
        engine = PhonemeEnhancedAdaptationEngine(quality_threshold=0.5)

        speaker_emb = mx.random.normal((192,))
        audio = mx.zeros((16000,))
        encoder_out = mx.random.normal((1, 100, 1280))
        decoder_out = MockDecoderOutput(text="Test", confidence=0.9)

        decision = engine.process_utterance(
            audio=audio,
            encoder_output=encoder_out,
            decoder_output=decoder_out,
            speaker_embedding=speaker_emb,
        )

        speaker_id = decision.speaker_id
        assert len(engine.get_training_data(speaker_id)) > 0

        # Clear speaker
        engine.clear_speaker(speaker_id)
        assert len(engine.get_training_data(speaker_id)) == 0

    def test_clear_all(self):
        """Test clearing all data."""
        engine = PhonemeEnhancedAdaptationEngine(quality_threshold=0.5)

        speaker_emb = mx.random.normal((192,))
        audio = mx.zeros((16000,))
        encoder_out = mx.random.normal((1, 100, 1280))
        decoder_out = MockDecoderOutput(text="Test", confidence=0.9)

        engine.process_utterance(
            audio=audio,
            encoder_output=encoder_out,
            decoder_output=decoder_out,
            speaker_embedding=speaker_emb,
        )

        assert engine._total_accepted > 0

        engine.clear_all()

        assert engine._total_accepted == 0
        assert engine._total_rejected == 0
        assert len(engine.speaker_states) == 0


class TestCreateAdaptationEngine:
    """Tests for factory function."""

    def test_create_default(self):
        """Test creating default engine."""
        engine = create_adaptation_engine()

        assert engine.phoneme_head is None
        assert engine.config.quality_threshold == 0.7

    def test_create_with_phoneme_head(self):
        """Test creating engine with phoneme head."""
        head = MockPhonemeHead()
        engine = create_adaptation_engine(phoneme_head=head)

        assert engine.phoneme_head is head

    def test_create_with_custom_threshold(self):
        """Test creating engine with custom threshold."""
        engine = create_adaptation_engine(quality_threshold=0.9)

        assert engine.config.quality_threshold == 0.9

    def test_create_with_custom_min_samples(self):
        """Test creating engine with custom min samples."""
        engine = create_adaptation_engine(min_samples_for_lora=50)

        assert engine.config.min_samples_for_lora == 50


class TestPhonemeScoring:
    """Tests for phoneme-based quality scoring."""

    def test_scoring_without_phoneme_head(self):
        """Test scoring falls back gracefully without phoneme head."""
        engine = PhonemeEnhancedAdaptationEngine()

        encoder_out = mx.random.normal((1, 100, 1280))
        score = engine._compute_phoneme_score(
            encoder_output=encoder_out,
            transcript="Hello world",
        )

        # Should return heuristic value
        assert 0.0 <= score <= 1.0

    def test_scoring_with_phoneme_head(self):
        """Test scoring with phoneme head."""
        head = MockPhonemeHead()
        engine = PhonemeEnhancedAdaptationEngine(phoneme_head=head)

        encoder_out = mx.random.normal((1, 100, 1280))
        score = engine._compute_phoneme_score(
            encoder_output=encoder_out,
            transcript="Hello world",
        )

        assert 0.0 <= score <= 1.0

    def test_scoring_empty_transcript(self):
        """Test scoring with empty transcript."""
        engine = PhonemeEnhancedAdaptationEngine()

        encoder_out = mx.random.normal((1, 100, 1280))
        score = engine._compute_phoneme_score(
            encoder_output=encoder_out,
            transcript="",
        )

        # Empty transcript should get lower score
        assert score < 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
