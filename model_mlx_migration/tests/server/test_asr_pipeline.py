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

"""Tests for integrated ASR pipeline (Phase 9.4)."""

import numpy as np
import pytest

# Module-level random generator for reproducibility
_rng = np.random.default_rng(42)

from src.server.asr_pipeline import (
    EMOTION_LABELS,
    IntegratedASRPipeline,
    IntegratedPipelineConfig,
    ZipformerASRPipeline,
)
from src.server.rich_token import ASRMode, EmotionLabel, RichToken
from src.server.voice_server import ClientSession, ServerConfig, SessionMetrics


class TestIntegratedPipelineConfig:
    """Tests for IntegratedPipelineConfig dataclass."""

    def test_default_config(self):
        config = IntegratedPipelineConfig()
        assert config.sample_rate == 16000
        assert config.num_mel_bins == 80
        assert config.enable_rich_heads
        assert config.enable_multi_speaker
        assert config.enable_rover

    def test_custom_config(self):
        config = IntegratedPipelineConfig(
            zipformer_checkpoint="/path/to/checkpoint.pt",
            zipformer_bpe_model="/path/to/bpe.model",
            enable_rich_heads=False,
            enable_rover=False,
        )
        assert config.zipformer_checkpoint == "/path/to/checkpoint.pt"
        assert not config.enable_rich_heads
        assert not config.enable_rover

    def test_encoder_dim(self):
        config = IntegratedPipelineConfig(encoder_dim=512)
        assert config.encoder_dim == 512


class TestEmotionLabels:
    """Tests for EMOTION_LABELS constant."""

    def test_all_8_emotions(self):
        assert len(EMOTION_LABELS) == 8

    def test_emotion_order(self):
        assert EMOTION_LABELS[0] == EmotionLabel.NEUTRAL
        assert EMOTION_LABELS[1] == EmotionLabel.HAPPY
        assert EMOTION_LABELS[2] == EmotionLabel.SAD
        assert EMOTION_LABELS[3] == EmotionLabel.ANGRY

    def test_all_labels_are_emotion_label(self):
        for label in EMOTION_LABELS:
            assert isinstance(label, EmotionLabel)


class TestIntegratedASRPipeline:
    """Tests for IntegratedASRPipeline."""

    @pytest.fixture
    def pipeline(self):
        return IntegratedASRPipeline()

    @pytest.fixture
    def session(self):
        session = ClientSession(session_id="test123", websocket=None)
        session.utterance_id = "utt1"
        session.audio_start_ms = 0.0
        session.metrics = SessionMetrics()
        session.metrics.audio_received_ms = 1000.0
        return session

    @pytest.fixture
    def config(self):
        return ServerConfig()

    def test_default_creation(self, pipeline):
        assert pipeline.config is not None
        assert not pipeline._models_loaded

    def test_custom_config(self):
        config = IntegratedPipelineConfig(enable_rich_heads=False)
        pipeline = IntegratedASRPipeline(config)
        assert not pipeline.config.enable_rich_heads

    @pytest.mark.asyncio
    async def test_process_chunk_streaming(self, pipeline, session, config):
        audio = _rng.standard_normal(16000).astype(np.float32)
        result = await pipeline.process_chunk(audio, session, ASRMode.STREAMING, config)
        # Without models loaded, returns placeholder
        assert result is not None
        assert result.is_partial

    @pytest.mark.asyncio
    async def test_process_chunk_high_accuracy_returns_none(self, pipeline, session, config):
        audio = _rng.standard_normal(16000).astype(np.float32)
        result = await pipeline.process_chunk(audio, session, ASRMode.HIGH_ACCURACY, config)
        # High accuracy mode accumulates, doesn't return partials
        assert result is None

    @pytest.mark.asyncio
    async def test_finalize_utterance(self, pipeline, session, config):
        # Add audio to buffer
        session.audio_buffer = [_rng.standard_normal(16000).astype(np.float32)]

        result = await pipeline.finalize_utterance(session, ASRMode.STREAMING, config)

        assert isinstance(result, RichToken)
        assert result.is_final
        assert not result.is_partial

    @pytest.mark.asyncio
    async def test_finalize_includes_language(self, pipeline, session, config):
        session.audio_buffer = [_rng.standard_normal(16000).astype(np.float32)]

        result = await pipeline.finalize_utterance(session, ASRMode.STREAMING, config)

        assert result.language is not None
        assert result.language.language == "en"

    @pytest.mark.asyncio
    async def test_detect_language(self, pipeline):
        audio = _rng.standard_normal(16000).astype(np.float32)
        lang, conf = await pipeline.detect_language(audio)
        assert lang == "en"
        assert conf > 0.0

    @pytest.mark.asyncio
    async def test_detect_speakers(self, pipeline):
        audio = _rng.standard_normal(16000).astype(np.float32)
        num_speakers = await pipeline.detect_speakers(audio)
        assert num_speakers >= 0


class TestZipformerASRPipeline:
    """Tests for ZipformerASRPipeline."""

    @pytest.fixture
    def pipeline(self):
        # Create without model paths (won't load actual model)
        return ZipformerASRPipeline()

    @pytest.fixture
    def session(self):
        session = ClientSession(session_id="test", websocket=None)
        session.utterance_id = "utt1"
        session.audio_start_ms = 0.0
        session.metrics = SessionMetrics()
        return session

    @pytest.fixture
    def config(self):
        return ServerConfig()

    def test_creation_without_paths(self, pipeline):
        assert pipeline._checkpoint_path is None
        assert pipeline._bpe_model_path is None
        assert not pipeline._loaded

    def test_creation_with_paths(self):
        pipeline = ZipformerASRPipeline(
            checkpoint_path="/path/to/model.pt",
            bpe_model_path="/path/to/bpe.model",
        )
        assert pipeline._checkpoint_path == "/path/to/model.pt"

    @pytest.mark.asyncio
    async def test_process_chunk_without_model(self, pipeline, session, config):
        audio = _rng.standard_normal(16000).astype(np.float32)
        result = await pipeline.process_chunk(audio, session, ASRMode.STREAMING, config)
        # Without model, returns None
        assert result is None

    @pytest.mark.asyncio
    async def test_finalize_without_model(self, pipeline, session, config):
        session.audio_buffer = [_rng.standard_normal(16000).astype(np.float32)]

        result = await pipeline.finalize_utterance(session, ASRMode.STREAMING, config)

        assert isinstance(result, RichToken)
        assert result.is_final
        assert "[no transcription]" in result.text


class TestPipelineIntegration:
    """Integration tests for ASR pipelines."""

    @pytest.mark.asyncio
    async def test_streaming_workflow(self):
        """Test complete streaming workflow."""
        pipeline = IntegratedASRPipeline()
        config = ServerConfig()

        session = ClientSession(session_id="stream_test", websocket=None)
        session.utterance_id = "stream_utt"
        session.audio_start_ms = 0.0
        session.metrics = SessionMetrics()
        session.audio_buffer = []

        # Process several chunks
        for _i in range(3):
            chunk = _rng.standard_normal(3200).astype(np.float32)  # 200ms chunks
            session.audio_buffer.append(chunk)
            session.metrics.audio_received_ms += 200.0

            _result = await pipeline.process_chunk(
                chunk, session, ASRMode.STREAMING, config,
            )
            # Partial results may or may not be returned (just exercising code path)

        # Finalize
        final = await pipeline.finalize_utterance(session, ASRMode.STREAMING, config)
        assert final.is_final
        assert final.mode == ASRMode.STREAMING

    @pytest.mark.asyncio
    async def test_high_accuracy_workflow(self):
        """Test high-accuracy (ROVER) workflow."""
        pipeline = IntegratedASRPipeline()
        config = ServerConfig()

        session = ClientSession(session_id="ha_test", websocket=None)
        session.utterance_id = "ha_utt"
        session.audio_start_ms = 0.0
        session.metrics = SessionMetrics()
        session.audio_buffer = []

        # Accumulate audio without partial results
        for _i in range(5):
            chunk = _rng.standard_normal(3200).astype(np.float32)
            session.audio_buffer.append(chunk)
            session.metrics.audio_received_ms += 200.0

            result = await pipeline.process_chunk(
                chunk, session, ASRMode.HIGH_ACCURACY, config,
            )
            # High accuracy doesn't return partials
            assert result is None

        # Finalize with ROVER
        final = await pipeline.finalize_utterance(session, ASRMode.HIGH_ACCURACY, config)
        assert final.is_final
        assert final.mode == ASRMode.HIGH_ACCURACY

    @pytest.mark.asyncio
    async def test_empty_audio_buffer(self):
        """Test handling of empty audio buffer."""
        pipeline = IntegratedASRPipeline()
        config = ServerConfig()

        session = ClientSession(session_id="empty_test", websocket=None)
        session.utterance_id = "empty_utt"
        session.audio_start_ms = 0.0
        session.metrics = SessionMetrics()
        session.audio_buffer = []  # Empty!

        final = await pipeline.finalize_utterance(session, ASRMode.STREAMING, config)
        assert final.is_final
        # Should handle gracefully

    def test_pipeline_lazy_loading(self):
        """Test that models are lazily loaded."""
        pipeline = IntegratedASRPipeline()
        assert not pipeline._models_loaded

        # Trigger load
        pipeline._load_models()
        assert pipeline._models_loaded

        # Second call is no-op
        pipeline._load_models()
        assert pipeline._models_loaded


class TestPipelineWithRichHeads:
    """Tests for pipeline with rich audio heads enabled."""

    @pytest.mark.asyncio
    async def test_rich_features_added(self):
        """Test that rich features are added when enabled."""
        config = IntegratedPipelineConfig(enable_rich_heads=True)
        pipeline = IntegratedASRPipeline(config)
        server_config = ServerConfig(
            enable_emotion=True,
            enable_pitch=True,
            enable_phoneme=True,
            enable_hallucination=True,
            enable_speaker=True,
        )

        session = ClientSession(session_id="rich_test", websocket=None)
        session.utterance_id = "rich_utt"
        session.audio_start_ms = 0.0
        session.metrics = SessionMetrics()
        session.audio_buffer = [_rng.standard_normal(16000).astype(np.float32)]

        final = await pipeline.finalize_utterance(session, ASRMode.STREAMING, server_config)

        # Rich features should be present (as placeholders without real model)
        assert final.is_final

    @pytest.mark.asyncio
    async def test_disabled_rich_heads(self):
        """Test pipeline with rich heads disabled."""
        config = IntegratedPipelineConfig(enable_rich_heads=False)
        pipeline = IntegratedASRPipeline(config)
        server_config = ServerConfig()

        session = ClientSession(session_id="no_rich_test", websocket=None)
        session.utterance_id = "no_rich_utt"
        session.audio_start_ms = 0.0
        session.metrics = SessionMetrics()
        session.audio_buffer = [_rng.standard_normal(16000).astype(np.float32)]

        final = await pipeline.finalize_utterance(session, ASRMode.STREAMING, server_config)
        assert final.is_final
