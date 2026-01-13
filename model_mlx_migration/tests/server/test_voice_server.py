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

"""Tests for VoiceServer."""

import numpy as np
import pytest

from src.server.rich_token import RichToken, create_partial_token
from src.server.voice_server import (
    ASRMode,
    ASRPipeline,
    ClientSession,
    MockASRPipeline,
    ServerConfig,
    ServerState,
    SessionMetrics,
    SessionState,
    VoiceServer,
)

# Module-level RNG for reproducible tests
_rng = np.random.default_rng(42)


class TestServerConfig:
    """Tests for ServerConfig dataclass."""

    def test_default_config(self):
        config = ServerConfig()
        assert config.host == "0.0.0.0"
        assert config.port == 8765
        assert config.sample_rate == 16000
        assert config.chunk_size_ms == 320
        assert config.default_mode == ASRMode.STREAMING

    def test_custom_config(self):
        config = ServerConfig(
            host="localhost",
            port=9000,
            max_connections=5,
            default_mode=ASRMode.HIGH_ACCURACY,
        )
        assert config.host == "localhost"
        assert config.port == 9000
        assert config.max_connections == 5
        assert config.default_mode == ASRMode.HIGH_ACCURACY

    def test_feature_flags(self):
        config = ServerConfig(
            enable_emotion=False,
            enable_pitch=True,
            enable_speaker=False,
        )
        assert not config.enable_emotion
        assert config.enable_pitch
        assert not config.enable_speaker


class TestSessionMetrics:
    """Tests for SessionMetrics dataclass."""

    def test_default_metrics(self):
        metrics = SessionMetrics()
        assert metrics.start_time == 0.0
        assert metrics.audio_received_ms == 0.0
        assert metrics.chunks_received == 0
        assert metrics.tokens_sent == 0

    def test_metrics_update(self):
        metrics = SessionMetrics()
        metrics.audio_received_ms += 320.0
        metrics.chunks_received += 1
        assert metrics.audio_received_ms == 320.0
        assert metrics.chunks_received == 1


class TestClientSession:
    """Tests for ClientSession dataclass."""

    def test_basic_session(self):
        session = ClientSession(
            session_id="test123",
            websocket=None,
        )
        assert session.session_id == "test123"
        assert session.state == SessionState.CONNECTED
        assert session.mode == ASRMode.STREAMING
        assert session.utterance_id == ""
        assert session.chunk_index == 0

    def test_session_state_transition(self):
        session = ClientSession(session_id="test", websocket=None)
        assert session.state == SessionState.CONNECTED

        session.state = SessionState.STREAMING
        assert session.state == SessionState.STREAMING

        session.state = SessionState.PROCESSING
        assert session.state == SessionState.PROCESSING

    def test_audio_buffer(self):
        session = ClientSession(session_id="test", websocket=None)
        assert session.audio_buffer == []

        # Add audio chunks
        chunk1 = np.zeros(1600, dtype=np.float32)
        chunk2 = np.ones(1600, dtype=np.float32)
        session.audio_buffer.append(chunk1)
        session.audio_buffer.append(chunk2)

        assert len(session.audio_buffer) == 2


class TestMockASRPipeline:
    """Tests for MockASRPipeline."""

    @pytest.fixture
    def pipeline(self):
        return MockASRPipeline()

    @pytest.fixture
    def session(self):
        session = ClientSession(session_id="test", websocket=None)
        session.utterance_id = "utt123"
        session.audio_start_ms = 0.0
        return session

    @pytest.fixture
    def config(self):
        return ServerConfig()

    @pytest.mark.asyncio
    async def test_process_chunk_returns_token(self, pipeline, session, config):
        # Create audio that's long enough to produce output
        audio = _rng.standard_normal(16000).astype(np.float32)  # 1 second

        result = await pipeline.process_chunk(audio, session, ASRMode.STREAMING, config)

        assert result is not None
        assert isinstance(result, RichToken)
        assert result.is_partial
        assert not result.is_final

    @pytest.mark.asyncio
    async def test_process_short_chunk_returns_none(self, pipeline, session, config):
        # Create audio that's too short
        audio = _rng.standard_normal(1600).astype(np.float32)  # 100ms

        result = await pipeline.process_chunk(audio, session, ASRMode.STREAMING, config)

        # Short chunks may return None
        # The mock returns None for <200ms audio
        assert result is None

    @pytest.mark.asyncio
    async def test_finalize_utterance(self, pipeline, session, config):
        session.metrics.audio_received_ms = 2000.0

        result = await pipeline.finalize_utterance(session, ASRMode.STREAMING, config)

        assert result is not None
        assert result.is_final
        assert not result.is_partial
        assert len(result.text) > 0

    @pytest.mark.asyncio
    async def test_finalize_includes_features(self, pipeline, session, config):
        session.metrics.audio_received_ms = 2000.0

        result = await pipeline.finalize_utterance(session, ASRMode.STREAMING, config)

        # Check rich features are included
        assert result.emotion is not None
        assert result.pitch is not None
        assert result.phonemes is not None
        assert result.language is not None
        assert result.hallucination is not None
        assert result.speaker is not None

    @pytest.mark.asyncio
    async def test_finalize_includes_word_timestamps(self, pipeline, session, config):
        session.metrics.audio_received_ms = 2000.0

        result = await pipeline.finalize_utterance(session, ASRMode.STREAMING, config)

        assert len(result.word_timestamps) > 0
        for wt in result.word_timestamps:
            assert wt.word
            assert wt.end_ms > wt.start_ms

    @pytest.mark.asyncio
    async def test_detect_language(self, pipeline):
        audio = _rng.standard_normal(16000).astype(np.float32)

        lang, confidence = await pipeline.detect_language(audio)

        assert lang == "en"  # Default
        assert confidence > 0.0

    @pytest.mark.asyncio
    async def test_detect_speakers(self, pipeline):
        audio = _rng.standard_normal(16000).astype(np.float32)

        num_speakers = await pipeline.detect_speakers(audio)

        assert num_speakers == 1  # Default


class TestVoiceServer:
    """Tests for VoiceServer (non-WebSocket functionality)."""

    def test_default_creation(self):
        server = VoiceServer()
        assert server.state == ServerState.STOPPED
        assert server.num_connections == 0
        assert isinstance(server.config, ServerConfig)
        assert isinstance(server.pipeline, MockASRPipeline)

    def test_custom_config(self):
        config = ServerConfig(port=9000)
        server = VoiceServer(config=config)
        assert server.config.port == 9000

    def test_custom_pipeline(self):
        pipeline = MockASRPipeline()
        server = VoiceServer(pipeline=pipeline)
        assert server.pipeline is pipeline

    def test_callbacks(self):
        server = VoiceServer()

        connected = []
        disconnected = []

        server.on_connect(lambda s: connected.append(s))
        server.on_disconnect(lambda s: disconnected.append(s))

        assert server._on_connect is not None
        assert server._on_disconnect is not None


class TestServerState:
    """Tests for ServerState enum."""

    def test_all_states(self):
        assert ServerState.STOPPED.value == "stopped"
        assert ServerState.STARTING.value == "starting"
        assert ServerState.RUNNING.value == "running"
        assert ServerState.STOPPING.value == "stopping"


class TestSessionState:
    """Tests for SessionState enum."""

    def test_all_states(self):
        assert SessionState.CONNECTED.value == "connected"
        assert SessionState.STREAMING.value == "streaming"
        assert SessionState.PROCESSING.value == "processing"
        assert SessionState.DISCONNECTED.value == "disconnected"


class TestASRModeConfig:
    """Tests for ASR mode configuration."""

    def test_streaming_mode_default(self):
        config = ServerConfig()
        assert config.default_mode == ASRMode.STREAMING

    def test_high_accuracy_mode(self):
        config = ServerConfig(
            default_mode=ASRMode.HIGH_ACCURACY,
            enable_high_accuracy=True,
        )
        assert config.default_mode == ASRMode.HIGH_ACCURACY
        assert config.enable_high_accuracy

    def test_disable_high_accuracy(self):
        config = ServerConfig(enable_high_accuracy=False)
        assert not config.enable_high_accuracy


class TestCustomASRPipeline:
    """Tests for custom ASRPipeline implementations."""

    @pytest.mark.asyncio
    async def test_custom_pipeline_interface(self):
        """Test that custom pipelines can be created."""

        class CustomPipeline(ASRPipeline):
            async def process_chunk(self, audio, session, mode, config):
                return create_partial_token(
                    text="custom",
                    start_ms=0.0,
                    end_ms=100.0,
                )

            async def finalize_utterance(self, session, mode, config):
                return RichToken(
                    text="custom result",
                    confidence=0.99,
                    start_time_ms=0.0,
                    end_time_ms=1000.0,
                    is_final=True,
                )

        pipeline = CustomPipeline()
        session = ClientSession(session_id="test", websocket=None)
        config = ServerConfig()

        # Test process_chunk
        audio = np.zeros(1600, dtype=np.float32)
        result = await pipeline.process_chunk(audio, session, ASRMode.STREAMING, config)
        assert result.text == "custom"

        # Test finalize_utterance
        result = await pipeline.finalize_utterance(session, ASRMode.STREAMING, config)
        assert result.text == "custom result"
        assert result.is_final


class TestFeatureConfiguration:
    """Tests for feature enable/disable configuration."""

    @pytest.mark.asyncio
    async def test_disabled_features(self):
        """Test that disabled features are not included."""
        config = ServerConfig(
            enable_emotion=False,
            enable_pitch=False,
            enable_phoneme=False,
            enable_paralinguistics=False,
            enable_language=False,
            enable_singing=False,
            enable_timestamps=False,
            enable_hallucination=False,
            enable_speaker=False,
        )

        pipeline = MockASRPipeline()
        session = ClientSession(session_id="test", websocket=None)
        session.metrics.audio_received_ms = 2000.0

        result = await pipeline.finalize_utterance(session, ASRMode.STREAMING, config)

        # With all features disabled, these should be None
        assert result.emotion is None
        assert result.pitch is None


class TestServerIntegration:
    """Integration tests for server components."""

    def test_server_with_all_components(self):
        """Test server with all components configured."""
        config = ServerConfig(
            host="localhost",
            port=9999,
            sample_rate=16000,
            chunk_size_ms=320,
            default_mode=ASRMode.STREAMING,
            enable_high_accuracy=True,
            enable_emotion=True,
            enable_pitch=True,
            enable_phoneme=True,
            enable_paralinguistics=True,
            enable_language=True,
            enable_singing=True,
            enable_timestamps=True,
            enable_hallucination=True,
            enable_speaker=True,
            enable_multi_speaker=True,
            max_speakers=4,
        )

        pipeline = MockASRPipeline()
        server = VoiceServer(config=config, pipeline=pipeline)

        assert server.state == ServerState.STOPPED
        assert server.config.enable_high_accuracy
        assert server.config.enable_multi_speaker
