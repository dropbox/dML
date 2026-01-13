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
Tests for DashVoice Prometheus metrics.

Tests:
- Metrics module availability
- Metric tracking context managers
- Metric recording functions
- FastAPI integration (metrics endpoint)
"""

import time

import pytest


class TestMetricsModule:
    """Test metrics module availability and basic functionality."""

    def test_metrics_module_imports(self):
        """Test that metrics module can be imported."""
        from tools.dashvoice import metrics
        assert hasattr(metrics, 'instrument_app')
        assert hasattr(metrics, 'track_tts_generation')
        assert hasattr(metrics, 'track_stt_transcription')
        assert hasattr(metrics, 'track_pipeline_processing')

    def test_prometheus_availability_flag(self):
        """Test prometheus availability flag is set correctly."""
        from tools.dashvoice.metrics import PROMETHEUS_AVAILABLE
        # This should be True if prometheus_client is installed
        # or False if not - either way it should be a boolean
        assert isinstance(PROMETHEUS_AVAILABLE, bool)

    def test_track_tts_generation_context_manager(self):
        """Test TTS generation tracking context manager."""
        from tools.dashvoice.metrics import track_tts_generation

        # Should not raise even if prometheus is not available
        with track_tts_generation(voice="af_bella"):
            time.sleep(0.01)

    def test_track_stt_transcription_context_manager(self):
        """Test STT transcription tracking context manager."""
        from tools.dashvoice.metrics import track_stt_transcription

        with track_stt_transcription():
            time.sleep(0.01)

    def test_track_pipeline_processing_context_manager(self):
        """Test pipeline processing tracking context manager."""
        from tools.dashvoice.metrics import track_pipeline_processing

        with track_pipeline_processing("vad+stt"):
            time.sleep(0.01)

    def test_track_vad_processing_context_manager(self):
        """Test VAD processing tracking context manager."""
        from tools.dashvoice.metrics import track_vad_processing

        with track_vad_processing():
            time.sleep(0.01)


class TestMetricRecording:
    """Test metric recording functions."""

    def test_record_tts_output(self):
        """Test TTS output recording."""
        from tools.dashvoice.metrics import record_tts_output

        # Should not raise
        record_tts_output(
            voice="af_bella",
            samples=24000,
            sample_rate=24000,
            generation_time_s=0.5,
        )

    def test_record_stt_output(self):
        """Test STT output recording."""
        from tools.dashvoice.metrics import record_stt_output

        record_stt_output(audio_duration_s=5.0, transcription="Hello world")

    def test_record_vad_segment(self):
        """Test VAD segment recording."""
        from tools.dashvoice.metrics import record_vad_segment

        record_vad_segment(is_speech=True)
        record_vad_segment(is_speech=False)

    def test_record_pipeline_output(self):
        """Test pipeline output recording."""
        from tools.dashvoice.metrics import record_pipeline_output

        record_pipeline_output(num_speakers=2, dashvoice_voice="af_bella")
        record_pipeline_output(num_speakers=1, dashvoice_voice=None)

    def test_record_model_loaded(self):
        """Test model loaded recording."""
        from tools.dashvoice.metrics import record_model_loaded

        record_model_loaded("tts", True)
        record_model_loaded("stt", False)

    def test_record_model_warmup(self):
        """Test model warmup recording."""
        from tools.dashvoice.metrics import record_model_warmup

        record_model_warmup("tts", 1.5)
        record_model_warmup("stt", 2.0)

    def test_record_prewarm_complete(self):
        """Test prewarm complete recording."""
        from tools.dashvoice.metrics import record_prewarm_complete

        record_prewarm_complete()

    def test_record_startup(self):
        """Test startup recording."""
        from tools.dashvoice.metrics import record_startup

        record_startup()


class TestWebSocketMetrics:
    """Test WebSocket session tracking metrics."""

    def test_websocket_session_tracking(self):
        """Test WebSocket session open/close tracking."""
        from tools.dashvoice.metrics import (
            websocket_session_closed,
            websocket_session_opened,
        )

        # Should not raise
        websocket_session_opened("stream")
        websocket_session_closed("stream")

    def test_websocket_message_tracking(self):
        """Test WebSocket message tracking."""
        from tools.dashvoice.metrics import record_websocket_message

        record_websocket_message("stream", "config")
        record_websocket_message("stream", "audio")
        record_websocket_message("transcribe", "end")

    def test_websocket_audio_tracking(self):
        """Test WebSocket audio bytes tracking."""
        from tools.dashvoice.metrics import record_websocket_audio

        record_websocket_audio("stream", 32000)
        record_websocket_audio("transcribe", 16000)


class TestMetricsEndpoint:
    """Test Prometheus metrics endpoint integration."""

    def test_instrument_app_function_exists(self):
        """Test that instrument_app function exists."""
        from tools.dashvoice.metrics import instrument_app
        assert callable(instrument_app)

    def test_metrics_summary(self):
        """Test get_metrics_summary function."""
        from tools.dashvoice.metrics import get_metrics_summary

        summary = get_metrics_summary()
        assert isinstance(summary, dict)


class TestSemaphoreMetrics:
    """Test semaphore tracking metrics."""

    def test_track_semaphore_wait_context_manager(self):
        """Test semaphore wait tracking context manager."""
        from tools.dashvoice.metrics import track_semaphore_wait

        with track_semaphore_wait("tts"):
            time.sleep(0.01)

    def test_semaphore_acquired_and_released(self):
        """Test semaphore acquire/release tracking."""
        from tools.dashvoice.metrics import semaphore_acquired, semaphore_released

        semaphore_acquired("tts")
        semaphore_released("tts")
        semaphore_acquired("pipeline")
        semaphore_released("pipeline")


class TestPrometheusMetrics:
    """Test Prometheus metrics when prometheus_client is available."""

    def test_metrics_registry_exists(self):
        """Test that metrics are registered in the registry."""
        from tools.dashvoice.metrics import PROMETHEUS_AVAILABLE, REGISTRY

        if PROMETHEUS_AVAILABLE:
            assert REGISTRY is not None
        else:
            pytest.skip("prometheus_client not installed")

    def test_metrics_definitions_exist(self):
        """Test that metric definitions exist."""
        from tools.dashvoice import metrics

        if not metrics.PROMETHEUS_AVAILABLE:
            pytest.skip("prometheus_client not installed")

        # Request metrics
        assert hasattr(metrics, 'REQUEST_LATENCY')
        assert hasattr(metrics, 'REQUEST_COUNT')
        assert hasattr(metrics, 'REQUEST_IN_PROGRESS')

        # TTS metrics
        assert hasattr(metrics, 'TTS_GENERATION_LATENCY')
        assert hasattr(metrics, 'TTS_AUDIO_DURATION')
        assert hasattr(metrics, 'TTS_RTF')
        assert hasattr(metrics, 'TTS_SAMPLES_GENERATED')

        # STT metrics
        assert hasattr(metrics, 'STT_TRANSCRIPTION_LATENCY')
        assert hasattr(metrics, 'STT_AUDIO_PROCESSED')
        assert hasattr(metrics, 'STT_TRANSCRIPTION_LENGTH')

        # VAD metrics
        assert hasattr(metrics, 'VAD_SEGMENTS_DETECTED')
        assert hasattr(metrics, 'VAD_PROCESSING_LATENCY')

        # Pipeline metrics
        assert hasattr(metrics, 'PIPELINE_PROCESSING_LATENCY')
        assert hasattr(metrics, 'PIPELINE_SPEAKERS_DETECTED')
        assert hasattr(metrics, 'PIPELINE_DASHVOICE_DETECTED')

        # WebSocket metrics
        assert hasattr(metrics, 'WEBSOCKET_SESSIONS_ACTIVE')
        assert hasattr(metrics, 'WEBSOCKET_MESSAGES_RECEIVED')
        assert hasattr(metrics, 'WEBSOCKET_AUDIO_BYTES_RECEIVED')

        # Model status metrics
        assert hasattr(metrics, 'MODEL_LOADED')
        assert hasattr(metrics, 'MODEL_WARMUP_LATENCY')
        assert hasattr(metrics, 'STARTUP_TIME')
        assert hasattr(metrics, 'PREWARM_COMPLETE')
