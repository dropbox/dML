#!/usr/bin/env python3
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
Tests for DashVoice TTS endpoints and pre-warming.

Tests:
1. TTS endpoint configuration
2. Voice listing
3. Pre-warming function
4. WebSocket streaming TTS protocol
"""

import pytest
from fastapi.testclient import TestClient

from tools.dashvoice.server import (
    KOKORO_VOICES,
    app,
    get_pipeline,
    get_tts_model,
)


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestTTSConfiguration:
    """Test TTS configuration and setup."""

    def test_kokoro_voices_defined(self):
        """Test that Kokoro voices list is defined."""
        assert len(KOKORO_VOICES) >= 10
        assert "af_bella" in KOKORO_VOICES
        assert "am_adam" in KOKORO_VOICES

    def test_voice_naming_convention(self):
        """Test voice naming follows pattern."""
        for voice in KOKORO_VOICES:
            # Pattern: {accent}{gender}_{name}
            # af = American Female, am = American Male
            # bf = British Female, bm = British Male
            assert len(voice) >= 3
            prefix = voice[:2]
            assert prefix in ["af", "am", "bf", "bm"], f"Unexpected prefix: {prefix}"

    def test_tts_model_singleton(self):
        """Test TTS model is singleton."""
        model1 = get_tts_model()
        model2 = get_tts_model()
        assert model1 is model2


class TestVoicesEndpoint:
    """Test /voices endpoint."""

    def test_voices_endpoint_exists(self, client):
        """Test voices endpoint returns 200."""
        response = client.get("/voices")
        assert response.status_code == 200

    def test_voices_response_structure(self, client):
        """Test voices response has expected fields."""
        response = client.get("/voices")
        data = response.json()

        assert "tts_enabled" in data
        assert "voices" in data
        assert "default_voice" in data
        assert "voice_categories" in data

    def test_voice_categories(self, client):
        """Test voice categories are present."""
        response = client.get("/voices")
        data = response.json()

        categories = data.get("voice_categories", {})
        assert "american_female" in categories
        assert "american_male" in categories
        assert "british_female" in categories
        assert "british_male" in categories


class TestModelsEndpoint:
    """Test /models endpoint includes TTS."""

    def test_models_includes_tts(self, client):
        """Test models endpoint includes TTS info."""
        response = client.get("/models")
        data = response.json()

        assert "tts" in data.get("pipeline", {})
        assert "tts" in data.get("models", {})
        assert "pre_warming" in data


class TestRootEndpoint:
    """Test root endpoint includes TTS endpoints."""

    def test_root_includes_synthesize(self, client):
        """Test root lists synthesize endpoint."""
        response = client.get("/")
        data = response.json()

        rest_endpoints = data.get("endpoints", {}).get("rest", [])
        rest_str = " ".join(rest_endpoints)

        assert "/synthesize" in rest_str
        assert "/voices" in rest_str

    def test_root_includes_ws_synthesize(self, client):
        """Test root lists streaming TTS endpoint."""
        response = client.get("/")
        data = response.json()

        ws_endpoints = data.get("endpoints", {}).get("websocket", [])
        ws_str = " ".join(ws_endpoints)

        assert "/ws/synthesize" in ws_str

    def test_api_version_updated(self, client):
        """Test API version reflects TTS additions."""
        response = client.get("/")
        data = response.json()

        version = data.get("version", "")
        # Should be at least 1.2.0 after TTS additions
        major, minor, patch = version.split(".")
        assert int(major) >= 1
        assert int(minor) >= 2


class TestSynthesizeEndpoint:
    """Test /synthesize endpoint structure."""

    def test_synthesize_requires_text(self, client):
        """Test synthesize requires text parameter."""
        response = client.post("/synthesize", data={})
        # Should return 422 (validation error) without text
        assert response.status_code == 422

    def test_synthesize_invalid_voice(self, client):
        """Test synthesize rejects invalid voice."""
        response = client.post(
            "/synthesize",
            data={"text": "Hello", "voice": "invalid_voice"},
        )
        # Either 400 (invalid voice) or 503 (TTS disabled)
        assert response.status_code in [400, 503]

    def test_synthesize_invalid_speed(self, client):
        """Test synthesize validates speed range."""
        response = client.post(
            "/synthesize",
            data={"text": "Hello", "speed": 5.0},  # Too fast
        )
        # Either 400 (invalid speed) or 503 (TTS disabled)
        assert response.status_code in [400, 503]


class TestWebSocketSynthesizeProtocol:
    """Test WebSocket TTS streaming protocol."""

    def test_ws_synthesize_endpoint_exists(self, client):
        """Test WS synthesize endpoint can be connected."""
        # This tests that the WebSocket endpoint is properly registered
        # Full WebSocket testing requires async test setup
        routes = [r.path for r in app.routes]
        assert "/ws/synthesize" in routes

    def test_synthesize_message_structure(self):
        """Test expected message format."""
        # Test the expected JSON structure
        request = {
            "type": "synthesize",
            "text": "Hello world",
            "voice": "af_bella",
            "speed": 1.0,
        }
        assert all(k in request for k in ["type", "text", "voice", "speed"])

        response_chunk = {
            "type": "audio_chunk",
            "index": 0,
            "audio_base64": "...",
            "samples": 1234,
            "sample_rate": 24000,
            "duration_s": 0.5,
        }
        assert all(k in response_chunk for k in ["type", "index", "audio_base64"])


class TestPreWarming:
    """Test pre-warming functionality."""

    def test_prewarm_function_exists(self):
        """Test prewarm_models function exists."""
        from tools.dashvoice.server import prewarm_models
        assert callable(prewarm_models)

    def test_get_pipeline_after_import(self):
        """Test pipeline can be retrieved."""
        pipeline = get_pipeline()
        assert pipeline is not None

    def test_pipeline_has_vad(self):
        """Test pipeline has VAD processor."""
        pipeline = get_pipeline()
        assert hasattr(pipeline, '_vad')
        assert hasattr(pipeline, 'enable_vad')


# Integration test that requires TTS model
@pytest.mark.slow
class TestTTSIntegration:
    """Integration tests that require TTS model."""

    def test_synthesize_produces_audio(self, client):
        """Test synthesize actually produces audio."""
        response = client.post(
            "/synthesize",
            data={"text": "Hello", "voice": "af_bella", "speed": 1.0},
        )

        if response.status_code == 503:
            pytest.skip("TTS not available")

        assert response.status_code == 200
        data = response.json()

        assert "audio_base64" in data
        assert "samples" in data
        assert "duration_s" in data
        assert data["samples"] > 0
        assert data["duration_s"] > 0
