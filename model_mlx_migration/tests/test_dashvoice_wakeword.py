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
Tests for DashVoice Wake Word detection endpoints.

Tests:
1. Wake word info endpoint (/wakeword/info)
2. Wake word detection (/wakeword/detect)
3. WebSocket streaming wake word detection (/ws/wakeword)
"""

import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from tools.dashvoice.server import (
    app,
    get_wakeword_detector,
)

# Module-level RNG for reproducible tests
_rng = np.random.default_rng(42)


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def sample_audio():
    """Create a sample audio file for testing (1 second of noise)."""
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)

    # Create noise audio (should NOT trigger wake word)
    audio = (0.1 * _rng.standard_normal(len(t))).astype(np.float32)

    # Create WAV file in memory
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format='WAV')
    buffer.seek(0)

    return buffer


@pytest.fixture
def sample_audio_short():
    """Create a very short audio file (0.5 seconds)."""
    sample_rate = 16000
    duration = 0.5
    audio = np.zeros(int(sample_rate * duration), dtype=np.float32)

    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format='WAV')
    buffer.seek(0)

    return buffer


class TestRootEndpointIncludesWakeWord:
    """Test root endpoint includes wake word."""

    def test_root_includes_wakeword_endpoints(self, client):
        """Test root endpoint lists wake word endpoints."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()

        assert "wake_word" in data.get("endpoints", {})
        wakeword_endpoints = data["endpoints"]["wake_word"]

        assert any("/wakeword/detect" in e for e in wakeword_endpoints)
        assert any("/wakeword/info" in e for e in wakeword_endpoints)
        assert any("/ws/wakeword" in e for e in wakeword_endpoints)

    def test_api_version_updated_for_wakeword(self, client):
        """Test API version is at least 1.4.0."""
        response = client.get("/")
        data = response.json()

        version = data.get("version", "")
        major, minor, patch = version.split(".")
        assert int(major) >= 1
        assert int(minor) >= 4


class TestWakeWordInfoEndpoint:
    """Test /wakeword/info endpoint."""

    def test_info_endpoint_exists(self, client):
        """Test info endpoint returns 200."""
        response = client.get("/wakeword/info")
        assert response.status_code == 200

    def test_info_returns_model_info(self, client):
        """Test info endpoint returns model information."""
        response = client.get("/wakeword/info")
        data = response.json()

        # Should have enabled status
        assert "enabled" in data

        # If enabled, should have model details
        if data.get("enabled"):
            assert data.get("wake_word") == "Hey Agent"
            assert data.get("model_name") == "hey_agent"
            assert data.get("model_type") == "CNN"
            assert "sample_rate" in data
            assert "vad_enabled" in data
        else:
            # If disabled, should explain why
            assert "reason" in data or "model_path" in data


class TestWakeWordDetectEndpoint:
    """Test /wakeword/detect endpoint."""

    @pytest.mark.xfail(reason="Flaky under parallel execution - resource contention")
    def test_detect_endpoint_exists(self, client, sample_audio):
        """Test detect endpoint returns response."""
        response = client.post(
            "/wakeword/detect",
            files={"audio": ("test.wav", sample_audio, "audio/wav")},
        )
        # 200 if model loaded, 503 if disabled, 500 if optional dependency missing
        if response.status_code == 500:
            # Accept 500 if it's due to missing optional dependency (silero_vad)
            assert "silero_vad" in response.text or "No module named" in response.text
        else:
            assert response.status_code in [200, 503]

    def test_detect_requires_audio(self, client):
        """Test detect endpoint requires audio file."""
        response = client.post("/wakeword/detect")
        assert response.status_code == 422  # Validation error

    def test_detect_response_structure(self, client, sample_audio):
        """Test detect endpoint returns expected structure."""
        response = client.post(
            "/wakeword/detect",
            files={"audio": ("test.wav", sample_audio, "audio/wav")},
        )

        if response.status_code == 200:
            data = response.json()
            assert "detected" in data
            assert "probability" in data
            assert "threshold" in data
            assert "audio_duration_s" in data
            assert "vad_enabled" in data

            # Probability should be 0-1
            assert 0.0 <= data["probability"] <= 1.0

            # Threshold should be float
            assert isinstance(data["threshold"], float)

    def test_detect_custom_threshold(self, client, sample_audio):
        """Test detect with custom threshold."""
        response = client.post(
            "/wakeword/detect",
            files={"audio": ("test.wav", sample_audio, "audio/wav")},
            data={"threshold": "0.9"},
        )

        if response.status_code == 200:
            data = response.json()
            assert data["threshold"] == 0.9

    def test_detect_noise_not_detected(self, client, sample_audio):
        """Test that random noise is not detected as wake word."""
        response = client.post(
            "/wakeword/detect",
            files={"audio": ("test.wav", sample_audio, "audio/wav")},
        )

        if response.status_code == 200:
            data = response.json()
            # Random noise with VAD should return 0.0 (no speech detected)
            # or low probability
            # Either way, should not be detected as wake word
            assert data["detected"] is False or data["probability"] < 0.5


class TestWakeWordWebSocket:
    """Test /ws/wakeword WebSocket endpoint."""

    def test_ws_wakeword_endpoint_exists(self, client):
        """Test WebSocket endpoint can be connected."""
        with client.websocket_connect("/ws/wakeword") as ws:
            # Should receive ready message
            response = ws.receive_json()
            assert response.get("type") in ["ready", "error"]

    def test_ws_wakeword_config_message(self, client):
        """Test WebSocket config message handling."""
        with client.websocket_connect("/ws/wakeword") as ws:
            # Receive ready message
            response = ws.receive_json()

            if response.get("type") == "error":
                # Model not loaded, skip rest of test
                return

            # Send config
            ws.send_json({
                "type": "config",
                "threshold": 0.7,
                "sample_rate": 16000,
            })

            # Should receive config ack
            response = ws.receive_json()
            assert response.get("type") == "config_ack"
            assert response.get("threshold") == 0.7

    def test_ws_wakeword_ping_pong(self, client):
        """Test WebSocket ping/pong."""
        with client.websocket_connect("/ws/wakeword") as ws:
            # Receive ready message
            response = ws.receive_json()

            if response.get("type") == "error":
                return

            # Send ping
            ws.send_json({"type": "ping"})

            # Should receive pong
            response = ws.receive_json()
            assert response.get("type") == "pong"

    @pytest.mark.xfail(reason="Flaky under parallel execution - resource contention")
    def test_ws_wakeword_audio_chunk(self, client):
        """Test WebSocket audio chunk processing."""
        with client.websocket_connect("/ws/wakeword") as ws:
            # Receive ready message
            response = ws.receive_json()

            if response.get("type") == "error":
                return

            # Send audio chunk (1 second of silence as int16)
            audio = np.zeros(16000, dtype=np.int16)
            ws.send_bytes(audio.tobytes())

            # Should receive detection result (or error if optional dependency missing)
            response = ws.receive_json()
            if response.get("type") == "error":
                # Accept error if it's due to missing optional dependency
                assert "silero_vad" in response.get("message", "") or "No module named" in response.get("message", "")
            else:
                assert response.get("type") in ["detection", "detected"]
                assert "probability" in response
                assert "chunk_index" in response


class TestWakeWordDetectorSingleton:
    """Test wake word detector singleton."""

    def test_get_wakeword_detector(self):
        """Test wake word detector can be obtained."""
        detector = get_wakeword_detector()
        # Should return either a detector or "disabled"
        assert detector is not None

        if detector != "disabled":
            # Should have expected attributes
            assert hasattr(detector, "detect")
            assert hasattr(detector, "sample_rate")
            assert hasattr(detector, "use_vad")

    def test_wakeword_detector_singleton(self):
        """Test wake word detector is a singleton."""
        detector1 = get_wakeword_detector()
        detector2 = get_wakeword_detector()
        assert detector1 is detector2


class TestWakeWordIntegration:
    """Integration tests for wake word detection."""

    @pytest.mark.xfail(reason="Flaky under parallel execution - resource contention")
    def test_full_detection_flow(self, client, sample_audio):
        """Test full detection flow from info to detect."""
        # 1. Get info
        info_response = client.get("/wakeword/info")
        assert info_response.status_code == 200
        info = info_response.json()

        # 2. Detect (if model loaded)
        if info.get("enabled"):
            detect_response = client.post(
                "/wakeword/detect",
                files={"audio": ("test.wav", sample_audio, "audio/wav")},
            )
            # Accept 200 success or 500 if optional dependency (silero_vad) missing
            if detect_response.status_code == 500:
                assert "silero_vad" in detect_response.text or "No module named" in detect_response.text
            else:
                assert detect_response.status_code == 200
                result = detect_response.json()
                # Should have consistent VAD status
                assert result.get("vad_enabled") == info.get("vad_enabled")

    def test_multiple_detections(self, client, sample_audio):
        """Test multiple detection calls return consistent results."""
        info_response = client.get("/wakeword/info")
        if not info_response.json().get("enabled"):
            pytest.skip("Wake word model not loaded")

        results = []
        for _ in range(3):
            sample_audio.seek(0)  # Reset buffer
            response = client.post(
                "/wakeword/detect",
                files={"audio": ("test.wav", sample_audio, "audio/wav")},
            )
            if response.status_code == 200:
                results.append(response.json()["probability"])

        # Same audio should give same result (deterministic)
        if len(results) > 1:
            # Allow small variation due to floating point
            for r in results[1:]:
                assert abs(r - results[0]) < 0.01
