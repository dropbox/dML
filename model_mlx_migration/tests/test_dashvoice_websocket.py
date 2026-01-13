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
Tests for DashVoice WebSocket streaming endpoints.

These tests verify the WebSocket protocol for real-time audio streaming.
"""


import numpy as np
import pytest

# Check if test client is available
try:
    from fastapi.testclient import TestClient  # noqa: F401
    TESTCLIENT_AVAILABLE = True
except ImportError:
    TESTCLIENT_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not TESTCLIENT_AVAILABLE, reason="fastapi.testclient not available (install httpx)",
)


class TestWebSocketProtocol:
    """Test WebSocket protocol implementation."""

    def test_server_routes_exist(self):
        """Verify WebSocket routes are registered."""
        from tools.dashvoice.server import app

        routes = [r.path for r in app.routes if hasattr(r, 'path')]
        assert "/ws/stream" in routes
        assert "/ws/transcribe" in routes
        assert "/ws/sessions" in routes

    def test_streaming_session_class(self):
        """Test StreamingSession initialization."""
        from tools.dashvoice.server import StreamingSession

        # Mock websocket
        class MockWebSocket:
            async def send_json(self, data):
                pass

        session = StreamingSession(
            websocket=MockWebSocket(),
            sample_rate=16000,
            enable_stt=True,
            enable_diarization=False,
            chunk_duration_ms=500,
        )

        assert session.sample_rate == 16000
        assert session.enable_stt is True
        assert session.enable_diarization is False
        assert session.chunk_samples == 8000  # 500ms at 16kHz
        assert session.is_active is True
        assert session.session_id.startswith("session_")

    def test_audio_data_format(self):
        """Test audio data format conversion."""
        # Create test int16 audio data
        audio_int16 = np.array([0, 16384, 32767, -32768, -16384], dtype=np.int16)
        audio_bytes = audio_int16.tobytes()

        # Convert back
        audio_decoded = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # Verify conversion
        expected = np.array([0.0, 0.5, 1.0 - 1/32768, -1.0, -0.5], dtype=np.float32)
        np.testing.assert_array_almost_equal(audio_decoded, expected, decimal=4)


class TestRESTEndpoints:
    """Test REST endpoints that support WebSocket functionality."""

    def test_sessions_endpoint(self):
        """Test /ws/sessions endpoint."""
        from fastapi.testclient import TestClient

        from tools.dashvoice.server import app

        client = TestClient(app)
        response = client.get("/ws/sessions")
        assert response.status_code == 200
        data = response.json()
        assert "active_sessions" in data
        assert "sessions" in data
        assert isinstance(data["sessions"], list)

    def test_root_endpoint_includes_websocket(self):
        """Test root endpoint lists WebSocket endpoints."""
        from fastapi.testclient import TestClient

        from tools.dashvoice.server import app

        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "websocket" in data.get("endpoints", {})
        ws_endpoints = data["endpoints"]["websocket"]
        assert any("/ws/stream" in e for e in ws_endpoints)
        assert any("/ws/transcribe" in e for e in ws_endpoints)


class TestWebSocketConnection:
    """Test actual WebSocket connections using FastAPI test client."""

    def test_websocket_stream_config(self):
        """Test WebSocket /ws/stream config message."""
        from fastapi.testclient import TestClient

        from tools.dashvoice.server import app

        client = TestClient(app)
        with client.websocket_connect("/ws/stream") as ws:
            # Send config
            ws.send_json({
                "type": "config",
                "sample_rate": 16000,
                "enable_stt": False,
                "enable_diarization": False,
                "enable_echo_cancel": False,
                "chunk_duration_ms": 500,
            })

            # Receive config ack
            response = ws.receive_json()
            assert response["type"] == "config_ack"
            assert "session_id" in response
            assert response["status"] == "ready"

            # Send end
            ws.send_json({"type": "end"})

            # Receive session end
            response = ws.receive_json()
            assert response["type"] == "session_end"
            assert response["total_duration_s"] == 0.0

    def test_websocket_stream_audio_chunk(self):
        """Test WebSocket /ws/stream with audio data."""
        from fastapi.testclient import TestClient

        from tools.dashvoice.server import app

        client = TestClient(app)
        with client.websocket_connect("/ws/stream") as ws:
            # Send config with small chunk duration
            ws.send_json({
                "type": "config",
                "sample_rate": 16000,
                "enable_stt": False,
                "enable_diarization": False,
                "enable_echo_cancel": False,
                "chunk_duration_ms": 100,  # 100ms chunks = 1600 samples
            })

            response = ws.receive_json()
            assert response["type"] == "config_ack"
            session_id = response["session_id"]

            # Send audio data (1600 samples = 100ms at 16kHz)
            # Generate simple sine wave
            t = np.linspace(0, 0.1, 1600, dtype=np.float32)
            audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
            audio_int16 = (audio * 32767).astype(np.int16)
            ws.send_bytes(audio_int16.tobytes())

            # Should receive a result (either segment or silence)
            response = ws.receive_json()
            assert response["type"] in ("segment", "silence")
            assert response["session_id"] == session_id
            assert "processing_time_ms" in response

            # End session
            ws.send_json({"type": "end"})
            response = ws.receive_json()
            assert response["type"] == "session_end"

    def test_websocket_ping_pong(self):
        """Test WebSocket ping/pong keep-alive."""
        from fastapi.testclient import TestClient

        from tools.dashvoice.server import app

        client = TestClient(app)
        with client.websocket_connect("/ws/stream") as ws:
            # Send ping
            ws.send_json({"type": "ping"})

            # Receive pong
            response = ws.receive_json()
            assert response["type"] == "pong"
            assert "timestamp" in response

            # End
            ws.send_json({"type": "end"})

    def test_websocket_transcribe_endpoint(self):
        """Test WebSocket /ws/transcribe endpoint."""
        from fastapi.testclient import TestClient

        from tools.dashvoice.server import app

        client = TestClient(app)
        with client.websocket_connect("/ws/transcribe") as ws:
            # First receive ready message with capabilities
            response = ws.receive_json()
            assert response["type"] == "ready"
            assert "mode" in response
            assert "features" in response

            # Send end immediately (no audio)
            ws.send_json({"type": "end"})

            # Should receive complete with empty text
            response = ws.receive_json()
            assert response["type"] == "complete"
            assert "full_text" in response
            assert "segments" in response

    def test_websocket_transcribe_streaming_config(self):
        """Test WebSocket /ws/transcribe with config message."""
        from fastapi.testclient import TestClient

        from tools.dashvoice.server import app

        client = TestClient(app)
        with client.websocket_connect("/ws/transcribe") as ws:
            # First receive ready message
            response = ws.receive_json()
            assert response["type"] == "ready"

            # Send config
            ws.send_json({
                "type": "config",
                "use_vad": True,
                "language": "en",
                "emit_partials": True,
                "min_chunk_duration": 0.5,
                "max_chunk_duration": 10.0,
                "silence_threshold": 0.5,
            })

            # Receive config ack
            response = ws.receive_json()
            assert response["type"] == "config_ack"

            # Send end
            ws.send_json({"type": "end"})

            # Should receive complete
            response = ws.receive_json()
            assert response["type"] == "complete"

    def test_websocket_transcribe_streaming_mode_detection(self):
        """Test that /ws/transcribe indicates streaming vs fallback mode."""
        from fastapi.testclient import TestClient

        from tools.dashvoice.server import app

        client = TestClient(app)
        with client.websocket_connect("/ws/transcribe") as ws:
            response = ws.receive_json()
            assert response["type"] == "ready"

            # Mode should be 'streaming' if WhisperMLX available, otherwise 'fallback'
            assert response["mode"] in ("streaming", "fallback")

            # Streaming mode should have VAD feature
            if response["mode"] == "streaming":
                assert "vad" in response["features"]
                assert "partial_results" in response["features"]
            else:
                assert "basic_chunked" in response["features"]

            ws.send_json({"type": "end"})
            response = ws.receive_json()
            assert response["type"] == "complete"


def test_websocket_integration():
    """Basic integration test for WebSocket functionality."""
    from tools.dashvoice.server import _streaming_sessions

    # Verify no sessions initially
    assert len(_streaming_sessions) == 0

    print("WebSocket integration test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
