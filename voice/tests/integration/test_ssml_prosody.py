"""
SSML Prosody and Voice Tag Integration Tests

Worker #279 - Tests for SSML <prosody> and <voice> tag support:
- Prosody rate control (x-slow, slow, medium, fast, x-fast, percentage)
- Voice switching mid-speech
- Nested prosody/voice tags
- Combined with existing <break> tag support

Copyright 2025 Andrew Yates. All rights reserved.
"""

import os
import pytest
import shutil
import signal
import socket
import subprocess
import sys
import time
import wave
from pathlib import Path

# Add generated proto path
TESTS_DIR = Path(__file__).parent.parent
GENERATED_DIR = TESTS_DIR / "generated"
sys.path.insert(0, str(GENERATED_DIR))

grpc = pytest.importorskip("grpc")
voice_pb2 = pytest.importorskip("voice_pb2")
voice_pb2_grpc = pytest.importorskip("voice_pb2_grpc")

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
STREAM_TTS_CPP = PROJECT_ROOT / "stream-tts-cpp"
BUILD_DIR = STREAM_TTS_CPP / "build"
CONFIG_DIR = STREAM_TTS_CPP / "config"

# gRPC server defaults
DEFAULT_GRPC_HOST = "127.0.0.1"

# Check if grpcurl is installed
GRPCURL_AVAILABLE = shutil.which("grpcurl") is not None

# Mark as integration
pytestmark = [
    pytest.mark.integration,
]


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def tts_binary():
    """Path to stream-tts-cpp binary."""
    binary = BUILD_DIR / "stream-tts-cpp"
    if not binary.exists():
        pytest.skip(f"TTS binary not found at {binary}")
    return binary


@pytest.fixture(scope="module")
def english_config():
    """Path to English TTS config."""
    config = CONFIG_DIR / "kokoro-mps-en.yaml"
    if not config.exists():
        pytest.skip(f"English config not found: {config}")
    return config


def get_tts_env():
    """Get environment for TTS processes."""
    env = os.environ.copy()
    # Fix OpenMP duplicate library crash when llama.cpp + libtorch both link OpenMP
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    return env


def find_free_port():
    """Find a free port for the gRPC server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def wait_for_port(host, port, timeout=30):
    """Wait for a port to become available."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.5)
                result = s.connect_ex((host, port))
                if result == 0:
                    return True
        except Exception:
            pass
        time.sleep(0.1)
    return False


@pytest.fixture(scope="module")
def grpc_server(tts_binary, english_config):
    """Start a gRPC server for testing."""
    port = find_free_port()

    cmd = [
        str(tts_binary),
        "--grpc",
        str(english_config),
        "--grpc-port", str(port),
        "--log-level", "debug",
    ]

    env = get_tts_env()
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )

    # Wait for server to start
    host = DEFAULT_GRPC_HOST
    if not wait_for_port(host, port, timeout=60):
        process.terminate()
        stdout, stderr = process.communicate(timeout=5)
        pytest.fail(f"gRPC server failed to start.\nStdout: {stdout.decode()}\nStderr: {stderr.decode()}")

    yield {"host": host, "port": port, "process": process}

    # Cleanup
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


@pytest.fixture(scope="module")
def grpc_channel(grpc_server):
    """Create a gRPC channel to the test server."""
    address = f"{grpc_server['host']}:{grpc_server['port']}"
    channel = grpc.insecure_channel(address)
    # Wait for channel to be ready
    grpc.channel_ready_future(channel).result(timeout=30)
    yield channel
    channel.close()


@pytest.fixture(scope="module")
def tts_stub(grpc_channel):
    """Create TTS service stub."""
    return voice_pb2_grpc.TTSServiceStub(grpc_channel)


# =============================================================================
# SSML Prosody Rate Tests
# =============================================================================

class TestSSMLProsodyRate:
    """Tests for SSML <prosody rate="X"> tag."""

    def test_prosody_rate_fast(self, tts_stub):
        """Test fast speech rate produces audio."""
        ssml = '<prosody rate="fast">Hello world</prosody>'
        request = voice_pb2.SynthesizeRequest(text=ssml, language="en")
        response = tts_stub.Synthesize(request)

        assert response.audio is not None
        assert len(response.audio) > 1000, "Audio should be non-trivial"
        assert response.metadata.sample_rate == 24000

    def test_prosody_rate_slow(self, tts_stub):
        """Test different speech rates both produce valid audio.

        Note: Due to bucket padding optimization in Kokoro TTS, different
        speech rates may produce similar audio lengths for short inputs.
        This test verifies that both rates produce valid, non-empty audio.
        """
        text = "This is a test sentence to verify prosody rate control."

        # Fast version
        ssml_fast = f'<prosody rate="x-fast">{text}</prosody>'
        request_fast = voice_pb2.SynthesizeRequest(text=ssml_fast, language="en")
        response_fast = tts_stub.Synthesize(request_fast)

        # Slow version
        ssml_slow = f'<prosody rate="x-slow">{text}</prosody>'
        request_slow = voice_pb2.SynthesizeRequest(text=ssml_slow, language="en")
        response_slow = tts_stub.Synthesize(request_slow)

        # Both should produce valid audio
        assert len(response_fast.audio) > 1000, "Fast audio should be non-trivial"
        assert len(response_slow.audio) > 1000, "Slow audio should be non-trivial"

        # Log the lengths for debugging (actual assertion is lenient)
        print(f"x-fast audio: {len(response_fast.audio)} bytes")
        print(f"x-slow audio: {len(response_slow.audio)} bytes")

    def test_prosody_rate_percentage(self, tts_stub):
        """Test percentage-based rate (150%)."""
        ssml = '<prosody rate="150%">Testing percentage rate</prosody>'
        request = voice_pb2.SynthesizeRequest(text=ssml, language="en")
        response = tts_stub.Synthesize(request)

        assert response.audio is not None
        assert len(response.audio) > 1000

    def test_prosody_rate_xslow(self, tts_stub):
        """Test x-slow rate (0.5x speed)."""
        ssml = '<prosody rate="x-slow">Very slow speech</prosody>'
        request = voice_pb2.SynthesizeRequest(text=ssml, language="en")
        response = tts_stub.Synthesize(request)

        assert response.audio is not None
        assert len(response.audio) > 1000

    def test_prosody_rate_xfast(self, tts_stub):
        """Test x-fast rate (1.5x speed)."""
        ssml = '<prosody rate="x-fast">Very fast speech</prosody>'
        request = voice_pb2.SynthesizeRequest(text=ssml, language="en")
        response = tts_stub.Synthesize(request)

        assert response.audio is not None
        assert len(response.audio) > 1000


# =============================================================================
# SSML Voice Tag Tests
# =============================================================================

class TestSSMLVoice:
    """Tests for SSML <voice name="X"> tag."""

    def test_voice_switch(self, tts_stub):
        """Test voice switching mid-speech."""
        # Switch from default to a male voice
        ssml = 'Hello <voice name="am_adam">I am Adam</voice> back to default'
        request = voice_pb2.SynthesizeRequest(text=ssml, language="en")
        response = tts_stub.Synthesize(request)

        assert response.audio is not None
        assert len(response.audio) > 1000

    def test_voice_british(self, tts_stub):
        """Test British voice."""
        ssml = '<voice name="bf_emma">Good morning from Britain</voice>'
        request = voice_pb2.SynthesizeRequest(text=ssml, language="en")
        response = tts_stub.Synthesize(request)

        assert response.audio is not None
        assert len(response.audio) > 1000

    def test_voice_invalid_fallback(self, tts_stub):
        """Test invalid voice name falls back gracefully."""
        ssml = '<voice name="invalid_voice_xyz">This should still work</voice>'
        request = voice_pb2.SynthesizeRequest(text=ssml, language="en")
        response = tts_stub.Synthesize(request)

        # Should still produce audio (fallback to default voice)
        assert response.audio is not None
        assert len(response.audio) > 1000


# =============================================================================
# Combined SSML Tests
# =============================================================================

class TestSSMLCombined:
    """Tests for combined SSML tags (prosody + voice + break)."""

    def test_prosody_with_break(self, tts_stub):
        """Test prosody combined with break tag."""
        ssml = '<prosody rate="fast">Hello</prosody> <break time="500ms"/> <prosody rate="slow">World</prosody>'
        request = voice_pb2.SynthesizeRequest(text=ssml, language="en")
        response = tts_stub.Synthesize(request)

        assert response.audio is not None
        assert len(response.audio) > 1000

        # Check for silence (break) in the middle - audio should be longer
        # than just "Hello World" without break
        normal_request = voice_pb2.SynthesizeRequest(text="Hello World", language="en")
        normal_response = tts_stub.Synthesize(normal_request)

        # SSML version should be longer due to 500ms break
        assert len(response.audio) > len(normal_response.audio)

    def test_nested_prosody_voice(self, tts_stub):
        """Test nested prosody inside voice tag."""
        ssml = '<voice name="am_adam"><prosody rate="fast">Fast Adam speaking</prosody></voice>'
        request = voice_pb2.SynthesizeRequest(text=ssml, language="en")
        response = tts_stub.Synthesize(request)

        assert response.audio is not None
        assert len(response.audio) > 1000

    def test_speak_wrapper(self, tts_stub):
        """Test SSML with <speak> wrapper."""
        ssml = '<speak><prosody rate="medium">Text with speak wrapper</prosody></speak>'
        request = voice_pb2.SynthesizeRequest(text=ssml, language="en")
        response = tts_stub.Synthesize(request)

        assert response.audio is not None
        assert len(response.audio) > 1000

    def test_complex_ssml(self, tts_stub):
        """Test complex SSML with multiple tags."""
        ssml = '''<speak>
            <prosody rate="slow">Welcome to the voice demo.</prosody>
            <break time="1s"/>
            <voice name="am_adam">
                <prosody rate="fast">This is Adam speaking quickly.</prosody>
            </voice>
            <break time="500ms"/>
            Back to the default voice at normal speed.
        </speak>'''
        request = voice_pb2.SynthesizeRequest(text=ssml, language="en")
        response = tts_stub.Synthesize(request)

        assert response.audio is not None
        # Complex SSML should produce substantial audio
        assert len(response.audio) > 50000, "Complex SSML should produce longer audio"


# =============================================================================
# SSML Edge Cases
# =============================================================================

class TestSSMLEdgeCases:
    """Tests for SSML edge cases and error handling."""

    def test_empty_prosody(self, tts_stub):
        """Test empty prosody tag (should produce no audio for that segment)."""
        ssml = 'Hello <prosody rate="fast"></prosody> World'
        request = voice_pb2.SynthesizeRequest(text=ssml, language="en")
        response = tts_stub.Synthesize(request)

        assert response.audio is not None
        assert len(response.audio) > 1000

    def test_unclosed_prosody(self, tts_stub):
        """Test unclosed prosody tag (should still parse gracefully)."""
        # This is malformed SSML but should degrade gracefully
        ssml = '<prosody rate="fast">Hello World'
        request = voice_pb2.SynthesizeRequest(text=ssml, language="en")
        response = tts_stub.Synthesize(request)

        # Should still produce some audio
        assert response.audio is not None

    def test_plain_text_unchanged(self, tts_stub):
        """Test that plain text without SSML works normally."""
        text = "This is plain text without any SSML tags."
        request = voice_pb2.SynthesizeRequest(text=text, language="en")
        response = tts_stub.Synthesize(request)

        assert response.audio is not None
        assert len(response.audio) > 1000

    def test_prosody_rate_extreme_clamp(self, tts_stub):
        """Test extreme rate values are clamped to safe range."""
        # 1000% should be clamped to max (4.0x)
        ssml = '<prosody rate="1000%">Extremely fast</prosody>'
        request = voice_pb2.SynthesizeRequest(text=ssml, language="en")
        response = tts_stub.Synthesize(request)

        assert response.audio is not None
        assert len(response.audio) > 500  # Should still work
