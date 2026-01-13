"""
WebSocket Server Integration Tests

Worker #177 - WebSocket TTS Streaming

Tests the WebSocket server endpoints:
- ws://host:port/       - WebSocket TTS (send JSON text, receive binary audio)

Protocol:
- Client sends JSON: {"text": "Hello", "translate": false}
- Server streams binary audio chunks (WAV format)
- Server sends JSON status: {"status": "done", "synthesis_ms": 123}

Copyright 2025 Andrew Yates. All rights reserved.
"""

import asyncio
import base64
import json
import os
import pytest
import signal
import subprocess
import struct
import time
from pathlib import Path

# Try to import websockets library
try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
STREAM_TTS_CPP = PROJECT_ROOT / "stream-tts-cpp"
BUILD_DIR = STREAM_TTS_CPP / "build"
CONFIG_DIR = STREAM_TTS_CPP / "config"
MODELS_DIR = PROJECT_ROOT / "models"

# WebSocket server defaults
DEFAULT_WS_HOST = "127.0.0.1"
DEFAULT_WS_PORT = 0  # Use ephemeral port for tests


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
    """Find a free port for the WebSocket server."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def parse_wav_header(data: bytes) -> dict:
    """Parse WAV header and return info dict."""
    if len(data) < 44:
        return None

    # Check RIFF header
    if data[:4] != b'RIFF' or data[8:12] != b'WAVE':
        return None

    # Parse fmt chunk
    fmt_offset = 12
    if data[fmt_offset:fmt_offset+4] != b'fmt ':
        return None

    chunk_size = struct.unpack('<I', data[fmt_offset+4:fmt_offset+8])[0]
    audio_format = struct.unpack('<H', data[fmt_offset+8:fmt_offset+10])[0]
    channels = struct.unpack('<H', data[fmt_offset+10:fmt_offset+12])[0]
    sample_rate = struct.unpack('<I', data[fmt_offset+12:fmt_offset+16])[0]
    byte_rate = struct.unpack('<I', data[fmt_offset+16:fmt_offset+20])[0]
    block_align = struct.unpack('<H', data[fmt_offset+20:fmt_offset+22])[0]
    bits_per_sample = struct.unpack('<H', data[fmt_offset+22:fmt_offset+24])[0]

    # Find data chunk
    data_offset = fmt_offset + 8 + chunk_size
    while data_offset < len(data) - 8:
        chunk_id = data[data_offset:data_offset+4]
        chunk_size = struct.unpack('<I', data[data_offset+4:data_offset+8])[0]
        if chunk_id == b'data':
            return {
                'audio_format': audio_format,
                'channels': channels,
                'sample_rate': sample_rate,
                'byte_rate': byte_rate,
                'block_align': block_align,
                'bits_per_sample': bits_per_sample,
                'data_size': chunk_size,
                'data_offset': data_offset + 8
            }
        data_offset += 8 + chunk_size

    return None


@pytest.fixture(scope="module")
def websocket_server(tts_binary, english_config):
    """Start WebSocket server and return ws:// URL.

    Uses ephemeral port to avoid conflicts.
    """
    if not WEBSOCKETS_AVAILABLE:
        pytest.skip("websockets library not installed (pip install websockets)")

    port = find_free_port()

    cmd = [
        str(tts_binary),
        "--websocket",
        "--ws-host", DEFAULT_WS_HOST,
        "--ws-port", str(port),
        str(english_config)
    ]

    env = get_tts_env()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env
    )

    # Wait for server to start
    max_wait = 30
    url = f"ws://{DEFAULT_WS_HOST}:{port}"
    server_ready = False

    for _ in range(max_wait * 10):
        # Check if process died
        if proc.poll() is not None:
            stdout, stderr = proc.communicate()
            pytest.fail(f"WebSocket server process died unexpectedly.\n"
                       f"stdout: {stdout.decode()}\nstderr: {stderr.decode()}")

        # Try to connect
        try:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.1)
                result = s.connect_ex((DEFAULT_WS_HOST, port))
                if result == 0:
                    server_ready = True
                    break
        except Exception:
            pass

        time.sleep(0.1)

    if not server_ready:
        proc.terminate()
        proc.wait(timeout=5)
        pytest.skip(f"WebSocket server failed to start on {url}")

    # Give server a moment to finish initializing
    time.sleep(0.5)

    yield url

    # Cleanup
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


# =============================================================================
# Test Classes
# =============================================================================

@pytest.mark.skipif(not WEBSOCKETS_AVAILABLE, reason="websockets library not installed")
class TestWebSocketConnection:
    """Test WebSocket connection and handshake."""

    @pytest.mark.asyncio
    async def test_websocket_connect(self, websocket_server):
        """Test basic WebSocket connection."""
        async with websockets.connect(websocket_server, close_timeout=5) as ws:
            # Connection is open if we reach this point without exception
            # websockets 15.x removed the .open attribute
            pass

    @pytest.mark.asyncio
    async def test_websocket_ping_pong(self, websocket_server):
        """Test WebSocket ping/pong."""
        async with websockets.connect(websocket_server, close_timeout=5) as ws:
            pong_waiter = await ws.ping()
            await asyncio.wait_for(pong_waiter, timeout=5)


@pytest.mark.skipif(not WEBSOCKETS_AVAILABLE, reason="websockets library not installed")
class TestWebSocketTTS:
    """Test WebSocket TTS synthesis."""

    @pytest.mark.asyncio
    async def test_simple_text_synthesis(self, websocket_server):
        """Test synthesizing simple text."""
        async with websockets.connect(websocket_server, close_timeout=30) as ws:
            # Send TTS request
            request = {"text": "Hello world"}
            await ws.send(json.dumps(request))

            # Collect all messages
            messages = []
            audio_data = bytearray()

            async for msg in ws:
                if isinstance(msg, bytes):
                    audio_data.extend(msg)
                else:
                    messages.append(json.loads(msg))
                    # Check for completion
                    if messages[-1].get("status") == "done":
                        break

            # Verify we got processing and done status
            statuses = [m.get("status") for m in messages]
            assert "processing" in statuses
            assert "done" in statuses

            # Verify we got audio data
            assert len(audio_data) > 44, "Should have received audio data"

            # Verify WAV format
            wav_info = parse_wav_header(bytes(audio_data))
            assert wav_info is not None, "Invalid WAV header"
            assert wav_info['sample_rate'] == 24000
            assert wav_info['channels'] == 1

    @pytest.mark.asyncio
    async def test_synthesis_timing_info(self, websocket_server):
        """Test synthesis returns timing info."""
        async with websockets.connect(websocket_server, close_timeout=30) as ws:
            request = {"text": "Testing timing info"}
            await ws.send(json.dumps(request))

            done_msg = None
            async for msg in ws:
                if isinstance(msg, str):
                    data = json.loads(msg)
                    if data.get("status") == "done":
                        done_msg = data
                        break

            assert done_msg is not None
            assert "synthesis_ms" in done_msg
            assert done_msg["synthesis_ms"] > 0
            assert "audio_size" in done_msg
            assert done_msg["audio_size"] > 0

    @pytest.mark.asyncio
    async def test_multiple_requests(self, websocket_server):
        """Test multiple sequential requests on same connection."""
        async with websockets.connect(websocket_server, close_timeout=60) as ws:
            texts = ["First message", "Second message"]

            for text in texts:
                request = {"text": text}
                await ws.send(json.dumps(request))

                # Wait for completion
                audio_size = 0
                async for msg in ws:
                    if isinstance(msg, bytes):
                        audio_size += len(msg)
                    else:
                        data = json.loads(msg)
                        if data.get("status") == "done":
                            break

                assert audio_size > 44, f"Should have audio for '{text}'"


@pytest.mark.skipif(not WEBSOCKETS_AVAILABLE, reason="websockets library not installed")
class TestWebSocketErrorHandling:
    """Test WebSocket error handling."""

    @pytest.mark.asyncio
    async def test_empty_text_returns_error(self, websocket_server):
        """Test that empty text returns error."""
        async with websockets.connect(websocket_server, close_timeout=10) as ws:
            request = {"text": ""}
            await ws.send(json.dumps(request))

            response = await ws.recv()
            data = json.loads(response)
            assert "error" in data

    @pytest.mark.asyncio
    async def test_missing_text_returns_error(self, websocket_server):
        """Test that missing text field returns error."""
        async with websockets.connect(websocket_server, close_timeout=10) as ws:
            request = {"translate": False}  # Missing 'text' field
            await ws.send(json.dumps(request))

            response = await ws.recv()
            data = json.loads(response)
            assert "error" in data

    @pytest.mark.asyncio
    async def test_invalid_json_returns_error(self, websocket_server):
        """Test that invalid JSON returns error."""
        async with websockets.connect(websocket_server, close_timeout=10) as ws:
            await ws.send("not valid json {{{")

            response = await ws.recv()
            data = json.loads(response)
            assert "error" in data

    @pytest.mark.asyncio
    async def test_binary_message_returns_error(self, websocket_server):
        """Test that binary messages are rejected."""
        async with websockets.connect(websocket_server, close_timeout=10) as ws:
            await ws.send(b'\x00\x01\x02\x03')

            response = await ws.recv()
            data = json.loads(response)
            assert "error" in data


@pytest.mark.skipif(not WEBSOCKETS_AVAILABLE, reason="websockets library not installed")
class TestWebSocketAudioQuality:
    """Test WebSocket audio quality."""

    @pytest.mark.asyncio
    async def test_audio_has_valid_pcm_data(self, websocket_server):
        """Test that audio contains valid PCM data."""
        async with websockets.connect(websocket_server, close_timeout=30) as ws:
            request = {"text": "Hello there"}
            await ws.send(json.dumps(request))

            audio_data = bytearray()
            async for msg in ws:
                if isinstance(msg, bytes):
                    audio_data.extend(msg)
                else:
                    data = json.loads(msg)
                    if data.get("status") == "done":
                        break

            # Parse WAV
            wav_info = parse_wav_header(bytes(audio_data))
            assert wav_info is not None

            # Extract PCM data
            pcm_start = wav_info['data_offset']
            pcm_data = audio_data[pcm_start:]

            # Verify PCM data is not all zeros
            assert len(pcm_data) > 0
            non_zero = sum(1 for b in pcm_data if b != 0)
            assert non_zero > len(pcm_data) * 0.01, "Audio should not be mostly silence"

    @pytest.mark.asyncio
    async def test_longer_text_produces_more_audio(self, websocket_server):
        """Test that longer text produces proportionally more audio."""
        async with websockets.connect(websocket_server, close_timeout=60) as ws:
            short_text = "Hi"
            long_text = "This is a much longer sentence that should produce significantly more audio output than the short text."

            audio_sizes = []
            for text in [short_text, long_text]:
                request = {"text": text}
                await ws.send(json.dumps(request))

                audio_size = 0
                async for msg in ws:
                    if isinstance(msg, bytes):
                        audio_size += len(msg)
                    else:
                        data = json.loads(msg)
                        if data.get("status") == "done":
                            break

                audio_sizes.append(audio_size)

            # Long text should produce more audio
            assert audio_sizes[1] > audio_sizes[0], \
                f"Long text ({audio_sizes[1]} bytes) should produce more audio than short ({audio_sizes[0]} bytes)"


@pytest.mark.skipif(not WEBSOCKETS_AVAILABLE, reason="websockets library not installed")
class TestWebSocketGracefulClose:
    """Test WebSocket graceful close."""

    @pytest.mark.asyncio
    async def test_client_close(self, websocket_server):
        """Test client-initiated close."""
        ws = await websockets.connect(websocket_server, close_timeout=5)
        # Connection established successfully (no exception)
        await ws.close()
        # websockets 15.x: check state instead of .closed attribute
        from websockets.protocol import State
        assert ws.state == State.CLOSED

    @pytest.mark.asyncio
    async def test_close_during_synthesis(self, websocket_server):
        """Test closing connection during synthesis doesn't crash server."""
        ws = await websockets.connect(websocket_server, close_timeout=5)

        # Send request
        request = {"text": "This is a longer text that will take some time to synthesize so we can close during processing."}
        await ws.send(json.dumps(request))

        # Wait briefly then close
        await asyncio.sleep(0.1)
        await ws.close()

        # Server should still accept new connections
        async with websockets.connect(websocket_server, close_timeout=10) as new_ws:
            # Connection is open if we reach this point without exception
            pass
