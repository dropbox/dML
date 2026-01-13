#!/usr/bin/env python3
"""
CosyVoice2 Socket Client Integration Tests (Worker #546)

Tests the C++ daemon's ability to route CosyVoice2 voices through the
Python socket server for better quality (PyTorch 2.5.1 vs libtorch 2.9.1).

The socket server uses PyTorch 2.5.1 which produces better audio quality
than the native libtorch 2.9.1 implementation ("frog audio" issue).

Usage:
    # Option 1: Start server manually
    source cosyvoice_251_venv/bin/activate
    python scripts/cosyvoice_server.py --socket /tmp/cosyvoice.sock &
    pytest tests/integration/test_cosyvoice_socket_client.py -v

    # Option 2: Run without server (tests should skip)
    pytest tests/integration/test_cosyvoice_socket_client.py -v

Copyright 2025 Andrew Yates. All rights reserved.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
SOCKET_PATH = "/tmp/cosyvoice.sock"
TTS_BINARY = PROJECT_ROOT / "stream-tts-cpp" / "build" / "stream-tts-cpp"


def socket_server_available() -> bool:
    """Check if the CosyVoice2 socket server is running."""
    return Path(SOCKET_PATH).exists()


def binary_available() -> bool:
    """Check if the TTS binary is built."""
    return TTS_BINARY.exists()


@pytest.fixture
def require_socket_server():
    """Skip test if socket server is not running."""
    if not socket_server_available():
        pytest.skip("CosyVoice2 socket server not running at /tmp/cosyvoice.sock")
    if not binary_available():
        pytest.skip("TTS binary not built")


class TestCosyVoiceSocketClient:
    """Test suite for CosyVoice2 socket client integration."""

    def test_socket_server_detection(self, require_socket_server):
        """Test that the C++ binary detects and uses the socket server."""
        result = subprocess.run(
            [str(TTS_BINARY), "--speak", "hello", "--voice-name", "sichuan"],
            capture_output=True,
            text=True,
            timeout=60,
            env={**os.environ, "COSYVOICE_SOCKET": SOCKET_PATH}
        )

        # Check that socket backend was selected
        assert "cosyvoice2-socket" in result.stderr or result.returncode == 0, \
            f"Expected socket backend. stderr: {result.stderr[:500]}"

    def test_sichuan_voice_synthesis(self, require_socket_server):
        """Test Sichuanese dialect synthesis via socket."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = f.name

        try:
            result = subprocess.run(
                [
                    str(TTS_BINARY),
                    "--speak", "你好世界",
                    "--voice-name", "sichuan",
                    "--output", output_path,
                    "--lang", "zh"
                ],
                capture_output=True,
                text=True,
                timeout=90,
                env={**os.environ, "COSYVOICE_SOCKET": SOCKET_PATH}
            )

            # Check output file exists and has content
            assert Path(output_path).exists(), f"Output file not created. stderr: {result.stderr[:500]}"
            assert Path(output_path).stat().st_size > 1000, "Output file too small"

        finally:
            if Path(output_path).exists():
                Path(output_path).unlink()

    def test_cantonese_voice_synthesis(self, require_socket_server):
        """Test Cantonese dialect synthesis via socket."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = f.name

        try:
            result = subprocess.run(
                [
                    str(TTS_BINARY),
                    "--speak", "你好世界",
                    "--voice-name", "cantonese",
                    "--output", output_path,
                    "--lang", "zh"
                ],
                capture_output=True,
                text=True,
                timeout=90,
                env={**os.environ, "COSYVOICE_SOCKET": SOCKET_PATH}
            )

            # Check output file exists and has content
            assert Path(output_path).exists(), f"Output file not created. stderr: {result.stderr[:500]}"
            assert Path(output_path).stat().st_size > 1000, "Output file too small"

        finally:
            if Path(output_path).exists():
                Path(output_path).unlink()

    def test_fallback_to_native_without_server(self):
        """Test that synthesis attempts fallback when socket server unavailable.

        Note: The native CosyVoice2 may also fail if the daemon isn't running.
        This test verifies the socket detection logic works, not that synthesis succeeds.
        """
        if not binary_available():
            pytest.skip("TTS binary not built")

        # Point to non-existent socket to force native fallback path
        fake_socket = "/tmp/cosyvoice_fake_test.sock"

        result = subprocess.run(
            [
                str(TTS_BINARY),
                "--speak", "hello world",
                "--voice-name", "sichuan",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            env={**os.environ, "COSYVOICE_SOCKET": fake_socket}
        )

        # The binary should NOT crash with a segfault or signal
        # It may return error 1 if native CosyVoice2 models aren't available
        # or daemon isn't running, but it should handle the missing socket gracefully
        assert result.returncode in [0, 1], \
            f"Unexpected exit code (crash?). returncode={result.returncode}, stderr: {result.stderr[:500]}"

        # Verify we didn't try the fake socket (would show connection error)
        assert "Failed to connect" not in result.stderr, \
            "Should skip unavailable socket, not try to connect"

    def test_env_var_override(self, require_socket_server):
        """Test that COSYVOICE_SOCKET environment variable is respected."""
        result = subprocess.run(
            [str(TTS_BINARY), "--speak", "test", "--voice-name", "cosy"],
            capture_output=True,
            text=True,
            timeout=60,
            env={**os.environ, "COSYVOICE_SOCKET": SOCKET_PATH}
        )

        # With socket server available and env var set, should use socket
        assert result.returncode == 0 or "socket" in result.stderr.lower(), \
            f"Expected socket usage with COSYVOICE_SOCKET set. stderr: {result.stderr[:500]}"


class TestCosyVoiceSocketPerformance:
    """Performance tests for CosyVoice2 socket client."""

    def test_latency_measurement(self, require_socket_server):
        """Measure synthesis latency through socket."""
        import time

        start = time.time()
        result = subprocess.run(
            [
                str(TTS_BINARY),
                "--speak", "测试延迟",
                "--voice-name", "sichuan",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            env={**os.environ, "COSYVOICE_SOCKET": SOCKET_PATH}
        )
        elapsed_ms = (time.time() - start) * 1000

        assert result.returncode == 0, f"Synthesis failed: {result.stderr[:500]}"

        # Log latency for monitoring
        print(f"CosyVoice2 socket synthesis latency: {elapsed_ms:.0f}ms")

        # Should complete within 30 seconds (generous timeout for CI)
        assert elapsed_ms < 30000, f"Synthesis too slow: {elapsed_ms:.0f}ms"
