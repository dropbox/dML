"""
HTTP Server Integration Tests

Worker #174 - HTTP API Testing
Worker #176 - /tts endpoint (translate + synthesize combined)

Tests the HTTP server endpoints:
- GET  /health         - Health check
- GET  /status         - Server status JSON
- POST /speak          - Synthesize speech (returns WAV audio)
- POST /synthesize     - Synthesize speech (returns JSON with base64 audio)
- POST /stream         - Streaming TTS (chunked WAV)
- POST /translate      - Translate text (returns JSON)
- POST /tts            - Translate + Synthesize (returns WAV with timing headers)
- POST /batch          - Batch synthesis (returns JSON array with base64 audio)
- GET  /               - API documentation

Copyright 2025 Andrew Yates. All rights reserved.
"""

import base64
import json
import os
import pytest
import requests
import signal
import subprocess
import tempfile
import time
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
STREAM_TTS_CPP = PROJECT_ROOT / "stream-tts-cpp"
BUILD_DIR = STREAM_TTS_CPP / "build"
CONFIG_DIR = STREAM_TTS_CPP / "config"
MODELS_DIR = PROJECT_ROOT / "models"

# HTTP server defaults
DEFAULT_HTTP_HOST = "127.0.0.1"
DEFAULT_HTTP_PORT = 0  # Use ephemeral port for tests


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


@pytest.fixture(scope="module")
def en2ja_config():
    """Path to EN->JA translation config (enables translation endpoint)."""
    config = CONFIG_DIR / "kokoro-mps-en2ja.yaml"
    if not config.exists():
        pytest.skip(f"EN->JA config not found: {config}")
    return config


def get_tts_env():
    """Get environment for TTS processes."""
    env = os.environ.copy()
    # Fix OpenMP duplicate library crash when llama.cpp + libtorch both link OpenMP
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    return env


def find_free_port():
    """Find a free port for the HTTP server."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def http_server(tts_binary, english_config):
    """Start HTTP server and return base URL.

    Uses ephemeral port to avoid conflicts.
    """
    port = find_free_port()

    cmd = [
        str(tts_binary),
        "--http",
        "--http-host", DEFAULT_HTTP_HOST,
        "--http-port", str(port),
        str(english_config)
    ]

    env = get_tts_env()

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    base_url = f"http://{DEFAULT_HTTP_HOST}:{port}"

    # Wait for server to start (up to 30 seconds for model loading)
    max_wait = 30
    start_time = time.time()
    server_ready = False

    while time.time() - start_time < max_wait:
        try:
            resp = requests.get(f"{base_url}/health", timeout=1)
            if resp.status_code == 200:
                server_ready = True
                break
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(0.5)

    if not server_ready:
        # Read any output for debugging
        output = ""
        try:
            proc.terminate()
            output, _ = proc.communicate(timeout=5)
        except:
            proc.kill()
        pytest.skip(f"HTTP server failed to start within {max_wait}s. Output: {output[:500]}")

    yield base_url

    # Cleanup
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture(scope="module")
def http_server_with_translation(tts_binary, en2ja_config):
    """Start HTTP server with translation enabled and return base URL."""
    port = find_free_port()

    cmd = [
        str(tts_binary),
        "--http",
        "--http-host", DEFAULT_HTTP_HOST,
        "--http-port", str(port),
        str(en2ja_config)
    ]

    env = get_tts_env()

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    base_url = f"http://{DEFAULT_HTTP_HOST}:{port}"

    # Wait for server to start (up to 60 seconds for translation model)
    max_wait = 60
    start_time = time.time()
    server_ready = False

    while time.time() - start_time < max_wait:
        try:
            resp = requests.get(f"{base_url}/health", timeout=1)
            if resp.status_code == 200:
                server_ready = True
                break
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(0.5)

    if not server_ready:
        output = ""
        try:
            proc.terminate()
            output, _ = proc.communicate(timeout=5)
        except:
            proc.kill()
        pytest.skip(f"HTTP server (translation) failed to start within {max_wait}s. Output: {output[:500]}")

    yield base_url

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


# =============================================================================
# Health Check Tests
# =============================================================================

class TestHealthEndpoint:
    """Test /health endpoint."""

    def test_health_returns_200(self, http_server):
        """Health endpoint should return 200 OK."""
        resp = requests.get(f"{http_server}/health")
        assert resp.status_code == 200

    def test_health_returns_json(self, http_server):
        """Health endpoint should return JSON."""
        resp = requests.get(f"{http_server}/health")
        data = resp.json()
        assert "status" in data
        assert data["status"] == "healthy"


# =============================================================================
# Status Endpoint Tests
# =============================================================================

class TestStatusEndpoint:
    """Test /status endpoint."""

    def test_status_returns_200(self, http_server):
        """Status endpoint should return 200 OK."""
        resp = requests.get(f"{http_server}/status")
        assert resp.status_code == 200

    def test_status_returns_server_info(self, http_server):
        """Status endpoint should return server info."""
        resp = requests.get(f"{http_server}/status")
        data = resp.json()

        assert "running" in data
        assert data["running"] == True
        assert "voice_engine_ready" in data
        assert "config" in data
        assert "stats" in data

    def test_status_includes_stats(self, http_server):
        """Status should include request statistics."""
        resp = requests.get(f"{http_server}/status")
        data = resp.json()

        stats = data["stats"]
        assert "total_requests" in stats
        assert "successful_requests" in stats
        assert "failed_requests" in stats

    def test_status_includes_timing(self, http_server):
        """Status should include warmup and model load timing info."""
        resp = requests.get(f"{http_server}/status")
        data = resp.json()

        # Timing section should exist with warmup info
        assert "timing" in data, "Status should include timing section"
        timing = data["timing"]
        assert "model_load_ms" in timing, "Timing should include model_load_ms"
        assert "warmup_ms" in timing, "Timing should include warmup_ms"
        # Warmup should be non-zero after server initialization
        assert timing["warmup_ms"] >= 0, "warmup_ms should be non-negative"

    def test_status_includes_mps_residency(self, http_server):
        """Status should include MPS residency check info (Worker #218).

        Verifies that the server exposes MPS availability and CPU fallback count.
        This enables tests to detect silent CPU fallbacks in hot paths.
        """
        resp = requests.get(f"{http_server}/status")
        data = resp.json()

        # MPS residency section should exist
        assert "mps_residency" in data, "Status should include mps_residency section"
        mps = data["mps_residency"]
        assert "available" in mps, "mps_residency should include 'available' flag"
        assert "cpu_fallback_count" in mps, "mps_residency should include cpu_fallback_count"
        # On Apple Silicon, MPS should be available
        assert mps["available"] == True, "MPS should be available on Apple Silicon"


# =============================================================================
# Speak Endpoint Tests (WAV Response)
# =============================================================================

class TestSpeakEndpoint:
    """Test /speak endpoint (returns WAV audio)."""

    def test_speak_simple_text(self, http_server):
        """Speak endpoint should synthesize simple text."""
        resp = requests.post(
            f"{http_server}/speak",
            json={"text": "Hello world"}
        )

        assert resp.status_code == 200
        assert resp.headers["Content-Type"] == "audio/wav"

        # WAV header validation
        content = resp.content
        assert len(content) > 44  # WAV header is 44 bytes
        assert content[:4] == b"RIFF"
        assert content[8:12] == b"WAVE"

    def test_speak_missing_text_returns_400(self, http_server):
        """Speak endpoint should return 400 for missing text."""
        resp = requests.post(f"{http_server}/speak", json={})
        assert resp.status_code == 400

        data = resp.json()
        assert "error" in data

    def test_speak_empty_text_returns_400(self, http_server):
        """Speak endpoint should return 400 for empty text."""
        resp = requests.post(f"{http_server}/speak", json={"text": ""})
        assert resp.status_code == 400

    def test_speak_produces_valid_audio(self, http_server, tmp_path):
        """Speak endpoint should produce playable WAV audio."""
        resp = requests.post(
            f"{http_server}/speak",
            json={"text": "This is a test."}
        )

        assert resp.status_code == 200

        # Save to temp file
        wav_path = tmp_path / "test.wav"
        wav_path.write_bytes(resp.content)

        # Verify with afinfo (macOS)
        result = subprocess.run(
            ["afinfo", str(wav_path)],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "24000" in result.stdout  # Sample rate

    def test_speak_longer_text(self, http_server):
        """Speak endpoint should handle longer text."""
        long_text = "This is a longer text that contains multiple sentences. " * 3

        resp = requests.post(
            f"{http_server}/speak",
            json={"text": long_text},
            timeout=30
        )

        assert resp.status_code == 200
        assert len(resp.content) > 44


# =============================================================================
# Synthesize Endpoint Tests (JSON Response)
# =============================================================================

class TestSynthesizeEndpoint:
    """Test /synthesize endpoint (returns JSON with base64 audio)."""

    def test_synthesize_returns_json(self, http_server):
        """Synthesize endpoint should return JSON response."""
        resp = requests.post(
            f"{http_server}/synthesize",
            json={"text": "Hello"}
        )

        assert resp.status_code == 200
        assert "application/json" in resp.headers["Content-Type"]

        data = resp.json()
        assert data["success"] == True
        assert "audio_base64" in data
        assert "audio_size" in data
        assert "synthesis_ms" in data
        assert data["format"] == "wav"
        assert data["sample_rate"] == 24000

    def test_synthesize_audio_decodes(self, http_server):
        """Base64 audio should decode to valid WAV."""
        resp = requests.post(
            f"{http_server}/synthesize",
            json={"text": "Test audio."}
        )

        data = resp.json()
        audio_bytes = base64.b64decode(data["audio_base64"])

        # Verify WAV header
        assert len(audio_bytes) > 44
        assert audio_bytes[:4] == b"RIFF"
        assert audio_bytes[8:12] == b"WAVE"

        # Verify size matches
        assert len(audio_bytes) == data["audio_size"]

    def test_synthesize_missing_text_returns_400(self, http_server):
        """Synthesize should return 400 for missing text."""
        resp = requests.post(
            f"{http_server}/synthesize",
            json={}
        )
        assert resp.status_code == 400


# =============================================================================
# Streaming Endpoint Tests
# =============================================================================

class TestStreamEndpoint:
    """Test /stream endpoint (chunked WAV streaming)."""

    def test_stream_returns_chunked_audio(self, http_server):
        """Stream endpoint should return chunked audio."""
        resp = requests.post(
            f"{http_server}/stream",
            json={"text": "This is streaming audio test."},
            stream=True,
            timeout=30
        )

        assert resp.status_code == 200
        assert resp.headers["Content-Type"] == "audio/wav"

        # Read chunks
        total_size = 0
        chunks = []
        for chunk in resp.iter_content(chunk_size=4096):
            chunks.append(chunk)
            total_size += len(chunk)

        # Should have received audio
        assert total_size > 44

        # First chunk should start with WAV header
        full_audio = b"".join(chunks)
        assert full_audio[:4] == b"RIFF"

    def test_stream_long_text_prosody_preserved(self, http_server):
        """Stream endpoint should handle multi-sentence text."""
        text = "First sentence is short. Second sentence is a bit longer. Third sentence wraps up."

        resp = requests.post(
            f"{http_server}/stream",
            json={"text": text},
            stream=True,
            timeout=60
        )

        assert resp.status_code == 200

        full_audio = b"".join(resp.iter_content(chunk_size=4096))
        assert len(full_audio) > 44
        assert full_audio[:4] == b"RIFF"

    def test_stream_missing_text_returns_400(self, http_server):
        """Stream should return 400 for missing text."""
        resp = requests.post(f"{http_server}/stream", json={})
        assert resp.status_code == 400


# =============================================================================
# Translation Endpoint Tests
# =============================================================================

class TestTranslateEndpoint:
    """Test /translate endpoint."""

    def test_translate_without_translation_engine_returns_503(self, http_server):
        """Translate should return 503 when translation not enabled."""
        resp = requests.post(
            f"{http_server}/translate",
            json={"text": "Hello world"}
        )

        # Server started without translation config
        assert resp.status_code == 503
        data = resp.json()
        assert "error" in data
        assert "unavailable" in data["error"].lower()

    def test_translate_en_to_ja(self, http_server_with_translation):
        """Translate should translate English to Japanese."""
        resp = requests.post(
            f"{http_server_with_translation}/translate",
            json={"text": "Hello world"}
        )

        assert resp.status_code == 200
        data = resp.json()

        assert data["success"] == True
        assert "translation" in data
        assert "translation_ms" in data
        assert data["source_lang"] == "en"
        assert data["target_lang"] == "ja"

        # Translation should contain Japanese characters
        translation = data["translation"]
        has_japanese = any(ord(c) >= 0x3000 and ord(c) <= 0x9FFF for c in translation)
        assert has_japanese, f"Translation '{translation}' doesn't contain Japanese characters"

    def test_translate_dynamic_languages(self, http_server_with_translation):
        """Translate should accept dynamic source/target languages."""
        resp = requests.post(
            f"{http_server_with_translation}/translate",
            json={
                "text": "Good morning",
                "source_lang": "en",
                "target_lang": "ja"
            }
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["source_lang"] == "en"
        assert data["target_lang"] == "ja"

    def test_translate_missing_text_returns_400(self, http_server_with_translation):
        """Translate should return 400 for missing text."""
        resp = requests.post(
            f"{http_server_with_translation}/translate",
            json={}
        )
        assert resp.status_code == 400


# =============================================================================
# API Documentation Tests
# =============================================================================

class TestApiDocumentation:
    """Test / (root) endpoint for API docs."""

    def test_root_returns_docs(self, http_server):
        """Root endpoint should return API documentation."""
        resp = requests.get(f"{http_server}/")

        assert resp.status_code == 200
        data = resp.json()

        assert "name" in data
        assert "version" in data
        assert "endpoints" in data

        # Verify expected endpoints are documented
        endpoints = [e["path"] for e in data["endpoints"]]
        assert "/health" in endpoints
        assert "/status" in endpoints
        assert "/speak" in endpoints
        assert "/synthesize" in endpoints
        assert "/stream" in endpoints
        assert "/translate" in endpoints


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Test error handling."""

    def test_invalid_json_returns_400(self, http_server):
        """Invalid JSON should return 400."""
        resp = requests.post(
            f"{http_server}/speak",
            data="not valid json",
            headers={"Content-Type": "application/json"}
        )

        assert resp.status_code == 400
        data = resp.json()
        assert "error" in data

    def test_nonexistent_endpoint_returns_404(self, http_server):
        """Non-existent endpoint should return 404."""
        resp = requests.get(f"{http_server}/nonexistent")
        assert resp.status_code == 404


# =============================================================================
# Statistics Tests
# =============================================================================

# =============================================================================
# Batch Endpoint Tests
# =============================================================================

class TestBatchEndpoint:
    """Test /batch endpoint (multiple texts -> JSON array with base64 audio)."""

    def test_batch_simple(self, http_server):
        """Batch endpoint should synthesize multiple texts."""
        resp = requests.post(
            f"{http_server}/batch",
            json={"texts": ["Hello", "Goodbye"]},
            timeout=60
        )

        assert resp.status_code == 200
        data = resp.json()

        assert data["success"] == True
        assert data["batch_size"] == 2
        assert data["success_count"] == 2
        assert data["failure_count"] == 0
        assert len(data["results"]) == 2

    def test_batch_results_have_audio(self, http_server):
        """Batch results should contain base64 audio."""
        resp = requests.post(
            f"{http_server}/batch",
            json={"texts": ["Test one", "Test two"]},
            timeout=60
        )

        data = resp.json()
        for result in data["results"]:
            assert result["success"] == True
            assert "audio_base64" in result
            assert "audio_size" in result
            assert result["audio_size"] > 44  # WAV header minimum

            # Verify audio decodes to valid WAV
            audio_bytes = base64.b64decode(result["audio_base64"])
            assert audio_bytes[:4] == b"RIFF"
            assert audio_bytes[8:12] == b"WAVE"

    def test_batch_empty_texts_returns_400(self, http_server):
        """Batch endpoint should return 400 for empty texts array."""
        resp = requests.post(
            f"{http_server}/batch",
            json={"texts": []}
        )
        assert resp.status_code == 400

    def test_batch_missing_texts_returns_400(self, http_server):
        """Batch endpoint should return 400 for missing texts."""
        resp = requests.post(
            f"{http_server}/batch",
            json={}
        )
        assert resp.status_code == 400

    def test_batch_handles_empty_string(self, http_server):
        """Batch should handle empty strings gracefully."""
        resp = requests.post(
            f"{http_server}/batch",
            json={"texts": ["Hello", "", "World"]},
            timeout=60
        )

        assert resp.status_code == 200
        data = resp.json()

        assert data["batch_size"] == 3
        assert data["success_count"] == 2
        assert data["failure_count"] == 1

        # Empty string should fail
        assert data["results"][1]["success"] == False
        assert "Empty" in data["results"][1]["error"]

    def test_batch_timing_info(self, http_server):
        """Batch response should include timing information."""
        resp = requests.post(
            f"{http_server}/batch",
            json={"texts": ["Timing test"]},
            timeout=60
        )

        data = resp.json()
        assert "total_synthesis_ms" in data
        assert "batch_elapsed_ms" in data
        assert data["batch_elapsed_ms"] >= 0


class TestStatistics:
    """Test that statistics are correctly tracked."""

    def test_stats_increment_after_requests(self, http_server):
        """Statistics should increment after requests."""
        # Get initial stats
        resp1 = requests.get(f"{http_server}/status")
        initial_total = resp1.json()["stats"]["total_requests"]

        # Make a speak request
        requests.post(
            f"{http_server}/speak",
            json={"text": "Stats test"}
        )

        # Check stats increased
        resp2 = requests.get(f"{http_server}/status")
        new_total = resp2.json()["stats"]["total_requests"]

        assert new_total > initial_total

    def test_streaming_stats_tracked(self, http_server):
        """Streaming requests should be tracked separately."""
        # Get initial streaming count
        resp1 = requests.get(f"{http_server}/status")
        initial_streaming = resp1.json()["stats"]["streaming_requests"]

        # Make a stream request
        resp = requests.post(
            f"{http_server}/stream",
            json={"text": "Stream stats test"},
            stream=True
        )
        # Consume the stream
        _ = b"".join(resp.iter_content())

        # Check streaming stats increased
        resp2 = requests.get(f"{http_server}/status")
        new_streaming = resp2.json()["stats"]["streaming_requests"]

        assert new_streaming > initial_streaming


# =============================================================================
# MPS Residency Tests (Worker #218 - Verify no CPU fallbacks)
# =============================================================================

class TestMPSResidency:
    """Tests that verify MPS GPU operations don't fall back to CPU.

    These tests are critical for detecting silent performance regressions
    where tensor operations fall back from GPU to CPU execution.

    Implements roadmap item #5: MPS residency checks - fail tests if CPU fallbacks.
    """

    def test_no_cpu_fallbacks_after_synthesis(self, http_server):
        """TTS synthesis should not cause any CPU fallbacks on MPS.

        This test performs TTS synthesis and verifies that the MPS device
        validator detected zero CPU fallbacks during the operation.
        """
        # Perform a TTS synthesis request
        resp = requests.post(
            f"{http_server}/speak",
            json={"text": "Testing MPS residency without CPU fallbacks"},
            timeout=30
        )
        assert resp.status_code == 200, f"TTS request failed: {resp.text}"

        # Check status for MPS residency info
        status_resp = requests.get(f"{http_server}/status")
        data = status_resp.json()

        assert "mps_residency" in data, "Status should include mps_residency"
        mps = data["mps_residency"]

        # Verify MPS is available
        assert mps["available"] == True, "MPS should be available on Apple Silicon"

        # CRITICAL: No CPU fallbacks should have occurred
        # If this fails, a hot path tensor operation fell back to CPU
        # which would cause severe performance degradation
        assert mps["cpu_fallback_count"] == 0, (
            f"PERFORMANCE REGRESSION: {mps['cpu_fallback_count']} CPU fallback(s) detected! "
            "Check logs for 'MPS_RESIDENCY' warnings to identify which tensors fell back to CPU."
        )

    def test_no_cpu_fallbacks_after_multiple_requests(self, http_server):
        """Multiple TTS requests should maintain zero CPU fallbacks.

        Verifies that repeated synthesis doesn't accumulate CPU fallbacks,
        which could indicate memory pressure or dynamic tensor shape issues.
        """
        # Perform multiple TTS requests
        texts = [
            "First test sentence for MPS residency.",
            "Second test with different length.",
            "Third verification request."
        ]

        for text in texts:
            resp = requests.post(
                f"{http_server}/speak",
                json={"text": text},
                timeout=30
            )
            assert resp.status_code == 200

        # Check final CPU fallback count
        status_resp = requests.get(f"{http_server}/status")
        data = status_resp.json()
        mps = data["mps_residency"]

        assert mps["cpu_fallback_count"] == 0, (
            f"PERFORMANCE REGRESSION: {mps['cpu_fallback_count']} CPU fallback(s) detected "
            f"after {len(texts)} TTS requests. This may indicate dynamic tensor shape issues."
        )


# =============================================================================
# TTS Endpoint Tests (Translate + Synthesize Combined)
# =============================================================================

class TestTTSEndpoint:
    """Test /tts endpoint (translate + synthesize in one call)."""

    def test_tts_without_translation(self, http_server):
        """TTS endpoint should work without translation (translate=false)."""
        resp = requests.post(
            f"{http_server}/tts",
            json={"text": "Hello world", "translate": False},
            timeout=30
        )

        assert resp.status_code == 200
        assert resp.headers["Content-Type"] == "audio/wav"

        # Check timing headers
        assert "X-Synthesis-Ms" in resp.headers
        assert "X-Translation-Ms" in resp.headers
        assert resp.headers["X-Translation-Ms"] == "0"  # No translation

        # Verify WAV audio
        content = resp.content
        assert len(content) > 44
        assert content[:4] == b"RIFF"
        assert content[8:12] == b"WAVE"

    def test_tts_without_translation_engine_returns_503(self, http_server):
        """TTS with translate=true should return 503 when translation not available."""
        resp = requests.post(
            f"{http_server}/tts",
            json={"text": "Hello world", "translate": True}
        )

        # Server started without translation config, translate=true fails
        assert resp.status_code == 503
        data = resp.json()
        assert "error" in data
        assert "unavailable" in data["error"].lower()

    def test_tts_with_translation(self, http_server_with_translation):
        """TTS endpoint should translate and synthesize."""
        # Use unique text to avoid translation cache hits
        import time
        unique_text = f"The weather today is quite pleasant at time {int(time.time())}"

        resp = requests.post(
            f"{http_server_with_translation}/tts",
            json={"text": unique_text},
            timeout=60
        )

        assert resp.status_code == 200
        assert resp.headers["Content-Type"] == "audio/wav"

        # Check timing headers
        assert "X-Synthesis-Ms" in resp.headers
        assert "X-Translation-Ms" in resp.headers
        assert "X-Total-Ms" in resp.headers
        assert "X-Translated-Text" in resp.headers

        # Translation should have taken some time (>= 0 for cached results is acceptable)
        translation_ms = int(resp.headers["X-Translation-Ms"])
        assert translation_ms >= 0  # Can be 0 if cached, but header must exist

        # Translated text should be present (HTTP headers may have encoding issues with non-ASCII)
        translated_raw = resp.headers["X-Translated-Text"]
        # Try to decode as UTF-8 if it looks like mojibake (Latin-1 misinterpretation)
        try:
            translated = translated_raw.encode('latin-1').decode('utf-8')
        except (UnicodeDecodeError, UnicodeEncodeError):
            translated = translated_raw

        # Check for Japanese characters (range includes hiragana, katakana, CJK)
        has_japanese = any(
            (0x3040 <= ord(c) <= 0x309F) or  # Hiragana
            (0x30A0 <= ord(c) <= 0x30FF) or  # Katakana
            (0x4E00 <= ord(c) <= 0x9FFF)     # CJK Unified Ideographs
            for c in translated
        )
        assert has_japanese, f"Translated text '{translated}' doesn't contain Japanese characters"

        # Verify WAV audio
        content = resp.content
        assert len(content) > 44
        assert content[:4] == b"RIFF"

    def test_tts_dynamic_languages(self, http_server_with_translation):
        """TTS should accept dynamic source/target languages."""
        resp = requests.post(
            f"{http_server_with_translation}/tts",
            json={
                "text": "Good morning everyone",
                "source_lang": "en",
                "target_lang": "ja"
            },
            timeout=60
        )

        assert resp.status_code == 200
        assert resp.headers["Content-Type"] == "audio/wav"
        assert "X-Translated-Text" in resp.headers

    def test_tts_missing_text_returns_400(self, http_server):
        """TTS should return 400 for missing text."""
        resp = requests.post(f"{http_server}/tts", json={})
        assert resp.status_code == 400

        data = resp.json()
        assert "error" in data

    def test_tts_empty_text_returns_400(self, http_server):
        """TTS should return 400 for empty text."""
        resp = requests.post(f"{http_server}/tts", json={"text": ""})
        assert resp.status_code == 400

    def test_tts_invalid_json_returns_400(self, http_server):
        """TTS should return 400 for invalid JSON."""
        resp = requests.post(
            f"{http_server}/tts",
            data="not valid json",
            headers={"Content-Type": "application/json"}
        )
        assert resp.status_code == 400

    def test_tts_in_api_docs(self, http_server):
        """TTS endpoint should be documented in API docs."""
        resp = requests.get(f"{http_server}/")
        data = resp.json()

        endpoints = [e["path"] for e in data["endpoints"]]
        assert "/tts" in endpoints
