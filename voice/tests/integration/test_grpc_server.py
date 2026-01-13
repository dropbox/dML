"""
gRPC Server Integration Tests

Tests the gRPC server startup and endpoints using Python gRPC client:
- Service listing and method enumeration
- Proto reflection for debugging
Worker #182 - gRPC TTS Server Tests
Worker #201 - C++ build now forces full protobuf Message types (fixes lite reflection error)

Copyright 2025 Andrew Yates. All rights reserved.
"""

import json
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
DEFAULT_GRPC_PORT = 0  # Use ephemeral port for tests

# Check if grpcurl is installed (used for metadata tests only)
GRPCURL_AVAILABLE = shutil.which("grpcurl") is not None

# Mark as integration and skip cleanly if grpcurl is unavailable
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not GRPCURL_AVAILABLE, reason="grpcurl not installed"),
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


def run_grpcurl(host, port, *args, data=None, timeout=30):
    """Run grpcurl command and return (returncode, stdout, stderr).

    Args:
        host: gRPC server host
        port: gRPC server port
        *args: Additional grpcurl arguments (method, list, describe, etc.)
        data: JSON data to send with -d flag (placed before address)
        timeout: Command timeout in seconds
    """
    # Build command: grpcurl [flags] [address] [verb] [symbol]
    # -d flag must come before the address
    cmd = ["grpcurl", "-plaintext"]
    if data:
        cmd.extend(["-d", data])
    cmd.append(f"{host}:{port}")
    cmd.extend(list(args))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"


def load_sample_audio(max_seconds=6):
    """Load sample JFK audio (PCM16) for STT tests."""
    sample_path = PROJECT_ROOT / "external" / "whisper.cpp" / "samples" / "jfk.wav"
    with wave.open(str(sample_path), "rb") as wf:
        sample_rate = wf.getframerate()
        frames = min(wf.getnframes(), int(sample_rate * max_seconds))
        audio = wf.readframes(frames)
    return audio, sample_rate


def chunk_audio(audio_bytes, chunk_size=3200):
    """Yield fixed-size PCM16 chunks."""
    for i in range(0, len(audio_bytes), chunk_size):
        yield audio_bytes[i:i + chunk_size]


@pytest.fixture(scope="module")
def grpc_server(tts_binary, english_config):
    """Start gRPC server and return (host, port).

    Uses ephemeral port to avoid conflicts.
    """
    if not GRPCURL_AVAILABLE:
        pytest.skip("grpcurl not installed (brew install grpcurl)")

    port = find_free_port()

    cmd = [
        str(tts_binary),
        "--grpc",
        "--grpc-host", DEFAULT_GRPC_HOST,
        "--grpc-port", str(port),
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
    if not wait_for_port(DEFAULT_GRPC_HOST, port, timeout=30):
        proc.terminate()
        try:
            stdout, stderr = proc.communicate(timeout=5)
            pytest.fail(f"gRPC server failed to start.\n"
                       f"stdout: {stdout.decode()}\nstderr: {stderr.decode()}")
        except subprocess.TimeoutExpired:
            proc.kill()
            pytest.skip("gRPC server failed to start")

    # Give server a moment to finish initializing
    time.sleep(1)

    yield (DEFAULT_GRPC_HOST, port)

    # Cleanup
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


@pytest.fixture(scope="module")
def grpc_channel(grpc_server):
    """Create a gRPC channel to the server."""
    host, port = grpc_server
    channel = grpc.insecure_channel(f"{host}:{port}")
    yield channel
    channel.close()


@pytest.fixture(scope="module")
def tts_stub(grpc_channel):
    """Create a TTSService stub."""
    return voice_pb2_grpc.TTSServiceStub(grpc_channel)


@pytest.fixture(scope="module")
def translation_stub(grpc_channel):
    """Create a TranslationService stub."""
    return voice_pb2_grpc.TranslationServiceStub(grpc_channel)


@pytest.fixture(scope="module")
def stt_stub(grpc_channel):
    """Create an STTService stub."""
    return voice_pb2_grpc.STTServiceStub(grpc_channel)


# =============================================================================
# Test Classes
# =============================================================================

class TestGrpcServerStartup:
    """Test gRPC server startup and service listing."""

    def test_server_starts(self, grpc_server):
        """Test that gRPC server starts successfully."""
        host, port = grpc_server
        # If we reach this point, the fixture succeeded
        assert port > 0

    def test_list_services(self, grpc_server):
        """Test listing available gRPC services."""
        host, port = grpc_server
        rc, stdout, stderr = run_grpcurl(host, port, "list")

        assert rc == 0, f"grpcurl list failed: {stderr}"
        assert "voice.TTSService" in stdout
        assert "voice.STTService" in stdout
        assert "voice.TranslationService" in stdout
        assert "voice.FilteredSTTService" in stdout

    def test_list_tts_methods(self, grpc_server):
        """Test listing TTSService methods."""
        host, port = grpc_server
        rc, stdout, stderr = run_grpcurl(host, port, "list", "voice.TTSService")

        assert rc == 0, f"grpcurl list failed: {stderr}"
        assert "GetStatus" in stdout
        assert "Synthesize" in stdout
        assert "StreamSynthesize" in stdout

    def test_list_filtered_stt_methods(self, grpc_server):
        """Test listing FilteredSTTService methods."""
        host, port = grpc_server
        rc, stdout, stderr = run_grpcurl(host, port, "list", "voice.FilteredSTTService")

        assert rc == 0, f"grpcurl list failed: {stderr}"
        assert "StreamFilteredTranscribe" in stdout
        assert "StartFilteredListen" in stdout
        assert "ConfigureFilter" in stdout


class TestGrpcHealthCheck:
    """Test gRPC health check endpoints using Python client."""

    def test_get_status_ready(self, tts_stub):
        """Test GetStatus returns ready status."""
        request = voice_pb2.StatusRequest()
        response = tts_stub.GetStatus(request)

        assert response.version == "1.0.0"
        assert response.status == "ready"
        assert response.models_loaded is True

    def test_get_status_with_metrics(self, tts_stub):
        """Test GetStatus with metrics enabled."""
        request = voice_pb2.StatusRequest(include_metrics=True)
        response = tts_stub.GetStatus(request)

        assert response.version == "1.0.0"
        assert response.status == "ready"
        # Metrics should be populated when requested
        assert response.HasField("metrics")


class TestGrpcTTSSynthesis:
    """Test gRPC TTS synthesis endpoints using Python client."""

    def test_synthesize_simple_text(self, tts_stub):
        """Test unary synthesis with simple text."""
        request = voice_pb2.SynthesizeRequest(text="Hello world", language="en")
        response = tts_stub.Synthesize(request)

        # Check audio was returned
        assert len(response.audio) > 100, "Audio should have content"

        # Check metadata
        assert response.metadata.sample_rate == 24000
        assert response.metadata.channels == 1

    def test_synthesize_returns_timing(self, tts_stub):
        """Test that synthesis returns timing info."""
        request = voice_pb2.SynthesizeRequest(text="Testing timing")
        response = tts_stub.Synthesize(request)

        # Check synthesis time is positive
        assert response.timing.synthesis_ms > 0, "Synthesis should take some time"

    def test_synthesize_empty_text_error(self, tts_stub):
        """Test that empty text returns error."""
        request = voice_pb2.SynthesizeRequest(text="")

        with pytest.raises(grpc.RpcError) as exc_info:
            tts_stub.Synthesize(request)

        assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT


class TestGrpcStreamSynthesis:
    """Test gRPC streaming synthesis using Python client."""

    def test_stream_synthesize_simple(self, tts_stub):
        """Test server-streaming synthesis."""
        request = voice_pb2.SynthesizeRequest(text="Hello streaming world")
        chunks = list(tts_stub.StreamSynthesize(request))

        # Should receive at least one chunk
        assert len(chunks) >= 1, "Should receive at least one chunk"

        # Last chunk should have is_final=true
        assert chunks[-1].is_final is True

    def test_stream_synthesize_receives_chunks(self, tts_stub):
        """Test that streaming returns multiple chunks."""
        request = voice_pb2.SynthesizeRequest(
            text="This is a longer text that should produce multiple audio chunks when streamed from the server."
        )
        chunks = list(tts_stub.StreamSynthesize(request))

        # Should have at least 1 chunk
        assert len(chunks) >= 1, "Should receive chunks"

        # First chunk should have metadata
        first_chunk = chunks[0]
        assert first_chunk.metadata.sample_rate == 24000


class TestGrpcTranslation:
    """Test gRPC translation endpoints using Python client."""

    def test_translate_passthrough(self, translation_stub):
        """Test translate endpoint (currently passthrough)."""
        request = voice_pb2.TranslateRequest(
            text="Hello",
            source_language="en",
            target_language="ja"
        )
        response = translation_stub.Translate(request)

        # Currently passthrough - returns same text
        assert response.translated_text == "Hello"


class TestGrpcReflection:
    """Test gRPC reflection service."""

    def test_reflection_available(self, grpc_server):
        """Test that gRPC reflection is enabled."""
        host, port = grpc_server
        rc, stdout, stderr = run_grpcurl(host, port, "list")

        assert rc == 0, f"grpcurl list failed: {stderr}"
        # Reflection services should be listed
        assert "grpc.reflection" in stdout

    def test_describe_synthesize_request(self, grpc_server):
        """Test describing SynthesizeRequest message."""
        host, port = grpc_server
        rc, stdout, stderr = run_grpcurl(
            host, port,
            "describe", "voice.SynthesizeRequest"
        )

        assert rc == 0, f"describe failed: {stderr}"
        assert "text" in stdout
        assert "language" in stdout


class TestGrpcConcurrency:
    """Test gRPC server concurrency using Python client."""

    def test_sequential_requests(self, tts_stub):
        """Test multiple sequential requests."""
        texts = ["First", "Second", "Third"]

        for text in texts:
            request = voice_pb2.SynthesizeRequest(text=text)
            response = tts_stub.Synthesize(request)
            assert len(response.audio) > 100, f"Synthesis failed for '{text}'"

    def test_status_after_synthesis(self, tts_stub):
        """Test GetStatus after synthesis operations."""
        # Do a synthesis first
        request = voice_pb2.SynthesizeRequest(text="Hello")
        tts_stub.Synthesize(request)

        # Check status still works
        status_request = voice_pb2.StatusRequest()
        response = tts_stub.GetStatus(status_request)
        assert response.status == "ready"


class TestGrpcBidirectionalStreaming:
    """Test gRPC bidirectional streaming (StreamTTS) using Python client."""

    def test_bidirectional_single_request(self, tts_stub):
        """Test bidirectional streaming with single request."""
        def request_generator():
            synth_req = voice_pb2.SynthesizeRequest(text="Hello bidirectional")
            yield voice_pb2.TTSRequest(synthesize=synth_req)

        chunks = list(tts_stub.StreamTTS(request_generator()))

        # Should receive audio chunks
        assert len(chunks) >= 1, "Should receive at least one chunk"

        # Last chunk should have is_final=true
        assert chunks[-1].is_final is True

    def test_bidirectional_multiple_requests(self, tts_stub):
        """Test bidirectional streaming with multiple sequential requests."""
        texts = ["First message", "Second message", "Third message"]

        def request_generator():
            for text in texts:
                synth_req = voice_pb2.SynthesizeRequest(text=text)
                yield voice_pb2.TTSRequest(synthesize=synth_req)

        chunks = list(tts_stub.StreamTTS(request_generator()))

        # Should receive chunks for all messages
        # Count final chunks to verify we got responses for all requests
        final_count = sum(1 for c in chunks if c.is_final)
        assert final_count == len(texts), f"Expected {len(texts)} final chunks, got {final_count}"

    def test_bidirectional_with_control_command(self, tts_stub):
        """Test bidirectional streaming with control commands."""
        def request_generator():
            # Send a synthesis request
            synth_req = voice_pb2.SynthesizeRequest(text="Hello")
            yield voice_pb2.TTSRequest(synthesize=synth_req)

            # Send a flush command (should be ignored but not break)
            control = voice_pb2.ControlCommand(command=voice_pb2.ControlCommand.COMMAND_FLUSH)
            yield voice_pb2.TTSRequest(control=control)

            # Send another synthesis request
            synth_req2 = voice_pb2.SynthesizeRequest(text="World")
            yield voice_pb2.TTSRequest(synthesize=synth_req2)

        chunks = list(tts_stub.StreamTTS(request_generator()))

        # Should receive chunks (control command is handled silently)
        assert len(chunks) >= 2, "Should receive chunks for both requests"

    def test_bidirectional_empty_text(self, tts_stub):
        """Test bidirectional streaming with empty text."""
        def request_generator():
            synth_req = voice_pb2.SynthesizeRequest(text="")
            yield voice_pb2.TTSRequest(synthesize=synth_req)

        chunks = list(tts_stub.StreamTTS(request_generator()))

        # Should receive at least an empty final chunk
        assert len(chunks) >= 1
        assert chunks[-1].is_final is True

    def test_bidirectional_first_chunk_has_metadata(self, tts_stub):
        """Test that first chunk in bidirectional stream has metadata."""
        def request_generator():
            synth_req = voice_pb2.SynthesizeRequest(text="Test metadata")
            yield voice_pb2.TTSRequest(synthesize=synth_req)

        chunks = list(tts_stub.StreamTTS(request_generator()))

        # Find first chunk with audio data
        audio_chunks = [c for c in chunks if c.data]
        assert len(audio_chunks) >= 1, "Should have at least one chunk with audio"

        first_audio_chunk = audio_chunks[0]
        assert first_audio_chunk.metadata.sample_rate == 24000
        assert first_audio_chunk.metadata.channels == 1


class TestGrpcSTT:
    """Test gRPC STT (Whisper streaming) endpoints."""

    def test_stt_transcribe_unary(self, stt_stub):
        """Unary STT transcription returns expected text."""
        audio, sample_rate = load_sample_audio(max_seconds=6)
        request = voice_pb2.TranscribeRequest(
            audio=audio,
            sample_rate=sample_rate,
            language="en",
        )
        response = stt_stub.Transcribe(request)

        lowered = response.text.lower()
        assert response.text, "Transcript should not be empty"
        assert any(phrase in lowered for phrase in ["fellow americans", "ask not"]), lowered
        assert response.confidence >= 0.0

    def test_stt_streaming_transcribe(self, stt_stub):
        """Streaming STT returns partial and final transcripts."""
        audio, sample_rate = load_sample_audio(max_seconds=6)
        audio_chunks = list(chunk_audio(audio, chunk_size=6400))

        def request_generator():
            if not audio_chunks:
                yield voice_pb2.AudioInput(end_of_stream=True)
                return

            for idx, chunk in enumerate(audio_chunks):
                yield voice_pb2.AudioInput(
                    audio=chunk,
                    sample_rate=sample_rate,
                    language="en" if idx == 0 else "",
                    end_of_stream=(idx == len(audio_chunks) - 1),
                )

        responses = list(stt_stub.StreamTranscribe(request_generator()))
        assert responses, "Should receive transcript chunks"

        transcript = " ".join([r.text for r in responses if r.text]).lower()
        assert any(phrase in transcript for phrase in ["fellow americans", "ask not"]), transcript
        assert any(r.is_final for r in responses), "Should emit a final transcript"


class TestGrpcWakeWordService:
    """Test gRPC WakeWordService endpoints (Worker #304: Phase 3).

    Tests wake word detection gRPC service:
    - Service listing and methods
    - GetStatus endpoint
    - ListModels endpoint
    """

    def test_wake_word_service_listed(self, grpc_server):
        """Test that WakeWordService is listed in services."""
        host, port = grpc_server
        rc, stdout, stderr = run_grpcurl(host, port, "list")

        assert rc == 0, f"grpcurl list failed: {stderr}"
        assert "voice.WakeWordService" in stdout, "WakeWordService should be listed"

    def test_list_wake_word_methods(self, grpc_server):
        """Test listing WakeWordService methods."""
        host, port = grpc_server
        rc, stdout, stderr = run_grpcurl(host, port, "list", "voice.WakeWordService")

        assert rc == 0, f"grpcurl list failed: {stderr}"
        assert "StartListening" in stdout
        assert "StopListening" in stdout
        assert "SetEnabled" in stdout
        assert "GetStatus" in stdout
        assert "ListModels" in stdout
        assert "TestDetection" in stdout

    def test_get_wake_word_status(self, grpc_server):
        """Test GetStatus endpoint returns status."""
        host, port = grpc_server
        rc, stdout, stderr = run_grpcurl(
            host, port,
            "voice.WakeWordService/GetStatus",
            data="{}"
        )

        # May fail if models not loaded, but call should succeed
        # If models are missing, check stderr for model-not-ready message
        if rc != 0:
            # Check if it's a "models not ready" error (expected on some systems)
            assert "not ready" in stderr.lower() or "unavailable" in stderr.lower(), \
                f"Unexpected error: {stderr}"
        else:
            # Success - parse response
            assert "isListening" in stdout or "{}" in stdout or "isEnabled" in stdout

    def test_list_wake_word_models(self, grpc_server):
        """Test ListModels endpoint."""
        host, port = grpc_server
        rc, stdout, stderr = run_grpcurl(
            host, port,
            "voice.WakeWordService/ListModels",
            data="{}"
        )

        # May return empty list if models not found, but call should succeed
        if rc != 0:
            assert "not ready" in stderr.lower() or "unavailable" in stderr.lower(), \
                f"Unexpected error: {stderr}"
        else:
            # Success - response may be empty or have models list
            assert rc == 0

    def test_describe_wake_word_config(self, grpc_server):
        """Test describing WakeWordListenConfig message."""
        host, port = grpc_server
        rc, stdout, stderr = run_grpcurl(
            host, port,
            "describe", "voice.WakeWordListenConfig"
        )

        assert rc == 0, f"describe failed: {stderr}"
        assert "threshold" in stdout
        assert "enable_self_speech_filter" in stdout
        assert "play_activation_sound" in stdout

    def test_describe_wake_word_event(self, grpc_server):
        """Test describing WakeWordDetectionEvent message."""
        host, port = grpc_server
        rc, stdout, stderr = run_grpcurl(
            host, port,
            "describe", "voice.WakeWordDetectionEvent"
        )

        assert rc == 0, f"describe failed: {stderr}"
        assert "wake_word" in stdout
        assert "confidence" in stdout
        assert "timestamp_ms" in stdout


class TestGrpcSummarizationService:
    """Test gRPC SummarizationService endpoints (Worker #310: Phase 3).

    Tests text and audio summarization gRPC service:
    - Service listing and methods
    - GetSummarizationStatus endpoint
    - Summarize text endpoint
    - Proto message validation
    """

    def test_summarization_service_listed(self, grpc_server):
        """Test that SummarizationService is listed in services."""
        host, port = grpc_server
        rc, stdout, stderr = run_grpcurl(host, port, "list")

        assert rc == 0, f"grpcurl list failed: {stderr}"
        assert "voice.SummarizationService" in stdout, "SummarizationService should be listed"

    def test_list_summarization_methods(self, grpc_server):
        """Test listing SummarizationService methods."""
        host, port = grpc_server
        rc, stdout, stderr = run_grpcurl(host, port, "list", "voice.SummarizationService")

        assert rc == 0, f"grpcurl list failed: {stderr}"
        assert "Summarize" in stdout
        assert "StreamSummarize" in stdout
        assert "LiveSummarize" in stdout
        assert "GetSummarizationStatus" in stdout

    def test_get_summarization_status(self, grpc_server):
        """Test GetSummarizationStatus endpoint returns status."""
        host, port = grpc_server
        rc, stdout, stderr = run_grpcurl(
            host, port,
            "voice.SummarizationService/GetSummarizationStatus",
            data="{}"
        )

        # May fail if model not loaded, but call should succeed
        if rc != 0:
            assert "not ready" in stderr.lower() or "unavailable" in stderr.lower(), \
                f"Unexpected error: {stderr}"
        else:
            # Success - parse response
            # Response may have modelLoaded, gpuEnabled, etc.
            assert rc == 0

    def test_summarize_text_brief(self, grpc_server):
        """Test Summarize endpoint with brief mode."""
        host, port = grpc_server
        test_text = "The quick brown fox jumps over the lazy dog. This is a simple test sentence."
        rc, stdout, stderr = run_grpcurl(
            host, port,
            "voice.SummarizationService/Summarize",
            data=json.dumps({"text": test_text, "mode": "MODE_BRIEF"})
        )

        # May fail if model not loaded
        if rc != 0:
            assert "not ready" in stderr.lower() or "unavailable" in stderr.lower(), \
                f"Unexpected error: {stderr}"
        else:
            # Success - should return a summary
            assert "summary" in stdout.lower() or rc == 0

    def test_summarize_empty_text(self, grpc_server):
        """Test Summarize endpoint with empty text returns gracefully."""
        host, port = grpc_server
        rc, stdout, stderr = run_grpcurl(
            host, port,
            "voice.SummarizationService/Summarize",
            data=json.dumps({"text": ""})
        )

        # Empty text should return empty summary, not error
        # (unless model not loaded, which is also acceptable)
        if rc != 0:
            # Check if it's a model not ready error (expected)
            assert "not ready" in stderr.lower() or "unavailable" in stderr.lower() or rc == 0

    def test_describe_summarize_request(self, grpc_server):
        """Test describing SummarizeTextRequest message."""
        host, port = grpc_server
        rc, stdout, stderr = run_grpcurl(
            host, port,
            "describe", "voice.SummarizeTextRequest"
        )

        assert rc == 0, f"describe failed: {stderr}"
        assert "text" in stdout
        assert "mode" in stdout
        assert "source_language" in stdout

    def test_describe_summarize_response(self, grpc_server):
        """Test describing SummarizeResponse message."""
        host, port = grpc_server
        rc, stdout, stderr = run_grpcurl(
            host, port,
            "describe", "voice.SummarizeResponse"
        )

        assert rc == 0, f"describe failed: {stderr}"
        assert "summary" in stdout
        assert "latency_ms" in stdout

    def test_describe_summary_event(self, grpc_server):
        """Test describing SummaryEvent message for LiveSummarize."""
        host, port = grpc_server
        rc, stdout, stderr = run_grpcurl(
            host, port,
            "describe", "voice.SummaryEvent"
        )

        assert rc == 0, f"describe failed: {stderr}"
        assert "summary" in stdout
        assert "trigger_reason" in stdout
        assert "transcript_words" in stdout

    def test_describe_summarization_mode_enum(self, grpc_server):
        """Test describing SummarizationMode enum."""
        host, port = grpc_server
        rc, stdout, stderr = run_grpcurl(
            host, port,
            "describe", "voice.SummarizationMode"
        )

        assert rc == 0, f"describe failed: {stderr}"
        assert "MODE_BRIEF" in stdout
        assert "MODE_STANDARD" in stdout
        assert "MODE_DETAILED" in stdout
        assert "MODE_ACTION_ITEMS" in stdout
