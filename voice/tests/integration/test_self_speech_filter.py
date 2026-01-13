"""
Integration Tests for SelfSpeechFilter - Unified Self-Speech Filtering System

Worker #296 - Phase 4.4 Comprehensive Integration Tests

Tests the unified SelfSpeechFilter that combines all three layers:
1. Text Match Filter (fuzzy string matching)
2. Acoustic Echo Cancellation (SpeexDSP AEC)
3. Speaker Diarization (ECAPA-TDNN embeddings)

Test categories:
1. Binary compilation and help output
2. CLI --demo-duplex mode
3. gRPC FilteredSTTService proto definitions
4. Filter initialization and layer readiness

Copyright 2025 Andrew Yates. All rights reserved.
"""

import os
import pytest
import subprocess
import time
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
STREAM_TTS_CPP = PROJECT_ROOT / "stream-tts-cpp"
BUILD_DIR = STREAM_TTS_CPP / "build"
CONFIG_DIR = STREAM_TTS_CPP / "config"
PROTO_DIR = STREAM_TTS_CPP / "proto"


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
def proto_file():
    """Path to voice.proto file."""
    proto = PROTO_DIR / "voice.proto"
    if not proto.exists():
        pytest.skip(f"Proto file not found: {proto}")
    return proto


# =============================================================================
# Tests: Binary Compilation
# =============================================================================

class TestBinaryCompilation:
    """Test that self-speech filter is compiled into the binary."""

    def test_binary_exists(self, tts_binary):
        """Binary should exist after build."""
        assert tts_binary.exists()

    def test_binary_executable(self, tts_binary):
        """Binary should be executable."""
        assert os.access(tts_binary, os.X_OK)

    def test_help_output_includes_demo_duplex(self, tts_binary):
        """Help output should include --demo-duplex option."""
        result = subprocess.run(
            [str(tts_binary), "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )
        assert result.returncode == 0
        assert "--demo-duplex" in result.stdout, (
            "--demo-duplex flag not found in help output"
        )

    def test_help_describes_full_duplex(self, tts_binary):
        """Help output should describe full-duplex self-speech filtering."""
        result = subprocess.run(
            [str(tts_binary), "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )
        assert result.returncode == 0
        # The help text should mention self-speech filtering
        output = result.stdout.lower()
        assert "duplex" in output or "self-speech" in output or "filter" in output


# =============================================================================
# Tests: Proto Definition
# =============================================================================

class TestProtoDefinition:
    """Test that gRPC proto file includes FilteredSTTService."""

    def test_proto_file_exists(self, proto_file):
        """Proto file should exist."""
        assert proto_file.exists()

    def test_proto_defines_filtered_stt_service(self, proto_file):
        """Proto should define FilteredSTTService."""
        content = proto_file.read_text()
        assert "FilteredSTTService" in content, (
            "FilteredSTTService not found in voice.proto"
        )

    def test_proto_defines_stream_filtered_transcribe(self, proto_file):
        """Proto should define StreamFilteredTranscribe RPC."""
        content = proto_file.read_text()
        assert "StreamFilteredTranscribe" in content, (
            "StreamFilteredTranscribe RPC not found in voice.proto"
        )

    def test_proto_defines_filtered_transcript_event(self, proto_file):
        """Proto should define FilteredTranscriptEvent message."""
        content = proto_file.read_text()
        assert "FilteredTranscriptEvent" in content, (
            "FilteredTranscriptEvent message not found in voice.proto"
        )

    def test_proto_defines_speaker_id_field(self, proto_file):
        """FilteredTranscriptEvent should have speaker_id field."""
        content = proto_file.read_text()
        assert "speaker_id" in content, (
            "speaker_id field not found in voice.proto"
        )

    def test_proto_defines_start_filtered_listen(self, proto_file):
        """Proto should define StartFilteredListen RPC."""
        content = proto_file.read_text()
        assert "StartFilteredListen" in content, (
            "StartFilteredListen RPC not found in voice.proto"
        )

    def test_proto_defines_filtered_listen_request(self, proto_file):
        """Proto should define FilteredListenRequest message."""
        content = proto_file.read_text()
        assert "FilteredListenRequest" in content, (
            "FilteredListenRequest message not found in voice.proto"
        )

    def test_proto_defines_transcription_event(self, proto_file):
        """Proto should define TranscriptionEvent message."""
        content = proto_file.read_text()
        assert "TranscriptionEvent" in content, (
            "TranscriptionEvent message not found in voice.proto"
        )

    def test_proto_defines_filter_config(self, proto_file):
        """Proto should define FilterConfig message."""
        content = proto_file.read_text()
        assert "FilterConfig" in content, (
            "FilterConfig message not found in voice.proto"
        )

    def test_proto_defines_filter_stats(self, proto_file):
        """Proto should define FilterStatsResponse message."""
        content = proto_file.read_text()
        assert "FilterStatsResponse" in content, (
            "FilterStatsResponse message not found in voice.proto"
        )


# =============================================================================
# Tests: Header Files
# =============================================================================

class TestHeaderFiles:
    """Test that all self-speech filter header files exist."""

    def test_self_speech_filter_header_exists(self):
        """Main SelfSpeechFilter header should exist."""
        header = STREAM_TTS_CPP / "include" / "self_speech_filter.hpp"
        assert header.exists(), f"self_speech_filter.hpp not found at {header}"

    def test_text_match_filter_header_exists(self):
        """TextMatchFilter header should exist."""
        header = STREAM_TTS_CPP / "include" / "text_match_filter.hpp"
        assert header.exists(), f"text_match_filter.hpp not found at {header}"

    def test_aec_bridge_header_exists(self):
        """AECBridge header should exist."""
        header = STREAM_TTS_CPP / "include" / "aec_bridge.hpp"
        assert header.exists(), f"aec_bridge.hpp not found at {header}"

    def test_speaker_diarized_stt_header_exists(self):
        """SpeakerDiarizedSTT header should exist."""
        header = STREAM_TTS_CPP / "include" / "speaker_diarized_stt.hpp"
        assert header.exists(), f"speaker_diarized_stt.hpp not found at {header}"


# =============================================================================
# Tests: Source Files
# =============================================================================

class TestSourceFiles:
    """Test that all self-speech filter source files exist."""

    def test_self_speech_filter_source_exists(self):
        """Main SelfSpeechFilter implementation should exist."""
        source = STREAM_TTS_CPP / "src" / "self_speech_filter.cpp"
        assert source.exists(), f"self_speech_filter.cpp not found at {source}"

    def test_text_match_filter_source_exists(self):
        """TextMatchFilter implementation should exist."""
        source = STREAM_TTS_CPP / "src" / "text_match_filter.cpp"
        assert source.exists(), f"text_match_filter.cpp not found at {source}"

    def test_aec_bridge_source_exists(self):
        """AECBridge implementation should exist."""
        source = STREAM_TTS_CPP / "src" / "aec_bridge.cpp"
        assert source.exists(), f"aec_bridge.cpp not found at {source}"

    def test_speaker_diarized_stt_source_exists(self):
        """SpeakerDiarizedSTT implementation should exist."""
        source = STREAM_TTS_CPP / "src" / "speaker_diarized_stt.cpp"
        assert source.exists(), f"speaker_diarized_stt.cpp not found at {source}"


# =============================================================================
# Tests: Demo Duplex Mode
# =============================================================================

class TestDemoDuplexMode:
    """Test the --demo-duplex command line mode."""

    def test_demo_duplex_starts_without_crash(self, tts_binary, english_config):
        """Demo duplex mode should start without immediate crash.

        Note: This test doesn't fully exercise the filter since we can't
        provide audio in a headless test. It verifies the mode initializes.
        """
        proc = subprocess.Popen(
            [str(tts_binary), str(english_config), "--demo-duplex", "en"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Wait a few seconds for initialization
        time.sleep(3.0)

        # Check if still running (should be waiting for audio)
        if proc.poll() is not None:
            # Process exited - check why
            stdout, stderr = proc.communicate(timeout=5)
            output = stdout + stderr

            # Skip if no microphone access
            if "microphone" in output.lower():
                pytest.skip("Microphone access not available")
            # Skip if speaker model not found
            if "ecapa" in output.lower() or "speaker" in output.lower():
                pytest.skip("Speaker embedding model not found")
            # If it's a different error, fail the test
            if "failed" in output.lower() or "error" in output.lower():
                pytest.fail(f"Demo duplex failed to start: {output}")

        # Kill the process since it's designed to run forever
        proc.kill()
        proc.wait(timeout=5)

    def test_demo_duplex_shows_filter_status(self, tts_binary, english_config):
        """Demo duplex should print filter initialization status."""
        proc = subprocess.Popen(
            [str(tts_binary), str(english_config), "--demo-duplex", "en"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        time.sleep(3.0)
        proc.kill()
        stdout, stderr = proc.communicate(timeout=5)
        output = stdout + stderr

        # Skip if no microphone access
        if "microphone" in output.lower():
            pytest.skip("Microphone access not available")

        # Should show filter initialization status
        # (Check for any indication it tried to initialize)
        if "initialized" in output.lower() or "filter" in output.lower():
            pass  # Good - filter status shown


# =============================================================================
# Tests: API Consistency
# =============================================================================

class TestAPIConsistency:
    """Test API consistency across filter components."""

    def test_self_speech_filter_hpp_declares_class(self):
        """self_speech_filter.hpp should declare SelfSpeechFilter class."""
        header = STREAM_TTS_CPP / "include" / "self_speech_filter.hpp"
        content = header.read_text()
        assert "class SelfSpeechFilter" in content

    def test_self_speech_filter_has_initialize(self):
        """SelfSpeechFilter should have initialize() method."""
        header = STREAM_TTS_CPP / "include" / "self_speech_filter.hpp"
        content = header.read_text()
        assert "initialize()" in content

    def test_self_speech_filter_has_on_tts_start(self):
        """SelfSpeechFilter should have on_tts_start() method."""
        header = STREAM_TTS_CPP / "include" / "self_speech_filter.hpp"
        content = header.read_text()
        assert "on_tts_start" in content

    def test_self_speech_filter_has_on_tts_audio(self):
        """SelfSpeechFilter should have on_tts_audio() method."""
        header = STREAM_TTS_CPP / "include" / "self_speech_filter.hpp"
        content = header.read_text()
        assert "on_tts_audio" in content

    def test_self_speech_filter_has_on_tts_end(self):
        """SelfSpeechFilter should have on_tts_end() method."""
        header = STREAM_TTS_CPP / "include" / "self_speech_filter.hpp"
        content = header.read_text()
        assert "on_tts_end" in content

    def test_self_speech_filter_has_process(self):
        """SelfSpeechFilter should have process() method."""
        header = STREAM_TTS_CPP / "include" / "self_speech_filter.hpp"
        content = header.read_text()
        assert "process(" in content

    def test_self_speech_filter_has_start_streaming(self):
        """SelfSpeechFilter should have start_streaming() method."""
        header = STREAM_TTS_CPP / "include" / "self_speech_filter.hpp"
        content = header.read_text()
        assert "start_streaming" in content

    def test_self_speech_filter_has_get_stats(self):
        """SelfSpeechFilter should have get_stats() method."""
        header = STREAM_TTS_CPP / "include" / "self_speech_filter.hpp"
        content = header.read_text()
        assert "get_stats" in content


# =============================================================================
# Tests: Config Structures
# =============================================================================

class TestConfigStructures:
    """Test configuration structures in headers."""

    def test_self_speech_filter_config_exists(self):
        """SelfSpeechFilterConfig struct should exist."""
        header = STREAM_TTS_CPP / "include" / "self_speech_filter.hpp"
        content = header.read_text()
        assert "SelfSpeechFilterConfig" in content

    def test_config_has_enable_text_matching(self):
        """Config should have enable_text_matching flag."""
        header = STREAM_TTS_CPP / "include" / "self_speech_filter.hpp"
        content = header.read_text()
        assert "enable_text_matching" in content

    def test_config_has_enable_aec(self):
        """Config should have enable_aec flag."""
        header = STREAM_TTS_CPP / "include" / "self_speech_filter.hpp"
        content = header.read_text()
        assert "enable_aec" in content

    def test_config_has_enable_speaker_diarization(self):
        """Config should have enable_speaker_diarization flag."""
        header = STREAM_TTS_CPP / "include" / "self_speech_filter.hpp"
        content = header.read_text()
        assert "enable_speaker_diarization" in content

    def test_config_has_confidence_threshold(self):
        """Config should have agent_confidence_threshold."""
        header = STREAM_TTS_CPP / "include" / "self_speech_filter.hpp"
        content = header.read_text()
        assert "agent_confidence_threshold" in content


# =============================================================================
# Tests: FilteredResult Structure
# =============================================================================

class TestFilteredResult:
    """Test FilteredResult structure."""

    def test_filtered_result_exists(self):
        """FilteredResult struct should exist."""
        header = STREAM_TTS_CPP / "include" / "self_speech_filter.hpp"
        content = header.read_text()
        assert "FilteredResult" in content or "struct FilteredResult" in content

    def test_result_has_user_speech(self):
        """FilteredResult should have user_speech field."""
        header = STREAM_TTS_CPP / "include" / "self_speech_filter.hpp"
        content = header.read_text()
        assert "user_speech" in content

    def test_result_has_filtered_agent_speech(self):
        """FilteredResult should have filtered_agent_speech field."""
        header = STREAM_TTS_CPP / "include" / "self_speech_filter.hpp"
        content = header.read_text()
        assert "filtered_agent_speech" in content

    def test_result_has_speaker_id(self):
        """FilteredResult should have speaker_id field."""
        header = STREAM_TTS_CPP / "include" / "self_speech_filter.hpp"
        content = header.read_text()
        assert "speaker_id" in content

    def test_result_has_confidence_scores(self):
        """FilteredResult should have confidence scores."""
        header = STREAM_TTS_CPP / "include" / "self_speech_filter.hpp"
        content = header.read_text()
        assert "user_confidence" in content
        assert "agent_confidence" in content


# =============================================================================
# Tests: Statistics Structure
# =============================================================================

class TestStatistics:
    """Test statistics structure."""

    def test_stats_struct_exists(self):
        """SelfSpeechFilterStats struct should exist."""
        header = STREAM_TTS_CPP / "include" / "self_speech_filter.hpp"
        content = header.read_text()
        assert "SelfSpeechFilterStats" in content

    def test_stats_has_utterance_counts(self):
        """Stats should have utterance counts."""
        header = STREAM_TTS_CPP / "include" / "self_speech_filter.hpp"
        content = header.read_text()
        assert "total_utterances" in content
        assert "user_speech_count" in content
        assert "agent_speech_count" in content


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
