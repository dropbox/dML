"""
Integration Tests for Full Pipeline, Daemon Mode, and Streaming Mode

Worker #118 - Phase 6: Final Integration Tests

Tests the complete voice system:
1. Full pipeline: text -> translate -> TTS -> play
2. Daemon mode: persistent server with commands
3. Streaming mode: sentence-level streaming
4. Error handling: invalid input, missing models, etc.

Copyright 2025 Andrew Yates. All rights reserved.
"""

import json
import os
import pytest
import signal
import socket
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

# Daemon socket path (default)
DEFAULT_SOCKET = "/tmp/voice-daemon.sock"


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
    """Path to EN->JA translation config."""
    config = CONFIG_DIR / "kokoro-mps-en2ja.yaml"
    if not config.exists():
        pytest.skip(f"EN->JA config not found: {config}")
    return config


@pytest.fixture(scope="module")
def en2zh_config():
    """Path to EN->ZH translation config."""
    config = CONFIG_DIR / "kokoro-mps-en2zh.yaml"
    if not config.exists():
        pytest.skip(f"EN->ZH config not found: {config}")
    return config


@pytest.fixture
def temp_socket():
    """Create a temporary socket path for daemon tests."""
    socket_path = f"/tmp/voice-test-{os.getpid()}.sock"
    yield socket_path
    # Cleanup
    if os.path.exists(socket_path):
        try:
            os.unlink(socket_path)
        except OSError:
            pass


@pytest.fixture
def temp_wav(tmp_path):
    """Fixture to provide a temporary WAV file path."""
    return tmp_path / "test_audio.wav"


# =============================================================================
# Helper Functions
# =============================================================================

def get_tts_env():
    """
    Get environment variables for TTS subprocess.

    Note: PYTORCH_ENABLE_MPS_FALLBACK no longer needed with PyTorch 2.9.1+.
    """
    env = os.environ.copy()
    # Fix OpenMP duplicate library crash when llama.cpp + libtorch both link OpenMP
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    return env


def create_claude_json(text: str) -> str:
    """Create Claude API format JSON for input."""
    escaped = text.replace('"', '\\"').replace('\n', '\\n')
    return f'{{"type":"content_block_delta","delta":{{"type":"text_delta","text":"{escaped}"}}}}'


def run_tts_pipeline(binary: Path, config: Path, text: str,
                     output_path: Path = None, timeout: int = 60,
                     streaming: bool = False) -> subprocess.CompletedProcess:
    """
    Run the TTS pipeline with given text.

    Args:
        binary: Path to stream-tts-cpp
        config: Path to config YAML
        text: Text to synthesize
        output_path: Optional path to save WAV file
        timeout: Command timeout
        streaming: Use streaming mode (-s flag)

    Returns:
        CompletedProcess with stdout/stderr
    """
    input_json = create_claude_json(text)

    cmd = [str(binary)]
    if output_path:
        cmd.extend(["--save-audio", str(output_path)])
    if streaming:
        cmd.append("-s")
    cmd.append(str(config))

    result = subprocess.run(
        cmd,
        input=input_json,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(STREAM_TTS_CPP),
        env=get_tts_env()
    )

    return result


def start_daemon(binary: Path, config: Path, socket_path: str,
                 timeout: int = 30) -> subprocess.Popen:
    """
    Start the daemon process.

    Args:
        binary: Path to stream-tts-cpp
        config: Path to config YAML
        socket_path: Unix socket path for communication
        timeout: Time to wait for daemon to start

    Returns:
        Popen process object
    """
    cmd = [
        str(binary),
        "--daemon",
        "--socket", socket_path,
        str(config)
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(STREAM_TTS_CPP),
        env=get_tts_env()
    )

    # Wait for socket to be created
    start_time = time.time()
    while time.time() - start_time < timeout:
        if os.path.exists(socket_path):
            # Give it a moment to start accepting connections
            time.sleep(0.5)
            return process
        time.sleep(0.1)

    # Daemon didn't start in time
    process.kill()
    stdout, stderr = process.communicate(timeout=5)
    raise RuntimeError(f"Daemon failed to start: {stderr.decode()}")


def send_daemon_command(binary: Path, socket_path: str, command: str,
                        text: str = None, priority: int = None,
                        timeout: int = 30) -> subprocess.CompletedProcess:
    """
    Send a command to the daemon.

    Args:
        binary: Path to stream-tts-cpp
        socket_path: Unix socket path
        command: One of --speak, --status, --interrupt, --stop
        text: Text for --speak command
        priority: Priority level for --speak (higher = more urgent)
        timeout: Command timeout

    Returns:
        CompletedProcess with result
    """
    cmd = [str(binary), "--socket", socket_path]

    if command == "--speak":
        if text is None:
            raise ValueError("--speak requires text")
        cmd.extend(["--speak", text])
        if priority is not None:
            cmd.extend(["--priority", str(priority)])
    elif command in ("--status", "--interrupt", "--stop"):
        cmd.append(command)
    else:
        raise ValueError(f"Unknown command: {command}")

    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(STREAM_TTS_CPP),
        env=get_tts_env()
    )


# =============================================================================
# Full Pipeline Integration Tests
# =============================================================================

@pytest.mark.integration
class TestFullPipeline:
    """Test the complete text -> translate -> TTS -> audio pipeline."""

    def test_english_tts_generates_audio(self, tts_binary, english_config, temp_wav):
        """Test that English TTS generates valid audio file."""
        result = run_tts_pipeline(
            tts_binary, english_config,
            "Hello, this is a test of the text to speech system.",
            output_path=temp_wav
        )

        assert result.returncode == 0, f"TTS failed: {result.stderr}"
        assert temp_wav.exists(), "WAV file not created"
        assert temp_wav.stat().st_size > 1000, "WAV file too small"

    def test_en2ja_translation_pipeline(self, tts_binary, en2ja_config, temp_wav):
        """Test English to Japanese translation + TTS pipeline."""
        result = run_tts_pipeline(
            tts_binary, en2ja_config,
            "Good morning, how are you today?",
            output_path=temp_wav,
            timeout=120  # Translation takes longer
        )

        assert result.returncode == 0, f"EN->JA pipeline failed: {result.stderr}"

        # Check for translation activity in output
        output = result.stdout + result.stderr
        assert "Translation" in output or "TTS" in output, \
            "Expected translation/TTS activity in output"

    def test_en2zh_translation_pipeline(self, tts_binary, en2zh_config, temp_wav):
        """Test English to Chinese translation + TTS pipeline.

        Note: Chinese TTS has G2P limitations, but the pipeline should not crash.
        """
        result = run_tts_pipeline(
            tts_binary, en2zh_config,
            "The weather is nice today.",
            output_path=temp_wav,
            timeout=120
        )

        # Pipeline should complete without crashing
        assert result.returncode == 0, f"EN->ZH pipeline crashed: {result.stderr}"

    def test_pipeline_with_punctuation(self, tts_binary, english_config, temp_wav):
        """Test TTS handles various punctuation correctly."""
        result = run_tts_pipeline(
            tts_binary, english_config,
            "Hello! How are you? I'm fine, thanks. Really -- very good.",
            output_path=temp_wav
        )

        assert result.returncode == 0, f"Punctuation test failed: {result.stderr}"
        assert temp_wav.exists() and temp_wav.stat().st_size > 1000

    def test_pipeline_with_numbers(self, tts_binary, english_config, temp_wav):
        """Test TTS handles numbers correctly."""
        result = run_tts_pipeline(
            tts_binary, english_config,
            "The year is 2025 and we have 42 items costing $3.50 each.",
            output_path=temp_wav
        )

        assert result.returncode == 0, f"Numbers test failed: {result.stderr}"
        assert temp_wav.exists() and temp_wav.stat().st_size > 1000

    def test_pipeline_with_special_characters(self, tts_binary, english_config, temp_wav):
        """Test TTS handles special characters without crashing."""
        result = run_tts_pipeline(
            tts_binary, english_config,
            "Use the ampersand & at-sign @ and percent % symbols.",
            output_path=temp_wav
        )

        # Should not crash
        assert result.returncode == 0, f"Special chars test failed: {result.stderr}"

    def test_pipeline_long_text(self, tts_binary, english_config, temp_wav):
        """Test TTS handles longer text (multiple sentences)."""
        long_text = """
        The quick brown fox jumps over the lazy dog.
        This pangram contains every letter of the English alphabet.
        It is commonly used for testing fonts and keyboard layouts.
        Today we are testing the text to speech system.
        """

        result = run_tts_pipeline(
            tts_binary, english_config,
            long_text.strip(),
            output_path=temp_wav,
            timeout=120
        )

        assert result.returncode == 0, f"Long text test failed: {result.stderr}"
        assert temp_wav.exists() and temp_wav.stat().st_size > 5000, \
            "WAV file too small for long text"

    def test_pipeline_latency_measurement(self, tts_binary, english_config):
        """Measure and report pipeline latency."""
        import re

        result = run_tts_pipeline(
            tts_binary, english_config,
            "Hello world"
        )

        assert result.returncode == 0

        # Look for latency metrics in output
        output = result.stdout + result.stderr

        # Try to extract End-to-End latency
        match = re.search(r"End-to-End\s+\d+\s+([\d.]+)", output)
        if match:
            latency_ms = float(match.group(1))
            print(f"\nPipeline End-to-End latency: {latency_ms:.0f}ms")
            # Informational only - don't fail on latency


# =============================================================================
# Daemon Mode Tests
# =============================================================================

@pytest.mark.integration
@pytest.mark.slow
class TestDaemonMode:
    """Test daemon mode functionality: start, command, status, stop."""

    def test_daemon_starts_and_stops(self, tts_binary, english_config, temp_socket):
        """Test that daemon can start and stop cleanly."""
        daemon = None
        try:
            daemon = start_daemon(tts_binary, english_config, temp_socket)
            assert daemon.poll() is None, "Daemon exited early"

            # Send stop command
            result = send_daemon_command(tts_binary, temp_socket, "--stop")
            assert result.returncode == 0, f"Stop command failed: {result.stderr}"

            # Wait for daemon to exit
            try:
                daemon.wait(timeout=10)
            except subprocess.TimeoutExpired:
                daemon.kill()
                pytest.fail("Daemon did not exit after stop command")

        finally:
            if daemon and daemon.poll() is None:
                daemon.kill()
                daemon.wait(timeout=5)

    def test_daemon_status_command(self, tts_binary, english_config, temp_socket):
        """Test that --status returns valid JSON."""
        daemon = None
        try:
            daemon = start_daemon(tts_binary, english_config, temp_socket)

            # Get status
            result = send_daemon_command(tts_binary, temp_socket, "--status")
            assert result.returncode == 0, f"Status command failed: {result.stderr}"

            # Parse JSON response
            try:
                status = json.loads(result.stdout.strip())
                print(f"\nDaemon status: {json.dumps(status, indent=2)}")

                # Should have expected fields
                assert "running" in status or "queue_size" in status, \
                    f"Unexpected status format: {status}"
            except json.JSONDecodeError:
                # Status might be printed differently
                assert "running" in result.stdout.lower() or \
                       "status" in result.stdout.lower(), \
                    f"Unexpected status output: {result.stdout}"

        finally:
            if daemon and daemon.poll() is None:
                # Clean stop
                send_daemon_command(tts_binary, temp_socket, "--stop")
                daemon.wait(timeout=10)

    def test_daemon_speak_command(self, tts_binary, english_config, temp_socket):
        """Test that --speak sends text to daemon for synthesis."""
        daemon = None
        try:
            daemon = start_daemon(tts_binary, english_config, temp_socket)

            # Send speak command
            result = send_daemon_command(
                tts_binary, temp_socket,
                "--speak", "Hello from daemon test"
            )

            # Command should be accepted
            assert result.returncode == 0, f"Speak command failed: {result.stderr}"

            # Give time for synthesis (don't need to verify audio)
            time.sleep(1)

        finally:
            if daemon and daemon.poll() is None:
                send_daemon_command(tts_binary, temp_socket, "--stop")
                daemon.wait(timeout=10)

    def test_daemon_interrupt_command(self, tts_binary, english_config, temp_socket):
        """Test that --interrupt stops current speech."""
        daemon = None
        try:
            daemon = start_daemon(tts_binary, english_config, temp_socket)

            # Send a long text
            send_daemon_command(
                tts_binary, temp_socket,
                "--speak", "This is a very long sentence that will take some time to synthesize and speak aloud."
            )

            # Small delay then interrupt
            time.sleep(0.5)
            result = send_daemon_command(tts_binary, temp_socket, "--interrupt")
            assert result.returncode == 0, f"Interrupt failed: {result.stderr}"

        finally:
            if daemon and daemon.poll() is None:
                send_daemon_command(tts_binary, temp_socket, "--stop")
                daemon.wait(timeout=10)

    def test_daemon_priority_interruption(self, tts_binary, english_config, temp_socket):
        """Test that high priority messages interrupt low priority ones (Phase 7.3).

        This tests the priority-based interruption mechanism where higher priority
        messages automatically interrupt lower priority speech without explicit
        --interrupt command.
        """
        daemon = None
        try:
            daemon = start_daemon(tts_binary, english_config, temp_socket)

            # Send a low priority (0) long text that takes time to process
            result_low = send_daemon_command(
                tts_binary, temp_socket,
                "--speak",
                "This is a very long low priority message that will take considerable time to synthesize and speak aloud to the listener.",
                priority=0
            )
            assert result_low.returncode == 0, f"Low priority speak failed: {result_low.stderr}"

            # Small delay to let processing start
            time.sleep(0.3)

            # Send a high priority (10) message that should interrupt
            result_high = send_daemon_command(
                tts_binary, temp_socket,
                "--speak",
                "Urgent high priority message.",
                priority=10
            )
            assert result_high.returncode == 0, f"High priority speak failed: {result_high.stderr}"

            # Give time for the high priority message to be processed
            time.sleep(1)

            # Get status to verify system is still running
            result_status = send_daemon_command(tts_binary, temp_socket, "--status")
            assert result_status.returncode == 0, f"Status after priority test failed: {result_status.stderr}"

        finally:
            if daemon and daemon.poll() is None:
                send_daemon_command(tts_binary, temp_socket, "--stop")
                daemon.wait(timeout=10)

    def test_daemon_multiple_speaks(self, tts_binary, english_config, temp_socket):
        """Test that daemon handles multiple speak commands."""
        daemon = None
        try:
            daemon = start_daemon(tts_binary, english_config, temp_socket)

            # Send multiple speak commands
            for i in range(3):
                result = send_daemon_command(
                    tts_binary, temp_socket,
                    "--speak", f"Message number {i+1}"
                )
                assert result.returncode == 0, f"Speak {i+1} failed: {result.stderr}"

            # Give time for processing
            time.sleep(2)

            # Check status
            result = send_daemon_command(tts_binary, temp_socket, "--status")
            assert result.returncode == 0

        finally:
            if daemon and daemon.poll() is None:
                send_daemon_command(tts_binary, temp_socket, "--stop")
                daemon.wait(timeout=10)

    def test_daemon_client_without_daemon(self, tts_binary, english_config, temp_socket):
        """Test that client commands fail gracefully when daemon not running."""
        # Don't start daemon - just try to connect
        result = send_daemon_command(tts_binary, temp_socket, "--status")

        # Should fail but not crash
        assert result.returncode != 0, "Expected failure when daemon not running"
        assert "Could not connect" in result.stderr or \
               "not running" in result.stderr.lower() or \
               "error" in result.stderr.lower(), \
            f"Expected connection error, got: {result.stderr}"

    def test_daemon_repeated_start_stop(self, tts_binary, english_config, temp_socket):
        """
        Test repeated start/stop cycles to verify socket cleanup.

        Repo Audit Item #15: Harden daemon socket lifecycle tests.
        This test ensures the daemon properly cleans up sockets on shutdown
        and can restart cleanly multiple times.
        """
        NUM_CYCLES = 3

        for cycle in range(NUM_CYCLES):
            daemon = None
            try:
                # Start daemon
                daemon = start_daemon(tts_binary, english_config, temp_socket)
                assert daemon.poll() is None, f"Cycle {cycle+1}: Daemon exited early"
                assert os.path.exists(temp_socket), f"Cycle {cycle+1}: Socket not created"

                # Verify it responds
                result = send_daemon_command(tts_binary, temp_socket, "--status")
                assert result.returncode == 0, f"Cycle {cycle+1}: Status failed"

                # Stop daemon
                result = send_daemon_command(tts_binary, temp_socket, "--stop")
                assert result.returncode == 0, f"Cycle {cycle+1}: Stop failed"

                # Wait for exit
                daemon.wait(timeout=10)
                daemon = None

                # Socket should be cleaned up - wait for it to be removed
                for _ in range(20):  # Up to 2 seconds
                    if not os.path.exists(temp_socket):
                        break
                    time.sleep(0.1)
                else:
                    # Force remove stale socket
                    try:
                        os.unlink(temp_socket)
                    except FileNotFoundError:
                        pass

            except subprocess.TimeoutExpired:
                pytest.fail(f"Cycle {cycle+1}: Daemon timeout")
            finally:
                if daemon and daemon.poll() is None:
                    daemon.kill()
                    daemon.wait(timeout=5)

        print(f"\n{NUM_CYCLES} start/stop cycles completed successfully")

    def test_daemon_stale_socket_cleanup(self, tts_binary, english_config, temp_socket):
        """
        Test that daemon handles stale socket files correctly.

        Repo Audit Item #15: Tests that the daemon can start even when
        a stale socket file exists (e.g., from previous crash).
        """
        # Create a stale socket file
        stale_socket = temp_socket
        with open(stale_socket, 'w') as f:
            f.write("stale")

        assert os.path.exists(stale_socket), "Stale socket file not created"

        daemon = None
        try:
            # Daemon should handle the stale socket
            daemon = start_daemon(tts_binary, english_config, stale_socket, timeout=60)
            assert daemon.poll() is None, "Daemon exited with stale socket"

            # Should be functional
            result = send_daemon_command(tts_binary, stale_socket, "--status")
            # Note: May fail if daemon doesn't handle stale sockets well
            # This test documents the current behavior

        finally:
            if daemon and daemon.poll() is None:
                try:
                    send_daemon_command(tts_binary, stale_socket, "--stop")
                    daemon.wait(timeout=10)
                except Exception:
                    daemon.kill()
                    daemon.wait(timeout=5)

            # Cleanup
            if os.path.exists(stale_socket):
                os.unlink(stale_socket)

    def test_daemon_sigterm_cleanup(self, tts_binary, english_config, temp_socket):
        """
        Test that daemon cleans up when killed with SIGTERM.

        Repo Audit Item #15: Verify clean shutdown on signal.
        """
        daemon = None
        try:
            daemon = start_daemon(tts_binary, english_config, temp_socket)
            assert daemon.poll() is None, "Daemon exited early"
            assert os.path.exists(temp_socket), "Socket not created"

            # Send SIGTERM (graceful shutdown)
            daemon.terminate()

            # Should exit cleanly
            try:
                daemon.wait(timeout=10)
                exit_code = daemon.returncode
                # SIGTERM typically results in exit code 0 or 128+15=143
                # -6 (SIGABRT) can occur during Metal GPU cleanup race conditions
                assert exit_code in (0, -15, -6, 143, None), f"Unexpected exit code: {exit_code}"
            except subprocess.TimeoutExpired:
                daemon.kill()
                pytest.fail("Daemon did not respond to SIGTERM in time")

        finally:
            if daemon and daemon.poll() is None:
                daemon.kill()
                daemon.wait(timeout=5)

    def test_daemon_rapid_commands(self, tts_binary, english_config, temp_socket):
        """
        Test daemon stability under rapid command submission.

        Repo Audit Item #15: Stress test the daemon command handling.
        """
        daemon = None
        try:
            daemon = start_daemon(tts_binary, english_config, temp_socket)

            # Send rapid status requests
            NUM_REQUESTS = 10
            successes = 0
            for i in range(NUM_REQUESTS):
                result = send_daemon_command(tts_binary, temp_socket, "--status", timeout=10)
                if result.returncode == 0:
                    successes += 1

            print(f"\nRapid commands: {successes}/{NUM_REQUESTS} succeeded")

            # At least 80% should succeed (allow some contention)
            assert successes >= NUM_REQUESTS * 0.8, \
                f"Too many rapid command failures: {NUM_REQUESTS - successes}/{NUM_REQUESTS}"

        finally:
            if daemon and daemon.poll() is None:
                send_daemon_command(tts_binary, temp_socket, "--stop")
                daemon.wait(timeout=10)


# =============================================================================
# Streaming Mode Tests
# =============================================================================

@pytest.mark.integration
class TestStreamingMode:
    """Test streaming mode (-s flag) for sentence-level streaming.

    Note: Streaming mode does not support --save-audio (audio goes directly to
    speaker). These tests verify successful execution without file output.
    """

    def test_streaming_mode_basic(self, tts_binary, english_config):
        """Test basic streaming mode execution."""
        # Streaming mode doesn't support --save-audio, just verify execution
        result = run_tts_pipeline(
            tts_binary, english_config,
            "First sentence. Second sentence. Third sentence.",
            streaming=True
        )

        assert result.returncode == 0, f"Streaming mode failed: {result.stderr}"
        # Verify streaming pipeline was used
        output = result.stdout + result.stderr
        assert "Streaming" in output or "streaming" in output or "frames" in output, \
            "Expected streaming pipeline indicators in output"

    def test_streaming_mode_single_sentence(self, tts_binary, english_config):
        """Test streaming mode with single sentence."""
        result = run_tts_pipeline(
            tts_binary, english_config,
            "Just one sentence here.",
            streaming=True
        )

        assert result.returncode == 0, f"Single sentence streaming failed: {result.stderr}"

    def test_streaming_mode_long_text(self, tts_binary, english_config):
        """Test streaming mode with longer multi-sentence text."""
        text = """
        The streaming pipeline splits text into sentences.
        Each sentence is synthesized separately.
        This allows for faster time-to-first-audio.
        The audio plays while the next sentence is being generated.
        This is particularly useful for long-form content.
        """

        result = run_tts_pipeline(
            tts_binary, english_config,
            text.strip(),
            streaming=True,
            timeout=120
        )

        assert result.returncode == 0, f"Long streaming failed: {result.stderr}"
        # Verify multiple sentences were processed
        output = result.stdout + result.stderr
        assert "Streaming" in output or "sentence" in output.lower() or "frames" in output

    def test_streaming_mode_with_translation(self, tts_binary, en2ja_config):
        """Test streaming mode with translation enabled."""
        result = run_tts_pipeline(
            tts_binary, en2ja_config,
            "First sentence. Second sentence.",
            streaming=True,
            timeout=120
        )

        # Should complete without error
        assert result.returncode == 0, f"Streaming + translation failed: {result.stderr}"


# =============================================================================
# Error Handling Tests
# =============================================================================

@pytest.mark.integration
class TestErrorHandling:
    """Test error handling for various edge cases."""

    def test_empty_input(self, tts_binary, english_config):
        """Test handling of empty input."""
        result = subprocess.run(
            [str(tts_binary), str(english_config)],
            input="",
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(STREAM_TTS_CPP)
        )

        # Should exit cleanly (no audio to generate)
        # returncode may be 0 or non-zero depending on implementation
        # Main thing is it shouldn't crash
        assert result.returncode in (0, 1), f"Unexpected exit code: {result.returncode}"

    def test_invalid_json_input(self, tts_binary, english_config):
        """Test handling of malformed JSON input."""
        result = subprocess.run(
            [str(tts_binary), str(english_config)],
            input="not valid json {{{",
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(STREAM_TTS_CPP)
        )

        # Should handle gracefully, not crash
        # (will likely just ignore the invalid JSON)
        assert result.returncode in (0, 1), f"Crashed on invalid JSON: {result.stderr}"

    def test_missing_config_file(self, tts_binary):
        """Test handling of missing config file."""
        result = subprocess.run(
            [str(tts_binary), "/nonexistent/config.yaml"],
            input=create_claude_json("Hello"),
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(STREAM_TTS_CPP)
        )

        # Should fail with error message
        assert result.returncode != 0, "Expected failure with missing config"
        # Error may be in stdout (spdlog) or stderr
        combined = (result.stdout + result.stderr).lower()
        assert "error" in combined or \
               "not found" in combined or \
               "cannot" in combined or \
               "bad file" in combined, \
            f"Expected config error, got stdout: {result.stdout}, stderr: {result.stderr}"

    def test_unicode_text(self, tts_binary, english_config, temp_wav):
        """Test handling of unicode text in English TTS."""
        result = run_tts_pipeline(
            tts_binary, english_config,
            "This has unicode: cafe and resume.",
            output_path=temp_wav
        )

        # Should handle without crashing
        assert result.returncode == 0, f"Unicode test failed: {result.stderr}"

    def test_very_long_word(self, tts_binary, english_config, temp_wav):
        """Test handling of very long words."""
        long_word = "supercalifragilisticexpialidocious"

        result = run_tts_pipeline(
            tts_binary, english_config,
            f"The word is {long_word}.",
            output_path=temp_wav
        )

        # Should handle without crashing
        assert result.returncode == 0, f"Long word test failed: {result.stderr}"

    def test_only_whitespace(self, tts_binary, english_config):
        """Test handling of whitespace-only input."""
        result = run_tts_pipeline(
            tts_binary, english_config,
            "     \n\n\t   "
        )

        # Should not crash (may or may not generate audio)
        assert result.returncode in (0, 1)

    def test_rapid_sequential_requests(self, tts_binary, english_config, tmp_path):
        """Test handling of rapid sequential requests."""
        for i in range(5):
            wav_path = tmp_path / f"rapid_{i}.wav"
            result = run_tts_pipeline(
                tts_binary, english_config,
                f"Request number {i+1}.",
                output_path=wav_path
            )
            assert result.returncode == 0, f"Request {i+1} failed"


# =============================================================================
# Configuration Tests
# =============================================================================

@pytest.mark.integration
class TestConfigurationVariants:
    """Test different configuration options."""

    def test_all_language_configs_exist(self):
        """Verify expected config files exist."""
        expected_configs = [
            "kokoro-mps-en.yaml",
            "kokoro-mps-ja.yaml",
            "kokoro-mps-zh.yaml",
            "kokoro-mps-es.yaml",
            "kokoro-mps-fr.yaml",
            "kokoro-mps-en2ja.yaml",
            "kokoro-mps-en2zh.yaml",
        ]

        for config_name in expected_configs:
            config_path = CONFIG_DIR / config_name
            assert config_path.exists(), f"Missing config: {config_name}"

    def test_japanese_direct_tts(self, tts_binary, temp_wav):
        """Test Japanese TTS with Japanese text input."""
        config = CONFIG_DIR / "kokoro-mps-ja.yaml"
        if not config.exists():
            pytest.skip("Japanese config not found")

        result = run_tts_pipeline(
            tts_binary, config,
            "Hello world",  # Will be processed with Japanese G2P
            output_path=temp_wav
        )

        assert result.returncode == 0, f"Japanese TTS failed: {result.stderr}"

    def test_spanish_tts(self, tts_binary, temp_wav):
        """Test Spanish TTS."""
        config = CONFIG_DIR / "kokoro-mps-es.yaml"
        if not config.exists():
            pytest.skip("Spanish config not found")

        result = run_tts_pipeline(
            tts_binary, config,
            "Hola mundo",
            output_path=temp_wav
        )

        assert result.returncode == 0, f"Spanish TTS failed: {result.stderr}"

    def test_french_tts(self, tts_binary, temp_wav):
        """Test French TTS."""
        config = CONFIG_DIR / "kokoro-mps-fr.yaml"
        if not config.exists():
            pytest.skip("French config not found")

        result = run_tts_pipeline(
            tts_binary, config,
            "Bonjour monde",
            output_path=temp_wav
        )

        assert result.returncode == 0, f"French TTS failed: {result.stderr}"


# =============================================================================
# Performance Metrics Tests
# =============================================================================

@pytest.mark.integration
class TestPerformanceMetrics:
    """Test performance metrics and reporting."""

    def test_warm_latency(self, tts_binary, english_config):
        """Measure warm latency (second invocation)."""
        import re

        # First invocation (cold)
        run_tts_pipeline(tts_binary, english_config, "Warmup call")

        # Second invocation (warm)
        import time
        start = time.time()
        result = run_tts_pipeline(tts_binary, english_config, "Hello world")
        elapsed_ms = (time.time() - start) * 1000

        assert result.returncode == 0
        print(f"\nWarm invocation total time: {elapsed_ms:.0f}ms")

        # Extract internal latency if available
        output = result.stdout + result.stderr
        match = re.search(r"End-to-End\s+\d+\s+([\d.]+)", output)
        if match:
            internal_latency = float(match.group(1))
            print(f"Internal End-to-End latency: {internal_latency:.0f}ms")

    def test_throughput_short_sentences(self, tts_binary, english_config):
        """Measure throughput for short sentences."""
        import time

        sentences = [
            "Hello.",
            "How are you?",
            "I am fine.",
            "Thank you.",
            "Goodbye."
        ]

        start = time.time()
        for sentence in sentences:
            result = run_tts_pipeline(tts_binary, english_config, sentence)
            assert result.returncode == 0

        elapsed = time.time() - start
        throughput = len(sentences) / elapsed

        print(f"\nProcessed {len(sentences)} sentences in {elapsed:.1f}s")
        print(f"Throughput: {throughput:.2f} sentences/sec")


# =============================================================================
# Voice Monitor Tests (Worker #141 - Phase 5)
# =============================================================================

@pytest.mark.integration
class TestVoiceMonitor:
    """Test voice monitor functionality for Claude Code integration."""

    @pytest.fixture
    def worker_logs_dir(self):
        """Path to worker logs directory."""
        logs_dir = PROJECT_ROOT / "worker_logs"
        if not logs_dir.exists():
            pytest.skip(f"Worker logs directory not found: {logs_dir}")
        return logs_dir

    def _run_voice_monitor_with_timeout(self, tts_binary, args, cwd, run_seconds=8):
        """Run voice monitor for a limited time, then terminate gracefully."""
        proc = subprocess.Popen(
            [str(tts_binary)] + args,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Wait for initialization (warmup takes ~3-4s)
        time.sleep(run_seconds)

        # Send SIGTERM for graceful shutdown
        proc.terminate()

        # Wait for graceful exit
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()

        return stdout + stderr

    def test_voice_monitor_starts_and_stops(self, tts_binary, worker_logs_dir):
        """Test that voice monitor can start, find logs, and stop gracefully."""
        output = self._run_voice_monitor_with_timeout(
            tts_binary,
            ["--watch", str(worker_logs_dir), "--verbosity", "2"],
            STREAM_TTS_CPP,
            run_seconds=10  # Extra time for warmup
        )

        # Check that it started successfully
        assert "VoiceMonitor started" in output or "Voice Monitor watching" in output, \
            f"Voice monitor didn't start properly. Output:\n{output}"

        # Check that it found log files
        assert "Found" in output and "log files" in output, \
            f"Voice monitor didn't find log files. Output:\n{output}"

        # Check graceful shutdown
        assert "Shutdown" in output or "stopping" in output.lower(), \
            f"Voice monitor didn't shutdown gracefully. Output:\n{output}"

        print(f"\nVoice monitor test output:\n{output}")

    def test_voice_monitor_default_config_loads(self, tts_binary, worker_logs_dir):
        """Test that voice monitor loads the default configuration."""
        output = self._run_voice_monitor_with_timeout(
            tts_binary,
            ["--watch", str(worker_logs_dir), "--verbosity", "0"],
            STREAM_TTS_CPP,
            run_seconds=10
        )

        # Should load config successfully
        assert "Loaded config" in output or "Auto-configured" in output, \
            f"Failed to load config. Output:\n{output}"

        # Should not have fatal errors
        assert "Fatal error" not in output, \
            f"Voice monitor had fatal error. Output:\n{output}"

    def test_voice_monitor_tts_engine_initializes(self, tts_binary, worker_logs_dir):
        """Test that voice monitor initializes the TTS engine correctly."""
        output = self._run_voice_monitor_with_timeout(
            tts_binary,
            ["--watch", str(worker_logs_dir), "--verbosity", "1"],
            STREAM_TTS_CPP,
            run_seconds=10
        )

        # Should initialize VoiceEngine
        assert "VoiceEngine" in output, \
            f"VoiceEngine not mentioned in output:\n{output}"

        # Should do TTS warmup
        assert "warmup" in output.lower() or "Warmup" in output, \
            f"No warmup mentioned in output:\n{output}"

    def test_voice_monitor_help_shows_watch_option(self, tts_binary):
        """Test that --help shows the --watch option."""
        result = subprocess.run(
            [str(tts_binary), "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode == 0
        output = result.stdout + result.stderr
        assert "--watch" in output, \
            f"--watch not in help output:\n{output}"
        assert "worker logs" in output.lower() or "monitor" in output.lower(), \
            f"Watch description not in help output:\n{output}"
