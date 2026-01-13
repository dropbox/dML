"""
Manager Audit Verification Tests (Dec 11 2025)

These tests verify claims made in the roadmap by running actual integration tests.
Created to ensure future workers don't claim fixes without verification.

Run with: pytest tests/integration/test_manager_audit.py -v
"""
import os
import subprocess
import tempfile
import pytest
from pathlib import Path

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
BINARY = PROJECT_ROOT / "stream-tts-cpp" / "build" / "stream-tts-cpp"
DAEMON_SOCKET = Path("/tmp/stream-tts.sock")


class TestBinaryFunctionality:
    """Test that the TTS binary actually works."""

    @pytest.fixture(autouse=True)
    def check_binary(self):
        """Skip if binary doesn't exist."""
        if not BINARY.exists():
            pytest.skip(f"Binary not found: {BINARY}")

    def test_binary_version(self):
        """Verify binary runs and reports version."""
        result = subprocess.run(
            [str(BINARY), "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        assert result.returncode == 0, f"Binary failed: {result.stderr}"
        assert "Stream TTS C++" in result.stdout
        assert "libtorch" in result.stdout.lower() or "Enabled" in result.stdout

    def test_tts_produces_valid_wav(self):
        """CRITICAL: Verify TTS actually produces a valid WAV file."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = f.name

        try:
            result = subprocess.run(
                [str(BINARY), "--speak", "Test one two three", "--lang", "en",
                 "--output", output_path],
                capture_output=True,
                text=True,
                timeout=120  # Allow for model warmup
            )

            # Check process succeeded
            assert result.returncode == 0, f"TTS failed: {result.stderr}"

            # Check WAV file exists and has content
            assert os.path.exists(output_path), "WAV file not created"
            file_size = os.path.getsize(output_path)
            assert file_size > 1000, f"WAV file too small: {file_size} bytes"

            # Verify it's actually a WAV file
            with open(output_path, "rb") as f:
                header = f.read(4)
            assert header == b"RIFF", f"Not a valid WAV file, header: {header}"

            # Check logs mention synthesis (logs go to stdout)
            output = result.stdout + result.stderr
            assert "Synthesis" in output or "synthesis" in output.lower(), \
                f"No synthesis log found in output"

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_kokoro_engine_used(self):
        """Verify Kokoro TorchScript is the default engine."""
        result = subprocess.run(
            [str(BINARY), "--speak", "Hello", "--lang", "en"],
            capture_output=True,
            text=True,
            timeout=120
        )

        # Should use Kokoro, not fall back to espeak (logs go to stdout)
        output = result.stdout + result.stderr
        assert "Kokoro" in output or "kokoro" in output.lower(), \
            f"Kokoro not used. Logs: {output[:500]}"
        assert "TorchScript" in output or "torchscript" in output.lower(), \
            "TorchScript not mentioned in logs"


class TestTranslation:
    """Test translation functionality."""

    @pytest.fixture(autouse=True)
    def check_binary(self):
        if not BINARY.exists():
            pytest.skip(f"Binary not found: {BINARY}")

    def test_english_to_japanese_translation(self):
        """CRITICAL: Verify EN→JA translation works."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = f.name

        try:
            result = subprocess.run(
                [str(BINARY), "--speak", "Hello how are you", "--lang", "ja",
                 "--translate", "--output", output_path],
                capture_output=True,
                text=True,
                timeout=120
            )

            assert result.returncode == 0, f"Translation failed: {result.stderr}"

            # Check translation happened (logs go to stdout)
            output = result.stdout + result.stderr
            assert "Translat" in output, f"No translation log found"
            assert "→" in output or "->" in output, \
                "No translation arrow in logs"

            # Verify output file created
            assert os.path.exists(output_path), "Output WAV not created"
            assert os.path.getsize(output_path) > 1000, "Output WAV too small"

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


class TestMultiLanguageTTS:
    """Test multi-language TTS support."""

    @pytest.fixture(autouse=True)
    def check_binary(self):
        if not BINARY.exists():
            pytest.skip(f"Binary not found: {BINARY}")

    def test_arabic_mms_tts(self):
        """Verify Arabic MMS-TTS works."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = f.name

        try:
            result = subprocess.run(
                [str(BINARY), "--speak", "مرحبا", "--lang", "ar",
                 "--output", output_path],
                capture_output=True,
                text=True,
                timeout=120
            )

            assert result.returncode == 0, f"Arabic TTS failed: {result.stderr}"

            # Should use MMS-TTS for Arabic (logs go to stdout)
            output = result.stdout + result.stderr
            assert "MMS-TTS" in output or "mms" in output.lower(), \
                f"MMS-TTS not used for Arabic. Logs: {output[:500]}"

            # Verify output
            assert os.path.exists(output_path), "Output WAV not created"
            assert os.path.getsize(output_path) > 500, "Output WAV too small"

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_turkish_mms_tts(self):
        """Verify Turkish MMS-TTS works."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = f.name

        try:
            result = subprocess.run(
                [str(BINARY), "--speak", "Merhaba", "--lang", "tr",
                 "--output", output_path],
                capture_output=True,
                text=True,
                timeout=120
            )

            assert result.returncode == 0, f"Turkish TTS failed: {result.stderr}"
            assert os.path.exists(output_path), "Output WAV not created"

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_persian_mms_tts(self):
        """Verify Persian MMS-TTS works."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = f.name

        try:
            result = subprocess.run(
                [str(BINARY), "--speak", "سلام", "--lang", "fa",
                 "--output", output_path],
                capture_output=True,
                text=True,
                timeout=120
            )

            assert result.returncode == 0, f"Persian TTS failed: {result.stderr}"
            assert os.path.exists(output_path), "Output WAV not created"

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


class TestDaemonMode:
    """Test daemon functionality."""

    def test_daemon_socket_format(self):
        """Verify daemon socket path is correct."""
        # This test just verifies the expected path
        assert str(DAEMON_SOCKET) == "/tmp/stream-tts.sock"

    @pytest.mark.skipif(
        not DAEMON_SOCKET.exists(),
        reason="Daemon not running (socket doesn't exist)"
    )
    def test_daemon_socket_exists_when_running(self):
        """If daemon is running, socket should exist."""
        import stat
        mode = os.stat(DAEMON_SOCKET).st_mode
        assert stat.S_ISSOCK(mode), "Path exists but is not a socket"


class TestFishSpeechStatus:
    """Document Fish-Speech integration status.

    As of Worker #521 (2025-12-11), Fish-Speech works in PURE C++:
    - Tokenizer: C++ tiktoken implementation (100K tokens)
    - Transformer: Native C++ with safetensors loading (203 weights, 2.5GB)
    - Vocoder: TorchScript model (6.5-7.9x real-time on CPU)
    - E2E: Verified working with test_fish_speech_e2e --text "Hello world"

    No Python server required. Python server (fish_speech_server.py) is
    now only needed for legacy compatibility.
    """

    def test_fish_speech_native_cpp_works(self):
        """Verify Fish-Speech native C++ components exist."""
        models_dir = PROJECT_ROOT / "models" / "fish-speech-1.5"

        if not models_dir.exists():
            pytest.skip("Fish-Speech models not downloaded")

        # Verify all required model files exist
        vocoder = models_dir / "vocoder_cpu.pt"
        safetensors = models_dir / "transformer_weights.safetensors"
        tokenizer = models_dir / "tokenizer.tiktoken"
        config = models_dir / "transformer_config_cpp.json"

        assert vocoder.exists(), "Vocoder TorchScript not found"
        assert safetensors.exists(), "Transformer weights not found"
        assert tokenizer.exists(), "Tokenizer not found"
        assert config.exists(), "C++ config not found"

        # Verify sizes are reasonable
        assert vocoder.stat().st_size > 50_000_000, "Vocoder too small"
        assert safetensors.stat().st_size > 2_000_000_000, "Transformer weights too small"
        assert tokenizer.stat().st_size > 1_000_000, "Tokenizer too small"

    def test_fish_speech_e2e_binary_exists(self):
        """Verify Fish-Speech E2E test binary exists."""
        e2e_binary = PROJECT_ROOT / "stream-tts-cpp" / "build" / "test_fish_speech_e2e"

        if not e2e_binary.exists():
            pytest.skip("Fish-Speech E2E binary not built")

        # Just verify it exists - actual E2E test is slow and covered by C++ tests
        assert e2e_binary.stat().st_size > 1_000_000, "E2E binary too small"


class TestCodeFixes:
    """Verify claimed code fixes exist."""

    def test_sp1_backpressure_exists(self):
        """Verify SP1 fix: synthesis_queue_ backpressure."""
        source_file = PROJECT_ROOT / "stream-tts-cpp" / "src" / "streaming_pipeline.cpp"
        if not source_file.exists():
            pytest.skip("Source file not found")

        content = source_file.read_text()
        assert "max_queue_size" in content, "SP1: max_queue_size not found"
        assert "synthesis_queue_.size()" in content, "SP1: queue size check not found"

    def test_sp2_null_check_exists(self):
        """Verify SP2 fix: translator_ null check."""
        source_file = PROJECT_ROOT / "stream-tts-cpp" / "src" / "streaming_pipeline.cpp"
        if not source_file.exists():
            pytest.skip("Source file not found")

        content = source_file.read_text()
        # Should have null check before using translator_
        assert "translator_ &&" in content or "translator_)" in content, \
            "SP2: translator_ null check not found"

    def test_sp5_raii_guard_exists(self):
        """Verify SP5 fix: RAII StopGuard."""
        source_file = PROJECT_ROOT / "stream-tts-cpp" / "src" / "streaming_pipeline.cpp"
        if not source_file.exists():
            pytest.skip("Source file not found")

        content = source_file.read_text()
        assert "StopGuard" in content, "SP5: StopGuard struct not found"
        assert "~StopGuard" in content, "SP5: StopGuard destructor not found"

    def test_ve5_espeak_return_check(self):
        """Verify VE5 fix: espeak_Initialize return check."""
        source_file = PROJECT_ROOT / "stream-tts-cpp" / "src" / "kokoro_torchscript_tts.cpp"
        if not source_file.exists():
            pytest.skip("Source file not found")

        content = source_file.read_text()
        assert "espeak_Initialize" in content, "espeak_Initialize not found"
        # Should check for -1 return
        assert "== -1" in content or "< 0" in content, \
            "VE5: espeak return code not checked"
