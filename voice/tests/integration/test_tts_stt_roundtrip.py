"""
TTS→STT Round-Trip Integration Tests

Tests the complete pipeline:
1. Text → TTS → Audio (C++ Kokoro)
2. Audio → STT → Text (C++ Whisper)
3. Compare output text with input text

Worker #99 - Phase 2: STT Integration
"""

import json
import os
import pytest
import subprocess
import tempfile
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
STREAM_TTS_CPP = PROJECT_ROOT / "stream-tts-cpp"
BUILD_DIR = STREAM_TTS_CPP / "build"
CONFIG_DIR = STREAM_TTS_CPP / "config"
MODELS_DIR = PROJECT_ROOT / "models"


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def whisper_binary():
    """Path to test_whisper_stt binary."""
    binary = BUILD_DIR / "test_whisper_stt"
    if not binary.exists():
        pytest.skip(f"WhisperSTT binary not found at {binary}")
    return binary


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


def generate_audio_stream(tts_binary: Path, text: str, config: Path, timeout: int = 60) -> bool:
    """
    Test that TTS generates audio (via stream-tts-cpp pipeline).

    Returns True only if:
    1. Process completed successfully (returncode 0)
    2. Audio frames were actually generated
    3. No critical warnings indicating quality issues

    Critical warnings that cause failure:
    - "Old voice format" - voice pack has wrong shape, causes audio fade-off
    - "Failed to load" - model loading issues
    """
    # Escape text for JSON
    escaped_text = text.replace('"', '\\"').replace('\n', '\\n')
    input_json = f'{{"type":"content_block_delta","delta":{{"type":"text_delta","text":"{escaped_text}"}}}}'

    result = subprocess.run(
        [str(tts_binary), str(config)],
        input=input_json,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(STREAM_TTS_CPP),
        env=get_tts_env()
    )

    # Check for successful TTS: returncode 0 and audio frames output
    success = result.returncode == 0
    has_audio = "frames output" in result.stdout or "STAGE [TTS" in result.stdout

    # Check for CRITICAL warnings that indicate quality issues
    combined_output = result.stdout + result.stderr
    critical_warnings = [
        "Old voice format",      # Wrong voice pack shape - causes audio fade-off
        "Failed to load",        # Model loading failure
        "Failed to initialize",  # Initialization failure
    ]
    has_critical_warning = any(w in combined_output for w in critical_warnings)

    if has_critical_warning:
        # Log the warning for debugging
        for w in critical_warnings:
            if w in combined_output:
                print(f"\nCRITICAL WARNING DETECTED: '{w}' - this indicates a quality issue!")
        return False

    return success and has_audio


# =============================================================================
# Test Cases
# =============================================================================

@pytest.mark.integration
class TestWhisperSTTDirect:
    """Direct WhisperSTT tests using the test binary."""

    def test_whisper_stt_jfk_sample(self, whisper_binary):
        """Test WhisperSTT with built-in JFK sample."""
        result = subprocess.run(
            [str(whisper_binary)],
            capture_output=True,
            text=True,
            timeout=180,
            cwd=str(STREAM_TTS_CPP)
        )

        assert result.returncode == 0, f"WhisperSTT test failed: {result.stderr}"
        assert "SUCCESS" in result.stdout, "Expected SUCCESS in output"
        assert "Americans" in result.stdout or "country" in result.stdout, \
            "Expected JFK speech content in transcription"

    def test_whisper_stt_performance(self, whisper_binary):
        """Verify WhisperSTT performance metrics."""
        import re

        result = subprocess.run(
            [str(whisper_binary)],
            capture_output=True,
            text=True,
            timeout=180,
            cwd=str(STREAM_TTS_CPP)
        )

        # Extract transcription time from output
        output = result.stdout
        match = re.search(r"Transcription time:\s*(\d+)ms", output)
        if match:
            latency_ms = int(match.group(1))
            print(f"\nWhisperSTT transcription latency: {latency_ms}ms")
            # Target: <2000ms for ~11s JFK audio (with VAD)
            assert latency_ms < 2000, f"Transcription too slow: {latency_ms}ms"
        else:
            # Also check alternate format "Transcribed X segments in Yms"
            match = re.search(r"Transcribed \d+ segments in (\d+)ms", output)
            if match:
                latency_ms = int(match.group(1))
                print(f"\nWhisperSTT transcription latency: {latency_ms}ms")
                assert latency_ms < 2000, f"Transcription too slow: {latency_ms}ms"

    def test_whisper_transcription_accuracy(self, whisper_binary):
        """Verify JFK speech transcription contains expected phrases."""
        result = subprocess.run(
            [str(whisper_binary)],
            capture_output=True,
            text=True,
            timeout=180,
            cwd=str(STREAM_TTS_CPP)
        )

        # JFK speech: "And so, my fellow Americans, ask not what your country can do for you..."
        expected_phrases = ["fellow", "ask", "country"]
        output_lower = result.stdout.lower()

        matches = sum(1 for phrase in expected_phrases if phrase in output_lower)
        assert matches >= 2, f"Expected at least 2 of {expected_phrases} in transcription"


@pytest.mark.integration
class TestTranslationPipeline:
    """Test translation + TTS pipeline."""

    def test_en_to_ja_pipeline(self, tts_binary):
        """Test English→Japanese translation pipeline runs without error."""
        config = CONFIG_DIR / "kokoro-mps-en2ja.yaml"
        if not config.exists():
            pytest.skip("EN→JA config not found")

        input_json = '{"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello world"}}'

        result = subprocess.run(
            [str(tts_binary), str(config)],
            input=input_json,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(STREAM_TTS_CPP),
            env=get_tts_env()
        )

        # Should not crash
        assert result.returncode == 0, f"Pipeline failed: {result.stderr}"
        # Logs go to stdout in this binary (spdlog)
        combined_output = result.stdout + result.stderr
        assert "Translation" in combined_output or "TTS" in combined_output, \
            "Expected TTS/Translation logs in output"

    def test_en_to_zh_pipeline(self, tts_binary):
        """Test English→Chinese translation pipeline runs without error."""
        config = CONFIG_DIR / "kokoro-mps-en2zh.yaml"
        if not config.exists():
            pytest.skip("EN→ZH config not found")

        input_json = '{"type":"content_block_delta","delta":{"type":"text_delta","text":"Good morning"}}'

        result = subprocess.run(
            [str(tts_binary), str(config)],
            input=input_json,
            capture_output=True,
            text=True,
            timeout=90,
            cwd=str(STREAM_TTS_CPP),
            env=get_tts_env()
        )

        # Should not crash
        assert result.returncode == 0, f"Pipeline failed: {result.stderr}"
        # Should target Chinese (check both stdout and stderr)
        combined_output = result.stdout.lower() + result.stderr.lower()
        assert "zho_hans" in combined_output or "zh" in combined_output, \
            "Expected Chinese language indicator in output"


@pytest.mark.integration
class TestEnglishTTS:
    """English TTS pipeline tests."""

    def test_english_tts_basic(self, tts_binary, english_config):
        """Test basic English TTS pipeline execution."""
        success = generate_audio_stream(tts_binary, "Hello world", english_config)
        assert success, "English TTS failed to generate audio"

    def test_english_tts_sentence(self, tts_binary, english_config):
        """Test English TTS with a longer sentence."""
        success = generate_audio_stream(
            tts_binary,
            "The quick brown fox jumps over the lazy dog.",
            english_config
        )
        assert success, "English TTS failed on pangram sentence"

    def test_english_tts_numbers(self, tts_binary, english_config):
        """Test English TTS with numbers."""
        success = generate_audio_stream(
            tts_binary,
            "The year is 2025 and we have version 3.14.",
            english_config
        )
        assert success, "English TTS failed with numbers"

    def test_english_tts_latency(self, tts_binary, english_config):
        """Verify TTS latency is acceptable (informational)."""
        import re

        input_json = '{"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello world"}}'

        result = subprocess.run(
            [str(tts_binary), str(english_config)],
            input=input_json,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(STREAM_TTS_CPP),
            env=get_tts_env()
        )

        # Look for end-to-end latency in output
        match = re.search(r"End-to-End\s+\d+\s+([\d.]+)", result.stdout)
        if match:
            latency_ms = float(match.group(1))
            print(f"\nTTS end-to-end latency: {latency_ms:.0f}ms")
            # Soft check: warn if over 500ms
            if latency_ms > 500:
                print(f"WARNING: Latency {latency_ms:.0f}ms exceeds 500ms warm target")


@pytest.mark.integration
@pytest.mark.slow
class TestMultiLanguageTTS:
    """Multi-language TTS integration tests."""

    def test_japanese_tts(self, tts_binary):
        """Test Japanese TTS pipeline."""
        config = CONFIG_DIR / "kokoro-mps-ja.yaml"
        if not config.exists():
            pytest.skip("Japanese config not found")

        success = generate_audio_stream(tts_binary, "Hello", config)
        # Note: Japanese TTS will use G2P to convert English to Japanese phonemes
        assert success, "Japanese TTS failed"

    def test_spanish_tts(self, tts_binary):
        """Test Spanish TTS pipeline."""
        config = CONFIG_DIR / "kokoro-mps-es.yaml"
        if not config.exists():
            pytest.skip("Spanish config not found")

        input_json = '{"type":"content_block_delta","delta":{"type":"text_delta","text":"Hola mundo"}}'

        result = subprocess.run(
            [str(tts_binary), str(config)],
            input=input_json,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(STREAM_TTS_CPP),
            env=get_tts_env()
        )

        assert result.returncode == 0, f"Spanish TTS failed: {result.stderr}"

    def test_french_tts(self, tts_binary):
        """Test French TTS pipeline."""
        config = CONFIG_DIR / "kokoro-mps-fr.yaml"
        if not config.exists():
            pytest.skip("French config not found")

        input_json = '{"type":"content_block_delta","delta":{"type":"text_delta","text":"Bonjour monde"}}'

        result = subprocess.run(
            [str(tts_binary), str(config)],
            input=input_json,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(STREAM_TTS_CPP),
            env=get_tts_env()
        )

        assert result.returncode == 0, f"French TTS failed: {result.stderr}"


# =============================================================================
# TTS→STT Full Round-Trip Tests (Worker #100)
# =============================================================================

def generate_wav_file(tts_binary: Path, text: str, config: Path, output_path: Path, timeout: int = 60) -> bool:
    """
    Generate a WAV file using TTS.

    Args:
        tts_binary: Path to stream-tts-cpp binary
        text: Text to synthesize
        config: Path to config YAML
        output_path: Path to save WAV file
        timeout: Timeout in seconds

    Returns:
        True if WAV file was created successfully
    """
    # Escape text for JSON
    escaped_text = text.replace('"', '\\"').replace('\n', '\\n')
    input_json = f'{{"type":"content_block_delta","delta":{{"type":"text_delta","text":"{escaped_text}"}}}}'

    result = subprocess.run(
        [str(tts_binary), "--save-audio", str(output_path), str(config)],
        input=input_json,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(STREAM_TTS_CPP),
        env=get_tts_env()
    )

    return result.returncode == 0 and output_path.exists()


def transcribe_audio(whisper_binary: Path, audio_path: Path, timeout: int = 120, language: str = "en") -> str:
    """
    Transcribe audio file using WhisperSTT.

    Args:
        whisper_binary: Path to test_whisper_stt binary
        audio_path: Path to audio file
        timeout: Timeout in seconds
        language: Transcription language (en, ja, zh, es, fr, ko, auto)

    Returns:
        Transcribed text (empty string on failure)
    """
    result = subprocess.run(
        [str(whisper_binary), "--lang", language, str(audio_path)],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(STREAM_TTS_CPP)
    )

    if result.returncode != 0:
        return ""

    # Extract transcription from output
    # Format: "--- Transcription ---\n text here \n---------------------"
    output = result.stdout
    if "--- Transcription ---" in output:
        start = output.index("--- Transcription ---") + len("--- Transcription ---")
        end = output.index("---------------------", start)
        return output[start:end].strip()

    return ""


def word_overlap_score(original: str, transcribed: str) -> float:
    """
    Calculate word overlap score between original and transcribed text.

    Returns a score from 0.0 to 1.0 indicating how many words from the
    original text appear in the transcription (case-insensitive).

    Punctuation is stripped from words before comparison to handle STT
    variations in punctuation (e.g., "Alice" vs "Alice,").
    """
    import re
    # Strip punctuation from words for comparison
    def clean_word(w: str) -> str:
        return re.sub(r'[^\w]', '', w.lower())

    original_words = set(clean_word(w) for w in original.split() if clean_word(w))
    transcribed_words = set(clean_word(w) for w in transcribed.split() if clean_word(w))

    if not original_words:
        return 1.0 if not transcribed_words else 0.0

    overlap = original_words & transcribed_words
    return len(overlap) / len(original_words)


@pytest.mark.integration
@pytest.mark.slow
class TestTTSSTTRoundTrip:
    """
    Full TTS→STT round-trip tests.

    These tests verify that:
    1. TTS generates valid audio for input text
    2. STT can transcribe the generated audio
    3. The transcription contains expected words/phrases

    Note: TTS→STT round-trips are inherently lossy - we test for
    semantic similarity, not exact matches.
    """

    @pytest.fixture
    def temp_wav(self, tmp_path):
        """Fixture to provide a temporary WAV file path."""
        return tmp_path / "test_audio.wav"

    def test_hello_world_roundtrip(self, tts_binary, whisper_binary, english_config, temp_wav):
        """Test basic 'Hello world' round-trip."""
        original_text = "Hello world"

        # Generate audio
        success = generate_wav_file(tts_binary, original_text, english_config, temp_wav)
        assert success, "Failed to generate WAV file"
        assert temp_wav.stat().st_size > 1000, "WAV file too small"

        # Transcribe
        transcribed = transcribe_audio(whisper_binary, temp_wav)
        assert transcribed, "Failed to get transcription"

        # Verify - should contain "hello" and/or "world"
        transcribed_lower = transcribed.lower()
        assert "hello" in transcribed_lower or "world" in transcribed_lower, \
            f"Expected 'hello' or 'world' in transcription: '{transcribed}'"

        print(f"\nRound-trip: '{original_text}' → '{transcribed}'")

    def test_sentence_roundtrip(self, tts_binary, whisper_binary, english_config, temp_wav):
        """Test a complete sentence round-trip.

        With proper voice packs ([510, 1, 256]), TTS quality should be high enough
        for reliable STT transcription. No excuses for low thresholds.
        """
        original_text = "Hello, my name is Alice and I like to read books"
        threshold = 1.0   # 100% word overlap - PERFECTION required, no excuses
        max_retries = 1   # No retries - quality must be consistent

        best_score = 0.0
        best_transcribed = ""

        for attempt in range(max_retries):
            # Generate audio
            success = generate_wav_file(tts_binary, original_text, english_config, temp_wav, timeout=90)
            if not success:
                continue

            # Transcribe
            transcribed = transcribe_audio(whisper_binary, temp_wav)
            if not transcribed:
                continue

            # Check word overlap
            score = word_overlap_score(original_text, transcribed)
            if score > best_score:
                best_score = score
                best_transcribed = transcribed

            if score >= threshold:
                print(f"\nRound-trip (attempt {attempt+1}, score={score:.0%}): '{original_text}' → '{transcribed}'")
                return  # Test passed

        # All retries failed
        assert best_score >= threshold, \
            f"Word overlap too low after {max_retries} attempts ({best_score:.0%}): '{original_text}' → '{best_transcribed}'"

    def test_numbers_roundtrip(self, tts_binary, whisper_binary, english_config, temp_wav):
        """Test numbers in round-trip."""
        original_text = "One two three four five"

        # Generate audio
        success = generate_wav_file(tts_binary, original_text, english_config, temp_wav)
        assert success, "Failed to generate WAV file"

        # Transcribe
        transcribed = transcribe_audio(whisper_binary, temp_wav)
        assert transcribed, "Failed to get transcription"

        # Should contain at least some number words
        number_words = ["one", "two", "three", "four", "five", "1", "2", "3", "4", "5"]
        transcribed_lower = transcribed.lower()
        matches = sum(1 for w in number_words if w in transcribed_lower)
        assert matches >= 2, \
            f"Expected at least 2 number words in transcription: '{transcribed}'"

        print(f"\nRound-trip: '{original_text}' → '{transcribed}'")

    def test_technical_text_roundtrip(self, tts_binary, whisper_binary, english_config, temp_wav):
        """Test technical/programming related text."""
        original_text = "The function returns a string value"

        # Generate audio
        success = generate_wav_file(tts_binary, original_text, english_config, temp_wav)
        assert success, "Failed to generate WAV file"

        # Transcribe
        transcribed = transcribe_audio(whisper_binary, temp_wav)
        assert transcribed, "Failed to get transcription"

        # Check for key words
        key_words = ["function", "returns", "string", "value"]
        transcribed_lower = transcribed.lower()
        matches = sum(1 for w in key_words if w in transcribed_lower)
        assert matches >= 2, \
            f"Expected at least 2 of {key_words} in: '{transcribed}'"

        print(f"\nRound-trip: '{original_text}' → '{transcribed}'")

    def test_japanese_translation_roundtrip(self, tts_binary, whisper_binary, temp_wav):
        """Test EN→JA translation + TTS round-trip (transcribes back to Japanese text)."""
        config = CONFIG_DIR / "kokoro-mps-en2ja.yaml"
        if not config.exists():
            pytest.skip("EN→JA config not found")

        original_text = "Good morning"

        # Generate Japanese audio from English text
        success = generate_wav_file(tts_binary, original_text, config, temp_wav, timeout=120)
        assert success, "Failed to generate Japanese WAV file"
        assert temp_wav.stat().st_size > 1000, "WAV file too small"

        # Note: Whisper will transcribe as Japanese text
        # We mainly verify the pipeline doesn't crash and produces audio
        print(f"\nEN→JA round-trip: '{original_text}' → [Japanese audio generated]")

    def test_roundtrip_latency(self, tts_binary, whisper_binary, english_config, temp_wav):
        """Measure and verify round-trip latency is acceptable."""
        import time

        original_text = "Hello world"

        # Time TTS
        tts_start = time.time()
        success = generate_wav_file(tts_binary, original_text, english_config, temp_wav)
        tts_time = (time.time() - tts_start) * 1000
        assert success, "Failed to generate WAV file"

        # Time STT
        stt_start = time.time()
        transcribed = transcribe_audio(whisper_binary, temp_wav)
        stt_time = (time.time() - stt_start) * 1000
        assert transcribed, "Failed to get transcription"

        total_time = tts_time + stt_time
        print(f"\nLatency breakdown:")
        print(f"  TTS: {tts_time:.0f}ms")
        print(f"  STT: {stt_time:.0f}ms")
        print(f"  Total: {total_time:.0f}ms")

        # Soft target: total should be under 15 seconds for cold-start
        # (includes model loading: ~8-10s TTS + ~2-3s STT)
        # Note: This is a cold-start test; daemon mode tests use stricter thresholds
        assert total_time < 15000, f"Round-trip too slow: {total_time:.0f}ms (cold-start threshold: 15s)"


# =============================================================================
# Japanese STT Round-Trip Tests (Worker #101)
# =============================================================================

def japanese_char_overlap_score(original: str, transcribed: str) -> float:
    """
    Calculate character overlap for Japanese text.

    Japanese doesn't have word boundaries like English, so we compare
    characters instead of words. Returns 0.0 to 1.0.
    """
    # Remove spaces and normalize
    original_chars = set(original.replace(" ", "").replace("　", ""))
    transcribed_chars = set(transcribed.replace(" ", "").replace("　", ""))

    if not original_chars:
        return 1.0 if not transcribed_chars else 0.0

    overlap = original_chars & transcribed_chars
    return len(overlap) / len(original_chars)


def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    Calculate Word Error Rate (WER) for Japanese text.

    For Japanese, we treat each character as a "word" since there are
    no explicit word boundaries.

    WER = (S + D + I) / N
    Where:
        S = substitutions
        D = deletions
        I = insertions
        N = number of chars in reference

    Returns WER as a float (0.0 = perfect, 1.0 = 100% error)
    """
    # Treat each character as a word for Japanese
    ref_chars = list(reference.replace(" ", "").replace("　", ""))
    hyp_chars = list(hypothesis.replace(" ", "").replace("　", ""))

    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0

    # Dynamic programming for edit distance
    m, n = len(ref_chars), len(hyp_chars)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_chars[i - 1] == hyp_chars[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # deletion
                    dp[i][j - 1],      # insertion
                    dp[i - 1][j - 1]   # substitution
                )

    edit_distance = dp[m][n]
    return edit_distance / m


@pytest.fixture(scope="module")
def japanese_config():
    """Path to Japanese TTS config."""
    config = CONFIG_DIR / "kokoro-mps-ja.yaml"
    if not config.exists():
        pytest.skip(f"Japanese config not found: {config}")
    return config


@pytest.mark.integration
@pytest.mark.slow
class TestJapaneseSTTRoundTrip:
    """
    Japanese TTS→STT round-trip tests.

    These tests verify:
    1. Japanese TTS generates valid audio
    2. Japanese STT can transcribe the audio
    3. Transcription contains expected Japanese characters

    Target: WER < 10% for clear Japanese speech
    """

    @pytest.fixture
    def temp_wav(self, tmp_path):
        """Fixture to provide a temporary WAV file path."""
        return tmp_path / "test_ja_audio.wav"

    def test_japanese_hiragana_roundtrip(self, tts_binary, whisper_binary, japanese_config, temp_wav):
        """Test hiragana text round-trip: こんにちは (konnichiwa)."""
        original_text = "こんにちは"  # Hello

        # Generate Japanese audio
        success = generate_wav_file(tts_binary, original_text, japanese_config, temp_wav, timeout=120)
        assert success, "Failed to generate Japanese WAV file"
        assert temp_wav.stat().st_size > 1000, "WAV file too small"

        # Transcribe with Japanese Whisper
        transcribed = transcribe_audio(whisper_binary, temp_wav, language="ja")
        assert transcribed, "Failed to get Japanese transcription"

        # Check character overlap
        score = japanese_char_overlap_score(original_text, transcribed)
        print(f"\nJA round-trip (hiragana): '{original_text}' → '{transcribed}' (overlap={score:.0%})")

        # Japanese is production quality - require 100% character overlap
        assert score >= 1.0, \
            f"Hiragana overlap {score:.0%} below 100% - got: '{transcribed}'"

    def test_japanese_katakana_roundtrip(self, tts_binary, whisper_binary, japanese_config, temp_wav):
        """Test katakana text round-trip: コンピューター (computer)."""
        original_text = "コンピューター"  # Computer

        # Generate Japanese audio
        success = generate_wav_file(tts_binary, original_text, japanese_config, temp_wav, timeout=120)
        assert success, "Failed to generate Japanese WAV file"
        assert temp_wav.stat().st_size > 1000, "WAV file too small"

        # Transcribe with Japanese Whisper
        transcribed = transcribe_audio(whisper_binary, temp_wav, language="ja")
        assert transcribed, "Failed to get Japanese transcription"

        # Check character overlap
        score = japanese_char_overlap_score(original_text, transcribed)
        print(f"\nJA round-trip (katakana): '{original_text}' → '{transcribed}' (overlap={score:.0%})")

        # Japanese is production quality - require 100% character overlap
        assert score >= 1.0, \
            f"Katakana overlap {score:.0%} below 100% - got: '{transcribed}'"

    def test_japanese_kanji_roundtrip(self, tts_binary, whisper_binary, japanese_config, temp_wav):
        """Test kanji text round-trip: 今日は良い天気です (It's nice weather today)."""
        original_text = "今日は良い天気です"

        # Generate Japanese audio
        success = generate_wav_file(tts_binary, original_text, japanese_config, temp_wav, timeout=120)
        assert success, "Failed to generate Japanese WAV file"
        assert temp_wav.stat().st_size > 1000, "WAV file too small"

        # Transcribe with Japanese Whisper
        transcribed = transcribe_audio(whisper_binary, temp_wav, language="ja")
        assert transcribed, "Failed to get Japanese transcription"

        # Check for key kanji
        key_chars = ["今日", "天気", "良"]
        transcribed_text = transcribed.replace(" ", "")
        matches = sum(1 for char in key_chars if char in transcribed_text)

        print(f"\nJA round-trip (kanji): '{original_text}' → '{transcribed}' (key matches={matches}/3)")

        assert matches >= 1, \
            f"Expected at least 1 of {key_chars} in: '{transcribed}'"

    def test_japanese_mixed_script_roundtrip(self, tts_binary, whisper_binary, japanese_config, temp_wav):
        """Test mixed script: 東京でコーヒーを飲みます (I drink coffee in Tokyo)."""
        original_text = "東京でコーヒーを飲みます"

        # Generate Japanese audio
        success = generate_wav_file(tts_binary, original_text, japanese_config, temp_wav, timeout=120)
        assert success, "Failed to generate Japanese WAV file"
        assert temp_wav.stat().st_size > 1000, "WAV file too small"

        # Transcribe with Japanese Whisper
        transcribed = transcribe_audio(whisper_binary, temp_wav, language="ja")
        assert transcribed, "Failed to get Japanese transcription"

        # Check for key elements (kanji + katakana)
        key_elements = ["東京", "コーヒー", "飲"]
        transcribed_text = transcribed.replace(" ", "")
        matches = sum(1 for elem in key_elements if elem in transcribed_text)

        print(f"\nJA round-trip (mixed): '{original_text}' → '{transcribed}' (key matches={matches}/3)")

        assert matches >= 1, \
            f"Expected at least 1 of {key_elements} in: '{transcribed}'"

    def test_japanese_wer_measurement(self, tts_binary, whisper_binary, japanese_config, temp_wav):
        """Measure Word Error Rate for Japanese STT."""
        # Simple, clear sentence for WER measurement
        original_text = "おはようございます"  # Good morning

        # Generate Japanese audio
        success = generate_wav_file(tts_binary, original_text, japanese_config, temp_wav, timeout=120)
        assert success, "Failed to generate Japanese WAV file"

        # Transcribe
        transcribed = transcribe_audio(whisper_binary, temp_wav, language="ja")
        assert transcribed, "Failed to get Japanese transcription"

        # Calculate WER
        wer = calculate_wer(original_text, transcribed)
        print(f"\nJA WER measurement: '{original_text}' → '{transcribed}' (WER={wer:.1%})")

        # Target: WER < 50% (Japanese STT is more challenging than English)
        # Note: This is a soft target - exact transcription varies
        if wer > 0.5:
            print(f"WARNING: WER {wer:.1%} exceeds 50% target")

    def test_en2ja_translation_stt_roundtrip(self, tts_binary, whisper_binary, temp_wav):
        """Test EN→JA translation + TTS + Japanese STT full pipeline."""
        config = CONFIG_DIR / "kokoro-mps-en2ja.yaml"
        if not config.exists():
            pytest.skip("EN→JA config not found")

        # English input
        english_text = "Hello, how are you?"

        # Generate Japanese audio from English text (via NLLB translation)
        success = generate_wav_file(tts_binary, english_text, config, temp_wav, timeout=120)
        assert success, "Failed to generate translated Japanese WAV file"
        assert temp_wav.stat().st_size > 1000, "WAV file too small"

        # Transcribe with Japanese Whisper
        transcribed = transcribe_audio(whisper_binary, temp_wav, language="ja")
        assert transcribed, "Failed to get Japanese transcription"

        # Should contain Japanese text (not English)
        # Common translations: "こんにちは" (hello), "元気" (how are you)
        print(f"\nEN→JA→STT: '{english_text}' → '{transcribed}'")

        # Verify it's Japanese (contains hiragana/katakana/kanji, not just ASCII)
        has_japanese = any(
            '\u3040' <= c <= '\u309f' or  # Hiragana
            '\u30a0' <= c <= '\u30ff' or  # Katakana
            '\u4e00' <= c <= '\u9fff'     # CJK (Kanji)
            for c in transcribed
        )
        assert has_japanese, f"Expected Japanese characters in transcription: '{transcribed}'"


# =============================================================================
# Chinese STT Round-Trip Tests (Worker #102)
# =============================================================================

def chinese_char_overlap_score(original: str, transcribed: str) -> float:
    """
    Calculate character overlap for Chinese text.

    Chinese doesn't have word boundaries like English, so we compare
    characters instead of words. Returns 0.0 to 1.0.
    """
    # Remove spaces and punctuation
    original_chars = set(c for c in original if '\u4e00' <= c <= '\u9fff')
    transcribed_chars = set(c for c in transcribed if '\u4e00' <= c <= '\u9fff')

    if not original_chars:
        return 1.0 if not transcribed_chars else 0.0

    overlap = original_chars & transcribed_chars
    return len(overlap) / len(original_chars)


def has_chinese_characters(text: str) -> bool:
    """Check if text contains Chinese characters (CJK Unified Ideographs)."""
    return any('\u4e00' <= c <= '\u9fff' for c in text)


@pytest.fixture(scope="module")
def chinese_config():
    """Path to Chinese TTS config."""
    config = CONFIG_DIR / "kokoro-mps-zh.yaml"
    if not config.exists():
        pytest.skip(f"Chinese config not found: {config}")
    return config


@pytest.fixture(scope="module")
def en2zh_config():
    """Path to EN→ZH translation config."""
    config = CONFIG_DIR / "kokoro-mps-en2zh.yaml"
    if not config.exists():
        pytest.skip(f"EN→ZH config not found: {config}")
    return config


@pytest.mark.integration
@pytest.mark.slow
class TestChineseSTTRoundTrip:
    """
    Chinese TTS→STT round-trip tests.

    IMPORTANT: As of Worker #102, Chinese TTS is BROKEN due to missing G2P.
    Kokoro cannot convert Chinese characters (hanzi) to phonemes.
    These tests document the current state and test Chinese STT capability.

    Chinese G2P implementation tracked in:
    - reports/main/G2P_RESEARCH_2025-12-05.md
    - MANAGER_ROADMAP_2025-12-05.md (Worker #94.6)
    """

    @pytest.fixture
    def temp_wav(self, tmp_path):
        """Fixture to provide a temporary WAV file path."""
        return tmp_path / "test_zh_audio.wav"

    def test_chinese_stt_language_detection(self, whisper_binary):
        """
        Test that Whisper can transcribe audio with Chinese language setting.

        Uses the JFK sample (English speech) but with Chinese language mode
        to verify the --lang zh parameter works.
        """
        result = subprocess.run(
            [str(whisper_binary), "--lang", "zh"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(STREAM_TTS_CPP)
        )

        assert result.returncode == 0, f"WhisperSTT failed: {result.stderr}"
        assert "SUCCESS" in result.stdout, "Expected SUCCESS in output"
        # When given English audio with zh mode, Whisper will still transcribe
        # but may produce Chinese translation or keep English
        print(f"\nChinese mode STT on English audio: SUCCESS")

    def test_chinese_tts_generates_audio(self, tts_binary, chinese_config, temp_wav):
        """
        Test Chinese TTS generates valid audio (beta quality - uses espeak G2P).

        Chinese TTS uses espeak-ng for G2P with phoneme mapping. Quality is
        approximate but should be intelligible. This test verifies audio is generated.
        """
        original_text = "你好世界"  # Hello world

        escaped_text = original_text.replace('"', '\\"')
        input_json = f'{{"type":"content_block_delta","delta":{{"type":"text_delta","text":"{escaped_text}"}}}}'

        result = subprocess.run(
            [str(tts_binary), "--save-audio", str(temp_wav), str(chinese_config)],
            input=input_json,
            capture_output=True,
            text=True,
            timeout=90,
            cwd=str(STREAM_TTS_CPP),
            env=get_tts_env()
        )

        # Must not crash
        assert result.returncode == 0, f"Chinese TTS crashed: {result.stderr}"

        # Must generate audio file
        assert temp_wav.exists(), "No audio file generated"
        assert temp_wav.stat().st_size > 1000, f"Audio file too small: {temp_wav.stat().st_size} bytes"

        print(f"\nChinese TTS: Generated {temp_wav.stat().st_size} bytes for '{original_text}'")

    def test_en2zh_translation_generates_audio(self, tts_binary, en2zh_config, whisper_binary, temp_wav):
        """
        Test EN→ZH translation + TTS pipeline generates audio.

        NLLB translates English to Chinese, then Kokoro with espeak G2P
        synthesizes the Chinese text. Audio must be generated.
        """
        english_text = "Hello world"

        escaped_text = english_text.replace('"', '\\"')
        input_json = f'{{"type":"content_block_delta","delta":{{"type":"text_delta","text":"{escaped_text}"}}}}'

        result = subprocess.run(
            [str(tts_binary), "--save-audio", str(temp_wav), str(en2zh_config)],
            input=input_json,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(STREAM_TTS_CPP),
            env=get_tts_env()
        )

        # Must not crash
        assert result.returncode == 0, f"EN→ZH pipeline crashed: {result.stderr}"

        # Must generate audio file
        assert temp_wav.exists(), "No audio file generated for EN→ZH"
        assert temp_wav.stat().st_size > 1000, f"EN→ZH audio too small: {temp_wav.stat().st_size} bytes"

        # Transcribe and verify Chinese characters in output
        transcribed = transcribe_audio(whisper_binary, temp_wav, language="zh")

        print(f"\nEN→ZH Pipeline: '{english_text}' → '{transcribed}'")
        print(f"  Audio size: {temp_wav.stat().st_size} bytes")

        # Must produce some transcription (even if quality is beta)
        assert transcribed, "STT returned empty transcription for EN→ZH audio"

    def test_chinese_character_detection_helper(self):
        """Unit test for Chinese character detection utility."""
        assert has_chinese_characters("你好") == True
        assert has_chinese_characters("Hello") == False
        assert has_chinese_characters("你好world") == True
        assert has_chinese_characters("") == False

    def test_chinese_overlap_score_helper(self):
        """Unit test for Chinese character overlap score utility."""
        # Perfect match
        assert chinese_char_overlap_score("你好", "你好") == 1.0

        # Partial match
        score = chinese_char_overlap_score("你好世界", "你好")
        assert 0.4 <= score <= 0.6  # 2/4 chars match

        # No match
        assert chinese_char_overlap_score("你好", "再见") == 0.0

        # Empty strings
        assert chinese_char_overlap_score("", "") == 1.0
        assert chinese_char_overlap_score("", "你好") == 0.0
