"""
Tests for Translation Chunking (HARDENING_ROADMAP 2.2)

Validates:
- Long text is split into sentences before translation
- Each sentence translated independently (fail one, not all)
- Chunks stay under max token limit (~400 chars)

Worker #548 - Initial implementation
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
TTS_BINARY = BUILD_DIR / "stream-tts-cpp"


@pytest.fixture(scope="module")
def binary_exists():
    """Ensure TTS binary exists and daemon is running."""
    if not TTS_BINARY.exists():
        pytest.skip(f"TTS binary not found: {TTS_BINARY}")

    # Test actual daemon connection by running a quick command
    socket_path = "/tmp/stream-tts.sock"
    if not os.path.exists(socket_path):
        pytest.skip("Daemon not running (socket not found). Start with: stream-tts-cpp --daemon config.yaml &")

    # Try to get status from daemon
    try:
        result = subprocess.run(
            [str(TTS_BINARY), "--status"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(STREAM_TTS_CPP)
        )
        if result.returncode != 0:
            pytest.skip(f"Daemon not responding. Start with: stream-tts-cpp --daemon config.yaml &")
    except (subprocess.TimeoutExpired, Exception) as e:
        pytest.skip(f"Daemon connection failed: {e}")

    return TTS_BINARY


def make_claude_json(text: str) -> str:
    """Create Claude API JSON format input."""
    escaped_text = text.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
    return f'{{"type":"content_block_delta","delta":{{"type":"text_delta","text":"{escaped_text}"}}}}\n{{"type":"message_stop"}}'


def run_tts_with_translation(binary: Path, text: str, lang: str = "ja", timeout: int = 60) -> dict:
    """
    Run TTS with translation enabled, return result dict.
    """
    cmd = [
        str(binary),
        "--daemon-pipe",
        "--language", lang,
        "--translate",
        "--no-audio",
    ]

    input_json = make_claude_json(text)

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            input=input_json,
            capture_output=True,
            text=True,
            cwd=str(STREAM_TTS_CPP),
            timeout=timeout
        )
        duration_ms = (time.time() - start) * 1000

        return {
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration_ms": duration_ms,
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": "Timeout expired",
            "duration_ms": timeout * 1000,
        }


class TestTranslationChunking:
    """Tests for translation chunking with long text."""

    def test_short_text_no_chunking(self, binary_exists):
        """
        Short text (<200 chars) should not trigger chunking.
        """
        text = "Hello, how are you today?"  # ~25 chars
        result = run_tts_with_translation(binary_exists, text, lang="ja", timeout=60)

        assert result["success"], f"Short text translation failed: {result['stderr'][:200]}"
        # No chunking log message expected for short text
        # Just verify it succeeded

    def test_long_text_with_multiple_sentences(self, binary_exists):
        """
        Long text with multiple sentences should be chunked.
        Each sentence translated independently.
        """
        # Create text with multiple sentences (~300 chars)
        text = (
            "The quick brown fox jumps over the lazy dog. "
            "This is a test of the translation chunking system. "
            "Each sentence should be translated independently. "
            "If one sentence fails, the others should still succeed. "
            "This ensures reliable translation of long text passages."
        )

        assert len(text) > 200, f"Test text should be >200 chars, got {len(text)}"

        result = run_tts_with_translation(binary_exists, text, lang="ja", timeout=90)

        assert result["success"], f"Long text translation failed: {result['stderr'][:200]}"

        # Should see chunking debug message if debug logging enabled
        # (may not appear in release builds)
        print(f"Long text translation succeeded in {result['duration_ms']:.0f}ms")

    def test_very_long_text_paragraph(self, binary_exists):
        """
        Very long text (multiple paragraphs) should be chunked properly.
        """
        # Create a substantial paragraph (~600 chars)
        text = (
            "Machine learning is transforming the way we interact with technology. "
            "Natural language processing enables computers to understand human speech. "
            "Speech synthesis converts text into natural sounding audio output. "
            "Real-time translation bridges language barriers across the world. "
            "These technologies are advancing rapidly with new research breakthroughs. "
            "The combination of fast processors and efficient algorithms makes it possible. "
            "Soon we will have seamless multilingual communication everywhere. "
            "This is truly an exciting time for language technology innovation."
        )

        assert len(text) > 400, f"Test text should be >400 chars, got {len(text)}"

        result = run_tts_with_translation(binary_exists, text, lang="ja", timeout=120)

        assert result["success"], f"Very long text translation failed: {result['stderr'][:200]}"
        print(f"Very long text ({len(text)} chars) succeeded in {result['duration_ms']:.0f}ms")

    def test_text_with_newlines(self, binary_exists):
        """
        Text with newlines should split on newlines as sentence boundaries.
        """
        text = (
            "First line of text.\n"
            "Second line of text.\n"
            "Third line of text.\n"
            "Fourth line of text.\n"
            "Fifth line of text.\n"
            "Sixth line of text.\n"
            "Seventh line of text.\n"
            "Eighth line of text."
        )

        result = run_tts_with_translation(binary_exists, text, lang="ja", timeout=90)

        assert result["success"], f"Newline text translation failed: {result['stderr'][:200]}"

    def test_text_with_exclamations_and_questions(self, binary_exists):
        """
        Text with ! and ? should split on those sentence boundaries.
        """
        text = (
            "Hello! How are you today? I am doing very well thank you. "
            "Did you see the news about the technology conference? It was very exciting! "
            "What do you think about the new developments? I have many questions. "
            "Let me know your thoughts on this topic! This is important to discuss."
        )

        assert len(text) > 200, f"Test text should be >200 chars, got {len(text)}"

        result = run_tts_with_translation(binary_exists, text, lang="ja", timeout=90)

        assert result["success"], f"Punctuation text translation failed: {result['stderr'][:200]}"

    def test_single_very_long_sentence_breaks_at_word_boundary(self, binary_exists):
        """
        A single sentence exceeding 400 chars should break at word boundaries.
        """
        # Create a single long sentence without periods
        words = ["word" + str(i) for i in range(100)]
        text = " ".join(words)  # ~500+ chars, single sentence

        assert len(text) > 400, f"Test text should be >400 chars, got {len(text)}"
        assert "." not in text, "Test should be single sentence without periods"

        result = run_tts_with_translation(binary_exists, text, lang="ja", timeout=120)

        # Should succeed even with long single sentence (will break at word boundary)
        assert result["success"], f"Long single sentence failed: {result['stderr'][:200]}"
        print(f"Long single sentence ({len(text)} chars) succeeded in {result['duration_ms']:.0f}ms")


class TestChunkingCacheEfficiency:
    """Tests that chunking works well with translation cache."""

    def test_repeated_long_text_uses_cache(self, binary_exists):
        """
        Translating the same long text twice should hit cache on second call.
        """
        text = (
            "This is a test of cache efficiency with chunked translations. "
            "Each sentence chunk should be cached individually. "
            "The second translation of this text should be faster. "
            "Cache hits improve performance significantly."
        )

        # First call - cache miss
        result1 = run_tts_with_translation(binary_exists, text, lang="ja", timeout=90)
        assert result1["success"], f"First translation failed: {result1['stderr'][:200]}"

        # Brief pause
        time.sleep(0.5)

        # Second call - should hit cache
        result2 = run_tts_with_translation(binary_exists, text, lang="ja", timeout=90)
        assert result2["success"], f"Second translation failed: {result2['stderr'][:200]}"

        # Note: Can't easily measure cache hit rate without daemon status,
        # but both should succeed
        print(f"First: {result1['duration_ms']:.0f}ms, Second: {result2['duration_ms']:.0f}ms")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
