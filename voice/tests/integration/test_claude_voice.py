#!/usr/bin/env python3
"""
Integration tests for Claude Code Voice Integration (Phase 7).
Tests sentence chunking, latency targets, and multilingual support.

Copyright 2025 Andrew Yates. All rights reserved.
"""

import os
import sys
import json
import subprocess
import tempfile
import time
import pytest

# Add scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))


def get_tts_binary() -> str:
    """Get path to TTS binary."""
    base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    binary = os.path.join(base, 'stream-tts-cpp', 'build', 'stream-tts-cpp')
    return binary


def tts_binary_exists() -> bool:
    """Check if TTS binary exists."""
    return os.path.exists(get_tts_binary())


# Sample Claude JSON messages for testing
SAMPLE_MESSAGES = [
    # Text delta - Claude streaming
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {
            "type": "text_delta",
            "text": "I'll help you implement this feature. "
        }
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {
            "type": "text_delta",
            "text": "First, let me check the existing code."
        }
    },
    # Complete message with text block
    {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "The implementation looks correct. I've verified all the tests pass."
                }
            ]
        }
    },
]


class TestSentenceBuffer:
    """Test sentence boundary detection and chunking."""

    def test_import(self):
        """Test that the module can be imported."""
        from claude_to_voice import SentenceBuffer, ClaudeJSONParser, TTSDaemon
        assert SentenceBuffer is not None
        assert ClaudeJSONParser is not None
        assert TTSDaemon is not None

    def test_sentence_chunking_english(self):
        """Test English sentence boundary detection."""
        from claude_to_voice import SentenceBuffer

        buffer = SentenceBuffer()
        sentences = []

        # Add text with sentence boundaries
        for s in buffer.add_text("This is the first sentence. "):
            sentences.append(s)
        for s in buffer.add_text("This is the second sentence. "):
            sentences.append(s)
        for s in buffer.add_text("And this is the third sentence."):
            sentences.append(s)

        # Flush remaining
        remaining = buffer.flush()
        if remaining:
            sentences.append(remaining)

        assert len(sentences) >= 2, f"Expected at least 2 sentences, got {len(sentences)}"

    def test_sentence_chunking_japanese(self):
        """Test Japanese sentence boundary detection."""
        from claude_to_voice import SentenceBuffer

        buffer = SentenceBuffer()
        sentences = []

        # Japanese text with sentence boundaries (。)
        text = "これは最初の文です。これは2番目の文です。これは3番目の文です。"
        for s in buffer.add_text(text):
            sentences.append(s)

        remaining = buffer.flush()
        if remaining:
            sentences.append(remaining)

        assert len(sentences) >= 2, f"Expected at least 2 Japanese sentences, got {len(sentences)}"

    def test_minimum_length_filter(self):
        """Test that short fragments are not emitted."""
        from claude_to_voice import SentenceBuffer

        buffer = SentenceBuffer()
        sentences = []

        # Short fragments should be filtered
        for s in buffer.add_text("Hi. OK. Yes. "):
            sentences.append(s)

        # These short phrases should not produce sentences
        assert len(sentences) == 0, f"Short fragments should be filtered, got {sentences}"

    def test_longer_sentence_emitted(self):
        """Test that sentences meeting minimum length are emitted."""
        from claude_to_voice import SentenceBuffer

        buffer = SentenceBuffer()
        sentences = []

        # This sentence should be long enough
        for s in buffer.add_text("This is a longer sentence that should definitely be spoken aloud. "):
            sentences.append(s)

        assert len(sentences) == 1, f"Expected 1 sentence, got {len(sentences)}"


class TestClaudeJSONParser:
    """Test Claude JSON parsing."""

    def test_extract_text_delta(self):
        """Test extracting text from streaming deltas."""
        from claude_to_voice import ClaudeJSONParser

        parser = ClaudeJSONParser()
        msg = SAMPLE_MESSAGES[0]  # text_delta

        texts = list(parser.extract_text(msg))
        assert len(texts) == 1
        assert "implement this feature" in texts[0]

    def test_extract_text_message(self):
        """Test extracting text from complete messages."""
        from claude_to_voice import ClaudeJSONParser

        parser = ClaudeJSONParser()
        msg = SAMPLE_MESSAGES[2]  # complete message

        texts = list(parser.extract_text(msg))
        assert len(texts) == 1
        assert "verified" in texts[0]

    def test_clean_text_code_blocks(self):
        """Test that code blocks are removed."""
        from claude_to_voice import ClaudeJSONParser

        parser = ClaudeJSONParser()
        text = "Here is code: ```python\nprint('hello')\n``` and more text."
        cleaned = parser.clean_text(text)
        assert "```" not in cleaned
        assert "print" not in cleaned
        assert "more text" in cleaned

    def test_clean_text_markdown(self):
        """Test that markdown formatting is removed."""
        from claude_to_voice import ClaudeJSONParser

        parser = ClaudeJSONParser()
        text = "This is **bold** and *italic* and `code`."
        cleaned = parser.clean_text(text)
        assert "**" not in cleaned
        assert "*" not in cleaned
        assert "`" not in cleaned
        assert "bold" in cleaned

    def test_skip_system_reminders(self):
        """Test that system reminders are skipped."""
        from claude_to_voice import ClaudeJSONParser

        parser = ClaudeJSONParser()
        assert parser.should_skip("<system-reminder>")
        assert parser.should_skip("</system-reminder>")
        assert parser.should_skip("Co-Authored-By: Claude")
        assert not parser.should_skip("Regular text message")


class TestTTSIntegration:
    """Integration tests requiring TTS binary."""

    @pytest.mark.skipif(not tts_binary_exists(), reason="TTS binary not built")
    def test_tts_binary_help(self):
        """Test that TTS binary responds to --help."""
        result = subprocess.run(
            [get_tts_binary(), '--help'],
            capture_output=True,
            text=True,
            timeout=10
        )
        # Binary should output help text
        assert 'speak' in result.stdout.lower() or 'speak' in result.stderr.lower()

    @pytest.mark.skipif(not tts_binary_exists(), reason="TTS binary not built")
    def test_tts_binary_list_languages(self):
        """Test that TTS binary can list supported languages."""
        result = subprocess.run(
            [get_tts_binary(), '--list-languages'],
            capture_output=True,
            text=True,
            timeout=10
        )
        # Should list language codes
        output = result.stdout + result.stderr
        # At minimum should show English
        assert 'en' in output.lower()


class TestLatencyTarget:
    """Test latency targets for Phase 7."""

    def test_sentence_processing_overhead(self):
        """Test that sentence processing adds minimal overhead."""
        from claude_to_voice import SentenceBuffer, ClaudeJSONParser

        parser = ClaudeJSONParser()
        buffer = SentenceBuffer()

        # Time processing 100 messages
        start = time.perf_counter()

        for _ in range(100):
            for msg in SAMPLE_MESSAGES:
                for text in parser.extract_text(msg):
                    cleaned = parser.clean_text(text)
                    if cleaned:
                        for _ in buffer.add_text(cleaned + ' '):
                            pass

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Processing 100 messages should take less than 100ms (< 1ms each)
        assert elapsed_ms < 100, f"Processing overhead too high: {elapsed_ms:.1f}ms for 100 messages"

    @pytest.mark.skipif(not tts_binary_exists(), reason="TTS binary not built")
    def test_latency_target_english(self):
        """
        Test that English TTS meets latency target (non-translated).

        Note: Cold start (first invocation) includes model loading time (~9s).
        Warm latency (daemon mode) is ~150ms. This test accepts cold start time
        but warns if it exceeds reasonable bounds.
        """
        binary = get_tts_binary()

        # Speak a short English sentence
        start = time.perf_counter()
        result = subprocess.run(
            [binary, '--speak', 'Hello, this is a test sentence.', '--lang', 'en'],
            capture_output=True,
            timeout=60
        )
        latency_ms = (time.perf_counter() - start) * 1000

        # Cold start includes model loading (~9s on M4 Max)
        # Daemon mode (warm) is ~150ms
        # Allow up to 30s for cold start to account for slower systems
        assert latency_ms < 30000, f"TTS cold start too slow: {latency_ms:.0f}ms (max: 30s)"

        # Report actual latency for monitoring
        if latency_ms > 10000:
            print(f"Info: Cold start latency {latency_ms:.0f}ms (model loading)")
        elif latency_ms > 500:
            print(f"Warning: TTS latency {latency_ms:.0f}ms exceeds 500ms target (may be cold start)")
        else:
            print(f"TTS latency: {latency_ms:.0f}ms (meets <500ms target)")


class TestMultilingualSupport:
    """Test multilingual support (ja, zh, hi)."""

    @pytest.mark.skipif(not tts_binary_exists(), reason="TTS binary not built")
    def test_japanese_language_flag(self):
        """Test that --lang ja is accepted."""
        binary = get_tts_binary()
        result = subprocess.run(
            [binary, '--speak', 'Hello', '--lang', 'ja', '--translate'],
            capture_output=True,
            timeout=60
        )
        # Should not crash (may fail if translation not available)
        # Accept either success or graceful failure
        assert result.returncode in [0, 1], f"Unexpected return code: {result.returncode}"

    @pytest.mark.skipif(not tts_binary_exists(), reason="TTS binary not built")
    def test_chinese_language_flag(self):
        """Test that --lang zh is accepted."""
        binary = get_tts_binary()
        result = subprocess.run(
            [binary, '--speak', 'Hello', '--lang', 'zh', '--translate'],
            capture_output=True,
            timeout=60
        )
        assert result.returncode in [0, 1], f"Unexpected return code: {result.returncode}"


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_full_pipeline_mock(self):
        """Test the full pipeline with mock input."""
        from claude_to_voice import ClaudeJSONParser, SentenceBuffer

        parser = ClaudeJSONParser()
        buffer = SentenceBuffer()
        sentences_to_speak = []

        # Simulate processing Claude output
        for msg in SAMPLE_MESSAGES:
            for text in parser.extract_text(msg):
                cleaned = parser.clean_text(text)
                if cleaned:
                    for sentence in buffer.add_text(cleaned + ' '):
                        sentences_to_speak.append(sentence)

        remaining = buffer.flush()
        if remaining:
            sentences_to_speak.append(remaining)

        # Should have extracted some sentences
        assert len(sentences_to_speak) >= 1, "No sentences extracted from mock input"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
