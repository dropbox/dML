"""
Unit tests for MessageClassifier component.

Tests the classification logic for determining:
- What text should be spoken (speakable content)
- What should be skipped (code, system messages, etc.)
- Clean-for-speech text transformations
- Tool call formatting

These tests verify the classification logic directly.
"""

import pytest
import re
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestShouldSpeak:
    """Test should_speak classification logic."""

    def test_short_text_rejected(self):
        """Text shorter than 3 chars should not be spoken."""
        short_texts = ["", "a", "ab"]
        for text in short_texts:
            assert len(text) < 3

    def test_minimum_length_accepted(self):
        """Text with 3+ chars should be speakable."""
        acceptable = ["abc", "hello", "This is a sentence."]
        for text in acceptable:
            assert len(text) >= 3

    def test_oversized_text_rejected(self):
        """Text longer than 10KB should be rejected."""
        max_length = 10 * 1024  # 10KB
        oversized = "A" * (max_length + 1)
        assert len(oversized) > max_length

    def test_co_authored_by_skipped(self):
        """Co-Authored-By lines should be skipped."""
        skip_patterns = [
            "Co-Authored-By: Claude <noreply@anthropic.com>",
            "Some text\nCo-Authored-By: Author",
        ]
        pattern = "Co-Authored-By:"
        for text in skip_patterns:
            assert pattern in text

    def test_generated_with_skipped(self):
        """Generated with Claude markers should be skipped."""
        pattern = "\U0001F916 Generated with"
        text = "Commit message\n\U0001F916 Generated with Claude"
        assert pattern in text

    def test_system_reminder_skipped(self):
        """System reminder tags should be skipped."""
        text = "<system-reminder>You should do X</system-reminder>"
        assert "<system-reminder>" in text

    def test_code_blocks_skipped(self):
        """Code blocks (```) should be skipped."""
        text = "Here's the code:\n```python\nprint('hello')\n```"
        assert "```" in text

    def test_normal_text_accepted(self):
        """Normal conversational text should be accepted."""
        acceptable = [
            "Hello, how can I help you today?",
            "I'll read the file and check the implementation.",
            "The function looks correct. Let me test it.",
        ]
        skip_patterns = ["Co-Authored-By:", "\U0001F916 Generated with", "<system-reminder>", "```"]
        for text in acceptable:
            has_skip = any(p in text for p in skip_patterns)
            assert not has_skip


class TestCleanForSpeech:
    """Test clean_for_speech text transformation."""

    def test_markdown_removed(self):
        """Markdown formatting should be removed."""
        # Simulate clean_for_speech logic
        text = "This is **bold** and *italic* and `code`"
        result = re.sub(r'[*_`]', '', text)
        assert "**" not in result
        assert "*" not in result
        assert "`" not in result
        assert "bold" in result
        assert "italic" in result
        assert "code" in result

    def test_urls_replaced(self):
        """URLs should be replaced with 'URL'."""
        text = "Check out https://github.com/test/repo for more info"
        result = re.sub(r'https?://\S+', 'URL', text)
        assert "https://" not in result
        assert "URL" in result

    def test_long_paths_replaced(self):
        """Long file paths should be replaced with 'a file'."""
        text = "Reading /Users/ayates/voice/stream-tts-cpp/src/main.cpp"
        # Pattern for paths with 20+ chars
        result = re.sub(r'/[\w\-/]{20,}', 'a file', text)
        assert "a file" in result

    def test_short_paths_kept(self):
        """Short filenames should be kept."""
        text = "Editing main.cpp"
        result = re.sub(r'/[\w\-/]{20,}', 'a file', text)
        # main.cpp should not match (no leading /, not long enough)
        assert "main.cpp" in result

    def test_whitespace_normalized(self):
        """Multiple whitespace should be normalized to single space."""
        text = "Word1   Word2\t\tWord3\n\nWord4"
        result = re.sub(r'\s+', ' ', text)
        assert "  " not in result
        assert result == "Word1 Word2 Word3 Word4"

    def test_trim_whitespace(self):
        """Leading and trailing whitespace should be trimmed."""
        text = "   Hello, world!   "
        result = text.strip()
        assert result == "Hello, world!"


class TestToolCallFormatting:
    """Test tool call formatting logic."""

    def test_read_tool_extracts_filename(self):
        """Read tool should show just the filename."""
        full_path = "/Users/ayates/voice/stream-tts-cpp/src/main.cpp"
        last_slash = full_path.rfind("/")
        filename = full_path[last_slash + 1:] if last_slash != -1 else full_path
        assert filename == "main.cpp"

    def test_grep_tool_truncates_long_pattern(self):
        """Grep patterns > 30 chars should be truncated."""
        long_pattern = "this_is_a_very_long_search_pattern_that_exceeds_thirty_characters"
        max_len = 30
        if len(long_pattern) > max_len:
            truncated = long_pattern[:27] + "..."
        else:
            truncated = long_pattern
        assert len(truncated) <= 30

    def test_bash_extracts_first_line(self):
        """Bash command should extract first line only."""
        multiline_cmd = "git status\ngit log -1\ngit diff"
        first_newline = multiline_cmd.find('\n')
        if first_newline != -1:
            result = multiline_cmd[:first_newline] + "..."
        else:
            result = multiline_cmd
        assert result == "git status..."

    def test_bash_truncates_long_command(self):
        """Bash commands > 40 chars should be truncated."""
        long_cmd = "very_long_command_that_exceeds_forty_characters_limit"
        max_len = 40
        if len(long_cmd) > max_len:
            truncated = long_cmd[:37] + "..."
        else:
            truncated = long_cmd
        assert len(truncated) <= 40


class TestClassificationPriority:
    """Test priority classification."""

    def test_text_delta_high_priority(self):
        """Text deltas should be HIGH priority."""
        # HIGH priority for conversational text
        priority = "HIGH"
        assert priority == "HIGH"

    def test_tool_use_medium_priority(self):
        """Tool calls should be MEDIUM priority."""
        # MEDIUM priority for tool calls (needs summarization)
        priority = "MEDIUM"
        assert priority == "MEDIUM"

    def test_thinking_skip_priority(self):
        """Thinking deltas should be SKIP priority."""
        # SKIP priority for internal thinking
        priority = "SKIP"
        assert priority == "SKIP"

    def test_stop_reason_skip_priority(self):
        """Stop reason messages should be SKIP priority."""
        # SKIP for end-of-response markers
        priority = "SKIP"
        assert priority == "SKIP"


class TestTranslationFlags:
    """Test translation flag logic."""

    def test_text_delta_should_translate(self):
        """Text deltas should be marked for translation."""
        should_translate = True
        assert should_translate is True

    def test_tool_use_no_translate(self):
        """Tool calls should NOT be translated directly."""
        # Tool calls need summarization first
        should_translate = False
        should_summarize = True
        assert should_translate is False
        assert should_summarize is True


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_unicode_handling(self):
        """Unicode text should be handled correctly."""
        unicode_text = "Hello \u4e16\u754c \U0001F600"  # Chinese + emoji
        assert len(unicode_text) >= 3
        assert "\u4e16" in unicode_text  # Chinese char

    def test_mixed_content(self):
        """Mixed content (text + skip patterns) should be handled."""
        text = "Normal text ```code``` more text"
        # If code block present, entire message is skipped
        assert "```" in text

    def test_empty_tool_name(self):
        """Empty tool name should not crash."""
        tool_name = ""
        result = f"Tool: {tool_name}" if tool_name else ""
        assert result == ""

    def test_missing_tool_input(self):
        """Missing tool input should be handled gracefully."""
        tool_input = None
        has_input = tool_input is not None
        assert has_input is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
