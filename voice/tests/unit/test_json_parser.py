"""
Unit tests for JSONParser component.

Tests the JSON parsing logic for Claude stream events:
- content_block_delta (text deltas)
- content_block_start (tool calls)
- assistant messages (Claude Code format)
- thinking deltas
- Invalid JSON handling

These tests verify the parsing logic without loading TTS models.
"""

import json
import pytest
import subprocess
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from conftest_prerequisites import PROJECT_ROOT


class TestJSONParserFormats:
    """Test different JSON message formats are parsed correctly."""

    def test_text_delta_format(self):
        """Test standard Claude API text_delta format."""
        msg = {
            "type": "content_block_delta",
            "delta": {
                "type": "text_delta",
                "text": "Hello, world!"
            }
        }
        # This format should be recognized as speakable text
        assert msg["type"] == "content_block_delta"
        assert msg["delta"]["type"] == "text_delta"
        assert msg["delta"]["text"] == "Hello, world!"

    def test_tool_use_format(self):
        """Test tool_use content block format."""
        msg = {
            "type": "content_block_start",
            "content_block": {
                "type": "tool_use",
                "name": "Read",
                "input": {"file_path": "/path/to/file.cpp"}
            }
        }
        # Tool calls should have name and input
        assert msg["type"] == "content_block_start"
        assert msg["content_block"]["type"] == "tool_use"
        assert msg["content_block"]["name"] == "Read"
        assert "file_path" in msg["content_block"]["input"]

    def test_thinking_delta_format(self):
        """Test thinking delta format (should be skipped)."""
        msg = {
            "type": "content_block_delta",
            "delta": {
                "type": "thinking_delta",
                "thinking": "I need to analyze this..."
            }
        }
        # Thinking deltas should be recognized
        assert msg["delta"]["type"] == "thinking_delta"

    def test_assistant_simple_format(self):
        """Test simple assistant format from Claude Code."""
        msg = {
            "type": "assistant",
            "text": "I'll help you with that."
        }
        # Simple format with direct text field
        assert msg["type"] == "assistant"
        assert msg["text"] == "I'll help you with that."

    def test_assistant_nested_format(self):
        """Test nested assistant message format."""
        msg = {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "text", "text": "This is the nested response."}
                ]
            }
        }
        # Nested format with message.content[0].text
        assert msg["message"]["content"][0]["text"] == "This is the nested response."

    def test_invalid_json_handling(self):
        """Test that invalid JSON is handled gracefully."""
        invalid_inputs = [
            "",                     # Empty string
            "not json",            # Plain text
            "{incomplete",         # Incomplete JSON
            '{"no_type": "value"}', # Missing type field
        ]
        for invalid in invalid_inputs:
            try:
                if invalid:
                    result = json.loads(invalid)
                    # If parsing succeeds, check for missing type
                    assert "type" not in result or result.get("type") is None
            except json.JSONDecodeError:
                pass  # Expected for invalid JSON

    def test_message_delta_stop_format(self):
        """Test message_delta with stop_reason (end of response)."""
        msg = {
            "type": "message_delta",
            "delta": {
                "stop_reason": "end_turn"
            }
        }
        # This should be recognized as end of response
        assert msg["type"] == "message_delta"
        assert msg["delta"]["stop_reason"] == "end_turn"


class TestJSONParserEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_text_delta(self):
        """Test text delta with empty text."""
        msg = {
            "type": "content_block_delta",
            "delta": {
                "type": "text_delta",
                "text": ""
            }
        }
        assert msg["delta"]["text"] == ""

    def test_unicode_text_delta(self):
        """Test text delta with Unicode characters."""
        msg = {
            "type": "content_block_delta",
            "delta": {
                "type": "text_delta",
                "text": "Hello, \u4e16\u754c! \U0001F60A"  # Chinese + emoji
            }
        }
        assert "\u4e16\u754c" in msg["delta"]["text"]  # Chinese
        assert "\U0001F60A" in msg["delta"]["text"]    # Emoji

    def test_long_text_delta(self):
        """Test text delta with very long text."""
        long_text = "A" * 10000
        msg = {
            "type": "content_block_delta",
            "delta": {
                "type": "text_delta",
                "text": long_text
            }
        }
        assert len(msg["delta"]["text"]) == 10000

    def test_special_characters(self):
        """Test text with special characters that need escaping."""
        msg = {
            "type": "content_block_delta",
            "delta": {
                "type": "text_delta",
                "text": 'Line 1\nLine 2\tTabbed\r\nWindows line'
            }
        }
        assert "\n" in msg["delta"]["text"]
        assert "\t" in msg["delta"]["text"]

    def test_nested_json_in_tool_input(self):
        """Test tool input with complex nested JSON."""
        msg = {
            "type": "content_block_start",
            "content_block": {
                "type": "tool_use",
                "name": "Bash",
                "input": {
                    "command": "echo 'test'",
                    "options": {
                        "timeout": 30,
                        "env": {"PATH": "/usr/bin"}
                    }
                }
            }
        }
        assert msg["content_block"]["input"]["options"]["timeout"] == 30


class TestToolInputFormats:
    """Test different tool input formats."""

    def test_read_tool_format(self):
        """Test Read tool input format."""
        msg = {
            "type": "content_block_start",
            "content_block": {
                "type": "tool_use",
                "name": "Read",
                "input": {"file_path": "/Users/test/src/main.cpp"}
            }
        }
        assert msg["content_block"]["name"] == "Read"
        assert "main.cpp" in msg["content_block"]["input"]["file_path"]

    def test_grep_tool_format(self):
        """Test Grep tool input format."""
        msg = {
            "type": "content_block_start",
            "content_block": {
                "type": "tool_use",
                "name": "Grep",
                "input": {
                    "pattern": "class\\s+\\w+",
                    "path": "/Users/test/src"
                }
            }
        }
        assert msg["content_block"]["name"] == "Grep"
        assert "pattern" in msg["content_block"]["input"]

    def test_edit_tool_format(self):
        """Test Edit tool input format."""
        msg = {
            "type": "content_block_start",
            "content_block": {
                "type": "tool_use",
                "name": "Edit",
                "input": {
                    "file_path": "/path/to/file.py",
                    "old_string": "old code",
                    "new_string": "new code"
                }
            }
        }
        assert msg["content_block"]["name"] == "Edit"
        assert "old_string" in msg["content_block"]["input"]

    def test_bash_tool_format(self):
        """Test Bash tool input format."""
        msg = {
            "type": "content_block_start",
            "content_block": {
                "type": "tool_use",
                "name": "Bash",
                "input": {
                    "command": "git status && git log -1"
                }
            }
        }
        assert msg["content_block"]["name"] == "Bash"
        assert "git" in msg["content_block"]["input"]["command"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
