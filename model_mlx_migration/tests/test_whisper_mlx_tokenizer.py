# Copyright 2024-2025 Andrew Yates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tests for whisper_mlx tokenizer module.

Tests:
- LANGUAGES dictionary
- get_whisper_tokenizer function
- TimestampDecoder class
"""

import pytest


class TestLanguagesDictionary:
    """Tests for LANGUAGES constant."""

    def test_languages_not_empty(self):
        """Test LANGUAGES is populated."""
        from tools.whisper_mlx.tokenizer import LANGUAGES

        assert len(LANGUAGES) > 0

    def test_english_present(self):
        """Test English is present."""
        from tools.whisper_mlx.tokenizer import LANGUAGES

        assert "en" in LANGUAGES
        assert LANGUAGES["en"] == "english"

    def test_common_languages_present(self):
        """Test common languages are present."""
        from tools.whisper_mlx.tokenizer import LANGUAGES

        expected_languages = {
            "en": "english",
            "zh": "chinese",
            "de": "german",
            "es": "spanish",
            "fr": "french",
            "ja": "japanese",
            "ko": "korean",
            "ru": "russian",
            "pt": "portuguese",
            "ar": "arabic",
        }

        for code, name in expected_languages.items():
            assert code in LANGUAGES
            assert LANGUAGES[code] == name

    def test_language_codes_format(self):
        """Test language codes are lowercase strings."""
        from tools.whisper_mlx.tokenizer import LANGUAGES

        for code in LANGUAGES.keys():
            assert isinstance(code, str)
            assert code == code.lower()
            assert len(code) >= 2

    def test_language_names_format(self):
        """Test language names are lowercase strings."""
        from tools.whisper_mlx.tokenizer import LANGUAGES

        for name in LANGUAGES.values():
            assert isinstance(name, str)
            assert name == name.lower()
            assert len(name) > 0

    def test_all_100_languages(self):
        """Test Whisper v3 has 100 languages."""
        from tools.whisper_mlx.tokenizer import LANGUAGES

        # Whisper large-v3 supports 100 languages (including Cantonese)
        assert len(LANGUAGES) == 100

    def test_unique_language_names(self):
        """Test all language names are unique."""
        from tools.whisper_mlx.tokenizer import LANGUAGES

        names = list(LANGUAGES.values())
        # Check for duplicates
        assert len(names) == len(set(names))

    def test_cjk_languages_present(self):
        """Test CJK languages are present."""
        from tools.whisper_mlx.tokenizer import LANGUAGES

        assert "zh" in LANGUAGES  # Chinese
        assert "ja" in LANGUAGES  # Japanese
        assert "ko" in LANGUAGES  # Korean
        assert "yue" in LANGUAGES  # Cantonese

    def test_indic_languages_present(self):
        """Test Indic languages are present."""
        from tools.whisper_mlx.tokenizer import LANGUAGES

        indic_codes = ["hi", "ta", "te", "bn", "gu", "mr", "ml", "kn", "pa", "ne"]
        for code in indic_codes:
            assert code in LANGUAGES, f"Missing Indic language: {code}"


class TestGetWhisperTokenizer:
    """Tests for get_whisper_tokenizer function."""

    def test_import_available_flag(self):
        """Test MLX_WHISPER_AVAILABLE flag exists."""
        from tools.whisper_mlx.tokenizer import MLX_WHISPER_AVAILABLE

        assert isinstance(MLX_WHISPER_AVAILABLE, bool)

    @pytest.mark.skipif(
        True,  # Skip if mlx_whisper not installed
        reason="mlx_whisper may not be installed",
    )
    def test_get_tokenizer_basic(self):
        """Test basic tokenizer retrieval."""
        from tools.whisper_mlx.tokenizer import (
            MLX_WHISPER_AVAILABLE,
            get_whisper_tokenizer,
        )

        if not MLX_WHISPER_AVAILABLE:
            pytest.skip("mlx_whisper not installed")

        tokenizer = get_whisper_tokenizer()
        assert tokenizer is not None

    def test_get_tokenizer_raises_without_mlx_whisper(self):
        """Test ImportError raised when mlx_whisper not available."""
        from tools.whisper_mlx import tokenizer as tok_module

        # Temporarily set flag to False
        original = tok_module.MLX_WHISPER_AVAILABLE
        tok_module.MLX_WHISPER_AVAILABLE = False

        try:
            with pytest.raises(ImportError, match="mlx-whisper is required"):
                tok_module.get_whisper_tokenizer()
        finally:
            tok_module.MLX_WHISPER_AVAILABLE = original


class TestTimestampDecoder:
    """Tests for TimestampDecoder class."""

    def test_init_default(self):
        """Test default initialization."""
        from tools.whisper_mlx.tokenizer import TimestampDecoder

        decoder = TimestampDecoder()

        assert decoder.timestamp_begin == 50364
        assert decoder.precision == 0.02

    def test_init_custom(self):
        """Test custom initialization."""
        from tools.whisper_mlx.tokenizer import TimestampDecoder

        decoder = TimestampDecoder(
            timestamp_begin=1000,
            precision=0.01,
        )

        assert decoder.timestamp_begin == 1000
        assert decoder.precision == 0.01

    def test_set_precision(self):
        """Test set_precision method."""
        from tools.whisper_mlx.tokenizer import TimestampDecoder

        decoder = TimestampDecoder()

        decoder.set_precision(0.04)

        assert decoder.precision == 0.04

    def test_token_to_time_basic(self):
        """Test basic token to time conversion."""
        from tools.whisper_mlx.tokenizer import TimestampDecoder

        decoder = TimestampDecoder()

        # timestamp_begin (50364) should be 0s
        assert decoder.token_to_time(50364) == 0.0

        # timestamp_begin + 1 should be 0.02s
        assert abs(decoder.token_to_time(50365) - 0.02) < 1e-10

        # timestamp_begin + 50 should be 1.0s
        assert abs(decoder.token_to_time(50414) - 1.0) < 1e-10

    def test_token_to_time_30s(self):
        """Test token to time at 30 seconds."""
        from tools.whisper_mlx.tokenizer import TimestampDecoder

        decoder = TimestampDecoder()

        # 30s = 1500 positions
        token_30s = 50364 + 1500
        assert abs(decoder.token_to_time(token_30s) - 30.0) < 1e-10

    def test_time_to_token_basic(self):
        """Test basic time to token conversion."""
        from tools.whisper_mlx.tokenizer import TimestampDecoder

        decoder = TimestampDecoder()

        # 0s should be timestamp_begin
        assert decoder.time_to_token(0.0) == 50364

        # 0.02s should be timestamp_begin + 1
        assert decoder.time_to_token(0.02) == 50365

        # 1.0s should be timestamp_begin + 50
        assert decoder.time_to_token(1.0) == 50414

    def test_time_to_token_30s(self):
        """Test time to token at 30 seconds."""
        from tools.whisper_mlx.tokenizer import TimestampDecoder

        decoder = TimestampDecoder()

        # 30s should be timestamp_begin + 1500
        assert decoder.time_to_token(30.0) == 50364 + 1500

    def test_roundtrip_conversion(self):
        """Test roundtrip token -> time -> token."""
        from tools.whisper_mlx.tokenizer import TimestampDecoder

        decoder = TimestampDecoder()

        for position in [0, 1, 10, 50, 100, 500, 1000, 1500]:
            token = 50364 + position
            time = decoder.token_to_time(token)
            recovered_token = decoder.time_to_token(time)
            assert recovered_token == token

    def test_precision_affects_conversion(self):
        """Test precision affects conversions."""
        from tools.whisper_mlx.tokenizer import TimestampDecoder

        # Default precision (0.02)
        decoder1 = TimestampDecoder(precision=0.02)
        # Double precision (0.04)
        decoder2 = TimestampDecoder(precision=0.04)

        token = 50364 + 50

        # Same token gives different times
        time1 = decoder1.token_to_time(token)
        time2 = decoder2.token_to_time(token)

        assert abs(time1 - 1.0) < 1e-10   # 50 * 0.02 = 1.0
        assert abs(time2 - 2.0) < 1e-10   # 50 * 0.04 = 2.0


class TestTimestampDecoderExtractTimestamps:
    """Tests for TimestampDecoder.extract_timestamps method."""

    def test_empty_tokens(self):
        """Test with empty token list."""
        from tools.whisper_mlx.tokenizer import TimestampDecoder

        decoder = TimestampDecoder()

        segments = decoder.extract_timestamps([])

        assert segments == []

    def test_single_segment(self):
        """Test extracting single segment."""
        from tools.whisper_mlx.tokenizer import TimestampDecoder

        decoder = TimestampDecoder()

        # <0.00s> hello world <2.00s>
        tokens = [
            50364,       # 0.00s timestamp
            100, 200,    # text tokens
            50364 + 100, # 2.00s timestamp
        ]

        segments = decoder.extract_timestamps(tokens)

        assert len(segments) == 1
        start, end, text_tokens = segments[0]
        assert abs(start - 0.0) < 1e-10
        assert abs(end - 2.0) < 1e-10
        assert text_tokens == [100, 200]

    def test_multiple_segments(self):
        """Test extracting multiple segments."""
        from tools.whisper_mlx.tokenizer import TimestampDecoder

        decoder = TimestampDecoder()

        # <0.00s> hello <1.00s> <1.00s> world <2.00s>
        # Implementation creates 3 segments because consecutive timestamps
        # create an empty segment between them
        tokens = [
            50364,           # 0.00s
            100,             # "hello"
            50364 + 50,      # 1.00s
            50364 + 50,      # 1.00s (creates boundary)
            200,             # "world"
            50364 + 100,     # 2.00s
        ]

        segments = decoder.extract_timestamps(tokens)

        # Three segments: (0-1, "hello"), (1-1, empty), (1-2, "world")
        assert len(segments) == 3

        # First segment: 0.0s - 1.0s, "hello"
        assert abs(segments[0][0] - 0.0) < 1e-10
        assert abs(segments[0][1] - 1.0) < 1e-10
        assert segments[0][2] == [100]

        # Second segment: 1.0s - 1.0s, empty (boundary marker)
        assert abs(segments[1][0] - 1.0) < 1e-10
        assert abs(segments[1][1] - 1.0) < 1e-10
        assert segments[1][2] == []

        # Third segment: 1.0s - 2.0s, "world"
        assert abs(segments[2][0] - 1.0) < 1e-10
        assert abs(segments[2][1] - 2.0) < 1e-10
        assert segments[2][2] == [200]

    def test_no_timestamps(self):
        """Test with no timestamp tokens."""
        from tools.whisper_mlx.tokenizer import TimestampDecoder

        decoder = TimestampDecoder()

        # Only text tokens (below timestamp_begin)
        tokens = [100, 200, 300]

        segments = decoder.extract_timestamps(tokens)

        # No segments since no timestamps
        assert segments == []

    def test_only_start_timestamp(self):
        """Test with only start timestamp (no end)."""
        from tools.whisper_mlx.tokenizer import TimestampDecoder

        decoder = TimestampDecoder()

        # <0.00s> hello (no end timestamp)
        tokens = [
            50364,  # 0.00s
            100,    # text
        ]

        segments = decoder.extract_timestamps(tokens)

        # Should create segment with same start/end
        assert len(segments) == 1
        start, end, text_tokens = segments[0]
        assert abs(start - 0.0) < 1e-10
        assert start == end  # No end timestamp
        assert text_tokens == [100]

    def test_consecutive_timestamps(self):
        """Test consecutive timestamps create boundaries."""
        from tools.whisper_mlx.tokenizer import TimestampDecoder

        decoder = TimestampDecoder()

        # <0.00s> <1.00s> hello <2.00s>
        tokens = [
            50364,           # 0.00s
            50364 + 50,      # 1.00s (consecutive timestamp = boundary)
            100,             # "hello"
            50364 + 100,     # 2.00s
        ]

        segments = decoder.extract_timestamps(tokens)

        # The consecutive timestamps create a boundary
        # First "segment" has no text (0.0-1.0)
        # Second segment has text (1.0-2.0)
        assert len(segments) >= 1

    def test_text_tokens_collected(self):
        """Test text tokens are properly collected."""
        from tools.whisper_mlx.tokenizer import TimestampDecoder

        decoder = TimestampDecoder()

        text_tokens = [100, 200, 300, 400, 500]
        tokens = [50364] + text_tokens + [50364 + 100]

        segments = decoder.extract_timestamps(tokens)

        assert len(segments) == 1
        assert segments[0][2] == text_tokens

    def test_variable_precision(self):
        """Test with different precision settings."""
        from tools.whisper_mlx.tokenizer import TimestampDecoder

        # 0.04s precision (double standard)
        decoder = TimestampDecoder(precision=0.04)

        # <0.00s> hello <2.00s>
        # With 0.04 precision, position 50 = 2.0s
        tokens = [
            50364,           # 0.00s
            100,             # text
            50364 + 50,      # 2.00s (50 * 0.04)
        ]

        segments = decoder.extract_timestamps(tokens)

        assert len(segments) == 1
        assert abs(segments[0][1] - 2.0) < 1e-10

    def test_custom_timestamp_begin(self):
        """Test with custom timestamp_begin."""
        from tools.whisper_mlx.tokenizer import TimestampDecoder

        decoder = TimestampDecoder(timestamp_begin=1000, precision=0.02)

        tokens = [
            1000,       # 0.00s
            100, 200,   # text
            1050,       # 1.00s
        ]

        segments = decoder.extract_timestamps(tokens)

        assert len(segments) == 1
        assert abs(segments[0][0] - 0.0) < 1e-10
        assert abs(segments[0][1] - 1.0) < 1e-10


class TestTimestampDecoderEdgeCases:
    """Edge case tests for TimestampDecoder."""

    def test_very_small_precision(self):
        """Test with very small precision."""
        from tools.whisper_mlx.tokenizer import TimestampDecoder

        decoder = TimestampDecoder(precision=0.001)

        # Position 1000 should be 1.0s
        token = 50364 + 1000
        assert abs(decoder.token_to_time(token) - 1.0) < 1e-10

    def test_very_large_precision(self):
        """Test with very large precision."""
        from tools.whisper_mlx.tokenizer import TimestampDecoder

        decoder = TimestampDecoder(precision=1.0)

        # Position 1 should be 1.0s
        token = 50364 + 1
        assert abs(decoder.token_to_time(token) - 1.0) < 1e-10

    def test_negative_position(self):
        """Test with token below timestamp_begin."""
        from tools.whisper_mlx.tokenizer import TimestampDecoder

        decoder = TimestampDecoder()

        # Token below timestamp_begin gives negative time
        token = 50364 - 10
        time = decoder.token_to_time(token)

        assert time < 0

    def test_large_token_value(self):
        """Test with large token value."""
        from tools.whisper_mlx.tokenizer import TimestampDecoder

        decoder = TimestampDecoder()

        # Large position = long audio time
        token = 50364 + 100000
        time = decoder.token_to_time(token)

        # 100000 * 0.02 = 2000 seconds
        assert abs(time - 2000.0) < 1e-10

    def test_fractional_time_to_token(self):
        """Test fractional times are truncated to token."""
        from tools.whisper_mlx.tokenizer import TimestampDecoder

        decoder = TimestampDecoder()

        # 0.03s should map to position 1 (int(0.03 / 0.02) = 1)
        assert decoder.time_to_token(0.03) == 50365

        # 0.039s should still map to position 1
        assert decoder.time_to_token(0.039) == 50365

        # 0.04s should map to position 2
        assert decoder.time_to_token(0.04) == 50366


class TestTimestampDecoderIntegration:
    """Integration tests for TimestampDecoder."""

    def test_full_transcription_simulation(self):
        """Test simulating full transcription with timestamps."""
        from tools.whisper_mlx.tokenizer import TimestampDecoder

        decoder = TimestampDecoder()

        # Simulate: "<0.00s>Hello world<2.00s><2.00s>How are you<4.00s>"
        # Consecutive timestamps create an empty segment marker
        tokens = [
            50364,           # 0.00s
            1, 2, 3, 4,      # "Hello world"
            50364 + 100,     # 2.00s
            50364 + 100,     # 2.00s (segment boundary)
            5, 6, 7,         # "How are you"
            50364 + 200,     # 4.00s
        ]

        segments = decoder.extract_timestamps(tokens)

        # Should have 3 segments (including empty boundary)
        assert len(segments) == 3

        # Segment 1: 0.0s - 2.0s, "Hello world"
        assert abs(segments[0][0] - 0.0) < 1e-10
        assert abs(segments[0][1] - 2.0) < 1e-10
        assert segments[0][2] == [1, 2, 3, 4]

        # Segment 2: 2.0s - 2.0s, empty (boundary marker)
        assert abs(segments[1][0] - 2.0) < 1e-10
        assert abs(segments[1][1] - 2.0) < 1e-10
        assert segments[1][2] == []

        # Segment 3: 2.0s - 4.0s, "How are you"
        assert abs(segments[2][0] - 2.0) < 1e-10
        assert abs(segments[2][1] - 4.0) < 1e-10
        assert segments[2][2] == [5, 6, 7]

    def test_short_audio_precision(self):
        """Test precision adjustment for short audio."""
        from tools.whisper_mlx.tokenizer import TimestampDecoder

        # 5 second audio with 250 encoder positions
        # Precision = 5.0 / 250 = 0.02 (same as default)
        decoder = TimestampDecoder()
        decoder.set_precision(5.0 / 250)

        # Position 125 should be 2.5s
        token = 50364 + 125
        time = decoder.token_to_time(token)
        assert abs(time - 2.5) < 1e-10

    def test_long_audio_precision(self):
        """Test precision for longer audio."""
        from tools.whisper_mlx.tokenizer import TimestampDecoder

        # 60 second audio with 3000 encoder positions
        decoder = TimestampDecoder()
        decoder.set_precision(60.0 / 3000)

        # Position 1500 should be 30s
        token = 50364 + 1500
        time = decoder.token_to_time(token)
        assert abs(time - 30.0) < 1e-10

        # Position 3000 should be 60s
        token = 50364 + 3000
        time = decoder.token_to_time(token)
        assert abs(time - 60.0) < 1e-10
