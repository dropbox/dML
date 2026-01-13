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
Tests for whisper_mlx/decoding.py module.

Tests logit filters and decoding utilities for WhisperMLX.
"""

from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock

import mlx.core as mx
import numpy as np
import pytest

from tools.whisper_mlx.decoding import (
    ApplyTimestampRules,
    DecodingOptions,
    DecodingResult,
    HallucinationResult,
    LogitFilter,
    SuppressBlank,
    SuppressTokens,
    apply_filters,
    build_logit_filters,
    compression_ratio,
    detect_hallucination,
    get_suppress_tokens,
    is_hallucination,
)


class TestCompressionRatio:
    """Tests for compression_ratio function."""

    def test_simple_text(self):
        """Test compression ratio of simple text."""
        text = "Hello, world!"
        ratio = compression_ratio(text)
        assert ratio > 0
        assert isinstance(ratio, float)

    def test_repetitive_text_high_ratio(self):
        """Repetitive text should have HIGHER compression ratio (compresses well)."""
        # compression_ratio = original_size / compressed_size
        # Higher ratio means better compression (smaller compressed output)
        repetitive = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        varied = "The quick brown fox jumps over the lazy dog"
        assert compression_ratio(repetitive) > compression_ratio(varied)

    def test_empty_string(self):
        """Empty string compresses to minimum zlib overhead."""
        # Empty string still gets zlib header, so ratio is 0/small = 0
        ratio = compression_ratio("")
        assert ratio == 0.0

    def test_unicode_text(self):
        """Test compression ratio with unicode text."""
        text = "你好世界"
        ratio = compression_ratio(text)
        assert ratio > 0

    def test_highly_repetitive_high_ratio(self):
        """Highly repetitive text has HIGH ratio (compresses very well)."""
        # compression_ratio = original_size / compressed_size
        # "a" * 1000 = 1000 bytes, compresses to ~12 bytes = ratio ~83
        text = "a" * 1000
        ratio = compression_ratio(text)
        assert ratio > 10  # Highly compressible


class TestDecodingOptions:
    """Tests for DecodingOptions dataclass."""

    def test_default_values(self):
        """Test default values are set correctly."""
        opts = DecodingOptions()
        assert opts.task == "transcribe"
        assert opts.language is None
        assert opts.temperature == 0.0
        assert opts.sample_len is None
        assert opts.suppress_tokens == "-1"
        assert opts.suppress_blank is True
        assert opts.without_timestamps is False
        assert opts.max_initial_timestamp == 1.0

    def test_custom_values(self):
        """Test custom values are stored correctly."""
        opts = DecodingOptions(
            task="translate",
            language="fr",
            temperature=0.7,
            sample_len=100,
            suppress_tokens=[1, 2, 3],
            suppress_blank=False,
            without_timestamps=True,
            max_initial_timestamp=0.5,
        )
        assert opts.task == "translate"
        assert opts.language == "fr"
        assert opts.temperature == 0.7
        assert opts.sample_len == 100
        assert opts.suppress_tokens == [1, 2, 3]
        assert opts.suppress_blank is False
        assert opts.without_timestamps is True
        assert opts.max_initial_timestamp == 0.5

    def test_frozen(self):
        """DecodingOptions should be frozen (immutable)."""
        opts = DecodingOptions()
        with pytest.raises(FrozenInstanceError):
            opts.task = "translate"


class TestDecodingResult:
    """Tests for DecodingResult dataclass."""

    def test_default_values(self):
        """Test default values are set correctly."""
        result = DecodingResult()
        assert result.tokens == []
        assert result.text == ""
        assert result.language == "en"
        assert np.isnan(result.avg_logprob)
        assert np.isnan(result.no_speech_prob)
        assert np.isnan(result.temperature)
        assert np.isnan(result.compression_ratio)

    def test_custom_values(self):
        """Test custom values are stored correctly."""
        result = DecodingResult(
            tokens=[1, 2, 3],
            text="Hello",
            language="fr",
            avg_logprob=-0.5,
            no_speech_prob=0.1,
            temperature=0.7,
            compression_ratio=1.5,
        )
        assert result.tokens == [1, 2, 3]
        assert result.text == "Hello"
        assert result.language == "fr"
        assert result.avg_logprob == -0.5
        assert result.no_speech_prob == 0.1
        assert result.temperature == 0.7
        assert result.compression_ratio == 1.5

    def test_mutable(self):
        """DecodingResult should be mutable."""
        result = DecodingResult()
        result.text = "Updated"
        assert result.text == "Updated"


class TestLogitFilter:
    """Tests for LogitFilter base class."""

    def test_apply_not_implemented(self):
        """Base class apply should raise NotImplementedError."""
        filt = LogitFilter()
        with pytest.raises(NotImplementedError):
            filt.apply(mx.zeros((1, 10)), mx.zeros((1, 5)))


class TestSuppressBlank:
    """Tests for SuppressBlank logit filter."""

    def _mock_tokenizer(self):
        """Create mock tokenizer for testing."""
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [220]  # space token
        tokenizer.eot = 50257
        return tokenizer

    def test_initialization(self):
        """Test SuppressBlank initialization."""
        tokenizer = self._mock_tokenizer()
        filt = SuppressBlank(tokenizer, sample_begin=3, n_vocab=51865)
        assert filt.sample_begin == 3
        assert filt.mask.shape == (51865,)

    def test_suppresses_at_sample_begin(self):
        """Test that blank tokens are suppressed at sample_begin."""
        tokenizer = self._mock_tokenizer()
        n_vocab = 51865
        filt = SuppressBlank(tokenizer, sample_begin=3, n_vocab=n_vocab)

        # Tokens at exactly sample_begin
        logits = mx.zeros((1, n_vocab))
        tokens = mx.zeros((1, 3), dtype=mx.int32)  # shape[1] == sample_begin

        result = filt.apply(logits, tokens)

        # Space token and EOT should be suppressed (-inf)
        assert result[0, 220] == float("-inf")  # space
        assert result[0, 50257] == float("-inf")  # EOT

    def test_no_suppression_after_sample_begin(self):
        """Test that no suppression happens after sample_begin."""
        tokenizer = self._mock_tokenizer()
        n_vocab = 51865
        filt = SuppressBlank(tokenizer, sample_begin=3, n_vocab=n_vocab)

        # Tokens after sample_begin
        logits = mx.zeros((1, n_vocab))
        tokens = mx.zeros((1, 5), dtype=mx.int32)  # shape[1] > sample_begin

        result = filt.apply(logits, tokens)

        # Should return unchanged logits
        assert mx.allclose(result, logits).item()


class TestSuppressTokens:
    """Tests for SuppressTokens logit filter."""

    def test_initialization(self):
        """Test SuppressTokens initialization."""
        suppress_tokens = [100, 200, 300]
        filt = SuppressTokens(suppress_tokens, n_vocab=51865)
        assert filt.mask.shape == (51865,)

    def test_suppresses_specified_tokens(self):
        """Test that specified tokens are suppressed."""
        suppress_tokens = [100, 200, 300]
        n_vocab = 1000
        filt = SuppressTokens(suppress_tokens, n_vocab=n_vocab)

        logits = mx.zeros((1, n_vocab))
        tokens = mx.zeros((1, 5), dtype=mx.int32)

        result = filt.apply(logits, tokens)

        # Suppressed tokens should be -inf
        assert result[0, 100] == float("-inf")
        assert result[0, 200] == float("-inf")
        assert result[0, 300] == float("-inf")

        # Other tokens should be unchanged
        assert result[0, 0] == 0.0
        assert result[0, 50] == 0.0
        assert result[0, 999] == 0.0

    def test_batch_processing(self):
        """Test that filter works with batched inputs."""
        suppress_tokens = [100]
        n_vocab = 500
        filt = SuppressTokens(suppress_tokens, n_vocab=n_vocab)

        logits = mx.ones((4, n_vocab))  # batch of 4
        tokens = mx.zeros((4, 5), dtype=mx.int32)

        result = filt.apply(logits, tokens)

        # All batches should have token 100 suppressed
        for i in range(4):
            assert result[i, 100] == float("-inf")
            assert result[i, 0] == 1.0


class TestFusedSuppressFilter:
    """Tests for FusedSuppressFilter (OPT-NEW-8)."""

    def _mock_tokenizer(self, eot=500, space_token=220):
        """Create mock tokenizer for testing."""
        tokenizer = MagicMock()
        tokenizer.encode = MagicMock(return_value=[space_token])  # space token
        tokenizer.eot = eot
        return tokenizer

    def test_initialization(self):
        """Test FusedSuppressFilter initialization."""
        from tools.whisper_mlx.decoding import FusedSuppressFilter

        tokenizer = self._mock_tokenizer(eot=50257)
        suppress_tokens = [100, 200, 300]
        filt = FusedSuppressFilter(tokenizer, sample_begin=3, suppress_tokens=suppress_tokens, n_vocab=51865)

        assert filt.sample_begin == 3
        assert filt.base_mask.shape == (51865,)
        assert filt.combined_mask.shape == (51865,)

    def test_suppresses_tokens_always(self):
        """Test that suppress_tokens are always suppressed."""
        from tools.whisper_mlx.decoding import FusedSuppressFilter

        n_vocab = 1000
        tokenizer = self._mock_tokenizer(eot=500, space_token=220)
        suppress_tokens = [100, 200]
        filt = FusedSuppressFilter(tokenizer, sample_begin=3, suppress_tokens=suppress_tokens, n_vocab=n_vocab)

        logits = mx.zeros((1, n_vocab))
        tokens = mx.zeros((1, 10), dtype=mx.int32)  # Not at sample_begin

        result = filt.apply(logits, tokens)

        # Suppress tokens should be -inf
        assert result[0, 100] == float("-inf")
        assert result[0, 200] == float("-inf")
        # Other tokens should be unchanged
        assert result[0, 0] == 0.0

    def test_suppresses_blank_at_sample_begin(self):
        """Test that blank/EOT are suppressed only at sample_begin."""
        from tools.whisper_mlx.decoding import FusedSuppressFilter

        n_vocab = 1000
        space_token = 220
        eot = 500
        tokenizer = self._mock_tokenizer(eot=eot, space_token=space_token)
        filt = FusedSuppressFilter(tokenizer, sample_begin=3, suppress_tokens=[100], n_vocab=n_vocab)

        logits = mx.zeros((1, n_vocab))
        tokens = mx.zeros((1, 3), dtype=mx.int32)  # At sample_begin (shape[1] == 3)

        result = filt.apply(logits, tokens)

        # At sample_begin: blank and EOT should be suppressed
        assert result[0, space_token] == float("-inf")
        assert result[0, eot] == float("-inf")
        # Suppress tokens still suppressed
        assert result[0, 100] == float("-inf")

    def test_no_blank_suppression_after_sample_begin(self):
        """Test that blank suppression only happens at sample_begin."""
        from tools.whisper_mlx.decoding import FusedSuppressFilter

        n_vocab = 1000
        space_token = 220
        eot = 500
        tokenizer = self._mock_tokenizer(eot=eot, space_token=space_token)
        filt = FusedSuppressFilter(tokenizer, sample_begin=3, suppress_tokens=[100], n_vocab=n_vocab)

        logits = mx.zeros((1, n_vocab))
        tokens = mx.zeros((1, 5), dtype=mx.int32)  # After sample_begin

        result = filt.apply(logits, tokens)

        # After sample_begin: blank should NOT be suppressed (only in combined_mask)
        # space_token is not in suppress_tokens, so should be 0
        assert result[0, space_token] == 0.0
        # EOT also not suppressed after sample_begin
        assert result[0, eot] == 0.0
        # Suppress tokens still suppressed
        assert result[0, 100] == float("-inf")


class TestApplyTimestampRules:
    """Tests for ApplyTimestampRules logit filter."""

    def _mock_tokenizer(self):
        """Create mock tokenizer for testing."""
        tokenizer = MagicMock()
        tokenizer.timestamp_begin = 50364
        tokenizer.eot = 50257
        tokenizer.no_timestamps = 50363
        return tokenizer

    def test_initialization(self):
        """Test ApplyTimestampRules initialization."""
        tokenizer = self._mock_tokenizer()
        filt = ApplyTimestampRules(
            tokenizer,
            sample_begin=3,
            max_initial_timestamp_index=50,
            n_vocab=51865,
        )
        assert filt.sample_begin == 3
        assert filt.max_initial_timestamp_index == 50
        assert filt.timestamp_begin == 50364
        assert filt.eot == 50257

    def test_suppresses_no_timestamps(self):
        """Test that <|notimestamps|> is suppressed."""
        tokenizer = self._mock_tokenizer()
        n_vocab = 51865
        filt = ApplyTimestampRules(
            tokenizer,
            sample_begin=3,
            max_initial_timestamp_index=50,
            n_vocab=n_vocab,
        )

        logits = mx.zeros((1, n_vocab))
        tokens = mx.zeros((1, 5), dtype=mx.int32)

        result = filt.apply(logits, tokens)

        # no_timestamps token should be suppressed
        assert result[0, 50363] == float("-inf")

    def test_forces_timestamp_at_beginning(self):
        """Test that timestamp is forced at sample_begin."""
        tokenizer = self._mock_tokenizer()
        n_vocab = 51865
        filt = ApplyTimestampRules(
            tokenizer,
            sample_begin=3,
            max_initial_timestamp_index=50,
            n_vocab=n_vocab,
        )

        # Create logits with higher text probability
        logits = mx.zeros((1, n_vocab))
        logits = logits.at[0, :50364].add(1.0)  # Higher text logits

        # At sample_begin, timestamp should be forced
        tokens = mx.zeros((1, 3), dtype=mx.int32)  # shape[1] == sample_begin

        result = filt.apply(logits, tokens)

        # Text tokens (before timestamp_begin) should be suppressed
        assert result[0, 0] == float("-inf")
        assert result[0, 100] == float("-inf")

    def test_max_initial_timestamp_index(self):
        """Test max_initial_timestamp_index enforcement."""
        tokenizer = self._mock_tokenizer()
        n_vocab = 51865
        filt = ApplyTimestampRules(
            tokenizer,
            sample_begin=3,
            max_initial_timestamp_index=10,  # Only first 10 timestamps allowed
            n_vocab=n_vocab,
        )

        logits = mx.zeros((1, n_vocab))
        tokens = mx.zeros((1, 3), dtype=mx.int32)  # At sample_begin

        result = filt.apply(logits, tokens)

        # Timestamps beyond max_initial should be suppressed
        timestamp_begin = 50364
        assert result[0, timestamp_begin + 11] == float("-inf")
        assert result[0, timestamp_begin + 20] == float("-inf")

    def test_max_audio_timestamp_index(self):
        """Test max_audio_timestamp_index enforcement for variable-length."""
        tokenizer = self._mock_tokenizer()
        n_vocab = 51865
        filt = ApplyTimestampRules(
            tokenizer,
            sample_begin=3,
            max_initial_timestamp_index=50,
            n_vocab=n_vocab,
            max_audio_timestamp_index=100,
        )

        logits = mx.zeros((1, n_vocab))
        tokens = mx.zeros((1, 5), dtype=mx.int32)

        result = filt.apply(logits, tokens)

        # Timestamps beyond audio duration should be suppressed
        timestamp_begin = 50364
        assert result[0, timestamp_begin + 101] == float("-inf")
        assert result[0, timestamp_begin + 200] == float("-inf")

    def test_timestamp_pairing_after_timestamp(self):
        """Test timestamp pairing rules after a timestamp token."""
        tokenizer = self._mock_tokenizer()
        n_vocab = 51865
        timestamp_begin = 50364
        filt = ApplyTimestampRules(
            tokenizer,
            sample_begin=3,
            max_initial_timestamp_index=50,
            n_vocab=n_vocab,
        )

        logits = mx.zeros((1, n_vocab))
        # Sequence with timestamp at the end: [0, 0, 0, timestamp]
        tokens = mx.array([[0, 0, 0, timestamp_begin + 10]], dtype=mx.int32)

        result = filt.apply(logits, tokens)

        # After one timestamp, text should be suppressed (need to sample another timestamp or EOT)
        # The rule is: if last was timestamp and penultimate wasn't, suppress text
        assert result[0, 0] == float("-inf")


class TestGetSuppressTokens:
    """Tests for get_suppress_tokens function."""

    def _mock_tokenizer(self):
        """Create mock tokenizer for testing."""
        tokenizer = MagicMock()
        tokenizer.transcribe = 50359
        tokenizer.translate = 50358
        tokenizer.sot = 50258
        tokenizer.sot_prev = 50361
        tokenizer.sot_lm = 50360
        tokenizer.no_speech = 50362
        tokenizer.non_speech_tokens = [1, 2, 3]
        tokenizer.all_language_tokens = [50259, 50260, 50261]
        return tokenizer

    def test_string_input(self):
        """Test parsing string input."""
        tokenizer = self._mock_tokenizer()
        result = get_suppress_tokens(tokenizer, "100,200,300")

        # Should include parsed tokens and special tokens
        assert 100 in result
        assert 200 in result
        assert 300 in result

    def test_list_input(self):
        """Test list input."""
        tokenizer = self._mock_tokenizer()
        result = get_suppress_tokens(tokenizer, [100, 200])

        assert 100 in result
        assert 200 in result

    def test_minus_one_expands_non_speech(self):
        """Test that -1 expands to non_speech_tokens."""
        tokenizer = self._mock_tokenizer()
        result = get_suppress_tokens(tokenizer, "-1")

        # Should include non_speech_tokens
        assert 1 in result
        assert 2 in result
        assert 3 in result
        # Should not include -1 itself
        assert -1 not in result

    def test_special_tokens_always_suppressed(self):
        """Test that special tokens are always suppressed."""
        tokenizer = self._mock_tokenizer()
        result = get_suppress_tokens(tokenizer, [])

        # Special task tokens should always be included
        assert 50359 in result  # transcribe
        assert 50358 in result  # translate
        assert 50258 in result  # sot
        assert 50361 in result  # sot_prev
        assert 50360 in result  # sot_lm
        assert 50362 in result  # no_speech

    def test_language_tokens_suppressed(self):
        """Test that language tokens are suppressed."""
        tokenizer = self._mock_tokenizer()
        result = get_suppress_tokens(tokenizer, [])

        # Language tokens should be suppressed
        assert 50259 in result
        assert 50260 in result
        assert 50261 in result

    def test_result_is_sorted_tuple(self):
        """Test that result is a sorted tuple."""
        tokenizer = self._mock_tokenizer()
        result = get_suppress_tokens(tokenizer, [300, 100, 200])

        assert isinstance(result, tuple)
        assert list(result) == sorted(result)

    def test_no_duplicates(self):
        """Test that result has no duplicates."""
        tokenizer = self._mock_tokenizer()
        # Pass same token multiple times
        result = get_suppress_tokens(tokenizer, [100, 100, 100])

        assert len(result) == len(set(result))

    def test_none_input(self):
        """Test None input."""
        tokenizer = self._mock_tokenizer()
        result = get_suppress_tokens(tokenizer, None)

        # Should only have special tokens
        assert isinstance(result, tuple)


class TestBuildLogitFilters:
    """Tests for build_logit_filters function."""

    def _mock_tokenizer(self):
        """Create mock tokenizer for testing."""
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [220]
        tokenizer.eot = 50257
        tokenizer.timestamp_begin = 50364
        tokenizer.no_timestamps = 50363
        tokenizer.transcribe = 50359
        tokenizer.translate = 50358
        tokenizer.sot = 50258
        tokenizer.sot_prev = 50361
        tokenizer.sot_lm = 50360
        tokenizer.no_speech = 50362
        tokenizer.non_speech_tokens = [1, 2, 3]
        tokenizer.all_language_tokens = [50259, 50260, 50261]
        return tokenizer

    def test_default_options(self):
        """Test filter building with default options."""
        tokenizer = self._mock_tokenizer()
        options = DecodingOptions()

        filters = build_logit_filters(
            tokenizer, options, sample_begin=3, n_vocab=51865,
        )

        # OPT-NEW-8: With both suppress_blank and suppress_tokens enabled,
        # FusedSuppressFilter is used instead of separate filters
        # Should have 2 filters: FusedSuppressFilter, ApplyTimestampRules
        assert len(filters) == 2
        from tools.whisper_mlx.decoding import FusedSuppressFilter
        assert isinstance(filters[0], FusedSuppressFilter)
        assert isinstance(filters[1], ApplyTimestampRules)

    def test_no_suppress_blank(self):
        """Test with suppress_blank=False."""
        tokenizer = self._mock_tokenizer()
        options = DecodingOptions(suppress_blank=False)

        filters = build_logit_filters(
            tokenizer, options, sample_begin=3, n_vocab=51865,
        )

        # Should not have SuppressBlank
        assert not any(isinstance(f, SuppressBlank) for f in filters)

    def test_no_suppress_tokens(self):
        """Test with suppress_tokens=None."""
        tokenizer = self._mock_tokenizer()
        options = DecodingOptions(suppress_tokens=None)

        filters = build_logit_filters(
            tokenizer, options, sample_begin=3, n_vocab=51865,
        )

        # Should not have SuppressTokens
        assert not any(isinstance(f, SuppressTokens) for f in filters)

    def test_without_timestamps(self):
        """Test with without_timestamps=True."""
        tokenizer = self._mock_tokenizer()
        options = DecodingOptions(without_timestamps=True)

        filters = build_logit_filters(
            tokenizer, options, sample_begin=3, n_vocab=51865,
        )

        # Should not have ApplyTimestampRules
        assert not any(isinstance(f, ApplyTimestampRules) for f in filters)

    def test_with_audio_duration(self):
        """Test with audio_duration for variable-length mode."""
        tokenizer = self._mock_tokenizer()
        options = DecodingOptions()

        filters = build_logit_filters(
            tokenizer,
            options,
            sample_begin=3,
            n_vocab=51865,
            audio_duration=5.0,  # 5 seconds
        )

        # Find the ApplyTimestampRules filter
        timestamp_filter = None
        for f in filters:
            if isinstance(f, ApplyTimestampRules):
                timestamp_filter = f
                break

        assert timestamp_filter is not None
        # 5.0 seconds / 0.02 precision = 250
        assert timestamp_filter.max_audio_timestamp_index == 250

    def test_precision_parameter(self):
        """Test precision parameter affects max_initial_timestamp_index."""
        tokenizer = self._mock_tokenizer()
        options = DecodingOptions(max_initial_timestamp=1.0)

        filters = build_logit_filters(
            tokenizer,
            options,
            sample_begin=3,
            n_vocab=51865,
            precision=0.04,  # Lower precision
        )

        timestamp_filter = None
        for f in filters:
            if isinstance(f, ApplyTimestampRules):
                timestamp_filter = f
                break

        # 1.0 / 0.04 = 25
        assert timestamp_filter.max_initial_timestamp_index == 25


class TestApplyFilters:
    """Tests for apply_filters function."""

    def test_empty_filters(self):
        """Test with no filters."""
        logits = mx.ones((1, 100))
        tokens = mx.zeros((1, 5), dtype=mx.int32)

        result = apply_filters(logits, tokens, [])

        assert mx.allclose(result, logits).item()

    def test_single_filter(self):
        """Test with single filter."""
        filt = SuppressTokens([50], n_vocab=100)
        logits = mx.zeros((1, 100))
        tokens = mx.zeros((1, 5), dtype=mx.int32)

        result = apply_filters(logits, tokens, [filt])

        assert result[0, 50] == float("-inf")

    def test_multiple_filters_chain(self):
        """Test that multiple filters are applied in sequence."""
        filt1 = SuppressTokens([10], n_vocab=100)
        filt2 = SuppressTokens([20], n_vocab=100)
        logits = mx.zeros((1, 100))
        tokens = mx.zeros((1, 5), dtype=mx.int32)

        result = apply_filters(logits, tokens, [filt1, filt2])

        # Both tokens should be suppressed
        assert result[0, 10] == float("-inf")
        assert result[0, 20] == float("-inf")


class TestSuppressBlankEdgeCases:
    """Edge case tests for SuppressBlank."""

    def _mock_tokenizer(self):
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [220]
        tokenizer.eot = 50257
        return tokenizer

    def test_batch_processing(self):
        """Test batch processing."""
        tokenizer = self._mock_tokenizer()
        n_vocab = 51865
        filt = SuppressBlank(tokenizer, sample_begin=3, n_vocab=n_vocab)

        logits = mx.zeros((4, n_vocab))  # Batch of 4
        tokens = mx.zeros((4, 3), dtype=mx.int32)

        result = filt.apply(logits, tokens)

        # All batches should have blank suppressed
        for i in range(4):
            assert result[i, 220] == float("-inf")
            assert result[i, 50257] == float("-inf")


class TestApplyTimestampRulesEdgeCases:
    """Edge case tests for ApplyTimestampRules."""

    def _mock_tokenizer(self):
        tokenizer = MagicMock()
        tokenizer.timestamp_begin = 50364
        tokenizer.eot = 50257
        tokenizer.no_timestamps = None  # No no_timestamps token
        return tokenizer

    def test_no_no_timestamps_token(self):
        """Test when tokenizer has no no_timestamps attribute."""
        tokenizer = self._mock_tokenizer()
        n_vocab = 51865
        filt = ApplyTimestampRules(
            tokenizer,
            sample_begin=3,
            max_initial_timestamp_index=50,
            n_vocab=n_vocab,
        )

        logits = mx.zeros((1, n_vocab))
        tokens = mx.zeros((1, 5), dtype=mx.int32)

        # Should not raise
        result = filt.apply(logits, tokens)
        assert result.shape == logits.shape

    def test_double_timestamp_suppresses_timestamps(self):
        """Test that two timestamps in a row suppress further timestamps."""
        tokenizer = self._mock_tokenizer()
        n_vocab = 51865
        timestamp_begin = 50364
        filt = ApplyTimestampRules(
            tokenizer,
            sample_begin=3,
            max_initial_timestamp_index=50,
            n_vocab=n_vocab,
        )

        logits = mx.zeros((1, n_vocab))
        # Two timestamps in a row at the end
        tokens = mx.array(
            [[0, 0, 0, timestamp_begin + 10, timestamp_begin + 11]],
            dtype=mx.int32,
        )

        result = filt.apply(logits, tokens)

        # After two timestamps, should suppress more timestamps
        assert result[0, timestamp_begin] == float("-inf")


class TestIntegration:
    """Integration tests for the decoding module."""

    def _mock_tokenizer(self):
        """Create comprehensive mock tokenizer."""
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [220]
        tokenizer.eot = 50257
        tokenizer.timestamp_begin = 50364
        tokenizer.no_timestamps = 50363
        tokenizer.transcribe = 50359
        tokenizer.translate = 50358
        tokenizer.sot = 50258
        tokenizer.sot_prev = 50361
        tokenizer.sot_lm = 50360
        tokenizer.no_speech = 50362
        tokenizer.non_speech_tokens = [1, 2, 3]
        tokenizer.all_language_tokens = [50259, 50260]
        return tokenizer

    def test_full_decoding_pipeline(self):
        """Test complete filter pipeline."""
        tokenizer = self._mock_tokenizer()
        options = DecodingOptions()
        n_vocab = 51865

        filters = build_logit_filters(
            tokenizer, options, sample_begin=3, n_vocab=n_vocab,
        )

        logits = mx.zeros((1, n_vocab))
        tokens = mx.zeros((1, 3), dtype=mx.int32)  # At sample_begin

        result = apply_filters(logits, tokens, filters)

        # Verify various suppressions
        assert result[0, 50257] == float("-inf")  # EOT from blank filter
        assert result[0, 50359] == float("-inf")  # transcribe token
        assert result[0, 50363] == float("-inf")  # no_timestamps

    def test_decoding_result_with_computed_values(self):
        """Test DecodingResult with computed compression ratio."""
        text = "Hello, how are you today?"
        result = DecodingResult(
            tokens=[50364, 100, 200, 300, 50365, 50257],
            text=text,
            language="en",
            avg_logprob=-0.3,
            no_speech_prob=0.01,
            temperature=0.0,
            compression_ratio=compression_ratio(text),
        )

        assert result.compression_ratio > 0
        assert len(result.tokens) == 6
        assert result.text == text


class TestHallucinationDetection:
    """Tests for E4 hallucination detection functions."""

    def test_normal_text_not_hallucination(self):
        """Normal speech text should not be flagged."""
        text = "The quick brown fox jumps over the lazy dog."
        result = detect_hallucination(text)
        assert not result.is_hallucination
        assert result.confidence < 0.5
        assert len(result.patterns_matched) == 0

    def test_thanks_for_watching_pattern(self):
        """Common YouTube ending pattern should be detected."""
        text = "That's all for today. Thanks for watching!"
        result = detect_hallucination(text)
        assert len(result.patterns_matched) > 0
        assert any("thanks" in p.lower() for p in result.patterns_matched)

    def test_subscribe_pattern(self):
        """Subscribe/like pattern should be detected."""
        text = "Please subscribe and hit the like button for more content."
        result = detect_hallucination(text)
        assert len(result.patterns_matched) > 0

    def test_repeated_phrase_detection(self):
        """Repeated phrases should be detected as hallucination."""
        text = "Thank you. Thank you. Thank you. Thank you. Thank you."
        result = detect_hallucination(text)
        assert result.is_hallucination
        assert result.confidence >= 0.5
        assert result.repeated_phrase is not None
        assert result.repetition_count >= 3

    def test_empty_brackets_pattern(self):
        """Empty brackets are a hallucination indicator."""
        text = "And then [ ] we have this [ ] situation."
        result = detect_hallucination(text)
        assert len(result.patterns_matched) > 0

    def test_music_notation_pattern(self):
        """Music notation (♪) is often hallucinated."""
        text = "♪ La la la ♪"
        result = detect_hallucination(text)
        assert len(result.patterns_matched) > 0

    def test_repeated_punctuation(self):
        """Excessive repeated punctuation should be detected."""
        text = "What is going on here......... I don't know."
        result = detect_hallucination(text)
        assert len(result.patterns_matched) > 0

    def test_short_text_skipped(self):
        """Very short text should not be analyzed."""
        text = "Hi"
        result = detect_hallucination(text)
        assert not result.is_hallucination
        assert result.confidence == 0.0

    def test_empty_text(self):
        """Empty text should not be flagged."""
        result = detect_hallucination("")
        assert not result.is_hallucination
        assert result.confidence == 0.0

    def test_hallucination_result_dataclass(self):
        """Test HallucinationResult dataclass fields."""
        result = HallucinationResult(
            is_hallucination=True,
            confidence=0.8,
            patterns_matched=["pattern1", "pattern2"],
            repeated_phrase="thank you",
            repetition_count=5,
        )
        assert result.is_hallucination is True
        assert result.confidence == 0.8
        assert len(result.patterns_matched) == 2
        assert result.repeated_phrase == "thank you"
        assert result.repetition_count == 5

    def test_is_hallucination_convenience_function(self):
        """Test is_hallucination() convenience wrapper."""
        normal = "The weather is nice today."
        # Need 5+ repetitions for high enough confidence
        halluc = "Thank you. Thank you. Thank you. Thank you. Thank you. Thank you."

        assert not is_hallucination(normal)
        assert is_hallucination(halluc)

    def test_custom_threshold(self):
        """Test custom threshold for is_hallucination()."""
        text = "Thanks for watching this video."
        # Lower threshold should flag this
        result = detect_hallucination(text)
        if result.confidence > 0.2:
            assert is_hallucination(text, threshold=0.2)

    def test_multiple_patterns_increase_confidence(self):
        """Multiple hallucination patterns should increase confidence."""
        # Single pattern
        single = "Thanks for watching!"
        single_result = detect_hallucination(single)

        # Multiple patterns
        multiple = "Thanks for watching! Please subscribe and hit the like button."
        multiple_result = detect_hallucination(multiple)

        # Multiple patterns should have higher confidence
        assert multiple_result.confidence >= single_result.confidence

    def test_high_compression_ratio_contributes(self):
        """High compression ratio (repetitive text) should add to confidence."""
        # Highly repetitive text
        repetitive = "hello " * 50
        result = detect_hallucination(repetitive)
        # Should be flagged due to high compression ratio
        assert result.confidence > 0

    def test_case_insensitive_patterns(self):
        """Pattern matching should be case-insensitive."""
        text1 = "THANKS FOR WATCHING"
        text2 = "thanks for watching"
        text3 = "Thanks For Watching"

        result1 = detect_hallucination(text1)
        result2 = detect_hallucination(text2)
        result3 = detect_hallucination(text3)

        # All should detect the pattern
        assert len(result1.patterns_matched) > 0
        assert len(result2.patterns_matched) > 0
        assert len(result3.patterns_matched) > 0
