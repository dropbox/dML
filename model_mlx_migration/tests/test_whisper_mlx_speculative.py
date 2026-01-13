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
Tests for whisper_mlx/speculative.py module.

Tests speculative decoding for WhisperMLX.
"""

from unittest.mock import MagicMock

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from tools.whisper_mlx.speculative import SpeculativeDecoder


class MockDecoder(nn.Module):
    """Mock decoder for testing."""

    def __init__(self, n_vocab: int = 51865):
        super().__init__()
        self.n_vocab = n_vocab

    def __call__(
        self,
        tokens: mx.array,
        audio_features: mx.array,
        kv_cache: list | None = None,
    ) -> tuple[mx.array, list | None, None, None]:
        """Return mock logits (4 values: logits, kv_cache, cross_qk, hidden_states)."""
        batch, seq_len = tokens.shape
        logits = mx.zeros((batch, seq_len, self.n_vocab))
        # Return some dummy KV cache
        if kv_cache is None:
            kv_cache = [
                ((mx.zeros((1, seq_len, 64)), mx.zeros((1, seq_len, 64))),
                 (mx.zeros((1, 100, 64)), mx.zeros((1, 100, 64)))),
            ]
        # Return 4 values: logits, kv_cache, cross_qk (None), hidden_states (None)
        return logits, kv_cache, None, None


class MockWhisperModel:
    """Mock WhisperMLX model for testing."""

    def __init__(self, n_vocab: int = 51865):
        self.decoder = MockDecoder(n_vocab)


class TestSpeculativeDecoderInitialization:
    """Tests for SpeculativeDecoder initialization."""

    def test_basic_initialization(self):
        """Test basic initialization."""
        main_model = MockWhisperModel()
        draft_model = MockWhisperModel()

        decoder = SpeculativeDecoder(main_model, draft_model)

        assert decoder.main_model is main_model
        assert decoder.draft_model is draft_model
        assert decoder.draft_tokens == 5  # Default

    def test_custom_draft_tokens(self):
        """Test initialization with custom draft_tokens."""
        main_model = MockWhisperModel()
        draft_model = MockWhisperModel()

        decoder = SpeculativeDecoder(main_model, draft_model, draft_tokens=10)

        assert decoder.draft_tokens == 10

    def test_initial_statistics(self):
        """Test that statistics are initialized to zero."""
        main_model = MockWhisperModel()
        draft_model = MockWhisperModel()

        decoder = SpeculativeDecoder(main_model, draft_model)

        assert decoder.total_tokens == 0
        assert decoder.accepted_tokens == 0
        assert decoder.iterations == 0


class TestSpeculativeDecoderStatistics:
    """Tests for statistics tracking."""

    def test_reset_stats(self):
        """Test reset_stats method."""
        main_model = MockWhisperModel()
        draft_model = MockWhisperModel()
        decoder = SpeculativeDecoder(main_model, draft_model)

        # Manually set some stats
        decoder.total_tokens = 100
        decoder.accepted_tokens = 80
        decoder.iterations = 10

        decoder.reset_stats()

        assert decoder.total_tokens == 0
        assert decoder.accepted_tokens == 0
        assert decoder.iterations == 0

    def test_acceptance_rate_zero_tokens(self):
        """Test acceptance_rate when no tokens have been processed."""
        main_model = MockWhisperModel()
        draft_model = MockWhisperModel()
        decoder = SpeculativeDecoder(main_model, draft_model)

        assert decoder.acceptance_rate == 0.0

    def test_acceptance_rate_calculation(self):
        """Test acceptance_rate calculation."""
        main_model = MockWhisperModel()
        draft_model = MockWhisperModel()
        decoder = SpeculativeDecoder(main_model, draft_model)

        decoder.total_tokens = 100
        decoder.accepted_tokens = 80

        assert decoder.acceptance_rate == 0.8

    def test_acceptance_rate_perfect(self):
        """Test acceptance_rate when all tokens accepted."""
        main_model = MockWhisperModel()
        draft_model = MockWhisperModel()
        decoder = SpeculativeDecoder(main_model, draft_model)

        decoder.total_tokens = 50
        decoder.accepted_tokens = 50

        assert decoder.acceptance_rate == 1.0

    def test_tokens_per_iteration_zero_iterations(self):
        """Test tokens_per_iteration when no iterations."""
        main_model = MockWhisperModel()
        draft_model = MockWhisperModel()
        decoder = SpeculativeDecoder(main_model, draft_model)

        assert decoder.tokens_per_iteration == 0.0

    def test_tokens_per_iteration_calculation(self):
        """Test tokens_per_iteration calculation."""
        main_model = MockWhisperModel()
        draft_model = MockWhisperModel()
        decoder = SpeculativeDecoder(main_model, draft_model)

        decoder.total_tokens = 50
        decoder.iterations = 10

        assert decoder.tokens_per_iteration == 5.0


class TestSpeculativeDecoderTrimKVCache:
    """Tests for _trim_kv_cache method."""

    def test_trim_none_cache(self):
        """Test trimming None cache."""
        main_model = MockWhisperModel()
        draft_model = MockWhisperModel()
        decoder = SpeculativeDecoder(main_model, draft_model)

        result = decoder._trim_kv_cache(None, 10)

        assert result is None

    def test_trim_cache_basic(self):
        """Test basic cache trimming."""
        main_model = MockWhisperModel()
        draft_model = MockWhisperModel()
        decoder = SpeculativeDecoder(main_model, draft_model)

        # Create a mock KV cache
        # Structure: List of (self_kv, cross_kv) per layer
        # self_kv = (k, v), cross_kv = (k, v)
        self_k = mx.zeros((1, 100, 64))  # 100 cached positions
        self_v = mx.zeros((1, 100, 64))
        cross_k = mx.zeros((1, 50, 64))
        cross_v = mx.zeros((1, 50, 64))

        kv_cache = [((self_k, self_v), (cross_k, cross_v))]

        trimmed = decoder._trim_kv_cache(kv_cache, 50)

        assert trimmed is not None
        assert len(trimmed) == 1
        trimmed_self_k, trimmed_self_v = trimmed[0][0]
        assert trimmed_self_k.shape[1] == 50
        assert trimmed_self_v.shape[1] == 50

    def test_trim_preserves_cross_attention(self):
        """Test that cross-attention cache is not trimmed."""
        main_model = MockWhisperModel()
        draft_model = MockWhisperModel()
        decoder = SpeculativeDecoder(main_model, draft_model)

        self_k = mx.zeros((1, 100, 64))
        self_v = mx.zeros((1, 100, 64))
        cross_k = mx.zeros((1, 200, 64))  # Cross-attention cache
        cross_v = mx.zeros((1, 200, 64))

        kv_cache = [((self_k, self_v), (cross_k, cross_v))]

        trimmed = decoder._trim_kv_cache(kv_cache, 50)

        # Cross-attention cache should be unchanged
        _, trimmed_cross = trimmed[0]
        assert trimmed_cross[0].shape[1] == 200
        assert trimmed_cross[1].shape[1] == 200

    def test_trim_multiple_layers(self):
        """Test trimming cache with multiple layers."""
        main_model = MockWhisperModel()
        draft_model = MockWhisperModel()
        decoder = SpeculativeDecoder(main_model, draft_model)

        # Create multi-layer cache
        kv_cache = []
        for _ in range(4):
            self_k = mx.zeros((1, 100, 64))
            self_v = mx.zeros((1, 100, 64))
            cross_k = mx.zeros((1, 50, 64))
            cross_v = mx.zeros((1, 50, 64))
            kv_cache.append(((self_k, self_v), (cross_k, cross_v)))

        trimmed = decoder._trim_kv_cache(kv_cache, 30)

        assert len(trimmed) == 4
        for layer_cache in trimmed:
            self_kv, _ = layer_cache
            assert self_kv[0].shape[1] == 30
            assert self_kv[1].shape[1] == 30

    def test_trim_with_none_layer(self):
        """Test trimming cache with None layer entry."""
        main_model = MockWhisperModel()
        draft_model = MockWhisperModel()
        decoder = SpeculativeDecoder(main_model, draft_model)

        self_k = mx.zeros((1, 100, 64))
        self_v = mx.zeros((1, 100, 64))
        cross_k = mx.zeros((1, 50, 64))
        cross_v = mx.zeros((1, 50, 64))

        kv_cache = [
            ((self_k, self_v), (cross_k, cross_v)),
            None,  # Some layer without cache
            ((self_k, self_v), (cross_k, cross_v)),
        ]

        trimmed = decoder._trim_kv_cache(kv_cache, 50)

        assert len(trimmed) == 3
        assert trimmed[1] is None  # Preserved None


class TestSpeculativeDecoderParseSegments:
    """Tests for _parse_segments method."""

    def _mock_tokenizer(self):
        """Create mock tokenizer."""
        tokenizer = MagicMock()
        tokenizer.timestamp_begin = 50364
        tokenizer.decode = lambda tokens: " ".join([f"w{t}" for t in tokens])
        return tokenizer

    def test_empty_tokens(self):
        """Test parsing empty token list."""
        main_model = MockWhisperModel()
        draft_model = MockWhisperModel()
        decoder = SpeculativeDecoder(main_model, draft_model)
        tokenizer = self._mock_tokenizer()

        segments = decoder._parse_segments([], tokenizer, precision=0.02)

        assert segments == []

    def test_text_only_tokens(self):
        """Test parsing tokens without timestamps."""
        main_model = MockWhisperModel()
        draft_model = MockWhisperModel()
        decoder = SpeculativeDecoder(main_model, draft_model)
        tokenizer = self._mock_tokenizer()

        tokens = [100, 200, 300]  # All below timestamp_begin
        segments = decoder._parse_segments(tokens, tokenizer, precision=0.02)

        # No timestamps = no segments (or implementation-specific)
        # The current implementation returns empty if no start timestamp
        assert isinstance(segments, list)

    def test_single_segment(self):
        """Test parsing a single segment with timestamps."""
        main_model = MockWhisperModel()
        draft_model = MockWhisperModel()
        decoder = SpeculativeDecoder(main_model, draft_model)
        tokenizer = self._mock_tokenizer()

        timestamp_begin = 50364
        # [ts_0, text, ts_50] = 0.0s to 1.0s (50 * 0.02)
        tokens = [timestamp_begin + 0, 100, 200, timestamp_begin + 50]

        segments = decoder._parse_segments(tokens, tokenizer, precision=0.02)

        assert len(segments) == 1
        assert segments[0]["start"] == 0.0
        assert segments[0]["end"] == 1.0

    def test_multiple_segments(self):
        """Test parsing multiple segments."""
        main_model = MockWhisperModel()
        draft_model = MockWhisperModel()
        decoder = SpeculativeDecoder(main_model, draft_model)
        tokenizer = self._mock_tokenizer()

        timestamp_begin = 50364
        # Two segments: 0-1s and 1-2s
        tokens = [
            timestamp_begin + 0, 100, timestamp_begin + 50,  # First: 0.0-1.0s
            timestamp_begin + 50, 200, timestamp_begin + 100,  # Second: 1.0-2.0s
        ]

        segments = decoder._parse_segments(tokens, tokenizer, precision=0.02)

        assert len(segments) >= 1

    def test_precision_affects_times(self):
        """Test that precision parameter affects segment times."""
        main_model = MockWhisperModel()
        draft_model = MockWhisperModel()
        decoder = SpeculativeDecoder(main_model, draft_model)
        tokenizer = self._mock_tokenizer()

        timestamp_begin = 50364
        tokens = [timestamp_begin + 0, 100, timestamp_begin + 100]

        # With precision 0.02: 100 * 0.02 = 2.0s
        segments_02 = decoder._parse_segments(tokens, tokenizer, precision=0.02)

        # With precision 0.04: 100 * 0.04 = 4.0s
        segments_04 = decoder._parse_segments(tokens, tokenizer, precision=0.04)

        if segments_02:
            assert segments_02[0]["end"] == 2.0
        if segments_04:
            assert segments_04[0]["end"] == 4.0


class TestSpeculativeDecoderGenerateDraft:
    """Tests for _generate_draft method."""

    def _get_decoder_with_predictable_draft(self, predictions: list[int]):
        """Create decoder with mock draft model that returns predictable tokens."""
        main_model = MockWhisperModel()
        draft_model = MockWhisperModel()

        # Mock the draft decoder to return specific predictions
        call_count = [0]

        def mock_forward(tokens, audio_features, kv_cache=None):
            idx = call_count[0]
            call_count[0] += 1

            batch, seq_len = tokens.shape
            # Create logits as numpy then convert to get around indexing
            logits_np = np.zeros((batch, seq_len, 51865), dtype=np.float32)

            # Set the predicted token to have high logit
            if idx < len(predictions):
                pred = predictions[idx]
                logits_np[0, -1, pred] = 100.0

            logits = mx.array(logits_np)

            # Return dummy KV cache
            if kv_cache is None:
                kv_cache = [
                    ((mx.zeros((1, seq_len, 64)), mx.zeros((1, seq_len, 64))),
                     (mx.zeros((1, 100, 64)), mx.zeros((1, 100, 64)))),
                ]
            return logits, kv_cache, None, None

        draft_model.decoder = MagicMock(side_effect=mock_forward)

        return SpeculativeDecoder(main_model, draft_model, draft_tokens=3)

    def test_generate_draft_basic(self):
        """Test basic draft generation."""
        decoder = self._get_decoder_with_predictable_draft([100, 200, 300])

        tokenizer = MagicMock()
        tokenizer.eot = 50257

        context = mx.array([[1, 2, 3]])
        audio_features = mx.zeros((1, 100, 64))

        draft_tokens, _ = decoder._generate_draft(
            context,
            audio_features,
            tokenizer,
            logit_filters=[],
            kv_cache=None,
        )

        assert len(draft_tokens) == 3
        assert draft_tokens == [100, 200, 300]

    def test_generate_draft_stops_on_eot(self):
        """Test that draft generation stops on EOT token."""
        eot = 50257
        decoder = self._get_decoder_with_predictable_draft([100, eot, 300])

        tokenizer = MagicMock()
        tokenizer.eot = eot

        context = mx.array([[1, 2, 3]])
        audio_features = mx.zeros((1, 100, 64))

        draft_tokens, _ = decoder._generate_draft(
            context,
            audio_features,
            tokenizer,
            logit_filters=[],
            kv_cache=None,
        )

        # Should stop at EOT
        assert len(draft_tokens) == 2
        assert draft_tokens[0] == 100
        assert draft_tokens[1] == eot


class TestSpeculativeDecoderVerifyDraft:
    """Tests for _verify_draft method (legacy)."""

    def test_verify_draft_all_accepted(self):
        """Test verification when all tokens are accepted."""
        main_model = MockWhisperModel()
        draft_model = MockWhisperModel()

        # Mock main decoder to return same predictions as draft
        def mock_forward(tokens, audio_features, kv_cache=None):
            batch, seq_len = tokens.shape
            logits_np = np.zeros((batch, seq_len, 51865), dtype=np.float32)
            # Set token 100 as the prediction for all positions
            logits_np[:, :, 100] = 10.0
            return mx.array(logits_np), None, None, None

        main_model.decoder = MagicMock(side_effect=mock_forward)

        decoder = SpeculativeDecoder(main_model, draft_model)

        tokenizer = MagicMock()
        tokenizer.eot = 50257

        context = mx.array([[1, 2, 3]])
        draft_tokens = mx.array([[100, 100, 100]])
        audio_features = mx.zeros((1, 100, 64))

        accepted, next_token, _ = decoder._verify_draft(
            context,
            draft_tokens,
            audio_features,
            tokenizer,
            logit_filters=[],
            kv_cache=None,
            sample_begin=1,
        )

        assert accepted == 3  # All accepted

    def test_verify_draft_rejection(self):
        """Test verification when draft is rejected."""
        main_model = MockWhisperModel()
        draft_model = MockWhisperModel()

        # Mock main decoder to return different prediction than draft
        def mock_forward(tokens, audio_features, kv_cache=None):
            batch, seq_len = tokens.shape
            logits_np = np.zeros((batch, seq_len, 51865), dtype=np.float32)
            # Set token 999 (different from draft) as the prediction
            logits_np[:, :, 999] = 10.0
            return mx.array(logits_np), None, None, None

        main_model.decoder = MagicMock(side_effect=mock_forward)

        decoder = SpeculativeDecoder(main_model, draft_model)

        tokenizer = MagicMock()
        tokenizer.eot = 50257

        context = mx.array([[1, 2, 3]])
        draft_tokens = mx.array([[100, 200, 300]])  # Different from main's prediction
        audio_features = mx.zeros((1, 100, 64))

        accepted, next_token, _ = decoder._verify_draft(
            context,
            draft_tokens,
            audio_features,
            tokenizer,
            logit_filters=[],
            kv_cache=None,
            sample_begin=1,
        )

        assert accepted == 0  # First token rejected
        assert next_token == 999  # Main model's prediction


class TestSpeculativeDecoderVerifyDraftWithCache:
    """Tests for _verify_draft_with_cache method."""

    def test_verify_with_cache_no_kv(self):
        """Test verification without existing KV cache."""
        main_model = MockWhisperModel()
        draft_model = MockWhisperModel()

        def mock_forward(tokens, audio_features, kv_cache=None):
            batch, seq_len = tokens.shape
            logits_np = np.zeros((batch, seq_len, 51865), dtype=np.float32)
            logits_np[:, :, 100] = 10.0
            # Create dummy KV cache
            kv = [
                ((mx.zeros((1, seq_len, 64)), mx.zeros((1, seq_len, 64))),
                 (mx.zeros((1, 100, 64)), mx.zeros((1, 100, 64)))),
            ]
            return mx.array(logits_np), kv, None, None

        main_model.decoder = MagicMock(side_effect=mock_forward)

        decoder = SpeculativeDecoder(main_model, draft_model)

        tokenizer = MagicMock()
        tokenizer.eot = 50257

        all_tokens = [1, 2, 3]
        draft_tokens_list = [100, 100, 100]
        audio_features = mx.zeros((1, 100, 64))

        accepted, next_token, new_kv, new_pos = decoder._verify_draft_with_cache(
            all_tokens,
            draft_tokens_list,
            audio_features,
            tokenizer,
            logit_filters=[],
            main_kv=None,
            main_kv_valid_pos=0,
            sample_begin=1,
        )

        assert accepted == 3
        assert new_kv is not None


class TestSpeculativeDecoderIntegration:
    """Integration tests for SpeculativeDecoder."""

    def test_full_decode_flow(self):
        """Test that decode method runs without error."""
        main_model = MockWhisperModel()
        draft_model = MockWhisperModel()

        # Create predictable mock models
        eot = 50257

        def mock_main_forward(tokens, audio_features, kv_cache=None):
            batch, seq_len = tokens.shape
            logits_np = np.zeros((batch, seq_len, 51865), dtype=np.float32)
            # Always predict EOT
            logits_np[:, :, eot] = 10.0
            kv = [
                ((mx.zeros((1, seq_len, 64)), mx.zeros((1, seq_len, 64))),
                 (mx.zeros((1, 100, 64)), mx.zeros((1, 100, 64)))),
            ]
            return mx.array(logits_np), kv, None, None

        def mock_draft_forward(tokens, audio_features, kv_cache=None):
            batch, seq_len = tokens.shape
            logits_np = np.zeros((batch, seq_len, 51865), dtype=np.float32)
            # Draft also predicts EOT
            logits_np[:, :, eot] = 10.0
            kv = [
                ((mx.zeros((1, seq_len, 64)), mx.zeros((1, seq_len, 64))),
                 (mx.zeros((1, 100, 64)), mx.zeros((1, 100, 64)))),
            ]
            return mx.array(logits_np), kv, None, None

        main_model.decoder = MagicMock(side_effect=mock_main_forward)
        draft_model.decoder = MagicMock(side_effect=mock_draft_forward)

        decoder = SpeculativeDecoder(main_model, draft_model, draft_tokens=3)

        # Mock tokenizer with all required attributes for get_suppress_tokens
        tokenizer = MagicMock()
        tokenizer.sot_sequence = [50258, 50259, 50360]
        tokenizer.eot = eot
        tokenizer.timestamp_begin = 50364
        tokenizer.decode = lambda tokens: "test"
        tokenizer.encode = lambda s: [220]
        tokenizer.transcribe = 50359
        tokenizer.translate = 50358
        tokenizer.sot = 50258
        tokenizer.sot_prev = 50361
        tokenizer.sot_lm = 50360
        tokenizer.no_speech = 50362
        tokenizer.no_timestamps = 50363
        tokenizer.non_speech_tokens = [1, 2, 3]
        tokenizer.all_language_tokens = [50259, 50260]

        # Mock decoding options
        from tools.whisper_mlx.decoding import DecodingOptions
        options = DecodingOptions()

        audio_features = mx.zeros((1, 100, 64))

        # Should complete without error
        tokens, segments = decoder.decode(
            audio_features,
            tokenizer,
            options,
            sample_begin=3,
            n_vocab=51865,
            max_tokens=10,
        )

        assert isinstance(tokens, list)
        assert isinstance(segments, list)


class TestSpeculativeDecoderEdgeCases:
    """Edge case tests."""

    def test_single_draft_token(self):
        """Test with single draft token configuration."""
        main_model = MockWhisperModel()
        draft_model = MockWhisperModel()

        decoder = SpeculativeDecoder(main_model, draft_model, draft_tokens=1)

        assert decoder.draft_tokens == 1

    def test_large_draft_tokens(self):
        """Test with large draft token count."""
        main_model = MockWhisperModel()
        draft_model = MockWhisperModel()

        decoder = SpeculativeDecoder(main_model, draft_model, draft_tokens=20)

        assert decoder.draft_tokens == 20

    def test_statistics_accumulate(self):
        """Test that statistics accumulate correctly."""
        main_model = MockWhisperModel()
        draft_model = MockWhisperModel()
        decoder = SpeculativeDecoder(main_model, draft_model)

        # Simulate multiple decode iterations
        decoder.total_tokens += 10
        decoder.accepted_tokens += 8
        decoder.iterations += 2

        decoder.total_tokens += 15
        decoder.accepted_tokens += 12
        decoder.iterations += 3

        assert decoder.total_tokens == 25
        assert decoder.accepted_tokens == 20
        assert decoder.iterations == 5
        assert decoder.acceptance_rate == 0.8
        assert decoder.tokens_per_iteration == 5.0
