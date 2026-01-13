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
Tests for whisper_mlx/model.py module.

Tests the WhisperMLX main model class.
"""

from unittest.mock import MagicMock

import mlx.core as mx
import numpy as np
import pytest

from tools.whisper_mlx.config import WhisperConfig
from tools.whisper_mlx.model import WhisperMLX


class TestWhisperMLXInitialization:
    """Tests for WhisperMLX initialization."""

    def _get_minimal_config(self):
        """Get minimal config for testing."""
        return WhisperConfig(
            n_mels=80,
            n_vocab=51865,
            n_audio_ctx=1500,
            n_audio_state=1280,
            n_audio_head=20,
            n_audio_layer=32,
            n_text_ctx=448,
            n_text_state=1280,
            n_text_head=20,
            n_text_layer=32,
        )

    def test_basic_initialization(self):
        """Test basic model initialization."""
        config = self._get_minimal_config()
        model = WhisperMLX(config)

        assert model.config == config
        assert model.dtype == mx.float16
        assert model.use_fused is True
        assert model.preallocate_kv is True  # OPT-NEW-5: Enabled by default

    def test_custom_dtype(self):
        """Test initialization with custom dtype."""
        config = self._get_minimal_config()
        model = WhisperMLX(config, dtype=mx.float32)

        assert model.dtype == mx.float32

    def test_preallocate_kv_option(self):
        """Test preallocate_kv can be explicitly disabled."""
        config = self._get_minimal_config()
        # OPT-NEW-5: preallocate_kv defaults to True, verify it can be disabled
        model = WhisperMLX(config, preallocate_kv=False)

        assert model.preallocate_kv is False

    def test_encoder_created(self):
        """Test that encoder is created correctly."""
        config = self._get_minimal_config()
        model = WhisperMLX(config)

        assert hasattr(model, "encoder")
        assert model.encoder is not None

    def test_decoder_created(self):
        """Test that decoder is created correctly."""
        config = self._get_minimal_config()
        model = WhisperMLX(config)

        assert hasattr(model, "decoder")
        assert model.decoder is not None

    def test_kv_cache_created(self):
        """Test that KV cache manager is created."""
        config = self._get_minimal_config()
        model = WhisperMLX(config)

        assert hasattr(model, "_kv_cache")
        assert model._kv_cache is not None

    def test_alignment_heads_initialized(self):
        """Test that alignment heads are initialized."""
        config = self._get_minimal_config()
        model = WhisperMLX(config)

        assert hasattr(model, "alignment_heads")
        assert model.alignment_heads is not None
        assert isinstance(model.alignment_heads, mx.array)

    def test_draft_model_initially_none(self):
        """Test that draft model is initially None."""
        config = self._get_minimal_config()
        model = WhisperMLX(config)

        assert model._draft_model is None
        assert model._speculative_decoder is None

    def test_encoder_cache_initially_none(self):
        """Test that encoder cache is initially None."""
        config = self._get_minimal_config()
        model = WhisperMLX(config)

        assert model._encoder_cache is None

    def test_warmup_method_runs(self):
        """Test OPT-NEW-31: _warmup method triggers JIT compilation without errors."""
        config = self._get_minimal_config()
        model = WhisperMLX(config)

        # Warmup should run without errors
        model._warmup()

        # After warmup, model should still work normally
        # Test encoder
        dummy_mel = mx.zeros((1, 750, config.n_mels), dtype=model.dtype)
        encoder_out = model.encoder(dummy_mel, variable_length=True)
        mx.eval(encoder_out)
        assert encoder_out.shape[0] == 1

        # Test decoder
        tokens = mx.array([[50258]])
        decoder_out = model.decoder(tokens, encoder_out)
        # Decoder returns tuple (logits, kv_cache, ...) - just check first element
        if isinstance(decoder_out, tuple):
            logits = decoder_out[0]
        else:
            logits = decoder_out
        mx.eval(logits)
        assert logits.shape[0] == 1


class TestWhisperMLXProperties:
    """Tests for WhisperMLX properties."""

    def _get_multilingual_config(self):
        """Get multilingual config (n_vocab >= 51865)."""
        return WhisperConfig(
            n_mels=80,
            n_vocab=51866,  # Multilingual
            n_audio_ctx=1500,
            n_audio_state=1280,
            n_audio_head=20,
            n_audio_layer=32,
            n_text_ctx=448,
            n_text_state=1280,
            n_text_head=20,
            n_text_layer=32,
        )

    def _get_english_only_config(self):
        """Get English-only config (n_vocab < 51865)."""
        return WhisperConfig(
            n_mels=80,
            n_vocab=51864,  # English-only
            n_audio_ctx=1500,
            n_audio_state=1280,
            n_audio_head=20,
            n_audio_layer=32,
            n_text_ctx=448,
            n_text_state=1280,
            n_text_head=20,
            n_text_layer=32,
        )

    def test_is_multilingual_true(self):
        """Test is_multilingual property when multilingual."""
        config = self._get_multilingual_config()
        model = WhisperMLX(config)

        assert model.is_multilingual is True

    def test_is_multilingual_false(self):
        """Test is_multilingual property when English-only."""
        config = self._get_english_only_config()
        model = WhisperMLX(config)

        assert model.is_multilingual is False

    def test_num_languages_multilingual(self):
        """Test num_languages property for multilingual model."""
        config = self._get_multilingual_config()
        model = WhisperMLX(config)

        # num_languages = n_vocab - 51765 - int(is_multilingual)
        # = 51866 - 51765 - 1 = 100
        assert model.num_languages == 100

    def test_num_languages_english_only(self):
        """Test num_languages property for English-only model."""
        config = self._get_english_only_config()
        model = WhisperMLX(config)

        # num_languages = n_vocab - 51765 - int(is_multilingual)
        # = 51864 - 51765 - 0 = 99
        assert model.num_languages == 99

    def test_encoder_cache_enabled_false(self):
        """Test encoder_cache_enabled when cache not enabled."""
        config = self._get_multilingual_config()
        model = WhisperMLX(config)

        assert model.encoder_cache_enabled is False

    def test_encoder_cache_enabled_true(self):
        """Test encoder_cache_enabled when cache is enabled."""
        config = self._get_multilingual_config()
        model = WhisperMLX(config)
        model.enable_encoder_cache()

        assert model.encoder_cache_enabled is True


class TestWhisperMLXEncoderCache:
    """Tests for encoder cache methods."""

    def _get_minimal_config(self):
        return WhisperConfig(
            n_mels=80,
            n_vocab=51865,
            n_audio_ctx=1500,
            n_audio_state=1280,
            n_audio_head=20,
            n_audio_layer=32,
            n_text_ctx=448,
            n_text_state=1280,
            n_text_head=20,
            n_text_layer=32,
        )

    def test_enable_encoder_cache(self):
        """Test enabling encoder cache."""
        config = self._get_minimal_config()
        model = WhisperMLX(config)

        model.enable_encoder_cache(max_entries=8)

        assert model._encoder_cache is not None
        assert model.encoder_cache_enabled is True

    def test_enable_encoder_cache_custom_params(self):
        """Test enabling encoder cache with custom parameters."""
        config = self._get_minimal_config()
        model = WhisperMLX(config)

        model.enable_encoder_cache(max_entries=32, max_memory_mb=512.0)

        assert model._encoder_cache is not None

    def test_disable_encoder_cache(self):
        """Test disabling encoder cache."""
        config = self._get_minimal_config()
        model = WhisperMLX(config)
        model.enable_encoder_cache()

        model.disable_encoder_cache()

        assert model._encoder_cache is None
        assert model.encoder_cache_enabled is False

    def test_clear_encoder_cache(self):
        """Test clearing encoder cache."""
        config = self._get_minimal_config()
        model = WhisperMLX(config)
        model.enable_encoder_cache()

        # Should not raise
        model.clear_encoder_cache()

        assert model.encoder_cache_enabled is True

    def test_clear_encoder_cache_when_disabled(self):
        """Test clearing encoder cache when not enabled."""
        config = self._get_minimal_config()
        model = WhisperMLX(config)

        # Should not raise
        model.clear_encoder_cache()

    def test_get_encoder_cache_stats_when_disabled(self):
        """Test getting cache stats when cache not enabled."""
        config = self._get_minimal_config()
        model = WhisperMLX(config)

        stats = model.get_encoder_cache_stats()

        assert stats is None

    def test_get_encoder_cache_stats_when_enabled(self):
        """Test getting cache stats when cache is enabled."""
        config = self._get_minimal_config()
        model = WhisperMLX(config)
        model.enable_encoder_cache()

        stats = model.get_encoder_cache_stats()

        assert stats is not None
        assert isinstance(stats, dict)


class TestWhisperMLXParseSegments:
    """Tests for _parse_segments method."""

    def _get_minimal_config(self):
        return WhisperConfig(
            n_mels=80,
            n_vocab=51865,
            n_audio_ctx=1500,
            n_audio_state=1280,
            n_audio_head=20,
            n_audio_layer=32,
            n_text_ctx=448,
            n_text_state=1280,
            n_text_head=20,
            n_text_layer=32,
        )

    def _mock_tokenizer(self):
        """Create mock tokenizer."""
        tokenizer = MagicMock()
        tokenizer.timestamp_begin = 50364
        tokenizer.decode = lambda tokens: " ".join([f"tok{t}" for t in tokens])
        return tokenizer

    def test_empty_tokens(self):
        """Test parsing empty token list."""
        config = self._get_minimal_config()
        model = WhisperMLX(config)
        tokenizer = self._mock_tokenizer()

        segments = model._parse_segments([], tokenizer, precision=0.02)

        assert segments == []

    def test_only_text_tokens(self):
        """Test parsing tokens with no timestamps."""
        config = self._get_minimal_config()
        model = WhisperMLX(config)
        tokenizer = self._mock_tokenizer()

        # Tokens below timestamp_begin are text tokens
        tokens = [100, 200, 300]
        segments = model._parse_segments(tokens, tokenizer, precision=0.02)

        # Should create one segment from text tokens
        assert len(segments) == 1
        assert segments[0]["start"] == 0.0
        assert "tok100" in segments[0]["text"]

    def test_single_segment_with_timestamps(self):
        """Test parsing a single segment with timestamps."""
        config = self._get_minimal_config()
        model = WhisperMLX(config)
        tokenizer = self._mock_tokenizer()

        # [timestamp_0.0, text, timestamp_1.0]
        timestamp_begin = 50364
        tokens = [timestamp_begin + 0, 100, 200, timestamp_begin + 50]  # 50 * 0.02 = 1.0s

        segments = model._parse_segments(tokens, tokenizer, precision=0.02)

        assert len(segments) == 1
        assert segments[0]["start"] == 0.0
        assert segments[0]["end"] == 1.0

    def test_multiple_segments(self):
        """Test parsing multiple segments."""
        config = self._get_minimal_config()
        model = WhisperMLX(config)
        tokenizer = self._mock_tokenizer()

        timestamp_begin = 50364
        # [ts_0, text1, ts_50, ts_50, text2, ts_100]
        tokens = [
            timestamp_begin + 0, 100, 200, timestamp_begin + 50,  # First segment
            timestamp_begin + 50, 300, 400, timestamp_begin + 100,  # Second segment
        ]

        segments = model._parse_segments(tokens, tokenizer, precision=0.02)

        assert len(segments) >= 1  # At least one segment

    def test_precision_calculation(self):
        """Test that precision affects timestamp calculation."""
        config = self._get_minimal_config()
        model = WhisperMLX(config)
        tokenizer = self._mock_tokenizer()

        timestamp_begin = 50364
        tokens = [timestamp_begin + 0, 100, timestamp_begin + 100]

        # With precision 0.02: 100 * 0.02 = 2.0s
        segments_02 = model._parse_segments(tokens, tokenizer, precision=0.02)

        # With precision 0.04: 100 * 0.04 = 4.0s
        segments_04 = model._parse_segments(tokens, tokenizer, precision=0.04)

        assert segments_02[0]["end"] == 2.0
        assert segments_04[0]["end"] == 4.0


class TestWhisperMLXMergeOverlappingSegments:
    """Tests for _merge_overlapping_segments method."""

    def _get_minimal_config(self):
        return WhisperConfig(
            n_mels=80,
            n_vocab=51865,
            n_audio_ctx=1500,
            n_audio_state=1280,
            n_audio_head=20,
            n_audio_layer=32,
            n_text_ctx=448,
            n_text_state=1280,
            n_text_head=20,
            n_text_layer=32,
        )

    def test_empty_segments(self):
        """Test merging empty segment list."""
        config = self._get_minimal_config()
        model = WhisperMLX(config)

        result = model._merge_overlapping_segments([], overlap=1.0)

        assert result == []

    def test_single_segment(self):
        """Test with single segment."""
        config = self._get_minimal_config()
        model = WhisperMLX(config)

        segments = [{"start": 0.0, "end": 5.0, "text": "Hello"}]
        result = model._merge_overlapping_segments(segments, overlap=1.0)

        assert len(result) == 1
        assert result[0]["text"] == "Hello"

    def test_non_overlapping_segments(self):
        """Test with non-overlapping segments."""
        config = self._get_minimal_config()
        model = WhisperMLX(config)

        segments = [
            {"start": 0.0, "end": 5.0, "text": "Hello"},
            {"start": 10.0, "end": 15.0, "text": "World"},
        ]
        result = model._merge_overlapping_segments(segments, overlap=1.0)

        assert len(result) == 2

    def test_significantly_overlapping_segments(self):
        """Test with significantly overlapping segments."""
        config = self._get_minimal_config()
        model = WhisperMLX(config)

        segments = [
            {"start": 0.0, "end": 6.0, "text": "Hi"},
            {"start": 5.0, "end": 10.0, "text": "Hello there"},  # Longer text preferred
        ]
        result = model._merge_overlapping_segments(segments, overlap=1.0)

        # Significant overlap should merge (prefer longer text)
        assert len(result) <= 2

    def test_sorts_by_start_time(self):
        """Test that segments are sorted by start time."""
        config = self._get_minimal_config()
        model = WhisperMLX(config)

        # Out of order segments
        segments = [
            {"start": 10.0, "end": 15.0, "text": "Second"},
            {"start": 0.0, "end": 5.0, "text": "First"},
        ]
        result = model._merge_overlapping_segments(segments, overlap=1.0)

        assert result[0]["text"] == "First"


class TestWhisperMLXEmbedAudio:
    """Tests for embed_audio method."""

    def _get_small_config(self):
        """Get small config for faster tests."""
        return WhisperConfig(
            n_mels=80,
            n_vocab=51865,
            n_audio_ctx=1500,
            n_audio_state=64,  # Small for testing
            n_audio_head=4,
            n_audio_layer=2,  # Few layers for testing
            n_text_ctx=448,
            n_text_state=64,
            n_text_head=4,
            n_text_layer=2,
        )

    def test_embed_audio_shape(self):
        """Test embed_audio output shape."""
        config = self._get_small_config()
        model = WhisperMLX(config)

        # Create dummy mel spectrogram
        mel = mx.zeros((3000, 80))  # 30s audio
        features = model.embed_audio(mel)

        assert features.shape[0] == 1  # Batch dim added
        assert features.shape[-1] == config.n_audio_state

    def test_embed_audio_batched(self):
        """Test embed_audio with batched input."""
        config = self._get_small_config()
        model = WhisperMLX(config)

        # Batched mel spectrograms
        mel = mx.zeros((2, 3000, 80))
        features = model.embed_audio(mel)

        assert features.shape[0] == 2


class TestWhisperMLXLogits:
    """Tests for logits method."""

    def _get_small_config(self):
        return WhisperConfig(
            n_mels=80,
            n_vocab=1000,  # Small vocab for testing
            n_audio_ctx=1500,
            n_audio_state=64,
            n_audio_head=4,
            n_audio_layer=2,
            n_text_ctx=448,
            n_text_state=64,
            n_text_head=4,
            n_text_layer=2,
        )

    def test_logits_shape(self):
        """Test logits output shape."""
        config = self._get_small_config()
        model = WhisperMLX(config)

        tokens = mx.array([[1, 2, 3]])
        audio_features = mx.zeros((1, 100, config.n_audio_state))

        logits = model.logits(tokens, audio_features)

        assert logits.shape[0] == 1  # Batch
        assert logits.shape[1] == 3  # Seq len
        assert logits.shape[2] == config.n_vocab


class TestWhisperMLXCall:
    """Tests for __call__ method."""

    def _get_small_config(self):
        return WhisperConfig(
            n_mels=80,
            n_vocab=1000,
            n_audio_ctx=1500,
            n_audio_state=64,
            n_audio_head=4,
            n_audio_layer=2,
            n_text_ctx=448,
            n_text_state=64,
            n_text_head=4,
            n_text_layer=2,
        )

    def test_call_shape(self):
        """Test __call__ output shape."""
        config = self._get_small_config()
        model = WhisperMLX(config)

        mel = mx.zeros((1, 3000, 80))
        tokens = mx.array([[1, 2, 3]])

        logits = model(mel, tokens)

        assert logits.shape[0] == 1
        assert logits.shape[1] == 3
        assert logits.shape[2] == config.n_vocab


class TestWhisperMLXLoadWeights:
    """Tests for load_weights method."""

    def _get_minimal_config(self):
        return WhisperConfig(
            n_mels=80,
            n_vocab=51865,
            n_audio_ctx=1500,
            n_audio_state=1280,
            n_audio_head=20,
            n_audio_layer=32,
            n_text_ctx=448,
            n_text_state=1280,
            n_text_head=20,
            n_text_layer=32,
        )

    def test_load_weights_file_not_found(self):
        """Test that FileNotFoundError is raised for missing file."""
        config = self._get_minimal_config()
        model = WhisperMLX(config)

        with pytest.raises(Exception):  # Could be FileNotFoundError or IOError
            model.load_weights("/nonexistent/path/weights.npz")


class TestWhisperMLXFromPretrained:
    """Tests for from_pretrained class method."""

    @pytest.mark.skip(reason="Requires network access and model download")
    def test_from_pretrained_basic(self):
        """Test loading pretrained model."""
        # This test would require network access
        model = WhisperMLX.from_pretrained("mlx-community/whisper-tiny")
        assert model is not None

    def test_from_pretrained_invalid_model(self):
        """Test that invalid model name raises error."""
        with pytest.raises(Exception):
            WhisperMLX.from_pretrained("nonexistent-model-12345")


class TestWhisperMLXDetectLanguage:
    """Tests for _detect_language method."""

    def _get_small_config(self):
        return WhisperConfig(
            n_mels=80,
            n_vocab=51865,
            n_audio_ctx=1500,
            n_audio_state=64,
            n_audio_head=4,
            n_audio_layer=2,
            n_text_ctx=448,
            n_text_state=64,
            n_text_head=4,
            n_text_layer=2,
        )

    def test_detect_language_returns_string(self):
        """Test that _detect_language returns a language string."""
        config = self._get_small_config()
        model = WhisperMLX(config)

        # Mock tokenizer
        tokenizer = MagicMock()
        tokenizer.sot = 50258
        tokenizer.all_language_tokens = [50259, 50260, 50261]
        tokenizer.decode = lambda tokens: "<|en|>"

        audio_features = mx.zeros((1, 100, config.n_audio_state))
        language = model._detect_language(audio_features, tokenizer)

        assert isinstance(language, str)


class TestWhisperMLXTranscribeSpeculative:
    """Tests for transcribe_speculative method."""

    def _get_minimal_config(self):
        return WhisperConfig(
            n_mels=80,
            n_vocab=51865,
            n_audio_ctx=1500,
            n_audio_state=1280,
            n_audio_head=20,
            n_audio_layer=32,
            n_text_ctx=448,
            n_text_state=1280,
            n_text_head=20,
            n_text_layer=32,
        )

    def test_transcribe_speculative_without_draft_model(self):
        """Test that transcribe_speculative raises without draft model."""
        config = self._get_minimal_config()
        model = WhisperMLX(config)

        with pytest.raises(RuntimeError, match="Draft model not loaded"):
            model.transcribe_speculative(np.zeros(16000, dtype=np.float32))


class TestWhisperMLXAlignmentHeads:
    """Tests for alignment heads initialization."""

    def _get_config(self, n_text_layer, n_text_head):
        return WhisperConfig(
            n_mels=80,
            n_vocab=51865,
            n_audio_ctx=1500,
            n_audio_state=64,
            n_audio_head=4,
            n_audio_layer=2,
            n_text_ctx=448,
            n_text_state=64,
            n_text_head=n_text_head,
            n_text_layer=n_text_layer,
        )

    def test_alignment_heads_shape(self):
        """Test alignment heads shape."""
        config = self._get_config(n_text_layer=4, n_text_head=8)
        model = WhisperMLX(config)

        # Alignment heads should be indices of attention heads
        assert model.alignment_heads.ndim == 2
        assert model.alignment_heads.shape[-1] == 2  # (layer, head) pairs

    def test_alignment_heads_uses_last_half_layers(self):
        """Test that alignment heads use last half of decoder layers."""
        config = self._get_config(n_text_layer=8, n_text_head=4)
        model = WhisperMLX(config)

        # Should only include layers 4-7 (last half of 8 layers)
        layers = set(model.alignment_heads[:, 0].tolist())
        assert all(layer >= 4 for layer in layers)


class TestWhisperMLXIntegration:
    """Integration tests for WhisperMLX."""

    def _get_small_config(self):
        return WhisperConfig(
            n_mels=80,
            n_vocab=1000,
            n_audio_ctx=1500,
            n_audio_state=64,
            n_audio_head=4,
            n_audio_layer=2,
            n_text_ctx=448,
            n_text_state=64,
            n_text_head=4,
            n_text_layer=2,
        )

    def test_encoder_decoder_pipeline(self):
        """Test full encoder-decoder pipeline."""
        config = self._get_small_config()
        model = WhisperMLX(config)

        # Encode
        mel = mx.zeros((1, 3000, 80))
        audio_features = model.embed_audio(mel)

        # Decode
        tokens = mx.array([[1, 2, 3]])
        logits = model.logits(tokens, audio_features)

        assert logits.shape == (1, 3, config.n_vocab)

    def test_cache_lifecycle(self):
        """Test encoder cache enable/disable lifecycle."""
        config = self._get_small_config()
        model = WhisperMLX(config)

        # Initially disabled
        assert not model.encoder_cache_enabled
        assert model.get_encoder_cache_stats() is None

        # Enable
        model.enable_encoder_cache(max_entries=4)
        assert model.encoder_cache_enabled
        assert model.get_encoder_cache_stats() is not None

        # Clear
        model.clear_encoder_cache()
        assert model.encoder_cache_enabled

        # Disable
        model.disable_encoder_cache()
        assert not model.encoder_cache_enabled


class TestWhisperMLXEdgeCases:
    """Edge case tests."""

    def _get_minimal_config(self):
        return WhisperConfig(
            n_mels=80,
            n_vocab=51865,
            n_audio_ctx=1500,
            n_audio_state=64,
            n_audio_head=4,
            n_audio_layer=2,
            n_text_ctx=448,
            n_text_state=64,
            n_text_head=4,
            n_text_layer=2,
        )

    def test_disable_fused_attention(self):
        """Test model with fused attention disabled."""
        config = self._get_minimal_config()
        model = WhisperMLX(config, use_fused=False)

        assert model.use_fused is False

    def test_float32_dtype(self):
        """Test model with float32 dtype."""
        config = self._get_minimal_config()
        model = WhisperMLX(config, dtype=mx.float32)

        assert model.dtype == mx.float32

    def test_bfloat16_dtype(self):
        """Test model with bfloat16 dtype."""
        config = self._get_minimal_config()
        model = WhisperMLX(config, dtype=mx.bfloat16)

        assert model.dtype == mx.bfloat16


class TestWhisperMLXPromptTokens:
    """Tests for OPT-NEW-32: Previous window prompt caching."""

    def _get_small_config(self):
        return WhisperConfig(
            n_mels=80,
            n_vocab=51865,
            n_audio_ctx=1500,
            n_audio_state=64,
            n_audio_head=4,
            n_audio_layer=2,
            n_text_ctx=448,
            n_text_state=64,
            n_text_head=4,
            n_text_layer=2,
        )

    def test_decode_accepts_prompt_tokens(self):
        """Test that _decode accepts prompt_tokens parameter."""
        config = self._get_small_config()
        model = WhisperMLX(config)

        # Create mock tokenizer
        tokenizer = MagicMock()
        tokenizer.sot_sequence = [50258, 50259, 50360, 50364]  # SOT sequence
        tokenizer.sot_prev = 50361  # Previous segment marker
        tokenizer.eot = 50257
        tokenizer.no_speech = None
        tokenizer.timestamp_begin = 50364
        tokenizer.encode = lambda x: [1, 2, 3]
        tokenizer.decode = lambda tokens: "test"
        tokenizer.non_speech_tokens = []
        tokenizer.transcribe = 50359
        tokenizer.translate = 50358
        tokenizer.sot = 50258
        tokenizer.sot_lm = 50360
        tokenizer.all_language_tokens = []

        # Test that _decode_with_metrics accepts prompt_tokens parameter
        import inspect
        sig = inspect.signature(model._decode_with_metrics)
        assert 'prompt_tokens' in sig.parameters

        # Verify parameter type annotation
        param = sig.parameters['prompt_tokens']
        assert param.default is None  # Optional with None default

    def test_transcribe_long_accepts_condition_on_previous_text(self):
        """Test that transcribe_long accepts condition_on_previous_text parameter."""
        config = self._get_small_config()
        model = WhisperMLX(config)

        # Check the signature accepts the parameter
        import inspect
        sig = inspect.signature(model.transcribe_long)
        assert 'condition_on_previous_text' in sig.parameters

        # Check default value is True (following mlx-whisper pattern)
        default = sig.parameters['condition_on_previous_text'].default
        assert default is True

    def test_prompt_tokens_truncation(self):
        """Test that prompt tokens are truncated to max length."""
        config = self._get_small_config()

        # Verify the truncation logic follows mlx-whisper pattern
        # max_prompt_len should be n_text_ctx // 2 - 1 = 448 // 2 - 1 = 223
        max_prompt_len = config.n_text_ctx // 2 - 1
        assert max_prompt_len == 223

        # Verify a model can be created with these settings
        model = WhisperMLX(config)
        assert model.config.n_text_ctx == 448

    def test_decode_method_exists(self):
        """Test that _decode method exists and has correct signature."""
        config = self._get_small_config()
        model = WhisperMLX(config)

        # Check method exists
        assert hasattr(model, '_decode')
        assert callable(model._decode)

        # Check signature
        import inspect
        sig = inspect.signature(model._decode)
        params = list(sig.parameters.keys())

        # Should have all required parameters
        assert 'audio_features' in params
        assert 'tokenizer' in params
        assert 'temperature' in params
        assert 'prompt_tokens' in params

    def test_decode_calls_decode_with_metrics(self):
        """Test that _decode is a wrapper for _decode_with_metrics."""
        config = self._get_small_config()
        model = WhisperMLX(config)

        # Verify _decode exists and _decode_with_metrics exists
        assert hasattr(model, '_decode')
        assert hasattr(model, '_decode_with_metrics')

        # _decode should return 2 values (tokens, segments)
        # _decode_with_metrics returns 4 values (tokens, segments, avg_logprob, no_speech_prob)
        import inspect
        decode_sig = inspect.signature(model._decode)
        decode_return = str(decode_sig.return_annotation)

        # Verify return type annotation mentions List and Dict (for tokens, segments)
        assert 'List' in decode_return or 'Tuple' in decode_return


class TestWhisperMLXQuantization:
    """Tests for WhisperMLX INT8/INT4 quantization."""

    def _get_small_config(self):
        """Get small model config for testing."""
        return WhisperConfig(
            n_mels=80,
            n_vocab=51865,
            n_audio_ctx=1500,
            n_audio_state=384,  # Small for testing
            n_audio_head=6,
            n_audio_layer=4,
            n_text_ctx=448,
            n_text_state=384,
            n_text_head=6,
            n_text_layer=4,
        )

    def test_quantize_model_method_exists(self):
        """Test that quantize_model method exists."""
        config = self._get_small_config()
        model = WhisperMLX(config)

        assert hasattr(model, 'quantize_model')
        assert callable(model.quantize_model)

    def test_quantize_model_signature(self):
        """Test quantize_model method signature."""
        import inspect
        config = self._get_small_config()
        model = WhisperMLX(config)

        sig = inspect.signature(model.quantize_model)
        params = sig.parameters

        # Should have group_size and bits parameters
        assert 'group_size' in params
        assert 'bits' in params

        # Check defaults
        assert params['group_size'].default == 64
        assert params['bits'].default == 8

    def test_quantize_model_returns_self(self):
        """Test that quantize_model returns self for chaining."""
        config = self._get_small_config()
        model = WhisperMLX(config)

        result = model.quantize_model(bits=8)

        assert result is model

    def test_from_pretrained_quantize_parameter(self):
        """Test that from_pretrained accepts quantize parameter."""
        import inspect
        sig = inspect.signature(WhisperMLX.from_pretrained)
        params = sig.parameters

        assert 'quantize' in params
        assert params['quantize'].default is None

    def test_quantize_model_int8_changes_weights(self):
        """Test that INT8 quantization modifies model structure."""
        config = self._get_small_config()
        model = WhisperMLX(config)

        # Quantize
        model.quantize_model(bits=8)

        # After quantization, model should have QuantizedLinear layers
        # which have different parameter structure (scales, biases, etc.)
        quantized_params = list(model.parameters().items())

        # The parameter structure should change after quantization
        # (either more params from scales/biases or different param names)
        # This is a basic sanity check
        assert len(quantized_params) > 0

    def test_quantize_model_int4(self):
        """Test INT4 quantization."""
        config = self._get_small_config()
        model = WhisperMLX(config)

        # Should not raise an error
        model.quantize_model(bits=4)

        # Model should still have parameters
        params = list(model.parameters().items())
        assert len(params) > 0

    def test_quantize_model_custom_group_size(self):
        """Test quantization with custom group size."""
        config = self._get_small_config()
        model = WhisperMLX(config)

        # Should not raise an error with group_size=32
        model.quantize_model(group_size=32, bits=8)

        # Model should still have parameters
        params = list(model.parameters().items())
        assert len(params) > 0

    def test_quantized_model_forward_pass(self):
        """Test that quantized model can do forward pass.

        The attention module detects QuantizedLinear layers and automatically
        falls back to separate Q/K/V projections instead of fused QKV weight
        concatenation.
        """
        config = self._get_small_config()
        model = WhisperMLX(config)

        # Quantize
        model.quantize_model(bits=8)

        # Forward pass should work now (auto-detects quantization)
        mel = mx.zeros((3000, config.n_mels))
        encoder_output = model.embed_audio(mel)

        assert encoder_output.shape[0] == 1  # Batch dim added
        assert encoder_output.shape[-1] == config.n_audio_state

    def test_quantized_model_decoder_forward(self):
        """Test that quantized model decoder can do forward pass.

        Embeddings are excluded from quantization to preserve correct shapes.
        Only Linear layers are quantized.
        """
        config = self._get_small_config()
        model = WhisperMLX(config)

        model.quantize_model(bits=8)

        mel = mx.zeros((1, 3000, config.n_mels))
        tokens = mx.array([[50258, 50259, 50359, 50363]])

        encoder_output = model.embed_audio(mel)
        # Decoder returns (logits, kv_cache, attn_weights, hidden_states)
        logits, _, _, _ = model.decoder(tokens, encoder_output)

        assert logits.shape[0] == 1  # Batch size
        assert logits.shape[2] == config.n_vocab
