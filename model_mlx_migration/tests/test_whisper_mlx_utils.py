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
Tests for WhisperMLX utility functions.

Tests weight loading, HuggingFace weight conversion, model download, and timestamp formatting.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import mlx.core as mx
import numpy as np
import pytest

from tools.whisper_mlx.utils import (
    convert_hf_weights,
    download_model,
    format_timestamp,
    load_weights_from_safetensors,
)


class TestFormatTimestamp:
    """Test format_timestamp utility function."""

    def test_zero_seconds(self):
        """Zero seconds formats correctly."""
        assert format_timestamp(0.0) == "00:00,000"

    def test_zero_seconds_with_hours(self):
        """Zero seconds with hours flag."""
        assert format_timestamp(0.0, always_include_hours=True) == "00:00:00,000"

    def test_milliseconds_only(self):
        """Sub-second timestamps."""
        assert format_timestamp(0.001) == "00:00,001"
        assert format_timestamp(0.5) == "00:00,500"
        assert format_timestamp(0.999) == "00:00,999"

    def test_seconds_only(self):
        """Seconds without minutes/hours."""
        assert format_timestamp(1.0) == "00:01,000"
        assert format_timestamp(30.0) == "00:30,000"
        assert format_timestamp(59.999) == "00:59,999"

    def test_minutes_and_seconds(self):
        """Minutes and seconds."""
        assert format_timestamp(60.0) == "01:00,000"
        assert format_timestamp(90.5) == "01:30,500"
        assert format_timestamp(599.0) == "09:59,000"

    def test_hours_auto_included(self):
        """Hours automatically included when >= 1 hour."""
        assert format_timestamp(3600.0) == "01:00:00,000"
        assert format_timestamp(3661.5) == "01:01:01,500"
        assert format_timestamp(7200.0) == "02:00:00,000"

    def test_always_include_hours(self):
        """always_include_hours flag includes hours even when zero."""
        assert format_timestamp(30.0, always_include_hours=True) == "00:00:30,000"
        assert format_timestamp(90.5, always_include_hours=True) == "00:01:30,500"

    def test_large_timestamp(self):
        """Large timestamps (many hours)."""
        # 10 hours + 30 mins + 45 secs + 123ms
        timestamp = 10 * 3600 + 30 * 60 + 45 + 0.123
        assert format_timestamp(timestamp) == "10:30:45,123"

    def test_rounding(self):
        """Milliseconds are rounded correctly."""
        assert format_timestamp(1.9999) == "00:02,000"
        assert format_timestamp(1.0001) == "00:01,000"
        # Python uses banker's rounding (round half to even)
        assert format_timestamp(1.0005) == "00:01,000"  # 0.5ms rounds to even
        assert format_timestamp(1.0015) == "00:01,002"  # 1.5ms rounds to 2

    def test_srt_format_compatible(self):
        """Output is compatible with SRT format (comma separator)."""
        result = format_timestamp(125.456)
        assert "," in result  # SRT uses comma for milliseconds
        assert result == "02:05,456"

    def test_typical_whisper_timestamps(self):
        """Typical Whisper-generated timestamps."""
        # First word at 0.02s
        assert format_timestamp(0.02) == "00:00,020"
        # End of typical chunk at 30s
        assert format_timestamp(30.0) == "00:30,000"
        # Long transcription under 1 hour (no hours in output)
        assert format_timestamp(3599.99) == "59:59,990"
        # Long transcription over 1 hour
        assert format_timestamp(3600.5) == "01:00:00,500"


class TestConvertHfWeights:
    """Test HuggingFace weight name conversion."""

    def test_empty_weights(self):
        """Empty input returns empty output."""
        result = convert_hf_weights({})
        assert result == {}

    def test_encoder_conv_layers(self):
        """Encoder conv layer weight name conversion."""
        hf_weights = {
            "model.encoder.conv1.weight": mx.zeros((3, 3)),
            "model.encoder.conv1.bias": mx.zeros((3,)),
            "model.encoder.conv2.weight": mx.zeros((3, 3)),
            "model.encoder.conv2.bias": mx.zeros((3,)),
        }
        result = convert_hf_weights(hf_weights)

        assert "encoder.conv1.weight" in result
        assert "encoder.conv1.bias" in result
        assert "encoder.conv2.weight" in result
        assert "encoder.conv2.bias" in result

    def test_encoder_positional_embedding(self):
        """Encoder positional embedding conversion."""
        hf_weights = {
            "model.encoder.embed_positions.weight": mx.zeros((1500, 1280)),
        }
        result = convert_hf_weights(hf_weights)
        assert "encoder._positional_embedding" in result

    def test_encoder_layer_norm(self):
        """Encoder layer norm conversion."""
        hf_weights = {
            "model.encoder.layer_norm.weight": mx.zeros((1280,)),
            "model.encoder.layer_norm.bias": mx.zeros((1280,)),
        }
        result = convert_hf_weights(hf_weights)
        assert "encoder.ln_post.weight" in result
        assert "encoder.ln_post.bias" in result

    def test_decoder_embeddings(self):
        """Decoder embedding conversion."""
        hf_weights = {
            "model.decoder.embed_tokens.weight": mx.zeros((51866, 1280)),
            "model.decoder.embed_positions.weight": mx.zeros((448, 1280)),
        }
        result = convert_hf_weights(hf_weights)
        assert "decoder.token_embedding.weight" in result
        assert "decoder.positional_embedding" in result

    def test_decoder_layer_norm(self):
        """Decoder final layer norm conversion."""
        hf_weights = {
            "model.decoder.layer_norm.weight": mx.zeros((1280,)),
            "model.decoder.layer_norm.bias": mx.zeros((1280,)),
        }
        result = convert_hf_weights(hf_weights)
        assert "decoder.ln.weight" in result
        assert "decoder.ln.bias" in result

    def test_encoder_layer_self_attention(self):
        """Encoder layer self-attention weight conversion."""
        hf_weights = {
            "model.encoder.layers.0.self_attn.q_proj.weight": mx.zeros((1280, 1280)),
            "model.encoder.layers.0.self_attn.q_proj.bias": mx.zeros((1280,)),
            "model.encoder.layers.0.self_attn.k_proj.weight": mx.zeros((1280, 1280)),
            "model.encoder.layers.0.self_attn.v_proj.weight": mx.zeros((1280, 1280)),
            "model.encoder.layers.0.self_attn.out_proj.weight": mx.zeros((1280, 1280)),
        }
        result = convert_hf_weights(hf_weights)

        assert "encoder.blocks.0.attn.query.weight" in result
        assert "encoder.blocks.0.attn.query.bias" in result
        assert "encoder.blocks.0.attn.key.weight" in result
        assert "encoder.blocks.0.attn.value.weight" in result
        assert "encoder.blocks.0.attn.out.weight" in result

    def test_encoder_layer_mlp(self):
        """Encoder layer MLP weight conversion."""
        hf_weights = {
            "model.encoder.layers.0.fc1.weight": mx.zeros((5120, 1280)),
            "model.encoder.layers.0.fc1.bias": mx.zeros((5120,)),
            "model.encoder.layers.0.fc2.weight": mx.zeros((1280, 5120)),
            "model.encoder.layers.0.fc2.bias": mx.zeros((1280,)),
        }
        result = convert_hf_weights(hf_weights)

        assert "encoder.blocks.0.mlp1.weight" in result
        assert "encoder.blocks.0.mlp1.bias" in result
        assert "encoder.blocks.0.mlp2.weight" in result
        assert "encoder.blocks.0.mlp2.bias" in result

    def test_encoder_layer_norms(self):
        """Encoder layer norm weight conversion."""
        hf_weights = {
            "model.encoder.layers.0.self_attn_layer_norm.weight": mx.zeros((1280,)),
            "model.encoder.layers.0.self_attn_layer_norm.bias": mx.zeros((1280,)),
            "model.encoder.layers.0.final_layer_norm.weight": mx.zeros((1280,)),
            "model.encoder.layers.0.final_layer_norm.bias": mx.zeros((1280,)),
        }
        result = convert_hf_weights(hf_weights)

        assert "encoder.blocks.0.attn_ln.weight" in result
        assert "encoder.blocks.0.attn_ln.bias" in result
        assert "encoder.blocks.0.mlp_ln.weight" in result
        assert "encoder.blocks.0.mlp_ln.bias" in result

    def test_decoder_layer_self_attention(self):
        """Decoder layer self-attention weight conversion."""
        hf_weights = {
            "model.decoder.layers.0.self_attn.q_proj.weight": mx.zeros((1280, 1280)),
            "model.decoder.layers.0.self_attn.k_proj.weight": mx.zeros((1280, 1280)),
            "model.decoder.layers.0.self_attn.v_proj.weight": mx.zeros((1280, 1280)),
            "model.decoder.layers.0.self_attn.out_proj.weight": mx.zeros((1280, 1280)),
        }
        result = convert_hf_weights(hf_weights)

        assert "decoder.blocks.0.attn.query.weight" in result
        assert "decoder.blocks.0.attn.key.weight" in result
        assert "decoder.blocks.0.attn.value.weight" in result
        assert "decoder.blocks.0.attn.out.weight" in result

    def test_decoder_layer_cross_attention(self):
        """Decoder layer cross-attention weight conversion."""
        hf_weights = {
            "model.decoder.layers.0.encoder_attn.q_proj.weight": mx.zeros((1280, 1280)),
            "model.decoder.layers.0.encoder_attn.k_proj.weight": mx.zeros((1280, 1280)),
            "model.decoder.layers.0.encoder_attn.v_proj.weight": mx.zeros((1280, 1280)),
            "model.decoder.layers.0.encoder_attn.out_proj.weight": mx.zeros((1280, 1280)),
        }
        result = convert_hf_weights(hf_weights)

        assert "decoder.blocks.0.cross_attn.query.weight" in result
        assert "decoder.blocks.0.cross_attn.key.weight" in result
        assert "decoder.blocks.0.cross_attn.value.weight" in result
        assert "decoder.blocks.0.cross_attn.out.weight" in result

    def test_decoder_layer_norms(self):
        """Decoder layer norm weight conversion."""
        hf_weights = {
            "model.decoder.layers.0.self_attn_layer_norm.weight": mx.zeros((1280,)),
            "model.decoder.layers.0.encoder_attn_layer_norm.weight": mx.zeros((1280,)),
            "model.decoder.layers.0.final_layer_norm.weight": mx.zeros((1280,)),
        }
        result = convert_hf_weights(hf_weights)

        assert "decoder.blocks.0.attn_ln.weight" in result
        assert "decoder.blocks.0.cross_attn_ln.weight" in result
        assert "decoder.blocks.0.mlp_ln.weight" in result

    def test_multiple_layers(self):
        """Conversion works for multiple layers."""
        hf_weights = {
            "model.encoder.layers.0.fc1.weight": mx.zeros((5120, 1280)),
            "model.encoder.layers.5.fc1.weight": mx.zeros((5120, 1280)),
            "model.encoder.layers.31.fc1.weight": mx.zeros((5120, 1280)),
            "model.decoder.layers.0.fc1.weight": mx.zeros((5120, 1280)),
            "model.decoder.layers.3.fc1.weight": mx.zeros((5120, 1280)),
        }
        result = convert_hf_weights(hf_weights)

        assert "encoder.blocks.0.mlp1.weight" in result
        assert "encoder.blocks.5.mlp1.weight" in result
        assert "encoder.blocks.31.mlp1.weight" in result
        assert "decoder.blocks.0.mlp1.weight" in result
        assert "decoder.blocks.3.mlp1.weight" in result

    def test_weight_values_preserved(self):
        """Weight values are preserved during conversion."""
        test_weight = mx.array([[1.0, 2.0], [3.0, 4.0]])
        hf_weights = {
            "model.encoder.conv1.weight": test_weight,
        }
        result = convert_hf_weights(hf_weights)

        assert mx.array_equal(result["encoder.conv1.weight"], test_weight)

    def test_unmapped_weights_ignored(self):
        """Weights that don't match any pattern are ignored."""
        hf_weights = {
            "random.unknown.weight": mx.zeros((3, 3)),
            "model.encoder.conv1.weight": mx.zeros((3, 3)),
        }
        result = convert_hf_weights(hf_weights)

        assert "encoder.conv1.weight" in result
        assert "random.unknown.weight" not in result


class TestDownloadModel:
    """Test download_model function."""

    @patch("huggingface_hub.snapshot_download")
    def test_download_standard_model(self, mock_download):
        """Standard model names get mlx-community prefix and -mlx suffix."""
        mock_download.return_value = "/path/to/model"

        result = download_model("large-v3")

        mock_download.assert_called_once()
        call_args = mock_download.call_args
        assert call_args[0][0] == "mlx-community/whisper-large-v3-mlx"
        assert result == Path("/path/to/model")

    @patch("huggingface_hub.snapshot_download")
    def test_download_turbo_model_no_mlx_suffix(self, mock_download):
        """Turbo models don't have -mlx suffix."""
        mock_download.return_value = "/path/to/model"

        download_model("large-v3-turbo")

        call_args = mock_download.call_args
        assert call_args[0][0] == "mlx-community/whisper-large-v3-turbo"
        # Note: no -mlx suffix for turbo models

    @patch("huggingface_hub.snapshot_download")
    def test_download_full_repo_name(self, mock_download):
        """Full repo names are used as-is."""
        mock_download.return_value = "/path/to/model"

        download_model("mlx-community/whisper-large-v3-mlx")

        call_args = mock_download.call_args
        assert call_args[0][0] == "mlx-community/whisper-large-v3-mlx"

    @patch("huggingface_hub.snapshot_download")
    def test_download_custom_cache_dir(self, mock_download):
        """Custom cache directory is passed through."""
        mock_download.return_value = "/path/to/model"

        download_model("large-v3", cache_dir="/custom/cache")

        call_args = mock_download.call_args
        assert call_args[1]["cache_dir"] == "/custom/cache"

    @patch("huggingface_hub.snapshot_download")
    def test_download_filters_files(self, mock_download):
        """Download filters to only safetensors and json files."""
        mock_download.return_value = "/path/to/model"

        download_model("large-v3")

        call_args = mock_download.call_args
        allowed_patterns = call_args[1]["allow_patterns"]
        assert "*.safetensors" in allowed_patterns
        assert "*.json" in allowed_patterns
        assert "config.json" in allowed_patterns

    @patch("huggingface_hub.snapshot_download")
    def test_download_returns_path_object(self, mock_download):
        """download_model returns a Path object."""
        mock_download.return_value = "/some/cache/path/to/model"

        result = download_model("base")

        assert isinstance(result, Path)
        assert str(result) == "/some/cache/path/to/model"


class TestLoadWeightsFromSafetensors:
    """Test load_weights_from_safetensors function."""

    @patch("safetensors.numpy.load_file")
    def test_load_weights_returns_counts(self, mock_load_file):
        """Function returns loaded and skipped counts."""
        mock_load_file.return_value = {
            "encoder.weight": np.zeros((3, 3)),
            "decoder.weight": np.zeros((3, 3)),
        }

        # Create mock model without spec restriction
        model = MagicMock()
        model.named_parameters.return_value = [
            ("encoder.weight", mx.zeros((3, 3))),
            ("decoder.weight", mx.zeros((3, 3))),
        ]

        loaded, skipped = load_weights_from_safetensors(
            model, "/fake/path.safetensors", strict=False,
        )

        # Both weights should be loaded
        assert loaded == 2
        assert skipped == 0

    @patch("safetensors.numpy.load_file")
    def test_load_weights_skips_unknown_in_non_strict(self, mock_load_file):
        """Non-strict mode skips unknown weights."""
        mock_load_file.return_value = {
            "known.weight": np.zeros((3, 3)),
            "unknown.weight": np.zeros((3, 3)),
        }

        model = MagicMock()
        model.named_parameters.return_value = [
            ("known.weight", mx.zeros((3, 3))),
        ]

        loaded, skipped = load_weights_from_safetensors(
            model, "/fake/path.safetensors", strict=False,
        )

        assert loaded == 1
        assert skipped == 1

    @patch("safetensors.numpy.load_file")
    def test_load_weights_strict_raises_on_unexpected(self, mock_load_file):
        """Strict mode raises on unexpected weights."""
        mock_load_file.return_value = {
            "unexpected.weight": np.zeros((3, 3)),
        }

        model = MagicMock()
        model.named_parameters.return_value = []

        with pytest.raises(KeyError) as exc_info:
            load_weights_from_safetensors(
                model, "/fake/path.safetensors", strict=True,
            )
        assert "unexpected" in str(exc_info.value).lower()


class TestUtilsIntegration:
    """Integration tests combining multiple utility functions."""

    def test_format_timestamp_srt_range(self):
        """Generate valid SRT timestamp ranges."""
        start = format_timestamp(10.5, always_include_hours=True)
        end = format_timestamp(15.25, always_include_hours=True)

        # Valid SRT format: HH:MM:SS,mmm --> HH:MM:SS,mmm
        srt_line = f"{start} --> {end}"
        assert srt_line == "00:00:10,500 --> 00:00:15,250"

    def test_full_hf_weight_conversion(self):
        """Full HuggingFace model weight conversion."""
        # Simulate a small HuggingFace model state dict
        hf_weights = {
            # Encoder
            "model.encoder.conv1.weight": mx.zeros((512, 128, 3)),
            "model.encoder.conv2.weight": mx.zeros((512, 512, 3)),
            "model.encoder.embed_positions.weight": mx.zeros((1500, 512)),
            "model.encoder.layers.0.self_attn.q_proj.weight": mx.zeros((512, 512)),
            "model.encoder.layers.0.self_attn.k_proj.weight": mx.zeros((512, 512)),
            "model.encoder.layers.0.self_attn.v_proj.weight": mx.zeros((512, 512)),
            "model.encoder.layers.0.self_attn.out_proj.weight": mx.zeros((512, 512)),
            "model.encoder.layers.0.fc1.weight": mx.zeros((2048, 512)),
            "model.encoder.layers.0.fc2.weight": mx.zeros((512, 2048)),
            "model.encoder.layer_norm.weight": mx.zeros((512,)),
            # Decoder
            "model.decoder.embed_tokens.weight": mx.zeros((51864, 512)),
            "model.decoder.embed_positions.weight": mx.zeros((448, 512)),
            "model.decoder.layers.0.self_attn.q_proj.weight": mx.zeros((512, 512)),
            "model.decoder.layers.0.encoder_attn.q_proj.weight": mx.zeros((512, 512)),
            "model.decoder.layers.0.fc1.weight": mx.zeros((2048, 512)),
            "model.decoder.layers.0.fc2.weight": mx.zeros((512, 2048)),
            "model.decoder.layer_norm.weight": mx.zeros((512,)),
        }

        result = convert_hf_weights(hf_weights)

        # Verify key conversions
        assert len(result) >= 15
        assert "encoder.conv1.weight" in result
        assert "encoder._positional_embedding" in result
        assert "encoder.blocks.0.attn.query.weight" in result
        assert "encoder.blocks.0.mlp1.weight" in result
        assert "encoder.ln_post.weight" in result
        assert "decoder.token_embedding.weight" in result
        assert "decoder.positional_embedding" in result
        assert "decoder.blocks.0.attn.query.weight" in result
        assert "decoder.blocks.0.cross_attn.query.weight" in result
        assert "decoder.ln.weight" in result
