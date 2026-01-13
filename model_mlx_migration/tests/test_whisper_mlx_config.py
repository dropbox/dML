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
Tests for WhisperMLX configuration module.

Tests the WhisperConfig dataclass, predefined configurations for all model sizes,
and the get_config() helper function.
"""

import pytest

from tools.whisper_mlx.config import (
    WHISPER_CONFIGS,
    WhisperConfig,
    get_config,
)


class TestWhisperConfig:
    """Test WhisperConfig dataclass and its properties."""

    def test_default_values(self):
        """Default values should match large-v3 model."""
        config = WhisperConfig()
        assert config.n_mels == 128
        assert config.n_audio_ctx == 1500
        assert config.n_audio_state == 1280
        assert config.n_audio_head == 20
        assert config.n_audio_layer == 32
        assert config.n_vocab == 51866
        assert config.n_text_ctx == 448
        assert config.n_text_state == 1280
        assert config.n_text_head == 20
        assert config.n_text_layer == 32
        assert config.sample_rate == 16000
        assert config.n_fft == 400
        assert config.hop_length == 160
        assert config.chunk_length == 30
        assert config.name == "large-v3"

    def test_custom_values(self):
        """Custom values should override defaults."""
        config = WhisperConfig(
            n_mels=80,
            n_audio_state=384,
            n_audio_head=6,
            n_audio_layer=4,
            name="tiny",
        )
        assert config.n_mels == 80
        assert config.n_audio_state == 384
        assert config.n_audio_head == 6
        assert config.n_audio_layer == 4
        assert config.name == "tiny"
        # Unchanged defaults
        assert config.n_audio_ctx == 1500
        assert config.sample_rate == 16000

    def test_frames_per_second_property(self):
        """frames_per_second should be sample_rate / hop_length."""
        config = WhisperConfig()
        # 16000 / 160 = 100 fps
        assert config.frames_per_second == 100.0

        # Custom sample rate and hop length
        config = WhisperConfig(sample_rate=48000, hop_length=480)
        assert config.frames_per_second == 100.0

        config = WhisperConfig(sample_rate=16000, hop_length=80)
        assert config.frames_per_second == 200.0

    def test_time_precision_property(self):
        """time_precision should be chunk_length / n_audio_ctx."""
        config = WhisperConfig()
        # 30 / 1500 = 0.02s
        assert config.time_precision == 0.02

        config = WhisperConfig(chunk_length=15, n_audio_ctx=750)
        assert config.time_precision == 0.02

        config = WhisperConfig(chunk_length=60, n_audio_ctx=3000)
        assert config.time_precision == 0.02

    def test_compute_precision_standard(self):
        """compute_precision for standard 30s audio."""
        config = WhisperConfig()
        precision = config.compute_precision(audio_duration=30.0, encoder_positions=1500)
        assert precision == 0.02

    def test_compute_precision_short_audio(self):
        """compute_precision for shorter audio clips."""
        config = WhisperConfig()
        # 10s audio with 500 encoder positions
        precision = config.compute_precision(audio_duration=10.0, encoder_positions=500)
        assert precision == 0.02

        # 5s audio with 250 encoder positions
        precision = config.compute_precision(audio_duration=5.0, encoder_positions=250)
        assert precision == 0.02

    def test_compute_precision_long_audio(self):
        """compute_precision for longer audio."""
        config = WhisperConfig()
        precision = config.compute_precision(audio_duration=60.0, encoder_positions=3000)
        assert precision == 0.02

    def test_compute_precision_zero_positions(self):
        """compute_precision returns default when encoder_positions is 0."""
        config = WhisperConfig()
        precision = config.compute_precision(audio_duration=10.0, encoder_positions=0)
        assert precision == config.time_precision  # Falls back to default

    def test_compute_precision_negative_positions(self):
        """compute_precision returns default for negative encoder_positions."""
        config = WhisperConfig()
        precision = config.compute_precision(audio_duration=10.0, encoder_positions=-1)
        assert precision == config.time_precision

    def test_compute_precision_non_standard_ratio(self):
        """compute_precision with non-standard audio/positions ratio."""
        config = WhisperConfig()
        # Variable-length input: 15s audio with 750 encoder positions
        precision = config.compute_precision(audio_duration=15.0, encoder_positions=750)
        assert precision == 0.02

        # Different ratio: 20s audio with 1000 positions
        precision = config.compute_precision(audio_duration=20.0, encoder_positions=1000)
        assert precision == 0.02


class TestWhisperConfigs:
    """Test predefined WHISPER_CONFIGS dictionary."""

    def test_all_model_sizes_present(self):
        """All expected model sizes should be in WHISPER_CONFIGS."""
        expected_models = [
            "tiny", "tiny.en",
            "base", "base.en",
            "small", "small.en",
            "medium", "medium.en",
            "large", "large-v2", "large-v3", "large-v3-turbo",
            "distil-large-v3", "distil-large-v2", "distil-medium.en",
        ]
        for model in expected_models:
            assert model in WHISPER_CONFIGS, f"Missing model: {model}"

    def test_v1v2_models_have_80_mels(self):
        """Whisper v1/v2 models should have 80 mel bands."""
        v1v2_models = [
            "tiny", "tiny.en",
            "base", "base.en",
            "small", "small.en",
            "medium", "medium.en",
            "large", "large-v2",
            "distil-large-v2", "distil-medium.en",
        ]
        for model in v1v2_models:
            assert WHISPER_CONFIGS[model].n_mels == 80, f"{model} should have 80 mels"

    def test_v3_models_have_128_mels(self):
        """Whisper v3 models should have 128 mel bands."""
        v3_models = ["large-v3", "large-v3-turbo", "distil-large-v3"]
        for model in v3_models:
            assert WHISPER_CONFIGS[model].n_mels == 128, f"{model} should have 128 mels"

    def test_tiny_dimensions(self):
        """Tiny model has smallest dimensions."""
        config = WHISPER_CONFIGS["tiny"]
        assert config.n_audio_state == 384
        assert config.n_audio_head == 6
        assert config.n_audio_layer == 4
        assert config.n_text_state == 384
        assert config.n_text_head == 6
        assert config.n_text_layer == 4

    def test_base_dimensions(self):
        """Base model dimensions."""
        config = WHISPER_CONFIGS["base"]
        assert config.n_audio_state == 512
        assert config.n_audio_head == 8
        assert config.n_audio_layer == 6
        assert config.n_text_state == 512
        assert config.n_text_head == 8
        assert config.n_text_layer == 6

    def test_small_dimensions(self):
        """Small model dimensions."""
        config = WHISPER_CONFIGS["small"]
        assert config.n_audio_state == 768
        assert config.n_audio_head == 12
        assert config.n_audio_layer == 12
        assert config.n_text_state == 768
        assert config.n_text_head == 12
        assert config.n_text_layer == 12

    def test_medium_dimensions(self):
        """Medium model dimensions."""
        config = WHISPER_CONFIGS["medium"]
        assert config.n_audio_state == 1024
        assert config.n_audio_head == 16
        assert config.n_audio_layer == 24
        assert config.n_text_state == 1024
        assert config.n_text_head == 16
        assert config.n_text_layer == 24

    def test_large_dimensions(self):
        """Large models have largest dimensions."""
        for model_name in ["large", "large-v2", "large-v3"]:
            config = WHISPER_CONFIGS[model_name]
            assert config.n_audio_state == 1280
            assert config.n_audio_head == 20
            assert config.n_audio_layer == 32
            assert config.n_text_state == 1280
            assert config.n_text_head == 20
            assert config.n_text_layer == 32

    def test_turbo_has_4_decoder_layers(self):
        """Turbo model has only 4 decoder layers."""
        config = WHISPER_CONFIGS["large-v3-turbo"]
        assert config.n_text_layer == 4
        assert config.n_audio_layer == 32  # Full encoder

    def test_distil_models_have_2_decoder_layers(self):
        """Distilled models have 2 decoder layers."""
        distil_models = ["distil-large-v3", "distil-large-v2", "distil-medium.en"]
        for model in distil_models:
            config = WHISPER_CONFIGS[model]
            assert config.n_text_layer == 2, f"{model} should have 2 decoder layers"

    def test_english_models_have_51864_vocab(self):
        """English-only models have 51864 vocab size."""
        english_models = [
            "tiny.en", "base.en", "small.en", "medium.en", "distil-medium.en",
        ]
        for model in english_models:
            config = WHISPER_CONFIGS[model]
            assert config.n_vocab == 51864, f"{model} should have 51864 vocab"

    def test_multilingual_v1v2_models_have_51865_vocab(self):
        """Multilingual v1/v2 models have 51865 vocab size."""
        multilingual_v1v2 = ["large", "large-v2", "distil-large-v2"]
        for model in multilingual_v1v2:
            config = WHISPER_CONFIGS[model]
            assert config.n_vocab == 51865, f"{model} should have 51865 vocab"

    def test_multilingual_v3_models_have_51866_vocab(self):
        """Multilingual v3 models have 51866 vocab size."""
        multilingual_v3 = ["large-v3", "large-v3-turbo", "distil-large-v3"]
        for model in multilingual_v3:
            config = WHISPER_CONFIGS[model]
            assert config.n_vocab == 51866, f"{model} should have 51866 vocab"

    def test_model_names_match_config_names(self):
        """Config name property should match dictionary key."""
        for model_name, config in WHISPER_CONFIGS.items():
            assert config.name == model_name

    def test_head_dimension_consistency(self):
        """hidden_state should be divisible by num_heads."""
        for model_name, config in WHISPER_CONFIGS.items():
            assert config.n_audio_state % config.n_audio_head == 0, \
                f"{model_name} audio state not divisible by heads"
            assert config.n_text_state % config.n_text_head == 0, \
                f"{model_name} text state not divisible by heads"


class TestGetConfig:
    """Test get_config() helper function."""

    def test_get_standard_models(self):
        """get_config returns correct config for standard model names."""
        config = get_config("large-v3")
        assert config.name == "large-v3"
        assert config.n_mels == 128

        config = get_config("tiny")
        assert config.name == "tiny"
        assert config.n_audio_state == 384

    def test_get_english_models(self):
        """get_config works with English-only model names."""
        config = get_config("tiny.en")
        assert config.name == "tiny.en"
        assert config.n_vocab == 51864

    def test_case_insensitive(self):
        """get_config is case-insensitive."""
        config_lower = get_config("large-v3")
        config_upper = get_config("LARGE-V3")
        config_mixed = get_config("Large-V3")

        assert config_lower.name == config_upper.name == config_mixed.name

    def test_whisper_prefix_stripped(self):
        """get_config strips 'whisper-' prefix."""
        config = get_config("whisper-large-v3")
        assert config.name == "large-v3"

        config = get_config("WHISPER-tiny")
        assert config.name == "tiny"

    def test_underscore_to_dash_conversion(self):
        """get_config converts underscores to dashes."""
        config = get_config("large_v3")
        assert config.name == "large-v3"

        config = get_config("large_v3_turbo")
        assert config.name == "large-v3-turbo"

    def test_unknown_model_raises(self):
        """get_config raises ValueError for unknown models."""
        with pytest.raises(ValueError) as exc_info:
            get_config("unknown-model")
        assert "Unknown model" in str(exc_info.value)
        assert "unknown-model" in str(exc_info.value)

    def test_unknown_model_lists_available(self):
        """get_config error message lists available models."""
        with pytest.raises(ValueError) as exc_info:
            get_config("not-a-model")
        error_msg = str(exc_info.value)
        # Should list at least some available models
        assert "large-v3" in error_msg
        assert "tiny" in error_msg

    def test_get_distil_models(self):
        """get_config works with distil model names."""
        config = get_config("distil-large-v3")
        assert config.name == "distil-large-v3"
        assert config.n_text_layer == 2

        config = get_config("distil-medium.en")
        assert config.name == "distil-medium.en"
        assert config.n_vocab == 51864


class TestConfigIntegrity:
    """Integration tests for configuration consistency."""

    def test_encoder_state_matches_text_state(self):
        """For all standard models, encoder and decoder states match."""
        for model_name, config in WHISPER_CONFIGS.items():
            assert config.n_audio_state == config.n_text_state, \
                f"{model_name} has mismatched encoder/decoder states"

    def test_encoder_heads_match_text_heads(self):
        """For all standard models, encoder and decoder heads match."""
        for model_name, config in WHISPER_CONFIGS.items():
            assert config.n_audio_head == config.n_text_head, \
                f"{model_name} has mismatched encoder/decoder heads"

    def test_all_configs_have_standard_audio_params(self):
        """All configs have standard audio processing parameters."""
        for model_name, config in WHISPER_CONFIGS.items():
            assert config.sample_rate == 16000, f"{model_name} has non-standard sample rate"
            assert config.n_fft == 400, f"{model_name} has non-standard n_fft"
            assert config.hop_length == 160, f"{model_name} has non-standard hop_length"
            assert config.chunk_length == 30, f"{model_name} has non-standard chunk_length"
            assert config.n_audio_ctx == 1500, f"{model_name} has non-standard n_audio_ctx"
            assert config.n_text_ctx == 448, f"{model_name} has non-standard n_text_ctx"

    def test_config_hierarchy_scaling(self):
        """Model dimensions should scale appropriately: tiny < base < small < medium < large."""
        tiny = WHISPER_CONFIGS["tiny"]
        base = WHISPER_CONFIGS["base"]
        small = WHISPER_CONFIGS["small"]
        medium = WHISPER_CONFIGS["medium"]
        large = WHISPER_CONFIGS["large"]

        # State dimensions should increase
        assert tiny.n_audio_state < base.n_audio_state
        assert base.n_audio_state < small.n_audio_state
        assert small.n_audio_state < medium.n_audio_state
        assert medium.n_audio_state < large.n_audio_state

        # Layer counts should increase
        assert tiny.n_audio_layer < base.n_audio_layer
        assert base.n_audio_layer < small.n_audio_layer
        assert small.n_audio_layer < medium.n_audio_layer
        assert medium.n_audio_layer < large.n_audio_layer

        # Head counts should increase
        assert tiny.n_audio_head < base.n_audio_head
        assert base.n_audio_head < small.n_audio_head
        assert small.n_audio_head < medium.n_audio_head
        assert medium.n_audio_head < large.n_audio_head
