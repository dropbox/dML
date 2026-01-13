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
Tests for whisper_mlx/presets.py module.

Tests the optimization presets and TranscriptionConfig.
"""

import pytest

from tools.whisper_mlx.presets import (
    PRESET_INFO,
    ModelVariant,
    OptimizationPreset,
    QualityThresholds,
    TranscriptionConfig,
    list_presets,
)


class TestModelVariant:
    """Tests for ModelVariant enum."""

    def test_model_variants_exist(self):
        """Test that all expected model variants exist."""
        assert ModelVariant.LARGE_V3 is not None
        assert ModelVariant.LARGE_V3_TURBO is not None
        assert ModelVariant.DISTIL_LARGE_V3 is not None

    def test_model_variant_values(self):
        """Test that model variants have valid HuggingFace paths."""
        assert "large-v3" in ModelVariant.LARGE_V3.value
        assert "turbo" in ModelVariant.LARGE_V3_TURBO.value
        assert "distil" in ModelVariant.DISTIL_LARGE_V3.value

    def test_model_variants_are_strings(self):
        """Test that model variant values are strings."""
        for variant in ModelVariant:
            assert isinstance(variant.value, str)


class TestOptimizationPreset:
    """Tests for OptimizationPreset enum."""

    def test_all_presets_exist(self):
        """Test that all expected presets exist."""
        assert OptimizationPreset.MAX_QUALITY is not None
        assert OptimizationPreset.BALANCED is not None
        assert OptimizationPreset.FAST is not None
        assert OptimizationPreset.ULTRA_FAST is not None

    def test_preset_count(self):
        """Test that we have exactly 4 presets."""
        assert len(list(OptimizationPreset)) == 4

    def test_all_presets_have_info(self):
        """Test that all presets have corresponding info."""
        for preset in OptimizationPreset:
            assert preset in PRESET_INFO
            info = PRESET_INFO[preset]
            assert "description" in info
            assert "model" in info
            assert "expected_wer" in info
            assert "speedup" in info


class TestQualityThresholds:
    """Tests for QualityThresholds dataclass."""

    def test_default_values(self):
        """Test default threshold values."""
        thresholds = QualityThresholds()
        assert thresholds.compression_ratio == 2.4
        assert thresholds.logprob == -1.0
        assert thresholds.no_speech == 0.6

    def test_custom_values(self):
        """Test custom threshold values."""
        thresholds = QualityThresholds(
            compression_ratio=3.0,
            logprob=-1.5,
            no_speech=0.8,
        )
        assert thresholds.compression_ratio == 3.0
        assert thresholds.logprob == -1.5
        assert thresholds.no_speech == 0.8

    def test_nullable_thresholds(self):
        """Test that thresholds can be None (disabled)."""
        thresholds = QualityThresholds(
            compression_ratio=None,
            logprob=None,
            no_speech=None,
        )
        assert thresholds.compression_ratio is None
        assert thresholds.logprob is None
        assert thresholds.no_speech is None


class TestTranscriptionConfig:
    """Tests for TranscriptionConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TranscriptionConfig()
        assert config.model_variant == ModelVariant.LARGE_V3
        assert config.weight_bits is None  # FP16
        assert config.quantize_kv is True  # INT8 KV (lossless)
        assert config.beam_size == 1
        assert config.temperature == 0.0
        assert config.condition_on_previous is True
        assert config.timestamps is True
        assert config.language is None  # Auto-detect
        assert config.use_vad is False

    def test_get_model_name(self):
        """Test get_model_name returns correct path."""
        config = TranscriptionConfig()
        assert "large-v3" in config.get_model_name()

        config = TranscriptionConfig(model_variant=ModelVariant.DISTIL_LARGE_V3)
        assert "distil" in config.get_model_name()

    def test_get_transcribe_kwargs(self):
        """Test get_transcribe_kwargs returns valid kwargs."""
        config = TranscriptionConfig(
            language="en",
            temperature=0.2,
        )
        kwargs = config.get_transcribe_kwargs()

        assert "language" in kwargs
        assert kwargs["language"] == "en"
        assert "temperature" in kwargs
        assert "compression_ratio_threshold" in kwargs
        assert "logprob_threshold" in kwargs
        assert "no_speech_threshold" in kwargs

    def test_get_transcribe_kwargs_temperature_fallback(self):
        """Test that temperature fallback is used when temperature is 0."""
        config = TranscriptionConfig(
            temperature=0.0,
            temperature_fallback=(0.0, 0.2, 0.4),
        )
        kwargs = config.get_transcribe_kwargs()

        # When temperature=0.0 and fallback is set, use fallback tuple
        assert kwargs["temperature"] == (0.0, 0.2, 0.4)

    def test_get_transcribe_long_kwargs(self):
        """Test get_transcribe_long_kwargs returns valid kwargs."""
        config = TranscriptionConfig(
            language="ja",
            condition_on_previous=False,
        )
        kwargs = config.get_transcribe_long_kwargs()

        assert kwargs["language"] == "ja"
        assert kwargs["condition_on_previous_text"] is False

    def test_describe(self):
        """Test describe returns readable string."""
        config = TranscriptionConfig()
        desc = config.describe()

        assert "WhisperMLX Configuration" in desc
        assert "Model:" in desc
        assert "Weights:" in desc
        assert "FP16" in desc  # Default weight type


class TestTranscriptionConfigFromPreset:
    """Tests for TranscriptionConfig.from_preset()."""

    def test_max_quality_preset(self):
        """Test MAX_QUALITY preset configuration."""
        config = TranscriptionConfig.from_preset(OptimizationPreset.MAX_QUALITY)

        assert config.model_variant == ModelVariant.LARGE_V3
        assert config.weight_bits is None  # FP16
        assert config.quantize_kv is True
        assert config.temperature == 0.0
        assert config.condition_on_previous is True
        assert config.timestamps is True

    def test_balanced_preset(self):
        """Test BALANCED preset configuration."""
        config = TranscriptionConfig.from_preset(OptimizationPreset.BALANCED)

        # BALANCED uses turbo model (4 decoder layers)
        assert config.model_variant == ModelVariant.LARGE_V3_TURBO
        assert config.weight_bits is None  # FP16
        assert config.quantize_kv is True
        # Relaxed quality thresholds
        assert config.quality_thresholds.compression_ratio == 2.6

    def test_fast_preset(self):
        """Test FAST preset configuration."""
        config = TranscriptionConfig.from_preset(OptimizationPreset.FAST)

        assert config.model_variant == ModelVariant.LARGE_V3_TURBO
        assert config.weight_bits == 8  # INT8 weights
        assert config.beam_size == 1  # Greedy
        assert config.condition_on_previous is False  # Skip for speed
        assert config.timestamps is False  # Skip for speed

    def test_ultra_fast_preset(self):
        """Test ULTRA_FAST preset configuration."""
        config = TranscriptionConfig.from_preset(OptimizationPreset.ULTRA_FAST)

        assert config.model_variant == ModelVariant.DISTIL_LARGE_V3
        assert config.weight_bits == 4  # INT4 weights
        assert config.beam_size == 1  # Greedy
        assert config.use_vad is True  # VAD enabled (not yet implemented)
        # Disabled quality thresholds
        assert config.quality_thresholds.compression_ratio is None
        assert config.quality_thresholds.logprob is None

    def test_all_presets_create_valid_config(self):
        """Test that all presets create valid configurations."""
        for preset in OptimizationPreset:
            config = TranscriptionConfig.from_preset(preset)
            assert config is not None
            assert isinstance(config.model_variant, ModelVariant)
            assert config.get_model_name() is not None

    def test_invalid_preset_raises(self):
        """Test that invalid preset raises ValueError."""
        with pytest.raises(ValueError):
            TranscriptionConfig.from_preset("invalid_preset")


class TestPresetQualitySpeedTradeoffs:
    """Tests verifying expected quality/speed relationships between presets."""

    def test_preset_ordering_by_expected_speed(self):
        """Test that presets are ordered by expected speed."""
        # MAX_QUALITY should be slowest (baseline)
        # ULTRA_FAST should be fastest

        max_q = TranscriptionConfig.from_preset(OptimizationPreset.MAX_QUALITY)
        balanced = TranscriptionConfig.from_preset(OptimizationPreset.BALANCED)
        fast = TranscriptionConfig.from_preset(OptimizationPreset.FAST)
        ultra = TranscriptionConfig.from_preset(OptimizationPreset.ULTRA_FAST)

        # Model layers: large-v3 (32) > turbo (4) > distil (2)
        # Weight bits: None (FP16) > 8 (INT8) > 4 (INT4)

        # MAX_QUALITY uses full large-v3
        assert max_q.model_variant == ModelVariant.LARGE_V3
        assert max_q.weight_bits is None

        # BALANCED uses turbo (faster decoder)
        assert balanced.model_variant == ModelVariant.LARGE_V3_TURBO

        # FAST adds INT8 quantization
        assert fast.weight_bits == 8

        # ULTRA_FAST uses distil + INT4
        assert ultra.model_variant == ModelVariant.DISTIL_LARGE_V3
        assert ultra.weight_bits == 4

    def test_preset_ordering_by_expected_quality(self):
        """Test that presets have progressively relaxed quality thresholds."""
        max_q = TranscriptionConfig.from_preset(OptimizationPreset.MAX_QUALITY)
        balanced = TranscriptionConfig.from_preset(OptimizationPreset.BALANCED)
        fast = TranscriptionConfig.from_preset(OptimizationPreset.FAST)
        ultra = TranscriptionConfig.from_preset(OptimizationPreset.ULTRA_FAST)

        # Compression ratio thresholds (higher = more lenient)
        # MAX_QUALITY: 2.4
        # BALANCED: 2.6
        # FAST: 3.0
        # ULTRA_FAST: None (disabled)

        assert max_q.quality_thresholds.compression_ratio == 2.4
        assert balanced.quality_thresholds.compression_ratio == 2.6
        assert fast.quality_thresholds.compression_ratio == 3.0
        assert ultra.quality_thresholds.compression_ratio is None


class TestListPresets:
    """Tests for list_presets() function."""

    def test_list_presets_returns_string(self):
        """Test that list_presets returns a string."""
        result = list_presets()
        assert isinstance(result, str)

    def test_list_presets_contains_all_presets(self):
        """Test that list_presets mentions all preset names."""
        result = list_presets()
        for preset in OptimizationPreset:
            assert preset.value.upper() in result

    def test_list_presets_contains_info(self):
        """Test that list_presets contains expected information."""
        result = list_presets()
        assert "WER" in result
        assert "RTF" in result or "Speedup" in result


class TestConfigCustomization:
    """Tests for customizing configs after creation from preset."""

    def test_can_override_language(self):
        """Test that language can be overridden after preset creation."""
        config = TranscriptionConfig.from_preset(OptimizationPreset.BALANCED)
        assert config.language is None  # Default: auto-detect

        config.language = "en"
        assert config.language == "en"

        kwargs = config.get_transcribe_kwargs()
        assert kwargs["language"] == "en"

    def test_can_override_temperature(self):
        """Test that temperature can be overridden."""
        config = TranscriptionConfig.from_preset(OptimizationPreset.MAX_QUALITY)
        config.temperature = 0.5

        assert config.temperature == 0.5

    def test_can_override_quality_thresholds(self):
        """Test that quality thresholds can be overridden."""
        config = TranscriptionConfig.from_preset(OptimizationPreset.FAST)

        # Override to use stricter thresholds
        config.quality_thresholds = QualityThresholds(
            compression_ratio=2.0,
            logprob=-0.5,
            no_speech=0.5,
        )

        kwargs = config.get_transcribe_kwargs()
        assert kwargs["compression_ratio_threshold"] == 2.0
        assert kwargs["logprob_threshold"] == -0.5


class TestPresetDocumentation:
    """Tests for preset documentation and info."""

    def test_preset_info_completeness(self):
        """Test that PRESET_INFO has all required fields for each preset."""
        required_fields = ["description", "model", "expected_wer", "expected_rtf", "speedup"]

        for preset in OptimizationPreset:
            info = PRESET_INFO[preset]
            for field in required_fields:
                assert field in info, f"Missing '{field}' in {preset.name} info"

    def test_preset_info_values_are_strings(self):
        """Test that preset info values are strings."""
        for preset, info in PRESET_INFO.items():
            for key, value in info.items():
                assert isinstance(value, str), f"{preset.name}.{key} should be string"
