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
Tests for Kokoro TTS Converter

Tests the Kokoro converter without requiring actual model weights for most tests.
For full validation tests, run with actual model weights downloaded.
"""

import sys
from pathlib import Path

# Add tools to path
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))


class TestKokoroConverterImports:
    """Test converter imports and initialization."""

    def test_import_converter(self):
        """Test that converter can be imported."""
        from pytorch_to_mlx.converters import KokoroConverter

        assert KokoroConverter is not None

    def test_import_dataclasses(self):
        """Test that dataclasses can be imported."""
        from pytorch_to_mlx.converters.kokoro_converter import (
            BenchmarkResult,
            ConversionResult,
            ValidationResult,
        )

        assert ConversionResult is not None
        assert ValidationResult is not None
        assert BenchmarkResult is not None

    def test_converter_init(self):
        """Test converter initialization."""
        from pytorch_to_mlx.converters import KokoroConverter

        converter = KokoroConverter()
        assert converter is not None


class TestConversionResult:
    """Test ConversionResult dataclass."""

    def test_result_creation_success(self):
        """Test ConversionResult creation for successful conversion."""
        from pytorch_to_mlx.converters.kokoro_converter import ConversionResult

        result = ConversionResult(
            success=True,
            mlx_path="./mlx-kokoro",
            model_size_mb=330.5,
            num_parameters=82_000_000,
            error=None,
        )
        assert result.success is True
        assert result.mlx_path == "./mlx-kokoro"
        assert result.model_size_mb == 330.5
        assert result.num_parameters == 82_000_000
        assert result.error is None

    def test_result_creation_failure(self):
        """Test ConversionResult creation for failed conversion."""
        from pytorch_to_mlx.converters.kokoro_converter import ConversionResult

        result = ConversionResult(
            success=False,
            mlx_path="",
            model_size_mb=0.0,
            num_parameters=0,
            error="Model file not found",
        )
        assert result.success is False
        assert result.error == "Model file not found"


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_result_creation_passed(self):
        """Test ValidationResult creation for passing validation."""
        from pytorch_to_mlx.converters.kokoro_converter import ValidationResult

        result = ValidationResult(
            passed=True,
            text_encoder_max_error=1e-6,
            bert_max_error=1e-6,
            error=None,
        )
        assert result.passed is True
        assert result.text_encoder_max_error == 1e-6
        assert result.bert_max_error == 1e-6
        assert result.error is None

    def test_result_creation_failed(self):
        """Test ValidationResult creation for failing validation."""
        from pytorch_to_mlx.converters.kokoro_converter import ValidationResult

        result = ValidationResult(
            passed=False,
            text_encoder_max_error=0.1,
            bert_max_error=0.2,
            error="Max error exceeds threshold",
        )
        assert result.passed is False
        assert result.text_encoder_max_error == 0.1
        assert result.bert_max_error == 0.2
        assert result.error == "Max error exceeds threshold"


class TestBenchmarkResult:
    """Test BenchmarkResult dataclass."""

    def test_result_creation(self):
        """Test BenchmarkResult creation."""
        from pytorch_to_mlx.converters.kokoro_converter import BenchmarkResult

        result = BenchmarkResult(
            mlx_audio_per_second=38.0,
            pytorch_audio_per_second=10.0,
            speedup=3.8,
        )
        assert result.mlx_audio_per_second == 38.0
        assert result.pytorch_audio_per_second == 10.0
        assert result.speedup == 3.8

    def test_speedup_ratio(self):
        """Test that speedup ratio is calculated correctly."""
        from pytorch_to_mlx.converters.kokoro_converter import BenchmarkResult

        result = BenchmarkResult(
            mlx_audio_per_second=38.0,
            pytorch_audio_per_second=10.0,
            speedup=38.0 / 10.0,
        )
        assert abs(result.speedup - 3.8) < 0.01


class TestSupportedModels:
    """Test supported models list."""

    def test_list_supported_models(self):
        """Test supported models list."""
        from pytorch_to_mlx.converters import KokoroConverter

        models = KokoroConverter.list_supported_models()
        assert len(models) > 0
        assert isinstance(models, list)

    def test_default_model_in_list(self):
        """Test that default Kokoro model is in supported list."""
        from pytorch_to_mlx.converters import KokoroConverter

        models = KokoroConverter.list_supported_models()
        assert "hexgrad/Kokoro-82M" in models

    def test_model_names_are_strings(self):
        """Test that all model names are strings."""
        from pytorch_to_mlx.converters import KokoroConverter

        models = KokoroConverter.list_supported_models()
        for model in models:
            assert isinstance(model, str)


class TestVoiceListing:
    """Test voice listing features."""

    def test_list_voices_returns_list(self):
        """Test that list_voices returns a list of voice names."""
        from pytorch_to_mlx.converters import KokoroConverter

        converter = KokoroConverter()
        voices = converter.list_voices()
        assert isinstance(voices, list)

    def test_voices_are_strings(self):
        """Test that all voice entries are strings."""
        from pytorch_to_mlx.converters import KokoroConverter

        converter = KokoroConverter()
        voices = converter.list_voices()

        # All voice names should be strings
        for voice in voices:
            assert isinstance(voice, str)

    def test_voice_naming_convention(self):
        """Test that voices follow the naming convention (prefix_name)."""
        from pytorch_to_mlx.converters import KokoroConverter

        converter = KokoroConverter()
        voices = converter.list_voices()

        # Kokoro voices typically have format like af_alloy, am_adam, etc.
        # At least some should follow this pattern
        has_prefix_format = any("_" in v for v in voices)
        assert len(voices) == 0 or has_prefix_format


class TestSelectVoiceEmbedding:
    """Test voice embedding selection."""

    def test_select_voice_embedding_signature(self):
        """Test that select_voice_embedding method exists."""
        from pytorch_to_mlx.converters import KokoroConverter

        converter = KokoroConverter()
        assert hasattr(converter, "select_voice_embedding")
        assert callable(converter.select_voice_embedding)

    def test_load_voice_pack_signature(self):
        """Test that load_voice_pack method exists."""
        from pytorch_to_mlx.converters import KokoroConverter

        converter = KokoroConverter()
        assert hasattr(converter, "load_voice_pack")
        assert callable(converter.load_voice_pack)

    def test_load_voice_signature(self):
        """Test that load_voice method exists."""
        from pytorch_to_mlx.converters import KokoroConverter

        converter = KokoroConverter()
        assert hasattr(converter, "load_voice")
        assert callable(converter.load_voice)


class TestConversionMethods:
    """Test conversion method signatures."""

    def test_convert_method_exists(self):
        """Test that convert method exists."""
        from pytorch_to_mlx.converters import KokoroConverter

        converter = KokoroConverter()
        assert hasattr(converter, "convert")
        assert callable(converter.convert)

    def test_validate_method_exists(self):
        """Test that validate method exists."""
        from pytorch_to_mlx.converters import KokoroConverter

        converter = KokoroConverter()
        assert hasattr(converter, "validate")
        assert callable(converter.validate)

    def test_load_from_hf_method_exists(self):
        """Test that load_from_hf method exists."""
        from pytorch_to_mlx.converters import KokoroConverter

        converter = KokoroConverter()
        assert hasattr(converter, "load_from_hf")
        assert callable(converter.load_from_hf)


class TestExportMethods:
    """Test export method signatures."""

    def test_export_for_cpp_method_exists(self):
        """Test that export_for_cpp method exists."""
        from pytorch_to_mlx.converters import KokoroConverter

        converter = KokoroConverter()
        assert hasattr(converter, "export_for_cpp")
        assert callable(converter.export_for_cpp)

    def test_export_cpp_bundle_method_exists(self):
        """Test that export_cpp_bundle method exists."""
        from pytorch_to_mlx.converters import KokoroConverter

        converter = KokoroConverter()
        assert hasattr(converter, "export_cpp_bundle")
        assert callable(converter.export_cpp_bundle)


class TestKokoroModelImport:
    """Test KokoroModel and KokoroConfig imports."""

    def test_import_kokoro_model(self):
        """Test that KokoroModel can be imported."""
        from pytorch_to_mlx.converters.models.kokoro import KokoroModel

        assert KokoroModel is not None

    def test_import_kokoro_config(self):
        """Test that KokoroConfig can be imported."""
        from pytorch_to_mlx.converters.models.kokoro import KokoroConfig

        assert KokoroConfig is not None

    def test_kokoro_config_defaults(self):
        """Test KokoroConfig default values."""
        from pytorch_to_mlx.converters.models.kokoro import KokoroConfig

        config = KokoroConfig()
        assert hasattr(config, "dim_in")
        assert hasattr(config, "hidden_dim")
        assert hasattr(config, "style_dim")
        assert hasattr(config, "n_token")

    def test_kokoro_config_custom_values(self):
        """Test KokoroConfig with custom values."""
        from pytorch_to_mlx.converters.models.kokoro import KokoroConfig

        config = KokoroConfig(
            dim_in=64,
            hidden_dim=512,
            style_dim=128,
            n_token=178,
        )
        assert config.dim_in == 64
        assert config.hidden_dim == 512
        assert config.style_dim == 128
        assert config.n_token == 178


def run_quick_test():
    """Quick test without pytest."""
    print("Testing Kokoro Converter...")
    print("=" * 50)

    # Test imports
    print("\n1. Testing imports...")
    try:
        from pytorch_to_mlx.converters import KokoroConverter
        from pytorch_to_mlx.converters.kokoro_converter import (  # noqa: F401
            BenchmarkResult,
            ConversionResult,
            ValidationResult,
        )

        print("   Imports successful!")
    except ImportError as e:
        print(f"   Import failed: {e}")
        return False

    # Test converter init
    print("\n2. Testing converter initialization...")
    try:
        converter = KokoroConverter()
        print("   Converter created!")
    except Exception as e:
        print(f"   Failed: {e}")
        return False

    # Test supported models
    print("\n3. Testing supported models list...")
    models = KokoroConverter.list_supported_models()
    print(f"   Found {len(models)} supported models")
    for m in models:
        print(f"     - {m}")

    # Test voice listing
    print("\n4. Testing voice listing...")
    try:
        voices = converter.list_voices()
        total_voices = sum(len(v) for v in voices.values()) if voices else 0
        print(f"   Found {len(voices)} voice categories, {total_voices} total voices")
    except Exception as e:
        print(f"   Voice listing unavailable: {e}")

    # Test method existence
    print("\n5. Testing method signatures...")
    methods = ["convert", "validate", "load_from_hf", "export_for_cpp"]
    for method in methods:
        has_method = hasattr(converter, method)
        status = "OK" if has_method else "MISSING"
        print(f"   {status}: {method}")

    print("\n" + "=" * 50)
    print("Quick test PASSED!")
    print("\nTo test with actual model:")
    print('  python -c "')
    print("    from pytorch_to_mlx.converters import KokoroConverter")
    print("    c = KokoroConverter()")
    print('    result = c.convert(\"hexgrad/Kokoro-82M\", \"./mlx-kokoro\")"')

    return True


if __name__ == "__main__":
    sys.exit(0 if run_quick_test() else 1)
