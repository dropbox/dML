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
Tests for NLLB Converter

Tests the NLLB converter without requiring actual model weights.
For full validation tests, run with HuggingFace authentication.
"""

import sys
from pathlib import Path

# Add tools to path
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))


class TestNLLBConverterImports:
    """Test converter imports and initialization."""

    def test_import_converter(self):
        """Test that converter can be imported."""
        from pytorch_to_mlx.converters import NLLBConverter

        assert NLLBConverter is not None

    def test_converter_init(self):
        """Test converter initialization."""
        from pytorch_to_mlx.converters import NLLBConverter

        converter = NLLBConverter()
        assert converter is not None

    def test_list_supported_models(self):
        """Test supported models list."""
        from pytorch_to_mlx.converters import NLLBConverter

        models = NLLBConverter.list_supported_models()
        assert len(models) > 0
        assert "facebook/nllb-200-distilled-600M" in models
        assert "facebook/nllb-200-1.3B" in models


class TestConversionResult:
    """Test ConversionResult dataclass."""

    def test_result_dataclass(self):
        """Test ConversionResult creation."""
        from pytorch_to_mlx.converters.nllb_converter import ConversionResult

        result = ConversionResult(
            success=True,
            mlx_path="/path/to/model",
            model_size_mb=600.0,
            num_parameters=600_000_000,
        )
        assert result.success
        assert result.num_parameters == 600_000_000

    def test_failed_result(self):
        """Test ConversionResult with failure."""
        from pytorch_to_mlx.converters.nllb_converter import ConversionResult

        result = ConversionResult(
            success=False,
            mlx_path="/path/to/model",
            model_size_mb=0,
            num_parameters=0,
            error="Model not found",
        )
        assert not result.success
        assert result.error == "Model not found"


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_validation_result(self):
        """Test ValidationResult creation."""
        from pytorch_to_mlx.converters.nllb_converter import ValidationResult

        result = ValidationResult(
            passed=True,
            encoder_max_error=1e-6,
            decoder_max_error=1e-5,
            top_token_match=True,
        )
        assert result.passed
        assert result.encoder_max_error < 1e-4

    def test_validation_failed(self):
        """Test ValidationResult with failure."""
        from pytorch_to_mlx.converters.nllb_converter import ValidationResult

        result = ValidationResult(
            passed=False,
            encoder_max_error=0.5,
            decoder_max_error=0.5,
            top_token_match=False,
            error="Large error",
        )
        assert not result.passed
        assert result.error == "Large error"


class TestBenchmarkResult:
    """Test BenchmarkResult dataclass."""

    def test_benchmark_result(self):
        """Test BenchmarkResult creation."""
        from pytorch_to_mlx.converters.nllb_converter import BenchmarkResult

        result = BenchmarkResult(
            mlx_tokens_per_second=150.0,
            pytorch_tokens_per_second=75.0,
            speedup=2.0,
            mlx_encode_time_ms=5.0,
            pytorch_encode_time_ms=10.0,
        )
        assert result.speedup == 2.0
        assert result.mlx_tokens_per_second > result.pytorch_tokens_per_second


class TestCLICommands:
    """Test NLLB CLI commands."""

    def test_nllb_list_command(self):
        """Test nllb list CLI command."""
        import subprocess

        result = subprocess.run(
            [".venv/bin/python", "-m", "pytorch_to_mlx.cli", "nllb", "list"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            env={"PYTHONPATH": "tools"},
        )
        assert result.returncode == 0
        assert "facebook/nllb-200-distilled-600M" in result.stdout

    def test_nllb_help_command(self):
        """Test nllb --help CLI command."""
        import subprocess

        result = subprocess.run(
            [".venv/bin/python", "-m", "pytorch_to_mlx.cli", "nllb", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            env={"PYTHONPATH": "tools"},
        )
        assert result.returncode == 0
        assert "convert" in result.stdout
        assert "validate" in result.stdout
        assert "benchmark" in result.stdout
        assert "translate" in result.stdout


def run_quick_test():
    """Quick test without pytest."""
    print("Testing NLLB Converter...")
    print("=" * 50)

    # Test imports
    print("\n1. Testing imports...")
    try:
        from pytorch_to_mlx.converters import NLLBConverter
        from pytorch_to_mlx.converters.nllb_converter import (  # noqa: F401
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
        NLLBConverter()
        print("   Converter created!")
    except Exception as e:
        print(f"   Failed: {e}")
        return False

    # Test supported models
    print("\n3. Testing supported models list...")
    models = NLLBConverter.list_supported_models()
    print(f"   Found {len(models)} supported models")
    for m in models:
        print(f"     - {m}")

    print("\n" + "=" * 50)
    print("Quick test PASSED!")
    print("\nTo test conversion with real models:")
    print('  python -c "')
    print("    from pytorch_to_mlx.converters import NLLBConverter")
    print("    c = NLLBConverter()")
    print("    c.convert('facebook/nllb-200-distilled-600M', './mlx-nllb')\"")

    return True


if __name__ == "__main__":
    sys.exit(0 if run_quick_test() else 1)
