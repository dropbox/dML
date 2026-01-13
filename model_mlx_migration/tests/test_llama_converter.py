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
Tests for LLaMA Converter

Tests the mlx-lm based LLaMA converter without requiring actual model weights.
For full validation tests, run with LLaMA weights available.
"""

import sys
from pathlib import Path

import pytest

# Add tools to path
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))


class TestLLaMAConverterImports:
    """Test converter imports and initialization."""

    def test_import_converter(self):
        """Test that converter can be imported."""
        from pytorch_to_mlx.converters import LLaMAConverter

        assert LLaMAConverter is not None

    def test_converter_init(self):
        """Test converter initialization."""
        from pytorch_to_mlx.converters import LLaMAConverter

        converter = LLaMAConverter()
        assert converter is not None

    def test_list_supported_models(self):
        """Test supported models list."""
        from pytorch_to_mlx.converters import LLaMAConverter

        models = LLaMAConverter.list_supported_models()
        assert len(models) > 0
        assert "meta-llama/Llama-3-8B" in models
        assert "mistralai/Mistral-7B-v0.1" in models


class TestConversionResult:
    """Test ConversionResult dataclass."""

    def test_result_dataclass(self):
        """Test ConversionResult creation."""
        from pytorch_to_mlx.converters.llama_converter import ConversionResult

        result = ConversionResult(
            success=True,
            mlx_path="/path/to/model",
            model_size_mb=1024.0,
            num_parameters=7_000_000_000,
        )
        assert result.success
        assert result.num_parameters == 7_000_000_000

    def test_failed_result(self):
        """Test ConversionResult with failure."""
        from pytorch_to_mlx.converters.llama_converter import ConversionResult

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
        from pytorch_to_mlx.converters.llama_converter import ValidationResult

        result = ValidationResult(
            passed=True,
            max_abs_error=1e-5,
            mean_abs_error=1e-6,
            tokens_compared=1000,
            mismatched_tokens=0,
        )
        assert result.passed
        assert result.max_abs_error < 1e-4


class TestBenchmarkResult:
    """Test BenchmarkResult dataclass."""

    def test_benchmark_result(self):
        """Test BenchmarkResult creation."""
        from pytorch_to_mlx.converters.llama_converter import BenchmarkResult

        result = BenchmarkResult(
            mlx_tokens_per_second=100.0,
            pytorch_tokens_per_second=50.0,
            speedup=2.0,
            mlx_time_to_first_token_ms=10.0,
            pytorch_time_to_first_token_ms=20.0,
        )
        assert result.speedup == 2.0
        assert result.mlx_tokens_per_second > result.pytorch_tokens_per_second


class TestBatchGenerationResult:
    """Test BatchGenerationResult dataclass."""

    def test_batch_result(self):
        """Test BatchGenerationResult creation."""
        from pytorch_to_mlx.converters.llama_converter import BatchGenerationResult

        result = BatchGenerationResult(
            texts=["Hello, world!", "How are you?"],
            total_tokens=100,
            total_time_ms=500.0,
            throughput_tokens_per_second=200.0,
        )
        assert len(result.texts) == 2
        assert result.total_tokens == 100
        assert result.throughput_tokens_per_second == 200.0


class TestSpeculativeGenerationResult:
    """Test SpeculativeGenerationResult dataclass."""

    def test_speculative_result(self):
        """Test SpeculativeGenerationResult creation."""
        from pytorch_to_mlx.converters.llama_converter import (
            SpeculativeGenerationResult,
        )

        result = SpeculativeGenerationResult(
            text="Generated text here",
            total_tokens=50,
            total_time_ms=250.0,
            tokens_per_second=200.0,
            acceptance_rate=0.85,
        )
        assert result.total_tokens == 50
        assert result.tokens_per_second == 200.0
        assert result.acceptance_rate == 0.85

    def test_speculative_result_no_acceptance_rate(self):
        """Test SpeculativeGenerationResult without acceptance rate."""
        from pytorch_to_mlx.converters.llama_converter import (
            SpeculativeGenerationResult,
        )

        result = SpeculativeGenerationResult(
            text="Generated text",
            total_tokens=50,
            total_time_ms=250.0,
            tokens_per_second=200.0,
        )
        assert result.acceptance_rate is None


class TestNewMethods:
    """Test new batch and speculative methods exist."""

    def test_generate_batch_method_exists(self):
        """Test that generate_batch method exists on converter."""
        from pytorch_to_mlx.converters import LLaMAConverter

        converter = LLaMAConverter()
        assert hasattr(converter, "generate_batch")
        assert callable(converter.generate_batch)

    def test_generate_speculative_method_exists(self):
        """Test that generate_speculative method exists on converter."""
        from pytorch_to_mlx.converters import LLaMAConverter

        converter = LLaMAConverter()
        assert hasattr(converter, "generate_speculative")
        assert callable(converter.generate_speculative)

    def test_benchmark_optimizations_method_exists(self):
        """Test that benchmark_optimizations method exists on converter."""
        from pytorch_to_mlx.converters import LLaMAConverter

        converter = LLaMAConverter()
        assert hasattr(converter, "benchmark_optimizations")
        assert callable(converter.benchmark_optimizations)


# Mark integration tests that require model weights
@pytest.mark.skipif(
    not Path("~/models/llama").expanduser().exists(),
    reason="LLaMA weights not available",
)
class TestLLaMAConversion:
    """Integration tests requiring actual model weights."""

    def test_conversion(self):
        """Test actual model conversion."""
        # This test requires HuggingFace authentication and model weights
        pytest.skip("Requires model weights and HF authentication")


def run_quick_test():
    """Quick test without pytest."""
    print("Testing LLaMA Converter...")
    print("=" * 50)

    # Test imports
    print("\n1. Testing imports...")
    try:
        from pytorch_to_mlx.converters import LLaMAConverter
        from pytorch_to_mlx.converters.llama_converter import (  # noqa: F401
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
        LLaMAConverter()
        print("   Converter created!")
    except Exception as e:
        print(f"   Failed: {e}")
        return False

    # Test supported models
    print("\n3. Testing supported models list...")
    models = LLaMAConverter.list_supported_models()
    print(f"   Found {len(models)} supported models")
    for m in models[:3]:
        print(f"     - {m}")

    print("\n" + "=" * 50)
    print("Quick test PASSED!")
    print("\nTo test conversion with real models:")
    print("  1. Get access to LLaMA weights on HuggingFace")
    print('  2. Run: python -c "')
    print("     from pytorch_to_mlx.converters import LLaMAConverter")
    print("     c = LLaMAConverter()")
    print("     c.convert('meta-llama/Llama-3.2-1B', './mlx-llama-1b')\"")

    return True


if __name__ == "__main__":
    sys.exit(0 if run_quick_test() else 1)
