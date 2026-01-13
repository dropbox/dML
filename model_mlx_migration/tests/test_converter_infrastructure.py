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
Tests for PyTorch to MLX Converter Infrastructure (Phase 1)

Tests the core components:
- TorchScriptAnalyzer
- OpMapper
- MLXCodeGenerator
- WeightConverter
- NumericalValidator
- Benchmark
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add tools to path
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))


class TestOpMapper:
    """Tests for the OpMapper component."""

    def test_direct_mapping(self):
        """Test direct op mappings."""
        from pytorch_to_mlx.analyzer.op_mapper import MappingType, OpMapper

        mapper = OpMapper()

        # Test common direct mappings
        mapping = mapper.map_op("aten::linear")
        assert mapping.mapping_type == MappingType.DIRECT
        assert mapping.mlx_op == "mx.nn.Linear"

        mapping = mapper.map_op("aten::relu")
        assert mapping.mapping_type == MappingType.DIRECT
        assert mapping.mlx_op == "mx.nn.relu"

        mapping = mapper.map_op("aten::softmax")
        assert mapping.mapping_type == MappingType.DIRECT
        assert mapping.mlx_op == "mx.softmax"

    def test_decomposed_ops(self):
        """Test decomposed operation mappings."""
        from pytorch_to_mlx.analyzer.op_mapper import MappingType, OpMapper

        mapper = OpMapper()

        # Test decomposed ops
        mapping = mapper.map_op("aten::scaled_dot_product_attention")
        assert mapping.mapping_type == MappingType.DECOMPOSED
        assert mapping.decomposition is not None
        assert "def scaled_dot_product_attention" in mapping.decomposition

    def test_unsupported_ops(self):
        """Test unsupported operation detection."""
        from pytorch_to_mlx.analyzer.op_mapper import MappingType, OpMapper

        mapper = OpMapper()

        # Test unknown op
        mapping = mapper.map_op("aten::some_unknown_op")
        assert mapping.mapping_type == MappingType.UNSUPPORTED

    def test_coverage_report(self):
        """Test coverage report generation."""
        from pytorch_to_mlx.analyzer.op_mapper import OpMapper

        mapper = OpMapper()

        ops = ["aten::linear", "aten::relu", "aten::softmax", "aten::unknown_op"]
        report = mapper.get_coverage_report(ops)

        assert report["total_ops"] == 4
        assert report["supported_ops"] == 3
        assert report["unsupported_ops"] == 1
        assert "aten::unknown_op" in report["unsupported_list"]
        assert report["coverage_percent"] == 75.0

    def test_list_supported_ops(self):
        """Test listing all supported operations."""
        from pytorch_to_mlx.analyzer.op_mapper import OpMapper

        supported = OpMapper.list_supported_ops()
        assert len(supported) > 50  # Should have many supported ops
        assert "aten::linear" in supported
        assert "aten::conv2d" in supported


class TestMLXCodeGenerator:
    """Tests for the MLXCodeGenerator component."""

    def test_sanitize_name(self):
        """Test name sanitization for Python identifiers."""
        from pytorch_to_mlx.generator.mlx_code_generator import MLXCodeGenerator

        generator = MLXCodeGenerator()

        assert generator._sanitize_name("my_model") == "MyModel"
        assert generator._sanitize_name("my-model") == "MyModel"
        assert generator._sanitize_name("123model") == "_123model"
        assert generator._sanitize_name("model.pt") == "Model"

    def test_infer_type(self):
        """Test Python type inference."""
        from pytorch_to_mlx.generator.mlx_code_generator import MLXCodeGenerator

        generator = MLXCodeGenerator()

        assert generator._infer_type(42) == "int"
        assert generator._infer_type(3.14) == "float"
        assert generator._infer_type(True) == "bool"
        assert generator._infer_type((1, 2, 3)) == "Tuple[int, int, int]"


class TestWeightConverter:
    """Tests for the WeightConverter component."""

    @pytest.fixture
    def simple_pytorch_model(self):
        """Create a simple PyTorch model for testing."""
        import torch.nn as nn

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)

            def forward(self, x):
                return self.linear(x)

        return SimpleModel()

    def test_name_conversion(self):
        """Test weight name conversion."""
        from pytorch_to_mlx.generator.weight_converter import WeightConverter

        converter = WeightConverter()

        # Test standard names (should be unchanged)
        assert converter.convert_name("linear.weight") == "linear.weight"
        assert converter.convert_name("linear.bias") == "linear.bias"

        # Test LayerNorm naming
        assert converter.convert_name("layer.gamma") == "layer.weight"
        assert converter.convert_name("layer.beta") == "layer.bias"

    def test_tensor_conversion(self):
        """Test tensor dtype conversion."""
        import torch
        from pytorch_to_mlx.generator.weight_converter import WeightConverter

        converter = WeightConverter()

        # Test float32
        t = torch.randn(10, 5)
        arr = converter.convert_tensor(t)
        assert arr.dtype == np.float32
        assert arr.shape == (10, 5)

        # Test float16
        t_fp16 = torch.randn(10, 5).half()
        arr_fp16 = converter.convert_tensor(t_fp16)
        assert arr_fp16.dtype == np.float16

    def test_conversion_with_target_dtype(self):
        """Test conversion with target dtype."""
        import torch
        from pytorch_to_mlx.generator.weight_converter import WeightConverter

        converter = WeightConverter(target_dtype="float16")

        t = torch.randn(10, 5)
        arr = converter.convert_tensor(t)
        assert arr.dtype == np.float16


class TestNumericalValidator:
    """Tests for the NumericalValidator component."""

    def test_array_comparison_pass(self):
        """Test array comparison with passing tolerance."""
        from pytorch_to_mlx.validator.numerical_validator import NumericalValidator

        validator = NumericalValidator(atol=1e-5, rtol=1e-4)

        # Arrays that should match
        arr1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        arr2 = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        comparison = validator._compare_arrays("test", arr1, arr2)

        assert comparison.shape_match
        assert comparison.max_abs_error < 1e-10
        assert comparison.num_mismatches == 0

    def test_array_comparison_fail(self):
        """Test array comparison with failing tolerance."""
        from pytorch_to_mlx.validator.numerical_validator import NumericalValidator

        validator = NumericalValidator(atol=1e-5, rtol=1e-4)

        # Arrays that should not match
        arr1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        arr2 = np.array([1.1, 2.1, 3.1], dtype=np.float32)  # 0.1 difference

        comparison = validator._compare_arrays("test", arr1, arr2)

        assert comparison.shape_match
        assert comparison.max_abs_error > 0.05
        assert comparison.num_mismatches > 0

    def test_shape_mismatch(self):
        """Test handling of shape mismatches."""
        from pytorch_to_mlx.validator.numerical_validator import NumericalValidator

        validator = NumericalValidator()

        arr1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        arr2 = np.array([1.0, 2.0], dtype=np.float32)

        comparison = validator._compare_arrays("test", arr1, arr2)

        assert not comparison.shape_match
        assert comparison.max_abs_error == float("inf")

    def test_create_test_inputs(self):
        """Test creation of test inputs."""
        from pytorch_to_mlx.validator.numerical_validator import NumericalValidator

        inputs = NumericalValidator.create_test_inputs(
            shapes=[(1, 10), (2, 20)], dtype="float32", seed=42,
        )

        assert len(inputs) == 2
        assert inputs[0]["input"].shape == (1, 10)
        assert inputs[1]["input"].shape == (2, 20)
        assert inputs[0]["input"].dtype == np.float32


class TestBenchmark:
    """Tests for the Benchmark component."""

    def test_latency_stats(self):
        """Test latency statistics computation."""
        from pytorch_to_mlx.validator.benchmark import Benchmark

        benchmark = Benchmark()

        times = [10.0, 11.0, 12.0, 13.0, 14.0]  # ms
        stats = benchmark._compute_latency_stats(times)

        assert stats.mean_ms == 12.0
        assert stats.min_ms == 10.0
        assert stats.max_ms == 14.0
        assert stats.samples == 5

    def test_create_test_input(self):
        """Test test input creation."""
        from pytorch_to_mlx.validator.benchmark import Benchmark

        test_input = Benchmark.create_test_input(
            shape=(2, 10), dtype="float32", seed=42,
        )

        assert "input" in test_input
        assert test_input["input"].shape == (2, 10)
        assert test_input["input"].dtype == np.float32


class TestTorchScriptAnalyzerBasics:
    """Basic tests for TorchScriptAnalyzer (without model files)."""

    def test_op_categories(self):
        """Test operation categorization."""
        from pytorch_to_mlx.analyzer.torchscript_analyzer import (
            OpCategory,
            TorchScriptAnalyzer,
        )

        analyzer = TorchScriptAnalyzer()

        # Layer ops
        assert analyzer._categorize_op("aten::linear") == OpCategory.LAYER
        assert analyzer._categorize_op("aten::conv2d") == OpCategory.LAYER

        # Activation ops
        assert analyzer._categorize_op("aten::relu") == OpCategory.ACTIVATION
        assert analyzer._categorize_op("aten::gelu") == OpCategory.ACTIVATION

        # Tensor ops
        assert analyzer._categorize_op("aten::reshape") == OpCategory.TENSOR
        assert analyzer._categorize_op("aten::transpose") == OpCategory.TENSOR


class TestCLI:
    """Tests for the CLI module."""

    def test_cli_help(self):
        """Test CLI help output."""
        import sys

        from pytorch_to_mlx.cli import main

        # Capture args
        old_argv = sys.argv
        sys.argv = ["pytorch_to_mlx", "--help"]

        try:
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 0
        finally:
            sys.argv = old_argv

    def test_cli_no_command(self):
        """Test CLI with no command."""
        import sys

        from pytorch_to_mlx.cli import main

        old_argv = sys.argv
        sys.argv = ["pytorch_to_mlx"]

        try:
            result = main()
            assert result == 1  # Should return error code
        finally:
            sys.argv = old_argv


class TestIntegration:
    """Integration tests using real PyTorch models."""

    @pytest.fixture
    def simple_model_file(self, tmp_path):
        """Create a simple TorchScript model file for testing."""
        import torch
        import torch.nn as nn

        class SimpleNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 20)
                self.fc2 = nn.Linear(20, 5)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.relu(self.fc1(x))
                return self.fc2(x)

        model = SimpleNet()
        model.eval()

        # Export to TorchScript using trace (more compatible with Python 3.14)
        example_input = torch.randn(1, 10)
        traced = torch.jit.trace(model, example_input)
        model_path = tmp_path / "simple_model.pt"
        torch.jit.save(traced, str(model_path))

        return model_path

    def test_analyzer_with_model(self, simple_model_file):
        """Test TorchScriptAnalyzer with a real model."""
        from pytorch_to_mlx.analyzer.torchscript_analyzer import TorchScriptAnalyzer

        analyzer = TorchScriptAnalyzer(str(simple_model_file))
        arch = analyzer.get_architecture()

        assert arch.name == "simple_model"
        assert arch.total_params > 0
        assert len(arch.weights) > 0

    def test_weight_converter_with_model(self, simple_model_file, tmp_path):
        """Test WeightConverter with a real model."""
        from pytorch_to_mlx.generator.weight_converter import WeightConverter

        converter = WeightConverter()
        output_path = tmp_path / "weights.npz"

        stats = converter.convert(str(simple_model_file), str(output_path))

        assert stats.converted_tensors > 0
        assert stats.total_bytes_output > 0
        assert output_path.exists()

    def test_full_conversion_pipeline(self, simple_model_file, tmp_path):
        """Test full conversion pipeline."""
        from pytorch_to_mlx.analyzer.op_mapper import OpMapper
        from pytorch_to_mlx.analyzer.torchscript_analyzer import TorchScriptAnalyzer
        from pytorch_to_mlx.generator.mlx_code_generator import MLXCodeGenerator
        from pytorch_to_mlx.generator.weight_converter import WeightConverter

        output_dir = tmp_path / "mlx_model"
        output_dir.mkdir()

        # Step 1: Analyze
        analyzer = TorchScriptAnalyzer(str(simple_model_file))
        arch = analyzer.get_architecture()
        assert arch.total_params > 0

        # Step 2: Check op coverage
        mapper = OpMapper()
        ops = [op.name for op in arch.ops]
        coverage = mapper.get_coverage_report(ops)
        # Simple model should have high coverage
        assert coverage["coverage_percent"] >= 50.0

        # Step 3: Generate code
        generator = MLXCodeGenerator()
        generator.generate_file(
            arch, str(output_dir / "model.py"), class_name="SimpleNetMLX",
        )
        assert (output_dir / "model.py").exists()

        # Step 4: Convert weights
        converter = WeightConverter()
        stats = converter.convert(
            str(simple_model_file), str(output_dir / "weights.npz"),
        )
        assert stats.converted_tensors > 0
        assert (output_dir / "weights.npz").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
