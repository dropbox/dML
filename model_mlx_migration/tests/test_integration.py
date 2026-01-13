#!/usr/bin/env python3
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
Integration Test for PyTorch to MLX Converter

This test validates the complete conversion pipeline:
1. Analyze a PyTorch model
2. Generate MLX code
3. Convert weights
4. Validate numerical equivalence
5. Benchmark performance

Prerequisites:
    pip install -r requirements.txt
    python tests/fixtures/create_test_model.py  # Create test models

Usage:
    pytest tests/test_integration.py -v
    # Or directly:
    python tests/test_integration.py
"""

import sys
from pathlib import Path

import pytest

# Add tools to path
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))


# Skip all tests if PyTorch not available
pytestmark = pytest.mark.skipif(
    not pytest.importorskip("torch", reason="PyTorch required"),
    reason="PyTorch not installed",
)


class TestFullConversionPipeline:
    """Integration tests for complete conversion pipeline."""

    @pytest.fixture
    def simple_model_path(self) -> Path:
        """Path to simple test model."""
        path = Path(__file__).parent / "fixtures" / "simple_linear.pt"
        if not path.exists():
            pytest.skip(
                "Test model not created. Run: python tests/fixtures/create_test_model.py",
            )
        return path

    @pytest.fixture
    def temp_output_dir(self, tmp_path: Path) -> Path:
        """Temporary output directory."""
        return tmp_path / "mlx_output"

    def test_analyze_model(self, simple_model_path):
        """Test model analysis."""
        from pytorch_to_mlx.analyzer.torchscript_analyzer import TorchScriptAnalyzer

        analyzer = TorchScriptAnalyzer(str(simple_model_path))
        architecture = analyzer.get_architecture()

        # Verify basic structure
        assert architecture.name == "simple_linear"
        assert architecture.total_params > 0
        assert len(architecture.layers) > 0
        assert len(architecture.ops) > 0
        assert len(architecture.weights) > 0

        # Should have linear layers
        layer_types = [layer.layer_type for layer in architecture.layers]
        assert "Linear" in layer_types

        # Print summary
        print("\n" + analyzer.summarize())

    def test_op_coverage(self, simple_model_path):
        """Test operation coverage analysis."""
        from pytorch_to_mlx.analyzer.op_mapper import OpMapper
        from pytorch_to_mlx.analyzer.torchscript_analyzer import TorchScriptAnalyzer

        analyzer = TorchScriptAnalyzer(str(simple_model_path))
        architecture = analyzer.get_architecture()

        mapper = OpMapper()
        op_names = [op.name for op in architecture.ops]
        coverage = mapper.get_coverage_report(op_names)

        # Simple model should have high coverage
        assert coverage["coverage_percent"] >= 90, (
            f"Coverage too low: {coverage['coverage_percent']}%"
        )
        print(f"\nOp coverage: {coverage['coverage_percent']:.1f}%")

    def test_generate_code(self, simple_model_path, temp_output_dir):
        """Test code generation."""
        from pytorch_to_mlx.analyzer.torchscript_analyzer import TorchScriptAnalyzer
        from pytorch_to_mlx.generator.mlx_code_generator import MLXCodeGenerator

        analyzer = TorchScriptAnalyzer(str(simple_model_path))
        architecture = analyzer.get_architecture()

        temp_output_dir.mkdir(parents=True, exist_ok=True)
        output_file = temp_output_dir / "model.py"

        generator = MLXCodeGenerator()
        generator.generate_file(
            architecture, str(output_file), class_name="SimpleModelMLX",
        )

        # Verify file was created
        assert output_file.exists()

        # Verify content
        content = output_file.read_text()
        assert "class SimpleModelMLX" in content
        assert "nn.Module" in content
        assert "def __call__" in content

        print(f"\nGenerated code preview:\n{content[:500]}...")

    def test_convert_weights(self, simple_model_path, temp_output_dir):
        """Test weight conversion."""
        from pytorch_to_mlx.generator.weight_converter import WeightConverter

        temp_output_dir.mkdir(parents=True, exist_ok=True)
        output_file = temp_output_dir / "weights.safetensors"

        converter = WeightConverter(target_dtype="float32")
        stats = converter.convert(str(simple_model_path), str(output_file))

        # Verify conversion
        assert stats.converted_tensors > 0
        assert stats.total_bytes_output > 0
        assert len(stats.errors) == 0

        # Verify file exists (either .safetensors or .npz)
        assert output_file.exists() or (temp_output_dir / "weights.npz").exists()

        print(f"\nConverted {stats.converted_tensors} tensors")
        print(f"Output size: {stats.total_bytes_output / 1024:.1f} KB")

    def test_verify_weights(self, simple_model_path, temp_output_dir):
        """Test weight verification."""
        from pytorch_to_mlx.generator.weight_converter import WeightConverter

        temp_output_dir.mkdir(parents=True, exist_ok=True)
        output_file = temp_output_dir / "weights.safetensors"

        converter = WeightConverter()
        converter.convert(str(simple_model_path), str(output_file))

        # Verify round-trip
        weights_path = (
            output_file if output_file.exists() else temp_output_dir / "weights.npz"
        )
        report = converter.verify_conversion(
            str(simple_model_path), str(weights_path), tolerance=1e-6,
        )

        assert report["verified"], f"Weight verification failed: {report}"
        print(
            f"\nWeight verification: {report['matched']}/{report['pytorch_tensors']} matched",
        )

    def test_full_pipeline_cli(self, simple_model_path, temp_output_dir):
        """Test full conversion via CLI."""
        import os
        import subprocess

        temp_output_dir.mkdir(parents=True, exist_ok=True)

        # Set up environment with tools in PYTHONPATH
        project_root = Path(__file__).parent.parent
        tools_path = project_root / "tools"
        env = os.environ.copy()
        env["PYTHONPATH"] = str(tools_path) + os.pathsep + env.get("PYTHONPATH", "")

        # Run analyze
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytorch_to_mlx",
                "analyze",
                "--input",
                str(simple_model_path),
            ],
            capture_output=True,
            text=True,
            cwd=str(project_root),
            env=env,
        )
        print(f"\nAnalyze output:\n{result.stdout}")
        assert result.returncode == 0, f"Analyze failed: {result.stderr}"

        # Run convert
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytorch_to_mlx",
                "convert",
                "--input",
                str(simple_model_path),
                "--output",
                str(temp_output_dir),
            ],
            capture_output=True,
            text=True,
            cwd=str(project_root),
            env=env,
        )
        print(f"\nConvert output:\n{result.stdout}")
        assert result.returncode == 0, f"Convert failed: {result.stderr}"

        # Verify outputs
        assert (temp_output_dir / "model.py").exists()
        assert (temp_output_dir / "weights.safetensors").exists() or (
            temp_output_dir / "weights.npz"
        ).exists()


class TestNumericalValidation:
    """Tests for numerical validation between PyTorch and MLX."""

    @pytest.fixture
    def simple_pytorch_model(self):
        """Create a simple PyTorch model."""
        import torch.nn as nn

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 32)

            def forward(self, x):
                return self.linear(x)

        model = SimpleModel()
        model.eval()
        return model

    def test_validator_pytorch_only(self, simple_pytorch_model):
        """Test validator with PyTorch model."""
        import torch
        from pytorch_to_mlx.validator.numerical_validator import NumericalValidator

        validator = NumericalValidator(atol=1e-5)

        # Create test inputs
        test_inputs = validator.create_test_inputs([(1, 64)])

        # For now, just validate PyTorch model runs
        with torch.no_grad():
            pt_input = torch.from_numpy(test_inputs[0]["input"])
            pt_output = simple_pytorch_model(pt_input)

        assert pt_output.shape == (1, 32)
        print(f"\nPyTorch output shape: {pt_output.shape}")


class TestBenchmark:
    """Tests for benchmark functionality."""

    @pytest.fixture
    def simple_pytorch_model(self):
        """Create a simple PyTorch model."""
        import torch.nn as nn

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 32)

            def forward(self, x):
                return self.linear(x)

        model = SimpleModel()
        model.eval()
        return model

    def test_benchmark_pytorch(self, simple_pytorch_model):
        """Test PyTorch benchmarking."""
        from pytorch_to_mlx.validator.benchmark import Benchmark

        benchmark = Benchmark(warmup_iterations=5, benchmark_iterations=20)
        test_input = benchmark.create_test_input((1, 64))

        result = benchmark.benchmark_pytorch(
            simple_pytorch_model, test_input, model_name="simple_model",
        )

        assert result.latency.mean_ms > 0
        assert result.throughput.samples_per_second > 0

        print("\nPyTorch benchmark:")
        print(f"  Latency: {result.latency.mean_ms:.2f} ms")
        print(f"  Throughput: {result.throughput.samples_per_second:.1f} samples/sec")


def run_quick_test():
    """Quick sanity test without pytest."""
    print("Running quick integration test...")
    print("=" * 50)

    # Test imports
    print("\n1. Testing imports...")
    try:
        from pytorch_to_mlx.analyzer.op_mapper import OpMapper  # noqa: F401
        from pytorch_to_mlx.analyzer.torchscript_analyzer import (
            TorchScriptAnalyzer,  # noqa: F401
        )
        from pytorch_to_mlx.generator.mlx_code_generator import (
            MLXCodeGenerator,  # noqa: F401
        )
        from pytorch_to_mlx.generator.weight_converter import (
            WeightConverter,  # noqa: F401
        )
        from pytorch_to_mlx.validator.benchmark import Benchmark  # noqa: F401
        from pytorch_to_mlx.validator.numerical_validator import (
            NumericalValidator,  # noqa: F401
        )

        print("   All imports successful!")
    except ImportError as e:
        print(f"   Import failed: {e}")
        return False

    # Test OpMapper
    print("\n2. Testing OpMapper...")
    mapper = OpMapper()
    supported = OpMapper.list_supported_ops()
    print(f"   Supported ops: {len(supported)}")

    coverage = mapper.get_coverage_report(
        ["aten::linear", "aten::relu", "aten::unknown"],
    )
    print(f"   Coverage test: {coverage['coverage_percent']:.1f}%")

    # Test code generator with mock architecture
    print("\n3. Testing MLXCodeGenerator...")
    from pytorch_to_mlx.analyzer.torchscript_analyzer import (
        LayerInfo,
        ModelArchitecture,
        OpCategory,
        OpInfo,
        WeightInfo,
    )

    mock_arch = ModelArchitecture(
        name="TestModel",
        layers=[
            LayerInfo(
                "layer1",
                "Linear",
                [],
                [],
                {"weight": (64, 32)},
                {"in_features": 64, "out_features": 32},
            ),
        ],
        ops=[OpInfo("aten::linear", OpCategory.LAYER, [], [], count=1)],
        weights=[WeightInfo("layer1.weight", (64, 32), "float32", True, 64 * 32 * 4)],
        input_shapes=[(1, 64)],
        output_shapes=[(1, 32)],
        total_params=64 * 32,
        total_size_bytes=64 * 32 * 4,
    )

    generator = MLXCodeGenerator()
    result = generator.generate(mock_arch, "TestMLX")
    assert "class TestMLX" in result.model_code
    print("   Code generation successful!")

    print("\n" + "=" * 50)
    print("Quick test PASSED!")
    print("\nTo run full integration tests with real models:")
    print("  1. pip install -r requirements.txt")
    print("  2. python tests/fixtures/create_test_model.py")
    print("  3. pytest tests/test_integration.py -v")

    return True


if __name__ == "__main__":
    # Run quick test without pytest
    success = run_quick_test()
    sys.exit(0 if success else 1)
