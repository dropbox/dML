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
Tests for pytorch_to_mlx validator modules.

Tests for:
- tools/pytorch_to_mlx/validator/numerical_validator.py
- tools/pytorch_to_mlx/validator/benchmark.py
"""

import numpy as np
import pytest

from tools.pytorch_to_mlx.validator.benchmark import (
    Benchmark,
    BenchmarkResult,
    LatencyStats,
    MemoryStats,
    ParallelStats,
    ThroughputStats,
)
from tools.pytorch_to_mlx.validator.numerical_validator import (
    NumericalValidator,
    TensorComparison,
    ValidationReport,
    ValidationStatus,
)

# Module-level RNG for reproducible tests
_rng = np.random.default_rng(42)

# =============================================================================
# ValidationStatus Tests
# =============================================================================


class TestValidationStatus:
    """Tests for ValidationStatus enum."""

    def test_status_values(self):
        """Test that all expected status values exist."""
        assert ValidationStatus.PASSED.value == "passed"
        assert ValidationStatus.FAILED.value == "failed"
        assert ValidationStatus.ERROR.value == "error"
        assert ValidationStatus.SKIPPED.value == "skipped"

    def test_status_from_string(self):
        """Test creating status from string value."""
        assert ValidationStatus("passed") == ValidationStatus.PASSED
        assert ValidationStatus("failed") == ValidationStatus.FAILED


# =============================================================================
# TensorComparison Tests
# =============================================================================


class TestTensorComparison:
    """Tests for TensorComparison dataclass."""

    def test_create_tensor_comparison(self):
        """Test creating a TensorComparison instance."""
        tc = TensorComparison(
            name="output_0",
            shape_match=True,
            dtype_match=True,
            max_abs_error=1e-6,
            mean_abs_error=1e-7,
            max_rel_error=1e-5,
            mean_rel_error=1e-6,
            num_mismatches=0,
            total_elements=1000,
            pytorch_dtype="float32",
            mlx_dtype="float32",
            pytorch_shape=(10, 100),
            mlx_shape=(10, 100),
        )
        assert tc.name == "output_0"
        assert tc.shape_match is True
        assert tc.max_abs_error == 1e-6
        assert tc.num_mismatches == 0

    def test_tensor_comparison_shape_mismatch(self):
        """Test TensorComparison with shape mismatch."""
        tc = TensorComparison(
            name="output",
            shape_match=False,
            dtype_match=True,
            max_abs_error=float("inf"),
            mean_abs_error=float("inf"),
            max_rel_error=float("inf"),
            mean_rel_error=float("inf"),
            num_mismatches=100,
            total_elements=100,
            pytorch_dtype="float32",
            mlx_dtype="float32",
            pytorch_shape=(10, 10),
            mlx_shape=(10, 20),
        )
        assert tc.shape_match is False
        assert tc.pytorch_shape != tc.mlx_shape


# =============================================================================
# ValidationReport Tests
# =============================================================================


class TestValidationReport:
    """Tests for ValidationReport dataclass."""

    def test_create_passing_report(self):
        """Test creating a passing validation report."""
        report = ValidationReport(
            status=ValidationStatus.PASSED,
            num_outputs=3,
            passed=3,
            failed=0,
            comparisons=[],
            overall_max_error=1e-6,
            overall_mean_error=1e-7,
        )
        assert report.status == ValidationStatus.PASSED
        assert report.passed == 3
        assert report.failed == 0

    def test_create_failing_report(self):
        """Test creating a failing validation report."""
        report = ValidationReport(
            status=ValidationStatus.FAILED,
            num_outputs=3,
            passed=2,
            failed=1,
            comparisons=[],
            overall_max_error=0.1,
            overall_mean_error=0.01,
            error_message=None,
        )
        assert report.status == ValidationStatus.FAILED
        assert report.failed == 1

    def test_create_error_report(self):
        """Test creating an error validation report."""
        report = ValidationReport(
            status=ValidationStatus.ERROR,
            num_outputs=0,
            passed=0,
            failed=0,
            comparisons=[],
            overall_max_error=float("inf"),
            overall_mean_error=float("inf"),
            error_message="Model loading failed",
        )
        assert report.status == ValidationStatus.ERROR
        assert "Model loading failed" in report.error_message

    def test_report_with_metadata(self):
        """Test validation report with metadata."""
        report = ValidationReport(
            status=ValidationStatus.PASSED,
            num_outputs=1,
            passed=1,
            failed=0,
            comparisons=[],
            overall_max_error=1e-6,
            overall_mean_error=1e-7,
            metadata={"model_name": "test_model", "device": "mps"},
        )
        assert report.metadata["model_name"] == "test_model"


# =============================================================================
# NumericalValidator Tests
# =============================================================================


class TestNumericalValidator:
    """Tests for NumericalValidator class."""

    def test_create_validator(self):
        """Test creating a NumericalValidator instance."""
        validator = NumericalValidator(atol=1e-5, rtol=1e-4, match_threshold=0.99)
        assert validator.atol == 1e-5
        assert validator.rtol == 1e-4
        assert validator.match_threshold == 0.99

    def test_default_tolerances(self):
        """Test default tolerance values."""
        validator = NumericalValidator()
        assert validator.atol == 1e-5
        assert validator.rtol == 1e-4
        assert validator.match_threshold == 0.99


class TestNumericalValidatorCompareArrays:
    """Tests for NumericalValidator._compare_arrays method."""

    def test_compare_identical_arrays(self):
        """Test comparing identical arrays."""
        validator = NumericalValidator()
        arr = _rng.standard_normal(10, 10).astype(np.float32)

        comparison = validator._compare_arrays("test", arr, arr.copy())

        assert comparison.shape_match is True
        assert comparison.dtype_match is True
        assert comparison.max_abs_error == 0.0
        assert comparison.mean_abs_error == 0.0
        assert comparison.num_mismatches == 0

    def test_compare_similar_arrays(self):
        """Test comparing similar arrays within tolerance."""
        validator = NumericalValidator(atol=1e-4)
        arr1 = np.ones((10, 10), dtype=np.float32)
        arr2 = arr1 + _rng.standard_normal(10, 10).astype(np.float32) * 1e-5

        comparison = validator._compare_arrays("test", arr1, arr2)

        assert comparison.shape_match is True
        assert comparison.max_abs_error < 1e-4

    def test_compare_different_arrays(self):
        """Test comparing significantly different arrays."""
        validator = NumericalValidator(atol=1e-5)
        arr1 = np.ones((10, 10), dtype=np.float32)
        arr2 = np.zeros((10, 10), dtype=np.float32)

        comparison = validator._compare_arrays("test", arr1, arr2)

        assert comparison.shape_match is True
        assert comparison.max_abs_error == 1.0
        assert comparison.num_mismatches > 0

    def test_compare_shape_mismatch(self):
        """Test comparing arrays with different shapes."""
        validator = NumericalValidator()
        arr1 = np.ones((10, 10), dtype=np.float32)
        arr2 = np.ones((10, 20), dtype=np.float32)

        comparison = validator._compare_arrays("test", arr1, arr2)

        assert comparison.shape_match is False
        assert comparison.max_abs_error == float("inf")

    def test_compare_different_dtypes(self):
        """Test comparing arrays with different dtypes."""
        validator = NumericalValidator()
        arr1 = np.ones((10, 10), dtype=np.float32)
        arr2 = np.ones((10, 10), dtype=np.float64)

        comparison = validator._compare_arrays("test", arr1, arr2)

        # Shape matches, dtype doesn't
        assert comparison.shape_match is True
        assert comparison.dtype_match is False
        # Values should still match
        assert comparison.max_abs_error < 1e-6


class TestNumericalValidatorIsPassing:
    """Tests for NumericalValidator._is_passing method."""

    def test_passing_comparison(self):
        """Test that a good comparison passes."""
        validator = NumericalValidator(atol=1e-4, rtol=1e-3, match_threshold=0.99)

        comparison = TensorComparison(
            name="test",
            shape_match=True,
            dtype_match=True,
            max_abs_error=1e-6,
            mean_abs_error=1e-7,
            max_rel_error=1e-5,
            mean_rel_error=1e-6,
            num_mismatches=0,
            total_elements=1000,
            pytorch_dtype="float32",
            mlx_dtype="float32",
            pytorch_shape=(10, 100),
            mlx_shape=(10, 100),
        )

        assert validator._is_passing(comparison) is True

    def test_failing_shape_mismatch(self):
        """Test that shape mismatch fails."""
        validator = NumericalValidator()

        comparison = TensorComparison(
            name="test",
            shape_match=False,
            dtype_match=True,
            max_abs_error=0.0,
            mean_abs_error=0.0,
            max_rel_error=0.0,
            mean_rel_error=0.0,
            num_mismatches=0,
            total_elements=1000,
            pytorch_dtype="float32",
            mlx_dtype="float32",
            pytorch_shape=(10, 10),
            mlx_shape=(10, 20),
        )

        assert validator._is_passing(comparison) is False

    def test_failing_too_many_mismatches(self):
        """Test that too many mismatches fails."""
        validator = NumericalValidator(match_threshold=0.99)

        comparison = TensorComparison(
            name="test",
            shape_match=True,
            dtype_match=True,
            max_abs_error=0.1,
            mean_abs_error=0.01,
            max_rel_error=0.1,
            mean_rel_error=0.01,
            num_mismatches=50,  # 5% mismatch
            total_elements=1000,
            pytorch_dtype="float32",
            mlx_dtype="float32",
            pytorch_shape=(10, 100),
            mlx_shape=(10, 100),
        )

        assert validator._is_passing(comparison) is False


class TestNumericalValidatorCreateTestInputs:
    """Tests for NumericalValidator.create_test_inputs static method."""

    def test_create_float32_inputs(self):
        """Test creating float32 test inputs."""
        inputs = NumericalValidator.create_test_inputs(
            shapes=[(10, 20), (5, 5)], dtype="float32", seed=42,
        )

        assert len(inputs) == 2
        assert inputs[0]["input"].shape == (10, 20)
        assert inputs[1]["input"].shape == (5, 5)
        assert inputs[0]["input"].dtype == np.float32

    def test_create_int64_inputs(self):
        """Test creating int64 test inputs."""
        inputs = NumericalValidator.create_test_inputs(
            shapes=[(100,)], dtype="int64", seed=42,
        )

        assert len(inputs) == 1
        assert inputs[0]["input"].dtype == np.int64
        assert np.all(inputs[0]["input"] >= 0)
        assert np.all(inputs[0]["input"] < 100)

    def test_reproducible_with_seed(self):
        """Test that inputs are reproducible with same seed."""
        inputs1 = NumericalValidator.create_test_inputs(
            shapes=[(10,)], dtype="float32", seed=123,
        )
        inputs2 = NumericalValidator.create_test_inputs(
            shapes=[(10,)], dtype="float32", seed=123,
        )

        np.testing.assert_array_equal(inputs1[0]["input"], inputs2[0]["input"])


class TestNumericalValidatorGenerateReport:
    """Tests for NumericalValidator.generate_report method."""

    def test_generate_passing_report(self):
        """Test generating report for passing validation."""
        validator = NumericalValidator()

        comparison = TensorComparison(
            name="output_0",
            shape_match=True,
            dtype_match=True,
            max_abs_error=1e-6,
            mean_abs_error=1e-7,
            max_rel_error=1e-5,
            mean_rel_error=1e-6,
            num_mismatches=0,
            total_elements=1000,
            pytorch_dtype="float32",
            mlx_dtype="float32",
            pytorch_shape=(10, 100),
            mlx_shape=(10, 100),
        )

        report = ValidationReport(
            status=ValidationStatus.PASSED,
            num_outputs=1,
            passed=1,
            failed=0,
            comparisons=[comparison],
            overall_max_error=1e-6,
            overall_mean_error=1e-7,
            pytorch_inference_time=0.01,
            mlx_inference_time=0.005,
        )

        report_str = validator.generate_report(report)

        assert "VALIDATION REPORT" in report_str
        assert "PASSED" in report_str
        assert "Speedup" in report_str
        assert "output_0" in report_str


# =============================================================================
# Benchmark Dataclass Tests
# =============================================================================


class TestLatencyStats:
    """Tests for LatencyStats dataclass."""

    def test_create_latency_stats(self):
        """Test creating LatencyStats instance."""
        stats = LatencyStats(
            mean_ms=10.5,
            std_ms=1.2,
            min_ms=8.0,
            max_ms=15.0,
            p50_ms=10.0,
            p95_ms=14.0,
            p99_ms=14.8,
            samples=100,
        )
        assert stats.mean_ms == 10.5
        assert stats.samples == 100


class TestThroughputStats:
    """Tests for ThroughputStats dataclass."""

    def test_create_throughput_stats(self):
        """Test creating ThroughputStats instance."""
        stats = ThroughputStats(
            samples_per_second=100.0,
            total_samples=500,
            total_time_seconds=5.0,
            batch_size=1,
        )
        assert stats.samples_per_second == 100.0
        assert stats.total_samples == 500


class TestMemoryStats:
    """Tests for MemoryStats dataclass."""

    def test_create_memory_stats(self):
        """Test creating MemoryStats instance."""
        stats = MemoryStats(peak_mb=512.0, allocated_mb=256.0, reserved_mb=512.0)
        assert stats.peak_mb == 512.0


class TestParallelStats:
    """Tests for ParallelStats dataclass."""

    def test_create_parallel_stats(self):
        """Test creating ParallelStats instance."""
        stats = ParallelStats(
            num_threads=4,
            total_samples=400,
            total_time_seconds=2.0,
            throughput=200.0,
            per_thread_throughput=50.0,
            scaling_efficiency=0.95,
        )
        assert stats.num_threads == 4
        assert stats.scaling_efficiency == 0.95


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_create_benchmark_result(self):
        """Test creating BenchmarkResult instance."""
        latency = LatencyStats(
            mean_ms=10.0,
            std_ms=1.0,
            min_ms=8.0,
            max_ms=12.0,
            p50_ms=10.0,
            p95_ms=11.5,
            p99_ms=11.9,
            samples=100,
        )
        throughput = ThroughputStats(
            samples_per_second=100.0,
            total_samples=500,
            total_time_seconds=5.0,
            batch_size=1,
        )
        result = BenchmarkResult(
            model_name="test_model",
            framework="mlx",
            latency=latency,
            throughput=throughput,
        )
        assert result.model_name == "test_model"
        assert result.framework == "mlx"


# =============================================================================
# Benchmark Class Tests
# =============================================================================


class TestBenchmarkInit:
    """Tests for Benchmark initialization."""

    def test_default_params(self):
        """Test default benchmark parameters."""
        bench = Benchmark()
        assert bench.warmup_iterations == 10
        assert bench.benchmark_iterations == 100
        assert bench.batch_size == 1

    def test_custom_params(self):
        """Test custom benchmark parameters."""
        bench = Benchmark(warmup_iterations=5, benchmark_iterations=50, batch_size=8)
        assert bench.warmup_iterations == 5
        assert bench.benchmark_iterations == 50
        assert bench.batch_size == 8


class TestBenchmarkComputeLatencyStats:
    """Tests for Benchmark._compute_latency_stats method."""

    def test_compute_latency_stats(self):
        """Test computing latency statistics."""
        bench = Benchmark()
        times = [10.0, 11.0, 9.0, 12.0, 10.5, 9.5, 11.5, 10.0, 10.2, 10.8]

        stats = bench._compute_latency_stats(times)

        assert stats.samples == 10
        assert stats.min_ms == 9.0
        assert stats.max_ms == 12.0
        assert 10.0 < stats.mean_ms < 11.0
        assert stats.p50_ms is not None
        assert stats.p95_ms is not None

    def test_compute_latency_stats_single_sample(self):
        """Test latency stats with single sample."""
        bench = Benchmark()
        times = [10.0]

        stats = bench._compute_latency_stats(times)

        assert stats.samples == 1
        assert stats.mean_ms == 10.0
        assert stats.std_ms == 0.0  # Can't compute std with 1 sample


class TestBenchmarkCreateTestInput:
    """Tests for Benchmark.create_test_input static method."""

    def test_create_float32_input(self):
        """Test creating float32 test input."""
        inp = Benchmark.create_test_input(shape=(10, 20), dtype="float32", seed=42)

        assert "input" in inp
        assert inp["input"].shape == (10, 20)
        assert inp["input"].dtype == np.float32

    def test_create_int32_input(self):
        """Test creating int32 test input."""
        inp = Benchmark.create_test_input(shape=(100,), dtype="int32", seed=42)

        assert inp["input"].dtype == np.int32
        assert np.all(inp["input"] >= 0)
        assert np.all(inp["input"] < 100)

    def test_reproducible_with_seed(self):
        """Test that input is reproducible with same seed."""
        inp1 = Benchmark.create_test_input(shape=(10,), dtype="float32", seed=123)
        inp2 = Benchmark.create_test_input(shape=(10,), dtype="float32", seed=123)

        np.testing.assert_array_equal(inp1["input"], inp2["input"])


class TestBenchmarkCompare:
    """Tests for Benchmark.compare method."""

    def test_compare_results(self):
        """Test comparing PyTorch and MLX benchmark results."""
        bench = Benchmark()

        pytorch_result = BenchmarkResult(
            model_name="model",
            framework="pytorch",
            latency=LatencyStats(
                mean_ms=20.0,
                std_ms=2.0,
                min_ms=18.0,
                max_ms=25.0,
                p50_ms=20.0,
                p95_ms=24.0,
                p99_ms=24.8,
                samples=100,
            ),
            throughput=ThroughputStats(
                samples_per_second=50.0,
                total_samples=250,
                total_time_seconds=5.0,
                batch_size=1,
            ),
            metadata={"device": "mps"},
        )

        mlx_result = BenchmarkResult(
            model_name="model",
            framework="mlx",
            latency=LatencyStats(
                mean_ms=10.0,
                std_ms=1.0,
                min_ms=8.0,
                max_ms=12.0,
                p50_ms=10.0,
                p95_ms=11.5,
                p99_ms=11.9,
                samples=100,
            ),
            throughput=ThroughputStats(
                samples_per_second=100.0,
                total_samples=500,
                total_time_seconds=5.0,
                batch_size=1,
            ),
            metadata={"device": "metal"},
        )

        comparison = bench.compare(pytorch_result, mlx_result)

        assert comparison["latency_speedup"] == 2.0  # 20.0 / 10.0
        assert comparison["throughput_speedup"] == 2.0  # 100.0 / 50.0
        assert comparison["pytorch_device"] == "mps"
        assert comparison["mlx_device"] == "metal"


class TestBenchmarkGenerateReport:
    """Tests for Benchmark.generate_report method."""

    def test_generate_single_result_report(self):
        """Test generating report with single result."""
        bench = Benchmark()

        result = BenchmarkResult(
            model_name="test_model",
            framework="mlx",
            latency=LatencyStats(
                mean_ms=10.0,
                std_ms=1.0,
                min_ms=8.0,
                max_ms=12.0,
                p50_ms=10.0,
                p95_ms=11.5,
                p99_ms=11.9,
                samples=100,
            ),
            throughput=ThroughputStats(
                samples_per_second=100.0,
                total_samples=500,
                total_time_seconds=5.0,
                batch_size=1,
            ),
        )

        report = bench.generate_report([result])

        assert "BENCHMARK REPORT" in report
        assert "test_model" in report
        assert "MLX" in report
        assert "Latency:" in report
        assert "Throughput:" in report

    def test_generate_comparison_report(self):
        """Test generating report with comparison."""
        bench = Benchmark()

        result = BenchmarkResult(
            model_name="test_model",
            framework="mlx",
            latency=LatencyStats(
                mean_ms=10.0,
                std_ms=1.0,
                min_ms=8.0,
                max_ms=12.0,
                p50_ms=10.0,
                p95_ms=11.5,
                p99_ms=11.9,
                samples=100,
            ),
            throughput=ThroughputStats(
                samples_per_second=100.0,
                total_samples=500,
                total_time_seconds=5.0,
                batch_size=1,
            ),
        )

        comparison = {
            "latency_speedup": 2.0,
            "throughput_speedup": 2.0,
        }

        report = bench.generate_report([result], comparison=comparison)

        assert "COMPARISON" in report
        assert "2.00x" in report

    def test_generate_report_with_memory(self):
        """Test generating report with memory stats."""
        bench = Benchmark()

        result = BenchmarkResult(
            model_name="test_model",
            framework="mlx",
            latency=LatencyStats(
                mean_ms=10.0,
                std_ms=1.0,
                min_ms=8.0,
                max_ms=12.0,
                p50_ms=10.0,
                p95_ms=11.5,
                p99_ms=11.9,
                samples=100,
            ),
            throughput=ThroughputStats(
                samples_per_second=100.0,
                total_samples=500,
                total_time_seconds=5.0,
                batch_size=1,
            ),
            memory=MemoryStats(peak_mb=512.0, allocated_mb=256.0, reserved_mb=512.0),
        )

        report = bench.generate_report([result])

        assert "Memory:" in report
        assert "512.0 MB" in report

    def test_generate_report_with_parallel(self):
        """Test generating report with parallel scaling stats."""
        bench = Benchmark()

        result = BenchmarkResult(
            model_name="test_model",
            framework="mlx",
            latency=LatencyStats(
                mean_ms=10.0,
                std_ms=1.0,
                min_ms=8.0,
                max_ms=12.0,
                p50_ms=10.0,
                p95_ms=11.5,
                p99_ms=11.9,
                samples=100,
            ),
            throughput=ThroughputStats(
                samples_per_second=100.0,
                total_samples=500,
                total_time_seconds=5.0,
                batch_size=1,
            ),
        )

        parallel_stats = [
            ParallelStats(
                num_threads=1,
                total_samples=100,
                total_time_seconds=1.0,
                throughput=100.0,
                per_thread_throughput=100.0,
                scaling_efficiency=1.0,
            ),
            ParallelStats(
                num_threads=4,
                total_samples=400,
                total_time_seconds=1.0,
                throughput=380.0,
                per_thread_throughput=95.0,
                scaling_efficiency=0.95,
            ),
        ]

        report = bench.generate_report([result], parallel_results=parallel_stats)

        assert "PARALLEL SCALING" in report
        assert "4 threads" in report
        assert "95.0%" in report


# =============================================================================
# Integration Tests (require actual frameworks)
# =============================================================================


class TestValidatorIntegration:
    """Integration tests for NumericalValidator with real frameworks."""

    @pytest.mark.slow
    def test_validate_simple_arrays_mlx(self):
        """Test validating simple arrays with MLX."""
        import mlx.core as mx

        # Create simple MLX arrays
        arr1 = mx.array([1.0, 2.0, 3.0])
        arr2 = mx.array([1.0, 2.0, 3.0])

        validator = NumericalValidator()

        # Compare as numpy
        np1 = np.array(arr1)
        np2 = np.array(arr2)

        comparison = validator._compare_arrays("test", np1, np2)

        assert comparison.shape_match is True
        assert comparison.max_abs_error == 0.0

    @pytest.mark.slow
    def test_validate_with_small_error_mlx(self):
        """Test validating arrays with small numerical error."""
        import mlx.core as mx

        arr1 = mx.array([1.0, 2.0, 3.0])
        # Add small noise
        arr2 = mx.array([1.0 + 1e-7, 2.0 - 1e-7, 3.0 + 1e-7])

        validator = NumericalValidator(atol=1e-5)

        np1 = np.array(arr1)
        np2 = np.array(arr2)

        comparison = validator._compare_arrays("test", np1, np2)

        assert comparison.shape_match is True
        assert comparison.max_abs_error < 1e-5
        assert validator._is_passing(comparison) is True


class TestBenchmarkIntegration:
    """Integration tests for Benchmark with real models."""

    @pytest.mark.slow
    def test_benchmark_simple_mlx_function(self):
        """Test benchmarking a simple MLX function."""
        import mlx.core as mx

        bench = Benchmark(warmup_iterations=2, benchmark_iterations=10)

        # Create simple MLX model-like function
        class SimpleModel:
            def __call__(self, x):
                # Use mx.maximum instead of mx.relu (which is in mlx.nn)
                return mx.maximum(x @ mx.ones((10, 10)) + mx.zeros((10,)), 0)

        model = SimpleModel()
        inputs = {"input": _rng.standard_normal(10, 10).astype(np.float32)}

        result = bench.benchmark_mlx(model, inputs, model_name="simple")

        assert result.framework == "mlx"
        assert result.latency.samples == 10
        assert result.latency.mean_ms > 0
        assert result.throughput.samples_per_second > 0
