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
Numerical Validator

Validates that converted MLX models produce outputs matching the original PyTorch model.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class ValidationStatus(Enum):
    """Result status of validation."""

    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class TensorComparison:
    """Comparison results for a single tensor."""

    name: str
    shape_match: bool
    dtype_match: bool
    max_abs_error: float
    mean_abs_error: float
    max_rel_error: float
    mean_rel_error: float
    num_mismatches: int  # Count of values exceeding tolerance
    total_elements: int
    pytorch_dtype: str
    mlx_dtype: str
    pytorch_shape: tuple[int, ...]
    mlx_shape: tuple[int, ...]


@dataclass
class ValidationReport:
    """Complete validation report."""

    status: ValidationStatus
    num_outputs: int
    passed: int
    failed: int
    comparisons: list[TensorComparison]
    overall_max_error: float
    overall_mean_error: float
    pytorch_inference_time: float | None = None  # seconds
    mlx_inference_time: float | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class NumericalValidator:
    """
    Validates numerical equivalence between PyTorch and MLX models.

    Compares model outputs with configurable tolerances for:
    - Absolute error
    - Relative error
    - Element-wise matching percentage
    """

    def __init__(
        self,
        atol: float = 1e-5,
        rtol: float = 1e-4,
        match_threshold: float = 0.99,
    ):
        """
        Initialize validator.

        Args:
            atol: Absolute tolerance for comparison
            rtol: Relative tolerance for comparison
            match_threshold: Required percentage of matching elements (0-1)
        """
        self.atol = atol
        self.rtol = rtol
        self.match_threshold = match_threshold

    def validate(
        self,
        pytorch_model: Any,
        mlx_model: Any,
        test_inputs: list[dict[str, Any]],
        input_converter: Callable[[dict[str, Any], str], dict[str, Any]] | None = None,
        output_converter: Callable[[Any, str], Any] | None = None,
    ) -> ValidationReport:
        """
        Validate MLX model against PyTorch model.

        Args:
            pytorch_model: Loaded PyTorch model
            mlx_model: Loaded MLX model
            test_inputs: List of input dictionaries
            input_converter: Function to convert inputs for each framework
            output_converter: Function to convert outputs to comparable format

        Returns:
            ValidationReport with comparison results
        """
        import time

        comparisons = []
        pytorch_times = []
        mlx_times = []

        for input_dict in test_inputs:
            try:
                # Run PyTorch inference
                pt_start = time.perf_counter()
                pt_output = self._run_pytorch(
                    pytorch_model, input_dict, input_converter,
                )
                pt_time = time.perf_counter() - pt_start
                pytorch_times.append(pt_time)

                # Run MLX inference
                mlx_start = time.perf_counter()
                mlx_output = self._run_mlx(mlx_model, input_dict, input_converter)
                mlx_time = time.perf_counter() - mlx_start
                mlx_times.append(mlx_time)

                # Convert outputs to comparable format
                if output_converter:
                    pt_output = output_converter(pt_output, "pytorch")
                    mlx_output = output_converter(mlx_output, "mlx")

                # Compare outputs
                output_comparisons = self._compare_outputs(pt_output, mlx_output)
                comparisons.extend(output_comparisons)

            except Exception as e:
                return ValidationReport(
                    status=ValidationStatus.ERROR,
                    num_outputs=0,
                    passed=0,
                    failed=0,
                    comparisons=[],
                    overall_max_error=float("inf"),
                    overall_mean_error=float("inf"),
                    error_message=str(e),
                )

        # Aggregate results
        passed = sum(1 for c in comparisons if self._is_passing(c))
        failed = len(comparisons) - passed

        overall_max = max((c.max_abs_error for c in comparisons), default=0.0)
        overall_mean = (
            np.mean([c.mean_abs_error for c in comparisons]) if comparisons else 0.0
        )

        status = ValidationStatus.PASSED if failed == 0 else ValidationStatus.FAILED

        return ValidationReport(
            status=status,
            num_outputs=len(comparisons),
            passed=passed,
            failed=failed,
            comparisons=comparisons,
            overall_max_error=overall_max,
            overall_mean_error=float(overall_mean),
            pytorch_inference_time=float(np.mean(pytorch_times))
            if pytorch_times
            else None,
            mlx_inference_time=float(np.mean(mlx_times)) if mlx_times else None,
        )

    def _run_pytorch(
        self,
        model: Any,
        inputs: dict[str, Any],
        converter: Callable[[dict[str, Any], str], dict[str, Any]] | None,
    ) -> Any:
        """Run PyTorch model inference."""
        import torch

        model.eval()

        # Convert inputs if converter provided
        if converter:
            inputs = converter(inputs, "pytorch")

        # Convert to tensors if needed
        pt_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, np.ndarray):
                pt_inputs[k] = torch.from_numpy(v)
            elif isinstance(v, torch.Tensor):
                pt_inputs[k] = v
            else:
                pt_inputs[k] = v

        with torch.no_grad():
            if len(pt_inputs) == 1:
                # Single input
                output = model(list(pt_inputs.values())[0])
            else:
                output = model(**pt_inputs)

        return output

    def _run_mlx(
        self,
        model: Any,
        inputs: dict[str, Any],
        converter: Callable[[dict[str, Any], str], dict[str, Any]] | None,
    ) -> Any:
        """Run MLX model inference."""
        import mlx.core as mx

        # Convert inputs if converter provided
        if converter:
            inputs = converter(inputs, "mlx")

        # Convert to MLX arrays if needed
        mlx_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, np.ndarray):
                mlx_inputs[k] = mx.array(v)
            elif hasattr(v, "__mlx_array__"):
                mlx_inputs[k] = v
            else:
                mlx_inputs[k] = v

        if len(mlx_inputs) == 1:
            output = model(list(mlx_inputs.values())[0])
        else:
            output = model(**mlx_inputs)

        # Evaluate lazy computation
        mx.eval(output)

        return output

    def _compare_outputs(
        self,
        pt_output: Any,
        mlx_output: Any,
        prefix: str = "output",
    ) -> list[TensorComparison]:
        """Compare PyTorch and MLX outputs recursively."""
        import mlx.core as mx
        import torch

        comparisons = []

        # Handle different output types
        if isinstance(pt_output, (tuple, list)):
            for i, (pt, mlx) in enumerate(zip(pt_output, mlx_output, strict=False)):
                comparisons.extend(self._compare_outputs(pt, mlx, f"{prefix}_{i}"))
        elif isinstance(pt_output, dict):
            for key in pt_output:
                comparisons.extend(
                    self._compare_outputs(
                        pt_output[key], mlx_output[key], f"{prefix}_{key}",
                    ),
                )
        elif isinstance(pt_output, torch.Tensor):
            pt_arr = pt_output.detach().cpu().numpy()
            if isinstance(mlx_output, mx.array):
                mlx_arr = np.array(mlx_output)
            else:
                mlx_arr = mlx_output

            comparison = self._compare_arrays(prefix, pt_arr, mlx_arr)
            comparisons.append(comparison)
        elif hasattr(pt_output, "__mlx_array__") or isinstance(pt_output, np.ndarray):
            pt_arr = np.array(pt_output)
            mlx_arr = np.array(mlx_output)
            comparison = self._compare_arrays(prefix, pt_arr, mlx_arr)
            comparisons.append(comparison)

        return comparisons

    def _compare_arrays(
        self,
        name: str,
        pt_arr: np.ndarray,
        mlx_arr: np.ndarray,
    ) -> TensorComparison:
        """Compare two numpy arrays."""
        # Check shapes
        shape_match = pt_arr.shape == mlx_arr.shape

        # Check dtypes
        dtype_match = pt_arr.dtype == mlx_arr.dtype

        # Cast to float64 for comparison
        pt_float = pt_arr.astype(np.float64)
        mlx_float = mlx_arr.astype(np.float64)

        # Handle shape mismatch
        if not shape_match:
            return TensorComparison(
                name=name,
                shape_match=False,
                dtype_match=dtype_match,
                max_abs_error=float("inf"),
                mean_abs_error=float("inf"),
                max_rel_error=float("inf"),
                mean_rel_error=float("inf"),
                num_mismatches=pt_arr.size,
                total_elements=pt_arr.size,
                pytorch_dtype=str(pt_arr.dtype),
                mlx_dtype=str(mlx_arr.dtype),
                pytorch_shape=pt_arr.shape,
                mlx_shape=mlx_arr.shape,
            )

        # Compute absolute errors
        abs_diff = np.abs(pt_float - mlx_float)
        max_abs_error = float(np.max(abs_diff))
        mean_abs_error = float(np.mean(abs_diff))

        # Compute relative errors (avoid division by zero)
        abs_pt = np.abs(pt_float)
        rel_diff = abs_diff / np.maximum(abs_pt, 1e-10)
        max_rel_error = float(np.max(rel_diff))
        mean_rel_error = float(np.mean(rel_diff))

        # Count mismatches
        mismatches = np.sum(abs_diff > self.atol + self.rtol * abs_pt)

        return TensorComparison(
            name=name,
            shape_match=shape_match,
            dtype_match=dtype_match,
            max_abs_error=max_abs_error,
            mean_abs_error=mean_abs_error,
            max_rel_error=max_rel_error,
            mean_rel_error=mean_rel_error,
            num_mismatches=int(mismatches),
            total_elements=pt_arr.size,
            pytorch_dtype=str(pt_arr.dtype),
            mlx_dtype=str(mlx_arr.dtype),
            pytorch_shape=pt_arr.shape,
            mlx_shape=mlx_arr.shape,
        )

    def _is_passing(self, comparison: TensorComparison) -> bool:
        """Check if a comparison passes validation criteria."""
        if not comparison.shape_match:
            return False

        # Check tolerance
        if comparison.max_abs_error > self.atol:
            # Check if it passes with relative tolerance
            if comparison.max_rel_error > self.rtol:
                return False

        # Check match percentage
        match_ratio = 1 - (comparison.num_mismatches / comparison.total_elements)
        if match_ratio < self.match_threshold:
            return False

        return True

    def generate_report(self, report: ValidationReport) -> str:
        """Generate human-readable validation report."""
        lines = [
            "=" * 60,
            "VALIDATION REPORT",
            "=" * 60,
            "",
            f"Status: {report.status.value.upper()}",
            f"Outputs compared: {report.num_outputs}",
            f"Passed: {report.passed}",
            f"Failed: {report.failed}",
            "",
            f"Overall max error: {report.overall_max_error:.2e}",
            f"Overall mean error: {report.overall_mean_error:.2e}",
            "",
        ]

        if report.pytorch_inference_time and report.mlx_inference_time:
            speedup = report.pytorch_inference_time / report.mlx_inference_time
            lines.extend(
                [
                    "Inference Time:",
                    f"  PyTorch: {report.pytorch_inference_time * 1000:.2f} ms",
                    f"  MLX:     {report.mlx_inference_time * 1000:.2f} ms",
                    f"  Speedup: {speedup:.2f}x",
                    "",
                ],
            )

        if report.error_message:
            lines.extend(
                [
                    "ERROR:",
                    f"  {report.error_message}",
                    "",
                ],
            )

        # Details for each comparison
        lines.append("Output Details:")
        lines.append("-" * 60)

        for c in report.comparisons:
            status = "PASS" if self._is_passing(c) else "FAIL"
            lines.extend(
                [
                    f"  {c.name}: {status}",
                    f"    Shape: {c.pytorch_shape} -> {c.mlx_shape} ({'OK' if c.shape_match else 'MISMATCH'})",
                    f"    Dtype: {c.pytorch_dtype} -> {c.mlx_dtype}",
                    f"    Max abs error: {c.max_abs_error:.2e}",
                    f"    Mean abs error: {c.mean_abs_error:.2e}",
                    f"    Mismatches: {c.num_mismatches}/{c.total_elements}",
                    "",
                ],
            )

        lines.append("=" * 60)

        return "\n".join(lines)

    @staticmethod
    def create_test_inputs(
        shapes: list[tuple[int, ...]],
        dtype: str = "float32",
        seed: int = 42,
    ) -> list[dict[str, np.ndarray]]:
        """
        Create random test inputs with given shapes.

        Args:
            shapes: List of input shapes
            dtype: Data type for inputs
            seed: Random seed for reproducibility

        Returns:
            List of input dictionaries
        """
        rng = np.random.default_rng(seed)

        dtype_map = {
            "float32": np.float32,
            "float16": np.float16,
            "int32": np.int32,
            "int64": np.int64,
        }
        np_dtype = dtype_map.get(dtype, np.float32)

        inputs = []
        for shape in shapes:
            if np.issubdtype(np_dtype, np.floating):
                arr = rng.standard_normal(shape).astype(np_dtype)
            else:
                arr = rng.integers(0, 100, size=shape, dtype=np_dtype)
            inputs.append({"input": arr})

        return inputs
