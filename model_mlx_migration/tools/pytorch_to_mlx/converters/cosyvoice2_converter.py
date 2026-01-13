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
CosyVoice2 TTS Model Converter

Converts CosyVoice2-0.5B (FunAudioLLM/CosyVoice2-0.5B) to MLX format.
Provides validation and benchmarking against PyTorch reference.

Architecture:
- LLM: Qwen2-based text encoder and token generator
- Flow: MaskedDiffWithXvec flow matching model
- Vocoder: HiFi-GAN for mel-to-waveform

Files:
- llm.pt: LLM component weights
- flow.pt: Flow matching model weights
- hift.pt: HiFi-GAN vocoder weights
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import mlx.core as mx  # noqa: F401
    import mlx.nn as nn  # noqa: F401

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class ConversionResult:
    """Result of model conversion."""

    success: bool
    mlx_path: str
    model_size_mb: float
    num_parameters: int
    error: str | None = None


@dataclass
class ValidationResult:
    """Result of output validation between PyTorch and MLX."""

    passed: bool
    llm_max_error: float
    flow_max_error: float
    vocoder_max_error: float
    error: str | None = None


@dataclass
class BenchmarkResult:
    """Benchmark comparison result."""

    mlx_audio_per_second: float
    pytorch_audio_per_second: float
    speedup: float
    first_token_latency_ms: float


class CosyVoice2Converter:
    """
    Converts CosyVoice2 TTS model to MLX format.

    Supports:
    - ModelScope/HuggingFace model conversion
    - Validation of numerical equivalence
    - Performance benchmarking
    - Streaming inference

    Example:
        converter = CosyVoice2Converter()
        result = converter.convert("FunAudioLLM/CosyVoice2-0.5B", "./mlx-cosyvoice2")
        if result.success:
            validation = converter.validate("./mlx-cosyvoice2")
    """

    SUPPORTED_MODELS = [
        "FunAudioLLM/CosyVoice2-0.5B",
        "FunAudioLLM/CosyVoice-300M",
    ]

    # Expected model files
    MODEL_FILES = {
        "llm": "llm.pt",
        "flow": "flow.pt",
        "vocoder": "hift.pt",
    }

    def __init__(self) -> None:
        """Initialize converter."""
        if not MLX_AVAILABLE:
            raise ImportError("MLX is required. Install with: pip install mlx")

    @staticmethod
    def list_supported_models() -> list[str]:
        """Return list of supported model IDs."""
        return CosyVoice2Converter.SUPPORTED_MODELS.copy()

    def locate_model_files(
        self,
        model_dir: Path,
    ) -> dict[str, Path]:
        """
        Locate model files in directory.

        Args:
            model_dir: Directory containing model files

        Returns:
            Dict mapping component name to file path
        """
        model_dir = Path(model_dir)
        files = {}

        for component, filename in self.MODEL_FILES.items():
            file_path = model_dir / filename
            if file_path.exists():
                files[component] = file_path
            else:
                print(f"Warning: {filename} not found in {model_dir}")

        return files

    def inspect_model(
        self,
        model_dir: str,
    ) -> dict[str, Any]:
        """
        Inspect model structure without full conversion.

        Args:
            model_dir: Path to model directory

        Returns:
            Dict with model inspection results
        """
        if not TORCH_AVAILABLE:
            raise ImportError("torch is required for inspection")

        model_path = Path(model_dir)
        files = self.locate_model_files(model_path)

        results = {}

        for component, file_path in files.items():
            print(f"\nInspecting {component} ({file_path.name})...")

            try:
                data = torch.load(file_path, map_location="cpu", weights_only=False)

                component_info = {
                    "file": str(file_path),
                    "file_size_mb": file_path.stat().st_size / (1024 * 1024),
                }

                if isinstance(data, dict):
                    # State dict
                    component_info["type"] = "state_dict"
                    component_info["num_keys"] = len(data)

                    # Get parameter shapes
                    shapes = {}
                    total_params = 0
                    for k, v in data.items():
                        if hasattr(v, "shape"):
                            shapes[k] = list(v.shape)
                            total_params += v.numel()

                    component_info["total_params"] = total_params
                    component_info["sample_keys"] = list(shapes.keys())[:20]
                    component_info["sample_shapes"] = {
                        k: shapes[k] for k in list(shapes.keys())[:20]
                    }

                elif hasattr(data, "state_dict"):
                    # Full model
                    sd = data.state_dict()
                    component_info["type"] = "model"
                    component_info["model_class"] = type(data).__name__
                    component_info["num_keys"] = len(sd)

                    total_params = sum(
                        p.numel() for p in sd.values() if hasattr(p, "numel")
                    )
                    component_info["total_params"] = total_params

                else:
                    component_info["type"] = type(data).__name__

                results[component] = component_info

            except Exception as e:
                results[component] = {"error": str(e)}

        return results

    def convert(
        self,
        model_id_or_path: str,
        output_path: str,
        cache_dir: str | None = None,
    ) -> ConversionResult:
        """
        Convert CosyVoice2 model to MLX format.

        Args:
            model_id_or_path: Model ID or local path
            output_path: Output directory for MLX model
            cache_dir: Cache directory for downloads

        Returns:
            ConversionResult with conversion status
        """
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Determine if path or model ID
        model_dir = Path(model_id_or_path)
        if not model_dir.exists():
            # Try to download
            return ConversionResult(
                success=False,
                mlx_path=str(output_dir),
                model_size_mb=0,
                num_parameters=0,
                error=f"Model not found at {model_id_or_path}. "
                "Use scripts/download_cosyvoice2.py to download.",
            )

        # Locate model files
        files = self.locate_model_files(model_dir)
        if not files:
            return ConversionResult(
                success=False,
                mlx_path=str(output_dir),
                model_size_mb=0,
                num_parameters=0,
                error=f"No model files found in {model_dir}",
            )

        # Conversion not yet implemented via this API.
        # MLX weights must be prepared separately using the validation scripts.
        try:
            inspection = self.inspect_model(str(model_dir))

            total_params = sum(
                info.get("total_params", 0)
                for info in inspection.values()
                if isinstance(info, dict)
            )

            total_size = sum(
                info.get("file_size_mb", 0)
                for info in inspection.values()
                if isinstance(info, dict)
            )

            # Save inspection results
            with open(output_dir / "inspection.json", "w") as f:
                json.dump(inspection, f, indent=2, default=str)

            return ConversionResult(
                success=False,
                mlx_path=str(output_dir),
                model_size_mb=total_size,
                num_parameters=total_params,
                error=(
                    "Automated conversion not yet implemented. "
                    "Use 'cosyvoice2 inspect' to examine model structure. "
                    "For synthesis, use 'cosyvoice2 synthesize' with pre-converted weights. "
                    "See scripts/download_cosyvoice2.py and scripts/validate_cosyvoice2_*.py "
                    "for manual conversion workflow."
                ),
            )

        except Exception as e:
            return ConversionResult(
                success=False,
                mlx_path=str(output_dir),
                model_size_mb=0,
                num_parameters=0,
                error=str(e),
            )

    def validate(
        self,
        mlx_model_path: str,
        pytorch_model_path: str | None = None,
        tolerance: float = 1e-3,
    ) -> ValidationResult:
        """
        Validate MLX model against PyTorch reference.

        Args:
            mlx_model_path: Path to MLX model
            pytorch_model_path: Path to PyTorch model (optional)
            tolerance: Numerical tolerance for comparison

        Returns:
            ValidationResult with validation status
        """
        # API stub - params reserved for future implementation
        del mlx_model_path, pytorch_model_path, tolerance
        # Validation via this API is not yet implemented.
        # Use the validation scripts for component-level validation.
        return ValidationResult(
            passed=False,
            llm_max_error=0.0,
            flow_max_error=0.0,
            vocoder_max_error=0.0,
            error=(
                "Validation via CLI not yet implemented. "
                "Use validation scripts directly: "
                "scripts/validate_cosyvoice2_llm.py, "
                "scripts/validate_cosyvoice2_flow.py, "
                "scripts/validate_cosyvoice2_vocoder.py, "
                "scripts/validate_cosyvoice2_full.py"
            ),
        )

    def benchmark(
        self,
        mlx_model_path: str,
        pytorch_model_path: str | None = None,
        num_runs: int = 10,
        warmup: int = 3,
    ) -> BenchmarkResult:
        """
        Benchmark MLX model performance.

        Args:
            mlx_model_path: Path to MLX model
            pytorch_model_path: Path to PyTorch model for comparison
            num_runs: Number of benchmark runs
            warmup: Number of warmup runs

        Returns:
            BenchmarkResult with performance metrics
        """
        # API stub - params reserved for future implementation
        del mlx_model_path, pytorch_model_path, num_runs, warmup
        # Benchmarking via this API is not yet implemented.
        # Use scripts/benchmark_cosyvoice2.py for performance measurements.
        return BenchmarkResult(
            mlx_audio_per_second=0.0,
            pytorch_audio_per_second=0.0,
            speedup=0.0,
            first_token_latency_ms=0.0,
        )


def print_inspection_report(inspection: dict[str, Any]) -> None:
    """Print a formatted inspection report."""
    print("\n" + "=" * 60)
    print("CosyVoice2 Model Inspection Report")
    print("=" * 60)

    for component, info in inspection.items():
        print(f"\n{component.upper()}")
        print("-" * 40)

        if "error" in info:
            print(f"  Error: {info['error']}")
            continue

        print(f"  File: {info.get('file', 'N/A')}")
        print(f"  Size: {info.get('file_size_mb', 0):.1f} MB")
        print(f"  Type: {info.get('type', 'N/A')}")
        print(f"  Parameters: {info.get('total_params', 0):,}")
        print(f"  Keys: {info.get('num_keys', 0)}")

        if "sample_shapes" in info:
            print("  Sample weights:")
            for k, shape in list(info["sample_shapes"].items())[:5]:
                print(f"    {k}: {shape}")
            if len(info["sample_shapes"]) > 5:
                print(f"    ... and {len(info['sample_shapes']) - 5} more")

    print("\n" + "=" * 60)
