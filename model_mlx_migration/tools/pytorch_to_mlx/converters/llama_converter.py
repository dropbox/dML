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
LLaMA Model Converter

Converts LLaMA models to MLX format using mlx-lm infrastructure.
Provides validation and benchmarking against HuggingFace transformers.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import mlx.core as mx
    import mlx.nn as nn  # noqa: F401
    from mlx_lm import batch_generate, convert, generate, load, stream_generate
    from mlx_lm.utils import TokenizerWrapper  # noqa: F401

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

import numpy as np


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
    max_abs_error: float
    mean_abs_error: float
    tokens_compared: int
    mismatched_tokens: int
    error: str | None = None


@dataclass
class BenchmarkResult:
    """Benchmark comparison result."""

    mlx_tokens_per_second: float
    pytorch_tokens_per_second: float
    speedup: float
    mlx_time_to_first_token_ms: float
    pytorch_time_to_first_token_ms: float


@dataclass
class BatchGenerationResult:
    """Result of batch generation."""

    texts: list[str]
    total_tokens: int
    total_time_ms: float
    throughput_tokens_per_second: float


@dataclass
class SpeculativeGenerationResult:
    """Result of speculative decoding."""

    text: str
    total_tokens: int
    total_time_ms: float
    tokens_per_second: float
    acceptance_rate: float | None = None


class LLaMAConverter:
    """
    Converts LLaMA models to MLX format using mlx-lm.

    Supports:
    - HuggingFace model conversion (meta-llama/Llama-3-8B, etc.)
    - Validation of numerical equivalence
    - Performance benchmarking

    Example:
        converter = LLaMAConverter()
        result = converter.convert("meta-llama/Llama-3-8B", "./mlx-llama")
        if result.success:
            validation = converter.validate("./mlx-llama", test_prompts)
    """

    def __init__(self):
        """Initialize converter."""
        if not MLX_AVAILABLE:
            raise ImportError("mlx-lm is required. Install with: pip install mlx-lm")

    def convert(
        self,
        hf_path: str,
        output_path: str,
        quantize: bool = False,
        q_bits: int = 4,
        dtype: str | None = None,
    ) -> ConversionResult:
        """
        Convert a HuggingFace LLaMA model to MLX format.

        Args:
            hf_path: HuggingFace model path (e.g., "meta-llama/Llama-3-8B")
            output_path: Directory to save converted model
            quantize: Whether to quantize the model
            q_bits: Quantization bits (4 or 8)
            dtype: Data type for weights (float16, bfloat16, float32)

        Returns:
            ConversionResult with conversion status and metadata
        """
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Use mlx-lm convert function
            convert(
                hf_path=hf_path,
                mlx_path=str(output_dir),
                quantize=quantize,
                q_bits=q_bits,
                dtype=dtype,
            )

            # Load to get model info
            model, tokenizer = load(str(output_dir))  # type: ignore[misc]

            # Count parameters
            num_params = sum(p.size for p in model.parameters().values())

            # Estimate size
            weights_file = output_dir / "model.safetensors"
            if not weights_file.exists():
                # Check for quantized weights
                weights_file = output_dir / "weights.npz"

            if weights_file.exists():
                model_size_mb = weights_file.stat().st_size / (1024 * 1024)
            else:
                model_size_mb = num_params * 2 / (1024 * 1024)  # Estimate for fp16

            return ConversionResult(
                success=True,
                mlx_path=str(output_dir),
                model_size_mb=model_size_mb,
                num_parameters=num_params,
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
        mlx_path: str,
        test_prompts: list[str],
        hf_path: str | None = None,
        max_tokens: int = 50,
        tolerance: float = 1e-3,
    ) -> ValidationResult:
        """
        Validate MLX model output against HuggingFace transformers.

        Compares logits/probabilities between MLX and PyTorch implementations.

        Args:
            mlx_path: Path to MLX converted model
            test_prompts: List of prompts to test
            hf_path: HuggingFace model path (inferred from mlx_path if not provided)
            max_tokens: Maximum tokens to generate
            tolerance: Acceptable numerical difference

        Returns:
            ValidationResult with comparison metrics
        """
        if not TRANSFORMERS_AVAILABLE:
            return ValidationResult(
                passed=False,
                max_abs_error=float("inf"),
                mean_abs_error=float("inf"),
                tokens_compared=0,
                mismatched_tokens=0,
                error="transformers not installed. Install with: pip install transformers",
            )

        try:
            # Load MLX model
            mlx_model, mlx_tokenizer = load(mlx_path)  # type: ignore[misc]

            # Infer HF path from config if not provided
            if hf_path is None:
                config_path = Path(mlx_path) / "config.json"
                if config_path.exists():
                    import json

                    with open(config_path) as f:
                        config = json.load(f)
                    hf_path = config.get("_name_or_path", None)

            if hf_path is None:
                return ValidationResult(
                    passed=False,
                    max_abs_error=float("inf"),
                    mean_abs_error=float("inf"),
                    tokens_compared=0,
                    mismatched_tokens=0,
                    error="Could not determine HuggingFace model path",
                )

            # Load PyTorch model
            pt_tokenizer = AutoTokenizer.from_pretrained(hf_path)
            # Use float32 for validation accuracy, device_map="auto" for Apple Silicon
            pt_model = AutoModelForCausalLM.from_pretrained(hf_path)
            pt_model = pt_model.to(torch.float32)  # type: ignore[arg-type]
            # Move to MPS if available
            if torch.backends.mps.is_available():
                pt_model = pt_model.to("mps")
            pt_model.eval()

            all_errors = []
            tokens_compared = 0
            mismatched = 0

            for prompt in test_prompts:
                # Get MLX output
                mlx_tokens = mlx_tokenizer.encode(prompt)
                mlx_input = mx.array(mlx_tokens)[None]
                mlx_logits = mlx_model(mlx_input)
                mx.eval(mlx_logits)
                mlx_probs = mx.softmax(mlx_logits, axis=-1)
                # Convert to float32 for numpy compatibility (bfloat16 doesn't convert directly)
                mlx_probs_f32 = mlx_probs.astype(mx.float32)
                mx.eval(mlx_probs_f32)
                mlx_np = np.array(mlx_probs_f32)

                # Get PyTorch output
                pt_inputs = pt_tokenizer(prompt, return_tensors="pt").to(
                    pt_model.device,
                )
                with torch.no_grad():
                    pt_outputs = pt_model(**pt_inputs)
                    pt_logits = pt_outputs.logits
                    pt_probs = torch.softmax(pt_logits, dim=-1)
                pt_np = pt_probs.cpu().numpy()

                # Compare shapes
                if mlx_np.shape != pt_np.shape:
                    # Pad/truncate to match
                    min_seq = min(mlx_np.shape[1], pt_np.shape[1])
                    mlx_np = mlx_np[:, :min_seq, :]
                    pt_np = pt_np[:, :min_seq, :]

                # Compare probabilities
                abs_diff = np.abs(mlx_np - pt_np)
                all_errors.extend(abs_diff.flatten().tolist())
                tokens_compared += mlx_np.shape[1]

                # Count mismatched token predictions
                mlx_preds = np.argmax(mlx_np, axis=-1)
                pt_preds = np.argmax(pt_np, axis=-1)
                mismatched += int(np.sum(mlx_preds != pt_preds))

            max_error = float(max(all_errors)) if all_errors else 0.0
            mean_error = float(np.mean(all_errors)) if all_errors else 0.0

            return ValidationResult(
                passed=max_error < tolerance,
                max_abs_error=max_error,
                mean_abs_error=mean_error,
                tokens_compared=tokens_compared,
                mismatched_tokens=mismatched,
            )

        except Exception as e:
            return ValidationResult(
                passed=False,
                max_abs_error=float("inf"),
                mean_abs_error=float("inf"),
                tokens_compared=0,
                mismatched_tokens=0,
                error=str(e),
            )

    def benchmark(
        self,
        mlx_path: str,
        test_prompts: list[str],
        hf_path: str | None = None,
        max_tokens: int = 100,
        runs: int = 3,
    ) -> BenchmarkResult:
        """
        Benchmark MLX model against HuggingFace transformers.

        Args:
            mlx_path: Path to MLX converted model
            test_prompts: List of prompts to benchmark
            hf_path: HuggingFace model path
            max_tokens: Tokens to generate per prompt
            runs: Number of runs for averaging

        Returns:
            BenchmarkResult with performance comparison
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers required for benchmarking")

        # Load MLX model
        mlx_model, mlx_tokenizer = load(mlx_path)  # type: ignore[misc]

        # Load PyTorch model
        if hf_path is None:
            config_path = Path(mlx_path) / "config.json"
            if config_path.exists():
                import json

                with open(config_path) as f:
                    config = json.load(f)
                hf_path = config.get("_name_or_path")

        pt_tokenizer = AutoTokenizer.from_pretrained(hf_path)  # type: ignore[arg-type]
        pt_model = AutoModelForCausalLM.from_pretrained(
            hf_path,  # type: ignore[arg-type]
            torch_dtype=torch.float16,
            device_map="auto",
        )
        pt_model.eval()

        mlx_times = []
        pt_times = []
        mlx_ttft = []
        pt_ttft = []

        for prompt in test_prompts:
            for _ in range(runs):
                # MLX benchmark
                start = time.perf_counter()
                generate(
                    mlx_model,
                    mlx_tokenizer,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    verbose=False,
                )
                mlx_time = time.perf_counter() - start
                mlx_times.append(mlx_time)
                # Estimate TTFT (first token is roughly 1/max_tokens of total)
                mlx_ttft.append(mlx_time / max_tokens)

                # PyTorch benchmark
                pt_inputs = pt_tokenizer(prompt, return_tensors="pt").to(
                    pt_model.device,
                )
                start = time.perf_counter()
                with torch.no_grad():
                    pt_model.generate(
                        **pt_inputs, max_new_tokens=max_tokens, do_sample=False,
                    )
                pt_time = time.perf_counter() - start
                pt_times.append(pt_time)
                pt_ttft.append(pt_time / max_tokens)

        # Calculate metrics
        mlx_total_tokens = len(test_prompts) * runs * max_tokens
        pt_total_tokens = len(test_prompts) * runs * max_tokens

        mlx_tps = mlx_total_tokens / sum(mlx_times)
        pt_tps = pt_total_tokens / sum(pt_times)

        return BenchmarkResult(
            mlx_tokens_per_second=mlx_tps,
            pytorch_tokens_per_second=pt_tps,
            speedup=mlx_tps / pt_tps if pt_tps > 0 else 0,
            mlx_time_to_first_token_ms=float(np.mean(mlx_ttft) * 1000),
            pytorch_time_to_first_token_ms=float(np.mean(pt_ttft) * 1000),
        )

    def load_mlx_model(self, mlx_path: str) -> tuple[Any, Any]:
        """
        Load a converted MLX model.

        Args:
            mlx_path: Path to MLX model directory

        Returns:
            Tuple of (model, tokenizer)
        """
        return load(mlx_path)  # type: ignore[return-value]

    def generate_batch(
        self,
        mlx_path: str,
        prompts: list[str],
        max_tokens: int = 256,
        **kwargs,
    ) -> BatchGenerationResult:
        """
        Generate responses for multiple prompts using mlx_lm.batch_generate.

        2-4x throughput improvement over sequential generation.

        Args:
            mlx_path: Path to MLX model directory
            prompts: List of text prompts to process
            max_tokens: Maximum tokens to generate per prompt
            **kwargs: Additional arguments passed to batch_generate

        Returns:
            BatchGenerationResult with texts and performance metrics
        """
        model, tokenizer = load(mlx_path)

        # Tokenize all prompts
        tokenized = [tokenizer.encode(p) for p in prompts]

        start = time.perf_counter()
        response = batch_generate(
            model,
            tokenizer,
            prompts=tokenized,
            max_tokens=max_tokens,
            **kwargs,
        )
        elapsed = time.perf_counter() - start

        # Extract texts from response
        texts = [r.text for r in response.responses]

        # Count total tokens generated
        total_tokens = sum(len(tokenizer.encode(t)) for t in texts)

        return BatchGenerationResult(
            texts=texts,
            total_tokens=total_tokens,
            total_time_ms=elapsed * 1000,
            throughput_tokens_per_second=total_tokens / elapsed if elapsed > 0 else 0,
        )

    def generate_speculative(
        self,
        mlx_path: str,
        prompt: str,
        draft_model_path: str = "mlx-community/Llama-3.2-1B-4bit",
        max_tokens: int = 256,
        **kwargs,
    ) -> SpeculativeGenerationResult:
        """
        Generate with speculative decoding for 1.5-2x faster inference.

        Uses small draft model to propose tokens, main model verifies.
        Zero quality loss - output is identical to non-speculative generation.

        Args:
            mlx_path: Path to main MLX model directory
            prompt: Text prompt
            draft_model_path: Path to smaller draft model (HuggingFace or local)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments passed to stream_generate

        Returns:
            SpeculativeGenerationResult with text and performance metrics
        """
        # Load main model
        model, tokenizer = load(mlx_path)

        # Load draft model
        draft_model, _ = load(draft_model_path)

        start = time.perf_counter()
        result_text = ""
        token_count = 0

        for response in stream_generate(
            model,
            tokenizer,
            prompt,
            max_tokens=max_tokens,
            draft_model=draft_model,
            **kwargs,
        ):
            result_text = response.text
            token_count = response.token_count

        elapsed = time.perf_counter() - start

        return SpeculativeGenerationResult(
            text=result_text,
            total_tokens=token_count,
            total_time_ms=elapsed * 1000,
            tokens_per_second=token_count / elapsed if elapsed > 0 else 0,
        )

    def benchmark_optimizations(
        self,
        mlx_path: str,
        prompts: list[str],
        draft_model_path: str | None = None,
        max_tokens: int = 100,
        runs: int = 3,
    ) -> dict:
        """
        Benchmark sequential vs batch vs speculative generation.

        Args:
            mlx_path: Path to MLX model directory
            prompts: List of prompts to test
            draft_model_path: Path to draft model for speculative decoding
            max_tokens: Tokens to generate per prompt
            runs: Number of runs for averaging

        Returns:
            Dict with benchmark results for each method
        """
        model, tokenizer = load(mlx_path)
        results = {}

        # Sequential baseline
        seq_times = []
        for _ in range(runs):
            start = time.perf_counter()
            for prompt in prompts:
                generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)
            seq_times.append(time.perf_counter() - start)

        seq_avg = sum(seq_times) / len(seq_times)
        results["sequential"] = {
            "avg_time_ms": seq_avg * 1000,
            "throughput_tokens_per_second": (len(prompts) * max_tokens) / seq_avg,
        }

        # Batch generation
        batch_times = []
        for _ in range(runs):
            tokenized = [tokenizer.encode(p) for p in prompts]
            start = time.perf_counter()
            batch_generate(model, tokenizer, prompts=tokenized, max_tokens=max_tokens)
            batch_times.append(time.perf_counter() - start)

        batch_avg = sum(batch_times) / len(batch_times)
        results["batch"] = {
            "avg_time_ms": batch_avg * 1000,
            "throughput_tokens_per_second": (len(prompts) * max_tokens) / batch_avg,
            "speedup_vs_sequential": seq_avg / batch_avg if batch_avg > 0 else 0,
        }

        # Speculative decoding (single prompt, latency focus)
        if draft_model_path:
            draft_model, _ = load(draft_model_path)
            spec_times = []
            for _ in range(runs):
                start = time.perf_counter()
                for _response in stream_generate(
                    model,
                    tokenizer,
                    prompts[0],
                    max_tokens=max_tokens,
                    draft_model=draft_model,
                ):
                    pass  # Consume generator
                spec_times.append(time.perf_counter() - start)

            spec_avg = sum(spec_times) / len(spec_times)

            # Compare to single prompt sequential
            single_times = []
            for _ in range(runs):
                start = time.perf_counter()
                generate(model, tokenizer, prompt=prompts[0], max_tokens=max_tokens)
                single_times.append(time.perf_counter() - start)
            single_avg = sum(single_times) / len(single_times)

            results["speculative"] = {
                "avg_time_ms": spec_avg * 1000,
                "tokens_per_second": max_tokens / spec_avg if spec_avg > 0 else 0,
                "speedup_vs_sequential": single_avg / spec_avg if spec_avg > 0 else 0,
            }

        return results

    @staticmethod
    def list_supported_models() -> list[str]:
        """
        List known supported LLaMA model variants.

        Returns:
            List of HuggingFace model paths
        """
        return [
            "meta-llama/Llama-2-7b-hf",
            "meta-llama/Llama-2-13b-hf",
            "meta-llama/Llama-2-70b-hf",
            "meta-llama/Llama-3-8B",
            "meta-llama/Llama-3-70B",
            "meta-llama/Llama-3.1-8B",
            "meta-llama/Llama-3.1-70B",
            "meta-llama/Llama-3.2-1B",
            "meta-llama/Llama-3.2-3B",
            "mistralai/Mistral-7B-v0.1",
            "mistralai/Mixtral-8x7B-v0.1",
        ]
