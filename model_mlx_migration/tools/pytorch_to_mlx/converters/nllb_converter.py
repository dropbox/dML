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
NLLB Model Converter

Converts NLLB (facebook/nllb-200-*) models to MLX format.
Provides validation and benchmarking against HuggingFace transformers.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import mlx.core as mx

    from .models.nllb import NLLBModel

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

try:
    import torch
    from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


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
    encoder_max_error: float
    decoder_max_error: float
    top_token_match: bool
    error: str | None = None


@dataclass
class BenchmarkResult:
    """Benchmark comparison result."""

    mlx_tokens_per_second: float
    pytorch_tokens_per_second: float
    speedup: float
    mlx_encode_time_ms: float
    pytorch_encode_time_ms: float


class NLLBConverter:
    """
    Converts NLLB models to MLX format.

    Supports:
    - HuggingFace model conversion (facebook/nllb-200-*)
    - Validation of numerical equivalence
    - Performance benchmarking
    - Translation

    Example:
        converter = NLLBConverter()
        result = converter.convert("facebook/nllb-200-distilled-600M", "./mlx-nllb")
        if result.success:
            validation = converter.validate("./mlx-nllb")
    """

    def __init__(self):
        """Initialize converter."""
        if not MLX_AVAILABLE:
            raise ImportError("MLX is required. Install with: pip install mlx")

    def convert(
        self,
        hf_path: str,
        output_path: str,
    ) -> ConversionResult:
        """
        Convert a HuggingFace NLLB model to MLX format.

        Args:
            hf_path: HuggingFace model path (e.g., "facebook/nllb-200-distilled-600M")
            output_path: Directory to save converted model

        Returns:
            ConversionResult with conversion status and metadata
        """
        if not TRANSFORMERS_AVAILABLE:
            return ConversionResult(
                success=False,
                mlx_path=output_path,
                model_size_mb=0,
                num_parameters=0,
                error="transformers not installed",
            )

        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Load HuggingFace config
            AutoConfig.from_pretrained(hf_path)

            # Load and convert model
            print(f"Loading {hf_path}...")
            model = NLLBModel.from_hf(hf_path)

            # Save config
            config_dict = {
                "vocab_size": model.config.vocab_size,
                "d_model": model.config.d_model,
                "encoder_layers": model.config.encoder_layers,
                "decoder_layers": model.config.decoder_layers,
                "encoder_attention_heads": model.config.encoder_attention_heads,
                "decoder_attention_heads": model.config.decoder_attention_heads,
                "encoder_ffn_dim": model.config.encoder_ffn_dim,
                "decoder_ffn_dim": model.config.decoder_ffn_dim,
                "activation_function": model.config.activation_function,
                "dropout": model.config.dropout,
                "max_position_embeddings": model.config.max_position_embeddings,
                "scale_embedding": model.config.scale_embedding,
                "pad_token_id": model.config.pad_token_id,
                "bos_token_id": model.config.bos_token_id,
                "eos_token_id": model.config.eos_token_id,
                "decoder_start_token_id": model.config.decoder_start_token_id,
                "_name_or_path": hf_path,
            }

            with open(output_dir / "config.json", "w") as f:
                json.dump(config_dict, f, indent=2)

            # Save weights
            from mlx.utils import tree_flatten

            weights = dict(tree_flatten(model.parameters()))
            mx.save_safetensors(str(output_dir / "weights.safetensors"), weights)

            # Copy tokenizer
            tokenizer = AutoTokenizer.from_pretrained(hf_path)
            tokenizer.save_pretrained(str(output_dir))

            # Calculate stats
            weights_file = output_dir / "weights.safetensors"
            model_size_mb = weights_file.stat().st_size / (1024 * 1024)
            num_params = sum(v.size for v in weights.values())

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
        test_texts: list[str] | None = None,
        hf_path: str | None = None,
        tolerance: float = 1e-4,
    ) -> ValidationResult:
        """
        Validate MLX model output against HuggingFace transformers.

        Args:
            mlx_path: Path to MLX converted model
            test_texts: List of texts to test
            hf_path: HuggingFace model path (inferred from config if not provided)
            tolerance: Acceptable numerical difference

        Returns:
            ValidationResult with comparison metrics
        """
        if not TRANSFORMERS_AVAILABLE:
            return ValidationResult(
                passed=False,
                encoder_max_error=float("inf"),
                decoder_max_error=float("inf"),
                top_token_match=False,
                error="transformers not installed",
            )

        if test_texts is None:
            test_texts = ["Hello, how are you?"]

        try:
            # Load config to get HF path
            config_path = Path(mlx_path) / "config.json"
            with open(config_path) as f:
                config = json.load(f)

            if hf_path is None:
                hf_path = config.get("_name_or_path")

            if hf_path is None:
                return ValidationResult(
                    passed=False,
                    encoder_max_error=float("inf"),
                    decoder_max_error=float("inf"),
                    top_token_match=False,
                    error="Could not determine HuggingFace model path",
                )

            # Load models
            hf_model = AutoModelForSeq2SeqLM.from_pretrained(hf_path)
            hf_model.eval()
            tokenizer = AutoTokenizer.from_pretrained(hf_path)

            mlx_model = NLLBModel.from_pretrained(mlx_path)

            all_encoder_errors = []
            all_decoder_errors = []
            all_top_match = []

            for text in test_texts:
                # Tokenize
                inputs = tokenizer(text, return_tensors="pt")
                input_ids_pt = inputs["input_ids"]
                input_ids_mlx = mx.array(input_ids_pt.numpy())

                # Encoder comparison
                with torch.no_grad():
                    pt_encoder_out = hf_model.model.encoder(
                        input_ids_pt,
                    ).last_hidden_state
                mlx_encoder_out = mlx_model.encode(input_ids_mlx)
                mx.eval(mlx_encoder_out)

                encoder_diff = np.abs(
                    pt_encoder_out.numpy() - np.array(mlx_encoder_out),
                )
                all_encoder_errors.append(encoder_diff.max())

                # Decoder comparison
                decoder_input_ids_pt = torch.tensor([[tokenizer.eos_token_id]])
                decoder_input_ids_mlx = mx.array([[tokenizer.eos_token_id]])

                with torch.no_grad():
                    pt_output = hf_model(
                        input_ids=input_ids_pt,
                        decoder_input_ids=decoder_input_ids_pt,
                    )
                pt_logits_np = pt_output.logits.numpy()

                mlx_logits, _ = mlx_model.decode(decoder_input_ids_mlx, mlx_encoder_out)
                mx.eval(mlx_logits)
                mlx_logits_np = np.array(mlx_logits)

                decoder_diff = np.abs(pt_logits_np - mlx_logits_np)
                all_decoder_errors.append(decoder_diff.max())

                # Top token comparison
                pt_top = np.argmax(pt_logits_np[0, 0])
                mlx_top = np.argmax(mlx_logits_np[0, 0])
                all_top_match.append(pt_top == mlx_top)

            encoder_max = max(all_encoder_errors)
            decoder_max = max(all_decoder_errors)
            top_match = all(all_top_match)

            return ValidationResult(
                passed=encoder_max < tolerance and top_match,
                encoder_max_error=encoder_max,
                decoder_max_error=decoder_max,
                top_token_match=top_match,
            )

        except Exception as e:
            return ValidationResult(
                passed=False,
                encoder_max_error=float("inf"),
                decoder_max_error=float("inf"),
                top_token_match=False,
                error=str(e),
            )

    def benchmark(
        self,
        mlx_path: str,
        test_texts: list[str] | None = None,
        hf_path: str | None = None,
        runs: int = 3,
    ) -> BenchmarkResult:
        """
        Benchmark MLX model against HuggingFace transformers.

        Args:
            mlx_path: Path to MLX converted model
            test_texts: List of texts to benchmark
            hf_path: HuggingFace model path
            runs: Number of runs for averaging

        Returns:
            BenchmarkResult with performance comparison
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers required for benchmarking")

        if test_texts is None:
            test_texts = [
                "Hello, how are you?",
                "Machine learning is transforming the world.",
            ]

        # Load config
        config_path = Path(mlx_path) / "config.json"
        with open(config_path) as f:
            config = json.load(f)

        if hf_path is None:
            hf_path = config.get("_name_or_path")

        # Load models
        hf_model = AutoModelForSeq2SeqLM.from_pretrained(hf_path)
        hf_model.eval()
        tokenizer = AutoTokenizer.from_pretrained(hf_path)

        mlx_model = NLLBModel.from_pretrained(mlx_path)

        mlx_encode_times = []
        pt_encode_times = []
        mlx_decode_times = []
        pt_decode_times = []

        for text in test_texts:
            inputs = tokenizer(text, return_tensors="pt")
            input_ids_pt = inputs["input_ids"]
            input_ids_mlx = mx.array(input_ids_pt.numpy())

            for _ in range(runs):
                # MLX encoding
                start = time.perf_counter()
                mlx_encoder_out = mlx_model.encode(input_ids_mlx)
                mx.eval(mlx_encoder_out)
                mlx_encode_times.append(time.perf_counter() - start)

                # PyTorch encoding
                start = time.perf_counter()
                with torch.no_grad():
                    hf_model.model.encoder(input_ids_pt)
                pt_encode_times.append(time.perf_counter() - start)

                # MLX decoding (10 tokens)
                decoder_ids_mlx = mx.array([[tokenizer.eos_token_id]])
                start = time.perf_counter()
                for _ in range(10):
                    logits, _ = mlx_model.decode(decoder_ids_mlx, mlx_encoder_out)
                    mx.eval(logits)
                mlx_decode_times.append(time.perf_counter() - start)

                # PyTorch decoding (10 tokens)
                decoder_ids_pt = torch.tensor([[tokenizer.eos_token_id]])
                start = time.perf_counter()
                with torch.no_grad():
                    for _ in range(10):
                        hf_model(
                            input_ids=input_ids_pt,
                            decoder_input_ids=decoder_ids_pt,
                        )
                pt_decode_times.append(time.perf_counter() - start)

        # Calculate metrics
        mlx_encode_avg = float(np.mean(mlx_encode_times) * 1000)
        pt_encode_avg = float(np.mean(pt_encode_times) * 1000)

        mlx_decode_tokens = 10 * len(test_texts) * runs
        pt_decode_tokens = 10 * len(test_texts) * runs

        mlx_tps = mlx_decode_tokens / sum(mlx_decode_times)
        pt_tps = pt_decode_tokens / sum(pt_decode_times)

        return BenchmarkResult(
            mlx_tokens_per_second=mlx_tps,
            pytorch_tokens_per_second=pt_tps,
            speedup=mlx_tps / pt_tps if pt_tps > 0 else 0,
            mlx_encode_time_ms=mlx_encode_avg,
            pytorch_encode_time_ms=pt_encode_avg,
        )

    def translate(
        self,
        mlx_path: str,
        text: str,
        src_lang: str = "eng_Latn",
        tgt_lang: str = "fra_Latn",
        max_tokens: int = 100,
    ) -> str:
        """
        Translate text using the MLX model.

        Args:
            mlx_path: Path to MLX model
            text: Text to translate
            src_lang: Source language code
            tgt_lang: Target language code
            max_tokens: Maximum tokens to generate

        Returns:
            Translated text
        """
        # Load model and tokenizer
        model = NLLBModel.from_pretrained(mlx_path)
        tokenizer = AutoTokenizer.from_pretrained(mlx_path)

        # Set source language
        tokenizer.src_lang = src_lang

        # Tokenize
        inputs = tokenizer(text, return_tensors="np")
        input_ids = mx.array(inputs["input_ids"])

        # Encode
        encoder_output = model.encode(input_ids)
        mx.eval(encoder_output)

        # NLLB uses a special generation pattern:
        # 1. Start with decoder_start_token_id (EOS=2)
        # 2. Force first generated token to be target language
        # 3. Then continue with greedy decoding
        decoder_start_id = tokenizer.eos_token_id  # 2
        tgt_lang_id = tokenizer.convert_tokens_to_ids(tgt_lang)

        # Start sequence with EOS + target language token
        generated = [decoder_start_id, tgt_lang_id]

        # Prime the cache with initial tokens
        decoder_ids = mx.array([generated])
        logits, cache = model.decode(decoder_ids, encoder_output, cache=None)
        mx.eval(logits, cache)

        # Generate tokens with KV cache (incremental decoding)
        for _ in range(max_tokens):
            next_token = int(mx.argmax(logits[0, -1]))
            generated.append(next_token)

            if next_token == tokenizer.eos_token_id:
                break

            # Decode next token with cache
            decoder_ids = mx.array([[next_token]])
            logits, cache = model.decode(decoder_ids, encoder_output, cache=cache)
            mx.eval(logits, cache)

        # Decode
        output_text = tokenizer.decode(generated, skip_special_tokens=True)
        return str(output_text)

    @staticmethod
    def list_supported_models() -> list[str]:
        """
        List known supported NLLB model variants.

        Returns:
            List of HuggingFace model paths
        """
        return [
            "facebook/nllb-200-distilled-600M",
            "facebook/nllb-200-distilled-1.3B",
            "facebook/nllb-200-1.3B",
            "facebook/nllb-200-3.3B",
        ]
