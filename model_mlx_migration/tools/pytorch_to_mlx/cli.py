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
PyTorch to MLX Converter - Command Line Interface

Main entry point for the conversion tool.

Usage:
    # Model-specific converters (recommended - fully functional):
    ./pytorch_to_mlx llama convert --hf-path meta-llama/... --output model_mlx/
    ./pytorch_to_mlx nllb convert --hf-path facebook/nllb-... --output model_mlx/
    ./pytorch_to_mlx kokoro convert --output model_mlx/
    ./pytorch_to_mlx cosyvoice2 synthesize --text "Hello world" --output audio.wav
    ./pytorch_to_mlx whisper transcribe --audio audio.wav

    # General converter (generates code templates, requires manual implementation):
    ./pytorch_to_mlx analyze --input model.pt
    ./pytorch_to_mlx convert --input model.pt --output model_mlx/

Note: The general 'convert' command generates model.py templates with placeholder
forward() implementations. For production use, use the model-specific converters above.
"""

import argparse
import json
import sys
from pathlib import Path


def cmd_analyze(args: argparse.Namespace) -> int:
    """Analyze a PyTorch/TorchScript model."""
    from .analyzer.op_mapper import OpMapper
    from .analyzer.torchscript_analyzer import TorchScriptAnalyzer

    print(f"Analyzing: {args.input}")

    try:
        analyzer = TorchScriptAnalyzer(args.input)
        architecture = analyzer.get_architecture()
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        return 1

    # Print summary
    print(analyzer.summarize())

    # Op coverage report
    op_mapper = OpMapper()
    op_names = [op.name for op in architecture.ops]
    coverage = op_mapper.get_coverage_report(op_names)

    print(f"\nOp Coverage: {coverage['coverage_percent']:.1f}%")
    print(f"  Direct mappings: {coverage['direct_mappings']}")
    print(f"  Decomposed: {coverage['decomposed_ops']}")
    print(f"  Custom: {coverage['custom_ops']}")
    print(f"  Unsupported: {coverage['unsupported_ops']}")

    if coverage["unsupported_list"]:
        print("\nUnsupported operations:")
        for op in coverage["unsupported_list"]:
            print(f"  - {op}")

    # Save analysis to JSON if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert architecture to serializable dict
        arch_dict = {
            "name": architecture.name,
            "total_params": architecture.total_params,
            "total_size_bytes": architecture.total_size_bytes,
            "ops": [
                {"name": op.name, "category": op.category.value, "count": op.count}
                for op in architecture.ops
            ],
            "layers": [
                {
                    "name": layer.name,
                    "type": layer.layer_type,
                    "attributes": layer.attributes,
                }
                for layer in architecture.layers
            ],
            "weights": [
                {
                    "name": weight.name,
                    "shape": list(weight.shape),
                    "dtype": weight.dtype,
                }
                for weight in architecture.weights
            ],
            "coverage": coverage,
        }

        with open(output_path, "w") as f:
            json.dump(arch_dict, f, indent=2)
        print(f"\nAnalysis saved to: {output_path}")

    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    """Convert PyTorch model to MLX format."""
    from .analyzer.torchscript_analyzer import TorchScriptAnalyzer
    from .generator.mlx_code_generator import MLXCodeGenerator
    from .generator.weight_converter import WeightConverter

    print(f"Converting: {args.input}")
    print(f"Output directory: {args.output}")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Analyze model
    print("\n[1/3] Analyzing model...")
    try:
        analyzer = TorchScriptAnalyzer(args.input)
        architecture = analyzer.get_architecture()
        print(
            f"  Found {len(architecture.layers)} layers, {architecture.total_params:,} parameters",
        )
    except Exception as e:
        print(f"Error analyzing model: {e}", file=sys.stderr)
        return 1

    # Step 2: Generate MLX code
    print("\n[2/3] Generating MLX code...")
    try:
        generator = MLXCodeGenerator()
        class_name = args.class_name or None
        generator.generate_file(
            architecture, str(output_dir / "model.py"), class_name=class_name,
        )
        print(f"  Generated: {output_dir / 'model.py'}")
    except Exception as e:
        print(f"Error generating code: {e}", file=sys.stderr)
        return 1

    # Step 3: Convert weights
    print("\n[3/3] Converting weights...")
    try:
        converter = WeightConverter(target_dtype=args.dtype)
        stats = converter.convert(
            args.input,
            str(output_dir / "weights.safetensors"),
            progress_callback=lambda name, cur, total: print(
                f"  [{cur}/{total}] {name}",
            )
            if args.verbose
            else None,
        )
        print(f"  Converted {stats.converted_tensors} tensors")
        print(f"  Output size: {stats.total_bytes_output / (1024 * 1024):.2f} MB")

        if stats.errors:
            print(f"  Warnings: {len(stats.errors)}")
            for err in stats.errors[:5]:
                print(f"    - {err}")
    except Exception as e:
        print(f"Error converting weights: {e}", file=sys.stderr)
        return 1

    # Optional: Validate
    if args.validate:
        print("\n[Optional] Validating conversion...")
        result = validate_conversion(args.input, output_dir, args.tolerance)
        if not result:
            print("  VALIDATION FAILED", file=sys.stderr)
            return 1
        print("  VALIDATION PASSED")

    # Optional: Benchmark
    if args.benchmark:
        print("\n[Optional] Running benchmarks...")
        run_benchmark(args.input, output_dir)

    print("\nConversion complete!")
    print("Output files:")
    print(
        f"  {output_dir / 'model.py'} (template - requires manual forward() implementation)",
    )
    print(f"  {output_dir / 'weights.safetensors'}")
    print()
    print("NOTE: The generated model.py is a template with placeholder forward() code.")
    print("For production use, consider model-specific converters:")
    print("  pytorch_to_mlx llama convert    # LLaMA/Mistral models")
    print("  pytorch_to_mlx nllb convert     # NLLB translation models")
    print("  pytorch_to_mlx kokoro convert   # Kokoro TTS models")
    print("  pytorch_to_mlx cosyvoice2 synthesize  # CosyVoice2 TTS")
    print("  pytorch_to_mlx whisper transcribe     # Whisper STT")

    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate converted model against original."""
    from .generator.weight_converter import WeightConverter

    print("Validating:")
    print(f"  PyTorch: {args.pytorch}")
    print(f"  MLX:     {args.mlx}")

    converter = WeightConverter()
    report = converter.verify_conversion(
        args.pytorch, args.mlx, tolerance=args.tolerance,
    )

    print("\nResults:")
    print(f"  PyTorch tensors: {report['pytorch_tensors']}")
    print(f"  MLX tensors:     {report['mlx_tensors']}")
    print(f"  Matched:         {report['matched']}")
    print(f"  Mismatched:      {report['mismatched']}")

    if report["missing_in_mlx"]:
        print(f"\nMissing in MLX ({len(report['missing_in_mlx'])}):")
        for name in report["missing_in_mlx"][:10]:
            print(f"  - {name}")

    if report["mismatched"] > 0:
        print("\nLargest errors:")
        sorted_errors = sorted(
            report["max_errors"].items(), key=lambda x: x[1], reverse=True,
        )
        for name, error in sorted_errors[:10]:
            print(f"  {name}: {error:.2e}")

    if report["verified"]:
        print("\n✓ VALIDATION PASSED")
        return 0
    print("\n✗ VALIDATION FAILED")
    return 1


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Benchmark model performance."""
    from .validator.benchmark import Benchmark

    print(f"Benchmarking: {args.mlx}")

    # Create test input
    shape = tuple(map(int, args.input_shape.split(",")))
    test_input = Benchmark.create_test_input(shape, dtype=args.dtype)

    benchmark = Benchmark(
        warmup_iterations=args.warmup,
        benchmark_iterations=args.iterations,
        batch_size=1,
    )

    results = []

    # Benchmark MLX model
    print("\nLoading MLX model...")
    try:
        import mlx.core as mx

        # Load model (this is a simplified loader - real implementation would load model.py)
        weights_path = Path(args.mlx) / "weights.safetensors"
        if weights_path.exists():
            weights = mx.load(str(weights_path))
            print(f"  Loaded {len(weights)} weight tensors")

        # For now, we can only benchmark if we have a model callable
        print("  Note: Full model benchmark requires loading model.py")

    except Exception as e:
        print(f"Error loading MLX model: {e}", file=sys.stderr)
        return 1

    # If PyTorch model provided, compare
    if args.pytorch:
        print("\nLoading PyTorch model for comparison...")
        try:
            import torch

            pt_model = torch.jit.load(args.pytorch, map_location="cpu")
            pt_model.eval()

            pt_result = benchmark.benchmark_pytorch(
                pt_model, test_input, model_name="pytorch",
            )
            results.append(pt_result)
            print(f"  PyTorch latency: {pt_result.latency.mean_ms:.2f} ms")

        except Exception as e:
            print(f"Error benchmarking PyTorch: {e}")

    # Generate report
    if results:
        print("\n" + benchmark.generate_report(results))

    return 0


def validate_conversion(pytorch_path: str, mlx_dir: Path, tolerance: float) -> bool:
    """Helper to validate conversion."""
    from .generator.weight_converter import WeightConverter

    weights_path = mlx_dir / "weights.safetensors"
    if not weights_path.exists():
        weights_path = mlx_dir / "weights.npz"

    if not weights_path.exists():
        print("  No weights file found to validate")
        return False

    converter = WeightConverter()
    report = converter.verify_conversion(pytorch_path, str(weights_path), tolerance)

    return bool(report["verified"])


def run_benchmark(pytorch_path: str, mlx_dir: Path) -> None:
    """Helper to run benchmark after conversion."""
    # This would run a quick benchmark comparison
    # Simplified for now
    print("  Benchmark comparison not yet implemented for full models")
    print("  Use 'pytorch_to_mlx benchmark' command for detailed benchmarking")


# ============================================================================
# LLaMA Commands
# ============================================================================


def cmd_llama_convert(args: argparse.Namespace) -> int:
    """Convert a LLaMA model using mlx-lm."""
    from .converters import LLaMAConverter

    print(f"Converting LLaMA model: {args.hf_path}")
    print(f"Output directory: {args.output}")

    converter = LLaMAConverter()
    result = converter.convert(
        hf_path=args.hf_path,
        output_path=args.output,
        quantize=args.quantize,
        q_bits=args.q_bits,
        dtype=args.dtype,
    )

    if result.success:
        print("\nConversion successful!")
        print(f"  Model size: {result.model_size_mb:.1f} MB")
        print(f"  Parameters: {result.num_parameters:,}")
        print(f"  Output: {result.mlx_path}")
        return 0
    print(f"\nConversion failed: {result.error}", file=sys.stderr)
    return 1


def cmd_llama_validate(args: argparse.Namespace) -> int:
    """Validate a converted LLaMA model."""
    from .converters import LLaMAConverter

    print(f"Validating MLX model: {args.mlx_path}")

    # Parse test prompts
    if args.prompts:
        test_prompts = args.prompts
    else:
        test_prompts = [
            "The quick brown fox",
            "In a galaxy far far away",
            "def fibonacci(n):",
        ]

    print(f"Test prompts: {len(test_prompts)}")

    converter = LLaMAConverter()
    result = converter.validate(
        mlx_path=args.mlx_path,
        test_prompts=test_prompts,
        hf_path=args.hf_path,
        tolerance=args.tolerance,
    )

    if result.error:
        print(f"\nValidation error: {result.error}", file=sys.stderr)
        return 1

    print("\nValidation Results:")
    print(f"  Max absolute error: {result.max_abs_error:.2e}")
    print(f"  Mean absolute error: {result.mean_abs_error:.2e}")
    print(f"  Tokens compared: {result.tokens_compared}")
    print(f"  Mismatched predictions: {result.mismatched_tokens}")
    print(f"  Status: {'PASSED' if result.passed else 'FAILED'}")

    return 0 if result.passed else 1


def cmd_llama_benchmark(args: argparse.Namespace) -> int:
    """Benchmark a converted LLaMA model."""
    from .converters import LLaMAConverter

    print(f"Benchmarking: {args.mlx_path}")

    # Parse test prompts
    if args.prompts:
        test_prompts = args.prompts
    else:
        test_prompts = ["The future of artificial intelligence is"]

    print(f"Test prompts: {len(test_prompts)}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Runs: {args.runs}")

    converter = LLaMAConverter()

    try:
        result = converter.benchmark(
            mlx_path=args.mlx_path,
            test_prompts=test_prompts,
            hf_path=args.hf_path,
            max_tokens=args.max_tokens,
            runs=args.runs,
        )

        print("\nBenchmark Results:")
        print(f"  MLX tokens/sec: {result.mlx_tokens_per_second:.1f}")
        print(f"  PyTorch tokens/sec: {result.pytorch_tokens_per_second:.1f}")
        print(f"  Speedup: {result.speedup:.2f}x")
        print(f"  MLX TTFT: {result.mlx_time_to_first_token_ms:.1f} ms")
        print(f"  PyTorch TTFT: {result.pytorch_time_to_first_token_ms:.1f} ms")

        return 0

    except Exception as e:
        print(f"\nBenchmark error: {e}", file=sys.stderr)
        return 1


def cmd_llama_list(args: argparse.Namespace) -> int:
    """List supported LLaMA models."""
    from .converters import LLaMAConverter

    print("Supported LLaMA/Mistral Models:")
    print("-" * 40)
    for model in LLaMAConverter.list_supported_models():
        print(f"  {model}")

    print("\nNote: Any HuggingFace model compatible with mlx-lm can be used.")
    print("These are just commonly tested models.")

    return 0


# ============================================================================
# NLLB Commands
# ============================================================================


def cmd_nllb_convert(args: argparse.Namespace) -> int:
    """Convert an NLLB model from HuggingFace."""
    from .converters import NLLBConverter

    print(f"Converting NLLB model: {args.hf_path}")
    print(f"Output directory: {args.output}")

    converter = NLLBConverter()
    result = converter.convert(
        hf_path=args.hf_path,
        output_path=args.output,
    )

    if result.success:
        print("\nConversion successful!")
        print(f"  Model size: {result.model_size_mb:.1f} MB")
        print(f"  Parameters: {result.num_parameters:,}")
        print(f"  Output: {result.mlx_path}")
        return 0
    print(f"\nConversion failed: {result.error}", file=sys.stderr)
    return 1


def cmd_nllb_validate(args: argparse.Namespace) -> int:
    """Validate a converted NLLB model."""
    from .converters import NLLBConverter

    print(f"Validating MLX model: {args.mlx_path}")

    # Parse test texts
    if args.texts:
        test_texts = args.texts
    else:
        test_texts = [
            "Hello, how are you?",
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is transforming the world.",
        ]

    print(f"Test texts: {len(test_texts)}")

    converter = NLLBConverter()
    result = converter.validate(
        mlx_path=args.mlx_path,
        test_texts=test_texts,
        hf_path=args.hf_path,
        tolerance=args.tolerance,
    )

    if result.error:
        print(f"\nValidation error: {result.error}", file=sys.stderr)
        return 1

    print("\nValidation Results:")
    print(f"  Encoder max error: {result.encoder_max_error:.2e}")
    print(f"  Decoder max error: {result.decoder_max_error:.2e}")
    print(f"  Top token match: {'Yes' if result.top_token_match else 'No'}")
    print(f"  Status: {'PASSED' if result.passed else 'FAILED'}")

    return 0 if result.passed else 1


def cmd_nllb_benchmark(args: argparse.Namespace) -> int:
    """Benchmark a converted NLLB model."""
    from .converters import NLLBConverter

    print(f"Benchmarking: {args.mlx_path}")

    # Parse test texts
    if args.texts:
        test_texts = args.texts
    else:
        test_texts = [
            "Hello, how are you?",
            "Machine learning is transforming the world.",
        ]

    print(f"Test texts: {len(test_texts)}")
    print(f"Runs: {args.runs}")

    converter = NLLBConverter()

    try:
        result = converter.benchmark(
            mlx_path=args.mlx_path,
            test_texts=test_texts,
            hf_path=args.hf_path,
            runs=args.runs,
        )

        print("\nBenchmark Results:")
        print(f"  MLX tokens/sec: {result.mlx_tokens_per_second:.1f}")
        print(f"  PyTorch tokens/sec: {result.pytorch_tokens_per_second:.1f}")
        print(f"  Speedup: {result.speedup:.2f}x")
        print(f"  MLX encode time: {result.mlx_encode_time_ms:.1f} ms")
        print(f"  PyTorch encode time: {result.pytorch_encode_time_ms:.1f} ms")

        return 0

    except Exception as e:
        print(f"\nBenchmark error: {e}", file=sys.stderr)
        return 1


def cmd_nllb_translate(args: argparse.Namespace) -> int:
    """Translate text using NLLB model."""
    from .converters import NLLBConverter

    converter = NLLBConverter()

    try:
        result = converter.translate(
            mlx_path=args.mlx_path,
            text=args.text,
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            max_tokens=args.max_tokens,
        )

        print(f"Source ({args.src_lang}): {args.text}")
        print(f"Target ({args.tgt_lang}): {result}")

        return 0

    except Exception as e:
        print(f"\nTranslation error: {e}", file=sys.stderr)
        return 1


def cmd_nllb_list(args: argparse.Namespace) -> int:
    """List supported NLLB models."""
    from .converters import NLLBConverter

    print("Supported NLLB Models:")
    print("-" * 40)
    for model in NLLBConverter.list_supported_models():
        print(f"  {model}")

    print("\nNote: Any HuggingFace NLLB-200 model can be used.")

    return 0


# ============================================================================
# Kokoro TTS Commands
# ============================================================================


def cmd_kokoro_convert(args: argparse.Namespace) -> int:
    """Convert a Kokoro TTS model from HuggingFace."""
    from .converters import KokoroConverter

    print(f"Converting Kokoro model: {args.hf_path}")
    print(f"Output directory: {args.output}")

    converter = KokoroConverter()
    result = converter.convert(
        model_id=args.hf_path,
        output_path=args.output,
    )

    if result.success:
        print("\nConversion successful!")
        print(f"  Model size: {result.model_size_mb:.1f} MB")
        print(f"  Parameters: {result.num_parameters:,}")
        print(f"  Output: {result.mlx_path}")
        return 0
    print(f"\nConversion failed: {result.error}", file=sys.stderr)
    return 1


def cmd_kokoro_validate(args: argparse.Namespace) -> int:
    """Validate a converted Kokoro model."""
    from .converters import KokoroConverter

    print(f"Validating Kokoro model: {args.hf_path}")

    converter = KokoroConverter()
    result = converter.validate(
        model_id=args.hf_path,
    )

    if result.error:
        print(f"\nValidation error: {result.error}", file=sys.stderr)
        return 1

    print("\nValidation Results:")
    print(f"  Text encoder max error: {result.text_encoder_max_error:.2e}")
    print(f"  BERT max error: {result.bert_max_error:.2e}")
    print(f"  Status: {'PASSED' if result.passed else 'FAILED'}")

    return 0 if result.passed else 1


def cmd_kokoro_list(args: argparse.Namespace) -> int:
    """List supported Kokoro models."""
    from .converters import KokoroConverter

    print("Supported Kokoro Models:")
    print("-" * 40)
    for model in KokoroConverter.list_supported_models():
        print(f"  {model}")

    return 0


def cmd_kokoro_synthesize(args: argparse.Namespace) -> int:
    """Synthesize speech from text using Kokoro TTS."""
    import time

    import mlx.core as mx
    import numpy as np

    from .converters.kokoro_converter import KokoroConverter

    print(f"Text: {args.text}")
    print(f"Voice: {args.voice}")
    print()

    # Load model
    print("Loading Kokoro model...")
    start = time.time()
    converter = KokoroConverter()
    model, config, _ = converter.load_from_hf(args.model_id)
    print(f"Loaded in {time.time() - start:.2f}s")
    print()

    # Phonemize text
    print("Phonemizing text...")
    try:
        from .converters.models.kokoro_phonemizer import phonemize_text

        phonemes, token_ids = phonemize_text(args.text)
        print(f"Phonemes: {phonemes}")
    except ImportError:
        print("Warning: Phonemizer not available, using dummy tokens")
        # Use dummy token sequence for demo
        token_ids = [50, 83, 54, 156, 31, 16, 65, 156, 87, 123, 54, 46]
        phonemes = "həlˈO wˈɜɹld"

    # Load voice
    # IMPORTANT: Use phoneme STRING length, not token count
    # PyTorch Kokoro: pack[len(ps) - 1] where ps is phoneme string
    print(f"Loading voice: {args.voice}...")
    ref_s = converter.load_voice(args.voice, phoneme_length=len(phonemes))

    # Synthesize
    print("Synthesizing...")
    start = time.time()
    tokens = mx.array([token_ids])
    audio = model(tokens, ref_s, speed=args.speed)
    mx.eval(audio)
    synth_time = time.time() - start

    audio_np = np.array(audio[0])
    audio_duration = len(audio_np) / 24000  # Kokoro uses 24kHz

    print(f"Generated {audio_duration:.2f}s of audio in {synth_time:.2f}s")
    print(
        f"Real-time factor: {synth_time / audio_duration:.3f}x ({1 / (synth_time / audio_duration):.1f}x faster than real-time)",
    )

    # Save output
    if args.output:
        try:
            import soundfile as sf

            # Normalize if requested
            if np.abs(audio_np).max() > 0:
                audio_np = audio_np / np.abs(audio_np).max() * 0.95
            sf.write(args.output, audio_np, 24000)
            print(f"Saved to: {args.output}")
        except ImportError:
            print("Warning: soundfile not available, cannot save audio")
            return 1
    else:
        print("No output file specified (use --output to save)")

    return 0


# ========================================================================
# CosyVoice2 Commands
# ========================================================================


def cmd_cosyvoice2_convert(args: argparse.Namespace) -> int:
    """Convert a CosyVoice2 TTS model."""
    from .converters import CosyVoice2Converter

    print(f"Converting CosyVoice2 model: {args.model_path}")
    print(f"Output directory: {args.output}")

    converter = CosyVoice2Converter()
    result = converter.convert(
        model_id_or_path=args.model_path,
        output_path=args.output,
    )

    if result.success:
        print("\nConversion successful!")
        print(f"  Model size: {result.model_size_mb:.1f} MB")
        print(f"  Parameters: {result.num_parameters:,}")
        print(f"  Output: {result.mlx_path}")
        return 0
    print(f"\nConversion status: {result.error}", file=sys.stderr)
    return 1


def cmd_cosyvoice2_inspect(args: argparse.Namespace) -> int:
    """Inspect a CosyVoice2 model structure."""
    from .converters.cosyvoice2_converter import (
        CosyVoice2Converter,
        print_inspection_report,
    )

    print(f"Inspecting CosyVoice2 model: {args.model_path}")

    converter = CosyVoice2Converter()
    try:
        inspection = converter.inspect_model(args.model_path)
        print_inspection_report(inspection)
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_cosyvoice2_validate(args: argparse.Namespace) -> int:
    """Validate a converted CosyVoice2 model."""
    from .converters import CosyVoice2Converter

    print(f"Validating CosyVoice2 model: {args.mlx_path}")

    converter = CosyVoice2Converter()
    result = converter.validate(
        mlx_model_path=args.mlx_path,
        pytorch_model_path=args.pytorch_path,
    )

    if result.error:
        print(f"\nValidation status: {result.error}", file=sys.stderr)
        return 1

    print("\nValidation Results:")
    print(f"  LLM max error: {result.llm_max_error:.2e}")
    print(f"  Flow max error: {result.flow_max_error:.2e}")
    print(f"  Vocoder max error: {result.vocoder_max_error:.2e}")
    print(f"  Status: {'PASSED' if result.passed else 'FAILED'}")

    return 0 if result.passed else 1


def cmd_cosyvoice2_synthesize(args: argparse.Namespace) -> int:
    """Synthesize speech from text using CosyVoice2."""
    import time
    from pathlib import Path

    import mlx.core as mx
    import numpy as np

    from .converters.models import CosyVoice2Model

    # Get model path
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        model_path = CosyVoice2Model.get_default_model_path()

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}", file=sys.stderr)
        print(
            "Download with: python scripts/download_cosyvoice2.py --source huggingface",
        )
        return 1

    print(f"Model: {model_path}")
    print(f"Text: {args.text}")
    print()

    # Load model
    print("Loading model...")
    start = time.time()
    model = CosyVoice2Model.from_pretrained(model_path)
    print(f"Loaded in {time.time() - start:.2f}s")
    print()

    # Generate speaker embedding
    print(f"Using speaker seed: {args.speaker_seed}")
    if model.tokenizer is None:
        print("Error: Model tokenizer not initialized", file=sys.stderr)
        return 1
    speaker_embedding = model.tokenizer.random_speaker_embedding(seed=args.speaker_seed)

    # Synthesize
    print("Synthesizing...")
    start = time.time()
    audio = model.synthesize_text(
        args.text,
        speaker_embedding=speaker_embedding,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    mx.eval(audio)
    synth_time = time.time() - start

    audio_np = np.array(audio)
    audio_duration = len(audio_np) / model.config.sample_rate

    print(f"Generated {audio_duration:.2f}s of audio in {synth_time:.2f}s")
    print(f"Real-time factor: {synth_time / audio_duration:.2f}x")

    # Save output
    if args.output:
        try:
            import soundfile as sf

            # Normalize
            audio_np = audio_np / max(abs(audio_np.max()), abs(audio_np.min()), 1e-8)
            sf.write(args.output, audio_np, model.config.sample_rate)
            print(f"Saved to: {args.output}")
        except ImportError:
            print("Warning: soundfile not installed, cannot save audio")
            print("Install with: pip install soundfile")
            return 1

    return 0


def cmd_cosyvoice2_list(args: argparse.Namespace) -> int:
    """List supported CosyVoice2 models."""
    from .converters import CosyVoice2Converter

    print("Supported CosyVoice2 Models:")
    print("-" * 40)
    for model in CosyVoice2Converter.list_supported_models():
        print(f"  {model}")

    print("\nTo download models, use:")
    print("  python scripts/download_cosyvoice2.py --source huggingface")

    return 0


# ============================================================================
# Whisper STT Commands
# ============================================================================


def cmd_whisper_transcribe(args: argparse.Namespace) -> int:
    """Transcribe audio to text using Whisper."""
    from .converters import WhisperConverter

    print(f"Transcribing: {args.audio}")
    print(f"Model: {args.model}")

    converter = WhisperConverter()
    result = converter.transcribe(
        audio_path=args.audio,
        model=args.model,
        language=args.language,
        verbose=args.verbose,
        word_timestamps=args.word_timestamps,
        initial_prompt=args.initial_prompt,
        temperature=args.temperature,
    )

    if not result.success:
        print(f"\nTranscription failed: {result.error}", file=sys.stderr)
        return 1

    # Print results
    if result.language:
        print(f"Detected language: {result.language}")
    print(f"Audio duration: {result.duration_seconds:.1f}s")
    print(f"Transcription time: {result.transcription_time_seconds:.2f}s")
    print(f"Real-time factor: {result.real_time_factor:.3f}x")
    print()

    # Format output
    output_text = converter.format_output(result, args.format)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output_text)
        print(f"Saved to: {args.output}")
    else:
        print("=" * 60)
        print(output_text)
        print("=" * 60)

    return 0


def cmd_whisper_benchmark(args: argparse.Namespace) -> int:
    """Benchmark Whisper transcription performance."""
    from .converters import WhisperConverter

    print(f"Benchmarking: {args.audio}")
    print(f"Model: {args.model}")
    print(f"Runs: {args.runs}")

    converter = WhisperConverter()

    try:
        result = converter.benchmark(
            audio_path=args.audio, model=args.model, runs=args.runs,
        )

        print("\nBenchmark Results:")
        print(f"  Audio duration: {result.audio_duration_seconds:.1f}s")
        print(f"  Avg transcription time: {result.transcription_time_seconds:.2f}s")
        print(f"  Real-time factor: {result.real_time_factor:.3f}x")
        print(f"  Words per second: {result.words_per_second:.1f}")
        print(f"  Model: {result.model}")

        speedup = 1.0 / result.real_time_factor if result.real_time_factor > 0 else 0
        print(f"  Speedup: {speedup:.1f}x faster than real-time")

        return 0

    except Exception as e:
        print(f"\nBenchmark error: {e}", file=sys.stderr)
        return 1


def cmd_whisper_list(args: argparse.Namespace) -> int:
    """List available Whisper models."""
    from .converters import WhisperConverter

    print("Available Whisper Models (mlx-community):")
    print("-" * 50)

    for model in WhisperConverter.list_models():
        info = WhisperConverter.get_model_info(model)
        multilingual = "multilingual" if info["multilingual"] else "English only"
        print(f"  {model}")
        print(f"      Size: {info['size']}, {multilingual}")

    print()
    print("Recommended models:")
    print("  - whisper-large-v3-turbo: Best speed/quality balance")
    print("  - whisper-large-v3: Best quality")
    print("  - whisper-small: Good for testing")

    return 0


# ============================================================================
# Wake Word Detection Commands
# ============================================================================


def cmd_wakeword_status(args: argparse.Namespace) -> int:
    """Check wake word model status."""
    from .converters import WakeWordConverter

    converter = WakeWordConverter(
        melspec_path=Path(args.melspec) if args.melspec else None,
        embedding_path=Path(args.embedding) if args.embedding else None,
        classifier_path=Path(args.classifier) if args.classifier else None,
    )

    status = converter.get_model_status()

    print("Wake Word Model Status")
    print("=" * 60)

    for model_name in ["melspec", "embedding", "classifier"]:
        info = status[model_name]
        exists = "FOUND" if info["exists"] else "NOT FOUND"
        size = f"{info['size_mb']:.2f} MB" if info["exists"] else "N/A"
        print(f"\n{model_name.upper()}:")
        print(f"  Path: {info['path']}")
        print(f"  Status: {exists}")
        print(f"  Size: {size}")

    print()
    print(
        f"ONNX Runtime: {'Available' if status['onnx_available'] else 'Not installed'}",
    )
    print(f"MLX: {'Available' if status['mlx_available'] else 'Not installed'}")

    if converter.models_available():
        print("\nAll models found. Ready for detection.")
        return 0
    print("\nSome models missing. See expected paths above.")
    print("\nExpected model paths:")
    for name, path in converter.get_expected_model_paths().items():
        print(f"  {name}: {path}")
    return 1


def cmd_wakeword_analyze(args: argparse.Namespace) -> int:
    """Analyze ONNX wake word models."""
    from .converters import WakeWordConverter

    converter = WakeWordConverter(
        melspec_path=Path(args.melspec) if args.melspec else None,
        embedding_path=Path(args.embedding) if args.embedding else None,
        classifier_path=Path(args.classifier) if args.classifier else None,
    )

    if not converter.models_available():
        print("Error: Wake word models not found.", file=sys.stderr)
        print("\nRun 'pytorch_to_mlx wakeword status' for details.")
        return 1

    analysis = converter.analyze_all_models()

    print("Wake Word Model Analysis")
    print("=" * 60)

    for model_name in ["melspec", "embedding", "classifier"]:
        info = analysis[model_name]
        print(f"\n{model_name.upper()} ({info['path']})")
        print("-" * 40)

        print("  Inputs:")
        for inp in info["inputs"]:
            print(f"    {inp['name']}: {inp['shape']} ({inp['dtype']})")

        print("  Outputs:")
        for out in info["outputs"]:
            print(f"    {out['name']}: {out['shape']} ({out['dtype']})")

        print(f"  Total ops: {info['total_ops']}")
        print("  Op breakdown:")
        for op, count in sorted(info["op_counts"].items(), key=lambda x: -x[1])[:10]:
            print(f"    {op}: {count}")

    if args.output:
        import json

        with open(args.output, "w") as f:
            json.dump(analysis, f, indent=2)
        print(f"\nAnalysis saved to: {args.output}")

    return 0


def cmd_wakeword_detect(args: argparse.Namespace) -> int:
    """Run wake word detection on audio file."""
    from .converters import WakeWordConverter

    try:
        import soundfile as sf
    except ImportError:
        print(
            "Error: soundfile required. Install with: pip install soundfile",
            file=sys.stderr,
        )
        return 1

    converter = WakeWordConverter(
        melspec_path=Path(args.melspec) if args.melspec else None,
        embedding_path=Path(args.embedding) if args.embedding else None,
        classifier_path=Path(args.classifier) if args.classifier else None,
    )

    if not converter.models_available():
        print("Error: Wake word models not found.", file=sys.stderr)
        print("\nRun 'pytorch_to_mlx wakeword status' for details.")
        return 1

    # Load audio
    print(f"Loading audio: {args.audio}")
    try:
        audio, sample_rate = sf.read(args.audio, dtype="float32")
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)  # Convert stereo to mono
    except Exception as e:
        print(f"Error loading audio: {e}", file=sys.stderr)
        return 1

    print(f"Audio: {len(audio) / sample_rate:.2f}s, {sample_rate}Hz")

    # Run detection
    result = converter.detect(audio, sample_rate)

    if not result.success:
        print(f"Detection failed: {result.error}", file=sys.stderr)
        return 1

    print()
    print("Detection Result:")
    print(f"  Detected: {'YES' if result.detected else 'NO'}")
    print(f"  Probability: {result.probability:.4f}")
    print(f"  Inference time: {result.inference_time_seconds * 1000:.2f}ms")

    return 0 if not args.require_detection or result.detected else 1


def cmd_wakeword_benchmark(args: argparse.Namespace) -> int:
    """Benchmark wake word detection performance."""
    from .converters import WakeWordConverter

    converter = WakeWordConverter(
        melspec_path=Path(args.melspec) if args.melspec else None,
        embedding_path=Path(args.embedding) if args.embedding else None,
        classifier_path=Path(args.classifier) if args.classifier else None,
    )

    if not converter.models_available():
        print("Error: Wake word models not found.", file=sys.stderr)
        print("\nRun 'pytorch_to_mlx wakeword status' for details.")
        return 1

    print(f"Benchmarking wake word detection ({args.duration}s synthetic audio)...")

    try:
        result = converter.benchmark(
            duration_seconds=args.duration,
            sample_rate=args.sample_rate,
        )

        print("\nBenchmark Results:")
        print(f"  Total frames: {result.total_frames}")
        print(f"  Total time: {result.total_time_seconds:.3f}s")
        print(f"  Avg frame time: {result.avg_frame_time_ms:.2f}ms")
        print(f"  FPS: {result.fps:.1f}")
        print(f"  Backend: {result.model_type}")

        # Calculate real-time factor (500ms chunks)
        chunk_duration_ms = 500
        rtf = result.avg_frame_time_ms / chunk_duration_ms
        print(f"  Real-time factor: {rtf:.3f}x")

        return 0

    except Exception as e:
        print(f"Benchmark error: {e}", file=sys.stderr)
        return 1


def cmd_wakeword_convert(args: argparse.Namespace) -> int:
    """Generate MLX conversion templates for wake word models."""
    from .converters import WakeWordConverter

    converter = WakeWordConverter(
        melspec_path=Path(args.melspec) if args.melspec else None,
        embedding_path=Path(args.embedding) if args.embedding else None,
        classifier_path=Path(args.classifier) if args.classifier else None,
    )

    if not converter.models_available():
        print("Error: Wake word models not found.", file=sys.stderr)
        print("\nRun 'pytorch_to_mlx wakeword status' for details.")
        return 1

    result = converter.convert_to_mlx(
        output_dir=Path(args.output) if args.output else None,
    )

    if not result["success"]:
        print(
            f"Conversion failed: {result.get('error', 'Unknown error')}",
            file=sys.stderr,
        )
        return 1

    print("MLX Conversion Analysis")
    print("=" * 60)

    print("\nNext steps:")
    for step in result["next_steps"]:
        print(f"  {step}")

    print("\nGenerated MLX templates:")
    for model_name, template in result["mlx_templates"].items():
        print(f"\n--- {model_name}.py ---")
        print(template)

    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="pytorch_to_mlx",
        description="Convert PyTorch models to MLX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Model-specific converters (recommended - fully functional):
  pytorch_to_mlx llama convert --hf-path meta-llama/... --output ./
  pytorch_to_mlx nllb convert --hf-path facebook/nllb-200-distilled-600M --output ./
  pytorch_to_mlx kokoro convert --output ./
  pytorch_to_mlx cosyvoice2 synthesize --text "Hello" --output audio.wav
  pytorch_to_mlx whisper transcribe --audio audio.wav

General converter (generates code templates, requires manual implementation):
  pytorch_to_mlx analyze --input model.pt
  pytorch_to_mlx convert --input model.pt --output model_mlx/
  pytorch_to_mlx validate --pytorch model.pt --mlx model_mlx/weights.safetensors
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a PyTorch model")
    analyze_parser.add_argument(
        "--input", "-i", required=True, help="Path to PyTorch model",
    )
    analyze_parser.add_argument("--output", "-o", help="Save analysis to JSON file")

    # Convert command
    convert_parser = subparsers.add_parser(
        "convert", help="Convert PyTorch model to MLX (generates template code)",
    )
    convert_parser.add_argument(
        "--input", "-i", required=True, help="Path to PyTorch model",
    )
    convert_parser.add_argument(
        "--output", "-o", required=True, help="Output directory for MLX model",
    )
    convert_parser.add_argument("--class-name", help="Class name for generated model")
    convert_parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float16"],
        help="Target dtype for weights",
    )
    convert_parser.add_argument(
        "--validate", action="store_true", help="Validate conversion",
    )
    convert_parser.add_argument(
        "--benchmark", action="store_true", help="Run benchmarks",
    )
    convert_parser.add_argument(
        "--tolerance", type=float, default=1e-5, help="Tolerance for validation",
    )
    convert_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose output",
    )

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate converted model")
    validate_parser.add_argument(
        "--pytorch", required=True, help="Path to original PyTorch model",
    )
    validate_parser.add_argument(
        "--mlx", required=True, help="Path to MLX weights file",
    )
    validate_parser.add_argument(
        "--tolerance", type=float, default=1e-5, help="Tolerance for comparison",
    )

    # Benchmark command
    benchmark_parser = subparsers.add_parser(
        "benchmark", help="Benchmark model performance",
    )
    benchmark_parser.add_argument(
        "--mlx", required=True, help="Path to MLX model directory",
    )
    benchmark_parser.add_argument(
        "--pytorch", help="Optional PyTorch model for comparison",
    )
    benchmark_parser.add_argument(
        "--input-shape", default="1,512", help="Input shape (comma-separated)",
    )
    benchmark_parser.add_argument("--dtype", default="float32", help="Input dtype")
    benchmark_parser.add_argument(
        "--warmup", type=int, default=10, help="Warmup iterations",
    )
    benchmark_parser.add_argument(
        "--iterations", type=int, default=100, help="Benchmark iterations",
    )

    # ========================================================================
    # LLaMA subcommand group
    # ========================================================================
    llama_parser = subparsers.add_parser(
        "llama", help="LLaMA model commands (using mlx-lm)",
    )
    llama_subparsers = llama_parser.add_subparsers(
        dest="llama_command", help="LLaMA command to run",
    )

    # llama convert
    llama_convert_parser = llama_subparsers.add_parser(
        "convert", help="Convert LLaMA from HuggingFace",
    )
    llama_convert_parser.add_argument(
        "--hf-path",
        required=True,
        help="HuggingFace model path (e.g., meta-llama/Llama-3.2-1B)",
    )
    llama_convert_parser.add_argument(
        "--output", "-o", required=True, help="Output directory",
    )
    llama_convert_parser.add_argument(
        "--quantize", action="store_true", help="Quantize the model",
    )
    llama_convert_parser.add_argument(
        "--q-bits",
        type=int,
        default=4,
        choices=[4, 8],
        help="Quantization bits (default: 4)",
    )
    llama_convert_parser.add_argument(
        "--dtype", choices=["float16", "bfloat16", "float32"], help="Weight dtype",
    )

    # llama validate
    llama_validate_parser = llama_subparsers.add_parser(
        "validate", help="Validate MLX vs PyTorch",
    )
    llama_validate_parser.add_argument(
        "--mlx-path", required=True, help="Path to MLX model",
    )
    llama_validate_parser.add_argument(
        "--hf-path", help="HuggingFace path (auto-detected if not set)",
    )
    llama_validate_parser.add_argument("--prompts", nargs="+", help="Test prompts")
    llama_validate_parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
        help="Tolerance for validation (default: 1e-3)",
    )

    # llama benchmark
    llama_benchmark_parser = llama_subparsers.add_parser(
        "benchmark", help="Benchmark MLX vs PyTorch",
    )
    llama_benchmark_parser.add_argument(
        "--mlx-path", required=True, help="Path to MLX model",
    )
    llama_benchmark_parser.add_argument(
        "--hf-path", help="HuggingFace path (auto-detected if not set)",
    )
    llama_benchmark_parser.add_argument("--prompts", nargs="+", help="Test prompts")
    llama_benchmark_parser.add_argument(
        "--max-tokens", type=int, default=100, help="Tokens to generate (default: 100)",
    )
    llama_benchmark_parser.add_argument(
        "--runs", type=int, default=3, help="Benchmark runs (default: 3)",
    )

    # llama list
    llama_subparsers.add_parser("list", help="List supported models")

    # ========================================================================
    # NLLB subcommand group
    # ========================================================================
    nllb_parser = subparsers.add_parser("nllb", help="NLLB translation model commands")
    nllb_subparsers = nllb_parser.add_subparsers(
        dest="nllb_command", help="NLLB command to run",
    )

    # nllb convert
    nllb_convert_parser = nllb_subparsers.add_parser(
        "convert", help="Convert NLLB from HuggingFace",
    )
    nllb_convert_parser.add_argument(
        "--hf-path",
        required=True,
        help="HuggingFace model path (e.g., facebook/nllb-200-distilled-600M)",
    )
    nllb_convert_parser.add_argument(
        "--output", "-o", required=True, help="Output directory",
    )

    # nllb validate
    nllb_validate_parser = nllb_subparsers.add_parser(
        "validate", help="Validate MLX vs PyTorch",
    )
    nllb_validate_parser.add_argument(
        "--mlx-path", required=True, help="Path to MLX model",
    )
    nllb_validate_parser.add_argument(
        "--hf-path", help="HuggingFace path (auto-detected if not set)",
    )
    nllb_validate_parser.add_argument("--texts", nargs="+", help="Test texts")
    nllb_validate_parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="Tolerance for validation (default: 1e-4)",
    )

    # nllb benchmark
    nllb_benchmark_parser = nllb_subparsers.add_parser(
        "benchmark", help="Benchmark MLX vs PyTorch",
    )
    nllb_benchmark_parser.add_argument(
        "--mlx-path", required=True, help="Path to MLX model",
    )
    nllb_benchmark_parser.add_argument(
        "--hf-path", help="HuggingFace path (auto-detected if not set)",
    )
    nllb_benchmark_parser.add_argument("--texts", nargs="+", help="Test texts")
    nllb_benchmark_parser.add_argument(
        "--runs", type=int, default=3, help="Benchmark runs (default: 3)",
    )

    # nllb translate
    nllb_translate_parser = nllb_subparsers.add_parser(
        "translate", help="Translate text",
    )
    nllb_translate_parser.add_argument(
        "--mlx-path", required=True, help="Path to MLX model",
    )
    nllb_translate_parser.add_argument(
        "--text", required=True, help="Text to translate",
    )
    nllb_translate_parser.add_argument(
        "--src-lang", default="eng_Latn", help="Source language (default: eng_Latn)",
    )
    nllb_translate_parser.add_argument(
        "--tgt-lang", default="fra_Latn", help="Target language (default: fra_Latn)",
    )
    nllb_translate_parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate (default: 100)",
    )

    # nllb list
    nllb_subparsers.add_parser("list", help="List supported models")

    # Kokoro subcommand
    kokoro_parser = subparsers.add_parser("kokoro", help="Kokoro TTS model commands")
    kokoro_subparsers = kokoro_parser.add_subparsers(
        dest="kokoro_command", help="Kokoro command to run",
    )

    # kokoro convert
    kokoro_convert_parser = kokoro_subparsers.add_parser(
        "convert", help="Convert Kokoro from HuggingFace",
    )
    kokoro_convert_parser.add_argument(
        "--hf-path",
        default="hexgrad/Kokoro-82M",
        help="HuggingFace model path (default: hexgrad/Kokoro-82M)",
    )
    kokoro_convert_parser.add_argument(
        "--output", "-o", required=True, help="Output directory",
    )

    # kokoro validate
    kokoro_validate_parser = kokoro_subparsers.add_parser(
        "validate", help="Validate MLX vs PyTorch",
    )
    kokoro_validate_parser.add_argument(
        "--hf-path",
        default="hexgrad/Kokoro-82M",
        help="HuggingFace path (default: hexgrad/Kokoro-82M)",
    )

    # kokoro list
    kokoro_subparsers.add_parser("list", help="List supported models")

    # kokoro synthesize
    kokoro_synth_parser = kokoro_subparsers.add_parser(
        "synthesize", help="Synthesize speech from text",
    )
    kokoro_synth_parser.add_argument(
        "--text", "-t", required=True, help="Text to synthesize",
    )
    kokoro_synth_parser.add_argument(
        "--voice", "-v", default="af_bella", help="Voice name (default: af_bella)",
    )
    kokoro_synth_parser.add_argument(
        "--output", "-o", help="Output audio file path (WAV)",
    )
    kokoro_synth_parser.add_argument(
        "--model-id",
        default="hexgrad/Kokoro-82M",
        help="HuggingFace model ID (default: hexgrad/Kokoro-82M)",
    )
    kokoro_synth_parser.add_argument(
        "--speed", type=float, default=1.0, help="Speech speed (default: 1.0)",
    )

    # ========================================================================
    # CosyVoice2 subcommand group
    # ========================================================================
    cosyvoice2_parser = subparsers.add_parser(
        "cosyvoice2", help="CosyVoice2 TTS model commands",
    )
    cosyvoice2_subparsers = cosyvoice2_parser.add_subparsers(
        dest="cosyvoice2_command", help="CosyVoice2 command to run",
    )

    # cosyvoice2 convert
    cosyvoice2_convert_parser = cosyvoice2_subparsers.add_parser(
        "convert", help="Convert CosyVoice2 model",
    )
    cosyvoice2_convert_parser.add_argument(
        "--model-path", "-m", required=True, help="Path to downloaded model directory",
    )
    cosyvoice2_convert_parser.add_argument(
        "--output", "-o", required=True, help="Output directory",
    )

    # cosyvoice2 inspect
    cosyvoice2_inspect_parser = cosyvoice2_subparsers.add_parser(
        "inspect", help="Inspect model structure",
    )
    cosyvoice2_inspect_parser.add_argument(
        "--model-path", "-m", required=True, help="Path to model directory",
    )

    # cosyvoice2 validate
    cosyvoice2_validate_parser = cosyvoice2_subparsers.add_parser(
        "validate", help="Validate MLX vs PyTorch",
    )
    cosyvoice2_validate_parser.add_argument(
        "--mlx-path", required=True, help="Path to MLX model",
    )
    cosyvoice2_validate_parser.add_argument(
        "--pytorch-path", help="Path to PyTorch model",
    )

    # cosyvoice2 synthesize
    cosyvoice2_synth_parser = cosyvoice2_subparsers.add_parser(
        "synthesize", help="Synthesize speech from text",
    )
    cosyvoice2_synth_parser.add_argument(
        "--model-path",
        "-m",
        help="Path to model directory (default: ~/.cache/cosyvoice2/cosyvoice2-0.5b)",
    )
    cosyvoice2_synth_parser.add_argument(
        "--text", "-t", required=True, help="Text to synthesize",
    )
    cosyvoice2_synth_parser.add_argument("--output", "-o", help="Output WAV file")
    cosyvoice2_synth_parser.add_argument(
        "--max-tokens",
        type=int,
        default=1000,
        help="Maximum speech tokens (default: 1000)",
    )
    cosyvoice2_synth_parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)",
    )
    cosyvoice2_synth_parser.add_argument(
        "--speaker-seed",
        type=int,
        default=42,
        help="Random seed for speaker embedding (default: 42)",
    )

    # cosyvoice2 list
    cosyvoice2_subparsers.add_parser("list", help="List supported models")

    # ========================================================================
    # Whisper STT subcommand group
    # ========================================================================
    whisper_parser = subparsers.add_parser(
        "whisper", help="Whisper STT commands (using mlx-whisper)",
    )
    whisper_subparsers = whisper_parser.add_subparsers(
        dest="whisper_command", help="Whisper command to run",
    )

    # whisper transcribe
    whisper_transcribe_parser = whisper_subparsers.add_parser(
        "transcribe", help="Transcribe audio to text",
    )
    whisper_transcribe_parser.add_argument(
        "--audio", "-a", required=True, help="Path to audio file",
    )
    whisper_transcribe_parser.add_argument(
        "--model",
        "-m",
        default="mlx-community/whisper-large-v3-turbo",
        help="Model to use (default: whisper-large-v3-turbo)",
    )
    whisper_transcribe_parser.add_argument(
        "--language", "-l", help="Language code (auto-detect if not set)",
    )
    whisper_transcribe_parser.add_argument(
        "--output", "-o", help="Save transcription to file",
    )
    whisper_transcribe_parser.add_argument(
        "--format",
        "-f",
        default="text",
        choices=["text", "json", "srt", "vtt"],
        help="Output format (default: text)",
    )
    whisper_transcribe_parser.add_argument(
        "--word-timestamps", action="store_true", help="Include word-level timestamps",
    )
    whisper_transcribe_parser.add_argument(
        "--initial-prompt", help="Initial prompt to condition the model",
    )
    whisper_transcribe_parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0 for greedy)",
    )
    whisper_transcribe_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show progress during transcription",
    )

    # whisper benchmark
    whisper_benchmark_parser = whisper_subparsers.add_parser(
        "benchmark", help="Benchmark transcription performance",
    )
    whisper_benchmark_parser.add_argument(
        "--audio", "-a", required=True, help="Path to audio file",
    )
    whisper_benchmark_parser.add_argument(
        "--model",
        "-m",
        default="mlx-community/whisper-large-v3-turbo",
        help="Model to benchmark",
    )
    whisper_benchmark_parser.add_argument(
        "--runs", type=int, default=3, help="Number of runs (default: 3)",
    )

    # whisper list
    whisper_subparsers.add_parser("list", help="List available models")

    # ========================================================================
    # Wake Word Detection subcommand group
    # ========================================================================
    wakeword_parser = subparsers.add_parser(
        "wakeword", help="Wake word detection commands (ONNX/MLX)",
    )
    wakeword_subparsers = wakeword_parser.add_subparsers(
        dest="wakeword_command", help="Wake word command to run",
    )

    # Common wakeword arguments
    def add_wakeword_model_args(parser):
        parser.add_argument("--melspec", help="Path to mel spectrogram ONNX model")
        parser.add_argument("--embedding", help="Path to embedding ONNX model")
        parser.add_argument("--classifier", help="Path to classifier ONNX model")

    # wakeword status
    wakeword_status_parser = wakeword_subparsers.add_parser(
        "status", help="Check model availability",
    )
    add_wakeword_model_args(wakeword_status_parser)

    # wakeword analyze
    wakeword_analyze_parser = wakeword_subparsers.add_parser(
        "analyze", help="Analyze ONNX model structure",
    )
    add_wakeword_model_args(wakeword_analyze_parser)
    wakeword_analyze_parser.add_argument(
        "--output", "-o", help="Save analysis to JSON file",
    )

    # wakeword detect
    wakeword_detect_parser = wakeword_subparsers.add_parser(
        "detect", help="Run wake word detection",
    )
    add_wakeword_model_args(wakeword_detect_parser)
    wakeword_detect_parser.add_argument(
        "--audio", "-a", required=True, help="Path to audio file",
    )
    wakeword_detect_parser.add_argument(
        "--require-detection",
        action="store_true",
        help="Return non-zero exit code if wake word not detected",
    )

    # wakeword benchmark
    wakeword_benchmark_parser = wakeword_subparsers.add_parser(
        "benchmark", help="Benchmark detection performance",
    )
    add_wakeword_model_args(wakeword_benchmark_parser)
    wakeword_benchmark_parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Duration of synthetic audio in seconds (default: 10)",
    )
    wakeword_benchmark_parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Sample rate in Hz (default: 16000)",
    )

    # wakeword convert
    wakeword_convert_parser = wakeword_subparsers.add_parser(
        "convert", help="Generate MLX conversion templates",
    )
    add_wakeword_model_args(wakeword_convert_parser)
    wakeword_convert_parser.add_argument(
        "--output", "-o", help="Output directory for MLX models",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    # Dispatch to command handler
    if args.command == "analyze":
        return cmd_analyze(args)
    if args.command == "convert":
        return cmd_convert(args)
    if args.command == "validate":
        return cmd_validate(args)
    if args.command == "benchmark":
        return cmd_benchmark(args)
    if args.command == "llama":
        if args.llama_command is None:
            llama_parser.print_help()
            return 1
        if args.llama_command == "convert":
            return cmd_llama_convert(args)
        if args.llama_command == "validate":
            return cmd_llama_validate(args)
        if args.llama_command == "benchmark":
            return cmd_llama_benchmark(args)
        if args.llama_command == "list":
            return cmd_llama_list(args)
    elif args.command == "nllb":
        if args.nllb_command is None:
            nllb_parser.print_help()
            return 1
        if args.nllb_command == "convert":
            return cmd_nllb_convert(args)
        if args.nllb_command == "validate":
            return cmd_nllb_validate(args)
        if args.nllb_command == "benchmark":
            return cmd_nllb_benchmark(args)
        if args.nllb_command == "translate":
            return cmd_nllb_translate(args)
        if args.nllb_command == "list":
            return cmd_nllb_list(args)
    elif args.command == "kokoro":
        if args.kokoro_command is None:
            kokoro_parser.print_help()
            return 1
        if args.kokoro_command == "convert":
            return cmd_kokoro_convert(args)
        if args.kokoro_command == "validate":
            return cmd_kokoro_validate(args)
        if args.kokoro_command == "list":
            return cmd_kokoro_list(args)
        if args.kokoro_command == "synthesize":
            return cmd_kokoro_synthesize(args)
    elif args.command == "cosyvoice2":
        if args.cosyvoice2_command is None:
            cosyvoice2_parser.print_help()
            return 1
        if args.cosyvoice2_command == "convert":
            return cmd_cosyvoice2_convert(args)
        if args.cosyvoice2_command == "inspect":
            return cmd_cosyvoice2_inspect(args)
        if args.cosyvoice2_command == "validate":
            return cmd_cosyvoice2_validate(args)
        if args.cosyvoice2_command == "synthesize":
            return cmd_cosyvoice2_synthesize(args)
        if args.cosyvoice2_command == "list":
            return cmd_cosyvoice2_list(args)
    elif args.command == "whisper":
        if args.whisper_command is None:
            whisper_parser.print_help()
            return 1
        if args.whisper_command == "transcribe":
            return cmd_whisper_transcribe(args)
        if args.whisper_command == "benchmark":
            return cmd_whisper_benchmark(args)
        if args.whisper_command == "list":
            return cmd_whisper_list(args)
    elif args.command == "wakeword":
        if args.wakeword_command is None:
            wakeword_parser.print_help()
            return 1
        if args.wakeword_command == "status":
            return cmd_wakeword_status(args)
        if args.wakeword_command == "analyze":
            return cmd_wakeword_analyze(args)
        if args.wakeword_command == "detect":
            return cmd_wakeword_detect(args)
        if args.wakeword_command == "benchmark":
            return cmd_wakeword_benchmark(args)
        if args.wakeword_command == "convert":
            return cmd_wakeword_convert(args)

    return 1


if __name__ == "__main__":
    sys.exit(main())
