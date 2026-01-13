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
WhisperMLX Optimization Presets - Configurable quality/speed tradeoffs.

This module provides a unified API for all WhisperMLX optimization dials,
allowing users to easily configure quality vs speed tradeoffs through
named presets or individual dial adjustments.

## Presets

- MAX_QUALITY: Best accuracy, slowest (large-v3, beam=5, FP16)
- BALANCED: Good balance (turbo, beam=3, FP16) - ~5x faster, +0.6% WER
- FAST: Prioritize speed (turbo, INT8 weights, greedy) - ~8x faster, +1.5% WER
- ULTRA_FAST: Maximum speed (distil, INT4, greedy, VAD) - ~20x faster, +4% WER

## Optimization Dials (10 total)

1. model_variant: Model architecture (large-v3, turbo, distil)
2. weight_bits: Weight quantization (None=FP16, 8=INT8, 4=INT4)
3. quantize_kv: INT8 KV cache (lossless, 50% memory reduction)
4. beam_size: Beam search width (1=greedy, >1=beam search via transcribe_beam)
5. temperature: Sampling temperature (0=greedy)
6. quality_thresholds: Compression ratio and logprob thresholds
7. condition_on_previous: Use previous window for context
8. timestamps: Whether to predict timestamps
9. language: Fixed language vs auto-detect
10. use_vad: Voice Activity Detection preprocessing [NOT YET IMPLEMENTED]

## Usage

```python
from tools.whisper_mlx import WhisperMLX
from tools.whisper_mlx.presets import OptimizationPreset, TranscriptionConfig

# Use a preset
model = WhisperMLX.from_preset(OptimizationPreset.BALANCED)
result = model.transcribe("audio.wav")

# Or customize individual dials
config = TranscriptionConfig.from_preset(OptimizationPreset.FAST)
config.temperature = 0.0  # Override to greedy
config.language = "en"    # Fixed language for speed
```
"""

from dataclasses import dataclass, field
from enum import Enum


class ModelVariant(Enum):
    """Available Whisper model variants with different speed/quality tradeoffs."""

    # Maximum quality - 32 decoder layers
    LARGE_V3 = "mlx-community/whisper-large-v3-mlx"

    # 4 decoder layers - 2.65x faster, +1.08% WER (from MODEL_VARIANT_BENCHMARK.md)
    LARGE_V3_TURBO = "mlx-community/whisper-large-v3-turbo"

    # 2 decoder layers - 3.03x faster, +0.68% WER (from MODEL_VARIANT_BENCHMARK.md)
    DISTIL_LARGE_V3 = "mlx-community/distil-whisper-large-v3"


class OptimizationPreset(Enum):
    """
    Named presets for common quality/speed configurations.

    Based on MANAGER research (commit e9ff49b):
    - MAX_QUALITY: Best WER, slowest
    - BALANCED: 5x speed, <1% WER loss (user philosophy: 'slightly degrade all')
    - FAST: 8x speed, ~1.5% WER loss
    - ULTRA_FAST: 20x speed, ~4% WER loss
    """
    MAX_QUALITY = "max_quality"
    BALANCED = "balanced"
    FAST = "fast"
    ULTRA_FAST = "ultra_fast"


@dataclass
class QualityThresholds:
    """
    Quality thresholds for temperature fallback.

    When transcription fails these thresholds, WhisperMLX will retry
    with higher temperature. Lower thresholds = more retries = better quality.
    """
    # If compression ratio > this, transcription likely has repetition loops
    compression_ratio: float | None = 2.4

    # If avg log probability < this, transcription is low confidence
    logprob: float | None = -1.0

    # If no_speech_prob > this and logprob < threshold, treat as silence
    no_speech: float | None = 0.6


@dataclass
class TranscriptionConfig:
    """
    Complete configuration for WhisperMLX transcription.

    This dataclass bundles all 10 optimization dials into a single
    configuration object that can be passed to transcribe().

    Attributes:
        model_variant: Which Whisper model to use
        weight_bits: Weight quantization (None=FP16, 8=INT8, 4=INT4)
        quantize_kv: Use INT8 KV cache (lossless, 50% memory reduction)
        beam_size: Beam search width (1=greedy, >1=beam search via transcribe_beam)
        temperature: Sampling temperature (0=deterministic greedy)
        temperature_fallback: Temperatures to try on quality failure
        quality_thresholds: Thresholds for temperature fallback
        condition_on_previous: Use previous window tokens as context
        timestamps: Whether to predict word/segment timestamps
        language: Fixed language code (None=auto-detect)
        use_vad: Use Voice Activity Detection [NOT IMPLEMENTED]
    """
    # Dial 1: Model selection (largest impact on speed/quality)
    model_variant: ModelVariant = ModelVariant.LARGE_V3

    # Dial 2: Weight quantization
    weight_bits: int | None = None  # None=FP16, 8=INT8, 4=INT4

    # Dial 3: KV cache quantization (lossless - verified 100% exact match)
    quantize_kv: bool = True

    # Dial 4: Beam search vs greedy
    # Use transcribe_beam() method for beam search decoding
    # beam_size > 1 improves quality at cost of beam_size * compute
    beam_size: int = 1

    # Dial 5: Temperature for sampling
    temperature: float = 0.0

    # Temperature fallback sequence for quality issues
    temperature_fallback: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

    # Dial 6: Quality thresholds
    quality_thresholds: QualityThresholds = field(default_factory=QualityThresholds)

    # Dial 7: Context conditioning for long audio
    condition_on_previous: bool = True

    # Dial 8: Timestamp prediction
    timestamps: bool = True

    # Dial 9: Language (None = auto-detect, string = fixed)
    language: str | None = None

    # Dial 10: VAD preprocessing
    # NOTE: NOT YET IMPLEMENTED in WhisperMLX
    # When implemented, will skip silence for 2-4x speedup
    use_vad: bool = False

    # Additional settings
    task: str = "transcribe"  # "transcribe" or "translate"
    max_initial_timestamp: float = 1.0
    verbose: bool = False

    @classmethod
    def from_preset(cls, preset: OptimizationPreset) -> "TranscriptionConfig":
        """
        Create a TranscriptionConfig from a named preset.

        Args:
            preset: One of the OptimizationPreset values

        Returns:
            TranscriptionConfig with appropriate dial settings

        Example:
            config = TranscriptionConfig.from_preset(OptimizationPreset.BALANCED)
            config.language = "en"  # Override auto-detect
        """
        if preset == OptimizationPreset.MAX_QUALITY:
            return cls(
                model_variant=ModelVariant.LARGE_V3,
                weight_bits=None,  # FP16
                quantize_kv=True,  # Lossless
                beam_size=5,  # Would use beam=5 if implemented
                temperature=0.0,
                temperature_fallback=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                quality_thresholds=QualityThresholds(
                    compression_ratio=2.4,
                    logprob=-1.0,
                    no_speech=0.6,
                ),
                condition_on_previous=True,
                timestamps=True,
                language=None,  # Auto-detect
                use_vad=False,
            )

        if preset == OptimizationPreset.BALANCED:
            # MANAGER's "slightly degrade all components" philosophy
            # turbo + beam=3 + fixed lang = 5x speed, <1% WER loss
            return cls(
                model_variant=ModelVariant.LARGE_V3_TURBO,
                weight_bits=None,  # FP16
                quantize_kv=True,
                beam_size=3,  # Would use beam=3 if implemented
                temperature=0.0,
                temperature_fallback=(0.0, 0.2, 0.4),  # Fewer fallbacks
                quality_thresholds=QualityThresholds(
                    compression_ratio=2.6,  # Slightly relaxed
                    logprob=-1.2,
                    no_speech=0.6,
                ),
                condition_on_previous=True,
                timestamps=True,
                language=None,
                use_vad=False,
            )

        if preset == OptimizationPreset.FAST:
            # turbo + INT8 weights + greedy = ~8x speed, +1.5% WER
            return cls(
                model_variant=ModelVariant.LARGE_V3_TURBO,
                weight_bits=8,  # INT8 weights
                quantize_kv=True,
                beam_size=1,  # Greedy
                temperature=0.0,
                temperature_fallback=(0.0,),  # No fallback
                quality_thresholds=QualityThresholds(
                    compression_ratio=3.0,  # Very relaxed
                    logprob=-1.5,
                    no_speech=0.7,
                ),
                condition_on_previous=False,  # Skip for speed
                timestamps=False,  # Skip for speed
                language=None,
                use_vad=False,
            )

        if preset == OptimizationPreset.ULTRA_FAST:
            # distil + INT4 + greedy + VAD = ~20x speed, +4% WER
            return cls(
                model_variant=ModelVariant.DISTIL_LARGE_V3,
                weight_bits=4,  # INT4 weights (aggressive quantization)
                quantize_kv=True,
                beam_size=1,  # Greedy
                temperature=0.0,
                temperature_fallback=(0.0,),  # No fallback
                quality_thresholds=QualityThresholds(
                    compression_ratio=None,  # Disabled
                    logprob=None,  # Disabled
                    no_speech=0.8,  # Very relaxed
                ),
                condition_on_previous=False,
                timestamps=False,
                language=None,
                use_vad=True,  # Would use VAD if implemented
            )

        raise ValueError(f"Unknown preset: {preset}")

    def get_model_name(self) -> str:
        """Get the HuggingFace model name for this config."""
        return self.model_variant.value

    def get_transcribe_kwargs(self) -> dict:
        """
        Get kwargs for WhisperMLX.transcribe() from this config.

        Returns:
            Dictionary of keyword arguments for transcribe()
        """
        # Build temperature tuple
        if self.temperature == 0.0 and len(self.temperature_fallback) > 0:
            temperature = self.temperature_fallback
        else:
            temperature = self.temperature

        return {
            "language": self.language,
            "task": self.task,
            "temperature": temperature,
            "max_initial_timestamp": self.max_initial_timestamp,
            "compression_ratio_threshold": self.quality_thresholds.compression_ratio,
            "logprob_threshold": self.quality_thresholds.logprob,
            "no_speech_threshold": self.quality_thresholds.no_speech,
            "verbose": self.verbose,
        }

    def get_transcribe_long_kwargs(self) -> dict:
        """
        Get kwargs for WhisperMLX.transcribe_long() from this config.

        Returns:
            Dictionary of keyword arguments for transcribe_long()
        """
        return {
            "language": self.language,
            "task": self.task,
            "temperature": self.temperature,
            "max_initial_timestamp": self.max_initial_timestamp,
            "condition_on_previous_text": self.condition_on_previous,
            "verbose": self.verbose,
        }

    def describe(self) -> str:
        """
        Get a human-readable description of this configuration.

        Returns:
            Multi-line string describing all dial settings
        """
        lines = [
            "WhisperMLX Configuration:",
            f"  Model: {self.model_variant.name}",
            f"  Weights: {'FP16' if self.weight_bits is None else f'INT{self.weight_bits}'}",
            f"  KV Cache: {'INT8' if self.quantize_kv else 'FP16'}",
            f"  Decoding: {'Greedy' if self.beam_size <= 1 else f'Beam {self.beam_size} (use transcribe_beam)'}",
            f"  Temperature: {self.temperature}",
            f"  Language: {'auto-detect' if self.language is None else self.language}",
            f"  Timestamps: {self.timestamps}",
            f"  Context: {self.condition_on_previous}",
            f"  VAD: {self.use_vad}" + (" [not implemented]" if self.use_vad else ""),
        ]
        if self.quality_thresholds.compression_ratio is not None:
            lines.append(f"  Compression threshold: {self.quality_thresholds.compression_ratio}")
        if self.quality_thresholds.logprob is not None:
            lines.append(f"  Logprob threshold: {self.quality_thresholds.logprob}")
        return "\n".join(lines)


# Preset descriptions for documentation
PRESET_INFO = {
    OptimizationPreset.MAX_QUALITY: {
        "description": "Maximum transcription quality, slowest speed",
        "model": "large-v3 (32 decoder layers)",
        "expected_wer": "~2.5% (baseline)",
        "expected_rtf": "~0.11x",
        "speedup": "1x (baseline)",
    },
    OptimizationPreset.BALANCED: {
        "description": "Good balance of speed and quality (recommended)",
        "model": "large-v3-turbo (4 decoder layers)",
        "expected_wer": "~3.1% (+0.6%)",
        "expected_rtf": "~0.04x",
        "speedup": "~2.7x",
    },
    OptimizationPreset.FAST: {
        "description": "Prioritize speed with acceptable quality loss",
        "model": "large-v3-turbo + INT8 weights",
        "expected_wer": "~4% (+1.5%)",
        "expected_rtf": "~0.03x",
        "speedup": "~4x",
    },
    OptimizationPreset.ULTRA_FAST: {
        "description": "Maximum speed for real-time applications",
        "model": "distil-large-v3 + INT4 weights",
        "expected_wer": "~6.5% (+4%)",
        "expected_rtf": "~0.02x",
        "speedup": "~5x",
    },
}


def list_presets() -> str:
    """
    Get a formatted list of all available presets.

    Returns:
        Multi-line string describing all presets
    """
    lines = ["Available WhisperMLX Optimization Presets:", ""]

    for preset in OptimizationPreset:
        info = PRESET_INFO[preset]
        lines.extend([
            f"  {preset.value.upper()}:",
            f"    {info['description']}",
            f"    Model: {info['model']}",
            f"    WER: {info['expected_wer']}, RTF: {info['expected_rtf']}",
            f"    Speedup: {info['speedup']}",
            "",
        ])

    return "\n".join(lines)
