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
WhisperMLX Quality Dial System
==============================

Unified 0-1 quality dial for each optimization dimension.

- 0.0 = Perfect quality, no optimization (slowest)
- 1.0 = Maximum optimization (fastest, quality cost)

Each dial has:
- scale: 0.0 to 1.0
- units: what the dial controls
- quality_cost: estimated WER increase at dial=1.0
- speed_gain: estimated speedup at dial=1.0

Usage:
    from tools.whisper_mlx.quality_dial import QualityPreset, WhisperQualityConfig

    # Use preset
    config = QualityPreset.BALANCED.config()

    # Or tune manually
    config = WhisperQualityConfig(
        model_dial=0.3,      # Use turbo model
        quant_dial=0.0,      # No quantization
        decode_dial=0.2,     # Beam=3
        vad_dial=0.5,        # Balanced VAD aggressiveness (VAD is always ON)
    )

    # Apply to model
    model = config.create_model()
    result = model.transcribe(audio, **config.transcribe_kwargs())

Note: VAD (Voice Activity Detection) is ALWAYS enabled (P0.1).
The vad_dial only controls aggressiveness (0=conservative, 3=aggressive).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


@dataclass
class DialSpec:
    """Specification for a single quality dial."""
    name: str
    description: str
    units: str
    quality_cost_at_1: float  # WER increase (percentage points) at dial=1.0
    speed_gain_at_1: float    # Speedup multiplier at dial=1.0

    # Mapping from dial value to actual parameter
    # List of (dial_threshold, param_value) tuples
    levels: list = field(default_factory=list)

    def get_value(self, dial: float) -> Any:
        """Get parameter value for given dial setting."""
        dial = max(0.0, min(1.0, dial))
        for threshold, value in reversed(self.levels):
            if dial >= threshold:
                return value
        return self.levels[0][1] if self.levels else None

    def get_quality_cost(self, dial: float) -> float:
        """Estimate quality cost (WER increase) at dial setting."""
        return self.quality_cost_at_1 * dial

    def get_speed_gain(self, dial: float) -> float:
        """Estimate speed gain at dial setting."""
        # Exponential interpolation for speed
        if self.speed_gain_at_1 <= 1.0:
            return 1.0
        return 1.0 + (self.speed_gain_at_1 - 1.0) * dial


# =============================================================================
# DIAL DEFINITIONS
# =============================================================================

MODEL_DIAL = DialSpec(
    name="model",
    description="Model size/architecture",
    units="decoder_layers",
    quality_cost_at_1=1.0,  # +1.0% WER at distil
    speed_gain_at_1=6.0,    # 6x speedup at distil
    levels=[
        (0.00, "large-v3"),        # 32 decoder layers - BEST
        (0.25, "large-v3-turbo"),  # 4 decoder layers
        (0.50, "distil-large-v3"), # 2 decoder layers
        (0.65, "medium"),          # 24 decoder layers (older arch)
        (0.80, "small"),           # 12 decoder layers
        (0.90, "base"),            # 6 decoder layers
        (1.00, "tiny"),            # 4 decoder layers (smallest)
    ],
)

QUANTIZATION_DIAL = DialSpec(
    name="quantization",
    description="Weight quantization precision",
    units="bits",
    quality_cost_at_1=2.0,  # +2.0% WER at INT4
    speed_gain_at_1=1.8,    # 1.8x speedup at INT4
    levels=[
        (0.00, 16),   # FP16 - no quantization
        (0.40, 8),    # INT8
        (0.70, 4),    # INT4
        (1.00, 2),    # INT2 (extreme, if supported)
    ],
)

DECODE_DIAL = DialSpec(
    name="decode",
    description="Decoding strategy (beam size)",
    units="beam_size",
    quality_cost_at_1=0.5,  # +0.5% WER at greedy
    speed_gain_at_1=2.0,    # 2x speedup at greedy
    levels=[
        (0.00, 5),    # Beam=5 (best quality)
        (0.25, 4),    # Beam=4
        (0.50, 3),    # Beam=3
        (0.75, 2),    # Beam=2
        (1.00, 1),    # Greedy (fastest)
    ],
)

VAD_DIAL = DialSpec(
    name="vad",
    description="Voice Activity Detection aggressiveness (VAD is ALWAYS ON)",
    units="aggressiveness(0-3)",
    quality_cost_at_1=0.1,  # +0.1% WER at aggressive VAD
    speed_gain_at_1=3.0,    # Up to 3x with 66% silence (audio dependent)
    levels=[
        # VAD is ALWAYS ON (P0.1) - dial controls aggressiveness only
        # 0=most conservative (keeps more audio), 3=most aggressive (filters more)
        (0.00, 0),     # Most conservative (keeps more audio)
        (0.25, 1),     # Conservative
        (0.50, 2),     # Balanced (default)
        (0.75, 3),     # Aggressive
        (1.00, 3),     # Aggressive (same as 0.75)
    ],
)

KV_CACHE_DIAL = DialSpec(
    name="kv_cache",
    description="KV cache quantization",
    units="quantize_kv",
    quality_cost_at_1=0.0,  # 0% WER impact (verified lossless)
    speed_gain_at_1=1.1,    # 1.1x speedup + 50% memory
    levels=[
        (0.00, False),  # FP16 KV cache
        (0.50, True),   # INT8 cross-attention KV (lossless)
        (1.00, True),   # INT8 all KV
    ],
)

THRESHOLD_DIAL = DialSpec(
    name="threshold",
    description="Quality thresholds (compression ratio, logprob)",
    units="strictness",
    quality_cost_at_1=0.2,  # +0.2% WER at relaxed
    speed_gain_at_1=1.1,    # 1.1x speedup (fewer retries)
    levels=[
        (0.00, {"compression_ratio": 2.4, "logprob": -1.0}),  # Strict
        (0.50, {"compression_ratio": 2.8, "logprob": -1.25}), # Moderate
        (1.00, {"compression_ratio": 3.5, "logprob": -1.5}),  # Relaxed
    ],
)

CONTEXT_DIAL = DialSpec(
    name="context",
    description="Condition on previous text",
    units="enabled",
    quality_cost_at_1=0.2,  # +0.2% WER without context
    speed_gain_at_1=1.1,    # 1.1x speedup
    levels=[
        (0.00, True),   # Use previous context
        (1.00, False),  # Independent segments
    ],
)

LANGUAGE_DIAL = DialSpec(
    name="language",
    description="Language detection vs fixed",
    units="detection",
    quality_cost_at_1=0.0,  # 0% if language known
    speed_gain_at_1=1.05,   # 1.05x speedup (skip detection)
    levels=[
        (0.00, None),   # Auto-detect
        (1.00, "en"),   # Fixed (set at runtime)
    ],
)

TIMESTAMP_DIAL = DialSpec(
    name="timestamp",
    description="Timestamp prediction",
    units="enabled",
    quality_cost_at_1=0.0,  # 0% WER impact
    speed_gain_at_1=1.15,   # 1.15x speedup
    levels=[
        (0.00, True),   # With timestamps
        (1.00, False),  # Without timestamps
    ],
)

COREML_DIAL = DialSpec(
    name="coreml",
    description="CoreML encoder backend (M4 Max: no speedup; M2/M3: may help)",
    units="backend",
    quality_cost_at_1=0.0,  # 0% WER impact (same encoder, different backend)
    speed_gain_at_1=1.04,   # 1.04x on M4 Max GPU mode (ANE is 3x slower)
    levels=[
        (0.00, "mlx"),          # Pure MLX (recommended for M4)
        (0.50, "coreml_gpu"),   # CoreML with GPU
        (1.00, "coreml_ane"),   # CoreML with ANE (slower on M4 Max!)
    ],
)

SLIDING_WINDOW_DIAL = DialSpec(
    name="sliding_window",
    description="Encoder sliding window attention (OPT-1.1-SW)",
    units="window_size",
    quality_cost_at_1=0.1,  # +0.1% WER at smallest window (needs verification)
    speed_gain_at_1=2.0,    # 2x encoder speedup at smallest window
    levels=[
        (0.00, None),   # Full attention (no sliding window)
        (0.25, 1024),   # Large window (conservative)
        (0.50, 512),    # Medium window (recommended)
        (0.75, 256),    # Small window (aggressive)
        (1.00, 128),    # Very small window (maximum speed)
    ],
)

# All dials
ALL_DIALS = {
    "model": MODEL_DIAL,
    "quantization": QUANTIZATION_DIAL,
    "decode": DECODE_DIAL,
    "vad": VAD_DIAL,
    "kv_cache": KV_CACHE_DIAL,
    "threshold": THRESHOLD_DIAL,
    "context": CONTEXT_DIAL,
    "language": LANGUAGE_DIAL,
    "timestamp": TIMESTAMP_DIAL,
    "coreml": COREML_DIAL,
    "sliding_window": SLIDING_WINDOW_DIAL,
}


# =============================================================================
# QUALITY CONFIG
# =============================================================================

@dataclass
class WhisperQualityConfig:
    """
    Configuration with 0-1 quality dials for each dimension.

    Each dial: 0.0 = best quality, 1.0 = maximum speed optimization
    """
    # Core dials (biggest impact)
    model_dial: float = 0.0       # 0=large-v3, 0.25=turbo, 0.5=distil, 1.0=tiny
    quantization_dial: float = 0.0  # 0=FP16, 0.4=INT8, 0.7=INT4
    decode_dial: float = 0.0      # 0=beam5, 0.5=beam3, 1.0=greedy
    vad_dial: float = 0.5         # VAD is ALWAYS ON. 0=conservative, 0.5=balanced, 1.0=aggressive

    # Fine-tuning dials (smaller impact)
    kv_cache_dial: float = 0.5    # 0=FP16, 0.5+=INT8 (lossless)
    threshold_dial: float = 0.0   # 0=strict, 1.0=relaxed
    context_dial: float = 0.0     # 0=use context, 1.0=independent
    language_dial: float = 0.0    # 0=auto-detect, 1.0=fixed
    timestamp_dial: float = 0.0   # 0=with timestamps, 1.0=without
    coreml_dial: float = 0.0      # 0=MLX (best for M4), 0.5=CoreML GPU, 1.0=CoreML ANE
    sliding_window_dial: float = 0.0  # 0=full attention, 0.5=512 window, 1.0=128 window

    # Override language (used when language_dial > 0)
    fixed_language: str | None = "en"

    def get_model_name(self) -> str:
        """Get model name for current dial setting."""
        return MODEL_DIAL.get_value(self.model_dial)

    def get_quantization_bits(self) -> int:
        """Get quantization bits for current dial setting."""
        return QUANTIZATION_DIAL.get_value(self.quantization_dial)

    def get_beam_size(self) -> int:
        """Get beam size for current dial setting."""
        return DECODE_DIAL.get_value(self.decode_dial)

    def get_vad_aggressiveness(self) -> int:
        """Get VAD aggressiveness for current dial setting.

        VAD is ALWAYS ON (P0.1). This returns the aggressiveness level:
        - 0: Most conservative (keeps more audio)
        - 1: Conservative
        - 2: Balanced (default)
        - 3: Aggressive (filters more)
        """
        return VAD_DIAL.get_value(self.vad_dial)

    def get_thresholds(self) -> dict[str, float]:
        """Get quality thresholds for current dial setting."""
        return THRESHOLD_DIAL.get_value(self.threshold_dial)

    def get_coreml_backend(self) -> str:
        """Get CoreML backend setting.

        Returns:
            "mlx" (pure MLX, recommended for M4),
            "coreml_gpu" (CoreML with GPU),
            "coreml_ane" (CoreML with ANE, slower on M4 Max!)
        """
        return COREML_DIAL.get_value(self.coreml_dial)

    def get_sliding_window_size(self) -> int | None:
        """Get encoder sliding window size for current dial setting.

        Returns:
            Window size (int) or None for full attention.
            Recommended values: 512 (balanced), 256 (aggressive)
        """
        return SLIDING_WINDOW_DIAL.get_value(self.sliding_window_dial)

    def estimate_quality_cost(self) -> float:
        """
        Estimate total quality cost (WER increase in percentage points).

        Returns:
            Estimated WER increase vs baseline (large-v3, no optimization)
        """
        total = 0.0
        total += MODEL_DIAL.get_quality_cost(self.model_dial)
        total += QUANTIZATION_DIAL.get_quality_cost(self.quantization_dial)
        total += DECODE_DIAL.get_quality_cost(self.decode_dial)
        total += VAD_DIAL.get_quality_cost(self.vad_dial)
        total += KV_CACHE_DIAL.get_quality_cost(self.kv_cache_dial)
        total += THRESHOLD_DIAL.get_quality_cost(self.threshold_dial)
        total += CONTEXT_DIAL.get_quality_cost(self.context_dial)
        total += SLIDING_WINDOW_DIAL.get_quality_cost(self.sliding_window_dial)
        # Language, timestamp, and coreml don't affect WER
        return total

    def estimate_speed_gain(self) -> float:
        """
        Estimate total speed gain multiplier.

        Returns:
            Estimated speedup vs baseline (large-v3, no optimization)
        """
        # Multiply gains (they compound)
        gain = 1.0
        gain *= MODEL_DIAL.get_speed_gain(self.model_dial)
        gain *= QUANTIZATION_DIAL.get_speed_gain(self.quantization_dial)
        gain *= DECODE_DIAL.get_speed_gain(self.decode_dial)
        gain *= VAD_DIAL.get_speed_gain(self.vad_dial)
        gain *= KV_CACHE_DIAL.get_speed_gain(self.kv_cache_dial)
        gain *= THRESHOLD_DIAL.get_speed_gain(self.threshold_dial)
        gain *= CONTEXT_DIAL.get_speed_gain(self.context_dial)
        gain *= LANGUAGE_DIAL.get_speed_gain(self.language_dial)
        gain *= TIMESTAMP_DIAL.get_speed_gain(self.timestamp_dial)
        # Sliding window provides encoder speedup (scaled by encoder fraction ~30%)
        sw_gain = SLIDING_WINDOW_DIAL.get_speed_gain(self.sliding_window_dial)
        gain *= 1.0 + (sw_gain - 1.0) * 0.3  # 30% of time is encoder
        # Note: CoreML speed gain is hardware-dependent
        # On M4 Max, coreml_ane is actually 0.35x (3x SLOWER)
        # On M2/M3, it may provide actual speedup
        # We don't include it in the estimate since it's so variable
        return gain

    def create_model(self):
        """
        Create WhisperMLX or HybridWhisperMLX model with current dial settings.

        When coreml_dial > 0, attempts to create HybridWhisperMLX (CoreML encoder + MLX decoder).
        Falls back to pure WhisperMLX if CoreML is unavailable.

        Returns:
            Configured WhisperMLX or HybridWhisperMLX model
        """
        from . import WhisperMLX

        model_name = self.get_model_name()
        quantize_kv = KV_CACHE_DIAL.get_value(self.kv_cache_dial)

        # Check if CoreML backend is requested
        coreml_backend = self.get_coreml_backend()
        use_coreml = coreml_backend in ("coreml_gpu", "coreml_ane")

        if use_coreml:
            model = self._create_hybrid_model(model_name, coreml_backend, quantize_kv)
            if model is not None:
                return model
            # Fall through to pure MLX if hybrid failed

        # Pure MLX model
        model = WhisperMLX.from_pretrained(
            model_name,
            quantize_kv=quantize_kv,
        )

        # Apply weight quantization if needed
        quant_bits = self.get_quantization_bits()
        if quant_bits < 16:
            model.quantize(bits=quant_bits)

        # Set VAD aggressiveness (VAD is always ON per P0.1)
        vad_agg = self.get_vad_aggressiveness()
        model._vad_aggressiveness = vad_agg

        # OPT-1.1-SW: Apply sliding window attention if enabled
        window_size = self.get_sliding_window_size()
        if window_size is not None:
            model.encoder.set_window_size(window_size)

        return model

    def _create_hybrid_model(self, model_name: str, coreml_backend: str, quantize_kv: bool):
        """
        Attempt to create HybridWhisperMLX with CoreML encoder.

        Args:
            model_name: Model name (e.g., "large-v3")
            coreml_backend: "coreml_gpu" or "coreml_ane"
            quantize_kv: Whether to quantize KV cache (unused for hybrid, KV cache is in MLX decoder)

        Returns:
            HybridWhisperMLX model, or None if unavailable
        """
        import logging
        logger = logging.getLogger(__name__)

        try:
            from ..whisper_ane import HybridWhisperMLX
        except ImportError:
            logger.warning(
                "HybridWhisperMLX not available (coremltools not installed?). "
                "Falling back to pure MLX.",
            )
            return None

        # Map backend to compute units
        compute_units = "CPU_AND_GPU" if coreml_backend == "coreml_gpu" else "CPU_AND_NE"

        try:
            model = HybridWhisperMLX.from_pretrained(
                model_name,
                compute_units=compute_units,
                auto_download=True,
            )
            logger.info("Created HybridWhisperMLX with compute_units=%s", compute_units)
            return model
        except Exception as e:
            logger.warning(
                "Failed to create HybridWhisperMLX: %s. Falling back to pure MLX.", e,
            )
            return None

    def transcribe_kwargs(self) -> dict[str, Any]:
        """
        Get transcribe() keyword arguments for current dial settings.

        Returns:
            Dict of kwargs to pass to model.transcribe()
        """
        kwargs = {}

        # Beam size
        kwargs["beam_size"] = self.get_beam_size()

        # Temperature (greedy = 0.0)
        kwargs["temperature"] = 0.0

        # VAD aggressiveness (VAD is always ON per P0.1)
        kwargs["vad_aggressiveness"] = self.get_vad_aggressiveness()

        # Thresholds
        thresholds = self.get_thresholds()
        kwargs["compression_ratio_threshold"] = thresholds["compression_ratio"]
        kwargs["logprob_threshold"] = thresholds["logprob"]

        # Context
        kwargs["condition_on_previous_text"] = CONTEXT_DIAL.get_value(self.context_dial)

        # Language
        if self.language_dial > 0.5:
            kwargs["language"] = self.fixed_language
        else:
            kwargs["language"] = None

        # Timestamps
        kwargs["word_timestamps"] = TIMESTAMP_DIAL.get_value(self.timestamp_dial)

        return kwargs

    def summary(self) -> str:
        """Get human-readable summary of current settings."""
        sw_size = self.get_sliding_window_size()
        sw_str = str(sw_size) if sw_size else "full"
        lines = [
            "WhisperMLX Quality Configuration",
            "=" * 40,
            "",
            "Dial Settings (0=quality, 1=speed):",
            f"  model:        {self.model_dial:.2f} → {self.get_model_name()}",
            f"  quantization: {self.quantization_dial:.2f} → {self.get_quantization_bits()}-bit",
            f"  decode:       {self.decode_dial:.2f} → beam={self.get_beam_size()}",
            f"  vad:          {self.vad_dial:.2f} → aggressiveness={self.get_vad_aggressiveness()}",
            f"  kv_cache:     {self.kv_cache_dial:.2f} → quantize={KV_CACHE_DIAL.get_value(self.kv_cache_dial)}",
            f"  threshold:    {self.threshold_dial:.2f}",
            f"  context:      {self.context_dial:.2f}",
            f"  language:     {self.language_dial:.2f}",
            f"  timestamp:    {self.timestamp_dial:.2f}",
            f"  coreml:       {self.coreml_dial:.2f} → {self.get_coreml_backend()}",
            f"  sliding_win:  {self.sliding_window_dial:.2f} → window={sw_str}",
            "",
            "Estimated Impact:",
            f"  Quality cost: +{self.estimate_quality_cost():.2f}% WER",
            f"  Speed gain:   {self.estimate_speed_gain():.1f}x",
        ]
        return "\n".join(lines)

    @classmethod
    def from_master_dial(cls, master: float, fixed_language: str = "en") -> "WhisperQualityConfig":
        """
        Create config from a single master dial (0-1).

        All individual dials are set proportionally to the master dial.

        Args:
            master: Master quality dial (0=best quality, 1=max speed)
            fixed_language: Language to use when language detection is disabled

        Returns:
            WhisperQualityConfig with all dials set
        """
        master = max(0.0, min(1.0, master))

        return cls(
            model_dial=master * 0.5,         # Max at turbo/distil, not tiny
            quantization_dial=master * 0.5,  # Max at INT8, not INT4
            decode_dial=master * 0.8,        # Allow greedy at high master
            vad_dial=master,                 # Full range
            kv_cache_dial=0.5,               # Always use INT8 KV (lossless)
            threshold_dial=master * 0.5,     # Moderate relaxation
            context_dial=master * 0.5,       # Moderate
            language_dial=master,            # Full range
            timestamp_dial=master * 0.5,     # Moderate
            fixed_language=fixed_language,
        )


# =============================================================================
# PRESETS
# =============================================================================

class QualityPreset(Enum):
    """Pre-defined quality presets."""

    # Maximum quality - no optimization
    MAXIMUM_QUALITY = "maximum_quality"

    # Maximum quality + LOSSLESS streaming optimizations
    MAXIMUM_QUALITY_STREAMING = "maximum_quality_streaming"

    # Balanced - good tradeoff
    BALANCED = "balanced"

    # Fast - prioritize speed
    FAST = "fast"

    # Ultra-fast - maximum speed
    ULTRA_FAST = "ultra_fast"

    # Streaming - optimized for streaming with silence
    STREAMING = "streaming"

    # Slightly degrade all - user's preferred approach
    SLIGHT_ALL = "slight_all"

    def config(self) -> WhisperQualityConfig:
        """Get WhisperQualityConfig for this preset."""
        if self == QualityPreset.MAXIMUM_QUALITY:
            return WhisperQualityConfig(
                model_dial=0.0,
                quantization_dial=0.0,
                decode_dial=0.0,
                vad_dial=0.0,         # Conservative VAD (VAD is always ON per P0.1)
                kv_cache_dial=0.5,    # Lossless optimization
                threshold_dial=0.0,
                context_dial=0.0,
                language_dial=0.0,
                timestamp_dial=0.0,
            )

        if self == QualityPreset.MAXIMUM_QUALITY_STREAMING:
            # BEST quality + ALL lossless optimizations for streaming
            # This is the "absolutely best performance, then optimize losslessly" preset
            return WhisperQualityConfig(
                model_dial=0.0,         # large-v3 (BEST model)
                quantization_dial=0.0,  # FP16 (no quantization)
                decode_dial=0.0,        # beam=5 (BEST decoding)
                vad_dial=1.0,           # FULL VAD (LOSSLESS - just skips silence)
                kv_cache_dial=0.5,      # INT8 KV cache (LOSSLESS - verified 0% WER impact)
                threshold_dial=0.0,     # Strict thresholds (BEST quality)
                context_dial=0.0,       # Use context (BEST quality)
                language_dial=1.0,      # Fixed language (LOSSLESS if language known)
                timestamp_dial=0.0,     # Keep timestamps (full features)
            )

        if self == QualityPreset.BALANCED:
            return WhisperQualityConfig(
                model_dial=0.25,       # turbo
                quantization_dial=0.0, # FP16
                decode_dial=0.5,       # beam=3
                vad_dial=0.5,          # balanced VAD
                kv_cache_dial=0.5,
                threshold_dial=0.25,
                context_dial=0.0,
                language_dial=0.5,     # fixed language
                timestamp_dial=0.0,
            )

        if self == QualityPreset.FAST:
            return WhisperQualityConfig(
                model_dial=0.25,        # turbo
                quantization_dial=0.4,  # INT8
                decode_dial=1.0,        # greedy
                vad_dial=0.75,          # aggressive VAD
                kv_cache_dial=0.5,
                threshold_dial=0.5,
                context_dial=1.0,       # no context
                language_dial=1.0,      # fixed language
                timestamp_dial=1.0,     # no timestamps
            )

        if self == QualityPreset.ULTRA_FAST:
            return WhisperQualityConfig(
                model_dial=0.5,         # distil
                quantization_dial=0.7,  # INT4
                decode_dial=1.0,        # greedy
                vad_dial=1.0,           # max VAD
                kv_cache_dial=0.5,
                threshold_dial=1.0,     # relaxed
                context_dial=1.0,
                language_dial=1.0,
                timestamp_dial=1.0,
            )

        if self == QualityPreset.STREAMING:
            return WhisperQualityConfig(
                model_dial=0.25,       # turbo
                quantization_dial=0.0, # FP16 for stability
                decode_dial=0.5,       # beam=3
                vad_dial=1.0,          # max VAD (lots of silence)
                kv_cache_dial=0.5,
                threshold_dial=0.0,    # strict for accuracy
                context_dial=0.0,      # use context for coherence
                language_dial=1.0,     # fixed language (faster)
                timestamp_dial=0.0,    # keep timestamps for streaming
            )

        if self == QualityPreset.SLIGHT_ALL:
            # User's philosophy: slightly degrade everything
            return WhisperQualityConfig(
                model_dial=0.25,        # turbo (4 decoder layers)
                quantization_dial=0.0,  # Keep FP16 (quality matters)
                decode_dial=0.5,        # beam=3
                vad_dial=0.5,           # balanced VAD
                kv_cache_dial=0.5,      # INT8 KV (lossless)
                threshold_dial=0.25,    # slightly relaxed
                context_dial=0.0,       # keep context
                language_dial=1.0,      # fixed language
                timestamp_dial=0.5,     # sometimes skip timestamps
            )

        raise ValueError(f"Unknown preset: {self}")

    def description(self) -> str:
        """Get human-readable description of preset."""
        descriptions = {
            QualityPreset.MAXIMUM_QUALITY: "Maximum quality. large-v3, beam=5, FP16, conservative VAD.",
            QualityPreset.MAXIMUM_QUALITY_STREAMING: "BEST quality + LOSSLESS streaming. large-v3, beam=5, VAD, INT8 KV.",
            QualityPreset.BALANCED: "Balanced quality/speed. turbo, beam=3, balanced VAD.",
            QualityPreset.FAST: "Prioritize speed. turbo, INT8, greedy, aggressive VAD.",
            QualityPreset.ULTRA_FAST: "Maximum speed. distil, INT4, greedy, aggressive VAD.",
            QualityPreset.STREAMING: "Optimized for streaming with silence. turbo, aggressive VAD, context.",
            QualityPreset.SLIGHT_ALL: "Slightly degrade all components. Best quality/speed ratio.",
        }
        return descriptions.get(self, "Unknown preset")


# =============================================================================
# DIAL EXPLORER
# =============================================================================

def explore_dial_space(dial_name: str, steps: int = 5) -> None:
    """
    Print table showing dial values and their effects.

    Args:
        dial_name: Name of dial to explore
        steps: Number of steps to show
    """
    if dial_name not in ALL_DIALS:
        print(f"Unknown dial: {dial_name}")
        print(f"Available: {list(ALL_DIALS.keys())}")
        return

    dial = ALL_DIALS[dial_name]

    print(f"\n{dial.name.upper()} DIAL")
    print(f"Description: {dial.description}")
    print(f"Units: {dial.units}")
    print(f"Max quality cost: +{dial.quality_cost_at_1}% WER")
    print(f"Max speed gain: {dial.speed_gain_at_1}x")
    print()
    print(f"{'Dial':>6} | {'Value':>20} | {'WER Cost':>10} | {'Speed':>8}")
    print("-" * 55)

    for i in range(steps + 1):
        d = i / steps
        value = dial.get_value(d)
        cost = dial.get_quality_cost(d)
        speed = dial.get_speed_gain(d)
        print(f"{d:>6.2f} | {str(value):>20} | +{cost:>8.2f}% | {speed:>7.2f}x")


def compare_presets() -> None:
    """Print comparison table of all presets."""
    print("\nQUALITY PRESET COMPARISON")
    print("=" * 80)
    print()
    print(f"{'Preset':<20} | {'Model':<15} | {'Quant':>6} | {'Beam':>5} | {'WER Cost':>10} | {'Speed':>8}")
    print("-" * 80)

    for preset in QualityPreset:
        cfg = preset.config()
        print(f"{preset.name:<20} | {cfg.get_model_name():<15} | {cfg.get_quantization_bits():>4}-bit | {cfg.get_beam_size():>5} | +{cfg.estimate_quality_cost():>8.2f}% | {cfg.estimate_speed_gain():>7.1f}x")

    print()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Demo the dial system
    print("WhisperMLX Quality Dial System Demo")
    print("=" * 50)

    # Show all presets
    compare_presets()

    # Explore model dial
    explore_dial_space("model")

    # Show master dial examples
    print("\nMASTER DIAL EXAMPLES")
    print("=" * 50)

    for master in [0.0, 0.25, 0.5, 0.75, 1.0]:
        cfg = WhisperQualityConfig.from_master_dial(master)
        print(f"\nMaster dial = {master:.2f}")
        print(f"  Model: {cfg.get_model_name()}")
        print(f"  Quant: {cfg.get_quantization_bits()}-bit")
        print(f"  Beam:  {cfg.get_beam_size()}")
        print(f"  Est. WER cost: +{cfg.estimate_quality_cost():.2f}%")
        print(f"  Est. speed:    {cfg.estimate_speed_gain():.1f}x")
