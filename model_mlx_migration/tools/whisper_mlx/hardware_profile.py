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
Hardware Profile Detection and Adaptive Settings

Auto-detects Apple Silicon capabilities and sets appropriate defaults.
Like video game graphics auto-detection.

Usage:
    from tools.whisper_mlx.hardware_profile import detect_hardware, get_recommended_settings

    profile = detect_hardware()
    settings = get_recommended_settings(profile)
"""

import json
import os
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class ChipTier(Enum):
    """Hardware capability tiers."""
    ULTRA = "ultra"      # M1/M2/M3/M4 Ultra - datacenter class
    MAX = "max"          # M1/M2/M3/M4 Max - workstation class
    PRO = "pro"          # M1/M2/M3/M4 Pro - professional class
    BASE = "base"        # M1/M2/M3/M4 base - consumer class
    UNKNOWN = "unknown"


class PerformanceMode(Enum):
    """Like video game presets."""
    ULTRA_QUALITY = "ultra_quality"  # Best accuracy, high resources
    QUALITY = "quality"              # Good accuracy, moderate resources
    BALANCED = "balanced"            # Balance of speed and accuracy
    PERFORMANCE = "performance"      # Fast, lean (DEFAULT)
    EFFICIENCY = "efficiency"        # Minimum resources


@dataclass
class HardwareProfile:
    """Detected hardware capabilities."""
    chip_name: str
    chip_tier: ChipTier
    chip_generation: int  # 1, 2, 3, 4 for M1, M2, M3, M4
    gpu_cores: int
    cpu_cores: int
    memory_gb: float
    memory_bandwidth_gbps: float
    neural_engine_tops: float
    has_media_engine: bool

    def __str__(self):
        return f"{self.chip_name} ({self.memory_gb:.0f}GB, {self.gpu_cores} GPU cores)"


@dataclass
class RecommendedSettings:
    """Adaptive settings based on hardware."""
    # Model selection
    whisper_model: str           # "large-v3", "medium", "small", "tiny"
    use_ctc_head: bool
    use_emotion_head: bool
    use_singing_head: bool
    use_pitch_head: bool

    # Quantization
    use_quantization: bool
    quantization_bits: int       # 4, 8, or 16

    # Resource limits
    max_memory_gb: float
    max_gpu_percent: int
    idle_memory_mb: int

    # Ensemble
    enable_ensemble: bool
    ensemble_models: list

    # Latency targets
    target_first_partial_ms: int
    target_final_ms: int

    # Surge capacity
    can_surge: bool              # Can temporarily use more resources
    surge_max_memory_gb: float
    surge_max_gpu_percent: int


def detect_hardware() -> HardwareProfile:
    """Auto-detect Apple Silicon hardware."""

    chip_name = "Unknown"
    chip_tier = ChipTier.UNKNOWN
    chip_generation = 0
    gpu_cores = 8
    cpu_cores = os.cpu_count() or 8
    memory_gb = 8.0
    memory_bandwidth = 100.0
    neural_engine_tops = 15.0
    has_media_engine = False

    try:
        # Get chip name
        result = subprocess.run(
            ['sysctl', '-n', 'machdep.cpu.brand_string'],
            capture_output=True, text=True,
        )
        chip_name = result.stdout.strip()

        # Get memory
        result = subprocess.run(
            ['sysctl', '-n', 'hw.memsize'],
            capture_output=True, text=True,
        )
        memory_gb = int(result.stdout.strip()) / (1024**3)

        # Parse chip info
        chip_lower = chip_name.lower()

        # Detect generation
        if 'm4' in chip_lower:
            chip_generation = 4
        elif 'm3' in chip_lower:
            chip_generation = 3
        elif 'm2' in chip_lower:
            chip_generation = 2
        elif 'm1' in chip_lower:
            chip_generation = 1

        # Detect tier
        if 'ultra' in chip_lower:
            chip_tier = ChipTier.ULTRA
        elif 'max' in chip_lower:
            chip_tier = ChipTier.MAX
        elif 'pro' in chip_lower:
            chip_tier = ChipTier.PRO
        elif chip_generation > 0:
            chip_tier = ChipTier.BASE

        # Estimate GPU cores and bandwidth based on chip
        gpu_bandwidth_map = {
            # (tier, generation): (gpu_cores, bandwidth_gbps, neural_tops)
            (ChipTier.ULTRA, 4): (80, 819, 38),
            (ChipTier.MAX, 4): (40, 546, 38),
            (ChipTier.PRO, 4): (20, 273, 38),
            (ChipTier.BASE, 4): (10, 120, 38),

            (ChipTier.ULTRA, 3): (76, 819, 35),
            (ChipTier.MAX, 3): (40, 400, 35),
            (ChipTier.PRO, 3): (18, 200, 35),
            (ChipTier.BASE, 3): (10, 100, 35),

            (ChipTier.ULTRA, 2): (76, 800, 31),
            (ChipTier.MAX, 2): (38, 400, 31),
            (ChipTier.PRO, 2): (19, 200, 31),
            (ChipTier.BASE, 2): (10, 100, 31),

            (ChipTier.ULTRA, 1): (64, 800, 16),
            (ChipTier.MAX, 1): (32, 400, 16),
            (ChipTier.PRO, 1): (16, 200, 16),
            (ChipTier.BASE, 1): (8, 68, 16),
        }

        key = (chip_tier, chip_generation)
        if key in gpu_bandwidth_map:
            gpu_cores, memory_bandwidth, neural_engine_tops = gpu_bandwidth_map[key]

        has_media_engine = chip_generation >= 1

    except Exception as e:
        print(f"Warning: Hardware detection failed: {e}")

    return HardwareProfile(
        chip_name=chip_name,
        chip_tier=chip_tier,
        chip_generation=chip_generation,
        gpu_cores=gpu_cores,
        cpu_cores=cpu_cores,
        memory_gb=memory_gb,
        memory_bandwidth_gbps=memory_bandwidth,
        neural_engine_tops=neural_engine_tops,
        has_media_engine=has_media_engine,
    )


def get_recommended_settings(
    profile: HardwareProfile,
    mode: PerformanceMode = PerformanceMode.BALANCED,
) -> RecommendedSettings:
    """
    Get recommended settings based on hardware and performance mode.

    Default mode is BALANCED (efficient, not wasteful).

    Philosophy: "Fast enough with minimum resources" not "fastest possible"
    Even on M4 Max, don't use 50% GPU if 10% achieves acceptable latency.
    """

    # Base settings by tier - CONSERVATIVE defaults
    # Philosophy: Use minimum resources needed, save headroom for surge
    tier_defaults = {
        ChipTier.ULTRA: {
            'whisper_model': 'large-v3',
            'use_ctc': True, 'use_emotion': True, 'use_singing': True, 'use_pitch': True,
            'quantization': False, 'quant_bits': 16,
            'max_memory': 2.0, 'max_gpu': 15, 'idle_memory': 50,  # Conservative normal
            'ensemble': False, 'ensemble_models': [],  # Only on surge
            'first_partial': 200, 'final': 400,
            'surge_memory': 8.0, 'surge_gpu': 50,  # Headroom for important tasks
        },
        ChipTier.MAX: {
            'whisper_model': 'large-v3',
            'use_ctc': True, 'use_emotion': True, 'use_singing': True, 'use_pitch': True,
            'quantization': False, 'quant_bits': 16,
            'max_memory': 2.0, 'max_gpu': 15, 'idle_memory': 50,  # Conservative normal
            'ensemble': False, 'ensemble_models': [],  # Only on surge
            'first_partial': 200, 'final': 400,
            'surge_memory': 6.0, 'surge_gpu': 40,  # Headroom for important tasks
        },
        ChipTier.PRO: {
            'whisper_model': 'large-v3',
            'use_ctc': True, 'use_emotion': True, 'use_singing': True, 'use_pitch': False,
            'quantization': False, 'quant_bits': 16,
            'max_memory': 2.0, 'max_gpu': 15, 'idle_memory': 50,
            'ensemble': False, 'ensemble_models': [],
            'first_partial': 200, 'final': 500,
            'surge_memory': 4.0, 'surge_gpu': 30,
        },
        ChipTier.BASE: {
            'whisper_model': 'medium',
            'use_ctc': True, 'use_emotion': True, 'use_singing': False, 'use_pitch': False,
            'quantization': True, 'quant_bits': 8,
            'max_memory': 2.0, 'max_gpu': 30, 'idle_memory': 50,
            'ensemble': False, 'ensemble_models': [],
            'first_partial': 250, 'final': 700,
            'surge_memory': 4.0, 'surge_gpu': 50,
        },
        ChipTier.UNKNOWN: {
            'whisper_model': 'small',
            'use_ctc': True, 'use_emotion': False, 'use_singing': False, 'use_pitch': False,
            'quantization': True, 'quant_bits': 4,
            'max_memory': 1.0, 'max_gpu': 20, 'idle_memory': 50,
            'ensemble': False, 'ensemble_models': [],
            'first_partial': 300, 'final': 1000,
            'surge_memory': 2.0, 'surge_gpu': 40,
        },
    }

    # Mode adjustments (multipliers)
    mode_adjustments = {
        PerformanceMode.ULTRA_QUALITY: {
            'memory_mult': 2.0, 'gpu_mult': 1.5, 'latency_mult': 2.0,
            'ensemble': True, 'quantization': False,
        },
        PerformanceMode.QUALITY: {
            'memory_mult': 1.5, 'gpu_mult': 1.2, 'latency_mult': 1.5,
            'ensemble': True, 'quantization': False,
        },
        PerformanceMode.BALANCED: {
            'memory_mult': 1.0, 'gpu_mult': 1.0, 'latency_mult': 1.0,
            'ensemble': None, 'quantization': None,  # Use tier default
        },
        PerformanceMode.PERFORMANCE: {  # DEFAULT
            'memory_mult': 0.7, 'gpu_mult': 0.7, 'latency_mult': 0.8,
            'ensemble': False, 'quantization': None,
        },
        PerformanceMode.EFFICIENCY: {
            'memory_mult': 0.5, 'gpu_mult': 0.5, 'latency_mult': 0.7,
            'ensemble': False, 'quantization': True,
        },
    }

    defaults = tier_defaults.get(profile.chip_tier, tier_defaults[ChipTier.UNKNOWN])
    adjustments = mode_adjustments[mode]

    # Apply adjustments
    max_memory = defaults['max_memory'] * adjustments['memory_mult']
    max_gpu = min(100, int(defaults['max_gpu'] * adjustments['gpu_mult']))
    first_partial = int(defaults['first_partial'] * adjustments['latency_mult'])
    final = int(defaults['final'] * adjustments['latency_mult'])

    # Handle ensemble and quantization overrides
    use_ensemble = adjustments['ensemble'] if adjustments['ensemble'] is not None else defaults['ensemble']
    use_quant = adjustments['quantization'] if adjustments['quantization'] is not None else defaults['quantization']

    # Memory constraints
    max_memory = min(max_memory, profile.memory_gb * 0.5)  # Never use more than 50% of RAM
    surge_memory = min(defaults['surge_memory'], profile.memory_gb * 0.75)

    return RecommendedSettings(
        whisper_model=defaults['whisper_model'],
        use_ctc_head=defaults['use_ctc'],
        use_emotion_head=defaults['use_emotion'],
        use_singing_head=defaults['use_singing'],
        use_pitch_head=defaults['use_pitch'],
        use_quantization=use_quant,
        quantization_bits=defaults['quant_bits'],
        max_memory_gb=max_memory,
        max_gpu_percent=max_gpu,
        idle_memory_mb=defaults['idle_memory'],
        enable_ensemble=use_ensemble,
        ensemble_models=defaults['ensemble_models'] if use_ensemble else [],
        target_first_partial_ms=first_partial,
        target_final_ms=final,
        can_surge=True,
        surge_max_memory_gb=surge_memory,
        surge_max_gpu_percent=defaults['surge_gpu'],
    )


def save_profile(profile: HardwareProfile, settings: RecommendedSettings, path: Path | None = None):
    """Save detected profile and settings."""
    if path is None:
        path = Path.home() / '.whisper_mlx' / 'hardware_profile.json'

    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        'hardware': {
            'chip_name': profile.chip_name,
            'chip_tier': profile.chip_tier.value,
            'chip_generation': profile.chip_generation,
            'gpu_cores': profile.gpu_cores,
            'cpu_cores': profile.cpu_cores,
            'memory_gb': profile.memory_gb,
        },
        'settings': {
            'whisper_model': settings.whisper_model,
            'use_ctc_head': settings.use_ctc_head,
            'use_emotion_head': settings.use_emotion_head,
            'use_singing_head': settings.use_singing_head,
            'use_pitch_head': settings.use_pitch_head,
            'use_quantization': settings.use_quantization,
            'max_memory_gb': settings.max_memory_gb,
            'max_gpu_percent': settings.max_gpu_percent,
            'enable_ensemble': settings.enable_ensemble,
            'target_first_partial_ms': settings.target_first_partial_ms,
            'target_final_ms': settings.target_final_ms,
        },
    }

    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

    return path


def print_profile(profile: HardwareProfile, settings: RecommendedSettings):
    """Pretty print hardware profile and settings."""
    print("=" * 60)
    print("HARDWARE PROFILE")
    print("=" * 60)
    print(f"Chip:        {profile.chip_name}")
    print(f"Tier:        {profile.chip_tier.value.upper()}")
    print(f"Generation:  M{profile.chip_generation}")
    print(f"GPU Cores:   {profile.gpu_cores}")
    print(f"CPU Cores:   {profile.cpu_cores}")
    print(f"Memory:      {profile.memory_gb:.0f} GB")
    print(f"Bandwidth:   {profile.memory_bandwidth_gbps:.0f} GB/s")
    print()
    print("=" * 60)
    print("RECOMMENDED SETTINGS (Performance Mode)")
    print("=" * 60)
    print(f"Model:       {settings.whisper_model}")
    print(f"CTC Head:    {'Yes' if settings.use_ctc_head else 'No'}")
    print(f"Emotion:     {'Yes' if settings.use_emotion_head else 'No'}")
    print(f"Singing:     {'Yes' if settings.use_singing_head else 'No'}")
    print(f"Pitch:       {'Yes' if settings.use_pitch_head else 'No'}")
    print(f"Quantized:   {'Yes' if settings.use_quantization else 'No'}")
    print(f"Ensemble:    {'Yes' if settings.enable_ensemble else 'No'}")
    print()
    print(f"Max Memory:  {settings.max_memory_gb:.1f} GB (normal)")
    print(f"Surge:       {settings.surge_max_memory_gb:.1f} GB (when needed)")
    print(f"Max GPU:     {settings.max_gpu_percent}% (normal)")
    print()
    print("Targets:")
    print(f"  First partial: <{settings.target_first_partial_ms}ms")
    print(f"  Final result:  <{settings.target_final_ms}ms")
    print("=" * 60)


if __name__ == "__main__":
    profile = detect_hardware()
    settings = get_recommended_settings(profile)
    print_profile(profile, settings)

    # Save to disk
    path = save_profile(profile, settings)
    print(f"\nSaved to: {path}")
