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
Audio Cleaning Pipeline - Phase 9

Implements adaptive audio preprocessing with ASR-informed routing:
- ConditionEstimator: SNR/reverb detection
- ContentClassifier: speech/singing/music/noise detection
- DenoiseWrapper: DeepFilterNet3 integration (with spectral subtraction fallback)
- AdaptiveRouter: ASR-informed enhancement selection
"""

from tools.audio_cleaning.adaptive_router import (
    AdaptiveRouter,
    DeepFilterNetDenoiser,
    EnhancementResult,
    SpectralSubtractionDenoiser,
)
from tools.audio_cleaning.condition_estimator import (
    AudioCondition,
    ConditionEstimator,
    ContentType,
    estimate_reverb_t60,
    estimate_snr_wada,
)
from tools.audio_cleaning.wpe_dereverb import (
    DereverbConfig,
    SpectralDereverberator,
    WPEConfig,
    WPEDereverberator,
)

__all__ = [
    # Condition estimation
    "AudioCondition",
    "ConditionEstimator",
    "ContentType",
    "estimate_snr_wada",
    "estimate_reverb_t60",
    # Adaptive routing
    "AdaptiveRouter",
    "EnhancementResult",
    "SpectralSubtractionDenoiser",
    "DeepFilterNetDenoiser",
    # Dereverberation
    "DereverbConfig",
    "SpectralDereverberator",
    "WPEConfig",
    "WPEDereverberator",
]
