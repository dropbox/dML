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
MLX Model Implementations

Contains MLX implementations of model architectures for conversion.
"""

from .audio_postprocess import (
    AudioQualityConfig,
    AudioQualityMetrics,
    AudioQualityPipeline,
    apply_limiter,
    dc_offset_removal,
    deess,
    fade_edges,
    normalize_loudness,
    remove_clicks,
)
from .cosyvoice2 import CosyVoice2Config, CosyVoice2Model
from .cosyvoice2_flow import (
    CausalMaskedDiffWithXvec,
    FlowMatchingConfig,
    MaskedDiffWithXvec,
)
from .cosyvoice2_llm import CosyVoice2LLM, Qwen2Config
from .cosyvoice2_tokenizer import CosyVoice2Tokenizer, CosyVoice2TokenizerConfig
from .cosyvoice2_vocoder import HiFiGANConfig, HiFiGANVocoder
from .kokoro import KokoroConfig, KokoroModel
from .kokoro_modules import (
    AdaIN,
    AdainResBlk1d,
    AdaINResBlock1dStyled,
    AdaLayerNorm,
    BiLSTM,
    CustomLayerNorm,
    ResBlock1d,
    WeightNormConv1d,
)
from .nllb import NLLBConfig, NLLBModel

__all__ = [
    "NLLBConfig",
    "NLLBModel",
    "KokoroConfig",
    "KokoroModel",
    "WeightNormConv1d",
    "CustomLayerNorm",
    "AdaIN",
    "AdaLayerNorm",
    "AdainResBlk1d",
    "AdaINResBlock1dStyled",
    "BiLSTM",
    "ResBlock1d",
    "HiFiGANConfig",
    "HiFiGANVocoder",
    "Qwen2Config",
    "CosyVoice2LLM",
    "FlowMatchingConfig",
    "MaskedDiffWithXvec",
    "CausalMaskedDiffWithXvec",
    "CosyVoice2Config",
    "CosyVoice2Model",
    "CosyVoice2Tokenizer",
    "CosyVoice2TokenizerConfig",
    "AudioQualityConfig",
    "AudioQualityPipeline",
    "AudioQualityMetrics",
    "remove_clicks",
    "normalize_loudness",
    "apply_limiter",
    "dc_offset_removal",
    "fade_edges",
    "deess",
]
