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
MLX implementation of Zipformer ASR model.

Ported from k2-fsa/icefall for streaming speech recognition.
"""

# Complete ASR Model
from .asr_model import ASRModel, ASRModelConfig, load_checkpoint
from .convert_weights import (
    analyze_checkpoint,
    convert_encoder_weights,
    convert_full_model,
    extract_encoder_config,
)

# Decoder and Joiner
from .decoder import Decoder, DecoderConfig

# Decoding algorithms
from .decoding import (
    DecodingResult,
    StreamingDecoderState,
    TokenDecoder,
    greedy_search,
    greedy_search_batch,
    greedy_search_streaming,
    modified_beam_search,
)
from .encoder import (
    BypassModule,
    ChunkCausalDepthwiseConv1d,
    CompactRelPositionalEncoding,
    ConvolutionModule,
    DownsampledZipformer2Encoder,
    FeedforwardModule,
    NonlinAttention,
    RelPositionMultiheadAttentionWeights,
    SelfAttention,
    SimpleDownsample,
    SimpleUpsample,
    Zipformer2Encoder,
    Zipformer2EncoderLayer,
)
from .encoder_pretrained import (
    BasicNorm as PretrainedNorm,
)
from .encoder_pretrained import (
    ConvolutionModule as PretrainedConvolution,
)
from .encoder_pretrained import (
    FeedforwardModule as PretrainedFeedforward,
)
from .encoder_pretrained import (
    PoolingModule as PretrainedPooling,
)
from .encoder_pretrained import (
    RelPositionMultiheadAttention as PretrainedAttention,
)

# Pretrained model support
from .encoder_pretrained import (
    ZipformerEncoderLayer as PretrainedEncoderLayer,
)
from .encoder_pretrained import (
    load_encoder_layer_weights,
)

# Feature extraction
from .features import (
    FbankConfig,
    FbankExtractor,
    load_audio,
)

# Inference pipeline
from .inference import (
    ASRConfig,
    ASRPipeline,
    StreamingState,
    transcribe_file,
)
from .joiner import Joiner, JoinerConfig
from .scaling import (
    ActivationDropoutAndLinear,
    Balancer,
    BiasNorm,
    Identity,
    ScaledLinear,
    Whiten,
)

# Full model implementation
from .zipformer import (
    Conv2dSubsampling,
    Zipformer,
    ZipformerConfig,
    load_zipformer_weights,
)
from .zipformer import (
    DownsampledZipformer2Encoder as FullDownsampledEncoder,
)
from .zipformer import (
    Zipformer2Encoder as FullEncoder,
)

__all__ = [
    # Scaling modules
    "BiasNorm",
    "ScaledLinear",
    "Balancer",
    "Whiten",
    "Identity",
    "ActivationDropoutAndLinear",
    # Encoder components
    "BypassModule",
    "ChunkCausalDepthwiseConv1d",
    "CompactRelPositionalEncoding",
    "ConvolutionModule",
    "DownsampledZipformer2Encoder",
    "FeedforwardModule",
    "NonlinAttention",
    "RelPositionMultiheadAttentionWeights",
    "SelfAttention",
    "SimpleDownsample",
    "SimpleUpsample",
    "Zipformer2Encoder",
    "Zipformer2EncoderLayer",
    # Pretrained model support
    "PretrainedEncoderLayer",
    "PretrainedAttention",
    "PretrainedPooling",
    "PretrainedFeedforward",
    "PretrainedConvolution",
    "PretrainedNorm",
    "load_encoder_layer_weights",
    # Weight conversion utilities
    "analyze_checkpoint",
    "extract_encoder_config",
    "convert_encoder_weights",
    "convert_full_model",
    # Full model
    "Zipformer",
    "ZipformerConfig",
    "Conv2dSubsampling",
    "FullEncoder",
    "FullDownsampledEncoder",
    "load_zipformer_weights",
    # Decoder and Joiner
    "Decoder",
    "DecoderConfig",
    "Joiner",
    "JoinerConfig",
    # Complete ASR Model
    "ASRModel",
    "ASRModelConfig",
    "load_checkpoint",
    # Decoding
    "greedy_search",
    "greedy_search_batch",
    "greedy_search_streaming",
    "modified_beam_search",
    "DecodingResult",
    "StreamingDecoderState",
    "TokenDecoder",
    # Feature extraction
    "FbankConfig",
    "FbankExtractor",
    "load_audio",
    # Inference pipeline
    "ASRConfig",
    "ASRPipeline",
    "StreamingState",
    "transcribe_file",
]
