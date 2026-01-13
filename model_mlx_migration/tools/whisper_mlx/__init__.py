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
WhisperMLX - Custom MLX Whisper Implementation

A high-performance Whisper implementation for Apple Silicon with:
- Dynamic chunk sizing (variable-length audio without padding)
- Adjustable timestamp precision (fix for decoder hallucinations)
- Preallocated KV-cache (zero allocation during decoding)
- Optional speculative decoding (1.5-2x speedup)
- Encoder output caching (2x speedup for repeated queries)
- Fused attention operations
- Real-time streaming transcription with VAD

Target: 8-10x speedup over mlx-whisper with 100% lossless quality.
"""

from .audio import compute_stft_and_mel
from .beam_search import (
    BeamSearchDecoder,
    BeamSearchResult,
    decode_with_beam_search,
)
from .config import WHISPER_CONFIGS, WhisperConfig
from .decoding import (
    DecodingOptions,
    DecodingResult,
    HallucinationResult,
    RepetitionPenalty,
    detect_hallucination,
    is_hallucination,
)
from .dual_stream import (
    EventType,
    RichDualPathStreamer,
    RichDualStreamConfig,
    RichStreamConsumer,
    StreamEvent,
    TokenState,
    create_dual_stream_pipeline,
    event_to_dict,
)
from .encoder_cache import EncoderCache
from .encoder_vad import (
    EncoderVADHead,
    SileroVADDistiller,
    create_encoder_vad_head,
    load_encoder_vad_head,
    save_encoder_vad_head,
)
from .fused_vad import FusedVAD, detect_speech_fused
from .medusa import (
    MedusaHead,
    MedusaHeadBlock,
    MedusaModule,
    MedusaTreeVerifier,
    TreeCandidate,
    build_tree_attention_mask,
    create_medusa_module,
)
from .medusa_training import (
    MedusaTrainer,
    MedusaTrainingConfig,
    TrainingBatch,
    compute_medusa_loss,
    kl_divergence_loss,
    load_medusa_weights,
    save_medusa_weights,
)
from .model import TranscriptionCallbacks, WhisperMLX
from .presets import (
    ModelVariant,
    OptimizationPreset,
    QualityThresholds,
    TranscriptionConfig,
    list_presets,
)
from .prosody_beam_search import (
    ProsodyBeamSearch,
    ProsodyBeamSearchDecoder,
    ProsodyBoostConfig,
    ProsodyFeatures,
    StreamingProsodyDecoder,
    create_prosody_beam_search,
)
from .roundtrip_verification import (
    MelSimilarity,
    RoundTripVerifier,
    VerificationResult,
)
from .silero_vad import (
    SileroVADProcessor,
    SpeechSegment,
    VADResult,
    get_vad_processor,
    preprocess_audio_with_vad,
)
from .speculative import SpeculativeDecoder
from .streaming import (
    StreamingConfig,
    StreamingResult,
    StreamingWhisper,
    StreamState,
    SyncStreamingWhisper,
)

__version__ = "0.4.0"

__all__ = [
    "WhisperMLX",
    "TranscriptionCallbacks",  # GAP 44: Callback system
    "WhisperConfig",
    "WHISPER_CONFIGS",
    "DecodingOptions",
    "DecodingResult",
    "HallucinationResult",
    "RepetitionPenalty",
    "detect_hallucination",
    "is_hallucination",
    "SpeculativeDecoder",
    "BeamSearchDecoder",
    "BeamSearchResult",
    "decode_with_beam_search",
    "EncoderCache",
    # Presets (OPT-2.5)
    "OptimizationPreset",
    "TranscriptionConfig",
    "ModelVariant",
    "QualityThresholds",
    "list_presets",
    # Streaming
    "StreamingWhisper",
    "SyncStreamingWhisper",
    "StreamingConfig",
    "StreamingResult",
    "StreamState",
    # Fused VAD (OPT-SHARED-FFT)
    "FusedVAD",
    "detect_speech_fused",
    "compute_stft_and_mel",
    # Medusa multi-token (Phase 2)
    "MedusaHead",
    "MedusaHeadBlock",
    "MedusaModule",
    "MedusaTreeVerifier",
    "TreeCandidate",
    "build_tree_attention_mask",
    "create_medusa_module",
    # Medusa training (Phase 2.3)
    "MedusaTrainer",
    "MedusaTrainingConfig",
    "TrainingBatch",
    "compute_medusa_loss",
    "kl_divergence_loss",
    "load_medusa_weights",
    "save_medusa_weights",
    # Encoder VAD head (Phase 3)
    "EncoderVADHead",
    "SileroVADDistiller",
    "create_encoder_vad_head",
    "load_encoder_vad_head",
    "save_encoder_vad_head",
    # Silero VAD preprocessing (P0.1)
    "SileroVADProcessor",
    "SpeechSegment",
    "VADResult",
    "get_vad_processor",
    "preprocess_audio_with_vad",
    # Round-trip verification
    "RoundTripVerifier",
    "VerificationResult",
    "MelSimilarity",
    # Prosody-conditioned beam search
    "ProsodyBeamSearch",
    "ProsodyBeamSearchDecoder",
    "ProsodyBoostConfig",
    "ProsodyFeatures",
    "StreamingProsodyDecoder",
    "create_prosody_beam_search",
    # Dual-stream rich audio (Phase 7)
    "EventType",
    "StreamEvent",
    "TokenState",
    "RichStreamConsumer",
    "RichDualPathStreamer",
    "RichDualStreamConfig",
    "create_dual_stream_pipeline",
    "event_to_dict",
]
