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
Rich audio heads for Zipformer encoder.

These heads attach to the Zipformer encoder output to produce
additional predictions beyond ASR (emotion, pitch, phoneme, etc.).

Phase 5 heads:
- 5.1: EmotionHead - 8-class utterance-level emotion classification
- 5.2: PitchHead - Frame-level F0 prediction
- 5.3: PhonemeHead - Frame-level IPA phoneme classification
- 5.4: ParalinguisticsHead - 50-class non-speech event detection
- 5.5: LanguageHead - 9+ language identification
- 5.6: SingingHead - Binary singing detection + 10 techniques
- 5.7: TimestampHead - Word-level timestamp prediction
- 5.8: HallucinationHead - Phoneme mismatch-based hallucination detection

Phase 6 heads:
- 6.1: SpeakerHead - DELULU-style speaker embeddings (<0.8% EER target)
"""

from .emotion import EmotionConfig, EmotionHead, EmotionLoss, emotion_loss
from .hallucination import (
    HallucinationConfig,
    HallucinationHead,
    HallucinationLoss,
    HallucinationResult,
    compute_hallucination_metrics,
    hallucination_loss,
)
from .language import (
    CORE_LANGUAGES,
    EXTENDED_LANGUAGES,
    LANGUAGE_NAMES,
    LanguageConfig,
    LanguageHead,
    LanguageLoss,
    language_loss,
)
from .paralinguistics import (
    PARALINGUISTIC_CLASSES,
    ParalinguisticsConfig,
    ParalinguisticsHead,
    ParalinguisticsLoss,
    paralinguistics_loss,
)
from .phoneme import (
    IPA_PHONEMES,
    PhonemeConfig,
    PhonemeFrameLoss,
    PhonemeHead,
    phoneme_ce_loss,
)
from .pitch import PitchConfig, PitchHead, PitchLoss, pitch_loss
from .rich_audio import RichAudioHeads, RichAudioHeadsConfig
from .singing import (
    SINGING_TECHNIQUES,
    SingingConfig,
    SingingHead,
    SingingLoss,
    singing_loss,
)
from .speaker import (
    SpeakerConfig,
    SpeakerHead,
    SpeakerLoss,
    aam_softmax_loss,
    speaker_loss,
    verification_eer,
)
from .timestamp import (
    TimestampConfig,
    TimestampHead,
    TimestampLoss,
    WordTimestamp,
    timestamp_loss,
)

__all__ = [
    # Emotion (Phase 5.1)
    "EmotionHead",
    "EmotionConfig",
    "EmotionLoss",
    "emotion_loss",
    # Pitch (Phase 5.2)
    "PitchHead",
    "PitchConfig",
    "PitchLoss",
    "pitch_loss",
    # Phoneme (Phase 5.3)
    "PhonemeHead",
    "PhonemeConfig",
    "PhonemeFrameLoss",
    "phoneme_ce_loss",
    "IPA_PHONEMES",
    # Paralinguistics (Phase 5.4)
    "ParalinguisticsHead",
    "ParalinguisticsConfig",
    "ParalinguisticsLoss",
    "paralinguistics_loss",
    "PARALINGUISTIC_CLASSES",
    # Language (Phase 5.5)
    "LanguageHead",
    "LanguageConfig",
    "LanguageLoss",
    "language_loss",
    "CORE_LANGUAGES",
    "EXTENDED_LANGUAGES",
    "LANGUAGE_NAMES",
    # Singing (Phase 5.6)
    "SingingHead",
    "SingingConfig",
    "SingingLoss",
    "singing_loss",
    "SINGING_TECHNIQUES",
    # Timestamp (Phase 5.7)
    "TimestampHead",
    "TimestampConfig",
    "TimestampLoss",
    "timestamp_loss",
    "WordTimestamp",
    # Hallucination (Phase 5.8)
    "HallucinationHead",
    "HallucinationConfig",
    "HallucinationLoss",
    "hallucination_loss",
    "HallucinationResult",
    "compute_hallucination_metrics",
    # Speaker (Phase 6.1 - DELULU)
    "SpeakerHead",
    "SpeakerConfig",
    "SpeakerLoss",
    "speaker_loss",
    "aam_softmax_loss",
    "verification_eer",
    # Multi-head wrapper
    "RichAudioHeads",
    "RichAudioHeadsConfig",
]
