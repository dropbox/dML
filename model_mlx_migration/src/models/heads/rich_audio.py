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
Rich audio multi-head wrapper for Zipformer encoder output.

This is a convenience module that instantiates and runs all Phase 5-6 heads
against a shared encoder representation:

Phase 5 (Rich Audio):
- Emotion, Paralinguistics, Language (utterance-level)
- Pitch, Phoneme (frame-level)
- Singing (utterance-level binary + technique)
- Timestamp (frame-level boundaries + optional offsets)
- Hallucination (derived from phoneme logits + optional text/energy)

Phase 6 (Speaker):
- Speaker embeddings (DELULU-style, <0.8% EER target)
"""

from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn

from .emotion import EmotionConfig, EmotionHead
from .hallucination import HallucinationConfig, HallucinationHead
from .language import LanguageConfig, LanguageHead
from .paralinguistics import ParalinguisticsConfig, ParalinguisticsHead
from .phoneme import PhonemeConfig, PhonemeHead
from .pitch import PitchConfig, PitchHead
from .singing import SingingConfig, SingingHead
from .speaker import SpeakerConfig, SpeakerHead
from .timestamp import TimestampConfig, TimestampHead


@dataclass
class RichAudioHeadsConfig:
    """Configuration for RichAudioHeads."""

    encoder_dim: int = 384
    timestamp_use_offset_regression: bool = True

    # Hallucination head thresholds/weights can be overridden if desired.
    hallucination: HallucinationConfig = field(default_factory=HallucinationConfig)

    # Speaker embedding configuration (Phase 6 - DELULU)
    speaker_enabled: bool = True
    speaker: SpeakerConfig = field(default_factory=SpeakerConfig)


class RichAudioHeads(nn.Module):
    """
    Multi-head wrapper that runs all Phase 5-6 rich audio heads.

    Returned dict intentionally uses explicit keys to keep downstream loss and
    evaluation code simple and stable.
    """

    def __init__(self, config: RichAudioHeadsConfig | None = None):
        super().__init__()
        if config is None:
            config = RichAudioHeadsConfig()
        self.config = config

        encoder_dim = config.encoder_dim

        # Phase 5 heads
        self.emotion_head = EmotionHead(EmotionConfig(encoder_dim=encoder_dim))
        self.pitch_head = PitchHead(PitchConfig(encoder_dim=encoder_dim))
        self.phoneme_head = PhonemeHead(PhonemeConfig(encoder_dim=encoder_dim))
        self.paralinguistics_head = ParalinguisticsHead(
            ParalinguisticsConfig(encoder_dim=encoder_dim),
        )
        self.language_head = LanguageHead(LanguageConfig(encoder_dim=encoder_dim))
        self.singing_head = SingingHead(SingingConfig(encoder_dim=encoder_dim))
        self.timestamp_head = TimestampHead(
            TimestampConfig(
                encoder_dim=encoder_dim,
                use_offset_regression=config.timestamp_use_offset_regression,
            ),
        )
        self.hallucination_head = HallucinationHead(config.hallucination)

        # Phase 6 heads
        self.speaker_head: SpeakerHead | None = None
        if config.speaker_enabled:
            speaker_config = SpeakerConfig(encoder_dim=encoder_dim)
            # Override with any user-provided config values
            if config.speaker:
                speaker_config = SpeakerConfig(
                    encoder_dim=encoder_dim,
                    embedding_dim=config.speaker.embedding_dim,
                    num_speakers=config.speaker.num_speakers,
                    res2net_scale=config.speaker.res2net_scale,
                    se_reduction=config.speaker.se_reduction,
                    hidden_dim=config.speaker.hidden_dim,
                    aam_margin=config.speaker.aam_margin,
                    aam_scale=config.speaker.aam_scale,
                )
            self.speaker_head = SpeakerHead(speaker_config)

    def __call__(
        self,
        encoder_out: mx.array,
        encoder_lengths: mx.array | None = None,
        *,
        asr_text: list[str] | None = None,
        audio_energy: mx.array | None = None,
    ) -> dict[str, mx.array]:
        """
        Run all heads.

        Args:
            encoder_out: Encoder output (batch, seq, encoder_dim)
            encoder_lengths: Optional lengths (batch,)
            asr_text: Optional ASR text strings (batch,)
            audio_energy: Optional frame-level energy in dB (batch, seq)

        Returns:
            Dict of outputs with keys:
              - emotion_logits: (batch, 8)
              - language_logits: (batch, num_languages)
              - paralinguistics_logits: (batch, num_classes)
              - pitch_f0_hz: (batch, seq)
              - pitch_voiced_logits: (batch, seq)
              - phoneme_logits: (batch, seq, num_phonemes)
              - singing_binary_logits: (batch, 1)
              - singing_technique_logits: (batch, num_techniques)
              - timestamp_boundary_logits: (batch, seq, 1)
              - timestamp_offset_preds: (batch, seq, 2) or None (omitted)
              - hallucination_scores: (batch,)
              - speaker_embeddings: (batch, embedding_dim) if speaker_enabled
        """
        outputs: dict[str, mx.array] = {}

        # Utterance-level
        outputs["emotion_logits"] = self.emotion_head(encoder_out, encoder_lengths)
        outputs["language_logits"] = self.language_head(encoder_out, encoder_lengths)
        outputs["paralinguistics_logits"] = self.paralinguistics_head(encoder_out, encoder_lengths)

        # Frame-level / mixed
        pitch_f0, pitch_voiced = self.pitch_head(encoder_out, encoder_lengths)
        outputs["pitch_f0_hz"] = pitch_f0
        outputs["pitch_voiced_logits"] = pitch_voiced

        phoneme_logits = self.phoneme_head(encoder_out, encoder_lengths)
        outputs["phoneme_logits"] = phoneme_logits

        singing_binary, singing_technique = self.singing_head(encoder_out, encoder_lengths)
        outputs["singing_binary_logits"] = singing_binary
        outputs["singing_technique_logits"] = singing_technique

        ts_boundary, ts_offsets = self.timestamp_head(encoder_out, encoder_lengths)
        outputs["timestamp_boundary_logits"] = ts_boundary
        if ts_offsets is not None:
            outputs["timestamp_offset_preds"] = ts_offsets

        outputs["hallucination_scores"] = self.hallucination_head(
            phoneme_logits,
            asr_text=asr_text,
            audio_energy=audio_energy,
        )

        # Phase 6: Speaker embeddings (DELULU)
        if self.speaker_head is not None:
            outputs["speaker_embeddings"] = self.speaker_head.extract_embedding(
                encoder_out, encoder_lengths,
            )

        return outputs
