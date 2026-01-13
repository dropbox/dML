# Copyright 2024-2026 Andrew Yates
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
RichToken output format for SOTA++ Voice Server.

RichToken is the unified output format that combines ASR transcription
with all rich audio features from Phase 5-6 heads:
- Text + confidence
- Emotion, pitch, phoneme
- Paralinguistics, language, singing
- Timestamps, speaker embedding
- Hallucination detection
"""

import json
from dataclasses import dataclass, field
from enum import Enum


class EmotionLabel(str, Enum):
    """8-class emotion labels (CREMA-D/RAVDESS)."""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEAR = "fear"
    DISGUST = "disgust"
    SURPRISE = "surprise"
    CONTEMPT = "contempt"


class ASRMode(str, Enum):
    """ASR inference mode."""
    STREAMING = "streaming"  # Zipformer only, <100ms latency
    HIGH_ACCURACY = "high_accuracy"  # ROVER voting, <1.5% WER


@dataclass
class WordTimestamp:
    """Word-level timestamp information."""
    word: str
    start_ms: float
    end_ms: float
    confidence: float = 1.0


@dataclass
class PitchInfo:
    """Pitch (F0) information for the utterance."""
    # Statistics over voiced frames
    mean_hz: float
    std_hz: float
    min_hz: float
    max_hz: float
    # Per-frame F0 values (optional, for detailed analysis)
    frame_f0_hz: list[float] | None = None
    # Voiced/unvoiced mask
    voiced_ratio: float = 1.0


@dataclass
class PhonemeInfo:
    """Phoneme-level information."""
    # Predicted phoneme sequence (IPA)
    phonemes: list[str]
    # Per-phoneme confidence scores
    confidences: list[float] = field(default_factory=list)
    # Frame-level phoneme posteriors (optional)
    frame_posteriors: list[list[float]] | None = None


@dataclass
class ParalinguisticsInfo:
    """Paralinguistic event detection (50 classes)."""
    # Top predicted event
    event: str
    confidence: float
    # All probabilities (optional)
    all_probs: dict[str, float] | None = None


@dataclass
class LanguageInfo:
    """Language identification results."""
    # Detected language code
    language: str
    confidence: float
    # All language probabilities (optional)
    all_probs: dict[str, float] | None = None


@dataclass
class SingingInfo:
    """Singing detection and technique identification."""
    is_singing: bool
    singing_confidence: float
    # Detected techniques if singing
    techniques: list[str] = field(default_factory=list)
    technique_confidences: list[float] = field(default_factory=list)


@dataclass
class SpeakerInfo:
    """Speaker embedding and identification."""
    # Speaker embedding vector (256-dim DELULU)
    embedding: list[float]
    # Matched speaker ID (if diarization active)
    speaker_id: str | None = None
    # Similarity to matched speaker
    similarity: float | None = None


@dataclass
class HallucinationInfo:
    """Hallucination detection results."""
    # Overall hallucination probability
    score: float
    # Flagged as likely hallucination
    is_hallucinated: bool
    # Components of the score
    phoneme_mismatch: float = 0.0
    energy_silence: float = 0.0
    repetition: float = 0.0


@dataclass
class RichToken:
    """
    Unified output format for SOTA++ Voice Server.

    Contains ASR transcription plus all rich audio features
    from the Zipformer encoder heads.
    """

    # Core ASR output
    text: str
    confidence: float

    # Timing
    start_time_ms: float
    end_time_ms: float

    # ASR metadata
    mode: ASRMode = ASRMode.STREAMING
    is_final: bool = False  # True when utterance complete
    is_partial: bool = False  # True for streaming partial results

    # Word-level timestamps
    word_timestamps: list[WordTimestamp] = field(default_factory=list)

    # Rich audio features (Phase 5)
    emotion: EmotionLabel | None = None
    emotion_confidence: float = 0.0
    emotion_all_probs: dict[str, float] | None = None

    pitch: PitchInfo | None = None
    phonemes: PhonemeInfo | None = None
    paralinguistics: ParalinguisticsInfo | None = None
    language: LanguageInfo | None = None
    singing: SingingInfo | None = None

    # Hallucination detection (Phase 5.8)
    hallucination: HallucinationInfo | None = None

    # Speaker embedding (Phase 6)
    speaker: SpeakerInfo | None = None

    # Chunk tracking for streaming
    chunk_index: int = 0
    utterance_id: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "text": self.text,
            "confidence": self.confidence,
            "start_time_ms": self.start_time_ms,
            "end_time_ms": self.end_time_ms,
            "mode": self.mode.value,
            "is_final": self.is_final,
            "is_partial": self.is_partial,
            "chunk_index": self.chunk_index,
            "utterance_id": self.utterance_id,
        }

        # Word timestamps
        if self.word_timestamps:
            result["word_timestamps"] = [
                {
                    "word": wt.word,
                    "start_ms": wt.start_ms,
                    "end_ms": wt.end_ms,
                    "confidence": wt.confidence,
                }
                for wt in self.word_timestamps
            ]

        # Emotion
        if self.emotion is not None:
            result["emotion"] = {
                "label": self.emotion.value,
                "confidence": self.emotion_confidence,
            }
            if self.emotion_all_probs:
                result["emotion"]["all_probs"] = self.emotion_all_probs

        # Pitch
        if self.pitch is not None:
            result["pitch"] = {
                "mean_hz": self.pitch.mean_hz,
                "std_hz": self.pitch.std_hz,
                "min_hz": self.pitch.min_hz,
                "max_hz": self.pitch.max_hz,
                "voiced_ratio": self.pitch.voiced_ratio,
            }
            if self.pitch.frame_f0_hz:
                result["pitch"]["frame_f0_hz"] = self.pitch.frame_f0_hz

        # Phonemes
        if self.phonemes is not None:
            result["phonemes"] = {
                "sequence": self.phonemes.phonemes,
                "confidences": self.phonemes.confidences,
            }

        # Paralinguistics
        if self.paralinguistics is not None:
            result["paralinguistics"] = {
                "event": self.paralinguistics.event,
                "confidence": self.paralinguistics.confidence,
            }
            if self.paralinguistics.all_probs:
                result["paralinguistics"]["all_probs"] = self.paralinguistics.all_probs

        # Language
        if self.language is not None:
            result["language"] = {
                "code": self.language.language,
                "confidence": self.language.confidence,
            }
            if self.language.all_probs:
                result["language"]["all_probs"] = self.language.all_probs

        # Singing
        if self.singing is not None:
            result["singing"] = {
                "is_singing": self.singing.is_singing,
                "confidence": self.singing.singing_confidence,
                "techniques": self.singing.techniques,
            }

        # Hallucination
        if self.hallucination is not None:
            result["hallucination"] = {
                "score": self.hallucination.score,
                "is_hallucinated": self.hallucination.is_hallucinated,
                "phoneme_mismatch": self.hallucination.phoneme_mismatch,
                "energy_silence": self.hallucination.energy_silence,
                "repetition": self.hallucination.repetition,
            }

        # Speaker
        if self.speaker is not None:
            result["speaker"] = {
                "embedding": self.speaker.embedding,
            }
            if self.speaker.speaker_id:
                result["speaker"]["speaker_id"] = self.speaker.speaker_id
                result["speaker"]["similarity"] = self.speaker.similarity

        return result

    def to_json(self, indent: int | None = None) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict) -> "RichToken":
        """Create RichToken from dictionary."""
        # Parse word timestamps
        word_timestamps = []
        if "word_timestamps" in data:
            for wt in data["word_timestamps"]:
                word_timestamps.append(WordTimestamp(
                    word=wt["word"],
                    start_ms=wt["start_ms"],
                    end_ms=wt["end_ms"],
                    confidence=wt.get("confidence", 1.0),
                ))

        # Parse emotion
        emotion = None
        emotion_confidence = 0.0
        emotion_all_probs = None
        if "emotion" in data:
            emotion = EmotionLabel(data["emotion"]["label"])
            emotion_confidence = data["emotion"]["confidence"]
            emotion_all_probs = data["emotion"].get("all_probs")

        # Parse pitch
        pitch = None
        if "pitch" in data:
            pitch = PitchInfo(
                mean_hz=data["pitch"]["mean_hz"],
                std_hz=data["pitch"]["std_hz"],
                min_hz=data["pitch"]["min_hz"],
                max_hz=data["pitch"]["max_hz"],
                voiced_ratio=data["pitch"].get("voiced_ratio", 1.0),
                frame_f0_hz=data["pitch"].get("frame_f0_hz"),
            )

        # Parse phonemes
        phonemes = None
        if "phonemes" in data:
            phonemes = PhonemeInfo(
                phonemes=data["phonemes"]["sequence"],
                confidences=data["phonemes"].get("confidences", []),
            )

        # Parse paralinguistics
        paralinguistics = None
        if "paralinguistics" in data:
            paralinguistics = ParalinguisticsInfo(
                event=data["paralinguistics"]["event"],
                confidence=data["paralinguistics"]["confidence"],
                all_probs=data["paralinguistics"].get("all_probs"),
            )

        # Parse language
        language = None
        if "language" in data:
            language = LanguageInfo(
                language=data["language"]["code"],
                confidence=data["language"]["confidence"],
                all_probs=data["language"].get("all_probs"),
            )

        # Parse singing
        singing = None
        if "singing" in data:
            singing = SingingInfo(
                is_singing=data["singing"]["is_singing"],
                singing_confidence=data["singing"]["confidence"],
                techniques=data["singing"].get("techniques", []),
            )

        # Parse hallucination
        hallucination = None
        if "hallucination" in data:
            hallucination = HallucinationInfo(
                score=data["hallucination"]["score"],
                is_hallucinated=data["hallucination"]["is_hallucinated"],
                phoneme_mismatch=data["hallucination"].get("phoneme_mismatch", 0.0),
                energy_silence=data["hallucination"].get("energy_silence", 0.0),
                repetition=data["hallucination"].get("repetition", 0.0),
            )

        # Parse speaker
        speaker = None
        if "speaker" in data:
            speaker = SpeakerInfo(
                embedding=data["speaker"]["embedding"],
                speaker_id=data["speaker"].get("speaker_id"),
                similarity=data["speaker"].get("similarity"),
            )

        return cls(
            text=data["text"],
            confidence=data["confidence"],
            start_time_ms=data["start_time_ms"],
            end_time_ms=data["end_time_ms"],
            mode=ASRMode(data.get("mode", "streaming")),
            is_final=data.get("is_final", False),
            is_partial=data.get("is_partial", False),
            word_timestamps=word_timestamps,
            emotion=emotion,
            emotion_confidence=emotion_confidence,
            emotion_all_probs=emotion_all_probs,
            pitch=pitch,
            phonemes=phonemes,
            paralinguistics=paralinguistics,
            language=language,
            singing=singing,
            hallucination=hallucination,
            speaker=speaker,
            chunk_index=data.get("chunk_index", 0),
            utterance_id=data.get("utterance_id", ""),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "RichToken":
        """Create RichToken from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class StreamingResponse:
    """Response wrapper for streaming API."""

    # Type of response
    type: str  # "partial", "final", "error", "metadata"

    # RichToken if type is partial/final
    token: RichToken | None = None

    # Error message if type is error
    error: str | None = None

    # Metadata (latency stats, etc.)
    metadata: dict | None = None

    # Sequence number for ordering
    sequence: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "type": self.type,
            "sequence": self.sequence,
        }
        if self.token:
            result["token"] = self.token.to_dict()
        if self.error:
            result["error"] = self.error
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    def to_json(self, indent: int | None = None) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


def create_partial_token(
    text: str,
    start_ms: float,
    end_ms: float,
    chunk_index: int = 0,
    utterance_id: str = "",
    confidence: float = 0.9,
) -> RichToken:
    """Create a partial (streaming) RichToken with minimal fields."""
    return RichToken(
        text=text,
        confidence=confidence,
        start_time_ms=start_ms,
        end_time_ms=end_ms,
        mode=ASRMode.STREAMING,
        is_final=False,
        is_partial=True,
        chunk_index=chunk_index,
        utterance_id=utterance_id,
    )


def create_final_token(
    text: str,
    start_ms: float,
    end_ms: float,
    word_timestamps: list[WordTimestamp] | None = None,
    utterance_id: str = "",
    confidence: float = 0.95,
    mode: ASRMode = ASRMode.STREAMING,
) -> RichToken:
    """Create a final (complete) RichToken."""
    return RichToken(
        text=text,
        confidence=confidence,
        start_time_ms=start_ms,
        end_time_ms=end_ms,
        mode=mode,
        is_final=True,
        is_partial=False,
        word_timestamps=word_timestamps or [],
        utterance_id=utterance_id,
    )
