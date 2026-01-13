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
Prosody-Conditioned Beam Search for Streaming ASR.

This module implements beam search scoring modifications based on prosody signals
(pitch contour and emotion detection) to improve punctuation prediction.

Key insight: Prosody carries punctuation cues that text-only models miss:
- Rising pitch slope often indicates questions (?)
- Falling pitch slope indicates declarative statements (.)
- Emotional intensity (surprise, joy, anger) often correlates with exclamations (!)

Architecture:
    Audio -> Encoder -> [Pitch Head, Emotion Head] -> Prosody Features
                    |
                    +-> Beam Search with Prosody-Conditioned Scoring

References:
- Whisper beam search: tools/whisper_mlx/beam_search.py
- Multi-head architecture: tools/whisper_mlx/multi_head.py
- Streaming integration: tools/whisper_mlx/streaming.py
"""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .beam_search import Beam, BeamSearchDecoder, BeamSearchResult
from .decoding import DecodingOptions

# Punctuation token IDs in Whisper vocabulary
# These are approximate - actual IDs depend on tokenizer
PUNCTUATION_TOKENS = {
    "?": None,  # Question mark - will be set from tokenizer
    ".": None,  # Period
    "!": None,  # Exclamation
    ",": None,  # Comma
    ";": None,  # Semicolon
    ":": None,  # Colon
}


@dataclass
class ProsodyFeatures:
    """
    Prosody features extracted from audio for beam search conditioning.

    These features are computed from the encoder output using pitch and emotion heads.
    """
    # Pitch features
    pitch_values: mx.array | None = None  # (T,) F0 in Hz per frame
    pitch_slope: float = 0.0  # Overall pitch trend: positive = rising, negative = falling
    pitch_range: float = 0.0  # Pitch variation (std dev)
    voicing_ratio: float = 0.0  # Fraction of voiced frames

    # Emotion features (probabilities for each emotion class)
    emotion_probs: dict[str, float] | None = None
    dominant_emotion: str = "neutral"
    emotion_confidence: float = 0.0

    # Intensity features
    intensity: float = 0.0  # Overall speech intensity [0, 1]

    # Timing
    audio_duration: float = 0.0  # Duration in seconds

    def __post_init__(self):
        if self.emotion_probs is None:
            self.emotion_probs = {}


@dataclass
class ProsodyBoostConfig:
    """Configuration for prosody-based score boosting."""

    # Question mark boosting
    question_rising_pitch_boost: float = 3.0  # Boost for '?' when pitch is rising
    question_pitch_threshold: float = 0.1  # Minimum pitch slope to trigger boost

    # Period boosting
    period_falling_pitch_boost: float = 1.5  # Boost for '.' when pitch is falling
    period_pitch_threshold: float = -0.05  # Maximum pitch slope for period boost

    # Exclamation boosting (emotion-based)
    exclamation_surprise_boost: float = 2.5  # Boost for '!' with surprise emotion
    exclamation_joy_boost: float = 2.5  # Boost for '!' with joy emotion
    exclamation_anger_boost: float = 2.0  # Boost for '!' with anger emotion
    exclamation_emotion_threshold: float = 0.3  # Minimum emotion probability

    # High intensity boosting (any punctuation)
    high_intensity_punctuation_boost: float = 1.2  # Boost for punctuation with high intensity
    high_intensity_threshold: float = 0.7  # Intensity threshold for boosting

    # Enable/disable specific rules
    enable_pitch_rules: bool = True
    enable_emotion_rules: bool = True
    enable_intensity_rules: bool = True


class ProsodyBeamSearch:
    """
    Beam search with prosody-conditioned scoring.

    Modifies standard beam search by applying multiplicative score factors
    based on prosody features (pitch, emotion, intensity).

    Usage:
        from tools.whisper_mlx.prosody_beam_search import ProsodyBeamSearch
        from tools.whisper_mlx.multi_head import CREPEPitchHead, EmotionHead

        # Create prosody beam search with pitch and emotion heads
        prosody_search = ProsodyBeamSearch(
            pitch_head=pitch_head,
            emotion_head=emotion_head,
        )

        # Decode with prosody conditioning
        result = prosody_search.decode_with_prosody(
            model=whisper_model,
            audio_features=encoder_output,
            tokenizer=tokenizer,
        )
    """

    def __init__(
        self,
        pitch_head: nn.Module | None = None,
        emotion_head: nn.Module | None = None,
        config: ProsodyBoostConfig | None = None,
    ):
        """
        Initialize prosody-conditioned beam search.

        Args:
            pitch_head: Pitch prediction head (CREPEPitchHead or PitchHeadMLP)
            emotion_head: Emotion classification head (EmotionHead)
            config: Prosody boost configuration
        """
        self.pitch_head = pitch_head
        self.emotion_head = emotion_head
        self.config = config or ProsodyBoostConfig()

        # Punctuation token cache (populated on first use)
        self._punctuation_ids: dict[str, int] = {}
        self._tokenizer_initialized = False

        # Statistics
        self.boosts_applied = 0
        self.total_tokens_scored = 0

    def _initialize_punctuation_ids(self, tokenizer) -> None:
        """Initialize punctuation token IDs from tokenizer."""
        if self._tokenizer_initialized:
            return

        # Common punctuation tokens
        punctuation = ["?", ".", "!", ",", ";", ":"]

        for punct in punctuation:
            try:
                # Encode single punctuation
                tokens = tokenizer.encode(punct)
                if tokens:
                    self._punctuation_ids[punct] = tokens[0]
            except Exception:
                pass

        self._tokenizer_initialized = True

    def extract_prosody_features(
        self,
        encoder_output: mx.array,
        frame_rate: float = 50.0,
    ) -> ProsodyFeatures:
        """
        Extract prosody features from encoder output.

        Args:
            encoder_output: (batch, T, d_model) encoder hidden states
            frame_rate: Frames per second (50 for Whisper)

        Returns:
            ProsodyFeatures with pitch and emotion information
        """
        features = ProsodyFeatures()

        # Compute audio duration
        T = encoder_output.shape[1]
        features.audio_duration = T / frame_rate

        # Extract pitch features
        if self.pitch_head is not None:
            pitch_hz, voicing_prob = self.pitch_head(encoder_output)
            mx.eval(pitch_hz, voicing_prob)

            features.pitch_values = pitch_hz[0]  # (T,)

            # Compute pitch slope (linear regression over voiced frames)
            pitch_np = np.array(pitch_hz[0].tolist())
            voicing_np = np.array(voicing_prob[0].tolist())

            voiced_mask = voicing_np > 0.5
            features.voicing_ratio = float(np.mean(voiced_mask))

            if np.sum(voiced_mask) > 10:  # Need enough voiced frames
                voiced_pitch = pitch_np[voiced_mask]
                voiced_times = np.arange(len(pitch_np))[voiced_mask]

                # Simple linear regression for slope
                if len(voiced_times) > 1:
                    # Normalize pitch to [0, 1] range for slope calculation
                    pitch_min, pitch_max = voiced_pitch.min(), voiced_pitch.max()
                    if pitch_max > pitch_min:
                        pitch_norm = (voiced_pitch - pitch_min) / (pitch_max - pitch_min)
                        # Compute slope using least squares
                        len(voiced_times)
                        x_mean = np.mean(voiced_times)
                        y_mean = np.mean(pitch_norm)
                        slope = np.sum((voiced_times - x_mean) * (pitch_norm - y_mean)) / \
                                (np.sum((voiced_times - x_mean) ** 2) + 1e-8)
                        features.pitch_slope = float(slope)

                features.pitch_range = float(np.std(voiced_pitch))

        # Extract emotion features
        if self.emotion_head is not None:
            emotion_logits = self.emotion_head(encoder_output)
            mx.eval(emotion_logits)

            emotion_probs = mx.softmax(emotion_logits, axis=-1)
            probs_np = np.array(emotion_probs[0].tolist())

            # Map to emotion names (using RAVDESS/extended taxonomy)
            emotion_names = [
                "neutral", "calm", "happy", "sad", "angry",
                "fearful", "disgust", "surprise",
            ]

            features.emotion_probs = {}
            for i, name in enumerate(emotion_names):
                if i < len(probs_np):
                    features.emotion_probs[name] = float(probs_np[i])

            # Map common names
            if "happy" in features.emotion_probs:
                features.emotion_probs["joy"] = features.emotion_probs["happy"]
            if "angry" in features.emotion_probs:
                features.emotion_probs["anger"] = features.emotion_probs["angry"]
            if "fearful" in features.emotion_probs:
                features.emotion_probs["fear"] = features.emotion_probs["fearful"]

            # Get dominant emotion
            dominant_idx = int(np.argmax(probs_np))
            if dominant_idx < len(emotion_names):
                features.dominant_emotion = emotion_names[dominant_idx]
                features.emotion_confidence = float(probs_np[dominant_idx])

        return features

    def prosody_score_factor(
        self,
        token_id: int,
        prosody_features: ProsodyFeatures,
        tokenizer=None,
    ) -> float:
        """
        Compute multiplicative factor for beam score based on prosody.

        Rules:
        - Rising pitch slope + token '?' -> boost 3x
        - Falling pitch slope + token '.' -> boost 1.5x
        - Emotion 'surprise' or 'joy' + token '!' -> boost 2.5x
        - Emotion 'anger' + token '!' -> boost 2.0x
        - High intensity + any punctuation -> boost 1.2x

        Args:
            token_id: Token ID to score
            prosody_features: Extracted prosody features
            tokenizer: Tokenizer for punctuation lookup

        Returns:
            Multiplicative score factor (1.0 = no change)
        """
        self.total_tokens_scored += 1
        factor = 1.0

        # Initialize punctuation IDs if needed
        if tokenizer is not None and not self._tokenizer_initialized:
            self._initialize_punctuation_ids(tokenizer)

        # Get punctuation character for this token
        token_punct = None
        for punct, pid in self._punctuation_ids.items():
            if pid == token_id:
                token_punct = punct
                break

        if token_punct is None:
            return factor  # Not a punctuation token

        config = self.config

        # Rule 1: Rising pitch + question mark
        if config.enable_pitch_rules and token_punct == "?":
            if prosody_features.pitch_slope > config.question_pitch_threshold:
                factor *= config.question_rising_pitch_boost
                self.boosts_applied += 1

        # Rule 2: Falling pitch + period
        if config.enable_pitch_rules and token_punct == ".":
            if prosody_features.pitch_slope < config.period_pitch_threshold:
                factor *= config.period_falling_pitch_boost
                self.boosts_applied += 1

        # Rule 3: Emotion-based exclamation boosting
        if config.enable_emotion_rules and token_punct == "!":
            emotion_probs = prosody_features.emotion_probs or {}

            # Surprise or joy -> strong boost
            surprise_prob = emotion_probs.get("surprise", 0.0)
            joy_prob = emotion_probs.get("joy", emotion_probs.get("happy", 0.0))
            if surprise_prob > config.exclamation_emotion_threshold:
                factor *= config.exclamation_surprise_boost
                self.boosts_applied += 1
            elif joy_prob > config.exclamation_emotion_threshold:
                factor *= config.exclamation_joy_boost
                self.boosts_applied += 1

            # Anger -> moderate boost
            anger_prob = emotion_probs.get("anger", emotion_probs.get("angry", 0.0))
            if anger_prob > config.exclamation_emotion_threshold:
                factor *= config.exclamation_anger_boost
                self.boosts_applied += 1

        # Rule 4: High intensity + any punctuation
        if config.enable_intensity_rules:
            if prosody_features.intensity > config.high_intensity_threshold:
                factor *= config.high_intensity_punctuation_boost
                self.boosts_applied += 1

        return factor

    def decode_with_prosody(
        self,
        model: nn.Module,
        audio_features: mx.array,
        tokenizer,
        beam_size: int = 5,
        length_penalty: float = 1.0,
        max_tokens: int = 224,
        max_initial_timestamp: float = 1.0,
        audio_duration: float | None = None,
    ) -> BeamSearchResult:
        """
        Run beam search with prosody conditioning.

        This wraps the standard beam search decoder but modifies scores
        based on prosody features.

        Args:
            model: WhisperMLX model
            audio_features: (1, T, D) encoded audio features
            tokenizer: Whisper tokenizer
            beam_size: Number of beams
            length_penalty: Length normalization penalty
            max_tokens: Maximum tokens to decode
            max_initial_timestamp: Maximum initial timestamp
            audio_duration: Audio duration for variable-length mode

        Returns:
            BeamSearchResult with prosody-conditioned decoding
        """
        # Initialize punctuation IDs
        self._initialize_punctuation_ids(tokenizer)

        # Extract prosody features from encoder output
        prosody_features = self.extract_prosody_features(audio_features)

        # Create modified beam search decoder with vocab size for precomputed boosts
        decoder = ProsodyBeamSearchDecoder(
            model=model,
            beam_size=beam_size,
            length_penalty=length_penalty,
            prosody_features=prosody_features,
            prosody_scorer=self,
            tokenizer=tokenizer,
            n_vocab=model.config.n_vocab,
        )

        # Build decoding options
        options = DecodingOptions(
            temperature=0.0,
            max_initial_timestamp=max_initial_timestamp,
            suppress_blank=True,
            suppress_tokens="-1",
            without_timestamps=False,
        )

        sample_begin = len(list(tokenizer.sot_sequence))

        return decoder.decode(
            audio_features=audio_features,
            tokenizer=tokenizer,
            options=options,
            sample_begin=sample_begin,
            n_vocab=model.config.n_vocab,
            precision=getattr(model.decoder, 'precision', 0.02),
            audio_duration=audio_duration,
            max_tokens=max_tokens,
        )


    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self.boosts_applied = 0
        self.total_tokens_scored = 0

    def get_stats(self) -> dict[str, int | float]:
        """Get statistics about prosody boosts applied."""
        return {
            "boosts_applied": self.boosts_applied,
            "total_tokens_scored": self.total_tokens_scored,
            "boost_rate": self.boosts_applied / max(1, self.total_tokens_scored),
        }


class ProsodyBeamSearchDecoder(BeamSearchDecoder):
    """
    Extended beam search decoder with prosody-conditioned scoring.

    Inherits from BeamSearchDecoder and modifies the candidate scoring
    to incorporate prosody-based boosting factors.

    OPTIMIZATION: Precomputes prosody boosts once before decoding starts,
    storing them as an MLX array that can be added directly to log_probs
    without per-step numpy conversions.
    """

    def __init__(
        self,
        model: nn.Module,
        beam_size: int = 5,
        length_penalty: float = 1.0,
        patience: float = 1.0,
        prosody_features: ProsodyFeatures | None = None,
        prosody_scorer: ProsodyBeamSearch | None = None,
        tokenizer=None,
        n_vocab: int = 51865,
    ):
        """
        Initialize prosody-conditioned beam search decoder.

        Args:
            model: WhisperMLX model
            beam_size: Number of beams
            length_penalty: Length normalization penalty
            patience: Early stopping patience
            prosody_features: Pre-computed prosody features
            prosody_scorer: ProsodyBeamSearch instance for scoring
            tokenizer: Tokenizer for punctuation lookup
            n_vocab: Vocabulary size for precomputed boost array
        """
        super().__init__(model, beam_size, length_penalty, patience)
        self.prosody_features = prosody_features
        self.prosody_scorer = prosody_scorer
        self.tokenizer = tokenizer
        self.n_vocab = n_vocab

        # OPTIMIZATION: Precompute prosody boosts as MLX array
        self._prosody_log_boosts: mx.array | None = None
        self._precompute_prosody_boosts()

    def _precompute_prosody_boosts(self) -> None:
        """
        Precompute all prosody boosts once before decoding.

        Creates an MLX array of shape (n_vocab,) with log-space boosts
        for all punctuation tokens. Non-punctuation tokens have 0.0 boost.
        """
        if self.prosody_scorer is None or self.prosody_features is None:
            return

        # Initialize boost array with zeros (no boost in log space)
        boosts = np.zeros(self.n_vocab, dtype=np.float32)

        # Compute boost for each punctuation token
        for token_id in self.prosody_scorer._punctuation_ids.values():
            if token_id is not None and token_id < self.n_vocab:
                factor = self.prosody_scorer.prosody_score_factor(
                    token_id=token_id,
                    prosody_features=self.prosody_features,
                    tokenizer=self.tokenizer,
                )
                # Convert multiplicative factor to log-space addition
                if factor != 1.0:
                    boosts[token_id] = np.log(factor)

        # Convert to MLX array once
        self._prosody_log_boosts = mx.array(boosts)

    def _apply_prosody_scoring(
        self,
        log_probs: mx.array,
    ) -> mx.array:
        """
        Apply prosody-based score modifications to log probabilities.

        OPTIMIZED: Uses precomputed MLX array for O(1) boost application.

        Args:
            log_probs: (vocab_size,) log probabilities from decoder

        Returns:
            Modified log probabilities with prosody boosts
        """
        if self._prosody_log_boosts is None:
            return log_probs

        # Direct MLX addition - no numpy conversion needed
        return log_probs + self._prosody_log_boosts

    def decode(
        self,
        audio_features: mx.array,
        tokenizer,
        options: DecodingOptions,
        sample_begin: int,
        n_vocab: int,
        precision: float = 0.02,
        audio_duration: float | None = None,
        max_tokens: int = 224,
    ) -> BeamSearchResult:
        """
        Decode with prosody-conditioned scoring.

        This fully reimplements the decode method to inject prosody scoring
        into the beam expansion loop. Prosody boosts are applied to punctuation
        token probabilities based on pitch slope and emotion features.
        """
        from .decoding import apply_filters, build_logit_filters, compute_logprobs

        # Build initial token sequence
        initial_tokens = list(tokenizer.sot_sequence)
        tokens = initial_tokens.copy()

        # Build logit filters
        logit_filters = build_logit_filters(
            tokenizer=tokenizer,
            options=options,
            sample_begin=sample_begin,
            n_vocab=n_vocab,
            precision=precision,
            audio_duration=audio_duration,
        )

        # Initialize beams with initial sequence
        beams = [Beam(tokens=tokens.copy(), log_prob=0.0)]
        finished_beams: list[Beam] = []

        # Initialize KV caches
        kv_caches: list[list | None] = [None]

        # Tracking for early stopping
        best_finished_score = float('-inf')
        patience_counter = 0

        # Initial forward pass to get first logits
        tokens_tensor = mx.array([tokens])
        logits, kv_cache, _, _ = self.model.decoder(
            tokens_tensor, audio_features, kv_cache=None,
        )
        logits = logits[:, -1].astype(mx.float32)

        # Capture no_speech_prob from first logits
        no_speech_prob = 0.0
        if hasattr(tokenizer, 'no_speech'):
            probs_at_sot = mx.softmax(logits, axis=-1)
            no_speech_prob = float(probs_at_sot[0, tokenizer.no_speech])
            beams[0].no_speech_prob = no_speech_prob

        filtered_logits = apply_filters(logits, tokens_tensor, logit_filters)

        # Get log probabilities
        log_probs = compute_logprobs(filtered_logits)

        # PROSODY SCORING: Apply prosody boosts to initial token selection
        if self.prosody_scorer is not None and self.prosody_features is not None:
            log_probs = self._apply_prosody_scoring(log_probs[0])[None, :]

        # Get top-k initial tokens
        top_k_indices = mx.argsort(-log_probs[0])[:self.beam_size]
        top_k_log_probs = log_probs[0][top_k_indices]

        # Initialize beams with top-k tokens
        beams = []
        kv_caches = []
        for k in range(self.beam_size):
            token = int(top_k_indices[k])
            token_lp = float(top_k_log_probs[k])
            beams.append(Beam(
                tokens=tokens + [token],
                log_prob=token_lp,
                no_speech_prob=no_speech_prob,
            ))
            if k == 0:
                kv_caches.append(kv_cache)
            else:
                kv_caches.append(self._clone_kv_cache(kv_cache))

        # Main decoding loop
        for _step in range(max_tokens - len(initial_tokens)):
            if not beams:
                break

            all_candidates = []

            for beam_idx, (beam, kv_cache_item) in enumerate(zip(beams, kv_caches, strict=False)):
                if beam.finished:
                    continue

                # Get prediction for this beam
                last_token = mx.array([[beam.tokens[-1]]])
                logits, new_kv, _, _ = self.model.decoder(
                    last_token, audio_features, kv_cache=kv_cache_item,
                )
                logits = logits[:, -1].astype(mx.float32)

                # Apply filters with current beam's token history
                token_context = mx.array([beam.tokens])
                filtered_logits = apply_filters(logits, token_context, logit_filters)

                # Get log probabilities
                log_probs_step = compute_logprobs(filtered_logits)

                # PROSODY SCORING: Apply prosody boosts during decoding
                if self.prosody_scorer is not None and self.prosody_features is not None:
                    log_probs_step = self._apply_prosody_scoring(log_probs_step[0])[None, :]

                # Get top-k candidates for this beam
                top_k_indices = mx.argsort(-log_probs_step[0])[:self.beam_size]
                top_k_log_probs = log_probs_step[0][top_k_indices]

                for k in range(self.beam_size):
                    token = int(top_k_indices[k])
                    token_lp = float(top_k_log_probs[k])
                    new_lp = beam.log_prob + token_lp

                    all_candidates.append((
                        beam_idx,
                        token,
                        new_lp,
                        beam,
                        new_kv,
                    ))

            # Sort candidates by log probability and keep top beam_size
            all_candidates.sort(key=lambda x: x[2], reverse=True)

            # Select best candidates
            new_beams: list[Beam] = []
            new_kv_caches: list[list | None] = []
            used_kv: dict[int, list] = {}

            for beam_idx, token, new_lp, parent_beam, new_kv in all_candidates:
                if len(new_beams) >= self.beam_size:
                    break

                new_tokens = parent_beam.tokens + [token]
                new_beam = Beam(
                    tokens=new_tokens,
                    log_prob=new_lp,
                    no_speech_prob=parent_beam.no_speech_prob,
                )

                if token == tokenizer.eot:
                    new_beam.finished = True
                    finished_beams.append(new_beam)

                    # Update best finished score for early stopping
                    score = self._compute_score(new_beam, sample_begin)
                    if score > best_finished_score:
                        best_finished_score = score
                        patience_counter = 0
                else:
                    new_beams.append(new_beam)

                    # Clone KV cache if already used by another beam
                    if beam_idx in used_kv:
                        new_kv_caches.append(self._clone_kv_cache(new_kv))
                    else:
                        new_kv_caches.append(new_kv)
                        used_kv[beam_idx] = new_kv

            beams = new_beams
            kv_caches = new_kv_caches

            # Early stopping check
            if finished_beams:
                best_active_lp = max(b.log_prob for b in beams) if beams else float('-inf')
                best_possible_score = self._compute_score(
                    Beam(tokens=beams[0].tokens if beams else [], log_prob=best_active_lp),
                    sample_begin,
                )

                if best_possible_score < best_finished_score:
                    break

                patience_counter += 1
                if patience_counter >= int(self.patience * self.beam_size):
                    break

        # Select best beam
        all_finished = finished_beams + list(beams)
        if not all_finished:
            return BeamSearchResult(tokens=initial_tokens[sample_begin:])

        # Score all beams
        scored_beams = [
            (beam, self._compute_score(beam, sample_begin))
            for beam in all_finished
        ]
        scored_beams.sort(key=lambda x: x[1], reverse=True)
        best_beam = scored_beams[0][0]

        # Compute average log probability
        n_tokens = len(best_beam.tokens) - sample_begin
        avg_logprob = best_beam.log_prob / max(1, n_tokens)

        return BeamSearchResult(
            tokens=best_beam.tokens[sample_begin:],
            log_prob=best_beam.log_prob,
            normalized_score=scored_beams[0][1],
            avg_logprob=avg_logprob,
            no_speech_prob=best_beam.no_speech_prob,
        )


@dataclass
class ProsodyStreamingResult:
    """Result from prosody-conditioned streaming transcription."""
    text: str
    is_final: bool
    is_partial: bool = False
    segment_start: float = 0.0
    segment_end: float = 0.0
    language: str | None = None
    processing_time: float = 0.0
    audio_duration: float = 0.0
    confirmed_text: str = ""
    speculative_text: str = ""
    is_confirmed: bool = False
    # Prosody-specific fields
    prosody_features: ProsodyFeatures | None = None
    prosody_boosts_applied: int = 0


class StreamingProsodyDecoder:
    """
    Streaming integration for prosody-conditioned decoding.

    Integrates with StreamingWhisper to provide real-time prosody-based
    punctuation enhancement.

    Usage:
        from tools.whisper_mlx import WhisperMLX
        from tools.whisper_mlx.streaming import StreamingWhisper, StreamingConfig
        from tools.whisper_mlx.prosody_beam_search import StreamingProsodyDecoder

        model = WhisperMLX.from_pretrained("large-v3")

        # Create streaming decoder with prosody conditioning
        prosody_decoder = StreamingProsodyDecoder(
            model=model,
            pitch_head=pitch_head,
            emotion_head=emotion_head,
        )

        # Process streaming audio
        async for result in prosody_decoder.transcribe_stream(audio_source):
            print(result.text)
    """

    def __init__(
        self,
        model: nn.Module,
        pitch_head: nn.Module | None = None,
        emotion_head: nn.Module | None = None,
        prosody_config: ProsodyBoostConfig | None = None,
        streaming_config=None,
        use_prosody_decoding: bool = True,
        beam_size: int = 5,
        length_penalty: float = 1.0,
    ):
        """
        Initialize streaming prosody decoder.

        Args:
            model: WhisperMLX model
            pitch_head: Pitch prediction head
            emotion_head: Emotion classification head
            prosody_config: Prosody boost configuration
            streaming_config: StreamingConfig for base streaming behavior
            use_prosody_decoding: If True, use prosody beam search; else greedy
            beam_size: Number of beams for prosody beam search
            length_penalty: Length penalty for beam search
        """
        self.model = model
        self.prosody_search = ProsodyBeamSearch(
            pitch_head=pitch_head,
            emotion_head=emotion_head,
            config=prosody_config,
        )
        self.use_prosody_decoding = use_prosody_decoding
        self.beam_size = beam_size
        self.length_penalty = length_penalty

        # Import streaming components
        from .streaming import StreamingConfig, StreamingWhisper

        self.streaming_config = streaming_config or StreamingConfig()
        self._streaming_whisper: StreamingWhisper | None = None

        # State for streaming
        self._audio_buffer: list[np.ndarray] = []
        self._speech_buffer: np.ndarray | None = None
        self._detected_language: str | None = None
        self._segment_start_time: float = 0.0
        self._total_audio_time: float = 0.0

        # LocalAgreement for partial results
        self._local_agreement = None

    def _get_streaming_whisper(self):
        """Lazy initialization of streaming whisper."""
        if self._streaming_whisper is None:
            from .streaming import StreamingWhisper
            self._streaming_whisper = StreamingWhisper(
                model=self.model,
                config=self.streaming_config,
            )
        return self._streaming_whisper

    def _reset_state(self, reset_prosody_stats: bool = False):
        """Reset streaming state.

        Args:
            reset_prosody_stats: If True, also reset prosody scoring stats.
                                Default False to preserve stats for post-streaming analysis.
        """
        self._audio_buffer = []
        self._speech_buffer = None
        self._segment_start_time = 0.0
        self._total_audio_time = 0.0
        if reset_prosody_stats:
            self.prosody_search.reset_stats()

    def _transcribe_with_prosody(
        self,
        audio: np.ndarray,
        language: str | None = None,
    ) -> tuple[dict, ProsodyFeatures | None]:
        """
        Transcribe audio with prosody-conditioned beam search.

        This exposes the encoder output for prosody feature extraction
        and uses ProsodyBeamSearchDecoder for decoding.

        Args:
            audio: Audio waveform (float32, 16kHz)
            language: Language code (auto-detect if None)

        Returns:
            Tuple of (transcription result dict, prosody features)
        """
        import time

        from .audio import log_mel_spectrogram
        from .tokenizer import get_whisper_tokenizer

        # Compute mel spectrogram
        mel = log_mel_spectrogram(audio, n_mels=self.model.config.n_mels)
        target_len = self.model.config.n_audio_ctx * 2
        if mel.shape[0] < target_len:
            mel = mx.pad(mel, [(0, target_len - mel.shape[0]), (0, 0)])
        elif mel.shape[0] > target_len:
            mel = mel[:target_len, :]
        self.model.decoder.reset_precision()
        mel = mel[None]  # Add batch dimension

        # Encode audio
        t0 = time.perf_counter()
        audio_features = self.model.embed_audio(mel, variable_length=False)
        encode_time = time.perf_counter() - t0

        # Extract prosody features
        prosody_features = None
        if self.use_prosody_decoding:
            prosody_features = self.prosody_search.extract_prosody_features(
                audio_features,
                frame_rate=50.0,
            )

        # Get tokenizer
        is_multilingual = self.model.config.n_vocab >= 51865
        num_langs = self.model.config.n_vocab - 51765 - int(is_multilingual)
        tokenizer = get_whisper_tokenizer(
            multilingual=is_multilingual,
            num_languages=num_langs,
            language=language or self._detected_language,
            task=self.streaming_config.task,
        )

        # Detect language if not specified
        if language is None and self._detected_language is None:
            self._detected_language = self.model._detect_language(
                audio_features, tokenizer,
            )

        # Decode with prosody conditioning
        if self.use_prosody_decoding and prosody_features is not None:
            # Use prosody beam search
            result = self.prosody_search.decode_with_prosody(
                model=self.model,
                audio_features=audio_features,
                tokenizer=tokenizer,
                beam_size=self.beam_size,
                length_penalty=self.length_penalty,
                max_tokens=224,
                max_initial_timestamp=1.0,
                audio_duration=len(audio) / 16000,
            )

            # Convert BeamSearchResult to dict
            # Filter out special tokens (timestamps, EOT, etc.)
            text_tokens = [t for t in result.tokens
                          if t < tokenizer.eot and t not in tokenizer.sot_sequence]
            text = tokenizer.decode(text_tokens)
            return {
                "text": text.strip(),
                "language": language or self._detected_language,
                "encode_time": encode_time,
                "beam_score": result.score if hasattr(result, 'score') else 0.0,
            }, prosody_features
        # Fall back to standard transcription
        result = self.model.transcribe(
            audio,
            language=language or self._detected_language,
            task=self.streaming_config.task,
            variable_length=False,
        )
        return result, prosody_features

    async def transcribe_stream(self, audio_source):
        """
        Transcribe streaming audio with prosody-conditioned decoding.

        Args:
            audio_source: Async iterator yielding audio chunks (np.ndarray)

        Yields:
            ProsodyStreamingResult with prosody-enhanced transcription
        """
        # If prosody decoding is disabled, fall back to base streaming
        if not self.use_prosody_decoding:
            streamer = self._get_streaming_whisper()
            async for result in streamer.transcribe_stream(audio_source):
                yield ProsodyStreamingResult(
                    text=result.text,
                    is_final=result.is_final,
                    is_partial=result.is_partial,
                    segment_start=result.segment_start,
                    segment_end=result.segment_end,
                    language=result.language,
                    processing_time=result.processing_time,
                    audio_duration=result.audio_duration,
                    confirmed_text=result.confirmed_text,
                    speculative_text=result.speculative_text,
                    is_confirmed=result.is_confirmed,
                    prosody_features=None,
                    prosody_boosts_applied=0,
                )
            return

        # Full prosody-conditioned streaming
        self._reset_state()

        # Initialize VAD if available
        vad = None
        try:
            from .silero_vad import SileroVAD
            vad = SileroVAD()
        except ImportError:
            pass

        silence_frames = 0
        speech_frames = 0

        async for chunk in audio_source:
            # Normalize audio
            if chunk.dtype == np.int16:
                chunk = chunk.astype(np.float32) / 32768.0
            elif chunk.dtype != np.float32:
                chunk = chunk.astype(np.float32)

            # Update time tracking
            chunk_duration = len(chunk) / self.streaming_config.sample_rate
            self._total_audio_time += chunk_duration

            # VAD check
            is_speech = True
            if vad is not None:
                try:
                    probs = vad(chunk)
                    is_speech = probs > 0.5
                except Exception:
                    is_speech = True

            # Accumulate speech
            if self._speech_buffer is None:
                if is_speech:
                    self._speech_buffer = chunk.copy()
                    self._segment_start_time = self._total_audio_time - chunk_duration
                    speech_frames = 1
                    silence_frames = 0
            else:
                self._speech_buffer = np.concatenate([self._speech_buffer, chunk])
                if is_speech:
                    speech_frames += 1
                    silence_frames = 0
                else:
                    silence_frames += 1

                # Check for segment end conditions
                current_duration = len(self._speech_buffer) / self.streaming_config.sample_rate
                silence_duration = silence_frames * chunk_duration

                # Endpoint detected (silence threshold)
                if silence_duration >= self.streaming_config.silence_threshold_duration:
                    async for result in self._process_prosody_segment(is_final=True):
                        yield result
                    self._speech_buffer = None
                    silence_frames = 0
                    speech_frames = 0

                # Max duration reached
                elif current_duration >= self.streaming_config.max_chunk_duration:
                    async for result in self._process_prosody_segment(is_final=True):
                        yield result
                    # Keep context for next segment
                    ctx_samples = int(
                        self.streaming_config.context_duration *
                        self.streaming_config.sample_rate,
                    )
                    if len(self._speech_buffer) > ctx_samples:
                        self._speech_buffer = self._speech_buffer[-ctx_samples:].copy()
                    self._segment_start_time = self._total_audio_time

                # Emit partial if configured
                elif self.streaming_config.emit_partials:
                    min_dur = self.streaming_config.min_chunk_duration
                    if current_duration >= min_dur:
                        async for result in self._process_prosody_segment(
                            is_final=False, is_partial=True,
                        ):
                            yield result

        # Finalize - process remaining audio
        if self._speech_buffer is not None:
            min_samples = int(
                self.streaming_config.min_chunk_duration *
                self.streaming_config.sample_rate,
            )
            if len(self._speech_buffer) >= min_samples:
                async for result in self._process_prosody_segment(is_final=True):
                    yield result

        self._reset_state()

    async def _process_prosody_segment(
        self,
        is_final: bool,
        is_partial: bool = False,
    ):
        """Process accumulated speech with prosody decoding."""
        import time

        if self._speech_buffer is None or len(self._speech_buffer) == 0:
            return

        min_samples = int(
            self.streaming_config.min_chunk_duration *
            self.streaming_config.sample_rate,
        )
        if len(self._speech_buffer) < min_samples:
            return

        audio_duration = len(self._speech_buffer) / self.streaming_config.sample_rate

        # Transcribe with prosody
        start_time = time.perf_counter()
        result, prosody_features = self._transcribe_with_prosody(
            self._speech_buffer,
            language=self.streaming_config.language or self._detected_language,
        )
        processing_time = time.perf_counter() - start_time

        # Cache detected language
        if self._detected_language is None and result.get("language"):
            self._detected_language = result["language"]

        text = result.get("text", "").strip()

        if text:
            # Get prosody stats
            stats = self.prosody_search.get_stats()

            yield ProsodyStreamingResult(
                text=text,
                is_final=is_final and not is_partial,
                is_partial=is_partial,
                segment_start=self._segment_start_time,
                segment_end=self._total_audio_time,
                language=result.get("language"),
                processing_time=processing_time,
                audio_duration=audio_duration,
                confirmed_text=text if (is_final and not is_partial) else "",
                speculative_text="" if (is_final and not is_partial) else text,
                is_confirmed=is_final and not is_partial,
                prosody_features=prosody_features,
                prosody_boosts_applied=stats.get("boosts_applied", 0),
            )

    def get_prosody_stats(self) -> dict[str, int | float]:
        """Get prosody scoring statistics."""
        return self.prosody_search.get_stats()


# Factory function for easy creation
def create_prosody_beam_search(
    model_size: str = "large-v3",
    pitch_checkpoint: str | None = None,
    emotion_checkpoint: str | None = None,
    config: ProsodyBoostConfig | None = None,
) -> ProsodyBeamSearch:
    """
    Factory function to create prosody beam search with trained heads.

    Args:
        model_size: Whisper model size for dimension matching
        pitch_checkpoint: Path to pitch head weights
        emotion_checkpoint: Path to emotion head weights
        config: Prosody boost configuration

    Returns:
        ProsodyBeamSearch ready for use
    """
    from .multi_head import CREPEPitchHead, EmotionHead, MultiHeadConfig

    # Model dimensions
    d_model_map = {
        "tiny": 384,
        "base": 512,
        "small": 768,
        "medium": 1024,
        "large": 1280,
        "large-v3": 1280,
        "turbo": 1280,
    }
    d_model = d_model_map.get(model_size, 1280)

    multi_config = MultiHeadConfig(d_model=d_model, use_crepe_pitch=True)

    # Create heads
    pitch_head = None
    emotion_head = None

    if pitch_checkpoint:
        pitch_head = CREPEPitchHead(multi_config)
        all_weights = mx.load(pitch_checkpoint)
        # Filter for pitch.* keys and strip prefix (multi-head checkpoint format)
        pitch_weights = {}
        for key, value in all_weights.items():
            if key.startswith("pitch."):
                new_key = key[6:]  # Strip "pitch." prefix
                pitch_weights[new_key] = value
        if pitch_weights:
            # Manually load all weights (Python lists not tracked by MLX)
            for k, v in pitch_weights.items():
                parts = k.split(".")
                if k.startswith("conv_layers."):
                    # conv_layers.X.conv.conv.Y -> _conv_modules[X].conv.Y
                    idx = int(parts[1])
                    if parts[2] == "conv" and parts[3] == "conv":
                        if parts[4] == "weight":
                            pitch_head._conv_modules[idx].conv.weight = v
                        elif parts[4] == "bias":
                            pitch_head._conv_modules[idx].conv.bias = v
                    elif parts[2] == "ln":
                        if parts[3] == "weight":
                            pitch_head._ln_modules[idx].weight = v
                        elif parts[3] == "bias":
                            pitch_head._ln_modules[idx].bias = v
                elif len(parts) == 2:
                    # input_proj.weight, ln_input.bias, etc.
                    module = getattr(pitch_head, parts[0])
                    setattr(module, parts[1], v)
        else:
            # Assume standalone checkpoint format (no prefix)
            pitch_head.update(all_weights)

    if emotion_checkpoint:
        emotion_head = EmotionHead(multi_config)
        all_weights = mx.load(emotion_checkpoint)
        # Filter for emotion.* keys and strip prefix (multi-head checkpoint format)
        emotion_weights = {}
        for key, value in all_weights.items():
            if key.startswith("emotion."):
                new_key = key[8:]  # Strip "emotion." prefix
                emotion_weights[new_key] = value
        if emotion_weights:
            # Manually load weights (fc1.weight, fc2.weight, ln.weight, etc.)
            for k, v in emotion_weights.items():
                parts = k.split(".")
                if len(parts) == 2:
                    module = getattr(emotion_head, parts[0])
                    setattr(module, parts[1], v)
        else:
            # Assume standalone checkpoint format (no prefix)
            emotion_head.update(all_weights)

    return ProsodyBeamSearch(
        pitch_head=pitch_head,
        emotion_head=emotion_head,
        config=config,
    )


__all__ = [
    "ProsodyBeamSearch",
    "ProsodyBeamSearchDecoder",
    "ProsodyFeatures",
    "ProsodyBoostConfig",
    "ProsodyStreamingResult",
    "StreamingProsodyDecoder",
    "create_prosody_beam_search",
]
