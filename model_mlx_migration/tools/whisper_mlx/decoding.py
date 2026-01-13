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
Decoding utilities for WhisperMLX.

Implements logit filters for proper transcription:
- SuppressBlank: Suppress blank outputs at beginning
- SuppressTokens: Suppress non-speech tokens
- ApplyTimestampRules: Enforce timestamp pair rules

Based on mlx-whisper decoding.py for compatibility.
"""

import re
import zlib
from collections.abc import Sequence
from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np

# =============================================================================
# OPT-NEW-1: mx.compile for decoder sampling functions
# These functions are hot paths in decoding - compilation optimizes execution
# =============================================================================


@mx.compile
def greedy_decode(logits: mx.array) -> mx.array:
    """
    Greedy decoding: select token with highest logit.

    Compiled for optimal Metal execution.
    """
    return mx.argmax(logits, axis=-1)


@mx.compile
def sample_with_temperature(logits: mx.array, temperature: float) -> mx.array:
    """
    Sample token from logits with temperature scaling.

    Compiled for optimal Metal execution.

    Args:
        logits: Raw logits from decoder, shape (batch, vocab_size)
        temperature: Temperature for sampling (higher = more random)

    Returns:
        Sampled token indices, shape (batch,)
    """
    probs = mx.softmax(logits / temperature, axis=-1)
    return mx.random.categorical(probs)


@mx.compile
def compute_logprobs(logits: mx.array) -> mx.array:
    """
    Compute log probabilities from logits.

    Compiled for optimal Metal execution.
    """
    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


def compression_ratio(text: str) -> float:
    """Calculate compression ratio for quality assessment."""
    text_bytes = text.encode("utf-8")
    return len(text_bytes) / len(zlib.compress(text_bytes))


# =============================================================================
# E4: Hallucination Detection - Pattern-based quality filter
# Detects common Whisper hallucination patterns in transcribed text
# =============================================================================

# Common Whisper hallucination phrases (case-insensitive patterns)
HALLUCINATION_PATTERNS = [
    # YouTube/video endings
    r"thanks?\s+for\s+watching",
    r"please\s+(like\s+and\s+)?subscribe",
    r"see\s+you\s+(in\s+the\s+)?next\s+(video|time)",
    r"don'?t\s+forget\s+to\s+subscribe",
    r"hit\s+(the\s+)?like\s+button",
    r"leave\s+a\s+comment",
    r"click\s+the\s+bell",
    r"notification\s+bell",
    # Presentation endings
    r"thank\s+you\s+for\s+(your\s+)?attention",
    r"any\s+questions\??",
    # Music/audio artifacts
    r"♪+",
    r"\[music\]",
    r"\(music\)",
    r"\[applause\]",
    r"\(applause\)",
    # Empty brackets/content markers
    r"\[\s*\]",
    r"\(\s*\)",
    # Repeated punctuation
    r"\.{4,}",
    r"\?{3,}",
    r"!{3,}",
]

# Pre-compiled regex patterns for efficiency
_HALLUCINATION_REGEX = [re.compile(p, re.IGNORECASE) for p in HALLUCINATION_PATTERNS]


@dataclass
class HallucinationResult:
    """Result of hallucination detection."""
    is_hallucination: bool
    confidence: float  # 0.0-1.0, higher = more likely hallucination
    patterns_matched: list[str]  # Which patterns matched
    repeated_phrase: str | None = None  # If phrase repetition detected
    repetition_count: int = 0


def detect_hallucination(
    text: str,
    compression_threshold: float = 2.4,
    min_repetition_count: int = 3,
    min_text_length: int = 10,
) -> HallucinationResult:
    """
    Detect common hallucination patterns in transcribed text.

    E4 Quality Filter: Pattern-based hallucination detection.

    Hallucinations in Whisper often manifest as:
    1. Repeated phrases (e.g., "Thank you. Thank you. Thank you.")
    2. Common stock phrases (e.g., "Thanks for watching", "Please subscribe")
    3. Very high compression ratio (highly repetitive text)
    4. Empty brackets or repeated punctuation

    Args:
        text: Transcribed text to check
        compression_threshold: Text with compression ratio above this is suspicious
        min_repetition_count: Minimum phrase repetitions to flag as hallucination
        min_text_length: Minimum text length to analyze (shorter texts skipped)

    Returns:
        HallucinationResult with detection details
    """
    if not text or len(text.strip()) < min_text_length:
        return HallucinationResult(
            is_hallucination=False,
            confidence=0.0,
            patterns_matched=[],
        )

    patterns_matched = []
    confidence_factors = []

    # Check against known hallucination patterns
    for i, pattern in enumerate(_HALLUCINATION_REGEX):
        if pattern.search(text):
            patterns_matched.append(HALLUCINATION_PATTERNS[i])
            confidence_factors.append(0.3)  # Each pattern adds 30% confidence

    # Check for phrase repetition using sliding window
    repeated_phrase = None
    repetition_count = 0

    # Normalize text for comparison (lowercase, collapse whitespace)
    normalized = " ".join(text.lower().split())
    words = normalized.split()

    if len(words) >= 6:
        # Check for 2-6 word phrase repetitions
        for phrase_len in range(2, min(7, len(words) // 2)):
            for start in range(len(words) - phrase_len + 1):
                phrase = " ".join(words[start:start + phrase_len])
                count = normalized.count(phrase)
                if count >= min_repetition_count and count > repetition_count:
                    # Found a repeated phrase
                    repetition_count = count
                    repeated_phrase = phrase

        if repeated_phrase:
            # Repetition is highly indicative of hallucination
            # Scale confidence based on repetition count
            rep_confidence = min(0.9, 0.3 + (repetition_count - min_repetition_count) * 0.15)
            confidence_factors.append(rep_confidence)

    # Check compression ratio (already computed elsewhere, but useful as backup)
    try:
        comp_ratio = compression_ratio(text)
        if comp_ratio > compression_threshold:
            # High compression = repetitive text
            excess = comp_ratio - compression_threshold
            comp_confidence = min(0.5, excess * 0.2)
            confidence_factors.append(comp_confidence)
    except Exception:
        pass  # Skip if compression check fails

    # Calculate overall confidence
    if confidence_factors:
        # Use a weighted combination - multiple factors increase confidence
        confidence = min(1.0, sum(confidence_factors) * (1 + 0.1 * len(confidence_factors)))
    else:
        confidence = 0.0

    # Determine if this is a hallucination based on confidence threshold
    is_hallucination = confidence >= 0.5

    return HallucinationResult(
        is_hallucination=is_hallucination,
        confidence=confidence,
        patterns_matched=patterns_matched,
        repeated_phrase=repeated_phrase,
        repetition_count=repetition_count,
    )


def is_hallucination(text: str, threshold: float = 0.5) -> bool:
    """
    Simple check if text is likely a hallucination.

    Convenience wrapper around detect_hallucination().

    Args:
        text: Transcribed text to check
        threshold: Confidence threshold (0.0-1.0)

    Returns:
        True if text is likely a hallucination
    """
    result = detect_hallucination(text)
    return result.confidence >= threshold


@dataclass(frozen=True)
class DecodingOptions:
    """Options for decoding."""
    task: str = "transcribe"
    language: str | None = None
    temperature: float = 0.0
    sample_len: int | None = None
    suppress_tokens: str | Sequence[int] | None = "-1"
    suppress_blank: bool = True
    without_timestamps: bool = False
    max_initial_timestamp: float | None = 1.0
    repetition_penalty: float | None = None  # Penalty for repeated tokens (>1.0 to discourage)


@dataclass
class DecodingResult:
    """Result of decoding."""
    tokens: list[int] = field(default_factory=list)
    text: str = ""
    language: str = "en"
    avg_logprob: float = float("nan")
    no_speech_prob: float = float("nan")
    temperature: float = float("nan")
    compression_ratio: float = float("nan")


class LogitFilter:
    """Base class for logit filters."""

    def apply(self, logits: mx.array, tokens: mx.array) -> mx.array:
        """Apply filter to logits."""
        raise NotImplementedError


class SuppressBlank(LogitFilter):
    """Suppress blank outputs at the beginning of transcription."""

    def __init__(self, tokenizer, sample_begin: int, n_vocab: int):
        self.sample_begin = sample_begin
        mask = np.zeros(n_vocab, np.float32)
        # Suppress space and EOT at the very beginning
        blank_token = tokenizer.encode(" ")
        mask[blank_token + [tokenizer.eot]] = -np.inf
        self.mask = mx.array(mask)

    def apply(self, logits: mx.array, tokens: mx.array) -> mx.array:
        if tokens.shape[1] == self.sample_begin:
            return logits + self.mask
        return logits


class SuppressTokens(LogitFilter):
    """Suppress specific tokens."""

    def __init__(self, suppress_tokens: Sequence[int], n_vocab: int):
        mask = np.zeros(n_vocab, np.float32)
        mask[list(suppress_tokens)] = -np.inf
        self.mask = mx.array(mask)

    def apply(self, logits: mx.array, tokens: mx.array) -> mx.array:
        return logits + self.mask


class FusedSuppressFilter(LogitFilter):
    """
    OPT-NEW-8: Fused filter combining SuppressBlank and SuppressTokens.

    Reduces two mask additions to one by precomputing combined masks.
    The blank suppression is applied conditionally at sample_begin.
    """

    def __init__(
        self,
        tokenizer,
        sample_begin: int,
        suppress_tokens: Sequence[int],
        n_vocab: int,
    ):
        self.sample_begin = sample_begin

        # Base mask: always applied (SuppressTokens)
        base_mask = np.zeros(n_vocab, np.float32)
        base_mask[list(suppress_tokens)] = -np.inf
        self.base_mask = mx.array(base_mask)

        # Combined mask: base + blank suppression (applied at sample_begin)
        combined_mask = base_mask.copy()
        blank_token = tokenizer.encode(" ")
        combined_mask[blank_token + [tokenizer.eot]] = -np.inf
        self.combined_mask = mx.array(combined_mask)

    def apply(self, logits: mx.array, tokens: mx.array) -> mx.array:
        """Apply fused suppression mask."""
        if tokens.shape[1] == self.sample_begin:
            # At sample_begin: apply combined mask (blank + tokens)
            return logits + self.combined_mask
        # After sample_begin: apply base mask only (tokens)
        return logits + self.base_mask


class RepetitionPenalty(LogitFilter):
    """
    Apply penalty to tokens that have already been generated.

    This discourages repetition and helps prevent hallucination loops.
    Follows the standard approach:
    - For positive logits: logit = logit / penalty
    - For negative logits: logit = logit * penalty

    A penalty > 1.0 discourages repetition (typical values: 1.1 to 1.5).
    A penalty of 1.0 has no effect.
    """

    def __init__(self, penalty: float):
        """
        Args:
            penalty: Repetition penalty factor (>1.0 to discourage repetition)
        """
        if penalty <= 0:
            raise ValueError(f"Repetition penalty must be positive, got {penalty}")
        self.penalty = penalty

    def apply(self, logits: mx.array, tokens: mx.array) -> mx.array:
        """Apply repetition penalty to logits for previously generated tokens."""
        if self.penalty == 1.0:
            return logits

        # tokens shape: (batch, seq_len), logits shape: (batch, vocab_size)
        batch_size = logits.shape[0]
        vocab_size = logits.shape[-1]

        # Convert tokens to numpy for efficient unique extraction
        tokens_np = np.array(tokens)

        # Build a mask of which tokens have been seen (1 for seen, 0 for unseen)
        seen_mask = np.zeros((batch_size, vocab_size), dtype=np.float32)
        for b in range(batch_size):
            seq = tokens_np[b]
            # Get unique tokens (excluding negative values which may be padding)
            unique_tokens = np.unique(seq[seq >= 0])
            # Mark these tokens in the mask (clip to vocab size)
            valid_tokens = unique_tokens[unique_tokens < vocab_size]
            seen_mask[b, valid_tokens.astype(np.int32)] = 1.0

        seen_mask_mx = mx.array(seen_mask)

        # Apply penalty using vectorized operations:
        # For seen tokens: if logit > 0, divide by penalty; if logit <= 0, multiply by penalty
        # Unseen tokens: unchanged

        # Create penalty multipliers:
        # For positive logits of seen tokens: multiply by 1/penalty
        # For negative logits of seen tokens: multiply by penalty
        # For unseen tokens: multiply by 1 (no change)

        # penalty_factor will be:
        # - 1/penalty where logit > 0 AND token was seen
        # - penalty where logit <= 0 AND token was seen
        # - 1.0 otherwise

        is_positive = logits > 0
        positive_penalty = mx.where(is_positive, 1.0 / self.penalty, self.penalty)

        # Apply penalty only to seen tokens
        penalty_factor = mx.where(seen_mask_mx > 0, positive_penalty, 1.0)

        return logits * penalty_factor


class ApplyTimestampRules(LogitFilter):
    """
    Apply timestamp rules:
    1. Suppress <|notimestamps|>
    2. Timestamps must appear in pairs (except before EOT)
    3. Timestamps shouldn't decrease
    4. Force timestamp at beginning
    5. If timestamp probability > text probability, force timestamp
    """

    def __init__(
        self,
        tokenizer,
        sample_begin: int,
        max_initial_timestamp_index: int | None,
        n_vocab: int,
        max_audio_timestamp_index: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.sample_begin = sample_begin
        self.max_initial_timestamp_index = max_initial_timestamp_index
        self.max_audio_timestamp_index = max_audio_timestamp_index
        self.n_vocab = n_vocab
        self.timestamp_begin = tokenizer.timestamp_begin
        self.eot = tokenizer.eot
        self.no_timestamps = getattr(tokenizer, 'no_timestamps', None)

    def apply(self, logits: mx.array, tokens: mx.array) -> mx.array:
        n_batch = logits.shape[0]
        mask = np.zeros((n_batch, self.n_vocab), np.float32)

        # Suppress <|notimestamps|>
        if self.no_timestamps is not None:
            mask[:, self.no_timestamps] = -np.inf

        # Timestamp pairing rules
        tokens_list = tokens.tolist()
        for k in range(n_batch):
            seq = tokens_list[k][self.sample_begin:]
            last_was_timestamp = (
                len(seq) >= 1 and seq[-1] >= self.timestamp_begin
            )
            penultimate_was_timestamp = (
                len(seq) < 2 or seq[-2] >= self.timestamp_begin
            )

            if last_was_timestamp:
                if penultimate_was_timestamp:
                    # Two timestamps in a row - must be non-timestamp
                    mask[k, self.timestamp_begin:] = -np.inf
                else:
                    # Timestamp after text - cannot be normal text
                    mask[k, :self.eot] = -np.inf

            # Timestamps shouldn't decrease
            timestamps = [
                i for i, v in enumerate(seq) if v > self.timestamp_begin
            ]
            if len(timestamps) > 0:
                last_timestamp = timestamps[-1]
                if not last_timestamp or penultimate_was_timestamp:
                    last_timestamp += 1
                # Forbid timestamps before the last one
                mask[k, self.timestamp_begin:self.timestamp_begin + last_timestamp] = -np.inf

        # Force timestamp at beginning
        if len(tokens_list[0]) == self.sample_begin:
            mask[:, :self.timestamp_begin] = -np.inf

            # Apply max_initial_timestamp
            if self.max_initial_timestamp_index is not None:
                last_allowed = self.timestamp_begin + self.max_initial_timestamp_index
                mask[:, last_allowed + 1:] = -np.inf

        # Always suppress timestamps beyond audio duration (for variable-length mode)
        if self.max_audio_timestamp_index is not None:
            max_allowed = self.timestamp_begin + self.max_audio_timestamp_index
            mask[:, max_allowed + 1:] = -np.inf

            # Force EOT when we've reached near the max audio timestamp
            # This prevents hallucination when decoder can't progress in time
            # Use a threshold to handle floating point precision issues
            threshold = 5  # Allow 5 timestamp indices (~0.1s) tolerance
            for k in range(n_batch):
                seq = tokens_list[k][self.sample_begin:]
                # Check if we've hit the max timestamp
                if len(seq) >= 2:
                    last_token = seq[-1]
                    penult_token = seq[-2]
                    # If last two tokens are both timestamps near max, force EOT
                    if last_token >= self.timestamp_begin and penult_token >= self.timestamp_begin:
                        last_ts_idx = last_token - self.timestamp_begin
                        penult_ts_idx = penult_token - self.timestamp_begin
                        # If both timestamps are near the max (within threshold), force EOT
                        near_max = self.max_audio_timestamp_index - threshold
                        if last_ts_idx >= near_max and penult_ts_idx >= near_max:
                            # Suppress everything except EOT
                            mask[k, :self.eot] = -np.inf
                            mask[k, self.eot + 1:] = -np.inf

        # Convert to MLX array
        mask_mx = mx.array(mask)

        # If timestamp probability > text probability, sample timestamp
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        timestamp_logprob = logprobs[:, self.timestamp_begin:].logsumexp(axis=-1, keepdims=True)
        max_text_logprob = logprobs[:, :self.timestamp_begin].max(axis=-1, keepdims=True)

        # Where timestamp is more likely, suppress text tokens
        timestamp_preferred = timestamp_logprob > max_text_logprob
        text_mask = mx.where(
            timestamp_preferred,
            mx.full(mask_mx[:, :self.timestamp_begin].shape, -mx.inf),
            mask_mx[:, :self.timestamp_begin],
        )

        # Build final mask
        final_mask = mx.concatenate([text_mask, mask_mx[:, self.timestamp_begin:]], axis=1)

        return logits + final_mask


def get_suppress_tokens(tokenizer, suppress_tokens_option) -> tuple[int, ...]:
    """Get the list of tokens to suppress."""
    suppress_tokens = suppress_tokens_option

    if isinstance(suppress_tokens, str):
        suppress_tokens = [int(t) for t in suppress_tokens.split(",")]

    if suppress_tokens is None:
        suppress_tokens = []

    # -1 means suppress non-speech tokens
    if -1 in suppress_tokens:
        suppress_tokens = [t for t in suppress_tokens if t >= 0]
        # Add non-speech tokens
        if hasattr(tokenizer, 'non_speech_tokens'):
            suppress_tokens.extend(tokenizer.non_speech_tokens)

    # Always suppress special task tokens
    suppress_tokens.extend([
        tokenizer.transcribe,
        tokenizer.translate,
        tokenizer.sot,
        tokenizer.sot_prev,
        tokenizer.sot_lm,
    ])

    # Suppress all language tokens - they should only appear in SOT sequence
    if hasattr(tokenizer, 'all_language_tokens'):
        suppress_tokens.extend(tokenizer.all_language_tokens)

    # Suppress no_speech if present
    if tokenizer.no_speech is not None:
        suppress_tokens.append(tokenizer.no_speech)

    return tuple(sorted(set(suppress_tokens)))


def build_logit_filters(
    tokenizer,
    options: DecodingOptions,
    sample_begin: int,
    n_vocab: int,
    precision: float = 0.02,
    audio_duration: float | None = None,
) -> list[LogitFilter]:
    """
    Build the list of logit filters based on options.

    OPT-NEW-8: Uses FusedSuppressFilter when both suppress_blank and
    suppress_tokens are enabled, reducing two filter applications to one.

    Args:
        tokenizer: Whisper tokenizer
        options: Decoding options
        sample_begin: Index where sampling starts (after SOT sequence)
        n_vocab: Vocabulary size
        precision: Time precision per encoder position (for max_initial_timestamp)
        audio_duration: Actual audio duration in seconds (for variable-length mode)

    Returns:
        List of LogitFilter instances
    """
    filters = []

    # OPT-NEW-8: Fuse SuppressBlank and SuppressTokens when both are enabled
    # This reduces two mask additions to one in the decode loop
    if options.suppress_blank and options.suppress_tokens:
        # Use fused filter for better performance
        suppress_tokens = get_suppress_tokens(tokenizer, options.suppress_tokens)
        filters.append(FusedSuppressFilter(
            tokenizer, sample_begin, suppress_tokens, n_vocab,
        ))
    else:
        # Fallback to individual filters
        if options.suppress_blank:
            filters.append(SuppressBlank(tokenizer, sample_begin, n_vocab))
        if options.suppress_tokens:
            suppress_tokens = get_suppress_tokens(tokenizer, options.suppress_tokens)
            filters.append(SuppressTokens(suppress_tokens, n_vocab))

    # Timestamp rules
    if not options.without_timestamps:
        max_initial_timestamp_index = None
        if options.max_initial_timestamp is not None:
            max_initial_timestamp_index = round(options.max_initial_timestamp / precision)

        # For variable-length mode, limit timestamps to actual audio duration
        max_audio_timestamp_index = None
        if audio_duration is not None:
            max_audio_timestamp_index = round(audio_duration / precision)

        filters.append(ApplyTimestampRules(
            tokenizer,
            sample_begin,
            max_initial_timestamp_index,
            n_vocab,
            max_audio_timestamp_index=max_audio_timestamp_index,
        ))

    # Repetition penalty (helps prevent hallucination loops)
    if options.repetition_penalty is not None and options.repetition_penalty != 1.0:
        filters.append(RepetitionPenalty(options.repetition_penalty))

    return filters


def apply_filters(
    logits: mx.array,
    tokens: mx.array,
    filters: list[LogitFilter],
) -> mx.array:
    """Apply all logit filters to logits."""
    for f in filters:
        logits = f.apply(logits, tokens)
    return logits
