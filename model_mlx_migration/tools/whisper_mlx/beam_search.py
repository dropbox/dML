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
Beam search decoding for WhisperMLX.

Implements beam search with proper handling of:
- Timestamp constraints (via logit filters)
- Variable beam sizes
- KV cache per beam (memory efficient via lazy cloning)
- Length normalization for fair comparison
- Early stopping when all beams finish

Quality improvement at cost of compute: beam_size=5 typically reduces WER by 0.5-1%.

References:
- OpenAI Whisper beam search implementation
- MLX-Whisper decode.py patterns
"""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .decoding import (
    DecodingOptions,
    apply_filters,
    build_logit_filters,
    compute_logprobs,
)


@dataclass
class Beam:
    """
    A single beam hypothesis in beam search.

    Tracks the token sequence, log probability, and state.
    """
    tokens: list[int]
    log_prob: float
    finished: bool = False
    no_speech_prob: float = 0.0
    # KV cache is stored separately to allow efficient batching


@dataclass
class BeamSearchResult:
    """Result from beam search decoding."""
    tokens: list[int]
    text: str = ""
    log_prob: float = 0.0
    normalized_score: float = 0.0
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0


class BeamSearchDecoder:
    """
    Beam search decoder for WhisperMLX.

    Maintains multiple hypotheses during decoding, selecting the best
    one based on log probability. Provides better transcription quality
    than greedy decoding at the cost of beam_size * compute.

    Key optimizations:
    - Batched forward passes when possible
    - Lazy KV cache cloning (only clone when beams diverge)
    - Early stopping when all beams finish
    - Length normalization for fair beam comparison
    """

    def __init__(
        self,
        model: nn.Module,
        beam_size: int = 5,
        length_penalty: float = 1.0,
        patience: float = 1.0,
    ):
        """
        Args:
            model: WhisperMLX model instance
            beam_size: Number of beams to maintain (default: 5)
            length_penalty: Length penalty for beam scoring (default: 1.0)
                           > 1.0 encourages longer sequences
                           < 1.0 encourages shorter sequences
            patience: Early stopping patience multiplier (default: 1.0)
                     Stop early if best beam score hasn't improved for
                     patience * beam_size consecutive steps
        """
        self.model = model
        self.beam_size = beam_size
        self.length_penalty = length_penalty
        self.patience = patience

        # Statistics
        self.decode_steps = 0
        self.final_beam_count = 0

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
        Decode audio features using beam search.

        Args:
            audio_features: Encoded audio from encoder, shape (1, T, D)
            tokenizer: Whisper tokenizer
            options: Decoding options (temperature must be 0 for beam search)
            sample_begin: Index where sampling starts (after SOT sequence)
            n_vocab: Vocabulary size
            precision: Timestamp precision (0.02 = 20ms)
            audio_duration: Actual audio duration for variable-length mode
            max_tokens: Maximum tokens to decode

        Returns:
            BeamSearchResult with best hypothesis
        """
        self.decode_steps = 0

        # Build initial tokens (SOT sequence)
        initial_tokens = list(tokenizer.sot_sequence)

        # Build logit filters
        logit_filters = build_logit_filters(
            tokenizer,
            options,
            sample_begin=sample_begin,
            n_vocab=n_vocab,
            precision=precision,
            audio_duration=audio_duration,
        )

        # Initialize beams
        beams = [Beam(tokens=list(initial_tokens), log_prob=0.0)]
        finished_beams: list[Beam] = []

        # KV caches - one per beam, start with None
        kv_caches: list[list | None] = [None]

        # First step: process initial tokens to get first prediction
        tokens_tensor = mx.array([initial_tokens])
        logits, kv_cache, _, _ = self.model.decoder(
            tokens_tensor, audio_features, kv_cache=None,
        )
        logits = logits[:, -1].astype(mx.float32)

        # Capture no_speech_prob from first logits
        no_speech_prob = 0.0
        if tokenizer.no_speech is not None:
            probs_at_sot = mx.softmax(logits, axis=-1)
            no_speech_prob = float(probs_at_sot[0, tokenizer.no_speech])

        # Apply filters
        filtered_logits = apply_filters(logits, tokens_tensor, logit_filters)

        # Get log probabilities
        log_probs = compute_logprobs(filtered_logits)

        # Initialize beams with top-k first tokens
        # MLX topk returns only values, so we use argsort to get indices
        top_k_indices = mx.argsort(-log_probs[0])[:self.beam_size]  # Sort descending
        top_k_log_probs = log_probs[0][top_k_indices]

        beams = []
        kv_caches = []
        for i in range(self.beam_size):
            token = int(top_k_indices[i])
            lp = float(top_k_log_probs[i])

            new_tokens = initial_tokens + [token]
            beam = Beam(tokens=new_tokens, log_prob=lp, no_speech_prob=no_speech_prob)

            if token == tokenizer.eot:
                beam.finished = True
                finished_beams.append(beam)
            else:
                beams.append(beam)
                # Clone KV cache for each beam
                kv_caches.append(self._clone_kv_cache(kv_cache))

        mx.eval(*[c for kv in kv_caches if kv for c in kv for pair in c if pair for t in pair])
        self.decode_steps = 1

        # Main decode loop
        max_steps = max_tokens - 1  # Already generated 1 token
        patience_counter = 0
        best_finished_score = float('-inf')

        for _step in range(max_steps):
            if not beams:
                break

            self.decode_steps += 1

            # Process each beam
            all_candidates: list[tuple[int, int, float, Beam, list | None]] = []

            for beam_idx, (beam, kv_cache) in enumerate(zip(beams, kv_caches, strict=False)):
                # Get prediction for this beam
                last_token = mx.array([[beam.tokens[-1]]])
                logits, new_kv, _, _ = self.model.decoder(
                    last_token, audio_features, kv_cache=kv_cache,
                )
                logits = logits[:, -1].astype(mx.float32)

                # Apply filters with current beam's token history
                token_context = mx.array([beam.tokens])
                filtered_logits = apply_filters(logits, token_context, logit_filters)

                # Get log probabilities
                log_probs = compute_logprobs(filtered_logits)

                # Get top-k candidates for this beam
                # MLX topk returns only values, so we use argsort to get indices
                top_k_indices = mx.argsort(-log_probs[0])[:self.beam_size]
                top_k_log_probs = log_probs[0][top_k_indices]

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
            used_kv: dict[int, list] = {}  # Track which KV caches we've used

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
                # Check if best active beam can beat best finished beam
                best_active_lp = max(b.log_prob for b in beams) if beams else float('-inf')
                # Optimistic estimate: assume all remaining tokens have log_prob = 0
                best_possible_score = self._compute_score(
                    Beam(tokens=beams[0].tokens if beams else [], log_prob=best_active_lp),
                    sample_begin,
                )

                if best_possible_score < best_finished_score:
                    # Can't beat best finished beam, stop early
                    break

                patience_counter += 1
                if patience_counter >= int(self.patience * self.beam_size):
                    break

        # Select best beam
        all_finished = finished_beams + list(beams)
        if not all_finished:
            # Shouldn't happen, but handle gracefully
            return BeamSearchResult(tokens=initial_tokens[sample_begin:])

        # Score all beams
        scored_beams = [
            (self._compute_score(b, sample_begin), b)
            for b in all_finished
        ]
        scored_beams.sort(key=lambda x: x[0], reverse=True)

        best_score, best_beam = scored_beams[0]
        output_tokens = best_beam.tokens[sample_begin:]

        # Remove trailing EOT
        while output_tokens and output_tokens[-1] == tokenizer.eot:
            output_tokens = output_tokens[:-1]

        # Compute average log probability
        n_tokens = len(output_tokens)
        avg_logprob = best_beam.log_prob / max(n_tokens, 1)

        self.final_beam_count = len(all_finished)

        return BeamSearchResult(
            tokens=output_tokens,
            text=tokenizer.decode(output_tokens),
            log_prob=best_beam.log_prob,
            normalized_score=best_score,
            avg_logprob=avg_logprob,
            no_speech_prob=best_beam.no_speech_prob,
        )

    def _compute_score(self, beam: Beam, sample_begin: int) -> float:
        """
        Compute normalized score for beam comparison.

        Uses length normalization to avoid bias toward shorter sequences.

        Args:
            beam: Beam hypothesis
            sample_begin: Index where sampling started

        Returns:
            Normalized score
        """
        # Number of generated tokens (excluding SOT sequence)
        n_tokens = len(beam.tokens) - sample_begin
        if n_tokens <= 0:
            return beam.log_prob

        # Length penalty: divide by length^alpha
        # alpha = 1.0 means pure average
        # alpha > 1.0 favors longer sequences
        # alpha < 1.0 favors shorter sequences
        length_factor = ((5 + n_tokens) / 6) ** self.length_penalty

        return beam.log_prob / length_factor

    def _clone_kv_cache(self, kv_cache: list | None) -> list | None:
        """
        Clone KV cache for beam branching.

        Creates a copy of the KV cache so modifications to one beam
        don't affect others.

        Args:
            kv_cache: KV cache to clone (list of layer caches)

        Returns:
            Cloned KV cache
        """
        if kv_cache is None:
            return None

        cloned = []
        for layer_cache in kv_cache:
            if layer_cache is None:
                cloned.append(None)
                continue

            self_kv, cross_kv = layer_cache
            self_k, self_v = self_kv

            # MLX arrays are copy-on-write, so this is efficient
            cloned_self_kv = (mx.array(self_k), mx.array(self_v))
            cloned_cross_kv = cross_kv  # Cross-attention cache is shared

            cloned.append((cloned_self_kv, cloned_cross_kv))

        return cloned


def decode_with_beam_search(
    model: nn.Module,
    audio_features: mx.array,
    tokenizer,
    beam_size: int = 5,
    length_penalty: float = 1.0,
    max_initial_timestamp: float = 1.0,
    audio_duration: float | None = None,
    max_tokens: int = 224,
) -> tuple[list[int], list[dict], float, float]:
    """
    Convenience function for beam search decoding.

    Args:
        model: WhisperMLX model instance
        audio_features: Encoded audio features
        tokenizer: Whisper tokenizer
        beam_size: Number of beams (default: 5)
        length_penalty: Length penalty for scoring
        max_initial_timestamp: Maximum initial timestamp
        audio_duration: Actual audio duration (for variable-length mode)
        max_tokens: Maximum tokens to decode

    Returns:
        Tuple of (tokens, segments, avg_logprob, no_speech_prob)
    """
    decoder = BeamSearchDecoder(
        model=model,
        beam_size=beam_size,
        length_penalty=length_penalty,
    )

    options = DecodingOptions(
        temperature=0.0,  # Beam search requires greedy selection
        max_initial_timestamp=max_initial_timestamp,
        suppress_blank=True,
        suppress_tokens="-1",
        without_timestamps=False,
    )

    # Get sample_begin from tokenizer
    sample_begin = len(list(tokenizer.sot_sequence))

    result = decoder.decode(
        audio_features=audio_features,
        tokenizer=tokenizer,
        options=options,
        sample_begin=sample_begin,
        n_vocab=model.config.n_vocab,
        precision=getattr(model.decoder, 'precision', 0.02),
        audio_duration=audio_duration,
        max_tokens=max_tokens,
    )

    # Parse segments from timestamp tokens
    segments = _parse_segments(result.tokens, tokenizer, getattr(model.decoder, 'precision', 0.02))

    return result.tokens, segments, result.avg_logprob, result.no_speech_prob


def _parse_segments(
    tokens: list[int],
    tokenizer,
    precision: float,
) -> list[dict]:
    """Parse timestamp tokens into segments."""
    timestamp_begin = tokenizer.timestamp_begin
    segments = []
    current_tokens = []
    start_time = None
    last_time = 0.0

    for token in tokens:
        if token >= timestamp_begin:
            time = (token - timestamp_begin) * precision
            if start_time is None:
                start_time = time
            else:
                if current_tokens:
                    text = tokenizer.decode(current_tokens).strip()
                    if text:
                        segments.append({
                            "start": start_time,
                            "end": time,
                            "text": text,
                        })
                current_tokens = []
                start_time = time
            last_time = time
        else:
            current_tokens.append(token)

    # Handle trailing tokens
    if current_tokens and start_time is not None:
        text = tokenizer.decode(current_tokens).strip()
        if text:
            segments.append({
                "start": start_time,
                "end": last_time,
                "text": text,
            })

    return segments


__all__ = [
    "BeamSearchDecoder",
    "BeamSearchResult",
    "Beam",
    "decode_with_beam_search",
]
