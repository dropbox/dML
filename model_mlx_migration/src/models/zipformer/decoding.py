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
Decoding algorithms for RNN-T ASR model.

Implements:
- Greedy search (fastest, simplest)
- Beam search (better accuracy, more compute)
"""

from dataclasses import dataclass

import mlx.core as mx

from .decoder import Decoder
from .joiner import Joiner


@dataclass
class DecodingResult:
    """Result from decoding a single utterance."""
    tokens: list[int]  # Token IDs (excluding blank)
    score: float = 0.0  # Log probability score


def greedy_search(
    decoder: Decoder,
    joiner: Joiner,
    encoder_out: mx.array,
    encoder_out_len: int | None = None,
    max_sym_per_frame: int = 1,
) -> DecodingResult:
    """
    Greedy search decoding for RNN-T.

    At each encoder frame, emit the most likely token until blank
    or max_sym_per_frame is reached, then move to the next frame.

    Args:
        decoder: Decoder (predictor) module.
        joiner: Joiner module.
        encoder_out: Encoder output of shape (T, encoder_dim).
        encoder_out_len: Optional length (defaults to T).
        max_sym_per_frame: Maximum non-blank symbols per encoder frame.

    Returns:
        DecodingResult with token list and score.
    """
    T = encoder_out.shape[0]
    if encoder_out_len is None:
        encoder_out_len = T

    blank_id = decoder.blank_id
    context_size = decoder.context_size

    # Initialize decoder context with blank tokens
    # Shape: (1, context_size - 1)
    context = mx.full((1, context_size - 1), blank_id, dtype=mx.int32)
    context_embed = decoder.embedding(context)  # (1, context_size-1, decoder_dim)

    hyps: list[int] = []
    total_score = 0.0

    # Current token (starts with blank)
    current_token = mx.array([[blank_id]], dtype=mx.int32)
    prev_was_blank = True

    t = 0
    while t < encoder_out_len:
        # Get encoder output for this frame
        encoder_frame = encoder_out[t:t+1, :]  # (1, encoder_dim)
        encoder_frame = mx.expand_dims(encoder_frame, axis=0)  # (1, 1, encoder_dim)

        sym_count = 0
        while sym_count < max_sym_per_frame:
            # Run decoder if previous output was not blank
            if prev_was_blank:
                # Use cached context
                decoder_out = decoder.forward_one_step(current_token, context_embed)
            else:
                # Update context with new token and run decoder
                decoder_out = decoder.forward_one_step(current_token, context_embed)

            # Run joiner
            logits = joiner.forward_streaming(encoder_frame, decoder_out)  # (1, 1, vocab)
            mx.eval(logits)

            # Get log probabilities and best token
            log_probs = mx.softmax(logits, axis=-1)
            log_probs = mx.log(log_probs + 1e-10)
            best_token = int(mx.argmax(logits[0, 0, :]).item())
            best_score = float(log_probs[0, 0, best_token].item())

            if best_token == blank_id:
                # Blank token - move to next frame
                prev_was_blank = True
                total_score += best_score
                break
            # Non-blank token - add to hypothesis
            hyps.append(best_token)
            total_score += best_score
            sym_count += 1
            prev_was_blank = False

            # Update context for next decoder step
            # The context should contain the PREVIOUS tokens (before the new one)
            # So we shift in the OLD current_token, not the new one
            new_token = mx.array([[best_token]], dtype=mx.int32)

            # Shift context: add the old current_token embedding
            if context_size > 1:
                old_current_embed = decoder.embedding(current_token)  # (1, 1, decoder_dim)
                context_embed = mx.concatenate(
                    [context_embed[:, 1:, :], old_current_embed],
                    axis=1,
                )
            current_token = new_token

        t += 1

    return DecodingResult(tokens=hyps, score=total_score)


@dataclass
class StreamingDecoderState:
    """State for streaming greedy search decoding."""
    context_embed: mx.array  # (1, context_size-1, decoder_dim)
    current_token: mx.array  # (1, 1) - last non-blank token emitted
    tokens: list[int]  # Accumulated tokens so far
    total_score: float  # Running log probability score
    prev_was_blank: bool  # Whether previous output was blank


def greedy_search_streaming(
    decoder: Decoder,
    joiner: Joiner,
    encoder_out: mx.array,
    state: StreamingDecoderState | None = None,
    max_sym_per_frame: int = 1,
) -> tuple[DecodingResult, StreamingDecoderState]:
    """
    Streaming greedy search decoding for RNN-T.

    Processes a chunk of encoder output and returns partial result with
    state for continuation. Call repeatedly with consecutive encoder chunks.

    Args:
        decoder: Decoder (predictor) module.
        joiner: Joiner module.
        encoder_out: Encoder output chunk of shape (T, encoder_dim) or (batch, T, encoder_dim).
        state: Previous decoder state. If None, initializes fresh state.
        max_sym_per_frame: Maximum non-blank symbols per encoder frame.

    Returns:
        Tuple of (DecodingResult with tokens from this chunk, updated state).
    """
    # Handle batched input (take first sample for now)
    if encoder_out.ndim == 3:
        encoder_out = encoder_out[0]

    T = encoder_out.shape[0]
    blank_id = decoder.blank_id
    context_size = decoder.context_size

    # Initialize state if needed
    if state is None:
        context = mx.full((1, context_size - 1), blank_id, dtype=mx.int32)
        context_embed = decoder.embedding(context)
        current_token = mx.array([[blank_id]], dtype=mx.int32)
        state = StreamingDecoderState(
            context_embed=context_embed,
            current_token=current_token,
            tokens=[],
            total_score=0.0,
            prev_was_blank=True,
        )

    context_embed = state.context_embed
    current_token = state.current_token
    prev_was_blank = state.prev_was_blank
    chunk_tokens: list[int] = []
    chunk_score = 0.0

    for t in range(T):
        encoder_frame = encoder_out[t:t+1, :]  # (1, encoder_dim)
        encoder_frame = mx.expand_dims(encoder_frame, axis=0)  # (1, 1, encoder_dim)

        sym_count = 0
        while sym_count < max_sym_per_frame:
            # Run decoder
            decoder_out = decoder.forward_one_step(current_token, context_embed)

            # Run joiner
            logits = joiner.forward_streaming(encoder_frame, decoder_out)
            mx.eval(logits)

            # Get best token
            log_probs = mx.log(mx.softmax(logits, axis=-1) + 1e-10)
            best_token = int(mx.argmax(logits[0, 0, :]).item())
            best_score = float(log_probs[0, 0, best_token].item())

            if best_token == blank_id:
                prev_was_blank = True
                chunk_score += best_score
                break
            # Non-blank token
            chunk_tokens.append(best_token)
            chunk_score += best_score
            sym_count += 1
            prev_was_blank = False

            # Update context for next decoder step
            # The context should contain the PREVIOUS tokens (before the new one)
            # So we shift in the OLD current_token, not the new one
            new_token = mx.array([[best_token]], dtype=mx.int32)

            if context_size > 1:
                old_current_embed = decoder.embedding(current_token)  # (1, 1, decoder_dim)
                context_embed = mx.concatenate(
                    [context_embed[:, 1:, :], old_current_embed],
                    axis=1,
                )
            current_token = new_token

    # Update state
    new_state = StreamingDecoderState(
        context_embed=context_embed,
        current_token=current_token,
        tokens=state.tokens + chunk_tokens,
        total_score=state.total_score + chunk_score,
        prev_was_blank=prev_was_blank,
    )

    # Return tokens from this chunk and new state
    return DecodingResult(tokens=chunk_tokens, score=chunk_score), new_state


def greedy_search_batch(
    decoder: Decoder,
    joiner: Joiner,
    encoder_out: mx.array,
    encoder_out_lens: mx.array,
    max_sym_per_frame: int = 1,
) -> list[DecodingResult]:
    """
    Batched greedy search decoding for RNN-T.

    For simplicity, this processes each utterance sequentially.
    For true batched processing, use streaming decoding.

    Args:
        decoder: Decoder (predictor) module.
        joiner: Joiner module.
        encoder_out: Encoder output of shape (batch, T, encoder_dim).
        encoder_out_lens: Length of each utterance (batch,).
        max_sym_per_frame: Maximum non-blank symbols per encoder frame.

    Returns:
        List of DecodingResult for each utterance.
    """
    batch_size = encoder_out.shape[0]
    results = []

    for i in range(batch_size):
        length = int(encoder_out_lens[i].item())
        result = greedy_search(
            decoder=decoder,
            joiner=joiner,
            encoder_out=encoder_out[i],
            encoder_out_len=length,
            max_sym_per_frame=max_sym_per_frame,
        )
        results.append(result)

    return results


def modified_beam_search(
    decoder: Decoder,
    joiner: Joiner,
    encoder_out: mx.array,
    encoder_out_len: int | None = None,
    beam_size: int = 4,
    max_sym_per_frame: int = 3,
) -> DecodingResult:
    """
    Modified beam search for RNN-T.

    Maintains multiple hypothesis beams and prunes based on score.
    Returns the best hypothesis after processing all frames.

    Args:
        decoder: Decoder (predictor) module.
        joiner: Joiner module.
        encoder_out: Encoder output of shape (T, encoder_dim).
        encoder_out_len: Optional length (defaults to T).
        beam_size: Number of beams to maintain.
        max_sym_per_frame: Maximum non-blank symbols per encoder frame.

    Returns:
        DecodingResult with best hypothesis tokens and score.
    """
    T = encoder_out.shape[0]
    if encoder_out_len is None:
        encoder_out_len = T

    blank_id = decoder.blank_id
    context_size = decoder.context_size

    # Beam: List of (tokens, context_embed, score)
    initial_context = mx.full((1, context_size - 1), blank_id, dtype=mx.int32)
    initial_context_embed = decoder.embedding(initial_context)

    # Each beam entry: (token_list, context_embed, score)
    beams = [([], initial_context_embed, 0.0)]

    for t in range(encoder_out_len):
        # Get encoder output for this frame
        encoder_frame = encoder_out[t:t+1, :]
        encoder_frame = mx.expand_dims(encoder_frame, axis=0)

        new_beams = []

        for tokens, context_embed, score in beams:
            # Get last token (or blank if empty)
            if len(tokens) == 0:
                current_token = mx.array([[blank_id]], dtype=mx.int32)
            else:
                current_token = mx.array([[tokens[-1]]], dtype=mx.int32)

            # Process this beam for this frame
            for _ in range(max_sym_per_frame + 1):  # +1 for blank
                # Run decoder
                decoder_out = decoder.forward_one_step(current_token, context_embed)

                # Run joiner
                logits = joiner.forward_streaming(encoder_frame, decoder_out)
                mx.eval(logits)

                # Get log probabilities
                log_probs = mx.log(mx.softmax(logits, axis=-1) + 1e-10)

                # Get top-k tokens - MLX topk returns only values, use argsort for indices
                top_k = min(beam_size, logits.shape[-1])
                log_probs_1d = log_probs[0, 0, :]
                # Sort in descending order by negating
                sorted_indices = mx.argsort(-log_probs_1d)
                top_indices = sorted_indices[:top_k]
                top_scores = log_probs_1d[top_indices]
                mx.eval(top_scores, top_indices)

                for k in range(top_k):
                    token = int(top_indices[k].item())
                    token_score = float(top_scores[k].item())
                    new_score = score + token_score

                    if token == blank_id:
                        # Blank - keep same hypothesis
                        new_beams.append((tokens.copy(), context_embed, new_score))
                    else:
                        # Non-blank - extend hypothesis
                        new_tokens = tokens + [token]
                        new_token_arr = mx.array([[token]], dtype=mx.int32)
                        new_embed = decoder.embedding(new_token_arr)
                        if context_size > 1:
                            new_context = mx.concatenate(
                                [context_embed[:, 1:, :], new_embed],
                                axis=1,
                            )
                        else:
                            new_context = new_embed
                        new_beams.append((new_tokens, new_context, new_score))

                # If best extension is blank, stop extending this beam
                if int(top_indices[0].item()) == blank_id:
                    break

                # Update for next symbol in same frame
                if int(top_indices[0].item()) != blank_id:
                    current_token = mx.array([[int(top_indices[0].item())]], dtype=mx.int32)
                    new_embed = decoder.embedding(current_token)
                    if context_size > 1:
                        context_embed = mx.concatenate(
                            [context_embed[:, 1:, :], new_embed],
                            axis=1,
                        )

        # Prune to beam_size
        new_beams.sort(key=lambda x: x[2], reverse=True)
        beams = new_beams[:beam_size]

    # Return best beam
    if beams:
        best_tokens, _, best_score = beams[0]
        return DecodingResult(tokens=best_tokens, score=best_score)
    return DecodingResult(tokens=[], score=float('-inf'))


# Simple token-to-text decoder using BPE
class TokenDecoder:
    """
    Decode token IDs to text using a vocabulary file.

    Supports sentencepiece/BPE-style vocabularies.
    """

    def __init__(self, vocab_path: str):
        """
        Initialize token decoder.

        Args:
            vocab_path: Path to vocabulary file (one token per line).
        """
        self.id_to_token = {}
        with open(vocab_path, encoding='utf-8') as f:
            for i, line in enumerate(f):
                # Handle BPE format: "token score" or just "token"
                parts = line.strip().split()
                if parts:
                    token = parts[0]
                    self.id_to_token[i] = token

    def decode(self, token_ids: list[int]) -> str:
        """
        Decode token IDs to text.

        Args:
            token_ids: List of token IDs.

        Returns:
            Decoded text string.
        """
        tokens = []
        for tid in token_ids:
            if tid in self.id_to_token:
                token = self.id_to_token[tid]
                # Handle sentencepiece special characters
                if token.startswith('▁'):
                    token = ' ' + token[1:]
                tokens.append(token)

        text = ''.join(tokens).strip()
        return text
