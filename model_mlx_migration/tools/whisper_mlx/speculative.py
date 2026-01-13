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
Speculative decoding for WhisperMLX.

Uses a smaller draft model (distil-whisper) to generate candidate tokens,
then verifies them with the main model in parallel. This can provide
1.5-2x speedup for autoregressive decoding.

Key insight: The main model can verify K tokens in a single forward pass,
while generating K tokens sequentially takes K forward passes. If the draft
model is accurate enough, most tokens will be accepted.

KV Cache Optimization (Worker #1189):
The critical optimization is to reuse KV cache between iterations:
- Context tokens have their KV cache entries computed once
- Verification only processes new draft tokens with existing cache
- On rejection, cache is "rolled back" by rebuilding from accepted tokens

References:
- Fast Inference from Transformers via Speculative Decoding (2022)
  https://arxiv.org/abs/2211.17192
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ctc_head import CTCDraftHead

import mlx.core as mx
import mlx.nn as nn

from .decoding import (
    DecodingOptions,
    apply_filters,
    build_logit_filters,
)


class SpeculativeDecoder:
    """
    Speculative decoding with draft model verification.

    The draft model (distil-whisper-large-v3) has 2 decoder layers vs 32.
    It generates candidate tokens quickly, which the main model verifies
    in parallel.

    Algorithm:
    1. Draft model generates K candidate tokens
    2. Main model computes logits for all K tokens in one forward pass
    3. For each position i (0 to K-1):
       - If main model agrees with draft: accept token
       - Else: reject and resample from main model at position i
    4. Continue from last accepted position

    This is LOSSLESS because we always use the main model's distribution
    for the final token selection.
    """

    def __init__(
        self,
        main_model: nn.Module,
        draft_model: nn.Module,
        draft_tokens: int = 5,
    ):
        """
        Args:
            main_model: Main whisper model (target)
            draft_model: Draft whisper model (distil)
            draft_tokens: Number of tokens to draft per iteration
        """
        self.main_model = main_model
        self.draft_model = draft_model
        self.draft_tokens = draft_tokens

        # Statistics tracking
        self.total_tokens = 0
        self.accepted_tokens = 0
        self.iterations = 0

    def reset_stats(self):
        """Reset acceptance statistics."""
        self.total_tokens = 0
        self.accepted_tokens = 0
        self.iterations = 0

    @property
    def acceptance_rate(self) -> float:
        """Return acceptance rate of draft tokens."""
        if self.total_tokens == 0:
            return 0.0
        return self.accepted_tokens / self.total_tokens

    @property
    def tokens_per_iteration(self) -> float:
        """Return average tokens generated per decode iteration."""
        if self.iterations == 0:
            return 0.0
        return self.total_tokens / self.iterations

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
        draft_audio_features: mx.array | None = None,
    ) -> tuple[list[int], list[dict]]:
        """
        Decode with speculative decoding and KV cache optimization.

        KV Cache Strategy (Worker #1189):
        - Build initial KV cache from context (SOT sequence)
        - During verification, use existing cache + process new tokens
        - On acceptance: cache is valid, continue
        - On rejection: rebuild cache from accepted tokens only

        Args:
            audio_features: Encoded audio from main model encoder
            tokenizer: Whisper tokenizer
            options: Decoding options
            sample_begin: Index where sampling starts
            n_vocab: Vocabulary size
            precision: Timestamp precision
            audio_duration: Actual audio duration (for variable-length mode)
            max_tokens: Maximum tokens to decode
            draft_audio_features: Encoded audio from draft model encoder (if None, use audio_features)

        Returns:
            Tuple of (token_list, segments_list)
        """
        self.reset_stats()

        # Use same audio features for both models if not specified
        if draft_audio_features is None:
            draft_audio_features = audio_features

        # Build initial tokens
        initial_tokens = list(tokenizer.sot_sequence)
        all_tokens = list(initial_tokens)

        # Build logit filters for both models
        logit_filters = build_logit_filters(
            tokenizer,
            options,
            sample_begin=sample_begin,
            n_vocab=n_vocab,
            precision=precision,
            audio_duration=audio_duration,
        )

        # Initialize KV caches
        # Build initial main KV cache from context (all but last token)
        # We need to keep the last token out so we can include it in verification
        main_kv = None
        if len(all_tokens) > 1:
            # Process context[:-1] to build KV cache
            context_tokens = mx.array([all_tokens[:-1]])
            _, main_kv, _, _ = self.main_model.decoder(
                context_tokens,
                audio_features,
                kv_cache=None,
            )
            mx.eval(main_kv)

        # Store position of last token with valid KV cache
        # This is len(all_tokens) - 1 initially (all context except last)
        main_kv_valid_pos = len(all_tokens) - 1 if main_kv else 0

        draft_kv = None

        # Decode loop
        while len(all_tokens) - sample_begin < max_tokens:
            self.iterations += 1

            # Generate draft tokens (using draft model's audio features)
            draft_tokens_list, draft_kv = self._generate_draft(
                mx.array([all_tokens]),
                draft_audio_features,
                tokenizer,
                logit_filters,
                draft_kv,
            )

            # Check for EOT in draft
            if tokenizer.eot in draft_tokens_list:
                eot_idx = draft_tokens_list.index(tokenizer.eot)
                draft_tokens_list = draft_tokens_list[:eot_idx + 1]

            if not draft_tokens_list:
                break

            # Verify draft tokens with main model using incremental KV cache
            accepted, next_token, main_kv, main_kv_valid_pos = self._verify_draft_with_cache(
                all_tokens,
                draft_tokens_list,
                audio_features,
                tokenizer,
                logit_filters,
                main_kv,
                main_kv_valid_pos,
                sample_begin,
            )

            # Update statistics
            self.total_tokens += len(draft_tokens_list)
            self.accepted_tokens += accepted

            # Add accepted tokens
            all_tokens.extend(draft_tokens_list[:accepted])

            # Handle rejection/continuation
            if accepted < len(draft_tokens_list):
                # Draft was rejected at position 'accepted'
                # Add the resampled token from main model
                all_tokens.append(next_token)

                # Reset draft KV cache (it's now invalid)
                draft_kv = None

                # Main KV cache is valid up to the rejected position
                # main_kv_valid_pos was updated in _verify_draft_with_cache
            else:
                # All tokens accepted - add the bonus token from main model
                all_tokens.append(next_token)

            # Check for EOT
            if all_tokens[-1] == tokenizer.eot:
                break

        # Remove trailing EOT for text processing
        output_tokens = all_tokens[sample_begin:]

        # Strip all trailing EOT tokens (there may be multiple)
        while output_tokens and output_tokens[-1] == tokenizer.eot:
            output_tokens = output_tokens[:-1]

        # Parse segments
        segments = self._parse_segments(output_tokens, tokenizer, precision)

        return output_tokens, segments

    def _generate_draft(
        self,
        context: mx.array,
        audio_features: mx.array,
        tokenizer,
        logit_filters: list,
        kv_cache: list | None,
    ) -> tuple[list[int], list]:
        """
        Generate draft tokens using the draft model.

        Args:
            context: Current token context, shape (1, seq_len)
            audio_features: Encoded audio
            tokenizer: Whisper tokenizer
            logit_filters: Logit filters to apply
            kv_cache: Draft model KV cache

        Returns:
            Tuple of (draft_token_list, updated_kv_cache)
        """
        draft_tokens = []
        tokens = context
        all_tokens_list = context[0].tolist()

        for _ in range(self.draft_tokens):
            # Get logits from draft model
            logits, kv_cache, _, _ = self.draft_model.decoder(
                tokens[:, -1:] if kv_cache else tokens,
                audio_features,
                kv_cache=kv_cache,
            )
            logits = logits[:, -1].astype(mx.float32)

            # Apply logit filters
            logits = apply_filters(logits, mx.array([all_tokens_list]), logit_filters)

            # Greedy selection (temperature=0)
            next_token = int(mx.argmax(logits, axis=-1)[0])

            draft_tokens.append(next_token)
            all_tokens_list.append(next_token)

            # Update for next iteration
            tokens = mx.array([[next_token]])

            # Stop on EOT
            if next_token == tokenizer.eot:
                break

        return draft_tokens, kv_cache

    def _verify_draft_with_cache(
        self,
        all_tokens: list[int],
        draft_tokens_list: list[int],
        audio_features: mx.array,
        tokenizer,
        logit_filters: list,
        main_kv: list | None,
        main_kv_valid_pos: int,
        sample_begin: int,
    ) -> tuple[int, int, list, int]:
        """
        Verify draft tokens with main model using incremental KV cache.

        KV Cache Optimization (Worker #1189):
        Instead of processing the full sequence each time, we:
        1. Use existing KV cache for already-verified tokens
        2. Process only [last_cached_token, draft_tokens] in the forward pass
        3. Get logits for all draft tokens in one pass
        4. On acceptance: KV cache grows, return updated position
        5. On rejection: trim KV cache to valid position

        Args:
            all_tokens: Current full token sequence (list)
            draft_tokens_list: Draft tokens to verify (list)
            audio_features: Encoded audio
            tokenizer: Whisper tokenizer
            logit_filters: Logit filters to apply
            main_kv: Main model KV cache (valid up to main_kv_valid_pos)
            main_kv_valid_pos: Number of tokens with valid KV cache entries
            sample_begin: Index where sampling begins (after SOT sequence)

        Returns:
            Tuple of:
            - Number of accepted tokens
            - Next token to use (either accepted or resampled)
            - Updated KV cache
            - New valid position for KV cache
        """
        k = len(draft_tokens_list)
        context_len = len(all_tokens)

        # Build the tokens to process:
        # - If we have KV cache, process [last_token_with_cache_+1, ..., last_context_token, draft_tokens]
        # - This gives us logits that predict all draft tokens
        if main_kv is not None and main_kv_valid_pos > 0:
            # Process tokens from main_kv_valid_pos onwards
            # tokens_to_process = all_tokens[main_kv_valid_pos:] + draft_tokens_list
            uncached_context = all_tokens[main_kv_valid_pos:]
            tokens_to_process = uncached_context + draft_tokens_list
            tokens_tensor = mx.array([tokens_to_process])

            # Forward pass with KV cache
            logits, new_kv, _, _ = self.main_model.decoder(
                tokens_tensor,
                audio_features,
                kv_cache=main_kv,
            )
            logits = logits.astype(mx.float32)

            # logits[i] predicts token at position (main_kv_valid_pos + i + 1)
            # We want to predict draft_tokens_list[j] at position (context_len + j)
            # So we need logits at position (context_len + j - 1)
            # Which is logits index: (context_len + j - 1) - main_kv_valid_pos = (context_len - main_kv_valid_pos - 1) + j

            # The offset in logits where we find predictions for draft tokens
            # uncached_context has (context_len - main_kv_valid_pos) tokens
            # logits[uncached_context_len - 1] predicts first draft token
            uncached_len = len(uncached_context)
            draft_logits_offset = uncached_len - 1  # Index in logits that predicts draft_tokens[0]

        else:
            # No valid KV cache, process full sequence
            verify_tokens = mx.array([all_tokens + draft_tokens_list])
            logits, new_kv, _, _ = self.main_model.decoder(
                verify_tokens,
                audio_features,
                kv_cache=None,
            )
            logits = logits.astype(mx.float32)

            # logits[context_len - 1] predicts first draft token
            draft_logits_offset = context_len - 1

        # Verify each draft token
        accepted = 0
        next_token = draft_tokens_list[0]

        for i in range(k):
            logit_idx = draft_logits_offset + i

            if logit_idx >= logits.shape[1]:
                # Shouldn't happen, but be safe
                break

            pos_logits = logits[:, logit_idx]

            # Build token context for filters (all_tokens + draft[:i])
            filter_context = all_tokens + draft_tokens_list[:i]
            pos_logits = apply_filters(
                pos_logits,
                mx.array([filter_context]),
                logit_filters,
            )

            # Get main model's prediction
            main_pred = int(mx.argmax(pos_logits, axis=-1)[0])

            if main_pred == draft_tokens_list[i]:
                # Accept this token
                accepted += 1
                if i == k - 1:
                    # All tokens accepted, get bonus token
                    bonus_idx = draft_logits_offset + k
                    if bonus_idx < logits.shape[1]:
                        bonus_logits = logits[:, bonus_idx]
                        bonus_filter_context = all_tokens + draft_tokens_list
                        bonus_logits = apply_filters(
                            bonus_logits,
                            mx.array([bonus_filter_context]),
                            logit_filters,
                        )
                        next_token = int(mx.argmax(bonus_logits, axis=-1)[0])
                    else:
                        next_token = main_pred
            else:
                # Reject - use main model's token
                next_token = main_pred
                break

        # Update KV cache position
        # After verification, we accept 'accepted' draft tokens + the 'next_token'
        # So new position = context_len + accepted
        # But we need to handle cache trimming on rejection
        if accepted < k:
            # Some tokens rejected - we need to trim the KV cache
            # The KV cache from new_kv is valid for all processed tokens
            # We need to slice it to only keep entries up to (context_len + accepted)
            new_valid_pos = context_len + accepted
            # Slice the KV cache to remove entries for rejected tokens
            new_kv = self._trim_kv_cache(new_kv, new_valid_pos)
        else:
            # All accepted - KV cache is valid for context + all draft tokens
            new_valid_pos = context_len + k

        # The next_token will be added, so valid_pos should account for it
        # But we return the position BEFORE adding next_token
        # The caller adds next_token to all_tokens, then next iteration uses this cache

        return accepted, next_token, new_kv, new_valid_pos

    def _trim_kv_cache(
        self,
        kv_cache: list,
        valid_pos: int,
    ) -> list:
        """
        Trim KV cache to only include entries up to valid_pos.

        This is needed when draft tokens are rejected - we need to remove
        the KV cache entries for rejected tokens.

        Args:
            kv_cache: List of (self_kv, cross_kv) per layer
            valid_pos: Number of valid tokens (slice to this position)

        Returns:
            Trimmed KV cache
        """
        if kv_cache is None:
            return None

        trimmed = []
        for layer_cache in kv_cache:
            if layer_cache is None:
                trimmed.append(None)
                continue

            self_kv, cross_kv = layer_cache
            self_k, self_v = self_kv

            # Trim self-attention cache
            trimmed_self_k = self_k[:, :valid_pos, :]
            trimmed_self_v = self_v[:, :valid_pos, :]

            # Cross-attention cache doesn't need trimming (always same size)
            trimmed.append(((trimmed_self_k, trimmed_self_v), cross_kv))

        return trimmed

    def _verify_draft(
        self,
        context: mx.array,
        draft_tokens: mx.array,
        audio_features: mx.array,
        tokenizer,
        logit_filters: list,
        kv_cache: list | None,
        sample_begin: int,
    ) -> tuple[int, int, list]:
        """
        Verify draft tokens with main model (LEGACY - no KV cache optimization).

        Kept for reference. Use _verify_draft_with_cache instead.

        Processes all draft tokens in a single forward pass, then
        checks which tokens the main model would have generated.

        Args:
            context: Current token context, shape (1, seq_len)
            draft_tokens: Draft tokens to verify, shape (1, K)
            audio_features: Encoded audio
            tokenizer: Whisper tokenizer
            logit_filters: Logit filters to apply
            kv_cache: Main model KV cache
            sample_begin: Index where sampling begins (after SOT sequence)

        Returns:
            Tuple of:
            - Number of accepted tokens
            - Next token to use (either accepted or resampled)
            - Updated KV cache
        """
        context_list = context[0].tolist()
        draft_list = draft_tokens[0].tolist()
        k = len(draft_list)

        # Build sequence with draft tokens
        verify_tokens = mx.concatenate([context, draft_tokens], axis=1)

        # Get logits for all positions in one pass
        # Always use full sequence for correct position embeddings
        logits, kv_cache, _, _ = self.main_model.decoder(
            verify_tokens,
            audio_features,
            kv_cache=None,  # Don't use KV cache for full sequence verification
        )
        logits = logits.astype(mx.float32)

        # Verify each draft token
        # logits[0, j] predicts the token at position j+1
        # verify_tokens = [context_list + draft_list]
        # draft_list[i] is at position len(context_list) + i
        # To predict draft_list[i], we need logits at position len(context_list) + i - 1
        context_len = len(context_list)
        accepted = 0
        next_token = draft_list[0]  # Default: first draft token

        for i in range(k):
            # Position in logits that predicts draft_list[i]
            logit_pos = context_len - 1 + i
            pos_logits = logits[:, logit_pos:logit_pos+1]

            # Build token context for filters (context + draft[:i])
            filter_context = context_list + draft_list[:i]
            pos_logits = apply_filters(
                pos_logits[:, 0],
                mx.array([filter_context]),
                logit_filters,
            )

            # Get main model's prediction
            main_pred = int(mx.argmax(pos_logits, axis=-1)[0])

            if main_pred == draft_list[i]:
                # Accept this token
                accepted += 1
                if i == k - 1:
                    # All tokens accepted, generate one more from main
                    # Need to get prediction for next position
                    next_logit_pos = context_len - 1 + k
                    if next_logit_pos < logits.shape[1]:
                        next_logits = logits[:, next_logit_pos]
                        next_filter_context = context_list + draft_list
                        next_logits = apply_filters(next_logits, mx.array([next_filter_context]), logit_filters)
                        next_token = int(mx.argmax(next_logits, axis=-1)[0])
                    else:
                        next_token = main_pred
            else:
                # Reject - use main model's token
                next_token = main_pred
                break

        return accepted, next_token, None  # Always return None for kv_cache since we didn't use it

    def _parse_segments(
        self,
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


class CTCSpeculativeDecoder(SpeculativeDecoder):
    """
    Speculative decoding using CTC draft instead of distil-whisper.

    Key advantage: CTC generates ALL draft tokens in ONE forward pass through
    a lightweight linear layer, instead of K sequential decoder forward passes.

    Architecture:
        Audio -> Encoder -> CTC Head -> Draft tokens (1 pass)
                       |
                       +-> Main Decoder <- Verify draft (parallel)

    This reuses the verification logic from SpeculativeDecoder, only replacing
    the draft generation step with CTC.

    Expected speedup vs distil-whisper draft:
    - Draft generation: ~K times faster (1 vs K forward passes)
    - Verification: Same (dominated by main decoder forward pass)
    - Overall: Depends on acceptance rate

    Target metrics:
    - CTC acceptance rate: >60%
    - Overall speedup: >4x (vs 3.31x with distil-whisper)
    """

    def __init__(
        self,
        main_model: nn.Module,
        ctc_head: "CTCDraftHead",
        draft_tokens: int = 20,  # CTC can generate more tokens cheaply
    ):
        """
        Args:
            main_model: Main whisper model (target)
            ctc_head: Trained CTC draft head
            draft_tokens: Number of tokens to draft per iteration
        """
        # Initialize parent without draft_model
        self.main_model = main_model
        self.draft_model = None  # Not used for CTC
        self.ctc_head = ctc_head
        self.draft_tokens = draft_tokens

        # Statistics tracking
        self.total_tokens = 0
        self.accepted_tokens = 0
        self.iterations = 0
        self.ctc_draft_time = 0.0  # Track CTC draft time
        self.verify_time = 0.0  # Track verification time

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
        draft_audio_features: mx.array | None = None,
    ) -> tuple[list[int], list[dict]]:
        """
        Decode with CTC draft + main model verification.

        The CTC head generates draft tokens from encoder hidden states in a
        single forward pass. The main decoder then verifies these tokens.

        Args:
            audio_features: Encoded audio from main model encoder
            tokenizer: Whisper tokenizer
            options: Decoding options
            sample_begin: Index where sampling starts
            n_vocab: Vocabulary size
            precision: Timestamp precision
            audio_duration: Actual audio duration
            max_tokens: Maximum tokens to decode
            draft_audio_features: Ignored (CTC uses main encoder output)

        Returns:
            Tuple of (token_list, segments_list)
        """
        import time

        self.reset_stats()
        self.ctc_draft_time = 0.0
        self.verify_time = 0.0

        # Generate CTC draft tokens ONCE at the start
        # CTC predicts the entire sequence in one shot
        ctc_start = time.perf_counter()
        ctc_logits = self.ctc_head(audio_features)
        all_ctc_tokens = self.ctc_head.decode_greedy(ctc_logits)
        mx.eval(ctc_logits)  # Ensure computation is done
        self.ctc_draft_time = time.perf_counter() - ctc_start

        # Build initial tokens
        initial_tokens = list(tokenizer.sot_sequence)
        all_tokens = list(initial_tokens)

        # Build logit filters
        logit_filters = build_logit_filters(
            tokenizer,
            options,
            sample_begin=sample_begin,
            n_vocab=n_vocab,
            precision=precision,
            audio_duration=audio_duration,
        )

        # Initialize main model KV cache
        main_kv = None
        if len(all_tokens) > 1:
            context_tokens = mx.array([all_tokens[:-1]])
            _, main_kv, _, _ = self.main_model.decoder(
                context_tokens,
                audio_features,
                kv_cache=None,
            )
            mx.eval(main_kv)

        main_kv_valid_pos = len(all_tokens) - 1 if main_kv else 0

        # Track CTC token consumption
        ctc_idx = 0

        # Decode loop - consume CTC tokens in chunks
        while len(all_tokens) - sample_begin < max_tokens:
            self.iterations += 1

            # Get next chunk of CTC draft tokens
            remaining_ctc = all_ctc_tokens[ctc_idx:]
            if not remaining_ctc:
                # CTC exhausted - fall back to greedy from main model
                break

            # Take up to draft_tokens from CTC output
            draft_tokens_list = remaining_ctc[:self.draft_tokens]

            # Filter out special tokens that CTC shouldn't predict
            # (timestamps, language tokens, etc.)
            draft_tokens_list = self._filter_ctc_tokens(
                draft_tokens_list, tokenizer,
            )

            if not draft_tokens_list:
                ctc_idx += self.draft_tokens
                continue

            # Check for EOT in draft
            if tokenizer.eot in draft_tokens_list:
                eot_idx = draft_tokens_list.index(tokenizer.eot)
                draft_tokens_list = draft_tokens_list[:eot_idx + 1]

            # Verify draft tokens with main model
            verify_start = time.perf_counter()
            accepted, next_token, main_kv, main_kv_valid_pos = self._verify_draft_with_cache(
                all_tokens,
                draft_tokens_list,
                audio_features,
                tokenizer,
                logit_filters,
                main_kv,
                main_kv_valid_pos,
                sample_begin,
            )
            self.verify_time += time.perf_counter() - verify_start

            # Update statistics
            self.total_tokens += len(draft_tokens_list)
            self.accepted_tokens += accepted

            # Add accepted tokens
            all_tokens.extend(draft_tokens_list[:accepted])

            # Update CTC index based on accepted tokens
            ctc_idx += accepted

            # Handle rejection/continuation
            if accepted < len(draft_tokens_list):
                # Draft rejected - use main model's token
                all_tokens.append(next_token)
                # Don't advance CTC index - we'll retry from here
                # Actually, the main model gave us a different token,
                # so we should skip the rejected CTC tokens
                ctc_idx += len(draft_tokens_list)  # Skip all proposed tokens
            else:
                # All accepted - add bonus token from main model
                all_tokens.append(next_token)
                ctc_idx += 1  # Account for bonus token position

            # Check for EOT
            if all_tokens[-1] == tokenizer.eot:
                break

        # Remove trailing EOT for text processing
        output_tokens = all_tokens[sample_begin:]
        while output_tokens and output_tokens[-1] == tokenizer.eot:
            output_tokens = output_tokens[:-1]

        # Parse segments
        segments = self._parse_segments(output_tokens, tokenizer, precision)

        return output_tokens, segments

    def _filter_ctc_tokens(
        self,
        tokens: list[int],
        tokenizer,
    ) -> list[int]:
        """
        Filter out tokens that CTC shouldn't predict.

        CTC is trained on text transcripts, so it may output:
        - Regular text tokens (valid)
        - Special tokens like timestamps (invalid for draft)
        - Language/task tokens (invalid for draft)

        Args:
            tokens: Raw CTC token predictions
            tokenizer: Whisper tokenizer

        Returns:
            Filtered token list with only valid text tokens
        """
        # Whisper special token ranges
        # 50257: <|endoftext|>
        # 50258-50363: special tokens (lang, task, etc.)
        # 50364+: timestamps
        timestamp_begin = tokenizer.timestamp_begin  # 50364

        filtered = []
        for token in tokens:
            # Skip timestamps
            if token >= timestamp_begin:
                continue
            # Skip special tokens (but allow EOT)
            if token >= 50258 and token != tokenizer.eot:
                continue
            # Skip padding/blank (CTC blank is usually 0)
            if token == 0:
                continue
            filtered.append(token)

        return filtered

    @property
    def ctc_acceptance_rate(self) -> float:
        """Return acceptance rate specifically for CTC draft tokens."""
        return self.acceptance_rate

    @property
    def timing_breakdown(self) -> dict[str, float]:
        """Return timing breakdown for CTC speculative decoding."""
        total = self.ctc_draft_time + self.verify_time
        return {
            "ctc_draft_ms": self.ctc_draft_time * 1000,
            "verify_ms": self.verify_time * 1000,
            "total_ms": total * 1000,
            "draft_fraction": self.ctc_draft_time / total if total > 0 else 0,
        }
