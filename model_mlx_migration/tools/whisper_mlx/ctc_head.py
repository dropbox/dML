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
CTC Draft Head for Speculative Decoding.

This module implements a CTC (Connectionist Temporal Classification) head
that generates draft tokens from Whisper encoder hidden states. The draft
tokens are then verified by the main decoder using speculative decoding.

Key insight: The encoder has bidirectional attention over the entire audio,
so it "knows" about future content that the causal decoder cannot see.
This addresses Medusa's fundamental limitation.

Architecture:
    Audio -> Encoder -> CTC Head -> Draft tokens (1 forward pass)
                    |
                    +-> Main Decoder <- Verify draft

References:
- Graves, A. et al. "Connectionist Temporal Classification" (2006)
- Leviathan, Y. et al. "Fast Inference from Transformers via Speculative Decoding" (2022)
"""


import mlx.core as mx
import mlx.nn as nn


class CTCDraftHead(nn.Module):
    """
    CTC head for generating draft tokens from encoder hidden states.

    The encoder outputs have shape (batch, T, d_model) where:
    - T = 1500 for 30s audio (50Hz frame rate after conv downsampling)
    - d_model = 1280 for whisper-large-v3

    CTC predicts one token per encoder frame, with blank tokens
    for alignment. We collapse blanks and repeats to get draft tokens.

    Attributes:
        d_model: Hidden dimension (1280 for large-v3)
        vocab_size: Vocabulary size (51865 for Whisper)
        blank_id: CTC blank token ID (default: 0)
    """

    # Whisper vocab size (without padding)
    WHISPER_VOCAB_SIZE = 51865

    # Default blank token (use padding token)
    DEFAULT_BLANK_ID = 0

    def __init__(
        self,
        d_model: int = 1280,
        vocab_size: int = 51865,
        blank_id: int = 0,
        use_layer_norm: bool = True,
    ):
        """
        Initialize CTC draft head.

        Args:
            d_model: Encoder hidden dimension
            vocab_size: Output vocabulary size
            blank_id: CTC blank token ID
            use_layer_norm: Apply layer norm before projection
        """
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.blank_id = blank_id
        self._use_layer_norm = use_layer_norm

        # Optional layer norm for stability
        if use_layer_norm:
            self.ln = nn.LayerNorm(d_model)

        # Simple projection: encoder_hidden -> vocab logits
        # This is the only trainable component (~65M parameters)
        self.proj = nn.Linear(d_model, vocab_size)

        # Statistics tracking
        self._total_frames = 0
        self._total_tokens = 0

    def __call__(self, encoder_output: mx.array) -> mx.array:
        """
        Forward pass: encoder hidden states -> vocabulary logits.

        Args:
            encoder_output: (batch, T, d_model) encoder hidden states

        Returns:
            logits: (batch, T, vocab_size) per-frame vocabulary logits
        """
        x = encoder_output

        # Optional layer norm
        if self._use_layer_norm:
            x = self.ln(x)

        # Project to vocabulary
        return self.proj(x)


    def decode_greedy(
        self,
        logits: mx.array,
        max_tokens: int | None = None,
    ) -> list[int]:
        """
        CTC greedy decoding with blank/repeat collapse.

        Args:
            logits: (batch, T, vocab_size) or (T, vocab_size)
            max_tokens: Maximum tokens to return (for speculative decoding)

        Returns:
            Collapsed token sequence (no blanks, no consecutive repeats)
        """
        # Handle both batched and unbatched input
        if logits.ndim == 3:
            logits = logits[0]  # Take first batch element

        # Greedy selection
        predictions = mx.argmax(logits, axis=-1)  # (T,)
        mx.eval(predictions)

        # Collapse blanks and consecutive repeats
        tokens = []
        prev_token = self.blank_id

        predictions_list = predictions.tolist()
        for token in predictions_list:
            if token != self.blank_id and token != prev_token:
                tokens.append(token)
                if max_tokens and len(tokens) >= max_tokens:
                    break
            prev_token = token

        # Update statistics
        self._total_frames += len(predictions_list)
        self._total_tokens += len(tokens)

        return tokens

    def decode_greedy_with_timestamps(
        self,
        logits: mx.array,
        frame_rate: float = 50.0,
        max_tokens: int | None = None,
    ) -> list[tuple[int, float]]:
        """
        CTC greedy decoding with approximate timestamps.

        Args:
            logits: (batch, T, vocab_size) or (T, vocab_size)
            frame_rate: Encoder frames per second (50 for Whisper)
            max_tokens: Maximum tokens to return

        Returns:
            List of (token, timestamp) tuples
        """
        if logits.ndim == 3:
            logits = logits[0]

        predictions = mx.argmax(logits, axis=-1)
        mx.eval(predictions)

        tokens_with_times = []
        prev_token = self.blank_id

        predictions_list = predictions.tolist()
        for frame_idx, token in enumerate(predictions_list):
            if token != self.blank_id and token != prev_token:
                timestamp = frame_idx / frame_rate
                tokens_with_times.append((token, timestamp))
                if max_tokens and len(tokens_with_times) >= max_tokens:
                    break
            prev_token = token

        return tokens_with_times

    @property
    def compression_ratio(self) -> float:
        """
        Return average compression ratio (frames per token).

        Higher ratio means more efficient compression.
        Typical: 15-30 frames per token for speech.
        """
        if self._total_tokens == 0:
            return 0.0
        return self._total_frames / self._total_tokens

    def reset_stats(self):
        """Reset decoding statistics."""
        self._total_frames = 0
        self._total_tokens = 0

    def save_weights(self, path: str):
        """
        Save CTC head weights to file.

        Args:
            path: Output path (.safetensors recommended)
        """
        # Flatten parameters to dict with string keys
        weights = dict(self.parameters())

        # Always use safetensors for reliability
        if not path.endswith(".safetensors"):
            path = path.rsplit(".", 1)[0] + ".safetensors"

        mx.save_safetensors(path, weights)

    @classmethod
    def load_weights(cls, path: str, d_model: int = 1280) -> "CTCDraftHead":
        """
        Load CTC head weights from file.

        Supports both:
        - Flat key format from training checkpoints (.npz): 'ln.weight', 'proj.bias'
        - Nested dict format from safetensors: {'ln': {'weight': ...}}

        Args:
            path: Path to weights file (.safetensors or .npz)
            d_model: Encoder hidden dimension

        Returns:
            CTCDraftHead with loaded weights
        """
        # Create model
        model = cls(d_model=d_model)

        # Load weights
        weights = mx.load(path)

        # Check if weights need unflattening (flat keys have dots)
        needs_unflatten = any("." in k for k in weights.keys())

        if needs_unflatten:
            # Unflatten: 'ln.weight' -> {'ln': {'weight': ...}}
            nested = {}
            for key, value in weights.items():
                parts = key.split(".")
                current = nested
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
            weights = nested

        # Update model parameters
        model.update(weights)

        return model


class CTCLoss(nn.Module):
    """
    CTC Loss implementation for MLX.

    Computes the CTC loss between encoder frame predictions and
    target token sequences, marginalizing over all valid alignments.

    Note: This is a simplified implementation. For production training,
    consider using a more optimized CTC implementation.
    """

    def __init__(self, blank_id: int = 0, reduction: str = "mean"):
        """
        Initialize CTC loss.

        Args:
            blank_id: CTC blank token ID
            reduction: Loss reduction method ("mean", "sum", or "none")
        """
        super().__init__()
        self.blank_id = blank_id
        self.reduction = reduction

    def __call__(
        self,
        logits: mx.array,
        targets: mx.array,
        input_lengths: mx.array,
        target_lengths: mx.array,
    ) -> mx.array:
        """
        Compute CTC loss.

        Args:
            logits: (batch, T, vocab_size) log probabilities
            targets: (batch, S) target token sequences
            input_lengths: (batch,) actual input lengths
            target_lengths: (batch,) actual target lengths

        Returns:
            CTC loss value
        """
        # Log softmax
        log_probs = mx.log(mx.softmax(logits, axis=-1) + 1e-10)

        batch_size = logits.shape[0]
        losses = []

        for b in range(batch_size):
            T = int(input_lengths[b])
            S = int(target_lengths[b])

            if S == 0:
                losses.append(mx.array(0.0))
                continue

            log_prob = log_probs[b, :T]  # (T, vocab_size)
            target = targets[b, :S]  # (S,)

            # Forward-backward algorithm for CTC
            loss = self._ctc_loss_single(log_prob, target)
            losses.append(loss)

        losses = mx.stack(losses)

        if self.reduction == "mean":
            return mx.mean(losses)
        if self.reduction == "sum":
            return mx.sum(losses)
        return losses

    def _ctc_loss_single(
        self,
        log_probs: mx.array,
        targets: mx.array,
    ) -> mx.array:
        """
        CTC loss for a single sequence using forward algorithm.

        Args:
            log_probs: (T, vocab_size) log probabilities
            targets: (S,) target tokens

        Returns:
            Negative log likelihood
        """
        T = log_probs.shape[0]
        S = targets.shape[0]

        # Extended label sequence with blanks: [blank, t0, blank, t1, blank, ...]
        # Length: 2*S + 1
        L = 2 * S + 1

        # Build extended labels
        extended_labels = mx.zeros((L,), dtype=mx.int32)
        for s in range(S):
            extended_labels = extended_labels.at[2 * s + 1].add(targets[s])
        # Blanks are already 0

        # Forward algorithm
        # alpha[t, l] = sum of probabilities of all paths ending at position l at time t
        alpha = mx.full((T, L), float("-inf"))

        # Initialize: can start with blank or first label
        alpha = alpha.at[0, 0].add(log_probs[0, self.blank_id])
        if L > 1:
            first_label = int(extended_labels[1])
            alpha = alpha.at[0, 1].add(log_probs[0, first_label])

        # Forward pass
        for t in range(1, T):
            for idx in range(L):
                label = int(extended_labels[idx])

                # Stay in same state
                score = alpha[t - 1, idx]

                # Transition from previous state
                if idx > 0:
                    score = mx.logaddexp(score, alpha[t - 1, idx - 1])

                # Skip blank if labels are different
                if idx > 1 and label != self.blank_id:
                    prev_label = int(extended_labels[idx - 2])
                    if label != prev_label:
                        score = mx.logaddexp(score, alpha[t - 1, idx - 2])

                alpha = alpha.at[t, idx].add(score + log_probs[t, label])

        # Total probability: sum of ending at last blank or last label
        log_prob = mx.logaddexp(alpha[T - 1, L - 1], alpha[T - 1, L - 2])

        return -log_prob


def create_ctc_draft_head(
    model_size: str = "large-v3",
    use_layer_norm: bool = True,
) -> CTCDraftHead:
    """
    Factory function to create CTC draft head for a Whisper model.

    Args:
        model_size: Whisper model size ("large-v3", "medium", "small", etc.)
        use_layer_norm: Apply layer norm before projection

    Returns:
        CTCDraftHead configured for the specified model
    """
    # Model dimensions
    d_model_map = {
        "tiny": 384,
        "base": 512,
        "small": 768,
        "medium": 1024,
        "large": 1280,
        "large-v2": 1280,
        "large-v3": 1280,
        "turbo": 1280,
    }

    d_model = d_model_map.get(model_size, 1280)

    return CTCDraftHead(
        d_model=d_model,
        vocab_size=CTCDraftHead.WHISPER_VOCAB_SIZE,
        blank_id=CTCDraftHead.DEFAULT_BLANK_ID,
        use_layer_norm=use_layer_norm,
    )
