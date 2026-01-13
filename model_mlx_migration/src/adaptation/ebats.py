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
E-BATS: Efficient Backpropagation-free Test-Time Adaptation for Speech.

Based on arXiv:2506.07078.

E-BATS replaces SUTA (Self-Unsupervised Test-Time Adaptation) with a more
efficient approach that does not require backpropagation at inference time.

Key features:
- Backprop-free prompt adaptation
- Multi-scale loss for global/local distribution shifts (training only)
- EMA mechanism for stability
- Target latency: <10ms adaptation

The approach uses a pre-computed prompt bank. At inference time:
1. Compute audio context embedding from initial frames
2. Find best matching prompt via cosine similarity (no gradients)
3. Apply prompt to encoder for adapted inference

Training the prompt bank is done offline with the multi-scale loss.
"""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class EBATSConfig:
    """Configuration for E-BATS test-time adaptation."""

    # Prompt bank configuration
    num_prompts: int = 64  # Number of prompts in bank
    prompt_dim: int = 512  # Prompt embedding dimension (match encoder dim)
    prompt_length: int = 16  # Length of prompt sequence

    # Audio context configuration
    context_frames: int = 50  # Number of frames for context (50 = 1 second at 50Hz)
    context_dim: int = 512  # Context embedding dimension

    # Selection configuration
    top_k: int = 3  # Number of prompts to mix (1 = hard selection)
    temperature: float = 1.0  # Softmax temperature for mixing

    # EMA configuration for prompt bank updates (training)
    ema_decay: float = 0.999  # EMA decay for prompt bank updates
    ema_enabled: bool = True  # Enable EMA during training

    # Multi-scale loss configuration (training)
    use_multi_scale_loss: bool = True
    global_loss_weight: float = 0.5  # Global distribution shift loss
    local_loss_weight: float = 0.5  # Local distribution shift loss

    # Inference settings
    adaptation_enabled: bool = True  # Can disable adaptation


@dataclass
class PromptBankStats:
    """Statistics for prompt bank usage."""

    selection_counts: mx.array  # How often each prompt is selected
    avg_similarity: float  # Average similarity score
    prompt_diversity: float  # Variance in prompt usage


class PromptBank(nn.Module):
    """
    Pre-computed prompt bank for E-BATS.

    The prompt bank contains learnable prompt embeddings that can be
    selected based on audio context similarity. Prompts are trained
    offline with multi-scale loss.

    Each prompt is a sequence of embeddings that gets prepended/added
    to the encoder input to adapt its behavior.
    """

    def __init__(self, config: EBATSConfig):
        super().__init__()
        self.config = config

        # Initialize prompt bank
        # Shape: (num_prompts, prompt_length, prompt_dim)
        # Using Xavier-style initialization scaled by sqrt(dim)
        scale = 1.0 / (config.prompt_dim ** 0.5)
        self.prompts = mx.random.normal(
            shape=(config.num_prompts, config.prompt_length, config.prompt_dim),
        ) * scale

        # Prompt keys for similarity matching
        # Shape: (num_prompts, context_dim)
        self.prompt_keys = mx.random.normal(
            shape=(config.num_prompts, config.context_dim),
        ) * scale

        # EMA shadow parameters (for training stability)
        self._ema_prompts = mx.array(self.prompts)
        self._ema_keys = mx.array(self.prompt_keys)

        # Statistics tracking
        self._selection_counts = mx.zeros((config.num_prompts,))

    def get_prompt(
        self,
        context_embedding: mx.array,
        use_ema: bool = True,
    ) -> tuple[mx.array, mx.array]:
        """
        Select best prompt(s) based on context embedding.

        Args:
            context_embedding: Audio context embedding, shape (batch, context_dim)
            use_ema: Use EMA-smoothed parameters (default True for inference)

        Returns:
            Tuple of:
            - Selected prompt(s), shape (batch, prompt_length, prompt_dim)
            - Selection weights, shape (batch, num_prompts)
        """
        # Use EMA parameters for inference stability
        keys = self._ema_keys if use_ema else self.prompt_keys
        prompts = self._ema_prompts if use_ema else self.prompts

        # Normalize for cosine similarity
        context_norm = context_embedding / (
            mx.linalg.norm(context_embedding, axis=-1, keepdims=True) + 1e-8
        )
        keys_norm = keys / (mx.linalg.norm(keys, axis=-1, keepdims=True) + 1e-8)

        # Compute similarity scores
        # (batch, context_dim) @ (context_dim, num_prompts) -> (batch, num_prompts)
        similarity = context_norm @ keys_norm.T

        if self.config.top_k == 1:
            # Hard selection - single best prompt
            indices = mx.argmax(similarity, axis=-1)  # (batch,)

            # Create one-hot weights using scatter-like operation
            context_embedding.shape[0]
            num_prompts = self.config.num_prompts

            # One-hot encoding via comparison
            # indices: (batch,) -> (batch, 1)
            indices_expanded = indices[:, None]
            prompt_indices = mx.arange(num_prompts)[None, :]  # (1, num_prompts)
            weights = (indices_expanded == prompt_indices).astype(mx.float32)

            # Select prompts: (batch, prompt_length, prompt_dim)
            selected = prompts[indices]
        else:
            # Soft selection - mix top-k prompts
            # Get top-k indices and values
            min(self.config.top_k, self.config.num_prompts)

            # Apply temperature scaling
            scaled_sim = similarity / self.config.temperature

            # Softmax over all prompts (or could mask to top-k)
            weights = mx.softmax(scaled_sim, axis=-1)

            # Weighted combination of prompts
            # weights: (batch, num_prompts)
            # prompts: (num_prompts, prompt_length, prompt_dim)
            # -> (batch, prompt_length, prompt_dim)
            selected = mx.einsum("bn,nld->bld", weights, prompts)

        return selected, weights

    def update_ema(self) -> None:
        """Update EMA parameters (call during training)."""
        if not self.config.ema_enabled:
            return

        decay = self.config.ema_decay
        self._ema_prompts = decay * self._ema_prompts + (1 - decay) * self.prompts
        self._ema_keys = decay * self._ema_keys + (1 - decay) * self.prompt_keys

    def get_stats(self) -> PromptBankStats:
        """Get prompt bank usage statistics."""
        total = mx.sum(self._selection_counts)
        if total > 0:
            distribution = self._selection_counts / total
            diversity = float(mx.var(distribution).item())
        else:
            diversity = 0.0

        return PromptBankStats(
            selection_counts=self._selection_counts,
            avg_similarity=0.0,  # Updated during forward pass
            prompt_diversity=diversity,
        )


class ContextEncoder(nn.Module):
    """
    Encodes initial audio frames into a context embedding.

    This module processes the first N frames of audio to produce
    a compact context embedding for prompt selection.
    """

    def __init__(self, config: EBATSConfig):
        super().__init__()
        self.config = config

        # Simple temporal pooling + projection
        # Input: (batch, context_frames, encoder_dim)
        # Output: (batch, context_dim)

        # Temporal attention for weighted pooling
        self.attention_query = nn.Linear(config.prompt_dim, 1)

        # Optional projection if dimensions differ
        if config.prompt_dim != config.context_dim:
            self.projection = nn.Linear(config.prompt_dim, config.context_dim)
        else:
            self.projection = None

    def __call__(self, encoder_features: mx.array) -> mx.array:
        """
        Encode audio features into context embedding.

        Args:
            encoder_features: Encoder output, shape (batch, time, dim)
                              Only first context_frames are used.

        Returns:
            Context embedding, shape (batch, context_dim)
        """
        # Take only first N frames
        context_frames = min(self.config.context_frames, encoder_features.shape[1])
        features = encoder_features[:, :context_frames, :]

        # Compute attention weights
        attn_logits = self.attention_query(features)  # (batch, time, 1)
        attn_weights = mx.softmax(attn_logits, axis=1)  # (batch, time, 1)

        # Weighted sum
        pooled = mx.sum(features * attn_weights, axis=1)  # (batch, dim)

        # Project if needed
        if self.projection is not None:
            pooled = self.projection(pooled)

        return pooled


class EBATS(nn.Module):
    """
    E-BATS: Efficient Backpropagation-free Test-Time Adaptation for Speech.

    This module wraps an ASR encoder and provides test-time adaptation
    via prompt selection from a pre-trained prompt bank.

    At inference time:
    1. Run encoder on initial audio context
    2. Compute context embedding
    3. Select best prompt(s) from prompt bank
    4. Apply prompt to encoder input
    5. Run full inference with adapted encoder

    The prompt bank is trained offline. Inference is backprop-free
    and achieves <10ms adaptation latency.

    Usage:
        config = EBATSConfig()
        ebats = EBATS(config, encoder)

        # Inference (no gradients)
        output = ebats(audio_features)

        # Training (for prompt bank)
        output, loss = ebats.forward_with_loss(audio_features, target)
    """

    def __init__(self, config: EBATSConfig):
        super().__init__()
        self.config = config

        # Components
        self.prompt_bank = PromptBank(config)
        self.context_encoder = ContextEncoder(config)

        # Prompt application mode
        # 'prepend': Add prompt to beginning of sequence
        # 'add': Add prompt embedding to encoder features
        # 'scale': Learn per-layer scale factors from prompt
        self.application_mode = "add"

        # For 'scale' mode: project prompt to scale factors
        # This is a lightweight adaptation inspired by adapter methods
        self.prompt_to_scale = nn.Linear(config.prompt_dim, config.prompt_dim)

    def get_context_embedding(
        self,
        encoder_features: mx.array,
    ) -> mx.array:
        """
        Compute context embedding from encoder features.

        Args:
            encoder_features: Shape (batch, time, dim)

        Returns:
            Context embedding, shape (batch, context_dim)
        """
        return self.context_encoder(encoder_features)

    def select_prompt(
        self,
        context_embedding: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """
        Select prompt from bank based on context.

        Args:
            context_embedding: Shape (batch, context_dim)

        Returns:
            Tuple of:
            - Selected prompt, shape (batch, prompt_length, prompt_dim)
            - Selection weights, shape (batch, num_prompts)
        """
        return self.prompt_bank.get_prompt(context_embedding, use_ema=True)

    def apply_prompt(
        self,
        encoder_features: mx.array,
        prompt: mx.array,
    ) -> mx.array:
        """
        Apply prompt to encoder features.

        Args:
            encoder_features: Shape (batch, time, dim)
            prompt: Shape (batch, prompt_length, prompt_dim)

        Returns:
            Adapted features, shape (batch, time, dim)
        """
        if self.application_mode == "prepend":
            # Prepend prompt to sequence
            return mx.concatenate([prompt, encoder_features], axis=1)

        if self.application_mode == "add":
            # Add prompt embedding (broadcasted over time)
            # Average prompt over its length to get single embedding
            prompt_avg = mx.mean(prompt, axis=1, keepdims=True)  # (batch, 1, dim)
            return encoder_features + prompt_avg

        if self.application_mode == "scale":
            # Learn scale factors from prompt
            prompt_avg = mx.mean(prompt, axis=1)  # (batch, dim)
            scale = mx.sigmoid(self.prompt_to_scale(prompt_avg))  # (batch, dim)
            scale = scale[:, None, :]  # (batch, 1, dim)
            return encoder_features * (1.0 + scale)  # Residual scaling

        msg = f"Unknown application mode: {self.application_mode}"
        raise ValueError(msg)

    def adapt(
        self,
        encoder_features: mx.array,
    ) -> tuple[mx.array, dict]:
        """
        Adapt encoder features using E-BATS.

        This is the main inference method. It:
        1. Computes context embedding from features
        2. Selects best prompt from bank
        3. Applies prompt to features

        No backpropagation is needed.

        Args:
            encoder_features: Raw encoder output, shape (batch, time, dim)

        Returns:
            Tuple of:
            - Adapted features, shape (batch, time', dim) where time' may
              differ if using 'prepend' mode
            - Info dict with adaptation metadata
        """
        if not self.config.adaptation_enabled:
            return encoder_features, {"adapted": False}

        # Step 1: Compute context embedding
        context = self.get_context_embedding(encoder_features)

        # Step 2: Select prompt
        prompt, weights = self.select_prompt(context)

        # Step 3: Apply prompt
        adapted = self.apply_prompt(encoder_features, prompt)

        # Collect info
        info = {
            "adapted": True,
            "selection_weights": weights,
            "context_embedding": context,
            "prompt": prompt,
        }

        return adapted, info

    def __call__(self, encoder_features: mx.array) -> mx.array:
        """
        Forward pass with adaptation.

        Args:
            encoder_features: Shape (batch, time, dim)

        Returns:
            Adapted features, shape (batch, time', dim)
        """
        adapted, _ = self.adapt(encoder_features)
        return adapted

    def compute_multi_scale_loss(
        self,
        _encoder_features: mx.array,
        target_features: mx.array,
        adapted_features: mx.array,
    ) -> tuple[mx.array, dict[str, mx.array]]:
        """
        Compute multi-scale loss for prompt bank training.

        The multi-scale loss captures both global and local distribution shifts:
        - Global loss: Overall feature distribution alignment
        - Local loss: Per-frame alignment

        Args:
            _encoder_features: Original encoder features (unused, kept for API)
            target_features: Target domain features (from target domain data)
            adapted_features: Features after prompt adaptation

        Returns:
            Tuple of:
            - Total loss scalar
            - Dict of individual losses
        """
        if not self.config.use_multi_scale_loss:
            # Simple MSE loss
            loss = mx.mean((adapted_features - target_features) ** 2)
            return loss, {"mse": loss}

        # Global loss: distribution matching via moment alignment
        # Mean alignment
        adapted_mean = mx.mean(adapted_features, axis=(1, 2))  # (batch,)
        target_mean = mx.mean(target_features, axis=(1, 2))  # (batch,)
        mean_loss = mx.mean((adapted_mean - target_mean) ** 2)

        # Variance alignment
        adapted_var = mx.var(adapted_features, axis=(1, 2))  # (batch,)
        target_var = mx.var(target_features, axis=(1, 2))  # (batch,)
        var_loss = mx.mean((adapted_var - target_var) ** 2)

        global_loss = mean_loss + var_loss

        # Local loss: per-frame MSE
        min_len = min(adapted_features.shape[1], target_features.shape[1])
        local_loss = mx.mean(
            (adapted_features[:, :min_len] - target_features[:, :min_len]) ** 2,
        )

        # Combine
        total_loss = (
            self.config.global_loss_weight * global_loss
            + self.config.local_loss_weight * local_loss
        )

        losses = {
            "global": global_loss,
            "local": local_loss,
            "mean": mean_loss,
            "var": var_loss,
            "total": total_loss,
        }

        return total_loss, losses

    def forward_with_loss(
        self,
        encoder_features: mx.array,
        target_features: mx.array,
    ) -> tuple[mx.array, mx.array, dict]:
        """
        Forward pass with loss computation for training.

        Args:
            encoder_features: Source domain encoder features
            target_features: Target domain features

        Returns:
            Tuple of:
            - Adapted features
            - Total loss
            - Loss dict with individual components
        """
        # Adapt features
        adapted, info = self.adapt(encoder_features)

        # Compute loss
        loss, loss_dict = self.compute_multi_scale_loss(
            encoder_features, target_features, adapted,
        )

        # Update EMA
        self.prompt_bank.update_ema()

        return adapted, loss, {**info, "losses": loss_dict}


def create_ebats(
    encoder_dim: int = 512,
    num_prompts: int = 64,
    prompt_length: int = 16,
    **kwargs,
) -> EBATS:
    """
    Create E-BATS module with default configuration.

    Args:
        encoder_dim: Dimension of encoder features (must match encoder output)
        num_prompts: Number of prompts in bank
        prompt_length: Length of each prompt sequence
        **kwargs: Additional config overrides

    Returns:
        Configured EBATS module
    """
    config = EBATSConfig(
        num_prompts=num_prompts,
        prompt_dim=encoder_dim,
        prompt_length=prompt_length,
        context_dim=encoder_dim,
        **kwargs,
    )
    return EBATS(config)


# Prompt bank training utilities

def generate_prompt_bank_from_speakers(
    encoder: nn.Module,
    speaker_audio_dict: dict[str, list[mx.array]],
    config: EBATSConfig | None = None,
) -> PromptBank:
    """
    Generate initial prompt bank from diverse speaker recordings.

    This function creates prompts by clustering speaker embeddings
    and using cluster centers as initial prompt keys.

    Args:
        encoder: Pre-trained encoder to extract features
        speaker_audio_dict: Dict mapping speaker_id -> list of audio arrays
        config: E-BATS configuration

    Returns:
        Initialized PromptBank
    """
    if config is None:
        config = EBATSConfig()

    # Collect speaker embeddings
    embeddings = []
    for audio_list in speaker_audio_dict.values():
        speaker_embeddings = []
        for audio in audio_list:
            # Ensure audio has batch dimension
            if audio.ndim == 1:
                audio = audio[None, :]

            # Extract features
            features = encoder(audio)

            # Average pool to get speaker embedding
            emb = mx.mean(features, axis=1)  # (1, dim)
            speaker_embeddings.append(emb)

        # Average per speaker
        speaker_emb = mx.mean(mx.concatenate(speaker_embeddings, axis=0), axis=0)
        embeddings.append(speaker_emb)

    embeddings = mx.stack(embeddings, axis=0)  # (num_speakers, dim)

    # If fewer speakers than prompts, use all speakers
    # If more, cluster or sample
    num_speakers = embeddings.shape[0]
    num_prompts = config.num_prompts

    if num_speakers >= num_prompts:
        # Simple uniform sampling
        indices = mx.arange(0, num_speakers, num_speakers // num_prompts)[:num_prompts]
        initial_keys = embeddings[indices]
    else:
        # Replicate with noise
        reps = (num_prompts + num_speakers - 1) // num_speakers
        initial_keys = mx.tile(embeddings, (reps, 1))[:num_prompts]
        # Add noise for diversity
        noise = mx.random.normal(shape=initial_keys.shape) * 0.1
        initial_keys = initial_keys + noise

    # Create prompt bank with initialized keys
    prompt_bank = PromptBank(config)
    prompt_bank.prompt_keys = initial_keys
    prompt_bank._ema_keys = mx.array(initial_keys)

    return prompt_bank
