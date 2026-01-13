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
DELULU-style speaker embedding head for Zipformer encoder.

Produces speaker embeddings with target <0.8% EER (62% better than
ECAPA-TDNN baseline at ~2% EER).

Architecture based on DELULU (arXiv:2510.17662) which combines:
- Squeeze-and-Excitation (SE) channel attention
- Res2Net-style multi-scale feature extraction
- Attentive statistical pooling
- Self-supervised learning objectives (masked prediction + denoising)

Training uses AAM-Softmax (Additive Angular Margin Softmax) loss
for speaker-discriminative embeddings.

Reference:
- DELULU: arXiv:2510.17662 - 62% relative EER improvement
- ECAPA-TDNN: Interspeech 2020 - baseline architecture
- ReDimNet: Foundation for DELULU embeddings
"""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class SpeakerConfig:
    """Configuration for DELULU-style speaker embedding head."""

    # Input dimension from encoder
    encoder_dim: int = 384

    # Embedding output dimension
    embedding_dim: int = 256

    # Number of speakers for AAM-Softmax (set during training)
    num_speakers: int = 7205  # VoxCeleb2 training speakers

    # Res2Net-style scale factor
    res2net_scale: int = 8

    # SE block reduction ratio
    se_reduction: int = 8

    # Number of attention heads for pooling
    num_attention_heads: int = 1

    # Hidden dimension for feature processing
    hidden_dim: int = 1536

    # Bottleneck dimension for SE blocks
    bottleneck_dim: int = 128

    # Dropout rate
    dropout_rate: float = 0.1

    # AAM-Softmax margin
    aam_margin: float = 0.2

    # AAM-Softmax scale
    aam_scale: float = 30.0

    # Use self-supervised objectives during training
    use_ssl: bool = True

    # Masked prediction ratio for SSL
    mask_ratio: float = 0.15

    # Denoising probability for SSL
    denoise_prob: float = 0.5


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention.

    Learns channel-wise attention weights to emphasize important
    channels and suppress less useful ones.

    Args:
        channels: Number of input channels.
        reduction: Reduction ratio for bottleneck.
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        bottleneck = max(channels // reduction, 8)

        self.squeeze = nn.Sequential(
            nn.Linear(channels, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, channels),
        )

    def __call__(self, x: mx.array) -> mx.array:
        """
        Apply SE attention.

        Args:
            x: Input of shape (batch, seq, channels)

        Returns:
            Channel-weighted output of same shape.
        """
        # Global average pooling
        weights = mx.mean(x, axis=1)  # (batch, channels)

        # Squeeze and excitation
        weights = self.squeeze(weights)  # (batch, channels)
        weights = mx.sigmoid(weights)

        # Apply attention
        return x * mx.expand_dims(weights, axis=1)


class Res2NetBlock(nn.Module):
    """
    Res2Net-style multi-scale feature extraction block.

    Splits features into multiple scales and processes them
    with hierarchical residual connections for richer multi-scale
    representations.

    Args:
        channels: Number of input/output channels.
        scale: Number of scales (splits).
        kernel_size: Convolution kernel size.
    """

    def __init__(
        self,
        channels: int,
        scale: int = 8,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.scale = scale
        self.width = channels // scale

        # 1D convolutions for each scale (excluding first)
        self.convs = [
            nn.Conv1d(
                self.width,
                self.width,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
            for _ in range(scale - 1)
        ]

        self.bn = nn.BatchNorm(channels)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Apply Res2Net multi-scale processing.

        Args:
            x: Input of shape (batch, seq, channels)

        Returns:
            Multi-scale features of same shape.
        """
        batch_size, seq_len, channels = x.shape

        # Split into scales
        splits = mx.split(x, self.scale, axis=-1)

        outputs = []
        prev = None

        for i, split in enumerate(splits):
            if i == 0:
                # First scale passes through directly
                outputs.append(split)
                prev = split
            else:
                # Add previous scale output
                combined = split + prev if prev is not None else split

                # Apply convolution (MLX Conv1d takes batch, seq, channels)
                conv_out = self.convs[i - 1](combined)
                conv_out = nn.relu(conv_out)

                outputs.append(conv_out)
                prev = conv_out

        # Concatenate scales
        out = mx.concatenate(outputs, axis=-1)

        # Batch normalization
        out = self.bn(out)

        return out


class SERes2NetBlock(nn.Module):
    """
    Combined SE + Res2Net block.

    Applies Res2Net multi-scale processing followed by SE attention.

    Args:
        channels: Number of channels.
        scale: Res2Net scale factor.
        reduction: SE reduction ratio.
    """

    def __init__(
        self,
        channels: int,
        scale: int = 8,
        reduction: int = 8,
    ):
        super().__init__()
        self.res2net = Res2NetBlock(channels, scale)
        self.se = SqueezeExcitation(channels, reduction)

    def __call__(self, x: mx.array) -> mx.array:
        """Apply SE-Res2Net processing."""
        out = self.res2net(x)
        out = self.se(out)
        return out + x  # Residual connection


class AttentiveStatisticsPooling(nn.Module):
    """
    Attentive statistics pooling for utterance-level embeddings.

    Computes attention-weighted mean and standard deviation,
    concatenated into a fixed-length representation.

    Args:
        in_dim: Input feature dimension.
        attention_dim: Attention bottleneck dimension.
        num_heads: Number of attention heads.
    """

    def __init__(
        self,
        in_dim: int,
        attention_dim: int = 128,
        num_heads: int = 1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads

        # Attention network
        self.attention = nn.Sequential(
            nn.Linear(in_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, num_heads),
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        """
        Compute attentive statistics pooling.

        Args:
            x: Input of shape (batch, seq, in_dim)
            mask: Optional mask of shape (batch, seq), True = masked

        Returns:
            Pooled features of shape (batch, in_dim * 2)
        """
        batch_size, seq_len, _ = x.shape

        # Compute attention weights
        attn_scores = self.attention(x)  # (batch, seq, num_heads)

        # Apply mask if provided
        if mask is not None:
            mask_expanded = mx.expand_dims(mask, axis=-1)
            attn_scores = mx.where(
                mask_expanded,
                mx.full(attn_scores.shape, -1e9),
                attn_scores,
            )

        # Softmax over sequence
        attn_weights = mx.softmax(attn_scores, axis=1)  # (batch, seq, num_heads)

        # Average attention across heads
        attn_weights = mx.mean(attn_weights, axis=-1, keepdims=True)  # (batch, seq, 1)

        # Weighted mean
        weighted_mean = mx.sum(x * attn_weights, axis=1)  # (batch, in_dim)

        # Weighted variance -> std
        diff = x - mx.expand_dims(weighted_mean, axis=1)
        weighted_var = mx.sum(diff * diff * attn_weights, axis=1)
        weighted_std = mx.sqrt(weighted_var + 1e-8)

        # Concatenate mean and std
        pooled = mx.concatenate([weighted_mean, weighted_std], axis=-1)

        return pooled


class SpeakerHead(nn.Module):
    """
    DELULU-style speaker embedding head for Zipformer encoder.

    Takes encoder output and produces speaker embeddings suitable
    for speaker verification with AAM-Softmax training.

    Architecture:
    1. Feature projection from encoder dim
    2. SE-Res2Net blocks for multi-scale processing
    3. Attentive statistics pooling
    4. Embedding projection with batch normalization

    Args:
        config: SpeakerConfig instance with hyperparameters.
    """

    def __init__(self, config: SpeakerConfig | None = None):
        super().__init__()
        if config is None:
            config = SpeakerConfig()

        self.config = config

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(config.encoder_dim, config.hidden_dim),
            nn.ReLU(),
            nn.BatchNorm(config.hidden_dim),
        )

        # SE-Res2Net blocks
        self.se_res2net_blocks = [
            SERes2NetBlock(
                config.hidden_dim,
                scale=config.res2net_scale,
                reduction=config.se_reduction,
            )
            for _ in range(3)
        ]

        # MFA (Multi-layer Feature Aggregation) attention
        # Combines outputs from all SE-Res2Net blocks
        self.mfa_weight = mx.ones((3,)) / 3.0

        # Attentive statistics pooling
        self.asp = AttentiveStatisticsPooling(
            in_dim=config.hidden_dim,
            attention_dim=config.bottleneck_dim,
            num_heads=config.num_attention_heads,
        )

        # Embedding projection (mean + std concatenated)
        self.embedding_proj = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.embedding_dim),
        )

        # Batch normalization for embeddings
        self.embedding_bn = nn.BatchNorm(config.embedding_dim)

        # AAM-Softmax classification layer (for training)
        self.classifier = nn.Linear(
            config.embedding_dim, config.num_speakers, bias=False,
        )

        # Layer norm before processing
        self.layer_norm = nn.LayerNorm(config.encoder_dim)

    def __call__(
        self,
        encoder_out: mx.array,
        encoder_lengths: mx.array | None = None,
        return_embeddings_only: bool = True,
    ) -> mx.array:
        """
        Extract speaker embeddings from encoder output.

        Args:
            encoder_out: Encoder output of shape (batch, seq, encoder_dim)
            encoder_lengths: Optional sequence lengths of shape (batch,)
            return_embeddings_only: If True, return only embeddings (inference).
                                   If False, return (embeddings, logits) for training.

        Returns:
            Speaker embeddings of shape (batch, embedding_dim)
            Or tuple (embeddings, logits) if return_embeddings_only=False
        """
        encoder_out.shape[0]
        seq_len = encoder_out.shape[1]

        # Create mask from lengths if provided
        mask = None
        if encoder_lengths is not None:
            positions = mx.arange(seq_len)
            mask = positions >= mx.expand_dims(encoder_lengths, axis=1)

        # Layer norm
        x = self.layer_norm(encoder_out)

        # Input projection
        x = self.input_proj(x)

        # SE-Res2Net blocks with MFA
        block_outputs = []
        for block in self.se_res2net_blocks:
            x = block(x)
            block_outputs.append(x)

        # Multi-layer feature aggregation
        mfa_weights = mx.softmax(self.mfa_weight, axis=0)
        x = sum(w * out for w, out in zip(mfa_weights, block_outputs, strict=False))

        # Attentive statistics pooling
        pooled = self.asp(x, mask=mask)

        # Embedding projection
        embeddings = self.embedding_proj(pooled)
        embeddings = self.embedding_bn(embeddings)

        # L2 normalize embeddings
        embeddings = embeddings / (
            mx.sqrt(mx.sum(embeddings * embeddings, axis=-1, keepdims=True)) + 1e-8
        )

        if return_embeddings_only:
            return embeddings

        # For training, compute AAM-Softmax logits
        # Normalize classifier weights
        weights = self.classifier.weight
        weights_norm = weights / (
            mx.sqrt(mx.sum(weights * weights, axis=-1, keepdims=True)) + 1e-8
        )

        # Cosine similarity (embeddings already normalized)
        logits = mx.matmul(embeddings, weights_norm.T) * self.config.aam_scale

        return embeddings, logits

    def extract_embedding(
        self,
        encoder_out: mx.array,
        encoder_lengths: mx.array | None = None,
    ) -> mx.array:
        """
        Extract normalized speaker embedding for verification.

        This is the inference-time method for extracting embeddings
        to be compared via cosine similarity.

        Args:
            encoder_out: Encoder output
            encoder_lengths: Optional sequence lengths

        Returns:
            L2-normalized speaker embedding of shape (batch, embedding_dim)
        """
        return self(encoder_out, encoder_lengths, return_embeddings_only=True)

    def similarity(
        self,
        embedding1: mx.array,
        embedding2: mx.array,
    ) -> mx.array:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding (batch, embedding_dim) or (embedding_dim,)
            embedding2: Second embedding (batch, embedding_dim) or (embedding_dim,)

        Returns:
            Cosine similarity score(s)
        """
        # Ensure 2D
        if embedding1.ndim == 1:
            embedding1 = mx.expand_dims(embedding1, axis=0)
        if embedding2.ndim == 1:
            embedding2 = mx.expand_dims(embedding2, axis=0)

        # Embeddings are already L2 normalized, so dot product = cosine similarity
        return mx.sum(embedding1 * embedding2, axis=-1)


def aam_softmax_loss(
    embeddings: mx.array,
    logits: mx.array,
    targets: mx.array,
    margin: float = 0.2,
    scale: float = 30.0,
) -> mx.array:
    """
    Additive Angular Margin Softmax (AAM-Softmax) loss.

    Adds an angular margin penalty to the target class logit,
    encouraging larger angular separations between classes.

    Args:
        embeddings: L2-normalized embeddings (batch, embedding_dim)
        logits: Pre-computed cosine similarity logits (batch, num_classes)
        targets: Target class indices (batch,)
        margin: Angular margin in radians
        scale: Scaling factor for logits

    Returns:
        Loss value (scalar)
    """
    batch_size = embeddings.shape[0]
    num_classes = logits.shape[-1]

    # Get cosine of angles (logits are already scaled cosine similarities)
    cos_theta = logits / scale  # Undo scaling to get raw cosines

    # Compute cos(theta + m) for target classes only
    # Using cos(a+b) = cos(a)cos(b) - sin(a)sin(b)
    cos_m = mx.cos(mx.array(margin))
    sin_m = mx.sin(mx.array(margin))

    # Get target class cosines
    batch_indices = mx.arange(batch_size)
    target_cos = cos_theta[batch_indices, targets]

    # sin(theta) = sqrt(1 - cos^2(theta))
    target_sin = mx.sqrt(mx.maximum(1.0 - target_cos * target_cos, 1e-8))

    # cos(theta + m)
    target_cos_m = target_cos * cos_m - target_sin * sin_m

    # Apply margin only when cos(theta) > cos(pi - m)
    # to avoid numerical issues
    threshold = mx.cos(mx.array(3.14159265 - margin))
    target_cos_m = mx.where(
        target_cos > threshold, target_cos_m, target_cos - margin * sin_m,
    )

    # Create modified logits with margin applied to targets
    # Use one-hot encoding for efficient update
    one_hot = mx.zeros((batch_size, num_classes))
    one_hot = one_hot.at[batch_indices, targets].add(1.0)

    modified_logits = scale * (
        cos_theta * (1.0 - one_hot) + target_cos_m[:, None] * one_hot
    )

    # Cross-entropy loss
    log_probs = modified_logits - mx.logsumexp(modified_logits, axis=-1, keepdims=True)
    loss = -log_probs[batch_indices, targets]

    return mx.mean(loss)


class SpeakerLoss(nn.Module):
    """AAM-Softmax loss as nn.Module wrapper."""

    def __init__(
        self,
        margin: float = 0.2,
        scale: float = 30.0,
    ):
        super().__init__()
        self.margin = margin
        self.scale = scale

    def __call__(
        self,
        embeddings: mx.array,
        logits: mx.array,
        targets: mx.array,
    ) -> mx.array:
        return aam_softmax_loss(
            embeddings=embeddings,
            logits=logits,
            targets=targets,
            margin=self.margin,
            scale=self.scale,
        )


def speaker_loss(
    embeddings: mx.array,
    logits: mx.array,
    targets: mx.array,
    margin: float = 0.2,
    scale: float = 30.0,
) -> mx.array:
    """Convenience function for AAM-Softmax loss."""
    return aam_softmax_loss(embeddings, logits, targets, margin, scale)


def verification_eer(
    similarities: mx.array,
    labels: mx.array,
) -> float:
    """
    Compute Equal Error Rate (EER) for speaker verification.

    Args:
        similarities: Cosine similarities for trial pairs
        labels: Binary labels (1 = same speaker, 0 = different)

    Returns:
        EER as a float (0.0 to 1.0)
    """
    # Convert to numpy for EER computation
    sims = similarities.tolist() if hasattr(similarities, "tolist") else list(similarities)
    labs = labels.tolist() if hasattr(labels, "tolist") else list(labels)

    # Sort by similarity descending
    pairs = sorted(zip(sims, labs, strict=False), reverse=True)

    # Count positives and negatives
    num_positive = sum(labs)
    num_negative = len(labs) - num_positive

    if num_positive == 0 or num_negative == 0:
        return 0.5  # Undefined, return random baseline

    # Compute FAR and FRR at each threshold
    far_prev = 0.0
    frr_prev = 1.0
    tp = 0
    fp = 0

    for sim, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1

        far = fp / num_negative  # False Accept Rate
        frr = 1.0 - tp / num_positive  # False Reject Rate

        # Check for EER crossing
        if far >= frr:
            # Linear interpolation
            if far == frr:
                return max(0.0, min(1.0, far))
            # EER is between previous and current thresholds
            eer = (far_prev * (far - frr) + far * (frr_prev - far)) / (
                (far - frr) + (frr_prev - far_prev) + 1e-10
            )
            return max(0.0, min(1.0, eer))  # Clamp to [0, 1]

        far_prev = far
        frr_prev = frr

    return max(0.0, min(1.0, far_prev))  # If no crossing found, return last FAR
