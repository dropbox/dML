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
MLX implementation of scaling modules from icefall/zipformer.

This module ports the custom normalization and scaling layers used in Zipformer
from PyTorch to MLX.

Reference: tools/third_party/icefall/egs/librispeech/ASR/zipformer/scaling.py
"""


import mlx.core as mx
import mlx.nn as nn


class BiasNorm(nn.Module):
    """
    MLX implementation of BiasNorm from icefall.

    A simpler replacement for LayerNorm. Instead of learned weight and bias
    applied after normalization, BiasNorm uses:
    - A trainable bias SUBTRACTED for computing the scale (but output uses original x)
    - A trainable log_scale on the output

    This addresses the observation that Transformer networks sometimes set one
    feature dimension to a large constant to "defeat" LayerNorm.

    Forward computation (from icefall):
        centered = x - bias
        scales = (mean(centered^2, dim=channel_dim) + eps)^(-0.5) * exp(log_scale)
        output = x * scales

    Note: The bias is used only for computing the scale. The output is x * scales,
    not centered * scales.

    Args:
        num_channels: Number of channels in the input.
        channel_dim: Dimension along which to normalize (default: -1).
        log_scale: Initial value for the learnable log scale (default: 1.0).
        log_scale_min: Minimum allowed value for log_scale during training.
        log_scale_max: Maximum allowed value for log_scale during training.
    """

    def __init__(
        self,
        num_channels: int,
        channel_dim: int = -1,
        log_scale: float = 1.0,
        log_scale_min: float = -1.5,
        log_scale_max: float = 1.5,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.channel_dim = channel_dim
        self.log_scale_min = log_scale_min
        self.log_scale_max = log_scale_max

        # Learnable parameters
        self.log_scale = mx.array(log_scale)
        self.bias = mx.random.normal(shape=(num_channels,)) * 1e-4

    def __call__(self, x: mx.array) -> mx.array:
        """
        Apply BiasNorm to input tensor.

        BiasNorm formula from icefall:
            centered = x - bias
            scales = (mean(centered^2))^(-0.5) * exp(log_scale)
            output = x * scales

        Note: The bias is subtracted for computing the scale, but the output
        is the original x multiplied by scales (not the centered value).

        Args:
            x: Input tensor with shape [..., num_channels, ...] where
               num_channels is at position channel_dim.

        Returns:
            Normalized tensor with same shape as input.
        """
        # Resolve negative channel_dim
        channel_dim = self.channel_dim
        if channel_dim < 0:
            channel_dim = x.ndim + channel_dim

        # Expand bias to match input dimensions
        bias = self.bias
        for _ in range(channel_dim + 1, x.ndim):
            bias = mx.expand_dims(bias, axis=-1)

        # Clamp log_scale to valid range during forward pass
        log_scale = mx.clip(self.log_scale, self.log_scale_min, self.log_scale_max)

        # Compute scale: (mean((x - bias)^2))^(-0.5) * exp(log_scale)
        centered = x - bias
        variance = mx.mean(centered * centered, axis=channel_dim, keepdims=True)
        # Add small epsilon to avoid division by zero
        scale = mx.rsqrt(variance + 1e-8) * mx.exp(log_scale)

        # Return original x times scale (not centered!)
        return x * scale


class ScaledLinear(nn.Linear):
    """
    A Linear layer with scaled initialization.

    This is a simple wrapper around nn.Linear that scales the initial
    weights by initial_scale. This can help with training stability
    in deep networks.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: If set to False, the layer will not learn an additive bias.
        initial_scale: Scale factor for weight initialization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        initial_scale: float = 1.0,
    ):
        super().__init__(in_features, out_features, bias)

        # Scale initial weights
        if initial_scale != 1.0:
            self.weight = self.weight * initial_scale
            if bias and hasattr(self, 'bias'):
                self.bias = self.bias * initial_scale


class Balancer(nn.Module):
    """
    MLX implementation of Balancer from icefall.

    A module that adds a loss term to encourage the mean absolute value
    of activations to be close to a target value. This helps with training
    stability and gradient flow.

    During inference, this is a no-op (identity function).

    Args:
        num_channels: Number of channels in the input.
        channel_dim: Dimension along which to compute statistics.
        min_positive: Minimum fraction of positive values (not used in inference).
        max_positive: Maximum fraction of positive values (not used in inference).
        min_abs: Minimum mean absolute value.
        max_abs: Maximum mean absolute value.
    """

    def __init__(
        self,
        num_channels: int,
        channel_dim: int = -1,
        min_positive: float = 0.05,
        max_positive: float = 0.95,
        min_abs: float = 0.2,
        max_abs: float = 100.0,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.channel_dim = channel_dim
        self.min_positive = min_positive
        self.max_positive = max_positive
        self.min_abs = min_abs
        self.max_abs = max_abs

    def __call__(self, x: mx.array) -> mx.array:
        """Identity function during inference."""
        return x


class Whiten(nn.Module):
    """
    MLX implementation of Whiten from icefall.

    A module that whitens (decorrelates) activations. During training,
    it encourages the covariance matrix of activations to be close to identity.

    During inference, this is a no-op (identity function).

    Args:
        num_groups: Number of groups to split channels into.
        whitening_limit: Maximum amount of whitening to apply.
        prob: Probability of applying whitening during training.
        grad_scale: Scale factor for gradients.
    """

    def __init__(
        self,
        num_groups: int,
        whitening_limit: float = 5.0,
        prob: float = 0.025,
        grad_scale: float = 0.01,
    ):
        super().__init__()
        self.num_groups = num_groups
        self.whitening_limit = whitening_limit
        self.prob = prob
        self.grad_scale = grad_scale

    def __call__(self, x: mx.array) -> mx.array:
        """Identity function during inference."""
        return x


class Identity(nn.Module):
    """Identity module (no-op)."""

    def __call__(self, x: mx.array) -> mx.array:
        return x


class ActivationDropoutAndLinear(nn.Module):
    """
    Combined activation, dropout, and linear layer.

    This combines SwooshL/SwooshR activation with dropout and linear projection
    into a single module for efficiency.

    Args:
        in_features: Input feature dimension.
        out_features: Output feature dimension.
        bias: Whether to use bias in linear layer.
        activation: Type of activation ('SwooshL' or 'SwooshR').
        dropout: Dropout probability.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: str = "SwooshL",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation_type = activation
        self.dropout_prob = dropout

        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        # Apply activation
        if self.activation_type == "SwooshL":
            x = self._swoosh_l(x)
        elif self.activation_type == "SwooshR":
            x = self._swoosh_r(x)

        # Note: Dropout not applied during inference in MLX
        # In training, would need to handle dropout separately

        return self.linear(x)

    @staticmethod
    def _swoosh_l(x: mx.array) -> mx.array:
        """SwooshL activation: x * sigmoid(x - 1) + 0.1 * sigmoid(-x)"""
        return x * mx.sigmoid(x - 1) + 0.1 * mx.sigmoid(-x)

    @staticmethod
    def _swoosh_r(x: mx.array) -> mx.array:
        """SwooshR activation: x * sigmoid(x + 1) - 0.1"""
        return x * mx.sigmoid(x + 1) - 0.1
