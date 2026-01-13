"""Audio Spectrogram Transformer (AST) Configuration for MLX.

This configuration matches the MIT AST AudioSet checkpoint.
Reference: https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593
Paper: https://arxiv.org/abs/2104.01778
"""

from dataclasses import dataclass


@dataclass
class ASTConfig:
    """Configuration for Audio Spectrogram Transformer.

    Architecture:
        - Patch embedding: Conv2d(1, hidden_size, kernel=patch_size, stride=stride)
        - Position embeddings: learnable (1, num_patches + 2, hidden_size)
        - cls_token and distillation_token
        - 12 Transformer encoder layers
        - LayerNorm + Classifier

    Attributes:
        hidden_size: Transformer hidden dimension
        num_hidden_layers: Number of transformer layers
        num_attention_heads: Number of attention heads
        intermediate_size: FFN intermediate dimension
        hidden_act: Activation function (gelu)
        hidden_dropout_prob: Dropout probability
        attention_probs_dropout_prob: Attention dropout probability
        patch_size: Patch size for embedding
        num_mel_bins: Number of mel frequency bins
        max_length: Maximum spectrogram length
        time_stride: Stride in time dimension
        frequency_stride: Stride in frequency dimension
        num_labels: Number of output classes
        qkv_bias: Whether to use bias in QKV projections
        layer_norm_eps: LayerNorm epsilon
    """

    # Transformer architecture
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0

    # Patch embedding
    patch_size: int = 16
    num_mel_bins: int = 128
    max_length: int = 1024
    time_stride: int = 10
    frequency_stride: int = 10

    # Output
    num_labels: int = 527
    qkv_bias: bool = True
    layer_norm_eps: float = 1e-12

    # Derived properties
    @property
    def num_patches(self) -> int:
        """Calculate number of patches based on input dimensions."""
        # Default AudioSet: 1024 time steps, 128 mel bins
        # With patch_size=16, time_stride=10, freq_stride=10:
        # time_patches = (1024 - 16) // 10 + 1 = 101.8 -> 101
        # freq_patches = (128 - 16) // 10 + 1 = 12.2 -> 12
        # Total = 101 * 12 = 1212 patches + 2 tokens = 1214
        time_patches = (self.max_length - self.patch_size) // self.time_stride + 1
        freq_patches = (self.num_mel_bins - self.patch_size) // self.frequency_stride + 1
        return time_patches * freq_patches

    @classmethod
    def audioset(cls) -> "ASTConfig":
        """Return configuration for AudioSet checkpoint (527 classes)."""
        return cls()

    @classmethod
    def from_dict(cls, config_dict: dict) -> "ASTConfig":
        """Create config from dictionary."""
        return cls(
            hidden_size=config_dict.get("hidden_size", 768),
            num_hidden_layers=config_dict.get("num_hidden_layers", 12),
            num_attention_heads=config_dict.get("num_attention_heads", 12),
            intermediate_size=config_dict.get("intermediate_size", 3072),
            hidden_act=config_dict.get("hidden_act", "gelu"),
            hidden_dropout_prob=config_dict.get("hidden_dropout_prob", 0.0),
            attention_probs_dropout_prob=config_dict.get("attention_probs_dropout_prob", 0.0),
            patch_size=config_dict.get("patch_size", 16),
            num_mel_bins=config_dict.get("num_mel_bins", 128),
            max_length=config_dict.get("max_length", 1024),
            time_stride=config_dict.get("time_stride", 10),
            frequency_stride=config_dict.get("frequency_stride", 10),
            num_labels=len(config_dict.get("id2label", {})) or 527,
            qkv_bias=config_dict.get("qkv_bias", True),
            layer_norm_eps=config_dict.get("layer_norm_eps", 1e-12),
        )
