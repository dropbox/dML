"""ECAPA-TDNN Configuration for MLX.

This configuration matches the SpeechBrain VoxLingua107 checkpoint exactly.
Reference: https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa
"""

from dataclasses import dataclass, field


@dataclass
class ECAPATDNNConfig:
    """Configuration for ECAPA-TDNN model.

    Architecture:
        Input: 60 mel filterbanks
        Block 0: Conv1d(60, 1024, kernel=5) + BatchNorm + ReLU
        Blocks 1-3: SE-Res2Block with dilations [2, 3, 4]
        MFA: Multi-Feature Aggregation (concatenate all block outputs)
        ASP: Attentive Statistics Pooling
        FC: Linear to embedding dimension

    Attributes:
        n_mels: Number of mel filterbank features (input dimension)
        channels: Channel dimensions for each block
        kernel_sizes: Kernel sizes for each block
        dilations: Dilation factors for each SE-Res2Block
        attention_channels: Hidden dimension for ASP attention
        lin_neurons: Output embedding dimension
        res2net_scale: Number of sub-bands in Res2Net block
        se_channels: Squeeze-and-Excitation bottleneck dimension
        num_languages: Number of output language classes
    """

    # Feature extraction
    n_mels: int = 60
    sample_rate: int = 16000

    # ECAPA-TDNN architecture
    channels: list[int] = field(default_factory=lambda: [1024, 1024, 1024, 1024, 3072])
    kernel_sizes: list[int] = field(default_factory=lambda: [5, 3, 3, 3, 1])
    dilations: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 1])

    # Attentive Statistics Pooling
    attention_channels: int = 128

    # Output embedding
    lin_neurons: int = 256

    # Res2Net parameters
    res2net_scale: int = 8  # Number of sub-bands (inferred from 1024 / 128 = 8)

    # SE block parameters
    se_channels: int = 128  # SE bottleneck dimension

    # Classifier
    num_languages: int = 107
    classifier_hidden: int = 512

    @classmethod
    def voxlingua107(cls) -> "ECAPATDNNConfig":
        """Return configuration for VoxLingua107 language ID checkpoint."""
        return cls()

    @classmethod
    def voxceleb_speaker(cls) -> "ECAPATDNNConfig":
        """Return configuration for VoxCeleb speaker verification checkpoint.

        This is for speaker embeddings/verification, not language identification.
        Produces 192-dimensional speaker embeddings.
        """
        return cls(
            n_mels=80,  # VoxCeleb uses 80 mels
            lin_neurons=192,  # 192-dim speaker embeddings
            num_languages=1,  # Not used for speaker verification
        )
