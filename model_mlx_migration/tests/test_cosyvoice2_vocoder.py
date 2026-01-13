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
Tests for CosyVoice2 HiFi-GAN Vocoder

Phase 5.2 tests - HiFi-GAN vocoder implementation.
"""

import mlx.core as mx


class TestHiFiGANConfig:
    """Test HiFiGANConfig."""

    def test_default_config(self):
        """Test default config values."""
        from tools.pytorch_to_mlx.converters.models import HiFiGANConfig

        config = HiFiGANConfig()
        assert config.mel_channels == 80
        assert config.sample_rate == 22050
        assert config.upsample_initial_channel == 512
        assert len(config.upsample_rates) == 3  # 3 upsample stages in CosyVoice2
        assert len(config.resblock_kernel_sizes) == 3

    def test_custom_config(self):
        """Test custom config values."""
        from tools.pytorch_to_mlx.converters.models import HiFiGANConfig

        config = HiFiGANConfig(mel_channels=64, sample_rate=24000)
        assert config.mel_channels == 64
        assert config.sample_rate == 24000


class TestWeightNormConv1d:
    """Test WeightNormConv1d layer."""

    def test_forward_shape(self):
        """Test output shape."""
        from tools.pytorch_to_mlx.converters.models.cosyvoice2_vocoder import (
            WeightNormConv1d,
        )

        conv = WeightNormConv1d(80, 256, 7, padding=3)
        x = mx.random.normal((2, 100, 80))
        y = conv(x)

        assert y.shape == (2, 100, 256)

    def test_with_stride(self):
        """Test with stride > 1."""
        from tools.pytorch_to_mlx.converters.models.cosyvoice2_vocoder import (
            WeightNormConv1d,
        )

        conv = WeightNormConv1d(80, 256, 4, stride=2, padding=1)
        x = mx.random.normal((2, 100, 80))
        y = conv(x)

        assert y.shape == (2, 50, 256)


class TestResBlock1d:
    """Test ResBlock1d layer."""

    def test_forward_shape(self):
        """Test output shape preservation."""
        from tools.pytorch_to_mlx.converters.models.cosyvoice2_vocoder import ResBlock1d

        resblock = ResBlock1d(256, kernel_size=3, dilations=(1, 3, 5))
        x = mx.random.normal((2, 100, 256))
        y = resblock(x)

        assert y.shape == x.shape

    def test_residual_connection(self):
        """Test that residual connections work."""
        from tools.pytorch_to_mlx.converters.models.cosyvoice2_vocoder import ResBlock1d

        resblock = ResBlock1d(64, kernel_size=3, dilations=(1,))
        x = mx.zeros((1, 10, 64))
        y = resblock(x)

        # With zero input, output should not be far from zero due to residual
        assert mx.abs(y).max() < 1.0


class TestHiFiGANVocoder:
    """Test HiFiGANVocoder model."""

    def test_init(self):
        """Test model initialization."""
        from tools.pytorch_to_mlx.converters.models import HiFiGANConfig, HiFiGANVocoder

        config = HiFiGANConfig()
        vocoder = HiFiGANVocoder(config)

        assert vocoder is not None
        assert len(vocoder.ups) == 3  # 3 upsample stages in CosyVoice2
        assert len(vocoder.resblocks) == 9  # 3 stages * 3 kernel sizes

    def test_forward_shape(self):
        """Test forward pass output shape."""
        from tools.pytorch_to_mlx.converters.models import HiFiGANConfig, HiFiGANVocoder

        config = HiFiGANConfig()
        vocoder = HiFiGANVocoder(config)

        mel = mx.random.normal((1, 50, 80))
        audio = vocoder(mel)

        # Audio should be longer than mel due to upsampling
        assert len(audio.shape) == 2
        assert audio.shape[0] == 1
        assert audio.shape[1] > 50

    def test_batch_processing(self):
        """Test batch processing."""
        from tools.pytorch_to_mlx.converters.models import HiFiGANConfig, HiFiGANVocoder

        config = HiFiGANConfig()
        vocoder = HiFiGANVocoder(config)

        mel = mx.random.normal((4, 50, 80))
        audio = vocoder(mel)

        assert audio.shape[0] == 4


class TestF0Predictor:
    """Test F0Predictor module."""

    def test_forward_shape(self):
        """Test F0 prediction shape."""
        from tools.pytorch_to_mlx.converters.models.cosyvoice2_vocoder import (
            F0Predictor,
            HiFiGANConfig,
        )

        config = HiFiGANConfig()
        predictor = F0Predictor(config)

        mel = mx.random.normal((2, 100, 80))
        f0 = predictor(mel)

        assert f0.shape == (2, 100, 1)
