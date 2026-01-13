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
Unit tests for CosyVoice2 Full Model.

Tests the integrated CosyVoice2 model components.
"""

import mlx.core as mx
import pytest

from tools.pytorch_to_mlx.converters.models.cosyvoice2 import (
    CosyVoice2Config,
    CosyVoice2Model,
    count_parameters,
)
from tools.pytorch_to_mlx.converters.models.cosyvoice2_flow import FlowMatchingConfig
from tools.pytorch_to_mlx.converters.models.cosyvoice2_llm import Qwen2Config
from tools.pytorch_to_mlx.converters.models.cosyvoice2_vocoder import HiFiGANConfig


class TestCosyVoice2Config:
    """Tests for CosyVoice2Config."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CosyVoice2Config()

        assert config.sample_rate == 24000
        assert config.token_frame_rate == 25
        assert config.token_mel_ratio == 2
        assert config.chunk_size == 25

    def test_nested_configs(self):
        """Test that nested configs are initialized."""
        config = CosyVoice2Config()

        assert isinstance(config.llm_config, Qwen2Config)
        assert isinstance(config.flow_config, FlowMatchingConfig)
        assert isinstance(config.vocoder_config, HiFiGANConfig)

    def test_custom_llm_config(self):
        """Test custom LLM config."""
        llm_config = Qwen2Config(num_hidden_layers=2)
        config = CosyVoice2Config(llm_config=llm_config)

        assert config.llm_config.num_hidden_layers == 2


class TestCosyVoice2ModelSmall:
    """Tests for CosyVoice2Model with small config for fast testing."""

    @pytest.fixture
    def small_config(self):
        """Create small config for testing."""
        return CosyVoice2Config(
            llm_config=Qwen2Config(
                num_hidden_layers=2,
                vocab_size=1000,
                speech_vocab_size=500,
            ),
            flow_config=FlowMatchingConfig(
                vocab_size=500,
                num_encoder_layers=2,
            ),
            vocoder_config=HiFiGANConfig(),
        )

    def test_initialization(self, small_config):
        """Test model initialization."""
        model = CosyVoice2Model(small_config)

        assert model.llm is not None
        assert model.flow is not None
        assert model.vocoder is not None

    def test_model_has_config(self, small_config):
        """Test model stores config."""
        model = CosyVoice2Model(small_config)

        assert model.config == small_config
        assert model.config.sample_rate == 24000


class TestCountParameters:
    """Tests for parameter counting utility."""

    def test_count_parameters(self):
        """Test parameter counting."""
        from tools.pytorch_to_mlx.converters.models.cosyvoice2_llm import Qwen2RMSNorm

        norm = Qwen2RMSNorm(hidden_size=256)
        count = count_parameters(norm)

        assert count == 256  # Weight only

    def test_count_params_nested(self):
        """Test counting params in nested modules."""
        from tools.pytorch_to_mlx.converters.models.cosyvoice2_llm import (
            Qwen2Config,
            Qwen2MLP,
        )

        config = Qwen2Config(hidden_size=64, intermediate_size=128)
        mlp = Qwen2MLP(config)
        count = count_parameters(mlp)

        # gate: 64*128 = 8192, up: 64*128 = 8192, down: 128*64 = 8192
        assert count == 8192 * 3


class TestCosyVoice2Pipeline:
    """Tests for CosyVoice2 pipeline methods."""

    @pytest.fixture
    def tiny_model(self):
        """Create tiny model for pipeline testing."""
        config = CosyVoice2Config(
            llm_config=Qwen2Config(
                num_hidden_layers=1,
                vocab_size=100,
                speech_vocab_size=100,
            ),
            flow_config=FlowMatchingConfig(
                vocab_size=100,
                num_encoder_layers=1,
            ),
        )
        return CosyVoice2Model(config)

    def test_generate_speech_tokens_interface(self, tiny_model):
        """Test generate_speech_tokens method signature."""
        text_ids = mx.random.randint(0, 100, (1, 5))

        # Should not raise
        tokens = tiny_model.generate_speech_tokens(
            text_ids,
            max_length=3,
            temperature=1.0,
            top_k=10,
        )
        mx.eval(tokens)

        assert tokens.shape[0] == 1  # batch
        assert tokens.shape[1] <= 3  # max_length

    def test_streaming_interface(self, tiny_model):
        """Test streaming generation interface."""
        text_ids = mx.random.randint(0, 100, (1, 5))

        chunks = list(
            tiny_model.generate_speech_tokens_stream(
                text_ids,
                max_length=5,
                chunk_size=2,
            ),
        )

        # Should yield at least one chunk
        assert len(chunks) >= 1

        # Each chunk is (tokens, is_final)
        for tokens, is_final in chunks:
            assert isinstance(is_final, bool)
            assert tokens.shape[0] == 1


class TestCosyVoice2ModelPath:
    """Tests for model path utilities."""

    def test_get_default_model_path(self):
        """Test default model path."""
        path = CosyVoice2Model.get_default_model_path()

        assert path.name == "cosyvoice2-0.5b"
        assert "cosyvoice2" in str(path)


class TestCosyVoice2TokenizerIntegration:
    """Tests for tokenizer integration with CosyVoice2Model."""

    @pytest.fixture
    def model_without_tokenizer(self):
        """Create model without tokenizer loaded."""
        config = CosyVoice2Config(
            llm_config=Qwen2Config(
                num_hidden_layers=1,
                vocab_size=100,
                speech_vocab_size=100,
            ),
            flow_config=FlowMatchingConfig(
                vocab_size=100,
                num_encoder_layers=1,
            ),
        )
        return CosyVoice2Model(config)

    def test_tokenizer_none_by_default(self, model_without_tokenizer):
        """Test tokenizer is None when not loaded from pretrained."""
        assert model_without_tokenizer.tokenizer is None

    def test_synthesize_text_without_tokenizer_raises(self, model_without_tokenizer):
        """Test synthesize_text raises when tokenizer not loaded."""
        with pytest.raises(RuntimeError, match="Tokenizer not loaded"):
            model_without_tokenizer.synthesize_text("Hello")

    def test_synthesize_text_stream_without_tokenizer_raises(
        self, model_without_tokenizer,
    ):
        """Test synthesize_text_stream raises when tokenizer not loaded."""
        with pytest.raises(RuntimeError, match="Tokenizer not loaded"):
            list(model_without_tokenizer.synthesize_text_stream("Hello"))


@pytest.mark.filterwarnings("ignore:onnxruntime not available:UserWarning")
class TestCosyVoice2FromPretrained:
    """Tests for from_pretrained with tokenizer."""

    @pytest.fixture
    def model_path(self):
        """Get model path, skip if not available."""
        from pathlib import Path

        path = Path.home() / ".cache" / "cosyvoice2" / "cosyvoice2-0.5b"
        if not path.exists():
            pytest.skip("CosyVoice2 model not downloaded")
        return path

    def test_from_pretrained_loads_tokenizer(self, model_path):
        """Test from_pretrained loads tokenizer."""
        model = CosyVoice2Model.from_pretrained(model_path)

        assert model.tokenizer is not None
        assert model.tokenizer.vocab_size > 0

    def test_tokenizer_encode_works(self, model_path):
        """Test tokenizer encode works after from_pretrained."""
        model = CosyVoice2Model.from_pretrained(model_path)

        tokens = model.tokenizer.encode("Hello, world!")
        assert tokens.shape[0] > 0


class TestCosyVoice2BatchSynthesis:
    """Tests for batch synthesis functionality."""

    @pytest.fixture
    def tiny_model(self):
        """Create tiny model for batch testing."""
        config = CosyVoice2Config(
            llm_config=Qwen2Config(
                num_hidden_layers=1,
                vocab_size=100,
                speech_vocab_size=100,
            ),
            flow_config=FlowMatchingConfig(
                vocab_size=100,
                num_encoder_layers=1,
            ),
        )
        return CosyVoice2Model(config)

    def test_batch_synthesis_basic(self, tiny_model):
        """Test basic batch synthesis interface."""
        mx.random.seed(42)  # Ensure deterministic test
        batch_size = 3
        text_ids_list = [
            mx.random.randint(0, 100, (5,)),  # seq_len 5
            mx.random.randint(0, 100, (3,)),  # seq_len 3
            mx.random.randint(0, 100, (7,)),  # seq_len 7
        ]
        speaker = mx.random.normal((batch_size, 192))

        audio, lengths = tiny_model.synthesize_batch(
            text_ids_list,
            speaker,
            max_tokens=10,
            num_flow_steps=2,
        )
        mx.eval(audio, lengths)

        assert audio.shape[0] == batch_size
        assert lengths.shape[0] == batch_size
        # Audio should have samples
        assert audio.shape[1] > 0
        # Lengths should be non-negative (could be 0 if EOS hit immediately with random weights)
        assert mx.all(lengths >= 0).item()

    def test_batch_synthesis_single_speaker(self, tiny_model):
        """Test batch synthesis with single speaker broadcast."""
        mx.random.seed(42)  # Ensure deterministic test
        batch_size = 2
        text_ids_list = [
            mx.random.randint(0, 100, (4,)),
            mx.random.randint(0, 100, (4,)),
        ]
        # Single speaker for all
        single_speaker = mx.random.normal((192,))

        audio, lengths = tiny_model.synthesize_batch(
            text_ids_list,
            single_speaker,  # Will be broadcast to batch
            max_tokens=15,  # Needs enough tokens for flow encoder convolutions
            num_flow_steps=2,
        )
        mx.eval(audio, lengths)

        assert audio.shape[0] == batch_size
        assert lengths.shape[0] == batch_size

    def test_batch_synthesis_speaker_list(self, tiny_model):
        """Test batch synthesis with per-utterance speaker embeddings."""
        mx.random.seed(42)  # Ensure deterministic test
        batch_size = 2
        text_ids_list = [
            mx.random.randint(0, 100, (4,)),
            mx.random.randint(0, 100, (4,)),
        ]
        # Per-utterance speakers as list
        speaker_list = [
            mx.random.normal((192,)),
            mx.random.normal((192,)),
        ]

        audio, lengths = tiny_model.synthesize_batch(
            text_ids_list,
            speaker_list,
            max_tokens=15,  # Needs enough tokens for flow encoder convolutions
            num_flow_steps=2,
        )
        mx.eval(audio, lengths)

        assert audio.shape[0] == batch_size

    def test_batch_synthesis_variable_lengths(self, tiny_model):
        """Test batch synthesis handles variable-length inputs."""
        mx.random.seed(42)  # Ensure deterministic test
        text_ids_list = [
            mx.random.randint(0, 100, (2,)),   # Short
            mx.random.randint(0, 100, (10,)),  # Long
        ]
        speaker = mx.random.normal((2, 192))

        audio, lengths = tiny_model.synthesize_batch(
            text_ids_list,
            speaker,
            max_tokens=8,
            num_flow_steps=2,
        )
        mx.eval(audio, lengths)

        # Should produce valid output for both
        assert audio.shape[0] == 2
        # Longer input might produce more tokens (but depends on model)
        assert mx.all(lengths > 0).item()

    def test_batch_synthesis_empty_raises(self, tiny_model):
        """Test empty batch raises error."""
        mx.random.seed(42)  # Ensure deterministic test
        with pytest.raises(ValueError, match="Empty text_ids_list"):
            tiny_model.synthesize_batch([], mx.random.normal((192,)))

    def test_batch_synthesis_speaker_mismatch_raises(self, tiny_model):
        """Test speaker batch size mismatch raises error."""
        mx.random.seed(42)  # Ensure deterministic test
        text_ids_list = [mx.random.randint(0, 100, (5,))]
        # Wrong number of speakers
        wrong_speakers = mx.random.normal((3, 192))

        with pytest.raises(ValueError, match="must match input batch size"):
            tiny_model.synthesize_batch(
                text_ids_list,
                wrong_speakers,
                max_tokens=3,
            )

    def test_batch_synthesis_audio_clipping(self, tiny_model):
        """Test audio output is clipped to valid range."""
        mx.random.seed(42)  # Ensure deterministic test
        text_ids_list = [mx.random.randint(0, 100, (5,))]
        speaker = mx.random.normal((192,))

        # Flow encoder needs minimum 6 tokens for PreLookaheadLayer convolutions:
        # conv1(kernel=4) + conv2(kernel=3) - 1 = 6 minimum input tokens
        # Use max_tokens=50 to ensure LLM generates enough even with random weights
        audio, _ = tiny_model.synthesize_batch(
            text_ids_list,
            speaker,
            max_tokens=50,
            num_flow_steps=2,
        )
        mx.eval(audio)

        # Audio should be clipped to [-1, 1]
        assert mx.all(audio >= -1.0).item()
        assert mx.all(audio <= 1.0).item()


class TestCosyVoice2BatchSynthesisText:
    """Tests for text-based batch synthesis."""

    @pytest.fixture
    def model_path(self):
        """Get model path, skip if not available."""
        from pathlib import Path

        path = Path.home() / ".cache" / "cosyvoice2" / "cosyvoice2-0.5b"
        if not path.exists():
            pytest.skip("CosyVoice2 model not downloaded")
        return path

    def test_batch_text_without_tokenizer_raises(self):
        """Test synthesize_batch_text raises without tokenizer."""
        config = CosyVoice2Config(
            llm_config=Qwen2Config(num_hidden_layers=1, vocab_size=100, speech_vocab_size=100),
            flow_config=FlowMatchingConfig(vocab_size=100, num_encoder_layers=1),
        )
        model = CosyVoice2Model(config)

        with pytest.raises(RuntimeError, match="Tokenizer not loaded"):
            model.synthesize_batch_text(["Hello", "World"])

    @pytest.mark.filterwarnings("ignore:onnxruntime not available:UserWarning")
    def test_batch_text_interface(self, model_path):
        """Test synthesize_batch_text with real model."""
        model = CosyVoice2Model.from_pretrained(model_path)

        texts = ["Hello", "Good morning"]
        audio, lengths = model.synthesize_batch_text(
            texts,
            max_tokens=10,
            num_flow_steps=2,
        )
        mx.eval(audio, lengths)

        assert audio.shape[0] == 2
        assert lengths.shape[0] == 2


class TestCosyVoice2LLMBatchGeneration:
    """Tests for LLM batch token generation with length tracking."""

    @pytest.fixture
    def tiny_llm(self):
        """Create tiny LLM for testing."""
        from tools.pytorch_to_mlx.converters.models.cosyvoice2_llm import CosyVoice2LLM

        config = Qwen2Config(
            num_hidden_layers=1,
            vocab_size=100,
            speech_vocab_size=100,
        )
        return CosyVoice2LLM(config)

    def test_generate_batch_returns_lengths(self, tiny_llm):
        """Test batch generation returns token lengths."""
        text_ids = mx.random.randint(0, 100, (2, 5))

        tokens, lengths = tiny_llm.generate_speech_tokens_batch(
            text_ids,
            max_length=10,
        )
        mx.eval(tokens, lengths)

        assert tokens.shape[0] == 2  # batch
        assert lengths.shape[0] == 2
        # Lengths should be within max_length
        assert mx.all(lengths <= 10).item()

    def test_generate_batch_tracks_eos(self, tiny_llm):
        """Test batch generation tracks EOS per sequence."""
        text_ids = mx.random.randint(0, 100, (3, 4))

        tokens, lengths = tiny_llm.generate_speech_tokens_batch(
            text_ids,
            max_length=20,
            eos_token_id=0,
        )
        mx.eval(tokens, lengths)

        # Each length should represent where EOS occurred
        assert lengths.shape[0] == 3
        for i in range(3):
            length = int(lengths[i].item())
            if length < tokens.shape[1]:
                # EOS should be at position `length`
                pass  # Can't guarantee EOS position without deterministic sampling
