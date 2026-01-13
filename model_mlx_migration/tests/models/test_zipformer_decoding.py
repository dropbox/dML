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
Tests for Zipformer RNN-T decoding algorithms.
"""

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from models.zipformer.decoder import Decoder, DecoderConfig
from models.zipformer.decoding import (
    DecodingResult,
    greedy_search,
    greedy_search_batch,
    modified_beam_search,
)
from models.zipformer.joiner import Joiner, JoinerConfig


class TestGreedySearch:
    """Tests for greedy search decoding."""

    @pytest.fixture
    def decoder(self):
        config = DecoderConfig(
            vocab_size=500,
            decoder_dim=512,
            blank_id=0,
            context_size=2,
        )
        return Decoder(config)

    @pytest.fixture
    def joiner(self):
        config = JoinerConfig(
            encoder_dim=256,
            decoder_dim=512,
            joiner_dim=512,
            vocab_size=500,
        )
        return Joiner(config)

    def test_greedy_search_basic(self, decoder, joiner):
        """Test basic greedy search execution."""
        # Random encoder output (10 frames)
        encoder_out = mx.random.normal((10, 256))

        result = greedy_search(
            decoder=decoder,
            joiner=joiner,
            encoder_out=encoder_out,
            max_sym_per_frame=1,
        )

        assert isinstance(result, DecodingResult)
        assert isinstance(result.tokens, list)
        # Result should have some tokens (with random weights, exact count varies)
        # All tokens should be non-blank (0)
        for token in result.tokens:
            assert token != 0
            assert 0 <= token < 500

    def test_greedy_search_with_length(self, decoder, joiner):
        """Test greedy search with explicit length."""
        encoder_out = mx.random.normal((20, 256))

        # Only use first 10 frames
        result = greedy_search(
            decoder=decoder,
            joiner=joiner,
            encoder_out=encoder_out,
            encoder_out_len=10,
            max_sym_per_frame=1,
        )

        assert isinstance(result, DecodingResult)

    def test_greedy_search_max_sym_per_frame(self, decoder, joiner):
        """Test that max_sym_per_frame limits symbols."""
        encoder_out = mx.random.normal((5, 256))

        result = greedy_search(
            decoder=decoder,
            joiner=joiner,
            encoder_out=encoder_out,
            max_sym_per_frame=3,
        )

        # With 5 frames and max 3 symbols per frame, max possible = 15
        assert len(result.tokens) <= 15

    def test_greedy_search_batch(self, decoder, joiner):
        """Test batched greedy search."""
        batch_size = 3
        max_T = 10
        encoder_out = mx.random.normal((batch_size, max_T, 256))
        encoder_out_lens = mx.array([10, 8, 5])

        results = greedy_search_batch(
            decoder=decoder,
            joiner=joiner,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
        )

        assert len(results) == batch_size
        for result in results:
            assert isinstance(result, DecodingResult)


class TestBeamSearch:
    """Tests for beam search decoding."""

    @pytest.fixture
    def decoder(self):
        config = DecoderConfig(
            vocab_size=500,
            decoder_dim=512,
            blank_id=0,
            context_size=2,
        )
        return Decoder(config)

    @pytest.fixture
    def joiner(self):
        config = JoinerConfig(
            encoder_dim=256,
            decoder_dim=512,
            joiner_dim=512,
            vocab_size=500,
        )
        return Joiner(config)

    def test_beam_search_basic(self, decoder, joiner):
        """Test basic beam search execution."""
        encoder_out = mx.random.normal((10, 256))

        result = modified_beam_search(
            decoder=decoder,
            joiner=joiner,
            encoder_out=encoder_out,
            beam_size=4,
            max_sym_per_frame=2,
        )

        assert isinstance(result, DecodingResult)
        assert isinstance(result.tokens, list)

    def test_beam_search_larger_beam(self, decoder, joiner):
        """Test beam search with larger beam size."""
        encoder_out = mx.random.normal((5, 256))

        result = modified_beam_search(
            decoder=decoder,
            joiner=joiner,
            encoder_out=encoder_out,
            beam_size=8,
        )

        assert isinstance(result, DecodingResult)


class TestPretrainedDecoding:
    """Tests with pretrained model weights."""

    @pytest.fixture
    def checkpoint_path(self):
        return Path('checkpoints/zipformer/en-streaming/exp/pretrained.pt')

    @pytest.mark.skipif(
        not Path('checkpoints/zipformer/en-streaming/exp/pretrained.pt').exists(),
        reason="Pretrained checkpoint not available",
    )
    def test_greedy_search_with_pretrained(self, checkpoint_path):
        """Test greedy search with pretrained weights produces valid output."""
        import torch

        # Load checkpoint
        ckpt = torch.load(str(checkpoint_path), map_location='cpu')
        model_dict = ckpt.get('model', ckpt)

        # Create and load decoder
        decoder_config = DecoderConfig(
            vocab_size=500,
            decoder_dim=512,
            blank_id=0,
            context_size=2,
        )
        decoder = Decoder(decoder_config)
        decoder.embedding.weight = mx.array(
            model_dict['decoder.embedding.weight'].numpy(),
        )
        conv_weight = model_dict['decoder.conv.weight'].numpy()
        decoder.conv.weight = mx.array(np.transpose(conv_weight, (0, 2, 1)))

        # Create and load joiner - get encoder_dim from checkpoint
        encoder_dim = model_dict['joiner.encoder_proj.weight'].shape[1]  # 512
        joiner_config = JoinerConfig(
            encoder_dim=encoder_dim,
            decoder_dim=512,
            joiner_dim=512,
            vocab_size=500,
        )
        joiner = Joiner(joiner_config)
        joiner.encoder_proj.weight = mx.array(
            model_dict['joiner.encoder_proj.weight'].numpy(),
        )
        joiner.encoder_proj.bias = mx.array(
            model_dict['joiner.encoder_proj.bias'].numpy(),
        )
        joiner.decoder_proj.weight = mx.array(
            model_dict['joiner.decoder_proj.weight'].numpy(),
        )
        joiner.decoder_proj.bias = mx.array(
            model_dict['joiner.decoder_proj.bias'].numpy(),
        )
        joiner.output_linear.weight = mx.array(
            model_dict['joiner.output_linear.weight'].numpy(),
        )
        joiner.output_linear.bias = mx.array(
            model_dict['joiner.output_linear.bias'].numpy(),
        )

        # Create fake encoder output (would come from actual encoder)
        # This should produce some tokens with pretrained weights
        encoder_out = mx.random.normal((50, encoder_dim)) * 0.1

        result = greedy_search(
            decoder=decoder,
            joiner=joiner,
            encoder_out=encoder_out,
        )

        print(f"Decoded tokens: {result.tokens}")
        print(f"Score: {result.score}")

        assert isinstance(result, DecodingResult)
        # With pretrained weights and random encoder output,
        # we expect some tokens to be emitted
        # (exact count depends on the joiner's output distribution)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
