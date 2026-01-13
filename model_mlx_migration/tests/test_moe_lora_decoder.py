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
Tests for MoE-LoRA Decoder (Phase 10.5).

Tests:
1. ExpertRouter - speaker embedding to expert weights
2. LoRAExpert - single expert LoRA computation
3. MoELoRADecoder - full decoder with routing
4. Parameter counting and state dict
5. Expert specialization (weights differentiate by speaker)
"""

import mlx.core as mx
import mlx.nn as nn
import pytest


class MockWhisperDecoder(nn.Module):
    """Mock Whisper decoder for testing."""

    def __init__(self, n_state: int = 1280, n_layer: int = 32, vocab_size: int = 51865):
        super().__init__()
        self.n_state = n_state
        self.n_layer = n_layer
        self.vocab_size = vocab_size

        # Simple projection for testing
        self.proj = nn.Linear(n_state, vocab_size)

    def __call__(self, tokens: mx.array, encoder_out: mx.array) -> mx.array:
        """Mock forward: tokens -> logits."""
        batch_size, seq_len = tokens.shape
        # Simple mock: project encoder output mean to vocab
        enc_mean = encoder_out.mean(axis=1, keepdims=True)  # [B, 1, D]
        enc_broadcast = mx.broadcast_to(enc_mean, (batch_size, seq_len, self.n_state))
        return self.proj(enc_broadcast)  # [B, T, vocab_size]


class TestExpertRouter:
    """Tests for ExpertRouter."""

    def test_router_output_shape(self):
        """Router outputs correct shape."""
        from tools.whisper_mlx.sota.moe_lora_decoder import ExpertRouter

        router = ExpertRouter(speaker_dim=192, n_experts=4, hidden_dim=64)

        # Single embedding
        emb = mx.random.normal((192,))
        weights = router(emb)
        assert weights.shape == (1, 4), f"Expected (1, 4), got {weights.shape}"

        # Batch of embeddings
        emb_batch = mx.random.normal((8, 192))
        weights_batch = router(emb_batch)
        assert weights_batch.shape == (8, 4), f"Expected (8, 4), got {weights_batch.shape}"

    def test_router_weights_sum_to_one(self):
        """Router outputs sum to 1 (softmax property)."""
        from tools.whisper_mlx.sota.moe_lora_decoder import ExpertRouter

        router = ExpertRouter(speaker_dim=192, n_experts=4)
        emb = mx.random.normal((4, 192))
        weights = router(emb)

        # Sum across experts should be 1 for each sample
        sums = weights.sum(axis=1)
        mx.eval(sums)

        for i, s in enumerate(sums.tolist()):
            assert abs(s - 1.0) < 1e-5, f"Sample {i} weights sum to {s}, expected 1.0"

    def test_router_different_embeddings_different_weights(self):
        """Different speaker embeddings produce different expert weights."""
        from tools.whisper_mlx.sota.moe_lora_decoder import ExpertRouter

        router = ExpertRouter(speaker_dim=192, n_experts=4)

        # Create two distinct embeddings
        emb1 = mx.random.normal((192,)) * 0.5
        emb2 = mx.random.normal((192,)) * 2.0

        weights1 = router(emb1)
        weights2 = router(emb2)

        mx.eval(weights1, weights2)

        # Weights should be different
        diff = mx.abs(weights1 - weights2).max().item()
        assert diff > 0.01, f"Weights too similar: max diff = {diff}"


class TestLoRAExpert:
    """Tests for LoRAExpert."""

    def test_expert_delta_shape(self):
        """Expert produces correct delta shapes."""
        from tools.whisper_mlx.sota.moe_lora_decoder import LoRAExpert

        expert = LoRAExpert(
            name="test_expert",
            n_state=1280,
            rank=8,
            n_layers=12,
        )

        hidden = mx.random.normal((2, 100, 1280))  # [B, T, D]

        q_delta, v_delta = expert.compute_delta(hidden, layer_idx=0)

        assert q_delta.shape == (2, 100, 1280), f"Q delta shape: {q_delta.shape}"
        assert v_delta.shape == (2, 100, 1280), f"V delta shape: {v_delta.shape}"

    def test_expert_param_count(self):
        """Expert parameter count is correct."""
        from tools.whisper_mlx.sota.moe_lora_decoder import LoRAExpert

        n_state = 1280
        rank = 8
        n_layers = 12

        expert = LoRAExpert(
            name="test",
            n_state=n_state,
            rank=rank,
            n_layers=n_layers,
        )

        # Each LoRA: A (n_state, rank) + B (rank, n_state) = 2 * n_state * rank
        # Q and V per layer = 2 * 2 * n_state * rank
        # Total = n_layers * 2 * 2 * n_state * rank
        expected = n_layers * 2 * 2 * n_state * rank
        actual = expert.param_count()

        assert actual == expected, f"Expected {expected} params, got {actual}"

    def test_expert_layer_idx_bounds(self):
        """Expert raises error for out-of-bounds layer index."""
        from tools.whisper_mlx.sota.moe_lora_decoder import LoRAExpert

        expert = LoRAExpert(name="test", n_state=1280, rank=8, n_layers=12)
        hidden = mx.random.normal((2, 100, 1280))

        with pytest.raises(ValueError, match="layer_idx"):
            expert.compute_delta(hidden, layer_idx=12)  # Out of bounds

    def test_expert_different_layers_different_deltas(self):
        """Different layers have different LoRA weights after initialization."""
        from tools.whisper_mlx.sota.moe_lora_decoder import LoRAExpert

        expert = LoRAExpert(name="test", n_state=1280, rank=8, n_layers=12)

        # Note: LoRA B matrices are initialized to zero, so initial deltas are zero
        # We verify that the A matrices (which determine delta after training) are different
        a0 = expert.q_loras[0].lora_A.weight
        a1 = expert.q_loras[1].lora_A.weight

        mx.eval(a0, a1)

        # Different layers have different A matrices (random init)
        diff = mx.abs(a0 - a1).max().item()
        assert diff > 0.01, f"Layer A matrices too similar: max diff = {diff}"


class TestMoELoRADecoder:
    """Tests for MoELoRADecoder."""

    def test_decoder_output_shape(self):
        """MoE-LoRA decoder produces correct output shape."""
        from tools.whisper_mlx.sota.moe_lora_decoder import MoELoRADecoder

        mock_decoder = MockWhisperDecoder(n_state=1280, vocab_size=51865)
        moe_decoder = MoELoRADecoder(
            whisper_decoder=mock_decoder,
            n_experts=4,
            lora_rank=8,
            speaker_dim=192,
        )

        encoder_out = mx.random.normal((2, 100, 1280))
        speaker_emb = mx.random.normal((2, 192))
        tokens = mx.zeros((2, 50), dtype=mx.int32)

        output = moe_decoder(encoder_out, speaker_emb, tokens)

        assert output.shape == (2, 50, 51865), f"Output shape: {output.shape}"

    def test_decoder_aux_loss(self):
        """MoE-LoRA decoder returns auxiliary load balancing loss."""
        from tools.whisper_mlx.sota.moe_lora_decoder import MoELoRADecoder

        mock_decoder = MockWhisperDecoder()
        moe_decoder = MoELoRADecoder(
            whisper_decoder=mock_decoder,
            n_experts=4,
            lora_rank=8,
        )

        encoder_out = mx.random.normal((2, 100, 1280))
        speaker_emb = mx.random.normal((2, 192))
        tokens = mx.zeros((2, 50), dtype=mx.int32)

        output, aux_loss = moe_decoder(encoder_out, speaker_emb, tokens, return_aux_loss=True)

        mx.eval(aux_loss)
        assert aux_loss.shape == (), f"Aux loss should be scalar, got shape {aux_loss.shape}"
        assert aux_loss.item() >= 0, f"Aux loss should be non-negative: {aux_loss.item()}"

    def test_decoder_different_speakers_different_routing(self):
        """Different speakers get routed to different expert weights."""
        from tools.whisper_mlx.sota.moe_lora_decoder import MoELoRADecoder

        mock_decoder = MockWhisperDecoder()
        moe_decoder = MoELoRADecoder(
            whisper_decoder=mock_decoder,
            n_experts=4,
            lora_rank=8,
        )

        encoder_out = mx.random.normal((1, 50, 1280))
        tokens = mx.zeros((1, 20), dtype=mx.int32)

        # Two different speakers
        speaker1 = mx.random.normal((1, 192)) * 0.5
        speaker2 = mx.random.normal((1, 192)) * 2.0

        # Forward pass to trigger routing
        _ = moe_decoder(encoder_out, speaker1, tokens)
        usage1 = moe_decoder.get_expert_usage()

        _ = moe_decoder(encoder_out, speaker2, tokens)
        usage2 = moe_decoder.get_expert_usage()

        mx.eval(usage1, usage2)

        # Router should produce different expert weights for different speakers
        max_diff = mx.abs(usage1 - usage2).max().item()
        assert max_diff > 0.01, f"Expert routing too similar for different speakers: {max_diff}"

    def test_decoder_output_changes_with_trained_lora(self):
        """After modifying LoRA weights, outputs change with speaker."""
        from tools.whisper_mlx.sota.moe_lora_decoder import MoELoRADecoder

        mock_decoder = MockWhisperDecoder()
        moe_decoder = MoELoRADecoder(
            whisper_decoder=mock_decoder,
            n_experts=4,
            lora_rank=8,
        )

        # Set non-zero B matrices in all experts to simulate training
        for expert in moe_decoder.experts:
            for q_lora in expert.q_loras:
                q_lora.lora_B.weight = mx.random.normal(
                    q_lora.lora_B.weight.shape,
                ) * 0.5
            for v_lora in expert.v_loras:
                v_lora.lora_B.weight = mx.random.normal(
                    v_lora.lora_B.weight.shape,
                ) * 0.5

        # Also set output proj to non-trivial values
        moe_decoder.output_proj.weight = mx.random.normal(
            moe_decoder.output_proj.weight.shape,
        ) * 0.1

        encoder_out = mx.random.normal((1, 50, 1280))
        tokens = mx.zeros((1, 20), dtype=mx.int32)

        # Two different speakers
        speaker1 = mx.random.normal((1, 192)) * 0.5
        speaker2 = mx.random.normal((1, 192)) * 2.0

        out1 = moe_decoder(encoder_out, speaker1, tokens)
        out2 = moe_decoder(encoder_out, speaker2, tokens)

        mx.eval(out1, out2)

        max_diff = mx.abs(out1 - out2).max().item()
        # After training (non-zero LoRA B), different speakers should produce different outputs
        assert max_diff > 0.001, f"Outputs too similar after training: {max_diff}"

    def test_decoder_trainable_params(self):
        """MoE-LoRA decoder returns correct trainable parameters."""
        from tools.whisper_mlx.sota.moe_lora_decoder import MoELoRADecoder

        mock_decoder = MockWhisperDecoder()
        moe_decoder = MoELoRADecoder(
            whisper_decoder=mock_decoder,
            n_experts=4,
            lora_rank=8,
        )

        params = moe_decoder.trainable_parameters()

        # Should have router params
        assert "router.fc1.weight" in params
        assert "router.fc2.weight" in params

        # Should have output projection params
        assert "output_proj.weight" in params
        assert "output_proj.bias" in params

        # Should have expert LoRA params
        assert "expert_0.layer_0.q_lora.A" in params
        assert "expert_0.layer_0.v_lora.B" in params
        assert "expert_3.layer_11.q_lora.A" in params  # Last expert, last layer

    def test_decoder_total_params_reasonable(self):
        """MoE-LoRA decoder has reasonable parameter count."""
        from tools.whisper_mlx.sota.moe_lora_decoder import MoELoRADecoder

        mock_decoder = MockWhisperDecoder()
        moe_decoder = MoELoRADecoder(
            whisper_decoder=mock_decoder,
            n_experts=4,
            lora_rank=8,
        )

        total = moe_decoder.total_params()

        # Parameter breakdown:
        # - Router: 192*64+64 + 64*4+4 = ~12K
        # - Output proj: 1280*51865 + 51865 = ~66M
        # - 4 experts * 12 layers * 2 loras * 2 matrices * 1280 * 8 = ~3.1M
        # Total: ~70M params
        # Output proj dominates but is necessary for logit space adaptation
        assert 60_000_000 < total < 80_000_000, f"Unexpected param count: {total}"
        print(f"Total trainable params: {total:,}")

    def test_decoder_state_dict_roundtrip(self):
        """State dict save/load preserves parameters."""
        from tools.whisper_mlx.sota.moe_lora_decoder import MoELoRADecoder

        mock_decoder = MockWhisperDecoder()
        moe_decoder1 = MoELoRADecoder(
            whisper_decoder=mock_decoder,
            n_experts=4,
            lora_rank=8,
        )

        # Get state dict
        state = moe_decoder1.state_dict()

        # Create new decoder
        mock_decoder2 = MockWhisperDecoder()
        moe_decoder2 = MoELoRADecoder(
            whisper_decoder=mock_decoder2,
            n_experts=4,
            lora_rank=8,
        )

        # Load state
        moe_decoder2.load_state_dict(state)

        # Compare parameters
        params1 = moe_decoder1.trainable_parameters()
        params2 = moe_decoder2.trainable_parameters()

        for key in params1:
            assert key in params2, f"Missing key: {key}"
            diff = mx.abs(params1[key] - params2[key]).max().item()
            assert diff < 1e-6, f"Parameter {key} differs: max diff = {diff}"

    def test_decoder_expert_usage_tracking(self):
        """Expert usage is tracked after forward pass."""
        from tools.whisper_mlx.sota.moe_lora_decoder import MoELoRADecoder

        mock_decoder = MockWhisperDecoder()
        moe_decoder = MoELoRADecoder(
            whisper_decoder=mock_decoder,
            n_experts=4,
            lora_rank=8,
        )

        encoder_out = mx.random.normal((4, 50, 1280))
        speaker_emb = mx.random.normal((4, 192))
        tokens = mx.zeros((4, 20), dtype=mx.int32)

        _ = moe_decoder(encoder_out, speaker_emb, tokens)

        usage = moe_decoder.get_expert_usage()
        assert usage is not None
        assert usage.shape == (4,), f"Usage shape: {usage.shape}"

        mx.eval(usage)
        total_usage = usage.sum().item()
        assert abs(total_usage - 1.0) < 1e-5, f"Usage should sum to 1: {total_usage}"


class TestMoELoRAConfig:
    """Tests for MoELoRAConfig."""

    def test_config_defaults(self):
        """Config has sensible defaults."""
        from tools.whisper_mlx.sota.moe_lora_decoder import MoELoRAConfig

        config = MoELoRAConfig()

        assert config.n_experts == 4
        assert config.lora_rank == 8
        assert config.speaker_dim == 192
        assert config.n_state == 1280
        assert config.adapt_start_layer == 20

    def test_config_custom_values(self):
        """Config accepts custom values."""
        from tools.whisper_mlx.sota.moe_lora_decoder import MoELoRAConfig

        config = MoELoRAConfig(
            n_experts=8,
            lora_rank=16,
            speaker_dim=256,
        )

        assert config.n_experts == 8
        assert config.lora_rank == 16
        assert config.speaker_dim == 256


class TestCreateHelper:
    """Tests for create_moe_lora_decoder helper."""

    def test_create_helper(self):
        """Helper creates valid decoder."""
        from tools.whisper_mlx.sota.moe_lora_decoder import create_moe_lora_decoder

        mock_decoder = MockWhisperDecoder()
        moe_decoder = create_moe_lora_decoder(
            whisper_decoder=mock_decoder,
            n_experts=4,
            lora_rank=8,
        )

        assert moe_decoder.n_experts == 4
        assert moe_decoder.lora_rank == 8
        assert len(moe_decoder.experts) == 4


class TestIntegration:
    """Integration tests with actual shapes."""

    def test_full_forward_pass(self):
        """Full forward pass with realistic shapes."""
        from tools.whisper_mlx.sota.moe_lora_decoder import MoELoRADecoder

        # Realistic shapes
        batch_size = 4
        encoder_len = 1500  # ~30s audio
        decoder_len = 448   # Whisper max tokens
        n_state = 1280
        vocab_size = 51865

        mock_decoder = MockWhisperDecoder(n_state=n_state, vocab_size=vocab_size)
        moe_decoder = MoELoRADecoder(
            whisper_decoder=mock_decoder,
            n_experts=4,
            lora_rank=8,
        )

        encoder_out = mx.random.normal((batch_size, encoder_len, n_state))
        speaker_emb = mx.random.normal((batch_size, 192))
        tokens = mx.zeros((batch_size, decoder_len), dtype=mx.int32)

        output, aux_loss = moe_decoder(
            encoder_out, speaker_emb, tokens, return_aux_loss=True,
        )

        mx.eval(output, aux_loss)

        assert output.shape == (batch_size, decoder_len, vocab_size)
        print(f"Output shape: {output.shape}")
        print(f"Aux loss: {aux_loss.item():.6f}")


if __name__ == "__main__":
    # Quick smoke test
    print("Running MoE-LoRA decoder tests...")

    # Test router
    test_router = TestExpertRouter()
    test_router.test_router_output_shape()
    test_router.test_router_weights_sum_to_one()
    test_router.test_router_different_embeddings_different_weights()
    print("Router tests passed")

    # Test expert
    test_expert = TestLoRAExpert()
    test_expert.test_expert_delta_shape()
    test_expert.test_expert_param_count()
    test_expert.test_expert_different_layers_different_deltas()
    print("Expert tests passed")

    # Test decoder
    test_decoder = TestMoELoRADecoder()
    test_decoder.test_decoder_output_shape()
    test_decoder.test_decoder_aux_loss()
    test_decoder.test_decoder_different_speakers_different_routing()
    test_decoder.test_decoder_output_changes_with_trained_lora()
    test_decoder.test_decoder_trainable_params()
    test_decoder.test_decoder_total_params_reasonable()
    test_decoder.test_decoder_state_dict_roundtrip()
    test_decoder.test_decoder_expert_usage_tracking()
    print("Decoder tests passed")

    # Integration
    test_int = TestIntegration()
    test_int.test_full_forward_pass()
    print("Integration tests passed")

    print("\nAll tests passed!")
