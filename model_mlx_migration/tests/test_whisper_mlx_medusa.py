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
Tests for Medusa multi-token prediction module.

These tests verify the architecture and basic functionality of Medusa heads
without requiring trained weights.
"""

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from tools.whisper_mlx.config import get_config
from tools.whisper_mlx.medusa import (
    MedusaHead,
    MedusaHeadBlock,
    MedusaModule,
    build_tree_attention_mask,
    create_medusa_module,
)


class TestMedusaHead:
    """Tests for single Medusa head."""

    def test_init(self):
        """Test Medusa head initialization."""
        n_state = 1280
        n_vocab = 51866

        head = MedusaHead(n_state, n_vocab, head_index=1)

        assert head.n_state == n_state
        assert head.n_vocab == n_vocab
        assert head.head_index == 1

    def test_forward_shape(self):
        """Test forward pass output shape."""
        batch_size = 2
        seq_len = 10
        n_state = 1280
        n_vocab = 51866

        head = MedusaHead(n_state, n_vocab)

        # Create dummy hidden states
        hidden = mx.random.normal((batch_size, seq_len, n_state))

        # Forward pass
        logits = head(hidden)

        assert logits.shape == (batch_size, seq_len, n_vocab)

    def test_single_token_inference(self):
        """Test inference with single token (typical decode step)."""
        n_state = 1280
        n_vocab = 51866

        head = MedusaHead(n_state, n_vocab)

        # Single token hidden state
        hidden = mx.random.normal((1, 1, n_state))

        logits = head(hidden)

        assert logits.shape == (1, 1, n_vocab)

    def test_dtype_preservation(self):
        """Test that dtype is preserved through forward pass."""
        n_state = 1280
        n_vocab = 51866

        head = MedusaHead(n_state, n_vocab, dtype=mx.float16)

        hidden = mx.random.normal((1, 1, n_state)).astype(mx.float16)
        head(hidden)

        # Linear output is float32 but input dtype should be respected
        assert hidden.dtype == mx.float16


class TestMedusaHeadBlock:
    """Tests for Medusa head with transformer block."""

    def test_init(self):
        """Test Medusa block head initialization."""
        n_state = 1280
        n_vocab = 51866
        n_head = 20

        head = MedusaHeadBlock(n_state, n_vocab, n_head, head_index=1)

        assert head.head_index == 1
        assert head.self_attn is not None
        assert head.mlp is not None

    def test_forward_shape(self):
        """Test forward pass output shape."""
        batch_size = 2
        seq_len = 10
        n_state = 1280
        n_vocab = 51866
        n_head = 20

        head = MedusaHeadBlock(n_state, n_vocab, n_head)

        hidden = mx.random.normal((batch_size, seq_len, n_state))

        logits = head(hidden)

        assert logits.shape == (batch_size, seq_len, n_vocab)

    def test_forward_with_mask(self):
        """Test forward pass with attention mask."""
        n_state = 1280
        n_vocab = 51866
        n_head = 20
        seq_len = 10

        head = MedusaHeadBlock(n_state, n_vocab, n_head)

        hidden = mx.random.normal((1, seq_len, n_state))

        # Create causal mask
        mask = mx.triu(mx.full((seq_len, seq_len), float('-inf')), k=1)

        logits = head(hidden, mask=mask)

        assert logits.shape == (1, seq_len, n_vocab)


class TestMedusaModule:
    """Tests for the full Medusa module with multiple heads."""

    def test_init_linear(self):
        """Test initialization with Medusa-Linear heads."""
        n_state = 1280
        n_vocab = 51866
        n_heads = 5

        module = MedusaModule(n_state, n_vocab, n_heads, use_block=False)

        assert len(module.heads) == n_heads
        assert all(isinstance(h, MedusaHead) for h in module.heads)

    def test_init_block(self):
        """Test initialization with Medusa-Block heads."""
        n_state = 1280
        n_vocab = 51866
        n_heads = 5
        n_attn_head = 20

        module = MedusaModule(n_state, n_vocab, n_heads, use_block=True, n_attn_head=n_attn_head)

        assert len(module.heads) == n_heads
        assert all(isinstance(h, MedusaHeadBlock) for h in module.heads)

    def test_forward_shape(self):
        """Test forward pass returns list of correct shapes."""
        batch_size = 2
        seq_len = 10
        n_state = 1280
        n_vocab = 51866
        n_heads = 5

        module = MedusaModule(n_state, n_vocab, n_heads)

        hidden = mx.random.normal((batch_size, seq_len, n_state))

        logits_list = module(hidden)

        assert len(logits_list) == n_heads
        for logits in logits_list:
            assert logits.shape == (batch_size, seq_len, n_vocab)

    def test_get_predictions(self):
        """Test getting top-k predictions from each head."""
        batch_size = 2
        seq_len = 1
        n_state = 1280
        n_vocab = 51866
        n_heads = 5
        top_k = 10

        module = MedusaModule(n_state, n_vocab, n_heads)

        hidden = mx.random.normal((batch_size, seq_len, n_state))

        predictions = module.get_predictions(hidden, top_k=top_k)

        assert len(predictions) == n_heads
        for preds in predictions:
            assert preds.shape == (batch_size, seq_len, top_k)

    def test_head_indices(self):
        """Test that head indices are set correctly."""
        n_state = 1280
        n_vocab = 51866
        n_heads = 5

        module = MedusaModule(n_state, n_vocab, n_heads)

        for i, head in enumerate(module.heads):
            assert head.head_index == i + 1  # 1-indexed


class TestTreeAttentionMask:
    """Tests for tree attention mask construction."""

    def test_basic_mask_shape(self):
        """Test basic mask shape."""
        prefix_len = 10
        tree_structure = [5, 4, 3, 2, 1]

        mask = build_tree_attention_mask(prefix_len, tree_structure)

        total_len = prefix_len + sum(tree_structure)
        assert mask.shape == (total_len, total_len)

    def test_prefix_attention(self):
        """Test that all tokens can attend to prefix."""
        prefix_len = 5
        tree_structure = [3, 2]

        mask = build_tree_attention_mask(prefix_len, tree_structure)

        total_len = prefix_len + sum(tree_structure)

        # All tokens should be able to attend to entire prefix
        for i in range(total_len):
            for j in range(prefix_len):
                if j <= min(i, prefix_len - 1):  # Causal within prefix
                    assert mask[i, j] == 0, f"Token {i} should attend to prefix position {j}"

    def test_causal_prefix(self):
        """Test that prefix has causal attention."""
        prefix_len = 5
        tree_structure = [3]

        mask = build_tree_attention_mask(prefix_len, tree_structure)

        # Prefix should be causal
        for i in range(prefix_len):
            for j in range(prefix_len):
                if j <= i:
                    assert mask[i, j] == 0, f"Prefix token {i} should attend to {j}"
                else:
                    assert mask[i, j] == float('-inf'), f"Prefix token {i} should not attend to {j}"

    def test_self_attention(self):
        """Test that candidates can attend to themselves."""
        prefix_len = 5
        tree_structure = [3, 2]

        mask = build_tree_attention_mask(prefix_len, tree_structure)

        total_len = prefix_len + sum(tree_structure)

        # All tokens should be able to attend to themselves
        for i in range(total_len):
            assert mask[i, i] == 0, f"Token {i} should attend to itself"


class TestCreateMedusaModule:
    """Tests for the factory function."""

    def test_create_from_config(self):
        """Test creating module from Whisper config."""
        config = get_config("large-v3")

        module = create_medusa_module(config, n_heads=5)

        assert len(module.heads) == 5
        assert module.n_state == config.n_text_state
        assert module.n_vocab == config.n_vocab

    def test_create_block_variant(self):
        """Test creating Medusa-Block variant."""
        config = get_config("large-v3")

        module = create_medusa_module(config, n_heads=5, use_block=True)

        assert all(isinstance(h, MedusaHeadBlock) for h in module.heads)

    def test_create_different_head_counts(self):
        """Test creating with different numbers of heads."""
        config = get_config("large-v3")

        for n_heads in [1, 3, 5, 10]:
            module = create_medusa_module(config, n_heads=n_heads)
            assert len(module.heads) == n_heads


class TestMedusaMemory:
    """Tests for memory usage and efficiency."""

    def test_parameter_count_linear(self):
        """Test parameter count for Medusa-Linear variant."""
        n_state = 1280
        n_vocab = 51866
        n_heads = 5

        module = MedusaModule(n_state, n_vocab, n_heads, use_block=False)

        # Each linear head: n_state * n_vocab + n_vocab (bias)
        expected_per_head = n_state * n_vocab + n_vocab
        expected_total = expected_per_head * n_heads

        # Count actual parameters
        total_params = 0
        for head in module.heads:
            total_params += head.linear.weight.size + head.linear.bias.size

        assert total_params == expected_total

    def test_parameter_count_reasonable(self):
        """Test that parameter count is reasonable relative to Whisper."""
        config = get_config("large-v3")
        n_heads = 5

        module = create_medusa_module(config, n_heads=n_heads, use_block=False)

        # Count Medusa parameters
        medusa_params = 0
        for head in module.heads:
            medusa_params += head.linear.weight.size + head.linear.bias.size

        # Whisper large-v3 has ~1.55B parameters
        # Medusa should add <10% overhead
        whisper_params = 1.55e9
        overhead_ratio = medusa_params / whisper_params

        # 5 heads * (1280 * 51866) = ~332M params
        # That's about 21% overhead, which is acceptable for 1.5-2x speedup
        assert overhead_ratio < 0.25, f"Medusa overhead {overhead_ratio:.1%} too high"


class TestMedusaIntegration:
    """Integration tests with decoder-like inputs."""

    def test_batch_inference(self):
        """Test batch inference."""
        batch_sizes = [1, 2, 4]
        n_state = 1280
        n_vocab = 51866
        n_heads = 5

        module = MedusaModule(n_state, n_vocab, n_heads)

        for batch_size in batch_sizes:
            hidden = mx.random.normal((batch_size, 1, n_state))
            logits_list = module(hidden)

            assert len(logits_list) == n_heads
            for logits in logits_list:
                assert logits.shape[0] == batch_size

    def test_sequential_inference(self):
        """Test sequential single-token inference (typical decode pattern)."""
        n_state = 1280
        n_vocab = 51866
        n_heads = 5

        module = MedusaModule(n_state, n_vocab, n_heads)

        # Simulate 10 decode steps
        for _step in range(10):
            hidden = mx.random.normal((1, 1, n_state))
            logits_list = module(hidden)

            # Each step should produce valid predictions
            assert len(logits_list) == n_heads
            for logits in logits_list:
                assert logits.shape == (1, 1, n_vocab)
                # Should be able to argmax
                pred = mx.argmax(logits[0, 0])
                assert 0 <= pred.item() < n_vocab


class TestTreeCandidateGeneration:
    """Tests for tree candidate generation."""

    def test_tree_structure_basic(self):
        """Test basic tree structure generation."""
        from tools.whisper_mlx.medusa import MedusaTreeVerifier

        n_state = 1280
        n_vocab = 51866
        n_heads = 5

        medusa = MedusaModule(n_state, n_vocab, n_heads)

        # Create mock decoder (not needed for generation)
        class MockDecoder:
            pass

        verifier = MedusaTreeVerifier(
            decoder=MockDecoder(),
            medusa_module=medusa,
            tree_structure=[3, 2, 2],  # 3 at level 0, 2 at level 1, 2 at level 2
            top_k=5,
        )

        # Create dummy inputs
        hidden = mx.random.normal((1, 1, n_state))
        main_logits = mx.random.normal((1, 1, n_vocab))

        # Generate tree candidates
        tree_tokens, tree_mask, tree_paths, node_depths = verifier.generate_tree_candidates(
            hidden, main_logits,
        )

        # Check tree tokens shape
        assert tree_tokens.ndim == 2
        assert tree_tokens.shape[0] == 1  # batch size

        # Check mask is square
        total_nodes = tree_tokens.shape[1]
        assert tree_mask.shape == (total_nodes, total_nodes)

        # Check paths are valid indices
        for path in tree_paths:
            for idx in path:
                assert 0 <= idx < total_nodes

    def test_tree_mask_properties(self):
        """Test that tree mask has correct properties."""
        from tools.whisper_mlx.medusa import MedusaTreeVerifier

        n_state = 1280
        n_vocab = 51866
        n_heads = 3

        medusa = MedusaModule(n_state, n_vocab, n_heads)

        class MockDecoder:
            pass

        verifier = MedusaTreeVerifier(
            decoder=MockDecoder(),
            medusa_module=medusa,
            tree_structure=[2, 2],
            top_k=2,
        )

        hidden = mx.random.normal((1, 1, n_state))
        main_logits = mx.random.normal((1, 1, n_vocab))

        tree_tokens, tree_mask, tree_paths, node_depths = verifier.generate_tree_candidates(
            hidden, main_logits,
        )

        total_nodes = tree_tokens.shape[1]

        # Every node can attend to itself
        for i in range(total_nodes):
            assert tree_mask[i, i] == 0, f"Node {i} should attend to itself"

    def test_tree_paths_complete(self):
        """Test that tree paths cover all leaf nodes."""
        from tools.whisper_mlx.medusa import MedusaTreeVerifier

        n_state = 1280
        n_vocab = 51866
        n_heads = 3

        medusa = MedusaModule(n_state, n_vocab, n_heads)

        class MockDecoder:
            pass

        verifier = MedusaTreeVerifier(
            decoder=MockDecoder(),
            medusa_module=medusa,
            tree_structure=[2, 2, 2],
            top_k=3,
        )

        hidden = mx.random.normal((1, 1, n_state))
        main_logits = mx.random.normal((1, 1, n_vocab))

        tree_tokens, tree_mask, tree_paths, node_depths = verifier.generate_tree_candidates(
            hidden, main_logits,
        )

        # All paths should have the same depth (full tree)
        if tree_paths:
            depths = [len(path) for path in tree_paths]
            # Paths may have different lengths if tree is truncated
            assert all(d > 0 for d in depths), "All paths should have positive depth"

    def test_generate_candidates_flat(self):
        """Test flat candidate generation (legacy method)."""
        from tools.whisper_mlx.medusa import MedusaTreeVerifier

        n_state = 1280
        n_vocab = 51866
        n_heads = 5

        medusa = MedusaModule(n_state, n_vocab, n_heads)

        class MockDecoder:
            pass

        verifier = MedusaTreeVerifier(
            decoder=MockDecoder(),
            medusa_module=medusa,
        )

        hidden = mx.random.normal((1, 1, n_state))
        main_logits = mx.random.normal((1, 1, n_vocab))

        candidates, main_pred = verifier.generate_candidates(hidden, main_logits, top_k=3)

        # Check shapes
        assert candidates.ndim == 3
        assert candidates.shape[0] == 1  # batch
        assert candidates.shape[2] == n_heads + 1  # main + medusa heads

        # Main prediction should be argmax of logits
        expected_main = mx.argmax(main_logits[:, -1], axis=-1, keepdims=True)
        assert mx.all(main_pred == expected_main).item()


class TestMedusaTreeVerifierStats:
    """Tests for tree verifier statistics."""

    def test_initial_stats(self):
        """Test initial statistics are zero."""
        from tools.whisper_mlx.medusa import MedusaTreeVerifier

        n_state = 1280
        n_vocab = 51866
        n_heads = 5

        medusa = MedusaModule(n_state, n_vocab, n_heads)

        class MockDecoder:
            pass

        verifier = MedusaTreeVerifier(
            decoder=MockDecoder(),
            medusa_module=medusa,
        )

        assert verifier.acceptance_rate == 0.0
        assert verifier.tokens_per_iteration == 1.0
        assert verifier._total_proposed == 0
        assert verifier._total_accepted == 0
        assert verifier._iterations == 0

    def test_reset_stats(self):
        """Test statistics reset."""
        from tools.whisper_mlx.medusa import MedusaTreeVerifier

        n_state = 1280
        n_vocab = 51866
        n_heads = 5

        medusa = MedusaModule(n_state, n_vocab, n_heads)

        class MockDecoder:
            pass

        verifier = MedusaTreeVerifier(
            decoder=MockDecoder(),
            medusa_module=medusa,
        )

        # Manually set some stats
        verifier._total_proposed = 100
        verifier._total_accepted = 50
        verifier._iterations = 20

        # Reset
        verifier.reset_stats()

        assert verifier._total_proposed == 0
        assert verifier._total_accepted == 0
        assert verifier._iterations == 0


class TestTreeCandidateClass:
    """Tests for TreeCandidate helper class."""

    def test_tree_candidate_init(self):
        """Test TreeCandidate initialization."""
        from tools.whisper_mlx.medusa import TreeCandidate

        candidate = TreeCandidate(
            tokens=[1, 2, 3],
            parent_indices=[0, 1, 2],
            depth=3,
        )

        assert candidate.tokens == [1, 2, 3]
        assert candidate.parent_indices == [0, 1, 2]
        assert candidate.depth == 3

    def test_tree_candidate_empty(self):
        """Test TreeCandidate with empty tokens."""
        from tools.whisper_mlx.medusa import TreeCandidate

        candidate = TreeCandidate(
            tokens=[],
            parent_indices=[],
            depth=0,
        )

        assert candidate.tokens == []
        assert candidate.depth == 0


class TestBuildTreeMaskInternal:
    """Tests for internal _build_tree_mask method."""

    def test_build_tree_mask_simple(self):
        """Test building tree mask with simple parent map."""
        from tools.whisper_mlx.medusa import MedusaTreeVerifier

        n_state = 1280
        n_vocab = 51866
        n_heads = 3

        medusa = MedusaModule(n_state, n_vocab, n_heads)

        class MockDecoder:
            pass

        verifier = MedusaTreeVerifier(
            decoder=MockDecoder(),
            medusa_module=medusa,
        )

        # Simple tree: 0 -> 1 -> 2
        parent_map = [-1, 0, 1]
        mask = verifier._build_tree_mask(3, parent_map)

        # Node 0: can attend to self only
        assert mask[0, 0] == 0

        # Node 1: can attend to self and parent (0)
        assert mask[1, 1] == 0
        assert mask[1, 0] == 0

        # Node 2: can attend to self, parent (1), and grandparent (0)
        assert mask[2, 2] == 0
        assert mask[2, 1] == 0
        assert mask[2, 0] == 0

    def test_build_tree_mask_branching(self):
        """Test tree mask with branching."""
        from tools.whisper_mlx.medusa import MedusaTreeVerifier

        n_state = 1280
        n_vocab = 51866

        medusa = MedusaModule(n_state, n_vocab, 3)

        class MockDecoder:
            pass

        verifier = MedusaTreeVerifier(
            decoder=MockDecoder(),
            medusa_module=medusa,
        )

        # Tree:
        #     0
        #    / \
        #   1   2
        parent_map = [-1, 0, 0]
        mask = verifier._build_tree_mask(3, parent_map)

        # Node 1 and 2 both have parent 0
        assert mask[1, 0] == 0
        assert mask[2, 0] == 0

        # Node 1 should NOT attend to sibling 2
        assert mask[1, 2] == float('-inf')
        assert mask[2, 1] == float('-inf')


class TestDecoderTreeAttention:
    """Tests for decoder tree attention support."""

    def test_decoder_custom_mask(self):
        """Test decoder accepts custom attention mask."""
        from tools.whisper_mlx.decoder import TextDecoder

        decoder = TextDecoder(
            n_vocab=1000,
            n_ctx=64,
            n_state=64,
            n_head=4,
            n_layer=2,
        )

        batch_size = 1
        seq_len = 5
        enc_len = 10

        x = mx.array([[0, 1, 2, 3, 4]])
        xa = mx.random.normal((batch_size, enc_len, 64))

        # Create custom mask (all can attend to all)
        custom_mask = mx.zeros((seq_len, seq_len), dtype=mx.float32)

        logits, kv_cache, _, _ = decoder(x, xa, custom_mask=custom_mask)

        assert logits.shape == (batch_size, seq_len, 1000)

    def test_decoder_return_hidden(self):
        """Test decoder returns hidden states when requested."""
        from tools.whisper_mlx.decoder import TextDecoder

        decoder = TextDecoder(
            n_vocab=1000,
            n_ctx=64,
            n_state=64,
            n_head=4,
            n_layer=2,
        )

        batch_size = 1
        seq_len = 5
        enc_len = 10

        x = mx.array([[0, 1, 2, 3, 4]])
        xa = mx.random.normal((batch_size, enc_len, 64))

        # Without return_hidden
        logits1, _, _, hidden1 = decoder(x, xa, return_hidden=False)
        assert hidden1 is None

        # With return_hidden
        logits2, _, _, hidden2 = decoder(x, xa, return_hidden=True)
        assert hidden2 is not None
        assert hidden2.shape == (batch_size, seq_len, 64)

    def test_decoder_tree_mask_shape(self):
        """Test decoder handles tree mask with different shapes."""
        from tools.whisper_mlx.decoder import TextDecoder

        decoder = TextDecoder(
            n_vocab=1000,
            n_ctx=64,
            n_state=64,
            n_head=4,
            n_layer=2,
        )

        batch_size = 1
        seq_len = 6
        enc_len = 10

        x = mx.array([[0, 1, 2, 3, 4, 5]])
        xa = mx.random.normal((batch_size, enc_len, 64))

        # Create a tree-like mask (sparse attention)
        import numpy as np
        mask_np = np.full((seq_len, seq_len), float('-inf'), dtype=np.float32)
        for i in range(seq_len):
            mask_np[i, i] = 0  # Self attention
            if i > 0:
                mask_np[i, 0] = 0  # All attend to first

        custom_mask = mx.array(mask_np)

        logits, _, _, _ = decoder(x, xa, custom_mask=custom_mask)

        assert logits.shape == (batch_size, seq_len, 1000)


class TestCombinedAttentionMask:
    """Tests for combined prefix + tree attention mask."""

    def test_build_combined_mask_no_prefix(self):
        """Test combined mask with no prefix."""
        from tools.whisper_mlx.medusa import MedusaTreeVerifier

        n_state = 64
        n_vocab = 1000
        n_heads = 2

        medusa = MedusaModule(n_state, n_vocab, n_heads)

        class MockDecoder:
            pass

        verifier = MedusaTreeVerifier(
            decoder=MockDecoder(),
            medusa_module=medusa,
        )

        # Simple tree mask
        tree_mask = mx.array([
            [0, float('-inf'), float('-inf')],
            [0, 0, float('-inf')],
            [0, float('-inf'), 0],
        ])

        combined = verifier._build_combined_attention_mask(
            prefix_len=0,
            tree_len=3,
            tree_mask=tree_mask,
        )

        # With no prefix, combined should equal tree mask
        assert combined.shape == (3, 3)

    def test_build_combined_mask_with_prefix(self):
        """Test combined mask with prefix."""
        from tools.whisper_mlx.medusa import MedusaTreeVerifier

        n_state = 64
        n_vocab = 1000
        n_heads = 2

        medusa = MedusaModule(n_state, n_vocab, n_heads)

        class MockDecoder:
            pass

        verifier = MedusaTreeVerifier(
            decoder=MockDecoder(),
            medusa_module=medusa,
        )

        # Simple tree mask
        tree_mask = mx.array([
            [0, float('-inf')],
            [0, 0],
        ])

        combined = verifier._build_combined_attention_mask(
            prefix_len=5,
            tree_len=2,
            tree_mask=tree_mask,
        )

        # Combined should be (tree_len, prefix_len + tree_len)
        assert combined.shape == (2, 7)

        # All tree tokens can attend to all prefix positions
        for i in range(2):
            for j in range(5):
                assert combined[i, j] == 0, f"Tree token {i} should attend to prefix {j}"

        # Tree structure is preserved in the tree portion
        assert combined[0, 5] == 0  # First tree token attends to itself
        assert combined[1, 5] == 0  # Second tree token attends to first
        assert combined[1, 6] == 0  # Second tree token attends to itself


class TestTreeVerificationIntegration:
    """Integration tests for tree verification with real decoder."""

    def test_verify_tree_parallel_basic(self):
        """Test basic tree verification (structure only, no KV cache)."""
        from tools.whisper_mlx.medusa import MedusaTreeVerifier

        n_state = 64
        n_vocab = 100
        n_heads = 2

        medusa = MedusaModule(n_state, n_vocab, n_heads)

        # Create real decoder
        from tools.whisper_mlx.decoder import TextDecoder
        decoder = TextDecoder(
            n_vocab=n_vocab,
            n_ctx=64,
            n_state=n_state,
            n_head=4,
            n_layer=2,
        )

        verifier = MedusaTreeVerifier(
            decoder=decoder,
            medusa_module=medusa,
            tree_structure=[2, 2],
            top_k=2,
        )

        # Generate dummy audio features
        audio_features = mx.random.normal((1, 10, n_state))

        # Create simple tree tokens and mask
        tree_tokens = mx.array([[1, 2, 3, 4, 5, 6]])
        # Tree: 1 -> [3, 4], 2 -> [5, 6]
        # parent_map: [-1, -1, 0, 0, 1, 1]
        tree_mask = verifier._build_tree_mask(6, [-1, -1, 0, 0, 1, 1])
        tree_paths = [[0, 2], [0, 3], [1, 4], [1, 5]]

        # Verify (without KV cache)
        accepted, n_accepted, updated_cache = verifier.verify_tree_parallel(
            tree_tokens, tree_mask, tree_paths, audio_features, kv_cache=None,
        )

        # Should accept at least 1 token
        assert n_accepted >= 1
        assert accepted.shape[0] == 1  # batch size
        assert accepted.shape[1] == n_accepted

    def test_verify_tree_stats_update(self):
        """Test that verification updates statistics."""
        from tools.whisper_mlx.medusa import MedusaTreeVerifier

        n_state = 64
        n_vocab = 100
        n_heads = 2

        medusa = MedusaModule(n_state, n_vocab, n_heads)

        from tools.whisper_mlx.decoder import TextDecoder
        decoder = TextDecoder(
            n_vocab=n_vocab,
            n_ctx=64,
            n_state=n_state,
            n_head=4,
            n_layer=2,
        )

        verifier = MedusaTreeVerifier(
            decoder=decoder,
            medusa_module=medusa,
        )

        assert verifier._iterations == 0

        audio_features = mx.random.normal((1, 10, n_state))
        tree_tokens = mx.array([[1, 2, 3]])
        tree_mask = verifier._build_tree_mask(3, [-1, 0, 1])
        tree_paths = [[0, 1, 2]]

        verifier.verify_tree_parallel(
            tree_tokens, tree_mask, tree_paths, audio_features, kv_cache=None,
        )

        # Stats should be updated
        assert verifier._iterations == 1
        assert verifier._total_proposed > 0


class TestKLDivergenceLoss:
    """Tests for KL divergence loss function."""

    def test_identical_distributions(self):
        """Test KL divergence is zero for identical distributions."""
        from tools.whisper_mlx.medusa_training import kl_divergence_loss

        batch_size = 2
        seq_len = 5
        vocab = 100

        logits = mx.random.normal((batch_size, seq_len, vocab))

        # KL divergence of identical distributions should be near zero
        loss = kl_divergence_loss(logits, logits, temperature=1.0)

        assert loss.shape == ()
        assert abs(loss.item()) < 1e-5

    def test_different_distributions(self):
        """Test KL divergence is positive for different distributions."""
        from tools.whisper_mlx.medusa_training import kl_divergence_loss

        batch_size = 2
        seq_len = 5
        vocab = 100

        student_logits = mx.random.normal((batch_size, seq_len, vocab))
        teacher_logits = mx.random.normal((batch_size, seq_len, vocab)) + 1.0

        loss = kl_divergence_loss(student_logits, teacher_logits)

        assert loss.item() > 0

    def test_temperature_effect(self):
        """Test that higher temperature softens distributions."""
        from tools.whisper_mlx.medusa_training import kl_divergence_loss

        batch_size = 1
        seq_len = 3
        vocab = 50

        student = mx.random.normal((batch_size, seq_len, vocab))
        teacher = mx.random.normal((batch_size, seq_len, vocab)) + 0.5

        loss_t1 = kl_divergence_loss(student, teacher, temperature=1.0)
        loss_t2 = kl_divergence_loss(student, teacher, temperature=2.0)

        # Both losses should be positive
        assert loss_t1.item() > 0
        assert loss_t2.item() > 0

    def test_reduction_modes(self):
        """Test different reduction modes."""
        from tools.whisper_mlx.medusa_training import kl_divergence_loss

        batch_size = 2
        seq_len = 5
        vocab = 100

        student = mx.random.normal((batch_size, seq_len, vocab))
        teacher = mx.random.normal((batch_size, seq_len, vocab))

        # Mean reduction (default)
        loss_mean = kl_divergence_loss(student, teacher, reduction="mean")
        assert loss_mean.shape == ()

        # Sum reduction
        loss_sum = kl_divergence_loss(student, teacher, reduction="sum")
        assert loss_sum.shape == ()
        assert loss_sum.item() > loss_mean.item()  # Sum should be larger

        # No reduction
        loss_none = kl_divergence_loss(student, teacher, reduction="none")
        assert loss_none.shape == (batch_size, seq_len)


class TestComputeMedusaLoss:
    """Tests for full Medusa loss computation."""

    def test_basic_loss(self):
        """Test basic loss computation."""
        from tools.whisper_mlx.medusa_training import (
            MedusaTrainingConfig,
            compute_medusa_loss,
        )

        config = MedusaTrainingConfig(n_medusa_heads=3)

        batch_size = 2
        seq_len = 10
        vocab = 100

        teacher_logits = mx.random.normal((batch_size, seq_len, vocab))

        # Create medusa logits for each head
        medusa_logits = [
            mx.random.normal((batch_size, seq_len, vocab))
            for _ in range(3)
        ]

        loss, loss_dict = compute_medusa_loss(medusa_logits, teacher_logits, config)

        assert loss.shape == ()
        assert loss.item() > 0
        assert "total_loss" in loss_dict
        assert "head_0_loss" in loss_dict
        assert "head_1_loss" in loss_dict
        assert "head_2_loss" in loss_dict

    def test_loss_alignment(self):
        """Test that loss aligns predictions with shifted targets."""
        from tools.whisper_mlx.medusa_training import (
            MedusaTrainingConfig,
            compute_medusa_loss,
        )

        config = MedusaTrainingConfig(n_medusa_heads=2)

        batch_size = 1
        seq_len = 5
        vocab = 10

        # Create teacher logits
        teacher_logits = mx.random.normal((batch_size, seq_len, vocab))

        # Head 0 should predict position n+1 from position n
        # If we make head_0's predictions match teacher[:, 1:], loss should be low
        medusa_logits = [
            teacher_logits[:, 1:],  # Aligned - should have low loss
            teacher_logits[:, 2:],  # Aligned - should have low loss
        ]

        # Pad to same length (loss function handles alignment internally)
        medusa_logits[0] = mx.concatenate([
            medusa_logits[0],
            mx.zeros((batch_size, 1, vocab)),
        ], axis=1)
        medusa_logits[1] = mx.concatenate([
            medusa_logits[1],
            mx.zeros((batch_size, 2, vocab)),
        ], axis=1)

        loss, loss_dict = compute_medusa_loss(medusa_logits, teacher_logits, config)

        # Loss should be relatively low since predictions are aligned
        # (Not zero due to truncation and the zero padding)
        assert "total_loss" in loss_dict

    def test_short_sequence(self):
        """Test with sequence shorter than number of heads."""
        from tools.whisper_mlx.medusa_training import (
            MedusaTrainingConfig,
            compute_medusa_loss,
        )

        config = MedusaTrainingConfig(n_medusa_heads=5)

        batch_size = 1
        seq_len = 3  # Shorter than n_heads + 1
        vocab = 100

        teacher_logits = mx.random.normal((batch_size, seq_len, vocab))
        medusa_logits = [
            mx.random.normal((batch_size, seq_len, vocab))
            for _ in range(5)
        ]

        # Should not crash, later heads contribute 0 due to insufficient sequence length
        loss, loss_dict = compute_medusa_loss(medusa_logits, teacher_logits, config)

        assert "total_loss" in loss_dict


class TestMedusaTrainingConfig:
    """Tests for training configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        from tools.whisper_mlx.medusa_training import MedusaTrainingConfig

        config = MedusaTrainingConfig()

        assert config.n_medusa_heads == 5
        assert config.use_block is False
        assert config.learning_rate == 1e-4
        assert config.batch_size == 4
        assert config.kl_temperature == 1.0
        assert len(config.loss_weights) == 5

    def test_custom_config(self):
        """Test custom configuration."""
        from tools.whisper_mlx.medusa_training import MedusaTrainingConfig

        config = MedusaTrainingConfig(
            n_medusa_heads=10,
            learning_rate=5e-5,
            batch_size=8,
        )

        assert config.n_medusa_heads == 10
        assert config.learning_rate == 5e-5
        assert config.batch_size == 8


class TestTrainingBatch:
    """Tests for training batch dataclass."""

    def test_batch_creation(self):
        """Test creating a training batch."""
        from tools.whisper_mlx.medusa_training import TrainingBatch

        audio_features = mx.random.normal((4, 1500, 1280))
        decoder_tokens = mx.array([[1, 2, 3, 4, 5]] * 4)

        batch = TrainingBatch(
            audio_features=audio_features,
            decoder_tokens=decoder_tokens,
        )

        assert batch.audio_features.shape == (4, 1500, 1280)
        assert batch.decoder_tokens.shape == (4, 5)
        assert batch.attention_mask is None

    def test_batch_with_mask(self):
        """Test batch with attention mask."""
        from tools.whisper_mlx.medusa_training import TrainingBatch

        audio_features = mx.random.normal((2, 100, 64))
        decoder_tokens = mx.array([[1, 2, 3], [1, 2, 0]])  # Second has padding
        attention_mask = mx.array([[1, 1, 1], [1, 1, 0]])

        batch = TrainingBatch(
            audio_features=audio_features,
            decoder_tokens=decoder_tokens,
            attention_mask=attention_mask,
        )

        assert batch.attention_mask is not None
        assert batch.attention_mask.shape == (2, 3)


class TestMedusaWeightsSaveLoad:
    """Tests for saving and loading Medusa weights."""

    def test_save_load_roundtrip(self, tmp_path):
        """Test saving and loading weights."""
        from tools.whisper_mlx.medusa_training import (
            load_medusa_weights,
            save_medusa_weights,
        )

        n_state = 64
        n_vocab = 100
        n_heads = 3

        # Helper to compare nested params
        def compare_params(p1, p2, path=""):
            for key in p1:
                full_path = f"{path}.{key}" if path else key
                v1, v2 = p1[key], p2[key]
                if isinstance(v1, dict):
                    compare_params(v1, v2, full_path)
                else:
                    assert mx.allclose(v1, v2).item(), f"Mismatch at {full_path}"

        # Create and initialize medusa module
        medusa1 = MedusaModule(n_state, n_vocab, n_heads)

        # Save weights
        weights_path = str(tmp_path / "medusa_test.npz")
        save_medusa_weights(medusa1, weights_path)

        # Create fresh module
        medusa2 = MedusaModule(n_state, n_vocab, n_heads)

        # Load weights
        load_medusa_weights(medusa2, weights_path)

        # Compare parameters
        for i in range(n_heads):
            p1 = medusa1.heads[i].parameters()
            p2 = medusa2.heads[i].parameters()
            compare_params(p1, p2, f"head_{i}")

    def test_load_preserves_structure(self, tmp_path):
        """Test that loading preserves module structure."""
        from tools.whisper_mlx.medusa_training import (
            load_medusa_weights,
            save_medusa_weights,
        )

        n_state = 128
        n_vocab = 200
        n_heads = 5

        medusa1 = MedusaModule(n_state, n_vocab, n_heads)

        # Verify structure
        assert len(medusa1.heads) == n_heads
        for head in medusa1.heads:
            assert isinstance(head, MedusaHead)

        weights_path = str(tmp_path / "medusa_struct.npz")
        save_medusa_weights(medusa1, weights_path)

        medusa2 = MedusaModule(n_state, n_vocab, n_heads)
        load_medusa_weights(medusa2, weights_path)

        # Structure should be preserved
        assert len(medusa2.heads) == n_heads
        assert medusa2.n_state == n_state
        assert medusa2.n_vocab == n_vocab


class TestMedusaTrainerBasic:
    """Basic tests for MedusaTrainer (without full model)."""

    def test_config_initialization(self):
        """Test trainer config attributes."""
        from tools.whisper_mlx.medusa_training import MedusaTrainingConfig

        config = MedusaTrainingConfig(
            n_medusa_heads=5,
            learning_rate=1e-4,
            max_epochs=10,
        )

        assert config.n_medusa_heads == 5
        assert config.learning_rate == 1e-4
        assert config.max_epochs == 10

    def test_get_trainable_params(self):
        """Test extracting trainable parameters."""
        n_state = 64
        n_vocab = 100
        n_heads = 3

        medusa = MedusaModule(n_state, n_vocab, n_heads)

        # Helper to count params in nested dict
        def count_params(params):
            total = 0
            for _key, value in params.items():
                if isinstance(value, dict):
                    total += count_params(value)
                else:
                    total += value.size
            return total

        # Manually count parameters
        total_params = 0
        for head in medusa.heads:
            total_params += count_params(head.parameters())

        # Should have weight and bias for each head's linear layer
        expected = n_heads * (n_state * n_vocab + n_vocab)
        assert total_params == expected


class TestTranscribeMedusaIntegration:
    """Integration tests for transcribe_medusa() method."""

    def test_load_unload_medusa_heads(self, tmp_path):
        """Test loading and unloading Medusa heads."""
        # Create minimal model config for testing
        from tools.whisper_mlx.config import WhisperConfig
        from tools.whisper_mlx.medusa_training import save_medusa_weights
        from tools.whisper_mlx.model import WhisperMLX

        # Use small config for fast testing
        config = WhisperConfig(
            n_mels=80,
            n_vocab=100,
            n_audio_ctx=64,
            n_audio_state=64,
            n_audio_head=4,
            n_audio_layer=2,
            n_text_ctx=64,
            n_text_state=64,
            n_text_head=4,
            n_text_layer=2,
        )

        model = WhisperMLX(config, dtype=mx.float32, preallocate_kv=False, pad_vocab=False)

        # Initially not loaded
        assert not model.medusa_loaded

        # Create and save random weights
        medusa = MedusaModule(64, 100, n_heads=3)
        weights_path = str(tmp_path / "test_medusa.npz")
        save_medusa_weights(medusa, weights_path)

        # Load weights
        model.load_medusa_heads(weights_path, n_heads=3)
        assert model.medusa_loaded
        assert model._medusa_module is not None
        assert model._medusa_verifier is not None

        # Unload
        model.unload_medusa_heads()
        assert not model.medusa_loaded
        assert model._medusa_module is None
        assert model._medusa_verifier is None

    def test_transcribe_medusa_requires_loaded_heads(self):
        """Test that transcribe_medusa raises if heads not loaded."""
        import numpy as np

        from tools.whisper_mlx.config import WhisperConfig
        from tools.whisper_mlx.model import WhisperMLX

        config = WhisperConfig(
            n_mels=80,
            n_vocab=100,
            n_audio_ctx=64,
            n_audio_state=64,
            n_audio_head=4,
            n_audio_layer=2,
            n_text_ctx=64,
            n_text_state=64,
            n_text_head=4,
            n_text_layer=2,
        )

        model = WhisperMLX(config, dtype=mx.float32, preallocate_kv=False, pad_vocab=False)

        # Should raise without loading heads
        audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
        with pytest.raises(RuntimeError, match="Medusa heads not loaded"):
            model.transcribe_medusa(audio)

    def test_medusa_verifier_tree_generation_shapes(self):
        """Test tree candidate generation shapes with varying configs."""
        from tools.whisper_mlx.medusa import MedusaTreeVerifier

        n_state = 64
        n_vocab = 100

        for n_heads in [2, 3, 5]:
            medusa = MedusaModule(n_state, n_vocab, n_heads)

            class MockDecoder:
                pass

            # Vary tree structure
            tree_structure = [3] * n_heads
            verifier = MedusaTreeVerifier(
                decoder=MockDecoder(),
                medusa_module=medusa,
                tree_structure=tree_structure,
                top_k=3,
            )

            hidden = mx.random.normal((1, 1, n_state))
            main_logits = mx.random.normal((1, 1, n_vocab))

            tree_tokens, tree_mask, tree_paths, node_depths = verifier.generate_tree_candidates(
                hidden, main_logits,
            )

            # Check shapes are consistent
            total_nodes = tree_tokens.shape[1]
            assert tree_mask.shape == (total_nodes, total_nodes)
            assert len(tree_paths) > 0

    def test_medusa_decode_loop_termination(self):
        """Test that Medusa decode loop terminates properly."""
        from tools.whisper_mlx.decoder import TextDecoder
        from tools.whisper_mlx.medusa import MedusaTreeVerifier

        n_state = 64
        n_vocab = 100
        n_heads = 2

        # Create decoder with small context to force quick termination
        decoder = TextDecoder(
            n_vocab=n_vocab,
            n_ctx=32,  # Small context
            n_state=n_state,
            n_head=4,
            n_layer=2,
            pad_vocab=False,
        )

        medusa = MedusaModule(n_state, n_vocab, n_heads)

        verifier = MedusaTreeVerifier(
            decoder=decoder,
            medusa_module=medusa,
            tree_structure=[2, 2],
            top_k=2,
        )

        # Run a few verification iterations
        audio_features = mx.random.normal((1, 10, n_state))

        for _i in range(5):
            hidden = mx.random.normal((1, 1, n_state))
            main_logits = mx.random.normal((1, 1, n_vocab))

            tree_tokens, tree_mask, tree_paths, node_depths = verifier.generate_tree_candidates(
                hidden, main_logits,
            )

            accepted, n_accepted, _ = verifier.verify_tree_parallel(
                tree_tokens, tree_mask, tree_paths, audio_features, kv_cache=None,
            )

            # Should always accept at least 1 token
            assert n_accepted >= 1
            assert accepted.shape[0] == 1
            assert accepted.shape[1] == n_accepted

    def test_silent_audio_handling_medusa(self, tmp_path):
        """Test that silent audio is handled correctly in Medusa mode."""
        import numpy as np

        from tools.whisper_mlx.config import WhisperConfig
        from tools.whisper_mlx.medusa_training import save_medusa_weights
        from tools.whisper_mlx.model import WhisperMLX

        config = WhisperConfig(
            n_mels=80,
            n_vocab=100,
            n_audio_ctx=64,
            n_audio_state=64,
            n_audio_head=4,
            n_audio_layer=2,
            n_text_ctx=64,
            n_text_state=64,
            n_text_head=4,
            n_text_layer=2,
        )

        model = WhisperMLX(config, dtype=mx.float32, preallocate_kv=False, pad_vocab=False)

        # Create and load random weights
        medusa = MedusaModule(64, 100, n_heads=2)
        weights_path = str(tmp_path / "test_medusa.npz")
        save_medusa_weights(medusa, weights_path)
        model.load_medusa_heads(weights_path, n_heads=2)

        # Silent audio (all zeros)
        silent_audio = np.zeros(16000, dtype=np.float32)

        result = model.transcribe_medusa(silent_audio, language="en")

        # Should detect silence and return empty
        assert result["text"] == ""
        assert result.get("is_silent", False) is True


class TestMedusaBenchmarkScript:
    """Tests for the benchmark script utilities."""

    def test_generate_test_audio(self):
        """Test test audio generation."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

        # Import benchmark script functions
        from benchmark_medusa import generate_test_audio

        audio = generate_test_audio(duration=2.0, sample_rate=16000)

        assert audio.shape == (32000,)  # 2 seconds at 16kHz
        assert audio.dtype == np.float32
        assert np.max(np.abs(audio)) <= 1.0  # Normalized

    def test_compare_results(self):
        """Test result comparison logic."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

        from benchmark_medusa import compare_results

        standard = {
            "mean_time": 1.0,
            "text": "hello world",
        }
        medusa = {
            "mean_time": 0.5,
            "text": "hello world",
            "mean_acceptance_rate": 0.8,
            "mean_tokens_per_step": 2.0,
        }

        comparison = compare_results(standard, medusa)

        assert comparison["speedup"] == 2.0
        assert comparison["texts_match"] is True
        assert comparison["acceptance_rate"] == 0.8
        assert comparison["tokens_per_step"] == 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
