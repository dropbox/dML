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
Medusa multi-token prediction for WhisperMLX.

Medusa adds extra "heads" to the decoder that predict multiple future tokens
simultaneously. This enables speculative decoding with self-generated drafts,
achieving 1.5-2.5x lossless speedup.

Architecture:
- Main decoder output: predicts token at position n
- Medusa head 1: predicts token at position n+1
- Medusa head 2: predicts token at position n+2
- ... etc.

The algorithm:
1. Generate candidates using all heads
2. Construct a "tree" of possible continuations
3. Verify all candidates in parallel using tree attention
4. Accept the longest valid prefix

References:
- Medusa paper: https://arxiv.org/abs/2401.10774
- Whisper-Medusa: https://github.com/aiola-lab/whisper-medusa
"""


import mlx.core as mx
import mlx.nn as nn


class MedusaResBlock(nn.Module):
    """
    Medusa residual block matching aiola/whisper-medusa architecture.

    Architecture:
    - Linear transformation: hidden_dim -> hidden_dim
    - SiLU activation
    - Residual connection (add input)

    This transforms hidden states while preserving information through
    the residual connection. The output must be passed through a shared
    proj_out layer for final vocabulary projection.

    Args:
        n_state: Hidden dimension (e.g., 1280 for large-v2/v3)
        head_index: Which future position this head predicts (1, 2, 3, ...)
    """

    def __init__(
        self,
        n_state: int,
        head_index: int = 1,
        dtype: mx.Dtype = mx.float16,
    ):
        super().__init__()
        self.head_index = head_index
        self.n_state = n_state

        # Linear transformation (hidden_dim -> hidden_dim)
        self.linear = nn.Linear(n_state, n_state)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """
        Forward pass through Medusa residual block.

        Args:
            hidden_states: Decoder hidden states
                          Shape: (batch, seq_len, n_state)

        Returns:
            Transformed hidden states (same shape)
            Shape: (batch, seq_len, n_state)

        Note:
            Output must be passed through proj_out for vocabulary logits.
        """
        # Linear transformation with SiLU activation
        transformed = self.linear(hidden_states)
        transformed = nn.silu(transformed)

        # Residual connection
        return hidden_states + transformed


class MedusaHead(nn.Module):
    """
    Single Medusa head for predicting one additional future token.

    Architecture (Medusa-Linear variant):
    - Residual connection from decoder hidden states
    - Single linear projection to vocabulary

    This is the simpler variant. A more complex Medusa-Block variant
    could use a shared transformer block before the linear projection.

    Args:
        n_state: Hidden dimension from decoder (e.g., 1280 for large-v3)
        n_vocab: Vocabulary size (e.g., 51866 for large-v3)
        head_index: Which future position this head predicts (1, 2, 3, ...)
    """

    def __init__(
        self,
        n_state: int,
        n_vocab: int,
        head_index: int = 1,
        dtype: mx.Dtype = mx.float16,
    ):
        super().__init__()
        self.head_index = head_index
        self.n_state = n_state
        self.n_vocab = n_vocab

        # Linear projection to vocabulary
        # Uses same hidden dim as decoder for residual compatibility
        self.linear = nn.Linear(n_state, n_vocab)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """
        Forward pass through Medusa head.

        Args:
            hidden_states: Decoder hidden states before final projection
                          Shape: (batch, seq_len, n_state)

        Returns:
            Logits for predicted future token
            Shape: (batch, seq_len, n_vocab)
        """
        # Simple linear projection
        # The hidden states already contain context from the full decoder
        return self.linear(hidden_states)


class MedusaHeadBlock(nn.Module):
    """
    Medusa head with shared transformer block (Medusa-Block variant).

    This variant uses an additional transformer block shared across all heads
    before the linear projection. It can capture more complex patterns but
    uses more compute.

    Architecture:
    - Self-attention over hidden states (no cross-attention)
    - Feed-forward network
    - Linear projection to vocabulary
    """

    def __init__(
        self,
        n_state: int,
        n_vocab: int,
        n_head: int,
        head_index: int = 1,
        dtype: mx.Dtype = mx.float16,
    ):
        super().__init__()
        self.head_index = head_index

        # Self-attention (no cross-attention for Medusa)
        self.self_attn = nn.MultiHeadAttention(n_state, n_head)
        self.ln1 = nn.LayerNorm(n_state)

        # Feed-forward
        self.mlp = nn.Sequential(
            nn.Linear(n_state, n_state * 4),
            nn.GELU(),
            nn.Linear(n_state * 4, n_state),
        )
        self.ln2 = nn.LayerNorm(n_state)

        # Projection to vocabulary
        self.linear = nn.Linear(n_state, n_vocab)

    def __call__(
        self,
        hidden_states: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        """
        Forward pass through Medusa block head.

        Args:
            hidden_states: Decoder hidden states
                          Shape: (batch, seq_len, n_state)
            mask: Optional causal attention mask

        Returns:
            Logits for predicted future token
            Shape: (batch, seq_len, n_vocab)
        """
        # Self-attention with residual
        x = hidden_states
        ln_x = self.ln1(x)
        # Self-attention: query, key, value are all the same
        attn_out = self.self_attn(ln_x, ln_x, ln_x, mask=mask)
        x = x + attn_out

        # Feed-forward with residual
        x = x + self.mlp(self.ln2(x))

        # Project to vocabulary
        return self.linear(x)


class MedusaModule(nn.Module):
    """
    Collection of Medusa heads for multi-token prediction.

    This module manages multiple Medusa heads, each predicting a different
    future token position. It can be added to an existing Whisper decoder.

    Supports three variants:
    - "linear": Direct hidden_dim -> vocab projection (self-trained)
    - "block": Transformer block + vocab projection (self-trained)
    - "aiola": Residual block (hidden_dim -> hidden_dim) + shared proj_out
              (compatible with aiola/whisper-medusa-v1 weights)

    Args:
        n_state: Hidden dimension from decoder
        n_vocab: Vocabulary size
        n_heads: Number of Medusa heads (default: 5)
        use_block: Use Medusa-Block variant (default: False for Medusa-Linear)
        use_aiola: Use aiola-compatible residual block architecture
        n_attn_head: Number of attention heads (only for Medusa-Block)
        proj_out: Shared vocabulary projection layer (required for aiola variant)
        dtype: Data type for computation
    """

    def __init__(
        self,
        n_state: int,
        n_vocab: int,
        n_heads: int = 5,
        use_block: bool = False,
        use_aiola: bool = False,
        n_attn_head: int = 20,
        proj_out: nn.Linear | None = None,
        dtype: mx.Dtype = mx.float16,
    ):
        super().__init__()
        self.n_state = n_state
        self.n_vocab = n_vocab
        self.n_heads = n_heads
        self.use_block = use_block
        self.use_aiola = use_aiola
        self._proj_out = proj_out  # Shared projection for aiola variant

        # Create Medusa heads based on variant
        if use_aiola:
            # Aiola variant: Residual block (hidden -> hidden), uses shared proj_out
            self.heads = [
                MedusaResBlock(n_state, head_index=i + 1, dtype=dtype)
                for i in range(n_heads)
            ]
        elif use_block:
            # Block variant: Transformer block + vocab projection
            self.heads = [
                MedusaHeadBlock(n_state, n_vocab, n_attn_head, head_index=i + 1, dtype=dtype)
                for i in range(n_heads)
            ]
        else:
            # Linear variant: Direct vocab projection
            self.heads = [
                MedusaHead(n_state, n_vocab, head_index=i + 1, dtype=dtype)
                for i in range(n_heads)
            ]

    def set_proj_out(self, proj_out: nn.Linear):
        """Set the shared projection layer for aiola variant."""
        self._proj_out = proj_out

    def __call__(
        self,
        hidden_states: mx.array,
        mask: mx.array | None = None,
    ) -> list[mx.array]:
        """
        Forward pass through all Medusa heads.

        Args:
            hidden_states: Decoder hidden states before final projection
                          Shape: (batch, seq_len, n_state)
            mask: Optional causal attention mask (for Medusa-Block)

        Returns:
            List of logits from each head, each shape (batch, seq_len, n_vocab)
        """
        if self.use_aiola:
            # Aiola variant: heads produce hidden states, then apply shared proj_out
            if self._proj_out is None:
                raise RuntimeError(
                    "Aiola variant requires proj_out layer. "
                    "Call set_proj_out() or pass proj_out in constructor.",
                )
            head_outputs = [head(hidden_states) for head in self.heads]
            return [self._proj_out(h) for h in head_outputs]
        if self.use_block:
            return [head(hidden_states, mask=mask) for head in self.heads]
        return [head(hidden_states) for head in self.heads]

    def get_predictions(
        self,
        hidden_states: mx.array,
        top_k: int = 10,
        mask: mx.array | None = None,
    ) -> list[mx.array]:
        """
        Get top-k predictions from each Medusa head.

        This is used during inference to generate candidate tokens.

        Args:
            hidden_states: Decoder hidden states (typically last position only)
                          Shape: (batch, 1, n_state) or (batch, seq_len, n_state)
            top_k: Number of top candidates per head
            mask: Optional attention mask

        Returns:
            List of top-k token indices per head
            Each element shape: (batch, seq_len, top_k)
        """
        all_logits = self(hidden_states, mask=mask)
        predictions = []

        for logits in all_logits:
            # Get top-k indices
            top_indices = mx.argpartition(-logits, top_k, axis=-1)[..., :top_k]
            # Sort by probability
            top_logits = mx.take_along_axis(logits, top_indices, axis=-1)
            sort_indices = mx.argsort(-top_logits, axis=-1)
            top_indices = mx.take_along_axis(top_indices, sort_indices, axis=-1)
            predictions.append(top_indices)

        return predictions


def build_tree_attention_mask(
    prefix_len: int,
    tree_structure: list[int],
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """
    Build attention mask for tree-based verification.

    In tree attention, each candidate can only attend to:
    1. The input prefix (with causal attention within prefix)
    2. Its ancestors in the tree (the path that led to this candidate)

    Args:
        prefix_len: Length of the input prefix (context tokens)
        tree_structure: Number of candidates at each depth level
                       e.g., [5, 4, 3, 2, 1] = 5 at depth 1, 4 at depth 2, etc.
        dtype: Data type for mask

    Returns:
        Attention mask of shape (total_tokens, total_tokens)
        where total_tokens = prefix_len + sum(tree_structure)
    """
    # Calculate total sequence length
    total_candidates = sum(tree_structure)
    total_len = prefix_len + total_candidates

    # Initialize mask (0 = can attend, -inf = cannot attend)
    mask = mx.full((total_len, total_len), float('-inf'), dtype=dtype)

    # Prefix tokens have causal attention among themselves
    for i in range(prefix_len):
        mask[i, :i + 1] = 0

    # Candidate tokens can attend to the full prefix (causal from their perspective)
    for i in range(prefix_len, total_len):
        mask[i, :prefix_len] = 0

    # Build tree structure
    # Each candidate at depth d can attend to its ancestor at depth d-1
    current_pos = prefix_len
    parent_positions = list(range(prefix_len))  # Start with prefix as "parents"

    for depth, n_candidates in enumerate(tree_structure):
        if depth == 0:
            # First level: all candidates attend to end of prefix
            for i in range(n_candidates):
                pos = current_pos + i
                mask[pos, pos] = 0  # Can attend to itself
            parent_positions = list(range(current_pos, current_pos + n_candidates))
        else:
            # Subsequent levels: each candidate attends to one parent
            new_parents = []
            for i in range(n_candidates):
                pos = current_pos + i
                # Assign parent (round-robin for simplicity)
                parent_idx = i % len(parent_positions)
                parent_pos = parent_positions[parent_idx]

                # Can attend to parent and parent's ancestors (in candidate region)
                for j in range(prefix_len, parent_pos + 1):
                    if mask[parent_pos, j] == 0:
                        mask[pos, j] = 0
                mask[pos, pos] = 0  # Can attend to itself

                new_parents.append(pos)

            parent_positions = new_parents

        current_pos += n_candidates

    return mask


class TreeCandidate:
    """
    Represents a candidate path in the Medusa tree.

    Each candidate is a sequence of token indices that forms a path
    from the root (main decoder prediction) through Medusa head predictions.
    """

    def __init__(
        self,
        tokens: list[int],
        parent_indices: list[int],
        depth: int,
    ):
        """
        Args:
            tokens: Token indices in this path
            parent_indices: Index of parent candidate at each depth (for tree structure)
            depth: Depth of this candidate in tree (0 = main prediction)
        """
        self.tokens = tokens
        self.parent_indices = parent_indices
        self.depth = depth


class MedusaTreeVerifier:
    """
    Verifies candidate sequences using tree attention.

    This class handles the construction of candidate trees and their
    parallel verification using the main decoder.

    Tree Verification Algorithm:
    1. Generate candidates: Get top-k predictions from each Medusa head
    2. Build tree structure: Create paths through head predictions
    3. Construct tree mask: Attention mask encoding parent-child relationships
    4. Parallel verification: Single forward pass with tree attention
    5. Accept longest valid prefix: Find best matching path
    """

    def __init__(
        self,
        decoder,
        medusa_module: MedusaModule,
        tree_structure: list[int] | None = None,
        top_k: int = 5,
    ):
        """
        Args:
            decoder: The main Whisper decoder
            medusa_module: MedusaModule with trained heads
            tree_structure: Candidates per level (default: [5, 4, 3, 2, 1])
            top_k: Number of top candidates per head position
        """
        self.decoder = decoder
        self.medusa = medusa_module
        self.top_k = top_k

        # Default tree: 5 heads, decreasing candidates per level
        # tree_structure[i] = number of top-k candidates to consider at depth i+1
        self.tree_structure = tree_structure or [5, 4, 3, 2, 1]

        # Statistics
        self._total_proposed = 0
        self._total_accepted = 0
        self._iterations = 0

    @property
    def acceptance_rate(self) -> float:
        """Average acceptance rate across all iterations."""
        if self._total_proposed == 0:
            return 0.0
        return self._total_accepted / self._total_proposed

    @property
    def tokens_per_iteration(self) -> float:
        """Average tokens accepted per iteration."""
        if self._iterations == 0:
            return 1.0
        return self._total_accepted / self._iterations

    def reset_stats(self):
        """Reset acceptance statistics."""
        self._total_proposed = 0
        self._total_accepted = 0
        self._iterations = 0

    def generate_tree_candidates(
        self,
        hidden_states: mx.array,
        main_logits: mx.array,
    ) -> tuple[mx.array, mx.array, list[list[int]], list[int]]:
        """
        Generate tree of candidate sequences from Medusa predictions.

        Creates a tree where:
        - Root: Main decoder's top prediction
        - Level i: Top-k predictions from Medusa head i, branching from level i-1

        Args:
            hidden_states: Last decoder hidden state (batch, 1, n_state)
            main_logits: Main decoder logits (batch, 1, n_vocab)

        Returns:
            Tuple of:
            - tree_tokens: All candidate tokens flattened (batch, total_nodes)
            - tree_mask: Attention mask for tree (total_nodes, total_nodes)
            - tree_paths: List of paths, each path is list of node indices
            - node_depths: Depth level of each node (0 for root, 1 for level 1, etc.)
        """
        _ = hidden_states.shape[0]  # batch_size (unused, for reference)
        n_heads = self.medusa.n_heads

        # Get main decoder's top-k predictions
        main_top_k = min(self.top_k, self.tree_structure[0]) if self.tree_structure else self.top_k
        main_top_indices = mx.argpartition(-main_logits[:, -1], main_top_k, axis=-1)[:, :main_top_k]
        main_top_logits = mx.take_along_axis(main_logits[:, -1], main_top_indices, axis=-1)
        main_sort_idx = mx.argsort(-main_top_logits, axis=-1)
        main_preds = mx.take_along_axis(main_top_indices, main_sort_idx, axis=-1)  # (batch, top_k)

        # Get Medusa head predictions for each head
        medusa_preds = self.medusa.get_predictions(hidden_states, top_k=self.top_k)  # List of (batch, 1, top_k)

        # Build tree structure
        # node_paths[i] = path from root to node i (list of node indices)
        # This tracks the path to every node, not just leaves

        tree_tokens_list = []
        node_paths = []  # node_paths[i] = path to node i
        parent_map = []  # parent_map[i] = index of parent node, -1 for root level
        node_depths = []  # node_depths[i] = depth level of node i

        # Level 0: Main decoder predictions
        n_level_0 = min(main_top_k, self.tree_structure[0]) if self.tree_structure else main_top_k
        for i in range(n_level_0):
            node_idx = len(tree_tokens_list)
            tree_tokens_list.append(main_preds[:, i:i+1])  # (batch, 1)
            node_paths.append([node_idx])  # Path to this root node is just itself
            parent_map.append(-1)  # Root level has no parent
            node_depths.append(0)  # Depth 0 for root level

        # Track which nodes are leaves (will be updated each level)
        leaf_nodes = list(range(n_level_0))

        # Levels 1+: Medusa head predictions branching from previous level
        for head_idx in range(min(n_heads, len(self.tree_structure) - 1)):
            head_preds = medusa_preds[head_idx][:, -1]  # (batch, top_k)
            n_branches = min(self.top_k, self.tree_structure[head_idx + 1]) if head_idx + 1 < len(self.tree_structure) else 1
            current_depth = head_idx + 1  # Depth starts at 1 for first Medusa head

            new_leaf_nodes = []
            for parent_node in leaf_nodes:
                # Each leaf node branches to top n_branches predictions
                for k in range(n_branches):
                    node_idx = len(tree_tokens_list)
                    tree_tokens_list.append(head_preds[:, k:k+1])  # (batch, 1)

                    # Build path by extending parent's path
                    parent_path = node_paths[parent_node].copy()
                    parent_path.append(node_idx)
                    node_paths.append(parent_path)
                    parent_map.append(parent_node)
                    node_depths.append(current_depth)

                    new_leaf_nodes.append(node_idx)

            leaf_nodes = new_leaf_nodes

        # Stack all tokens
        tree_tokens = mx.concatenate(tree_tokens_list, axis=-1)  # (batch, total_nodes)

        # Build tree attention mask
        total_nodes = tree_tokens.shape[-1]
        tree_mask = self._build_tree_mask(total_nodes, parent_map)

        # Extract paths to leaf nodes only
        tree_paths = [node_paths[leaf] for leaf in leaf_nodes]

        return tree_tokens, tree_mask, tree_paths, node_depths

    def _build_tree_mask(
        self,
        total_nodes: int,
        parent_map: list[int],
        dtype: mx.Dtype = mx.float32,
    ) -> mx.array:
        """
        Build attention mask for tree verification.

        Each node can attend to:
        1. Itself
        2. All its ancestors (via parent_map traversal)

        Args:
            total_nodes: Total number of nodes in tree
            parent_map: parent_map[i] = index of parent, -1 for roots
            dtype: Data type for mask

        Returns:
            Attention mask (total_nodes, total_nodes) where 0 = attend, -inf = mask
        """
        import numpy as np

        # Build mask in numpy (MLX doesn't support .at[] indexing)
        mask_np = np.full((total_nodes, total_nodes), float('-inf'), dtype=np.float32)

        for i in range(total_nodes):
            # Can attend to self
            mask_np[i, i] = 0

            # Can attend to all ancestors
            parent = parent_map[i]
            while parent >= 0:
                mask_np[i, parent] = 0
                parent = parent_map[parent]

        return mx.array(mask_np, dtype=dtype)

    def generate_candidates(
        self,
        hidden_states: mx.array,
        main_logits: mx.array,
        top_k: int = 5,
    ) -> tuple[mx.array, mx.array]:
        """
        Generate candidate token sequences from Medusa predictions.

        This is a simpler flat candidate generation (no tree structure).
        Use generate_tree_candidates for proper tree verification.

        Args:
            hidden_states: Last decoder hidden state (batch, 1, n_state)
            main_logits: Main decoder logits (batch, 1, n_vocab)
            top_k: Number of top candidates per position

        Returns:
            Tuple of:
            - candidates: Token candidates (batch, n_candidates, max_depth)
            - main_pred: Main decoder's prediction
        """
        # Get main decoder's prediction (most likely next token)
        main_pred = mx.argmax(main_logits[:, -1], axis=-1, keepdims=True)  # (batch, 1)

        # Get Medusa head predictions
        medusa_preds = self.medusa.get_predictions(hidden_states, top_k=top_k)

        # Build candidate sequences
        _ = len(medusa_preds)  # n_heads (unused)

        # Combine predictions into candidate sequences
        # Each candidate is: [main_pred, head1_pred, head2_pred, ...]
        all_candidates = []

        # Greedy path: just the top prediction from each head
        greedy_path = [main_pred] + [head_preds[:, -1, 0:1] for head_preds in medusa_preds]
        greedy_candidate = mx.concatenate(greedy_path, axis=-1)  # (batch, n_heads+1)
        all_candidates.append(greedy_candidate)

        # Alternative paths: vary one position at a time
        for head_idx, _head_preds in enumerate(medusa_preds):
            for k in range(1, min(top_k, 3)):  # Top 3 alternatives per head
                path = [main_pred]
                for j, h_preds in enumerate(medusa_preds):
                    if j == head_idx:
                        path.append(h_preds[:, -1, k:k + 1])
                    else:
                        path.append(h_preds[:, -1, 0:1])
                candidate = mx.concatenate(path, axis=-1)
                all_candidates.append(candidate)

        # Stack all candidates
        candidates = mx.stack(all_candidates, axis=1)  # (batch, n_candidates, depth)

        return candidates, main_pred

    def verify_tree_parallel(
        self,
        tree_tokens: mx.array,
        tree_mask: mx.array,
        tree_paths: list[list[int]],
        audio_features: mx.array,
        kv_cache: list | None = None,
        node_depths: list[int] | None = None,
    ) -> tuple[mx.array, int, list | None]:
        """
        Verify all tree candidates in a single parallel forward pass.

        This is the key optimization: instead of verifying each candidate
        sequentially, we process all candidates at once using tree attention.

        The decoder processes all tree tokens in one forward pass with a custom
        attention mask that enforces the tree structure. Each node can only
        attend to its ancestors, not siblings.

        Args:
            tree_tokens: All candidate tokens (batch, total_nodes)
            tree_mask: Tree attention mask (total_nodes, total_nodes)
            tree_paths: List of paths, each path is list of node indices
            audio_features: Encoder output (batch, encoder_len, n_state)
            kv_cache: Optional KV cache from previous decode steps
            node_depths: Optional depth level of each tree node. Used to compute
                        correct positional embeddings where all nodes at depth L
                        share position prefix_len + L. If None, computes from paths.

        Returns:
            Tuple of:
            - accepted_tokens: Verified tokens to add (batch, n_accepted)
            - n_accepted: Number of accepted tokens
            - updated_kv_cache: Cache updated with accepted tokens only
        """
        _ = tree_tokens.shape[0]  # batch_size (unused)
        total_nodes = tree_tokens.shape[1]

        # Build combined attention mask for prefix + tree
        # The mask needs to account for:
        # 1. KV cache prefix (all tree tokens can attend to prefix)
        # 2. Tree structure (each node attends to ancestors only)
        prefix_len = kv_cache[0][0][0].shape[1] if kv_cache else 0
        # Use dtype from KV cache (this matches what attention uses internally)
        # Note: audio_features may be float32 but attention uses float16
        mask_dtype = kv_cache[0][0][0].dtype if kv_cache else mx.float16
        combined_mask = self._build_combined_attention_mask(
            prefix_len, total_nodes, tree_mask, dtype=mask_dtype,
        )

        # Compute custom positional embedding indices based on tree depth
        # All nodes at depth L should use position prefix_len + L
        # This prevents positional embedding overflow when KV cache is large
        if node_depths is not None:
            # Use provided node depths
            custom_positions = mx.array([prefix_len + d for d in node_depths], dtype=mx.int32)
        else:
            # Fallback: compute depths from path lengths (for backwards compatibility)
            # This is less efficient but works if node_depths not provided
            _ = max(len(p) for p in tree_paths)  # max_depth (unused)
            node_depth_map = {
                node_idx: depth
                for path in tree_paths
                for depth, node_idx in enumerate(path)
            }
            custom_positions = mx.array(
                [prefix_len + node_depth_map.get(i, 0) for i in range(total_nodes)],
                dtype=mx.int32,
            )

        # Process all tree tokens in a SINGLE forward pass with tree attention
        # This is the key speedup: O(1) forward passes instead of O(total_nodes)
        #
        # NOTE: The decoder will modify kv_cache in place with all tree tokens.
        # After verification, we select only the accepted token entries from the cache.
        all_logits, _, _, _ = self.decoder(
            tree_tokens,
            audio_features,
            kv_cache=kv_cache,
            custom_mask=combined_mask,
            custom_positions=custom_positions,
        )
        # all_logits shape: (batch, total_nodes, vocab)

        # Find best path by checking each path for validity
        best_path_tokens = None
        best_length = 0

        for path in tree_paths:
            _ = len(path)  # path_length (unused)
            valid_length = 0

            for i, node_idx in enumerate(path):
                # Get the token at this node
                node_token = tree_tokens[:, node_idx]  # (batch,)

                if i == 0:
                    # First token: always valid (it's the main decoder's prediction)
                    # Verify by checking if decoder would predict this token
                    # Since we're using the decoder's top prediction, it should match
                    valid_length = 1
                else:
                    # Subsequent tokens: check if parent's output predicts this token
                    parent_idx = path[i - 1]
                    parent_logits = all_logits[:, parent_idx]  # (batch, vocab)
                    parent_pred = mx.argmax(parent_logits, axis=-1)  # (batch,)

                    # Check if prediction matches the candidate token
                    if mx.all(parent_pred == node_token).item():
                        valid_length = i + 1
                    else:
                        break

            if valid_length > best_length:
                best_length = valid_length
                best_path_tokens = mx.stack(
                    [tree_tokens[:, path[i]] for i in range(valid_length)],
                    axis=1,
                )  # (batch, valid_length)

        # Update statistics
        self._iterations += 1
        max_path_len = max(len(p) for p in tree_paths)
        self._total_proposed += max_path_len
        self._total_accepted += best_length

        # Update KV cache efficiently by selecting accepted entries from tree cache
        # The decoder has already computed KV entries for all tree tokens.
        # We just need to select the entries corresponding to accepted tokens.

        # Handle case where kv_cache is None (no cache to update)
        if kv_cache is None:
            if best_path_tokens is None:
                best_path_tokens = tree_tokens[:, 0:1]
                best_length = 1
            return best_path_tokens, best_length, None

        if best_path_tokens is not None and best_length > 0:
            # Find the path indices that were accepted
            best_path = None
            for path in tree_paths:
                valid_length = 0
                for i, node_idx in enumerate(path):
                    if i == 0:
                        valid_length = 1
                    else:
                        parent_idx = path[i - 1]
                        parent_logits = all_logits[:, parent_idx]
                        parent_pred = mx.argmax(parent_logits, axis=-1)
                        if mx.all(parent_pred == tree_tokens[:, node_idx]).item():
                            valid_length = i + 1
                        else:
                            break
                if valid_length == best_length:
                    best_path = path[:valid_length]
                    break

            # Select KV cache entries for accepted path
            # kv_cache structure: [(self_kv, cross_kv), ...] per layer
            # Each kv is (key, value) of shape (batch, prefix_len + tree_len, n_state)
            updated_cache = []
            for (k, v), (ck, cv) in kv_cache:
                # Self-attention KV: select prefix + accepted tree positions
                # Original prefix_len positions + positions corresponding to best_path
                if best_path is not None:
                    # The tree tokens were appended after the prefix
                    # prefix indices: 0:prefix_len
                    # tree indices: prefix_len:prefix_len+total_nodes
                    # We want: prefix + accepted tree positions in order
                    accepted_indices = list(range(prefix_len)) + [prefix_len + idx for idx in best_path]
                    accepted_indices = mx.array(accepted_indices)
                    new_k = k[:, accepted_indices, :]
                    new_v = v[:, accepted_indices, :]
                else:
                    # Fallback: just use original cache + first tree token
                    new_k = k[:, :prefix_len + 1, :]
                    new_v = v[:, :prefix_len + 1, :]

                # Cross-attention KV: unchanged (just audio features)
                updated_cache.append(((new_k, new_v), (ck, cv)))
        else:
            # Fallback: original cache + first token
            updated_cache = []
            for (k, v), (ck, cv) in kv_cache:
                new_k = k[:, :prefix_len + 1, :]
                new_v = v[:, :prefix_len + 1, :]
                updated_cache.append(((new_k, new_v), (ck, cv)))
            best_path_tokens = tree_tokens[:, 0:1]
            best_length = 1

        return best_path_tokens, best_length, updated_cache

    def _build_combined_attention_mask(
        self,
        prefix_len: int,
        tree_len: int,
        tree_mask: mx.array,
        dtype: mx.Dtype = mx.float32,
    ) -> mx.array:
        """
        Build combined attention mask for prefix context + tree structure.

        The combined mask allows:
        1. All tree tokens to attend to the full prefix (causal from prefix end)
        2. Tree tokens to attend to ancestors within tree (from tree_mask)

        Args:
            prefix_len: Length of KV cache prefix (context from previous decoding)
            tree_len: Number of tree tokens
            tree_mask: Tree attention mask (tree_len, tree_len)
            dtype: Data type for mask

        Returns:
            Combined mask of shape (tree_len, prefix_len + tree_len)
            where 0 = attend, -inf = mask
        """
        import numpy as np

        total_kv_len = prefix_len + tree_len

        # Build mask in numpy for easier indexing
        mask_np = np.full((tree_len, total_kv_len), float('-inf'), dtype=np.float32)

        # All tree tokens can attend to full prefix (causal attention to context)
        if prefix_len > 0:
            mask_np[:, :prefix_len] = 0

        # Tree tokens follow tree structure for attending to each other
        # Copy the tree mask for the tree portion
        tree_mask_np = np.array(tree_mask, dtype=np.float32)
        mask_np[:, prefix_len:] = tree_mask_np

        return mx.array(mask_np, dtype=dtype)

    def verify_candidates(
        self,
        candidates: mx.array,
        audio_features: mx.array,
        prefix_tokens: mx.array,
        kv_cache: list | None = None,
    ) -> tuple[mx.array, int, list | None]:
        """
        Verify candidate sequences using the main decoder.

        This is the legacy sequential verification method.
        For better performance, use verify_tree_parallel with tree candidates.

        Args:
            candidates: Candidate sequences (batch, n_candidates, depth)
            audio_features: Encoder output (batch, encoder_len, n_state)
            prefix_tokens: Context tokens (batch, prefix_len)
            kv_cache: Optional KV cache

        Returns:
            Tuple of:
            - accepted_tokens: Verified tokens to add
            - n_accepted: Number of accepted tokens
            - updated_kv_cache
        """
        batch_size, n_candidates, depth = candidates.shape

        # For each candidate, verify one token at a time
        best_candidate = None
        best_length = 0

        for c in range(n_candidates):
            candidate = candidates[:, c, :]  # (batch, depth)

            # Try to verify as many tokens as possible
            valid_length = 0
            temp_cache = kv_cache

            for d in range(depth):
                token = candidate[:, d:d + 1]  # (batch, 1)

                # Run through decoder
                logits, temp_cache, _, _ = self.decoder(
                    token, audio_features, kv_cache=temp_cache,
                )

                # Get decoder's prediction
                decoder_pred = mx.argmax(logits[:, -1], axis=-1)

                # Check if candidate matches (or if this is the first token)
                if d == 0:
                    # First token always uses decoder's prediction
                    valid_length += 1
                else:
                    # Check if decoder agrees with candidate
                    if mx.all(decoder_pred == candidate[:, d]).item():
                        valid_length += 1
                    else:
                        break

            if valid_length > best_length:
                best_length = valid_length
                best_candidate = candidate[:, :valid_length]

        # Update statistics
        self._iterations += 1
        self._total_proposed += depth
        self._total_accepted += best_length

        return best_candidate, best_length, kv_cache


def create_medusa_module(
    config,
    n_heads: int = 5,
    use_block: bool = False,
    use_aiola: bool = False,
    proj_out: nn.Linear | None = None,
    dtype: mx.Dtype = mx.float16,
) -> MedusaModule:
    """
    Create a MedusaModule for a Whisper config.

    Args:
        config: WhisperConfig with n_text_state and n_vocab
        n_heads: Number of Medusa heads
        use_block: Use Medusa-Block variant
        use_aiola: Use aiola-compatible residual block architecture
        proj_out: Shared projection layer (required for aiola variant)
        dtype: Data type

    Returns:
        MedusaModule instance
    """
    return MedusaModule(
        n_state=config.n_text_state,
        n_vocab=config.n_vocab,
        n_heads=n_heads,
        use_block=use_block,
        use_aiola=use_aiola,
        n_attn_head=config.n_text_head,
        proj_out=proj_out,
        dtype=dtype,
    )
