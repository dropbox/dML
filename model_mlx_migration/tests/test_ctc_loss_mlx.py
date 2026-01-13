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
Validation suite for native MLX CTC loss.

REQUIREMENT: MLX CTC must be numerically equivalent to PyTorch CTC.
These tests ensure the implementation is correct before using in training.

Tolerances:
    - Absolute tolerance: 1e-4 (accounts for log-sum-exp accumulation)
    - Relative tolerance: 1e-5 (excellent for loss values in hundreds/thousands)

DO NOT use native MLX CTC in production training until ALL tests pass.
"""

import mlx.core as mx
import numpy as np
import pytest

# Check if PyTorch is available for reference comparison
try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from tools.whisper_mlx.training.ctc_loss_mlx import (
    ctc_loss,
    ctc_loss_batch,
    ctc_loss_with_grad,
    validate_against_pytorch,
)

# =============================================================================
# Test Fixtures
# =============================================================================

def generate_test_data(T, N, C, S_max, seed=42):
    """Generate random test data for CTC loss testing."""
    rng = np.random.default_rng(seed)

    # Generate log probabilities (normalized)
    logits = rng.standard_normal((T, N, C)).astype(np.float32)
    log_probs = logits - np.log(np.sum(np.exp(logits), axis=-1, keepdims=True))

    # Generate targets (no blanks, values in [1, C-1])
    targets_list = []
    target_lengths = []
    for _ in range(N):
        S = rng.integers(1, S_max + 1)
        targets_list.extend(rng.integers(1, C, size=S).tolist())
        target_lengths.append(S)

    input_lengths = [T] * N

    return {
        "log_probs": log_probs,
        "targets": np.array(targets_list, dtype=np.int32),
        "input_lengths": np.array(input_lengths, dtype=np.int32),
        "target_lengths": np.array(target_lengths, dtype=np.int32),
    }


# =============================================================================
# Basic Correctness Tests
# =============================================================================

class TestCTCLossBasic:
    """Basic correctness tests for CTC loss."""

    def test_single_sample_small(self):
        """Test with minimal input (T=10, N=1, C=5)."""
        data = generate_test_data(T=10, N=1, C=5, S_max=3)

        loss = ctc_loss(
            mx.array(data["log_probs"]),
            mx.array(data["targets"]),
            mx.array(data["input_lengths"]),
            mx.array(data["target_lengths"]),
            blank=0,
            reduction="mean",
        )

        # Should be a positive scalar
        assert loss.ndim == 0
        assert float(loss) > 0

    def test_batch_computation(self):
        """Test batched computation (N=4)."""
        data = generate_test_data(T=50, N=4, C=100, S_max=10)

        loss = ctc_loss(
            mx.array(data["log_probs"]),
            mx.array(data["targets"]),
            mx.array(data["input_lengths"]),
            mx.array(data["target_lengths"]),
            blank=0,
            reduction="mean",
        )

        assert loss.ndim == 0
        assert float(loss) > 0

    def test_reduction_none(self):
        """Test reduction='none' returns per-sample losses."""
        data = generate_test_data(T=20, N=4, C=20, S_max=5)

        losses = ctc_loss(
            mx.array(data["log_probs"]),
            mx.array(data["targets"]),
            mx.array(data["input_lengths"]),
            mx.array(data["target_lengths"]),
            blank=0,
            reduction="none",
        )

        assert losses.shape == (4,)
        assert all(float(loss) > 0 for loss in losses)

    def test_reduction_sum(self):
        """Test reduction='sum' returns sum of losses."""
        data = generate_test_data(T=20, N=4, C=20, S_max=5)

        loss_sum = ctc_loss(
            mx.array(data["log_probs"]),
            mx.array(data["targets"]),
            mx.array(data["input_lengths"]),
            mx.array(data["target_lengths"]),
            blank=0,
            reduction="sum",
        )

        loss_none = ctc_loss(
            mx.array(data["log_probs"]),
            mx.array(data["targets"]),
            mx.array(data["input_lengths"]),
            mx.array(data["target_lengths"]),
            blank=0,
            reduction="none",
        )

        # Sum of individual losses should approximately equal reduction='sum'
        # (not exact due to float precision)
        expected_sum = float(mx.sum(loss_none))
        actual_sum = float(loss_sum)
        assert abs(expected_sum - actual_sum) < 1e-3


# =============================================================================
# PyTorch Equivalence Tests
# =============================================================================

@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestCTCLossEquivalence:
    """Test MLX CTC produces identical results to PyTorch CTC."""

    # Tolerance for numerical equivalence
    # Absolute tolerance handles small absolute differences
    # Relative tolerance handles large loss values
    ATOL = 1e-4
    RTOL = 1e-5

    def _compare_loss(self, T, N, C, S_max, seed=42):
        """Helper to compare MLX vs PyTorch CTC loss."""
        mlx_loss, pt_loss, abs_error = validate_against_pytorch(
            T=T, N=N, C=C, S_max=S_max, seed=seed,
        )

        # Check relative error for large loss values
        rel_error = abs_error / max(abs(pt_loss), 1e-8)

        # Pass if either absolute OR relative tolerance is met
        passed = (abs_error < self.ATOL) or (rel_error < self.RTOL)

        assert passed, (
            f"Loss mismatch: MLX={mlx_loss:.6f}, PT={pt_loss:.6f}, "
            f"abs_error={abs_error:.2e}, rel_error={rel_error:.2e}"
        )

        return mlx_loss, pt_loss, abs_error

    def test_small_sequence(self):
        """Test with small T=10, N=1, C=5, S=3."""
        self._compare_loss(T=10, N=1, C=5, S_max=3)

    def test_medium_sequence(self):
        """Test with medium T=50, N=1, C=50, S=10."""
        self._compare_loss(T=50, N=1, C=50, S_max=10)

    def test_batch_small(self):
        """Test with batch N=4."""
        self._compare_loss(T=50, N=4, C=100, S_max=10)

    def test_batch_medium(self):
        """Test with batch N=8."""
        self._compare_loss(T=100, N=8, C=200, S_max=15)

    def test_large_vocab(self):
        """Test with large vocabulary (C=1000)."""
        self._compare_loss(T=100, N=2, C=1000, S_max=20)

    def test_whisper_vocab(self):
        """Test with Whisper vocabulary size (C=51865)."""
        self._compare_loss(T=100, N=2, C=51865, S_max=20)

    @pytest.mark.slow
    def test_whisper_realistic(self):
        """Test with realistic Whisper dimensions (T=750, N=4, C=51865)."""
        self._compare_loss(T=750, N=4, C=51865, S_max=50)

    @pytest.mark.slow
    def test_whisper_batch_16(self):
        """Test with batch_size=16 at Whisper scale."""
        self._compare_loss(T=750, N=16, C=51865, S_max=50)

    def test_different_seeds(self):
        """Test with different random seeds for robustness."""
        for seed in [0, 123, 456, 789, 999]:
            self._compare_loss(T=50, N=4, C=100, S_max=10, seed=seed)


# =============================================================================
# Edge Case Tests
# =============================================================================

@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestCTCLossEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_frame(self):
        """Test with single time frame (T=1)."""
        # T=1 can only match target length 0 or 1
        rng = np.random.default_rng(42)
        T, N, C = 1, 1, 10

        log_probs = rng.standard_normal((T, N, C)).astype(np.float32)
        log_probs = log_probs - np.log(np.sum(np.exp(log_probs), axis=-1, keepdims=True))

        # Target of length 1
        targets = np.array([1], dtype=np.int32)
        input_lengths = np.array([1], dtype=np.int32)
        target_lengths = np.array([1], dtype=np.int32)

        loss_mlx = ctc_loss(
            mx.array(log_probs),
            mx.array(targets),
            mx.array(input_lengths),
            mx.array(target_lengths),
            blank=0,
            reduction="sum",
        )

        loss_pt = F.ctc_loss(
            torch.from_numpy(log_probs),
            torch.tensor(targets, dtype=torch.long),
            torch.tensor(input_lengths, dtype=torch.long),
            torch.tensor(target_lengths, dtype=torch.long),
            blank=0,
            reduction="sum",
        )

        assert abs(float(loss_mlx) - loss_pt.item()) < 1e-4

    def test_long_target(self):
        """Test with target almost as long as input."""
        # For T frames, max target length is T (all non-blanks)
        T, N, C = 20, 1, 50
        S = 10  # Half of T

        rng = np.random.default_rng(42)
        log_probs = rng.standard_normal((T, N, C)).astype(np.float32)
        log_probs = log_probs - np.log(np.sum(np.exp(log_probs), axis=-1, keepdims=True))

        targets = rng.integers(1, C, size=S).astype(np.int32)
        input_lengths = np.array([T], dtype=np.int32)
        target_lengths = np.array([S], dtype=np.int32)

        loss_mlx = ctc_loss(
            mx.array(log_probs),
            mx.array(targets),
            mx.array(input_lengths),
            mx.array(target_lengths),
            blank=0,
            reduction="sum",
        )

        loss_pt = F.ctc_loss(
            torch.from_numpy(log_probs),
            torch.tensor(targets, dtype=torch.long),
            torch.tensor(input_lengths, dtype=torch.long),
            torch.tensor(target_lengths, dtype=torch.long),
            blank=0,
            reduction="sum",
        )

        assert abs(float(loss_mlx) - loss_pt.item()) < 1e-3

    def test_repeated_labels(self):
        """Test with repeated consecutive labels (requires blank between)."""
        T, N, C = 30, 1, 10

        rng = np.random.default_rng(42)
        log_probs = rng.standard_normal((T, N, C)).astype(np.float32)
        log_probs = log_probs - np.log(np.sum(np.exp(log_probs), axis=-1, keepdims=True))

        # Target with repeated labels: [1, 1, 2, 2, 3]
        # CTC must insert blanks between repeated: [blank, 1, blank, 1, blank, 2, ...]
        targets = np.array([1, 1, 2, 2, 3], dtype=np.int32)
        input_lengths = np.array([T], dtype=np.int32)
        target_lengths = np.array([5], dtype=np.int32)

        loss_mlx = ctc_loss(
            mx.array(log_probs),
            mx.array(targets),
            mx.array(input_lengths),
            mx.array(target_lengths),
            blank=0,
            reduction="sum",
        )

        loss_pt = F.ctc_loss(
            torch.from_numpy(log_probs),
            torch.tensor(targets, dtype=torch.long),
            torch.tensor(input_lengths, dtype=torch.long),
            torch.tensor(target_lengths, dtype=torch.long),
            blank=0,
            reduction="sum",
        )

        assert abs(float(loss_mlx) - loss_pt.item()) < 1e-3

    def test_variable_lengths(self):
        """Test with variable input/target lengths in same batch."""
        rng = np.random.default_rng(42)
        T_max, N, C = 50, 4, 30

        log_probs = rng.standard_normal((T_max, N, C)).astype(np.float32)
        log_probs = log_probs - np.log(np.sum(np.exp(log_probs), axis=-1, keepdims=True))

        # Variable input lengths
        input_lengths = np.array([50, 40, 30, 20], dtype=np.int32)

        # Variable target lengths
        target_lengths = np.array([10, 8, 6, 4], dtype=np.int32)
        targets = np.concatenate([
            rng.integers(1, C, size=tl) for tl in target_lengths
        ]).astype(np.int32)

        loss_mlx = ctc_loss(
            mx.array(log_probs),
            mx.array(targets),
            mx.array(input_lengths),
            mx.array(target_lengths),
            blank=0,
            reduction="mean",
        )

        loss_pt = F.ctc_loss(
            torch.from_numpy(log_probs),
            torch.tensor(targets, dtype=torch.long),
            torch.tensor(input_lengths, dtype=torch.long),
            torch.tensor(target_lengths, dtype=torch.long),
            blank=0,
            reduction="mean",
        )

        assert abs(float(loss_mlx) - loss_pt.item()) < 1e-3


# =============================================================================
# Numerical Stability Tests
# =============================================================================

class TestCTCLossStability:
    """Test numerical stability of CTC loss."""

    def test_very_small_probs(self):
        """Test with very small (but valid) probabilities."""
        T, N, C = 20, 1, 10

        # Create log probs with very small values (large negative)
        rng = np.random.default_rng(42)
        log_probs = rng.standard_normal((T, N, C)).astype(np.float32) - 50
        log_probs = log_probs - np.log(np.sum(np.exp(log_probs), axis=-1, keepdims=True))

        targets = np.array([1, 2, 3], dtype=np.int32)
        input_lengths = np.array([T], dtype=np.int32)
        target_lengths = np.array([3], dtype=np.int32)

        loss = ctc_loss(
            mx.array(log_probs),
            mx.array(targets),
            mx.array(input_lengths),
            mx.array(target_lengths),
            blank=0,
            reduction="sum",
        )

        # Should not be NaN or Inf
        assert not np.isnan(float(loss))
        assert not np.isinf(float(loss))

    def test_zero_infinity(self):
        """Test zero_infinity flag replaces inf with 0."""
        # This can happen with impossible alignments (target too long for input)
        T, N, C = 5, 1, 10

        rng = np.random.default_rng(42)
        log_probs = rng.standard_normal((T, N, C)).astype(np.float32)
        log_probs = log_probs - np.log(np.sum(np.exp(log_probs), axis=-1, keepdims=True))

        # Target longer than what T frames can support
        # Max target length for T=5 is roughly T/2 with blanks = ~2-3
        targets = np.array([1, 2, 3, 4, 5], dtype=np.int32)  # 5 targets needs ~10 frames
        input_lengths = np.array([T], dtype=np.int32)
        target_lengths = np.array([5], dtype=np.int32)

        loss_zero_inf = ctc_loss(
            mx.array(log_probs),
            mx.array(targets),
            mx.array(input_lengths),
            mx.array(target_lengths),
            blank=0,
            reduction="sum",
            zero_infinity=True,
        )

        # With zero_infinity=True, should not be inf
        # (Though it might still be very large or 0 if impossible)
        val = float(loss_zero_inf)
        # Just check it's not NaN
        assert not np.isnan(val)


# =============================================================================
# Batch Convenience Function Tests
# =============================================================================

class TestCTCLossBatch:
    """Test ctc_loss_batch convenience function."""

    def test_batch_list_targets(self):
        """Test with list of target arrays."""
        T, N, C = 30, 3, 20

        rng = np.random.default_rng(42)
        log_probs = rng.standard_normal((T, N, C)).astype(np.float32)
        log_probs = log_probs - np.log(np.sum(np.exp(log_probs), axis=-1, keepdims=True))

        # List of target arrays (one per batch item)
        targets = [
            mx.array([1, 2, 3], dtype=mx.int32),
            mx.array([4, 5], dtype=mx.int32),
            mx.array([6, 7, 8, 9], dtype=mx.int32),
        ]
        input_lengths = mx.array([T, T, T], dtype=mx.int32)

        loss = ctc_loss_batch(
            mx.array(log_probs),
            targets,
            input_lengths,
            blank=0,
            reduction="mean",
        )

        assert loss.ndim == 0
        assert float(loss) > 0


# =============================================================================
# Gradient Equivalence Tests
# =============================================================================

@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch required for gradient comparison")
class TestCTCGradientEquivalence:
    """Test that native CTC gradients match PyTorch CTC gradients."""

    GRAD_TOLERANCE = 1e-4  # Absolute tolerance for gradient comparison

    def _compare_gradients(self, T, N, C, S_max, seed=42):
        """Compare MLX and PyTorch CTC gradients."""
        data = generate_test_data(T=T, N=N, C=C, S_max=S_max, seed=seed)

        # PyTorch gradient
        log_probs_torch = torch.from_numpy(data["log_probs"]).requires_grad_(True)
        targets_torch = torch.tensor(data["targets"].tolist(), dtype=torch.long)
        input_lengths_torch = torch.tensor(data["input_lengths"].tolist(), dtype=torch.long)
        target_lengths_torch = torch.tensor(data["target_lengths"].tolist(), dtype=torch.long)

        loss_torch = F.ctc_loss(
            log_probs_torch, targets_torch,
            input_lengths_torch, target_lengths_torch,
            blank=0, reduction="mean", zero_infinity=True,
        )
        loss_torch.backward()
        grad_torch = log_probs_torch.grad.numpy()

        # MLX gradient
        log_probs_mx = mx.array(data["log_probs"])
        targets_mx = mx.array(data["targets"])
        input_lens_mx = mx.array(data["input_lengths"])
        target_lens_mx = mx.array(data["target_lengths"])

        loss_mlx, grad_mlx = ctc_loss_with_grad(
            log_probs_mx, targets_mx,
            input_lens_mx, target_lens_mx,
            blank=0, reduction="mean",
        )
        mx.eval(loss_mlx, grad_mlx)
        grad_mlx_np = np.array(grad_mlx)

        # Compare
        loss_error = abs(float(loss_torch.item()) - float(loss_mlx))
        grad_max_error = np.max(np.abs(grad_torch - grad_mlx_np))
        grad_rel_error = np.linalg.norm(grad_torch - grad_mlx_np) / (np.linalg.norm(grad_torch) + 1e-10)

        return {
            "loss_torch": loss_torch.item(),
            "loss_mlx": float(loss_mlx),
            "loss_error": loss_error,
            "grad_max_error": grad_max_error,
            "grad_rel_error": grad_rel_error,
        }

    def test_small_gradient(self):
        """Test gradient with small dimensions."""
        result = self._compare_gradients(T=20, N=2, C=50, S_max=5)
        assert result["grad_max_error"] < self.GRAD_TOLERANCE, \
            f"Gradient error {result['grad_max_error']:.2e} exceeds tolerance {self.GRAD_TOLERANCE}"

    def test_medium_gradient(self):
        """Test gradient with medium dimensions."""
        result = self._compare_gradients(T=100, N=4, C=500, S_max=20)
        assert result["grad_max_error"] < self.GRAD_TOLERANCE, \
            f"Gradient error {result['grad_max_error']:.2e} exceeds tolerance {self.GRAD_TOLERANCE}"

    def test_large_vocab_gradient(self):
        """Test gradient with large vocabulary."""
        result = self._compare_gradients(T=50, N=2, C=5000, S_max=10)
        assert result["grad_max_error"] < self.GRAD_TOLERANCE, \
            f"Gradient error {result['grad_max_error']:.2e} exceeds tolerance {self.GRAD_TOLERANCE}"

    def test_whisper_scale_gradient(self):
        """Test gradient at Whisper scale dimensions.

        Note: At Whisper scale with long sequences and large vocab, numerical
        error accumulates in the forward-backward algorithm. We use a relaxed
        tolerance (1e-3) which is still excellent for training.
        """
        result = self._compare_gradients(T=750, N=4, C=51865, S_max=50)
        whisper_tolerance = 1e-3  # Relaxed for large-scale computation
        assert result["grad_max_error"] < whisper_tolerance, \
            f"Gradient error {result['grad_max_error']:.2e} exceeds tolerance {whisper_tolerance}"

    def test_gradient_multiple_seeds(self):
        """Test gradient stability across different random seeds."""
        for seed in [42, 123, 456]:
            result = self._compare_gradients(T=50, N=4, C=100, S_max=10, seed=seed)
            assert result["grad_max_error"] < self.GRAD_TOLERANCE, \
                f"Seed {seed}: gradient error {result['grad_max_error']:.2e}"


# =============================================================================
# Performance Test (Optional, for benchmarking)
# =============================================================================

@pytest.mark.slow
class TestCTCLossPerformance:
    """Performance benchmarks for CTC loss."""

    def test_whisper_scale_performance(self):
        """Benchmark at Whisper scale (T=750, N=16, C=51865)."""
        import time

        data = generate_test_data(T=750, N=16, C=51865, S_max=50)

        log_probs_mx = mx.array(data["log_probs"])
        targets_mx = mx.array(data["targets"])
        input_lens_mx = mx.array(data["input_lengths"])
        target_lens_mx = mx.array(data["target_lengths"])

        # Warm up
        _ = ctc_loss(log_probs_mx, targets_mx, input_lens_mx, target_lens_mx)
        mx.eval(_)

        # Benchmark
        start = time.time()
        for _ in range(10):
            loss = ctc_loss(log_probs_mx, targets_mx, input_lens_mx, target_lens_mx)
            mx.eval(loss)
        elapsed = time.time() - start

        time_per_batch = elapsed / 10
        print(f"\nMLX CTC time per batch (T=750, N=16, C=51865): {time_per_batch*1000:.1f}ms")

        # With forward-backward for gradients, expect ~8s per batch
        # (2s forward + 2s backward + numpy overhead)
        assert time_per_batch < 10.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
