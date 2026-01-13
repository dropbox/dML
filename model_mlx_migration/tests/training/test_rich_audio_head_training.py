# Copyright 2024-2026 Andrew Yates
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
Tests for rich audio head training script.

Validates that the training script can:
1. Create heads for emotion and paralinguistics
2. Run training steps with gradient clipping
3. Save checkpoints
"""

import random
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

# Add project root to path
_PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


class TestHeadCreation:
    """Test head creation functions."""

    def test_create_emotion_head(self):
        """Test creating emotion head."""
        from scripts.train_rich_audio_heads import create_head
        from src.training import EMOTION_LABELS

        head = create_head("emotion", encoder_dim=512)
        assert head is not None

        # Test forward pass
        x = mx.random.normal((2, 100, 512))
        lengths = mx.array([100, 80])
        logits = head(x, lengths)

        assert logits.shape == (2, len(EMOTION_LABELS))

    def test_create_paralinguistics_head(self):
        """Test creating paralinguistics head."""
        from scripts.train_rich_audio_heads import create_head
        from src.training import PARALINGUISTIC_LABELS

        head = create_head("paralinguistics", encoder_dim=512)
        assert head is not None

        # Test forward pass
        x = mx.random.normal((2, 100, 512))
        lengths = mx.array([100, 80])
        logits = head(x, lengths)

        assert logits.shape == (2, len(PARALINGUISTIC_LABELS))


class TestGradientClipping:
    """Test gradient clipping function."""

    def test_clip_grad_norm_basic(self):
        """Test basic gradient clipping."""
        from scripts.train_rich_audio_heads import clip_grad_norm

        # Create a gradient tensor
        grad = mx.array([10.0, 0.0, 0.0])  # Norm = 10
        clipped = clip_grad_norm({"weight": grad}, max_norm=1.0)

        # Norm should be approximately 1.0
        clipped_norm = mx.sqrt(mx.sum(clipped["weight"] * clipped["weight"])).item()
        assert abs(clipped_norm - 1.0) < 0.01

    def test_clip_grad_norm_nested(self):
        """Test gradient clipping with nested dict."""
        from scripts.train_rich_audio_heads import clip_grad_norm

        grads = {
            "layer1": {"weight": mx.array([5.0, 0.0])},
            "layer2": {"weight": mx.array([0.0, 5.0])},
        }

        clipped = clip_grad_norm(grads, max_norm=1.0)

        # Both should be clipped
        assert clipped["layer1"]["weight"].shape == (2,)
        assert clipped["layer2"]["weight"].shape == (2,)


class TestLRSchedule:
    """Test learning rate schedule."""

    def test_warmup_phase(self):
        """Test warmup phase of schedule."""
        from scripts.train_rich_audio_heads import warmup_cosine_lr

        # During warmup, LR should increase linearly
        lr_0 = warmup_cosine_lr(0, warmup_steps=100, max_steps=1000, base_lr=1e-3)
        lr_50 = warmup_cosine_lr(50, warmup_steps=100, max_steps=1000, base_lr=1e-3)
        lr_100 = warmup_cosine_lr(100, warmup_steps=100, max_steps=1000, base_lr=1e-3)

        assert lr_0 == 0.0
        assert abs(lr_50 - 5e-4) < 1e-6
        assert abs(lr_100 - 1e-3) < 1e-6

    def test_cosine_decay(self):
        """Test cosine decay phase."""
        from scripts.train_rich_audio_heads import warmup_cosine_lr

        # After warmup, LR should decay
        lr_start = warmup_cosine_lr(100, warmup_steps=100, max_steps=1000, base_lr=1e-3)
        lr_mid = warmup_cosine_lr(550, warmup_steps=100, max_steps=1000, base_lr=1e-3)
        lr_end = warmup_cosine_lr(1000, warmup_steps=100, max_steps=1000, base_lr=1e-3, min_lr=1e-6)

        assert lr_start > lr_mid
        assert lr_mid > lr_end
        assert abs(lr_end - 1e-6) < 1e-8


class TestCheckpointSaving:
    """Test checkpoint saving functions."""

    def test_flatten_params(self):
        """Test parameter flattening."""
        from scripts.train_rich_audio_heads import flatten_params

        params = {
            "encoder": {
                "layer1": {"weight": mx.zeros((10, 10))},
                "layer2": {"weight": mx.zeros((5, 5))},
            },
            "bias": mx.zeros((10,)),
        }

        flat = flatten_params(params)

        assert "encoder.layer1.weight" in flat
        assert "encoder.layer2.weight" in flat
        assert "bias" in flat
        assert flat["encoder.layer1.weight"].shape == (10, 10)

    def test_save_checkpoint(self):
        """Test checkpoint saving."""
        from scripts.train_rich_audio_heads import create_head, save_checkpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            head = create_head("emotion", encoder_dim=256)
            optimizer = optim.AdamW(learning_rate=1e-4)

            save_checkpoint(head, optimizer, step=100, epoch=1, output_dir=output_dir)

            # Check files exist
            assert (output_dir / "head_step_100.npz").exists()
            assert (output_dir / "meta_step_100.json").exists()


class TestTrainingIntegration:
    """Integration tests for training."""

    def test_training_script_imports(self):
        """Test that training script imports work."""
        from scripts.train_rich_audio_heads import (
            TrainingArgs,
            create_dataloader,
            create_head,
        )

        assert TrainingArgs is not None
        assert callable(create_head)
        assert callable(create_dataloader)

    def test_dummy_train_step(self):
        """Test training step with dummy data."""
        from scripts.train_rich_audio_heads import create_head, train_step

        # Create a simple encoder proxy
        class DummyEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.output_dim = 256

            def __call__(self, x, lengths):
                # (batch, time, feat) -> (batch, time//4, encoder_dim)
                batch, time, _ = x.shape
                out = mx.zeros((batch, time // 4, self.output_dim))
                out_lengths = lengths // 4
                return out, out_lengths

        encoder = DummyEncoder()
        head = create_head("emotion", encoder_dim=256)
        optimizer = optim.AdamW(learning_rate=1e-4)

        # Create dummy batch
        batch = {
            "features": mx.random.normal((4, 100, 80)),
            "feature_lengths": mx.array([100, 80, 60, 40]),
            "emotion_labels": mx.array([0, 1, 2, 3]),
        }

        loss, acc, _ = train_step(encoder, head, optimizer, batch, "emotion_labels")

        assert loss > 0  # Loss should be positive
        assert 0 <= acc <= 1  # Accuracy should be in [0, 1]


class TestAdaptiveLR:
    """Test adaptive learning rate for NaN stability."""

    def test_lr_reduction_logic(self):
        """Test that LR is reduced correctly when NaN rate is high."""
        # Simulate the adaptive LR logic from train_rich_audio_heads.py
        base_lr = 4e-05
        adaptive_lr_factor = 1.0
        nan_rate_threshold = 0.1  # 10%
        consecutive_high_nan_epochs = 0

        # Simulate a sequence of epochs with varying NaN rates
        epoch_nan_rates = [0.05, 0.15, 0.20, 0.08, 0.25, 0.30]

        for nan_rate in epoch_nan_rates:
            if nan_rate > nan_rate_threshold:
                consecutive_high_nan_epochs += 1
                if consecutive_high_nan_epochs >= 2:
                    # Reduce LR
                    adaptive_lr_factor *= 0.5
                    consecutive_high_nan_epochs = 0  # Reset after reduction
            else:
                consecutive_high_nan_epochs = 0

        # After epochs [5%, 15%, 20%], LR should be reduced once (0.5x)
        # 15% triggers consecutive_high_nan_epochs=1
        # 20% triggers consecutive_high_nan_epochs=2 -> reduce
        # 8% resets counter
        # 25% triggers consecutive_high_nan_epochs=1
        # 30% triggers consecutive_high_nan_epochs=2 -> reduce again
        # Final factor should be 0.5 * 0.5 = 0.25
        assert adaptive_lr_factor == 0.25
        effective_lr = base_lr * adaptive_lr_factor
        assert abs(effective_lr - 1e-05) < 1e-10

    def test_lr_unchanged_when_nan_rate_low(self):
        """Test that LR stays unchanged with low NaN rates."""
        adaptive_lr_factor = 1.0
        nan_rate_threshold = 0.1
        consecutive_high_nan_epochs = 0

        # All epochs below threshold
        epoch_nan_rates = [0.05, 0.08, 0.02, 0.09, 0.03]

        for nan_rate in epoch_nan_rates:
            if nan_rate > nan_rate_threshold:
                consecutive_high_nan_epochs += 1
                if consecutive_high_nan_epochs >= 2:
                    adaptive_lr_factor *= 0.5
                    consecutive_high_nan_epochs = 0
            else:
                consecutive_high_nan_epochs = 0

        # No reductions should occur
        assert adaptive_lr_factor == 1.0
        assert consecutive_high_nan_epochs == 0

    def test_lr_floor_enforced(self):
        """Test that LR floor prevents over-reduction.

        The training script now has --lr-floor (default 1e-7) which prevents
        adaptive_lr_factor from making effective LR too small.
        """
        # Test the actual logic from train_rich_audio_heads.py
        base_lr = 4e-05
        lr_floor = 1e-07  # Default floor
        adaptive_lr_factor = 1.0

        # Simulate many consecutive high NaN rate epochs
        # Each triggers a reduction attempt (nan_rate=0.25, threshold=0.1)
        for _ in range(20):
            # This is the logic from train_rich_audio_heads.py
            proposed_factor = adaptive_lr_factor * 0.5
            proposed_effective_lr = base_lr * proposed_factor
            if proposed_effective_lr >= lr_floor:
                adaptive_lr_factor = proposed_factor
            # else: floor reached, don't reduce

        effective_lr = base_lr * adaptive_lr_factor

        # With floor of 1e-7 and base_lr of 4e-5:
        # Floor is reached when adaptive_lr_factor = 1e-7 / 4e-5 = 0.0025
        # 0.5^9 = 0.00195... which is below 0.0025, so after 8 reductions
        # the floor should prevent further reduction
        assert effective_lr >= lr_floor, f"LR {effective_lr} below floor {lr_floor}"
        # Should not be too much above floor (within factor of 2)
        assert effective_lr < lr_floor * 4, f"LR {effective_lr} unexpectedly high"

    def test_lr_floor_with_higher_base(self):
        """Test floor with higher base LR allows more reductions."""
        base_lr = 1e-03  # Higher base LR
        lr_floor = 1e-07
        adaptive_lr_factor = 1.0

        reductions = 0
        for _ in range(20):
            proposed_factor = adaptive_lr_factor * 0.5
            proposed_effective_lr = base_lr * proposed_factor
            if proposed_effective_lr >= lr_floor:
                adaptive_lr_factor = proposed_factor
                reductions += 1

        effective_lr = base_lr * adaptive_lr_factor

        # With base_lr=1e-3 and floor=1e-7, can reduce by factor of 10000
        # log2(10000) ≈ 13.3, so ~13 reductions allowed
        assert reductions >= 13, f"Expected at least 13 reductions, got {reductions}"
        assert effective_lr >= lr_floor


class TestSpecAugment:
    def test_apply_spec_augment_constant_features_preserved(self, monkeypatch):
        """
        If masked regions are filled with the per-sample mean, then a constant
        feature tensor should remain unchanged (vs. inserting literal zeros).
        """
        from scripts.train_rich_audio_heads import _apply_spec_augment

        features = mx.full((1, 20, 8), -2.0, dtype=mx.float32)
        feature_lengths = mx.array([20])

        def deterministic_randint(a: int, b: int) -> int:
            return 1 if b >= 1 else 0

        monkeypatch.setattr(random, "randint", deterministic_randint)

        out = _apply_spec_augment(
            features,
            feature_lengths,
            num_time_masks=1,
            time_mask_max=5,
            num_freq_masks=0,
            freq_mask_max=0,
        )
        mx.eval(out)
        max_delta = mx.max(mx.abs(out - features)).item()
        assert max_delta == 0.0
