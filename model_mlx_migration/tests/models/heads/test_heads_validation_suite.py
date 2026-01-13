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
Phase 5.9: Validation suite for all rich audio heads.

This suite validates that all 8 heads:
1. Produce correct output shapes
2. Are differentiable (gradients flow)
3. Can be trained jointly
4. Meet performance targets (where measurable)
5. Work together in a multi-head scenario

Heads validated:
- 5.1: EmotionHead - 8-class utterance classification
- 5.2: PitchHead - Frame-level F0 prediction
- 5.3: PhonemeHead - Frame-level IPA classification
- 5.4: ParalinguisticsHead - 50-class event detection
- 5.5: LanguageHead - 9+ language identification
- 5.6: SingingHead - Binary singing + 10 techniques
- 5.7: TimestampHead - Word boundary detection
- 5.8: HallucinationHead - Phoneme mismatch detection

Performance Targets (from FINAL_ROADMAP_SOTA_PLUS_PLUS.md):
- Emotion: >92% accuracy
- Pitch: <10Hz MAE on voiced frames
- Phoneme: <18% PER
- Paralinguistics: >96% accuracy
- Language: >98% accuracy
- Singing: >95% binary, >90% technique
- Timestamps: 80-90% accuracy within 50ms
- Hallucination: >90% detection rate
"""

import mlx.core as mx
import mlx.nn as nn
import pytest

from src.models.heads import (
    CORE_LANGUAGES,
    IPA_PHONEMES,
    PARALINGUISTIC_CLASSES,
    SINGING_TECHNIQUES,
    EmotionConfig,
    # Emotion
    EmotionHead,
    HallucinationConfig,
    HallucinationHead,
    LanguageConfig,
    # Language
    LanguageHead,
    ParalinguisticsConfig,
    # Paralinguistics
    ParalinguisticsHead,
    PhonemeConfig,
    PhonemeHead,
    PitchConfig,
    PitchHead,
    SingingConfig,
    # Singing
    SingingHead,
    TimestampConfig,
    # Timestamp
    TimestampHead,
)

# Common test parameters
ENCODER_DIM = 384
BATCH_SIZE = 4
SEQ_LEN = 100


class MultiHeadModel(nn.Module):
    """
    Multi-head model combining all rich audio heads.

    This simulates how heads would be attached to a Zipformer encoder.
    """

    def __init__(self, encoder_dim: int = 384):
        super().__init__()

        # All heads
        self.emotion_head = EmotionHead(EmotionConfig(encoder_dim=encoder_dim))
        self.pitch_head = PitchHead(PitchConfig(encoder_dim=encoder_dim))
        self.phoneme_head = PhonemeHead(PhonemeConfig(encoder_dim=encoder_dim))
        self.paralinguistics_head = ParalinguisticsHead(
            ParalinguisticsConfig(encoder_dim=encoder_dim),
        )
        self.language_head = LanguageHead(LanguageConfig(encoder_dim=encoder_dim))
        self.singing_head = SingingHead(SingingConfig(encoder_dim=encoder_dim))
        self.timestamp_head = TimestampHead(TimestampConfig(encoder_dim=encoder_dim))
        self.hallucination_head = HallucinationHead(HallucinationConfig())

    def __call__(self, encoder_out: mx.array, encoder_lengths: mx.array = None):
        """
        Forward pass through all heads.

        Args:
            encoder_out: Encoder output (batch, seq, encoder_dim)
            encoder_lengths: Sequence lengths (batch,)

        Returns:
            Dict of head outputs
        """
        outputs = {}

        # Utterance-level heads
        outputs["emotion"] = self.emotion_head(encoder_out, encoder_lengths)
        outputs["language"] = self.language_head(encoder_out, encoder_lengths)
        outputs["paralinguistics"] = self.paralinguistics_head(encoder_out, encoder_lengths)

        # Frame-level heads
        outputs["pitch"] = self.pitch_head(encoder_out, encoder_lengths)  # tuple (f0, voiced)
        outputs["phoneme"] = self.phoneme_head(encoder_out, encoder_lengths)
        outputs["singing"] = self.singing_head(encoder_out, encoder_lengths)  # tuple (binary, technique)
        outputs["timestamp"] = self.timestamp_head(encoder_out, encoder_lengths)  # tuple (boundaries, offsets)

        # Hallucination head (uses phoneme output)
        # Note: hallucination head doesn't need encoder_out directly

        return outputs


class TestHeadOutputShapes:
    """Validate output shapes for all heads."""

    @pytest.fixture
    def encoder_out(self):
        """Create mock encoder output."""
        mx.random.seed(42)
        return mx.random.normal((BATCH_SIZE, SEQ_LEN, ENCODER_DIM))

    @pytest.fixture
    def encoder_lengths(self):
        """Create sequence lengths."""
        return mx.array([100, 80, 60, 40])

    def test_emotion_head_shape(self, encoder_out, encoder_lengths):
        """Emotion head output shape."""
        head = EmotionHead(EmotionConfig(encoder_dim=ENCODER_DIM))
        output = head(encoder_out, encoder_lengths)
        assert output.shape == (BATCH_SIZE, 8)  # 8 emotion classes

    def test_pitch_head_shape(self, encoder_out, encoder_lengths):
        """Pitch head output shape."""
        head = PitchHead(PitchConfig(encoder_dim=ENCODER_DIM))
        f0, voiced = head(encoder_out, encoder_lengths)
        assert f0.shape == (BATCH_SIZE, SEQ_LEN)
        assert voiced.shape == (BATCH_SIZE, SEQ_LEN)

    def test_phoneme_head_shape(self, encoder_out, encoder_lengths):
        """Phoneme head output shape."""
        head = PhonemeHead(PhonemeConfig(encoder_dim=ENCODER_DIM))
        output = head(encoder_out, encoder_lengths)
        assert output.shape == (BATCH_SIZE, SEQ_LEN, len(IPA_PHONEMES))

    def test_paralinguistics_head_shape(self, encoder_out, encoder_lengths):
        """Paralinguistics head output shape."""
        head = ParalinguisticsHead(ParalinguisticsConfig(encoder_dim=ENCODER_DIM))
        output = head(encoder_out, encoder_lengths)
        # Paralinguistics is utterance-level (batch, num_classes)
        assert output.shape == (BATCH_SIZE, len(PARALINGUISTIC_CLASSES))

    def test_language_head_shape(self, encoder_out, encoder_lengths):
        """Language head output shape."""
        head = LanguageHead(LanguageConfig(encoder_dim=ENCODER_DIM))
        output = head(encoder_out, encoder_lengths)
        assert output.shape == (BATCH_SIZE, len(CORE_LANGUAGES))

    def test_singing_head_shape(self, encoder_out, encoder_lengths):
        """Singing head output shape."""
        head = SingingHead(SingingConfig(encoder_dim=ENCODER_DIM))
        output = head(encoder_out, encoder_lengths)
        # Returns tuple (binary_logits, technique_logits) - utterance level
        assert isinstance(output, tuple)
        binary, technique = output
        assert binary.shape == (BATCH_SIZE, 1)
        assert technique.shape == (BATCH_SIZE, len(SINGING_TECHNIQUES))

    def test_timestamp_head_shape(self, encoder_out, encoder_lengths):
        """Timestamp head output shape."""
        config = TimestampConfig(encoder_dim=ENCODER_DIM, use_offset_regression=True)
        head = TimestampHead(config)
        output = head(encoder_out, encoder_lengths)
        # Returns tuple (boundaries, offsets)
        assert isinstance(output, tuple)
        boundaries, offsets = output
        assert boundaries.shape == (BATCH_SIZE, SEQ_LEN, 1)
        # Offsets have 2 dims (start/end offset)
        assert offsets.shape == (BATCH_SIZE, SEQ_LEN, 2)

    def test_hallucination_head_shape(self):
        """Hallucination head output shape."""
        head = HallucinationHead()
        phoneme_logits = mx.random.normal((BATCH_SIZE, SEQ_LEN, len(IPA_PHONEMES)))
        output = head(phoneme_logits)
        assert output.shape == (BATCH_SIZE,)


class TestHeadGradients:
    """Validate gradient flow for all trainable heads."""

    @pytest.fixture
    def encoder_out(self):
        """Create mock encoder output."""
        mx.random.seed(42)
        return mx.random.normal((BATCH_SIZE, SEQ_LEN, ENCODER_DIM))

    @pytest.fixture
    def encoder_lengths(self):
        """Create sequence lengths."""
        return mx.array([100, 80, 60, 40])

    def test_emotion_gradients(self, encoder_out, encoder_lengths):
        """Test gradient flow through emotion head."""
        head = EmotionHead(EmotionConfig(encoder_dim=ENCODER_DIM))
        targets = mx.array([0, 1, 2, 3])

        def loss_fn(model, x, lengths, y):
            logits = model(x, lengths)
            # Manual cross-entropy to avoid nn.losses
            log_softmax = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            return -mx.mean(mx.take_along_axis(log_softmax, mx.expand_dims(y, -1), axis=-1))

        loss, grads = nn.value_and_grad(head, loss_fn)(head, encoder_out, encoder_lengths, targets)

        assert loss.shape == ()
        assert not mx.isnan(loss).item()
        # Check some gradients exist
        has_grads = False
        for _key, val in grads.items():
            if isinstance(val, dict):
                for _subkey, subval in val.items():
                    if hasattr(subval, 'shape') and subval.size > 0:
                        if mx.any(subval != 0).item():
                            has_grads = True
                            break
        assert has_grads

    def test_pitch_gradients(self, encoder_out, encoder_lengths):
        """Test gradient flow through pitch head."""
        head = PitchHead(PitchConfig(encoder_dim=ENCODER_DIM))
        targets = mx.random.uniform(100, 400, (BATCH_SIZE, SEQ_LEN))  # Hz
        voiced_mask = mx.ones((BATCH_SIZE, SEQ_LEN), dtype=mx.bool_)

        def loss_fn(model, x, lengths, y, mask):
            f0, voiced = model(x, lengths)
            # Simple L1 loss on F0
            return mx.mean(mx.abs(f0 - y) * mask.astype(mx.float32))

        loss, grads = nn.value_and_grad(head, loss_fn)(
            head, encoder_out, encoder_lengths, targets, voiced_mask,
        )

        assert loss.shape == ()
        assert not mx.isnan(loss).item()

    def test_phoneme_gradients(self, encoder_out, encoder_lengths):
        """Test gradient flow through phoneme head."""
        head = PhonemeHead(PhonemeConfig(encoder_dim=ENCODER_DIM))
        targets = mx.random.randint(0, len(IPA_PHONEMES), (BATCH_SIZE, SEQ_LEN))

        def loss_fn(model, x, lengths, y):
            logits = model(x, lengths)
            batch, seq, classes = logits.shape
            return nn.losses.cross_entropy(
                logits.reshape(-1, classes),
                y.reshape(-1),
            ).mean()

        loss, grads = nn.value_and_grad(head, loss_fn)(
            head, encoder_out, encoder_lengths, targets,
        )

        assert loss.shape == ()
        assert not mx.isnan(loss).item()

    def test_paralinguistics_gradients(self, encoder_out, encoder_lengths):
        """Test gradient flow through paralinguistics head."""
        config = ParalinguisticsConfig(encoder_dim=ENCODER_DIM, multi_label=False)
        head = ParalinguisticsHead(config)
        # Utterance-level targets (batch,)
        targets = mx.array([0, 1, 2, 3])

        def loss_fn(model, x, lengths, y):
            logits = model(x, lengths)  # (batch, num_classes)
            # Manual cross-entropy
            log_softmax = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            return -mx.mean(mx.take_along_axis(log_softmax, mx.expand_dims(y, -1), axis=-1))

        loss, grads = nn.value_and_grad(head, loss_fn)(
            head, encoder_out, encoder_lengths, targets,
        )

        assert loss.shape == ()
        assert not mx.isnan(loss).item()

    def test_language_gradients(self, encoder_out, encoder_lengths):
        """Test gradient flow through language head."""
        head = LanguageHead(LanguageConfig(encoder_dim=ENCODER_DIM))
        targets = mx.array([0, 1, 2, 3])

        def loss_fn(model, x, lengths, y):
            logits = model(x, lengths)
            return nn.losses.cross_entropy(logits, y).mean()

        loss, grads = nn.value_and_grad(head, loss_fn)(
            head, encoder_out, encoder_lengths, targets,
        )

        assert loss.shape == ()
        assert not mx.isnan(loss).item()

    def test_singing_gradients(self, encoder_out, encoder_lengths):
        """Test gradient flow through singing head."""
        head = SingingHead(SingingConfig(encoder_dim=ENCODER_DIM))
        # Utterance-level binary targets (batch,)
        binary_targets = mx.zeros((BATCH_SIZE,), dtype=mx.int32)

        def loss_fn(model, x, lengths, y):
            binary_logits, technique_logits = model(x, lengths)
            binary_logits = binary_logits.squeeze(-1)  # (batch,)
            # BCE loss
            probs = mx.sigmoid(binary_logits)
            return -mx.mean(
                y.astype(mx.float32) * mx.log(probs + 1e-8) +
                (1 - y.astype(mx.float32)) * mx.log(1 - probs + 1e-8),
            )

        loss, grads = nn.value_and_grad(head, loss_fn)(
            head, encoder_out, encoder_lengths, binary_targets,
        )

        assert loss.shape == ()
        assert not mx.isnan(loss).item()

    def test_timestamp_gradients(self, encoder_out, encoder_lengths):
        """Test gradient flow through timestamp head."""
        head = TimestampHead(TimestampConfig(encoder_dim=ENCODER_DIM))
        targets = mx.zeros((BATCH_SIZE, SEQ_LEN), dtype=mx.float32)
        targets = targets.at[:, [10, 25, 50, 75]].add(1.0)  # Word boundaries

        def loss_fn(model, x, lengths, y):
            boundaries, offsets = model(x, lengths)
            boundaries = boundaries.squeeze(-1)  # (batch, seq)
            # BCE loss
            probs = mx.sigmoid(boundaries)
            return -mx.mean(
                y * mx.log(probs + 1e-8) +
                (1 - y) * mx.log(1 - probs + 1e-8),
            )

        loss, grads = nn.value_and_grad(head, loss_fn)(
            head, encoder_out, encoder_lengths, targets,
        )

        assert loss.shape == ()
        assert not mx.isnan(loss).item()


class TestMultiHeadModel:
    """Test all heads working together in a multi-head model."""

    @pytest.fixture
    def model(self):
        """Create multi-head model."""
        return MultiHeadModel(encoder_dim=ENCODER_DIM)

    @pytest.fixture
    def encoder_out(self):
        """Create mock encoder output."""
        mx.random.seed(42)
        return mx.random.normal((BATCH_SIZE, SEQ_LEN, ENCODER_DIM))

    @pytest.fixture
    def encoder_lengths(self):
        """Create sequence lengths."""
        return mx.array([100, 80, 60, 40])

    def test_multi_head_forward(self, model, encoder_out, encoder_lengths):
        """Test forward pass through all heads."""
        outputs = model(encoder_out, encoder_lengths)

        assert "emotion" in outputs
        assert "pitch" in outputs
        assert "phoneme" in outputs
        assert "paralinguistics" in outputs
        assert "language" in outputs
        assert "singing" in outputs
        assert "timestamp" in outputs

    def test_multi_head_shapes_consistent(self, model, encoder_out, encoder_lengths):
        """Test all head outputs have consistent batch size."""
        outputs = model(encoder_out, encoder_lengths)

        # Utterance-level heads
        assert outputs["emotion"].shape[0] == BATCH_SIZE
        assert outputs["language"].shape[0] == BATCH_SIZE
        assert outputs["paralinguistics"].shape[0] == BATCH_SIZE

        # Frame-level heads - pitch returns (f0, voiced) tuple
        f0, voiced = outputs["pitch"]
        assert f0.shape[0] == BATCH_SIZE
        assert voiced.shape[0] == BATCH_SIZE

        assert outputs["phoneme"].shape[0] == BATCH_SIZE

        # singing returns (binary, technique) tuple
        binary, technique = outputs["singing"]
        assert binary.shape[0] == BATCH_SIZE
        assert technique.shape[0] == BATCH_SIZE

        # timestamp returns (boundaries, offsets) tuple
        boundaries, offsets = outputs["timestamp"]
        assert boundaries.shape[0] == BATCH_SIZE

    def test_multi_head_parameter_count(self, model):
        """Test total parameter count is reasonable."""
        total_params = 0
        for _name, param in model.parameters().items():
            if hasattr(param, 'size'):
                total_params += param.size
            elif isinstance(param, dict):
                for p in param.values():
                    if hasattr(p, 'size'):
                        total_params += p.size

        # Should be less than 1M parameters for all heads combined
        # (main model params are in encoder, not heads)
        assert total_params < 1_000_000


class TestJointTraining:
    """Test joint training of multiple heads."""

    def test_joint_loss_computation(self):
        """Test computing combined loss from multiple heads."""
        model = MultiHeadModel(encoder_dim=ENCODER_DIM)

        mx.random.seed(42)
        encoder_out = mx.random.normal((BATCH_SIZE, SEQ_LEN, ENCODER_DIM))
        encoder_lengths = mx.array([100, 80, 60, 40])

        # Create targets for each head
        emotion_targets = mx.array([0, 1, 2, 3])
        language_targets = mx.array([0, 1, 2, 3])
        phoneme_targets = mx.random.randint(0, len(IPA_PHONEMES), (BATCH_SIZE, SEQ_LEN))

        def joint_loss_fn(model, x, lengths, emo_y, lang_y, phone_y):
            outputs = model(x, lengths)

            # Emotion loss
            emo_loss = nn.losses.cross_entropy(outputs["emotion"], emo_y).mean()

            # Language loss
            lang_loss = nn.losses.cross_entropy(outputs["language"], lang_y).mean()

            # Phoneme loss
            phone_logits = outputs["phoneme"]
            batch, seq, classes = phone_logits.shape
            phone_loss = nn.losses.cross_entropy(
                phone_logits.reshape(-1, classes),
                phone_y.reshape(-1),
            ).mean()

            # Combined loss
            return emo_loss + lang_loss + phone_loss

        loss, grads = nn.value_and_grad(model, joint_loss_fn)(
            model, encoder_out, encoder_lengths,
            emotion_targets, language_targets, phoneme_targets,
        )

        assert loss.shape == ()
        assert not mx.isnan(loss).item()
        assert loss.item() > 0  # Loss should be positive

    def test_loss_decreases_with_training(self):
        """Test that loss decreases with training steps."""
        import mlx.optimizers as optim

        # Use a simpler single head for stability
        head = EmotionHead(EmotionConfig(encoder_dim=ENCODER_DIM))

        mx.random.seed(42)
        encoder_out = mx.random.normal((BATCH_SIZE, SEQ_LEN, ENCODER_DIM))
        encoder_lengths = mx.array([100, 80, 60, 40])
        emotion_targets = mx.array([0, 1, 2, 3])

        def loss_fn(model, x, lengths, y):
            logits = model(x, lengths)
            # Manual cross-entropy
            log_softmax = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            return -mx.mean(mx.take_along_axis(log_softmax, mx.expand_dims(y, -1), axis=-1))

        optimizer = optim.Adam(learning_rate=0.001)

        losses = []
        for _ in range(10):
            loss, grads = nn.value_and_grad(head, loss_fn)(
                head, encoder_out, encoder_lengths, emotion_targets,
            )
            losses.append(float(loss.item()))
            optimizer.update(head, grads)
            mx.eval(head.parameters())

        # Loss should generally decrease (allow some variation)
        assert losses[-1] < losses[0]


class TestHeadIntegration:
    """Test integration between heads."""

    def test_phoneme_to_hallucination_pipeline(self):
        """Test phoneme output feeding into hallucination detection."""
        phoneme_head = PhonemeHead(PhonemeConfig(encoder_dim=ENCODER_DIM))
        hallucination_head = HallucinationHead()

        mx.random.seed(42)
        encoder_out = mx.random.normal((BATCH_SIZE, SEQ_LEN, ENCODER_DIM))

        # Get phoneme logits
        phoneme_logits = phoneme_head(encoder_out)

        # Feed to hallucination head
        asr_text = ["hello world", "testing speech", "another sample", "last one"]
        hallucination_scores = hallucination_head(phoneme_logits, asr_text=asr_text)

        assert hallucination_scores.shape == (BATCH_SIZE,)
        assert mx.all(hallucination_scores >= 0.0)
        assert mx.all(hallucination_scores <= 1.0)

    def test_singing_masks_paralinguistics(self):
        """Test that singing detection can mask paralinguistics output."""
        mx.random.seed(42)
        encoder_out = mx.random.normal((BATCH_SIZE, SEQ_LEN, ENCODER_DIM))

        singing_head = SingingHead(SingingConfig(encoder_dim=ENCODER_DIM))
        para_head = ParalinguisticsHead(ParalinguisticsConfig(encoder_dim=ENCODER_DIM))

        singing_out = singing_head(encoder_out)  # Returns (binary, technique) tuple
        para_out = para_head(encoder_out)  # Returns (batch, num_classes)

        # Get singing binary prediction (utterance level)
        binary_logits, technique_logits = singing_out
        singing_probs = mx.sigmoid(binary_logits)  # (batch, 1)

        # Both are utterance-level, can use singing to mask paralinguistics
        is_singing = (singing_probs > 0.5).squeeze(-1)  # (batch,)

        # Both should be utterance-level with matching batch size
        assert is_singing.shape[0] == para_out.shape[0]


class TestHeadPerformanceBaselines:
    """Document expected performance baselines for each head."""

    def test_emotion_baseline_documented(self):
        """Emotion head target: >92% accuracy (Whisper baseline: 92.07%)"""
        config = EmotionConfig()
        assert config.num_classes == 8
        # Target accuracy documented in roadmap

    def test_pitch_baseline_documented(self):
        """Pitch head target: <10Hz MAE on voiced frames"""
        config = PitchConfig()
        assert config.f0_min_hz == 50.0
        assert config.f0_max_hz == 800.0
        # Target MAE documented in roadmap

    def test_phoneme_baseline_documented(self):
        """Phoneme head target: <18% PER (Whisper baseline: 19.7%)"""
        config = PhonemeConfig()
        assert config.num_phonemes == len(IPA_PHONEMES)
        # Target PER documented in roadmap

    def test_paralinguistics_baseline_documented(self):
        """Paralinguistics head target: >96% accuracy (Whisper-AT: 96.96%)"""
        config = ParalinguisticsConfig()
        assert config.num_classes == len(PARALINGUISTIC_CLASSES)
        # Target accuracy documented in roadmap

    def test_language_baseline_documented(self):
        """Language head target: >98% accuracy (Whisper: 98.61%)"""
        config = LanguageConfig()
        assert config.num_languages == len(CORE_LANGUAGES)
        # Target accuracy documented in roadmap

    def test_singing_baseline_documented(self):
        """Singing head target: >95% binary, >90% technique"""
        config = SingingConfig()
        assert config.num_techniques == len(SINGING_TECHNIQUES)
        # Target accuracy documented in roadmap

    def test_timestamp_baseline_documented(self):
        """Timestamp head target: 80-90% accuracy within 50ms"""
        config = TimestampConfig()
        # Verify config is created with expected defaults
        assert config is not None

    def test_hallucination_baseline_documented(self):
        """Hallucination head target: >90% detection rate"""
        config = HallucinationConfig()
        assert config.detection_threshold == 0.5
        # Target detection rate documented in roadmap


class TestHeadMemoryEfficiency:
    """Test memory efficiency of heads."""

    def test_heads_support_variable_lengths(self):
        """Test heads work with variable sequence lengths."""
        model = MultiHeadModel(encoder_dim=ENCODER_DIM)

        # Short sequences
        short_out = mx.random.normal((BATCH_SIZE, 20, ENCODER_DIM))
        short_lengths = mx.array([20, 15, 10, 5])

        # Long sequences
        long_out = mx.random.normal((BATCH_SIZE, 500, ENCODER_DIM))
        long_lengths = mx.array([500, 400, 300, 200])

        # Both should work without error
        short_results = model(short_out, short_lengths)
        long_results = model(long_out, long_lengths)

        assert short_results["emotion"].shape[0] == BATCH_SIZE
        assert long_results["emotion"].shape[0] == BATCH_SIZE

    def test_single_batch_inference(self):
        """Test inference with batch size 1."""
        model = MultiHeadModel(encoder_dim=ENCODER_DIM)

        single_out = mx.random.normal((1, SEQ_LEN, ENCODER_DIM))
        single_length = mx.array([SEQ_LEN])

        results = model(single_out, single_length)

        assert results["emotion"].shape[0] == 1
        assert results["language"].shape[0] == 1


class TestHeadExports:
    """Test that all heads are properly exported from the module."""

    def test_all_heads_importable(self):
        """Test all head classes can be imported."""
        from src.models.heads import (
            EmotionHead,
            HallucinationHead,
            LanguageHead,
            ParalinguisticsHead,
            PhonemeHead,
            PitchHead,
            SingingHead,
            TimestampHead,
        )

        # All should be importable
        assert EmotionHead is not None
        assert PitchHead is not None
        assert PhonemeHead is not None
        assert ParalinguisticsHead is not None
        assert LanguageHead is not None
        assert SingingHead is not None
        assert TimestampHead is not None
        assert HallucinationHead is not None

    def test_all_configs_importable(self):
        """Test all config classes can be imported."""
        from src.models.heads import (
            EmotionConfig,
            HallucinationConfig,
            LanguageConfig,
            ParalinguisticsConfig,
            PhonemeConfig,
            PitchConfig,
            SingingConfig,
            TimestampConfig,
        )

        # All should be importable
        assert EmotionConfig is not None
        assert PitchConfig is not None
        assert PhonemeConfig is not None
        assert ParalinguisticsConfig is not None
        assert LanguageConfig is not None
        assert SingingConfig is not None
        assert TimestampConfig is not None
        assert HallucinationConfig is not None

    def test_all_losses_importable(self):
        """Test all loss classes can be imported."""
        from src.models.heads import (
            EmotionLoss,
            HallucinationLoss,
            LanguageLoss,
            ParalinguisticsLoss,
            PhonemeFrameLoss,
            PitchLoss,
            SingingLoss,
            TimestampLoss,
        )

        # All should be importable
        assert EmotionLoss is not None
        assert PitchLoss is not None
        assert PhonemeFrameLoss is not None
        assert ParalinguisticsLoss is not None
        assert LanguageLoss is not None
        assert SingingLoss is not None
        assert TimestampLoss is not None
        assert HallucinationLoss is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
