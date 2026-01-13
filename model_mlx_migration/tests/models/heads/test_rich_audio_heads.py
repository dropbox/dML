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
Tests for RichAudioHeads multi-head wrapper.

This is a lightweight integration test to ensure the shared encoder output can be
fanned out to all Phase 5 heads with consistent keying and shapes.
"""

import mlx.core as mx

from src.models.heads import (
    CORE_LANGUAGES,
    IPA_PHONEMES,
    PARALINGUISTIC_CLASSES,
    SINGING_TECHNIQUES,
    RichAudioHeads,
    RichAudioHeadsConfig,
)


def test_rich_audio_heads_output_shapes():
    mx.random.seed(0)

    batch_size = 3
    seq_len = 80
    encoder_dim = 384

    model = RichAudioHeads(RichAudioHeadsConfig(encoder_dim=encoder_dim))

    encoder_out = mx.random.normal((batch_size, seq_len, encoder_dim))
    encoder_lengths = mx.array([80, 60, 40])

    asr_text = ["hello world", "testing speech", ""]
    audio_energy = mx.full((batch_size, seq_len), -10.0)

    outputs = model(
        encoder_out,
        encoder_lengths,
        asr_text=asr_text,
        audio_energy=audio_energy,
    )

    assert outputs["emotion_logits"].shape == (batch_size, 8)
    assert outputs["language_logits"].shape == (batch_size, len(CORE_LANGUAGES))
    assert outputs["paralinguistics_logits"].shape == (batch_size, len(PARALINGUISTIC_CLASSES))

    assert outputs["pitch_f0_hz"].shape == (batch_size, seq_len)
    assert outputs["pitch_voiced_logits"].shape == (batch_size, seq_len)

    assert outputs["phoneme_logits"].shape == (batch_size, seq_len, len(IPA_PHONEMES))

    assert outputs["singing_binary_logits"].shape == (batch_size, 1)
    assert outputs["singing_technique_logits"].shape == (batch_size, len(SINGING_TECHNIQUES))

    assert outputs["timestamp_boundary_logits"].shape == (batch_size, seq_len, 1)
    assert outputs["timestamp_offset_preds"].shape == (batch_size, seq_len, 2)

    assert outputs["hallucination_scores"].shape == (batch_size,)
    assert mx.all(outputs["hallucination_scores"] >= 0.0)
    assert mx.all(outputs["hallucination_scores"] <= 1.0)

    # Phase 6: Speaker embeddings
    assert outputs["speaker_embeddings"].shape == (batch_size, 256)  # Default embedding dim
    # Check embeddings are normalized (L2 norm ~= 1)
    norms = mx.sqrt(mx.sum(outputs["speaker_embeddings"] ** 2, axis=-1))
    assert mx.allclose(norms, mx.ones(batch_size), atol=1e-5)


def test_rich_audio_heads_speaker_disabled():
    """Test that speaker head can be disabled."""
    mx.random.seed(1)

    batch_size = 2
    seq_len = 50
    encoder_dim = 384

    config = RichAudioHeadsConfig(encoder_dim=encoder_dim, speaker_enabled=False)
    model = RichAudioHeads(config)

    encoder_out = mx.random.normal((batch_size, seq_len, encoder_dim))

    outputs = model(encoder_out)

    # Speaker embeddings should not be present when disabled
    assert "speaker_embeddings" not in outputs

    # Other outputs should still be present
    assert "emotion_logits" in outputs
    assert "phoneme_logits" in outputs

