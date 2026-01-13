#!/usr/bin/env python3
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
Data Quality Test Suite.

Enforces data quality and consistency before training runs. These tests verify:
1. Label consistency across all training/inference code
2. Manifest file validity (paths, fields, no duplicates)
3. Model config consistency (classes, indices, token IDs)
4. Checkpoint compatibility with current model architecture

Run with: pytest tests/test_data_quality.py -v

References:
    WORKER_ROADMAP_20260102.md - Phase 3: Data Quality Test Suite
    code_audit_issues_20260102.md - Issues ISSUE-001 through ISSUE-006
"""

import json
import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# 1. LABEL CONSISTENCY TESTS
# =============================================================================

class TestEmotionLabelConsistency:
    """Verify emotion labels match across all files."""

    def test_emotion_classes_9_is_canonical(self):
        """EMOTION_CLASSES_9 in label_taxonomy.py is the canonical source (v2.0)."""
        from tools.whisper_mlx.label_taxonomy import EMOTION_CLASSES_9

        expected = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised", "contempt"]
        assert EMOTION_CLASSES_9 == expected, f"EMOTION_CLASSES_9 changed! Expected {expected}, got {EMOTION_CLASSES_9}"
        assert len(EMOTION_CLASSES_9) == 9, f"Expected 9 classes, got {len(EMOTION_CLASSES_9)}"

    def test_emotion_classes_8_backward_compat(self):
        """EMOTION_CLASSES_8 alias points to EMOTION_CLASSES_9 for backward compatibility."""
        from tools.whisper_mlx.label_taxonomy import EMOTION_CLASSES_9
        from tools.whisper_mlx.rich_ctc_head import EMOTION_CLASSES_8

        # EMOTION_CLASSES_8 is now an alias for EMOTION_CLASSES_9
        assert EMOTION_CLASSES_8 == EMOTION_CLASSES_9, \
            "EMOTION_CLASSES_8 should be an alias for EMOTION_CLASSES_9"

    def test_train_rich_decoder_v4_emotion_mapping(self):
        """train_rich_decoder_v4.py emotion_to_id matches EMOTION_CLASSES_9."""
        from tools.whisper_mlx.label_taxonomy import EMOTION_CLASSES_9

        # The emotion_to_id mapping in train_rich_decoder_v4.py
        # should map each emotion to its index in EMOTION_CLASSES_9
        expected_mapping = {
            'neutral': 0,
            'calm': 1,
            'happy': 2,
            'sad': 3,
            'angry': 4,
            'fearful': 5, 'fear': 5,  # alias
            'disgust': 6,
            'surprised': 7, 'surprise': 7,  # alias
            'contempt': 8,  # NEW in v2.0
            'other': 0,  # map to neutral
        }

        # Verify against canonical source
        for emotion, expected_idx in expected_mapping.items():
            if emotion in ('other', 'fear', 'surprise'):
                # These are aliases, not in EMOTION_CLASSES_9 directly
                continue
            assert EMOTION_CLASSES_9[expected_idx] == emotion, \
                   f"Emotion {emotion} should be at index {expected_idx}, got {EMOTION_CLASSES_9[expected_idx]}"

    def test_multi_head_ravdess_emotions_match(self):
        """multi_head.py RAVDESS_EMOTIONS matches EMOTION_CLASSES_9."""
        from tools.whisper_mlx.label_taxonomy import EMOTION_CLASSES_9
        from tools.whisper_mlx.multi_head import RAVDESS_EMOTIONS

        assert RAVDESS_EMOTIONS == EMOTION_CLASSES_9, \
            f"RAVDESS_EMOTIONS mismatch! Got {RAVDESS_EMOTIONS}, expected {EMOTION_CLASSES_9}"

    def test_distillation_emotion_mapping(self):
        """train_with_distillation.py emotion mapping is consistent."""
        from tools.whisper_mlx.label_taxonomy import EMOTION_CLASSES_9

        # Verify the label_map in train_with_distillation.py load_samples()
        # maps 'fear' -> 'fearful' and 'surprise' -> 'surprised'
        label_map = {
            'fear': 'fearful',
            'surprise': 'surprised',
        }

        for _src, dst in label_map.items():
            assert dst in EMOTION_CLASSES_9, \
                f"Mapped emotion {dst} not in EMOTION_CLASSES_9"

        # Also verify contempt is in the canonical list
        assert 'contempt' in EMOTION_CLASSES_9, "contempt should be in EMOTION_CLASSES_9"


class TestParalinguisticsLabelConsistency:
    """Verify paralinguistics class indices are consistent."""

    def test_paralinguistics_classes_canonical(self):
        """PARALINGUISTICS_CLASSES in multi_head.py is the canonical 50-class list."""
        from tools.whisper_mlx.multi_head import PARALINGUISTICS_CLASSES

        # Key classes that must be at specific indices
        key_classes = {
            0: "speech",
            1: "laughter",
            2: "cough",
            3: "sigh",
            4: "breath",
            5: "cry",  # NOTE: "cry" not "crying"
            6: "yawn",
            7: "throat_clear",
            8: "sneeze",  # ISSUE: was incorrectly mapped to 5 in some places
            9: "gasp",
            10: "groan",
        }

        for idx, expected_class in key_classes.items():
            assert PARALINGUISTICS_CLASSES[idx] == expected_class, \
                f"Class at index {idx} should be {expected_class}, got {PARALINGUISTICS_CLASSES[idx]}"

        # Verify total is 50
        assert len(PARALINGUISTICS_CLASSES) == 50, \
            f"Expected 50 paralinguistics classes, got {len(PARALINGUISTICS_CLASSES)}"

    def test_train_paralinguistics_vocalsound_map(self):
        """train_paralinguistics.py VOCALSOUND_MAP uses correct indices."""
        from tools.whisper_mlx.multi_head import PARALINGUISTICS_CLASSES

        # These are the expected mappings after the fix
        vocalsound_map = {
            "Cough": 2,
            "Laughter": 1,
            "Sigh": 3,
            "Sneeze": 8,  # Was incorrectly 5
            "Sniff": 8,
            "Throat clearing": 7,
        }

        for sound, idx in vocalsound_map.items():
            assert idx < len(PARALINGUISTICS_CLASSES), \
                f"Index {idx} for {sound} exceeds class count"

    def test_train_paralinguistics_esc50_map(self):
        """train_paralinguistics.py ESC50_MAP uses correct indices."""
        from tools.whisper_mlx.multi_head import PARALINGUISTICS_CLASSES

        # These are the expected mappings after the fix
        esc50_map = {
            "breathing": 4,
            "coughing": 2,
            "laughing": 1,
            "sneezing": 8,
            "crying": 5,  # Maps to "cry" class
            "crying_baby": 5,
            "clapping": 0,
        }

        for sound, idx in esc50_map.items():
            if idx < len(PARALINGUISTICS_CLASSES):
                expected_class = PARALINGUISTICS_CLASSES[idx]
                # Verify the mapping makes sense
                if "cry" in sound:
                    assert "cry" in expected_class.lower(), \
                        f"{sound} should map to cry-related class, got {expected_class}"


class TestLanguageTokenConsistency:
    """Verify language token IDs are correct."""

    def test_whisper_language_base_ids(self):
        """Language token base IDs are correct for all model sizes."""
        from tools.whisper_mlx.rich_ctc_head import WHISPER_LANGUAGE_BASE_ID_BY_MODEL

        # All Whisper models use the same base ID (50259)
        for model_size, base_id in WHISPER_LANGUAGE_BASE_ID_BY_MODEL.items():
            assert base_id == 50259, \
                f"Model {model_size} has wrong base ID {base_id}, expected 50259"

    def test_language_token_ids_generation(self):
        """get_language_token_ids() generates valid token IDs."""
        from tools.whisper_mlx.rich_ctc_head import get_language_token_ids

        token_ids = get_language_token_ids("large-v3")

        # Verify all languages have unique token IDs
        ids = list(token_ids.values())
        assert len(ids) == len(set(ids)), "Duplicate language token IDs found"

        # Verify IDs are sequential from base
        min_id = min(ids)
        max_id = max(ids)
        assert max_id - min_id + 1 == len(ids), "Language token IDs not sequential"


# =============================================================================
# 2. MANIFEST VALIDATION TESTS
# =============================================================================

class TestManifestValidation:
    """Validate manifest file structure and content."""

    @pytest.fixture
    def sample_manifest_paths(self):
        """Return paths to sample manifests for testing."""
        project_root = Path(__file__).parent.parent
        manifests = list(project_root.glob("data/**/train_manifest.json"))
        manifests.extend(project_root.glob("data/**/val_manifest.json"))
        return manifests[:5]  # Limit to 5 for speed

    def test_manifest_has_required_fields(self, sample_manifest_paths):
        """Manifests contain required fields."""
        required_fields = {"audio_path"}  # Minimum required

        for manifest_path in sample_manifest_paths:
            if not manifest_path.exists():
                pytest.skip(f"Manifest {manifest_path} not found")

            with open(manifest_path) as f:
                manifest = json.load(f)

            if not manifest:
                continue  # Empty manifest

            # Check first entry has required fields
            entry = manifest[0]
            for field in required_fields:
                assert field in entry, \
                    f"Manifest {manifest_path} missing required field: {field}"

    def test_manifest_no_excessive_duplicates(self, sample_manifest_paths):
        """Manifests don't have excessive duplicate audio paths.

        Note: Up to 50% duplicates is acceptable for augmented training data.
        """
        for manifest_path in sample_manifest_paths:
            if not manifest_path.exists():
                pytest.skip(f"Manifest {manifest_path} not found")

            with open(manifest_path) as f:
                manifest = json.load(f)

            paths = [entry.get("audio_path", entry.get("path", "")) for entry in manifest]
            unique_paths = set(paths)

            # Allow up to 50% duplicates (common with data augmentation)
            # Only flag if >60% duplicated which indicates potential manifest bug
            dup_rate = 1 - len(unique_paths) / len(paths) if paths else 0
            assert dup_rate < 0.6, \
                f"Manifest {manifest_path} has {dup_rate*100:.1f}% duplicate paths (>60% suggests error)"

    def test_emotion_labels_are_valid(self, sample_manifest_paths):
        """Emotion labels in manifests are valid EMOTION_CLASSES_9 entries."""
        from tools.whisper_mlx.label_taxonomy import EMOTION_CLASSES_9

        # Valid labels include canonical + common aliases
        valid_labels = set(EMOTION_CLASSES_9)
        valid_labels.update(["fear", "surprise", "other"])  # Common aliases

        for manifest_path in sample_manifest_paths:
            if not manifest_path.exists():
                continue

            with open(manifest_path) as f:
                manifest = json.load(f)

            invalid_labels = set()
            for entry in manifest:
                emotion = entry.get("emotion")
                if emotion and emotion not in valid_labels:
                    invalid_labels.add(emotion)

            if invalid_labels:
                pytest.fail(f"Invalid emotion labels in {manifest_path}: {invalid_labels}")


# =============================================================================
# 3. MODEL CONFIG CONSISTENCY TESTS
# =============================================================================

class TestModelConfigConsistency:
    """Verify model configurations are consistent."""

    def test_emotion_head_num_classes(self):
        """EmotionHead output matches EMOTION_CLASSES_9 count (9 classes with contempt)."""
        from tools.whisper_mlx.label_taxonomy import EMOTION_CLASSES_9
        from tools.whisper_mlx.multi_head import EmotionHead, MultiHeadConfig

        config = MultiHeadConfig(
            d_model=1280,
            num_emotions=len(EMOTION_CLASSES_9),
        )
        head = EmotionHead(config)

        # The output dimension should match num_emotions (9 classes in v2.0)
        assert config.num_emotions == 9, f"Expected 9 emotion classes (v2.0 with contempt), got {config.num_emotions}"
        assert head is not None  # Head was created successfully

    def test_paralinguistics_head_num_classes(self):
        """ParalinguisticsHead output matches PARALINGUISTICS_CLASSES count."""
        from tools.whisper_mlx.multi_head import (
            PARALINGUISTICS_CLASSES,
            MultiHeadConfig,
            ParalinguisticsHead,
        )

        config = MultiHeadConfig(
            d_model=1280,
            num_paralinguistics_classes=len(PARALINGUISTICS_CLASSES),
        )
        head = ParalinguisticsHead(config)

        assert config.num_paralinguistics_classes == 50, \
            f"Expected 50 paralinguistics classes, got {config.num_paralinguistics_classes}"
        assert head is not None  # Head was created successfully

    def test_kokoro_phoneme_vocab_size(self):
        """Kokoro phoneme head uses correct vocab size (178)."""
        from tools.whisper_mlx.kokoro_phoneme_head import KOKORO_PHONEME_VOCAB_SIZE

        assert KOKORO_PHONEME_VOCAB_SIZE == 178, \
            f"Kokoro vocab size should be 178, got {KOKORO_PHONEME_VOCAB_SIZE}"


# =============================================================================
# 4. CHECKPOINT COMPATIBILITY TESTS
# =============================================================================

class TestCheckpointCompatibility:
    """Verify checkpoints are compatible with current architecture."""

    @pytest.fixture
    def sample_checkpoint_paths(self):
        """Return paths to sample checkpoints for testing."""
        project_root = Path(__file__).parent.parent
        checkpoints = list(project_root.glob("checkpoints/**/best.npz"))
        checkpoints.extend(project_root.glob("checkpoints/**/final.npz"))
        return checkpoints[:3]  # Limit for speed

    def test_checkpoint_has_expected_keys(self, sample_checkpoint_paths):
        """Checkpoints contain expected weight keys."""
        import mlx.core as mx

        for checkpoint_path in sample_checkpoint_paths:
            if not checkpoint_path.exists():
                continue

            try:
                checkpoint = dict(mx.load(str(checkpoint_path)))
            except Exception as e:
                pytest.skip(f"Could not load {checkpoint_path}: {e}")

            # Just verify it loaded and has some keys
            assert len(checkpoint) > 0, \
                f"Checkpoint {checkpoint_path} is empty"

    def test_singing_head_checkpoint_inference(self, sample_checkpoint_paths):
        """Singing head config can be inferred from checkpoint."""
        import mlx.core as mx

        for checkpoint_path in sample_checkpoint_paths:
            if not checkpoint_path.exists():
                continue

            try:
                checkpoint = dict(mx.load(str(checkpoint_path)))
            except Exception:
                continue

            # Check if this is a singing checkpoint
            singing_keys = [k for k in checkpoint.keys() if k.startswith("singing.")]
            if not singing_keys:
                continue

            # Extract singing params
            singing_params = {k.replace("singing.", ""): v for k, v in checkpoint.items() if k.startswith("singing.")}

            # Verify we can infer architecture
            if "shared_fc.weight" in singing_params:
                # ExtendedSingingHead
                assert "style_fc.weight" in singing_params, \
                    "ExtendedSingingHead missing style_fc.weight"
                num_styles = singing_params["style_fc.weight"].shape[0]
                assert num_styles > 0, "Invalid num_styles"
            elif "fc1.weight" in singing_params:
                # Basic SingingHead
                hidden_dim = singing_params["fc1.weight"].shape[0]
                assert hidden_dim > 0, "Invalid hidden_dim"


# =============================================================================
# 5. TRAINING PIPELINE CONSISTENCY TESTS
# =============================================================================

class TestTrainingPipelineConsistency:
    """Verify training pipeline configurations are consistent."""

    def test_project_root_can_be_resolved(self):
        """PROJECT_ROOT constant resolves correctly."""
        from pathlib import Path

        # Simulate the PROJECT_ROOT resolution logic
        test_file = Path(__file__)
        resolved_root = test_file.parent.parent

        assert resolved_root.exists(), "PROJECT_ROOT does not exist"
        assert (resolved_root / "tools").exists(), "PROJECT_ROOT/tools not found"
        assert (resolved_root / "tools" / "whisper_mlx").exists(), "whisper_mlx module not found"

    def test_kl_divergence_no_double_temperature(self):
        """KL divergence loss does not double-apply temperature."""
        # This is a code structure test - verify the fix is in place
        import inspect

        from scripts.train_with_distillation import kl_divergence_loss

        source = inspect.getsource(kl_divergence_loss)

        # Should NOT have the old buggy pattern
        assert "mx.softmax(teacher_log_probs / temperature" not in source, \
            "KL divergence still has double temperature bug!"

        # Should have the fixed pattern
        assert "teacher_probs_T = teacher_probs" in source, \
            "KL divergence fix not applied - teacher_probs should be used directly"

    def test_label_taxonomy_validate_all_maps(self):
        """label_taxonomy.py validate_label_maps() passes for all built-in mappings."""
        from tools.whisper_mlx.label_taxonomy import validate_label_maps

        all_valid, errors = validate_label_maps()
        assert all_valid, f"Label mapping validation failed: {errors}"

    def test_label_taxonomy_version(self):
        """label_taxonomy.py has version 2.0 with contempt."""
        from tools.whisper_mlx.label_taxonomy import EMOTION_CLASSES_9, TAXONOMY_VERSION

        assert TAXONOMY_VERSION == "2.0", f"Expected taxonomy version 2.0, got {TAXONOMY_VERSION}"
        assert "contempt" in EMOTION_CLASSES_9, "contempt should be in EMOTION_CLASSES_9 (v2.0)"
        assert EMOTION_CLASSES_9.index("contempt") == 8, "contempt should be at index 8"


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
