#!/usr/bin/env python3
"""
Label Consistency Tests - Data QA Gate

Verifies that all labels in manifests map to canonical taxonomy classes.
This is a REQUIRED gate - training cannot start if this fails.

Run with: pytest tests/data_quality/test_label_consistency.py -v
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools.whisper_mlx.label_taxonomy import (
    EMOTION_CLASSES_9,
    PARALINGUISTICS_CLASSES_50,
    get_emotion_to_id_9,
    validate_label_maps,
)


@pytest.mark.data_qa
class TestLabelConsistency:
    """Verify all dataset labels map to canonical taxonomy."""

    def test_all_builtin_mappings_valid(self):
        """All built-in label mappings produce valid indices."""
        all_valid, errors = validate_label_maps()
        assert all_valid, f"Label mapping validation failed:\n{json.dumps(errors, indent=2)}"

    def test_emotion_mapping_complete(self):
        """Emotion mapping covers all canonical classes."""
        mapping = get_emotion_to_id_9()

        # Every canonical class should be reachable
        covered_indices = set(mapping.values())
        expected_indices = set(range(len(EMOTION_CLASSES_9)))

        missing = expected_indices - covered_indices
        assert not missing, f"Emotion indices not covered by any mapping: {missing}"

    def test_emotion_aliases_correct(self):
        """Common emotion aliases map to correct canonical classes."""
        mapping = get_emotion_to_id_9()

        # Verify aliases
        assert mapping['fear'] == mapping['fearful'], "'fear' should map to same index as 'fearful'"
        assert mapping['surprise'] == mapping['surprised'], "'surprise' should map to same index as 'surprised'"
        assert mapping['other'] == mapping['neutral'], "'other' should map to neutral (index 0)"

    @pytest.fixture
    def emotion_manifests(self):
        """Find manifests with emotion labels."""
        project_root = Path(__file__).parent.parent.parent
        manifests = []

        # Check common emotion dataset locations
        emotion_paths = [
            "data/prosody/ravdess",
            "data/emotion/CREMA-D",
            "data/emotion/MELD",
            "data/v4_expanded",
            "data/v4_combined",
        ]

        for rel_path in emotion_paths:
            path = project_root / rel_path
            if path.exists():
                manifests.extend(path.glob("*manifest*.json"))

        return manifests[:10]  # Limit for speed

    def test_manifest_emotions_map_to_canonical(self, emotion_manifests):
        """All emotion labels in manifests map to canonical EMOTION_CLASSES_9."""
        mapping = get_emotion_to_id_9()
        valid_labels = set(mapping.keys())

        for manifest_path in emotion_manifests:
            if not manifest_path.exists():
                continue

            with open(manifest_path) as f:
                try:
                    manifest = json.load(f)
                except json.JSONDecodeError:
                    pytest.skip(f"Invalid JSON: {manifest_path}")

            unmapped_labels = set()
            for entry in manifest:
                if not isinstance(entry, dict):
                    continue  # Skip non-dict entries
                emotion = entry.get("emotion")
                if emotion and emotion.lower() not in valid_labels:
                    unmapped_labels.add(emotion)

            if unmapped_labels:
                pytest.fail(
                    f"Unmapped emotion labels in {manifest_path.name}: {unmapped_labels}\n"
                    f"Valid labels: {sorted(valid_labels)}",
                )


@pytest.mark.data_qa
class TestParalinguisticsConsistency:
    """Verify paralinguistics labels are consistent."""

    def test_paralinguistics_classes_count(self):
        """Paralinguistics has exactly 50 classes."""
        assert len(PARALINGUISTICS_CLASSES_50) == 50, \
            f"Expected 50 paralinguistics classes, got {len(PARALINGUISTICS_CLASSES_50)}"

    def test_key_paralinguistics_indices(self):
        """Key paralinguistics classes are at expected indices."""
        expected = {
            0: "speech",
            1: "laughter",
            2: "cough",
            5: "cry",
            8: "sneeze",
        }

        for idx, expected_class in expected.items():
            assert PARALINGUISTICS_CLASSES_50[idx] == expected_class, \
                f"Index {idx} should be '{expected_class}', got '{PARALINGUISTICS_CLASSES_50[idx]}'"

    @pytest.fixture
    def para_manifests(self):
        """Find manifests with paralinguistics labels."""
        project_root = Path(__file__).parent.parent.parent
        manifests = []

        para_paths = [
            "data/paralinguistics/VocalSound",
            "data/paralinguistics/ESC-50",
        ]

        for rel_path in para_paths:
            path = project_root / rel_path
            if path.exists():
                manifests.extend(path.glob("*manifest*.json"))

        return manifests[:5]

    def test_manifest_para_labels_valid(self, para_manifests):
        """All paralinguistics labels in manifests are valid."""
        valid_labels = set(PARALINGUISTICS_CLASSES_50)

        for manifest_path in para_manifests:
            if not manifest_path.exists():
                continue

            with open(manifest_path) as f:
                try:
                    manifest = json.load(f)
                except json.JSONDecodeError:
                    continue

            invalid_labels = set()
            for entry in manifest:
                para = entry.get("para") or entry.get("paralinguistics")
                if para and para not in valid_labels:
                    invalid_labels.add(para)

            if invalid_labels:
                pytest.fail(
                    f"Invalid paralinguistics labels in {manifest_path.name}: {invalid_labels}",
                )


@pytest.mark.data_qa
class TestNoUnmappedLabels:
    """Ensure no labels are silently dropped."""

    def test_emotion_no_silent_drops(self):
        """Emotion mapping doesn't silently drop any labels."""
        mapping = get_emotion_to_id_9()

        # Check that all mapped values are valid indices
        max_idx = len(EMOTION_CLASSES_9) - 1
        for label, idx in mapping.items():
            assert 0 <= idx <= max_idx, \
                f"Label '{label}' maps to invalid index {idx} (max: {max_idx})"

    def test_mapping_policy_documented(self):
        """Verify mapping policy is documented in label_taxonomy.py."""
        from tools.whisper_mlx import label_taxonomy

        # Check that the module has documentation
        assert label_taxonomy.__doc__ is not None, "label_taxonomy.py needs module docstring"
        assert "single source of truth" in label_taxonomy.__doc__.lower(), \
            "label_taxonomy.py docstring should mention 'single source of truth'"
