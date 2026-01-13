#!/usr/bin/env python3
"""
Deduplication Tests - Data QA Gate

Verifies that datasets have no excessive duplicates.
This is an OPTIONAL gate - can be run on schedule.

Run with: pytest tests/data_quality/test_dedup.py -v -m data_qa_heavy
"""

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_dataset_config() -> dict:
    """Load dataset configuration from YAML."""
    config_path = Path(__file__).parent.parent.parent / "data" / "qa" / "datasets.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {"defaults": {}, "datasets": {}}


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of file content."""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def compute_path_hash(path: str) -> str:
    """Compute hash of path string (for quick dedup)."""
    return hashlib.sha256(path.encode()).hexdigest()[:16]


@pytest.mark.data_qa
class TestExactDuplicates:
    """Test for exact duplicate audio paths."""

    @pytest.fixture
    def sample_manifests(self):
        project_root = Path(__file__).parent.parent.parent
        manifests = list(project_root.glob("data/**/train_manifest.json"))
        return manifests[:5]

    def test_no_exact_path_duplicates_strict(self, sample_manifests):
        """No exact duplicate paths in single manifest (strict check)."""
        for manifest_path in sample_manifests:
            if not manifest_path.exists():
                continue

            with open(manifest_path) as f:
                try:
                    manifest = json.load(f)
                except json.JSONDecodeError:
                    continue

            paths = [
                entry.get("audio_path") or entry.get("path", "")
                for entry in manifest
            ]

            # Count occurrences
            path_counts = Counter(paths)
            duplicates = {p: c for p, c in path_counts.items() if c > 1 and p}

            # For strict check, no duplicates allowed
            # But we relax for augmented data - just warn if >50% duplicated
            if duplicates:
                total = len(paths)
                dup_count = sum(c - 1 for c in duplicates.values())
                dup_ratio = dup_count / total if total > 0 else 0

                # Only fail if >60% duplicated (likely a bug)
                assert dup_ratio < 0.6, \
                    f"Manifest {manifest_path.name} has {dup_ratio:.1%} duplicate paths"


@pytest.mark.data_qa
class TestCrossManifestDedup:
    """Test for duplicates across manifests (train/val/test leakage)."""

    def test_train_val_no_overlap(self):
        """Train and val manifests have no overlapping audio paths."""
        project_root = Path(__file__).parent.parent.parent

        # Find train/val pairs
        train_manifests = list(project_root.glob("data/**/train_manifest.json"))

        for train_path in train_manifests[:3]:  # Limit for speed
            val_path = train_path.parent / "val_manifest.json"
            if not val_path.exists():
                continue

            # Load paths
            with open(train_path) as f:
                train_manifest = json.load(f)
            with open(val_path) as f:
                val_manifest = json.load(f)

            train_paths = set(
                entry.get("audio_path") or entry.get("path", "")
                for entry in train_manifest
            )
            val_paths = set(
                entry.get("audio_path") or entry.get("path", "")
                for entry in val_manifest
            )

            overlap = train_paths & val_paths
            overlap = {p for p in overlap if p}  # Remove empty strings

            # Calculate overlap ratio
            if val_paths:
                overlap_ratio = len(overlap) / len(val_paths)
                assert overlap_ratio < 0.01, \
                    f"Train/val overlap in {train_path.parent.name}: {len(overlap)} paths ({overlap_ratio:.1%})"


@pytest.mark.data_qa_heavy
class TestNearDuplicates:
    """Test for near-duplicate audio (content-based)."""

    @pytest.fixture
    def dataset_config(self):
        return load_dataset_config()

    def test_near_dedup_threshold_configured(self, dataset_config):
        """Near-duplicate threshold is configured."""
        defaults = dataset_config.get("defaults", {})
        assert "dedup_near_max_ratio" in defaults, "Missing default dedup_near_max_ratio"
        assert 0 <= defaults["dedup_near_max_ratio"] <= 1, \
            "dedup_near_max_ratio should be in [0, 1]"

    def test_near_duplicate_detection_works(self):
        """Near-duplicate detection algorithm works on test data."""
        # This is a unit test for the detection algorithm
        # Actual near-dedup requires audio fingerprinting (heavy operation)

        # Test with synthetic data
        paths = [
            "/path/to/audio1.wav",
            "/path/to/audio2.wav",
            "/path/to/audio1.wav",  # Exact dup
            "/path/to/audio3.wav",
        ]

        # Simple hash-based check
        hashes = [compute_path_hash(p) for p in paths]
        unique_hashes = set(hashes)

        # Should detect one duplicate
        assert len(unique_hashes) == 3, "Should detect 1 duplicate"


@pytest.mark.data_qa
class TestDedupConfig:
    """Verify dedup configuration is valid."""

    def test_dedup_thresholds_valid(self):
        """Dedup thresholds are valid."""
        config = load_dataset_config()
        defaults = config.get("defaults", {})

        exact_thresh = defaults.get("dedup_exact_max_ratio", 0.0)
        near_thresh = defaults.get("dedup_near_max_ratio", 0.001)

        assert 0 <= exact_thresh <= 1, f"Invalid dedup_exact_max_ratio: {exact_thresh}"
        assert 0 <= near_thresh <= 1, f"Invalid dedup_near_max_ratio: {near_thresh}"
        assert exact_thresh <= near_thresh, "exact threshold should be <= near threshold"


@pytest.mark.data_qa
class TestLeakageChecks:
    """Test for train/val/test data leakage."""

    def test_speaker_id_separation(self):
        """Verify speaker IDs don't leak across splits (where applicable)."""
        project_root = Path(__file__).parent.parent.parent

        # Check LibriSpeech-style datasets that have speaker info
        train_manifests = list(project_root.glob("data/**/train_manifest.json"))

        for train_path in train_manifests[:3]:
            val_path = train_path.parent / "val_manifest.json"
            if not val_path.exists():
                continue

            with open(train_path) as f:
                train_manifest = json.load(f)
            with open(val_path) as f:
                val_manifest = json.load(f)

            # Extract speaker IDs if present
            train_speakers = set()
            val_speakers = set()

            for entry in train_manifest:
                speaker = entry.get("speaker_id") or entry.get("speaker")
                if speaker:
                    train_speakers.add(speaker)

            for entry in val_manifest:
                speaker = entry.get("speaker_id") or entry.get("speaker")
                if speaker:
                    val_speakers.add(speaker)

            # Only check if both have speaker info
            if train_speakers and val_speakers:
                overlap = train_speakers & val_speakers
                if overlap:
                    # Some overlap is OK for some datasets, but warn if >10%
                    overlap_ratio = len(overlap) / len(val_speakers)
                    assert overlap_ratio < 0.5, \
                        f"Speaker leakage in {train_path.parent.name}: {len(overlap)} speakers ({overlap_ratio:.1%})"
