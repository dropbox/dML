#!/usr/bin/env python3
"""
File Integrity Tests - Data QA Gate

Verifies that audio files exist, are readable, and meet format requirements.
This is a REQUIRED gate - training cannot start if this fails.

Run with: pytest tests/data_quality/test_integrity.py -v
"""

import json
import sys
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


@pytest.mark.data_qa
class TestFileIntegrity:
    """Verify audio files exist and are valid."""

    @pytest.fixture
    def dataset_config(self):
        return load_dataset_config()

    @pytest.fixture
    def sample_manifests(self):
        """Find sample manifests for testing."""
        project_root = Path(__file__).parent.parent.parent
        manifests = list(project_root.glob("data/**/train_manifest.json"))
        manifests.extend(project_root.glob("data/**/manifest.json"))
        return manifests[:5]  # Limit for speed

    def test_manifest_audio_paths_exist(self, sample_manifests):
        """Audio files referenced in manifests exist."""
        project_root = Path(__file__).parent.parent.parent
        missing_count = 0
        checked_count = 0
        max_check = 100  # Sample check, not exhaustive

        for manifest_path in sample_manifests:
            if not manifest_path.exists():
                continue

            with open(manifest_path) as f:
                try:
                    manifest = json.load(f)
                except json.JSONDecodeError:
                    continue

            for entry in manifest[:max_check]:
                audio_path = entry.get("audio_path") or entry.get("path")
                if not audio_path:
                    continue

                checked_count += 1

                # Resolve relative paths
                if not audio_path.startswith("/"):
                    full_path = project_root / audio_path
                else:
                    full_path = Path(audio_path)

                if not full_path.exists():
                    missing_count += 1

        if checked_count > 0:
            missing_ratio = missing_count / checked_count
            assert missing_ratio < 0.05, \
                f"Too many missing audio files: {missing_count}/{checked_count} ({missing_ratio:.1%})"

    def test_manifest_has_required_fields(self, sample_manifests):
        """Manifests contain minimum required fields."""
        required_fields = {"audio_path"}  # At minimum, audio path is required
        alternative_fields = {"path"}  # Some manifests use 'path' instead

        for manifest_path in sample_manifests:
            if not manifest_path.exists():
                continue

            with open(manifest_path) as f:
                try:
                    manifest = json.load(f)
                except json.JSONDecodeError:
                    pytest.skip(f"Invalid JSON: {manifest_path}")

            if not manifest:
                continue

            # Check first few entries
            for entry in manifest[:10]:
                has_required = any(
                    field in entry for field in required_fields.union(alternative_fields)
                )
                assert has_required, \
                    f"Entry in {manifest_path.name} missing audio path field: {entry.keys()}"


@pytest.mark.data_qa
class TestAudioFormat:
    """Verify audio format meets requirements."""

    def test_wav_file_readable(self):
        """Sample WAV files are readable."""
        project_root = Path(__file__).parent.parent.parent

        # Find a few sample WAV files
        wav_files = list(project_root.glob("tests/*.wav"))[:3]

        for wav_path in wav_files:
            if not wav_path.exists():
                continue

            # Basic readability check - file should be non-empty
            assert wav_path.stat().st_size > 44, \
                f"WAV file too small (possibly empty): {wav_path}"

            # Check WAV header
            with open(wav_path, 'rb') as f:
                header = f.read(4)
                assert header == b'RIFF', f"Invalid WAV header in {wav_path}"


@pytest.mark.data_qa
class TestDurationBounds:
    """Verify audio durations are within expected bounds."""

    @pytest.fixture
    def dataset_config(self):
        return load_dataset_config()

    def test_duration_bounds_configured(self, dataset_config):
        """Dataset config has duration bounds."""
        defaults = dataset_config.get("defaults", {})

        assert "duration_min_sec" in defaults, "Missing default duration_min_sec"
        assert "duration_max_sec" in defaults, "Missing default duration_max_sec"

        assert defaults["duration_min_sec"] > 0, "duration_min_sec must be positive"
        assert defaults["duration_max_sec"] > defaults["duration_min_sec"], \
            "duration_max_sec must be > duration_min_sec"


@pytest.mark.data_qa
class TestManifestConsistency:
    """Verify manifest internal consistency."""

    @pytest.fixture
    def sample_manifests(self):
        project_root = Path(__file__).parent.parent.parent
        manifests = list(project_root.glob("data/**/train_manifest.json"))
        return manifests[:5]

    def test_no_null_entries(self, sample_manifests):
        """Manifests have no null entries."""
        for manifest_path in sample_manifests:
            if not manifest_path.exists():
                continue

            with open(manifest_path) as f:
                try:
                    manifest = json.load(f)
                except json.JSONDecodeError:
                    continue

            null_count = sum(1 for entry in manifest if entry is None)
            assert null_count == 0, \
                f"Manifest {manifest_path.name} has {null_count} null entries"

    def test_no_empty_audio_paths(self, sample_manifests):
        """Audio paths are non-empty strings."""
        for manifest_path in sample_manifests:
            if not manifest_path.exists():
                continue

            with open(manifest_path) as f:
                try:
                    manifest = json.load(f)
                except json.JSONDecodeError:
                    continue

            for i, entry in enumerate(manifest[:100]):
                audio_path = entry.get("audio_path") or entry.get("path", "")
                assert audio_path and isinstance(audio_path, str), \
                    f"Entry {i} in {manifest_path.name} has invalid audio path: {audio_path!r}"
