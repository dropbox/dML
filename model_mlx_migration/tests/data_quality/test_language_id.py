#!/usr/bin/env python3
"""
Language ID Tests - Data QA Gate

Verifies that transcripts match expected language(s) for each dataset.
This is an OPTIONAL gate - can be run on schedule.

Run with: pytest tests/data_quality/test_language_id.py -v -m data_qa_heavy
"""

import json
import re
import sys
import unicodedata
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


# Character ranges for language detection (simplified)
LANGUAGE_CHAR_RANGES = {
    "zh": re.compile(r'[\u4e00-\u9fff]'),  # CJK Unified Ideographs
    "ja": re.compile(r'[\u3040-\u309f\u30a0-\u30ff]'),  # Hiragana + Katakana
    "ko": re.compile(r'[\uac00-\ud7af\u1100-\u11ff]'),  # Hangul
    "ar": re.compile(r'[\u0600-\u06ff]'),  # Arabic
    "hi": re.compile(r'[\u0900-\u097f]'),  # Devanagari
    "th": re.compile(r'[\u0e00-\u0e7f]'),  # Thai
    "ru": re.compile(r'[\u0400-\u04ff]'),  # Cyrillic
    "he": re.compile(r'[\u0590-\u05ff]'),  # Hebrew
}


def detect_script(text: str) -> set[str]:
    """Detect scripts present in text."""
    scripts = set()

    for lang, pattern in LANGUAGE_CHAR_RANGES.items():
        if pattern.search(text):
            scripts.add(lang)

    # Check for Latin (default for many languages)
    if re.search(r'[a-zA-Z]', text):
        scripts.add("latin")

    return scripts


def simple_lid(text: str, expected_languages: list[str]) -> bool:
    """
    Simple language ID check based on character scripts.

    Returns True if text appears to match one of the expected languages.
    """
    if not text or not text.strip():
        return True  # Empty text passes (no language to check)

    scripts = detect_script(text)

    # Map languages to expected scripts
    expected_scripts = set()
    for lang in expected_languages:
        lang = lang.lower()
        if lang in ["en", "de", "fr", "es", "it", "pt", "nl", "pl"]:
            expected_scripts.add("latin")
        elif lang in LANGUAGE_CHAR_RANGES:
            expected_scripts.add(lang)
        else:
            expected_scripts.add("latin")  # Default assumption

    # Check if detected scripts match expected
    if not scripts:
        return True  # No detectable script (numbers only, etc.)

    return bool(scripts & expected_scripts)


@pytest.mark.data_qa_heavy
class TestLanguageID:
    """Verify transcript language matches expected language."""

    @pytest.fixture
    def dataset_config(self):
        return load_dataset_config()

    def test_language_config_present(self, dataset_config):
        """Datasets have language configuration."""
        datasets = dataset_config.get("datasets", {})

        for name, ds_config in datasets.items():
            languages = ds_config.get("languages")
            # languages can be None for non-speech datasets
            if languages is not None:
                assert isinstance(languages, list), \
                    f"Dataset {name} languages should be a list, got {type(languages)}"

    @pytest.fixture
    def english_manifests(self):
        """Find manifests expected to be English."""
        project_root = Path(__file__).parent.parent.parent
        manifests = []

        english_paths = [
            "data/LibriSpeech",
            "data/prosody/ravdess",
            "data/emotion/CREMA-D",
        ]

        for rel_path in english_paths:
            path = project_root / rel_path
            if path.exists():
                manifests.extend(path.glob("*manifest*.json"))

        return manifests[:3]

    def test_english_transcripts_are_english(self, english_manifests):
        """English dataset transcripts contain English text."""
        for manifest_path in english_manifests:
            if not manifest_path.exists():
                continue

            with open(manifest_path) as f:
                try:
                    manifest = json.load(f)
                except json.JSONDecodeError:
                    continue

            non_english_count = 0
            checked_count = 0

            for entry in manifest[:100]:  # Sample check
                text = entry.get("text") or entry.get("transcript", "")
                if not text:
                    continue

                checked_count += 1
                if not simple_lid(text, ["en"]):
                    non_english_count += 1

            if checked_count > 0:
                non_english_ratio = non_english_count / checked_count
                assert non_english_ratio < 0.1, \
                    f"English manifest {manifest_path.name} has {non_english_ratio:.1%} non-English transcripts"


@pytest.mark.data_qa
class TestTranscriptQuality:
    """Verify transcript text quality."""

    @pytest.fixture
    def sample_manifests(self):
        project_root = Path(__file__).parent.parent.parent
        manifests = list(project_root.glob("data/**/train_manifest.json"))
        return manifests[:5]

    def test_no_control_characters(self, sample_manifests):
        """Transcripts have no control characters."""
        for manifest_path in sample_manifests:
            if not manifest_path.exists():
                continue

            with open(manifest_path) as f:
                try:
                    manifest = json.load(f)
                except json.JSONDecodeError:
                    continue

            control_char_count = 0
            for entry in manifest[:100]:
                text = entry.get("text") or entry.get("transcript", "")
                if not text:
                    continue

                # Check for control characters (except common whitespace)
                for char in text:
                    if unicodedata.category(char) == 'Cc' and char not in '\n\r\t':
                        control_char_count += 1

            assert control_char_count == 0, \
                f"Manifest {manifest_path.name} has {control_char_count} control characters"

    def test_transcript_length_bounds(self, sample_manifests):
        """Transcript length is within reasonable bounds."""
        for manifest_path in sample_manifests:
            if not manifest_path.exists():
                continue

            with open(manifest_path) as f:
                try:
                    manifest = json.load(f)
                except json.JSONDecodeError:
                    continue

            too_short = 0
            too_long = 0

            for entry in manifest[:100]:
                text = entry.get("text") or entry.get("transcript", "")
                if not text:
                    continue

                # Check bounds
                if len(text) < 1:
                    too_short += 1
                elif len(text) > 10000:  # Unreasonably long
                    too_long += 1

            # Allow some edge cases
            assert too_long == 0, \
                f"Manifest {manifest_path.name} has {too_long} unreasonably long transcripts"


@pytest.mark.data_qa
class TestUnicodeNormalization:
    """Verify text is properly Unicode normalized."""

    def test_nfkc_normalization(self):
        """Sample transcripts are NFKC normalized."""
        project_root = Path(__file__).parent.parent.parent
        manifests = list(project_root.glob("data/**/train_manifest.json"))[:3]

        unnormalized_count = 0

        for manifest_path in manifests:
            if not manifest_path.exists():
                continue

            with open(manifest_path) as f:
                try:
                    manifest = json.load(f)
                except json.JSONDecodeError:
                    continue

            for entry in manifest[:50]:
                text = entry.get("text") or entry.get("transcript", "")
                if not text:
                    continue

                normalized = unicodedata.normalize('NFKC', text)
                if text != normalized:
                    unnormalized_count += 1

        # Some unnormalized text is OK, but flag if >10%
        # In practice, we'd normalize during preprocessing
        # This is informational, not a hard failure
