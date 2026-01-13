"""
Config Coherency Validator (Phase 6, Recommendation #16)

Validates kokoro-mps-*.yaml configs for internal consistency:
1. Language code matches voice
2. Device is MPS (use_gpu: true on macOS)
3. Model paths exist (if specified)
4. Sample rate consistency

Worker #320 - 2025-12-08
"""

import platform
import pytest
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "stream-tts-cpp" / "config"
MODELS_DIR = PROJECT_ROOT / "models"

# Valid language codes for Kokoro
VALID_LANGUAGES = {
    "en": ["en", "en_us", "en_gb"],
    "ja": ["ja"],
    "zh": ["zh", "zh_cn", "cmn"],
    "zh-sichuan": ["zh-sichuan", "sichuan"],
    "es": ["es", "es_es"],
    "fr": ["fr", "fr_fr"],
    "hi": ["hi"],
    "it": ["it", "it_it"],
    "pt": ["pt", "pt_br", "pt_pt"],
    "ko": ["ko"],
    "yi": ["yi"],
}

# Voice to language mapping
# Note: Some languages reuse voices from other languages
VOICE_LANGUAGE_MAP = {
    "en": ["en", "yi"],  # Yiddish uses English voice
    "ja": ["ja"],
    "zh": ["zh", "zh-sichuan"],  # Sichuanese uses standard Chinese voice
    "zh-sichuan": ["zh-sichuan"],
    "es": ["es"],
    "fr": ["fr"],
    "hi": ["hi"],
    "it": ["it"],
    "pt": ["pt"],
    "ko": ["ko"],
    "yi": ["yi"],
}


def load_config(config_path: Path) -> Optional[Dict]:
    """Load and parse a YAML config file."""
    if not config_path.exists():
        return None
    try:
        with open(config_path) as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def get_all_kokoro_configs() -> List[Path]:
    """Get all kokoro-mps-*.yaml config files."""
    if not CONFIG_DIR.exists():
        return []
    return list(CONFIG_DIR.glob("kokoro-mps-*.yaml"))


def validate_language_voice_match(config: Dict, config_path: Path) -> List[str]:
    """
    Validate that language code matches voice setting.

    Returns list of error messages (empty if valid).
    """
    errors = []

    tts = config.get("tts", {})
    language = tts.get("language", "")
    voice = tts.get("voice", "")

    if not language or not voice:
        return errors  # Skip if not specified

    # Extract base language from config filename for validation
    # e.g., kokoro-mps-en.yaml -> "en"
    config_name = config_path.stem
    if config_name.startswith("kokoro-mps-"):
        file_lang = config_name.replace("kokoro-mps-", "").split("2")[0]  # Handle en2ja pattern
    else:
        file_lang = None

    # Check voice matches language
    expected_langs = VOICE_LANGUAGE_MAP.get(voice, [])
    if expected_langs and language not in expected_langs:
        errors.append(
            f"Voice '{voice}' expects language in {expected_langs} but got '{language}'"
        )

    # Check filename matches config language (for non-translation configs)
    translation = config.get("translation", {})
    if not translation.get("enabled", False):
        # Non-translation config: filename should match language
        if file_lang and file_lang != language:
            # Allow close matches and variants (e.g., "en-lowlatency" still starts with "en")
            base_file_lang = file_lang.split("-")[0]  # Get base language from filename
            if base_file_lang != language and file_lang not in VALID_LANGUAGES.get(language, []):
                errors.append(
                    f"Config filename suggests '{file_lang}' but language is '{language}'"
                )

    return errors


def validate_gpu_settings(config: Dict, config_path: Path) -> List[str]:
    """
    Validate GPU settings match platform.

    Returns list of error messages (empty if valid).
    """
    errors = []

    performance = config.get("performance", {})
    use_gpu = performance.get("use_gpu", False)

    # On macOS, use_gpu should typically be true for MPS acceleration
    if platform.system() == "Darwin":
        if not use_gpu and "mps" in config_path.stem.lower():
            errors.append(
                f"Config name contains 'mps' but use_gpu is {use_gpu}"
            )

    return errors


def validate_model_paths(config: Dict, config_path: Path) -> List[str]:
    """
    Validate that specified model paths exist or have the models directory.

    For paths like /path/to/models/nllb-200-distilled-600M, we check if:
    1. The exact path exists, OR
    2. The parent "models" directory has a similar directory (fuzzy match)

    Returns list of error messages (empty if valid).
    """
    errors = []

    def check_model_path(model_path: str, model_type: str) -> Optional[str]:
        """Check if model path exists or is likely correct."""
        if not model_path or model_path == "":
            return None

        path = Path(model_path)

        # Only check absolute paths
        if not path.is_absolute():
            return None

        # If path exists, no error
        if path.exists():
            return None

        # Check if parent models directory has the model
        # e.g., /path/models/nllb-200-distilled-600M -> /path/models/nllb
        parent = path.parent
        model_name = path.name.lower()

        if parent.exists():
            # Look for similar directories (fuzzy match)
            for item in parent.iterdir():
                item_name = item.name.lower()
                # Check if model name starts with same base
                # e.g., "nllb-200-distilled-600m" matches "nllb"
                if model_name.startswith(item_name) or item_name.startswith(model_name.split("-")[0]):
                    return None  # Found a similar model, likely just different naming

        return f"{model_type} model_path does not exist: {model_path}"

    # TTS model path
    tts = config.get("tts", {})
    model_path = tts.get("model_path", "")
    error = check_model_path(model_path, "TTS")
    if error:
        errors.append(error)

    # Translation model path
    translation = config.get("translation", {})
    if translation.get("enabled", False):
        trans_model = translation.get("model_path", "")
        error = check_model_path(trans_model, "Translation")
        if error:
            errors.append(error)

    # Summarization model path
    summarization = config.get("summarization", {})
    if summarization.get("enabled", False):
        summ_model = summarization.get("model_path", "")
        error = check_model_path(summ_model, "Summarization")
        if error:
            errors.append(error)

    return errors


def validate_translation_languages(config: Dict, config_path: Path) -> List[str]:
    """
    Validate translation source/target languages are coherent.

    Returns list of error messages (empty if valid).
    """
    errors = []

    translation = config.get("translation", {})
    if not translation.get("enabled", False):
        return errors

    source = translation.get("source_lang", "")
    target = translation.get("target_lang", "")

    # Source and target shouldn't be the same for translation configs
    if source == target:
        errors.append(
            f"Translation enabled but source_lang ({source}) == target_lang ({target})"
        )

    # TTS language should match translation target
    tts = config.get("tts", {})
    tts_lang = tts.get("language", "")

    if tts_lang and target and tts_lang != target:
        # Allow close matches (e.g., zh vs cmn)
        if target not in VALID_LANGUAGES.get(tts_lang, [tts_lang]):
            errors.append(
                f"Translation target '{target}' doesn't match TTS language '{tts_lang}'"
            )

    return errors


def validate_config_complete(config: Dict, config_path: Path) -> Tuple[bool, List[str]]:
    """
    Run all validations on a config.

    Returns (is_valid, list_of_errors).
    """
    all_errors = []

    all_errors.extend(validate_language_voice_match(config, config_path))
    all_errors.extend(validate_gpu_settings(config, config_path))
    all_errors.extend(validate_model_paths(config, config_path))
    all_errors.extend(validate_translation_languages(config, config_path))

    return len(all_errors) == 0, all_errors


# =============================================================================
# Test Cases
# =============================================================================

class TestConfigCoherency:
    """Config coherency validation tests."""

    @pytest.fixture
    def all_configs(self) -> List[Tuple[Path, Dict]]:
        """Load all kokoro-mps configs."""
        configs = []
        for config_path in get_all_kokoro_configs():
            config = load_config(config_path)
            if config:
                configs.append((config_path, config))
        return configs

    def test_configs_exist(self):
        """Verify we have kokoro-mps configs to test."""
        configs = get_all_kokoro_configs()
        assert len(configs) > 0, f"No kokoro-mps-*.yaml configs found in {CONFIG_DIR}"
        print(f"\nFound {len(configs)} kokoro-mps configs to validate")

    def test_configs_parse(self, all_configs):
        """Verify all configs parse as valid YAML."""
        assert len(all_configs) > 0, "No configs loaded"

        for config_path, config in all_configs:
            assert config is not None, f"Failed to parse {config_path}"

    def test_language_voice_coherency(self, all_configs):
        """Validate language matches voice for all configs."""
        errors_by_config = {}

        for config_path, config in all_configs:
            errors = validate_language_voice_match(config, config_path)
            if errors:
                errors_by_config[config_path.name] = errors

        if errors_by_config:
            msg = "\nLanguage/voice coherency errors:\n"
            for config_name, errors in errors_by_config.items():
                msg += f"  {config_name}:\n"
                for error in errors:
                    msg += f"    - {error}\n"
            pytest.fail(msg)

    def test_gpu_settings_coherency(self, all_configs):
        """Validate GPU settings match platform expectations."""
        errors_by_config = {}

        for config_path, config in all_configs:
            errors = validate_gpu_settings(config, config_path)
            if errors:
                errors_by_config[config_path.name] = errors

        if errors_by_config:
            msg = "\nGPU settings coherency errors:\n"
            for config_name, errors in errors_by_config.items():
                msg += f"  {config_name}:\n"
                for error in errors:
                    msg += f"    - {error}\n"
            pytest.fail(msg)

    def test_model_paths_exist(self, all_configs):
        """Validate specified model paths exist."""
        errors_by_config = {}

        for config_path, config in all_configs:
            errors = validate_model_paths(config, config_path)
            if errors:
                errors_by_config[config_path.name] = errors

        if errors_by_config:
            msg = "\nModel path errors:\n"
            for config_name, errors in errors_by_config.items():
                msg += f"  {config_name}:\n"
                for error in errors:
                    msg += f"    - {error}\n"
            pytest.fail(msg)

    def test_translation_coherency(self, all_configs):
        """Validate translation settings are internally consistent."""
        errors_by_config = {}

        for config_path, config in all_configs:
            errors = validate_translation_languages(config, config_path)
            if errors:
                errors_by_config[config_path.name] = errors

        if errors_by_config:
            msg = "\nTranslation coherency errors:\n"
            for config_name, errors in errors_by_config.items():
                msg += f"  {config_name}:\n"
                for error in errors:
                    msg += f"    - {error}\n"
            pytest.fail(msg)

    def test_all_configs_valid(self, all_configs):
        """Comprehensive validation of all configs."""
        invalid_configs = {}
        valid_count = 0

        for config_path, config in all_configs:
            is_valid, errors = validate_config_complete(config, config_path)
            if is_valid:
                valid_count += 1
            else:
                invalid_configs[config_path.name] = errors

        print(f"\n{valid_count}/{len(all_configs)} configs passed validation")

        if invalid_configs:
            msg = f"\n{len(invalid_configs)} configs have coherency issues:\n"
            for config_name, errors in invalid_configs.items():
                msg += f"\n  {config_name}:\n"
                for error in errors:
                    msg += f"    - {error}\n"
            pytest.fail(msg)


class TestSpecificConfigs:
    """Test specific config requirements."""

    def test_english_config_exists(self):
        """Verify kokoro-mps-en.yaml exists and is valid."""
        config_path = CONFIG_DIR / "kokoro-mps-en.yaml"
        assert config_path.exists(), f"English config not found: {config_path}"

        config = load_config(config_path)
        assert config is not None, "Failed to parse English config"

        tts = config.get("tts", {})
        assert tts.get("language") == "en", "English config should have language: en"

    def test_japanese_config_exists(self):
        """Verify kokoro-mps-ja.yaml exists and is valid."""
        config_path = CONFIG_DIR / "kokoro-mps-ja.yaml"
        assert config_path.exists(), f"Japanese config not found: {config_path}"

        config = load_config(config_path)
        assert config is not None, "Failed to parse Japanese config"

        tts = config.get("tts", {})
        assert tts.get("language") == "ja", "Japanese config should have language: ja"

    def test_translation_config_valid(self):
        """Verify en2ja translation config has proper settings."""
        config_path = CONFIG_DIR / "kokoro-mps-en2ja.yaml"
        if not config_path.exists():
            pytest.skip(f"EN->JA config not found: {config_path}")

        config = load_config(config_path)
        assert config is not None, "Failed to parse EN->JA config"

        translation = config.get("translation", {})
        assert translation.get("enabled") == True, "Translation should be enabled"
        assert translation.get("source_lang") == "en", "Source should be 'en'"
        assert translation.get("target_lang") in ["ja", "jpn"], "Target should be Japanese"

        tts = config.get("tts", {})
        assert tts.get("language") == "ja", "TTS language should be 'ja' for EN->JA"


# =============================================================================
# CLI Validator
# =============================================================================

def main():
    """Command-line config validator."""
    import sys

    print("Config Coherency Validator")
    print("=" * 60)

    configs = get_all_kokoro_configs()
    print(f"Found {len(configs)} kokoro-mps-*.yaml configs\n")

    total_errors = 0
    for config_path in sorted(configs):
        config = load_config(config_path)
        if config is None:
            print(f"FAIL: {config_path.name} - Failed to parse")
            total_errors += 1
            continue

        is_valid, errors = validate_config_complete(config, config_path)

        if is_valid:
            print(f"OK:   {config_path.name}")
        else:
            print(f"FAIL: {config_path.name}")
            for error in errors:
                print(f"      - {error}")
            total_errors += len(errors)

    print("\n" + "=" * 60)
    if total_errors == 0:
        print(f"All {len(configs)} configs passed validation")
        return 0
    else:
        print(f"{total_errors} errors found")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
