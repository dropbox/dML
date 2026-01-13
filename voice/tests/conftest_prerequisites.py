"""
Prerequisite Validation for Voice Project Tests

This module provides fail-fast validation of test prerequisites:
- Binaries (stream-tts-cpp)
- Model files (kokoro, whisper, nllb)
- Config files (kokoro-mps-*.yaml)

Philosophy:
- REQUIRED resources: Fail with clear error and fix instructions
- OPTIONAL resources: Skip test with informational message
- ALWAYS provide actionable guidance (what to run to fix)

Usage:
    # In conftest.py
    from conftest_prerequisites import require_binary, require_model, require_config

    @pytest.fixture
    def cpp_binary():
        return require_binary("stream-tts-cpp")
"""

from pathlib import Path
from typing import Optional


# =============================================================================
# Project Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
STREAM_TTS_CPP = PROJECT_ROOT / "stream-tts-cpp"
BUILD_DIR = STREAM_TTS_CPP / "build"
MODELS_DIR = PROJECT_ROOT / "models"
CONFIG_DIR = STREAM_TTS_CPP / "config"


# =============================================================================
# Error Messages with Fix Instructions
# =============================================================================

BINARY_NOT_FOUND_MSG = """
===============================================================================
PREREQUISITE MISSING: Binary not found
===============================================================================

Binary:   {binary_name}
Path:     {binary_path}

HOW TO FIX:
    cd {build_instructions_dir}
    mkdir -p build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j8

Or use the helper:
    make build

After building, re-run tests:
    pytest {test_path} -v

===============================================================================
"""

MODEL_NOT_FOUND_MSG = """
===============================================================================
PREREQUISITE MISSING: Model file not found
===============================================================================

Model:    {model_name}
Path:     {model_path}
Min size: {min_size} bytes

HOW TO FIX:
    # Download models using the setup script:
    ./scripts/download_models.sh

    # Or manually download the specific model:
    # {download_instructions}

After downloading, re-run tests:
    pytest {test_path} -v

===============================================================================
"""

CONFIG_NOT_FOUND_MSG = """
===============================================================================
PREREQUISITE MISSING: Config file not found
===============================================================================

Config:   {config_name}
Path:     {config_path}

HOW TO FIX:
    # Config files should exist in stream-tts-cpp/config/
    # If missing, check that the submodule is up to date:

    git submodule update --init --recursive

    # Or restore from git:
    cd stream-tts-cpp
    git checkout HEAD -- config/

After restoring, re-run tests:
    pytest {test_path} -v

===============================================================================
"""

MODEL_TOO_SMALL_MSG = """
===============================================================================
PREREQUISITE INVALID: Model file corrupted or incomplete
===============================================================================

Model:    {model_name}
Path:     {model_path}
Expected: >= {min_size} bytes
Actual:   {actual_size} bytes

HOW TO FIX:
    # The model file appears corrupted or incomplete.
    # Delete and re-download:

    rm -f {model_path}
    ./scripts/download_models.sh

    # Or manually download the model again.

After fixing, re-run tests:
    pytest {test_path} -v

===============================================================================
"""


# =============================================================================
# Prerequisite Categories
# =============================================================================

# Required binaries - tests FAIL if missing
REQUIRED_BINARIES = {
    "stream-tts-cpp": {
        "path": BUILD_DIR / "stream-tts-cpp",
        "description": "Main TTS binary",
    },
}

# Optional binaries - tests SKIP if missing
# Note: test_kokoro_torchscript removed - it's a diagnostic tool, not a test dependency
# Build it manually with: cmake --build . --target test_kokoro_torchscript
OPTIONAL_BINARIES = {}

# Required models - tests FAIL if missing
REQUIRED_MODELS = {
    "kokoro/kokoro_mps.pt": {
        "min_size": 300_000_000,  # ~328MB
        "description": "Kokoro TTS model (MPS)",
        "download": "Download from Kokoro GitHub releases",
    },
    "kokoro/voice_af_heart.pt": {
        "min_size": 500_000,  # ~523KB
        "description": "Default English voice",
        "download": "Download from Kokoro GitHub releases",
    },
}

# Optional models - tests SKIP if missing
OPTIONAL_MODELS = {
    "whisper/ggml-large-v3.bin": {
        "min_size": 3_000_000_000,  # ~3GB
        "description": "Whisper STT model (large-v3)",
        "download": "Download from Hugging Face (ggerganov/whisper.cpp)",
    },
    "whisper/ggml-silero-v6.2.0.bin": {
        "min_size": 800_000,  # ~885KB
        "description": "Silero VAD model",
        "download": "Download from Silero models",
    },
    "nllb/nllb-200-distilled-600m.pt": {
        "min_size": 2_000_000_000,  # ~2.4GB
        "description": "NLLB translation model",
        "download": "Download from Meta NLLB releases",
    },
}

# Required configs - tests FAIL if missing
REQUIRED_CONFIGS = {
    "kokoro-mps-en.yaml": "English TTS config",
}

# Optional configs - tests SKIP if missing (language-specific)
OPTIONAL_CONFIGS = {
    "kokoro-mps-ja.yaml": "Japanese TTS config",
    "kokoro-mps-zh.yaml": "Chinese TTS config",
    "kokoro-mps-es.yaml": "Spanish TTS config",
    "kokoro-mps-fr.yaml": "French TTS config",
    "kokoro-mps-hi.yaml": "Hindi TTS config",
    "kokoro-mps-it.yaml": "Italian TTS config",
    "kokoro-mps-pt.yaml": "Portuguese TTS config",
    "kokoro-mps-ko.yaml": "Korean TTS config",
    "kokoro-mps-yi.yaml": "Yiddish TTS config",
    "kokoro-mps-zh-sichuan.yaml": "Sichuanese TTS config",
}


# =============================================================================
# Validation Functions
# =============================================================================

class PrerequisiteError(Exception):
    """Raised when a required prerequisite is missing."""
    pass


class PrerequisiteSkip(Exception):
    """Raised when an optional prerequisite is missing (use with pytest.skip)."""
    pass


def require_binary(name: str, test_path: str = "tests/") -> Path:
    """
    Require a binary to exist. Fails with clear error if missing.

    Args:
        name: Binary name (key in REQUIRED_BINARIES or OPTIONAL_BINARIES)
        test_path: Test path for error message

    Returns:
        Path to the binary

    Raises:
        PrerequisiteError: If required binary is missing
        PrerequisiteSkip: If optional binary is missing
    """
    # Check required binaries first
    if name in REQUIRED_BINARIES:
        info = REQUIRED_BINARIES[name]
        binary_path = info["path"]
        if not binary_path.exists():
            raise PrerequisiteError(BINARY_NOT_FOUND_MSG.format(
                binary_name=name,
                binary_path=binary_path,
                build_instructions_dir=STREAM_TTS_CPP,
                test_path=test_path,
            ))
        return binary_path

    # Check optional binaries
    if name in OPTIONAL_BINARIES:
        info = OPTIONAL_BINARIES[name]
        binary_path = info["path"]
        if not binary_path.exists():
            raise PrerequisiteSkip(f"Optional binary not found: {name} ({info['description']})")
        return binary_path

    # Unknown binary - try to find it in build dir
    binary_path = BUILD_DIR / name
    if not binary_path.exists():
        raise PrerequisiteError(f"Unknown binary requested: {name}")
    return binary_path


def require_model(model_path: str, test_path: str = "tests/") -> Path:
    """
    Require a model file to exist with minimum size.

    Args:
        model_path: Relative path from models/ directory
        test_path: Test path for error message

    Returns:
        Full path to the model file

    Raises:
        PrerequisiteError: If required model is missing or invalid
        PrerequisiteSkip: If optional model is missing
    """
    full_path = MODELS_DIR / model_path

    # Check required models
    if model_path in REQUIRED_MODELS:
        info = REQUIRED_MODELS[model_path]
        if not full_path.exists():
            raise PrerequisiteError(MODEL_NOT_FOUND_MSG.format(
                model_name=model_path,
                model_path=full_path,
                min_size=info["min_size"],
                download_instructions=info["download"],
                test_path=test_path,
            ))

        actual_size = full_path.stat().st_size
        if actual_size < info["min_size"]:
            raise PrerequisiteError(MODEL_TOO_SMALL_MSG.format(
                model_name=model_path,
                model_path=full_path,
                min_size=info["min_size"],
                actual_size=actual_size,
                test_path=test_path,
            ))
        return full_path

    # Check optional models
    if model_path in OPTIONAL_MODELS:
        info = OPTIONAL_MODELS[model_path]
        if not full_path.exists():
            raise PrerequisiteSkip(f"Optional model not found: {model_path} ({info['description']})")

        actual_size = full_path.stat().st_size
        if actual_size < info["min_size"]:
            raise PrerequisiteSkip(
                f"Optional model incomplete: {model_path} "
                f"({actual_size} < {info['min_size']} bytes)"
            )
        return full_path

    # Unknown model - just check existence
    if not full_path.exists():
        raise PrerequisiteError(f"Model not found: {model_path}")
    return full_path


def require_config(config_name: str, test_path: str = "tests/") -> Path:
    """
    Require a config file to exist.

    Args:
        config_name: Config filename (e.g., "kokoro-mps-en.yaml")
        test_path: Test path for error message

    Returns:
        Full path to the config file

    Raises:
        PrerequisiteError: If required config is missing
        PrerequisiteSkip: If optional config is missing
    """
    config_path = CONFIG_DIR / config_name

    # Check required configs
    if config_name in REQUIRED_CONFIGS:
        if not config_path.exists():
            raise PrerequisiteError(CONFIG_NOT_FOUND_MSG.format(
                config_name=config_name,
                config_path=config_path,
                test_path=test_path,
            ))
        return config_path

    # Check optional configs
    if config_name in OPTIONAL_CONFIGS:
        if not config_path.exists():
            raise PrerequisiteSkip(
                f"Optional config not found: {config_name} "
                f"({OPTIONAL_CONFIGS[config_name]})"
            )
        return config_path

    # Unknown config - just check existence
    if not config_path.exists():
        raise PrerequisiteError(f"Config not found: {config_name}")
    return config_path


def check_all_prerequisites() -> dict:
    """
    Check all prerequisites and return status report.

    Returns:
        Dict with 'missing_required' and 'missing_optional' lists
    """
    missing_required = []
    missing_optional = []

    # Check binaries
    for name, info in REQUIRED_BINARIES.items():
        if not info["path"].exists():
            missing_required.append(f"Binary: {name}")

    for name, info in OPTIONAL_BINARIES.items():
        if not info["path"].exists():
            missing_optional.append(f"Binary: {name}")

    # Check models
    for path, info in REQUIRED_MODELS.items():
        full_path = MODELS_DIR / path
        if not full_path.exists():
            missing_required.append(f"Model: {path}")
        elif full_path.stat().st_size < info["min_size"]:
            missing_required.append(f"Model (incomplete): {path}")

    for path, info in OPTIONAL_MODELS.items():
        full_path = MODELS_DIR / path
        if not full_path.exists():
            missing_optional.append(f"Model: {path}")

    # Check configs
    for name, desc in REQUIRED_CONFIGS.items():
        if not (CONFIG_DIR / name).exists():
            missing_required.append(f"Config: {name}")

    for name, desc in OPTIONAL_CONFIGS.items():
        if not (CONFIG_DIR / name).exists():
            missing_optional.append(f"Config: {name}")

    return {
        "missing_required": missing_required,
        "missing_optional": missing_optional,
    }
