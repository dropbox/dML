#!/usr/bin/env python3
"""
Legacy wrapper for the audio quality checker.

The canonical implementation now lives at tests/audio_quality.py inside the
pytest tree. This wrapper forwards imports and CLI execution to that file to
avoid code drift while keeping existing references working.
"""

from pathlib import Path
import sys

# Add tests/ to import path for legacy callers
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = PROJECT_ROOT / "tests"
sys.path.insert(0, str(TESTS_DIR))

# Re-export all public symbols
from audio_quality import *  # noqa: F401,F403,E402


if __name__ == "__main__":
    # Delegate CLI entrypoint to the canonical implementation
    main()
