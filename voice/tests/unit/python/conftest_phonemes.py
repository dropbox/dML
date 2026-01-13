"""
Legacy helpers for phoneme tests.

The main fixtures now live in tests/conftest.py. This module keeps the
previous helper import path available for compatibility.
"""

from tests.conftest import run_cpp_tts  # noqa: F401 re-export for legacy imports
