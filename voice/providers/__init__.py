#!/usr/bin/env python3
"""
TTS Providers Package

Provides a unified interface for different TTS engines:
- KokoroProvider: Fast multi-language TTS (82M params)
- VoicevoxProvider: High-quality Japanese TTS
- OpenAITTSProvider: Cloud-based TTS (requires API key)
"""

from .base import TtsProvider, TtsProviderBase, play_audio

__all__ = [
    "TtsProvider",
    "TtsProviderBase",
    "play_audio",
]
