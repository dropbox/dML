#!/usr/bin/env python3
"""
TTS Provider Interface

All TTS providers must implement this protocol.
Provides both Protocol (structural typing) and ABC (inheritance) patterns.
"""

import asyncio
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Protocol, runtime_checkable


def play_audio(audio_bytes: bytes, verbose: bool = False) -> bool:
    """
    Play audio bytes using the platform-appropriate audio player.

    Cross-platform support:
    - macOS: afplay
    - Linux: aplay or paplay (PulseAudio)
    - Windows: Windows Media Player via os.startfile

    Args:
        audio_bytes: WAV audio bytes to play
        verbose: If True, print status messages

    Returns:
        True if playback succeeded, False otherwise
    """
    import os
    import subprocess
    import tempfile

    # Write to temporary file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        f.write(audio_bytes)
        temp_path = f.name

    try:
        if sys.platform == 'darwin':
            # macOS
            result = subprocess.run(['afplay', temp_path], capture_output=True)
            return result.returncode == 0
        elif sys.platform.startswith('linux'):
            # Linux - try aplay first, then paplay (PulseAudio)
            for player in ['aplay', 'paplay']:
                try:
                    result = subprocess.run([player, temp_path], capture_output=True)
                    if result.returncode == 0:
                        return True
                except FileNotFoundError:
                    continue
            if verbose:
                print("ERROR: No audio player found (tried aplay, paplay)", file=sys.stderr)
            return False
        elif sys.platform == 'win32':
            # Windows
            os.startfile(temp_path)  # type: ignore[attr-defined]
            return True
        else:
            if verbose:
                print(f"ERROR: Unsupported platform: {sys.platform}", file=sys.stderr)
            return False
    except Exception as e:
        if verbose:
            print(f"ERROR playing audio: {e}", file=sys.stderr)
        return False
    finally:
        # Clean up temp file after playback
        try:
            os.unlink(temp_path)
        except OSError:
            pass


@runtime_checkable
class TtsProvider(Protocol):
    """Protocol for TTS providers (structural typing)."""

    name: str

    def synthesize(self, text: str, lang: str, voice_id: str = "default") -> bytes:
        """
        Synthesize speech from text.

        Args:
            text: Text to synthesize
            lang: Language code (en, ja, zh, etc.)
            voice_id: Voice identifier (provider-specific)

        Returns:
            Audio bytes (WAV format, 24kHz mono)
        """
        ...

    async def synthesize_async(self, text: str, lang: str, voice_id: str = "default") -> bytes:
        """Async version of synthesize."""
        ...

    def get_supported_languages(self) -> list[str]:
        """Return list of supported language codes."""
        ...

    def get_voices(self, lang: str = None) -> list[str]:
        """Return available voices, optionally filtered by language."""
        ...


class TtsProviderBase(ABC):
    """
    Abstract base class for TTS providers.

    Provides common functionality:
    - Consistent logging
    - Error handling with proper exceptions
    - File I/O helpers
    - Timing/metrics

    Subclasses must implement:
    - name (class attribute)
    - _synthesize_impl(text, lang, voice_id) -> bytes
    - get_supported_languages() -> list[str]
    - get_voices(lang) -> list[str]
    """

    name: str = "base"  # Override in subclass
    sample_rate: int = 24000  # Default sample rate

    def __init__(self, verbose: bool = True):
        """
        Initialize provider.

        Args:
            verbose: If True, print status messages to stderr
        """
        self.verbose = verbose
        self._initialized = False

    def _log(self, message: str) -> None:
        """Log message to stderr if verbose."""
        if self.verbose:
            print(f"[{self.name}] {message}", file=sys.stderr)

    @abstractmethod
    def _synthesize_impl(self, text: str, lang: str, voice_id: str) -> bytes:
        """
        Internal synthesis implementation.

        Args:
            text: Text to synthesize
            lang: Language code
            voice_id: Voice identifier

        Returns:
            Audio bytes (WAV format)

        Raises:
            RuntimeError: On synthesis failure
        """
        ...

    def synthesize(self, text: str, lang: str, voice_id: str = "default") -> bytes:
        """
        Synthesize speech from text.

        Args:
            text: Text to synthesize
            lang: Language code (en, ja, zh, etc.)
            voice_id: Voice identifier (provider-specific)

        Returns:
            Audio bytes (WAV format, 24kHz mono)

        Raises:
            ValueError: If text is empty or language unsupported
            RuntimeError: On synthesis failure
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        supported = self.get_supported_languages()
        if lang not in supported:
            raise ValueError(f"Language '{lang}' not supported. Supported: {supported}")

        preview = text[:50] + "..." if len(text) > 50 else text
        self._log(f"Synthesizing ({lang}/{voice_id}): {preview}")

        start = time.time()
        audio_bytes = self._synthesize_impl(text, lang, voice_id)
        elapsed = time.time() - start

        # Estimate duration from WAV size (44 header + 16-bit samples at sample_rate)
        audio_samples = (len(audio_bytes) - 44) // 2
        duration = audio_samples / self.sample_rate

        self._log(f"Synthesized in {elapsed*1000:.0f}ms ({duration:.1f}s audio)")

        return audio_bytes

    async def synthesize_async(self, text: str, lang: str, voice_id: str = "default") -> bytes:
        """Async version of synthesize."""
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.synthesize(text, lang, voice_id)
        )

    def synthesize_to_file(
        self, text: str, output_path: str, lang: str, voice_id: str = "default"
    ) -> bool:
        """
        Synthesize speech and save to file.

        Args:
            text: Text to synthesize
            output_path: Path to save WAV file
            lang: Language code
            voice_id: Voice identifier

        Returns:
            True if successful, False on error
        """
        try:
            audio_bytes = self.synthesize(text, lang, voice_id)
            Path(output_path).write_bytes(audio_bytes)
            return True
        except (ValueError, RuntimeError) as e:
            self._log(f"ERROR: {e}")
            return False
        except Exception as e:
            self._log(f"ERROR: {e}")
            import traceback
            traceback.print_exc(file=sys.stderr)
            return False

    def play_audio(self, audio_bytes: bytes) -> bool:
        """
        Play audio bytes using the platform-appropriate audio player.

        Cross-platform support:
        - macOS: afplay
        - Linux: aplay or paplay (PulseAudio)
        - Windows: Windows Media Player via os.startfile

        Args:
            audio_bytes: WAV audio bytes to play

        Returns:
            True if playback succeeded, False otherwise
        """
        # Use the standalone play_audio function with verbose=True for providers
        return play_audio(audio_bytes, verbose=self.verbose)

    @abstractmethod
    def get_supported_languages(self) -> list[str]:
        """Return list of supported language codes."""
        ...

    @abstractmethod
    def get_voices(self, lang: str = None) -> list[str]:
        """Return available voices, optionally filtered by language."""
        ...
