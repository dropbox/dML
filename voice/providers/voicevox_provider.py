#!/usr/bin/env python3
"""
VOICEVOX TTS Provider

Japanese-native TTS engine with high-quality neural voices.

Popular Style IDs:
    2  - 四国めたん ノーマル
    3  - ずんだもん ノーマル (default)
    8  - 春日部つむぎ ノーマル
    9  - 波音リツ ノーマル
    14 - 冥鳴ひまり ノーマル

Performance:
    - Model load: ~2-5s
    - Synthesis: ~200-500ms
    - Sample rate: 24kHz

Credit: VOICEVOX:ずんだもん (or appropriate character)
"""

import os
import sys

from .base import TtsProviderBase

# VOICEVOX paths (relative to project root)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOICEVOX_DIR = os.path.join(_PROJECT_ROOT, "self_host", "voicevox")
OPEN_JTALK_DICT = os.path.join(VOICEVOX_DIR, "open_jtalk_dic_utf_8-1.11")
ONNXRUNTIME_PATH = os.path.join(
    VOICEVOX_DIR,
    "voicevox_onnxruntime-osx-arm64-1.17.3",
    "lib",
    "libvoicevox_onnxruntime.dylib",
)
MODEL_DIR = os.path.join(VOICEVOX_DIR, "models")

# Default style: ずんだもん ノーマル
DEFAULT_STYLE_ID = 3

# Style ID to character name mapping (for credits and --list-voices)
STYLE_NAMES = {
    0: "四国めたん あまあま",
    1: "ずんだもん あまあま",
    2: "四国めたん ノーマル",
    3: "ずんだもん ノーマル",
    4: "四国めたん セクシー",
    5: "ずんだもん セクシー",
    6: "四国めたん ツンツン",
    7: "ずんだもん ツンツン",
    8: "春日部つむぎ ノーマル",
    9: "波音リツ ノーマル",
    10: "雨晴はう ノーマル",
    14: "冥鳴ひまり ノーマル",
}


class VoicevoxProvider(TtsProviderBase):
    """VOICEVOX TTS provider for Japanese."""

    name = "voicevox"
    sample_rate = 24000

    def __init__(self, verbose: bool = True, style_id: int = DEFAULT_STYLE_ID):
        """
        Initialize VOICEVOX provider.

        Args:
            verbose: Print status messages
            style_id: Default voice style ID (3 = ずんだもん ノーマル)
        """
        super().__init__(verbose)
        self.default_style_id = style_id
        self._synthesizer = None

    def _get_synthesizer(self):
        """Get or create VOICEVOX synthesizer (cached)."""
        if self._synthesizer is not None:
            return self._synthesizer

        from voicevox_core.blocking import (
            Onnxruntime,
            OpenJtalk,
            Synthesizer,
            VoiceModelFile,
        )

        self._log("Loading VOICEVOX...")

        # Load ONNX runtime
        if not os.path.exists(ONNXRUNTIME_PATH):
            raise RuntimeError(f"ONNX runtime not found: {ONNXRUNTIME_PATH}")
        onnxruntime = Onnxruntime.load_once(filename=ONNXRUNTIME_PATH)

        # Load Open JTalk dictionary
        if not os.path.exists(OPEN_JTALK_DICT):
            raise RuntimeError(f"Open JTalk dict not found: {OPEN_JTALK_DICT}")
        open_jtalk = OpenJtalk(OPEN_JTALK_DICT)

        # Create synthesizer
        self._synthesizer = Synthesizer(onnxruntime, open_jtalk)

        # Load voice models
        if os.path.exists(MODEL_DIR):
            for vvm_file in sorted(os.listdir(MODEL_DIR)):
                if vvm_file.endswith(".vvm"):
                    vvm_path = os.path.join(MODEL_DIR, vvm_file)
                    self._log(f"  Loading model: {vvm_file}")
                    with VoiceModelFile.open(vvm_path) as model:
                        self._synthesizer.load_voice_model(model)

        self._initialized = True
        return self._synthesizer

    def _synthesize_impl(self, text: str, lang: str, voice_id: str) -> bytes:
        """
        Synthesize speech using VOICEVOX.

        voice_id is interpreted as style ID (integer string).
        Returns WAV bytes at 24kHz mono.
        """
        synth = self._get_synthesizer()

        # Parse voice_id as style ID
        if voice_id == "default" or not voice_id:
            style_id = self.default_style_id
        else:
            try:
                style_id = int(voice_id)
            except ValueError:
                # Try to find by name
                style_id = self.default_style_id
                for sid, name in STYLE_NAMES.items():
                    if voice_id.lower() in name.lower():
                        style_id = sid
                        break

        # Generate audio
        audio_bytes = synth.tts(text, style_id)

        return audio_bytes

    def get_supported_languages(self) -> list[str]:
        """VOICEVOX only supports Japanese."""
        return ["ja", "jp"]

    def get_voices(self, lang: str = None) -> list[str]:
        """Return available style IDs as voice names."""
        # Return style IDs with names
        voices = []
        for style_id, name in sorted(STYLE_NAMES.items()):
            voices.append(f"{style_id}:{name}")
        return voices

    def list_all_voices(self):
        """Print detailed voice list from loaded models."""
        synth = self._get_synthesizer()
        metas = synth.metas()

        print("Available voices:")
        for meta in metas:
            print(f"\n{meta.name}:")
            for style in meta.styles:
                print(f"  [{style.id:3d}] {style.name}")


# CLI support
def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="VOICEVOX TTS Provider")
    parser.add_argument("text", nargs="?", help="Text to synthesize")
    parser.add_argument("-o", "--output", help="Output WAV file")
    parser.add_argument(
        "-s", "--style", type=int, default=DEFAULT_STYLE_ID, help="Style ID"
    )
    parser.add_argument("--play", action="store_true", help="Play after synthesis")
    parser.add_argument("--list-voices", action="store_true", help="List all voices")

    args = parser.parse_args()

    provider = VoicevoxProvider(style_id=args.style)

    if args.list_voices:
        provider.list_all_voices()
        return

    if not args.text:
        text = sys.stdin.read().strip()
    else:
        text = args.text

    if not text:
        print("Error: No text provided", file=sys.stderr)
        sys.exit(1)

    if not args.output:
        print("Error: -o/--output is required", file=sys.stderr)
        sys.exit(1)

    success = provider.synthesize_to_file(text, args.output, "ja", str(args.style))

    if success and args.play:
        import subprocess

        subprocess.run(["afplay", args.output])

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
