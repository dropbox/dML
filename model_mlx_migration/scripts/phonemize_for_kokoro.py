#!/usr/bin/env python3
# Copyright 2024-2025 Andrew Yates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Phonemize text for Kokoro TTS using Misaki G2P.

Supports multiple languages: en, ja, zh, ko

Usage:
    # From .venv_phonemizer environment (Python 3.13):
    python scripts/phonemize_for_kokoro.py "Hello world"
    python scripts/phonemize_for_kokoro.py --lang ja "こんにちは"
    python scripts/phonemize_for_kokoro.py --lang zh "你好世界"

    # Output: JSON with phonemes and token_ids
"""

import json
import sys
import warnings
from pathlib import Path
from typing import Any, Optional

warnings.filterwarnings('ignore')

# Language-specific G2P loaders
G2P_LOADERS = {}

def _load_en_g2p():
    from misaki import en
    return en.G2P(trf=False, british=False, fallback=None)

def _load_ja_g2p():
    from misaki import ja
    return ja.JAG2P()

def _load_zh_g2p():
    from misaki import zh
    return zh.ZHG2P()

def _load_ko_g2p():
    from misaki import ko
    return ko.KOG2P()

G2P_LOADERS = {
    'en': _load_en_g2p,
    'ja': _load_ja_g2p,
    'zh': _load_zh_g2p,
    'ko': _load_ko_g2p,
}

SUPPORTED_LANGUAGES = list(G2P_LOADERS.keys())


def load_kokoro_vocab(vocab_path: Optional[Path] = None) -> dict[str, int]:
    """Load Kokoro phoneme vocabulary from exported artifacts or HuggingFace config.

    Search order:
    1. Explicit vocab_path if provided
    2. misaki_export/vocab.json (repo artifact)
    3. kokoro_cpp_export/g2p/vocab.json (C++ artifact)
    4. ~/models/kokoro/config.json (HuggingFace download)

    Returns:
        Dict mapping phoneme characters to token IDs
    """
    # Find repo root (directory containing misaki_export/)
    repo_root = Path(__file__).parent.parent
    if not (repo_root / "misaki_export").exists():
        repo_root = Path(__file__).parent
        while repo_root.parent != repo_root:
            if (repo_root / "misaki_export").exists():
                break
            repo_root = repo_root.parent

    search_paths = []
    if vocab_path:
        search_paths.append(vocab_path)
    search_paths.extend(
        [
            repo_root / "misaki_export" / "vocab.json",
            repo_root / "kokoro_cpp_export" / "g2p" / "vocab.json",
            Path.home() / "models" / "kokoro" / "config.json",
        ]
    )

    for path in search_paths:
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            # Handle both vocab.json (direct dict) and config.json (nested)
            vocab = data.get("vocab", data) if isinstance(data, dict) else data
            if isinstance(vocab, dict) and len(vocab) > 50:  # Sanity check
                return vocab

    raise FileNotFoundError(
        f"Kokoro vocab not found. Searched: {[str(p) for p in search_paths]}"
    )


def phonemize(text: str, language: str = "en", vocab: Optional[dict[str, int]] = None) -> dict[str, Any]:
    """
    Convert text to Kokoro phonemes and token IDs.

    Args:
        text: Input text to phonemize
        language: Language code (en, ja, zh, ko)
        vocab: Optional vocabulary dict (loaded if not provided)

    Returns:
        dict with keys: text, phonemes, token_ids, unknown_chars, language
    """
    if vocab is None:
        vocab = load_kokoro_vocab()

    if language not in G2P_LOADERS:
        raise ValueError(f"Unsupported language: {language}. Supported: {SUPPORTED_LANGUAGES}")

    # Initialize G2P for the specified language
    g2p = G2P_LOADERS[language]()

    # Get phonemes - handle different return types
    result = g2p(text)

    # English returns (phonemes, tokens), others return just phonemes string
    if isinstance(result, tuple):
        phonemes = result[0]
    else:
        phonemes = result

    # Convert to token IDs
    token_ids = []
    unknown_chars = []

    for char in phonemes:
        if char in vocab:
            token_ids.append(vocab[char])
        else:
            unknown_chars.append(char)
            token_ids.append(0)  # Unknown token

    # Add BOS (0) and EOS (0) tokens - required by Kokoro model
    # PyTorch Kokoro: input_ids = [0, ...phoneme_tokens..., 0]
    token_ids = [0] + token_ids + [0]

    return {
        "text": text,
        "phonemes": phonemes,
        "token_ids": token_ids,
        "unknown_chars": unknown_chars,
        "token_count": len(token_ids),
        "language": language,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Phonemize text for Kokoro TTS")
    parser.add_argument("text", nargs="*", help="Text to phonemize")
    parser.add_argument("--lang", "-l", default="en", choices=SUPPORTED_LANGUAGES,
                        help=f"Language code (default: en). Supported: {SUPPORTED_LANGUAGES}")
    args = parser.parse_args()

    if not args.text:
        print("Usage: python phonemize_for_kokoro.py [--lang LANG] 'text to phonemize'")
        print()
        print(f"Supported languages: {SUPPORTED_LANGUAGES}")
        print()
        print("Examples:")
        print("  python phonemize_for_kokoro.py 'Hello world'")
        print("  python phonemize_for_kokoro.py --lang ja 'こんにちは'")
        print("  python phonemize_for_kokoro.py --lang zh '你好世界'")
        sys.exit(1)

    text = " ".join(args.text)

    try:
        result = phonemize(text, language=args.lang)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
