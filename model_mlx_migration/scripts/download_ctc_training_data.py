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
Download training data for CTC heads and other Whisper auxiliary heads.

This script downloads high-quality ASR datasets for multiple languages
to improve CTC streaming transcription quality.

Datasets:
    - Common Voice: 100+ languages, crowdsourced
    - Multilingual LibriSpeech (MLS): 8 languages, read speech
    - VoxPopuli: 23 languages, European Parliament
    - GigaSpeech: English, 10k hours
    - TED-LIUM 3: English, TED talks

Usage:
    # Download English data
    python scripts/download_ctc_training_data.py --english

    # Download all CTC language data
    python scripts/download_ctc_training_data.py --all-languages

    # Download specific language
    python scripts/download_ctc_training_data.py --language zh

    # List available datasets
    python scripts/download_ctc_training_data.py --list
"""

import argparse
import sys
from pathlib import Path

# Language codes to names
LANGUAGE_MAP = {
    "en": "English",
    "zh": "Chinese",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "ja": "Japanese",
    "ko": "Korean",
    "hi": "Hindi",
    "ru": "Russian",
    "ar": "Arabic",
    "pt": "Portuguese",
    "it": "Italian",
    "nl": "Dutch",
    "pl": "Polish",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "th": "Thai",
    "id": "Indonesian",
    "sv": "Swedish",
    "cs": "Czech",
    "ro": "Romanian",
    "hu": "Hungarian",
    "fi": "Finnish",
    "uk": "Ukrainian",
    "el": "Greek",
    "he": "Hebrew",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "ur": "Urdu",
}

# Common Voice language availability (v17.0)
COMMON_VOICE_LANGS = {
    "en", "zh-CN", "zh-TW", "fr", "de", "es", "ja", "ko", "hi", "ru",
    "ar", "pt", "it", "nl", "pl", "tr", "vi", "th", "id", "sv", "cs",
    "ro", "hu", "fi", "uk", "el", "he", "bn", "ta", "te", "mr", "gu",
    "kn", "ml", "pa", "ur", "fa", "ca", "eu", "gl", "cy", "br", "mt",
}

# MLS languages
MLS_LANGS = {"en", "de", "nl", "fr", "es", "it", "pt", "pl"}

# VoxPopuli languages
VOXPOPULI_LANGS = {
    "en", "de", "fr", "es", "pl", "it", "ro", "hu", "cs", "nl", "fi",
    "hr", "sk", "sl", "et", "lt", "pt", "bg", "el", "lv", "mt", "sv", "da"
}


def download_common_voice(lang: str, output_dir: str, streaming: bool = True):
    """
    Download Common Voice dataset for a language.

    Uses HuggingFace streaming to avoid downloading the full dataset.
    """
    from datasets import load_dataset

    # Map standard codes to Common Voice codes
    cv_lang = lang
    if lang == "zh":
        cv_lang = "zh-CN"  # Use simplified Chinese

    if cv_lang not in COMMON_VOICE_LANGS:
        print(f"  [SKIP] {lang} not available in Common Voice")
        return False

    print(f"  Loading Common Voice {lang}...")

    try:
        # Common Voice 17.0 - use parquet format
        dataset = load_dataset(
            "mozilla-foundation/common_voice_17_0",
            cv_lang,
            split="train",
            streaming=streaming,
        )

        # Save info
        info_path = Path(output_dir) / f"common_voice_{lang}_info.txt"
        info_path.parent.mkdir(parents=True, exist_ok=True)

        with open(info_path, "w") as f:
            f.write("Dataset: Common Voice 17.0\n")
            f.write(f"Language: {lang}\n")
            f.write(f"Streaming: {streaming}\n")
            f.write("Status: Available\n")

        print(f"  [OK] Common Voice {lang} available (streaming mode)")
        return True

    except Exception as e:
        # Try Common Voice 16.0 as fallback
        try:
            dataset = load_dataset(
                "mozilla-foundation/common_voice_16_0",
                cv_lang,
                split="train",
                streaming=streaming,
            )
            print(f"  [OK] Common Voice 16.0 {lang} available (streaming mode)")
            return True
        except:
            pass
        print(f"  [ERROR] Common Voice {lang}: {e}")
        return False


def download_mls(lang: str, output_dir: str, streaming: bool = True):
    """Download Multilingual LibriSpeech for a language."""
    from datasets import load_dataset

    if lang not in MLS_LANGS:
        print(f"  [SKIP] {lang} not available in MLS")
        return False

    print(f"  Loading MLS {lang}...")

    try:
        dataset = load_dataset(
            "facebook/multilingual_librispeech",
            lang,
            split="train",
            streaming=streaming,
        )

        print(f"  [OK] MLS {lang} available (streaming mode)")
        return True

    except Exception as e:
        print(f"  [ERROR] MLS {lang}: {e}")
        return False


def download_voxpopuli(lang: str, output_dir: str, streaming: bool = True):
    """Download VoxPopuli for a language."""
    from datasets import load_dataset

    if lang not in VOXPOPULI_LANGS:
        print(f"  [SKIP] {lang} not available in VoxPopuli")
        return False

    print(f"  Loading VoxPopuli {lang}...")

    try:
        dataset = load_dataset(
            "facebook/voxpopuli",
            lang,
            split="train",
            streaming=streaming,
        )

        print(f"  [OK] VoxPopuli {lang} available (streaming mode)")
        return True

    except Exception as e:
        print(f"  [ERROR] VoxPopuli {lang}: {e}")
        return False


def download_gigaspeech(output_dir: str, subset: str = "s"):
    """
    Download GigaSpeech English dataset.

    Subsets:
        xs: 10 hours
        s: 250 hours
        m: 1000 hours
        l: 2500 hours
        xl: 10000 hours
    """
    from datasets import load_dataset

    print(f"  Loading GigaSpeech ({subset})...")

    try:
        # GigaSpeech requires access request
        dataset = load_dataset(
            "speechcolab/gigaspeech",
            subset,
            split="train",
            streaming=True,
        )

        print(f"  [OK] GigaSpeech {subset} available")
        return True

    except Exception as e:
        print(f"  [ERROR] GigaSpeech: {e}")
        print("  NOTE: GigaSpeech requires access request at HuggingFace")
        return False


def download_librispeech(output_dir: str, subset: str = "train.clean.360"):
    """Download LibriSpeech subset."""
    from datasets import load_dataset

    print(f"  Loading LibriSpeech {subset}...")

    try:
        dataset = load_dataset(
            "openslr/librispeech_asr",
            "clean",  # or "other" for harder samples
            split="train.clean.360",
            streaming=True,
        )

        print(f"  [OK] LibriSpeech {subset} available")
        return True

    except Exception as e:
        print(f"  [ERROR] LibriSpeech: {e}")
        return False


def download_tedlium(output_dir: str):
    """Download TED-LIUM 3."""
    from datasets import load_dataset

    print("  Loading TED-LIUM 3...")

    try:
        dataset = load_dataset(
            "LIUM/tedlium",
            "release3",
            split="train",
            streaming=True,
        )

        print("  [OK] TED-LIUM 3 available")
        return True

    except Exception as e:
        print(f"  [ERROR] TED-LIUM: {e}")
        return False


def download_peoples_speech(output_dir: str):
    """Download People's Speech dataset (30k hours English)."""
    from datasets import load_dataset

    print("  Loading People's Speech...")

    try:
        dataset = load_dataset(
            "MLCommons/peoples_speech",
            "clean",
            split="train",
            streaming=True,
        )

        print("  [OK] People's Speech available")
        return True

    except Exception as e:
        print(f"  [ERROR] People's Speech: {e}")
        print("  NOTE: May require access approval")
        return False


def download_fleurs(lang: str, output_dir: str, streaming: bool = True):
    """Download FLEURS dataset (Google's multilingual benchmark)."""
    from datasets import load_dataset

    # FLEURS language codes
    fleurs_lang = f"{lang}_xx" if len(lang) == 2 else lang

    print(f"  Loading FLEURS {lang}...")

    try:
        dataset = load_dataset(
            "google/fleurs",
            fleurs_lang,
            split="train",
            streaming=streaming,
        )

        print(f"  [OK] FLEURS {lang} available")
        return True

    except Exception as e:
        print(f"  [ERROR] FLEURS {lang}: {e}")
        return False


def download_language_data(lang: str, output_dir: str):
    """Download all available datasets for a language."""
    print(f"\n{'='*60}")
    print(f"Downloading data for: {LANGUAGE_MAP.get(lang, lang)}")
    print(f"{'='*60}")

    results = {}

    # Try each data source
    results["Common Voice"] = download_common_voice(lang, output_dir)
    results["MLS"] = download_mls(lang, output_dir)
    results["VoxPopuli"] = download_voxpopuli(lang, output_dir)
    results["FLEURS"] = download_fleurs(lang, output_dir)

    # English-only datasets
    if lang == "en":
        results["GigaSpeech"] = download_gigaspeech(output_dir, "s")
        results["TED-LIUM"] = download_tedlium(output_dir)
        results["LibriSpeech"] = download_librispeech(output_dir)
        results["People's Speech"] = download_peoples_speech(output_dir)

    # Summary
    print(f"\nResults for {lang}:")
    for name, success in results.items():
        status = "OK" if success else "SKIP/ERROR"
        print(f"  {name}: {status}")

    return results


def list_datasets():
    """List all available datasets and their language coverage."""
    print("\n" + "="*70)
    print("Available Datasets for CTC Training")
    print("="*70)

    print("\n1. Common Voice 17.0 (Mozilla)")
    print(f"   Languages: {len(COMMON_VOICE_LANGS)}+")
    print("   Type: Crowdsourced read speech")
    print("   Access: Open")

    print("\n2. Multilingual LibriSpeech (Facebook)")
    print(f"   Languages: {', '.join(sorted(MLS_LANGS))}")
    print("   Type: Read audiobooks")
    print("   Access: Open")

    print("\n3. VoxPopuli (Facebook)")
    print(f"   Languages: {', '.join(sorted(VOXPOPULI_LANGS))}")
    print("   Type: European Parliament speeches")
    print("   Access: Open")

    print("\n4. FLEURS (Google)")
    print("   Languages: 102 languages")
    print("   Type: Read sentences")
    print("   Access: Open")

    print("\n5. GigaSpeech (English only)")
    print("   Size: 10,000 hours")
    print("   Type: YouTube, podcasts, audiobooks")
    print("   Access: Requires approval")

    print("\n6. TED-LIUM 3 (English)")
    print("   Size: 450 hours")
    print("   Type: TED talks")
    print("   Access: Open")

    print("\n7. People's Speech (English)")
    print("   Size: 30,000 hours")
    print("   Type: Various public domain")
    print("   Access: Requires approval")

    print("\n" + "="*70)
    print("Current CTC Languages Trained:")
    print("="*70)
    trained = ["en", "zh", "fr", "de", "es", "ja", "ko", "hi"]
    for lang in trained:
        name = LANGUAGE_MAP.get(lang, lang)
        datasets = []
        if lang in COMMON_VOICE_LANGS or f"{lang}-CN" in COMMON_VOICE_LANGS:
            datasets.append("CV")
        if lang in MLS_LANGS:
            datasets.append("MLS")
        if lang in VOXPOPULI_LANGS:
            datasets.append("VP")
        datasets.append("FLEURS")
        print(f"  {lang}: {name} - Available: {', '.join(datasets)}")


def main():
    parser = argparse.ArgumentParser(
        description="Download training data for CTC heads",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download English data
    python scripts/download_ctc_training_data.py --english

    # Download Chinese data
    python scripts/download_ctc_training_data.py --language zh

    # Download all trained CTC languages
    python scripts/download_ctc_training_data.py --all-languages

    # List available datasets
    python scripts/download_ctc_training_data.py --list
        """
    )

    parser.add_argument("--list", action="store_true",
                       help="List available datasets")
    parser.add_argument("--english", action="store_true",
                       help="Download English datasets")
    parser.add_argument("--language", "-l", type=str,
                       help="Download data for specific language (2-letter code)")
    parser.add_argument("--all-languages", action="store_true",
                       help="Download data for all CTC languages")
    parser.add_argument("--output-dir", "-o", type=str,
                       default="data/ctc_training",
                       help="Output directory")

    args = parser.parse_args()

    if args.list:
        list_datasets()
        return

    if not any([args.english, args.language, args.all_languages]):
        parser.print_help()
        print("\n" + "="*60)
        print("Quick start: python scripts/download_ctc_training_data.py --list")
        print("="*60)
        return

    # Import datasets library
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: datasets library required")
        print("Install: pip install datasets")
        sys.exit(1)

    print("="*70)
    print("CTC Training Data Downloader")
    print("="*70)
    print(f"Output directory: {args.output_dir}")

    if args.english:
        download_language_data("en", args.output_dir)

    if args.language:
        download_language_data(args.language, args.output_dir)

    if args.all_languages:
        languages = ["en", "zh", "fr", "de", "es", "ja", "ko", "hi"]
        for lang in languages:
            download_language_data(lang, args.output_dir)

    print("\n" + "="*70)
    print("Download Complete")
    print("="*70)
    print("\nNote: Datasets are in streaming mode by default.")
    print("To train, use the HuggingFace datasets streaming API:")
    print("  from datasets import load_dataset")
    print("  ds = load_dataset('mozilla-foundation/common_voice_17_0', 'en', streaming=True)")
    print("="*70)


if __name__ == "__main__":
    main()
