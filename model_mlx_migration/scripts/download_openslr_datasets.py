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
Download ASR datasets from OpenSLR for CTC training.

All datasets here have commercial-friendly licenses (CC BY, Apache 2.0, etc.)

Usage:
    # List all available datasets
    python scripts/download_openslr_datasets.py --list

    # Download specific dataset
    python scripts/download_openslr_datasets.py --dataset aishell

    # Download all Chinese datasets
    python scripts/download_openslr_datasets.py --language zh

    # Download all Korean datasets
    python scripts/download_openslr_datasets.py --language ko
"""

import argparse
import subprocess
from pathlib import Path

# Dataset definitions: (name, SLR#, language, license, size_gb, files)
DATASETS = {
    # Chinese/Mandarin datasets
    "aishell": {
        "slr": 33,
        "language": "zh",
        "license": "Apache 2.0",
        "size_gb": 15,
        "description": "AISHELL-1: 178 hours Mandarin from 400 speakers",
        "files": ["data_aishell.tgz"],
    },
    "aishell3": {
        "slr": 93,
        "language": "zh",
        "license": "Apache 2.0",
        "size_gb": 19,
        "description": "AISHELL-3: Multi-speaker Mandarin TTS corpus",
        "files": ["data_aishell3.tgz"],
    },
    "st_cmds": {
        "slr": 38,
        "language": "zh",
        "license": "CC BY-SA 4.0",
        "size_gb": 8,
        "description": "Free ST Chinese Mandarin: 102,600 utterances from 855 speakers",
        "files": ["ST-CMDS-20170001_1-OS.tar.gz"],
    },
    "primewords": {
        "slr": 47,
        "language": "zh",
        "license": "CC BY-SA 3.0",
        "size_gb": 10,
        "description": "Primewords: 100 hours of Mandarin speech",
        "files": ["primewords_md_2018_set1.tar.gz"],
    },
    "aidatatang": {
        "slr": 62,
        "language": "zh",
        "license": "CC BY-NC-ND 4.0",  # NC - non-commercial!
        "size_gb": 20,
        "description": "aidatatang_200zh: 200 hours from 600 speakers (NON-COMMERCIAL)",
        "files": ["aidatatang_200zh.tgz"],
    },
    "magicdata": {
        "slr": 68,
        "language": "zh",
        "license": "CC BY-NC-ND 4.0",  # NC - non-commercial!
        "size_gb": 52,
        "description": "MAGICDATA: 755 hours Mandarin (NON-COMMERCIAL)",
        "files": ["train_set.tar.gz", "dev_set.tar.gz", "test_set.tar.gz"],
    },

    # Korean datasets
    "zeroth_korean": {
        "slr": 40,
        "language": "ko",
        "license": "CC BY 4.0",
        "size_gb": 5,
        "description": "Zeroth-Korean: ASR corpus",
        "files": ["zeroth_korean.tar.gz"],
    },
    "pansori": {
        "slr": 58,
        "language": "ko",
        "license": "CC BY-NC-ND 4.0",  # NC - non-commercial!
        "size_gb": 3,
        "description": "Pansori-TEDxKR: Korean TED talks (NON-COMMERCIAL)",
        "files": ["pansori_tedxkr.tar.gz"],
    },
    "deeply_korean": {
        "slr": 97,
        "language": "ko",
        "license": "CC BY-SA 4.0",
        "size_gb": 0.3,  # 281MB - this is 1% subset
        "description": "Deeply Korean: Read speech (1% subset, 281MB)",
        "files": ["KoreanReadSpeechCorpus.tar.gz"],
    },
    "seoul_corpus": {
        "slr": 113,
        "language": "ko",
        "license": "CC BY-NC 2.0",  # NC - non-commercial!
        "size_gb": 2.5,
        "description": "SEOUL CORPUS: Spontaneous Korean speech (NON-COMMERCIAL)",
        "files": ["sound.tgz", "label.tgz"],
    },

    # Hindi datasets
    "hindi_speech": {
        "slr": 116,
        "language": "hi",
        "license": "CC BY 4.0",
        "size_gb": 5,
        "description": "Hindi speech corpus",
        "files": ["hindi.tar.gz"],
    },

    # Spanish datasets
    "heroico": {
        "slr": 39,
        "language": "es",
        "license": "CC BY-SA 4.0",
        "size_gb": 1,
        "description": "HEROICO: Mexican Spanish speech",
        "files": ["heroico.tgz"],
    },

    # French datasets
    "african_french": {
        "slr": 57,
        "language": "fr",
        "license": "CC BY 4.0",
        "size_gb": 3,
        "description": "African Accented French",
        "files": ["african_accented_french.tar.gz"],
    },

    # German datasets
    "german_speech": {
        "slr": 95,
        "language": "de",
        "license": "CC BY 4.0",
        "size_gb": 8,
        "description": "German speech corpus",
        "files": ["german.tar.gz"],
    },

    # Russian datasets
    "russian_librispeech": {
        "slr": 96,
        "language": "ru",
        "license": "CC BY 4.0",
        "size_gb": 9.1,
        "description": "Russian LibriSpeech (RuLS): 98 hours from LibriVox",
        "files": ["ruls_data.tar.gz"],
    },

    # Japanese datasets
    "japanese_speech": {
        "slr": 105,
        "language": "ja",
        "license": "CC BY 4.0",
        "size_gb": 10,
        "description": "Japanese speech corpus",
        "files": ["japanese.tar.gz"],
    },

    # Kashmiri
    "kashmiri": {
        "slr": 122,
        "language": "ks",
        "license": "GPL-3.0",
        "size_gb": 0.4,
        "description": "Kashmiri Data Corpus",
        "files": ["kashmiri.tar.gz"],
    },
}

# Commercial-friendly datasets only
COMMERCIAL_OK = {
    name for name, info in DATASETS.items()
    if "NC" not in info["license"] and "GPL" not in info["license"]
}


def download_dataset(name: str, output_dir: str, mirror: str = "us"):
    """Download a dataset from OpenSLR."""
    if name not in DATASETS:
        print(f"Unknown dataset: {name}")
        return False

    info = DATASETS[name]
    slr = info["slr"]

    # Check license
    if "NC" in info["license"]:
        print(f"WARNING: {name} has non-commercial license: {info['license']}")
        response = input("Continue anyway? [y/N] ")
        if response.lower() != 'y':
            return False

    output_path = Path(output_dir) / info["language"] / name
    output_path.mkdir(parents=True, exist_ok=True)

    # OpenSLR mirror URLs
    mirrors = {
        "us": f"https://us.openslr.org/resources/{slr}",
        "eu": f"https://openslr.elda.org/resources/{slr}",
        "cn": f"https://openslr.magicdatatech.com/resources/{slr}",
    }
    base_url = mirrors.get(mirror, mirrors["us"])

    print(f"\n{'='*60}")
    print(f"Downloading: {name}")
    print(f"  SLR: {slr}")
    print(f"  Language: {info['language']}")
    print(f"  License: {info['license']}")
    print(f"  Size: ~{info['size_gb']} GB")
    print(f"  Output: {output_path}")
    print(f"{'='*60}")

    success = True
    for filename in info["files"]:
        url = f"{base_url}/{filename}"
        dest = output_path / filename

        if dest.exists():
            print(f"  [SKIP] {filename} already exists")
            continue

        print(f"  [DOWNLOAD] {filename}...")
        try:
            # Use wget for large files with resume support
            # --no-check-certificate needed due to OpenSLR certificate issues
            cmd = ["wget", "-c", "--no-check-certificate", "-O", str(dest), url]
            result = subprocess.run(cmd, check=True)
            print(f"  [OK] {filename}")
        except subprocess.CalledProcessError as e:
            print(f"  [ERROR] {filename}: {e}")
            success = False
        except FileNotFoundError:
            # wget not available, try curl
            try:
                # -k to skip SSL verification (OpenSLR cert issues)
                cmd = ["curl", "-k", "-L", "-C", "-", "-o", str(dest), url]
                result = subprocess.run(cmd, check=True)
                print(f"  [OK] {filename}")
            except Exception as e:
                print(f"  [ERROR] {filename}: {e}")
                success = False

    return success


def list_datasets(commercial_only: bool = False):
    """List all available datasets."""
    print("\n" + "="*70)
    print("Available OpenSLR Datasets for CTC Training")
    print("="*70)

    # Group by language
    by_language = {}
    for name, info in DATASETS.items():
        lang = info["language"]
        if lang not in by_language:
            by_language[lang] = []
        by_language[lang].append((name, info))

    for lang in sorted(by_language.keys()):
        datasets = by_language[lang]
        print(f"\n{lang.upper()}:")
        for name, info in datasets:
            commercial = "NC" not in info["license"] and "GPL" not in info["license"]
            if commercial_only and not commercial:
                continue
            status = "OK" if commercial else "NC"
            print(f"  [{status}] {name}: {info['description']}")
            print(f"       License: {info['license']}, Size: ~{info['size_gb']} GB")

    print("\n" + "="*70)
    print("Legend: [OK] = Commercial use allowed, [NC] = Non-commercial only")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Download OpenSLR datasets")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument("--commercial-only", action="store_true",
                       help="Show only commercially usable datasets")
    parser.add_argument("--dataset", "-d", type=str, help="Dataset name to download")
    parser.add_argument("--language", "-l", type=str,
                       help="Download all datasets for a language (zh, ko, etc.)")
    parser.add_argument("--output-dir", "-o", type=str, default="data/openslr",
                       help="Output directory")
    parser.add_argument("--mirror", "-m", type=str, default="us",
                       choices=["us", "eu", "cn"],
                       help="Download mirror (us, eu, cn)")

    args = parser.parse_args()

    if args.list:
        list_datasets(args.commercial_only)
        return

    if args.dataset:
        download_dataset(args.dataset, args.output_dir, args.mirror)
        return

    if args.language:
        for name, info in DATASETS.items():
            if info["language"] == args.language:
                # Skip NC datasets unless explicitly requested
                if "NC" in info["license"]:
                    print(f"Skipping {name} (non-commercial license)")
                    continue
                download_dataset(name, args.output_dir, args.mirror)
        return

    parser.print_help()
    print("\n" + "="*60)
    print("Examples:")
    print("  python scripts/download_openslr_datasets.py --list")
    print("  python scripts/download_openslr_datasets.py --dataset aishell")
    print("  python scripts/download_openslr_datasets.py --language zh")
    print("="*60)


if __name__ == "__main__":
    main()
