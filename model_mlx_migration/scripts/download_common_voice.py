#!/usr/bin/env python3
"""
Download all Common Voice datasets from Mozilla Data Collective.

IMPORTANT: Before downloading, you must accept the terms for each dataset on the website:
    https://datacollective.mozillafoundation.org/datasets

Usage:
    # List all available datasets
    python scripts/download_common_voice.py --list-only

    # Accept terms for all datasets (opens browser)
    python scripts/download_common_voice.py --accept-terms

    # Download all Common Voice datasets
    python scripts/download_common_voice.py

    # Download a specific language
    python scripts/download_common_voice.py --filter russian

    # Download by ID
    python scripts/download_common_voice.py --id cmj8u48ey004xnxzpphv4udzz

Environment variables:
    MDC_API_KEY: Your Mozilla Data Collective API key (required)
    MDC_DOWNLOAD_PATH: Download directory (default: ./data/common_voice)
"""

import os
import sys
import time
import argparse
import webbrowser
from pathlib import Path
from typing import Any

# Ensure the API key is set
if not os.environ.get('MDC_API_KEY'):
    # Try loading from .env file
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value

from datacollective import save_dataset_to_disk, get_dataset_details
from datacollective.api_utils import api_request, DEFAULT_API_URL

# Set default download path
DEFAULT_DOWNLOAD_PATH = Path(__file__).parent.parent / 'data' / 'common_voice'

# All known Common Voice datasets from Mozilla Data Collective
# Updated: 2024-12-29
# Source: https://datacollective.mozillafoundation.org/datasets
# NOTE: Dataset IDs are CUIDs and cannot be enumerated - must be discovered from the website
ALL_CV_DATASETS = [
    # Common Voice Scripted Speech 24.0 (main corpus - large datasets)
    {"id": "cmj8u3p1w0075nxxbe8bedl00", "name": "Common Voice Scripted Speech 24.0 - English", "locale": "en", "size_mb": 89800},
    {"id": "cmj8u3q2n00vhnxxbzrjcugwc", "name": "Common Voice Scripted Speech 24.0 - Chinese (China)", "locale": "zh-CN", "size_mb": 21800},

    # Common Voice Spontaneous Speech 2.0 datasets
    {"id": "cmj8u48ej004lnxzp8sdt5z8c", "name": "Common Voice Spontaneous Speech 2.0 - Sabah Malay", "locale": "msi", "size_mb": 275.8},
    {"id": "cmj8u48eo004pnxzp991piql1", "name": "Common Voice Spontaneous Speech 2.0 - Western Penan", "locale": "pne", "size_mb": 247.1},
    {"id": "cmj8u48et004tnxzps28psruc", "name": "Common Voice Spontaneous Speech 2.0 - Puno Quechua", "locale": "qxp", "size_mb": 178.7},
    {"id": "cmj8u48ey004xnxzpphv4udzz", "name": "Common Voice Spontaneous Speech 2.0 - Russian", "locale": "ru", "size_mb": 50.0},
    {"id": "cmj8u48f60051nxzplpe1elj4", "name": "Common Voice Spontaneous Speech 2.0 - Ruuli", "locale": "ruc", "size_mb": 360.9},
    {"id": "cmj8u48fb0055nxzpqba4if8t", "name": "Common Voice Spontaneous Speech 2.0 - Amba", "locale": "rwm", "size_mb": 260.9},
    {"id": "cmj8u48fg0059nxzp1nhmykyr", "name": "Common Voice Spontaneous Speech 2.0 - Scots", "locale": "sco", "size_mb": 227.8},
    {"id": "cmj8u48fp005dnxzpmf26m8xo", "name": "Common Voice Spontaneous Speech 2.0 - Serian Bidayuh", "locale": "sdo", "size_mb": 199.9},
    {"id": "cmj8u48fu005hnxzp78hiv9ll", "name": "Common Voice Spontaneous Speech 2.0 - Sena", "locale": "seh", "size_mb": 24.6},
    {"id": "cmj8u48g4005lnxzp98cpr7b2", "name": "Common Voice Spontaneous Speech 2.0 - Tashlhiyt", "locale": "shi", "size_mb": 6.5},
    {"id": "cmj8u48ga005pnxzpmsz76k9o", "name": "Common Voice Spontaneous Speech 2.0 - Shona", "locale": "sn", "size_mb": 1.5},
    {"id": "cmj8u48gg005tnxzp9wqdltrl", "name": "Common Voice Spontaneous Speech 2.0 - snv", "locale": "snv", "size_mb": 212.7},
    {"id": "cmj8u48gl005xnxzpqzvf4ovg", "name": "Common Voice Spontaneous Speech 2.0 - Thai", "locale": "th", "size_mb": 0.1},
    {"id": "cmj8u48gq0061nxzpvl67iu91", "name": "Common Voice Spontaneous Speech 2.0 - Toba Qom", "locale": "tob", "size_mb": 172.4},
    {"id": "cmj8u48gv0065nxzpb3y7vi7v", "name": "Common Voice Spontaneous Speech 2.0 - Papantla Totonac", "locale": "top", "size_mb": 205.5},
    {"id": "cmj8u48h10069nxzpo6tghopr", "name": "Common Voice Spontaneous Speech 2.0 - Turkish", "locale": "tr", "size_mb": 4.2},
    {"id": "cmj8u48h7006dnxzp3y4uqb69", "name": "Common Voice Spontaneous Speech 2.0 - Rutoro", "locale": "ttj", "size_mb": 272.6},
    {"id": "cmj8u48hc006hnxzprn4k1cxx", "name": "Common Voice Spontaneous Speech 2.0 - Kuku", "locale": "ukv", "size_mb": 233.8},
    {"id": "cmj8u48hj006lnxzpnj14uhpz", "name": "Common Voice Spontaneous Speech 2.0 - Ushojo", "locale": "ush", "size_mb": 102.8},
    {"id": "cmj8u48hr006pnxzp3s43beqr", "name": "Common Voice Spontaneous Speech 2.0 - Kenyah", "locale": "xkl", "size_mb": 212.1},
]

# Other speech datasets on MDC (not Common Voice but related)
OTHER_SPEECH_DATASETS = [
    {"id": "cmjepxo6t08nmmk07iauvua6v", "name": "DhoNam: Dholuo Speech dataset", "locale": "Luo", "size_mb": 2546.8},
    {"id": "cmjk758i00cfumk070r7nwve7", "name": "Bamun-French Parallel Corpus", "locale": "bax", "size_mb": 0.1},
]


def fetch_dataset_details(dataset_id: str) -> dict[str, Any] | None:
    """Fetch details for a dataset."""
    try:
        return get_dataset_details(dataset_id)
    except Exception as e:
        print(f"Error fetching details for {dataset_id}: {e}")
        return None


def check_download_access(dataset_id: str) -> tuple[bool, str]:
    """Check if we can download a dataset (terms accepted)."""
    url = f"{DEFAULT_API_URL}/datasets/{dataset_id}/download"
    try:
        resp = api_request("POST", url, raise_known_errors=False)
        if resp.status_code == 200:
            return True, "OK"
        elif resp.status_code == 403:
            data = resp.json()
            return False, data.get("error", "Access denied")
        else:
            return False, f"HTTP {resp.status_code}"
    except Exception as e:
        return False, str(e)


def download_dataset(dataset_id: str, name: str, download_dir: Path,
                    overwrite: bool = False) -> bool:
    """Download a single dataset."""
    try:
        print(f"\n{'='*60}")
        print(f"Downloading: {name}")
        print(f"ID: {dataset_id}")
        print(f"{'='*60}")

        # Get dataset details first
        details = fetch_dataset_details(dataset_id)
        if details:
            size_bytes = int(details.get('sizeBytes', 0))
            size_mb = size_bytes / (1024 * 1024)
            print(f"Size: {size_mb:.1f} MB")

        # Download the dataset
        path = save_dataset_to_disk(
            dataset_id,
            download_directory=str(download_dir),
            show_progress=True,
            overwrite_existing=overwrite
        )

        print(f"Downloaded to: {path}")
        return True

    except PermissionError as e:
        print(f"Permission denied: {e}")
        print("  -> You must accept terms on the website first:")
        print(f"     https://datacollective.mozillafoundation.org/datasets/{dataset_id}")
        return False
    except FileNotFoundError:
        print(f"Dataset not found: {dataset_id}")
        return False
    except Exception as e:
        print(f"Error downloading {name}: {e}")
        return False


def accept_terms_in_browser(datasets: list[dict]) -> None:
    """Open browser tabs to accept terms for each dataset."""
    print("\nOpening browser to accept terms for each dataset...")
    print("Please accept the terms for each dataset in the browser windows that open.")
    print("Press Enter after accepting terms to continue downloading.\n")

    for i, d in enumerate(datasets, 1):
        url = f"https://datacollective.mozillafoundation.org/datasets/{d['id']}"
        print(f"  {i}/{len(datasets)}: {d['name']}")
        webbrowser.open(url)
        time.sleep(1)  # Small delay between opening tabs

    input("\nPress Enter after accepting all terms to start downloading...")


def main():
    parser = argparse.ArgumentParser(
        description="Download Common Voice datasets from Mozilla Data Collective",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # List all available datasets
    python scripts/download_common_voice.py --list-only

    # Check which datasets have terms accepted
    python scripts/download_common_voice.py --check-access

    # Accept terms for all datasets (opens browser)
    python scripts/download_common_voice.py --accept-terms

    # Download all datasets
    python scripts/download_common_voice.py

    # Download only Russian
    python scripts/download_common_voice.py --filter russian

    # Download by locale code
    python scripts/download_common_voice.py --locale ru
        """
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=DEFAULT_DOWNLOAD_PATH,
        help=f"Output directory (default: {DEFAULT_DOWNLOAD_PATH})"
    )
    parser.add_argument(
        '--overwrite', '-f',
        action='store_true',
        help="Overwrite existing files"
    )
    parser.add_argument(
        '--list-only', '-l',
        action='store_true',
        help="Only list datasets without downloading"
    )
    parser.add_argument(
        '--filter', '-F',
        type=str,
        default=None,
        help="Filter datasets by name (case-insensitive substring match)"
    )
    parser.add_argument(
        '--locale',
        type=str,
        default=None,
        help="Filter by locale code (e.g., 'ru' for Russian)"
    )
    parser.add_argument(
        '--id',
        type=str,
        default=None,
        help="Download a specific dataset by ID"
    )
    parser.add_argument(
        '--accept-terms',
        action='store_true',
        help="Open browser to accept terms for all datasets"
    )
    parser.add_argument(
        '--check-access',
        action='store_true',
        help="Check which datasets have terms accepted"
    )
    parser.add_argument(
        '--include-other',
        action='store_true',
        help="Include other speech datasets (not just Common Voice)"
    )
    parser.add_argument(
        '--refresh-list',
        action='store_true',
        help="Fetch fresh dataset details from API"
    )

    args = parser.parse_args()

    # Check API key
    if not os.environ.get('MDC_API_KEY'):
        print("Error: MDC_API_KEY environment variable not set")
        print("Set it in .env file or export MDC_API_KEY=your_key")
        print("\nTo get an API key:")
        print("  1. Go to https://datacollective.mozillafoundation.org")
        print("  2. Sign in or create an account")
        print("  3. Go to Profile > API > Create API Key")
        sys.exit(1)

    # Build dataset list
    if args.id:
        # Download specific dataset by ID
        details = fetch_dataset_details(args.id)
        if details:
            datasets = [{
                "id": args.id,
                "name": details.get('name', args.id),
                "locale": details.get('locale', 'unknown'),
                "size_mb": int(details.get('sizeBytes', 0)) / (1024*1024)
            }]
        else:
            print(f"Error: Could not fetch dataset {args.id}")
            sys.exit(1)
    else:
        datasets = ALL_CV_DATASETS.copy()
        if args.include_other:
            datasets.extend(OTHER_SPEECH_DATASETS)

    # Apply filters
    if args.filter:
        filter_lower = args.filter.lower()
        datasets = [d for d in datasets if filter_lower in d['name'].lower()]

    if args.locale:
        locale_lower = args.locale.lower()
        datasets = [d for d in datasets if d.get('locale', '').lower() == locale_lower]

    # Refresh from API if requested
    if args.refresh_list:
        print("Refreshing dataset details from API...")
        for d in datasets:
            details = fetch_dataset_details(d['id'])
            if details:
                d['name'] = details.get('name', d['name'])
                d['locale'] = details.get('locale', d.get('locale', ''))
                d['size_mb'] = int(details.get('sizeBytes', 0)) / (1024*1024)
            time.sleep(0.3)

    # Calculate total size
    total_size_mb = sum(d.get('size_mb', 0) for d in datasets)

    print(f"\n{'='*60}")
    print("Mozilla Data Collective - Common Voice Datasets")
    print(f"{'='*60}")
    print(f"Found {len(datasets)} datasets (total: {total_size_mb:.1f} MB)")

    # List mode
    if args.list_only:
        print("\nDatasets:")
        for i, d in enumerate(datasets, 1):
            size = d.get('size_mb', 0)
            locale = d.get('locale', '')
            print(f"  {i:3d}. {d['name']}")
            print(f"       ID: {d['id']} | Locale: {locale} | Size: {size:.1f} MB")
        print(f"\nTotal: {total_size_mb:.1f} MB")
        return

    # Check access mode
    if args.check_access:
        print("\nChecking access to datasets...")
        accessible = []
        blocked = []
        for d in datasets:
            can_download, reason = check_download_access(d['id'])
            status = "OK" if can_download else f"BLOCKED: {reason}"
            print(f"  {d['name']}: {status}")
            if can_download:
                accessible.append(d)
            else:
                blocked.append(d)
            time.sleep(0.3)  # Rate limiting

        print(f"\nSummary: {len(accessible)} accessible, {len(blocked)} blocked")
        if blocked:
            print("\nTo unblock datasets, accept terms at:")
            for d in blocked[:5]:  # Show first 5
                print(f"  https://datacollective.mozillafoundation.org/datasets/{d['id']}")
            if len(blocked) > 5:
                print(f"  ... and {len(blocked) - 5} more")
        return

    # Accept terms mode
    if args.accept_terms:
        accept_terms_in_browser(datasets)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Download all datasets
    print(f"\nDownloading to: {args.output_dir}")
    print("Note: You must accept terms on the website before downloading each dataset.")

    success = 0
    failed = 0
    needs_terms = []

    for d in datasets:
        result = download_dataset(
            d['id'],
            d['name'],
            args.output_dir,
            args.overwrite
        )
        if result:
            success += 1
        else:
            failed += 1
            needs_terms.append(d)

        # Rate limiting between downloads
        time.sleep(1)

    # Summary
    print(f"\n{'='*60}")
    print("Download Summary")
    print(f"{'='*60}")
    print(f"  Success: {success}")
    print(f"  Failed:  {failed}")
    print(f"  Total:   {len(datasets)}")
    print(f"\nFiles saved to: {args.output_dir}")

    if needs_terms:
        print(f"\n{len(needs_terms)} datasets need terms accepted. Run:")
        print(f"  python {sys.argv[0]} --accept-terms")
        print("\nOr manually visit:")
        for d in needs_terms[:5]:
            print(f"  https://datacollective.mozillafoundation.org/datasets/{d['id']}")
        if len(needs_terms) > 5:
            print(f"  ... and {len(needs_terms) - 5} more")


if __name__ == "__main__":
    main()
