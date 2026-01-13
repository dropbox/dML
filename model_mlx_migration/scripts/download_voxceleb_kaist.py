#!/usr/bin/env python3
"""
VoxCeleb Download from KAIST Mirror

Downloads VoxCeleb1 and VoxCeleb2 from the KAIST mirror with:
- aria2c for fast, resumable downloads (multi-connection)
- wget fallback
- Automatic concatenation and extraction
- Checksum verification

Usage:
    # With username/password from VoxCeleb registration
    python scripts/download_voxceleb_kaist.py --user YOUR_USER --password YOUR_PASS

    # Resume interrupted download
    python scripts/download_voxceleb_kaist.py --user YOUR_USER --password YOUR_PASS --resume

    # Download only VoxCeleb1
    python scripts/download_voxceleb_kaist.py --user YOUR_USER --password YOUR_PASS --vox1-only

    # Extract only (after download complete)
    python scripts/download_voxceleb_kaist.py --extract-only
"""

import argparse
import hashlib
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from datetime import datetime

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
VOX1_DIR = DATA_DIR / "voxceleb1"
VOX2_DIR = DATA_DIR / "voxceleb2"

# KAIST Mirror Base URL
BASE_URL = "http://cnode01.mm.kaist.ac.kr/voxceleb"

# File definitions with MD5 checksums
VOX1_PARTS = [
    ("vox1a/vox1_dev_wav_partaa", "vox1_dev_wav_partaa", None),
    ("vox1a/vox1_dev_wav_partab", "vox1_dev_wav_partab", None),
    ("vox1a/vox1_dev_wav_partac", "vox1_dev_wav_partac", None),
    ("vox1a/vox1_dev_wav_partad", "vox1_dev_wav_partad", None),
]

VOX1_FILES = [
    ("vox1a/vox1_test_wav.zip", "vox1_test_wav.zip", "185fdc63c3c739954633d50379a3d102"),
]

VOX2_PARTS = [
    ("vox1a/vox2_dev_aac_partaa", "vox2_dev_aac_partaa", None),
    ("vox1a/vox2_dev_aac_partab", "vox2_dev_aac_partab", None),
    ("vox1a/vox2_dev_aac_partac", "vox2_dev_aac_partac", None),
    ("vox1a/vox2_dev_aac_partad", "vox2_dev_aac_partad", None),
    ("vox1a/vox2_dev_aac_partae", "vox2_dev_aac_partae", None),
    ("vox1a/vox2_dev_aac_partaf", "vox2_dev_aac_partaf", None),
    ("vox1a/vox2_dev_aac_partag", "vox2_dev_aac_partag", None),
    ("vox1a/vox2_dev_aac_partah", "vox2_dev_aac_partah", None),
]

VOX2_FILES = [
    ("vox1a/vox2_test_aac.zip", "vox2_test_aac.zip", "0d2b3ea430a821c33263b5ea37ede312"),
]

# Combined file checksums
COMBINED_CHECKSUMS = {
    "vox1_dev_wav.zip": "ae63e55b951748cc486645f532ba230b",
    "vox2_dev_aac.zip": "bbc063c46078a602ca71605645c2a402",
}


def log(msg: str, level: str = "INFO"):
    """Print timestamped log message."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}")


def check_tools():
    """Check for required tools."""
    tools = {
        "aria2c": shutil.which("aria2c"),
        "wget": shutil.which("wget"),
        "ffmpeg": shutil.which("ffmpeg"),
    }

    if not tools["aria2c"] and not tools["wget"]:
        log("Neither aria2c nor wget found. Install one of them.", "ERROR")
        sys.exit(1)

    if tools["aria2c"]:
        log("Using aria2c for downloads (fast, multi-connection)")
    else:
        log("Using wget for downloads (aria2c not found)")

    return tools


def compute_md5(filepath: Path) -> str:
    """Compute MD5 hash of file."""
    md5 = hashlib.md5()
    size = filepath.stat().st_size
    done = 0

    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192 * 1024), b""):  # 8MB chunks
            md5.update(chunk)
            done += len(chunk)
            pct = done * 100 // size
            print(f"\r  Verifying: {pct}%", end="", flush=True)

    print()
    return md5.hexdigest()


def download_aria2c(url: str, output: Path, user: str, password: str) -> bool:
    """Download using aria2c with resume and multi-connection."""
    cmd = [
        "aria2c",
        "--continue=true",           # Resume
        "--max-connection-per-server=4",  # Multiple connections
        "--split=4",
        "--min-split-size=10M",
        "--http-user=" + user,
        "--http-passwd=" + password,
        "--dir=" + str(output.parent),
        "--out=" + output.name,
        "--console-log-level=notice",
        url,
    ]

    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError:
        return False


def download_wget(url: str, output: Path, user: str, password: str) -> bool:
    """Download using wget with resume."""
    cmd = [
        "wget",
        "-c",  # Continue/resume
        "--user=" + user,
        "--password=" + password,
        "-O", str(output),
        url,
    ]

    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError:
        return False


def download_file(url_path: str, filename: str, output_dir: Path,
                  user: str, password: str, tools: dict) -> bool:
    """Download a single file."""
    url = f"{BASE_URL}/{url_path}"
    output = output_dir / filename

    if output.exists():
        size_mb = output.stat().st_size / (1024 * 1024)
        log(f"  {filename}: exists ({size_mb:.1f}MB), skipping")
        return True

    log(f"  Downloading {filename}...")

    if tools["aria2c"]:
        return download_aria2c(url, output, user, password)
    else:
        return download_wget(url, output, user, password)


def concatenate_parts(parts: list, output: Path, output_dir: Path) -> bool:
    """Concatenate file parts into single file."""
    if output.exists():
        log(f"  {output.name} already exists, skipping concatenation")
        return True

    log(f"  Concatenating {len(parts)} parts into {output.name}...")

    try:
        with open(output, "wb") as outf:
            for _, filename, _ in parts:
                part_path = output_dir / filename
                if not part_path.exists():
                    log(f"  Missing part: {filename}", "ERROR")
                    return False

                log(f"    Adding {filename}...")
                with open(part_path, "rb") as inf:
                    shutil.copyfileobj(inf, outf)

        return True
    except Exception as e:
        log(f"  Concatenation failed: {e}", "ERROR")
        return False


def verify_checksum(filepath: Path, expected_md5: str) -> bool:
    """Verify file checksum."""
    if not expected_md5:
        return True

    log(f"  Verifying {filepath.name}...")
    actual_md5 = compute_md5(filepath)

    if actual_md5 == expected_md5:
        log(f"  Checksum OK: {expected_md5}")
        return True
    else:
        log(f"  Checksum MISMATCH: expected {expected_md5}, got {actual_md5}", "ERROR")
        return False


def extract_zip(zip_path: Path, output_dir: Path) -> bool:
    """Extract zip file."""
    log(f"  Extracting {zip_path.name}...")

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            total = len(zf.namelist())
            for i, name in enumerate(zf.namelist()):
                if i % 1000 == 0:
                    pct = i * 100 // total
                    print(f"\r    Progress: {pct}%", end="", flush=True)
                zf.extract(name, output_dir)
            print()
        return True
    except Exception as e:
        log(f"  Extraction failed: {e}", "ERROR")
        return False


def convert_m4a_to_wav(input_dir: Path, sample_rate: int = 16000):
    """Convert M4A files to WAV using ffmpeg."""
    m4a_files = list(input_dir.rglob("*.m4a"))
    if not m4a_files:
        log("No M4A files found")
        return

    log(f"Converting {len(m4a_files)} M4A files to WAV...")

    for i, m4a in enumerate(m4a_files):
        wav = m4a.with_suffix(".wav")
        if wav.exists():
            continue

        if i % 500 == 0:
            pct = i * 100 // len(m4a_files)
            log(f"  Progress: {pct}% ({i}/{len(m4a_files)})")

        cmd = [
            "ffmpeg", "-i", str(m4a),
            "-ar", str(sample_rate),
            "-ac", "1",
            "-y", "-loglevel", "error",
            str(wav)
        ]
        subprocess.run(cmd, capture_output=True)

    log("  Conversion complete!")


def download_voxceleb1(user: str, password: str, tools: dict):
    """Download VoxCeleb1."""
    log("=== Downloading VoxCeleb1 ===")
    VOX1_DIR.mkdir(parents=True, exist_ok=True)

    # Download parts
    log("Downloading dev parts...")
    for url_path, filename, _ in VOX1_PARTS:
        download_file(url_path, filename, VOX1_DIR, user, password, tools)

    # Download test
    log("Downloading test set...")
    for url_path, filename, md5 in VOX1_FILES:
        download_file(url_path, filename, VOX1_DIR, user, password, tools)

    # Concatenate
    combined = VOX1_DIR / "vox1_dev_wav.zip"
    concatenate_parts(VOX1_PARTS, combined, VOX1_DIR)

    # Verify
    verify_checksum(combined, COMBINED_CHECKSUMS["vox1_dev_wav.zip"])

    log("VoxCeleb1 download complete!")


def download_voxceleb2(user: str, password: str, tools: dict):
    """Download VoxCeleb2."""
    log("=== Downloading VoxCeleb2 ===")
    VOX2_DIR.mkdir(parents=True, exist_ok=True)

    # Download parts
    log("Downloading dev parts (8 parts)...")
    for url_path, filename, _ in VOX2_PARTS:
        download_file(url_path, filename, VOX2_DIR, user, password, tools)

    # Download test
    log("Downloading test set...")
    for url_path, filename, md5 in VOX2_FILES:
        download_file(url_path, filename, VOX2_DIR, user, password, tools)

    # Concatenate
    combined = VOX2_DIR / "vox2_dev_aac.zip"
    concatenate_parts(VOX2_PARTS, combined, VOX2_DIR)

    # Verify
    verify_checksum(combined, COMBINED_CHECKSUMS["vox2_dev_aac.zip"])

    log("VoxCeleb2 download complete!")


def extract_voxceleb1():
    """Extract VoxCeleb1."""
    log("=== Extracting VoxCeleb1 ===")

    dev_zip = VOX1_DIR / "vox1_dev_wav.zip"
    test_zip = VOX1_DIR / "vox1_test_wav.zip"

    if dev_zip.exists():
        extract_zip(dev_zip, VOX1_DIR)

    if test_zip.exists():
        extract_zip(test_zip, VOX1_DIR)

    log("VoxCeleb1 extraction complete!")


def extract_voxceleb2():
    """Extract VoxCeleb2."""
    log("=== Extracting VoxCeleb2 ===")

    dev_zip = VOX2_DIR / "vox2_dev_aac.zip"
    test_zip = VOX2_DIR / "vox2_test_aac.zip"

    if dev_zip.exists():
        extract_zip(dev_zip, VOX2_DIR)

    if test_zip.exists():
        extract_zip(test_zip, VOX2_DIR)

    log("VoxCeleb2 extraction complete!")


def print_stats():
    """Print download statistics."""
    log("=== Statistics ===")

    for name, path in [("VoxCeleb1", VOX1_DIR), ("VoxCeleb2", VOX2_DIR)]:
        if not path.exists():
            log(f"{name}: Not downloaded")
            continue

        # Count speakers and files
        wav_files = list(path.rglob("*.wav"))
        m4a_files = list(path.rglob("*.m4a"))

        speakers = set()
        for f in wav_files + m4a_files:
            try:
                rel = f.relative_to(path)
                if len(rel.parts) >= 1 and rel.parts[0].startswith("id"):
                    speakers.add(rel.parts[0])
            except:
                pass

        total_size = sum(f.stat().st_size for f in wav_files + m4a_files)

        log(f"{name}:")
        log(f"  Speakers: {len(speakers)}")
        log(f"  WAV files: {len(wav_files)}")
        log(f"  M4A files: {len(m4a_files)}")
        log(f"  Total size: {total_size / (1024**3):.1f} GB")


def main():
    parser = argparse.ArgumentParser(description="Download VoxCeleb from KAIST mirror")
    parser.add_argument("--user", "-u", help="VoxCeleb username")
    parser.add_argument("--password", "-p", help="VoxCeleb password")
    parser.add_argument("--vox1-only", action="store_true", help="Download only VoxCeleb1")
    parser.add_argument("--vox2-only", action="store_true", help="Download only VoxCeleb2")
    parser.add_argument("--extract-only", action="store_true", help="Extract only (skip download)")
    parser.add_argument("--convert", action="store_true", help="Convert M4A to WAV after extraction")
    parser.add_argument("--stats", action="store_true", help="Print statistics and exit")
    args = parser.parse_args()

    log("VoxCeleb Downloader (KAIST Mirror)")
    log(f"Output: {DATA_DIR}")

    if args.stats:
        print_stats()
        return

    # Determine what to download
    do_vox1 = not args.vox2_only
    do_vox2 = not args.vox1_only

    if not args.extract_only:
        if not args.user or not args.password:
            log("Username and password required!", "ERROR")
            log("Get credentials from VoxCeleb registration")
            log("Usage: python download_voxceleb_kaist.py --user USER --password PASS")
            sys.exit(1)

        tools = check_tools()

        if do_vox1:
            download_voxceleb1(args.user, args.password, tools)

        if do_vox2:
            download_voxceleb2(args.user, args.password, tools)

    # Extract
    if do_vox1:
        extract_voxceleb1()

    if do_vox2:
        extract_voxceleb2()

    # Convert M4A to WAV
    if args.convert and do_vox2:
        convert_m4a_to_wav(VOX2_DIR)

    print_stats()
    log("=== Done ===")


if __name__ == "__main__":
    main()
