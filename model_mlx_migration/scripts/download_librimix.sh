#!/bin/bash
# Download and generate LibriMix dataset for source separation training
# License: CC BY 4.0 (inherits from LibriSpeech)
# Source: https://github.com/JorisCos/LibriMix

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
cd "$BASE_DIR"

VENV_DIR="${VENV_DIR:-venv}"
if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
    echo "ERROR: venv not found at ${VENV_DIR}/bin/activate" >&2
    echo "Create it with: python3 -m venv ${VENV_DIR} && source ${VENV_DIR}/bin/activate" >&2
    exit 2
fi

# Activate virtual environment
source "${VENV_DIR}/bin/activate"

download_file() {
    local url="$1"
    local out="$2"
    if command -v wget >/dev/null 2>&1; then
        wget -c -O "${out}" "${url}"
    else
        curl -L --fail --retry 5 --retry-delay 2 -C - -o "${out}" "${url}"
    fi
}

echo "=== LibriMix Download and Generation ==="
echo "Base directory: $BASE_DIR"
echo ""

# Create directories
mkdir -p data/librimix
mkdir -p data/wham_noise

# Step 1: Clone LibriMix repo if needed
echo "=== Step 1: Clone LibriMix repository ==="
if [ ! -d "data/librimix/LibriMix" ]; then
    git clone https://github.com/JorisCos/LibriMix.git data/librimix/LibriMix
fi

# Step 2: Download WHAM noise
echo ""
echo "=== Step 2: Download WHAM noise (~35GB) ==="
cd data/wham_noise

if [ ! -f "wham_noise.zip" ] && [ ! -d "wham_noise" ]; then
    echo "Downloading WHAM noise..."
    # WHAM noise from the official source
    download_file "https://storage.googleapis.com/whisper-public/wham_noise.zip" "wham_noise.zip"
    echo "Extracting WHAM noise..."
    unzip -q wham_noise.zip
else
    echo "WHAM noise already exists, skipping download."
fi

cd "$BASE_DIR"

# Step 3: Ensure LibriSpeech is available
echo ""
echo "=== Step 3: Check LibriSpeech ==="
if [ -d "data/LibriSpeech/train-clean-100" ]; then
    echo "Found LibriSpeech train-clean-100"
    LIBRISPEECH_DIR="$BASE_DIR/data/LibriSpeech"
else
    echo "ERROR: LibriSpeech train-clean-100 not found!"
    echo "Please ensure data/LibriSpeech/train-clean-100 exists."
    exit 1
fi

# Step 4: Install Python requirements
echo ""
echo "=== Step 4: Install Python requirements ==="
python -m pip -q install pandas soundfile pyloudnorm tqdm

# Step 5: Generate LibriMix
echo ""
echo "=== Step 5: Generate LibriMix ==="
cd data/librimix/LibriMix

# Create output directory
mkdir -p ../Libri2Mix
mkdir -p ../Libri3Mix

# Generate Libri2Mix (2-speaker mixtures)
echo "Generating Libri2Mix..."
python scripts/create_librimix_from_metadata.py \
    --librispeech_dir "$LIBRISPEECH_DIR" \
    --wham_dir "$BASE_DIR/data/wham_noise/wham_noise" \
    --metadata_dir metadata/Libri2Mix \
    --librimix_outdir ../Libri2Mix \
    --n_src 2 \
    --freqs 16k \
    --modes min max \
    --types mix_clean mix_both mix_single

# Generate Libri3Mix (3-speaker mixtures) - optional, takes longer
if [[ "${LIBRIMIX_GENERATE_LIBRI3MIX:-0}" == "1" ]]; then
    echo ""
    echo "Generating Libri3Mix..."
    python scripts/create_librimix_from_metadata.py \
        --librispeech_dir "$LIBRISPEECH_DIR" \
        --wham_dir "$BASE_DIR/data/wham_noise/wham_noise" \
        --metadata_dir metadata/Libri3Mix \
        --librimix_outdir ../Libri3Mix \
        --n_src 3 \
        --freqs 16k \
        --modes min max \
        --types mix_clean mix_both mix_single
else
    echo ""
    echo "Skipping Libri3Mix (set LIBRIMIX_GENERATE_LIBRI3MIX=1 to enable)."
fi

cd "$BASE_DIR"

echo ""
echo "=== LibriMix Generation Complete ==="
echo ""
echo "Output directories:"
echo "  data/librimix/Libri2Mix/ - 2-speaker mixtures"
echo "  data/librimix/Libri3Mix/ - 3-speaker mixtures"
echo ""
du -sh data/librimix/Libri*Mix 2>/dev/null || echo "Calculating sizes..."
