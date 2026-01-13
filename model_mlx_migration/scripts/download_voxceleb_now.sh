#!/bin/bash
# Download VoxCeleb using stored credentials - ROBUST with aria2c
set -e

# Load credentials
source /Users/ayates/model_mlx_migration/.env

if [ -z "$VOXCELEB_BASE_URL" ] || [ -z "$VOXCELEB_KEY" ]; then
    echo "ERROR: VOXCELEB_BASE_URL and VOXCELEB_KEY must be set"
    exit 1
fi

cd /Users/ayates/model_mlx_migration/data

mkdir -p voxceleb1 voxceleb2

# Function to download with aria2c (multi-connection, resume)
download() {
    local url="$1"
    local output="$2"
    local dir=$(dirname "$output")
    local file=$(basename "$output")

    if [ -f "$output" ]; then
        echo "EXISTS: $output"
        return 0
    fi

    echo "DOWNLOADING: $file"
    aria2c \
        --continue=true \
        --max-connection-per-server=8 \
        --split=8 \
        --min-split-size=5M \
        --max-tries=10 \
        --retry-wait=5 \
        --timeout=60 \
        --connect-timeout=30 \
        --dir="$dir" \
        --out="$file" \
        --console-log-level=notice \
        "$url" || {
            echo "aria2c failed, trying wget..."
            wget -c -O "$output" "$url"
        }
}

echo "=== VoxCeleb Download ==="
echo "Base URL: $VOXCELEB_BASE_URL"
echo ""

# VoxCeleb1 Dev Parts
echo "=== VoxCeleb1 Dev Parts ==="
download "${VOXCELEB_BASE_URL}?key=${VOXCELEB_KEY}&file=vox1_dev_wav_partaa" "voxceleb1/vox1_dev_wav_partaa"
download "${VOXCELEB_BASE_URL}?key=${VOXCELEB_KEY}&file=vox1_dev_wav_partab" "voxceleb1/vox1_dev_wav_partab"
download "${VOXCELEB_BASE_URL}?key=${VOXCELEB_KEY}&file=vox1_dev_wav_partac" "voxceleb1/vox1_dev_wav_partac"
download "${VOXCELEB_BASE_URL}?key=${VOXCELEB_KEY}&file=vox1_dev_wav_partad" "voxceleb1/vox1_dev_wav_partad"

# VoxCeleb1 Test
echo "=== VoxCeleb1 Test ==="
download "${VOXCELEB_BASE_URL}?key=${VOXCELEB_KEY}&file=vox1_test_wav.zip" "voxceleb1/vox1_test_wav.zip"

# VoxCeleb2 Dev Parts
echo "=== VoxCeleb2 Dev Parts (8 parts) ==="
download "${VOXCELEB_BASE_URL}?key=${VOXCELEB_KEY}&file=vox2_dev_aac_partaa" "voxceleb2/vox2_dev_aac_partaa"
download "${VOXCELEB_BASE_URL}?key=${VOXCELEB_KEY}&file=vox2_dev_aac_partab" "voxceleb2/vox2_dev_aac_partab"
download "${VOXCELEB_BASE_URL}?key=${VOXCELEB_KEY}&file=vox2_dev_aac_partac" "voxceleb2/vox2_dev_aac_partac"
download "${VOXCELEB_BASE_URL}?key=${VOXCELEB_KEY}&file=vox2_dev_aac_partad" "voxceleb2/vox2_dev_aac_partad"
download "${VOXCELEB_BASE_URL}?key=${VOXCELEB_KEY}&file=vox2_dev_aac_partae" "voxceleb2/vox2_dev_aac_partae"
download "${VOXCELEB_BASE_URL}?key=${VOXCELEB_KEY}&file=vox2_dev_aac_partaf" "voxceleb2/vox2_dev_aac_partaf"
download "${VOXCELEB_BASE_URL}?key=${VOXCELEB_KEY}&file=vox2_dev_aac_partag" "voxceleb2/vox2_dev_aac_partag"
download "${VOXCELEB_BASE_URL}?key=${VOXCELEB_KEY}&file=vox2_dev_aac_partah" "voxceleb2/vox2_dev_aac_partah"

# VoxCeleb2 Test
echo "=== VoxCeleb2 Test ==="
download "${VOXCELEB_BASE_URL}?key=${VOXCELEB_KEY}&file=vox2_test_aac.zip" "voxceleb2/vox2_test_aac.zip"

echo ""
echo "=== Download Complete ==="
echo ""

# Concatenate VoxCeleb1
echo "=== Concatenating VoxCeleb1 ==="
cd voxceleb1
if [ ! -f vox1_dev_wav.zip ]; then
    cat vox1_dev_wav_parta* > vox1_dev_wav.zip
    echo "Created vox1_dev_wav.zip"
fi

# Verify checksum
echo "Verifying VoxCeleb1 checksum..."
EXPECTED="ae63e55b951748cc486645f532ba230b"
ACTUAL=$(md5 -q vox1_dev_wav.zip 2>/dev/null || md5sum vox1_dev_wav.zip | cut -d' ' -f1)
if [ "$ACTUAL" = "$EXPECTED" ]; then
    echo "VoxCeleb1 checksum OK!"
else
    echo "WARNING: VoxCeleb1 checksum mismatch (expected $EXPECTED, got $ACTUAL)"
fi

# Extract
echo "Extracting VoxCeleb1..."
unzip -q -n vox1_dev_wav.zip
unzip -q -n vox1_test_wav.zip
cd ..

# Concatenate VoxCeleb2
echo "=== Concatenating VoxCeleb2 ==="
cd voxceleb2
if [ ! -f vox2_dev_aac.zip ]; then
    cat vox2_dev_aac_parta* > vox2_dev_aac.zip
    echo "Created vox2_dev_aac.zip"
fi

# Verify checksum
echo "Verifying VoxCeleb2 checksum..."
EXPECTED="bbc063c46078a602ca71605645c2a402"
ACTUAL=$(md5 -q vox2_dev_aac.zip 2>/dev/null || md5sum vox2_dev_aac.zip | cut -d' ' -f1)
if [ "$ACTUAL" = "$EXPECTED" ]; then
    echo "VoxCeleb2 checksum OK!"
else
    echo "WARNING: VoxCeleb2 checksum mismatch (expected $EXPECTED, got $ACTUAL)"
fi

# Extract
echo "Extracting VoxCeleb2..."
unzip -q -n vox2_dev_aac.zip
unzip -q -n vox2_test_aac.zip
cd ..

echo ""
echo "=== All Done ==="
echo ""
echo "VoxCeleb1: $(du -sh voxceleb1)"
echo "VoxCeleb2: $(du -sh voxceleb2)"
