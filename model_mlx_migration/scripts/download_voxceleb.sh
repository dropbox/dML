#!/bin/bash
# Download VoxCeleb1/VoxCeleb2 audio archives.
#
# IMPORTANT: Do not hardcode time-limited download keys in this repo.
# Provide credentials via environment variables:
#   - VOXCELEB_BASE_URL (e.g. "https://example.com/download/voxceleb")
#   - VOXCELEB_KEY (time-limited token)
#
# Example:
#   VOXCELEB_BASE_URL="https://..." VOXCELEB_KEY="..." ./scripts/download_voxceleb.sh

set -euo pipefail

BASE_URL="${VOXCELEB_BASE_URL:-}"
KEY="${VOXCELEB_KEY:-}"

if [[ -z "${BASE_URL}" || -z "${KEY}" ]]; then
  cat << 'EOF'
ERROR: VoxCeleb download credentials not provided.

Set both environment variables and retry:
  export VOXCELEB_BASE_URL="https://<provider>/download/voxceleb"
  export VOXCELEB_KEY="<time-limited-token>"
  ./scripts/download_voxceleb.sh

Tip: Put these in `.env` (gitignored) and `source .env` before running.
EOF
  exit 2
fi

md5_check() {
  local expected="$1"
  local file="$2"
  if command -v md5sum >/dev/null 2>&1; then
    echo "${expected}  ${file}" | md5sum -c -
  elif command -v md5 >/dev/null 2>&1; then
    local got
    got="$(md5 -q "${file}")"
    [[ "${got}" == "${expected}" ]]
  else
    echo "WARNING: No md5 tool available; skipping checksum for ${file}" >&2
    return 0
  fi
}

# Create directories
mkdir -p data/voxceleb1
mkdir -p data/voxceleb2

echo "=== Downloading VoxCeleb1 (audio) ==="
cd data/voxceleb1

# VoxCeleb1 Dev (4 parts: partaa, partab, partac, partad)
for part in a b c d; do
    file="vox1_dev_wav_parta${part}"
    echo "Downloading $file..."
    curl -k -L --fail --retry 5 --retry-delay 2 -o "${file}" "${BASE_URL}?key=${KEY}&file=${file}"
done

# VoxCeleb1 Test
echo "Downloading vox1_test_wav.zip..."
curl -k -L --fail --retry 5 --retry-delay 2 -o "vox1_test_wav.zip" "${BASE_URL}?key=${KEY}&file=vox1_test_wav.zip"

# Concatenate dev parts and extract
echo "Concatenating VoxCeleb1 dev parts..."
cat vox1_dev_wav_parta* > vox1_dev_wav.zip
echo "Extracting VoxCeleb1 dev..."
unzip -q vox1_dev_wav.zip
echo "Extracting VoxCeleb1 test..."
unzip -q vox1_test_wav.zip

# Verify checksums
echo "Verifying VoxCeleb1 checksums..."
md5_check "ae63e55b951748cc486645f532ba230b" "vox1_dev_wav.zip" || echo "WARNING: VoxCeleb1 dev checksum mismatch"
md5_check "185fdc63c3c739954633d50379a3d102" "vox1_test_wav.zip" || echo "WARNING: VoxCeleb1 test checksum mismatch"

cd ../voxceleb2

echo ""
echo "=== Downloading VoxCeleb2 (audio AAC) ==="

# VoxCeleb2 Dev (8 parts: partaa through partah)
for part in a b c d e f g h; do
    file="vox2_dev_aac_parta${part}"
    echo "Downloading $file..."
    curl -k -L --fail --retry 5 --retry-delay 2 -o "${file}" "${BASE_URL}?key=${KEY}&file=${file}"
done

# VoxCeleb2 Test
echo "Downloading vox2_test_aac.zip..."
curl -k -L --fail --retry 5 --retry-delay 2 -o "vox2_test_aac.zip" "${BASE_URL}?key=${KEY}&file=vox2_test_aac.zip"

# Concatenate dev parts and extract
echo "Concatenating VoxCeleb2 dev parts..."
cat vox2_dev_aac_parta* > vox2_aac.zip
echo "Extracting VoxCeleb2 dev..."
unzip -q vox2_aac.zip
echo "Extracting VoxCeleb2 test..."
unzip -q vox2_test_aac.zip

# Verify checksums
echo "Verifying VoxCeleb2 checksums..."
md5_check "bbc063c46078a602ca71605645c2a402" "vox2_aac.zip" || echo "WARNING: VoxCeleb2 dev checksum mismatch"
md5_check "0d2b3ea430a821c33263b5ea37ede312" "vox2_test_aac.zip" || echo "WARNING: VoxCeleb2 test checksum mismatch"

cd ../..

echo ""
echo "=== Download Complete ==="
echo "VoxCeleb1: data/voxceleb1/"
echo "VoxCeleb2: data/voxceleb2/"
echo ""
echo "Note: VoxCeleb2 audio is in AAC format (m4a)."
echo "You may want to convert to WAV with:"
echo "  find data/voxceleb2 -name '*.m4a' -exec ffmpeg -i {} -ar 16000 {}.wav \;"
