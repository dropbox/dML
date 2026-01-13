#!/bin/bash
# Sync models from another machine
# Usage: ./scripts/sync_models.sh <user@host> [--minimal|--full]
#
# Model sets:
#   --minimal  Kokoro TTS only (~1.5GB) - English TTS
#   --full     All models (~5GB) - TTS + translation + whisper (default)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info() { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# Parse arguments
SOURCE=""
MODEL_SET="full"

while [[ $# -gt 0 ]]; do
    case $1 in
        --minimal)
            MODEL_SET="minimal"
            shift
            ;;
        --full)
            MODEL_SET="full"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 <user@host> [--minimal|--full]"
            echo ""
            echo "Sync models from another machine running Voice TTS."
            echo ""
            echo "Options:"
            echo "  --minimal  Kokoro TTS only (~1.5GB)"
            echo "  --full     All models (~5GB) - default"
            echo ""
            echo "Examples:"
            echo "  $0 ayates@main-macbook.local"
            echo "  $0 ayates@192.168.1.100 --minimal"
            exit 0
            ;;
        *)
            if [[ -z "$SOURCE" ]]; then
                SOURCE="$1"
            else
                error "Unknown option: $1"
            fi
            shift
            ;;
    esac
done

if [[ -z "$SOURCE" ]]; then
    error "Source machine required. Usage: $0 <user@host>"
fi

info "Syncing models from $SOURCE (set: $MODEL_SET)"
echo ""

# Create models directory
mkdir -p models

# Define what to sync based on model set
if [[ "$MODEL_SET" == "minimal" ]]; then
    info "Syncing minimal model set (Kokoro TTS only)..."

    # Kokoro TTS - required
    rsync -avP --compress \
        "$SOURCE:~/voice/models/kokoro/" \
        models/kokoro/

elif [[ "$MODEL_SET" == "full" ]]; then
    info "Syncing full model set..."

    # Sync everything, excluding large optional models
    rsync -avP --compress \
        --exclude='*.gguf' \
        --exclude='nllb-3.3b/' \
        --exclude='cosyvoice/' \
        --exclude='fish-speech*/' \
        --exclude='orpheus-tts*/' \
        --exclude='vibevoice/' \
        --exclude='openaudio*/' \
        --exclude='test_output/' \
        --exclude='*.wav' \
        "$SOURCE:~/voice/models/" \
        models/
fi

echo ""

# Verify what we got
info "Verifying models..."

check_model() {
    local path="$1"
    local name="$2"
    local required="$3"

    if [[ -f "$path" ]]; then
        SIZE=$(stat -f%z "$path" 2>/dev/null || stat -c%s "$path" 2>/dev/null)
        SIZE_MB=$((SIZE / 1024 / 1024))
        echo -e "  ${GREEN}[OK]${NC} $name (${SIZE_MB}MB)"
    elif [[ "$required" == "required" ]]; then
        echo -e "  ${RED}[MISSING]${NC} $name"
    else
        echo -e "  ${YELLOW}[SKIP]${NC} $name (optional)"
    fi
}

check_model "models/kokoro/kokoro_mps.pt" "Kokoro TTS model" "required"
check_model "models/kokoro/voice_af_heart.pt" "Kokoro voice (af_heart)" "required"
check_model "models/nllb/nllb-encoder-mps.pt" "NLLB encoder" "optional"
check_model "models/nllb/nllb-decoder-mps.pt" "NLLB decoder" "optional"
check_model "models/whisper/ggml-large-v3-turbo.bin" "Whisper STT" "optional"

echo ""

# Calculate total size
TOTAL_SIZE=$(du -sh models 2>/dev/null | cut -f1)
info "Total models size: $TOTAL_SIZE"

echo ""
echo "=============================================="
echo -e "${GREEN}Model sync complete!${NC}"
echo "=============================================="
echo ""
echo "Test with:"
echo "  ./stream-tts-cpp/build/stream-tts-cpp stream-tts-cpp/config/kokoro-mps-en.yaml --speak \"Hello\""
echo ""
