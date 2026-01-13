#!/bin/bash
# Extract all downloaded data archives
# Run with: ./scripts/extract_all_data.sh 2>&1 | tee logs/extraction.log

set -e
LOGFILE="logs/extraction_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOGFILE"
}

extract_tar() {
    local archive="$1"
    local dest="$2"

    if [ ! -f "$archive" ]; then
        log "SKIP: $archive not found"
        return
    fi

    local basename=$(basename "$archive" .tar.gz)
    local destdir="$dest/$basename"

    if [ -d "$destdir" ] && [ "$(ls -A $destdir 2>/dev/null)" ]; then
        log "SKIP: $destdir already extracted"
        return
    fi

    log "EXTRACTING: $archive -> $dest"
    mkdir -p "$dest"
    tar -xzf "$archive" -C "$dest" && log "DONE: $archive" || log "FAILED: $archive"
}

log "========================================"
log "Starting data extraction"
log "========================================"

# MLS (Multilingual LibriSpeech) - 7 languages
log ""
log "=== MLS Datasets ==="
for f in data/mls/mls_*.tar.gz; do
    extract_tar "$f" "data/mls"
done

# CommonVoice v24 - 4 languages
log ""
log "=== CommonVoice Datasets ==="
for f in data/commonvoice/mcv-*.tar.gz; do
    extract_tar "$f" "data/commonvoice"
done

# Multilingual - Various
log ""
log "=== Multilingual Datasets ==="

# Hindi
for f in data/multilingual/hindi/*.tar.gz; do
    extract_tar "$f" "data/multilingual/hindi"
done

# Hindi MUCS
for f in data/multilingual/hindi_mucs/*.tar.gz; do
    extract_tar "$f" "data/multilingual/hindi_mucs"
done

# Korean
for f in data/multilingual/korean/*.tar.gz; do
    extract_tar "$f" "data/multilingual/korean"
done

# Chinese
for f in data/multilingual/chinese/*.tar.gz; do
    extract_tar "$f" "data/multilingual/chinese"
done

# Kashmiri
for f in data/multilingual/kashmiri/*.tar.gz; do
    extract_tar "$f" "data/multilingual/kashmiri"
done

# OpenSLR datasets that might need extraction
log ""
log "=== OpenSLR Datasets ==="
for lang in zh ko ru de es fr; do
    for f in data/openslr/$lang/*.tar.gz data/openslr/$lang/*.tgz 2>/dev/null; do
        [ -f "$f" ] && extract_tar "$f" "data/openslr/$lang"
    done
done

# Singing datasets
log ""
log "=== Singing Datasets ==="
if [ -f "data/singing/VocalSet.zip" ]; then
    if [ ! -d "data/singing/vocalset_full" ]; then
        log "EXTRACTING: VocalSet.zip"
        unzip -q "data/singing/VocalSet.zip" -d "data/singing/vocalset_full" && log "DONE: VocalSet.zip"
    else
        log "SKIP: vocalset_full already extracted"
    fi
fi

# Emotion punctuation - MELD
log ""
log "=== Emotion/Punctuation Datasets ==="
if [ -f "data/emotion_punctuation/MELD.Raw.tar.gz" ]; then
    if [ ! -d "data/emotion_punctuation/MELD.Raw" ] || [ ! "$(ls -A data/emotion_punctuation/MELD.Raw 2>/dev/null)" ]; then
        extract_tar "data/emotion_punctuation/MELD.Raw.tar.gz" "data/emotion_punctuation"
    else
        log "SKIP: MELD.Raw already extracted"
    fi
fi

log ""
log "========================================"
log "Extraction complete!"
log "========================================"

# Summary
log ""
log "=== Extraction Summary ==="
log "MLS extracted directories:"
ls -d data/mls/mls_*_opus 2>/dev/null | wc -l | xargs echo "  Count:"

log "CommonVoice extracted directories:"
ls -d data/commonvoice/cv-corpus-* data/commonvoice/mcv-* 2>/dev/null | grep -v tar.gz | wc -l | xargs echo "  Count:"

log "Multilingual extracted:"
find data/multilingual -type d -mindepth 2 -maxdepth 2 2>/dev/null | wc -l | xargs echo "  Count:"
