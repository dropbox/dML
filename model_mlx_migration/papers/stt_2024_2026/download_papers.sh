#!/bin/bash
# STT SOTA Paper Download Script
# Generated: 2026-01-02
# Downloads all papers from PAPER_INDEX.md

set -e

PAPER_DIR="$(dirname "$0")/pdfs"
mkdir -p "$PAPER_DIR"

echo "Downloading STT SOTA papers to $PAPER_DIR..."

# Function to download with retry
download_paper() {
    local arxiv_id=$1
    local name=$2
    local url="https://arxiv.org/pdf/${arxiv_id}.pdf"
    local output="$PAPER_DIR/${arxiv_id}_${name}.pdf"

    if [ -f "$output" ]; then
        echo "  [SKIP] $arxiv_id ($name) - already exists"
        return 0
    fi

    echo "  [DOWNLOADING] $arxiv_id ($name)..."
    if wget -q --timeout=30 -O "$output" "$url" 2>/dev/null; then
        echo "  [OK] $arxiv_id"
    else
        echo "  [FAIL] $arxiv_id - retrying..."
        sleep 2
        wget -q --timeout=60 -O "$output" "$url" || echo "  [FAILED] $arxiv_id"
    fi
}

echo ""
echo "=== Category 1: Foundation Models ==="
download_paper "2303.01037" "USM"
download_paper "2312.08553" "USM-Lite"
download_paper "2401.16658" "OWSM-v3.1"
download_paper "2402.12654" "OWSM-CTC"
download_paper "2308.11596" "SeamlessM4T"
download_paper "2312.05187" "SeamlessM4T-v2"
download_paper "2512.22165" "Marco-ASR"
download_paper "2512.23808" "MiMo-Audio"

echo ""
echo "=== Category 2: Speaker-Adaptive ASR ==="
download_paper "2511.18774" "Context-Aware-Whisper"
download_paper "2510.18374" "Fairness-Prompted"
download_paper "2510.10401" "Knowledge-Decoupled"
download_paper "2509.20397" "Variational-LoRA"
download_paper "2407.06310" "Homogeneous-Features"
download_paper "2505.20006" "MAS-LoRA"
download_paper "2512.16401" "Privacy-LoRA"
download_paper "2511.07253" "Omni-AVSR"
download_paper "2508.12301" "CarelessWhisper"
download_paper "2406.09873" "Perceiver-Prompt"

echo ""
echo "=== Category 3: Test-Time Adaptation ==="
download_paper "2408.05769" "LI-TTA"
download_paper "2409.13095" "SUTA-SGEM-Child"
download_paper "2506.07078" "E-BATS"
download_paper "2508.01847" "TTT-SE"
download_paper "2509.25495" "EMO-TTA"

echo ""
echo "=== Category 4: Multi-Speaker ASR ==="
download_paper "2511.16046" "JEDIS-LLM"
download_paper "2510.03723" "DiCoW"
download_paper "2508.06372" "SpeakerLM"
download_paper "2310.04863" "SA-Paraformer"
download_paper "2509.13093" "GLAD-MoE"
download_paper "2509.15612" "Cocktail-Party-CoT"
download_paper "2302.11824" "MossFormer"
download_paper "2312.11825" "MossFormer2"
download_paper "2510.12275" "TFGA-Net"
download_paper "2510.23320" "LibriConvo"

echo ""
echo "=== Category 5: Speech Enhancement ==="
download_paper "2512.17562" "Denoising-Hurts-Medical"
download_paper "2510.04157" "GDiffuSE"
download_paper "2509.21522" "Shortcut-Flow"
download_paper "2509.04851" "Quantum-Fourier"
download_paper "2506.22001" "WTFormer"

echo ""
echo "=== Category 6: Self-Supervised Speech ==="
download_paper "2512.21204" "SpidR-Adapt"
download_paper "2511.12690" "MauBERT"
download_paper "2510.25955" "SPEAR"
download_paper "2509.23238" "WavJEPA"
download_paper "2510.17662" "DELULU"

echo ""
echo "=== Category 7: CTC Improvements ==="
download_paper "2410.05101" "CR-CTC"
download_paper "2508.07315" "FlexCTC"
download_paper "2506.22846" "LLM-CTC"
download_paper "2508.03937" "LCS-CTC"
download_paper "2406.02560" "Label-Priors-CTC"

echo ""
echo "=== Category 8: Streaming ASR ==="
download_paper "2512.11543" "All-in-One-ASR"
download_paper "2506.14434" "Zipformer-Unified"
download_paper "2409.07165" "Linear-Conformers"

echo ""
echo "=== Category 9: Speaker Embeddings ==="
download_paper "2409.15782" "M-Vec"
download_paper "2501.16542" "UniPET-SPK"
download_paper "2410.05037" "Contrastive-SPK"
download_paper "2505.14561" "SSPS"

echo ""
echo "=== Category 10: Hallucination Mitigation ==="
download_paper "2511.14219" "Listen-Like-Teacher"
download_paper "2501.11378" "BoH-Investigation"
download_paper "2503.09905" "Quantization-Hallucination"
download_paper "2510.12851" "Adaptive-Steering"

echo ""
echo "=== Category 11: VAD ==="
download_paper "2512.17281" "LibriVAD"
download_paper "2508.20885" "SincQDR-VAD"
download_paper "2507.22157" "Tiny-VAD"
download_paper "2403.05772" "sVAD-SNN"
download_paper "2312.16613" "SSL-VAD"

echo ""
echo "=== Category 12: Architecture ==="
download_paper "2509.01087" "NoisyD-CT"
download_paper "2505.21245" "One-bit-ASR"
download_paper "2409.00481" "DCIM-AVSR"

echo ""
echo "=== Download Complete ==="
TOTAL=$(ls -1 "$PAPER_DIR"/*.pdf 2>/dev/null | wc -l)
echo "Total papers downloaded: $TOTAL"
echo "Location: $PAPER_DIR"
