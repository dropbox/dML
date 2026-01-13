#!/bin/bash
# Download GigaSpeech dataset
# License: See dataset page (may be gated and/or non-commercial)
# Source: https://github.com/SpeechColab/GigaSpeech

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
cd "$BASE_DIR"

VENV_DIR="${VENV_DIR:-venv}"
if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  source "${VENV_DIR}/bin/activate"
fi

echo "=== Downloading GigaSpeech ==="
echo ""
echo "GigaSpeech is a 10,000 hour English ASR corpus."
echo "Subset options: xs (10h), s (250h), m (1000h), l (2500h), xl (10000h)"
echo ""

# Create directory
mkdir -p data/gigaspeech

# Access via HuggingFace datasets.
# Defaults to STREAMING mode to avoid large downloads; set GIGASPEECH_DOWNLOAD=1 to download.
python3 << 'EOF'
from datasets import load_dataset
import os

config = os.environ.get("GIGASPEECH_CONFIG", "xs")  # xs/s/m/l/xl
do_download = os.environ.get("GIGASPEECH_DOWNLOAD", "0") == "1"
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

print(f"Config: {config}")
print(f"Mode: {'download' if do_download else 'streaming'}")
print("")

os.makedirs("data/gigaspeech", exist_ok=True)

try:
    dataset = load_dataset(
        "speechcolab/gigaspeech",
        config,
        streaming=not do_download,
        token=token,
        cache_dir="data/gigaspeech/cache" if do_download else None,
    )

    if do_download:
        print(f"Downloaded {len(dataset['train'])} training examples")
        print(f"Downloaded {len(dataset['validation'])} validation examples")
        print(f"Downloaded {len(dataset['test'])} test examples")
    else:
        train_iter = iter(dataset["train"])
        first = next(train_iter)
        print("Streaming access OK. First example keys:")
        print(sorted(first.keys()))

    # Save info
    with open("data/gigaspeech/info.txt", "w") as f:
        f.write(f"Config: {config}\n")
        f.write(f"Mode: {'download' if do_download else 'streaming'}\n")

    print("")
    print("=== Download Complete ===")
    if do_download:
        print("Data saved to: data/gigaspeech/cache/")
    else:
        print("No data downloaded (streaming mode). Set GIGASPEECH_DOWNLOAD=1 to download.")

except Exception as e:
    print(f"ERROR: {e}")
    print("")
    print("If you see a 'gated dataset' error:")
    print("1. Visit: https://huggingface.co/datasets/speechcolab/gigaspeech")
    print("2. Request/accept access per the dataset instructions")
    print("3. Export HF_TOKEN with your HuggingFace token and retry")
EOF

echo ""
echo "Checking download size..."
du -sh data/gigaspeech/ 2>/dev/null || echo "Directory empty or not created"
