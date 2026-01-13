#!/usr/bin/env python3
"""
Download SOTA Models for Comparison and Knowledge Distillation

Models to download:
1. emotion2vec-large - Emotion recognition SOTA
2. wav2vec2-xlsr-SER - Speech emotion recognition
3. BEATs - Audio classification SOTA (paralinguistics)
4. AST - Audio Spectrogram Transformer
5. ECAPA-TDNN - Language identification
6. wav2vec2-large-xlsr-53 - General audio features

Usage:
    python scripts/download_sota_models.py --all
    python scripts/download_sota_models.py --model emotion2vec
    python scripts/download_sota_models.py --model beats
"""

import argparse
from pathlib import Path
from datetime import datetime

# Models to download
SOTA_MODELS = {
    "emotion2vec": {
        "repo_id": "emotion2vec/emotion2vec_base",
        "description": "Emotion recognition SOTA - self-supervised on emotion",
        "size": "~380MB",
        "task": "emotion",
        "source": "https://github.com/ddlBoJack/emotion2vec"
    },
    "wav2vec2-xlsr-ser": {
        "repo_id": "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        "description": "wav2vec2 fine-tuned for speech emotion recognition",
        "size": "~1.2GB",
        "task": "emotion",
        "metric": "82.23% on RAVDESS"
    },
    "beats": {
        "repo_id": "microsoft/beats",
        "description": "Audio classification SOTA - bidirectional encoder",
        "size": "~90MB",
        "task": "paralinguistics",
        "metric": "96.4% on ESC-50"
    },
    "ast": {
        "repo_id": "MIT/ast-finetuned-audioset-10-10-0.4593",
        "description": "Audio Spectrogram Transformer",
        "size": "~87MB",
        "task": "paralinguistics",
        "metric": "95.6% on ESC-50"
    },
    "ecapa-tdnn": {
        "repo_id": "speechbrain/lang-id-voxlingua107-ecapa",
        "description": "Language identification on 107 languages",
        "size": "~80MB",
        "task": "language_id",
        "metric": "93.3% on VoxLingua107"
    },
    "wav2vec2-xlsr": {
        "repo_id": "facebook/wav2vec2-large-xlsr-53",
        "description": "Multilingual wav2vec2 for phoneme recognition",
        "size": "~1.2GB",
        "task": "phoneme",
        "metric": "SOTA on SUPERB"
    },
    "wavlm-large": {
        "repo_id": "microsoft/wavlm-large",
        "description": "WavLM for speech understanding tasks",
        "size": "~1.2GB",
        "task": "general",
        "metric": "SOTA on SUPERB benchmark"
    }
}


def download_huggingface_model(repo_id: str, output_dir: Path) -> bool:
    """Download a model from HuggingFace Hub."""
    try:
        from huggingface_hub import snapshot_download
        print(f"  Downloading {repo_id}...")
        local_path = snapshot_download(
            repo_id=repo_id,
            local_dir=output_dir,
            local_dir_use_symlinks=False
        )
        print(f"  Downloaded to: {local_path}")
        return True
    except Exception as e:
        print(f"  Error downloading {repo_id}: {e}")
        return False


def download_beats_official() -> bool:
    """Download BEATs from HuggingFace mirror (lpepino/beats_ckpts)."""
    print("  Downloading BEATs from HuggingFace (lpepino/beats_ckpts)...")

    output_dir = Path("models/sota/beats")
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / "BEATs_iter3_plus_AS2M.pt"

    if checkpoint_path.exists():
        print(f"  BEATs checkpoint already exists at {checkpoint_path}")
        return True

    try:
        from huggingface_hub import hf_hub_download

        # Download the best BEATs model (iter3+ AS2M - achieves 96.4% on ESC-50)
        print("  Downloading BEATs_iter3_plus_AS2M.pt...")
        path = hf_hub_download(
            repo_id='lpepino/beats_ckpts',
            filename='BEATs_iter3_plus_AS2M.pt',
            local_dir=str(output_dir)
        )
        print(f"  Downloaded to: {path}")

        # Also download the fine-tuned version
        print("  Downloading BEATs_iter3_finetuned_on_AS2M_cpt1.pt...")
        path2 = hf_hub_download(
            repo_id='lpepino/beats_ckpts',
            filename='BEATs_iter3_finetuned_on_AS2M_cpt1.pt',
            local_dir=str(output_dir)
        )
        print(f"  Downloaded to: {path2}")
        return True
    except Exception as e:
        print(f"  Error downloading BEATs: {e}")
        return False


def download_model(model_name: str, output_base: Path) -> bool:
    """Download a specific SOTA model."""
    if model_name not in SOTA_MODELS:
        print(f"Unknown model: {model_name}")
        print(f"Available: {list(SOTA_MODELS.keys())}")
        return False

    model_info = SOTA_MODELS[model_name]
    print(f"\nDownloading {model_name}...")
    print(f"  Description: {model_info['description']}")
    print(f"  Size: {model_info['size']}")
    print(f"  Task: {model_info['task']}")

    output_dir = output_base / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Special handling for BEATs
    if model_name == "beats":
        return download_beats_official()

    # Standard HuggingFace download
    return download_huggingface_model(model_info["repo_id"], output_dir)


def main():
    parser = argparse.ArgumentParser(description="Download SOTA Models")
    parser.add_argument("--model", type=str, help="Specific model to download")
    parser.add_argument("--all", action="store_true", help="Download all SOTA models")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--output-dir", type=str, default="models/sota",
                        help="Output directory for downloaded models")
    args = parser.parse_args()

    if args.list:
        print("Available SOTA models:")
        print("-" * 80)
        for name, info in SOTA_MODELS.items():
            print(f"\n{name}:")
            print(f"  Repo: {info['repo_id']}")
            print(f"  Description: {info['description']}")
            print(f"  Size: {info['size']}")
            print(f"  Task: {info['task']}")
            if "metric" in info:
                print(f"  Metric: {info['metric']}")
        return

    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SOTA Model Downloader")
    print("=" * 60)
    print(f"Output directory: {output_base}")
    print(f"Timestamp: {datetime.now().isoformat()}")

    results = {}

    if args.all:
        print(f"\nDownloading all {len(SOTA_MODELS)} models...")
        for model_name in SOTA_MODELS:
            results[model_name] = download_model(model_name, output_base)
    elif args.model:
        results[args.model] = download_model(args.model, output_base)
    else:
        # Default: download most useful models for our tasks
        priority_models = ["emotion2vec", "beats", "ecapa-tdnn"]
        print(f"\nDownloading priority models: {priority_models}")
        for model_name in priority_models:
            results[model_name] = download_model(model_name, output_base)

    # Summary
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    for model, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"  {model}: {status}")

    # Create index file
    index_path = output_base / "index.json"
    import json
    index = {
        "downloaded": datetime.now().isoformat(),
        "models": {name: {**info, "downloaded": results.get(name, False)}
                   for name, info in SOTA_MODELS.items()}
    }
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"\nIndex saved to: {index_path}")


if __name__ == "__main__":
    main()
