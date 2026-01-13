#!/usr/bin/env python3
"""Download CREMA-D dataset from HuggingFace."""

import os

try:
    from datasets import load_dataset
except ImportError:
    print("Installing datasets library...")
    os.system("pip install datasets soundfile")
    from datasets import load_dataset

def main():
    output_dir = "data/emotion/crema-d_hf"
    os.makedirs(output_dir, exist_ok=True)

    print("Downloading CREMA-D from HuggingFace...")
    print("This may take a while (~5GB)...")

    # Try NoahMartinezXiang/CREMA-D first
    try:
        dataset = load_dataset("NoahMartinezXiang/CREMA-D")
        print(f"Downloaded dataset with {len(dataset)} splits")

        # Save to disk
        dataset.save_to_disk(output_dir)
        print(f"Saved to {output_dir}")

        # Print info
        for split, data in dataset.items():
            print(f"  {split}: {len(data)} samples")
            if len(data) > 0:
                print(f"    Features: {list(data.features.keys())}")

    except Exception as e:
        print(f"Error with NoahMartinezXiang/CREMA-D: {e}")
        print("Trying MahiA/CREMA-D...")

        dataset = load_dataset("MahiA/CREMA-D")
        dataset.save_to_disk(output_dir)
        print(f"Saved to {output_dir}")

if __name__ == "__main__":
    main()
