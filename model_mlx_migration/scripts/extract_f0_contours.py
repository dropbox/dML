#!/usr/bin/env python3
# Copyright 2024-2025 Andrew Yates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Extract F0 contours from all prosody training samples.
This prepares data for Path C contour training.

Usage:
    python scripts/extract_f0_contours.py --input data/prosody/multilingual_train.json --output data/prosody/contours_train.json
"""

import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_f0_contour(audio_path: str, sr: int = 24000) -> dict:
    """Extract F0 contour from a single audio file."""
    import librosa

    try:
        # Load audio
        y, _ = librosa.load(audio_path, sr=sr)

        # Extract F0 using YIN (faster than PYIN)
        f0 = librosa.yin(
            y,
            fmin=50,
            fmax=500,
            sr=sr,
            frame_length=1024,
            hop_length=256
        )

        # Handle invalid values
        f0_clean = np.copy(f0)
        f0_clean[(f0_clean < 50) | (f0_clean > 500)] = np.nan

        # Interpolate unvoiced regions
        valid_indices = np.where(~np.isnan(f0_clean))[0]
        if len(valid_indices) > 2:
            all_indices = np.arange(len(f0_clean))
            f0_interp = np.interp(all_indices, valid_indices, f0_clean[valid_indices])
        else:
            # Not enough voiced frames
            mean_f0 = np.nanmean(f0_clean) if not np.all(np.isnan(f0_clean)) else 150.0
            f0_interp = np.full_like(f0_clean, mean_f0)

        # Compute statistics
        f0_valid = f0_clean[~np.isnan(f0_clean)]
        if len(f0_valid) > 0:
            f0_mean = float(np.mean(f0_valid))
            f0_std = float(np.std(f0_valid))
            f0_min = float(np.min(f0_valid))
            f0_max = float(np.max(f0_valid))
        else:
            f0_mean, f0_std, f0_min, f0_max = 150.0, 30.0, 100.0, 200.0

        # Normalize contour to [0, 1] range for portability
        if f0_max > f0_min:
            f0_normalized = (f0_interp - f0_min) / (f0_max - f0_min)
        else:
            f0_normalized = np.zeros_like(f0_interp) + 0.5

        # Downsample contour to fixed length (50 points)
        target_len = 50
        if len(f0_normalized) > target_len:
            indices = np.linspace(0, len(f0_normalized) - 1, target_len, dtype=int)
            f0_downsampled = f0_normalized[indices]
        else:
            f0_downsampled = np.pad(f0_normalized, (0, target_len - len(f0_normalized)), mode='edge')

        voiced_count = len(valid_indices)
        total_frames = len(f0)

        return {
            'f0_contour': f0_downsampled.tolist(),
            'f0_mean': f0_mean,
            'f0_std': f0_std,
            'f0_min': f0_min,
            'f0_max': f0_max,
            'f0_range': f0_max - f0_min,
            'duration_frames': total_frames,
            'voiced_ratio': float(voiced_count / total_frames) if total_frames > 0 else 0.0,
        }
    except Exception:
        return None


def process_sample(sample: dict) -> dict:
    """Process a single sample, adding F0 contour data."""
    audio_path = sample.get('audio_path')
    if not audio_path or not os.path.exists(audio_path):
        return None

    contour_data = extract_f0_contour(audio_path)
    if contour_data is None:
        return None

    # Merge contour data into sample
    result = {**sample, **contour_data}
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Extract F0 contours from prosody samples')
    parser.add_argument('--input', default='data/prosody/multilingual_train.json', help='Input JSON file')
    parser.add_argument('--output', default='data/prosody/contours_train.json', help='Output JSON file')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of samples (for testing)')
    args = parser.parse_args()

    # Load input data
    logger.info(f"Loading {args.input}...")
    with open(args.input, 'r') as f:
        samples = json.load(f)

    if args.limit:
        samples = samples[:args.limit]

    logger.info(f"Processing {len(samples)} samples with {args.workers} workers...")

    results = []
    failed = 0

    # Process in parallel
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_sample, s): i for i, s in enumerate(samples)}

        for i, future in enumerate(as_completed(futures)):
            if (i + 1) % 5000 == 0:
                pct = 100 * (i + 1) / len(samples)
                logger.info(f"Progress: {i + 1}/{len(samples)} ({pct:.1f}%) - {len(results)} success, {failed} failed")

            try:
                result = future.result()
                if result:
                    results.append(result)
                else:
                    failed += 1
            except Exception:
                failed += 1

    logger.info(f"Processed {len(results)} samples successfully, {failed} failed")

    # Save output
    logger.info(f"Saving to {args.output}...")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f)

    # Print summary statistics
    if results:
        f0_means = [r['f0_mean'] for r in results if r.get('f0_mean')]
        logger.info("F0 statistics:")
        logger.info(f"  Mean F0: {np.mean(f0_means):.1f} Hz")
        logger.info(f"  Std F0: {np.std(f0_means):.1f} Hz")
        logger.info(f"  Min F0: {np.min(f0_means):.1f} Hz")
        logger.info(f"  Max F0: {np.max(f0_means):.1f} Hz")

    logger.info("Done!")


if __name__ == '__main__':
    main()
