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
MADLAD Translation Generation Script

Generates translations from MADLAD-3B, 7B, 10B for quality validation.
Uses NTREX benchmark which has human reference translations.

Usage:
    python scripts/generate_madlad_translations.py --models 3b,7b,10b --samples 500
    python scripts/generate_madlad_translations.py --models 3b --samples 100 --languages fr,de

Output: data/translations/madlad_translations_YYYY-MM-DD.json
"""

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class TranslationResult:
    """Single translation result."""
    id: int
    source_text: str
    source_lang: str
    target_lang: str
    model_size: str
    model_translation: str
    reference_translation: str
    latency_ms: float
    tokens_generated: int


@dataclass
class ValidationDataset:
    """Complete validation dataset."""
    metadata: dict
    translations: List[dict]


# Language code mapping (NTREX to MADLAD format)
LANG_MAP = {
    "fra": "fr",  # French
    "deu": "de",  # German
    "spa": "es",  # Spanish
    "zho_hans": "zh",  # Chinese Simplified
    "jpn": "ja",  # Japanese
    "kor": "ko",  # Korean
    "arb": "ar",  # Arabic
    "heb": "he",  # Hebrew
    "rus": "ru",  # Russian
    "por": "pt",  # Portuguese
    "vie": "vi",  # Vietnamese
    "tha": "th",  # Thai
    "ita": "it",  # Italian
    "hin": "hi",  # Hindi
}

# Priority languages for validation
PRIORITY_LANGS = ["fr", "de", "es", "zh", "ja", "ko", "he", "ar", "ru"]


def load_ntrex_benchmark(benchmark_path: str, max_samples: int = None) -> List[dict]:
    """Load NTREX benchmark data."""
    with open(benchmark_path) as f:
        data = json.load(f)

    samples = data["data"]
    if max_samples:
        samples = samples[:max_samples]

    return samples


def generate_translations(
    model_sizes: List[str],
    samples: List[dict],
    target_langs: List[str],
    quantize: int = 8,
) -> List[TranslationResult]:
    """Generate translations for all model sizes and languages."""
    from tools.pytorch_to_mlx.converters.madlad_converter import MADLADConverter

    results = []
    model_ids = {
        "3b": "google/madlad400-3b-mt",
        "7b": "google/madlad400-7b-mt",
        "10b": "google/madlad400-10b-mt",
    }

    for model_size in model_sizes:
        print(f"\n{'='*60}")
        print(f"Loading MADLAD-{model_size.upper()}")
        print("=" * 60)

        # Load model
        model_id = model_ids[model_size]
        converter = MADLADConverter(model_path=model_id, quantize=quantize)
        converter.load()

        # Warmup
        print("Warming up...")
        _ = converter.translate("Hello world.", "fr")

        # Process each language
        for tgt_lang in target_langs:
            # Find corresponding NTREX language code
            ntrex_lang = None
            for ntrex_code, madlad_code in LANG_MAP.items():
                if madlad_code == tgt_lang:
                    ntrex_lang = ntrex_code
                    break

            if not ntrex_lang:
                print(f"  WARNING: No NTREX reference for language {tgt_lang}, skipping")
                continue

            print(f"\n  Translating to {tgt_lang} ({len(samples)} samples)...")

            for i, sample in enumerate(samples):
                source_text = sample["eng"]
                reference = sample.get(ntrex_lang, "")

                if not reference:
                    continue

                # Translate
                result = converter.translate(source_text, tgt_lang)

                results.append(TranslationResult(
                    id=sample["id"],
                    source_text=source_text,
                    source_lang="en",
                    target_lang=tgt_lang,
                    model_size=model_size,
                    model_translation=result.text,
                    reference_translation=reference,
                    latency_ms=result.latency_ms,
                    tokens_generated=result.tokens_generated,
                ))

                if (i + 1) % 100 == 0:
                    print(f"    Progress: {i+1}/{len(samples)}")

        # Cleanup
        del converter
        gc.collect()

    return results


def main():
    parser = argparse.ArgumentParser(description="Generate MADLAD translations for quality validation")
    parser.add_argument(
        "--models",
        default="3b,7b,10b",
        help="Comma-separated model sizes (3b, 7b, 10b)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=500,
        help="Number of NTREX samples to use (default: 500, max: 1997)",
    )
    parser.add_argument(
        "--languages",
        default="fr,de,es,zh,ja,ko,he,ar,ru",
        help="Comma-separated target languages",
    )
    parser.add_argument(
        "--quantize",
        type=int,
        default=8,
        choices=[4, 8],
        help="Quantization bits (default: 8)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path (default: data/translations/madlad_translations_YYYY-MM-DD.json)",
    )
    parser.add_argument(
        "--benchmark",
        default="data/benchmarks/ntrex/ntrex128.json",
        help="Path to NTREX benchmark file",
    )
    args = parser.parse_args()

    # Parse arguments
    model_sizes = [s.strip().lower() for s in args.models.split(",")]
    target_langs = [lang.strip().lower() for lang in args.languages.split(",")]

    # Validate
    valid_sizes = ["3b", "7b", "10b"]
    for size in model_sizes:
        if size not in valid_sizes:
            print(f"ERROR: Invalid model size '{size}'. Valid: {valid_sizes}")
            return 1

    # Load benchmark
    print(f"Loading NTREX benchmark from {args.benchmark}...")
    if not os.path.exists(args.benchmark):
        print(f"ERROR: Benchmark file not found: {args.benchmark}")
        return 1

    samples = load_ntrex_benchmark(args.benchmark, args.samples)
    print(f"Loaded {len(samples)} samples")

    # Generate translations
    print("\nGenerating translations:")
    print(f"  Models: {model_sizes}")
    print(f"  Languages: {target_langs}")
    print(f"  Samples: {len(samples)}")
    print(f"  Total translations: {len(model_sizes) * len(target_langs) * len(samples)}")

    start_time = time.time()
    results = generate_translations(
        model_sizes=model_sizes,
        samples=samples,
        target_langs=target_langs,
        quantize=args.quantize,
    )
    elapsed = time.time() - start_time

    print(f"\nGenerated {len(results)} translations in {elapsed/60:.1f} minutes")

    # Save results
    if args.output:
        output_path = args.output
    else:
        date_str = datetime.now().strftime("%Y-%m-%d")
        output_path = f"data/translations/madlad_translations_{date_str}.json"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    dataset = ValidationDataset(
        metadata={
            "generated_at": datetime.now().isoformat(),
            "models": model_sizes,
            "languages": target_langs,
            "samples": len(samples),
            "total_translations": len(results),
            "quantization": args.quantize,
            "benchmark_source": args.benchmark,
            "generation_time_minutes": elapsed / 60,
        },
        translations=[asdict(r) for r in results],
    )

    with open(output_path, "w") as f:
        json.dump(asdict(dataset), f, indent=2, ensure_ascii=False)

    print(f"Saved to: {output_path}")

    # Summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("=" * 60)

    by_model = {}
    by_lang = {}
    for r in results:
        by_model.setdefault(r.model_size, []).append(r)
        by_lang.setdefault(r.target_lang, []).append(r)

    print("\nBy Model:")
    for model, items in sorted(by_model.items()):
        avg_latency = sum(i.latency_ms for i in items) / len(items)
        avg_tokens = sum(i.tokens_generated for i in items) / len(items)
        print(f"  {model.upper()}: {len(items)} translations, avg {avg_latency:.0f}ms, avg {avg_tokens:.1f} tokens")

    print("\nBy Language:")
    for lang, items in sorted(by_lang.items()):
        print(f"  {lang}: {len(items)} translations")

    return 0


if __name__ == "__main__":
    sys.exit(main())
