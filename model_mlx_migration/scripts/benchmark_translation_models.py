#!/usr/bin/env /Users/ayates/model_mlx_migration/.venv/bin/python3
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
Translation Model Benchmark Script

Compares multiple translation models using GPT-5.2 as judge.
Uses 0-100 rating scale for semantic accuracy, fluency, and overall quality.

Usage:
    python scripts/benchmark_translation_models.py \
        --models m2m100,opus-mt --languages zh,ja,de
    python scripts/benchmark_translation_models.py --all --limit 50
"""

import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass
class TranslationResult:
    """Single translation result with evaluation."""
    model: str
    source_lang: str
    target_lang: str
    source_text: str
    translation: str
    latency_ms: float
    rating: Optional[int] = None
    accuracy: Optional[int] = None
    fluency: Optional[int] = None
    explanation: Optional[str] = None


@dataclass
class ModelBenchmark:
    """Benchmark results for a single model."""
    model_name: str
    model_id: str
    total_translations: int
    avg_rating: float
    avg_accuracy: float
    avg_fluency: float
    avg_latency_ms: float
    per_language: dict


class TranslationModel:
    """Base class for translation models."""

    def __init__(self, name: str, model_id: str):
        self.name = name
        self.model_id = model_id
        self.model = None
        self.tokenizer = None

    def load(self):
        raise NotImplementedError

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        raise NotImplementedError


class M2M100Model(TranslationModel):
    """Facebook M2M-100 multilingual model."""

    LANG_MAP = {
        'zh': 'zh', 'ja': 'ja', 'ko': 'ko', 'de': 'de', 'fr': 'fr',
        'es': 'es', 'pt': 'pt', 'ru': 'ru', 'ar': 'ar', 'hi': 'hi',
        'en': 'en', 'it': 'it', 'nl': 'nl', 'pl': 'pl', 'tr': 'tr',
        'vi': 'vi', 'th': 'th', 'he': 'he'
    }

    def __init__(self):
        super().__init__("M2M-100", "facebook/m2m100_1.2B")

    def load(self):
        from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
        print(f"Loading {self.name}...")
        self.tokenizer = M2M100Tokenizer.from_pretrained(self.model_id)
        self.model = M2M100ForConditionalGeneration.from_pretrained(self.model_id)
        print(f"  {self.name} loaded")

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        src = self.LANG_MAP.get(src_lang, src_lang)
        tgt = self.LANG_MAP.get(tgt_lang, tgt_lang)

        self.tokenizer.src_lang = src
        encoded = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        )
        generated = self.model.generate(
            **encoded,
            forced_bos_token_id=self.tokenizer.get_lang_id(tgt),
            max_length=512
        )
        return self.tokenizer.decode(generated[0], skip_special_tokens=True)


class OpusMTModel(TranslationModel):
    """Helsinki-NLP OPUS-MT models (language-pair specific)."""

    # Available OPUS-MT models (en -> X)
    # Note: Not all language pairs exist. Korean (ko) not available.
    MODELS = {
        'zh': 'Helsinki-NLP/opus-mt-en-zh',
        'ja': 'Helsinki-NLP/opus-mt-en-jap',  # Uses 'jap' not 'ja'
        'de': 'Helsinki-NLP/opus-mt-en-de',
        'fr': 'Helsinki-NLP/opus-mt-en-fr',
        'es': 'Helsinki-NLP/opus-mt-en-es',
        'pt': 'Helsinki-NLP/opus-mt-en-ROMANCE',
        'ru': 'Helsinki-NLP/opus-mt-en-ru',
        'ar': 'Helsinki-NLP/opus-mt-en-ar',
        # 'ko': NOT AVAILABLE - use M2M-100 for Korean
    }

    def __init__(self, target_lang: str):
        default_id = f'Helsinki-NLP/opus-mt-en-{target_lang}'
        model_id = self.MODELS.get(target_lang, default_id)
        super().__init__(f"OPUS-MT-{target_lang}", model_id)
        self.target_lang = target_lang

    def load(self):
        from transformers import MarianMTModel, MarianTokenizer
        print(f"Loading {self.name}...")
        try:
            self.tokenizer = MarianTokenizer.from_pretrained(self.model_id)
            self.model = MarianMTModel.from_pretrained(self.model_id)
            print(f"  {self.name} loaded")
        except Exception as e:
            print(f"  {self.name} failed to load: {e}")
            self.model = None

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        if self.model is None:
            return "[MODEL NOT AVAILABLE]"
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        )
        outputs = self.model.generate(**inputs, max_length=512)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


class MADLADModel(TranslationModel):
    """Google MADLAD-400 3B model (our baseline)."""

    def __init__(self):
        super().__init__("MADLAD-400", "google/madlad400-3b-mt")

    def load(self):
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        print(f"Loading {self.name}...")
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_id)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_id)
        print(f"  {self.name} loaded")

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        # MADLAD uses <2xx> prefix format
        prompt = f"<2{tgt_lang}> {text}"
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        )
        outputs = self.model.generate(**inputs, max_length=512)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


class GPT52Judge:
    """GPT-5.2 quality evaluator."""

    def __init__(self):
        from openai import OpenAI
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        self.client = OpenAI(api_key=api_key)

    def evaluate(
        self, source: str, translation: str, src_lang: str, tgt_lang: str
    ) -> dict:
        """Evaluate translation quality using GPT-5.2."""
        lang_names = {
            'en': 'English', 'zh': 'Chinese', 'ja': 'Japanese', 'ko': 'Korean',
            'de': 'German', 'fr': 'French', 'es': 'Spanish', 'pt': 'Portuguese',
            'ru': 'Russian', 'ar': 'Arabic', 'hi': 'Hindi', 'it': 'Italian',
            'nl': 'Dutch', 'pl': 'Polish', 'tr': 'Turkish', 'vi': 'Vietnamese',
            'th': 'Thai', 'he': 'Hebrew'
        }

        src_name = lang_names.get(src_lang, src_lang)
        tgt_name = lang_names.get(tgt_lang, tgt_lang)

        prompt = f"""Rate this translation from {src_name} to {tgt_name}.

Source ({src_name}): {source}
Translation ({tgt_name}): {translation}

Return JSON with these fields:
- rating (0-100): Overall translation quality
- accuracy (0-100): How well the meaning is preserved
- fluency (0-100): How natural it sounds in {tgt_name}
- explanation: Brief explanation (1-2 sentences)

Be strict but fair. A perfect human translation would be 90-100.
Machine translations typically score 70-85 for good quality.
"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-5.2",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_completion_tokens=300,
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"  GPT-5.2 error: {e}")
            return {
                "rating": 0, "accuracy": 0, "fluency": 0,
                "explanation": f"Error: {e}"
            }


def load_benchmark_data(
    path: str = "tests/quality/translation_benchmark_data.json"
) -> dict:
    """Load benchmark sentences."""
    with open(path, 'r') as f:
        return json.load(f)


def run_benchmark(
    models: list[TranslationModel],
    target_languages: list[str],
    sentences: list[str],
    judge: GPT52Judge,
    limit: Optional[int] = None
) -> list[TranslationResult]:
    """Run benchmark across all models and languages."""
    results = []

    if limit and len(sentences) > limit:
        sentences = random.sample(sentences, limit)

    total = len(models) * len(target_languages) * len(sentences)
    n_models = len(models)
    n_langs = len(target_languages)
    n_sents = len(sentences)
    print(f"\nRunning benchmark: {n_models} models x {n_langs} languages "
          f"x {n_sents} sentences = {total} translations")

    for model in models:
        print(f"\n{'='*60}")
        print(f"Model: {model.name}")
        print(f"{'='*60}")

        for lang in target_languages:
            print(f"\n  Language: en -> {lang}")
            lang_results = []

            for i, sentence in enumerate(sentences):
                # Translate
                start = time.time()
                try:
                    translation = model.translate(sentence, 'en', lang)
                except Exception as e:
                    translation = f"[ERROR: {e}]"
                latency = (time.time() - start) * 1000

                # Evaluate with GPT-5.2
                eval_result = judge.evaluate(sentence, translation, 'en', lang)

                result = TranslationResult(
                    model=model.name,
                    source_lang='en',
                    target_lang=lang,
                    source_text=sentence,
                    translation=translation,
                    latency_ms=latency,
                    rating=eval_result.get('rating'),
                    accuracy=eval_result.get('accuracy'),
                    fluency=eval_result.get('fluency'),
                    explanation=eval_result.get('explanation')
                )
                results.append(result)
                lang_results.append(result)

                # Progress
                if (i + 1) % 10 == 0:
                    avg = sum(r.rating or 0 for r in lang_results) / len(lang_results)
                    print(f"    {i+1}/{len(sentences)}: avg rating = {avg:.1f}")

            # Language summary
            avg_rating = sum(r.rating or 0 for r in lang_results) / len(lang_results)
            avg_latency = sum(r.latency_ms for r in lang_results) / len(lang_results)
            print(f"    {lang} complete: avg={avg_rating:.1f}, "
                  f"latency={avg_latency:.0f}ms")

    return results


def generate_report(results: list[TranslationResult], output_path: str):
    """Generate markdown report from benchmark results."""
    # Aggregate by model
    model_stats = {}
    for r in results:
        if r.model not in model_stats:
            model_stats[r.model] = {
                'ratings': [], 'accuracies': [], 'fluencies': [], 'latencies': [],
                'by_lang': {}
            }

        stats = model_stats[r.model]
        if r.rating is not None:
            stats['ratings'].append(r.rating)
            stats['accuracies'].append(r.accuracy or 0)
            stats['fluencies'].append(r.fluency or 0)
        stats['latencies'].append(r.latency_ms)

        if r.target_lang not in stats['by_lang']:
            stats['by_lang'][r.target_lang] = {'ratings': [], 'latencies': []}
        if r.rating is not None:
            stats['by_lang'][r.target_lang]['ratings'].append(r.rating)
        stats['by_lang'][r.target_lang]['latencies'].append(r.latency_ms)

    # Generate report
    lines = [
        "# Translation Model Benchmark Results",
        "",
        f"**Date**: {time.strftime('%Y-%m-%d')}",
        "**Judge**: GPT-5.2",
        f"**Total Translations**: {len(results)}",
        "",
        "---",
        "",
        "## Summary",
        "",
        "| Model | Avg Rating | Avg Accuracy | Avg Fluency | Avg Latency |",
        "|-------|------------|--------------|-------------|-------------|",
    ]

    # Sort by rating
    def avg_rating(x):
        ratings = x[1]['ratings']
        return sum(ratings) / len(ratings) if ratings else 0
    sorted_models = sorted(model_stats.items(), key=avg_rating, reverse=True)

    for model, stats in sorted_models:
        if stats['ratings']:
            avg_r = sum(stats['ratings']) / len(stats['ratings'])
            avg_a = sum(stats['accuracies']) / len(stats['accuracies'])
            avg_f = sum(stats['fluencies']) / len(stats['fluencies'])
            avg_l = sum(stats['latencies']) / len(stats['latencies'])
            row = f"| {model} | {avg_r:.1f} | {avg_a:.1f} | {avg_f:.1f} "
            lines.append(row + f"| {avg_l:.0f}ms |")

    lines.extend([
        "",
        "---",
        "",
        "## Per-Language Results",
        "",
    ])

    for model, stats in sorted_models:
        lines.append(f"### {model}")
        lines.append("")
        lines.append("| Language | Avg Rating | Samples | Avg Latency |")
        lines.append("|----------|------------|---------|-------------|")

        for lang, lang_stats in sorted(stats['by_lang'].items()):
            if lang_stats['ratings']:
                avg_r = sum(lang_stats['ratings']) / len(lang_stats['ratings'])
                avg_l = sum(lang_stats['latencies']) / len(lang_stats['latencies'])
                n_samples = len(lang_stats['ratings'])
                lines.append(f"| {lang} | {avg_r:.1f} | {n_samples} | {avg_l:.0f}ms |")

        lines.append("")

    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"\nReport saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark translation models")
    parser.add_argument(
        '--models', default='m2m100',
        help='Comma-separated: m2m100,opus-mt,madlad'
    )
    parser.add_argument(
        '--languages', default='zh,ja,de',
        help='Comma-separated target languages'
    )
    parser.add_argument(
        '--limit', type=int, default=20,
        help='Max sentences per language'
    )
    parser.add_argument(
        '--output', default='reports/main/TRANSLATION_BENCHMARK_RESULTS.md',
        help='Output report path'
    )
    parser.add_argument(
        '--all', action='store_true',
        help='Run all available models'
    )
    args = parser.parse_args()

    # Load benchmark data
    data = load_benchmark_data()
    all_sentences = []
    for category, sents in data['sentences'].items():
        all_sentences.extend(sents)
    print(f"Loaded {len(all_sentences)} benchmark sentences")

    # Initialize judge
    print("Initializing GPT-5.2 judge...")
    judge = GPT52Judge()

    # Parse languages
    languages = args.languages.split(',')

    # Load models
    models = []
    if args.all:
        model_names = ['m2m100', 'opus-mt', 'madlad']
    else:
        model_names = args.models.lower().split(',')

    if 'm2m100' in model_names:
        m = M2M100Model()
        m.load()
        models.append(m)

    if 'opus-mt' in model_names:
        for lang in languages:
            m = OpusMTModel(lang)
            m.load()
            if m.model is not None:
                models.append(m)

    if 'madlad' in model_names:
        m = MADLADModel()
        m.load()
        models.append(m)

    if not models:
        print("No models loaded!")
        return

    print(f"\nLoaded {len(models)} models: {[m.name for m in models]}")

    # Run benchmark
    results = run_benchmark(models, languages, all_sentences, judge, limit=args.limit)

    # Generate report
    generate_report(results, args.output)

    # Save raw results
    raw_path = args.output.replace('.md', '_raw.json')
    with open(raw_path, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"Raw results saved to: {raw_path}")


if __name__ == '__main__':
    main()
