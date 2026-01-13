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
LLM Ensemble Evaluation Script for MADLAD Translations

Evaluates translation quality using GPT-5 and Claude Opus 4.5 as judges.
Produces ensemble scores with disagreement detection.

Usage:
    python scripts/llm_ensemble_evaluate.py --translations data/translations/madlad_translations.json
    python scripts/llm_ensemble_evaluate.py --translations data/translations/madlad_translations.json --parallel 10

Requires:
    - OPENAI_API_KEY environment variable
    - ANTHROPIC_API_KEY environment variable

Output: data/metrics/llm_ensemble_scores_YYYY-MM-DD.json
"""

import argparse
import asyncio
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
class EvaluationScore:
    """Score from a single LLM judge."""
    accuracy: int  # 0-100: semantic preservation
    fluency: int   # 0-100: grammatical correctness
    overall: int   # 0-100: combined assessment
    explanation: str


@dataclass
class EnsembleResult:
    """Combined result from both judges."""
    translation_id: int
    source_text: str
    target_lang: str
    model_size: str
    model_translation: str
    reference_translation: str

    # GPT-5 scores
    gpt5_accuracy: int
    gpt5_fluency: int
    gpt5_overall: int
    gpt5_explanation: str

    # Opus 4.5 scores
    opus_accuracy: int
    opus_fluency: int
    opus_overall: int
    opus_explanation: str

    # Ensemble scores
    ensemble_accuracy: float
    ensemble_fluency: float
    ensemble_overall: float

    # Disagreement flag (>15 point difference)
    disagreement: bool


EVALUATION_PROMPT = """Rate this machine translation from English to {target_lang}.

Source (English): {source}
Machine Translation: {translation}
Human Reference: {reference}

Score each dimension from 0-100:

1. **Semantic Accuracy** (0-100): Does the translation preserve the meaning of the source?
   - 90-100: Perfect or near-perfect meaning preservation
   - 70-89: Minor meaning differences that don't affect understanding
   - 50-69: Some meaning lost or changed
   - 30-49: Significant meaning distortion
   - 0-29: Completely wrong or incomprehensible

2. **Fluency** (0-100): Is the translation grammatically correct and natural?
   - 90-100: Native-quality, completely natural
   - 70-89: Minor grammatical issues
   - 50-69: Noticeable but understandable errors
   - 30-49: Frequent errors, hard to read
   - 0-29: Broken grammar, unreadable

3. **Overall Quality** (0-100): Combined assessment considering both dimensions.

Respond with ONLY valid JSON in this exact format:
{{"accuracy": <int>, "fluency": <int>, "overall": <int>, "explanation": "<brief 1-2 sentence explanation>"}}"""


class GPT5Judge:
    """GPT-5 (or GPT-4o) judge for translation evaluation."""

    def __init__(self, model: str = "gpt-4o"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model

    async def evaluate(self, source: str, translation: str, reference: str,
                       target_lang: str) -> EvaluationScore:
        """Evaluate a single translation."""
        prompt = EVALUATION_PROMPT.format(
            target_lang=target_lang,
            source=source,
            translation=translation,
            reference=reference,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=200,
            )
            content = response.choices[0].message.content.strip()

            # Parse JSON response
            result = json.loads(content)
            return EvaluationScore(
                accuracy=int(result.get("accuracy", 0)),
                fluency=int(result.get("fluency", 0)),
                overall=int(result.get("overall", 0)),
                explanation=result.get("explanation", ""),
            )
        except Exception as e:
            print(f"  GPT-5 error: {e}")
            return EvaluationScore(accuracy=0, fluency=0, overall=0, explanation=f"Error: {e}")


class OpusJudge:
    """Claude Opus 4.5 judge for translation evaluation."""

    def __init__(self, model: str = "claude-opus-4-5-20251101"):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model

    async def evaluate(self, source: str, translation: str, reference: str,
                       target_lang: str) -> EvaluationScore:
        """Evaluate a single translation."""
        prompt = EVALUATION_PROMPT.format(
            target_lang=target_lang,
            source=source,
            translation=translation,
            reference=reference,
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.content[0].text.strip()

            # Parse JSON response
            result = json.loads(content)
            return EvaluationScore(
                accuracy=int(result.get("accuracy", 0)),
                fluency=int(result.get("fluency", 0)),
                overall=int(result.get("overall", 0)),
                explanation=result.get("explanation", ""),
            )
        except Exception as e:
            print(f"  Opus error: {e}")
            return EvaluationScore(accuracy=0, fluency=0, overall=0, explanation=f"Error: {e}")


class EnsembleJudge:
    """Ensemble of GPT-5 and Opus 4.5 for translation evaluation."""

    def __init__(self, gpt_model: str = "gpt-4o", opus_model: str = "claude-opus-4-5-20251101"):
        self.gpt5 = GPT5Judge(model=gpt_model)
        self.opus = OpusJudge(model=opus_model)

    async def evaluate(self, translation_data: dict) -> EnsembleResult:
        """Evaluate a translation using both judges."""
        source = translation_data["source_text"]
        translation = translation_data["model_translation"]
        reference = translation_data["reference_translation"]
        target_lang = translation_data["target_lang"]

        # Get scores from both judges
        gpt5_score = await self.gpt5.evaluate(source, translation, reference, target_lang)
        opus_score = await self.opus.evaluate(source, translation, reference, target_lang)

        # Calculate ensemble scores
        ensemble_accuracy = (gpt5_score.accuracy + opus_score.accuracy) / 2
        ensemble_fluency = (gpt5_score.fluency + opus_score.fluency) / 2
        ensemble_overall = (gpt5_score.overall + opus_score.overall) / 2

        # Check for disagreement (>15 point difference on overall score)
        disagreement = abs(gpt5_score.overall - opus_score.overall) > 15

        return EnsembleResult(
            translation_id=translation_data["id"],
            source_text=source,
            target_lang=target_lang,
            model_size=translation_data["model_size"],
            model_translation=translation,
            reference_translation=reference,
            gpt5_accuracy=gpt5_score.accuracy,
            gpt5_fluency=gpt5_score.fluency,
            gpt5_overall=gpt5_score.overall,
            gpt5_explanation=gpt5_score.explanation,
            opus_accuracy=opus_score.accuracy,
            opus_fluency=opus_score.fluency,
            opus_overall=opus_score.overall,
            opus_explanation=opus_score.explanation,
            ensemble_accuracy=ensemble_accuracy,
            ensemble_fluency=ensemble_fluency,
            ensemble_overall=ensemble_overall,
            disagreement=disagreement,
        )


async def evaluate_batch(
    judge: EnsembleJudge,
    translations: List[dict],
    batch_size: int = 10,
    delay_ms: int = 100,
) -> List[EnsembleResult]:
    """Evaluate translations in batches with rate limiting."""
    results = []

    for i in range(0, len(translations), batch_size):
        batch = translations[i:i + batch_size]

        # Process batch
        tasks = [judge.evaluate(t) for t in batch]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in batch_results:
            if isinstance(result, Exception):
                print(f"  Error: {result}")
            else:
                results.append(result)

        # Progress
        print(f"  Evaluated {min(i + batch_size, len(translations))}/{len(translations)}")

        # Rate limiting
        await asyncio.sleep(delay_ms / 1000)

    return results


def compute_statistics(results: List[EnsembleResult]) -> dict:
    """Compute aggregate statistics from evaluation results."""
    if not results:
        return {}

    # Overall statistics
    accuracies = [r.ensemble_accuracy for r in results]
    fluencies = [r.ensemble_fluency for r in results]
    overalls = [r.ensemble_overall for r in results]
    disagreements = sum(1 for r in results if r.disagreement)

    stats = {
        "total_evaluations": len(results),
        "disagreements": disagreements,
        "disagreement_rate": disagreements / len(results),
        "overall": {
            "mean_accuracy": sum(accuracies) / len(accuracies),
            "mean_fluency": sum(fluencies) / len(fluencies),
            "mean_overall": sum(overalls) / len(overalls),
        },
    }

    # By model
    by_model = {}
    for r in results:
        if r.model_size not in by_model:
            by_model[r.model_size] = []
        by_model[r.model_size].append(r)

    stats["by_model"] = {}
    for model, model_results in by_model.items():
        model_accuracies = [r.ensemble_accuracy for r in model_results]
        model_fluencies = [r.ensemble_fluency for r in model_results]
        model_overalls = [r.ensemble_overall for r in model_results]
        stats["by_model"][model] = {
            "count": len(model_results),
            "mean_accuracy": sum(model_accuracies) / len(model_accuracies),
            "mean_fluency": sum(model_fluencies) / len(model_fluencies),
            "mean_overall": sum(model_overalls) / len(model_overalls),
        }

    # By language
    by_lang = {}
    for r in results:
        if r.target_lang not in by_lang:
            by_lang[r.target_lang] = []
        by_lang[r.target_lang].append(r)

    stats["by_language"] = {}
    for lang, lang_results in by_lang.items():
        lang_accuracies = [r.ensemble_accuracy for r in lang_results]
        lang_fluencies = [r.ensemble_fluency for r in lang_results]
        lang_overalls = [r.ensemble_overall for r in lang_results]
        stats["by_language"][lang] = {
            "count": len(lang_results),
            "mean_accuracy": sum(lang_accuracies) / len(lang_accuracies),
            "mean_fluency": sum(lang_fluencies) / len(lang_fluencies),
            "mean_overall": sum(lang_overalls) / len(lang_overalls),
        }

    return stats


async def main_async(args):
    """Async main function."""
    # Load translations
    print(f"Loading translations from {args.translations}...")
    with open(args.translations) as f:
        data = json.load(f)

    translations = data["translations"]
    print(f"Loaded {len(translations)} translations")

    # Sample if needed
    if args.sample and args.sample < len(translations):
        import random
        random.seed(42)
        translations = random.sample(translations, args.sample)
        print(f"Sampled {len(translations)} translations for evaluation")

    # Check API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set")
        return 1
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable not set")
        return 1

    # Initialize judge
    print("\nInitializing ensemble judge...")
    print(f"  GPT model: {args.gpt_model}")
    print(f"  Opus model: {args.opus_model}")
    judge = EnsembleJudge(gpt_model=args.gpt_model, opus_model=args.opus_model)

    # Evaluate
    print(f"\nEvaluating translations (batch size: {args.batch_size})...")
    start_time = time.time()

    results = await evaluate_batch(
        judge=judge,
        translations=translations,
        batch_size=args.batch_size,
        delay_ms=args.delay_ms,
    )

    elapsed = time.time() - start_time
    print(f"\nEvaluated {len(results)} translations in {elapsed/60:.1f} minutes")

    # Compute statistics
    stats = compute_statistics(results)

    # Save results
    if args.output:
        output_path = args.output
    else:
        date_str = datetime.now().strftime("%Y-%m-%d")
        output_path = f"data/metrics/llm_ensemble_scores_{date_str}.json"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    output_data = {
        "metadata": {
            "evaluated_at": datetime.now().isoformat(),
            "translations_file": args.translations,
            "total_evaluated": len(results),
            "gpt_model": args.gpt_model,
            "opus_model": args.opus_model,
            "evaluation_time_minutes": elapsed / 60,
        },
        "statistics": stats,
        "evaluations": [asdict(r) for r in results],
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Saved to: {output_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print("=" * 60)

    print(f"\nOverall (n={stats['total_evaluations']}):")
    print(f"  Mean Accuracy: {stats['overall']['mean_accuracy']:.1f}")
    print(f"  Mean Fluency:  {stats['overall']['mean_fluency']:.1f}")
    print(f"  Mean Overall:  {stats['overall']['mean_overall']:.1f}")
    print(f"  Disagreements: {stats['disagreements']} ({stats['disagreement_rate']*100:.1f}%)")

    print("\nBy Model:")
    for model, model_stats in sorted(stats["by_model"].items()):
        print(f"  {model.upper()}: acc={model_stats['mean_accuracy']:.1f}, "
              f"flu={model_stats['mean_fluency']:.1f}, "
              f"ovr={model_stats['mean_overall']:.1f} (n={model_stats['count']})")

    print("\nBy Language:")
    for lang, lang_stats in sorted(stats["by_language"].items()):
        print(f"  {lang}: acc={lang_stats['mean_accuracy']:.1f}, "
              f"flu={lang_stats['mean_fluency']:.1f}, "
              f"ovr={lang_stats['mean_overall']:.1f} (n={lang_stats['count']})")

    return 0


def main():
    parser = argparse.ArgumentParser(description="Evaluate MADLAD translations with LLM ensemble")
    parser.add_argument(
        "--translations",
        required=True,
        help="Path to translations JSON file",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path (default: data/metrics/llm_ensemble_scores_YYYY-MM-DD.json)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Randomly sample N translations for evaluation (default: all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of concurrent evaluations (default: 10)",
    )
    parser.add_argument(
        "--delay-ms",
        type=int,
        default=100,
        help="Delay between batches in milliseconds (default: 100)",
    )
    parser.add_argument(
        "--gpt-model",
        default="gpt-4o",
        help="OpenAI model to use (default: gpt-4o)",
    )
    parser.add_argument(
        "--opus-model",
        default="claude-opus-4-5-20251101",
        help="Anthropic model to use (default: claude-opus-4-5-20251101)",
    )
    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.translations):
        print(f"ERROR: Translations file not found: {args.translations}")
        return 1

    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
