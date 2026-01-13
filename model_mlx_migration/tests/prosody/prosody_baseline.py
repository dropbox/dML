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
Kokoro Prosody Baseline Testing Framework

Phase 0 of the Prosody Annotation Roadmap:
Measure what Kokoro already responds to before building annotation support.

Test Categories:
1. Punctuation Tests (!, ?, ., ...)
2. Context Tests ("Alert!" vs "the alert was triggered")
3. Capitalization Tests ("IMPORTANT" vs "important")
4. Stress Pattern Tests (noun/verb stress)
5. Sentence Position Tests (fronted emphasis)
6. Question Type Tests (yes/no vs wh-questions)
7. IPA Marker Tests (primary/secondary stress)

Each test generates audio for multiple variants and analyzes:
- F0 (pitch) mean, std, contour shape
- Duration of the utterance
- Significant differences between variants

Usage:
    # Run all tests
    python -m tests.prosody.prosody_baseline

    # Run specific category
    python -m tests.prosody.prosody_baseline --category punctuation

    # Save audio files for manual review
    python -m tests.prosody.prosody_baseline --save-audio

    # Generate HTML report
    python -m tests.prosody.prosody_baseline --html-report
"""

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# F0 and Duration Extraction
# =============================================================================

def extract_f0(audio: np.ndarray, sr: int = 24000) -> dict:
    """Extract F0 (pitch) features from audio using librosa's pyin.

    Returns:
        dict with keys:
        - f0: raw F0 array (Hz, with NaN for unvoiced)
        - f0_mean: mean F0 (Hz, voiced frames only)
        - f0_std: std F0 (Hz, voiced frames only)
        - f0_min: minimum F0 (Hz)
        - f0_max: maximum F0 (Hz)
        - f0_range: f0_max - f0_min (Hz)
        - voiced_ratio: fraction of voiced frames
        - f0_contour: smoothed F0 contour for visualization
    """
    import librosa

    # Use pyin for robust F0 estimation
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio,
        fmin=50,  # Hz
        fmax=500,  # Hz
        sr=sr,
        frame_length=1024,
        hop_length=256,
    )

    # Get voiced frames only
    voiced_f0 = f0[~np.isnan(f0)]

    if len(voiced_f0) == 0:
        return {
            "f0": f0,
            "f0_mean": 0.0,
            "f0_std": 0.0,
            "f0_min": 0.0,
            "f0_max": 0.0,
            "f0_range": 0.0,
            "voiced_ratio": 0.0,
            "f0_contour": f0,
        }

    return {
        "f0": f0,
        "f0_mean": float(np.mean(voiced_f0)),
        "f0_std": float(np.std(voiced_f0)),
        "f0_min": float(np.min(voiced_f0)),
        "f0_max": float(np.max(voiced_f0)),
        "f0_range": float(np.max(voiced_f0) - np.min(voiced_f0)),
        "voiced_ratio": float(len(voiced_f0) / len(f0)),
        "f0_contour": f0,
    }


def extract_duration(audio: np.ndarray, sr: int = 24000) -> dict:
    """Extract duration features from audio.

    Returns:
        dict with keys:
        - duration_s: total duration in seconds
        - duration_samples: total samples
        - rms_energy: root mean square energy
    """
    duration_samples = len(audio)
    duration_s = duration_samples / sr
    rms_energy = float(np.sqrt(np.mean(audio ** 2)))

    return {
        "duration_s": duration_s,
        "duration_samples": duration_samples,
        "rms_energy": rms_energy,
    }


def analyze_f0_difference(f0_a: dict, f0_b: dict) -> dict:
    """Analyze the difference between two F0 extractions.

    Returns:
        dict with difference metrics
    """
    return {
        "mean_diff_hz": f0_a["f0_mean"] - f0_b["f0_mean"],
        "mean_diff_percent": ((f0_a["f0_mean"] - f0_b["f0_mean"]) / f0_b["f0_mean"] * 100)
            if f0_b["f0_mean"] > 0 else 0.0,
        "std_diff_hz": f0_a["f0_std"] - f0_b["f0_std"],
        "range_diff_hz": f0_a["f0_range"] - f0_b["f0_range"],
    }


def are_significantly_different(metrics_a: dict, metrics_b: dict,
                                 f0_threshold_pct: float = 5.0,
                                 duration_threshold_pct: float = 10.0) -> dict:
    """Determine if two audio samples are significantly different.

    Args:
        metrics_a: F0 and duration metrics for sample A
        metrics_b: F0 and duration metrics for sample B
        f0_threshold_pct: Percent difference in F0 mean to consider significant
        duration_threshold_pct: Percent difference in duration to consider significant

    Returns:
        dict with significance analysis
    """
    f0_diff_pct = abs(metrics_a["f0"]["f0_mean"] - metrics_b["f0"]["f0_mean"]) / \
                  max(metrics_a["f0"]["f0_mean"], metrics_b["f0"]["f0_mean"], 1) * 100

    dur_diff_pct = abs(metrics_a["duration"]["duration_s"] - metrics_b["duration"]["duration_s"]) / \
                   max(metrics_a["duration"]["duration_s"], metrics_b["duration"]["duration_s"], 0.001) * 100

    f0_std_diff = abs(metrics_a["f0"]["f0_std"] - metrics_b["f0"]["f0_std"])

    return {
        "f0_different": f0_diff_pct > f0_threshold_pct,
        "duration_different": dur_diff_pct > duration_threshold_pct,
        "f0_diff_percent": f0_diff_pct,
        "duration_diff_percent": dur_diff_pct,
        "f0_std_diff_hz": f0_std_diff,
        "overall_different": f0_diff_pct > f0_threshold_pct or dur_diff_pct > duration_threshold_pct,
    }


# =============================================================================
# Test Case Definitions
# =============================================================================

@dataclass
class ProsodyTestCase:
    """A single prosody test case."""
    id: int
    category: str
    description: str
    variants: list[str]  # Different text variants to compare
    expected_difference: str  # What we expect to be different
    labels: list[str] = field(default_factory=list)  # Labels for each variant


# Test cases from PROSODY_DESIGN.md
PROSODY_TEST_CASES = [
    # === A. Punctuation Tests ===
    ProsodyTestCase(
        id=1,
        category="punctuation",
        description="Terminal punctuation: period vs exclamation vs question",
        variants=["Hello.", "Hello!", "Hello?"],
        expected_difference="Different intonation patterns",
        labels=["period", "exclamation", "question"],
    ),
    ProsodyTestCase(
        id=2,
        category="punctuation",
        description="Command vs exclaim vs question",
        variants=["Stop.", "Stop!", "Stop?"],
        expected_difference="Different emphasis and intonation",
        labels=["command", "exclaim", "question"],
    ),
    ProsodyTestCase(
        id=3,
        category="punctuation",
        description="Single word punctuation",
        variants=["Really.", "Really!", "Really?"],
        expected_difference="Different prosody on single word",
        labels=["statement", "exclamation", "question"],
    ),
    ProsodyTestCase(
        id=4,
        category="punctuation",
        description="Ellipsis vs period vs exclamation",
        variants=["I see...", "I see.", "I see!"],
        expected_difference="Ellipsis should have trailing/thoughtful tone",
        labels=["ellipsis", "period", "exclamation"],
    ),
    ProsodyTestCase(
        id=5,
        category="punctuation",
        description="Comma pause",
        variants=["Wait, what?", "Wait what?"],
        expected_difference="Comma should introduce pause",
        labels=["with_comma", "without_comma"],
    ),
    ProsodyTestCase(
        id=6,
        category="punctuation",
        description="Em-dash vs comma vs no punctuation",
        variants=["No—never", "No, never", "No never"],
        expected_difference="Different pause patterns",
        labels=["em_dash", "comma", "no_punct"],
    ),

    # === B. Context Tests ===
    ProsodyTestCase(
        id=7,
        category="context",
        description="Word function: exclamation vs noun",
        variants=["Alert!", "The alert was triggered"],
        expected_difference="'Alert!' should be more emphatic",
        labels=["exclaim", "noun"],
    ),
    ProsodyTestCase(
        id=8,
        category="context",
        description="Urgency in context",
        variants=["Help!", "I need help with this"],
        expected_difference="'Help!' should be urgent",
        labels=["urgent", "casual"],
    ),
    ProsodyTestCase(
        id=9,
        category="context",
        description="Command vs verb",
        variants=["Fire!", "Fire the employee"],
        expected_difference="Different stress and urgency",
        labels=["command", "verb"],
    ),
    ProsodyTestCase(
        id=10,
        category="context",
        description="Command vs noun",
        variants=["Run!", "Go for a run"],
        expected_difference="Different stress patterns",
        labels=["command", "noun"],
    ),
    ProsodyTestCase(
        id=11,
        category="context",
        description="Standalone vs greeting",
        variants=["Hey!", "Hey, how are you?"],
        expected_difference="Standalone more emphatic",
        labels=["standalone", "greeting"],
    ),

    # === C. Capitalization Tests ===
    ProsodyTestCase(
        id=12,
        category="capitalization",
        description="All-caps emphasis",
        variants=["This is IMPORTANT", "This is important"],
        expected_difference="All-caps should be emphasized",
        labels=["caps", "lowercase"],
    ),
    ProsodyTestCase(
        id=13,
        category="capitalization",
        description="Caps with punctuation",
        variants=["NO!", "No."],
        expected_difference="Different emphasis level",
        labels=["caps_exclaim", "lower_period"],
    ),
    ProsodyTestCase(
        id=14,
        category="capitalization",
        description="Caps question",
        variants=["WHY?", "Why?"],
        expected_difference="Caps should add intensity",
        labels=["caps", "lowercase"],
    ),

    # === D. Stress Pattern Tests (noun/verb) ===
    ProsodyTestCase(
        id=15,
        category="stress",
        description="Record: noun vs verb stress",
        variants=["the record", "to record"],
        expected_difference="REcord (noun) vs reCORD (verb)",
        labels=["noun", "verb"],
    ),
    ProsodyTestCase(
        id=16,
        category="stress",
        description="Present: noun vs verb stress",
        variants=["a present", "to present"],
        expected_difference="PREsent (noun) vs preSENT (verb)",
        labels=["noun", "verb"],
    ),
    ProsodyTestCase(
        id=17,
        category="stress",
        description="Contract: noun vs verb stress",
        variants=["the contract", "to contract"],
        expected_difference="CONtract (noun) vs conTRACT (verb)",
        labels=["noun", "verb"],
    ),

    # === E. Sentence Position Tests ===
    ProsodyTestCase(
        id=18,
        category="position",
        description="Fronted emphasis",
        variants=["Important: do this now", "This is important"],
        expected_difference="Fronted word more emphasized",
        labels=["fronted", "embedded"],
    ),
    ProsodyTestCase(
        id=19,
        category="position",
        description="Position effect",
        variants=["Listen. This matters.", "This matters, listen"],
        expected_difference="Different emphasis patterns",
        labels=["front", "end"],
    ),

    # === F. Question Type Tests ===
    ProsodyTestCase(
        id=20,
        category="question",
        description="Yes/no question (rising)",
        variants=["Is this correct?"],
        expected_difference="Rising intonation",
        labels=["yes_no"],
    ),
    ProsodyTestCase(
        id=21,
        category="question",
        description="Wh-question (falling)",
        variants=["What is this?"],
        expected_difference="Falling intonation",
        labels=["wh_question"],
    ),
    ProsodyTestCase(
        id=22,
        category="question",
        description="Confirmation question (rising)",
        variants=["You're sure?"],
        expected_difference="Rising intonation",
        labels=["confirmation"],
    ),
    ProsodyTestCase(
        id=23,
        category="question",
        description="Incredulous question (stronger rising)",
        variants=["This is what you want?"],
        expected_difference="Strong rising intonation",
        labels=["incredulous"],
    ),
    ProsodyTestCase(
        id=24,
        category="question",
        description="Echo question (rising on wh-word)",
        variants=["You did what?"],
        expected_difference="Rising on 'what'",
        labels=["echo"],
    ),

    # === G. IPA Marker Tests ===
    # Note: These test whether Kokoro respects phonetic input
    # If Kokoro uses eSpeak for G2P, these may already work
    ProsodyTestCase(
        id=25,
        category="ipa",
        description="Primary stress marker",
        variants=["record", "Record"],  # Testing case sensitivity
        expected_difference="Case might affect stress",
        labels=["lowercase", "capitalized"],
    ),
    ProsodyTestCase(
        id=26,
        category="ipa",
        description="Vowel length",
        variants=["beat", "bit"],
        expected_difference="Different vowel length",
        labels=["long", "short"],
    ),
    ProsodyTestCase(
        id=27,
        category="ipa",
        description="Break marker (pipe)",
        variants=["hello world", "hello | world"],
        expected_difference="Pipe should insert pause",
        labels=["no_break", "with_break"],
    ),
]


# =============================================================================
# Test Runner
# =============================================================================

@dataclass
class TestResult:
    """Result of a single prosody test."""
    test_case: ProsodyTestCase
    variant_results: list[dict]  # Each variant's metrics
    comparisons: list[dict]  # Pairwise comparisons
    responds: str  # "yes", "partial", "no"
    notes: str = ""


class ProsodyTester:
    """Runs prosody baseline tests on Kokoro TTS."""

    def __init__(self, voice: str = "af_bella", save_audio: bool = False,
                 output_dir: str = "outputs/prosody_tests"):
        self.voice = voice
        self.save_audio = save_audio
        self.output_dir = Path(output_dir)
        self.model = None
        self.sample_rate = 24000  # Kokoro default

    def load_model(self):
        """Load the Kokoro TTS model."""
        if self.model is not None:
            return

        print("Loading Kokoro TTS model...")
        from mlx_audio.tts.utils import load_model
        self.model = load_model("prince-canuma/Kokoro-82M")

        # Warmup
        print("Warming up model...")
        for _ in range(3):
            for _result in self.model.generate(text="Hello.", voice=self.voice):
                pass
        print("Model ready.")

    def synthesize(self, text: str) -> np.ndarray:
        """Synthesize text to audio."""
        audio_chunks = [
            result.audio
            for result in self.model.generate(
                text=text,
                voice=self.voice,
                speed=1.0,
                verbose=False,
            )
        ]

        # Concatenate all chunks
        audio = np.concatenate(audio_chunks, axis=-1)
        return audio.flatten()

    def analyze_audio(self, audio: np.ndarray) -> dict:
        """Extract all metrics from audio."""
        return {
            "f0": extract_f0(audio, sr=self.sample_rate),
            "duration": extract_duration(audio, sr=self.sample_rate),
        }

    def run_test(self, test_case: ProsodyTestCase) -> TestResult:
        """Run a single test case."""
        print(f"  Test {test_case.id}: {test_case.description}")

        variant_results = []
        for i, text in enumerate(test_case.variants):
            label = test_case.labels[i] if i < len(test_case.labels) else f"variant_{i}"

            # Synthesize
            audio = self.synthesize(text)

            # Save audio if requested
            if self.save_audio:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                audio_path = self.output_dir / f"test_{test_case.id:02d}_{label}.wav"
                sf.write(audio_path, audio, self.sample_rate)

            # Analyze
            metrics = self.analyze_audio(audio)
            metrics["text"] = text
            metrics["label"] = label
            variant_results.append(metrics)

        # Compare variants pairwise
        comparisons = []
        for i in range(len(variant_results)):
            for j in range(i + 1, len(variant_results)):
                diff = are_significantly_different(
                    variant_results[i],
                    variant_results[j],
                )
                diff["variant_a"] = variant_results[i]["label"]
                diff["variant_b"] = variant_results[j]["label"]
                comparisons.append(diff)

        # Determine if Kokoro responds to this prosody feature
        any_different = any(c["overall_different"] for c in comparisons)
        all_different = all(c["overall_different"] for c in comparisons) if comparisons else False

        if len(test_case.variants) == 1:
            responds = "n/a"  # Single variant, can't compare
        elif all_different:
            responds = "yes"
        elif any_different:
            responds = "partial"
        else:
            responds = "no"

        return TestResult(
            test_case=test_case,
            variant_results=variant_results,
            comparisons=comparisons,
            responds=responds,
        )

    def run_all_tests(self, categories: list[str] | None = None) -> list[TestResult]:
        """Run all or selected test cases."""
        self.load_model()

        results = []
        test_cases = PROSODY_TEST_CASES

        if categories:
            test_cases = [tc for tc in test_cases if tc.category in categories]

        print(f"\nRunning {len(test_cases)} prosody tests...")

        for test_case in test_cases:
            result = self.run_test(test_case)
            results.append(result)

        return results

    def generate_report(self, results: list[TestResult]) -> str:
        """Generate a markdown report of test results."""
        lines = [
            "# Kokoro Prosody Baseline Test Report",
            "",
            f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Voice**: {self.voice}",
            "**Model**: prince-canuma/Kokoro-82M (MLX)",
            "",
            "---",
            "",
            "## Summary",
            "",
        ]

        # Count by category
        categories = {}
        for r in results:
            cat = r.test_case.category
            if cat not in categories:
                categories[cat] = {"yes": 0, "partial": 0, "no": 0, "n/a": 0}
            categories[cat][r.responds] += 1

        total_yes = sum(c["yes"] for c in categories.values())
        total_partial = sum(c["partial"] for c in categories.values())
        total_no = sum(c["no"] for c in categories.values())
        total = len(results)

        lines.extend([
            "| Category | Responds | Partial | No Response | Total |",
            "|----------|----------|---------|-------------|-------|",
        ])

        for cat, counts in sorted(categories.items()):
            total_cat = counts["yes"] + counts["partial"] + counts["no"] + counts["n/a"]
            lines.append(
                f"| {cat.capitalize()} | {counts['yes']} | {counts['partial']} | {counts['no']} | {total_cat} |",
            )

        lines.extend([
            f"| **Total** | **{total_yes}** | **{total_partial}** | **{total_no}** | **{total}** |",
            "",
            "---",
            "",
            "## Detailed Results",
            "",
        ])

        # Group by category
        for cat in sorted(categories.keys()):
            cat_results = [r for r in results if r.test_case.category == cat]
            lines.append(f"### {cat.capitalize()}")
            lines.append("")

            for r in cat_results:
                tc = r.test_case
                status_emoji = {"yes": "✅", "partial": "⚠️", "no": "❌", "n/a": "ℹ️"}[r.responds]

                lines.append(f"#### Test {tc.id}: {tc.description} {status_emoji}")
                lines.append("")
                lines.append(f"**Variants**: {', '.join(repr(v) for v in tc.variants)}")
                lines.append(f"**Expected**: {tc.expected_difference}")
                lines.append(f"**Response**: {r.responds.upper()}")
                lines.append("")

                # Metrics table
                lines.append("| Variant | Text | Duration (s) | F0 Mean (Hz) | F0 Std (Hz) | F0 Range (Hz) |")
                lines.append("|---------|------|--------------|--------------|-------------|---------------|")

                lines.extend(
                    f"| {vr['label']} | {vr['text']!r} | "
                    f"{vr['duration']['duration_s']:.3f} | "
                    f"{vr['f0']['f0_mean']:.1f} | "
                    f"{vr['f0']['f0_std']:.1f} | "
                    f"{vr['f0']['f0_range']:.1f} |"
                    for vr in r.variant_results
                )

                lines.append("")

                # Comparisons
                if r.comparisons:
                    lines.append("**Comparisons**:")
                    for comp in r.comparisons:
                        sig = "Different" if comp["overall_different"] else "Similar"
                        lines.append(
                            f"- {comp['variant_a']} vs {comp['variant_b']}: {sig} "
                            f"(F0: {comp['f0_diff_percent']:.1f}%, Duration: {comp['duration_diff_percent']:.1f}%)",
                        )
                    lines.append("")

                lines.append("---")
                lines.append("")

        return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Kokoro Prosody Baseline Tests")
    parser.add_argument("--category", "-c", type=str, nargs="+",
                        help="Test categories to run (punctuation, context, capitalization, stress, position, question, ipa)")
    parser.add_argument("--voice", type=str, default="af_bella",
                        help="Kokoro voice to use (default: af_bella)")
    parser.add_argument("--save-audio", action="store_true",
                        help="Save audio files for manual review")
    parser.add_argument("--output-dir", type=str, default="outputs/prosody_tests",
                        help="Directory to save outputs")
    parser.add_argument("--html-report", action="store_true",
                        help="Generate HTML report (requires markdown)")

    args = parser.parse_args()

    tester = ProsodyTester(
        voice=args.voice,
        save_audio=args.save_audio,
        output_dir=args.output_dir,
    )

    results = tester.run_all_tests(categories=args.category)

    # Generate report
    report = tester.generate_report(results)

    # Save markdown report
    report_dir = Path(args.output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"prosody_baseline_{time.strftime('%Y%m%d_%H%M%S')}.md"
    report_path.write_text(report)
    print(f"\nReport saved to: {report_path}")

    # Print summary
    total_yes = sum(1 for r in results if r.responds == "yes")
    total_partial = sum(1 for r in results if r.responds == "partial")
    total_no = sum(1 for r in results if r.responds == "no")

    print("\n=== Summary ===")
    print(f"Responds: {total_yes}/{len(results)}")
    print(f"Partial:  {total_partial}/{len(results)}")
    print(f"No:       {total_no}/{len(results)}")

    # Return results for programmatic use
    return results


if __name__ == "__main__":
    main()
