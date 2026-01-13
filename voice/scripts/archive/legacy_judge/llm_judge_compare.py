#!/usr/bin/env python3
"""LLM-as-Judge comparison between torch versions."""

import os
import sys
import base64
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parent.parent / ".env")
client = OpenAI()

OUTPUT_DIR = Path(__file__).parent.parent / "models" / "cosyvoice" / "test_output"

FILES = [
    ("torch 2.9.1 (bad)", OUTPUT_DIR / "compare_torch_2_9_1.wav"),
    ("torch 2.5.1 (fresh)", OUTPUT_DIR / "compare_torch_2_5_1.wav"),
    ("Dec 8 golden ref", OUTPUT_DIR / "cosyvoice_sichuan_grandma.wav"),
]


def llm_judge(audio_path: Path) -> dict:
    """Use GPT-4o-audio-preview to evaluate audio quality."""
    if not audio_path.exists():
        return {"error": f"File not found: {audio_path}"}

    with open(audio_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode()

    response = client.chat.completions.create(
        model="gpt-4o-audio-preview",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {"data": audio_b64, "format": "wav"}
                },
                {
                    "type": "text",
                    "text": """Rate this TTS audio on a scale of 1-10 for quality.
Focus on:
1. Naturalness (does it sound human-like?)
2. Clarity (is the speech clear and understandable?)
3. Artifacts (any robotic sounds, distortion, "dying frog" croaking?)

Return ONLY a JSON object:
{"score": <1-10>, "frog": <true/false>, "issues": "<brief description>"}"""
                }
            ]
        }],
        max_tokens=150
    )

    content = response.choices[0].message.content.strip()
    if "```" in content:
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()

    try:
        return json.loads(content)
    except:
        import re
        score_match = re.search(r'"score":\s*(\d+)', content)
        frog_match = re.search(r'"frog":\s*(true|false)', content, re.IGNORECASE)
        if score_match:
            return {
                "score": int(score_match.group(1)),
                "frog": frog_match.group(1).lower() == "true" if frog_match else False,
                "issues": content[:80]
            }
        return {"score": 0, "frog": True, "issues": f"Parse error: {content[:50]}"}


def main():
    print("=" * 70)
    print("LLM-as-Judge Torch Version Comparison")
    print("=" * 70)

    results = []
    for name, path in FILES:
        print(f"\nEvaluating: {name}")
        print(f"File: {path.name}")

        result = llm_judge(path)
        if "error" in result:
            print(f"  ERROR: {result['error']}")
        else:
            print(f"  Score: {result['score']}/10")
            print(f"  Frog: {result['frog']}")
            print(f"  Issues: {result.get('issues', 'None')[:50]}")

        results.append({"name": name, "file": path.name, **result})

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Version':<25} {'Score':<8} {'Frog':<8} Issues")
    print("-" * 70)
    for r in results:
        if "error" in r:
            print(f"{r['name']:<25} ERROR: {r['error'][:40]}")
        else:
            issues = r.get('issues', '')[:30] + "..." if r.get('issues') else ""
            print(f"{r['name']:<25} {r['score']:<8} {str(r['frog']):<8} {issues}")

    # Save results
    results_path = OUTPUT_DIR / "torch_comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
