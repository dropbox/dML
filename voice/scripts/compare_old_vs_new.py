#!/usr/bin/env python3
"""Compare old (good?) vs new (bad?) popo samples."""

import os
import base64
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import openai

OUTPUT_DIR = Path(__file__).parent.parent / "models" / "cosyvoice" / "test_output"

# Old files from Dec 8 vs new files from Dec 9
COMPARE_FILES = [
    ("cosyvoice_sichuan_grandma.wav", "OLD - Dec 8 Sichuan Grandma"),
    ("cosyvoice_sichuan_motherlnlaw.wav", "OLD - Dec 8 Mother-in-law"),
    ("popo_sichuan_happy.wav", "NEW - Dec 9 Sichuan Happy"),
    ("popo_sichuan_angry.wav", "NEW - Dec 9 Sichuan Angry"),
]

QUICK_EVAL_PROMPT = """Listen to this Chinese audio. Rate overall quality 1-10.

Does it sound like:
- A dying frog / croaking?
- Robotic / glitchy?
- Natural grandmother speech?

Reply EXACTLY:
SCORE: X/10
DYING_FROG: YES or NO
NATURAL: YES or NO
BRIEF: [one line]
"""


def evaluate_one(audio_path: Path) -> str:
    client = openai.OpenAI()

    with open(audio_path, "rb") as f:
        audio_data = base64.standard_b64encode(f.read()).decode("utf-8")

    response = client.chat.completions.create(
        model="gpt-audio-2025-08-28",
        modalities=["text"],
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": QUICK_EVAL_PROMPT},
                {"type": "input_audio", "input_audio": {"data": audio_data, "format": "wav"}}
            ]
        }]
    )
    return response.choices[0].message.content


def main():
    print("=" * 60)
    print("OLD vs NEW Popo Samples Comparison")
    print("=" * 60)

    for filename, description in COMPARE_FILES:
        audio_path = OUTPUT_DIR / filename
        if not audio_path.exists():
            print(f"\n{description}: FILE NOT FOUND")
            continue

        # Get file info
        stat = audio_path.stat()
        size_kb = stat.st_size / 1024

        print(f"\n{'='*60}")
        print(f"{description}")
        print(f"File: {filename} ({size_kb:.1f} KB)")
        print("=" * 60)

        try:
            result = evaluate_one(audio_path)
            print(result)
        except Exception as e:
            print(f"ERROR: {e}")


if __name__ == "__main__":
    main()
