#!/usr/bin/env python3
"""Evaluate each CosyVoice2 audio file individually with GPT-5."""

import os
import sys
import base64
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import openai

OUTPUT_DIR = Path(__file__).parent.parent / "models" / "cosyvoice" / "test_output"

AUDIO_FILES = [
    ("popo_sichuan_happy.wav", "Sichuanese Popo - Very Happy"),
    ("popo_mandarin_happy.wav", "Formal Mandarin - Very Happy"),
    ("popo_sichuan_angry.wav", "Sichuanese Popo - Very Angry"),
    ("popo_mandarin_angry.wav", "Formal Mandarin - Very Angry"),
    ("popo_sichuan_singing.wav", "Sichuanese Popo - Singing"),
    ("popo_mandarin_singing.wav", "Formal Mandarin - Singing"),
]

QUICK_EVAL_PROMPT = """Listen to this audio. Rate it 1-10 on overall quality.

Does it have any of these problems?
- Dying frog / croaking sounds
- Robotic glitches
- Distortion
- Unnatural speech

Reply in this EXACT format:
SCORE: X/10
DYING_FROG: YES or NO
GLITCHY: YES or NO
BRIEF: [one sentence description]
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
    print("Quick Quality Check - All 6 Files")
    print("=" * 60)

    for filename, description in AUDIO_FILES:
        audio_path = OUTPUT_DIR / filename
        if not audio_path.exists():
            print(f"\n{description}: FILE NOT FOUND")
            continue

        print(f"\n--- {description} ---")
        print(f"File: {filename}")

        try:
            result = evaluate_one(audio_path)
            print(result)
        except Exception as e:
            print(f"ERROR: {e}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
