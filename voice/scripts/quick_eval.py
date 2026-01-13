#!/usr/bin/env python3
"""Quick evaluation of audio files."""

import os
import base64
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

import openai

OUTPUT_DIR = Path(__file__).parent.parent / "models" / "cosyvoice" / "test_output"

FILES_TO_EVAL = [
    # Old good ones
    "cosyvoice_sichuan_grandma.wav",
    "cosyvoice_sichuan_motherlnlaw.wav",
    # Simple instruction tests
    "simple_nagging.wav",
    "simple_happy.wav",
    "simple_angry.wav",
    "simple_singing.wav",
    # Tag-based tests
    "tags_laughter.wav",
    "tags_strong.wav",
    "tags_sigh.wav",
]


def eval_audio(audio_path: Path) -> str:
    client = openai.OpenAI()
    with open(audio_path, "rb") as f:
        audio_data = base64.standard_b64encode(f.read()).decode("utf-8")

    # More explicit prompt
    response = client.chat.completions.create(
        model="gpt-audio-2025-08-28",
        modalities=["text"],
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {"data": audio_data, "format": "wav"}
                },
                {
                    "type": "text",
                    "text": """I just played you a Chinese speech audio file. Please evaluate:

1. SCORE (1-10): Overall audio quality
2. DYING_FROG: Does it sound like a dying frog or have croaking distortion? (YES/NO)
3. NATURAL: Does it sound like natural human speech? (YES/NO)

Reply in format:
SCORE: X/10
DYING_FROG: YES or NO
NATURAL: YES or NO
"""
                }
            ]
        }]
    )
    return response.choices[0].message.content


def main():
    print("=" * 60)
    print("Audio Quality Evaluation")
    print("=" * 60)

    for filename in FILES_TO_EVAL:
        audio_path = OUTPUT_DIR / filename
        if not audio_path.exists():
            print(f"\n{filename}: NOT FOUND")
            continue

        print(f"\n{filename}:")
        try:
            result = eval_audio(audio_path)
            # Print only first 200 chars to keep output manageable
            print(result[:200] if len(result) > 200 else result)
        except Exception as e:
            print(f"ERROR: {e}")


if __name__ == "__main__":
    main()
