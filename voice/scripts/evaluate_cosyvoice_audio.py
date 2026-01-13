#!/usr/bin/env python3
"""Evaluate CosyVoice2 audio quality using GPT-4o as judge."""

import os
import sys
import base64
from pathlib import Path

# Load API key from .env
from dotenv import load_dotenv
load_dotenv()

OUTPUT_DIR = Path(__file__).parent.parent / "models" / "cosyvoice" / "test_output"

AUDIO_FILES = [
    ("popo_sichuan_happy.wav", "Sichuanese Popo - Very Happy"),
    ("popo_mandarin_happy.wav", "Formal Mandarin - Very Happy"),
    ("popo_sichuan_angry.wav", "Sichuanese Popo - Very Angry"),
    ("popo_mandarin_angry.wav", "Formal Mandarin - Very Angry"),
    ("popo_sichuan_singing.wav", "Sichuanese Popo - Singing"),
    ("popo_mandarin_singing.wav", "Formal Mandarin - Singing"),
]

EVALUATION_PROMPT = """You are an expert audio quality evaluator. Listen to this audio sample and evaluate it.

This audio is supposed to be: {description}

The text content should be a grandmother nagging story in Chinese (either Sichuanese dialect or formal Mandarin).

Please evaluate on a scale of 1-10 for each category:

1. **Audio Quality** (1-10): Is the audio clear? Any distortion, artifacts, robotic sounds, glitches?
2. **Naturalness** (1-10): Does it sound like natural human speech or robotic/synthesized?
3. **Emotion Match** (1-10): Does the emotion match what was requested? (happy/angry/singing)
4. **Dialect Accuracy** (1-10): For Sichuanese - does it sound like Sichuan dialect? For Mandarin - is it standard Putonghua?
5. **Overall Quality** (1-10): Overall impression

Also note any specific problems:
- Dying frog sounds?
- Robotic/glitchy artifacts?
- Wrong language/dialect?
- Monotone when should be emotional?
- Speaking when should be singing?

Give your response in this format:
AUDIO_QUALITY: X/10
NATURALNESS: X/10
EMOTION_MATCH: X/10
DIALECT_ACCURACY: X/10
OVERALL: X/10
PROBLEMS: [list any specific problems]
NOTES: [any additional observations]
"""


def evaluate_audio(audio_path: Path, description: str) -> dict:
    """Evaluate a single audio file using GPT-4o."""
    import openai

    client = openai.OpenAI()

    # Read and encode audio
    with open(audio_path, "rb") as f:
        audio_data = base64.standard_b64encode(f.read()).decode("utf-8")

    response = client.chat.completions.create(
        model="gpt-audio-2025-08-28",  # "GPT-5" audio model - better than gpt-4o-audio-preview
        modalities=["text"],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": EVALUATION_PROMPT.format(description=description)
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_data,
                            "format": "wav"
                        }
                    }
                ]
            }
        ]
    )

    return response.choices[0].message.content


def main():
    print("=" * 70)
    print("CosyVoice2 Audio Quality Evaluation (GPT-4o Judge)")
    print("=" * 70)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found in environment or .env file")
        return 1

    results = []

    for filename, description in AUDIO_FILES:
        audio_path = OUTPUT_DIR / filename
        if not audio_path.exists():
            print(f"WARNING: {filename} not found, skipping")
            continue

        print(f"\n{'='*70}")
        print(f"Evaluating: {description}")
        print(f"File: {filename}")
        print("=" * 70)

        try:
            evaluation = evaluate_audio(audio_path, description)
            print(evaluation)
            results.append({
                "file": filename,
                "description": description,
                "evaluation": evaluation
            })
        except Exception as e:
            print(f"ERROR evaluating {filename}: {e}")
            results.append({
                "file": filename,
                "description": description,
                "evaluation": f"ERROR: {e}"
            })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for r in results:
        print(f"\n{r['description']}:")
        # Extract scores if possible
        eval_text = r['evaluation']
        if "ERROR" not in eval_text:
            for line in eval_text.split('\n'):
                if any(x in line for x in ['AUDIO_QUALITY:', 'NATURALNESS:', 'EMOTION_MATCH:', 'DIALECT_ACCURACY:', 'OVERALL:', 'PROBLEMS:']):
                    print(f"  {line}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
