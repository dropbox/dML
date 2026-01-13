#!/usr/bin/env python3
"""LLM-as-Judge v2: Match working test pattern."""

import base64
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
import openai

OUTPUT_DIR = Path(__file__).parent.parent / "models" / "cosyvoice" / "test_output"

FILES = [
    ("cosyvoice_sichuan_grandma.wav", "OLD Dec 8"),
    ("cosyvoice_sichuan_motherlnlaw.wav", "NEW regenerated"),
]

PROMPT = """Evaluate this Chinese speech audio for quality issues.

Rate from 1-10 where:
- 10 = Perfect natural human speech
- 5 = Acceptable but some issues
- 1 = Unlistenable distortion

Check for these specific problems:
- "Dying frog" croaking sounds
- Robotic/glitchy artifacts
- Distortion or warbling

Output ONLY valid JSON:
{"score": <1-10>, "frog": <true/false>, "issues": "<description>"}"""

def eval_audio(path: Path) -> str:
    client = openai.OpenAI()
    with open(path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")

    resp = client.chat.completions.create(
        model="gpt-audio-2025-08-28",
        modalities=["text"],
        messages=[
            {
                "role": "system",
                "content": "You are an expert audio quality evaluator that ONLY outputs valid JSON."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "input_audio", "input_audio": {"data": data, "format": "wav"}}
                ]
            }
        ]
    )
    return resp.choices[0].message.content

print("=" * 50)
print("LLM-as-Judge v2: OLD vs NEW")
print("=" * 50)

for fname, desc in FILES:
    path = OUTPUT_DIR / fname
    if not path.exists():
        print(f"\n{desc}: NOT FOUND")
        continue

    import os
    from datetime import datetime
    mtime = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%m-%d %H:%M")

    print(f"\n{desc} ({mtime}):")
    try:
        result = eval_audio(path)
        print(result)
    except Exception as e:
        print(f"ERROR: {e}")
