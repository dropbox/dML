#!/usr/bin/env python3
"""
Compare Python Kokoro vs C++ Kokoro audio using LLM-as-Judge.
"""

import os
import json
import base64
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

def load_env():
    env_path = PROJECT_ROOT / '.env'
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ.setdefault(key, value)

load_env()

def compare_audio(client, audio1_path: Path, audio2_path: Path, text: str, language: str) -> dict:
    """Compare two audio files using GPT-4o-audio."""
    with open(audio1_path, 'rb') as f:
        audio1_b64 = base64.b64encode(f.read()).decode('utf-8')
    with open(audio2_path, 'rb') as f:
        audio2_b64 = base64.b64encode(f.read()).decode('utf-8')

    prompt = f"""You are an expert TTS audio evaluator. Compare these two TTS audio samples.

Expected text: "{text}"
Language: {language}

AUDIO 1 is the first sample.
AUDIO 2 is the second sample.

Evaluate each for:
1. Pronunciation accuracy (1-5)
2. Naturalness/prosody (1-5)
3. Overall quality (1-5)

For the specific text, check:
- Japanese: Does it sound like "konichiwa-a" (extra syllable) or clean "konnichiwa"?
- English statements: Does it end with falling intonation (correct) or rising (incorrect)?
- English questions: Does it end with rising intonation (correct) or falling (incorrect)?
- Chinese: Are the tones correct? 你好 should be rising then dipping.

Respond in JSON:
{{
    "audio1": {{
        "accuracy": 1-5,
        "naturalness": 1-5,
        "quality": 1-5,
        "issues": "describe any issues"
    }},
    "audio2": {{
        "accuracy": 1-5,
        "naturalness": 1-5,
        "quality": 1-5,
        "issues": "describe any issues"
    }},
    "winner": "audio1|audio2|tie",
    "reasoning": "why one is better"
}}
"""

    response = client.chat.completions.create(
        model="gpt-audio-2025-08-28",  # GPT-5 based audio model
        modalities=["text"],
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "input_audio", "input_audio": {"data": audio1_b64, "format": "wav"}},
                {"type": "input_audio", "input_audio": {"data": audio2_b64, "format": "wav"}}
            ]
        }],
        max_tokens=1500
    )

    result_text = response.choices[0].message.content
    json_start = result_text.find('{')
    json_end = result_text.rfind('}') + 1
    if json_start >= 0 and json_end > json_start:
        return json.loads(result_text[json_start:json_end])
    return {"error": "No JSON found", "raw": result_text}


def main():
    from openai import OpenAI
    client = OpenAI()

    comparison_dir = PROJECT_ROOT / "reports" / "prosody_comparison"

    test_cases = [
        ("ja", "こんにちは", "japanese_konnichiwa.wav"),
        ("en", "The quick brown fox jumps over the lazy dog.", "english_statement.wav"),
        ("en", "Are you coming to the party tonight?", "english_question.wav"),
        ("zh", "你好", "chinese_nihao.wav"),
        ("zh", "今天天气很好", "chinese_sentence.wav"),
    ]

    results = []
    for lang, text, filename in test_cases:
        python_path = comparison_dir / f"python_{filename}"
        cpp_path = comparison_dir / f"cpp_{filename}"

        if not python_path.exists() or not cpp_path.exists():
            print(f"SKIP: Missing files for {filename}")
            continue

        print(f"\n{'='*60}")
        print(f"Comparing: {text} ({lang})")
        print(f"{'='*60}")

        result = compare_audio(client, python_path, cpp_path, text, lang)
        result["language"] = lang
        result["text"] = text
        results.append(result)

        print(f"Python (Audio1): accuracy={result.get('audio1', {}).get('accuracy')}, "
              f"naturalness={result.get('audio1', {}).get('naturalness')}, "
              f"quality={result.get('audio1', {}).get('quality')}")
        print(f"  Issues: {result.get('audio1', {}).get('issues')}")

        print(f"C++ (Audio2): accuracy={result.get('audio2', {}).get('accuracy')}, "
              f"naturalness={result.get('audio2', {}).get('naturalness')}, "
              f"quality={result.get('audio2', {}).get('quality')}")
        print(f"  Issues: {result.get('audio2', {}).get('issues')}")

        print(f"Winner: {result.get('winner')}")
        print(f"Reasoning: {result.get('reasoning')}")

    # Summary
    print("\n\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    python_wins = sum(1 for r in results if r.get('winner') == 'audio1')
    cpp_wins = sum(1 for r in results if r.get('winner') == 'audio2')
    ties = sum(1 for r in results if r.get('winner') == 'tie')

    print(f"Python wins: {python_wins}")
    print(f"C++ wins: {cpp_wins}")
    print(f"Ties: {ties}")

    # Save results
    output_path = comparison_dir / "comparison_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
