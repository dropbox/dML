#!/usr/bin/env python3
"""
LLM-as-Judge for TTS Quality Evaluation (FIXED v2)

DESIGN PRINCIPLE: Don't help the AI cheat.
- Don't tell Whisper the language - it must detect it
- Don't tell GPT-4o the expected text - it must transcribe blind
- Compare transcription to expected text programmatically

This prevents hallucination and ensures honest evaluation.

Usage:
    python scripts/llm_judge_tts.py /path/to/audio.wav "expected text" [expected_language]
"""

import os
import sys
import base64
from pathlib import Path
from difflib import SequenceMatcher


def load_api_key():
    """Load OpenAI API key from environment or .env file."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        env_file = Path(__file__).parent.parent / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("OPENAI_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
                    break
    return api_key


def whisper_transcribe_blind(client, audio_path: str) -> dict:
    """
    Use Whisper to transcribe audio WITHOUT telling it the language.

    This is intentional - we want Whisper to:
    1. Detect the language itself (proves the audio has recognizable speech)
    2. Transcribe what it actually hears (no bias from expected text)

    If Whisper can't detect the language or transcribes gibberish,
    that's a real signal about audio quality.
    """
    with open(audio_path, "rb") as f:
        # Don't specify language - let Whisper detect it
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json"  # Get detected language
        )

    return {
        "text": result.text,
        "language": getattr(result, "language", "unknown"),
    }


def text_similarity(a: str, b: str) -> float:
    """Calculate text similarity (0-1)."""
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def evaluate_tts(audio_path: str, expected_text: str, expected_language: str = None) -> dict:
    """
    Evaluate TTS audio quality with BLIND evaluation.

    Key principle: Don't help the AI cheat.
    - Whisper transcribes without knowing expected language
    - We compare transcription to expected programmatically
    - This catches gibberish that sounds plausible but is wrong
    """
    from openai import OpenAI

    api_key = load_api_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key)

    # Step 1: BLIND Whisper transcription (no language hint)
    print("Transcribing with Whisper (BLIND - no language hint)...")
    print("(This forces Whisper to prove it understands the audio)")
    whisper_result = whisper_transcribe_blind(client, audio_path)
    transcription = whisper_result["text"]
    detected_language = whisper_result["language"]

    # Step 2: Calculate text accuracy
    similarity = text_similarity(expected_text, transcription)
    text_accuracy = int(similarity * 10)  # 0-10 scale

    # Step 3: Language check (if expected language provided)
    language_match = True
    if expected_language:
        # Normalize language names
        lang_map = {
            "turkish": "tr", "tr": "tr",
            "arabic": "ar", "ar": "ar",
            "persian": "fa", "farsi": "fa", "fa": "fa",
            "english": "en", "en": "en",
            "japanese": "ja", "ja": "ja",
            "chinese": "zh", "zh": "zh",
        }
        expected_code = lang_map.get(expected_language.lower(), expected_language.lower()[:2])
        detected_code = detected_language.lower()[:2] if detected_language else "??"
        language_match = (expected_code == detected_code)

    # Step 4: Determine match status
    if similarity >= 0.85:
        match = "YES"
    elif similarity >= 0.5:
        match = "PARTIAL"
    else:
        match = "NO"

    # Step 5: Build result
    if match == "NO":
        # Complete failure - transcription doesn't match expected
        result = {
            "transcription": transcription,
            "expected": expected_text,
            "detected_language": detected_language,
            "expected_language": expected_language,
            "language_match": language_match,
            "match": match,
            "similarity": similarity,
            "text_accuracy": text_accuracy,
            "pronunciation": 1,
            "naturalness": 1,
            "clarity": 1,
            "overall": 1,
            "verdict": "FAIL",
            "notes": f"MISMATCH: Whisper heard '{transcription}' (detected: {detected_language}) but expected '{expected_text}'"
        }
    else:
        # Text matches - now rate quality
        with open(audio_path, "rb") as f:
            audio_data = base64.standard_b64encode(f.read()).decode("utf-8")

        # BLIND quality rating - don't tell it the expected text
        quality_prompt = """Listen to this audio and rate the TTS quality.

DO NOT tell me what you think the text says - I already have a Whisper transcription.
Just rate the AUDIO QUALITY:

Rate 1-10 WITH EXPLANATIONS:
- PRONUNCIATION: How clear are the sounds? (1=mumbled, 10=crystal clear)
- NATURALNESS: Does it sound human? (1=robotic, 10=indistinguishable from human)
- CLARITY: Audio quality? (1=distorted, 10=studio quality)

For each rating, explain WHY you gave that score. Be specific about what you heard.

Format your response as:
PRONUNCIATION: [score]/10 - [explanation]
NATURALNESS: [score]/10 - [explanation]
CLARITY: [score]/10 - [explanation]

Example:
PRONUNCIATION: 8/10 - Clear consonants, slight vowel distortion on 'e' sounds
NATURALNESS: 7/10 - Good prosody but robotic pauses between phrases
CLARITY: 9/10 - Clean audio, no artifacts or noise"""

        response = client.chat.completions.create(
            model="gpt-4o-audio-preview",
            modalities=["text"],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": quality_prompt},
                        {"type": "input_audio", "input_audio": {"data": audio_data, "format": "wav"}}
                    ]
                }
            ]
        )

        # Parse scores and explanations
        scores_text = response.choices[0].message.content.strip()
        quality_explanation = scores_text  # Store full explanation
        try:
            # Handle various formats: "8,7,9" or "8/10, 7/10, 9/10" etc
            import re
            numbers = re.findall(r'\d+', scores_text)[:3]
            pronunciation, naturalness, clarity = [int(n) for n in numbers]
        except:
            pronunciation, naturalness, clarity = 5, 5, 5  # Default

        # Clamp to 1-10
        pronunciation = max(1, min(10, pronunciation))
        naturalness = max(1, min(10, naturalness))
        clarity = max(1, min(10, clarity))

        # Overall = weighted average (text accuracy most important)
        overall = int((text_accuracy * 0.4 + pronunciation * 0.2 + naturalness * 0.2 + clarity * 0.2))

        # Language mismatch is a warning but not automatic fail
        notes = ""
        if not language_match:
            notes = f"WARNING: Expected {expected_language} but Whisper detected {detected_language}"

        result = {
            "transcription": transcription,
            "expected": expected_text,
            "detected_language": detected_language,
            "expected_language": expected_language,
            "language_match": language_match,
            "match": match,
            "similarity": similarity,
            "text_accuracy": text_accuracy,
            "pronunciation": pronunciation,
            "naturalness": naturalness,
            "clarity": clarity,
            "overall": overall,
            "verdict": "PASS" if overall >= 7 else "FAIL",
            "notes": notes,
            "quality_explanation": quality_explanation
        }

    return result


def main():
    if len(sys.argv) < 3:
        print("Usage: python llm_judge_tts.py <audio_file> <expected_text> [expected_language]")
        print("Example: python llm_judge_tts.py /tmp/test.wav 'Merhaba' Turkish")
        print()
        print("NOTE: Language is optional - if provided, we verify Whisper detects it correctly")
        sys.exit(1)

    audio_path = sys.argv[1]
    expected_text = sys.argv[2]
    expected_language = sys.argv[3] if len(sys.argv) > 3 else None

    if not os.path.exists(audio_path):
        print(f"Error: File not found: {audio_path}")
        sys.exit(1)

    print(f"=== LLM-as-Judge TTS Evaluation (BLIND) ===")
    print(f"Audio: {audio_path}")
    print(f"Expected text: {expected_text}")
    if expected_language:
        print(f"Expected language: {expected_language}")
    print()
    print("Design: Don't help the AI cheat - Whisper must detect language and transcribe blind")
    print()

    result = evaluate_tts(audio_path, expected_text, expected_language)

    print(f"DETECTED_LANGUAGE: {result['detected_language']}")
    if result['expected_language']:
        print(f"EXPECTED_LANGUAGE: {result['expected_language']} (match: {result['language_match']})")
    print(f"TRANSCRIPTION: {result['transcription']}")
    print(f"EXPECTED: {result['expected']}")
    print(f"MATCH: {result['match']} (similarity: {result['similarity']:.2f})")
    print(f"TEXT_ACCURACY: {result['text_accuracy']}/10")
    print(f"PRONUNCIATION: {result['pronunciation']}/10")
    print(f"NATURALNESS: {result['naturalness']}/10")
    print(f"CLARITY: {result['clarity']}/10")
    print(f"OVERALL: {result['overall']}/10")
    print(f"VERDICT: {result['verdict']}")
    if result['notes']:
        print(f"NOTES: {result['notes']}")
    if result.get('quality_explanation'):
        print()
        print("=== GPT-4o Quality Explanation ===")
        print(result['quality_explanation'])
    print()

    return result


if __name__ == "__main__":
    main()
