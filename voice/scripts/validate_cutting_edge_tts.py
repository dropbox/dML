#!/usr/bin/env python3
"""
Cutting-Edge TTS Validation Script

Phase 1 validation using MMS-TTS (Meta) as the baseline.
Tests Arabic, Turkish, and Persian with LLM-as-Judge.

Per MANAGER directive: All languages must pass 7/10 for Phase 2.

Usage:
    python scripts/validate_cutting_edge_tts.py

Copyright 2025 Andrew Yates. All rights reserved.
"""

import base64
import json
import os
import tempfile
import time
import wave
from pathlib import Path

import numpy as np
import torch
import torchaudio
from transformers import VitsModel, AutoTokenizer

# Load environment variables
PROJECT_ROOT = Path(__file__).parent.parent
env_path = PROJECT_ROOT / '.env'
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key] = value

# MMS-TTS model configs
MODELS = {
    "ar": {"model_id": "facebook/mms-tts-ara", "name": "Arabic", "sample_rate": 16000},
    "tr": {"model_id": "facebook/mms-tts-tur", "name": "Turkish", "sample_rate": 16000},
    "fa": {"model_id": "facebook/mms-tts-fas", "name": "Persian", "sample_rate": 16000},
}

# Test texts
TEST_TEXTS = {
    "ar": "مرحبا، كيف حالك اليوم؟ أتمنى لك يوما سعيدا",
    "tr": "Merhaba, bugün hava çok güzel. Umarım iyi bir gün geçirirsiniz.",
    "fa": "سلام، امروز هوا بسیار خوب است. امیدوارم روز خوبی داشته باشید.",
}

# Quality threshold
MIN_SCORE = 7


class MMSTTSSynthesizer:
    """MMS-TTS synthesizer for validation."""

    def __init__(self, language: str):
        config = MODELS[language]
        self.language = language
        self.name = config["name"]
        self.sample_rate = config["sample_rate"]

        print(f"Loading {self.name} MMS-TTS ({config['model_id']})...")
        self.model = VitsModel.from_pretrained(config["model_id"])
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_id"])

        # Move to MPS if available
        if torch.backends.mps.is_available():
            self.model = self.model.to("mps")
            print(f"  Using MPS acceleration")

    def synthesize(self, text: str) -> bytes:
        """Synthesize text to WAV bytes."""
        inputs = self.tokenizer(text, return_tensors="pt")

        if torch.backends.mps.is_available():
            inputs = {k: v.to("mps") for k, v in inputs.items()}

        with torch.no_grad():
            output = self.model(**inputs).waveform

        # Convert to numpy
        audio = output.squeeze().cpu().numpy()

        # Convert to 16-bit PCM
        audio_int16 = (audio * 32767).astype(np.int16)

        # Write to WAV bytes
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name

        try:
            with wave.open(tmp_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_int16.tobytes())

            with open(tmp_path, 'rb') as f:
                return f.read()
        finally:
            os.unlink(tmp_path)


def evaluate_with_llm_judge(wav_data: bytes, text: str, language: str, language_name: str) -> dict:
    """Evaluate audio quality using GPT-4o-audio."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return {"error": "OPENAI_API_KEY not set", "overall": 0}

    try:
        from openai import OpenAI
    except ImportError:
        return {"error": "openai package not installed", "overall": 0}

    audio_base64 = base64.b64encode(wav_data).decode("utf-8")

    prompts = {
        "ar": f"""Listen to this Arabic speech synthesis.
Rate the following aspects 1-10:
1. Arabic pronunciation accuracy
2. Natural intonation and rhythm
3. Overall quality and clarity

A score of 8+ means: "Native Arabic speakers would find this natural."
A score of 6-7 means: "Clearly Arabic but with minor issues."
Below 6 means: "Significant problems, not production ready."

Expected text: "{text}"

Output JSON: {{"pronunciation": X, "intonation": X, "quality": X, "overall": X, "issues": "brief description or none"}}""",

        "tr": f"""Listen to this Turkish speech synthesis.
Rate the following aspects 1-10:
1. Turkish pronunciation accuracy (vowel harmony, consonant clarity)
2. Natural intonation and emotional expression
3. Overall quality and clarity

A score of 8+ means: "Native Turkish speakers would be impressed."
A score of 6-7 means: "Clearly Turkish but with minor issues."
Below 6 means: "Significant problems, not production ready."

Expected text: "{text}"

Output JSON: {{"pronunciation": X, "intonation": X, "quality": X, "overall": X, "issues": "brief description or none"}}""",

        "fa": f"""Listen to this Persian (Farsi) speech synthesis.
Rate the following aspects 1-10:
1. Persian pronunciation accuracy (ezafe construction, consonant clusters)
2. Natural intonation and flow
3. Overall quality and clarity

A score of 8+ means: "Native Persian speakers would find this natural."
A score of 6-7 means: "Clearly Persian but with minor issues."
Below 6 means: "Significant problems, not production ready."

Expected text: "{text}"

Output JSON: {{"pronunciation": X, "intonation": X, "quality": X, "overall": X, "issues": "brief description or none"}}""",
    }

    prompt = prompts.get(language, prompts["ar"])

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-audio-preview",
        modalities=["text"],
        messages=[
            {
                "role": "system",
                "content": f"You are an expert {language_name} TTS audio evaluator. Output ONLY valid JSON."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "input_audio", "input_audio": {"data": audio_base64, "format": "wav"}}
                ]
            }
        ],
        max_tokens=300
    )

    result_text = response.choices[0].message.content.strip()

    # Parse JSON
    try:
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
            result_text = result_text.strip()

        json_start = result_text.find('{')
        json_end = result_text.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            return json.loads(result_text[json_start:json_end])
    except json.JSONDecodeError:
        pass

    return {
        "pronunciation": 0,
        "intonation": 0,
        "quality": 0,
        "overall": 0,
        "issues": f"Failed to parse: {result_text[:100]}"
    }


def main():
    print("=" * 60)
    print("CUTTING-EDGE TTS QUALITY VALIDATION")
    print("Using MMS-TTS (Meta) as baseline")
    print("=" * 60)
    print()

    results = {}
    output_dir = PROJECT_ROOT / "tests" / "output" / "mms_validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    for lang, text in TEST_TEXTS.items():
        config = MODELS[lang]
        print(f"\n--- {config['name'].upper()} ---")
        print(f"Text: {text}")

        # Synthesize
        synth = MMSTTSSynthesizer(lang)
        start = time.time()
        wav_data = synth.synthesize(text)
        latency = time.time() - start

        # Save audio file
        output_path = output_dir / f"{lang}_mms_validation.wav"
        output_path.write_bytes(wav_data)
        print(f"Saved: {output_path}")
        print(f"Latency: {latency:.2f}s, Size: {len(wav_data)} bytes")

        # LLM-Judge evaluation
        print("Running LLM-Judge evaluation...")
        result = evaluate_with_llm_judge(wav_data, text, lang, config["name"])
        results[lang] = result

        print(f"  Pronunciation: {result.get('pronunciation', 0)}/10")
        print(f"  Intonation: {result.get('intonation', 0)}/10")
        print(f"  Quality: {result.get('quality', 0)}/10")
        print(f"  Overall: {result.get('overall', 0)}/10")
        print(f"  Issues: {result.get('issues', 'N/A')}")

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    all_pass = True
    for lang, result in results.items():
        config = MODELS[lang]
        overall = result.get("overall", 0)
        status = "PASS" if overall >= MIN_SCORE else "FAIL"
        if overall < MIN_SCORE:
            all_pass = False
        print(f"{config['name']}: {overall}/10 [{status}]")

    print("=" * 60)
    print(f"Threshold: {MIN_SCORE}/10 (Phase 1 validation)")
    print(f"Result: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print("=" * 60)

    # Save results JSON
    results_path = output_dir / "validation_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "engine": "MMS-TTS (Meta)",
            "threshold": MIN_SCORE,
            "all_pass": all_pass,
            "results": results
        }, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {results_path}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    exit(main())
