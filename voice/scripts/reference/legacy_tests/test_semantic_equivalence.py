#!/usr/bin/env python3
"""
Semantic Equivalence Test

This is the ULTIMATE test for the translation + TTS pipeline.
It verifies that meaning is preserved through the entire roundtrip:

    English → Translate → Japanese → TTS → Audio → STT → Japanese → Translate → English

If the final English is semantically similar to the original, the pipeline works.

Usage:
    python tests/test_semantic_equivalence.py --pipeline "./pipeline.sh"

Exit codes:
    0 - Semantic similarity >= 70% (PASS)
    1 - Semantic similarity < 70% (FAIL)
"""

import argparse
import subprocess
import tempfile
import os
import sys

def get_sentence_embedding(text: str, model=None):
    """Get sentence embedding using sentence-transformers."""
    try:
        from sentence_transformers import SentenceTransformer
        if model is None:
            model = SentenceTransformer('all-MiniLM-L6-v2')
        return model.encode(text), model
    except ImportError:
        print("WARNING: sentence-transformers not installed")
        print("Install with: pip install sentence-transformers")
        return None, None

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    import numpy as np
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def transcribe_audio(wav_path: str, language: str = "ja") -> str:
    """Transcribe audio using Whisper."""
    try:
        import whisper
    except ImportError:
        print("ERROR: Whisper not installed")
        return ""

    model = whisper.load_model("base")
    result = model.transcribe(wav_path, language=language)
    return result["text"]

def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    """
    Translate text using available translation service.
    Falls back to simple comparison if no translator available.
    """
    # Try to use the project's NLLB translation
    try:
        # Check if we can call the daemon
        result = subprocess.run(
            ["./stream-tts-cpp", "--translate", text],
            capture_output=True, timeout=10, cwd="stream-tts-cpp/build"
        )
        if result.returncode == 0:
            return result.stdout.decode().strip()
    except:
        pass

    # Fallback: use transformers NLLB
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        model_name = "facebook/nllb-200-distilled-600M"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # Language codes
        lang_codes = {"en": "eng_Latn", "ja": "jpn_Jpan"}
        src_code = lang_codes.get(source_lang, source_lang)
        tgt_code = lang_codes.get(target_lang, target_lang)

        inputs = tokenizer(text, return_tensors="pt")
        translated = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id[tgt_code]
        )
        return tokenizer.decode(translated[0], skip_special_tokens=True)
    except:
        print("WARNING: No translation service available")
        return text  # Return original as fallback

def test_semantic_equivalence(
    original_text: str,
    audio_path: str,
    threshold: float = 0.70
) -> tuple:
    """
    Test semantic equivalence through the full pipeline.

    Returns:
        tuple: (passed, similarity_score, details)
    """
    print(f"\n{'='*60}")
    print("SEMANTIC EQUIVALENCE TEST")
    print(f"{'='*60}")

    print(f"\n1. Original English: '{original_text}'")

    # Step 1: Transcribe Japanese audio
    print("\n2. Transcribing Japanese audio...")
    ja_transcribed = transcribe_audio(audio_path, language="ja")
    print(f"   Japanese transcription: '{ja_transcribed}'")

    # Step 2: Translate back to English
    print("\n3. Translating back to English...")
    en_back = translate_text(ja_transcribed, "ja", "en")
    print(f"   Back-translated: '{en_back}'")

    # Step 3: Calculate semantic similarity
    print("\n4. Calculating semantic similarity...")
    emb_original, model = get_sentence_embedding(original_text)
    emb_back, _ = get_sentence_embedding(en_back, model)

    if emb_original is None or emb_back is None:
        # Fallback: simple word overlap
        orig_words = set(original_text.lower().split())
        back_words = set(en_back.lower().split())
        if orig_words:
            similarity = len(orig_words & back_words) / len(orig_words)
        else:
            similarity = 0.0
        print("   (Using word overlap - sentence-transformers not available)")
    else:
        similarity = cosine_similarity(emb_original, emb_back)

    print(f"   Similarity: {similarity:.1%}")

    passed = similarity >= threshold

    print(f"\n{'='*60}")
    if passed:
        print(f"✅ PASS - Semantic similarity {similarity:.0%} >= {threshold:.0%}")
    else:
        print(f"❌ FAIL - Semantic similarity {similarity:.0%} < {threshold:.0%}")
    print(f"{'='*60}")

    return passed, similarity, {
        "original": original_text,
        "ja_transcribed": ja_transcribed,
        "en_back": en_back
    }

def main():
    parser = argparse.ArgumentParser(description="Semantic Equivalence Test")
    parser.add_argument("--audio", required=True, help="Path to Japanese audio file")
    parser.add_argument("--original", required=True, help="Original English text")
    parser.add_argument("--threshold", type=float, default=0.70,
                       help="Minimum similarity threshold (default: 0.70)")

    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print(f"ERROR: Audio file not found: {args.audio}")
        sys.exit(1)

    passed, similarity, details = test_semantic_equivalence(
        args.original, args.audio, args.threshold
    )

    sys.exit(0 if passed else 1)

if __name__ == "__main__":
    main()
