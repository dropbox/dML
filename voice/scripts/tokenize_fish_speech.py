#!/usr/bin/env python3
"""
tokenize_fish_speech.py - Pre-tokenize text for Fish-Speech C++ inference

Creates JSON files with tokenized prompts that can be loaded by the C++ test.

Usage:
    python3 scripts/tokenize_fish_speech.py --text "Hello world" --output tokens.json
    python3 scripts/tokenize_fish_speech.py --file input.txt --output tokens.json
"""

import argparse
import json
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Tokenize text for Fish-Speech")
    parser.add_argument("--text", type=str, help="Text to tokenize")
    parser.add_argument("--file", type=str, help="File containing text to tokenize")
    parser.add_argument("--model", type=str, default="models/fish-speech-1.5",
                       help="Path to fish-speech model directory")
    parser.add_argument("--output", type=str, default="tokens.json",
                       help="Output JSON file")
    args = parser.parse_args()

    if not args.text and not args.file:
        print("Error: Must provide --text or --file", file=sys.stderr)
        return 1

    text = args.text
    if args.file:
        with open(args.file) as f:
            text = f.read().strip()

    print(f"Input text: {text}")

    # Load tokenizer
    try:
        from fish_speech.tokenizer import FishTokenizer
    except ImportError:
        print("Error: fish_speech package not installed", file=sys.stderr)
        print("Install with: pip install fish-speech", file=sys.stderr)
        return 1

    tokenizer = FishTokenizer.from_pretrained(args.model)

    # Load special tokens
    special_tokens_path = Path(args.model) / "special_tokens.json"
    with open(special_tokens_path) as f:
        special_tokens = json.load(f)

    # Get key token IDs
    im_start = special_tokens["<|im_start|>"]
    im_end = special_tokens["<|im_end|>"]
    text_token = special_tokens["<|text|>"]
    voice_token = special_tokens["<|voice|>"]

    # Tokenize the text
    text_ids = tokenizer.encode(text)

    print(f"Text tokens ({len(text_ids)}): {text_ids}")

    # Build the full prompt
    # Fish-Speech format: <|im_start|><|text|>[text tokens]<|voice|>
    prompt_tokens = [im_start, text_token] + text_ids + [voice_token]

    print(f"Full prompt tokens ({len(prompt_tokens)}): {prompt_tokens}")

    # Save to JSON
    output = {
        "text": text,
        "prompt_tokens": prompt_tokens,
        "text_tokens": text_ids,
        "special_tokens": {
            "im_start": im_start,
            "im_end": im_end,
            "text": text_token,
            "voice": voice_token
        }
    }

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Saved to: {args.output}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
