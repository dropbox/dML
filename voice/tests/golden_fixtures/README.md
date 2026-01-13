# Golden Audio Test Fixtures

Reference audio files generated from working TTS backends.

## Purpose

1. **Integration testing**: Verify TTS optimizations don't degrade quality
2. **Regression detection**: Compare new output against known-good samples
3. **Cross-validation**: Ensure C++ exports match Python originals

## Directory Structure

```
golden_fixtures/
├── kokoro/
│   ├── en_hello.wav          # English: "Hello, how are you?"
│   ├── ja_konnichiwa.wav     # Japanese: "こんにちは"
│   └── manifest.json         # Metadata for each file
├── melotts/
│   └── ...
├── openai/
│   └── ...
└── styletts2/
    └── ...
```

## manifest.json Format

```json
{
  "files": [
    {
      "filename": "en_hello.wav",
      "text": "Hello, how are you?",
      "language": "en",
      "voice": "af_heart",
      "expected_stt": "Hello, how are you?",
      "stt_accuracy": 1.0,
      "generated_at": "2025-12-04T12:00:00Z"
    }
  ]
}
```

## Generating Fixtures

```bash
# Generate from working Python backend
python scripts/generate_golden_fixtures.py --backend kokoro --langs en,ja

# Verify fixture with STT
python tests/test_japanese_stt.py golden_fixtures/kokoro/ja_konnichiwa.wav "こんにちは" medium
```

## Using in Tests

```python
def test_tts_matches_golden():
    """Ensure new TTS output is similar to golden reference."""
    golden = load_audio("golden_fixtures/kokoro/en_hello.wav")
    new = synthesize("Hello, how are you?", lang="en")

    # Compare using spectral similarity
    similarity = compute_mel_similarity(golden, new)
    assert similarity > 0.9, f"Output diverged from golden: {similarity}"
```
