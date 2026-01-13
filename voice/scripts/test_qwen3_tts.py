#!/usr/bin/env python3
"""Test Qwen3-TTS via DashScope API.

Sichuanese voices:
- Eric (四川-程川) - Male Sichuanese
- Sunny (四川-晴儿) - Female Sichuanese
"""

import os
import sys
from pathlib import Path

# Load .env if exists
env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())

import dashscope
from dashscope.audio.tts_v2 import SpeechSynthesizer, AudioFormat, ResultCallback

# Configuration
OUTPUT_DIR = Path(__file__).parent.parent / "models" / "qwen3_tts_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set up authentication
# DashScope SDK uses DASHSCOPE_API_KEY environment variable
# AccessKey auth is set via ALIBABA_CLOUD_ACCESS_KEY_ID/SECRET env vars
API_KEY = os.environ.get("DASHSCOPE_API_KEY")
ACCESS_KEY_ID = os.environ.get("ALIBABA_CLOUD_ACCESS_KEY_ID")
ACCESS_KEY_SECRET = os.environ.get("ALIBABA_CLOUD_ACCESS_KEY_SECRET")

if API_KEY:
    dashscope.api_key = API_KEY
    print(f"Using DashScope API key: {API_KEY[:8]}...")
elif ACCESS_KEY_ID and ACCESS_KEY_SECRET:
    # AccessKey auth - SDK reads from environment variables automatically
    print(f"Using AccessKey: {ACCESS_KEY_ID[:8]}...")
    print("Note: SDK will read ALIBABA_CLOUD_ACCESS_KEY_ID/SECRET from env")
else:
    print("ERROR: No credentials found!")
    print("Set DASHSCOPE_API_KEY or ALIBABA_CLOUD_ACCESS_KEY_ID/SECRET")
    sys.exit(1)

# Available Sichuanese voices
SICHUAN_VOICES = {
    "Eric": "四川-程川 (Male)",
    "Sunny": "四川-晴儿 (Female)",
}

def test_voice(voice_id: str, text: str):
    """Test a single voice."""
    print(f"\nTesting voice: {voice_id} - {SICHUAN_VOICES.get(voice_id, '')}")
    print(f"Text: {text}")

    output_file = OUTPUT_DIR / f"qwen3_tts_{voice_id.lower()}.wav"

    # Collect audio chunks
    audio_data = bytearray()

    class Callback(ResultCallback):
        def on_event(self, result):
            if result.get_audio_frame() is not None:
                audio_data.extend(result.get_audio_frame())

    try:
        synthesizer = SpeechSynthesizer(
            model="cosyvoice-v1",
            voice=voice_id,
            format=AudioFormat.WAV_24000HZ_MONO_16BIT,
            callback=Callback(),
        )

        synthesizer.call(text)

        if len(audio_data) > 0:
            with open(output_file, "wb") as f:
                f.write(audio_data)
            print(f"Saved: {output_file} ({len(audio_data)} bytes)")
            return output_file
        else:
            print("No audio data received")
            return None

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("=" * 60)
    print("Qwen3-TTS Test - Sichuanese Voices")
    print("=" * 60)

    # Test Sichuanese voices
    test_text = "你好，我是四川人！今天天气真好啊。"

    print("\n--- Testing Sichuanese Voices ---")
    for voice_id in SICHUAN_VOICES:
        output = test_voice(voice_id, test_text)
        if output:
            print(f"Playing {voice_id}...")
            os.system(f"afplay '{output}'")

    print("\n--- Done ---")
    print(f"Output files: {OUTPUT_DIR}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
