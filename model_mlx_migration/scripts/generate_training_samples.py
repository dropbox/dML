#!/usr/bin/env python3
# Copyright 2024-2025 Andrew Yates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Generate training sample audio for emotion review.
Uses baseline Kokoro (no prosody conditioning) so we can hear the raw text.

This generates 20+ samples per emotion for proper training coverage.
"""

import os
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import mlx.core as mx

# Extended sample sentences - 20+ per emotion for proper training
SAMPLES = {
    "neutral": [
        "The weather forecast shows partly cloudy skies tomorrow.",
        "Please remember to submit your report by Friday.",
        "The meeting has been rescheduled to three o'clock.",
        "Your order has been shipped and will arrive next week.",
        "The instructions are listed on the back of the package.",
        "Turn left at the next intersection and continue straight.",
        "The library closes at eight o'clock on weekdays.",
        "Your appointment is confirmed for Monday at two.",
        "The document has been saved to your desktop folder.",
        "Press enter to continue with the installation process.",
        "The total comes to forty seven dollars and sixty cents.",
        "Please hold while I transfer your call to the next available agent.",
        "The building is located on the corner of Main and Fifth Street.",
        "Your password must contain at least eight characters.",
        "The estimated delivery time is three to five business days.",
        "Chapter three begins on page forty two.",
        "The office will be closed for the holiday weekend.",
        "Please enter your confirmation number to proceed.",
        "The next train departs in approximately fifteen minutes.",
        "You can find more information on our website.",
    ],
    "angry": [
        "How DARE you say that to me after everything I have done!",
        "I am absolutely SICK of your constant excuses!",
        "You had ONE job and you completely FAILED!",
        "This is UNACCEPTABLE and I will NOT tolerate it!",
        "Get OUT of my sight RIGHT NOW!",
        "I cannot BELIEVE you would do something so STUPID!",
        "What the HELL were you thinking?!",
        "Do NOT test my patience any further!",
        "I have HAD it with your incompetence!",
        "This is the LAST time I will warn you!",
        "How MANY times do I have to tell you?!",
        "Your behavior is absolutely DISGRACEFUL!",
        "I DEMAND an explanation for this mess!",
        "Stop making excuses and FIX this immediately!",
        "I am FURIOUS about what you have done!",
        "This is completely OUTRAGEOUS!",
        "Don't you DARE walk away from me!",
        "I trusted you and THIS is how you repay me?!",
        "Enough is ENOUGH! I am done with this!",
        "You have gone TOO far this time!",
    ],
    "sad": [
        "I just received the worst news of my entire life.",
        "Everything I worked for is gone now.",
        "I don't know how I can go on after this.",
        "They said there's nothing more they can do.",
        "I feel so empty and alone.",
        "I miss them more than words can ever express.",
        "Nothing will ever be the same again.",
        "I wish I could go back and change things.",
        "The pain just won't go away no matter what I do.",
        "I feel like I've lost everything that mattered.",
        "Why did this have to happen to us?",
        "I can barely find the strength to get through the day.",
        "All those memories just make it hurt more.",
        "I don't think I'll ever truly recover from this.",
        "Sometimes I wonder if it's even worth trying anymore.",
        "The house feels so empty without them here.",
        "I keep hoping this is all just a bad dream.",
        "No one really understands what I'm going through.",
        "I feel so helpless and lost right now.",
        "Every reminder just brings back all the pain.",
    ],
    "happy": [
        "Congratulations! You have won the grand prize!",
        "This is the most wonderful day of my life!",
        "I am so incredibly proud of what you've achieved!",
        "We did it! We finally made it happen!",
        "Thank you so much for this amazing surprise!",
        "I couldn't be happier right now!",
        "What a beautiful and perfect moment this is!",
        "You have made all my dreams come true!",
        "This calls for a celebration!",
        "Everything is working out perfectly!",
        "I am so grateful for all of you!",
        "This is exactly what I've always wanted!",
        "Life is absolutely wonderful right now!",
        "I love every single moment of this!",
        "What an incredible blessing this has been!",
        "My heart is so full of joy right now!",
        "You have made me the happiest person alive!",
        "This is truly a dream come true!",
        "I can't stop smiling about this!",
        "Everything is coming together beautifully!",
    ],
    "excited": [
        "Oh my god, I can't believe this is really happening!",
        "Quick! Something incredible is about to start!",
        "This is insane! Did you see what just happened?",
        "I've never been more nervous in my entire life!",
        "The countdown has begun! Are you ready?",
        "Holy cow! That was absolutely AMAZING!",
        "Can you believe what we're about to witness?!",
        "I am SO pumped for this!",
        "This is going to be EPIC!",
        "I can barely contain my excitement right now!",
        "OMG it's actually happening! It's happening!",
        "Get ready because here we GO!",
        "I have been waiting SO long for this moment!",
        "My heart is RACING right now!",
        "This is the most thrilling thing I've ever experienced!",
        "Are you seeing this?! This is incredible!",
        "I cannot wait to see what happens next!",
        "The anticipation is absolutely KILLING me!",
        "This is going to change EVERYTHING!",
        "I am literally on the edge of my seat!",
    ],
}


def save_wav(audio: np.ndarray, path: str, sample_rate: int = 24000):
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


def main():
    from tools.pytorch_to_mlx.converters import KokoroConverter
    from tools.pytorch_to_mlx.converters.models.kokoro_phonemizer import phonemize_text

    output_dir = Path("/Users/ayates/voice/emotion_training_samples")

    print("Loading Kokoro model...")
    converter = KokoroConverter()
    model, config, _ = converter.load_from_hf()

    total_samples = sum(len(sentences) for sentences in SAMPLES.values())
    print(f"\nGenerating {total_samples} total samples across {len(SAMPLES)} emotions...")

    for emotion, sentences in SAMPLES.items():
        print(f"\n=== {emotion.upper()} ({len(sentences)} samples) ===")
        emotion_dir = output_dir / emotion
        emotion_dir.mkdir(parents=True, exist_ok=True)

        # Remove old files
        for old_file in emotion_dir.glob("*.wav"):
            old_file.unlink()

        for i, text in enumerate(sentences):
            print(f"  {i+1:2d}. {text[:45]}{'...' if len(text) > 45 else ''}")

            try:
                phonemes, token_ids = phonemize_text(text, language="en")
                input_ids = mx.array([token_ids])
                voice = converter.load_voice("af_heart", phoneme_length=len(phonemes))
                mx.eval(voice)

                audio = model.synthesize(input_ids, voice)
                mx.eval(audio)
                audio_np = np.array(audio).flatten()

                filename = f"en_{emotion}_{i+1:02d}.wav"
                save_wav(audio_np, str(emotion_dir / filename))
            except Exception as e:
                print(f"     ERROR: {e}")

    print(f"\n\n{'='*60}")
    print(f"Generated {total_samples} training samples in: {output_dir}")
    print(f"{'='*60}")
    print("\nSamples per emotion:")
    for emotion, sentences in SAMPLES.items():
        print(f"  {emotion}: {len(sentences)} samples")
    print("\nTo review, play each emotion folder:")
    for emotion in SAMPLES:
        print(f"  afplay {output_dir}/{emotion}/*.wav")


if __name__ == "__main__":
    main()
