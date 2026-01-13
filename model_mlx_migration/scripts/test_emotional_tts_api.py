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
Test Emotional TTS API Endpoint.

This script tests the /synthesize_emotional endpoint of the DashVoice server.
It validates that emotional prosody is working correctly via the API.

Usage:
    # Start server first:
    python -m tools.dashvoice.server

    # Then run this test:
    python scripts/test_emotional_tts_api.py

    # Or test without server (standalone):
    python scripts/test_emotional_tts_api.py --standalone
"""

import argparse
import base64
import json
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_standalone():
    """Test emotional TTS directly without server."""
    import mlx.core as mx
    import numpy as np

    from tools.pytorch_to_mlx.converters import KokoroConverter
    from tools.pytorch_to_mlx.converters.models.kokoro_phonemizer import phonemize_text

    print("=" * 70)
    print("Emotional TTS Standalone Test")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    converter = KokoroConverter()
    model, config, _ = converter.load_from_hf()
    model.set_deterministic(True)

    # Load prosody weights
    contour_weights = Path("models/prosody_contour_v2.4/best_model.npz")
    embedding_path = Path("models/prosody_embeddings_orthogonal/final.safetensors")
    duration_energy_weights = Path("models/prosody_duration_energy_v3/best_model.npz")

    if not all(p.exists() for p in [contour_weights, embedding_path, duration_energy_weights]):
        print("ERROR: Prosody weights not found!")
        print(f"  {contour_weights}: {contour_weights.exists()}")
        print(f"  {embedding_path}: {embedding_path.exists()}")
        print(f"  {duration_energy_weights}: {duration_energy_weights.exists()}")
        return 1

    model.enable_prosody_contour_v2()
    model.load_prosody_contour_v2_weights(contour_weights, embedding_path)
    model.enable_prosody_duration_energy()
    model.load_prosody_duration_energy_weights(duration_energy_weights, embedding_path)
    print("  Prosody trifecta enabled")

    # Load voice pack
    voice_pack = converter.load_voice_pack("af_heart")
    mx.eval(voice_pack)

    # Test text
    text = "Hello, how are you doing today?"
    emotions = ["neutral", "angry", "sad", "excited"]

    print(f"\nGenerating audio for: '{text}'")
    print("-" * 70)

    output_dir = Path("/tmp/emotional_tts_api_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for emotion in emotions:
        # Phonemize
        phonemes, tokens = phonemize_text(text)

        # Prosody mask
        prosody_ids = {"neutral": 0, "angry": 40, "sad": 41, "excited": 42}
        prosody_id = prosody_ids[emotion]
        prosody_mask = mx.array([[prosody_id] * len(tokens)], dtype=mx.int32)

        # Voice embedding
        voice = converter.select_voice_embedding(voice_pack, len(tokens))

        # Generate
        tokens_mx = mx.array([tokens])
        audio = model(tokens_mx, voice, prosody_mask=prosody_mask)
        mx.eval(audio)
        audio_np = np.array(audio).flatten()

        # Stats
        duration = len(audio_np) / 24000
        rms = np.sqrt(np.mean(audio_np ** 2))

        results[emotion] = {"duration": duration, "rms": rms}

        # Save
        audio_path = output_dir / f"{emotion}.wav"
        audio_int16 = (audio_np * 32767).astype(np.int16)
        with wave.open(str(audio_path), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(audio_int16.tobytes())

        print(f"  {emotion:12s} duration={duration:.3f}s, rms={rms:.4f}, saved to {audio_path}")

    # Evaluate
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)

    neutral = results["neutral"]
    all_pass = True

    for emotion in ["angry", "sad", "excited"]:
        r = results[emotion]
        dur_ratio = r["duration"] / neutral["duration"]
        rms_ratio = r["rms"] / neutral["rms"]

        # Expected: angry=faster+louder, sad=slower+quieter, excited=faster+louder
        if emotion == "angry":
            dur_pass = dur_ratio < 1.0
            rms_pass = rms_ratio > 1.0
        elif emotion == "sad":
            dur_pass = dur_ratio > 1.0
            rms_pass = rms_ratio < 1.0
        elif emotion == "excited":
            dur_pass = dur_ratio < 1.0
            rms_pass = rms_ratio > 1.0
        else:
            dur_pass = rms_pass = True

        status = "PASS" if (dur_pass and rms_pass) else "FAIL"
        if not (dur_pass and rms_pass):
            all_pass = False

        print(f"  {emotion:12s} dur_ratio={dur_ratio:.3f}, rms_ratio={rms_ratio:.3f} [{status}]")

    print("\n" + "=" * 70)
    if all_pass:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 70)

    return 0 if all_pass else 1


def test_api(host: str = "localhost", port: int = 8000):
    """Test emotional TTS via API."""
    import requests

    print("=" * 70)
    print("Emotional TTS API Test")
    print("=" * 70)

    base_url = f"http://{host}:{port}"

    # Test /voices endpoint
    print("\n1. Testing /voices endpoint...")
    try:
        resp = requests.get(f"{base_url}/voices", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        print(f"   TTS enabled: {data.get('tts_enabled')}")
        print(f"   Emotional TTS: {data.get('emotional_tts', {})}")
    except requests.RequestException as e:
        print(f"   ERROR: {e}")
        print("   Is the server running? Start with: python -m tools.dashvoice.server")
        return 1

    # Test /synthesize_emotional endpoint
    print("\n2. Testing /synthesize_emotional endpoint...")
    text = "Hello, how are you doing today?"
    emotions = ["neutral", "angry", "sad", "excited"]

    output_dir = Path("/tmp/emotional_tts_api_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for emotion in emotions:
        try:
            resp = requests.post(
                f"{base_url}/synthesize_emotional",
                data={
                    "text": text,
                    "voice": "af_heart",
                    "emotion": emotion,
                    "speed": 1.0,
                },
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()

            results[emotion] = {
                "duration": data["duration_s"],
                "generation_time_ms": data["generation_time_ms"],
                "prosody_id": data["prosody_id"],
            }

            # Save audio
            audio_bytes = base64.b64decode(data["audio_base64"])
            audio_path = output_dir / f"api_{emotion}.wav"
            with open(audio_path, "wb") as f:
                f.write(audio_bytes)

            print(f"   {emotion:12s} duration={data['duration_s']:.3f}s, gen_time={data['generation_time_ms']:.0f}ms, saved to {audio_path}")

        except requests.RequestException as e:
            print(f"   {emotion:12s} ERROR: {e}")
            return 1

    # Evaluate
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)

    neutral = results["neutral"]
    all_pass = True

    for emotion in ["angry", "sad", "excited"]:
        r = results[emotion]
        dur_ratio = r["duration"] / neutral["duration"]

        # Expected: angry=faster, sad=slower, excited=faster
        if emotion == "angry":
            dur_pass = dur_ratio < 1.0
        elif emotion == "sad":
            dur_pass = dur_ratio > 1.0
        elif emotion == "excited":
            dur_pass = dur_ratio < 1.0
        else:
            dur_pass = True

        status = "PASS" if dur_pass else "FAIL"
        if not dur_pass:
            all_pass = False

        print(f"   {emotion:12s} dur_ratio={dur_ratio:.3f} [{status}]")

    print("\n" + "=" * 70)
    if all_pass:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print(f"Audio files saved to: {output_dir}")
    print("=" * 70)

    return 0 if all_pass else 1


def test_websocket(host: str = "localhost", port: int = 8000):
    """Test emotional TTS via WebSocket streaming."""
    import asyncio

    import websockets

    print("=" * 70)
    print("Emotional TTS WebSocket Streaming Test")
    print("=" * 70)

    async def run_ws_test():
        ws_url = f"ws://{host}:{port}/ws/synthesize_emotional"

        output_dir = Path("/tmp/emotional_tts_ws_test")
        output_dir.mkdir(parents=True, exist_ok=True)

        text = "Hello, how are you? I hope you are having a wonderful day."
        emotions = ["neutral", "angry", "sad", "excited"]

        results = {}
        all_pass = True

        for emotion in emotions:
            try:
                async with websockets.connect(ws_url) as ws:
                    # Send synthesis request
                    request = {
                        "type": "synthesize",
                        "text": text,
                        "voice": "af_heart",
                        "emotion": emotion,
                        "speed": 1.0,
                    }
                    await ws.send(json.dumps(request))

                    # Collect responses
                    chunks = []
                    ttfa = None
                    total_duration = 0
                    total_time_ms = 0

                    while True:
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=30)
                            data = json.loads(msg)

                            if data["type"] == "synthesis_started":
                                print(f"   {emotion}: Started synthesis...")

                            elif data["type"] == "audio_chunk":
                                chunks.append(data)
                                if ttfa is None:
                                    ttfa = data["elapsed_ms"]
                                total_duration += data["duration_s"]
                                segment_text = data.get("segment_text", "")[:30]
                                print(f"   {emotion}: Chunk {data['index']} ({data['duration_s']:.2f}s) '{segment_text}...'")

                            elif data["type"] == "complete":
                                total_time_ms = data["total_time_ms"]
                                break

                            elif data["type"] == "error":
                                print(f"   {emotion}: ERROR: {data['message']}")
                                all_pass = False
                                break

                        except asyncio.TimeoutError:
                            print(f"   {emotion}: Timeout waiting for response")
                            all_pass = False
                            break

                    results[emotion] = {
                        "chunks": len(chunks),
                        "duration": total_duration,
                        "ttfa_ms": ttfa,
                        "total_time_ms": total_time_ms,
                    }

                    # Save combined audio
                    if chunks:
                        all_audio = b""
                        for chunk in chunks:
                            audio_bytes = base64.b64decode(chunk["audio_base64"])
                            # Skip WAV header for concatenation (header is 44 bytes)
                            if len(all_audio) == 0:
                                all_audio = audio_bytes  # Keep first header
                            else:
                                all_audio += audio_bytes[44:]  # Skip headers for subsequent

                        audio_path = output_dir / f"ws_{emotion}.wav"
                        with open(audio_path, "wb") as f:
                            f.write(all_audio)
                        print(f"   {emotion}: Saved to {audio_path}")

            except Exception as e:
                print(f"   {emotion}: Exception: {e}")
                all_pass = False
                results[emotion] = {"error": str(e)}

        # Evaluate results
        print("\n" + "=" * 70)
        print("Results")
        print("=" * 70)

        if "neutral" in results and "duration" in results["neutral"]:
            neutral_dur = results["neutral"]["duration"]

            for emotion in ["angry", "sad", "excited"]:
                if emotion not in results or "duration" not in results[emotion]:
                    print(f"   {emotion:12s} MISSING")
                    all_pass = False
                    continue

                r = results[emotion]
                dur_ratio = r["duration"] / neutral_dur if neutral_dur > 0 else 0

                # Expected: angry=faster, sad=slower, excited=faster
                if emotion == "angry":
                    dur_pass = dur_ratio < 1.0
                elif emotion == "sad":
                    dur_pass = dur_ratio > 1.0
                elif emotion == "excited":
                    dur_pass = dur_ratio < 1.0
                else:
                    dur_pass = True

                status = "PASS" if dur_pass else "FAIL"
                if not dur_pass:
                    all_pass = False

                print(f"   {emotion:12s} chunks={r['chunks']}, dur_ratio={dur_ratio:.3f}, ttfa={r['ttfa_ms']:.0f}ms [{status}]")
        else:
            print("   ERROR: No neutral baseline available")
            all_pass = False

        print("\n" + "=" * 70)
        if all_pass:
            print("ALL WEBSOCKET TESTS PASSED")
        else:
            print("SOME WEBSOCKET TESTS FAILED")
        print(f"Audio files saved to: {output_dir}")
        print("=" * 70)

        return all_pass

    try:
        return 0 if asyncio.run(run_ws_test()) else 1
    except Exception as e:
        print(f"WebSocket test error: {e}")
        return 1


def main():
    parser = argparse.ArgumentParser(description="Test Emotional TTS API")
    parser.add_argument("--standalone", action="store_true", help="Test without server")
    parser.add_argument("--websocket", action="store_true", help="Test WebSocket streaming")
    parser.add_argument("--all", action="store_true", help="Run all tests (API + WebSocket)")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    args = parser.parse_args()

    if args.standalone:
        return test_standalone()
    elif args.websocket:
        return test_websocket(args.host, args.port)
    elif args.all:
        api_result = test_api(args.host, args.port)
        ws_result = test_websocket(args.host, args.port)
        return 0 if (api_result == 0 and ws_result == 0) else 1
    else:
        return test_api(args.host, args.port)


if __name__ == "__main__":
    sys.exit(main())
