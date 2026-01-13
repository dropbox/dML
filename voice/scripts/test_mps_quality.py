#!/usr/bin/env python3
"""Test CosyVoice2 quality with MPS on torch 2.5.1."""

import os
import sys
import time
import base64
import json
from pathlib import Path

# Add CosyVoice repo to path
COSYVOICE_REPO = os.path.join(os.path.dirname(__file__), '..', 'cosyvoice_repo')
sys.path.insert(0, COSYVOICE_REPO)
sys.path.insert(0, os.path.join(COSYVOICE_REPO, 'third_party', 'Matcha-TTS'))

MODEL_DIR = Path(__file__).parent.parent / "models" / "cosyvoice" / "CosyVoice2-0.5B"
OUTPUT_DIR = Path(__file__).parent.parent / "models" / "cosyvoice" / "test_output"
VOICE_SAMPLE = Path(__file__).parent.parent / "tests" / "golden" / "hello.wav"

# Test prompts
TEST_PROMPTS = [
    {"text": "你好，我是四川婆婆。", "instruction": "用四川话说", "name": "sichuan"},
    {"text": "今天天气真好，阳光明媚。", "instruction": "", "name": "mandarin"},
    {"text": "Hello, how are you today?", "instruction": "", "name": "english"},
    {"text": "我很开心见到你！", "instruction": "开心地说", "name": "happy"},
    {"text": "这个消息让我很难过。", "instruction": "悲伤地说", "name": "sad"},
]


def llm_judge(audio_path: Path) -> dict:
    """Use GPT-4o-audio-preview to evaluate audio quality."""
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(Path(__file__).parent.parent / ".env")
    client = OpenAI()

    with open(audio_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode()

    response = client.chat.completions.create(
        model="gpt-4o-audio-preview",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {"data": audio_b64, "format": "wav"}
                },
                {
                    "type": "text",
                    "text": """Rate this TTS audio on a scale of 1-10 for quality.
Focus on:
1. Naturalness (does it sound human-like?)
2. Clarity (is the speech clear and understandable?)
3. Artifacts (any robotic sounds, distortion, "dying frog" croaking?)

Return ONLY a JSON object:
{"score": <1-10>, "frog": <true/false>, "issues": "<brief description of any issues>"}"""
                }
            ]
        }],
        max_tokens=200
    )

    content = response.choices[0].message.content.strip()
    # Extract JSON from response
    if "```" in content:
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]

    try:
        return json.loads(content)
    except:
        return {"score": 0, "frog": True, "issues": f"Failed to parse: {content}"}


def main():
    import torch
    import torchaudio

    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")

    from cosyvoice.cli.cosyvoice import CosyVoice2
    from cosyvoice.utils.file_utils import load_wav

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Test both CPU and MPS
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")

    results = []

    for device in devices:
        print(f"\n{'='*60}")
        print(f"Testing on device: {device.upper()}")
        print(f"{'='*60}")

        print("\nLoading model...")
        # CosyVoice2 loads to CPU by default, we may need to move it
        cosyvoice = CosyVoice2(str(MODEL_DIR), load_jit=False, load_trt=False, fp16=False)
        prompt = load_wav(str(VOICE_SAMPLE), 16000)

        # Try to move to MPS if requested
        if device == "mps":
            try:
                cosyvoice.model.to(torch.device("mps"))
                print(f"Moved model to MPS")
            except Exception as e:
                print(f"Could not move to MPS: {e}")
                print("Using CPU instead")
                device = "cpu"

        print(f"Device: {cosyvoice.model.device}")

        for prompt_info in TEST_PROMPTS:
            text = prompt_info["text"]
            instruction = prompt_info["instruction"]
            name = prompt_info["name"]

            output_path = OUTPUT_DIR / f"mps_test_{device}_{name}.wav"

            print(f"\n--- Generating: {name} ---")
            print(f"Text: {text}")
            print(f"Instruction: {instruction or '(none)'}")

            start = time.time()
            all_speech = []

            try:
                if instruction:
                    for result in cosyvoice.inference_instruct2(text, instruction, prompt, stream=False):
                        all_speech.append(result['tts_speech'])
                else:
                    for result in cosyvoice.inference_sft(text, 'Narrator Female 1', stream=False):
                        all_speech.append(result['tts_speech'])

                full_speech = torch.cat(all_speech, dim=1)
                torchaudio.save(str(output_path), full_speech.cpu(), cosyvoice.sample_rate)

                duration = full_speech.shape[1] / cosyvoice.sample_rate
                gen_time = time.time() - start
                rtf = gen_time / duration

                print(f"Generated: {duration:.1f}s in {gen_time:.1f}s (RTF: {rtf:.2f}x)")
                print(f"Output: {output_path}")

                # LLM judge evaluation
                print("Evaluating with LLM-as-Judge...")
                judge_result = llm_judge(output_path)
                print(f"Score: {judge_result['score']}/10, Frog: {judge_result['frog']}")
                print(f"Issues: {judge_result['issues']}")

                results.append({
                    "device": device,
                    "name": name,
                    "text": text,
                    "instruction": instruction,
                    "duration": duration,
                    "gen_time": gen_time,
                    "rtf": rtf,
                    "output": str(output_path),
                    "llm_score": judge_result["score"],
                    "llm_frog": judge_result["frog"],
                    "llm_issues": judge_result["issues"]
                })

            except Exception as e:
                print(f"ERROR: {e}")
                results.append({
                    "device": device,
                    "name": name,
                    "error": str(e)
                })

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Device':<8} {'Name':<12} {'Score':<6} {'Frog':<6} {'RTF':<8} Issues")
    print("-" * 70)
    for r in results:
        if "error" in r:
            print(f"{r['device']:<8} {r['name']:<12} ERROR: {r['error']}")
        else:
            print(f"{r['device']:<8} {r['name']:<12} {r['llm_score']:<6} {str(r['llm_frog']):<6} {r['rtf']:<8.2f} {r['llm_issues'][:30]}...")

    # Save results
    results_path = OUTPUT_DIR / "mps_quality_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
