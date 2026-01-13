#!/usr/bin/env python3
"""Test CosyVoice2 quality with torch 2.5.1 on CPU (MPS not supported)."""

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

# Test prompts using instruct2 mode only (best quality)
TEST_PROMPTS = [
    {"text": "你好，我是四川婆婆。", "instruction": "用四川话说", "name": "sichuan"},
    {"text": "今天天气真好，阳光明媚。", "instruction": "用标准普通话说", "name": "mandarin"},
    {"text": "我很开心见到你！", "instruction": "开心地说", "name": "happy"},
    {"text": "这个消息让我很难过。", "instruction": "悲伤地说", "name": "sad"},
    {"text": "你确定要这样做吗？", "instruction": "生气地说", "name": "angry"},
    {"text": "Hello, how are you today?", "instruction": "用英语说", "name": "english"},
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

Return ONLY a JSON object, nothing else:
{"score": <1-10>, "frog": <true/false>, "issues": "<brief description>"}"""
                }
            ]
        }],
        max_tokens=150
    )

    content = response.choices[0].message.content.strip()
    # Clean up response
    if "```" in content:
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()

    try:
        return json.loads(content)
    except:
        # Try to extract score from text
        import re
        score_match = re.search(r'"score":\s*(\d+)', content)
        frog_match = re.search(r'"frog":\s*(true|false)', content)
        if score_match:
            return {
                "score": int(score_match.group(1)),
                "frog": frog_match.group(1) == "true" if frog_match else False,
                "issues": content[:100]
            }
        return {"score": 0, "frog": True, "issues": f"Parse error: {content[:50]}"}


def main():
    import torch
    import torchaudio

    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: CPU (MPS not supported for CosyVoice2)")

    from cosyvoice.cli.cosyvoice import CosyVoice2
    from cosyvoice.utils.file_utils import load_wav

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading model...")
    cosyvoice = CosyVoice2(str(MODEL_DIR), load_jit=False, load_trt=False, fp16=False)
    prompt = load_wav(str(VOICE_SAMPLE), 16000)

    results = []

    print(f"\n{'='*60}")
    print(f"Testing torch {torch.__version__} Quality (CPU)")
    print(f"{'='*60}")

    for prompt_info in TEST_PROMPTS:
        text = prompt_info["text"]
        instruction = prompt_info["instruction"]
        name = prompt_info["name"]

        output_path = OUTPUT_DIR / f"torch251_{name}.wav"

        print(f"\n--- {name.upper()} ---")
        print(f"Text: {text}")
        print(f"Instruction: {instruction}")

        start = time.time()
        all_speech = []

        try:
            for result in cosyvoice.inference_instruct2(text, instruction, prompt, stream=False):
                all_speech.append(result['tts_speech'])

            full_speech = torch.cat(all_speech, dim=1)
            torchaudio.save(str(output_path), full_speech.cpu(), cosyvoice.sample_rate)

            duration = full_speech.shape[1] / cosyvoice.sample_rate
            gen_time = time.time() - start
            rtf = gen_time / duration

            print(f"Generated: {duration:.1f}s in {gen_time:.1f}s (RTF: {rtf:.2f}x)")

            # LLM judge evaluation
            judge_result = llm_judge(output_path)
            print(f"Score: {judge_result['score']}/10, Frog: {judge_result['frog']}")
            print(f"Issues: {judge_result.get('issues', 'None')[:60]}")

            results.append({
                "name": name,
                "text": text,
                "instruction": instruction,
                "duration": round(duration, 2),
                "gen_time": round(gen_time, 2),
                "rtf": round(rtf, 2),
                "llm_score": judge_result["score"],
                "llm_frog": judge_result["frog"],
                "llm_issues": judge_result.get("issues", "")
            })

        except Exception as e:
            print(f"ERROR: {e}")
            results.append({"name": name, "error": str(e)})

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY - PyTorch 2.5.1 CPU Quality Test")
    print(f"{'='*60}")

    success_results = [r for r in results if "error" not in r]
    if success_results:
        avg_score = sum(r["llm_score"] for r in success_results) / len(success_results)
        frog_count = sum(1 for r in success_results if r["llm_frog"])
        avg_rtf = sum(r["rtf"] for r in success_results) / len(success_results)

        print(f"\nAverage Score: {avg_score:.1f}/10")
        print(f"Frog Detections: {frog_count}/{len(success_results)}")
        print(f"Average RTF: {avg_rtf:.2f}x")

        print(f"\n{'Name':<12} {'Score':<6} {'Frog':<6} {'RTF':<8} Issues")
        print("-" * 60)
        for r in results:
            if "error" in r:
                print(f"{r['name']:<12} ERROR: {r['error'][:40]}")
            else:
                issues = r.get('llm_issues', '')[:25] + "..." if r.get('llm_issues') else ""
                print(f"{r['name']:<12} {r['llm_score']:<6} {str(r['llm_frog']):<6} {r['rtf']:<8.2f} {issues}")

    # Save results
    results_path = OUTPUT_DIR / "torch251_quality_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "pytorch_version": torch.__version__,
            "device": "cpu",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": results
        }, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
