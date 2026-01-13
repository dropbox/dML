#!/usr/bin/env python3
"""Benchmark CosyVoice2 on M4 Max with various configurations."""

import sys
import time
import os

# Use relative paths from script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, 'cosyvoice_repo'))

import torch
import torchaudio
from pathlib import Path

# Check MPS availability
print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

# Configuration
MODEL_DIR = os.path.join(PROJECT_DIR, 'models/cosyvoice/CosyVoice2-0.5B')
PROMPT_WAV = os.path.join(PROJECT_DIR, 'tests/golden/hello.wav')
OUTPUT_DIR = Path(os.path.join(PROJECT_DIR, 'models/cosyvoice/test_output'))

# Test texts of varying lengths
TEST_TEXTS = [
    ("short", "你好，今天天气真好！"),
    ("medium", "哎呀！你啷个还在屋头坐起嘛？要得不嘛！快点去超市买点东西回来！"),
    ("long", "今天我给你讲个故事。从前有座山，山上有座庙，庙里有个老和尚在给小和尚讲故事。他讲的什么故事呢？就是这个故事。"),
]

INSTRUCTION = "用四川话说这段话"

def benchmark_config(load_jit: bool, fp16: bool, device: str):
    """Benchmark a specific configuration."""
    config_name = f"jit={load_jit}_fp16={fp16}_device={device}"
    print(f"\n{'='*60}")
    print(f"Configuration: {config_name}")
    print(f"{'='*60}")

    try:
        from cosyvoice.cli.cosyvoice import CosyVoice2

        # Time model loading
        load_start = time.time()
        cosyvoice = CosyVoice2(
            MODEL_DIR,
            load_jit=load_jit,
            load_trt=False,  # No TensorRT on Mac
            load_vllm=False,  # No vLLM needed
            fp16=fp16
        )
        load_time = time.time() - load_start
        print(f"Model load time: {load_time:.2f}s")

        # Load prompt audio
        prompt_speech, sr = torchaudio.load(PROMPT_WAV)
        if sr != 16000:
            prompt_speech = torchaudio.functional.resample(prompt_speech, sr, 16000)

        results = []

        # Warmup
        print("\nWarmup run...")
        warmup_start = time.time()
        for _ in cosyvoice.inference_instruct2(
            "测试", INSTRUCTION, prompt_speech, stream=False
        ):
            pass
        warmup_time = time.time() - warmup_start
        print(f"Warmup time: {warmup_time:.2f}s")

        # Benchmark each text length
        for text_type, text in TEST_TEXTS:
            char_count = len(text)

            # Time inference
            start = time.time()
            audio_output = None
            for result in cosyvoice.inference_instruct2(
                text, INSTRUCTION, prompt_speech, stream=False
            ):
                audio_output = result['tts_speech']
            inference_time = time.time() - start

            # Calculate RTF
            audio_duration = audio_output.shape[1] / cosyvoice.sample_rate
            rtf = inference_time / audio_duration if audio_duration > 0 else float('inf')

            results.append({
                'text_type': text_type,
                'chars': char_count,
                'inference_time': inference_time,
                'audio_duration': audio_duration,
                'rtf': rtf
            })

            print(f"  {text_type}: {char_count} chars, {inference_time:.2f}s → {audio_duration:.2f}s audio (RTF: {rtf:.3f})")

        return {
            'config': config_name,
            'load_time': load_time,
            'warmup_time': warmup_time,
            'results': results,
            'avg_rtf': sum(r['rtf'] for r in results) / len(results)
        }

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("CosyVoice2 M4 Max Benchmark")
    print(f"Model: {MODEL_DIR}")
    print(f"Device: Apple M4 Max")

    benchmarks = []

    # Config 1: No JIT, fp32 (baseline)
    result = benchmark_config(load_jit=False, fp16=False, device='cpu')
    if result:
        benchmarks.append(result)

    # Config 2: JIT enabled, fp32
    result = benchmark_config(load_jit=True, fp16=False, device='cpu')
    if result:
        benchmarks.append(result)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for b in benchmarks:
        print(f"\n{b['config']}:")
        print(f"  Load time: {b['load_time']:.2f}s")
        print(f"  Warmup: {b['warmup_time']:.2f}s")
        print(f"  Average RTF: {b['avg_rtf']:.3f} (< 1.0 means faster than real-time)")

    if benchmarks:
        best = min(benchmarks, key=lambda x: x['avg_rtf'])
        print(f"\nBEST CONFIG: {best['config']} (RTF: {best['avg_rtf']:.3f})")


if __name__ == '__main__':
    main()
