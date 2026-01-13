#!/usr/bin/env python3
"""CosyVoice2 MPS Benchmark - Tests MPS with torch.compile for Apple Silicon.

MANAGER directive: torch.compile achieves RTF 0.83 on MPS (faster than real-time).
This benchmark tests three configurations:
1. CPU baseline (no compile)
2. MPS without compile
3. MPS with torch.compile(mode='reduce-overhead')
"""

import sys
import os
import time
import argparse
from contextlib import nullcontext

# Must patch BEFORE importing CosyVoice
import torch

# Check MPS availability
if not torch.backends.mps.is_available():
    print("ERROR: MPS not available on this system")
    sys.exit(1)

MPS_DEVICE = torch.device('mps')
print(f"MPS device available: {MPS_DEVICE}")
print(f"PyTorch version: {torch.__version__}")

# Patch torch.cuda functions for MPS compatibility
_original_cuda_is_available = torch.cuda.is_available
torch.cuda.is_available = lambda: False  # Force CPU path initially

# Use relative paths from script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, 'cosyvoice_repo'))

# Now import CosyVoice (will use CPU since we disabled CUDA check)
from cosyvoice.cli.cosyvoice import CosyVoice2
import torchaudio

MODEL_DIR = os.path.join(PROJECT_DIR, 'models/cosyvoice/CosyVoice2-0.5B')
PROMPT_WAV = os.path.join(PROJECT_DIR, 'tests/golden/hello.wav')

# Parse arguments
parser = argparse.ArgumentParser(description='CosyVoice2 MPS Benchmark')
parser.add_argument('--compile', action='store_true', help='Use torch.compile on ALL components (LLM, Flow, HiFT)')
parser.add_argument('--compile-llm-only', action='store_true', help='Use torch.compile on LLM only (legacy mode)')
parser.add_argument('--cpu', action='store_true', help='Run on CPU (baseline)')
parser.add_argument('--warmup', type=int, default=3, help='Number of warmup runs')
args = parser.parse_args()

print("\n=== Loading CosyVoice2 ===")
load_start = time.time()
cosyvoice = CosyVoice2(MODEL_DIR, load_jit=False, load_trt=False, fp16=False)
cpu_load_time = time.time() - load_start
print(f"Loaded on CPU in {cpu_load_time:.2f}s")

if not args.cpu:
    # Move to MPS
    print("\n=== Moving to MPS ===")
    move_start = time.time()
    cosyvoice.model.device = MPS_DEVICE
    cosyvoice.model.llm.to(MPS_DEVICE)
    cosyvoice.model.flow.to(MPS_DEVICE)
    cosyvoice.model.hift.to(MPS_DEVICE)
    cosyvoice.frontend.device = MPS_DEVICE
    move_time = time.time() - move_start
    print(f"Moved to MPS in {move_time:.2f}s")

    if args.compile or args.compile_llm_only:
        # Apply torch.compile
        print("\n=== Applying torch.compile ===")
        compile_start = time.time()

        # Always compile LLM
        print("  Compiling LLM...")
        cosyvoice.model.llm.llm = torch.compile(
            cosyvoice.model.llm.llm,
            mode='reduce-overhead',
            backend='inductor'
        )

        if args.compile:
            # Full compilation: LLM + Flow + HiFT (best performance)
            print("  Compiling Flow...")
            cosyvoice.model.flow = torch.compile(
                cosyvoice.model.flow,
                mode='reduce-overhead',
                backend='inductor'
            )

            print("  Compiling HiFT...")
            cosyvoice.model.hift = torch.compile(
                cosyvoice.model.hift,
                mode='reduce-overhead',
                backend='inductor'
            )
            print("  All components compiled (LLM + Flow + HiFT)")
        else:
            print("  LLM only compiled (use --compile for full optimization)")

        compile_time = time.time() - compile_start
        print(f"  torch.compile setup in {compile_time:.2f}s")

# Verify devices
device_str = "CPU" if args.cpu else "MPS"
print(f"\n  Running on: {device_str}")
print(f"  LLM device: {next(cosyvoice.model.llm.parameters()).device}")
print(f"  Flow device: {next(cosyvoice.model.flow.parameters()).device}")
print(f"  HiFT device: {next(cosyvoice.model.hift.parameters()).device}")
if args.compile:
    print(f"  torch.compile: ALL COMPONENTS (best performance)")
elif args.compile_llm_only:
    print(f"  torch.compile: LLM ONLY")

# Load prompt audio
prompt_speech, sr = torchaudio.load(PROMPT_WAV)
if sr != 16000:
    prompt_speech = torchaudio.functional.resample(prompt_speech, sr, 16000)

# Test texts
TEST_CASES = [
    ("你好！", "用四川话说"),
    ("你好，今天天气真好！", "用四川话说这段话"),
]

# Warmup runs (important for torch.compile - compilation happens on first run)
if args.warmup > 0:
    print(f"\n=== Warmup Runs ({args.warmup}) ===")
    for i in range(args.warmup):
        warmup_text, warmup_instruction = TEST_CASES[0]
        print(f"  Warmup {i+1}/{args.warmup}...", end=" ", flush=True)
        warmup_start = time.time()
        for result in cosyvoice.inference_instruct2(warmup_text, warmup_instruction, prompt_speech, stream=False):
            _ = result['tts_speech']
        warmup_time = time.time() - warmup_start
        print(f"done ({warmup_time:.2f}s)")
    print("  Warmup complete - JIT compilation done")

print("\n=== Running Benchmarks ===")
results = []

for text, instruction in TEST_CASES:
    print(f"\nText: {text} ({len(text)} chars)")
    print(f"Instruction: {instruction}")

    # Run inference
    start = time.time()
    audio_output = None
    for result in cosyvoice.inference_instruct2(text, instruction, prompt_speech, stream=False):
        audio_output = result['tts_speech']
    inference_time = time.time() - start

    audio_duration = audio_output.shape[1] / cosyvoice.sample_rate
    rtf = inference_time / audio_duration

    print(f"  Inference: {inference_time:.2f}s")
    print(f"  Audio: {audio_duration:.2f}s")
    print(f"  RTF: {rtf:.3f}")

    results.append({
        'chars': len(text),
        'inference_time': inference_time,
        'audio_duration': audio_duration,
        'rtf': rtf
    })

# Summary
mode_str = "CPU" if args.cpu else ("MPS + torch.compile" if args.compile else "MPS")
print("\n" + "="*60)
print(f"SUMMARY ({mode_str})")
print("="*60)
avg_rtf = sum(r['rtf'] for r in results) / len(results)
print(f"Average RTF: {avg_rtf:.3f}")
print(f"\nComparison:")
print(f"  CPU baseline RTF:    1.086")
print(f"  MPS no-compile RTF:  1.211")
print(f"  Your RTF ({mode_str}): {avg_rtf:.3f}")
if avg_rtf < 1.0:
    print(f"  STATUS: FASTER than real-time!")
elif avg_rtf < 1.086:
    print(f"  STATUS: Faster than CPU baseline")
else:
    print(f"  STATUS: Slower than CPU baseline")
