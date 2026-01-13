#!/usr/bin/env python3
"""Export CosyVoice2 models optimized for MPS.

This script exports:
1. Flow model to TorchScript (MPS optimized)
2. HiFT (vocoder) to TorchScript (MPS optimized)
3. Attempts torch.export on LLM (experimental)

The exported models can be loaded in C++ via libtorch.
"""

import sys
import os
import time
import torch

# Patch CUDA before importing CosyVoice
torch.cuda.is_available = lambda: False

# Use relative paths from script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, 'cosyvoice_repo'))
from cosyvoice.cli.cosyvoice import CosyVoice2
import torchaudio

MODEL_DIR = os.path.join(PROJECT_DIR, 'models/cosyvoice/CosyVoice2-0.5B')
EXPORT_DIR = os.path.join(PROJECT_DIR, 'models/cosyvoice/exported_mps')
MPS_DEVICE = torch.device('mps')

os.makedirs(EXPORT_DIR, exist_ok=True)

print('Loading CosyVoice2...')
cosyvoice = CosyVoice2(MODEL_DIR, load_jit=False, load_trt=False, fp16=False)

# Move to MPS
print('Moving to MPS...')
cosyvoice.model.device = MPS_DEVICE
cosyvoice.model.llm.to(MPS_DEVICE)
cosyvoice.model.flow.to(MPS_DEVICE)
cosyvoice.model.hift.to(MPS_DEVICE)
cosyvoice.frontend.device = MPS_DEVICE

print()
print('=== Model Analysis ===')
print(f'LLM type: {type(cosyvoice.model.llm.llm).__name__}')
print(f'Flow type: {type(cosyvoice.model.flow).__name__}')
print(f'HiFT type: {type(cosyvoice.model.hift).__name__}')

# Try to export HiFT (vocoder) - should be straightforward
print()
print('=== Exporting HiFT (vocoder) ===')
try:
    hift = cosyvoice.model.hift
    hift.eval()

    # HiFT takes mel spectrogram and optional source
    # Input shape: (batch, channels, time)
    example_mel = torch.randn(1, 80, 100, device=MPS_DEVICE)

    # Trace the forward method
    with torch.no_grad():
        traced_hift = torch.jit.trace(hift, example_mel)

    hift_path = os.path.join(EXPORT_DIR, 'hift_mps.pt')
    traced_hift.save(hift_path)
    print(f'  Saved: {hift_path}')
    print(f'  Size: {os.path.getsize(hift_path) / 1024 / 1024:.1f} MB')
except Exception as e:
    print(f'  FAILED: {e}')

# Try to export Flow
print()
print('=== Exporting Flow ===')
try:
    flow = cosyvoice.model.flow
    flow.eval()

    # Flow is more complex - let's see what it needs
    print(f'  Flow class: {flow.__class__.__name__}')
    print(f'  Flow methods: {[m for m in dir(flow) if not m.startswith("_")]}')

    # The flow has an encoder that we can try to export
    if hasattr(flow, 'encoder'):
        encoder = flow.encoder
        print(f'  Flow encoder: {type(encoder).__name__}')

        # Try to trace encoder
        # Encoder typically takes token_embedding and token_embedding_len
        example_emb = torch.randn(1, 100, 512, device=MPS_DEVICE)  # (batch, seq, dim)
        example_len = torch.tensor([100], device=MPS_DEVICE)

        try:
            with torch.no_grad():
                traced_encoder = torch.jit.trace(encoder, (example_emb, example_len))
            encoder_path = os.path.join(EXPORT_DIR, 'flow_encoder_mps.pt')
            traced_encoder.save(encoder_path)
            print(f'  Saved encoder: {encoder_path}')
            print(f'  Size: {os.path.getsize(encoder_path) / 1024 / 1024:.1f} MB')
        except Exception as e:
            print(f'  Encoder trace failed: {e}')

except Exception as e:
    print(f'  FAILED: {e}')

# Try torch.compile AOT export on LLM
print()
print('=== Testing LLM Export Options ===')

# First, apply torch.compile
print('  Applying torch.compile...')
cosyvoice.model.llm.llm = torch.compile(
    cosyvoice.model.llm.llm,
    mode='reduce-overhead',
    backend='inductor'
)

# Warmup to trigger compilation
print('  Warming up...')
PROMPT_WAV = os.path.join(PROJECT_DIR, 'tests/golden/hello.wav')
prompt_speech, sr = torchaudio.load(PROMPT_WAV)
if sr != 16000:
    prompt_speech = torchaudio.functional.resample(prompt_speech, sr, 16000)

for i in range(2):
    for result in cosyvoice.inference_instruct2('你好', '用四川话说', prompt_speech, stream=False):
        _ = result['tts_speech']
print('  Warmup done')

# Try to get the compiled artifacts
print('  Checking compiled model...')
llm = cosyvoice.model.llm.llm
print(f'  LLM type after compile: {type(llm).__name__}')

# Check if we can use torch.export
try:
    from torch.export import export

    # The LLM is Qwen2ForCausalLM - we need proper example inputs
    # This is complex due to the autoregressive nature
    print('  torch.export is available but LLM has generator loops - cannot export directly')
except Exception as e:
    print(f'  torch.export test: {e}')

# Final benchmark
print()
print('=== Final Benchmark (compiled) ===')
texts = ['你好', '今天天气真好', '这是一个测试句子用来验证性能']
for text in texts:
    start = time.time()
    for result in cosyvoice.inference_instruct2(text, '用四川话说', prompt_speech, stream=False):
        audio = result['tts_speech']
    elapsed = time.time() - start
    audio_len = audio.shape[1] / 24000
    rtf = elapsed / audio_len
    print(f'  {len(text):2d} chars: {elapsed:.2f}s / {audio_len:.2f}s = RTF {rtf:.3f}')

print()
print('=== Summary ===')
print('Exported files in:', EXPORT_DIR)
for f in os.listdir(EXPORT_DIR):
    path = os.path.join(EXPORT_DIR, f)
    if os.path.isfile(path):
        print(f'  {f}: {os.path.getsize(path) / 1024 / 1024:.1f} MB')
