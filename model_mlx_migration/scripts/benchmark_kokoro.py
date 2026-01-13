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
Kokoro TTS Model Benchmark

Compares MLX performance against PyTorch for text encoder and BERT.
"""

import sys
import time

import mlx.core as mx
import numpy as np
import torch

sys.path.insert(0, "/Users/ayates/model_mlx_migration")
from tools.pytorch_to_mlx.converters.kokoro_converter import KokoroConverter


def benchmark_mlx_text_encoder(model, input_ids, num_runs=100, warmup=10):
    """Benchmark MLX text encoder."""
    # Warmup
    for _ in range(warmup):
        out = model.text_encoder(input_ids)
        mx.eval(out)

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        out = model.text_encoder(input_ids)
        mx.eval(out)
        end = time.perf_counter()
        times.append(end - start)

    return np.array(times)


def benchmark_mlx_bert(model, input_ids, num_runs=100, warmup=10):
    """Benchmark MLX BERT."""
    # Warmup
    for _ in range(warmup):
        out = model.bert(input_ids)
        mx.eval(out)

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        out = model.bert(input_ids)
        mx.eval(out)
        end = time.perf_counter()
        times.append(end - start)

    return np.array(times)


def benchmark_pytorch_text_encoder(pt_state, input_ids_pt, num_runs=100, warmup=10):
    """Benchmark PyTorch text encoder embedding + conv."""
    te_state = pt_state["text_encoder"]
    embed_weight = te_state["module.embedding.weight"]

    # Create conv layer with weight norm
    conv = torch.nn.Conv1d(512, 512, 5, padding=2)
    weight_g = te_state["module.cnn.0.0.weight_g"]
    weight_v = te_state["module.cnn.0.0.weight_v"]
    bias = te_state["module.cnn.0.0.bias"]

    # Compute normalized weight
    v_norm = torch.sqrt(torch.sum(weight_v**2, dim=(1, 2), keepdim=True) + 1e-12)
    conv.weight = torch.nn.Parameter(weight_g * weight_v / v_norm)
    conv.bias = torch.nn.Parameter(bias)

    # MPS if available
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    embed_weight = embed_weight.to(device)
    conv = conv.to(device)
    input_ids_pt = input_ids_pt.to(device)

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            embed = torch.nn.functional.embedding(input_ids_pt, embed_weight)
            conv(embed.transpose(1, 2)).transpose(1, 2)

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        with torch.no_grad():
            embed = torch.nn.functional.embedding(input_ids_pt, embed_weight)
            conv(embed.transpose(1, 2)).transpose(1, 2)
            if device == "mps":
                torch.mps.synchronize()
        end = time.perf_counter()
        times.append(end - start)

    return np.array(times)


def benchmark_pytorch_bert_embedding(pt_state, input_ids_pt, num_runs=100, warmup=10):
    """Benchmark PyTorch BERT embedding layer."""
    bert_state = pt_state["bert"]

    word_weight = bert_state["module.embeddings.word_embeddings.weight"]
    pos_weight = bert_state["module.embeddings.position_embeddings.weight"]
    type_weight = bert_state["module.embeddings.token_type_embeddings.weight"]
    ln_weight = bert_state["module.embeddings.LayerNorm.weight"]
    ln_bias = bert_state["module.embeddings.LayerNorm.bias"]

    # MPS if available
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    word_weight = word_weight.to(device)
    pos_weight = pos_weight.to(device)
    type_weight = type_weight.to(device)
    ln_weight = ln_weight.to(device)
    ln_bias = ln_bias.to(device)
    input_ids_pt = input_ids_pt.to(device)

    seq_len = input_ids_pt.shape[1]
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    token_type_ids = torch.zeros_like(input_ids_pt, device=device)

    ln = torch.nn.LayerNorm(128, eps=1e-5).to(device)
    ln.weight = torch.nn.Parameter(ln_weight)
    ln.bias = torch.nn.Parameter(ln_bias)

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            word = torch.nn.functional.embedding(input_ids_pt, word_weight)
            pos = torch.nn.functional.embedding(position_ids, pos_weight)
            typ = torch.nn.functional.embedding(token_type_ids, type_weight)
            embed = word + pos + typ
            ln(embed)

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        with torch.no_grad():
            word = torch.nn.functional.embedding(input_ids_pt, word_weight)
            pos = torch.nn.functional.embedding(position_ids, pos_weight)
            typ = torch.nn.functional.embedding(token_type_ids, type_weight)
            embed = word + pos + typ
            ln(embed)
            if device == "mps":
                torch.mps.synchronize()
        end = time.perf_counter()
        times.append(end - start)

    return np.array(times)


def main():
    print("=" * 60)
    print("Kokoro TTS Model Benchmark")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    converter = KokoroConverter()
    model, config, pt_state = converter.load_from_hf("hexgrad/Kokoro-82M")

    # Test inputs
    test_input = list(range(1, 65))  # 64 token sequence
    input_ids = mx.array([test_input])
    input_ids_pt = torch.tensor([test_input])

    print(f"\nSequence length: {len(test_input)}")
    print(f"PyTorch device: {'mps' if torch.backends.mps.is_available() else 'cpu'}")
    print("Runs: 100, Warmup: 10")

    # Benchmark text encoder
    print("\n" + "-" * 60)
    print("Text Encoder Benchmark")
    print("-" * 60)

    mlx_times = benchmark_mlx_text_encoder(model, input_ids)
    pt_times = benchmark_pytorch_text_encoder(pt_state, input_ids_pt)

    print("\nMLX Text Encoder:")
    print(f"  Mean: {mlx_times.mean() * 1000:.3f} ms")
    print(f"  Std:  {mlx_times.std() * 1000:.3f} ms")
    print(f"  Min:  {mlx_times.min() * 1000:.3f} ms")
    print(f"  Max:  {mlx_times.max() * 1000:.3f} ms")

    print("\nPyTorch Text Encoder (embedding + conv):")
    print(f"  Mean: {pt_times.mean() * 1000:.3f} ms")
    print(f"  Std:  {pt_times.std() * 1000:.3f} ms")
    print(f"  Min:  {pt_times.min() * 1000:.3f} ms")
    print(f"  Max:  {pt_times.max() * 1000:.3f} ms")

    te_speedup = pt_times.mean() / mlx_times.mean()
    print(f"\nText Encoder Speedup: {te_speedup:.2f}x")

    # Benchmark BERT
    print("\n" + "-" * 60)
    print("BERT Embedding Benchmark")
    print("-" * 60)

    mlx_bert_times = benchmark_mlx_bert(model, input_ids)
    pt_bert_times = benchmark_pytorch_bert_embedding(pt_state, input_ids_pt)

    print("\nMLX BERT (full):")
    print(f"  Mean: {mlx_bert_times.mean() * 1000:.3f} ms")
    print(f"  Std:  {mlx_bert_times.std() * 1000:.3f} ms")
    print(f"  Min:  {mlx_bert_times.min() * 1000:.3f} ms")
    print(f"  Max:  {mlx_bert_times.max() * 1000:.3f} ms")

    print("\nPyTorch BERT (embedding only):")
    print(f"  Mean: {pt_bert_times.mean() * 1000:.3f} ms")
    print(f"  Std:  {pt_bert_times.std() * 1000:.3f} ms")
    print(f"  Min:  {pt_bert_times.min() * 1000:.3f} ms")
    print(f"  Max:  {pt_bert_times.max() * 1000:.3f} ms")

    # Summary
    print("\n" + "=" * 60)
    print("Benchmark Summary")
    print("=" * 60)
    print(f"\nText Encoder Speedup: {te_speedup:.2f}x vs PyTorch MPS")
    print(f"MLX BERT full forward: {mlx_bert_times.mean() * 1000:.3f} ms")

    return 0


if __name__ == "__main__":
    sys.exit(main())
