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
OPT-10: Analyze potential for confidence-based adaptive early exit.

This script measures confidence at each decoder layer to determine
if early exit would be beneficial for translation models.

Key question: How often is the model confident enough at early layers
that we could skip remaining computation?

Confidence metrics:
- Max probability: P(argmax)
- Entropy: -sum(p * log(p))
- Margin: P(top1) - P(top2)
"""

import argparse
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

# Add tools to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.pytorch_to_mlx.converters.madlad_converter import MADLADConverter


def softmax(logits, axis=-1):
    """Numerically stable softmax."""
    logits = logits - mx.max(logits, axis=axis, keepdims=True)
    exp_logits = mx.exp(logits)
    return exp_logits / mx.sum(exp_logits, axis=axis, keepdims=True)


def compute_confidence_metrics(logits):
    """
    Compute confidence metrics from logits.

    Returns:
        dict with:
        - max_prob: Maximum probability
        - entropy: Entropy of distribution
        - margin: Difference between top-2 probabilities
        - top_token: Most likely token ID
    """
    # Convert to float32 for numpy compatibility
    logits = logits.astype(mx.float32)
    probs = softmax(logits)
    mx.eval(probs)
    probs_np = np.array(probs)

    # Max probability
    max_prob = float(np.max(probs_np))

    # Entropy (avoid log(0))
    probs_clipped = np.clip(probs_np, 1e-10, 1.0)
    entropy = float(-np.sum(probs_np * np.log(probs_clipped)))

    # Margin between top-2
    top2_idx = np.argpartition(probs_np, -2)[-2:]
    top2_probs = probs_np[top2_idx]
    margin = float(np.max(top2_probs) - np.min(top2_probs))

    # Top token
    top_token = int(np.argmax(probs_np))

    return {
        "max_prob": max_prob,
        "entropy": entropy,
        "margin": margin,
        "top_token": top_token,
    }


def analyze_layer_confidence(
    converter,
    text: str,
    tgt_lang: str = "de",
    exit_layers: list = None,
):
    """
    Analyze confidence at each decoder layer during generation.

    This probes intermediate layers to see when the model becomes confident.
    """
    if exit_layers is None:
        exit_layers = [4, 8, 12, 16, 20, 24]  # For 24-layer MADLAD

    converter.load()
    model = converter.model
    tokenizer = converter.tokenizer

    # Prepare input
    input_text = f"<2{tgt_lang}> {text}"
    input_ids = tokenizer.encode(input_text, return_tensors="np")
    input_ids = mx.array(input_ids)

    # Encode
    encoder_output = model.encode(input_ids)
    mx.eval(encoder_output)

    # Setup decoder
    decoder_start_id = 0  # MADLAD uses 0
    decoder_ids = [[decoder_start_id]]
    decoder_ids = mx.array(decoder_ids)

    # Initialize full cache (24 layers)
    cache = model.decoder.make_cache()

    # Track per-token confidence at each layer
    token_layer_confidence = []
    generated_tokens = []

    max_tokens = 128
    eos_id = tokenizer.eos_token_id

    for step in range(max_tokens):
        # Get full model output first
        inputs_emb = model.wte(decoder_ids)
        T = inputs_emb.shape[1]

        if cache[0] is not None:
            offset = cache[0].offset
        else:
            offset = 0

        # Setup position bias and mask
        T_total = offset + T
        pos_bias = model.decoder.relative_attention_bias(T_total, T_total, offset=offset)

        if T > 1:
            query_pos = mx.arange(T) + offset
            key_pos = mx.arange(T_total)
            causal_mask = key_pos[None, :] > query_pos[:, None]
            mask = mx.where(causal_mask, float("-inf"), 0.0).astype(inputs_emb.dtype)
            mask = mask + pos_bias
        else:
            mask = pos_bias

        # Run through all layers, collecting intermediate outputs
        layer_outputs = []
        x = inputs_emb

        for i, layer in enumerate(model.decoder.layers):
            x, cache[i] = layer(x, encoder_output, mask, memory_mask=None, cache=cache[i])

            # Check if this is an exit layer
            layer_num = i + 1
            if layer_num in exit_layers:
                # Apply layer norm and get logits
                x_normed = model.decoder.ln(x)
                if not model.tie_word_embeddings:
                    logits = model.lm_head(x_normed)
                else:
                    logits = x_normed * model.model_dim**-0.5 @ model.wte.weight.T

                layer_outputs.append({
                    "layer": layer_num,
                    "logits": logits[0, -1],  # Last token logits
                })

        # Final output
        x_final = model.decoder.ln(x)
        if not model.tie_word_embeddings:
            final_logits = model.lm_head(x_final)
        else:
            final_logits = x_final * model.model_dim**-0.5 @ model.wte.weight.T

        mx.eval(final_logits, cache, *[lo["logits"] for lo in layer_outputs])

        # Compute confidence at each layer
        step_confidence = {"step": step, "layers": []}

        for lo in layer_outputs:
            metrics = compute_confidence_metrics(lo["logits"])
            step_confidence["layers"].append({
                "layer": lo["layer"],
                **metrics
            })

        # Final layer confidence
        final_metrics = compute_confidence_metrics(final_logits[0, -1])
        step_confidence["final"] = final_metrics

        token_layer_confidence.append(step_confidence)

        # Get next token (from final layer)
        next_token = int(mx.argmax(final_logits[0, -1]))
        generated_tokens.append(next_token)

        if next_token == eos_id:
            break

        decoder_ids = mx.array([[next_token]])

    # Decode output
    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return {
        "input": text,
        "output": output_text,
        "token_confidence": token_layer_confidence,
        "exit_layers": exit_layers,
    }


def summarize_results(results, confidence_threshold=0.95):
    """
    Summarize analysis results.

    Reports:
    - For each layer, % of tokens where it matches final layer prediction
    - For each layer, % of tokens with confidence > threshold
    - Potential speedup from early exit
    """
    print("\n" + "="*70)
    print("ADAPTIVE EARLY EXIT ANALYSIS")
    print("="*70)

    print(f"\nInput: {results['input']}")
    print(f"Output: {results['output']}")
    print(f"Tokens generated: {len(results['token_confidence'])}")
    print(f"Confidence threshold: {confidence_threshold}")

    exit_layers = results["exit_layers"]
    num_tokens = len(results["token_confidence"])

    # Per-layer statistics
    print("\n" + "-"*70)
    print(f"{'Layer':<8} {'Match%':<10} {'ConfHigh%':<12} {'AvgProb':<10} {'AvgEntropy':<12}")
    print("-"*70)

    layer_stats = {}
    for layer in exit_layers:
        matches = 0
        high_conf = 0
        probs = []
        entropies = []

        for tc in results["token_confidence"]:
            final_token = tc["final"]["top_token"]

            for lc in tc["layers"]:
                if lc["layer"] == layer:
                    probs.append(lc["max_prob"])
                    entropies.append(lc["entropy"])

                    if lc["top_token"] == final_token:
                        matches += 1
                    if lc["max_prob"] >= confidence_threshold:
                        high_conf += 1
                    break

        match_pct = 100 * matches / num_tokens if num_tokens > 0 else 0
        conf_pct = 100 * high_conf / num_tokens if num_tokens > 0 else 0
        avg_prob = np.mean(probs) if probs else 0
        avg_entropy = np.mean(entropies) if entropies else 0

        layer_stats[layer] = {
            "match_pct": match_pct,
            "conf_pct": conf_pct,
            "avg_prob": avg_prob,
            "avg_entropy": avg_entropy,
        }

        print(f"{layer:<8} {match_pct:<10.1f} {conf_pct:<12.1f} {avg_prob:<10.3f} {avg_entropy:<12.3f}")

    # Calculate potential speedup
    print("\n" + "-"*70)
    print("POTENTIAL SPEEDUP ANALYSIS")
    print("-"*70)

    total_layers = max(exit_layers)

    for layer in exit_layers:
        # Speedup if we exit at this layer when confident
        match_rate = layer_stats[layer]["match_pct"] / 100
        conf_rate = layer_stats[layer]["conf_pct"] / 100

        # Effective early exit rate = min(match_rate, conf_rate) * conf_rate
        # (we only exit when confident AND correct)
        early_exit_rate = match_rate * conf_rate

        # Speedup = weighted average of layers used
        # exit_rate at layer L means we do L layers instead of total_layers
        # More conservative: speedup factor
        effective_layers = layer * early_exit_rate + total_layers * (1 - early_exit_rate)
        speedup = total_layers / effective_layers if effective_layers > 0 else 1.0

        print(f"Layer {layer}: Exit rate={early_exit_rate*100:.1f}%, "
              f"Effective layers={effective_layers:.1f}, "
              f"Speedup={speedup:.2f}x")

    return layer_stats


def main():
    parser = argparse.ArgumentParser(description="Analyze adaptive early exit potential")
    parser.add_argument("--model", default=None, help="Model path (default: HuggingFace)")
    parser.add_argument("--text", default="Hello, how are you today? I hope you are doing well.",
                        help="Text to translate")
    parser.add_argument("--tgt-lang", default="de", help="Target language")
    parser.add_argument("--quantize", type=int, default=8, help="Quantization bits (8, 4, or 0 for none)")
    parser.add_argument("--threshold", type=float, default=0.95,
                        help="Confidence threshold for early exit")
    args = parser.parse_args()

    quantize = args.quantize if args.quantize > 0 else None

    print("OPT-10: Adaptive Early Exit Analysis")
    print("="*70)
    print(f"Model: {args.model or 'default MADLAD-400'}")
    print(f"Quantization: {quantize}-bit" if quantize else "none")
    print(f"Target language: {args.tgt_lang}")

    # Load model
    converter = MADLADConverter(
        model_path=args.model,
        quantize=quantize,
    )

    print(f"\nAnalyzing text: {args.text}")

    # Run analysis
    results = analyze_layer_confidence(
        converter,
        text=args.text,
        tgt_lang=args.tgt_lang,
    )

    # Summarize
    layer_stats = summarize_results(results, confidence_threshold=args.threshold)

    return layer_stats


if __name__ == "__main__":
    main()
