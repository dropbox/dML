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
Train Early Exit Head for Speculative Decoding (OPT-4)

Trains a linear layer to predict full model token distributions from early layer
hidden states. This enables high-acceptance-rate speculative decoding.

Key concepts:
1. Full model runs normally, generating token predictions
2. We save hidden states from layer N (e.g., layer 4 of 24)
3. We train early_exit_head to match full model's soft labels using KL divergence

The trained head allows faster draft token generation during inference.

Usage:
    python scripts/train_early_exit_head.py --model google/madlad400-3b-mt --exit-layer 4
"""

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EarlyExitHead(nn.Module):
    """
    Linear projection from early layer hidden states to vocabulary logits.

    Can optionally include a small feedforward transformation for better matching.
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        use_ffn: bool = False,
        ffn_dim: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.use_ffn = use_ffn

        if use_ffn:
            ffn_dim = ffn_dim or d_model
            self.ffn = nn.Linear(d_model, ffn_dim, bias=False)
            self.out = nn.Linear(ffn_dim, vocab_size, bias=False)
        else:
            self.out = nn.Linear(d_model, vocab_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: Hidden states from early layer [B, T, d_model]

        Returns:
            logits: [B, T, vocab_size]
        """
        if self.use_ffn:
            x = nn.gelu(self.ffn(x))
        return self.out(x)


def collect_training_data(
    model,
    tokenizer,
    texts: List[str],
    tgt_lang: str,
    exit_layer: int,
    max_tokens: int = 128,
    batch_size: int = 1,
) -> List[Dict]:
    """
    Collect (early_hidden_states, full_model_logits) pairs for training.

    This runs the full model and captures intermediate activations.
    """
    samples = []

    for i, text in enumerate(texts):
        if (i + 1) % 10 == 0:
            logger.info(f"Collecting data: {i + 1}/{len(texts)}")

        # Prepare input with language tag
        input_text = f"<2{tgt_lang}> {text}"
        inputs = tokenizer(input_text, return_tensors="np")
        input_ids = mx.array(inputs["input_ids"])

        # Encode source
        encoder_output = model.encode(input_ids)

        # Run full decode to generate translation
        decoder_start_id = getattr(model.config, "decoder_start_token_id", 0)
        eos_token_id = tokenizer.eos_token_id

        generated = [decoder_start_id]
        cache = None

        # Collect hidden states and logits during decoding
        hidden_states_list = []
        full_logits_list = []

        # We need to capture intermediate hidden states
        # Decode one token at a time to collect training pairs
        for _ in range(max_tokens):
            decoder_ids = mx.array([[generated[-1]]])

            # Get both early hidden states and full logits
            early_hidden, full_logits, cache = decode_with_intermediate(
                model, decoder_ids, encoder_output, cache, exit_layer
            )
            mx.eval(early_hidden, full_logits, cache)

            # Store training pair (exclude first token which is decoder_start)
            if len(generated) > 1 or True:  # Include all for now
                hidden_states_list.append(early_hidden[0, -1])  # [d_model]
                full_logits_list.append(full_logits[0, -1])      # [vocab_size]

            # Get next token
            next_token = int(mx.argmax(full_logits[0, -1]))
            generated.append(next_token)

            if next_token == eos_token_id:
                break

        if len(hidden_states_list) > 1:  # Need at least 2 tokens
            samples.append({
                "input_text": text,
                "hidden_states": mx.stack(hidden_states_list),  # [T, d_model]
                "target_logits": mx.stack(full_logits_list),    # [T, vocab_size]
            })

    logger.info(f"Collected {len(samples)} training samples")
    return samples


def decode_with_intermediate(
    model,
    inputs: mx.array,
    memory: mx.array,
    cache,
    exit_layer: int,
) -> Tuple[mx.array, mx.array, list]:
    """
    Decode and return both early layer hidden states and full model logits.

    This is a modified version of decode that captures intermediate states.
    """
    x = model.wte(inputs)
    T = x.shape[1]

    # Setup cache and mask
    if cache is not None and cache[0] is not None:
        offset = cache[0].offset
    else:
        offset = 0
        from tools.pytorch_to_mlx.converters.models.t5_mlx import T5KVCache
        cache = [T5KVCache() for _ in range(len(model.decoder.layers))]

    # Causal mask
    if T > 1:
        total_kv_len = offset + T
        query_pos = mx.arange(T) + offset
        key_pos = mx.arange(total_kv_len)
        causal_mask = key_pos[None, :] > query_pos[:, None]
        mask = mx.where(causal_mask, float("-inf"), 0.0).astype(x.dtype)
    else:
        mask = None

    T_full = offset + T
    pos_bias = model.decoder.relative_attention_bias(T_full, T_full, offset=offset)
    if mask is not None:
        mask = mask + pos_bias
    else:
        mask = pos_bias

    # Run through all decoder layers, capturing state at exit_layer
    early_hidden = None
    for i, layer in enumerate(model.decoder.layers):
        x, cache[i] = layer(x, memory, mask, memory_mask=None, cache=cache[i])
        if i == exit_layer - 1:  # 0-indexed, so layer 4 = index 3
            # Create copy using array slicing (MLX doesn't have .copy())
            early_hidden = model.decoder.ln(x[:])

    # Final layer norm
    x = model.decoder.ln(x)

    # Project to logits
    if not model.tie_word_embeddings:
        full_logits = model.lm_head(x)
    else:
        full_logits = x * model.model_dim**-0.5
        full_logits = full_logits @ model.wte.weight.T

    return early_hidden, full_logits, cache


def kl_divergence_loss(pred_logits: mx.array, target_logits: mx.array, temperature: float = 1.0) -> mx.array:
    """
    KL divergence loss for knowledge distillation.

    Args:
        pred_logits: Predicted logits from early_exit_head [B, T, V]
        target_logits: Target logits from full model [B, T, V]
        temperature: Softmax temperature (higher = softer labels)

    Returns:
        KL divergence loss (scalar)
    """
    # Apply temperature scaling
    pred_scaled = pred_logits / temperature
    target_scaled = target_logits / temperature

    # Compute softmax probabilities
    pred_probs = mx.softmax(pred_scaled, axis=-1)
    target_probs = mx.softmax(target_scaled, axis=-1)

    # KL divergence: sum(target * log(target / pred))
    # = sum(target * log(target)) - sum(target * log(pred))
    # The first term is constant (entropy of target), so we minimize:
    # -sum(target * log(pred)) + temperature^2 scale factor

    # Use cross-entropy form: -sum(target * log_softmax(pred))
    log_pred = mx.log(pred_probs + 1e-10)
    kl = -mx.sum(target_probs * log_pred, axis=-1)

    return mx.mean(kl) * (temperature ** 2)


def cross_entropy_with_soft_labels(pred_logits: mx.array, target_logits: mx.array) -> mx.array:
    """
    Cross-entropy loss with soft labels from teacher model.

    This is simpler and often works as well as KL divergence.
    """
    target_probs = mx.softmax(target_logits, axis=-1)
    log_pred = mx.log_softmax(pred_logits, axis=-1)

    loss = -mx.sum(target_probs * log_pred, axis=-1)
    return mx.mean(loss)


def train_epoch(
    head: EarlyExitHead,
    optimizer: optim.Optimizer,
    samples: List[Dict],
    batch_size: int = 8,
    temperature: float = 2.0,
) -> float:
    """Train for one epoch, return average loss."""
    random.shuffle(samples)
    total_loss = 0.0
    num_batches = 0

    def loss_fn(head, hidden_states, target_logits):
        pred_logits = head(hidden_states)
        return kl_divergence_loss(pred_logits, target_logits, temperature)

    loss_and_grad = nn.value_and_grad(head, loss_fn)

    # Process samples (each sample is a sequence)
    for sample in samples:
        hidden = sample["hidden_states"]
        target = sample["target_logits"]

        # Add batch dimension
        hidden = hidden[None, :, :]  # [1, T, d_model]
        target = target[None, :, :]  # [1, T, vocab_size]

        loss, grads = loss_and_grad(head, hidden, target)
        optimizer.update(head, grads)
        mx.eval(head.parameters(), optimizer.state)

        total_loss += float(loss)
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


def evaluate_acceptance_rate(
    model,
    head: EarlyExitHead,
    tokenizer,
    test_texts: List[str],
    tgt_lang: str,
    exit_layer: int,
) -> Tuple[float, int, int]:
    """
    Evaluate token acceptance rate with trained head.

    Returns:
        (acceptance_rate, accepted_count, total_count)
    """
    accepted = 0
    total = 0

    for text in test_texts:
        input_text = f"<2{tgt_lang}> {text}"
        inputs = tokenizer(input_text, return_tensors="np")
        input_ids = mx.array(inputs["input_ids"])

        encoder_output = model.encode(input_ids)
        decoder_start_id = getattr(model.config, "decoder_start_token_id", 0)
        eos_token_id = tokenizer.eos_token_id

        generated = [decoder_start_id]
        cache = None

        for _ in range(64):  # Short sequences for eval
            decoder_ids = mx.array([[generated[-1]]])

            # Get early hidden and full logits
            early_hidden, full_logits, cache = decode_with_intermediate(
                model, decoder_ids, encoder_output, cache, exit_layer
            )
            mx.eval(early_hidden, full_logits, cache)

            # Predict with early exit head
            early_logits = head(early_hidden)
            mx.eval(early_logits)

            # Compare predictions
            early_token = int(mx.argmax(early_logits[0, -1]))
            full_token = int(mx.argmax(full_logits[0, -1]))

            if early_token == full_token:
                accepted += 1
            total += 1

            generated.append(full_token)
            if full_token == eos_token_id:
                break

    rate = accepted / total if total > 0 else 0.0
    return rate, accepted, total


def get_training_texts(lang_pair: str = "en-de", num_samples: int = 100) -> List[str]:
    """
    Get training texts for distillation.

    Uses diverse sentences to cover vocabulary well.
    """
    # A mix of sentence types for good coverage
    # These are English sentences to translate to German
    texts = [
        # Simple sentences
        "Hello, how are you today?",
        "The weather is nice.",
        "I would like a cup of coffee.",
        "What time is it?",
        "Thank you very much.",

        # Medium complexity
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming many industries.",
        "Climate change affects weather patterns globally.",
        "The meeting has been rescheduled to next week.",
        "Please review the attached document carefully.",

        # Technical content
        "Neural networks process information in layers.",
        "The algorithm achieved state-of-the-art results.",
        "Memory allocation is handled automatically.",
        "The database query returned no results.",
        "Server response time exceeded the threshold.",

        # Conversational
        "I'm looking forward to seeing you tomorrow.",
        "Could you please send me the report?",
        "That sounds like a great idea!",
        "I completely understand your concerns.",
        "Let me know if you need any help.",

        # News-like
        "The company announced record profits today.",
        "Scientists discovered a new species of fish.",
        "The election results will be announced tonight.",
        "Stock markets closed higher on positive news.",
        "The government proposed new tax reforms.",

        # Legal/Formal
        "All parties must comply with the agreement.",
        "The terms and conditions are subject to change.",
        "Unauthorized access is strictly prohibited.",
        "The contract expires at the end of the year.",
        "Please sign and return the enclosed forms.",

        # Literary
        "The sunset painted the sky in shades of gold.",
        "Memories of childhood lingered in his mind.",
        "The old house stood silent on the hill.",
        "Music filled the air on that summer evening.",
        "Dreams are the language of the soul.",

        # Questions
        "How can I improve my productivity?",
        "Where should we go for dinner tonight?",
        "Why did the system fail to respond?",
        "What are the main advantages of this approach?",
        "When will the project be completed?",

        # Instructions
        "First, open the application and log in.",
        "Enter your password and click submit.",
        "Wait for the process to complete.",
        "Review the output and verify the results.",
        "Save your work before closing the window.",

        # Numbers and dates
        "The meeting is scheduled for March 15th.",
        "Sales increased by 25% last quarter.",
        "The price is $99.99 plus tax.",
        "We need to order 500 units by Friday.",
        "The report covers the period from 2020 to 2023.",
    ]

    # Extend with variations if needed
    while len(texts) < num_samples:
        idx = random.randint(0, len(texts) - 1)
        texts.append(texts[idx])

    return texts[:num_samples]


def main():
    parser = argparse.ArgumentParser(description="Train Early Exit Head for Speculative Decoding")
    parser.add_argument("--model", default="google/madlad400-3b-mt", help="Model to train on")
    parser.add_argument("--exit-layer", type=int, default=4, help="Early exit layer (1-indexed)")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--temperature", type=float, default=2.0, help="Distillation temperature")
    parser.add_argument("--num-samples", type=int, default=50, help="Number of training samples")
    parser.add_argument("--tgt-lang", default="de", help="Target language for training data")
    parser.add_argument("--use-ffn", action="store_true", help="Use FFN in early exit head")
    parser.add_argument("--output-dir", default="models/early_exit_heads", help="Output directory")
    args = parser.parse_args()

    logger.info(f"Training early exit head for {args.model}")
    logger.info(f"Exit layer: {args.exit_layer}, Epochs: {args.epochs}, LR: {args.lr}")

    # Import converter
    from tools.pytorch_to_mlx.converters.madlad_converter import MADLADConverter

    # Load model
    logger.info("Loading model...")
    converter = MADLADConverter(model_path=args.model, dtype="bfloat16", quantize=8)
    converter.load()
    model = converter.model
    tokenizer = converter.tokenizer

    # Get model dimensions
    d_model = model.config.d_model
    vocab_size = model.config.vocab_size
    logger.info(f"Model dimensions: d_model={d_model}, vocab_size={vocab_size}")

    # Create early exit head
    head = EarlyExitHead(d_model, vocab_size, use_ffn=args.use_ffn)
    mx.eval(head.parameters())

    # Get training texts
    logger.info(f"Getting {args.num_samples} training texts...")
    train_texts = get_training_texts(num_samples=args.num_samples)
    test_texts = train_texts[:10]  # Use first 10 for eval

    # Collect training data
    logger.info("Collecting training data (this may take a while)...")
    start = time.time()
    samples = collect_training_data(
        model=model,
        tokenizer=tokenizer,
        texts=train_texts,
        tgt_lang=args.tgt_lang,
        exit_layer=args.exit_layer,
        max_tokens=64,
    )
    logger.info(f"Data collection took {time.time() - start:.1f}s")

    # Evaluate baseline acceptance rate (before training)
    logger.info("Evaluating baseline acceptance rate...")
    baseline_rate, _, _ = evaluate_acceptance_rate(
        model, head, tokenizer, test_texts[:5], args.tgt_lang, args.exit_layer
    )
    logger.info(f"Baseline acceptance rate (random init): {baseline_rate:.1%}")

    # Setup optimizer
    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=0.01)

    # Training loop
    logger.info("Starting training...")
    best_rate = baseline_rate
    best_epoch = 0

    for epoch in range(args.epochs):
        start = time.time()
        loss = train_epoch(head, optimizer, samples, temperature=args.temperature)
        epoch_time = time.time() - start

        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0:
            rate, accepted, total = evaluate_acceptance_rate(
                model, head, tokenizer, test_texts[:5], args.tgt_lang, args.exit_layer
            )
            logger.info(f"Epoch {epoch + 1}: loss={loss:.4f}, acceptance={rate:.1%} ({accepted}/{total}), time={epoch_time:.1f}s")

            if rate > best_rate:
                best_rate = rate
                best_epoch = epoch + 1
        else:
            logger.info(f"Epoch {epoch + 1}: loss={loss:.4f}, time={epoch_time:.1f}s")

    # Final evaluation
    logger.info("Final evaluation on test set...")
    final_rate, accepted, total = evaluate_acceptance_rate(
        model, head, tokenizer, test_texts, args.tgt_lang, args.exit_layer
    )
    logger.info(f"Final acceptance rate: {final_rate:.1%} ({accepted}/{total})")
    logger.info(f"Best rate: {best_rate:.1%} at epoch {best_epoch}")
    logger.info(f"Improvement from baseline: {baseline_rate:.1%} -> {final_rate:.1%}")

    # Save trained head
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = args.model.split("/")[-1]
    output_path = output_dir / f"{model_name}_exit{args.exit_layer}_head.safetensors"

    # Flatten parameters for saving
    params = dict(tree_flatten(head.parameters()))
    mx.save_safetensors(str(output_path), params)
    logger.info(f"Saved trained head to {output_path}")

    # Save config
    config = {
        "model": args.model,
        "exit_layer": args.exit_layer,
        "d_model": d_model,
        "vocab_size": vocab_size,
        "use_ffn": args.use_ffn,
        "epochs": args.epochs,
        "temperature": args.temperature,
        "final_acceptance_rate": float(final_rate),
        "baseline_acceptance_rate": float(baseline_rate),
    }
    config_path = output_dir / f"{model_name}_exit{args.exit_layer}_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved config to {config_path}")

    return final_rate


if __name__ == "__main__":
    main()
