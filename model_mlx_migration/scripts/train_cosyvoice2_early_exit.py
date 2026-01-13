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
Train Early Exit Head for CosyVoice2 Speculative Decoding (OPT-4)

Trains the early exit head to predict full model token distributions from early layer
hidden states. This enables high-acceptance-rate speculative decoding for the LLM
component, which is the main bottleneck (87% of CosyVoice2 inference time).

Expected speedup: 1.5-2x for decoder-only models (vs 0.92x for encoder-decoder T5)

Usage:
    python scripts/train_cosyvoice2_early_exit.py
    python scripts/train_cosyvoice2_early_exit.py --exit-layer 4 --epochs 20
    python scripts/train_cosyvoice2_early_exit.py --num-samples 50
"""

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Training texts - diverse sentences for speech synthesis
TRAINING_TEXTS = [
    "Hello, how are you today?",
    "The weather is nice outside.",
    "Please speak clearly and slowly.",
    "This is a test of the speech system.",
    "Good morning, welcome to our show.",
    "Thank you for your patience.",
    "The meeting starts at three o'clock.",
    "I need to make a phone call.",
    "Can you repeat that please?",
    "That sounds like a great idea.",
    "Let me think about it for a moment.",
    "The project deadline is next week.",
    "Would you like some coffee?",
    "The restaurant is around the corner.",
    "I'll send you an email later.",
    "She is an excellent speaker.",
    "We should discuss this further.",
    "The train arrives at six fifteen.",
    "Happy birthday to you!",
    "Congratulations on your promotion!",
    "I apologize for the confusion.",
    "Please take a seat over there.",
    "The book was very interesting.",
    "Can I help you with anything?",
    "It was nice meeting you.",
    "Have a wonderful day ahead.",
    "The music is too loud.",
    "I prefer tea over coffee.",
    "The flight has been delayed.",
    "Let's get started right away.",
]


def collect_training_data(
    model,
    texts: List[str],
    exit_layer: int,
    max_tokens: int = 100,
) -> List[Dict]:
    """
    Collect (early_hidden_states, full_model_logits) pairs for training.

    Runs the LLM and captures intermediate activations.
    """
    samples = []

    for i, text in enumerate(texts):
        if (i + 1) % 5 == 0:
            logger.info(f"Collecting data: {i + 1}/{len(texts)}")

        # Get text IDs from tokenizer
        text_ids = model.tokenizer.encode(text)
        if text_ids.ndim == 1:
            text_ids = text_ids[None, :]  # Add batch dimension

        # Process text through LLM to get initial hidden states
        llm = model.llm
        batch = text_ids.shape[0]

        # Initial forward pass to process text
        _, _, cache = llm(text_ids, cache=None, use_early_exit=False)
        mx.eval(cache)

        # Start generation with SOS token
        next_input = mx.zeros((batch, 1), dtype=mx.int32)

        # Collect hidden states and logits during decoding
        hidden_states_list = []
        full_logits_list = []

        for step in range(max_tokens):
            # Get full model output with all layers
            _, full_speech_logits, new_cache = llm(next_input, cache=cache, use_early_exit=False)
            mx.eval(full_speech_logits, new_cache)

            # Get early hidden states by running with stop_at_layer
            early_hidden, _ = llm.llm(
                next_input,
                cache=[c if c is not None else None for c in cache] if cache else None,
                stop_at_layer=exit_layer
            )
            mx.eval(early_hidden)

            # Store training pair
            hidden_states_list.append(early_hidden[0, -1])  # [hidden_size]
            full_logits_list.append(full_speech_logits[0, -1])  # [speech_vocab_size]

            # Sample next token (greedy for consistency)
            next_token = mx.argmax(full_speech_logits[0, -1])
            next_input = next_token[None, None]
            cache = new_cache
            mx.eval(next_input)

            # Check for EOS (token 0)
            if int(next_token) == 0:
                break

        if len(hidden_states_list) > 2:  # Need at least 3 tokens
            samples.append({
                "text": text,
                "hidden_states": mx.stack(hidden_states_list),  # [T, hidden_size]
                "target_logits": mx.stack(full_logits_list),    # [T, speech_vocab_size]
            })

    logger.info(f"Collected {len(samples)} training samples")
    return samples


def log_softmax(x: mx.array, axis: int = -1) -> mx.array:
    """Compute log softmax in a numerically stable way."""
    max_x = mx.max(x, axis=axis, keepdims=True)
    shifted = x - max_x
    return shifted - mx.log(mx.sum(mx.exp(shifted), axis=axis, keepdims=True))


def kl_divergence_loss(
    pred_logits: mx.array,
    target_logits: mx.array,
    temperature: float = 2.0,
) -> mx.array:
    """
    KL divergence loss with temperature scaling.

    Uses soft labels from the full model for knowledge distillation.
    """
    # Apply temperature scaling
    pred_scaled = pred_logits / temperature
    target_scaled = target_logits / temperature

    # Compute KL divergence: sum_i p_i * (log(p_i) - log(q_i))
    # Where p = softmax(target), q = softmax(pred)
    target_probs = mx.softmax(target_scaled, axis=-1)
    log_pred = log_softmax(pred_scaled, axis=-1)
    log_target = log_softmax(target_scaled, axis=-1)

    kl = mx.sum(target_probs * (log_target - log_pred), axis=-1)

    return mx.mean(kl) * (temperature ** 2)


def train_epoch(
    head: nn.Module,
    optimizer: optim.Optimizer,
    samples: List[Dict],
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

    for sample in samples:
        hidden = sample["hidden_states"]
        target = sample["target_logits"]

        # Add batch dimension
        hidden = hidden[None, :, :]  # [1, T, hidden_size]
        target = target[None, :, :]  # [1, T, speech_vocab_size]

        loss, grads = loss_and_grad(head, hidden, target)
        optimizer.update(head, grads)
        mx.eval(head.parameters(), optimizer.state)

        total_loss += float(loss)
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


def evaluate_acceptance_rate(
    model,
    exit_layer: int,
    test_texts: List[str],
    max_tokens: int = 50,
) -> Tuple[float, int, int]:
    """
    Evaluate token acceptance rate with the current early exit head.

    Returns:
        (acceptance_rate, accepted_count, total_count)
    """
    accepted = 0
    total = 0
    llm = model.llm

    for text in test_texts[:5]:  # Use subset for faster eval
        text_ids = model.tokenizer.encode(text)
        if text_ids.ndim == 1:
            text_ids = text_ids[None, :]

        batch = text_ids.shape[0]

        # Process text
        _, _, cache = llm(text_ids, cache=None, use_early_exit=False)
        mx.eval(cache)

        next_input = mx.zeros((batch, 1), dtype=mx.int32)

        for _ in range(max_tokens):
            # Get both early and full predictions
            _, early_logits, _ = llm(next_input, cache=cache, use_early_exit=True)
            _, full_logits, cache = llm(next_input, cache=cache, use_early_exit=False)
            mx.eval(early_logits, full_logits, cache)

            # Compare predictions
            early_token = int(mx.argmax(early_logits[0, -1]))
            full_token = int(mx.argmax(full_logits[0, -1]))

            total += 1
            if early_token == full_token:
                accepted += 1

            # Use full token for next step
            next_input = mx.array([[full_token]])

            if full_token == 0:  # EOS
                break

    rate = accepted / total if total > 0 else 0.0
    return rate, accepted, total


def main():
    parser = argparse.ArgumentParser(description="Train CosyVoice2 Early Exit Head")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to CosyVoice2 model (default: auto-detect)",
    )
    parser.add_argument(
        "--exit-layer",
        type=int,
        default=6,
        help="Layer to use for early exit (default: 6 of 24)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of training epochs (default: 15)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=30,
        help="Number of training samples (default: 30)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=2.0,
        help="Temperature for knowledge distillation (default: 2.0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/early_exit_heads",
        help="Output directory for trained head",
    )
    args = parser.parse_args()

    from tools.pytorch_to_mlx.converters.models import CosyVoice2Model

    # Load model
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        model_path = CosyVoice2Model.get_default_model_path()

    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        return 1

    logger.info(f"Loading CosyVoice2 from {model_path}")
    model = CosyVoice2Model.from_pretrained(model_path)

    # Get model dimensions
    llm = model.llm
    hidden_size = llm.config.hidden_size
    speech_vocab_size = llm.config.speech_vocab_size
    num_layers = llm.config.num_hidden_layers

    logger.info(f"Model config: hidden_size={hidden_size}, vocab={speech_vocab_size}, layers={num_layers}")
    logger.info(f"Using early exit at layer {args.exit_layer}/{num_layers}")

    # Update model's early exit layer
    llm.early_exit_layer = args.exit_layer

    # Prepare training data
    texts = TRAINING_TEXTS[:args.num_samples]
    logger.info(f"Collecting training data from {len(texts)} texts...")

    start_time = time.time()
    samples = collect_training_data(model, texts, args.exit_layer, max_tokens=100)
    collect_time = time.time() - start_time
    logger.info(f"Data collection took {collect_time:.1f}s")

    if not samples:
        logger.error("No training samples collected!")
        return 1

    # Initialize early exit head from llm_decoder
    logger.info("Initializing early exit head from llm_decoder...")
    llm.initialize_early_exit_head()

    # Check initial acceptance rate
    logger.info("Evaluating initial acceptance rate...")
    init_rate, init_acc, init_total = evaluate_acceptance_rate(
        model, args.exit_layer, texts[:5]
    )
    logger.info(f"Initial acceptance rate: {init_rate:.1%} ({init_acc}/{init_total})")

    # Set up optimizer for early exit head only
    optimizer = optim.Adam(learning_rate=args.lr)

    # Training loop
    logger.info(f"Training for {args.epochs} epochs...")
    best_rate = init_rate

    for epoch in range(args.epochs):
        loss = train_epoch(llm.early_exit_head, optimizer, samples, args.temperature)

        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            rate, acc, total = evaluate_acceptance_rate(model, args.exit_layer, texts[:5])
            logger.info(
                f"Epoch {epoch + 1:3d}/{args.epochs} | "
                f"Loss: {loss:.4f} | "
                f"Acceptance: {rate:.1%} ({acc}/{total})"
            )

            if rate > best_rate:
                best_rate = rate
                logger.info(f"  -> New best! Rate: {best_rate:.1%}")
        else:
            logger.info(f"Epoch {epoch + 1:3d}/{args.epochs} | Loss: {loss:.4f}")

    # Save trained head
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    head_path = output_dir / f"cosyvoice2_exit{args.exit_layer}_head.safetensors"
    config_path = output_dir / f"cosyvoice2_exit{args.exit_layer}_config.json"

    # Save weights
    weights = dict(llm.early_exit_head.parameters())
    flat_weights = dict(tree_flatten(weights))
    mx.savez(str(head_path).replace(".safetensors", ".npz"), **flat_weights)

    # Save config
    config = {
        "model": "cosyvoice2",
        "exit_layer": args.exit_layer,
        "hidden_size": hidden_size,
        "speech_vocab_size": speech_vocab_size,
        "initial_acceptance_rate": init_rate,
        "final_acceptance_rate": best_rate,
        "epochs": args.epochs,
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Saved early exit head to {head_path}")
    logger.info(f"Acceptance rate improved: {init_rate:.1%} -> {best_rate:.1%}")

    # Final summary
    improvement = (best_rate - init_rate) / init_rate * 100 if init_rate > 0 else 0
    logger.info(f"\n{'='*60}")
    logger.info("Training Complete!")
    logger.info(f"{'='*60}")
    logger.info(f"Exit layer: {args.exit_layer}/{num_layers}")
    logger.info(f"Initial acceptance: {init_rate:.1%}")
    logger.info(f"Final acceptance: {best_rate:.1%}")
    logger.info(f"Improvement: {improvement:+.1f}%")
    logger.info(f"Output: {head_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
