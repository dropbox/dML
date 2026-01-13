#!/usr/bin/env python3
"""
Profile CTC training to identify bottlenecks.

Usage:
    python -m scripts.profile_ctc_training \
        --data-dir data/LibriSpeech_full \
        --checkpoint checkpoints/ctc_english_full/step_42500.npz \
        --num-steps 5
"""

import argparse
import time
from pathlib import Path
from typing import List, Dict
import random

import numpy as np
import mlx.core as mx

# Import training components
from tools.whisper_mlx.audio import load_audio, log_mel_spectrogram
from tools.whisper_mlx.model import WhisperMLX
from tools.whisper_mlx.ctc_head import CTCDraftHead

HAS_TORCH = False
try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    pass


def spec_augment(
    mel: mx.array,
    freq_mask_param: int = 27,
    time_mask_param: int = 100,
    num_freq_masks: int = 2,
    num_time_masks: int = 2,
) -> mx.array:
    """Apply SpecAugment data augmentation."""
    T, F_dim = mel.shape
    mel_np = np.array(mel)

    # Frequency masking
    for _ in range(num_freq_masks):
        f = random.randint(0, min(freq_mask_param, F_dim))
        f0 = random.randint(0, F_dim - f)
        mel_np[:, f0:f0 + f] = 0

    # Time masking
    for _ in range(num_time_masks):
        t = random.randint(0, min(time_mask_param, T))
        t0 = random.randint(0, T - t)
        mel_np[t0:t0 + t, :] = 0

    return mx.array(mel_np)


class TimingProfiler:
    """Simple profiler to measure time spent in each component."""

    def __init__(self):
        self.timings: Dict[str, List[float]] = {}

    def time(self, name: str):
        """Context manager to time a block."""
        return _TimingContext(self, name)

    def record(self, name: str, duration: float):
        if name not in self.timings:
            self.timings[name] = []
        self.timings[name].append(duration)

    def report(self):
        """Print timing report."""
        print("\n" + "=" * 70)
        print("TIMING PROFILE")
        print("=" * 70)

        total = sum(sum(v) for v in self.timings.values())

        # Sort by total time
        sorted_items = sorted(
            self.timings.items(),
            key=lambda x: sum(x[1]),
            reverse=True
        )

        for name, times in sorted_items:
            total_time = sum(times)
            avg_time = total_time / len(times)
            pct = 100 * total_time / total if total > 0 else 0
            print(f"{name:40s}: {avg_time*1000:8.2f}ms avg | {total_time:8.3f}s total | {pct:5.1f}%")

        print("-" * 70)
        print(f"{'TOTAL':40s}: {total:8.3f}s")
        print("=" * 70)


class _TimingContext:
    def __init__(self, profiler: TimingProfiler, name: str):
        self.profiler = profiler
        self.name = name
        self.start = None

    def __enter__(self):
        mx.synchronize()  # Ensure MLX operations are complete
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        mx.synchronize()  # Ensure MLX operations are complete
        duration = time.perf_counter() - self.start
        self.profiler.record(self.name, duration)


def find_audio_samples(data_dir: Path, limit: int = 100):
    """Find audio samples in LibriSpeech format."""
    import subprocess
    samples = []

    # Use find command to follow symlinks
    result = subprocess.run(
        ["find", "-L", str(data_dir), "-name", "*.trans.txt"],
        capture_output=True, text=True
    )

    trans_files = result.stdout.strip().split("\n")

    for trans_file_str in trans_files:
        if not trans_file_str:
            continue
        trans_file = Path(trans_file_str)
        try:
            with open(trans_file) as f:
                for line in f:
                    parts = line.strip().split(" ", 1)
                    if len(parts) == 2:
                        file_id, transcript = parts
                        audio_path = trans_file.parent / f"{file_id}.flac"
                        if audio_path.exists():
                            samples.append({
                                "audio_path": str(audio_path),
                                "transcript": transcript
                            })
                            if len(samples) >= limit:
                                return samples
        except Exception:
            continue

    return samples


def profile_training(
    data_dir: Path,
    checkpoint_path: Path,
    num_steps: int = 5,
    batch_size: int = 4,
):
    """Profile a few training steps."""

    profiler = TimingProfiler()

    # Load model
    print("Loading Whisper large-v3 model...")
    with profiler.time("1_model_load"):
        whisper_model = WhisperMLX.from_pretrained(
            "large-v3",
            warmup=False,  # Skip warmup for profiling
        )
        mx.eval(whisper_model.parameters())

    # Load CTC head
    print(f"Loading CTC head from {checkpoint_path}...")
    with profiler.time("2_ctc_head_load"):
        # large-v3 has d_model=1280 (n_audio_state)
        ctc_head = CTCDraftHead(
            d_model=whisper_model.config.n_audio_state,
            vocab_size=51865,
            use_layer_norm=True,
        )
        checkpoint = mx.load(str(checkpoint_path))
        ctc_params = {k.replace("ctc_head.", ""): v for k, v in checkpoint.items() if k.startswith("ctc_head.")}
        ctc_head.update(ctc_params)
        mx.eval(ctc_head.parameters())

    # Find samples
    print(f"Finding audio samples in {data_dir}...")
    samples = find_audio_samples(data_dir, limit=batch_size * num_steps * 2)
    random.shuffle(samples)
    print(f"Found {len(samples)} samples")

    # Profile training steps
    print(f"\nProfiling {num_steps} training steps with batch_size={batch_size}...")

    for step in range(num_steps):
        batch_samples = samples[step * batch_size:(step + 1) * batch_size]

        print(f"\n--- Step {step + 1}/{num_steps} ---")

        # 1. Load audio files
        audios = []
        with profiler.time("3_audio_load"):
            for sample in batch_samples:
                audio = load_audio(sample["audio_path"])
                audios.append(audio)

        # 2. Compute mel spectrograms
        mel_specs = []
        with profiler.time("4_mel_spectrogram"):
            for audio in audios:
                # Truncate to 30s max
                max_samples = int(30 * 16000)
                if len(audio) > max_samples:
                    audio = audio[:max_samples]
                mel = log_mel_spectrogram(audio, n_mels=128)
                mel_specs.append(mel)

        # 3. Apply SpecAugment
        augmented_mels = []
        with profiler.time("5_spec_augment"):
            for mel in mel_specs:
                mel_mx = mx.array(mel)
                mel_aug = spec_augment(mel_mx)
                augmented_mels.append(mel_aug)

        # 4. Pad to 3000 frames
        with profiler.time("6_padding"):
            padded_mels = []
            actual_frames = []
            for mel in augmented_mels:
                actual = mel.shape[0]
                actual_frames.append(actual)
                if mel.shape[0] < 3000:
                    mel = mx.pad(mel, [(0, 3000 - mel.shape[0]), (0, 0)])
                else:
                    mel = mel[:3000]
                padded_mels.append(mel)
            mel_batch = mx.stack(padded_mels)

        # 5. Encoder forward pass
        with profiler.time("7_encoder_forward"):
            encoder_output = whisper_model.encoder(mel_batch)
            mx.eval(encoder_output)

        # 6. CTC head forward pass
        with profiler.time("8_ctc_head_forward"):
            logits = ctc_head(encoder_output)
            mx.eval(logits)

        # 7. Convert to PyTorch for CTC loss
        if HAS_TORCH:
            with profiler.time("9_mlx_to_numpy"):
                max_input_len = max(f // 2 for f in actual_frames)
                logits_trimmed = logits[:, :max_input_len, :]
                mx.eval(logits_trimmed)
                logits_np = np.array(logits_trimmed)

            with profiler.time("10_numpy_to_torch"):
                logits_torch = torch.from_numpy(logits_np).float().requires_grad_(True)

            # 8. Compute CTC loss in PyTorch
            with profiler.time("11_pytorch_ctc_loss"):
                log_probs = torch.nn.functional.log_softmax(logits_torch, dim=-1)
                log_probs_t = log_probs.transpose(0, 1).contiguous()

                # Simple tokenization (just for profiling)
                target_tokens = [[50257, 50258, 50259, 50360, 50261] for _ in batch_samples]
                flat_targets = []
                target_lengths = []
                for t in target_tokens:
                    flat_targets.extend(t)
                    target_lengths.append(len(t))

                input_lengths = [f // 2 for f in actual_frames]

                targets_torch = torch.tensor(flat_targets, dtype=torch.long)
                input_lengths_torch = torch.tensor(input_lengths, dtype=torch.long)
                target_lengths_torch = torch.tensor(target_lengths, dtype=torch.long)

                loss = F.ctc_loss(
                    log_probs_t,
                    targets_torch,
                    input_lengths_torch,
                    target_lengths_torch,
                    blank=0,
                    reduction="mean",
                    zero_infinity=True,
                )

            # 9. Backward pass
            with profiler.time("12_pytorch_backward"):
                loss.backward()

            # 10. Get gradients back
            with profiler.time("13_grad_to_mlx"):
                logits_grad_np = logits_torch.grad.numpy()
                logits_grad_mx = mx.array(logits_grad_np)
                mx.eval(logits_grad_mx)

            print(f"  Loss: {loss.item():.4f}")

        # Clean up
        del mel_batch, encoder_output, logits
        if HAS_TORCH:
            del logits_torch, logits_grad_mx

    # Print report
    profiler.report()


def main():
    parser = argparse.ArgumentParser(description="Profile CTC training")
    parser.add_argument("--data-dir", type=Path, default=Path("data/LibriSpeech_full"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/ctc_english_full/step_42500.npz"))
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)

    args = parser.parse_args()

    profile_training(
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
