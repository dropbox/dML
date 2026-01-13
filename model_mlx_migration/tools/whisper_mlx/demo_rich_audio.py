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
Rich Audio Understanding Demo - Terminal Visualization.

Real-time demo showing streaming rich audio understanding:
- Live transcription with confidence scores
- Emotion detection (8 RAVDESS classes)
- Pitch tracking (F0 Hz)
- Paralinguistics detection (laughter, cough, fillers)
- Phoneme output

Usage:
    # Process a single audio file with visualization
    python -m tools.whisper_mlx.demo_rich_audio data/LibriSpeech/dev-clean/1272/128104/1272-128104-0000.flac

    # Process multiple files
    python -m tools.whisper_mlx.demo_rich_audio audio1.wav audio2.wav

    # Batch mode (no visualization, just results)
    python -m tools.whisper_mlx.demo_rich_audio --batch data/*.wav

    # Use specific emotion sample
    python -m tools.whisper_mlx.demo_rich_audio data/emotion/RAVDESS/Actor_01/03-01-01-01-01-01-01.wav

The demo shows:
- Real-time transcription with word-level confidence
- Dominant emotion per segment
- Pitch contour (average F0)
- Paralinguistic events (laughter, fillers, etc.)
- Per-frame processing rate

Terminal visualization uses ANSI colors for:
- Green: High confidence (>0.8)
- Yellow: Medium confidence (0.5-0.8)
- Red: Low confidence (<0.5)
- Cyan: Emotion
- Magenta: Pitch
- Blue: Paralinguistics
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# =============================================================================
# ANSI Color Codes for Terminal
# =============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright foreground colors
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"

    # Background colors
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"

    @staticmethod
    def confidence_color(confidence: float) -> str:
        """Get color based on confidence score."""
        if confidence >= 0.8:
            return Colors.BRIGHT_GREEN
        if confidence >= 0.5:
            return Colors.YELLOW
        return Colors.RED


# =============================================================================
# Emotion emoji mapping
# =============================================================================

EMOTION_DISPLAY = {
    "neutral": ("Neutral", "😐"),
    "calm": ("Calm", "😌"),
    "happy": ("Happy", "😊"),
    "sad": ("Sad", "😢"),
    "angry": ("Angry", "😠"),
    "fearful": ("Fear", "😨"),
    "disgust": ("Disgust", "🤢"),
    "surprised": ("Surprised", "😲"),
}


# =============================================================================
# Paralinguistics display
# =============================================================================

PARA_DISPLAY = {
    "speech": ("Speech", "💬"),
    "laughter": ("Laugh", "😂"),
    "cough": ("Cough", "🤧"),
    "sigh": ("Sigh", "😮‍💨"),
    "breath": ("Breath", "💨"),
    "cry": ("Cry", "😭"),
    "yawn": ("Yawn", "🥱"),
    "throat_clear": ("Clear", "🗣️"),
    "sneeze": ("Sneeze", "🤧"),
    "gasp": ("Gasp", "😯"),
    "groan": ("Groan", "😩"),
    "um_en": ("Um", "🤔"),
    "uh_en": ("Uh", "🤔"),
    "hmm_en": ("Hmm", "🤔"),
    "er_en": ("Er", "🤔"),
    "ah_en": ("Ah", "🤔"),
}


# =============================================================================
# Demo Results
# =============================================================================

@dataclass
class DemoResult:
    """Results from processing a single audio file."""
    audio_path: str
    duration_s: float

    # Transcription
    text: str
    mean_confidence: float

    # Emotion (dominant)
    emotion: str
    emotion_confidence: float
    emotion_distribution: dict[str, float]

    # Pitch
    mean_pitch_hz: float
    pitch_range: tuple[float, float]  # (min, max)
    voiced_ratio: float  # Fraction of frames with detected pitch

    # Paralinguistics
    para_events: list[tuple[str, float, float]]  # [(class, start_s, end_s), ...]
    dominant_para: str

    # Processing metrics
    total_latency_ms: float
    rtf: float


# =============================================================================
# Terminal Visualization
# =============================================================================

def print_header():
    """Print demo header."""
    print()
    print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}╔══════════════════════════════════════════════════════════════════════╗{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}║           Rich Audio Understanding Demo - Phase 9                    ║{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}║         Text + Emotion + Pitch + Paralinguistics                     ║{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}╚══════════════════════════════════════════════════════════════════════╝{Colors.RESET}")
    print()


def print_audio_info(path: str, duration_s: float, sample_rate: int):
    """Print audio file information."""
    print(f"{Colors.BOLD}Audio:{Colors.RESET} {Path(path).name}")
    print(f"  Duration: {duration_s:.2f}s ({int(duration_s * sample_rate):,} samples @ {sample_rate}Hz)")
    print()


def print_transcription(text: str, confidence: float):
    """Print transcription with confidence coloring."""
    color = Colors.confidence_color(confidence)
    print(f"{Colors.BOLD}Transcription:{Colors.RESET}")
    print(f"  {color}{text}{Colors.RESET}")
    print(f"  {Colors.DIM}Confidence: {confidence:.1%}{Colors.RESET}")
    print()


def print_emotion(emotion: str, confidence: float, distribution: dict[str, float]):
    """Print emotion detection results."""
    name, emoji = EMOTION_DISPLAY.get(emotion, (emotion, "❓"))

    print(f"{Colors.BOLD}{Colors.CYAN}Emotion:{Colors.RESET}")
    print(f"  Dominant: {emoji} {name} ({confidence:.1%})")

    # Show top 3 emotions
    sorted_emotions = sorted(distribution.items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"  {Colors.DIM}Distribution:{Colors.RESET}")
    for emo, prob in sorted_emotions:
        bar_len = int(prob * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        emo_name = EMOTION_DISPLAY.get(emo, (emo, ""))[0]
        print(f"    {emo_name:12s} {Colors.CYAN}{bar}{Colors.RESET} {prob:.1%}")
    print()


def print_pitch(mean_hz: float, pitch_range: tuple[float, float], voiced_ratio: float):
    """Print pitch detection results."""
    print(f"{Colors.BOLD}{Colors.MAGENTA}Pitch:{Colors.RESET}")

    if mean_hz > 0:
        # Convert Hz to musical note approximation
        note = hz_to_note(mean_hz)
        print(f"  Mean F0: {mean_hz:.1f} Hz ({note})")
        print(f"  Range: {pitch_range[0]:.1f} - {pitch_range[1]:.1f} Hz")
        print(f"  Voiced: {voiced_ratio:.1%} of frames")
    else:
        print(f"  {Colors.DIM}No voiced speech detected{Colors.RESET}")
    print()


def hz_to_note(hz: float) -> str:
    """Convert frequency to musical note name."""
    if hz <= 0:
        return "?"

    # A4 = 440 Hz
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    semitones_from_a4 = 12 * np.log2(hz / 440.0)
    note_index = int(round(semitones_from_a4)) + 9  # A is at index 9
    octave = 4 + (note_index + 3) // 12
    note_name = notes[note_index % 12]

    return f"{note_name}{octave}"


def print_paralinguistics(events: list[tuple[str, float, float]], dominant: str):
    """Print paralinguistics detection results."""
    print(f"{Colors.BOLD}{Colors.BLUE}Paralinguistics:{Colors.RESET}")

    if events:
        name, emoji = PARA_DISPLAY.get(dominant, (dominant, "❓"))
        print(f"  Dominant: {emoji} {name}")

        if len(events) > 0:
            print(f"  {Colors.DIM}Events:{Colors.RESET}")
            for cls, start, end in events[:5]:  # Show first 5
                cls_name, cls_emoji = PARA_DISPLAY.get(cls, (cls, ""))
                print(f"    {cls_emoji} {cls_name} @ {start:.2f}s - {end:.2f}s")
    else:
        print(f"  {Colors.DIM}Pure speech detected{Colors.RESET}")
    print()


def print_performance(latency_ms: float, rtf: float, duration_s: float):
    """Print performance metrics."""
    print(f"{Colors.BOLD}Performance:{Colors.RESET}")
    print(f"  Total latency: {latency_ms:.1f}ms")
    print(f"  RTF: {rtf:.3f} ({1/rtf:.1f}x real-time)")
    print(f"  Throughput: {duration_s / (latency_ms / 1000):.1f} audio-sec/wall-sec")
    print()


def print_separator():
    """Print section separator."""
    print(f"{Colors.DIM}{'─' * 70}{Colors.RESET}")


# =============================================================================
# Audio Processing
# =============================================================================

def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Load audio file and resample to 16kHz if needed."""
    import soundfile as sf

    audio, sr = sf.read(path)

    # Convert stereo to mono
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Resample to 16kHz if needed
    if sr != 16000:
        from scipy import signal
        audio = signal.resample(audio, int(len(audio) * 16000 / sr))
        sr = 16000

    return audio.astype(np.float32), sr


def process_audio(
    audio: np.ndarray,
    sr: int,
    whisper_model,
    rich_head,
    tokenizer,
) -> DemoResult:
    """Process audio through rich audio understanding pipeline."""
    import mlx.core as mx

    from .audio import log_mel_spectrogram
    from .rich_ctc_head import EMOTION_CLASSES_8, PARA_CLASSES_INV

    audio_duration_s = len(audio) / sr

    # Mel spectrogram
    t0 = time.time()
    mel = log_mel_spectrogram(audio, n_mels=whisper_model.config.n_mels)
    mel_tensor = mx.expand_dims(mx.array(mel), axis=0)
    mel_time = time.time() - t0

    # Encoder
    t0 = time.time()
    encoder_output = whisper_model.encoder(mel_tensor, variable_length=True)
    mx.eval(encoder_output)
    encoder_time = time.time() - t0

    # Rich CTC Head
    t0 = time.time()
    outputs = rich_head(encoder_output)
    mx.eval(outputs)
    head_time = time.time() - t0

    # Decode text
    t0 = time.time()
    tokens, tokens_with_timing = rich_head.decode_text_greedy(outputs)
    text = tokenizer.decode(tokens)
    decode_time = time.time() - t0

    # Compute confidence from softmax of text logits
    text_logits = outputs["text_logits"]
    if text_logits.ndim == 3:
        text_logits = text_logits[0]
    text_probs = mx.softmax(text_logits, axis=-1)
    max_probs = mx.max(text_probs, axis=-1)
    mean_confidence = float(np.mean(np.array(max_probs)))

    # Extract emotion
    emotion_logits = outputs.get("emotion")
    if emotion_logits is not None:
        emotion_probs = mx.softmax(emotion_logits, axis=-1)
        emotion_probs_np = np.array(emotion_probs[0])  # (T, num_emotions)

        # Average over time
        avg_probs = emotion_probs_np.mean(axis=0)

        # Get emotion classes
        num_emotions = avg_probs.shape[0]
        if num_emotions <= 8:
            emotion_classes = EMOTION_CLASSES_8[:num_emotions]
        else:
            from .rich_ctc_head import EMOTION_CLASSES_34
            emotion_classes = EMOTION_CLASSES_34[:num_emotions]

        dominant_idx = int(np.argmax(avg_probs))
        dominant_emotion = emotion_classes[dominant_idx]
        emotion_confidence = float(avg_probs[dominant_idx])

        emotion_distribution = {
            cls: float(avg_probs[i])
            for i, cls in enumerate(emotion_classes)
        }
    else:
        dominant_emotion = "unknown"
        emotion_confidence = 0.0
        emotion_distribution = {}

    # Extract pitch
    pitch_hz = outputs.get("pitch_hz")
    if pitch_hz is not None:
        pitch_np = np.array(pitch_hz[0, :, 0])  # (T,)

        # Filter voiced frames (pitch > 50 Hz)
        voiced_mask = pitch_np > 50
        voiced_frames = pitch_np[voiced_mask]

        if len(voiced_frames) > 0:
            mean_pitch = float(np.mean(voiced_frames))
            pitch_range = (float(np.min(voiced_frames)), float(np.max(voiced_frames)))
            voiced_ratio = float(np.sum(voiced_mask)) / len(pitch_np)
        else:
            mean_pitch = 0.0
            pitch_range = (0.0, 0.0)
            voiced_ratio = 0.0
    else:
        mean_pitch = 0.0
        pitch_range = (0.0, 0.0)
        voiced_ratio = 0.0

    # Extract paralinguistics
    para_logits = outputs.get("para")
    if para_logits is not None:
        para_probs = mx.softmax(para_logits, axis=-1)
        para_np = np.array(para_probs[0])  # (T, 50)

        # Get predicted class per frame
        para_classes = np.argmax(para_np, axis=-1)

        # Find non-speech segments (class != 0 which is "speech")
        para_events = []
        current_class = None
        start_frame = 0

        for i, cls in enumerate(para_classes):
            if cls != current_class:
                if current_class is not None and current_class != 0:
                    # End of non-speech segment
                    cls_name = PARA_CLASSES_INV.get(current_class, f"class_{current_class}")
                    start_s = start_frame * 0.02  # 50Hz = 20ms per frame
                    end_s = i * 0.02
                    if end_s - start_s > 0.1:  # Only show events > 100ms
                        para_events.append((cls_name, start_s, end_s))
                current_class = cls
                start_frame = i

        # Handle last segment
        if current_class is not None and current_class != 0:
            cls_name = PARA_CLASSES_INV.get(current_class, f"class_{current_class}")
            start_s = start_frame * 0.02
            end_s = len(para_classes) * 0.02
            if end_s - start_s > 0.1:
                para_events.append((cls_name, start_s, end_s))

        # Dominant paralinguistic (excluding speech)
        non_speech_mask = para_classes != 0
        if np.any(non_speech_mask):
            non_speech_classes = para_classes[non_speech_mask]
            unique, counts = np.unique(non_speech_classes, return_counts=True)
            dominant_para_idx = unique[np.argmax(counts)]
            dominant_para = PARA_CLASSES_INV.get(int(dominant_para_idx), "unknown")
        else:
            dominant_para = "speech"
    else:
        para_events = []
        dominant_para = "unknown"

    total_time_ms = (mel_time + encoder_time + head_time + decode_time) * 1000
    rtf = (total_time_ms / 1000) / audio_duration_s

    return DemoResult(
        audio_path="",
        duration_s=audio_duration_s,
        text=text,
        mean_confidence=mean_confidence,
        emotion=dominant_emotion,
        emotion_confidence=emotion_confidence,
        emotion_distribution=emotion_distribution,
        mean_pitch_hz=mean_pitch,
        pitch_range=pitch_range,
        voiced_ratio=voiced_ratio,
        para_events=para_events,
        dominant_para=dominant_para,
        total_latency_ms=total_time_ms,
        rtf=rtf,
    )


# =============================================================================
# Model Loading
# =============================================================================

def load_models(verbose: bool = True):
    """Load Whisper encoder and RichCTCHead."""
    if verbose:
        print(f"{Colors.DIM}Loading models...{Colors.RESET}")

    t0 = time.time()

    # Import MLX modules
    from .model import WhisperMLX
    from .rich_ctc_head import RichCTCHead
    from .tokenizer import get_whisper_tokenizer

    # Load Whisper model (uses cached HuggingFace model)
    if verbose:
        print(f"{Colors.DIM}  Loading Whisper large-v3...{Colors.RESET}")
    whisper_model = WhisperMLX.from_pretrained("large-v3")
    whisper_model.eval()

    # Load RichCTCHead with trained weights
    if verbose:
        print(f"{Colors.DIM}  Loading RichCTCHead...{Colors.RESET}")
    rich_head = RichCTCHead.from_pretrained()

    # Load tokenizer
    if verbose:
        print(f"{Colors.DIM}  Loading tokenizer...{Colors.RESET}")
    tokenizer = get_whisper_tokenizer(multilingual=True, task="transcribe")

    load_time = time.time() - t0

    if verbose:
        print(f"{Colors.DIM}Models loaded in {load_time:.2f}s{Colors.RESET}")
        print()

    return whisper_model, rich_head, tokenizer


# =============================================================================
# Main Demo
# =============================================================================

def run_demo(
    audio_paths: list[str],
    batch_mode: bool = False,
    verbose: bool = True,
):
    """Run the demo on audio files."""
    # Load models
    whisper_model, rich_head, tokenizer = load_models(verbose=not batch_mode)

    if not batch_mode:
        print_header()

    results = []

    for audio_path in audio_paths:
        if not os.path.exists(audio_path):
            print(f"{Colors.RED}Error: File not found: {audio_path}{Colors.RESET}")
            continue

        try:
            # Load audio
            audio, sr = load_audio(audio_path)
            duration_s = len(audio) / sr

            if not batch_mode:
                print_audio_info(audio_path, duration_s, sr)

            # Process
            result = process_audio(audio, sr, whisper_model, rich_head, tokenizer)
            result.audio_path = audio_path
            results.append(result)

            if not batch_mode:
                # Visualize results
                print_transcription(result.text, result.mean_confidence)
                print_emotion(result.emotion, result.emotion_confidence, result.emotion_distribution)
                print_pitch(result.mean_pitch_hz, result.pitch_range, result.voiced_ratio)
                print_paralinguistics(result.para_events, result.dominant_para)
                print_performance(result.total_latency_ms, result.rtf, result.duration_s)
                print_separator()
            else:
                # Batch mode: compact output
                print(f"{Path(audio_path).name}: \"{result.text[:50]}...\" "
                      f"[{result.emotion}] pitch={result.mean_pitch_hz:.0f}Hz "
                      f"RTF={result.rtf:.3f}")

        except Exception as e:
            print(f"{Colors.RED}Error processing {audio_path}: {e}{Colors.RESET}")
            if verbose:
                import traceback
                traceback.print_exc()

    # Summary
    if len(results) > 1 and not batch_mode:
        print()
        print(f"{Colors.BOLD}Summary ({len(results)} files):{Colors.RESET}")
        total_audio = sum(r.duration_s for r in results)
        total_latency = sum(r.total_latency_ms for r in results)
        avg_rtf = np.mean([r.rtf for r in results])
        print(f"  Total audio: {total_audio:.1f}s")
        print(f"  Total processing: {total_latency:.1f}ms")
        print(f"  Average RTF: {avg_rtf:.3f}")
        print()

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Rich Audio Understanding Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single file
    python -m tools.whisper_mlx.demo_rich_audio audio.wav

    # Multiple files
    python -m tools.whisper_mlx.demo_rich_audio *.wav

    # Batch mode (compact output)
    python -m tools.whisper_mlx.demo_rich_audio --batch data/LibriSpeech/dev-clean/1272/128104/*.flac
        """,
    )
    parser.add_argument(
        "audio_files",
        nargs="+",
        help="Audio file(s) to process",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch mode: compact output without visualization",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages",
    )

    args = parser.parse_args()

    run_demo(
        audio_paths=args.audio_files,
        batch_mode=args.batch,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
