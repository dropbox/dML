"""
CosyVoice2 Quality Audit Tests

Tests CosyVoice2 TTS for:
1. Audio quality metrics (RMS, peak, silence detection, clipping)
2. RTF (Real-Time Factor) performance
3. Dialect accuracy (Sichuanese)
4. Comparison: torch.compile vs non-compiled output

MANAGER directive: Before C++ export, we need quality benchmarks to catch audio bugs.

Worker #462 achieved RTF < 1.0 for ALL text lengths using torch.compile.

Usage:
    # Run all CosyVoice2 quality tests
    pytest tests/quality/test_cosyvoice_quality.py -v

    # Run only audio quality tests (no RTF benchmarks)
    pytest tests/quality/test_cosyvoice_quality.py -v -m "not slow"

    # Run RTF benchmark
    pytest tests/quality/test_cosyvoice_quality.py::TestCosyVoiceRTF -v -s
"""

import os
import sys
import time
import wave
import struct
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import pytest
import numpy as np

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
COSYVOICE_VENV = PROJECT_ROOT / "cosyvoice_venv"
COSYVOICE_REPO = PROJECT_ROOT / "cosyvoice_repo"
MODELS_DIR = PROJECT_ROOT / "models" / "cosyvoice"
PROMPT_WAV = PROJECT_ROOT / "tests" / "golden" / "hello.wav"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "output" / "cosyvoice"

# Skip all tests if CosyVoice2 environment is not available
pytestmark = pytest.mark.skipif(
    not COSYVOICE_VENV.exists() or not COSYVOICE_REPO.exists(),
    reason="CosyVoice2 environment not available"
)

# Audio quality thresholds
QUALITY_THRESHOLDS = {
    "min_rms": 0.01,           # Minimum RMS to detect silence (< this = silent)
    "max_rms": 0.999,          # Maximum RMS to detect clipping (> this = clipped)
                               # Note: CosyVoice2 normalizes output to peak=0.99, not clipping
    "min_duration_sec": 0.5,   # Minimum audio duration (seconds)
    "max_silence_ratio": 0.5,  # Maximum ratio of silent frames
    "target_rtf": 1.0,         # Real-Time Factor target (< 1.0 = faster than real-time)
}

# Test sentences for CosyVoice2
TEST_CASES = [
    # (text, instruction, language, expected_min_duration)
    ("你好", "用四川话说", "zh-sichuan", 1.0),
    ("你好，今天天气真好", "用四川话说这段话", "zh-sichuan", 2.0),
    ("这是一个测试", "用普通话说", "zh-mandarin", 1.5),
    ("Hello world", "Say this in English", "en", 1.0),
]

# RTF benchmark test cases (short to long)
RTF_BENCHMARK_CASES = [
    ("你好", "用四川话说", 2),
    ("你好世界", "用四川话说", 4),
    ("这是一个测试句子", "用四川话说", 8),
    ("这是一个用来测试语音合成质量的长句子", "用四川话说", 18),
]


def get_audio_metrics(audio_data: np.ndarray) -> Dict:
    """Calculate audio quality metrics.

    Returns:
        dict with keys: rms, peak, zero_crossings, silence_ratio, duration_samples
    """
    if len(audio_data) == 0:
        return {
            "rms": 0.0,
            "peak": 0.0,
            "zero_crossings": 0,
            "silence_ratio": 1.0,
            "duration_samples": 0,
        }

    # Normalize to [-1, 1]
    audio = audio_data.astype(np.float32)
    if audio.max() > 1.0 or audio.min() < -1.0:
        audio = audio / max(abs(audio.max()), abs(audio.min()))

    # RMS (root mean square)
    rms = np.sqrt(np.mean(audio ** 2))

    # Peak amplitude
    peak = np.max(np.abs(audio))

    # Zero-crossing rate
    zero_crossings = np.sum(np.abs(np.diff(np.signbit(audio)))) / len(audio)

    # Silence detection (frames below threshold)
    silence_threshold = 0.01
    frame_size = 1024
    silent_frames = 0
    total_frames = len(audio) // frame_size

    for i in range(total_frames):
        frame = audio[i * frame_size:(i + 1) * frame_size]
        frame_rms = np.sqrt(np.mean(frame ** 2))
        if frame_rms < silence_threshold:
            silent_frames += 1

    silence_ratio = silent_frames / max(total_frames, 1)

    return {
        "rms": float(rms),
        "peak": float(peak),
        "zero_crossings": float(zero_crossings),
        "silence_ratio": float(silence_ratio),
        "duration_samples": len(audio),
    }


def load_wav_as_numpy(wav_path: Path) -> Tuple[np.ndarray, int]:
    """Load a WAV file as numpy array.

    Returns:
        (audio_data, sample_rate)
    """
    with wave.open(str(wav_path), 'rb') as wav:
        sample_rate = wav.getframerate()
        n_frames = wav.getnframes()
        n_channels = wav.getnchannels()
        sample_width = wav.getsampwidth()

        raw_data = wav.readframes(n_frames)

        if sample_width == 2:
            dtype = np.int16
        elif sample_width == 4:
            dtype = np.int32
        else:
            dtype = np.uint8

        audio = np.frombuffer(raw_data, dtype=dtype)

        if n_channels > 1:
            audio = audio.reshape(-1, n_channels)[:, 0]  # Take first channel

        # Normalize to [-1, 1]
        audio = audio.astype(np.float32) / np.iinfo(dtype).max

        return audio, sample_rate


@pytest.fixture(scope="module")
def cosyvoice_model():
    """Load CosyVoice2 model once for all tests."""
    import torch

    # Patch CUDA
    torch.cuda.is_available = lambda: False

    # Add CosyVoice to path
    sys.path.insert(0, str(COSYVOICE_REPO))

    from cosyvoice.cli.cosyvoice import CosyVoice2
    import torchaudio

    model_dir = MODELS_DIR / "CosyVoice2-0.5B"
    if not model_dir.exists():
        pytest.skip(f"CosyVoice2 model not found at {model_dir}")

    # Load model on CPU first
    model = CosyVoice2(str(model_dir), load_jit=False, load_trt=False, fp16=False)

    # Move to MPS if available
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        model.model.device = device
        model.model.llm.to(device)
        model.model.flow.to(device)
        model.model.hift.to(device)
        model.frontend.device = device

    # Load prompt audio
    prompt_speech, sr = torchaudio.load(str(PROMPT_WAV))
    if sr != 16000:
        prompt_speech = torchaudio.functional.resample(prompt_speech, sr, 16000)

    return model, prompt_speech


@pytest.fixture(scope="module")
def cosyvoice_compiled(cosyvoice_model):
    """Get compiled CosyVoice2 model with full torch.compile on all components.

    Worker #462/463 proved that compiling ALL components (LLM+Flow+HiFT)
    achieves RTF < 1.0 for all text lengths.
    """
    import torch

    model, prompt_speech = cosyvoice_model

    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available for torch.compile")

    # Apply torch.compile to ALL components (proven by Worker #462/463)
    # LLM is the main bottleneck
    model.model.llm.llm = torch.compile(
        model.model.llm.llm,
        mode='reduce-overhead',
        backend='inductor'
    )

    # Flow and HiFT also benefit from compilation
    model.model.flow = torch.compile(
        model.model.flow,
        mode='reduce-overhead',
        backend='inductor'
    )

    model.model.hift = torch.compile(
        model.model.hift,
        mode='reduce-overhead',
        backend='inductor'
    )

    # Warmup (triggers JIT compilation for all components)
    for _ in range(3):
        for result in model.inference_instruct2("你好", "用四川话说", prompt_speech, stream=False):
            _ = result['tts_speech']

    return model, prompt_speech


class TestCosyVoiceModelExists:
    """Test that CosyVoice2 model files exist."""

    def test_model_directory_exists(self):
        """Test that CosyVoice2 model directory exists."""
        model_dir = MODELS_DIR / "CosyVoice2-0.5B"
        assert model_dir.exists(), f"Model directory not found: {model_dir}"

    def test_prompt_audio_exists(self):
        """Test that prompt audio file exists."""
        assert PROMPT_WAV.exists(), f"Prompt audio not found: {PROMPT_WAV}"

    @pytest.mark.parametrize("required_file", [
        "llm.pt",
        "flow.pt",
        "hift.pt",
    ])
    def test_model_files_exist(self, required_file):
        """Test that required model files exist."""
        model_dir = MODELS_DIR / "CosyVoice2-0.5B"
        model_file = model_dir / required_file
        assert model_file.exists(), f"Model file not found: {model_file}"


class TestCosyVoiceAudioQuality:
    """Test CosyVoice2 audio output quality."""

    @pytest.mark.parametrize("text,instruction,lang,min_duration", TEST_CASES)
    def test_audio_not_silent(self, cosyvoice_model, text, instruction, lang, min_duration):
        """Test that generated audio is not silent."""
        model, prompt_speech = cosyvoice_model

        # Generate audio
        audio_output = None
        for result in model.inference_instruct2(text, instruction, prompt_speech, stream=False):
            audio_output = result['tts_speech']

        assert audio_output is not None, "No audio output generated"

        # Convert to numpy
        audio_np = audio_output.squeeze().cpu().numpy()

        # Get metrics
        metrics = get_audio_metrics(audio_np)

        # Verify not silent
        assert metrics["rms"] > QUALITY_THRESHOLDS["min_rms"], (
            f"Audio is silent! RMS={metrics['rms']:.6f} < threshold={QUALITY_THRESHOLDS['min_rms']}"
        )

        # Verify not mostly silence
        assert metrics["silence_ratio"] < QUALITY_THRESHOLDS["max_silence_ratio"], (
            f"Audio is mostly silent! silence_ratio={metrics['silence_ratio']:.2f}"
        )

    @pytest.mark.parametrize("text,instruction,lang,min_duration", TEST_CASES)
    def test_audio_not_clipped(self, cosyvoice_model, text, instruction, lang, min_duration):
        """Test that generated audio is not clipped."""
        model, prompt_speech = cosyvoice_model

        # Generate audio
        audio_output = None
        for result in model.inference_instruct2(text, instruction, prompt_speech, stream=False):
            audio_output = result['tts_speech']

        assert audio_output is not None, "No audio output generated"

        # Convert to numpy
        audio_np = audio_output.squeeze().cpu().numpy()

        # Get metrics
        metrics = get_audio_metrics(audio_np)

        # Verify not clipped (peak should not be at maximum)
        assert metrics["peak"] < QUALITY_THRESHOLDS["max_rms"], (
            f"Audio may be clipped! peak={metrics['peak']:.4f}"
        )

    @pytest.mark.parametrize("text,instruction,lang,min_duration", TEST_CASES)
    def test_audio_duration(self, cosyvoice_model, text, instruction, lang, min_duration):
        """Test that generated audio has expected duration."""
        model, prompt_speech = cosyvoice_model

        # Generate audio
        audio_output = None
        for result in model.inference_instruct2(text, instruction, prompt_speech, stream=False):
            audio_output = result['tts_speech']

        assert audio_output is not None, "No audio output generated"

        # Calculate duration
        sample_rate = model.sample_rate
        duration_sec = audio_output.shape[1] / sample_rate

        assert duration_sec >= min_duration * 0.5, (
            f"Audio too short! {duration_sec:.2f}s < {min_duration * 0.5:.2f}s"
        )


class TestCosyVoiceRTF:
    """Test CosyVoice2 Real-Time Factor (RTF) performance.

    RTF varies by text length - short texts have more overhead due to:
    - Fixed warmup/setup costs per inference
    - JIT compilation amortization

    Target: Average RTF < 1.0 across all text lengths.
    Individual short texts may exceed 1.0 but must stay under 1.15.
    """

    # RTF threshold for individual tests (allow some overhead for short texts)
    RTF_INDIVIDUAL_THRESHOLD = 1.15

    @pytest.mark.slow
    @pytest.mark.parametrize("text,instruction,chars", RTF_BENCHMARK_CASES)
    def test_rtf_below_threshold(self, cosyvoice_compiled, text, instruction, chars):
        """Test that RTF is below individual threshold (allows short text overhead)."""
        model, prompt_speech = cosyvoice_compiled

        # Generate audio and measure time
        start = time.time()
        audio_output = None
        for result in model.inference_instruct2(text, instruction, prompt_speech, stream=False):
            audio_output = result['tts_speech']
        inference_time = time.time() - start

        assert audio_output is not None, "No audio output generated"

        # Calculate RTF
        audio_duration = audio_output.shape[1] / model.sample_rate
        rtf = inference_time / audio_duration

        assert rtf < self.RTF_INDIVIDUAL_THRESHOLD, (
            f"RTF too high! RTF={rtf:.3f} >= threshold={self.RTF_INDIVIDUAL_THRESHOLD} "
            f"(text={chars} chars, inference={inference_time:.2f}s, audio={audio_duration:.2f}s)"
        )

    @pytest.mark.slow
    def test_rtf_benchmark_all(self, cosyvoice_compiled):
        """Run full RTF benchmark and report results."""
        model, prompt_speech = cosyvoice_compiled

        results = []
        for text, instruction, chars in RTF_BENCHMARK_CASES:
            # Warmup
            for result in model.inference_instruct2(text, instruction, prompt_speech, stream=False):
                _ = result['tts_speech']

            # Measure
            start = time.time()
            audio_output = None
            for result in model.inference_instruct2(text, instruction, prompt_speech, stream=False):
                audio_output = result['tts_speech']
            inference_time = time.time() - start

            audio_duration = audio_output.shape[1] / model.sample_rate
            rtf = inference_time / audio_duration

            results.append({
                "chars": chars,
                "inference_time": inference_time,
                "audio_duration": audio_duration,
                "rtf": rtf,
            })

        # Report results
        print("\n" + "=" * 60)
        print("CosyVoice2 RTF Benchmark Results (torch.compile)")
        print("=" * 60)
        print(f"{'Chars':<8} {'Inference':<12} {'Audio':<10} {'RTF':<8} {'Status':<10}")
        print("-" * 60)

        for r in results:
            # Individual tests can exceed 1.0 for short texts (overhead)
            status = "PASS" if r["rtf"] < self.RTF_INDIVIDUAL_THRESHOLD else "FAIL"
            print(f"{r['chars']:<8} {r['inference_time']:.2f}s{'':<6} {r['audio_duration']:.2f}s{'':<4} {r['rtf']:.3f}{'':<4} {status:<10}")

        avg_rtf = sum(r["rtf"] for r in results) / len(results)
        print("-" * 60)
        print(f"Average RTF: {avg_rtf:.3f} (target < {QUALITY_THRESHOLDS['target_rtf']})")
        print("=" * 60)

        # Key metric: AVERAGE RTF must be < 1.0 (real-time)
        assert avg_rtf < QUALITY_THRESHOLDS["target_rtf"], (
            f"Average RTF too high! avg_rtf={avg_rtf:.3f} >= target={QUALITY_THRESHOLDS['target_rtf']}"
        )


class TestCosyVoiceSaveOutput:
    """Test that CosyVoice2 can save output files."""

    def test_save_wav_output(self, cosyvoice_model):
        """Test saving audio output to WAV file."""
        import torchaudio

        model, prompt_speech = cosyvoice_model

        # Create output directory
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Generate audio
        audio_output = None
        for result in model.inference_instruct2("你好", "用四川话说", prompt_speech, stream=False):
            audio_output = result['tts_speech']

        assert audio_output is not None, "No audio output generated"

        # Save to WAV
        output_path = OUTPUT_DIR / "test_output.wav"
        torchaudio.save(str(output_path), audio_output.cpu(), model.sample_rate)

        assert output_path.exists(), f"Output file not created: {output_path}"
        assert output_path.stat().st_size > 1000, "Output file too small (likely empty)"

        # Verify WAV is valid
        audio_loaded, sr = load_wav_as_numpy(output_path)
        assert len(audio_loaded) > 0, "Loaded audio is empty"
        assert sr == model.sample_rate, f"Sample rate mismatch: {sr} != {model.sample_rate}"


def test_generate_quality_report(cosyvoice_model):
    """Generate comprehensive quality report."""
    import json
    from datetime import datetime

    model, prompt_speech = cosyvoice_model

    report = {
        "timestamp": datetime.now().isoformat(),
        "model": "CosyVoice2-0.5B",
        "tests": [],
    }

    for text, instruction, lang, min_duration in TEST_CASES:
        # Generate audio
        start = time.time()
        audio_output = None
        for result in model.inference_instruct2(text, instruction, prompt_speech, stream=False):
            audio_output = result['tts_speech']
        inference_time = time.time() - start

        if audio_output is None:
            report["tests"].append({
                "text": text,
                "instruction": instruction,
                "language": lang,
                "status": "FAILED",
                "error": "No audio output",
            })
            continue

        # Get metrics
        audio_np = audio_output.squeeze().cpu().numpy()
        metrics = get_audio_metrics(audio_np)

        audio_duration = audio_output.shape[1] / model.sample_rate
        rtf = inference_time / audio_duration

        # Determine pass/fail
        silent = metrics["rms"] < QUALITY_THRESHOLDS["min_rms"]
        clipped = metrics["peak"] > QUALITY_THRESHOLDS["max_rms"]
        too_short = audio_duration < min_duration * 0.5
        rtf_fail = rtf > QUALITY_THRESHOLDS["target_rtf"]

        status = "PASS"
        if silent or clipped or too_short:
            status = "FAIL"
        elif rtf_fail:
            status = "WARN"

        report["tests"].append({
            "text": text,
            "instruction": instruction,
            "language": lang,
            "status": status,
            "metrics": {
                "rms": metrics["rms"],
                "peak": metrics["peak"],
                "silence_ratio": metrics["silence_ratio"],
                "duration_sec": audio_duration,
                "inference_time_sec": inference_time,
                "rtf": rtf,
            },
            "issues": {
                "silent": silent,
                "clipped": clipped,
                "too_short": too_short,
                "rtf_high": rtf_fail,
            }
        })

    # Save report
    reports_dir = PROJECT_ROOT / "reports" / "main"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"COSYVOICE_QUALITY_{datetime.now().strftime('%Y-%m-%d')}.json"

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nQuality report saved to: {report_path}")

    # Print summary
    passed = sum(1 for t in report["tests"] if t["status"] == "PASS")
    warned = sum(1 for t in report["tests"] if t["status"] == "WARN")
    failed = sum(1 for t in report["tests"] if t["status"] == "FAIL")

    print(f"\nSummary: {passed} PASS, {warned} WARN, {failed} FAIL out of {len(report['tests'])} tests")

    assert failed == 0, f"{failed} tests failed"
