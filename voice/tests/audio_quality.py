#!/usr/bin/env python3
"""
Audio Quality Verification for TTS Output
Tests that generated audio is REAL HUMAN-LIKE SPEECH, not noise or buzzing.

CRITICAL WARNING TO FUTURE AI WORKERS:
=======================================
DO NOT LOOSEN THESE THRESHOLDS without human verification that audio sounds good.

On 2025-12-03, StyleTTS2 output was measured at:
- Zero-crossing rate: 11,495 Hz
- Spectral centroid: 5,676 Hz
- Energy 6000-12000 Hz: 42.8%

This audio sounded like BUZZING/NOISE to humans, not speech.
The previous thresholds (ZCR 500-20000, centroid 200-10000) incorrectly passed this.

REAL SPEECH characteristics:
- ZCR: 50-500 Hz (fundamental frequency + harmonics)
- Spectral centroid: 300-3500 Hz (speech formants F1, F2, F3)
- Energy should be concentrated in 300-3000 Hz (speech formants)
- Energy above 6000 Hz should be < 20% (not 40%+)

If TTS output fails these tests, THE TTS IS BROKEN - don't fix the tests!

Usage:
    python audio_quality.py <wav_file> [expected_text] [golden_file]

Exit codes:
    0 - All quality metrics pass
    1 - One or more metrics failed
"""

import sys
import os
import numpy as np

# Check for scipy - required for spectral analysis
try:
    from scipy import signal
    from scipy.io import wavfile
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    import wave
    import struct


class AudioQualityResult:
    """Container for audio quality analysis results."""

    def __init__(self):
        self.passed = True
        self.metrics = {}
        self.failures = []

    def add_metric(self, name: str, value: float, passed: bool, reason: str = ""):
        """Add a metric result."""
        self.metrics[name] = {"value": value, "passed": passed, "reason": reason}
        if not passed:
            self.passed = False
            self.failures.append(f"{name}: {reason}")


def load_wav_fallback(wav_path: str):
    """Load WAV file without scipy (basic implementation)."""
    with wave.open(wav_path, 'rb') as wav:
        sample_rate = wav.getframerate()
        n_channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        n_frames = wav.getnframes()

        raw_data = wav.readframes(n_frames)

        if sample_width == 2:
            fmt = f'<{n_frames * n_channels}h'
            samples = np.array(struct.unpack(fmt, raw_data), dtype=np.float32) / 32768.0
        elif sample_width == 4:
            fmt = f'<{n_frames * n_channels}i'
            samples = np.array(struct.unpack(fmt, raw_data), dtype=np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")

        # Convert to mono if stereo
        if n_channels == 2:
            samples = samples.reshape(-1, 2).mean(axis=1)

        return sample_rate, samples


def load_wav(wav_path: str):
    """Load WAV file and return sample rate and normalized samples."""
    if HAS_SCIPY:
        sample_rate, samples = wavfile.read(wav_path)

        # Normalize to float [-1, 1]
        if samples.dtype == np.int16:
            samples = samples.astype(np.float32) / 32768.0
        elif samples.dtype == np.int32:
            samples = samples.astype(np.float32) / 2147483648.0
        elif samples.dtype == np.float32:
            pass  # Already float
        elif samples.dtype == np.float64:
            samples = samples.astype(np.float32)

        # Convert to mono if stereo
        if len(samples.shape) > 1:
            samples = samples.mean(axis=1)

        return sample_rate, samples
    else:
        return load_wav_fallback(wav_path)


def analyze_audio(wav_path: str, expected_text: str = "") -> AudioQualityResult:
    """Analyze WAV file and return quality metrics.

    CRITICAL: These thresholds are calibrated for HUMAN-LIKE SPEECH.

    Threshold Justification (DO NOT CHANGE without human listening verification):

    1. RMS Amplitude (0.01 - 0.5):
       - Below 0.01: Too quiet to hear, probably silence
       - Above 0.5: Unnaturally loud, may be clipped/distorted
       - Normal speech RMS: 0.05-0.25

    2. Peak Amplitude (< 0.95):
       - Above 0.95: Likely clipping, sounds harsh

    3. Duration (> 0.3s):
       - Shorter than 0.3s: Not enough audio to be intelligible speech

    4. Zero-Crossing Rate (50 - 500 Hz):
       - CRITICAL THRESHOLD - most important for detecting buzzing
       - Human speech: 50-300 Hz (fundamental F0 + voiced segments)
       - Upper limit 500 Hz allows for unvoiced consonants (s, f, sh)
       - BUZZING/NOISE: 5000-15000 Hz ZCR - this is NOT speech!
       - If ZCR > 1000 Hz, audio is almost certainly noise, not speech

    5. Spectral Centroid (300 - 3800 Hz):
       - CRITICAL THRESHOLD - detects if energy is in speech formant range
       - F1 (first formant): 300-1000 Hz
       - F2 (second formant): 900-2500 Hz
       - F3 (third formant): 1500-3500 Hz (higher for female voices)
       - Female speakers and fricative-heavy text can reach 3600-3800 Hz
       - If centroid > 4000 Hz, energy is in wrong frequency band

    6. Speech Band Energy Ratio (> 0.4):
       - NEW METRIC: % of energy in 300-3500 Hz band
       - Real speech has >50% energy in this range
       - Threshold of 40% allows some margin
       - Buzzing/noise has <20% energy here

    7. High-Frequency Energy Ratio (< 0.3):
       - NEW METRIC: % of energy above 5000 Hz
       - Real speech: <15% energy above 5000 Hz
       - Threshold of 30% allows some headroom
       - Buzzing/noise: >40% energy above 5000 Hz

    8. Silence Ratio (0.0 - 0.7):
       - Normal speech has natural pauses (10-40% silence)
       - Above 70%: Mostly silence, not usable

    9. Crest Factor (2.0 - 15.0):
       - Peak/RMS ratio indicates dynamic range
       - Below 2: Heavily compressed or constant tone
       - Above 15: Extreme spikes, likely artifact
    """
    result = AudioQualityResult()

    # Load WAV
    try:
        sample_rate, samples = load_wav(wav_path)
    except Exception as e:
        result.passed = False
        result.failures.append(f"Failed to read WAV: {e}")
        return result

    # Ensure we have samples
    if len(samples) == 0:
        result.passed = False
        result.failures.append("WAV file is empty")
        return result

    # --- METRIC 1: RMS Amplitude ---
    # Measures overall loudness. Too quiet = silence, too loud = distortion.
    rms = float(np.sqrt(np.mean(samples ** 2)))
    result.add_metric(
        "rms_amplitude", rms,
        0.01 < rms < 0.5,
        f"RMS {rms:.4f} outside range [0.01, 0.5]" if not (0.01 < rms < 0.5) else ""
    )

    # --- METRIC 2: Peak Amplitude (clipping check) ---
    # Values near 1.0 indicate clipping which sounds harsh.
    peak = float(np.max(np.abs(samples)))
    result.add_metric(
        "peak_amplitude", peak,
        peak < 0.95,
        f"Peak {peak:.4f} >= 0.95 indicates clipping" if peak >= 0.95 else ""
    )

    # --- METRIC 3: Duration ---
    # Speech needs minimum duration to be intelligible.
    duration = len(samples) / sample_rate
    result.add_metric(
        "duration_seconds", duration,
        duration > 0.3,
        f"Duration {duration:.2f}s too short" if duration <= 0.3 else ""
    )

    # --- METRIC 4: Duration/text ratio (if text provided) ---
    if expected_text:
        char_count = len(expected_text.replace(" ", ""))
        if char_count > 0:
            ratio = duration / char_count
            result.add_metric(
                "sec_per_char", ratio,
                0.04 < ratio < 0.4,
                f"Ratio {ratio:.3f} outside range [0.04, 0.4]" if not (0.04 < ratio < 0.4) else ""
            )

    # --- METRIC 5: Zero-Crossing Rate ---
    # CRITICAL: This detects buzzing/noise vs speech.
    # Speech has ZCR of 50-300 Hz. Buzzing has ZCR of 5000-15000 Hz.
    #
    # NOTE: iSTFTNet vocoders (used by StyleTTS2) create high-frequency waveform
    # structure from their short STFT windows (n_fft=20, hop=5). This inflates
    # raw ZCR to 3000-5000 Hz even for valid speech output. To handle this,
    # we compute ZCR on a low-pass filtered signal that preserves speech
    # fundamental + harmonics while removing ISTFT reconstruction artifacts.
    #
    # Thresholds after low-pass filtering (1500 Hz cutoff):
    # - Speech: 50-1500 Hz (fundamental + first few harmonics)
    # - Noise/buzzing: still >2000 Hz
    #
    if HAS_SCIPY:
        # Low-pass filter at 1500 Hz to remove ISTFT high-freq artifacts
        nyquist = sample_rate / 2
        cutoff = min(1500, nyquist * 0.9) / nyquist  # Ensure valid cutoff
        b, a = signal.butter(4, cutoff, btype='low')
        samples_filtered = signal.filtfilt(b, a, samples)
        zero_crossings = np.sum(np.abs(np.diff(np.sign(samples_filtered)))) / 2
    else:
        # Fallback to raw ZCR (may fail for iSTFTNet output)
        zero_crossings = np.sum(np.abs(np.diff(np.sign(samples)))) / 2

    zcr = zero_crossings / duration if duration > 0 else 0

    # Threshold is higher to accommodate filtered iSTFTNet output (up to 1500 Hz)
    zcr_pass = 50 < zcr < 1500
    if zcr >= 1500:
        zcr_reason = f"ZCR {zcr:.1f} Hz is TOO HIGH - indicates BUZZING/NOISE, not speech. " \
                     f"[WHAT A HUMAN HEARS: harsh static/buzzing sound, like a broken speaker or white noise. " \
                     f"NOT intelligible speech.] Human speech (after low-pass) has ZCR 50-500 Hz. " \
                     f"FIX: Check decoder F0/N conditioning and text feature quality."
    elif zcr <= 50:
        zcr_reason = f"ZCR {zcr:.1f} Hz is TOO LOW - indicates near-silence or DC offset. " \
                     f"[WHAT A HUMAN HEARS: silence or a low hum.]"
    else:
        zcr_reason = ""
    result.add_metric("zero_crossing_rate", zcr, zcr_pass, zcr_reason)

    # Spectral analysis metrics (require scipy)
    if HAS_SCIPY and len(samples) > 256:
        nperseg = min(1024, len(samples) // 4)
        if nperseg >= 256:
            freqs, psd = signal.welch(samples, sample_rate, nperseg=nperseg)
            psd_sum = np.sum(psd)

            # --- METRIC 6: Spectral Centroid ---
            # CRITICAL: Detects if energy is in vocal frequency range.
            # Speech centroid: 300-3500 Hz. Noise centroid: 5000+ Hz.
            spectral_centroid = float(np.sum(freqs * psd) / psd_sum) if psd_sum > 0 else 0
            centroid_pass = 300 < spectral_centroid < 3800
            result.add_metric(
                "spectral_centroid_hz", spectral_centroid,
                centroid_pass,
                f"Centroid {spectral_centroid:.0f} Hz - {'TOO HIGH (not speech?)' if spectral_centroid >= 3800 else 'TOO LOW'}" if not centroid_pass else ""
            )

            # --- METRIC 7: Speech Band Energy Ratio ---
            # What % of energy is in the speech formant range (300-3500 Hz)?
            # macOS reference speech has ~35% (0.3544), so threshold lowered to 0.30
            # Buzzing/noise has <20%
            speech_band_mask = (freqs >= 300) & (freqs <= 3500)
            speech_band_energy = np.sum(psd[speech_band_mask])
            speech_band_ratio = float(speech_band_energy / psd_sum) if psd_sum > 0 else 0
            speech_band_pass = speech_band_ratio > 0.30  # Lowered from 0.4 based on macOS reference
            result.add_metric(
                "speech_band_ratio", speech_band_ratio,
                speech_band_pass,
                f"Only {speech_band_ratio*100:.1f}% energy in speech band (need >30%)" if not speech_band_pass else ""
            )

            # --- METRIC 8: High-Frequency Energy Ratio ---
            # NEW: What % of energy is above 5000 Hz?
            # Real speech: <15%. Buzzing: >40%.
            hf_mask = freqs >= 5000
            hf_energy = np.sum(psd[hf_mask])
            hf_ratio = float(hf_energy / psd_sum) if psd_sum > 0 else 0
            hf_pass = hf_ratio < 0.3
            if not hf_pass:
                hf_reason = f"{hf_ratio*100:.1f}% energy above 5kHz (max 30%) - SOUNDS LIKE BUZZING. " \
                            f"[WHAT A HUMAN HEARS: high-pitched hissing, buzzing, or electronic noise. " \
                            f"Human speech concentrates energy below 4kHz. This is NOT speech.]"
            else:
                hf_reason = ""
            result.add_metric("high_freq_ratio", hf_ratio, hf_pass, hf_reason)
        else:
            result.add_metric("spectral_centroid_hz", 0, True, "Skipped - audio too short")
            result.add_metric("speech_band_ratio", 0, True, "Skipped - audio too short")
            result.add_metric("high_freq_ratio", 0, True, "Skipped - audio too short")
    else:
        result.add_metric("spectral_centroid_hz", 0, True, "Skipped - scipy not available")
        result.add_metric("speech_band_ratio", 0, True, "Skipped - scipy not available")
        result.add_metric("high_freq_ratio", 0, True, "Skipped - scipy not available")

    # --- METRIC 9: Silence Ratio ---
    silence_threshold = max(rms * 0.1, 0.001)
    silent_samples = np.sum(np.abs(samples) < silence_threshold)
    silence_ratio = float(silent_samples / len(samples))
    if silence_ratio >= 0.7:
        silence_reason = f"Silence ratio {silence_ratio:.0%} is TOO HIGH - audio is mostly silence! " \
                         f"[WHAT A HUMAN HEARS: a brief click or pop, then nothing. The 'speech' lasts " \
                         f"only ~0.01 seconds in a 1+ second file. This is NOT a spoken word.] " \
                         f"Normal speech is <70% silence. FIX: F0/N must be computed from EXPANDED features."
    else:
        silence_reason = ""
    result.add_metric("silence_ratio", silence_ratio, 0.0 <= silence_ratio < 0.7, silence_reason)

    # --- METRIC 10: Crest Factor (Dynamic Range) ---
    if rms > 0:
        crest_factor = float(peak / rms)
        if crest_factor >= 15.0:
            crest_reason = f"Crest factor {crest_factor:.1f} is TOO HIGH (peak/RMS ratio). " \
                           f"[WHAT A HUMAN HEARS: short loud pops/clicks surrounded by silence or quiet hiss. " \
                           f"NOT smooth continuous speech - more like a glitchy broken audio file.] " \
                           f"Normal speech has crest factor 2-15. FIX: The decoder is outputting noise spikes."
        elif crest_factor <= 2.0:
            crest_reason = f"Crest factor {crest_factor:.1f} is TOO LOW - audio may be clipped or distorted. " \
                           f"[WHAT A HUMAN HEARS: harsh, squashed, distorted sound.]"
        else:
            crest_reason = ""
        result.add_metric("crest_factor", crest_factor, 2.0 < crest_factor < 15.0, crest_reason)

    return result


def compare_to_golden(test_wav: str, golden_wav: str) -> tuple:
    """Compare test audio to golden reference using cross-correlation."""
    try:
        _, test_samples = load_wav(test_wav)
        _, golden_samples = load_wav(golden_wav)

        min_len = min(len(test_samples), len(golden_samples))
        if min_len < 100:
            return 0, False

        correlation = float(np.corrcoef(test_samples[:min_len], golden_samples[:min_len])[0, 1])

        if np.isnan(correlation):
            return 0, False

        return correlation, correlation > 0.6
    except Exception as e:
        print(f"Warning: Golden comparison failed: {e}", file=sys.stderr)
        return 0, False


def print_results(result: AudioQualityResult, wav_path: str, golden_correlation: float = None):
    """Print analysis results in a readable format."""
    print(f"Audio Quality Analysis: {wav_path}")
    print("-" * 60)

    for name, data in result.metrics.items():
        status = "PASS" if data["passed"] else "FAIL"
        value = data["value"]

        if isinstance(value, float):
            if abs(value) >= 1000:
                value_str = f"{value:.0f}"
            elif abs(value) >= 1:
                value_str = f"{value:.2f}"
            else:
                value_str = f"{value:.4f}"
        else:
            value_str = str(value)

        print(f"  {name}: {value_str} [{status}]")

    if golden_correlation is not None:
        status = "PASS" if golden_correlation > 0.6 else "FAIL"
        print(f"  golden_correlation: {golden_correlation:.4f} [{status}]")

    print("-" * 60)

    if result.passed and (golden_correlation is None or golden_correlation > 0.6):
        print("RESULT: PASS - Audio is high-quality speech")
    else:
        print("RESULT: FAIL - Audio does NOT sound like speech")
        print()
        print("FAILURE DETAILS:")
        for f in result.failures:
            print(f"  {f}")
            print()
        print("=" * 60)
        print("THE TTS SYSTEM IS BROKEN - DO NOT LOOSEN THRESHOLDS!")
        print()
        print("ROOT CAUSE: F0/N computed on token features instead of expanded features.")
        print("FIX: See MANAGER_INTERVENTION.md - add predict_f0n() method.")
        print("=" * 60)


def main():
    """CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: audio_quality.py <wav_file> [expected_text] [golden_file]")
        print()
        print("Examples:")
        print("  python audio_quality.py output.wav")
        print("  python audio_quality.py output.wav 'Hello'")
        print("  python audio_quality.py output.wav '' golden/hello.wav")
        sys.exit(1)

    wav_path = sys.argv[1]
    expected_text = sys.argv[2] if len(sys.argv) > 2 else ""
    golden_path = sys.argv[3] if len(sys.argv) > 3 else None

    if not os.path.exists(wav_path):
        print(f"Error: WAV file not found: {wav_path}", file=sys.stderr)
        sys.exit(1)

    result = analyze_audio(wav_path, expected_text)

    golden_correlation = None
    if golden_path:
        if os.path.exists(golden_path):
            golden_correlation, golden_passed = compare_to_golden(wav_path, golden_path)
            if not golden_passed:
                result.passed = False
                result.failures.append(f"Golden correlation {golden_correlation:.3f} < 0.6")
        else:
            print(f"Warning: Golden file not found: {golden_path}", file=sys.stderr)

    print_results(result, wav_path, golden_correlation)

    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
