"""
Multi-Voice TTS Integration Tests

Tests for generating and mixing audio from multiple voices/languages.

Usage:
    pytest tests/integration/test_multi_voice.py -v
    pytest tests/integration/test_multi_voice.py -v -s  # With audio playback
"""

import json
import os
import sys
import pytest
import subprocess
import wave
import numpy as np
from pathlib import Path

# Project paths
PROJECT_ROOT_STR = str(Path(__file__).parent.parent.parent)
if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)

from providers import play_audio

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
STREAM_TTS_CPP = PROJECT_ROOT / "stream-tts-cpp"
BUILD_DIR = STREAM_TTS_CPP / "build"
CONFIG_DIR = STREAM_TTS_CPP / "config"

MIN_RMS = 0.01
MIN_NONZERO_RATIO = 0.2
NONZERO_EPS = 1e-4
PEAK_LIMIT = 1.0


# Voice configurations for multi-voice tests
VOICE_CONFIGS = {
    "en": {
        "config": "kokoro-mps-en.yaml",
        "name": "English",
        "phrase": "Hello, how are you today?"
    },
    "ja": {
        "config": "kokoro-mps-ja.yaml",
        "name": "Japanese",
        "phrase": "こんにちは、元気ですか？"
    },
    "zh": {
        "config": "kokoro-mps-zh.yaml",
        "name": "Chinese",
        "phrase": "你好，你今天好吗？"
    },
    "es": {
        "config": "kokoro-mps-es.yaml",
        "name": "Spanish",
        "phrase": "Hola, cómo estás hoy?"
    },
    "fr": {
        "config": "kokoro-mps-fr.yaml",
        "name": "French",
        "phrase": "Bonjour, comment allez-vous?"
    },
}


def get_tts_env():
    """Get environment variables for TTS subprocess."""
    env = os.environ.copy()
    # Fix OpenMP duplicate library crash when llama.cpp + libtorch both link OpenMP
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    return env


def generate_wav(tts_binary: Path, text: str, config: Path, output_path: Path,
                 timeout: int = 90) -> bool:
    """Generate a WAV file using TTS."""
    escaped_text = text.replace('"', '\\"').replace('\n', '\\n')
    input_json = f'{{"type":"content_block_delta","delta":{{"type":"text_delta","text":"{escaped_text}"}}}}'

    try:
        result = subprocess.run(
            [str(tts_binary), "--save-audio", str(output_path), str(config)],
            input=input_json.encode('utf-8'),
            capture_output=True,
            timeout=timeout,
            cwd=str(STREAM_TTS_CPP),
            env=get_tts_env()
        )
        return result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 1000
    except Exception as e:
        print(f"TTS generation error: {e}")
        return False


def read_wav(path: Path) -> tuple:
    """Read WAV file and return samples as float array."""
    with wave.open(str(path), 'rb') as wf:
        frames = wf.readframes(wf.getnframes())
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        return samples, wf.getframerate()


def write_wav(path: Path, samples: np.ndarray, sample_rate: int = 24000):
    """Write samples to WAV file."""
    samples = np.clip(samples, -1.0, 1.0)
    int_samples = (samples * 32767).astype(np.int16)
    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int_samples.tobytes())


def mix_audio_sequential(audio_list: list, silence_duration: float = 0.3, sample_rate: int = 24000) -> np.ndarray:
    """Mix multiple audio arrays sequentially with silence between clips."""
    silence = np.zeros(int(sample_rate * silence_duration))
    result = []
    for audio in audio_list:
        if result:
            result.append(silence)
        result.append(audio)
    return np.concatenate(result) if result else np.array([])


def mix_audio_simultaneous(audio_list: list) -> np.ndarray:
    """Mix multiple audio arrays simultaneously (average mix)."""
    if not audio_list:
        return np.array([])

    # Pad all to same length
    max_len = max(len(a) for a in audio_list)
    padded = [np.pad(a, (0, max_len - len(a))) for a in audio_list]

    # Average mix with normalization
    mixed = np.sum(padded, axis=0) / len(audio_list)

    # Normalize to prevent clipping
    max_val = np.max(np.abs(mixed))
    if max_val > 0.9:
        mixed = mixed * 0.9 / max_val

    return mixed


def audio_signal_stats(samples: np.ndarray, nonzero_threshold: float = NONZERO_EPS) -> tuple:
    """Compute basic audio stats for validation."""
    if samples.size == 0:
        return 0.0, 0.0, 0.0

    abs_samples = np.abs(samples)
    rms = float(np.sqrt(np.mean(samples ** 2)))
    peak = float(np.max(abs_samples))
    nonzero_ratio = float(np.count_nonzero(abs_samples > nonzero_threshold) / len(samples))
    return rms, peak, nonzero_ratio


def assert_audio_signal(samples: np.ndarray, label: str,
                        min_rms: float = MIN_RMS,
                        min_nonzero_ratio: float = MIN_NONZERO_RATIO) -> tuple:
    """Assert that audio contains real signal (not silence or clipping)."""
    rms, peak, nonzero_ratio = audio_signal_stats(samples)

    assert rms > min_rms, f"{label} RMS too low (rms={rms:.5f}, threshold={min_rms})"
    assert nonzero_ratio > min_nonzero_ratio, \
        f"{label} mostly silent (nonzero ratio={nonzero_ratio:.3f}, threshold={min_nonzero_ratio})"
    assert peak <= PEAK_LIMIT, f"{label} clipping detected (peak={peak:.3f})"

    return rms, peak, nonzero_ratio


@pytest.fixture(scope="module")
def tts_binary():
    """Path to stream-tts-cpp binary."""
    binary = BUILD_DIR / "stream-tts-cpp"
    if not binary.exists():
        pytest.skip(f"TTS binary not found at {binary}")
    return binary


@pytest.mark.integration
class TestMultiVoice:
    """Tests for multi-voice TTS generation and mixing."""

    def test_two_voices_sequential(self, tts_binary, tmp_path):
        """Test generating and mixing 2 voices sequentially (English + Japanese)."""
        languages = ["en", "ja"]
        audio_clips = []

        print("\n=== Two-Voice Sequential Test (EN + JA) ===")

        for lang in languages:
            voice_info = VOICE_CONFIGS[lang]
            config = CONFIG_DIR / voice_info["config"]

            if not config.exists():
                pytest.skip(f"{voice_info['name']} config not found")

            wav_path = tmp_path / f"voice_{lang}.wav"
            success = generate_wav(tts_binary, voice_info["phrase"], config, wav_path)

            assert success, f"Failed to generate {voice_info['name']} audio"

            samples, sr = read_wav(wav_path)
            audio_clips.append(samples)
            print(f"  {voice_info['name']}: {len(samples)} samples ({len(samples)/sr:.2f}s)")

        # Mix sequentially
        mixed = mix_audio_sequential(audio_clips)
        output_path = tmp_path / "two_voices_sequential.wav"
        write_wav(output_path, mixed)

        # Verify output
        assert output_path.exists()
        assert output_path.stat().st_size > len(audio_clips) * 1000  # Each clip should add significant size

        mixed_samples, _ = read_wav(output_path)
        expected_min_samples = sum(len(c) for c in audio_clips)  # At least sum of all clips
        assert len(mixed_samples) >= expected_min_samples, \
            f"Mixed audio too short: {len(mixed_samples)} < {expected_min_samples}"

        print(f"  Mixed: {len(mixed_samples)} samples ({len(mixed_samples)/24000:.2f}s)")
        print(f"  Output: {output_path}")

    def test_two_voices_simultaneous(self, tts_binary, tmp_path):
        """Test mixing 2 voices simultaneously."""
        languages = ["en", "ja"]
        audio_clips = []

        print("\n=== Two-Voice Simultaneous Test (EN + JA) ===")

        for lang in languages:
            voice_info = VOICE_CONFIGS[lang]
            config = CONFIG_DIR / voice_info["config"]

            if not config.exists():
                pytest.skip(f"{voice_info['name']} config not found")

            wav_path = tmp_path / f"voice_{lang}_sim.wav"
            success = generate_wav(tts_binary, voice_info["phrase"], config, wav_path)

            assert success, f"Failed to generate {voice_info['name']} audio"

            samples, sr = read_wav(wav_path)
            audio_clips.append(samples)

        # Mix simultaneously
        mixed = mix_audio_simultaneous(audio_clips)
        output_path = tmp_path / "two_voices_simultaneous.wav"
        write_wav(output_path, mixed)

        # Verify output
        assert output_path.exists()
        mixed_samples, _ = read_wav(output_path)

        # Simultaneous mix should be length of longest clip
        expected_len = max(len(c) for c in audio_clips)
        assert len(mixed_samples) == expected_len, \
            f"Mixed audio wrong length: {len(mixed_samples)} != {expected_len}"

        # Verify mix has real signal without clipping
        rms, peak, nonzero_ratio = assert_audio_signal(
            mixed_samples,
            "Simultaneous mix (2 voices)"
        )

        print(f"  Mixed: {len(mixed_samples)} samples ({len(mixed_samples)/24000:.2f}s)")
        print(f"  Stats: rms={rms:.4f}, peak={peak:.3f}, nonzero={nonzero_ratio:.2%}")
        print(f"  Output: {output_path}")

    def test_five_voices_sequential(self, tts_binary, tmp_path):
        """Test generating and mixing all 5 voices sequentially."""
        audio_clips = []
        successful_voices = []

        print("\n=== Five-Voice Sequential Test (All Languages) ===")

        for lang, voice_info in VOICE_CONFIGS.items():
            config = CONFIG_DIR / voice_info["config"]

            if not config.exists():
                print(f"  SKIP: {voice_info['name']} config not found")
                continue

            wav_path = tmp_path / f"voice_{lang}_5.wav"
            success = generate_wav(tts_binary, voice_info["phrase"], config, wav_path)

            if success:
                samples, sr = read_wav(wav_path)
                audio_clips.append(samples)
                successful_voices.append(voice_info["name"])
                print(f"  {voice_info['name']}: {len(samples)} samples ({len(samples)/sr:.2f}s)")
            else:
                print(f"  FAIL: {voice_info['name']}")

        # Must have at least 3 voices for meaningful test
        assert len(audio_clips) >= 3, \
            f"Only {len(audio_clips)} voices succeeded, need at least 3"

        # Mix sequentially
        mixed = mix_audio_sequential(audio_clips)
        output_path = tmp_path / "five_voices_sequential.wav"
        write_wav(output_path, mixed)

        assert output_path.exists()
        assert output_path.stat().st_size > 5000

        print(f"  Successful voices: {', '.join(successful_voices)}")
        print(f"  Mixed: {len(mixed)} samples ({len(mixed)/24000:.2f}s)")
        print(f"  Output: {output_path}")

    def test_five_voices_simultaneous(self, tts_binary, tmp_path):
        """Test mixing all 5 voices simultaneously."""
        audio_clips = []

        print("\n=== Five-Voice Simultaneous Test (All Languages) ===")

        for lang, voice_info in VOICE_CONFIGS.items():
            config = CONFIG_DIR / voice_info["config"]

            if not config.exists():
                continue

            wav_path = tmp_path / f"voice_{lang}_5sim.wav"
            success = generate_wav(tts_binary, voice_info["phrase"], config, wav_path)

            if success:
                samples, _ = read_wav(wav_path)
                audio_clips.append(samples)

        assert len(audio_clips) >= 3, \
            f"Only {len(audio_clips)} voices succeeded, need at least 3"

        # Mix simultaneously
        mixed = mix_audio_simultaneous(audio_clips)
        output_path = tmp_path / "five_voices_simultaneous.wav"
        write_wav(output_path, mixed)

        assert output_path.exists()

        # Verify the mix is meaningful (handles padded mixed audio)
        rms, peak, nonzero_ratio = assert_audio_signal(
            mixed,
            "Simultaneous mix (5 voices)"
        )

        print(f"  Mixed {len(audio_clips)} voices: {len(mixed)} samples ({len(mixed)/24000:.2f}s)")
        print(f"  Stats: rms={rms:.4f}, peak={peak:.3f}, nonzero={nonzero_ratio:.2%}")
        print(f"  Output: {output_path}")

    def test_same_language_multiple_phrases(self, tts_binary, tmp_path):
        """Test 3 different English phrases sequentially."""
        phrases = [
            "Hello, how are you today?",
            "The weather is beautiful.",
            "I love programming."
        ]
        audio_clips = []

        print("\n=== Same Language Multiple Phrases Test (English x3) ===")

        config = CONFIG_DIR / "kokoro-mps-en.yaml"
        if not config.exists():
            pytest.skip("English config not found")

        for i, phrase in enumerate(phrases):
            wav_path = tmp_path / f"phrase_{i}.wav"
            success = generate_wav(tts_binary, phrase, config, wav_path)

            assert success, f"Failed to generate phrase {i}: {phrase}"

            samples, sr = read_wav(wav_path)
            audio_clips.append(samples)
            print(f"  Phrase {i+1}: '{phrase[:30]}...' - {len(samples)/sr:.2f}s")

        # Mix sequentially
        mixed = mix_audio_sequential(audio_clips)
        output_path = tmp_path / "three_english_sequential.wav"
        write_wav(output_path, mixed)

        assert output_path.exists()
        assert len(mixed) > sum(len(c) for c in audio_clips) * 0.9  # Account for rounding

        print(f"  Total: {len(mixed)/24000:.2f}s")
        print(f"  Output: {output_path}")

    def test_voice_isolation(self, tts_binary, tmp_path):
        """Test that different voice configs produce distinct audio."""
        # Generate same phrase with two different languages
        phrase = "Hello"

        config_en = CONFIG_DIR / "kokoro-mps-en.yaml"
        config_fr = CONFIG_DIR / "kokoro-mps-fr.yaml"

        if not config_en.exists() or not config_fr.exists():
            pytest.skip("Required configs not found")

        wav_en = tmp_path / "isolation_en.wav"
        wav_fr = tmp_path / "isolation_fr.wav"

        success_en = generate_wav(tts_binary, phrase, config_en, wav_en)
        success_fr = generate_wav(tts_binary, phrase, config_fr, wav_fr)

        assert success_en and success_fr, "Failed to generate audio"

        samples_en, _ = read_wav(wav_en)
        samples_fr, _ = read_wav(wav_fr)

        # Different voices should produce different audio lengths or waveforms
        # They might be similar in length but different in content
        min_len = min(len(samples_en), len(samples_fr))

        if min_len > 0:
            # Calculate correlation between the two waveforms
            corr = np.corrcoef(samples_en[:min_len], samples_fr[:min_len])[0, 1]

            # Different languages should not be highly correlated
            # (though they might have some similarity in prosody)
            print(f"\n  EN vs FR correlation: {corr:.3f}")

            # They shouldn't be identical (correlation ~1.0)
            assert corr < 0.99, "Different language configs produced nearly identical audio"


@pytest.mark.integration
class TestMultiVoicePlayback:
    """Tests that play audio for manual verification (run with -s flag)."""

    @pytest.mark.skip(reason="Manual playback test - run with pytest -s --runplayback")
    def test_play_two_voices(self, tts_binary, tmp_path):
        """Generate and play 2-voice demo."""
        audio_clips = []

        for lang in ["en", "ja"]:
            voice_info = VOICE_CONFIGS[lang]
            config = CONFIG_DIR / voice_info["config"]
            if not config.exists():
                pytest.skip(f"{voice_info['name']} config not found")

            wav_path = tmp_path / f"play_{lang}.wav"
            if generate_wav(tts_binary, voice_info["phrase"], config, wav_path):
                samples, _ = read_wav(wav_path)
                audio_clips.append(samples)

        if len(audio_clips) == 2:
            # Sequential
            mixed = mix_audio_sequential(audio_clips)
            output_seq = tmp_path / "play_sequential.wav"
            write_wav(output_seq, mixed)

            print(f"\nPlaying sequential mix...")
            with open(output_seq, 'rb') as f:
                play_audio(f.read())

            # Simultaneous
            mixed_sim = mix_audio_simultaneous(audio_clips)
            output_sim = tmp_path / "play_simultaneous.wav"
            write_wav(output_sim, mixed_sim)

            print(f"Playing simultaneous mix...")
            with open(output_sim, 'rb') as f:
                play_audio(f.read())


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
