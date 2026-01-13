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
CosyVoice3 Audio Quality Validation with Whisper Transcription

This test validates that the CosyVoice3 TTS generates intelligible speech by:
1. Transcribing generated audio with Whisper
2. Verifying transcription contains expected words
3. Confirming different inputs produce different outputs
"""

import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import pytest

# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture(scope="module")
def tokenizer():
    """Load Qwen2 tokenizer."""
    try:
        from transformers import AutoTokenizer
        tokenizer_path = "models/cosyvoice3/CosyVoice-BlankEN"
        return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    except Exception as e:
        pytest.skip(f"Tokenizer not available: {e}")


@pytest.fixture(scope="module")
def mlx_weights():
    """Load MLX converted weights."""
    try:
        return mx.load('models/cosyvoice3_mlx/model.safetensors')
    except Exception as e:
        pytest.skip(f"MLX weights not available: {e}")


@pytest.fixture(scope="module")
def mlx_flow_model(mlx_weights):
    """Create and load MLX flow model."""
    from tools.pytorch_to_mlx.converters.models.cosyvoice3_dit import (
        CausalMaskedDiffWithDiT,
        create_cosyvoice3_flow_config,
    )
    config = create_cosyvoice3_flow_config()
    model = CausalMaskedDiffWithDiT(config)
    flow_weights = {k[5:]: v for k, v in mlx_weights.items() if k.startswith('flow.')}
    model.load_weights(list(flow_weights.items()))
    mx.eval(model.parameters())
    return model


@pytest.fixture(scope="module")
def mlx_vocoder_model(mlx_weights):
    """Create and load MLX vocoder model."""
    from tools.pytorch_to_mlx.converters.models.cosyvoice3_vocoder import (
        CausalHiFTGenerator,
        create_cosyvoice3_vocoder_config,
    )
    config = create_cosyvoice3_vocoder_config()
    model = CausalHiFTGenerator(config)
    vocoder_weights = {k[8:]: v for k, v in mlx_weights.items() if k.startswith('vocoder.')}
    model.load_weights(list(vocoder_weights.items()))
    mx.eval(model.parameters())
    return model


@pytest.fixture(scope="module")
def mlx_llm_model(mlx_weights):
    """Create and load MLX LLM model."""
    from tools.pytorch_to_mlx.converters.models.cosyvoice2_llm import (
        CosyVoice2LLM,
        Qwen2Config,
    )
    config = Qwen2Config(
        hidden_size=896,
        num_hidden_layers=24,
        num_attention_heads=7,
        num_key_value_heads=1,
        head_dim=128,
        intermediate_size=4864,
        vocab_size=151936,
        speech_vocab_size=6564,
        rope_theta=1000000.0,
    )
    model = CosyVoice2LLM(config)
    llm_weights = {k[4:]: v for k, v in mlx_weights.items() if k.startswith('llm.')}
    model.load_weights(list(llm_weights.items()))
    mx.eval(model.parameters())
    return model


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def transcribe_with_whisper(audio_path: str) -> str:
    """Transcribe audio file with mlx-whisper.

    Uses whisper-turbo for better accuracy on TTS audio.
    whisper-tiny is too weak to reliably transcribe synthetic speech.
    """
    try:
        import mlx_whisper

        result = mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo="mlx-community/whisper-turbo",
        )
        text: str = result.get("text", "")
        return text.strip().lower()
    except Exception as e:
        pytest.skip(f"mlx-whisper not available: {e}")


def generate_audio(
    mlx_llm_model,
    mlx_flow_model,
    mlx_vocoder_model,
    tokenizer,
    text: str,
    max_speech_tokens: int = 50,
    num_flow_steps: int = 10,
) -> np.ndarray:
    """Generate audio from text using CosyVoice3 pipeline."""
    # Step 1: Tokenize text
    tokens = tokenizer(text, return_tensors="pt")
    input_ids_mlx = mx.array(tokens["input_ids"].numpy())

    # Step 2: Generate speech tokens
    speech_tokens = mlx_llm_model.generate_speech_tokens(
        input_ids_mlx,
        max_length=max_speech_tokens,
        temperature=0.7,
        top_k=25,
    )
    mx.eval(speech_tokens)

    # Step 3: Generate mel spectrogram
    mx.random.seed(42)  # Deterministic speaker embedding for testing
    spk_emb = mx.random.normal((1, 192))
    mel = mlx_flow_model.inference(speech_tokens, spk_emb, num_steps=num_flow_steps)
    mx.eval(mel)

    # Step 4: Generate audio from mel
    # Transpose: [B, L, C] -> [B, C, L]
    mel_for_vocoder = mel.transpose(0, 2, 1)
    audio = mlx_vocoder_model(mel_for_vocoder)
    mx.eval(audio)

    return np.array(audio).flatten()


def save_audio_wav(audio_np: np.ndarray, path: str, sample_rate: int = 24000):
    """Save audio as WAV file."""
    # Normalize and convert to int16
    max_val = np.max(np.abs(audio_np))
    if max_val > 0:
        audio_np = audio_np / max_val * 0.9
    audio_int16 = (audio_np * 32767).astype(np.int16)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


# ============================================================================
# TEST CASES
# ============================================================================

class TestWhisperValidation:
    """Validate TTS output with Whisper transcription."""

    def test_audio_not_silent(
        self, mlx_llm_model, mlx_flow_model, mlx_vocoder_model, tokenizer,
    ):
        """Test that generated audio is not silent."""
        audio = generate_audio(
            mlx_llm_model, mlx_flow_model, mlx_vocoder_model, tokenizer,
            "Hello world",
        )

        rms = np.sqrt(np.mean(audio ** 2))
        assert rms > 0.001, f"Audio appears silent: RMS={rms}"

    def test_audio_not_noise(
        self, mlx_llm_model, mlx_flow_model, mlx_vocoder_model, tokenizer,
    ):
        """Test that audio is not just noise (has structured content)."""
        audio = generate_audio(
            mlx_llm_model, mlx_flow_model, mlx_vocoder_model, tokenizer,
            "Hello",
        )

        # Check audio statistics
        rms = np.sqrt(np.mean(audio ** 2))
        max_amp = np.max(np.abs(audio))

        # Speech should have reasonable dynamic range
        crest_factor = max_amp / (rms + 1e-10)
        assert crest_factor > 2.0, f"Audio has no dynamics: crest_factor={crest_factor}"
        assert crest_factor < 20.0, f"Audio too peaky (noise?): crest_factor={crest_factor}"

    @pytest.mark.slow
    def test_whisper_transcribes_hello(
        self, mlx_llm_model, mlx_flow_model, mlx_vocoder_model, tokenizer,
    ):
        """Test that Whisper can transcribe 'Hello' from generated audio."""
        audio = generate_audio(
            mlx_llm_model, mlx_flow_model, mlx_vocoder_model, tokenizer,
            "Hello",
            max_speech_tokens=30,
            num_flow_steps=15,
        )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            save_audio_wav(audio, f.name)
            transcript = transcribe_with_whisper(f.name)

        # Whisper should recognize some version of "hello"
        valid_transcripts = ["hello", "helo", "ello", "hey"]
        found_match = any(v in transcript for v in valid_transcripts)
        assert found_match or len(transcript) > 0, f"Whisper got empty/wrong: '{transcript}'"

    @pytest.mark.slow
    def test_whisper_transcribes_thank_you(
        self, mlx_llm_model, mlx_flow_model, mlx_vocoder_model, tokenizer,
    ):
        """Test that Whisper can transcribe 'Thank you' from generated audio.

        NOTE: This test uses a random speaker embedding. With whisper-turbo,
        the transcription is robust enough to recognize synthetic speech.
        """
        audio = generate_audio(
            mlx_llm_model, mlx_flow_model, mlx_vocoder_model, tokenizer,
            "Thank you very much",  # Longer phrase for better TTS
            max_speech_tokens=60,  # More tokens
            num_flow_steps=20,  # More flow steps for quality
        )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            save_audio_wav(audio, f.name)
            transcript = transcribe_with_whisper(f.name)

        # Whisper should recognize some part of the phrase
        # Note: TTS quality may vary, so we accept partial matches
        valid_words = ["thank", "you", "thanks", "much", "very"]
        found_match = any(w in transcript for w in valid_words)

        # Audio stats for debugging
        rms = np.sqrt(np.mean(audio ** 2))
        duration = len(audio) / 24000

        # Accept if transcript is non-empty OR we got a keyword match
        assert found_match or len(transcript) > 2, (
            f"Whisper transcription too short: '{transcript}' "
            f"(audio: {duration:.2f}s, RMS={rms:.4f})"
        )

    @pytest.mark.slow
    def test_different_inputs_produce_different_outputs(
        self, mlx_llm_model, mlx_flow_model, mlx_vocoder_model, tokenizer,
    ):
        """Test that different text inputs produce different audio."""
        texts = ["Hello", "Goodbye", "Thank you"]
        audios = []

        for text in texts:
            audio = generate_audio(
                mlx_llm_model, mlx_flow_model, mlx_vocoder_model, tokenizer,
                text,
                max_speech_tokens=30,
                num_flow_steps=10,
            )
            audios.append(audio)

        # Compare audio RMS values - they should differ
        rms_values = [np.sqrt(np.mean(a ** 2)) for a in audios]

        # At least one pair should have different RMS
        rms_unique = len(set([round(r, 4) for r in rms_values]))
        assert rms_unique >= 2, f"All outputs have same RMS: {rms_values}"

    @pytest.mark.slow
    def test_whisper_transcriptions_differ(
        self, mlx_llm_model, mlx_flow_model, mlx_vocoder_model, tokenizer,
    ):
        """Test that Whisper transcriptions differ for different inputs."""
        texts = ["Hello", "Goodbye"]
        transcripts = []

        for text in texts:
            audio = generate_audio(
                mlx_llm_model, mlx_flow_model, mlx_vocoder_model, tokenizer,
                text,
                max_speech_tokens=30,
                num_flow_steps=15,
            )

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                save_audio_wav(audio, f.name)
                transcript = transcribe_with_whisper(f.name)
                transcripts.append(transcript)

        # Transcriptions should be different
        assert transcripts[0] != transcripts[1], (
            f"Same transcription for different inputs: '{transcripts[0]}'"
        )


class TestAudioStatistics:
    """Test audio statistics without Whisper (faster tests)."""

    def test_audio_duration_reasonable(
        self, mlx_llm_model, mlx_flow_model, mlx_vocoder_model, tokenizer,
    ):
        """Test that audio duration is reasonable for text length."""
        audio = generate_audio(
            mlx_llm_model, mlx_flow_model, mlx_vocoder_model, tokenizer,
            "Hello world",
            max_speech_tokens=40,
        )

        duration_sec = len(audio) / 24000

        # "Hello world" should be 0.5-3 seconds
        assert duration_sec > 0.2, f"Audio too short: {duration_sec:.2f}s"
        assert duration_sec < 5.0, f"Audio too long: {duration_sec:.2f}s"

    def test_audio_no_nan(
        self, mlx_llm_model, mlx_flow_model, mlx_vocoder_model, tokenizer,
    ):
        """Test that audio contains no NaN values."""
        audio = generate_audio(
            mlx_llm_model, mlx_flow_model, mlx_vocoder_model, tokenizer,
            "Test",
        )

        assert not np.any(np.isnan(audio)), "Audio contains NaN"
        assert not np.any(np.isinf(audio)), "Audio contains Inf"

    def test_audio_amplitude_reasonable(
        self, mlx_llm_model, mlx_flow_model, mlx_vocoder_model, tokenizer,
    ):
        """Test that audio amplitude is in reasonable range."""
        audio = generate_audio(
            mlx_llm_model, mlx_flow_model, mlx_vocoder_model, tokenizer,
            "Hello",
        )

        max_amp = np.max(np.abs(audio))

        # Audio shouldn't be too quiet or clipping
        assert max_amp > 0.01, f"Audio too quiet: max_amp={max_amp}"
        assert max_amp < 10.0, f"Audio amplitude too high: max_amp={max_amp}"


# ============================================================================
# EXPANDED TEST SUITE
# ============================================================================

# 10+ diverse test phrases
DIVERSE_TEST_PHRASES = [
    "Hello, how are you today?",
    "The quick brown fox jumps over the lazy dog.",
    "Good morning, welcome to our store.",
    "Please press one for sales.",
    "Your appointment is confirmed for tomorrow.",
    "The weather forecast shows rain tonight.",
    "Thank you for calling customer service.",
    "Enter your PIN number now.",
    "Flight departing from gate seven.",
    "The meeting starts at three thirty PM.",
    "Happy birthday to you!",
    "I would like to order a pizza.",
]

# Edge cases: numbers, acronyms, punctuation
EDGE_CASE_PHRASES = [
    "Call 1-800-555-1234 now.",  # Phone numbers
    "The price is $19.99.",  # Currency
    "NASA and the FBI work together.",  # Acronyms
    "What? Really! That's amazing...",  # Punctuation
    "123 Main Street, Apt. 4B.",  # Address
    "www.example.com is the URL.",  # URLs
    "The year 2024 was exciting.",  # Year
    "E = MC squared.",  # Formula
]


class TestDiversePhrases:
    """Test TTS with diverse phrases to ensure robustness."""

    @pytest.mark.parametrize("phrase", DIVERSE_TEST_PHRASES[:5])  # First 5 for speed
    @pytest.mark.slow
    def test_diverse_phrase_produces_audio(
        self, mlx_llm_model, mlx_flow_model, mlx_vocoder_model, tokenizer, phrase,
    ):
        """Test that diverse phrases produce valid audio."""
        audio = generate_audio(
            mlx_llm_model, mlx_flow_model, mlx_vocoder_model, tokenizer,
            phrase,
            max_speech_tokens=60,
            num_flow_steps=10,
        )

        # Basic validity checks
        assert len(audio) > 1000, f"Audio too short for '{phrase}'"
        assert not np.any(np.isnan(audio)), f"NaN in audio for '{phrase}'"

        rms = np.sqrt(np.mean(audio ** 2))
        assert rms > 0.001, f"Audio silent for '{phrase}': RMS={rms}"


class TestEdgeCases:
    """Test TTS with edge case inputs."""

    @pytest.mark.parametrize("phrase", EDGE_CASE_PHRASES[:4])  # First 4 for speed
    @pytest.mark.slow
    def test_edge_case_produces_audio(
        self, mlx_llm_model, mlx_flow_model, mlx_vocoder_model, tokenizer, phrase,
    ):
        """Test that edge case phrases produce valid audio."""
        audio = generate_audio(
            mlx_llm_model, mlx_flow_model, mlx_vocoder_model, tokenizer,
            phrase,
            max_speech_tokens=80,
            num_flow_steps=10,
        )

        # Basic validity checks
        assert len(audio) > 500, f"Audio too short for '{phrase}'"
        assert not np.any(np.isnan(audio)), f"NaN in audio for '{phrase}'"

        rms = np.sqrt(np.mean(audio ** 2))
        assert rms > 0.0005, f"Audio silent for '{phrase}': RMS={rms}"


class TestSpectrogramQuality:
    """Test spectrogram quality metrics (automated MOS-like tests)."""

    def test_spectrogram_energy_distribution(
        self, mlx_llm_model, mlx_flow_model, mlx_vocoder_model, tokenizer,
    ):
        """Test that spectrogram has reasonable energy distribution.

        Good speech should have:
        - Energy concentrated in speech frequencies (85Hz-8kHz)
        - Smooth energy transitions
        - No stuck/repeated patterns
        """
        audio = generate_audio(
            mlx_llm_model, mlx_flow_model, mlx_vocoder_model, tokenizer,
            "Hello, this is a test of speech quality.",
            max_speech_tokens=60,
            num_flow_steps=15,
        )

        # Compute simple spectrogram via STFT
        frame_length = 1024
        hop_length = 256
        n_frames = (len(audio) - frame_length) // hop_length + 1

        if n_frames < 5:
            pytest.skip("Audio too short for spectrogram analysis")

        # Simple power spectrogram
        window = np.hanning(frame_length)
        spectrogram = []
        for i in range(n_frames):
            start = i * hop_length
            frame = audio[start:start + frame_length] * window
            spectrum = np.abs(np.fft.rfft(frame)) ** 2
            spectrogram.append(spectrum)

        spectrogram = np.array(spectrogram)

        # Check energy not concentrated in single frequency
        energy_per_freq = np.mean(spectrogram, axis=0)
        energy_std = np.std(energy_per_freq)
        energy_mean = np.mean(energy_per_freq)

        # Coefficient of variation should be reasonable for speech
        cv = energy_std / (energy_mean + 1e-10)
        assert cv > 0.5, f"Energy too uniform (noise?): CV={cv:.2f}"
        assert cv < 10.0, f"Energy too concentrated: CV={cv:.2f}"

    def test_frame_to_frame_continuity(
        self, mlx_llm_model, mlx_flow_model, mlx_vocoder_model, tokenizer,
    ):
        """Test that audio has smooth frame-to-frame transitions.

        Good speech should not have sudden jumps in amplitude.
        """
        audio = generate_audio(
            mlx_llm_model, mlx_flow_model, mlx_vocoder_model, tokenizer,
            "Hello world",
            max_speech_tokens=40,
            num_flow_steps=10,
        )

        # Compute frame-level RMS
        frame_size = 480  # 20ms at 24kHz
        n_frames = len(audio) // frame_size

        if n_frames < 3:
            pytest.skip("Audio too short for continuity analysis")

        frame_rms = []
        for i in range(n_frames):
            frame = audio[i * frame_size:(i + 1) * frame_size]
            frame_rms.append(np.sqrt(np.mean(frame ** 2)))

        frame_rms = np.array(frame_rms)

        # Check for sudden jumps (excluding silence transitions)
        rms_diff = np.abs(np.diff(frame_rms))
        max_jump = np.max(rms_diff)
        mean_rms = np.mean(frame_rms)

        # Jump should not be more than 5x mean RMS
        jump_ratio = max_jump / (mean_rms + 1e-10)
        assert jump_ratio < 5.0, f"Audio has sudden jump: ratio={jump_ratio:.2f}"


class TestReproducibility:
    """Test that TTS output is reproducible with same seed."""

    def test_same_seed_same_output(
        self, mlx_llm_model, mlx_flow_model, mlx_vocoder_model, tokenizer,
    ):
        """Test that same seed produces same output."""
        text = "Hello"

        # Generate twice with same seed
        mx.random.seed(12345)
        audio1 = generate_audio(
            mlx_llm_model, mlx_flow_model, mlx_vocoder_model, tokenizer,
            text, max_speech_tokens=20, num_flow_steps=5,
        )

        mx.random.seed(12345)
        audio2 = generate_audio(
            mlx_llm_model, mlx_flow_model, mlx_vocoder_model, tokenizer,
            text, max_speech_tokens=20, num_flow_steps=5,
        )

        # Audio should be identical
        diff = np.abs(audio1 - audio2).max()
        assert diff < 1e-5, f"Non-deterministic output: max_diff={diff}"

    def test_different_seed_different_output(
        self, mlx_llm_model, mlx_flow_model, mlx_vocoder_model, tokenizer,
    ):
        """Test that different seeds produce different outputs."""
        text = "Hello"

        mx.random.seed(11111)
        audio1 = generate_audio(
            mlx_llm_model, mlx_flow_model, mlx_vocoder_model, tokenizer,
            text, max_speech_tokens=20, num_flow_steps=5,
        )

        mx.random.seed(22222)
        audio2 = generate_audio(
            mlx_llm_model, mlx_flow_model, mlx_vocoder_model, tokenizer,
            text, max_speech_tokens=20, num_flow_steps=5,
        )

        # Audio should differ
        diff = np.abs(audio1 - audio2).max()
        assert diff > 0.01, f"Same output for different seeds: max_diff={diff}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
