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
DashVoice Voice Fingerprint Database (Layer 2)

Pre-computes and stores speaker embeddings for all known DashVoice voices.
Used to recognize and optionally filter:
1. Our own generated audio (from this instance)
2. Audio from other DashVoice instances (default voices)
3. Custom cloned voices

Embedding Model: Resemblyzer (MIT license)
- 256-dimensional speaker embeddings
- Trained on VoxCeleb
- Fast inference on CPU
"""

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class VoiceFingerprint:
    """A voice fingerprint with metadata."""

    name: str
    embedding: np.ndarray
    source: str  # "kokoro", "cosyvoice2", "custom"
    created_at: str = ""
    reference_text: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "embedding": self.embedding.tolist(),
            "source": self.source,
            "created_at": self.created_at,
            "reference_text": self.reference_text,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VoiceFingerprint":
        return cls(
            name=data["name"],
            embedding=np.array(data["embedding"]),
            source=data["source"],
            created_at=data.get("created_at", ""),
            reference_text=data.get("reference_text", ""),
        )


class VoiceDatabase:
    """
    Database of voice fingerprints for all known DashVoice voices.

    Used for Layer 2 self-voice filtering: recognize any DashVoice-generated
    audio (our instance or others) and flag it for optional filtering.
    """

    # Standard reference text for generating voice samples
    REFERENCE_TEXT = "The quick brown fox jumps over the lazy dog. How are you today?"

    # Kokoro voice names (all available voices from prince-canuma/Kokoro-82M)
    # Prefixes: af=American Female, am=American Male, bf=British Female, bm=British Male
    #           ef=European Female, em=European Male, ff=French Female, hf=Hindi Female, hm=Hindi Male
    #           if=Italian Female, im=Italian Male, jf=Japanese Female, jm=Japanese Male
    #           pf=Portuguese Female, pm=Portuguese Male, zf=Chinese Female, zm=Chinese Male
    KOKORO_VOICES = [
        # American Female
        "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica",
        "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
        # American Male
        "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
        "am_michael", "am_onyx", "am_puck", "am_santa",
        # British Female
        "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
        # British Male
        "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
        # European
        "ef_dora", "em_alex", "em_santa",
        # French
        "ff_siwis",
        # Hindi
        "hf_alpha", "hf_beta", "hm_omega", "hm_psi",
        # Italian
        "if_sara", "im_nicola",
        # Japanese
        "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo",
        # Portuguese
        "pf_dora", "pm_alex", "pm_santa",
        # Chinese
        "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
        "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang",
    ]

    def __init__(self, db_path: str = "models/voices/voice_database.json"):
        """Initialize voice database.

        Args:
            db_path: Path to store the voice fingerprint database
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.encoder = None  # Lazy load
        self.fingerprints: dict[str, VoiceFingerprint] = {}

        # Load existing database if available
        if self.db_path.exists():
            self._load_database()

    def _load_encoder(self):
        """Load the speaker embedding model."""
        if self.encoder is None:
            from resemblyzer import VoiceEncoder

            self.encoder = VoiceEncoder()
            print("Loaded Resemblyzer speaker encoder")

    def _load_database(self):
        """Load fingerprints from disk."""
        try:
            with open(self.db_path) as f:
                data = json.load(f)
            for name, fp_data in data.get("fingerprints", {}).items():
                self.fingerprints[name] = VoiceFingerprint.from_dict(fp_data)
            print(f"Loaded {len(self.fingerprints)} voice fingerprints from {self.db_path}")
        except Exception as e:
            print(f"Warning: Could not load voice database: {e}")

    def _save_database(self):
        """Save fingerprints to disk."""
        data = {
            "version": "1.0",
            "fingerprints": {name: fp.to_dict() for name, fp in self.fingerprints.items()},
        }
        with open(self.db_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(self.fingerprints)} voice fingerprints to {self.db_path}")

    def extract_embedding(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Extract speaker embedding from audio.

        Args:
            audio: Audio waveform (mono, float32)
            sample_rate: Sample rate (default 16000)

        Returns:
            256-dimensional speaker embedding
        """
        self._load_encoder()

        # Resemblyzer expects 16kHz
        if sample_rate != 16000:
            import librosa

            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)

        # Ensure float32
        audio = audio.astype(np.float32)

        return self.encoder.embed_utterance(audio)

    def add_fingerprint(
        self,
        name: str,
        audio: np.ndarray,
        sample_rate: int = 16000,
        source: str = "custom",
        reference_text: str = "",
    ):
        """Add a voice fingerprint to the database.

        Args:
            name: Unique name for this voice
            audio: Audio waveform (mono, float32)
            sample_rate: Sample rate
            source: Source of voice ("kokoro", "cosyvoice2", "custom")
            reference_text: Text used to generate the audio
        """
        from datetime import datetime

        embedding = self.extract_embedding(audio, sample_rate)

        self.fingerprints[name] = VoiceFingerprint(
            name=name,
            embedding=embedding,
            source=source,
            created_at=datetime.now().isoformat(),
            reference_text=reference_text,
        )

        self._save_database()
        print(f"Added voice fingerprint: {name}")

    def is_dashvoice(
        self, audio: np.ndarray, sample_rate: int = 16000, threshold: float = 0.85,
    ) -> tuple[bool, str | None, float]:
        """Check if audio matches any known DashVoice voice.

        Args:
            audio: Audio waveform to check
            sample_rate: Sample rate
            threshold: Similarity threshold (0.85 = high confidence)

        Returns:
            Tuple of (is_match, best_match_name, similarity_score)
        """
        if not self.fingerprints:
            return False, None, 0.0

        embedding = self.extract_embedding(audio, sample_rate)

        best_match = None
        best_score = 0.0

        for name, fp in self.fingerprints.items():
            # Cosine similarity
            similarity = float(
                np.dot(embedding, fp.embedding)
                / (np.linalg.norm(embedding) * np.linalg.norm(fp.embedding) + 1e-8),
            )

            if similarity > best_score:
                best_score = similarity
                best_match = name

        is_match = best_score >= threshold
        return is_match, best_match if is_match else None, best_score

    def generate_kokoro_fingerprints(self, voices: list[str] | None = None):
        """Generate fingerprints for Kokoro default voices.

        Args:
            voices: List of voice names to process, or None for all voices
        """
        import glob
        import tempfile

        import soundfile as sf
        from mlx_audio.tts.generate import generate_audio

        print("Generating Kokoro voice fingerprints...")

        voices_to_process = voices or self.KOKORO_VOICES
        generated = 0
        skipped = 0

        for voice in voices_to_process:
            fingerprint_name = f"kokoro_{voice}"
            if fingerprint_name in self.fingerprints:
                print(f"  Skipping {voice} (already exists)")
                skipped += 1
                continue

            print(f"  Processing {voice}...")
            try:
                # Generate reference audio to temp file
                with tempfile.TemporaryDirectory() as tmpdir:
                    prefix = os.path.join(tmpdir, "voice")
                    generate_audio(
                        text=self.REFERENCE_TEXT,
                        voice=voice,
                        play=False,
                        verbose=False,
                        file_prefix=prefix,
                    )

                    # Find the generated wav file (format: voice_000.wav)
                    wav_files = glob.glob(os.path.join(tmpdir, "voice_*.wav"))
                    if not wav_files:
                        raise FileNotFoundError("No audio file generated")

                    # Read audio
                    audio, sample_rate = sf.read(wav_files[0])

                    # Add fingerprint
                    self.add_fingerprint(
                        name=fingerprint_name,
                        audio=audio,
                        sample_rate=sample_rate,
                        source="kokoro",
                        reference_text=self.REFERENCE_TEXT,
                    )
                    generated += 1
            except Exception as e:
                print(f"    Warning: Failed to generate {voice}: {e}")

        total = len([k for k in self.fingerprints if "kokoro" in k])
        print(f"Generated {generated} new fingerprints, skipped {skipped}, total Kokoro: {total}")

    # Default speaker seeds for CosyVoice2
    # These create reproducible "default" voices using deterministic random embeddings
    COSYVOICE2_DEFAULT_SEEDS = list(range(10))  # Seeds 0-9 for 10 default voices

    def generate_cosyvoice_fingerprints(
        self, seeds: list[int] | None = None, force: bool = False,
    ):
        """Generate fingerprints for CosyVoice2 default speakers.

        CosyVoice2 doesn't have pre-defined voices like Kokoro. Instead, it uses
        speaker embeddings extracted from reference audio. For fingerprinting,
        we create "default" voices using deterministic random speaker embeddings
        with specific seeds.

        Args:
            seeds: List of seeds to use for speaker embeddings (default: 0-9)
            force: If True, regenerate even if fingerprint exists
        """
        print("Generating CosyVoice2 voice fingerprints...")

        # Check if CosyVoice2 is available
        try:
            from tools.pytorch_to_mlx.converters.models.cosyvoice2 import (
                CosyVoice2Model,
            )
        except ImportError:
            print("  Error: CosyVoice2 model not available")
            return

        # Check if model exists
        model_path = CosyVoice2Model.get_default_model_path()
        if not model_path.exists():
            print(f"  Error: CosyVoice2 model not found at {model_path}")
            return

        seeds_to_process = seeds or self.COSYVOICE2_DEFAULT_SEEDS
        generated = 0
        skipped = 0

        # Load model once
        print("  Loading CosyVoice2 model...")
        model = CosyVoice2Model.from_pretrained(str(model_path))

        for seed in seeds_to_process:
            fingerprint_name = f"cosyvoice2_speaker_{seed}"

            if fingerprint_name in self.fingerprints and not force:
                print(f"  Skipping seed {seed} (already exists)")
                skipped += 1
                continue

            print(f"  Processing seed {seed}...")
            try:
                # Generate speaker embedding with specific seed
                speaker_embedding = model.tokenizer.random_speaker_embedding(seed=seed)

                # Generate reference audio
                audio = model.synthesize_text(
                    text=self.REFERENCE_TEXT,
                    speaker_embedding=speaker_embedding,
                    max_tokens=500,
                    num_flow_steps=10,
                )

                # Convert to numpy (CosyVoice2 outputs 24kHz)
                import numpy as np

                audio_np = np.array(audio).squeeze().astype(np.float32)
                sample_rate = 24000

                # Add fingerprint
                self.add_fingerprint(
                    name=fingerprint_name,
                    audio=audio_np,
                    sample_rate=sample_rate,
                    source="cosyvoice2",
                    reference_text=self.REFERENCE_TEXT,
                )
                generated += 1

            except Exception as e:
                print(f"    Warning: Failed to generate seed {seed}: {e}")

        total = len([k for k in self.fingerprints if "cosyvoice2" in k])
        print(
            f"Generated {generated} new fingerprints, skipped {skipped}, "
            f"total CosyVoice2: {total}",
        )

    def list_fingerprints(self) -> list[str]:
        """List all available voice fingerprints."""
        return list(self.fingerprints.keys())

    def get_fingerprint(self, name: str) -> VoiceFingerprint | None:
        """Get a specific voice fingerprint."""
        return self.fingerprints.get(name)

    def compare_voices(self, name1: str, name2: str) -> float:
        """Compare similarity between two stored voices."""
        fp1 = self.fingerprints.get(name1)
        fp2 = self.fingerprints.get(name2)

        if fp1 is None or fp2 is None:
            raise ValueError(f"Voice not found: {name1 if fp1 is None else name2}")

        return float(
            np.dot(fp1.embedding, fp2.embedding)
            / (np.linalg.norm(fp1.embedding) * np.linalg.norm(fp2.embedding) + 1e-8),
        )


def main():
    """Generate voice fingerprints for all default voices."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate voice fingerprints")
    parser.add_argument(
        "--kokoro-only", action="store_true", help="Only generate Kokoro fingerprints",
    )
    parser.add_argument(
        "--cosyvoice-only",
        action="store_true",
        help="Only generate CosyVoice2 fingerprints",
    )
    parser.add_argument(
        "--force", action="store_true", help="Regenerate existing fingerprints",
    )
    args = parser.parse_args()

    db = VoiceDatabase()

    print("=" * 60)
    print("DashVoice Voice Fingerprint Generator")
    print("=" * 60)

    if not args.cosyvoice_only:
        # Generate Kokoro fingerprints
        db.generate_kokoro_fingerprints()

    if not args.kokoro_only:
        # Generate CosyVoice2 fingerprints
        print()
        db.generate_cosyvoice_fingerprints(force=args.force)

    # Summary
    print("\n" + "=" * 60)
    print("Voice Database Summary")
    print("=" * 60)
    print(f"Total fingerprints: {len(db.fingerprints)}")

    # Group by source
    sources: dict[str, list[str]] = {}
    for name in sorted(db.fingerprints.keys()):
        fp = db.fingerprints[name]
        if fp.source not in sources:
            sources[fp.source] = []
        sources[fp.source].append(name)

    for source, names in sorted(sources.items()):
        print(f"\n{source.upper()} ({len(names)} voices):")
        for name in names[:5]:  # Show first 5
            print(f"  - {name}")
        if len(names) > 5:
            print(f"  ... and {len(names) - 5} more")


if __name__ == "__main__":
    main()
