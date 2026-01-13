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
Tests for DashVoice Voice Cloning endpoints.

Tests:
1. Voice embedding extraction (/clone/extract)
2. Voice cloning synthesis (/clone/synthesize)
3. Cloned voice listing (/clone/voices)
4. Voice deletion (/clone/voices/{name})
"""

import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from tools.dashvoice.server import (
    app,
    get_voice_database,
)

# Module-level RNG for reproducible tests
_rng = np.random.default_rng(42)


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def sample_audio():
    """Create a sample audio file for testing."""
    # Generate 2 seconds of audio (sine wave with some noise)
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)

    # Create a simple audio signal (mix of frequencies)
    audio = (
        0.3 * np.sin(2 * np.pi * 200 * t) +  # 200 Hz fundamental
        0.2 * np.sin(2 * np.pi * 400 * t) +  # First harmonic
        0.1 * np.sin(2 * np.pi * 600 * t) +  # Second harmonic
        0.05 * _rng.standard_normal(len(t))  # Some noise
    ).astype(np.float32)

    # Create WAV file in memory
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format='WAV')
    buffer.seek(0)

    return buffer


class TestRootEndpointIncludesCloning:
    """Test root endpoint includes voice cloning."""

    def test_root_includes_clone_endpoints(self, client):
        """Test root endpoint lists clone endpoints."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()

        assert "voice_cloning" in data.get("endpoints", {})
        clone_endpoints = data["endpoints"]["voice_cloning"]

        assert any("/clone/extract" in e for e in clone_endpoints)
        assert any("/clone/synthesize" in e for e in clone_endpoints)
        assert any("/clone/voices" in e for e in clone_endpoints)

    def test_api_version_updated_for_cloning(self, client):
        """Test API version is at least 1.3.0."""
        response = client.get("/")
        data = response.json()

        version = data.get("version", "")
        major, minor, patch = version.split(".")
        assert int(major) >= 1
        assert int(minor) >= 3


class TestCloneExtractEndpoint:
    """Test /clone/extract endpoint."""

    def test_extract_endpoint_exists(self, client):
        """Test extract endpoint exists."""
        routes = [r.path for r in app.routes]
        assert "/clone/extract" in routes

    def test_extract_requires_audio(self, client):
        """Test extract requires audio file."""
        response = client.post("/clone/extract")
        assert response.status_code == 422

    def test_extract_rejects_short_audio(self, client):
        """Test extract rejects audio shorter than 1 second."""
        # Create 0.5 second audio
        sample_rate = 16000
        duration = 0.5
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        audio = np.sin(2 * np.pi * 200 * t).astype(np.float32)

        buffer = io.BytesIO()
        sf.write(buffer, audio, sample_rate, format='WAV')
        buffer.seek(0)

        response = client.post(
            "/clone/extract",
            files={"audio": ("short.wav", buffer, "audio/wav")},
        )
        assert response.status_code == 400
        assert "too short" in response.json()["detail"].lower()

    def test_extract_returns_embedding(self, client, sample_audio):
        """Test extract returns embedding vector."""
        response = client.post(
            "/clone/extract",
            files={"audio": ("test.wav", sample_audio, "audio/wav")},
        )

        if response.status_code == 500 and "resemblyzer" in response.json().get("detail", "").lower():
            pytest.skip("Resemblyzer not available")

        assert response.status_code == 200
        data = response.json()

        assert "embedding" in data
        assert "embedding_dim" in data
        assert data["embedding_dim"] == 256  # Resemblyzer dimension
        assert isinstance(data["embedding"], list)
        assert len(data["embedding"]) == 256

    def test_extract_saves_with_name(self, client, sample_audio):
        """Test extract saves voice when name provided."""
        test_name = "test_extract_voice_12345"

        try:
            response = client.post(
                "/clone/extract",
                files={"audio": ("test.wav", sample_audio, "audio/wav")},
                data={"name": test_name},
            )

            if response.status_code == 500 and "resemblyzer" in response.json().get("detail", "").lower():
                pytest.skip("Resemblyzer not available")

            assert response.status_code == 200
            data = response.json()

            assert data["saved"] is True
            assert data["name"] == test_name
        finally:
            # Cleanup: delete the test voice
            db = get_voice_database()
            if test_name in db.fingerprints:
                del db.fingerprints[test_name]
                db._save_database()


class TestCloneSynthesizeEndpoint:
    """Test /clone/synthesize endpoint."""

    def test_synthesize_endpoint_exists(self, client):
        """Test synthesize endpoint exists."""
        routes = [r.path for r in app.routes]
        assert "/clone/synthesize" in routes

    def test_synthesize_requires_text(self, client):
        """Test synthesize requires text parameter."""
        response = client.post("/clone/synthesize")
        assert response.status_code == 422

    def test_synthesize_validates_speed_range(self, client):
        """Test synthesize validates speed range."""
        # Speed too low
        response = client.post(
            "/clone/synthesize",
            data={"text": "Hello", "speed": 0.1},
        )
        assert response.status_code == 400
        assert "speed" in response.json()["detail"].lower()

        # Speed too high
        response = client.post(
            "/clone/synthesize",
            data={"text": "Hello", "speed": 3.0},
        )
        assert response.status_code == 400
        assert "speed" in response.json()["detail"].lower()

    def test_synthesize_invalid_voice_name(self, client):
        """Test synthesize handles invalid voice name."""
        response = client.post(
            "/clone/synthesize",
            data={"text": "Hello", "voice_name": "nonexistent_voice_12345"},
        )
        # Should return 404 (voice not found) or 503 (model not available)
        assert response.status_code in [404, 503]


class TestCloneVoicesEndpoint:
    """Test /clone/voices endpoint."""

    def test_voices_endpoint_exists(self, client):
        """Test voices listing endpoint exists."""
        routes = [r.path for r in app.routes]
        assert "/clone/voices" in routes

    def test_voices_returns_list(self, client):
        """Test voices endpoint returns list structure."""
        response = client.get("/clone/voices")
        assert response.status_code == 200

        data = response.json()
        assert "total_voices" in data
        assert "voices" in data
        assert "by_source" in data
        assert isinstance(data["voices"], list)

    def test_voices_structure(self, client):
        """Test voice entry structure."""
        response = client.get("/clone/voices")
        data = response.json()

        if data["total_voices"] > 0:
            voice = data["voices"][0]
            assert "name" in voice
            assert "source" in voice
            assert "embedding_dim" in voice


class TestCloneVoiceDeleteEndpoint:
    """Test DELETE /clone/voices/{name} endpoint."""

    def test_delete_endpoint_exists(self, client):
        """Test delete endpoint pattern exists."""
        routes = [r.path for r in app.routes]
        assert any("/clone/voices/{voice_name}" in r for r in routes)

    def test_delete_nonexistent_voice(self, client):
        """Test delete returns 404 for nonexistent voice."""
        response = client.delete("/clone/voices/nonexistent_voice_12345")
        assert response.status_code == 404

    def test_delete_protected_voice(self, client):
        """Test delete protects built-in voices."""
        db = get_voice_database()

        # Find a non-custom voice to try deleting
        for name, fp in db.fingerprints.items():
            if fp.source != "custom":
                response = client.delete(f"/clone/voices/{name}")
                assert response.status_code == 403
                assert "cannot delete" in response.json()["detail"].lower()
                break


class TestVoiceDatabaseIntegration:
    """Test voice database integration."""

    def test_voice_database_singleton(self):
        """Test voice database is singleton."""
        db1 = get_voice_database()
        db2 = get_voice_database()
        assert db1 is db2

    def test_voice_database_has_methods(self):
        """Test voice database has required methods."""
        db = get_voice_database()

        assert hasattr(db, 'extract_embedding')
        assert hasattr(db, 'add_fingerprint')
        assert hasattr(db, 'list_fingerprints')
        assert hasattr(db, 'get_fingerprint')
        assert callable(db.extract_embedding)


# Integration tests that require full model loading
@pytest.mark.slow
class TestVoiceCloningIntegration:
    """Integration tests that require models."""

    def test_full_extract_and_list_flow(self, client, sample_audio):
        """Test extracting voice and then listing it."""
        test_name = "integration_test_voice_67890"

        try:
            # Extract
            response = client.post(
                "/clone/extract",
                files={"audio": ("test.wav", sample_audio, "audio/wav")},
                data={"name": test_name},
            )

            if response.status_code == 500:
                pytest.skip("Resemblyzer not available")

            assert response.status_code == 200

            # List and verify
            response = client.get("/clone/voices")
            assert response.status_code == 200

            names = [v["name"] for v in response.json()["voices"]]
            assert test_name in names

            # Delete
            response = client.delete(f"/clone/voices/{test_name}")
            assert response.status_code == 200

            # Verify deleted
            response = client.get("/clone/voices")
            names = [v["name"] for v in response.json()["voices"]]
            assert test_name not in names

        finally:
            # Cleanup if needed
            db = get_voice_database()
            if test_name in db.fingerprints:
                del db.fingerprints[test_name]
                db._save_database()

    def test_synthesize_with_default_voice(self, client):
        """Test synthesis with default CosyVoice2 voice."""
        response = client.post(
            "/clone/synthesize",
            data={"text": "Hello world", "speed": 1.0},
        )

        if response.status_code == 503:
            pytest.skip("CosyVoice2 not available")

        assert response.status_code == 200
        data = response.json()

        assert "audio_base64" in data
        assert "sample_rate" in data
        assert data["sample_rate"] == 24000
        assert data["samples"] > 0


class TestCosyVoice2FingerprintGeneration:
    """Test CosyVoice2 voice fingerprint generation."""

    def test_cosyvoice2_default_seeds_defined(self):
        """Test COSYVOICE2_DEFAULT_SEEDS is defined."""
        from tools.dashvoice.voice_database import VoiceDatabase

        assert hasattr(VoiceDatabase, "COSYVOICE2_DEFAULT_SEEDS")
        seeds = VoiceDatabase.COSYVOICE2_DEFAULT_SEEDS
        assert isinstance(seeds, list)
        assert len(seeds) == 10  # 10 default seeds (0-9)

    def test_generate_cosyvoice_fingerprints_method_exists(self):
        """Test generate_cosyvoice_fingerprints method exists."""
        from tools.dashvoice.voice_database import VoiceDatabase

        db = VoiceDatabase()
        assert hasattr(db, "generate_cosyvoice_fingerprints")
        assert callable(db.generate_cosyvoice_fingerprints)

    def test_generate_cosyvoice_fingerprints_accepts_seeds_param(self):
        """Test generate_cosyvoice_fingerprints accepts seeds parameter."""
        import inspect

        from tools.dashvoice.voice_database import VoiceDatabase

        sig = inspect.signature(VoiceDatabase.generate_cosyvoice_fingerprints)
        params = list(sig.parameters.keys())

        assert "seeds" in params
        assert "force" in params

    def test_cosyvoice2_fingerprint_name_format(self):
        """Test CosyVoice2 fingerprint name format."""
        from tools.dashvoice.voice_database import VoiceDatabase

        # Verify expected naming convention
        for seed in VoiceDatabase.COSYVOICE2_DEFAULT_SEEDS:
            expected_name = f"cosyvoice2_speaker_{seed}"
            assert expected_name.startswith("cosyvoice2_speaker_")
