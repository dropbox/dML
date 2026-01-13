#!/usr/bin/env python3
"""
TTS Quality Test Suite with proper CJK handling and latency metrics.
Tests XTTS v2 and Kokoro for quality comparison.

NOTE: This is a legacy reference script. For current testing, use pytest tests in tests/
"""

import torch
import os
import sys
import json
import time
from datetime import datetime
from difflib import SequenceMatcher

# Use relative paths from script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))

# Patch torch.load for PyTorch 2.6 compatibility
original_load = torch.load
def patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = patched_load

# Test sentences from real coding context
TEST_SENTENCES = {
    'en': [
        "The function was refactored to improve performance.",
        "Starting the build process now.",
        "All tests passed successfully.",
        "Error detected in authentication module.",
    ],
    'ja': [
        "関数のリファクタリングが完了しました。",
        "ビルドプロセスを開始します。",
        "すべてのテストが成功しました。",
        "認証モジュールでエラーが検出されました。",
    ],
    'zh': [
        "函数重构已完成。",
        "正在启动构建过程。",
        "所有测试都通过了。",
        "认证模块检测到错误。",
    ],
    'es': [
        "La función ha sido refactorizada.",
        "Iniciando el proceso de compilación.",
        "Todas las pruebas pasaron exitosamente.",
        "Error detectado en el módulo de autenticación.",
    ],
    'de': [
        "Die Funktion wurde refaktoriert.",
        "Starte den Build-Prozess.",
        "Alle Tests erfolgreich bestanden.",
        "Fehler im Authentifizierungsmodul erkannt.",
    ],
}

def character_similarity(s1, s2):
    """Calculate character-level similarity (better for CJK)."""
    # Normalize whitespace and punctuation
    s1 = ''.join(s1.lower().split())
    s2 = ''.join(s2.lower().split())
    return SequenceMatcher(None, s1, s2).ratio() * 100

def word_similarity(s1, s2):
    """Calculate word-level overlap."""
    w1 = set(s1.lower().split())
    w2 = set(s2.lower().split())
    if not w1:
        return 0
    return len(w1 & w2) / len(w1) * 100

class TTSTester:
    def __init__(self):
        self.xtts = None
        self.kokoro = None
        self.whisper = None
        self.ref_audio = '/tmp/ref_audio.wav'

    def load_xtts(self):
        """Load XTTS v2 model."""
        if self.xtts is None:
            from TTS.api import TTS
            print("Loading XTTS v2...")
            self.xtts = TTS('tts_models/multilingual/multi-dataset/xtts_v2', progress_bar=False)
        return self.xtts

    def load_kokoro(self):
        """Load Kokoro TTS."""
        if self.kokoro is None:
            try:
                import kokoro
                print("Loading Kokoro TTS...")
                self.kokoro = kokoro
            except ImportError:
                print("Kokoro not installed")
                return None
        return self.kokoro

    def load_whisper(self, model='large-v3'):
        """Load Whisper for STT verification. Default to large-v3 for SOTA."""
        if self.whisper is None:
            import whisper
            print(f"Loading Whisper ({model})...")
            self.whisper = whisper.load_model(model)
        return self.whisper

    def ensure_ref_audio(self):
        """Create reference audio for XTTS voice cloning."""
        if not os.path.exists(self.ref_audio):
            print("Creating reference audio...")
            # Use a simple tone as reference
            import subprocess
            subprocess.run([
                'python', os.path.join(PROJECT_DIR, 'scripts/kokoro_tts.py'),
                'Hello world, this is a reference audio for voice cloning.',
                '-o', self.ref_audio, '-l', 'en'
            ], capture_output=True)

    def test_xtts(self, text, lang, output_path):
        """Test XTTS v2 synthesis."""
        tts = self.load_xtts()
        self.ensure_ref_audio()

        start = time.time()
        tts.tts_to_file(
            text=text,
            speaker_wav=self.ref_audio,
            language=lang if lang != 'zh' else 'zh-cn',
            file_path=output_path
        )
        latency = time.time() - start

        # Get audio duration
        import torchaudio
        audio, sr = torchaudio.load(output_path)
        duration = audio.shape[1] / sr
        rtf = latency / duration

        return {'latency': latency, 'duration': duration, 'rtf': rtf}

    def test_kokoro(self, text, lang, output_path):
        """Test Kokoro synthesis (EN/JA only)."""
        if lang not in ['en', 'ja']:
            return None

        start = time.time()
        import subprocess
        result = subprocess.run([
            'python', os.path.join(PROJECT_DIR, 'scripts/kokoro_tts.py'),
            text, '-o', output_path, '-l', lang
        ], capture_output=True, text=True)

        if result.returncode != 0:
            return None

        latency = time.time() - start

        # Get audio duration
        import torchaudio
        audio, sr = torchaudio.load(output_path)
        duration = audio.shape[1] / sr
        rtf = latency / duration

        return {'latency': latency, 'duration': duration, 'rtf': rtf}

    def transcribe(self, audio_path, lang):
        """Transcribe audio with Whisper."""
        whisper = self.load_whisper()
        whisper_lang = lang.split('-')[0]
        result = whisper.transcribe(audio_path, language=whisper_lang)
        return result['text'].strip()

    def run_comparison(self, lang='ja'):
        """Compare XTTS v2 vs Kokoro for a language."""
        if lang not in TEST_SENTENCES:
            print(f"No test sentences for {lang}")
            return

        sentences = TEST_SENTENCES[lang]
        results = []

        print(f"\n{'='*60}")
        print(f"COMPARISON: XTTS v2 vs Kokoro ({lang.upper()})")
        print(f"{'='*60}")

        for i, text in enumerate(sentences):
            print(f"\n[{i+1}/{len(sentences)}] {text[:50]}...")

            # Test XTTS
            xtts_path = f'/tmp/xtts_test_{lang}_{i}.wav'
            xtts_metrics = self.test_xtts(text, lang, xtts_path)
            xtts_transcription = self.transcribe(xtts_path, lang)

            # Calculate similarity (character-level for CJK)
            if lang in ['ja', 'zh', 'ko']:
                xtts_sim = character_similarity(text, xtts_transcription)
            else:
                xtts_sim = word_similarity(text, xtts_transcription)

            print(f"  XTTS: {xtts_sim:.0f}% acc, {xtts_metrics['latency']:.2f}s latency, RTF={xtts_metrics['rtf']:.2f}")

            # Test Kokoro (if available for this language)
            kokoro_path = f'/tmp/kokoro_test_{lang}_{i}.wav'
            kokoro_metrics = self.test_kokoro(text, lang, kokoro_path)

            if kokoro_metrics:
                kokoro_transcription = self.transcribe(kokoro_path, lang)
                if lang in ['ja', 'zh', 'ko']:
                    kokoro_sim = character_similarity(text, kokoro_transcription)
                else:
                    kokoro_sim = word_similarity(text, kokoro_transcription)
                print(f"  Kokoro: {kokoro_sim:.0f}% acc, {kokoro_metrics['latency']:.2f}s latency, RTF={kokoro_metrics['rtf']:.2f}")
            else:
                kokoro_sim = None
                kokoro_transcription = None
                print(f"  Kokoro: N/A for {lang}")

            results.append({
                'text': text,
                'xtts': {
                    'transcription': xtts_transcription,
                    'similarity': xtts_sim,
                    **xtts_metrics
                },
                'kokoro': {
                    'transcription': kokoro_transcription,
                    'similarity': kokoro_sim,
                    **(kokoro_metrics or {})
                } if kokoro_metrics else None
            })

        # Summary
        print(f"\n{'='*60}")
        print(f"SUMMARY ({lang.upper()})")
        print(f"{'='*60}")

        xtts_avg_sim = sum(r['xtts']['similarity'] for r in results) / len(results)
        xtts_avg_latency = sum(r['xtts']['latency'] for r in results) / len(results)
        xtts_avg_rtf = sum(r['xtts']['rtf'] for r in results) / len(results)

        print(f"XTTS v2: Avg accuracy={xtts_avg_sim:.0f}%, Avg latency={xtts_avg_latency:.2f}s, Avg RTF={xtts_avg_rtf:.2f}")

        kokoro_results = [r for r in results if r['kokoro']]
        if kokoro_results:
            kokoro_avg_sim = sum(r['kokoro']['similarity'] for r in kokoro_results) / len(kokoro_results)
            kokoro_avg_latency = sum(r['kokoro']['latency'] for r in kokoro_results) / len(kokoro_results)
            kokoro_avg_rtf = sum(r['kokoro']['rtf'] for r in kokoro_results) / len(kokoro_results)
            print(f"Kokoro:  Avg accuracy={kokoro_avg_sim:.0f}%, Avg latency={kokoro_avg_latency:.2f}s, Avg RTF={kokoro_avg_rtf:.2f}")

            if kokoro_avg_sim > xtts_avg_sim:
                print(f"\n>>> WINNER: Kokoro (better accuracy)")
            elif kokoro_avg_sim == xtts_avg_sim and kokoro_avg_rtf < xtts_avg_rtf:
                print(f"\n>>> WINNER: Kokoro (same accuracy, faster)")
            else:
                print(f"\n>>> WINNER: XTTS v2")

        return results

def main():
    tester = TTSTester()

    # Run comparison for Japanese
    ja_results = tester.run_comparison('ja')

    # Run comparison for English
    en_results = tester.run_comparison('en')

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'japanese': ja_results,
        'english': en_results
    }

    output_path = os.path.join(PROJECT_DIR, f'reports/main/tts_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")

if __name__ == '__main__':
    main()
