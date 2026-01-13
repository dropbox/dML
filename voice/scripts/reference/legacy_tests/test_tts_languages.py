#!/usr/bin/env python3
"""
Comprehensive TTS/STT/Translation test for top 20 world languages.
Tests: TTS -> STT roundtrip and translation quality.

NOTE: This is a legacy reference script. For current testing, use pytest tests in tests/
"""

import torch
import os
import sys
import json
from datetime import datetime

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

# XTTS v2 supported languages with ISO codes
XTTS_LANGUAGES = {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'pt': 'Portuguese',
    'pl': 'Polish',
    'tr': 'Turkish',
    'ru': 'Russian',
    'nl': 'Dutch',
    'cs': 'Czech',
    'ar': 'Arabic',
    'zh-cn': 'Chinese',
    'hu': 'Hungarian',
    'ko': 'Korean',
    'ja': 'Japanese',
    'hi': 'Hindi',
}

# Real test sentences (from coding context)
TEST_SENTENCES = {
    'en': "The function was successfully refactored to improve performance and reduce memory usage.",
    'es': "La función fue refactorizada con éxito para mejorar el rendimiento y reducir el uso de memoria.",
    'fr': "La fonction a été refactorisée avec succès pour améliorer les performances et réduire l'utilisation de la mémoire.",
    'de': "Die Funktion wurde erfolgreich refaktoriert, um die Leistung zu verbessern und den Speicherverbrauch zu reduzieren.",
    'it': "La funzione è stata refactorizzata con successo per migliorare le prestazioni e ridurre l'utilizzo della memoria.",
    'pt': "A função foi refatorada com sucesso para melhorar o desempenho e reduzir o uso de memória.",
    'pl': "Funkcja została pomyślnie zrefaktoryzowana w celu poprawy wydajności i zmniejszenia zużycia pamięci.",
    'tr': "Fonksiyon, performansı artırmak ve bellek kullanımını azaltmak için başarıyla yeniden düzenlendi.",
    'ru': "Функция была успешно рефакторизована для улучшения производительности и уменьшения использования памяти.",
    'nl': "De functie is succesvol gerefactored om de prestaties te verbeteren en het geheugengebruik te verminderen.",
    'cs': "Funkce byla úspěšně refaktorována pro zlepšení výkonu a snížení využití paměti.",
    'ar': "تمت إعادة هيكلة الوظيفة بنجاح لتحسين الأداء وتقليل استخدام الذاكرة.",
    'zh-cn': "该函数已成功重构，以提高性能并减少内存使用。",
    'hu': "A funkciót sikeresen átdolgoztuk a teljesítmény javítása és a memóriahasználat csökkentése érdekében.",
    'ko': "함수가 성능을 개선하고 메모리 사용량을 줄이기 위해 성공적으로 리팩토링되었습니다.",
    'ja': "関数はパフォーマンスを向上させ、メモリ使用量を削減するために正常にリファクタリングされました。",
    'hi': "फ़ंक्शन को प्रदर्शन में सुधार और मेमोरी उपयोग को कम करने के लिए सफलतापूर्वक रिफैक्टर किया गया था।",
}

def load_tts():
    """Load XTTS v2 model."""
    from TTS.api import TTS
    print("Loading XTTS v2 model...")
    tts = TTS('tts_models/multilingual/multi-dataset/xtts_v2', progress_bar=False)
    return tts

def load_stt():
    """Load Whisper model."""
    import whisper
    print("Loading Whisper model...")
    return whisper.load_model('base')

def test_tts_stt(tts, stt, ref_audio):
    """Test TTS -> STT roundtrip for all languages."""
    results = {}

    for lang_code, lang_name in XTTS_LANGUAGES.items():
        if lang_code not in TEST_SENTENCES:
            continue

        text = TEST_SENTENCES[lang_code]
        wav_path = f'/tmp/tts_test_{lang_code}.wav'

        print(f"\n[{lang_name}] Testing...")
        print(f"  Input: {text[:60]}...")

        try:
            # TTS
            tts.tts_to_file(
                text=text,
                speaker_wav=ref_audio,
                language=lang_code,
                file_path=wav_path
            )

            # STT
            whisper_lang = lang_code.split('-')[0]  # zh-cn -> zh
            result = stt.transcribe(wav_path, language=whisper_lang)
            transcription = result['text'].strip()

            # Calculate similarity (simple word overlap)
            input_words = set(text.lower().split())
            output_words = set(transcription.lower().split())
            if input_words:
                overlap = len(input_words & output_words) / len(input_words) * 100
            else:
                overlap = 0

            print(f"  Output: {transcription[:60]}...")
            print(f"  Similarity: {overlap:.0f}%")

            results[lang_code] = {
                'language': lang_name,
                'input': text,
                'output': transcription,
                'similarity': overlap,
                'status': 'PASS' if overlap > 50 else 'FAIL'
            }

        except Exception as e:
            print(f"  ERROR: {e}")
            results[lang_code] = {
                'language': lang_name,
                'status': 'ERROR',
                'error': str(e)
            }

    return results

def main():
    # Create reference audio if needed
    ref_audio = '/tmp/ref_audio.wav'
    if not os.path.exists(ref_audio):
        print("Creating reference audio with Kokoro...")
        import subprocess
        subprocess.run([
            'python', os.path.join(PROJECT_DIR, 'scripts/kokoro_tts.py'),
            'Hello world, this is a reference audio for voice cloning.',
            '-o', ref_audio, '-l', 'en'
        ], capture_output=True)

    tts = load_tts()
    stt = load_stt()

    print("\n" + "="*60)
    print("TTS -> STT ROUNDTRIP TEST")
    print("="*60)

    results = test_tts_stt(tts, stt, ref_audio)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    passed = sum(1 for r in results.values() if r.get('status') == 'PASS')
    failed = sum(1 for r in results.values() if r.get('status') == 'FAIL')
    errors = sum(1 for r in results.values() if r.get('status') == 'ERROR')

    print(f"PASSED: {passed}/{len(results)}")
    print(f"FAILED: {failed}/{len(results)}")
    print(f"ERRORS: {errors}/{len(results)}")

    # Save results
    output_path = os.path.join(PROJECT_DIR, f'reports/main/tts_test_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")

if __name__ == '__main__':
    main()
