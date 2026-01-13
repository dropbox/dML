#!/usr/bin/env python3
"""
LLM-as-Judge Audio Quality Evaluation.
Uses GPT-5 to evaluate TTS audio quality on multiple dimensions.
Provides MOS-style scores for naturalness, clarity, prosody, etc.
"""

import os
import sys
import json
import base64
import time
from datetime import datetime
from pathlib import Path
import argparse

# Load .env file if exists
def load_env():
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value

load_env()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

def encode_audio_base64(audio_path: str) -> str:
    """Encode audio file to base64 for API."""
    with open(audio_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def get_audio_format(audio_path: str) -> str:
    """Get audio format from file extension."""
    ext = Path(audio_path).suffix.lower()
    formats = {'.wav': 'wav', '.mp3': 'mp3', '.m4a': 'm4a', '.ogg': 'ogg'}
    return formats.get(ext, 'wav')

EVALUATION_SYSTEM_PROMPT = """You are an expert TTS audio evaluator that ONLY outputs valid JSON. Never include any text before or after the JSON object. Your response must be parseable by json.loads()."""

EVALUATION_PROMPT = """Rate this audio on 3 dimensions (1-5 scale):

1. Accuracy (1-5): Does it say the EXACT words? (5=perfect, 1=wrong words)
2. Naturalness (1-5): Does it sound like a REAL HUMAN? (5=human, 1=robotic)
3. Quality (1-5): Overall quality (5=excellent, 1=unacceptable)

Expected text: "{expected_text}"
Language: {language}

OUTPUT ONLY THIS JSON (no other text):
{{"accuracy": <1-5>, "naturalness": <1-5>, "quality": <1-5>, "transcription": "<what you heard>", "issues": "<brief issues or 'none'>"}}"""

def evaluate_audio_openai(audio_path: str, expected_text: str, language: str = 'en', max_retries: int = 3) -> dict:
    """Evaluate audio using OpenAI GPT-5 with retry logic for non-deterministic transcription.

    GPT-5 audio transcription is non-deterministic - it may occasionally mis-transcribe words.
    This function retries up to max_retries times and returns the best result (highest accuracy).
    """
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
    except ImportError:
        return {"error": "openai package not installed. Run: pip install openai"}

    if not OPENAI_API_KEY:
        return {"error": "OPENAI_API_KEY environment variable not set"}

    audio_base64 = encode_audio_base64(audio_path)
    audio_format = get_audio_format(audio_path)

    prompt = EVALUATION_PROMPT.format(expected_text=expected_text, language=language)

    best_result = None
    all_results = []

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-audio-2025-08-28",  # GPT-5 based audio model (better than gpt-4o-audio-preview)
                modalities=["text"],
                messages=[
                    {
                        "role": "system",
                        "content": EVALUATION_SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": audio_base64,
                                    "format": audio_format
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )

            result_text = response.choices[0].message.content

            # Parse JSON from response
            json_start = result_text.find('{')
            json_end = result_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                result = json.loads(result_text[json_start:json_end])
                result['raw_response'] = result_text
                result['attempt'] = attempt + 1
                all_results.append(dict(result))  # Copy to avoid circular ref

                # If perfect accuracy, return immediately
                if result.get('accuracy', 0) >= 5:
                    result['total_attempts'] = attempt + 1
                    return result

                # Track best result by accuracy
                if best_result is None or result.get('accuracy', 0) > best_result.get('accuracy', 0):
                    best_result = result
            else:
                all_results.append({"error": "No JSON found", "raw_response": result_text, "attempt": attempt + 1})

        except json.JSONDecodeError as e:
            all_results.append({"error": f"JSON parse error: {e}", "attempt": attempt + 1})
        except Exception as e:
            all_results.append({"error": str(e), "attempt": attempt + 1})

    # Return best result or last error
    if best_result:
        best_result['total_attempts'] = max_retries
        best_result['all_attempts'] = all_results
        return best_result

    return all_results[-1] if all_results else {"error": "No results after retries"}

def evaluate_audio_anthropic(audio_path: str, expected_text: str, language: str = 'en') -> dict:
    """Evaluate audio using Anthropic Claude (if audio support available)."""
    # Note: As of now, Claude doesn't have direct audio input
    # This is a placeholder for future capability
    return {"error": "Anthropic Claude does not currently support audio input"}


COMPARISON_SYSTEM_PROMPT = """You are an expert TTS audio evaluator that ONLY outputs valid JSON. Never include any text before or after the JSON object. Your response must be parseable by json.loads()."""

COMPARISON_PROMPT = """Compare these two audio samples of the same text.

Expected text: "{expected_text}"
Language: {language}

For EACH audio, rate: Accuracy (1-5), Naturalness (1-5), Quality (1-5)
Then determine winner or tie.

OUTPUT ONLY THIS JSON (no other text):
{{"audio1": {{"accuracy": <1-5>, "naturalness": <1-5>, "quality": <1-5>, "issues": "<issues or none>"}}, "audio2": {{"accuracy": <1-5>, "naturalness": <1-5>, "quality": <1-5>, "issues": "<issues or none>"}}, "winner": "<audio1|audio2|tie>", "reasoning": "<why>"}}"""


def _single_comparison_openai(client, audio1_base64: str, audio1_format: str,
                               audio2_base64: str, audio2_format: str,
                               expected_text: str, language: str) -> dict:
    """Run a single comparison (internal helper)."""
    prompt = COMPARISON_PROMPT.format(expected_text=expected_text, language=language)

    response = client.chat.completions.create(
        model="gpt-audio-2025-08-28",  # GPT-5 based audio model
        modalities=["text"],
        messages=[
            {
                "role": "system",
                "content": COMPARISON_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "text", "text": "Audio 1:"},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio1_base64,
                            "format": audio1_format
                        }
                    },
                    {"type": "text", "text": "Audio 2:"},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio2_base64,
                            "format": audio2_format
                        }
                    }
                ]
            }
        ],
        max_tokens=500
    )

    result_text = response.choices[0].message.content

    # Parse JSON from response
    json_start = result_text.find('{')
    json_end = result_text.rfind('}') + 1
    if json_start >= 0 and json_end > json_start:
        result = json.loads(result_text[json_start:json_end])
        result['raw_response'] = result_text
        return result
    else:
        return {"error": "No JSON found in response", "raw_response": result_text}


def compare_audio_pair_openai(audio1_path: str, audio2_path: str, expected_text: str, language: str = 'en') -> dict:
    """Compare two audio files using OpenAI GPT-4o audio with position-bias correction.

    IMPORTANT: GPT-4o has a position bias (tends to favor audio2). To correct this,
    we run the comparison TWICE with swapped order and aggregate the results.

    Args:
        audio1_path: Path to first audio file (typically C++)
        audio2_path: Path to second audio file (typically Python)
        expected_text: Expected text content
        language: Language code

    Returns:
        Dict with audio1 scores, audio2 scores, winner, and reasoning
    """
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
    except ImportError:
        return {"error": "openai package not installed. Run: pip install openai"}

    if not OPENAI_API_KEY:
        return {"error": "OPENAI_API_KEY environment variable not set"}

    audio1_base64 = encode_audio_base64(audio1_path)
    audio2_base64 = encode_audio_base64(audio2_path)
    audio1_format = get_audio_format(audio1_path)
    audio2_format = get_audio_format(audio2_path)

    try:
        # Run comparison in BOTH orders to eliminate position bias
        # Round 1: audio1 as Audio1, audio2 as Audio2
        result1 = _single_comparison_openai(
            client, audio1_base64, audio1_format, audio2_base64, audio2_format,
            expected_text, language
        )

        # Round 2: audio2 as Audio1, audio1 as Audio2 (swapped)
        result2 = _single_comparison_openai(
            client, audio2_base64, audio2_format, audio1_base64, audio1_format,
            expected_text, language
        )

        # Check for errors
        if "error" in result1:
            return result1
        if "error" in result2:
            return result2

        # Aggregate scores: average the scores from both rounds
        # In round 2, "audio1" refers to our audio2 and vice versa
        a1_scores = result1.get('audio1', {})
        a2_scores = result1.get('audio2', {})
        # In round 2: result2['audio1'] is our audio2, result2['audio2'] is our audio1
        a1_scores_r2 = result2.get('audio2', {})  # Our audio1 was in position 2
        a2_scores_r2 = result2.get('audio1', {})  # Our audio2 was in position 1

        def avg_score(s1, s2, key):
            v1 = s1.get(key, 0) if isinstance(s1, dict) else 0
            v2 = s2.get(key, 0) if isinstance(s2, dict) else 0
            if v1 and v2:
                return round((v1 + v2) / 2, 1)
            return v1 or v2

        final_audio1 = {
            'accuracy': avg_score(a1_scores, a1_scores_r2, 'accuracy'),
            'naturalness': avg_score(a1_scores, a1_scores_r2, 'naturalness'),
            'quality': avg_score(a1_scores, a1_scores_r2, 'quality'),
            'issues': a1_scores.get('issues', '') or a1_scores_r2.get('issues', '')
        }

        final_audio2 = {
            'accuracy': avg_score(a2_scores, a2_scores_r2, 'accuracy'),
            'naturalness': avg_score(a2_scores, a2_scores_r2, 'naturalness'),
            'quality': avg_score(a2_scores, a2_scores_r2, 'quality'),
            'issues': a2_scores.get('issues', '') or a2_scores_r2.get('issues', '')
        }

        # Determine winner based on aggregated scores
        score1 = final_audio1['accuracy'] + final_audio1['naturalness'] + final_audio1['quality']
        score2 = final_audio2['accuracy'] + final_audio2['naturalness'] + final_audio2['quality']

        if abs(score1 - score2) <= 0.5:  # Within 0.5 points = tie
            winner = 'tie'
        elif score1 > score2:
            winner = 'audio1'
        else:
            winner = 'audio2'

        # Count wins from both rounds
        r1_winner = result1.get('winner', 'tie')
        r2_winner = result2.get('winner', 'tie')
        # Map r2 winner back (audio1 in r2 = our audio2)
        r2_winner_mapped = {'audio1': 'audio2', 'audio2': 'audio1', 'tie': 'tie'}.get(r2_winner, 'tie')

        return {
            'audio1': final_audio1,
            'audio2': final_audio2,
            'winner': winner,
            'reasoning': f"Position-corrected comparison: Round1={r1_winner}, Round2={r2_winner_mapped} -> Aggregated scores: a1={score1:.1f}, a2={score2:.1f}",
            'raw_response': f"R1: {result1.get('raw_response', '')}\n\nR2: {result2.get('raw_response', '')}",
            'round1': result1,
            'round2': result2
        }

    except Exception as e:
        return {"error": str(e)}

def evaluate_accent_consensus(audio_path: str, text: str, language: str, num_runs: int = 5) -> dict:
    """Run multiple accent evaluations and return consensus result with strong-vote filtering.

    GPT audio models are non-deterministic. To reduce flakiness we:
      - Run multiple evaluations (default 5, overridable via LLM_JUDGE_RUNS)
      - Count a strong English vote only when accent_origin is English *and* scores indicate non-native speech
      - Stop early once a majority decision is reached

    Returns a summary with vote counts, averaged scores, and all raw results.
    """
    # Import here to avoid circular import when used from test files
    from pathlib import Path
    import base64

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
    except ImportError:
        return {"error": "openai package not installed. Run: pip install openai"}

    if not OPENAI_API_KEY:
        return {"error": "OPENAI_API_KEY environment variable not set"}

    try:
        num_runs = int(os.environ.get("LLM_JUDGE_RUNS", num_runs))
    except ValueError:
        pass

    results = []
    valid_results = []
    english_votes = 0
    english_strong_votes = 0
    native_scores = []
    pronunciation_scores = []
    prosody_scores = []

    def _avg(values):
        return sum(values) / len(values) if values else 0.0

    for run_idx in range(num_runs):
        try:
            with open(audio_path, 'rb') as f:
                audio_base64 = base64.b64encode(f.read()).decode('utf-8')

            ext = Path(audio_path).suffix.lower()
            audio_format = {'.wav': 'wav', '.mp3': 'mp3', '.m4a': 'm4a'}.get(ext, 'wav')

            prompt = f"""Analyze this {language} TTS audio for accent/pronunciation.

Text: "{text}"
Target language: {language}

Rate 1-5:
1. native_accent: How native does it sound? (5=native speaker, 1=heavy foreign accent)
2. pronunciation_accuracy: Correct sounds? (5=perfect, 1=wrong)
3. prosody_authenticity: Natural rhythm/intonation? (5=native, 1=foreign)

The expected speaker is a native {language} voice. Minor TTS artifacts, slight flatness, or studio tone DO NOT mean English accent.
Only mark an English accent if you clearly hear English phoneme/stress patterns (e.g., r/l confusion, incorrect tones/vowels consistent with English).
If unsure between native vs English accent, choose native.

Also identify: Does this sound like an ENGLISH speaker trying to speak {language}?

OUTPUT ONLY JSON:
{{"native_accent": <1-5>, "accent_origin": "<detected accent or 'native'>", "pronunciation_accuracy": <1-5>, "prosody_authenticity": <1-5>, "issues": "<issues or 'none'>"}}"""

            response = client.chat.completions.create(
                model="gpt-audio-2025-08-28",
                modalities=["text"],
                messages=[
                    {"role": "system", "content": "You are an expert TTS audio evaluator. Output ONLY valid JSON."},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "input_audio", "input_audio": {"data": audio_base64, "format": audio_format}}
                    ]}
                ],
                max_tokens=300
            )

            result_text = response.choices[0].message.content
            json_start = result_text.find('{')
            json_end = result_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                result = json.loads(result_text[json_start:json_end])
                result['run'] = run_idx + 1
                results.append(result)
                valid_results.append(result)

                native_score = result.get('native_accent', 0)
                pron_score = result.get('pronunciation_accuracy', 0)
                prosody_score = result.get('prosody_authenticity', 0)

                native_scores.append(native_score)
                pronunciation_scores.append(pron_score)
                prosody_scores.append(prosody_score)

                accent = result.get('accent_origin', '').lower()
                is_english = any(x in accent for x in ['english', 'american', 'british'])
                if is_english:
                    english_votes += 1
                    if native_score <= 3 or pron_score <= 3 or prosody_score <= 3:
                        english_strong_votes += 1

            else:
                results.append({"error": "No JSON found in response", "run": run_idx + 1, "raw": result_text})
        except Exception as e:
            results.append({'error': str(e), 'run': run_idx + 1})

    valid_results = [r for r in results if 'error' not in r]

    if len(valid_results) < 2:
        return {'error': f'Only {len(valid_results)} successful evaluations (need at least 2)', 'all_results': results}

    native_votes = len(valid_results) - english_votes
    required_majority = max(2, len(valid_results) // 2 + 1)
    strong_threshold = max(1, required_majority - 1)
    has_english_accent = (english_votes >= required_majority) and (english_strong_votes >= strong_threshold)

    avg_native = _avg(native_scores)
    avg_pron = _avg(pronunciation_scores)
    avg_prosody = _avg(prosody_scores)

    return {
        'has_english_accent': has_english_accent,
        'votes': {'english': english_votes, 'english_strong': english_strong_votes, 'native': native_votes},
        'avg_native_accent': avg_native,
        'avg_pronunciation_accuracy': avg_pron,
        'avg_prosody_authenticity': avg_prosody,
        'valid_evaluations': len(valid_results),
        'runs_completed': len(results),
        'confidence': max(english_votes, native_votes) / len(valid_results),
        'all_results': results
    }


def compare_tts_systems(texts: list, languages: list, tts_systems: dict) -> dict:
    """Compare multiple TTS systems using LLM judge.

    Args:
        texts: List of texts to synthesize
        languages: List of language codes
        tts_systems: Dict of {name: synthesize_function}
    """
    results = []

    for text, lang in zip(texts, languages):
        print(f"\nEvaluating: {text[:50]}... ({lang})")

        text_results = {"text": text, "language": lang, "systems": {}}

        for system_name, synthesize_fn in tts_systems.items():
            print(f"  Testing {system_name}...")

            # Synthesize audio
            audio_path = f"/tmp/eval_{system_name}_{hash(text) % 10000}.wav"
            try:
                synth_result = synthesize_fn(text, lang, audio_path)
                if not synth_result.get('success', False):
                    text_results["systems"][system_name] = {
                        "error": synth_result.get('error', 'Synthesis failed')
                    }
                    continue
            except Exception as e:
                text_results["systems"][system_name] = {"error": str(e)}
                continue

            # Evaluate with LLM
            eval_result = evaluate_audio_openai(audio_path, text, lang)
            eval_result['latency'] = synth_result.get('latency', 0)
            eval_result['rtf'] = synth_result.get('rtf', 0)
            text_results["systems"][system_name] = eval_result

            if 'overall_mos' in eval_result:
                print(f"    MOS: {eval_result['overall_mos']}/5, "
                      f"Naturalness: {eval_result.get('naturalness', 'N/A')}/5")

        results.append(text_results)

    return {"evaluations": results, "timestamp": datetime.now().isoformat()}

def aggregate_scores(results: dict) -> dict:
    """Aggregate scores across all evaluations."""
    system_scores = {}

    for eval_item in results.get("evaluations", []):
        for system_name, scores in eval_item.get("systems", {}).items():
            if "error" in scores:
                continue

            if system_name not in system_scores:
                system_scores[system_name] = {
                    "accuracy": [], "naturalness": [], "quality": [],
                    "latency": [], "rtf": []
                }

            for metric in system_scores[system_name]:
                if metric in scores and scores[metric] is not None:
                    system_scores[system_name][metric].append(scores[metric])

    # Calculate averages
    summary = {}
    for system_name, metrics in system_scores.items():
        summary[system_name] = {}
        for metric, values in metrics.items():
            if values:
                summary[system_name][metric] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values)
                }

    return summary

def main():
    parser = argparse.ArgumentParser(description='LLM-as-Judge Audio Quality Evaluation')
    parser.add_argument('--audio', type=str, help='Audio file to evaluate')
    parser.add_argument('--text', type=str, help='Expected text')
    parser.add_argument('--language', '-l', type=str, default='en', help='Language code')
    parser.add_argument('--compare', action='store_true', help='Compare TTS systems')
    parser.add_argument('--output', '-o', type=str, help='Output JSON file')
    args = parser.parse_args()

    if args.audio and args.text:
        # Single audio evaluation
        print(f"Evaluating: {args.audio}")
        result = evaluate_audio_openai(args.audio, args.text, args.language)
        print(json.dumps(result, indent=2, ensure_ascii=False))

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

    elif args.compare:
        # Compare TTS systems
        print("Comparing TTS systems with LLM judge...")

        # Import TTS server for synthesis
        sys.path.insert(0, str(Path(__file__).parent))
        from tts_server import HybridTTSServer

        tts = HybridTTSServer()

        def kokoro_synth(text, lang, output):
            if lang not in ['en', 'ja']:
                return {'success': False, 'error': 'Kokoro only supports en/ja'}
            return tts._synthesize_kokoro(text, lang, output, time.time())

        def xtts_synth(text, lang, output):
            return tts._synthesize_xtts(text, lang, output, time.time())

        # Test sentences
        test_cases = [
            ("The function was successfully refactored.", "en"),
            ("All tests passed without any errors.", "en"),
            ("関数のリファクタリングが完了しました。", "ja"),
            ("すべてのテストが成功しました。", "ja"),
        ]

        texts = [t[0] for t in test_cases]
        langs = [t[1] for t in test_cases]

        systems = {
            "kokoro": kokoro_synth,
            "xtts": xtts_synth
        }

        results = compare_tts_systems(texts, langs, systems)
        summary = aggregate_scores(results)

        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)

        for system, metrics in summary.items():
            print(f"\n{system.upper()}:")
            for metric, stats in metrics.items():
                if metric in ['accuracy', 'naturalness', 'quality']:
                    print(f"  {metric}: {stats['mean']:.2f}/5 (n={stats['count']})")
                elif metric in ['latency', 'rtf']:
                    print(f"  {metric}: {stats['mean']:.2f}")

        # Determine winner
        if len(summary) >= 2:
            systems_list = list(summary.keys())
            quality_scores = {s: summary[s].get('quality', {}).get('mean', 0) for s in systems_list}
            winner = max(quality_scores, key=quality_scores.get)
            print(f"\n>>> WINNER (by Quality): {winner.upper()} ({quality_scores[winner]:.2f}/5)")

        script_dir = Path(__file__).parent
        project_dir = script_dir.parent
        output_path = args.output or str(project_dir / f'reports/main/llm_judge_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        results['summary'] = summary
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_path}")

    else:
        parser.print_help()
        print("\nExamples:")
        print("  # Evaluate single audio:")
        print("  python llm_audio_judge.py --audio test.wav --text 'Hello world' -l en")
        print("")
        print("  # Compare TTS systems:")
        print("  OPENAI_API_KEY=sk-... python llm_audio_judge.py --compare")

if __name__ == '__main__':
    main()
