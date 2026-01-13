#!/usr/bin/env python3
"""
Compare audio durations between the C++ Kokoro TTS binary and the reference
Python Kokoro pipeline for the same input text.

This is a DEVELOPMENT/DIAGNOSTIC script (Python is not used in production).
It generates WAV files from both implementations and reports duration deltas.

Usage:
    python scripts/compare_kokoro_durations.py "Hello world" --lang en
    python scripts/compare_kokoro_durations.py --text-file samples.txt --lang ja
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable


DEFAULT_VOICES = {
    "en": "af_heart",
    "ja": "jf_alpha",
    "es": "ef_dora",   # Spanish uses 'ef_dora' in Kokoro
    "fr": "ff_siwis",
    "hi": "hf_alpha",
    "it": "if_alice",
    "pt": "pf_dora",
    "zh": "zf_xiaobei",
}


def run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Run a subprocess and raise on failure."""
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


def measure_wav(path: Path) -> tuple[int, int]:
    """Return (num_samples, sample_rate) for a WAV file."""
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("soundfile is required to measure WAV durations") from exc

    data, sample_rate = sf.read(path)
    return data.shape[0], sample_rate


def load_texts(args: argparse.Namespace) -> list[str]:
    if args.text:
        return [args.text]
    if args.text_file:
        text_path = Path(args.text_file)
        lines = [line.strip() for line in text_path.read_text().splitlines() if line.strip()]
        if not lines:
            raise ValueError(f"No text found in {text_path}")
        return lines
    raise ValueError("Provide text via positional argument or --text-file")


def synthesize_python(text: str, lang: str, voice: str, output: Path, kokoro_script: Path) -> None:
    cmd = [
        sys.executable,
        str(kokoro_script),
        text,
        "-o",
        str(output),
        "-l",
        lang,
    ]
    if voice:
        cmd += ["-v", voice]
    run(cmd, cwd=kokoro_script.parent)


def synthesize_cpp(
    text: str,
    lang: str,
    voice: str,
    output: Path,
    cpp_binary: Path,
    cpp_cwd: Path,
    debug: bool,
) -> None:
    cmd = [
        str(cpp_binary),
        "--speak",
        text,
        "--lang",
        lang,
        "--save-audio",
        str(output),
    ]
    if voice:
        cmd += ["--voice-name", voice]
    if debug:
        cmd.append("--debug")
    run(cmd, cwd=cpp_cwd)


def format_seconds(samples: int, sample_rate: int) -> str:
    return f"{samples / sample_rate:.3f}s"


def compare_once(
    text: str,
    lang: str,
    voice: str,
    cpp_binary: Path,
    cpp_cwd: Path,
    kokoro_script: Path,
    debug: bool,
    keep: bool,
) -> dict:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        py_path = tmp / "python.wav"
        cpp_path = tmp / "cpp.wav"

        synthesize_python(text, lang, voice, py_path, kokoro_script)
        synthesize_cpp(text, lang, voice, cpp_path, cpp_binary, cpp_cwd, debug)

        py_samples, py_sr = measure_wav(py_path)
        cpp_samples, cpp_sr = measure_wav(cpp_path)

        if keep:
            dest_dir = Path(keep)
            dest_dir.mkdir(parents=True, exist_ok=True)
            py_out = dest_dir / f"python_{lang}.wav"
            cpp_out = dest_dir / f"cpp_{lang}.wav"
            py_out.write_bytes(py_path.read_bytes())
            cpp_out.write_bytes(cpp_path.read_bytes())

        return {
            "text": text,
            "lang": lang,
            "voice": voice,
            "python_samples": py_samples,
            "python_sr": py_sr,
            "cpp_samples": cpp_samples,
            "cpp_sr": cpp_sr,
        }


def report(results: Iterable[dict]) -> None:
    print("Kokoro duration comparison (C++ vs Python)")
    for idx, res in enumerate(results):
        py_dur = res["python_samples"] / res["python_sr"]
        cpp_dur = res["cpp_samples"] / res["cpp_sr"]
        delta = cpp_dur - py_dur
        pct = (delta / py_dur * 100.0) if py_dur > 0 else 0.0
        print(f"[{idx}] \"{res['text'][:60]}\"")
        print(f"     Python: {format_seconds(res['python_samples'], res['python_sr'])}")
        print(f"     C++   : {format_seconds(res['cpp_samples'], res['cpp_sr'])}")
        print(f"     Δ     : {delta:.3f}s ({pct:+.1f}%)  voice={res['voice']} lang={res['lang']}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Compare Kokoro C++ vs Python durations")
    parser.add_argument("text", nargs="?", help="Text to synthesize")
    parser.add_argument("--text-file", help="File with one utterance per line")
    parser.add_argument("--lang", default="en", help="Language code (default: en)")
    parser.add_argument("--voice", help="Voice name (defaults to Kokoro voice for lang)")
    parser.add_argument(
        "--cpp-binary",
        default=repo_root / "stream-tts-cpp" / "build" / "stream-tts-cpp",
        type=Path,
        help="Path to stream-tts-cpp binary",
    )
    parser.add_argument(
        "--cpp-cwd",
        default=repo_root / "stream-tts-cpp",
        type=Path,
        help="Working directory for C++ binary (model paths resolved relative here)",
    )
    parser.add_argument(
        "--kokoro-script",
        default=repo_root / "scripts" / "kokoro_tts.py",
        type=Path,
        help="Path to Python kokoro_tts.py helper",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging for C++ binary")
    parser.add_argument(
        "--keep-wavs",
        metavar="DIR",
        help="Copy generated WAVs into DIR for inspection",
    )

    args = parser.parse_args()

    texts = load_texts(args)
    voice = args.voice or DEFAULT_VOICES.get(args.lang, "af_heart")

    results = []
    for text in texts:
        results.append(
            compare_once(
                text=text,
                lang=args.lang,
                voice=voice,
                cpp_binary=args.cpp_binary,
                cpp_cwd=args.cpp_cwd,
                kokoro_script=args.kokoro_script,
                debug=args.debug,
                keep=args.keep_wavs,
            )
        )

    report(results)


if __name__ == "__main__":
    main()
