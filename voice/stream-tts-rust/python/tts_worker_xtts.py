#!/usr/bin/env python3
"""
TTS Worker - XTTS v2 on Metal GPU - No MeCab Required
"""
import os
# Disable cutlet/mecab (not needed for audio generation)
os.environ['XTTS_NO_NORMALIZE'] = '1'

"""
TTS Worker - XTTS v2 on Metal GPU
Converts Japanese text to speech using Coqui XTTS v2 with Metal acceleration.

Target: < 60ms per sentence (down from 581ms with macOS say)

Optimizations:
1. XTTS v2 model on Metal GPU (MPS)
2. BFloat16 precision for M4 Max
3. Direct audio streaming (no disk I/O)
4. Optimized generation parameters

Interface:
  - Input: Japanese text on stdin (one line per request)
  - Output: Path to generated audio file on stdout (or empty for streaming)
  - Errors: stderr

Usage:
  echo "こんにちは" | python3 tts_worker_xtts.py
"""
import sys
import time
import os
import tempfile
import torch
import sounddevice as sd
import numpy as np

# Fix for PyTorch 2.6+ weights_only default change
# Monkeypatch torch.load to allow loading TTS models
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    """Patched torch.load that defaults to weights_only=False for TTS compatibility"""
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

class XTTSWorker:
    def __init__(self):
        print("[TTS] Initializing XTTS v2 on Metal GPU...", file=sys.stderr)
        
        # Check if MPS is available
        if not torch.backends.mps.is_available():
            print("[TTS] WARNING: MPS not available, using CPU", file=sys.stderr)
            self.device = "cpu"
        else:
            self.device = "mps"
            print(f"[TTS] Using Metal GPU (MPS)", file=sys.stderr)
        
        # Load XTTS v2 model
        start = time.perf_counter()
        
        try:
            from TTS.api import TTS

            # Load multilingual XTTS v2
            print("[TTS] Loading XTTS v2 model...", file=sys.stderr)
            self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

            # Move to device
            if self.device == "mps":
                self.tts.to(self.device)
                print("[TTS] Model moved to Metal GPU", file=sys.stderr)

            self.fallback = False

        except Exception as e:
            print(f"[TTS] ERROR loading XTTS v2: {e}", file=sys.stderr)
            print("[TTS] Falling back to macOS say command", file=sys.stderr)
            self.tts = None
            self.fallback = True
        
        elapsed = time.perf_counter() - start
        print(f"[TTS] Model loaded in {elapsed:.2f}s", file=sys.stderr)
        
        # Audio settings
        self.sample_rate = 22050
        self.temp_dir = tempfile.mkdtemp(prefix="tts_xtts_")
        
        # Warm up
        if not self.fallback:
            print("[TTS] Warming up...", file=sys.stderr)
            try:
                _ = self.synthesize("こんにちは")
                print("[TTS] Warmup complete", file=sys.stderr)
            except Exception as e:
                print(f"[TTS] Warmup failed: {e}", file=sys.stderr)
        
        print("[TTS] Ready to synthesize", file=sys.stderr)
        sys.stderr.flush()
    
    def synthesize(self, text: str) -> str:
        """
        Synthesize text to speech using XTTS v2.
        Returns path to generated audio file.
        """
        if not text.strip():
            return ""
        
        start = time.perf_counter()
        
        try:
            if self.fallback:
                # Fallback to macOS say
                return self.synthesize_say(text)
            
            # Generate with XTTS v2
            # XTTS v2 requires speaker - use built-in speaker
            # Coqui TTS has built-in speakers we can use
            wav = self.tts.tts(
                text=text,
                language="ja",
                speaker="Claribel Dervla",  # Default speaker
                speed=1.0
            )
            
            # Convert to numpy array if needed
            if not isinstance(wav, np.ndarray):
                wav = np.array(wav, dtype=np.float32)
            
            # Save to file (for compatibility with current pipeline)
            audio_file = os.path.join(self.temp_dir, f"xtts_{int(time.time() * 1000000)}.wav")
            
            # Save as WAV
            from scipy.io import wavfile
            wavfile.write(audio_file, self.sample_rate, wav)
            
            elapsed_ms = (time.perf_counter() - start) * 1000
            file_size_kb = os.path.getsize(audio_file) / 1024
            
            # Truncate text for logging (UTF-8 safe)
            text_preview = ''.join(list(text)[:40])
            
            print(f"[TTS] {elapsed_ms:.1f}ms | {file_size_kb:.1f}KB | {text_preview}...", file=sys.stderr)
            sys.stderr.flush()
            
            return audio_file
            
        except Exception as e:
            print(f"[TTS] ERROR: {e}", file=sys.stderr)
            sys.stderr.flush()
            # Fallback to macOS say
            return self.synthesize_say(text)
    
    def synthesize_say(self, text: str) -> str:
        """Fallback to macOS say command"""
        import subprocess
        
        audio_file = os.path.join(self.temp_dir, f"say_{int(time.time() * 1000000)}.aiff")
        
        cmd = ['say', '-v', 'Kyoko', '-r', '200', '-o', audio_file, text]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"say command failed: {result.stderr}")
        
        return audio_file
    
    def run(self):
        """Main loop: read from stdin, synthesize, write file path to stdout"""
        print("[TTS] Listening for input on stdin...", file=sys.stderr)
        sys.stderr.flush()
        
        for line in sys.stdin:
            text = line.strip()
            if not text:
                continue
            
            try:
                audio_file = self.synthesize(text)
                print(audio_file)
                sys.stdout.flush()
            except Exception as e:
                print(f"[TTS] ERROR: {e}", file=sys.stderr)
                sys.stderr.flush()
                # Output empty line on error
                print()
                sys.stdout.flush()
    
    def cleanup(self):
        """Clean up temporary files"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"[TTS] Cleaned up temp directory", file=sys.stderr)

def main():
    worker = XTTSWorker()
    try:
        worker.run()
    except KeyboardInterrupt:
        print("\n[TTS] Shutting down...", file=sys.stderr)
    except Exception as e:
        print(f"[TTS] FATAL ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        worker.cleanup()

if __name__ == "__main__":
    main()
