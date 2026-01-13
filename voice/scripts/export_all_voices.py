#!/usr/bin/env python3
"""
Re-export all Kokoro voice files to match pipeline's load_voice().

The original exported voice files had different values than what
KPipeline.load_voice() returns, causing audio duration differences.

Usage:
    source .venv_kokoro/bin/activate
    python scripts/export_all_voices.py
"""

import torch
from kokoro import KPipeline
import os

# All available voices from Kokoro
VOICES = [
    # American female
    'af_heart', 'af_alloy', 'af_aoede', 'af_bella', 'af_jessica',
    'af_kore', 'af_nicole', 'af_nova', 'af_river', 'af_sarah', 'af_sky',
    # American male
    'am_adam', 'am_echo', 'am_eric', 'am_fenrir', 'am_liam',
    'am_michael', 'am_onyx', 'am_puck', 'am_santa',
    # British female
    'bf_alice', 'bf_emma', 'bf_isabella', 'bf_lily',
    # British male
    'bm_daniel', 'bm_fable', 'bm_george', 'bm_lewis',
    # Spanish
    'ef_dora', 'em_alex', 'em_santa',
    # French
    'ff_siwis',
    # Hindi
    'hf_alpha', 'hf_beta', 'hm_omega', 'hm_psi',
    # Italian
    'if_sara', 'im_nicola',
    # Japanese
    'jf_alpha', 'jf_gongitsune', 'jf_nezumi', 'jf_tebukuro', 'jm_kumo',
    # Portuguese
    'pf_dora', 'pm_alex', 'pm_santa',
    # Chinese
    'zf_xiaobei', 'zf_xiaoni', 'zf_xiaoxiao', 'zf_xiaoyi',
    'zm_yunjian', 'zm_yunxi', 'zm_yunxia', 'zm_yunyang',
]

def main():
    # Use relative path from script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    output_dir = os.path.join(project_dir, 'models/kokoro')
    os.makedirs(output_dir, exist_ok=True)

    print("Loading Kokoro pipeline...")
    pipe = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')

    print(f"\nExporting {len(VOICES)} voice files...")
    success = 0
    failed = []

    for voice in VOICES:
        try:
            voice_data = pipe.load_voice(voice)
            output_path = os.path.join(output_dir, f'voice_{voice}.pt')
            torch.save(voice_data, output_path)
            print(f"  {voice}: OK ({voice_data.shape})")
            success += 1
        except Exception as e:
            print(f"  {voice}: FAILED - {e}")
            failed.append(voice)

    print(f"\nExport complete: {success}/{len(VOICES)} voices exported")
    if failed:
        print(f"Failed voices: {failed}")

if __name__ == '__main__':
    main()
