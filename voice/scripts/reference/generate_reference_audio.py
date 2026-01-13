#!/usr/bin/env python3
"""
Generate reference audio using Python StyleTTS2 to compare with C++ output.
"""
import sys
import os

# Use relative paths from script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, 'models/styletts2'))

from providers import play_audio

import torch
import numpy as np
from scipy.io import wavfile
import phonemizer

# Import StyleTTS2 components
from models import build_model
from text_utils import TextCleaner
from Utils.PLBERT.util import load_plbert
from Modules.diffusion.sampler import DiffusionSampler
import yaml

def load_models():
    """Load all StyleTTS2 models."""
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    model_path = os.path.join(PROJECT_DIR, 'models/styletts2/Models/LJSpeech')
    config_path = os.path.join(model_path, 'config.yml')

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Build model
    model = build_model(config['model_params'], text_aligner=None, pitch_extractor=None, plbert=None)

    # Load checkpoint
    checkpoint_path = os.path.join(model_path, 'epoch_2nd_00100.pth')
    params = torch.load(checkpoint_path, map_location='cpu')
    for key in model:
        if key in params:
            try:
                model[key].load_state_dict(params[key])
            except:
                from collections import OrderedDict
                state_dict = params[key]
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] if k.startswith('module.') else k
                    new_state_dict[name] = v
                model[key].load_state_dict(new_state_dict, strict=False)

    # Move to device
    for key in model:
        model[key] = model[key].to(device)
        model[key].eval()

    # Load PLBERT
    plbert_config_path = os.path.join(PROJECT_DIR, 'models/styletts2/Utils/PLBERT/config.yml')
    plbert_path = os.path.join(PROJECT_DIR, 'models/styletts2/Utils/PLBERT/step_1000000.t7')
    plbert = load_plbert(plbert_config_path, plbert_path)
    plbert = plbert.to(device)

    return model, plbert, config, device


def synthesize(text, model, plbert, config, device, alpha=0.3, beta=0.7, diffusion_steps=5):
    """Synthesize speech from text."""
    text_cleaner = TextCleaner()

    # Get reference style from mean embeddings
    ref_s = model['style_encoder'].mean_style

    # Phonemize
    global_phonemizer = phonemizer.backend.EspeakBackend(
        language='en-us',
        preserve_punctuation=True,
        with_stress=True
    )
    ps = global_phonemizer.phonemize([text])
    ps = ps[0].strip()

    # Convert to tokens
    tokens = text_cleaner(ps)
    tokens = torch.LongTensor(tokens).unsqueeze(0).to(device)

    with torch.no_grad():
        # BERT features
        bert_input_ids = tokens  # Simplified - real impl uses tokenizer
        bert_out = plbert(bert_input_ids)

        # Text encoding
        t_en = model['text_encoder'](tokens, bert_out)

        # Style
        s = ref_s if ref_s is not None else torch.randn(1, 128).to(device)

        # Duration prediction
        d = model['predictor'](t_en, s, bert_out)
        d = torch.clamp(d, min=0)

        # Length regulation
        # ... (simplified)

        # Decode
        audio = model['decoder'](t_en, s)

    return audio.cpu().numpy()


def main():
    text = "Hello, this is a test of the text to speech system."

    print("Loading models...")
    model, plbert, config, device = load_models()

    print(f"Synthesizing: {text}")
    audio = synthesize(text, model, plbert, config, device)

    # Save
    output_path = '/tmp/tts_python_reference.wav'
    wavfile.write(output_path, 24000, (audio * 32767).astype(np.int16))
    print(f"Saved to: {output_path}")

    # Play
    with open(output_path, 'rb') as f:
        play_audio(f.read())


if __name__ == '__main__':
    main()
