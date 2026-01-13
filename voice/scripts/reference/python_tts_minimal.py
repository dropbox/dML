#!/usr/bin/env python3
"""
Minimal StyleTTS2 inference script - based on working notebook code.
Uses diffusion sampling for style vectors (the key to quality).

Usage:
    cd <project>/models/styletts2
    python <project>/scripts/reference/python_tts_minimal.py "Hello world" -o /tmp/output.wav
"""

import sys
import os
import argparse

# Use relative paths from script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
STYLETTS2_DIR = os.path.join(PROJECT_DIR, "models/styletts2")
os.chdir(STYLETTS2_DIR)
sys.path.insert(0, STYLETTS2_DIR)

import torch
import numpy as np
import yaml
from munch import Munch
from scipy.io import wavfile
from nltk.tokenize import word_tokenize
import phonemizer

# Fix for PyTorch 2.6+ weights_only default
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# Set seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)

def main():
    parser = argparse.ArgumentParser(description="StyleTTS2 inference")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("-o", "--output", default="/tmp/python_tts.wav", help="Output WAV path")
    parser.add_argument("--steps", type=int, default=5, help="Diffusion steps")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode")
    args = parser.parse_args()

    # Device selection
    if args.cpu:
        device = 'cpu'
    elif torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Using device: {device}")

    # Load phonemizer
    print("Loading phonemizer...")
    global_phonemizer = phonemizer.backend.EspeakBackend(
        language='en-us', preserve_punctuation=True, with_stress=True)

    from text_utils import TextCleaner
    textcleaner = TextCleaner()

    # Load config
    print("Loading config...")
    config = yaml.safe_load(open("Models/LJSpeech/config.yml"))

    # Import models module
    from models import build_model, load_ASR_models, load_F0_models
    from utils import recursive_munch

    # Load pretrained ASR model
    print("Loading ASR model...")
    ASR_config = config.get('ASR_config', False)
    ASR_path = config.get('ASR_path', False)
    text_aligner = load_ASR_models(ASR_path, ASR_config)

    # Load pretrained F0 model
    print("Loading F0 model...")
    F0_path = config.get('F0_path', False)
    pitch_extractor = load_F0_models(F0_path)

    # Load BERT model
    print("Loading BERT model...")
    from Utils.PLBERT.util import load_plbert
    BERT_path = config.get('PLBERT_dir', False)
    plbert = load_plbert(BERT_path)

    # Build model
    print("Building model...")
    model = build_model(recursive_munch(config['model_params']), text_aligner, pitch_extractor, plbert)
    _ = [model[key].eval() for key in model]
    _ = [model[key].to(device) for key in model]

    # Load weights
    print("Loading weights...")
    params_whole = torch.load("Models/LJSpeech/epoch_2nd_00100.pth", map_location='cpu')
    params = params_whole['net']

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

    _ = [model[key].eval() for key in model]

    # Load diffusion sampler
    print("Loading diffusion sampler...")
    from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
    sampler = DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
        clamp=False
    )

    # Helper function
    def length_to_mask(lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask+1, lengths.unsqueeze(1))
        return mask

    # Inference function
    def inference(text, noise, diffusion_steps=5, embedding_scale=1):
        text = text.strip().replace('"', '')
        ps = global_phonemizer.phonemize([text])
        ps = word_tokenize(ps[0])
        ps = ' '.join(ps)
        print(f"Phonemes: {ps}")

        tokens = textcleaner(ps)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)

        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(tokens.device)
            text_mask = length_to_mask(input_lengths).to(tokens.device)

            t_en = model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

            # KEY: Use diffusion sampler for style vectors!
            s_pred = sampler(noise,
                  embedding=bert_dur[0].unsqueeze(0), num_steps=diffusion_steps,
                  embedding_scale=embedding_scale).squeeze(0)

            s = s_pred[:, 128:]   # prosody style
            ref = s_pred[:, :128] # decoder reference

            d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

            x, _ = model.predictor.lstm(d)
            duration = model.predictor.duration_proj(x)
            duration = torch.sigmoid(duration).sum(axis=-1)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)

            pred_dur[-1] += 5  # Extend last phoneme

            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)

            # Encode prosody
            en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
            F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

            # Decode audio
            out = model.decoder((t_en @ pred_aln_trg.unsqueeze(0).to(device)),
                                    F0_pred, N_pred, ref.squeeze().unsqueeze(0))

        return out.squeeze().cpu().numpy()

    # Run inference
    print(f"\nSynthesizing: {args.text}")
    noise = torch.randn(1, 1, 256).to(device)
    wav = inference(args.text, noise, diffusion_steps=args.steps, embedding_scale=1)

    # Normalize and save output
    max_amp = np.abs(wav).max()
    print(f"Raw max amplitude: {max_amp:.4f}")
    if max_amp > 0.01:
        wav = wav / max_amp * 0.707  # Normalize to -3dB
    wav_int16 = (wav * 32767).astype(np.int16)
    wavfile.write(args.output, 24000, wav_int16)
    print(f"\nSaved to: {args.output}")
    print(f"Duration: {len(wav)/24000:.2f}s")
    print(f"Normalized max amplitude: {np.abs(wav).max():.4f}")

if __name__ == "__main__":
    main()
