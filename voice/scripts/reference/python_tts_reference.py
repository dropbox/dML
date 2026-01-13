#!/usr/bin/env python3
"""
Python StyleTTS2 Reference Implementation
Generates known-good audio for comparison with C++ implementation.

Usage:
    cd <project>/models/styletts2
    python <project>/scripts/reference/python_tts_reference.py "Hello world"
"""

import sys
import os
import torch
import numpy as np
import yaml
from pathlib import Path

# Use relative paths from script location
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_DIR = SCRIPT_DIR.parent.parent
STYLETTS2_DIR = PROJECT_DIR / "models/styletts2"
os.chdir(STYLETTS2_DIR)
sys.path.insert(0, str(STYLETTS2_DIR))

from munch import Munch
from models import build_model
from utils import recursive_munch
from text_utils import TextCleaner
import phonemizer
from nltk.tokenize import word_tokenize
from scipy.io import wavfile

# Initialize
torch.manual_seed(0)
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

# Load phonemizer
global_phonemizer = phonemizer.backend.EspeakBackend(
    language='en-us', preserve_punctuation=True, with_stress=True)
textcleaner = TextCleaner()

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

def load_model():
    """Load StyleTTS2 model."""
    config = yaml.safe_load(open("Models/LJSpeech/config.yml"))

    # Load pretrained ASR model
    from utils import load_ASR_models, load_F0_models
    ASR_config = config.get('ASR_config', False)
    ASR_path = config.get('ASR_path', False)
    text_aligner = load_ASR_models(ASR_path, ASR_config)

    # Load pretrained F0 model
    F0_path = config.get('F0_path', False)
    pitch_extractor = load_F0_models(F0_path)

    # Load BERT model
    from Utils.PLBERT.util import load_plbert
    BERT_path = config.get('PLBERT_dir', False)
    plbert = load_plbert(BERT_path)

    # Build model
    model = build_model(recursive_munch(config['model_params']), text_aligner, pitch_extractor, plbert)
    _ = [model[key].eval() for key in model]
    _ = [model[key].to(device) for key in model]

    # Load weights
    params_whole = torch.load("Models/LJSpeech/epoch_2nd_00100.pth", map_location='cpu')
    params = params_whole['net']

    for key in model:
        if key in params:
            print(f'{key} loaded')
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
    return model

def inference_with_logging(model, text, output_wav_path):
    """Run inference with detailed logging of intermediate tensors."""
    text = text.strip().replace('"', '')

    # Phonemize
    ps = global_phonemizer.phonemize([text])
    ps = word_tokenize(ps[0])
    ps = ' '.join(ps)
    print(f"Phonemes: {ps}")

    # Tokenize
    tokens = textcleaner(ps)
    tokens.insert(0, 0)  # Add start token
    tokens_tensor = torch.LongTensor(tokens).to(device).unsqueeze(0)
    print(f"Tokens: {tokens}")
    print(f"Token tensor shape: {tokens_tensor.shape}")

    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens_tensor.shape[-1]]).to(tokens_tensor.device)
        text_mask = length_to_mask(input_lengths).to(tokens_tensor.device)

        # 1. Text Encoder
        t_en = model.text_encoder(tokens_tensor, input_lengths, text_mask)
        print(f"\n=== TEXT ENCODER OUTPUT ===")
        print(f"Shape: {t_en.shape}")
        print(f"Range: [{t_en.min().item():.4f}, {t_en.max().item():.4f}]")
        print(f"Mean: {t_en.mean().item():.4f}")

        # 2. BERT
        bert_dur = model.bert(tokens_tensor, attention_mask=(~text_mask).int())
        print(f"\n=== BERT OUTPUT ===")
        print(f"Shape: {bert_dur.shape}")
        print(f"Range: [{bert_dur.min().item():.4f}, {bert_dur.max().item():.4f}]")
        print(f"Mean: {bert_dur.mean().item():.4f}")

        # 3. BERT Encoder (produces 512-dim)
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
        print(f"\n=== BERT ENCODER OUTPUT (d_en) ===")
        print(f"Shape: {d_en.shape}")
        print(f"Range: [{d_en.min().item():.4f}, {d_en.max().item():.4f}]")
        print(f"Mean: {d_en.mean().item():.4f}")

        # 4. Style - use fixed style instead of diffusion for determinism
        # Zero noise gives consistent style
        s = torch.zeros(1, 128).to(device)  # style vector
        ref = torch.zeros(1, 128).to(device)  # reference embedding
        print(f"\n=== STYLE (s) ===")
        print(f"Shape: {s.shape}")
        print(f"Using zero style for deterministic comparison")

        # 5. Duration Encoder (predictor.text_encoder) - produces 640-dim
        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        print(f"\n=== DURATION ENCODER OUTPUT (d) ===")
        print(f"Shape: {d.shape}")
        print(f"Range: [{d.min().item():.4f}, {d.max().item():.4f}]")
        print(f"Mean: {d.mean().item():.4f}")

        # 6. LSTM on Duration Encoder output
        x, _ = model.predictor.lstm(d)
        print(f"\n=== LSTM OUTPUT (x) ===")
        print(f"Shape: {x.shape}")
        print(f"Range: [{x.min().item():.4f}, {x.max().item():.4f}]")
        print(f"Mean: {x.mean().item():.4f}")

        # 7. Duration projection
        duration_logits = model.predictor.duration_proj(x)
        print(f"\n=== DURATION LOGITS ===")
        print(f"Shape: {duration_logits.shape}")
        print(f"Range: [{duration_logits.min().item():.4f}, {duration_logits.max().item():.4f}]")
        print(f"Mean: {duration_logits.mean().item():.4f}")

        # 8. Duration computation
        duration = torch.sigmoid(duration_logits).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)
        pred_dur[-1] += 5  # Add padding to last phoneme
        print(f"\n=== PREDICTED DURATIONS ===")
        print(f"Durations: {pred_dur.tolist()}")
        print(f"Total frames: {int(pred_dur.sum().item())}")

        # 9. Build alignment matrix
        pred_aln_trg = torch.zeros(input_lengths.item(), int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        # 10. Expand duration features for F0/N prediction
        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
        print(f"\n=== EXPANDED FEATURES (en) FOR F0/N ===")
        print(f"Shape: {en.shape}")
        print(f"Range: [{en.min().item():.4f}, {en.max().item():.4f}]")
        print(f"Mean: {en.mean().item():.4f}")

        # 11. F0 and N prediction
        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
        print(f"\n=== F0 PREDICTION ===")
        print(f"Shape: {F0_pred.shape}")
        print(f"Range: [{F0_pred.min().item():.4f}, {F0_pred.max().item():.4f}]")
        print(f"Mean: {F0_pred.mean().item():.4f}")

        print(f"\n=== N PREDICTION ===")
        print(f"Shape: {N_pred.shape}")
        print(f"Range: [{N_pred.min().item():.4f}, {N_pred.max().item():.4f}]")
        print(f"Mean: {N_pred.mean().item():.4f}")

        # 12. Expand text encoder output
        aln = pred_aln_trg.unsqueeze(0).to(device)
        t_en_expanded = t_en @ aln
        print(f"\n=== EXPANDED TEXT ENCODER ===")
        print(f"Shape: {t_en_expanded.shape}")

        # 13. Decode
        out = model.decoder(t_en_expanded, F0_pred, N_pred, ref.squeeze().unsqueeze(0))
        print(f"\n=== DECODER OUTPUT ===")
        print(f"Shape: {out.shape}")
        print(f"Range: [{out.min().item():.4f}, {out.max().item():.4f}]")

        # Convert to audio
        wav = out.squeeze().cpu().numpy()
        print(f"\n=== AUDIO OUTPUT ===")
        print(f"Samples: {len(wav)}")
        print(f"Duration: {len(wav)/24000:.2f}s")
        print(f"Range: [{wav.min():.4f}, {wav.max():.4f}]")

        # Save WAV
        wav_int16 = (wav * 32767).astype(np.int16)
        wavfile.write(output_wav_path, 24000, wav_int16)
        print(f"\nSaved to: {output_wav_path}")

        return wav

def main():
    text = sys.argv[1] if len(sys.argv) > 1 else "Hello world"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/python_tts_reference.wav"

    print(f"Text: {text}")
    print(f"Output: {output_path}")
    print("=" * 60)

    model = load_model()
    wav = inference_with_logging(model, text, output_path)

    print("\n" + "=" * 60)
    print("DONE - Play with: afplay " + output_path)

if __name__ == "__main__":
    main()
