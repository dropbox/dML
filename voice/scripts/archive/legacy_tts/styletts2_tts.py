#!/usr/bin/env python3
"""
StyleTTS2 TTS Wrapper

SOTA English TTS with MOS 4.3+ quality using style diffusion.

Usage:
    python scripts/styletts2_tts.py "Text to speak" -o output.wav
    python scripts/styletts2_tts.py "Hello world" -o output.wav --diffusion-steps 10

Language: English only (no Japanese support)

Performance:
    - Model load: ~3-5s
    - Synthesis: ~100-500ms (5 diffusion steps)
    - Quality: SOTA (MOS 4.3+)
"""

import argparse
import sys
import time
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Global cache
_model = None
_phonemizer = None
_sampler = None
_text_cleaner = None
_device = None

# StyleTTS2 repo path
STYLETTS2_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "styletts2")
MODEL_PATH = os.path.join(STYLETTS2_PATH, "Models", "LJSpeech")


def load_model():
    """Load StyleTTS2 model (cached)."""
    global _model, _phonemizer, _sampler, _text_cleaner, _device

    if _model is not None:
        return _model, _phonemizer, _sampler, _text_cleaner, _device

    print("Loading StyleTTS2 model...", file=sys.stderr)
    start = time.time()

    # Change to StyleTTS2 directory and add to path
    original_cwd = os.getcwd()
    os.chdir(STYLETTS2_PATH)
    sys.path.insert(0, STYLETTS2_PATH)

    import torch
    import yaml
    from munch import Munch
    import phonemizer

    # Import StyleTTS2 modules
    from models import build_model, load_ASR_models, load_F0_models
    from utils import recursive_munch
    from text_utils import TextCleaner
    from Utils.PLBERT.util import load_plbert
    from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

    # Set device - use CPU for stable results (MPS has numerical issues with some ops)
    _device = 'cpu'  # MPS has issues with diffusion sampling
    print(f"Using device: {_device}", file=sys.stderr)

    # Load config
    config_path = os.path.join(MODEL_PATH, "config.yml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Override paths for utils
    config['F0_path'] = os.path.join(STYLETTS2_PATH, config['F0_path'])
    config['ASR_path'] = os.path.join(STYLETTS2_PATH, config['ASR_path'])
    config['ASR_config'] = os.path.join(STYLETTS2_PATH, config['ASR_config'])
    config['PLBERT_dir'] = os.path.join(STYLETTS2_PATH, config['PLBERT_dir'])

    # Load auxiliary models
    text_aligner = load_ASR_models(config['ASR_path'], config['ASR_config'])
    pitch_extractor = load_F0_models(config['F0_path'])
    plbert = load_plbert(config['PLBERT_dir'])

    # Build model
    _model = build_model(recursive_munch(config['model_params']), text_aligner, pitch_extractor, plbert)

    # Load checkpoint
    checkpoint_path = os.path.join(MODEL_PATH, "epoch_2nd_00100.pth")
    print(f"Loading checkpoint: {checkpoint_path}", file=sys.stderr)
    params_whole = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    params = params_whole['net']

    # Load weights
    for key in _model:
        if key in params:
            try:
                _model[key].load_state_dict(params[key])
            except Exception:
                # Handle module. prefix
                from collections import OrderedDict
                state_dict = params[key]
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] if k.startswith('module.') else k
                    new_state_dict[name] = v
                _model[key].load_state_dict(new_state_dict, strict=False)

        _model[key].to(_device)
        _model[key].eval()

    # Load phonemizer
    _phonemizer = phonemizer.backend.EspeakBackend(
        language='en-us',
        preserve_punctuation=True,
        with_stress=True
    )

    # Load text cleaner
    _text_cleaner = TextCleaner()

    # Create sampler
    _sampler = DiffusionSampler(
        _model['diffusion'].diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
        clamp=False
    )

    print(f"Model loaded in {time.time()-start:.1f}s", file=sys.stderr)

    return _model, _phonemizer, _sampler, _text_cleaner, _device


def length_to_mask(lengths, device):
    """Create attention mask from lengths."""
    import torch
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask


def synthesize(text: str, output_path: str, diffusion_steps: int = 5, embedding_scale: float = 1.0) -> bool:
    """
    Synthesize speech from text.

    Args:
        text: English text to synthesize
        output_path: Path to save WAV file
        diffusion_steps: Number of diffusion steps (more = better quality, slower)
        embedding_scale: Style embedding scale (higher = more expressive)

    Returns:
        True if successful
    """
    try:
        import torch
        import soundfile as sf
        from nltk.tokenize import word_tokenize

        model, phonemizer, sampler, text_cleaner, device = load_model()

        print(f"Synthesizing: {text[:60]}...", file=sys.stderr)
        start = time.time()

        # Preprocess text
        text = text.strip().replace('"', '')
        ps = phonemizer.phonemize([text])
        ps = word_tokenize(ps[0])
        ps = ' '.join(ps)

        # Tokenize
        tokens = text_cleaner(ps)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)

        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
            text_mask = length_to_mask(input_lengths, device)

            # Text encoding
            t_en = model['text_encoder'](tokens, input_lengths, text_mask)
            bert_dur = model['bert'](tokens, attention_mask=(~text_mask).int())
            d_en = model['bert_encoder'](bert_dur).transpose(-1, -2)

            # Style diffusion
            noise = torch.randn(1, 1, 256).to(device)
            s_pred = sampler(
                noise,
                embedding=bert_dur[0].unsqueeze(0),
                num_steps=diffusion_steps,
                embedding_scale=embedding_scale
            ).squeeze(0)

            s = s_pred[:, 128:]
            ref = s_pred[:, :128]

            # Duration prediction
            d = model['predictor'].text_encoder(d_en, s, input_lengths, text_mask)
            x, _ = model['predictor'].lstm(d)
            duration = model['predictor'].duration_proj(x)
            duration = torch.sigmoid(duration).sum(axis=-1)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)
            pred_dur[-1] += 5  # Add padding at end

            # Alignment
            pred_aln_trg = torch.zeros(input_lengths.item(), int(pred_dur.sum().item()))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].item())] = 1
                c_frame += int(pred_dur[i].item())
            pred_aln_trg = pred_aln_trg.to(device)

            # Prosody encoding
            en = d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0)
            F0_pred, N_pred = model['predictor'].F0Ntrain(en, s)

            # Decode
            out = model['decoder'](
                t_en @ pred_aln_trg.unsqueeze(0),
                F0_pred, N_pred,
                ref.squeeze().unsqueeze(0)
            )

            wav = out.squeeze().cpu().numpy()

        # Save audio
        sf.write(output_path, wav, 24000)

        duration_sec = len(wav) / 24000
        elapsed = time.time() - start
        rtf = elapsed / duration_sec

        print(f"Synthesized {duration_sec:.2f}s audio in {elapsed:.2f}s (RTF={rtf:.2f})", file=sys.stderr)
        print(f"Saved to: {output_path}", file=sys.stderr)

        return True

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="StyleTTS2 TTS - SOTA English speech synthesis")
    parser.add_argument("text", nargs='?', help="Text to synthesize")
    parser.add_argument("-o", "--output", required=True, help="Output WAV file path")
    parser.add_argument("-l", "--language", default="en", help="Language (en only)")
    parser.add_argument("--diffusion-steps", type=int, default=5, help="Diffusion steps (5-10)")
    parser.add_argument("--embedding-scale", type=float, default=1.0, help="Style scale (1.0-2.0)")
    parser.add_argument("--play", action="store_true", help="Play audio after synthesis")

    args = parser.parse_args()

    # Get text from stdin if not provided
    if args.text:
        text = args.text
    else:
        text = sys.stdin.read().strip()

    if not text:
        print("Error: No text provided", file=sys.stderr)
        sys.exit(1)

    if args.language.lower() not in ('en', 'eng', 'english'):
        print("Warning: StyleTTS2 only supports English", file=sys.stderr)

    success = synthesize(
        text,
        args.output,
        diffusion_steps=args.diffusion_steps,
        embedding_scale=args.embedding_scale
    )

    if success and args.play:
        import subprocess
        subprocess.run(["afplay", args.output])

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
