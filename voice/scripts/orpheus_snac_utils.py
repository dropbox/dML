#!/usr/bin/env python3
"""
Utilities for converting Orpheus-TTS audio tokens to SNAC codebooks.

Official decoder mapping (from canopyai/Orpheus-TTS):
- Index 0           -> codebook 0 (1 code per frame)
- Indices 1 and 4   -> codebook 1 (2 codes per frame)
- Indices 2,3,5,6   -> codebook 2 (4 codes per frame)

Orpheus emits 7 codes per audio frame starting at token id 128262.
SNAC expects 3 codebooks (1 + 2 + 4 codes), so we redistribute
the interleaved 7-code layout into SNAC tensors.
"""

from typing import List, Sequence

import torch

AUDIO_TOKEN_START = 128262
AUDIO_END_TOKEN = 128257
TOKENS_PER_FRAME = 7
CODEBOOK_SIZE = 4096


def extract_audio_codes(generated_ids: Sequence[int]) -> List[int]:
    """Return raw audio codes (token ids minus offset) from generated ids."""
    return [t - AUDIO_TOKEN_START for t in generated_ids if t >= AUDIO_TOKEN_START]


def snac_tensors_from_interleaved(raw_codes: Sequence[int], device: torch.device) -> list[torch.Tensor]:
    """
    Convert interleaved 7-code frames to SNAC's 3-codebook tensors.

    Drops any trailing partial frame.
    """
    num_frames = len(raw_codes) // TOKENS_PER_FRAME
    if num_frames == 0:
        raise ValueError("No full audio frames to decode (need at least 7 codes)")

    cb0, cb1, cb2 = [], [], []
    for i in range(num_frames):
        base = i * TOKENS_PER_FRAME
        cb0.append(raw_codes[base] % CODEBOOK_SIZE)
        cb1.extend([
            raw_codes[base + 1] % CODEBOOK_SIZE,
            raw_codes[base + 4] % CODEBOOK_SIZE,
        ])
        cb2.extend([
            raw_codes[base + 2] % CODEBOOK_SIZE,
            raw_codes[base + 3] % CODEBOOK_SIZE,
            raw_codes[base + 5] % CODEBOOK_SIZE,
            raw_codes[base + 6] % CODEBOOK_SIZE,
        ])

    return [
        torch.tensor(cb0, device=device).unsqueeze(0),
        torch.tensor(cb1, device=device).unsqueeze(0),
        torch.tensor(cb2, device=device).unsqueeze(0),
    ]
