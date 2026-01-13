#!/usr/bin/env python3
# Copyright 2024-2025 Andrew Yates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Canonical Label Taxonomy for Whisper SST Training.

This module is the SINGLE SOURCE OF TRUTH for all label taxonomies used in
training and inference. All training scripts, evaluation code, and tests
should import from this module.

References:
    - DATA_QA_AND_LABEL_TAXONOMY_PLAN_2026-01-02-14-34.md
    - WORKER_ROADMAP_20260102.md

Version: 2.0 (9-class emotion taxonomy with contempt)
"""


# =============================================================================
# EMOTION TAXONOMY
# =============================================================================

# Canonical 9-class emotion taxonomy
# Extends RAVDESS 8-class with contempt as a first-class label
EMOTION_CLASSES_9: list[str] = [
    "neutral",    # 0 - No emotional expression
    "calm",       # 1 - Relaxed, peaceful state
    "happy",      # 2 - Joy, pleasure, contentment
    "sad",        # 3 - Sorrow, grief, unhappiness
    "angry",      # 4 - Anger, frustration, irritation
    "fearful",    # 5 - Fear, anxiety, worry
    "disgust",    # 6 - Disgust, revulsion, distaste
    "surprised",  # 7 - Surprise, astonishment, shock
    "contempt",   # 8 - Contempt, scorn, disdain (NEW in v2.0)
]

# Backward compatibility alias
EMOTION_CLASSES_8 = EMOTION_CLASSES_9[:8]  # Without contempt

# Extended 34-class emotion taxonomy for Expresso-style datasets
EMOTION_CLASSES_34: list[str] = [
    # Core emotions (0-8) - matches EMOTION_CLASSES_9
    "neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised", "contempt",
    # Extended emotions (9-33)
    "excited",      # 9
    "bored",        # 10
    "confused",     # 11
    "disappointed", # 12
    "embarrassed",  # 13
    "frustrated",   # 14
    "guilty",       # 15
    "hopeful",      # 16
    "hurt",         # 17
    "jealous",      # 18
    "lonely",       # 19
    "nervous",      # 20
    "nostalgic",    # 21
    "proud",        # 22
    "relieved",     # 23
    "romantic",     # 24
    "skeptical",    # 25
    "sympathetic",  # 26
    "tender",       # 27
    "tired",        # 28
    "amused",       # 29
    "annoyed",      # 30
    "curious",      # 31
    "determined",   # 32
    "grateful",     # 33
]

# =============================================================================
# EMOTION LABEL MAPPINGS (Dataset -> Canonical)
# =============================================================================

def get_emotion_to_id_9() -> dict[str, int]:
    """Get the canonical emotion-to-ID mapping for 9-class taxonomy.

    Includes common aliases like 'fear' -> 'fearful', 'surprise' -> 'surprised'.
    """
    mapping = {emotion: idx for idx, emotion in enumerate(EMOTION_CLASSES_9)}

    # Add common aliases
    mapping['fear'] = mapping['fearful']        # 5
    mapping['surprise'] = mapping['surprised']  # 7
    mapping['other'] = mapping['neutral']       # Map unknown to neutral

    return mapping


# RAVDESS dataset mapping (native 8-class, no contempt)
RAVDESS_TO_EMOTION_9: dict[str, int] = {
    'neutral': 0,
    'calm': 1,
    'happy': 2,
    'sad': 3,
    'angry': 4,
    'fearful': 5,
    'disgust': 6,
    'surprised': 7,
    # RAVDESS doesn't have contempt, so no mapping for index 8
}

# CREMA-D dataset mapping
CREMA_D_TO_EMOTION_9: dict[str, int] = {
    'NEU': 0,  # neutral
    'HAP': 2,  # happy
    'SAD': 3,  # sad
    'ANG': 4,  # angry
    'FEA': 5,  # fearful
    'DIS': 6,  # disgust
    # CREMA-D doesn't have calm, surprised, or contempt
}

# MELD dataset mapping (from Friends TV show)
MELD_TO_EMOTION_9: dict[str, int] = {
    'neutral': 0,
    'joy': 2,       # -> happy
    'sadness': 3,   # -> sad
    'anger': 4,     # -> angry
    'fear': 5,      # -> fearful
    'disgust': 6,
    'surprise': 7,  # -> surprised
    # MELD doesn't have calm or contempt
}

# EmoV-DB dataset mapping
EMOV_DB_TO_EMOTION_9: dict[str, int] = {
    'neutral': 0,
    'amused': 2,    # -> happy (closest match)
    'angry': 4,
    'disgusted': 6, # -> disgust
    'sleepy': 1,    # -> calm (closest match)
}

# CMU-MOSEI dataset mapping
CMU_MOSEI_TO_EMOTION_9: dict[str, int] = {
    'happiness': 2,  # -> happy
    'sadness': 3,    # -> sad
    'anger': 4,      # -> angry
    'fear': 5,       # -> fearful
    'disgust': 6,
    'surprise': 7,   # -> surprised
}


# =============================================================================
# PARALINGUISTICS TAXONOMY (50-class)
# =============================================================================

PARALINGUISTICS_CLASSES_50: list[str] = [
    # Universal non-verbal (0-10)
    "speech",         # 0 - Normal speech (default)
    "laughter",       # 1 - Laughing sounds
    "cough",          # 2 - Coughing
    "sigh",           # 3 - Sighing (emotional exhale)
    "breath",         # 4 - Breathing (in/out)
    "cry",            # 5 - Crying/sobbing
    "yawn",           # 6 - Yawning
    "throat_clear",   # 7 - Throat clearing
    "sneeze",         # 8 - Sneezing
    "gasp",           # 9 - Sharp inhale (surprise)
    "groan",          # 10 - Groaning

    # English fillers (11-15)
    "um_en",          # 11 - English "um"
    "uh_en",          # 12 - English "uh"
    "hmm_en",         # 13 - English "hmm"
    "er_en",          # 14 - English "er"
    "ah_en",          # 15 - English "ah"

    # Mandarin Chinese fillers (16-19)
    "nage_zh",        # 16 - 那个 (nàge)
    "zhege_zh",       # 17 - 这个 (zhège)
    "jiushi_zh",      # 18 - 就是 (jiùshì)
    "en_zh",          # 19 - 嗯 (en)

    # Japanese fillers (20-24)
    "eto_ja",         # 20 - えと (eto)
    "ano_ja",         # 21 - あの (ano)
    "ee_ja",          # 22 - ええ (ee)
    "maa_ja",         # 23 - まあ (maa)
    "un_ja",          # 24 - うん (un)

    # Korean fillers (25-28)
    "eo_ko",          # 25 - 어 (eo)
    "eum_ko",         # 26 - 음 (eum)
    "geuge_ko",       # 27 - 그게 (geuge)
    "mwo_ko",         # 28 - 뭐 (mwo)

    # Hindi fillers (29-32)
    "matlab_hi",      # 29 - मतलब (matlab)
    "wo_hi",          # 30 - वो (wo)
    "yeh_hi",         # 31 - ये (yeh)
    "haan_hi",        # 32 - हाँ (haan)

    # Other major languages (33-39)
    "este_es",        # 33 - Spanish "this/um"
    "pues_es",        # 34 - Spanish "well"
    "euh_fr",         # 35 - French "uh"
    "ben_fr",         # 36 - French "well"
    "aeh_de",         # 37 - German "uh"
    "also_de",        # 38 - German "so/well"
    "yani_ar",        # 39 - Arabic يعني (ya'ni)

    # Singing vocalizations (40-49)
    "sing_a",         # 40 - "ahhh" vowel
    "sing_e",         # 41 - "ehh" vowel
    "sing_i",         # 42 - "eee" vowel
    "sing_o",         # 43 - "ooo" vowel
    "sing_u",         # 44 - "ooo" (as in "you")
    "vibrato",        # 45 - vibrato technique
    "trill",          # 46 - trill/trillo
    "vocal_fry",      # 47 - creaky voice
    "falsetto",       # 48 - head voice
    "belt",           # 49 - chest voice projection
]


# =============================================================================
# PARALINGUISTICS LABEL MAPPINGS (Dataset -> Canonical)
# =============================================================================

# VocalSound dataset mapping
VOCALSOUND_TO_PARA_50: dict[str, int] = {
    "Cough": 2,
    "Laughter": 1,
    "Sigh": 3,
    "Sneeze": 8,
    "Sniff": 8,  # Map to sneeze (closest)
    "Throat clearing": 7,
}

# ESC-50 dataset mapping (environmental sounds)
ESC50_TO_PARA_50: dict[str, int] = {
    "breathing": 4,
    "coughing": 2,
    "laughing": 1,
    "sneezing": 8,
    "snoring": 10,   # -> groan (closest non-speech)
    "crying": 5,     # -> cry
    "crying_baby": 5,
    "clapping": 0,   # -> speech (no good match)
}

# CoughVID dataset mapping
COUGHVID_TO_PARA_50: dict[str, int] = {
    "cough": 2,
    "breath": 4,
    "speech": 0,
}

# Fillers dataset mapping
FILLERS_TO_PARA_50: dict[str, int] = {
    "um": 11,    # um_en
    "uh": 12,    # uh_en
    "hmm": 13,   # hmm_en
    "er": 14,    # er_en
    "ah": 15,    # ah_en
}


# =============================================================================
# VALIDATION HELPERS
# =============================================================================

def validate_emotion_mapping(mapping: dict[str, int], allow_unmapped: bool = False) -> tuple[bool, list[str]]:
    """Validate that an emotion mapping produces valid indices.

    Args:
        mapping: Label string to index mapping
        allow_unmapped: If True, allow indices outside [0, 8]

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []
    max_idx = len(EMOTION_CLASSES_9) - 1

    for label, idx in mapping.items():
        if not isinstance(idx, int):
            errors.append(f"Label '{label}' has non-integer index: {idx}")
        elif idx < 0:
            errors.append(f"Label '{label}' has negative index: {idx}")
        elif not allow_unmapped and idx > max_idx:
            errors.append(f"Label '{label}' has index {idx} > max {max_idx}")

    return len(errors) == 0, errors


def validate_paralinguistics_mapping(mapping: dict[str, int]) -> tuple[bool, list[str]]:
    """Validate that a paralinguistics mapping produces valid indices.

    Args:
        mapping: Label string to index mapping

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []
    max_idx = len(PARALINGUISTICS_CLASSES_50) - 1

    for label, idx in mapping.items():
        if not isinstance(idx, int):
            errors.append(f"Label '{label}' has non-integer index: {idx}")
        elif idx < 0 or idx > max_idx:
            errors.append(f"Label '{label}' has invalid index {idx} (valid: 0-{max_idx})")

    return len(errors) == 0, errors


def validate_label_maps() -> tuple[bool, dict[str, list[str]]]:
    """Validate all defined label mappings.

    Returns:
        Tuple of (all_valid, dict of mapping_name -> error messages)
    """
    all_errors = {}

    # Emotion mappings
    emotion_maps = {
        'RAVDESS_TO_EMOTION_9': RAVDESS_TO_EMOTION_9,
        'CREMA_D_TO_EMOTION_9': CREMA_D_TO_EMOTION_9,
        'MELD_TO_EMOTION_9': MELD_TO_EMOTION_9,
        'EMOV_DB_TO_EMOTION_9': EMOV_DB_TO_EMOTION_9,
        'CMU_MOSEI_TO_EMOTION_9': CMU_MOSEI_TO_EMOTION_9,
    }

    for name, mapping in emotion_maps.items():
        valid, errors = validate_emotion_mapping(mapping)
        if not valid:
            all_errors[name] = errors

    # Paralinguistics mappings
    para_maps = {
        'VOCALSOUND_TO_PARA_50': VOCALSOUND_TO_PARA_50,
        'ESC50_TO_PARA_50': ESC50_TO_PARA_50,
        'COUGHVID_TO_PARA_50': COUGHVID_TO_PARA_50,
        'FILLERS_TO_PARA_50': FILLERS_TO_PARA_50,
    }

    for name, mapping in para_maps.items():
        valid, errors = validate_paralinguistics_mapping(mapping)
        if not valid:
            all_errors[name] = errors

    return len(all_errors) == 0, all_errors


# =============================================================================
# VERSION INFO
# =============================================================================

TAXONOMY_VERSION = "2.0"
TAXONOMY_DATE = "2026-01-02"
TAXONOMY_CHANGES = [
    "v2.0: Added contempt as 9th emotion class",
    "v2.0: Created single source of truth module",
    "v1.0: Original 8-class RAVDESS taxonomy",
]
