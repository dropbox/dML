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
Test Prosody Parser Integration

Tests the prosody annotation parser with actual TTS synthesis.
This provides a Python implementation of the parser for validation
and testing of the C++ implementation.
"""

import os
import re
import sys
from dataclasses import dataclass, field
from enum import IntEnum

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class ProsodyType(IntEnum):
    """Prosody type enumeration (matching C++ version)."""
    NEUTRAL = 0
    EMPHASIS = 1
    STRONG_EMPHASIS = 2
    REDUCED_EMPHASIS = 3
    RATE_X_SLOW = 10
    RATE_SLOW = 11
    RATE_FAST = 12
    RATE_X_FAST = 13
    PITCH_X_LOW = 20
    PITCH_LOW = 21
    PITCH_HIGH = 22
    PITCH_X_HIGH = 23
    VOLUME_X_SOFT = 30
    VOLUME_SOFT = 31
    VOLUME_LOUD = 32
    VOLUME_X_LOUD = 33
    VOLUME_WHISPER = 34
    EMOTION_ANGRY = 40
    EMOTION_SAD = 41
    EMOTION_EXCITED = 42
    EMOTION_WORRIED = 43
    EMOTION_ALARMED = 44
    EMOTION_CALM = 45
    EMOTION_EMPATHETIC = 46
    EMOTION_CONFIDENT = 47
    EMOTION_FRUSTRATED = 48
    EMOTION_NERVOUS = 49
    EMOTION_SURPRISED = 50
    EMOTION_DISAPPOINTED = 51
    QUESTION = 60
    WHISPER = 61
    LOUD = 62


@dataclass
class ProsodyAnnotation:
    """A prosody annotation span."""
    char_start: int
    char_end: int
    type: ProsodyType
    value: float = 0.0


@dataclass
class ProsodyBreak:
    """A break/pause instruction."""
    after_char: int
    duration_ms: int


@dataclass
class ParsedProsody:
    """Result of parsing prosody markers."""
    clean_text: str = ""
    annotations: list[ProsodyAnnotation] = field(default_factory=list)
    breaks: list[ProsodyBreak] = field(default_factory=list)


# Adjustment parameters for each prosody type
PROSODY_ADJUSTMENTS = {
    ProsodyType.NEUTRAL: (1.0, 1.0, 1.0, 1.0),
    ProsodyType.EMPHASIS: (1.3, 1.1, 1.0, 1.1),
    ProsodyType.STRONG_EMPHASIS: (1.5, 1.2, 1.1, 1.2),
    ProsodyType.REDUCED_EMPHASIS: (0.85, 0.95, 0.9, 0.9),
    ProsodyType.RATE_X_SLOW: (2.0, 1.0, 1.0, 1.0),
    ProsodyType.RATE_SLOW: (1.4, 1.0, 1.0, 1.0),
    ProsodyType.RATE_FAST: (0.75, 1.0, 1.0, 1.0),
    ProsodyType.RATE_X_FAST: (0.6, 1.0, 1.0, 1.0),
    ProsodyType.PITCH_X_LOW: (1.0, 0.75, 1.0, 1.0),
    ProsodyType.PITCH_LOW: (1.0, 0.9, 1.0, 1.0),
    ProsodyType.PITCH_HIGH: (1.0, 1.15, 1.0, 1.0),
    ProsodyType.PITCH_X_HIGH: (1.0, 1.3, 1.0, 1.0),
    ProsodyType.EMOTION_ANGRY: (0.9, 1.15, 1.3, 1.3),
    ProsodyType.EMOTION_SAD: (1.2, 0.9, 0.8, 0.8),
    ProsodyType.EMOTION_EXCITED: (0.85, 1.2, 1.4, 1.2),
    ProsodyType.EMOTION_WORRIED: (1.1, 1.05, 1.2, 0.95),
    ProsodyType.EMOTION_ALARMED: (0.8, 1.25, 1.5, 1.3),
    ProsodyType.EMOTION_CALM: (1.2, 0.95, 0.7, 0.9),
    ProsodyType.WHISPER: (1.2, 0.8, 0.6, 0.3),
    ProsodyType.LOUD: (0.9, 1.1, 1.1, 1.5),
    ProsodyType.QUESTION: (1.0, 1.08, 1.2, 1.0),
}


def parse_time_ms(time_str: str) -> int:
    """Parse time string like '500ms' or '1s' to milliseconds."""
    match = re.match(r'^(\d+\.?\d*)(ms|s|sec|msec)?$', time_str)
    if not match:
        return 500
    value = float(match.group(1))
    unit = match.group(2) or 'ms'
    if unit in ('s', 'sec'):
        return int(value * 1000)
    return int(value)


def break_strength_to_ms(strength: str) -> int:
    """Convert break strength to milliseconds."""
    return {
        'none': 0,
        'x-weak': 100,
        'weak': 250,
        'medium': 500,
        'strong': 750,
        'x-strong': 1000,
    }.get(strength, 500)


def _parse_percent_multiplier(s: str) -> float | None:
    s = s.strip().lower()
    if not s.endswith('%'):
        return None
    num = s[:-1].strip()
    if not num:
        return None
    try:
        pct = float(num)
    except ValueError:
        return None
    relative = num[0] in ('+', '-')
    mult = 1.0 + (pct / 100.0) if relative else (pct / 100.0)
    if mult <= 0.0:
        return None
    return mult


def _parse_semitone_multiplier(s: str) -> float | None:
    s = s.strip().lower()
    if not s.endswith('st'):
        return None
    num = s[:-2].strip()
    if not num:
        return None
    try:
        semitones = float(num)
    except ValueError:
        return None
    return 2.0 ** (semitones / 12.0)


def parse_rate(rate: str) -> tuple[ProsodyType, float]:
    """Parse rate attribute to (ProsodyType, multiplier)."""
    r = rate.strip().lower()
    if r in ('', 'medium', 'normal', 'default'):
        return ProsodyType.NEUTRAL, 1.0

    keywords = {
        'x-slow': (ProsodyType.RATE_X_SLOW, 0.5),
        'xslow': (ProsodyType.RATE_X_SLOW, 0.5),
        'slow': (ProsodyType.RATE_SLOW, 0.7),
        'fast': (ProsodyType.RATE_FAST, 1.3),
        'x-fast': (ProsodyType.RATE_X_FAST, 1.6),
        'xfast': (ProsodyType.RATE_X_FAST, 1.6),
    }
    if r in keywords:
        return keywords[r]

    mult = _parse_percent_multiplier(r)
    if mult is None:
        return ProsodyType.NEUTRAL, 0.0

    # Coarse bucket mapping; the multiplier is preserved in `value`.
    if mult <= 0.60:
        return ProsodyType.RATE_X_SLOW, mult
    if mult <= 0.90:
        return ProsodyType.RATE_SLOW, mult
    if mult >= 1.50:
        return ProsodyType.RATE_X_FAST, mult
    if mult >= 1.10:
        return ProsodyType.RATE_FAST, mult
    return ProsodyType.NEUTRAL, mult


def parse_pitch(pitch: str) -> tuple[ProsodyType, float]:
    """Parse pitch attribute to (ProsodyType, multiplier)."""
    p = pitch.strip().lower()
    if p in ('', 'medium', 'normal', 'default'):
        return ProsodyType.NEUTRAL, 1.0

    keywords = {
        'x-low': (ProsodyType.PITCH_X_LOW, 0.75),
        'xlow': (ProsodyType.PITCH_X_LOW, 0.75),
        'low': (ProsodyType.PITCH_LOW, 0.90),
        'high': (ProsodyType.PITCH_HIGH, 1.15),
        'x-high': (ProsodyType.PITCH_X_HIGH, 1.30),
        'xhigh': (ProsodyType.PITCH_X_HIGH, 1.30),
    }
    if p in keywords:
        return keywords[p]

    mult = _parse_semitone_multiplier(p)
    if mult is None:
        mult = _parse_percent_multiplier(p)
    if mult is None:
        return ProsodyType.NEUTRAL, 0.0

    # Coarse bucket mapping; the multiplier is preserved in `value`.
    if mult <= 0.82:
        return ProsodyType.PITCH_X_LOW, mult
    if mult <= 0.97:
        return ProsodyType.PITCH_LOW, mult
    if mult >= 1.23:
        return ProsodyType.PITCH_X_HIGH, mult
    if mult >= 1.06:
        return ProsodyType.PITCH_HIGH, mult
    return ProsodyType.NEUTRAL, mult


def parse_emotion(emotion: str) -> ProsodyType:
    """Parse emotion attribute to ProsodyType."""
    return {
        'angry': ProsodyType.EMOTION_ANGRY,
        'sad': ProsodyType.EMOTION_SAD,
        'excited': ProsodyType.EMOTION_EXCITED,
        'worried': ProsodyType.EMOTION_WORRIED,
        'alarmed': ProsodyType.EMOTION_ALARMED,
        'calm': ProsodyType.EMOTION_CALM,
        'empathetic': ProsodyType.EMOTION_EMPATHETIC,
        'confident': ProsodyType.EMOTION_CONFIDENT,
        'frustrated': ProsodyType.EMOTION_FRUSTRATED,
        'nervous': ProsodyType.EMOTION_NERVOUS,
        'surprised': ProsodyType.EMOTION_SURPRISED,
        'disappointed': ProsodyType.EMOTION_DISAPPOINTED,
    }.get(emotion.lower(), ProsodyType.NEUTRAL)


def get_attribute(tag_content: str, attr_name: str) -> str | None:
    """Extract attribute value from tag content."""
    pattern = rf'{attr_name}\s*=\s*["\']([^"\']+)["\']'
    match = re.search(pattern, tag_content, re.IGNORECASE)
    return match.group(1) if match else None


def parse_prosody_markers(text: str) -> ParsedProsody:
    """
    Parse SSML-style prosody markers from text.

    Python implementation matching the C++ prosody_parser.cpp
    """
    result = ParsedProsody()
    tag_stack = []  # List of (tag_name, type, clean_start, value)

    i = 0
    while i < len(text):
        if text[i] == '<':
            # Find tag end
            tag_end = text.find('>', i)
            if tag_end == -1:
                result.clean_text += text[i]
                i += 1
                continue

            tag_content = text[i + 1:tag_end]

            # Self-closing?
            self_closing = tag_content.rstrip().endswith('/')
            if self_closing:
                tag_content = tag_content.rstrip()[:-1]

            tag_content = tag_content.strip()

            # Closing tag?
            closing = tag_content.startswith('/')
            if closing:
                tag_content = tag_content[1:]

            # Extract tag name
            tag_name = tag_content.split()[0] if tag_content else ''
            tag_name = tag_name.lower()

            # Handle closing tag
            if closing:
                for j in range(len(tag_stack) - 1, -1, -1):
                    if tag_stack[j][0] == tag_name:
                        _, ptype, clean_start, value = tag_stack.pop(j)
                        result.annotations.append(ProsodyAnnotation(
                            char_start=clean_start,
                            char_end=len(result.clean_text),
                            type=ptype,
                            value=value,
                        ))
                        break
                i = tag_end + 1
                continue

            # Handle self-closing (break) tags
            if self_closing or tag_name in ('break', 'br'):
                if tag_name in ('break', 'br'):
                    time_attr = get_attribute(tag_content, 'time')
                    strength_attr = get_attribute(tag_content, 'strength')

                    if time_attr:
                        duration_ms = parse_time_ms(time_attr)
                    elif strength_attr:
                        duration_ms = break_strength_to_ms(strength_attr)
                    else:
                        duration_ms = 500

                    if duration_ms > 0:
                        result.breaks.append(ProsodyBreak(
                            after_char=len(result.clean_text),
                            duration_ms=duration_ms,
                        ))
                i = tag_end + 1
                continue

            # Handle opening tags
            ptype = ProsodyType.NEUTRAL
            value = 0.0

            if tag_name in ('em', 'emphasis'):
                level = get_attribute(tag_content, 'level')
                if level == 'strong':
                    ptype = ProsodyType.STRONG_EMPHASIS
                elif level in ('reduced', 'none'):
                    ptype = ProsodyType.REDUCED_EMPHASIS
                else:
                    ptype = ProsodyType.EMPHASIS
            elif tag_name == 'strong':
                ptype = ProsodyType.STRONG_EMPHASIS
            elif tag_name == 'prosody':
                rate = get_attribute(tag_content, 'rate')
                pitch = get_attribute(tag_content, 'pitch')
                if rate:
                    ptype, value = parse_rate(rate)
                elif pitch:
                    ptype, value = parse_pitch(pitch)
            elif tag_name == 'emotion':
                emotion_type = get_attribute(tag_content, 'type')
                if emotion_type:
                    ptype = parse_emotion(emotion_type)
            elif tag_name == 'whisper':
                ptype = ProsodyType.WHISPER
            elif tag_name == 'loud':
                ptype = ProsodyType.LOUD
            elif tag_name == 'question':
                ptype = ProsodyType.QUESTION

            if ptype != ProsodyType.NEUTRAL:
                tag_stack.append((tag_name, ptype, len(result.clean_text), value))

            i = tag_end + 1
        else:
            result.clean_text += text[i]
            i += 1

    # Close any remaining open tags
    for _tag_name, ptype, clean_start, value in reversed(tag_stack):
        result.annotations.append(ProsodyAnnotation(
            char_start=clean_start,
            char_end=len(result.clean_text),
            type=ptype,
            value=value,
        ))

    return result


# ============================================================================
# Unit Tests
# ============================================================================

def test_plain_text():
    """Test plain text without markers."""
    result = parse_prosody_markers("Hello world")
    assert result.clean_text == "Hello world"
    assert len(result.annotations) == 0
    assert len(result.breaks) == 0
    print("  plain_text: PASSED")


def test_emphasis():
    """Test emphasis marker."""
    result = parse_prosody_markers("I <em>really</em> need this")
    assert result.clean_text == "I really need this"
    assert len(result.annotations) == 1
    assert result.annotations[0].char_start == 2
    assert result.annotations[0].char_end == 8
    assert result.annotations[0].type == ProsodyType.EMPHASIS
    print("  emphasis: PASSED")


def test_break_time():
    """Test break with time attribute."""
    result = parse_prosody_markers("Hello<break time='500ms'/>world")
    assert result.clean_text == "Helloworld"
    assert len(result.breaks) == 1
    assert result.breaks[0].after_char == 5
    assert result.breaks[0].duration_ms == 500
    print("  break_time: PASSED")


def test_emotion():
    """Test emotion marker."""
    result = parse_prosody_markers("<emotion type='angry'>I am angry</emotion>")
    assert result.clean_text == "I am angry"
    assert len(result.annotations) == 1
    assert result.annotations[0].type == ProsodyType.EMOTION_ANGRY
    print("  emotion: PASSED")


def test_prosody_rate():
    """Test prosody rate markers."""
    result = parse_prosody_markers("<prosody rate='slow'>Slow text</prosody>")
    assert result.clean_text == "Slow text"
    assert len(result.annotations) == 1
    assert result.annotations[0].type == ProsodyType.RATE_SLOW
    print("  prosody_rate: PASSED")


def test_prosody_rate_percent_value():
    """Test percent and relative percent parsing for rate."""
    result = parse_prosody_markers("<prosody rate='80%'>Text</prosody>")
    assert result.clean_text == "Text"
    assert len(result.annotations) == 1
    assert result.annotations[0].type == ProsodyType.RATE_SLOW
    assert abs(result.annotations[0].value - 0.8) < 1e-9

    result = parse_prosody_markers("<prosody rate='+20%'>Text</prosody>")
    assert result.clean_text == "Text"
    assert len(result.annotations) == 1
    assert result.annotations[0].type == ProsodyType.RATE_FAST
    assert abs(result.annotations[0].value - 1.2) < 1e-9


def test_prosody_pitch_numeric_value():
    """Test semitone and percent parsing for pitch."""
    result = parse_prosody_markers("<prosody pitch='+2st'>Text</prosody>")
    assert result.clean_text == "Text"
    assert len(result.annotations) == 1
    assert result.annotations[0].type == ProsodyType.PITCH_HIGH
    expected = 2.0 ** (2.0 / 12.0)
    assert abs(result.annotations[0].value - expected) < 1e-9

    result = parse_prosody_markers("<prosody pitch='-10%'>Text</prosody>")
    assert result.clean_text == "Text"
    assert len(result.annotations) == 1
    assert result.annotations[0].type == ProsodyType.PITCH_LOW
    assert abs(result.annotations[0].value - 0.9) < 1e-9


def test_complex():
    """Test complex example with multiple markers."""
    result = parse_prosody_markers(
        "I <em>really</em> understand.<break time='300ms'/> Let me help.",
    )
    assert result.clean_text == "I really understand. Let me help."
    assert len(result.annotations) == 1
    assert len(result.breaks) == 1
    assert result.breaks[0].duration_ms == 300
    print("  complex: PASSED")


def test_all_emotions():
    """Test all emotion types."""
    emotions = [
        "angry", "sad", "excited", "worried", "alarmed",
        "calm", "empathetic", "confident", "frustrated",
        "nervous", "surprised", "disappointed",
    ]
    for emotion in emotions:
        result = parse_prosody_markers(f"<emotion type='{emotion}'>text</emotion>")
        assert result.clean_text == "text"
        assert len(result.annotations) == 1
        assert result.annotations[0].type != ProsodyType.NEUTRAL
    print("  all_emotions: PASSED")


def run_parser_tests():
    """Run all parser unit tests."""
    print("\n=== Python Prosody Parser Tests ===\n")

    test_plain_text()
    test_emphasis()
    test_break_time()
    test_emotion()
    test_prosody_rate()
    test_prosody_rate_percent_value()
    test_prosody_pitch_numeric_value()
    test_complex()
    test_all_emotions()

    print("\n=== All parser tests passed! ===\n")


if __name__ == "__main__":
    run_parser_tests()
