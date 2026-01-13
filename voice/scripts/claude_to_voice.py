#!/usr/bin/env python3
"""
Claude Code Voice Integration - Phase 7
Streaming translation and TTS for Claude Code output.

Parses Claude JSON stream, buffers text to sentence boundaries,
then speaks each sentence via the TTS daemon with optional translation.

Usage:
    claude --json | python scripts/claude_to_voice.py --lang ja
    claude --json | python scripts/claude_to_voice.py --lang zh
    claude --json | python scripts/claude_to_voice.py --lang hi

Copyright 2025 Andrew Yates. All rights reserved.
"""

import sys
import os
import json
import re
import subprocess
import time
import argparse
from datetime import datetime
from typing import Optional, Generator

# ANSI colors for terminal output
BLUE = '\033[94m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
CYAN = '\033[96m'
MAGENTA = '\033[95m'
BOLD = '\033[1m'
DIM = '\033[2m'
RESET = '\033[0m'


class SentenceBuffer:
    """Buffers text and yields complete sentences."""

    # Sentence-ending punctuation patterns
    SENTENCE_ENDINGS = re.compile(
        r'(?<=[.!?])\s+|'          # English: period/exclamation/question + space
        r'(?<=[.!?])$|'            # English: end of string
        r'(?<=[。！？])|'           # Japanese/Chinese: full-width punctuation
        r'(?<=\n\n)'               # Double newline (paragraph break)
    )

    # Minimum sentence length to avoid speaking fragments
    MIN_SENTENCE_LENGTH = 10

    def __init__(self):
        self.buffer = ""
        self.last_flush_time = time.time()
        # Maximum time to hold text before forcing output
        self.max_hold_time = 3.0  # seconds

    def add_text(self, text: str) -> Generator[str, None, None]:
        """Add text to buffer, yield complete sentences."""
        self.buffer += text

        # Check for sentence boundaries
        parts = self.SENTENCE_ENDINGS.split(self.buffer)

        if len(parts) > 1:
            # Yield all complete sentences except the last incomplete one
            for sentence in parts[:-1]:
                sentence = sentence.strip()
                if len(sentence) >= self.MIN_SENTENCE_LENGTH:
                    self.last_flush_time = time.time()
                    yield sentence

            # Keep the incomplete last part in buffer
            self.buffer = parts[-1]

        # Force flush if buffer has been held too long
        elif (time.time() - self.last_flush_time > self.max_hold_time
              and len(self.buffer.strip()) >= self.MIN_SENTENCE_LENGTH):
            sentence = self.buffer.strip()
            self.buffer = ""
            self.last_flush_time = time.time()
            yield sentence

    def flush(self) -> Optional[str]:
        """Flush any remaining text in the buffer."""
        if len(self.buffer.strip()) >= self.MIN_SENTENCE_LENGTH:
            sentence = self.buffer.strip()
            self.buffer = ""
            return sentence
        self.buffer = ""
        return None


class ClaudeJSONParser:
    """Parse Claude's stream-json output format."""

    # System noise to filter out
    SKIP_PATTERNS = [
        'Co-Authored-By:',
        'Generated with',
        '<system-reminder>',
        '</system-reminder>',
        'you should consider whether it would be considered malware',
    ]

    # Content to clean from text
    CLEAN_PATTERNS = [
        (re.compile(r'```[\s\S]*?```'), ''),  # Code blocks
        (re.compile(r'`[^`]+`'), ''),          # Inline code
        (re.compile(r'\*\*([^*]+)\*\*'), r'\1'),  # Bold
        (re.compile(r'\*([^*]+)\*'), r'\1'),      # Italic
        (re.compile(r'https?://\S+'), 'URL'),     # URLs
        (re.compile(r'/[\w\-./]+'), ''),          # File paths
    ]

    def __init__(self):
        self.text_buffers = {}  # Track streaming text blocks by index

    def should_skip(self, text: str) -> bool:
        """Check if text should be skipped."""
        for pattern in self.SKIP_PATTERNS:
            if pattern in text:
                return True
        return False

    def clean_text(self, text: str) -> str:
        """Clean text for speech - remove code, markdown, etc."""
        for pattern, replacement in self.CLEAN_PATTERNS:
            text = pattern.sub(replacement, text)

        # Normalize whitespace
        text = ' '.join(text.split())
        return text.strip()

    def extract_text(self, msg: dict) -> Generator[str, None, None]:
        """Extract speakable text from a message."""
        msg_type = msg.get('type')

        # Handle streaming text deltas
        if msg_type == 'content_block_delta':
            delta = msg.get('delta', {})
            if delta.get('type') == 'text_delta':
                text = delta.get('text', '')
                if text and not self.should_skip(text):
                    yield text
            return

        # Handle complete messages
        content = None
        if 'message' in msg:
            content = msg['message'].get('content', [])
        elif 'content' in msg:
            content = msg.get('content', [])

        if not content:
            return

        # Handle string content
        if isinstance(content, str):
            if not self.should_skip(content):
                yield content
            return

        # Handle content blocks array
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get('type') == 'text':
                    text = block.get('text', '')
                    if text and not self.should_skip(text):
                        yield text


class TTSDaemon:
    """Interface to the stream-tts-cpp TTS daemon."""

    def __init__(self, binary_path: str, lang: str = 'en', translate: bool = True):
        self.binary_path = binary_path
        self.lang = lang
        self.translate = translate
        self.total_latency = 0
        self.sentence_count = 0

    def speak(self, text: str) -> float:
        """
        Speak text using one-shot mode.
        Returns latency in milliseconds.
        """
        start = time.perf_counter()

        cmd = [self.binary_path, '--speak', text, '--lang', self.lang]
        if self.translate:
            cmd.append('--translate')

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            latency_ms = (time.perf_counter() - start) * 1000
            self.total_latency += latency_ms
            self.sentence_count += 1

            if result.returncode != 0:
                print(f"{RED}[TTS Error]{RESET} {result.stderr[:100]}", file=sys.stderr)

            return latency_ms

        except subprocess.TimeoutExpired:
            print(f"{RED}[TTS Timeout]{RESET} Sentence took too long", file=sys.stderr)
            return 30000  # 30s timeout
        except Exception as e:
            print(f"{RED}[TTS Error]{RESET} {e}", file=sys.stderr)
            return 0

    def get_stats(self) -> dict:
        """Get performance statistics."""
        avg_latency = self.total_latency / self.sentence_count if self.sentence_count > 0 else 0
        return {
            'sentences': self.sentence_count,
            'total_latency_ms': self.total_latency,
            'avg_latency_ms': avg_latency
        }


def timestamp() -> str:
    """Get current timestamp for logging."""
    return datetime.now().strftime('%H:%M:%S.%f')[:-3]


def main():
    parser = argparse.ArgumentParser(
        description='Stream Claude Code output to translated TTS'
    )
    parser.add_argument(
        '--lang', '-l',
        default='ja',
        choices=['ja', 'zh', 'hi', 'ko', 'es', 'fr', 'de', 'en'],
        help='Target language for translation (default: ja)'
    )
    parser.add_argument(
        '--no-translate', '-n',
        action='store_true',
        help='Skip translation, speak in English'
    )
    parser.add_argument(
        '--binary', '-b',
        default='stream-tts-cpp/build/stream-tts-cpp',
        help='Path to TTS binary'
    )
    parser.add_argument(
        '--show-text', '-t',
        action='store_true',
        help='Print text to stderr as it is spoken'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show latency and stats'
    )

    args = parser.parse_args()

    # Find TTS binary
    binary_path = args.binary
    if not os.path.isabs(binary_path):
        # Try relative to script, then cwd
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        binary_path = os.path.join(script_dir, args.binary)
        if not os.path.exists(binary_path):
            binary_path = args.binary

    if not os.path.exists(binary_path):
        print(f"{RED}Error: TTS binary not found at {binary_path}{RESET}", file=sys.stderr)
        print(f"Build it with: cd stream-tts-cpp && ./build.sh", file=sys.stderr)
        sys.exit(1)

    # Initialize components
    json_parser = ClaudeJSONParser()
    sentence_buffer = SentenceBuffer()
    tts = TTSDaemon(binary_path, args.lang, not args.no_translate)

    print(f"{BOLD}{CYAN}[Voice]{RESET} Claude to Voice started", file=sys.stderr)
    print(f"{DIM}  Language: {args.lang}", file=sys.stderr)
    print(f"  Translation: {'disabled' if args.no_translate else 'enabled'}", file=sys.stderr)
    print(f"  Binary: {binary_path}{RESET}", file=sys.stderr)

    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Extract and process text
            for raw_text in json_parser.extract_text(msg):
                # Clean for speech
                clean = json_parser.clean_text(raw_text)
                if not clean:
                    continue

                # Buffer and yield sentences
                for sentence in sentence_buffer.add_text(clean + ' '):
                    if args.show_text:
                        print(f"{DIM}[{timestamp()}]{RESET} {BLUE}{sentence[:60]}...{RESET}", file=sys.stderr)

                    # Speak the sentence
                    latency = tts.speak(sentence)

                    if args.verbose:
                        print(f"{DIM}  Latency: {latency:.0f}ms{RESET}", file=sys.stderr)

        # Flush remaining buffer
        remaining = sentence_buffer.flush()
        if remaining:
            if args.show_text:
                print(f"{DIM}[{timestamp()}]{RESET} {BLUE}{remaining[:60]}...{RESET}", file=sys.stderr)
            tts.speak(remaining)

        # Print stats
        stats = tts.get_stats()
        if stats['sentences'] > 0:
            print(f"\n{BOLD}{GREEN}[Voice Complete]{RESET}", file=sys.stderr)
            print(f"{DIM}  Sentences: {stats['sentences']}", file=sys.stderr)
            print(f"  Avg Latency: {stats['avg_latency_ms']:.0f}ms", file=sys.stderr)
            print(f"  Total Time: {stats['total_latency_ms']/1000:.1f}s{RESET}", file=sys.stderr)

    except KeyboardInterrupt:
        print(f"\n{YELLOW}[Voice Interrupted]{RESET}", file=sys.stderr)
        sys.exit(0)
    except BrokenPipeError:
        sys.exit(0)


if __name__ == '__main__':
    main()
