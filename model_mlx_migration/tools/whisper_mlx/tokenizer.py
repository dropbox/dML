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
Tokenizer wrapper for WhisperMLX.

Reuses the tokenizer from mlx-whisper for compatibility.
Provides utilities for working with timestamps.
"""


# Try to import from mlx_whisper, fall back to tiktoken
try:
    from mlx_whisper.tokenizer import Tokenizer, get_tokenizer
    MLX_WHISPER_AVAILABLE = True
except ImportError:
    MLX_WHISPER_AVAILABLE = False
    Tokenizer = None
    get_tokenizer = None


# Whisper special tokens
LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
    "yue": "cantonese",
}


def get_whisper_tokenizer(
    multilingual: bool = True,
    num_languages: int = 99,
    language: str | None = None,
    task: str = "transcribe",
):
    """
    Get a Whisper tokenizer.

    Args:
        multilingual: Whether model is multilingual
        num_languages: Number of languages (for token offset calculation)
        language: Target language code (e.g., "en")
        task: Task type ("transcribe" or "translate")

    Returns:
        Tokenizer instance

    Raises:
        ImportError: If mlx_whisper is not installed
    """
    if not MLX_WHISPER_AVAILABLE:
        raise ImportError(
            "mlx-whisper is required for tokenization. "
            "Install with: pip install mlx-whisper",
        )

    return get_tokenizer(
        multilingual=multilingual,
        num_languages=num_languages,
        language=language,
        task=task,
    )


class TimestampDecoder:
    """
    Decode timestamp tokens to time values.

    Handles conversion between:
    - Timestamp token IDs
    - Encoder positions
    - Actual time in seconds

    Works with dynamic precision for variable-length audio.
    """

    def __init__(
        self,
        timestamp_begin: int = 50364,  # Standard Whisper timestamp_begin
        precision: float = 0.02,  # Default 30s / 1500 positions
    ):
        """
        Args:
            timestamp_begin: First timestamp token ID
            precision: Time per encoder position in seconds
        """
        self.timestamp_begin = timestamp_begin
        self.precision = precision

    def set_precision(self, precision: float):
        """Update precision for variable-length audio."""
        self.precision = precision

    def token_to_time(self, token_id: int) -> float:
        """
        Convert timestamp token to time in seconds.

        Args:
            token_id: Timestamp token ID

        Returns:
            Time in seconds
        """
        position = token_id - self.timestamp_begin
        return position * self.precision

    def time_to_token(self, time_seconds: float) -> int:
        """
        Convert time to timestamp token ID.

        Args:
            time_seconds: Time in seconds

        Returns:
            Timestamp token ID
        """
        position = int(time_seconds / self.precision)
        return self.timestamp_begin + position

    def extract_timestamps(
        self,
        tokens: list[int],
    ) -> list[tuple[float, float, list[int]]]:
        """
        Extract timestamp segments from token sequence.

        Args:
            tokens: List of token IDs (without SOT sequence)

        Returns:
            List of (start_time, end_time, text_tokens) tuples
        """
        segments = []
        current_tokens = []
        start_time = None

        for token in tokens:
            if token >= self.timestamp_begin:
                # Timestamp token
                time = self.token_to_time(token)
                if start_time is None:
                    start_time = time
                else:
                    # End of segment
                    segments.append((start_time, time, current_tokens))
                    current_tokens = []
                    start_time = time
            else:
                # Text token
                current_tokens.append(token)

        # Handle trailing tokens without end timestamp
        if current_tokens and start_time is not None:
            # Use audio duration as end if available
            segments.append((start_time, start_time, current_tokens))

        return segments
