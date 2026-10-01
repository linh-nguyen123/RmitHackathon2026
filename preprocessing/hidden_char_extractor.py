"""
Hidden Character & Homoglyph Extractor.
Detects and purges invisible Unicode control characters, zero-width characters,
and normalizes adversarial homoglyphs (lookalikes) to standard Latin characters.
"""

import re
import unicodedata
from typing import List, Dict, Any, Tuple


class HiddenCharExtractor:
    """
    Extracts, inspects, and strips invisible zero-width characters and
    normalizes homoglyph substitutions used in adversarial evasion attacks.
    """

    # Comprehensive list of zero-width, invisible format, and bi-di override characters
    ZERO_WIDTH_CHARS: Dict[str, str] = {
        "\u200B": "Zero Width Space (ZWSP)",
        "\u200C": "Zero Width Non-Joiner (ZWNJ)",
        "\u200D": "Zero Width Joiner (ZWJ)",
        "\uFEFF": "Zero Width No-Break Space / Byte Order Mark (BOM)",
        "\u00AD": "Soft Hyphen (SHY)",
        "\u200E": "Left-to-Right Mark (LRM)",
        "\u200F": "Right-to-Left Mark (RLM)",
        "\u202A": "Left-to-Right Embedding (LRE)",
        "\u202B": "Right-to-Left Embedding (RLE)",
        "\u202C": "Pop Directional Formatting (PDF)",
        "\u202D": "Left-to-Right Override (LRO)",
        "\u202E": "Right-to-Left Override (RLO)",
        "\u2060": "Word Joiner (WJ)",
        "\u2061": "Function Application",
        "\u2062": "Invisible Times",
        "\u2063": "Invisible Separator",
        "\u2064": "Invisible Plus",
        "\u180E": "Mongolian Vowel Separator",
    }

    # Common Cyrillic, Greek, and Fullwidth lookalikes mapped to Latin ASCII
    HOMOGLYPH_MAP: Dict[str, str] = {
        # Cyrillic lowercase to Latin
        "а": "a", "с": "c", "е": "e", "о": "o", "р": "p", "ѕ": "s", "і": "i",
        "ј": "j", "х": "x", "у": "y", "в": "b", "п": "n", "т": "t", "г": "r",
        "к": "k", "м": "m", "н": "h",
        # Cyrillic uppercase to Latin
        "А": "A", "В": "B", "С": "C", "Е": "E", "Н": "H", "І": "I", "Ј": "J",
        "К": "K", "М": "M", "О": "O", "Р": "P", "Ѕ": "S", "Т": "T", "Х": "X",
        "Ү": "Y",
        # Greek lowercase/uppercase to Latin
        "α": "a", "β": "b", "ε": "e", "ι": "i", "κ": "k", "ν": "v", "ο": "o",
        "ρ": "p", "τ": "t", "υ": "u", "χ": "x",
        "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K",
        "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
        # Fullwidth ASCII (U+FF01 to U+FF5E)
        "ａ": "a", "ｂ": "b", "ｃ": "c", "ｄ": "d", "ｅ": "e", "ｆ": "f",
        "ｇ": "g", "ｈ": "h", "ｉ": "i", "ｊ": "j", "ｋ": "k", "ｌ": "l",
        "ｍ": "m", "ｎ": "n", "ｏ": "o", "ｐ": "p", "ｑ": "q", "ｒ": "r",
        "ｓ": "s", "ｔ": "t", "ｕ": "u", "ｖ": "v", "ｗ": "w", "ｘ": "x",
        "ｙ": "y", "ｚ": "z",
        "Ａ": "A", "Ｂ": "B", "Ｃ": "C", "Ｄ": "D", "Ｅ": "E", "Ｆ": "F",
        "Ｇ": "G", "Ｈ": "H", "Ｉ": "I", "Ｊ": "J", "Ｋ": "K", "Ｌ": "L",
        "Ｍ": "M", "Ｎ": "N", "Ｏ": "O", "Ｐ": "P", "Ｑ": "Q", "Ｒ": "R",
        "Ｓ": "S", "Ｔ": "T", "Ｕ": "U", "Ｖ": "V", "Ｗ": "W", "Ｘ": "X",
        "Ｙ": "Y", "Ｚ": "Z",
    }

    # Regex for fast detection and removal
    _ZERO_WIDTH_PATTERN = re.compile(
        "[" + "".join(re.escape(ch) for ch in ZERO_WIDTH_CHARS.keys()) + "]"
    )

    @classmethod
    def extract_hidden_chars(cls, text: str) -> List[Dict[str, Any]]:
        """
        Inspects text and returns detailed records of hidden/zero-width characters found.
        """
        records: List[Dict[str, Any]] = []
        if not text:
            return records

        for idx, char in enumerate(text):
            if char in cls.ZERO_WIDTH_CHARS:
                records.append({
                    "index": idx,
                    "character": char,
                    "hex_code": f"U+{ord(char):04X}",
                    "name": cls.ZERO_WIDTH_CHARS[char],
                })
        return records

    @classmethod
    def has_hidden_chars(cls, text: str) -> bool:
        """Checks if text contains any zero-width or hidden characters."""
        if not text:
            return False
        return bool(cls._ZERO_WIDTH_PATTERN.search(text))

    @classmethod
    def strip_hidden_chars(cls, text: str) -> str:
        """Removes all zero-width, invisible format, and directional override characters."""
        if not text:
            return ""
        return cls._ZERO_WIDTH_PATTERN.sub("", text)

    @classmethod
    def normalize_homoglyphs(cls, text: str) -> str:
        """
        Replaces visually indistinguishable Cyrillic/Greek/Fullwidth homoglyphs
        with standard Latin characters.
        """
        if not text:
            return ""

        result = []
        for ch in text:
            result.append(cls.HOMOGLYPH_MAP.get(ch, ch))
        return "".join(result)

    @classmethod
    def clean(cls, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Performs complete de-obfuscation:
        1. Logs hidden character occurrences.
        2. Strips zero-width characters.
        3. Normalizes homoglyphs to Latin equivalents.
        """
        if not text:
            return "", []

        hidden_records = cls.extract_hidden_chars(text)
        cleaned_text = cls.strip_hidden_chars(text)
        cleaned_text = cls.normalize_homoglyphs(cleaned_text)

        return cleaned_text, hidden_records
