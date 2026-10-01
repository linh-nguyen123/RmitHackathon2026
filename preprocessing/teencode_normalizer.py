"""
Teencode & Slang Normalizer for Vietnamese.
Standardizes Vietnamese chat abbreviations, elongated words (vowel stretching),
and informal social media text into formal Vietnamese tokens.
"""

import re
from typing import Dict, Optional


class TeencodeNormalizer:
    """
    Translates Vietnamese teencode, slang, and chat shortcuts into standard Vietnamese.
    Also handles elongated expressive characters (e.g., 'ngonnnn' -> 'ngon').
    """

    TEENCODE_DICT: Dict[str, str] = {
        # Negation & Question words
        "ko": "không", "k": "không", "hok": "không", "hổng": "không", "kh": "không",
        "hong": "không", "hem": "không", "hẻm": "không", "khum": "không", "khom": "không",
        "k0": "không", "kh0ng": "không", "h0k": "không",
        "j": "gì", "gi": "gì", "chi": "gì", "jz": "gì vậy", "jday": "gì đây",
        "sao": "sao", "s": "sao", "ntn": "như thế nào", "sao z": "sao vậy",
        # Pronouns
        "mk": "mình", "mik": "mình", "minh": "mình", "m": "mày", "tao": "tao",
        "t": "tao", "tui": "tôi", "tớ": "tớ", "c": "chị", "a": "anh", "e": "em",
        "ngta": "người ta", "ng": "người", "mn": "mọi người", "b": "bạn", "bn": "bạn",
        "b@n": "bạn", "th@n": "thân", "b4n": "bạn",
        "m1nh": "mình", "m1k": "mình",
        "mng": "mọi người", "ae": "anh em", "cr": "crush", "gđ": "gia đình",
        # Common verbs & adjectives
        "dc": "được", "đc": "được", "dk": "được", "đk": "được",
        "vs": "với", "zới": "với", "cung": "cũng", "cug": "cũng",
        "wa": "quá", "wá": "quá", "qua": "quá",
        "r": "rồi", "roi": "rồi", "rùi": "rồi", "oy": "rồi", "roài": "rồi",
        "thik": "thích", "thjx": "thích", "thjk": "thích", "iu": "yêu",
        "bit": "biết", "bik": "biết", "bít": "biết", "b1t": "biết", "hỉu": "hiểu", "hiu": "hiểu",
        "thui": "thôi", "thoai": "thôi", "thoy": "thôi",
        "z": "vậy", "v": "vậy", "vay": "vậy", "zậy": "vậy", "zo": "vào", "vô": "vào",
        "cx": "cũng", "lun": "luôn", "luon": "luôn",
        "h": "giờ", "hqa": "hôm qua", "hnay": "hôm nay", "mngay": "mỗi ngày",
        "bt": "biết", "bth": "bình thường", "bthg": "bình thường",
        "chx": "chưa", "ch": "chưa", "chuwa": "chưa",
        "oke": "đồng ý", "ok": "đồng ý", "okie": "đồng ý", "oki": "đồng ý",
        "tks": "cảm ơn", "thx": "cảm ơn", "ty": "cảm ơn", "cmon": "cảm ơn", "camon": "cảm ơn",
        "pls": "làm ơn", "plz": "làm ơn",
        "acc": "tài khoản", "inbox": "nhắn tin", "ib": "nhắn tin", "rep": "trả lời",
        "cmt": "bình luận", "stt": "trạng thái", "avt": "ảnh đại diện",
        "chug": "chung", "nhug": "nhưng", "nhg": "nhưng",
        "ms": "mới", "bh": "bây giờ", "bg": "bây giờ", "lquan": "liên quan",
        # Leetspeak / slang words
        "ch0i": "chơi", "choi": "chơi", "ch0": "cho", "c0": "có",
        "đj": "đi", "dj": "đi", "d1": "đi", "đ1": "đi",
        "l4m": "làm", "đ@ng": "đang",
        # Internet slang & expressive words
        "vcl": "rất nhiều", "vl": "rất nhiều", "vler": "rất nhiều",
        "haizz": "thở dài", "chẹp": "thở dài", "ahihi": "cười", "huhu": "khóc",
        "gato": "ghen tị", "pro": "chuyên nghiệp", "noob": "nghiệp dư",
        "ngáo": "ngớ ngẩn", "toang": "hỏng", "xu cà na": "xui xẻo",
    }

    # Regex for collapsing elongated characters: ngonnnnn -> ngon
    # Match any character repeated 3 or more times
    _ELONGATED_PATTERN = re.compile(r"([a-zA-ZÀ-ỹ])\1{2,}", re.IGNORECASE)

    # Regex for repeated punctuation: ?????? -> ?, !!!!!! -> !
    _PUNCTUATION_PATTERN = re.compile(r"([?!.,~])\1{2,}")

    @classmethod
    def normalize_elongated_chars(cls, text: str) -> str:
        """
        Collapses expressive repeated characters to a single or double instance.
        E.g., 'ngonnnnn' -> 'ngon', 'quaaaaa' -> 'quá', 'đẹpppp' -> 'đẹp'.
        """
        if not text:
            return ""

        # Replace 3+ consecutive identical characters with 1 instance
        # For Vietnamese, double characters are sometimes meaningful (e.g. 'oo', 'ee' in English loanwords)
        # but 3+ are always expressive elongation.
        return cls._ELONGATED_PATTERN.sub(r"\1", text)

    @classmethod
    def normalize_symbol_leetspeak(cls, text: str) -> str:
        """
        De-obfuscates symbol and number replacements inside words (Leetspeak):
        - '@' within/adjacent to words -> 'a' (e.g. 'b@n' -> 'ban', 'th@n' -> 'than')
        - '0' adjacent to letters -> 'o' (e.g. 'ch0i' -> 'choi', 'k0' -> 'ko', 'ch0' -> 'cho')
        - '1' adjacent to letters -> 'i' (e.g. 'm1nh' -> 'minh', 'd1' -> 'di')
        - '3' adjacent to letters -> 'e' (e.g. 'm3' -> 'me')
        - '4' adjacent to letters -> 'a' (e.g. 'b4n' -> 'ban')
        - '7' adjacent to letters -> 't' (e.g. '7oi' -> 'toi')
        - 'đj' / 'dj' -> 'đi'
        Preserves valid email addresses and standalone numbers (e.g. 2026, 100).
        """
        if not text:
            return ""

        # 1. Protect email addresses
        emails = []
        def stash_email(m):
            emails.append(m.group(0))
            return f"__EMAIL_TOKEN_{len(emails) - 1}__"

        text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", stash_email, text)

        # 1.5. Direct mapping for high-frequency Vietnamese @ teencode words
        at_words = {
            "b@n": "bạn", "th@n": "thân", "đ@ng": "đang", "d@ng": "đang",
            "l@m": "làm", "c@i": "cái", "n@y": "này", "v@i": "vãi", "qu@": "quá"
        }
        for k, v in at_words.items():
            text = re.sub(rf"(?i)\b{re.escape(k)}\b", v, text)

        # 2. @ as 'a' when inside or adjacent to letters
        text = re.sub(r"(?<=[a-zA-ZÀ-ỹ])@(?=[a-zA-ZÀ-ỹ])", "a", text)
        text = re.sub(r"\b@(?=[a-zA-ZÀ-ỹ])", "a", text)
        text = re.sub(r"(?<=[a-zA-ZÀ-ỹ])@\b", "a", text)

        # 3. 0 as 'o' when adjacent to letters (preserves 2026, 100, etc.)
        text = re.sub(r"(?<=[a-zA-ZÀ-ỹ])0(?=[a-zA-ZÀ-ỹ])", "o", text)
        text = re.sub(r"(?<=[a-zA-ZÀ-ỹ])0\b", "o", text)
        text = re.sub(r"\b0(?=[a-zA-ZÀ-ỹ])", "o", text)

        # 4. 1 as 'i' when adjacent to letters
        text = re.sub(r"(?<=[a-zA-ZÀ-ỹ])1", "i", text)
        text = re.sub(r"1(?=[a-zA-ZÀ-ỹ])", "i", text)

        # 5. 3 as 'e', 4 as 'a', 7 as 't' when adjacent to letters
        text = re.sub(r"(?<=[a-zA-ZÀ-ỹ])3", "e", text)
        text = re.sub(r"3(?=[a-zA-ZÀ-ỹ])", "e", text)
        text = re.sub(r"(?<=[a-zA-ZÀ-ỹ])4", "a", text)
        text = re.sub(r"4(?=[a-zA-ZÀ-ỹ])", "a", text)
        text = re.sub(r"(?<=[a-zA-ZÀ-ỹ])7", "t", text)
        text = re.sub(r"7(?=[a-zA-ZÀ-ỹ])", "t", text)

        # 6. 'j' used as 'i' (e.g., 'đj' -> 'đi', 'dj' -> 'đi', 'chj' -> 'chị', 'laj' -> 'lại')
        text = re.sub(r"\b[dđ]j\b", "đi", text, flags=re.IGNORECASE)
        text = re.sub(r"(?<=[b-df-tv-zđB-DF-TV-ZĐ])j\b", "i", text)

        # 7. Restore emails
        for idx, email in enumerate(emails):
            text = text.replace(f"__EMAIL_TOKEN_{idx}__", email)

        return text

    @classmethod
    def normalize_teencode(cls, text: str) -> str:
        """
        Replaces teencode tokens with formal equivalents using regex word boundaries.
        Case-insensitive matching that preserves surrounding punctuation.
        """
        if not text:
            return ""

        tokens = text.split()
        normalized_tokens = []

        for token in tokens:
            # Separate leading/trailing punctuation, allowing '@' in word tokens
            match = re.match(r"^([^\wÀ-ỹ@]*)((?:[\wÀ-ỹ]|@)+)([^\wÀ-ỹ@]*)$", token, re.UNICODE)
            if match:
                prefix, word, suffix = match.groups()
                lower_word = word.lower()
                if lower_word in cls.TEENCODE_DICT:
                    replacement = cls.TEENCODE_DICT[lower_word]
                    # Preserve uppercase if the original was all caps
                    if word.isupper() and len(word) > 1:
                        replacement = replacement.upper()
                    elif word[0].isupper():
                        replacement = replacement.capitalize()
                    normalized_tokens.append(f"{prefix}{replacement}{suffix}")
                else:
                    normalized_tokens.append(token)
            else:
                normalized_tokens.append(token)

        return " ".join(normalized_tokens)

    @classmethod
    def clean_punctuation(cls, text: str, keep_emojis: bool = True) -> str:
        """
        Standardizes punctuation repetitions (e.g., '????' -> '?').
        Optionally removes or retains emojis.
        """
        if not text:
            return ""

        # Collapse repeated punctuation
        text = cls._PUNCTUATION_PATTERN.sub(r"\1", text)

        # Standardize multiple spaces
        text = re.sub(r"\s+", " ", text).strip()

        if not keep_emojis:
            # Emoji pattern range
            emoji_pattern = re.compile(
                "["
                "\U0001F600-\U0001F64F"  # emoticons
                "\U0001F300-\U0001F5FF"  # symbols & pictographs
                "\U0001F680-\U0001F6FF"  # transport & map symbols
                "\U0001F1E0-\U0001F1FF"  # flags (iOS)
                "\U00002702-\U000027B0"
                "\U000024C2-\U0001F251"
                "]+",
                flags=re.UNICODE,
            )
            text = emoji_pattern.sub(r"", text)

        return text

    @classmethod
    def normalize(cls, text: str) -> str:
        """
        Runs complete teencode normalization pipeline:
        1. Collapse elongated characters (ngonnnnn -> ngon)
        2. De-obfuscate leetspeak and embedded symbols (b@n -> ban, ch0i -> choi)
        3. Normalize teencode dictionary terms (mik -> mình, choi -> chơi, vs -> với)
        4. Clean up repetitive punctuation
        """
        if not text:
            return ""
        text = cls.normalize_elongated_chars(text)
        text = cls.normalize_symbol_leetspeak(text)
        text = cls.normalize_teencode(text)
        text = cls.clean_punctuation(text)
        return text
