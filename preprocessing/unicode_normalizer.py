"""
Unicode Normalizer for Vietnamese Text.
Supports NFC/NFD normalization, modern/legacy tone mark positioning,
and legacy font encodings (TCVN3 / VNI / UTF-8 Mojibake).
"""

import unicodedata
import re
from typing import Dict, Optional


class VietnameseUnicodeNormalizer:
    """
    Normalizes Vietnamese Unicode text formats, diacritics placement,
    and legacy encoding artifacts.
    """

    # Vowels with diacritics table: base_vowel -> {tone: accented_char}
    # Tone index: 0=ngang (none), 1=huyen (\), 2=sac (/), 3=hoi (?), 4=nga (~), 5=nang (.)
    TONE_MAP: Dict[str, str] = {
        "a": "aàáảãạ",
        "ă": "ăằắẳẵặ",
        "â": "âầấẩẫậ",
        "e": "eèéẻẽẹ",
        "ê": "êềếểễệ",
        "i": "iìíỉĩị",
        "o": "oòóỏõọ",
        "ô": "ôồốổỗộ",
        "ơ": "ơờớởỡợ",
        "u": "uùúủũụ",
        "ư": "ưừứửữự",
        "y": "yỳýỷỹỵ",
        "A": "AÀÁẢÃẠ",
        "Ă": "ĂẰẮẲẴẶ",
        "Â": "ÂẦẤẨẪẬ",
        "E": "EÈÉẺẼẸ",
        "Ê": "ÊỀẾỂỄỆ",
        "I": "IÌÍỈĨỊ",
        "O": "OÒÓỎÕỌ",
        "Ô": "ÔỒỐỔỖỘ",
        "Ơ": "ƠỜỚỞỠỢ",
        "U": "UÙÚỦŨỤ",
        "Ư": "ƯỪỨỬỮỰ",
        "Y": "YỲÝỶỸỴ",
    }

    # Tone to index map
    CHAR_TO_TONE: Dict[str, int] = {}
    for base, chars in TONE_MAP.items():
        for tone_idx, char in enumerate(chars):
            CHAR_TO_TONE[char] = tone_idx

    # Old vs New tone placement rules
    # Old: hoà, hoè, thuỷ | New: hòa, hòe, thủy
    OLD_TO_NEW_TONE_PATTERNS = [
        (re.compile(r"(?i)\b([b-df-tv-zđ]*)o([aáàảãạ])([b-df-tv-zđ]*)\b"), r"\1o\2\3"),
        (re.compile(r"o([aàảãạ])", re.IGNORECASE), r"o\1"),
    ]

    # Common TCVN3 (ABC) to Unicode mapping
    TCVN3_MAP = {
        "µ": "à", "¸": "á", "¶": "ả", "·": "ã", "¹": "ạ",
        "¨": "ă", "»": "ằ", "¾": "ắ", "¼": "ẳ", "½": "ẵ", "Æ": "ặ",
        "©": "â", "Ç": "ầ", "Ê": "ấ", "È": "ẩ", "É": "ẫ", "Ë": "ậ",
        "®": "đ", "Đ": "Đ",
        "Ì": "è", "Ð": "é", "Î": "ẻ", "Ï": "ẽ", "Ñ": "ẹ",
        "ª": "ê", "Ò": "ề", "Õ": "ế", "Ó": "ể", "Ô": "ễ", "Ö": "ệ",
        "×": "ì", "Ý": "í", "Ø": "ỉ", "Ü": "ĩ", "Þ": "ị",
        "ß": "ò", "á": "ó", "à": "ỏ", "ã": "õ", "ä": "ọ",
        "«": "ô", "å": "ồ", "è": "ố", "æ": "ổ", "ç": "ỗ", "é": "ộ",
        "¬": "ơ", "ê": "ờ", "í": "ớ", "ë": "ở", "ì": "ỡ", "î": "ợ",
        "ï": "ù", "ó": "ú", "ñ": "ủ", "ò": "ũ", "ô": "ụ",
        "­": "ư", "õ": "ừ", "ø": "ứ", "ö": "ử", "÷": "ữ", "ù": "ự",
        "ú": "ỳ", "ý": "ý", "û": "ỷ", "ü": "ỹ", "þ": "ỵ"
    }

    # Common UTF-8 Mojibake patterns
    MOJIBAKE_MAP = {
        "Ã¡": "á", "Ã ": "à", "Ã£": "ã", "Ã¢": "â", "Ã©": "é", "Ã¨": "è",
        "Ãª": "ê", "Ã­": "í", "Ã¬": "ì", "Ã³": "ó", "Ã²": "ò", "Ã´": "ô",
        "Ãµ": "õ", "Ãº": "ú", "Ã¹": "ù", "Ã½": "ý", "Ä": "đ", "Ä": "Đ",
        "Æ°": "ư", "Æ¡": "ơ"
    }

    @classmethod
    def normalize_form(cls, text: str, form: str = "NFC") -> str:
        """
        Normalizes Unicode form to standard NFC (precomposed) or NFD (decomposed).
        Defaults to NFC, which is the required standard for Hugging Face tokenizers.
        """
        if not text:
            return ""
        return unicodedata.normalize(form, text)

    @classmethod
    def standardize_tone_marks(cls, text: str, typing_style: str = "new") -> str:
        """
        Standardizes tone marks in Vietnamese words.
        'new' (default): hòa, hòe, thủy (accent on secondary vowel in diphthongs).
        'old': hoà, hoè, thuỷ (accent on first vowel).
        """
        if not text:
            return ""

        text = cls.normalize_form(text, form="NFC")

        # Standard replacements for common diphthong tone misplacements
        old_to_new = {
            "oà": "òa", "oá": "óa", "oả": "ỏa", "oã": "õa", "oạ": "ọa",
            "oè": "òe", "oé": "óe", "oẻ": "ỏe", "oẽ": "õe", "oẹ": "ọe",
            "uỳ": "ùy", "uý": "úy", "uỷ": "ủy", "uỹ": "ũy", "uỵ": "ụy",
            "OÀ": "ÒA", "OÁ": "ÓA", "OẢ": "ỎA", "OÃ": "ÕA", "OẠ": "ỌA",
            "OÈ": "ÒE", "OÉ": "ÓE", "OẢ": "ỎE", "OẼ": "ÕE", "OẸ": "ỌE",
            "UỲ": "ÙY", "UÝ": "ÚY", "UỶ": "ỦY", "UỸ": "ŨY", "UỴ": "ỤY"
        }

        new_to_old = {v: k for k, v in old_to_new.items()}

        mapping = old_to_new if typing_style.lower() == "new" else new_to_old
        for src, tgt in mapping.items():
            text = text.replace(src, tgt)

        return text

    @classmethod
    def fix_legacy_encoding(cls, text: str) -> str:
        """
        Repairs corrupted encodings such as TCVN3 (ABC font) and common UTF-8 Mojibake.
        """
        if not text:
            return ""

        # Step 1: Repair UTF-8 Mojibake
        for bad_char, good_char in cls.MOJIBAKE_MAP.items():
            if bad_char in text:
                text = text.replace(bad_char, good_char)

        # Step 2: Repair TCVN3 if recognizable characters are found
        # Check if text contains high density of specific TCVN3 characters
        tcvn_markers = sum(1 for ch in text if ch in "µ¸¶·¹¨»¾¼½Æ©ÇÊÈÉË®ÌÐÎÏÑªÒÕÓÔÖ×ÝØÜÞßáàãä«åèæçé¬êíëìîïóñòô­õøö÷ùúýûüþ")
        if tcvn_markers > 2 and len(text) > 0 and (tcvn_markers / len(text)) > 0.15:
            res = []
            for ch in text:
                res.append(cls.TCVN3_MAP.get(ch, ch))
            text = "".join(res)

        return cls.normalize_form(text, form="NFC")

    @classmethod
    def normalize_vietnamese_text(cls, text: str, typing_style: str = "new") -> str:
        """
        Full pipeline for Unicode normalization:
        1. Fix legacy / mojibake
        2. Normalize Unicode form (NFC)
        3. Standardize diacritics placement
        """
        if not text:
            return ""
        text = cls.fix_legacy_encoding(text)
        text = cls.normalize_form(text, form="NFC")
        text = cls.standardize_tone_marks(text, typing_style=typing_style)
        return text

    # Specific mapping for Vietnamese characters that NFD does not strip (e.g. đ, Đ)
    _VIETNAMESE_ACCENT_TABLE = str.maketrans({
        "à": "a", "á": "a", "ả": "a", "ã": "a", "ạ": "a",
        "ă": "a", "ằ": "a", "ắ": "a", "ẳ": "a", "ẵ": "a", "ặ": "a",
        "â": "a", "ầ": "a", "ấ": "a", "ẩ": "a", "ẫ": "a", "ậ": "a",
        "đ": "d",
        "è": "e", "é": "e", "ẻ": "e", "ẽ": "e", "ẹ": "e",
        "ê": "e", "ề": "e", "ế": "e", "ể": "e", "ễ": "e", "ệ": "e",
        "ì": "i", "í": "i", "ỉ": "i", "ĩ": "i", "ị": "i",
        "ò": "o", "ó": "o", "ỏ": "o", "õ": "o", "ọ": "o",
        "ô": "o", "ồ": "o", "ố": "o", "ổ": "o", "ỗ": "o", "ộ": "o",
        "ơ": "o", "ờ": "o", "ớ": "o", "ở": "o", "ỡ": "o", "ợ": "o",
        "ù": "u", "ú": "u", "ủ": "u", "ũ": "u", "ụ": "u",
        "ư": "u", "ừ": "u", "ứ": "u", "ử": "u", "ữ": "u", "ự": "u",
        "ỳ": "y", "ý": "y", "ỷ": "y", "ỹ": "y", "ỵ": "y",
        "À": "A", "Á": "A", "Ả": "A", "Ã": "A", "Ạ": "A",
        "Ă": "A", "Ằ": "A", "Ắ": "A", "Ẳ": "A", "Ẵ": "A", "Ặ": "A",
        "Â": "A", "Ầ": "A", "Ấ": "A", "Ẩ": "A", "Ẫ": "A", "Ậ": "A",
        "Đ": "D",
        "È": "E", "É": "E", "Ẻ": "E", "Ẽ": "E", "Ẹ": "E",
        "Ê": "E", "Ề": "E", "Ế": "E", "Ể": "E", "Ễ": "E", "Ệ": "E",
        "Ì": "I", "Í": "I", "Ỉ": "I", "Ĩ": "I", "Ị": "I",
        "Ò": "O", "Ó": "O", "Ỏ": "O", "Õ": "O", "Ọ": "O",
        "Ô": "O", "Ồ": "O", "Ố": "O", "Ổ": "O", "Ỗ": "O", "Ộ": "O",
        "Ơ": "O", "Ờ": "O", "Ớ": "O", "Ở": "O", "Ỡ": "O", "Ợ": "O",
        "Ù": "U", "Ú": "U", "Ủ": "U", "Ũ": "U", "Ụ": "U",
        "Ư": "U", "Ừ": "U", "Ứ": "U", "Ử": "U", "Ữ": "U", "Ự": "U",
        "Ỳ": "Y", "Ý": "Y", "Ỷ": "Y", "Ỹ": "Y", "Ỵ": "Y",
    })

    # Lexicon for unaccented -> accented restoration (focusing on prompt safety, core tokens, and syntax)
    RESTORATION_LEXICON_PHRASES = {
        "bo qua": "bỏ qua",
        "tat ca": "tất cả",
        "huong dan": "hướng dẫn",
        "chi thi": "chỉ thị",
        "he thong": "hệ thống",
        "an toan": "an toàn",
        "bao mat": "bảo mật",
        "kiem tra": "kiểm tra",
        "quy tac": "quy tắc",
        "noi dung": "nội dung",
        "thong tin": "thông tin",
        "du lieu": "dữ liệu",
        "mat khau": "mật khẩu",
        "tai khoan": "tài khoản",
        "tan cong": "tấn công",
        "khai thac": "khai thác",
        "lo hong": "lỗ hổng",
        "hoc sinh": "học sinh",
        "sinh vien": "sinh viên",
        "dai hoc": "đại học",
        "nguoi dung": "người dùng",
        "khach hang": "khách hàng",
        "tro ly": "trợ lý",
        "lap trinh": "lập trình",
        "tri tue nhan tao": "trí tuệ nhân tạo",
        "may hoc": "máy học",
        "tu gio": "từ giờ",
        "tu bay gio": "từ bây giờ",
        "in ra": "in ra",
        "xuat ra": "xuất ra",
        "tra loi": "trả lời",
        "khong gioi han": "không giới hạn",
        "khong bi rang buoc": "không bị ràng buộc",
        "dong vai": "đóng vai",
        "chuyen gia": "chuyên gia",
        "nghien cuu": "nghiên cứu",
        "tieng viet": "tiếng việt",
        "nhu the nao": "như thế nào",
        "co the": "có thể",
        "lam the nao": "làm thế nào",
        "tu choi": "từ chối",
        "toan bo": "toàn bộ",
        "truy cap": "truy cập",
        "trai phep": "trái phép",
        "xam nhap": "xâm nhập",
        "ma doc": "mã độc",
        "may chu": "máy chủ",
    }

    RESTORATION_LEXICON_WORDS = {
        "toi": "tôi", "ban": "bạn", "minh": "mình", "nguoi": "người", "chung": "chúng",
        "khong": "không", "duoc": "được", "biet": "biết", "thich": "thích",
        "phai": "phải", "muon": "muốn", "can": "cần", "nen": "nên", "nay": "này",
        "do": "đó", "kia": "kia", "ay": "ấy", "trong": "trong", "ngoai": "ngoài",
        "tren": "trên", "duoi": "dưới", "truoc": "trước", "sau": "sau",
        "nhung": "nhưng", "vi": "vì", "neu": "nếu", "thi": "thì", "va": "và",
        "hoac": "hoặc", "voi": "với", "cho": "cho", "ve": "về", "tu": "từ",
        "den": "đến", "la": "là", "co": "có", "se": "sẽ", "da": "đã", "dang": "đang",
    }

    # Distinctive unaccented Vietnamese marker syllables (high probability of Vietnamese language)
    _VIET_UNACCENTED_MARKERS = {
        "khong", "duoc", "nguoi", "thong", "nghiep", "truoc", "chuyen", "chinh",
        "huong", "phuong", "quyen", "nghi", "nguyen", "nhieu", "nhung", "trong",
        "thanh", "thuc", "khoang", "thoi", "gian", "cuoc", "song"
    }

    @classmethod
    def remove_diacritics(cls, text: str) -> str:
        """
        Strips all Vietnamese accents and diacritics, returning clean ASCII text.
        Converts 'đ'/'Đ' to 'd'/'D' and removes combining diacritics.
        E.g., 'Học sinh RMIT' -> 'Hoc sinh RMIT'.
        """
        if not text:
            return ""

        # Normalize to NFC first, then translate mapped characters
        text = cls.normalize_form(text, form="NFC")
        text = text.translate(cls._VIETNAMESE_ACCENT_TABLE)

        # Decompose any remaining combining characters and filter
        nfd = unicodedata.normalize("NFD", text)
        cleaned = "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")
        return unicodedata.normalize("NFC", cleaned)

    @classmethod
    def is_unaccented(cls, text: str, threshold: float = 0.15) -> bool:
        """
        Heuristically determines whether the input string is unaccented Vietnamese text.
        Checks for the absence of Vietnamese accented vowels combined with the presence
        of characteristic Vietnamese phonetic syllables.
        """
        if not text or len(text.strip()) == 0:
            return False

        # Check if text contains accented Vietnamese characters (tone > 0 or special Vietnamese diacritics)
        has_accents = any(
            cls.CHAR_TO_TONE.get(ch, 0) > 0 or ch in "đĐăĂâÂêÊôÔơƠưƯ"
            for ch in text
        )
        if has_accents:
            return False

        # Tokenize words into lower-case alphanumeric tokens
        tokens = [w.strip().lower() for w in re.findall(r"\b[a-zA-Z]+\b", text)]
        if not tokens:
            return False

        # Build comprehensive vocabulary of unaccented words (from markers, words, and phrases)
        viet_vocab = set(cls._VIET_UNACCENTED_MARKERS) | set(cls.RESTORATION_LEXICON_WORDS.keys())
        for phrase in cls.RESTORATION_LEXICON_PHRASES.keys():
            for p_word in phrase.split():
                viet_vocab.add(p_word)

        # Count match with characteristic Vietnamese syllable markers
        marker_count = sum(1 for t in tokens if t in viet_vocab)
        ratio = marker_count / len(tokens)

        return ratio >= threshold

    @classmethod
    def restore_diacritics(cls, text: str) -> str:
        """
        Restores Vietnamese accents for unaccented text using offline multi-word and
        unigram phrase matching.
        E.g., 'bo qua tat ca huong dan' -> 'bỏ qua tất cả hướng dẫn'.
        """
        if not text:
            return ""

        result = text

        # Step 1: Replace longer multi-word phrases first (e.g. 'tri tue nhan tao' -> 'trí tuệ nhân tạo')
        # Sort by descending length so longest phrases match first
        sorted_phrases = sorted(cls.RESTORATION_LEXICON_PHRASES.keys(), key=lambda x: len(x), reverse=True)
        for phrase in sorted_phrases:
            accented_phrase = cls.RESTORATION_LEXICON_PHRASES[phrase]
            pattern = re.compile(rf"\b{re.escape(phrase)}\b", re.IGNORECASE)

            def make_replacement(match, target=accented_phrase):
                matched_str = match.group(0)
                if matched_str.isupper():
                    return target.upper()
                elif matched_str[0].isupper():
                    return target.capitalize()
                return target

            result = pattern.sub(make_replacement, result)

        # Step 2: Replace individual frequent function words
        words = result.split()
        restored_words = []
        for word in words:
            # Preserve punctuation around word
            match = re.match(r"^([^\w]*)([\w]+)([^\w]*)$", word, re.UNICODE)
            if match:
                prefix, w, suffix = match.groups()
                w_lower = w.lower()
                if w_lower in cls.RESTORATION_LEXICON_WORDS:
                    replacement = cls.RESTORATION_LEXICON_WORDS[w_lower]
                    if w.isupper() and len(w) > 1:
                        replacement = replacement.upper()
                    elif w[0].isupper():
                        replacement = replacement.capitalize()
                    restored_words.append(f"{prefix}{replacement}{suffix}")
                else:
                    restored_words.append(word)
            else:
                restored_words.append(word)

        return " ".join(restored_words)
