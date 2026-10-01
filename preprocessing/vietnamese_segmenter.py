"""
Vietnamese Word Segmenter with Offline Fallback.
Crucial for PhoBERT which requires segmented words joined by underscores ('_').
Includes an automatic fallback to pure-Python Maximum Matching algorithm
when external packages (pyvi / underthesea) are unavailable in offline Kaggle environments.
"""

import re
from typing import List, Set, Optional


class VietnameseSegmenter:
    """
    Vietnamese Word Segmenter supporting PyVi, Underthesea, and an internal
    Maximum Matching algorithm as a 100% offline pure-Python fallback.
    """

    # Internal core compound word lexicon for offline fallback (top frequent multi-word tokens)
    CORE_COMPOUNDS: Set[str] = {
        "học sinh", "sinh viên", "giáo viên", "giảng viên", "trường học", "đại học",
        "bản thân", "bạn thân", "gia đình", "bạn bè", "xã hội", "nhà nước", "chính phủ",
        "thông tin", "công nghệ", "mạng xã hội", "an toàn", "bảo mật", "an ninh mạng",
        "trí tuệ nhân tạo", "máy học", "dữ liệu", "hệ thống", "phát triển", "nghiên cứu",
        "thành phố", "đất nước", "quốc gia", "thế giới", "kinh tế", "văn hóa",
        "thời gian", "tương lai", "quá khứ", "hiện tại", "cuộc sống", "công việc",
        "sức khỏe", "tinh thần", "thể chất", "bệnh tật", "thuốc men", "bác sĩ",
        "pháp luật", "quy định", "chính sách", "quyền lợi", "nghĩa vụ", "trách nhiệm",
        "tấn công", "phòng thủ", "phát hiện", "ngăn chặn", "đối kháng", "bảo vệ",
        "người dùng", "khách hàng", "dịch vụ", "sản phẩm", "chất lượng", "hiệu quả",
        "cảm ơn", "xin lỗi", "chúc mừng", "tạm biệt", "xin chào", "đồng ý",
        "bình thường", "đặc biệt", "quan trọng", "cần thiết", "rõ ràng", "chính xác",
        "hôm nay", "ngày mai", "hôm qua", "tuần trước", "tháng sau", "năm nay",
        "tất cả", "mọi người", "ai đó", "cái gì", "tại sao", "như thế nào",
    }

    _pyvi_available: Optional[bool] = None
    _underthesea_available: Optional[bool] = None

    @classmethod
    def _check_backends(cls):
        """Checks availability of external segmentation packages."""
        if cls._pyvi_available is None:
            try:
                import pyvi  # noqa: F401
                cls._pyvi_available = True
            except ImportError:
                cls._pyvi_available = False

        if cls._underthesea_available is None:
            try:
                import underthesea  # noqa: F401
                cls._underthesea_available = True
            except ImportError:
                cls._underthesea_available = False

    @classmethod
    def segment_max_match(cls, text: str) -> str:
        """
        Pure-Python Forward Maximum Matching algorithm for offline fallback.
        Requires zero external C/C++ or pip dependencies.
        """
        if not text:
            return ""

        words = text.split()
        n = len(words)
        result = []
        i = 0

        # Maximum compound word length in syllables
        max_len = 4

        while i < n:
            matched = False
            for length in range(min(max_len, n - i), 1, -1):
                phrase = " ".join(words[i : i + length]).lower()
                # Clean punctuation for lookup
                clean_phrase = re.sub(r"[^\w\sÀ-ỹ]", "", phrase).strip()
                if clean_phrase in cls.CORE_COMPOUNDS:
                    # Join tokens with underscore
                    segmented_word = "_".join(words[i : i + length])
                    result.append(segmented_word)
                    i += length
                    matched = True
                    break

            if not matched:
                result.append(words[i])
                i += 1

        return " ".join(result)

    @classmethod
    def segment(cls, text: str, engine: str = "auto") -> str:
        """
        Segments Vietnamese text.
        Tokens of compound words are connected with underscores (e.g., 'học_sinh').

        Args:
            text: Input Vietnamese text.
            engine: 'auto', 'pyvi', 'underthesea', or 'fallback'.
        """
        if not text:
            return ""

        cls._check_backends()

        if engine == "pyvi" or (engine == "auto" and cls._pyvi_available):
            try:
                from pyvi import ViTokenizer
                return ViTokenizer.tokenize(text)
            except Exception:
                pass

        if engine == "underthesea" or (engine == "auto" and cls._underthesea_available):
            try:
                from underthesea import word_tokenize
                # underthesea returns tokens, format with '_'
                tokens = word_tokenize(text, format="text")
                return tokens
            except Exception:
                pass

        # Fallback to internal Max-Match algorithm
        return cls.segment_max_match(text)

    @classmethod
    def desegment(cls, text: str) -> str:
        """
        Reverts segmented text back to standard whitespace-separated words.
        E.g., 'học_sinh RMIT' -> 'học sinh RMIT'.
        """
        if not text:
            return ""
        return text.replace("_", " ")
