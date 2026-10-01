"""
Unified Preprocessing Pipeline for Vietnamese NLP & Kaggle Submissions.
Orchestrates hidden character removal, homoglyph normalization,
Unicode normalization, teencode standardization, and word segmentation.
"""

from typing import List, Optional, Union
from tqdm import tqdm

from .unicode_normalizer import VietnameseUnicodeNormalizer
from .hidden_char_extractor import HiddenCharExtractor
from .teencode_normalizer import TeencodeNormalizer
from .vietnamese_segmenter import VietnameseSegmenter


class TextPreprocessingPipeline:
    """
    Modular, chainable pipeline for cleaning and normalizing Vietnamese text
    for transformer models (mDeBERTa, PhoBERT).
    """

    DEFAULT_STEPS = [
        "clean_hidden",
        "normalize_homoglyphs",
        "normalize_unicode",
        "normalize_teencode",
    ]

    def __init__(
        self,
        steps: Optional[List[str]] = None,
        segment_words: bool = False,
        segment_engine: str = "auto",
        typing_style: str = "new",
        keep_emojis: bool = True,
    ):
        """
        Args:
            steps: List of step names to run in order.
                   Available steps:
                   - 'clean_hidden': strips zero-width & invisible controls
                   - 'normalize_homoglyphs': maps lookalike Cyrillic to Latin
                   - 'normalize_unicode': fixes legacy encodings & standardizes NFC
                   - 'normalize_teencode': expands chat abbreviations & slang
                   - 'remove_diacritics': converts accented Vietnamese to unaccented ASCII
                   - 'restore_diacritics': restores accents for unaccented Vietnamese
                   - 'auto_restore_diacritics': restores accents only if text is unaccented
                   - 'segment': performs word segmentation (with underscores) for PhoBERT
            segment_words: If True and 'segment' is not in steps, appends 'segment' at the end.
                           (Crucial: Set True for PhoBERT, False for mDeBERTa).
            segment_engine: 'auto', 'pyvi', 'underthesea', or 'fallback'.
            typing_style: 'new' (hòa) or 'old' (hoà).
            keep_emojis: Whether to retain emoji characters.
        """
        self.steps = list(steps or self.DEFAULT_STEPS)
        if segment_words and "segment" not in self.steps:
            self.steps.append("segment")

        self.segment_engine = segment_engine
        self.typing_style = typing_style
        self.keep_emojis = keep_emojis

    def transform(self, text: str) -> str:
        """
        Executes all configured preprocessing steps sequentially on a single string.
        """
        if not text or not isinstance(text, str):
            return ""

        current_text = text

        for step in self.steps:
            if step == "clean_hidden":
                current_text = HiddenCharExtractor.strip_hidden_chars(current_text)
            elif step == "normalize_homoglyphs":
                current_text = HiddenCharExtractor.normalize_homoglyphs(current_text)
            elif step == "normalize_unicode":
                current_text = VietnameseUnicodeNormalizer.normalize_vietnamese_text(
                    current_text, typing_style=self.typing_style
                )
            elif step == "normalize_teencode":
                current_text = TeencodeNormalizer.normalize(current_text)
                if not self.keep_emojis:
                    current_text = TeencodeNormalizer.clean_punctuation(
                        current_text, keep_emojis=False
                    )
            elif step == "remove_diacritics":
                current_text = VietnameseUnicodeNormalizer.remove_diacritics(current_text)
            elif step == "restore_diacritics":
                current_text = VietnameseUnicodeNormalizer.restore_diacritics(current_text)
            elif step == "auto_restore_diacritics":
                if VietnameseUnicodeNormalizer.is_unaccented(current_text):
                    current_text = VietnameseUnicodeNormalizer.restore_diacritics(current_text)
            elif step == "segment":
                current_text = VietnameseSegmenter.segment(
                    current_text, engine=self.segment_engine
                )

        return current_text.strip()

    def transform_batch(
        self, texts: List[str], show_progress: bool = True, desc: str = "Preprocessing"
    ) -> List[str]:
        """
        Transforms a batch or list of texts with an optional tqdm progress indicator.
        """
        iterator = tqdm(texts, desc=desc) if show_progress else texts
        return [self.transform(t) for t in iterator]

    def __call__(self, text_or_texts: Union[str, List[str]]) -> Union[str, List[str]]:
        """Callable shorthand for transform / transform_batch."""
        if isinstance(text_or_texts, str):
            return self.transform(text_or_texts)
        return self.transform_batch(text_or_texts)
