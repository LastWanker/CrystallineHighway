"""中文分词与层级拆分。"""

from __future__ import annotations

import importlib.util
import re
from dataclasses import dataclass
from typing import Iterable, List

from .text_utils import normalize_text


def _split_with_delimiters(text: str, pattern: re.Pattern[str]) -> List[str]:
    """按正则切分文本并保留标点作为尾巴。"""

    results: List[str] = []
    last_index = 0
    for match in pattern.finditer(text):
        end_index = match.end()
        segment = text[last_index:end_index].strip()
        if segment:
            results.append(segment)
        last_index = end_index
    tail = text[last_index:].strip()
    if tail:
        results.append(tail)
    return results


@dataclass
class SegmentedUnit:
    """文本拆分后的单元，保留展示文本与规范化版本。"""

    display_text: str
    normalized_text: str


class ChineseSegmenter:
    """中文分词与层级拆分入口。

    后端优先顺序：jieba（轻量成熟）→ LTP（更智能但依赖重）→ simple（内置退化方案）。
    """

    sentence_split = re.compile(r"[。！？!?]+")
    clause_split = re.compile(r"[，,;；]+")

    def __init__(self, backend: str | None = None) -> None:
        self.backend = backend or self._detect_backend()
        self._ltp_model = None

    def segment_morphemes(self, text: str) -> List[str]:
        """最小词素切分，用于提前注册。"""

        if not text:
            return []
        if self.backend == "jieba":
            import jieba

            # jieba 的 search 模式更细碎，适合作为“最小词素”近似。
            return [token for token in jieba.cut_for_search(text) if normalize_text(token)]
        if self.backend == "ltp":
            tokens = self._ltp_segment(text)
            return [token for token in tokens if normalize_text(token)]
        return self._simple_morpheme_segment(text)

    def segment_words(self, text: str) -> List[str]:
        """常规词语切分，用于短语级背诵。"""

        if not text:
            return []
        if self.backend == "jieba":
            import jieba

            # 常规模式输出更稳定的词组，是“短语”级输入的默认选择。
            return [token for token in jieba.cut(text, cut_all=False) if normalize_text(token)]
        if self.backend == "ltp":
            tokens = self._ltp_segment(text)
            return [token for token in tokens if normalize_text(token)]
        return self._simple_word_segment(text)

    def split_paragraphs(self, text: str) -> List[SegmentedUnit]:
        """按段落拆分，保留原始标点。"""

        paragraphs = [segment.strip() for segment in text.split("\n") if segment.strip()]
        return [
            SegmentedUnit(display_text=paragraph, normalized_text=normalize_text(paragraph))
            for paragraph in paragraphs
        ]

    def split_long_sentences(self, paragraphs: Iterable[str]) -> List[SegmentedUnit]:
        """按长句拆分（主要按句号/问号等）。"""

        sentences: List[SegmentedUnit] = []
        for para in paragraphs:
            parts = _split_with_delimiters(para, self.sentence_split)
            for part in parts:
                normalized = normalize_text(part)
                if normalized:
                    sentences.append(SegmentedUnit(display_text=part, normalized_text=normalized))
        return sentences

    def split_short_sentences(self, long_sentences: Iterable[str]) -> List[SegmentedUnit]:
        """按短句拆分（主要按逗号/分号）。"""

        clauses: List[SegmentedUnit] = []
        for sentence in long_sentences:
            parts = _split_with_delimiters(sentence, self.clause_split)
            for part in parts:
                normalized = normalize_text(part)
                if normalized:
                    clauses.append(SegmentedUnit(display_text=part, normalized_text=normalized))
        return clauses

    def _ltp_segment(self, text: str) -> List[str]:
        if self._ltp_model is None:
            from ltp import LTP

            self._ltp_model = LTP()
        seg, _ = self._ltp_model.seg([text])
        return seg[0]

    def _simple_word_segment(self, text: str) -> List[str]:
        tokens: List[str] = []
        buffer = ""
        for char in text:
            if "\u4e00" <= char <= "\u9fff":
                if buffer:
                    tokens.append(buffer)
                    buffer = ""
                tokens.append(char)
            elif char.isalnum():
                buffer += char
            else:
                if buffer:
                    tokens.append(buffer)
                    buffer = ""
        if buffer:
            tokens.append(buffer)
        return [token for token in tokens if normalize_text(token)]

    def _simple_char_segment(self, text: str) -> List[str]:
        return [char for char in text if normalize_text(char)]

    def _simple_morpheme_segment(self, text: str) -> List[str]:
        tokens: List[str] = []
        chinese_buffer = ""
        alnum_buffer = ""
        for char in text:
            if "\u4e00" <= char <= "\u9fff":
                if alnum_buffer:
                    tokens.append(alnum_buffer)
                    alnum_buffer = ""
                chinese_buffer += char
                continue
            if char.isalnum():
                if chinese_buffer:
                    tokens.extend(self._split_chinese_morphemes(chinese_buffer))
                    chinese_buffer = ""
                alnum_buffer += char
                continue
            if chinese_buffer:
                tokens.extend(self._split_chinese_morphemes(chinese_buffer))
                chinese_buffer = ""
            if alnum_buffer:
                tokens.append(alnum_buffer)
                alnum_buffer = ""
        if chinese_buffer:
            tokens.extend(self._split_chinese_morphemes(chinese_buffer))
        if alnum_buffer:
            tokens.append(alnum_buffer)
        return [token for token in tokens if normalize_text(token)]

    def _split_chinese_morphemes(self, text: str) -> List[str]:
        if len(text) <= 1:
            return [text] if text else []
        chunks = [text[index : index + 2] for index in range(0, len(text), 2)]
        if len(chunks) >= 2 and len(chunks[-1]) == 1:
            chunks[-2] += chunks[-1]
            chunks.pop()
        return chunks

    def _detect_backend(self) -> str:
        if importlib.util.find_spec("jieba") is not None:
            return "jieba"
        if importlib.util.find_spec("ltp") is not None:
            return "ltp"
        return "simple"
