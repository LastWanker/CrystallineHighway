"""背诵调度与文本拆分。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .segmentation import ChineseSegmenter, SegmentedUnit
from .text_utils import normalize_text


@dataclass
class RecitationUnit:
    """背诵单元。

    display_text: 用于展示与固化输出（保留标点，保证可读性）。
    normalized_text: 用于检索与词典键（去标点，满足“寻找时忽略标点”）。
    """

    display_text: str
    normalized_text: str
    label: str
    tokens: List[str]


class RecitationPlanner:
    """将文本拆分成短语/短句/长句/段落/全文单元。

    说明：
    - 拆分时保留标点，确保固化后的文本“好看”；
    - 寻找与匹配使用 normalized_text，确保标点不影响检索逻辑。
    """

    def __init__(self, segmenter: ChineseSegmenter | None = None) -> None:
        self.segmenter = segmenter or ChineseSegmenter()

    def build_plan(self, text: str) -> List[RecitationUnit]:
        """先拆分、再倒序输出用于背诵的单元序列。

        约定：所有更小的单元必须来自上一层的切分结果，不再另行拆分。
        """

        # 1) 段落：仅按换行拆分，保留原标点。
        paragraphs = self.segmenter.split_paragraphs(text)
        # 2) 长句：只按句号/问号/感叹号等句末标点拆分。
        long_sentences: List[SegmentedUnit] = []
        paragraph_to_longs: List[List[SegmentedUnit]] = []
        for para in paragraphs:
            longs = self.segmenter.split_long_sentences([para.display_text])
            paragraph_to_longs.append(longs)
            long_sentences.extend(longs)
        # 3) 短句：只按逗号/分号拆分，保持“短句只是标点拆分”的约定。
        short_sentences: List[SegmentedUnit] = []
        long_to_shorts: List[List[SegmentedUnit]] = []
        for sentence in long_sentences:
            shorts = self.segmenter.split_short_sentences([sentence.display_text])
            long_to_shorts.append(shorts)
            short_sentences.extend(shorts)

        plan: List[RecitationUnit] = []
        # 4) 短语：由分词器输出的词条序列（比最小词素更大）。
        phrases_by_clause: List[List[str]] = []
        all_phrases: List[str] = []
        phrase_morphemes: List[List[str]] = []
        for clause in short_sentences:
            phrases = self.segmenter.segment_words(clause.display_text)
            phrases_by_clause.append(phrases)
            all_phrases.extend(phrases)
            for phrase in phrases:
                phrase_morphemes.append(self.segmenter.segment_morphemes(phrase))

        # 5) 最小词素：仅从短语拆出，确保来源一致。
        morphemes: List[str] = [token for group in phrase_morphemes for token in group]

        for morpheme in morphemes:
            plan.append(
                RecitationUnit(
                    display_text=morpheme,
                    normalized_text=normalize_text(morpheme),
                    label="morpheme",
                    tokens=[],
                )
            )

        for phrase, morpheme_tokens in zip(all_phrases, phrase_morphemes):
            plan.append(
                RecitationUnit(
                    display_text=phrase,
                    normalized_text=normalize_text(phrase),
                    label="短语",
                    tokens=morpheme_tokens,
                )
            )

        for clause, phrases in zip(short_sentences, phrases_by_clause):
            plan.append(
                RecitationUnit(
                    display_text=clause.display_text,
                    normalized_text=clause.normalized_text,
                    label="short_sentence",
                    tokens=phrases,
                )
            )
        for sentence, clauses in zip(long_sentences, long_to_shorts):
            clause_texts = [clause.display_text for clause in clauses]
            plan.append(
                RecitationUnit(
                    display_text=sentence.display_text,
                    normalized_text=sentence.normalized_text,
                    label="long_sentence",
                    tokens=clause_texts,
                )
            )
        for para, sentences in zip(paragraphs, paragraph_to_longs):
            sentence_texts = [sentence.display_text for sentence in sentences]
            plan.append(
                RecitationUnit(
                    display_text=para.display_text,
                    normalized_text=para.normalized_text,
                    label="paragraph",
                    tokens=sentence_texts,
                )
            )
        # 5) 段落与全文：直接作为宏观背诵单元，用于贴标签与检索配额。
        if text.strip():
            paragraph_texts = [para.display_text for para in paragraphs]
            plan.append(
                RecitationUnit(
                    display_text=text.strip(),
                    normalized_text=normalize_text(text.strip()),
                    label="full_text",
                    tokens=paragraph_texts or [text.strip()],
                )
            )
        return plan
