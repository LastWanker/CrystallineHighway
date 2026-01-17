"""背诵调度与文本拆分。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .segmentation import ChineseSegmenter
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


class RecitationPlanner:
    """将文本拆分成短语/短句/长句/段落/全文单元。

    说明：
    - 拆分时保留标点，确保固化后的文本“好看”；
    - 寻找与匹配使用 normalized_text，确保标点不影响检索逻辑。
    """

    def __init__(self, segmenter: ChineseSegmenter | None = None) -> None:
        self.segmenter = segmenter or ChineseSegmenter()

    def build_plan(self, text: str) -> List[RecitationUnit]:
        """先拆分、再倒序输出用于背诵的单元序列。"""

        # 1) 段落：仅按换行拆分，保留原标点。
        paragraphs = self.segmenter.split_paragraphs(text)
        # 2) 长句：只按句号/问号/感叹号等句末标点拆分。
        long_sentences = self.segmenter.split_long_sentences(
            [para.display_text for para in paragraphs]
        )
        # 3) 短句：只按逗号/分号拆分，保持“短句只是标点拆分”的约定。
        short_sentences = self.segmenter.split_short_sentences(
            [sentence.display_text for sentence in long_sentences]
        )

        plan: List[RecitationUnit] = []
        # 4) 短语：由分词器输出的词条序列（比最小词素更大）。
        for clause in short_sentences:
            for phrase in self.segmenter.segment_words(clause.display_text):
                plan.append(
                    RecitationUnit(
                        display_text=phrase,
                        normalized_text=normalize_text(phrase),
                        label="phrase",
                    )
                )
        for sentence in short_sentences:
            plan.append(
                RecitationUnit(
                    display_text=sentence.display_text,
                    normalized_text=sentence.normalized_text,
                    label="short_sentence",
                )
            )
        for sentence in long_sentences:
            plan.append(
                RecitationUnit(
                    display_text=sentence.display_text,
                    normalized_text=sentence.normalized_text,
                    label="long_sentence",
                )
            )
        for para in paragraphs:
            plan.append(
                RecitationUnit(
                    display_text=para.display_text,
                    normalized_text=para.normalized_text,
                    label="paragraph",
                )
            )
        # 5) 段落与全文：直接作为宏观背诵单元，用于贴标签与检索配额。
        if text.strip():
            plan.append(
                RecitationUnit(
                    display_text=text.strip(),
                    normalized_text=normalize_text(text.strip()),
                    label="full_text",
                )
            )
        return plan
