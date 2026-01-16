"""背诵调度与文本拆分。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class RecitationUnit:
    """背诵单元。"""

    text: str
    label: str


class RecitationPlanner:
    """将文本拆分成短句/长句/段落等单元。"""

    sentence_split = re.compile(r"[。！？!?]+")
    clause_split = re.compile(r"[，,;；]+")

    def build_plan(self, text: str) -> List[RecitationUnit]:
        """先拆分、再倒序输出用于背诵的单元序列。"""

        paragraphs = [segment.strip() for segment in text.split("\n") if segment.strip()]
        long_sentences = self._split_sentences(paragraphs)
        short_sentences = self._split_clauses(long_sentences)
        phrases = self._split_phrases(short_sentences)

        plan: List[RecitationUnit] = []
        for phrase in phrases:
            plan.append(RecitationUnit(text=phrase, label="phrase"))
        for sentence in short_sentences:
            plan.append(RecitationUnit(text=sentence, label="short_sentence"))
        for sentence in long_sentences:
            plan.append(RecitationUnit(text=sentence, label="long_sentence"))
        for para in paragraphs:
            plan.append(RecitationUnit(text=para, label="paragraph"))
        if text.strip():
            plan.append(RecitationUnit(text=text.strip(), label="full_text"))
        return plan

    def _split_sentences(self, paragraphs: Iterable[str]) -> List[str]:
        sentences: List[str] = []
        for para in paragraphs:
            parts = [part.strip() for part in self.sentence_split.split(para) if part.strip()]
            sentences.extend(parts)
        return sentences

    def _split_clauses(self, sentences: Iterable[str]) -> List[str]:
        clauses: List[str] = []
        for sentence in sentences:
            parts = [part.strip() for part in self.clause_split.split(sentence) if part.strip()]
            clauses.extend(parts)
        return clauses

    def _split_phrases(self, clauses: Iterable[str]) -> List[str]:
        phrases: List[str] = []
        for clause in clauses:
            words = clause.split()
            if words:
                phrases.extend(words)
            else:
                phrases.append(clause)
        return phrases
