"""全局词频入口，基于 wordfreq。"""

from __future__ import annotations

import importlib.util
import statistics
import warnings
from typing import List


class GlobalFrequencyProvider:
    """提供全局词频与典型频率估计。"""

    def __init__(
        self,
        language: str = "zh",
        sample_size: int = 2000,
        fallback_frequency: float = 1e-5,
    ) -> None:
        self.language = language
        self.sample_size = sample_size
        self.fallback_frequency = fallback_frequency
        self._available = importlib.util.find_spec("wordfreq") is not None

    def word_frequency(self, word: str) -> float:
        if not self._available:
            warnings.warn(
                "缺少 wordfreq，使用兜底频率值。",
                RuntimeWarning,
            )
            return self.fallback_frequency
        from wordfreq import word_frequency

        return float(word_frequency(word, self.language))

    def top_words(self) -> List[str]:
        if not self._available:
            return []
        from wordfreq import top_n_list

        return list(top_n_list(self.language, n=self.sample_size))

    def typical_frequency(self) -> float:
        words = self.top_words()
        if not words:
            return self.fallback_frequency
        frequencies = [self.word_frequency(word) for word in words]
        if not frequencies:
            return self.fallback_frequency
        return float(statistics.median(frequencies))
