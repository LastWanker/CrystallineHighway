"""外部词向量提供器（占位实现）。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List


class WordVectorProvider:
    """词向量加载与查询。

    说明：
    - 默认假设使用腾讯中文词向量（200 维）。
    - 大型向量库应提前构建索引，这里只做最小可用占位。
    """

    def __init__(self, dim: int, path: str | None, lazy: bool = True) -> None:
        self.dim = dim
        self.path = Path(path) if path else None
        self.lazy = lazy
        self.cache: Dict[str, List[float]] = {}

    def get_vector(self, word: str) -> List[float]:
        """获取词向量，若不可用则返回全 0 向量。"""

        if word in self.cache:
            return self.cache[word]
        if self.path and self.path.exists() and self.lazy:
            vec = self._scan_file(word)
            if vec is not None:
                self.cache[word] = vec
                return vec
        return [0.0 for _ in range(self.dim)]

    def _scan_file(self, word: str) -> List[float] | None:
        """扫描文件查找词向量（简化实现）。"""

        if not self.path:
            return None
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    if parts[0] == word and len(parts) == self.dim + 1:
                        return [float(value) for value in parts[1:]]
        except OSError:
            return None
        return None
