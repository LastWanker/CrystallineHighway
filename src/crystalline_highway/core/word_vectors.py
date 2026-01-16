"""外部词向量提供器。"""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path
from typing import Dict, List
from gensim.models import KeyedVectors

class WordVectorProvider:
    """词向量加载与查询。

    说明：
    - 支持文本格式与 word2vec 二进制格式（.bin）。
    - 维度会根据文件头自动识别，不匹配时以检测结果为准。
    - 大型向量库建议提前构建索引或使用内存映射工具。
    """

    def __init__(self, dim: int, path: str | None, lazy: bool = True) -> None:
        self.dim = dim
        self.path = Path(path) if path else None
        self.lazy = lazy
        self.cache: Dict[str, List[float]] = {}
        self._binary_model = None

        if self.path and self.path.exists():
            detected_dim = self._detect_dim()
            if detected_dim and detected_dim != self.dim:
                self.dim = detected_dim

    def get_vector(self, word: str) -> List[float]:
        """获取词向量，若不可用则返回全 0 向量。"""

        if word in self.cache:
            return self.cache[word]
        if self.path and self.path.exists():
            if self._is_binary():
                vec = self._get_from_binary(word)
            else:
                if not self.lazy:
                    self._load_text_cache()
                vec = self._scan_file(word) if self.lazy else self.cache.get(word)
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
                for line_number, line in enumerate(handle):
                    parts = line.strip().split()
                    if not parts:
                        continue
                    if line_number == 0 and self._is_header_line(parts):
                        continue
                    if parts[0] == word and len(parts) == self.dim + 1:
                        return [float(value) for value in parts[1:]]
        except OSError:
            return None
        return None

    def _load_text_cache(self) -> None:
        if not self.path or self.cache:
            return
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle):
                    parts = line.strip().split()
                    if not parts:
                        continue
                    if line_number == 0 and self._is_header_line(parts):
                        continue
                    if len(parts) != self.dim + 1:
                        continue
                    self.cache[parts[0]] = [float(value) for value in parts[1:]]
        except OSError:
            return

    def _get_from_binary(self, word: str) -> List[float] | None:
        model = self._ensure_binary_model()
        if model is None:
            return None
        if word in model:
            return model[word].tolist()
        return None

    def _ensure_binary_model(self):
        if self._binary_model is not None:
            return self._binary_model
        if not self.path:
            return None
        if importlib.util.find_spec("gensim") is None:
            warnings.warn(
                "缺少 gensim，无法加载二进制词向量文件；请安装 gensim 后重试。",
                RuntimeWarning,
            )
            return None
        from gensim.models import KeyedVectors

        self._binary_model = KeyedVectors.load_word2vec_format(str(self.path), binary=True)
        if self._binary_model.vector_size != self.dim:
            self.dim = self._binary_model.vector_size
        return self._binary_model

    def _detect_dim(self) -> int | None:
        if not self.path:
            return None
        if self._is_binary():
            return self._detect_dim_binary()
        return self._detect_dim_text()

    def _detect_dim_text(self) -> int | None:
        if not self.path:
            return None
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle):
                    parts = line.strip().split()
                    if not parts:
                        continue
                    if line_number == 0 and self._is_header_line(parts):
                        continue
                    if len(parts) > 1:
                        return len(parts) - 1
        except OSError:
            return None
        return None

    def _detect_dim_binary(self) -> int | None:
        if not self.path:
            return None
        try:
            with self.path.open("rb") as handle:
                header = handle.readline()
            header_text = header.decode("utf-8", errors="ignore").strip()
            parts = header_text.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1])
        except OSError:
            return None
        return None

    def _is_binary(self) -> bool:
        return self.path is not None and self.path.suffix.lower() == ".bin"

    @staticmethod
    def _is_header_line(parts: List[str]) -> bool:
        return len(parts) == 2 and all(part.isdigit() for part in parts)

    def _load_text_cache(self) -> None:
        if not self.path or self.cache:
            return
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle):
                    parts = line.strip().split()
                    if not parts:
                        continue
                    if line_number == 0 and self._is_header_line(parts):
                        continue
                    if len(parts) != self.dim + 1:
                        continue
                    self.cache[parts[0]] = [float(value) for value in parts[1:]]
        except OSError:
            return

    def _get_from_binary(self, word: str) -> List[float] | None:
        model = self._ensure_binary_model()
        if model is None:
            return None
        if word in model:
            return model[word].tolist()
        return None

    def _ensure_binary_model(self):
        if self._binary_model is not None:
            return self._binary_model
        if not self.path:
            return None
        if importlib.util.find_spec("gensim") is None:
            warnings.warn(
                "缺少 gensim，无法加载二进制词向量文件；请安装 gensim 后重试。",
                RuntimeWarning,
            )
            return None
        from gensim.models import KeyedVectors

        self._binary_model = KeyedVectors.load_word2vec_format(str(self.path), binary=True)
        if self._binary_model.vector_size != self.dim:
            self.dim = self._binary_model.vector_size
        return self._binary_model

    def _detect_dim(self) -> int | None:
        if not self.path:
            return None
        if self._is_binary():
            return self._detect_dim_binary()
        return self._detect_dim_text()

    def _detect_dim_text(self) -> int | None:
        if not self.path:
            return None
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle):
                    parts = line.strip().split()
                    if not parts:
                        continue
                    if line_number == 0 and self._is_header_line(parts):
                        continue
                    if len(parts) > 1:
                        return len(parts) - 1
        except OSError:
            return None
        return None

    def _detect_dim_binary(self) -> int | None:
        if not self.path:
            return None
        try:
            with self.path.open("rb") as handle:
                header = handle.readline()
            header_text = header.decode("utf-8", errors="ignore").strip()
            parts = header_text.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1])
        except OSError:
            return None
        return None

    def _is_binary(self) -> bool:
        return self.path is not None and self.path.suffix.lower() == ".bin"

    @staticmethod
    def _is_header_line(parts: List[str]) -> bool:
        return len(parts) == 2 and all(part.isdigit() for part in parts)
