"""外部词向量提供器与索引加载逻辑。"""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path
from typing import Dict, Iterable, List

from src.crystalline_highway.core.segmentation import ChineseSegmenter


class WordVectorProvider:
    """词向量加载、索引与向量聚合。

    设计要点（对应“指导文件/完整想法.docx”与“对一些问题的回答.txt”）：
    - 词向量扮演“范畴场/星图”的来源，只提供方向性与弱牵引，并不承担精确语义推理。
    - 默认支持腾讯词向量的 word2vec 二进制格式（.bin），并提供“索引化”缓存以加速加载。
    - 允许按需聚合词向量，形成短语/句子/段落的向量表示，用于粗略范畴定位。
    """

    def __init__(
            self,
            dim: int,
            path: str | None,
            lazy: bool = True,
            index_path: str | None = None,
            auto_build_index: bool = True,
    ) -> None:
        self.dim = dim
        self.path = Path(path) if path else None
        self.lazy = lazy
        self.index_path = Path(index_path) if index_path else None
        self.auto_build_index = auto_build_index
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

    def get_tokens_vector(self, tokens: Iterable[str]) -> List[float]:
        """聚合一组词向量，得到短语/句子/段落向量。

        说明：
        - 采用“平均池化”作为最稳健的默认方法，避免单一高频词拉扯过猛。
        - 若全部词都无向量，则返回全 0 向量（与系统“找不到就新建”的逻辑一致）。
        """

        vectors = [self.get_vector(token) for token in tokens if token]
        if not vectors:
            return [0.0 for _ in range(self.dim)]
        sums = [0.0 for _ in range(self.dim)]
        count = 0
        for vec in vectors:
            if not vec:
                continue
            count += 1
            for index, value in enumerate(vec):
                sums[index] += value
        if count == 0:
            return [0.0 for _ in range(self.dim)]
        return [value / count for value in sums]

    def get_text_vector(self, text: str, segmenter: ChineseSegmenter | None = None) -> List[float]:
        """直接计算整段文本的向量（适用于句子/段落）。"""

        if not text:
            return [0.0 for _ in range(self.dim)]
        segmenter = segmenter or ChineseSegmenter()
        tokens = segmenter.segment_words(text)
        return self.get_tokens_vector(tokens)

    def build_binary_index(self) -> Path | None:
        """为二进制词向量构建索引缓存（gensim KeyedVectors）。

        这是目前最成熟、最通用的做法：
        - 首次加载 .bin 并保存为 .kv；
        - 后续用 mmap 只读方式加载，大幅降低启动成本。
        """

        if not self.path or not self._is_binary():
            return None
        if importlib.util.find_spec("gensim") is None:
            warnings.warn(
                "缺少 gensim，无法构建词向量索引；请安装 gensim 后重试。",
                RuntimeWarning,
            )
            return None
        from gensim.models import KeyedVectors

        target_path = self.index_path or self.path.with_suffix(".kv")
        model = KeyedVectors.load_word2vec_format(str(self.path), binary=True)
        model.save(str(target_path))
        self.dim = model.vector_size
        return target_path

    def _scan_file(self, word: str) -> List[float] | None:
        """扫描文本格式文件查找词向量（简化实现）。"""

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

        index_path = self.index_path or (self.path.with_suffix(".kv") if self.path else None)
        if index_path and index_path.exists():
            self._binary_model = KeyedVectors.load(str(index_path), mmap="r")
        else:
            if self.auto_build_index:
                self.build_binary_index()
                if index_path and index_path.exists():
                    self._binary_model = KeyedVectors.load(str(index_path), mmap="r")
            if self._binary_model is None:
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
