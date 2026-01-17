"""词典与实例注册表管理。"""

from __future__ import annotations

import itertools
from typing import List

from ..config import MemoryConfig
from . import vector
from .text_utils import normalize_text
from .word_vectors import WordVectorProvider
from ..models.instance import InstanceNode
from ..models.meta import MetaEntry
from ..storage.in_memory import InMemoryStore


class Registry:
    """统一管理元与实例。"""

    def __init__(self, store: InMemoryStore, config: MemoryConfig) -> None:
        self.store = store
        self.config = config
        self._meta_counter = itertools.count(1)
        self._node_counter = itertools.count(1)
        # 词向量用于“范畴场/星图”，这里对接外部词向量库（腾讯中文词向量）。
        # 这符合“寻找驱动建构”里“先有粗糙范畴力场”的设定。
        self.vector_provider = WordVectorProvider(
            dim=config.vector_dim,
            path=config.tencent_vector_path,
            lazy=config.word_vector_lazy,
            index_path=config.tencent_vector_index_path,
            auto_build_index=config.word_vector_auto_index,
        )
        if self.vector_provider.dim != self.config.vector_dim:
            self.config.vector_dim = self.vector_provider.dim

    def ensure_meta(
        self,
        text: str,
        global_freq: float = 1.0,
        normalized_text: str | None = None,
    ) -> MetaEntry:
        """获取或创建元条目。

        关键逻辑说明：
        - 元是“寻找的目标”，其身份不是数据库 ID，而是“规范化后的文本”。
        - 寻找时忽略标点，因此词典键使用 normalized_text；
        - 固化时保留标点，因此显示文本仍用原 text。
        """

        normalized_key = normalized_text or normalize_text(text)
        if normalized_key in self.store.meta_table:
            meta = self.store.meta_table[normalized_key]
            meta.private_freq += 1.0
            # 如果新的显示文本更完整（例如包含标点），则更新展示文本，
            # 以满足“固化时保留标点”的可读性需求。
            if text and len(text) >= len(meta.text):
                meta.text = text
            return meta
        meta_id = f"meta-{next(self._meta_counter)}"
        entry = MetaEntry(
            meta_id=meta_id,
            text=text,
            normalized_text=normalized_key,
            global_freq=global_freq,
            private_freq=1.0,
            category_vector=self.vector_provider.get_vector(text),
        )
        self.store.meta_table[normalized_key] = entry
        return entry

    def create_instance(
        self,
        meta: MetaEntry,
        base_pos: List[float],
        bias_vector: List[float],
        jitter_scale: float,
    ) -> InstanceNode:
        """创建新的实例节点，并加入实例册。"""

        node_id = f"node-{next(self._node_counter)}"
        pos = vector.add(base_pos, bias_vector)
        pos = vector.add(pos, vector.jitter(self.config.vector_dim, jitter_scale))
        node = InstanceNode(node_id=node_id, meta_id=meta.meta_id, vector_pos=pos)
        self.store.instance_table[node_id] = node
        meta.instances.add(node_id)
        return node

    def get_instance(self, node_id: str) -> InstanceNode:
        return self.store.instance_table[node_id]
