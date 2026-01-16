"""词典与实例注册表管理。"""

from __future__ import annotations

import itertools
from typing import List

from src.crystalline_highway.config import MemoryConfig
from src.crystalline_highway.core import vector
from src.crystalline_highway.core.word_vectors import WordVectorProvider
from src.crystalline_highway.models.instance import InstanceNode
from src.crystalline_highway.models.meta import MetaEntry
from src.crystalline_highway.storage.in_memory import InMemoryStore


class Registry:
    """统一管理元与实例。"""

    def __init__(self, store: InMemoryStore, config: MemoryConfig) -> None:
        self.store = store
        self.config = config
        self._meta_counter = itertools.count(1)
        self._node_counter = itertools.count(1)
        self.vector_provider = WordVectorProvider(
            dim=config.vector_dim,
            path=config.tencent_vector_path,
            lazy=config.word_vector_lazy,
        )
        if self.vector_provider.dim != self.config.vector_dim:
            self.config.vector_dim = self.vector_provider.dim

    def ensure_meta(self, text: str, global_freq: float = 1.0) -> MetaEntry:
        """获取或创建元条目。"""

        if text in self.store.meta_table:
            meta = self.store.meta_table[text]
            meta.private_freq += 1.0
            return meta
        meta_id = f"meta-{next(self._meta_counter)}"
        entry = MetaEntry(
            meta_id=meta_id,
            text=text,
            global_freq=global_freq,
            private_freq=1.0,
            category_vector=self.vector_provider.get_vector(text),
        )
        self.store.meta_table[text] = entry
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
