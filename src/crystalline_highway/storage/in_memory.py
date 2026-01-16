"""内存级存储实现。"""

from __future__ import annotations

from typing import Dict

from crystalline_highway.models.graph import Graph
from crystalline_highway.models.instance import InstanceNode
from crystalline_highway.models.meta import MetaEntry


class InMemoryStore:
    """简单内存存储：词典、实例册、图三套结构。"""

    def __init__(self) -> None:
        self.meta_table: Dict[str, MetaEntry] = {}
        self.instance_table: Dict[str, InstanceNode] = {}
        self.graph = Graph()
