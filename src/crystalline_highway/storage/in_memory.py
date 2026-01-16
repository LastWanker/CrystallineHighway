"""内存级存储实现。"""

from __future__ import annotations

from typing import Dict

from ..models.graph import Graph
from ..models.instance import InstanceNode
from ..models.meta import MetaEntry


class InMemoryStore:
    """简单内存存储：词典、实例册、图三套结构。"""

    def __init__(self) -> None:
        # meta_table 的 key 是“规范化文本”，符合“寻找时忽略标点”的规则。
        self.meta_table: Dict[str, MetaEntry] = {}
        self.instance_table: Dict[str, InstanceNode] = {}
        self.graph = Graph()
