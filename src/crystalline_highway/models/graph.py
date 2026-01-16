"""图结构与边信息。"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple


class EdgeType(str, Enum):
    """路径类型。

    horizontal: 背诵/构筑生成的横向路径
    vertical: 固化生成的纵向索引路径
    """

    horizontal = "horizontal"
    vertical = "vertical"


@dataclass
class EdgeData:
    """边的数据。"""

    edge_type: EdgeType
    walk_count: int = 0


class Graph:
    """图结构：只管实例之间的路。"""

    def __init__(self) -> None:
        self.out_edges: Dict[str, Dict[str, EdgeData]] = {}
        self.in_edges: Dict[str, Dict[str, EdgeData]] = {}

    def _ensure_node(self, node_id: str) -> None:
        self.out_edges.setdefault(node_id, {})
        self.in_edges.setdefault(node_id, {})

    def add_edge(self, src_id: str, dst_id: str, edge_type: EdgeType) -> None:
        """新增或强化一条边，并增加走过次数。"""

        self._ensure_node(src_id)
        self._ensure_node(dst_id)
        edge = self.out_edges[src_id].get(dst_id)
        if edge is None:
            edge = EdgeData(edge_type=edge_type, walk_count=0)
            self.out_edges[src_id][dst_id] = edge
            self.in_edges[dst_id][src_id] = edge
        edge.walk_count += 1

    def get_edge(self, src_id: str, dst_id: str) -> EdgeData | None:
        return self.out_edges.get(src_id, {}).get(dst_id)

    def neighbors(self, node_id: str) -> Dict[str, EdgeData]:
        return self.out_edges.get(node_id, {})

    def downgrade_edge(self, src_id: str, dst_id: str, decrement: int) -> None:
        """固化时迁移次数：削减旧路计数。"""

        edge = self.out_edges.get(src_id, {}).get(dst_id)
        if edge is None:
            return
        edge.walk_count = max(0, edge.walk_count - decrement)

    def reset_edge(self, src_id: str, dst_id: str, count: int) -> None:
        """固化后将旧路计数重置为指定值。"""

        edge = self.out_edges.get(src_id, {}).get(dst_id)
        if edge is None:
            return
        edge.walk_count = max(0, count)
