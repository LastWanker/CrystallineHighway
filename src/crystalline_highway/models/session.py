"""会话态模型。"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class SessionState:
    """一次检索/对话的会话状态。"""

    ttl_budget: float
    # 点亮次数：会话态注意力
    light_count: Dict[str, int] = field(default_factory=dict)
    # 交汇记录（用于多源交集）
    hit_sources: Dict[str, List[str]] = field(default_factory=dict)

    def touch(self, node_id: str, source: str) -> None:
        """点亮节点，并记录来源。"""

        self.light_count[node_id] = self.light_count.get(node_id, 0) + 1
        self.hit_sources.setdefault(node_id, []).append(source)
