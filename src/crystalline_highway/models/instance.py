"""实例节点模型。"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class InstanceStats:
    """实例统计数据。"""

    use_count: int = 0
    pass_count: int = 0
    # 不应期：参与固化后可抵消一次固化触发
    refractory: int = 0


@dataclass
class InstanceNode:
    """实例节点。

    说明：实例册存坐标与元归属，是“个体定位”。
    """

    node_id: str
    meta_id: str
    vector_pos: List[float]
    stats: InstanceStats = field(default_factory=InstanceStats)
    # hub 惩罚用于检索态 TTL 折寿
    hub_penalty: float = 0.0
    # 额外信息：例如固化元指向的原始文本片段等
    payload: Dict[str, str] = field(default_factory=dict)
