"""元（词典条目）模型。"""

from dataclasses import dataclass, field
from typing import List, Set


@dataclass
class MetaEntry:
    """元条目。

    说明：词典只管“物种登记”，不保存坐标。
    """

    meta_id: str
    text: str
    global_freq: float
    private_freq: float
    level: int = 0
    instances: Set[str] = field(default_factory=set)
    # 固化元数量标签（初始元为 1，固化时相加）
    crystallized_count: int = 1
    # 标签：短句/长句/段落/全文等
    labels: Set[str] = field(default_factory=set)
    # 范畴向量（星图）用于给新建实例提供方向性
    category_vector: List[float] = field(default_factory=list)
