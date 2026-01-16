"""全局配置与默认参数。"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MemoryConfig:
    """系统可调参数集合。

    注意：这里的数值只是初始默认值，真实项目中应由配置文件或实验调整覆盖。
    """

    # 词向量维度：腾讯中文词向量为 200 维
    vector_dim: int = 200
    # 频率与半径的基本常量，用于将频率映射成容忍度
    radius_base: float = 1.0
    radius_floor: float = 0.05
    radius_ceiling: float = 5.0
    # 新建实例时在上下文附近的随机扰动尺度
    jitter_scale: float = 0.05
    # 固化阈值：一段路径被走过的次数达到该值触发固化
    crystallize_threshold: int = 2
    # 检索时的 TTL 基础步数
    retrieval_ttl: int = 4
    # hub 惩罚的最大上限，避免绝对封死
    hub_penalty_cap: float = 2.0
    # 固化元初始容忍度放大系数
    crystallize_radius_multiplier: float = 2.0
    # 固化元向更离心位置偏移的幅度
    crystallize_offset_scale: float = 0.2
    # 外部词向量文件路径（默认腾讯中文词向量 200 维）
    tencent_vector_path: str | None = field(
        default_factory=lambda: str(
            Path(__file__).resolve().parents[2]
            / "data"
            / "word_vectors"
            / "light_Tencent_AILab_ChineseEmbedding.bin"
        )
    )
    # 词向量索引路径（gensim KeyedVectors，建议与 .bin 同级）
    tencent_vector_index_path: str | None = None
    # 词向量是否使用懒加载扫描（大型文件建议先做索引）
    word_vector_lazy: bool = True
    # 是否自动构建 .kv 索引（第一次加载会较慢）
    word_vector_auto_index: bool = True
    # 背诵循环的最大轮次，避免无限循环
    recitation_max_rounds: int = 5
    # 检索输出配额：短句、长句、段落、相关记忆、可能相关记忆
    retrieval_quota_short: int = 9
    retrieval_quota_long: int = 6
    retrieval_quota_paragraph: int = 3
    retrieval_quota_memory: int = 1
    retrieval_quota_possible: int = 2
