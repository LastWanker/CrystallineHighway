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
    radius_floor: float = 0.0
    radius_ceiling: float = 100.0
    # 频率映射命中概率（典型词命中率）
    frequency_hit_probability: float = 0.5
    # 私人频率影响强度与上限
    frequency_private_beta: float = 0.5
    frequency_private_cap: float = 5.0
    # 私人频率典型值（无样本时使用）
    frequency_private_typical_default: float = 10.0
    # 频率安全下限
    frequency_eps: float = 1e-12
    # 频率语言
    frequency_language: str = "zh"
    # 频率校准：分位数点位（距离分位数）
    frequency_distance_quantiles: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)
    # 频率校准：词样本与距离样本规模
    frequency_word_sample_size: int = 2000
    frequency_distance_sample_size: int = 4000
    # 频率校准结果保存路径（覆盖式）
    frequency_calibration_path: str = field(
        default_factory=lambda: str(
            Path(__file__).resolve().parents[2] / "data" / "frequency_calibration.json"
        )
    )
    # 是否自动进行频率校准（需词向量与 wordfreq）
    frequency_auto_calibrate: bool = False
    # 无校准文件时的兜底典型频率与距离尺度
    frequency_fallback_avg_freq: float = 1e-5
    frequency_fallback_distance_scale: float = 1.0
    # 新建实例时在上下文附近的随机扰动尺度
    jitter_scale: float = 0.05
    # 固化阈值：一段路径被走过的次数达到该值触发固化
    crystallize_threshold: int = 2
    # 检索时的 TTL 基础步数
    retrieval_ttl: int = 10
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
    # 存储后端：sqlite 或 memory
    storage_backend: str = "sqlite"
    # SQLite 存储路径（默认 data/crystalline_highway.db）
    storage_path: str = field(
        default_factory=lambda: str(Path(__file__).resolve().parents[2] / "data" / "crystalline_highway.db")
    )
    # 写入后是否自动保存
    storage_auto_save: bool = True
