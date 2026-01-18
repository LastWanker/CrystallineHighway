"""生成频率相关文件。"""

from __future__ import annotations

import importlib.util

from src.crystalline_highway.config import MemoryConfig
from src.crystalline_highway.core.registry import Registry
from src.crystalline_highway.frequency.calibration import (
    FrequencyCalibrator,
    fallback_frequency_calibration,
    save_frequency_calibration,
)
from src.crystalline_highway.frequency.global_frequency import GlobalFrequencyProvider
from src.crystalline_highway.storage.in_memory import InMemoryStore


def main() -> None:
    config = MemoryConfig()
    frequency_provider = GlobalFrequencyProvider(
        language=config.frequency_language,
        sample_size=config.frequency_word_sample_size,
        fallback_frequency=config.frequency_fallback_avg_freq,
    )
    registry = Registry(InMemoryStore(), config, frequency_provider)

    has_wordfreq = importlib.util.find_spec("wordfreq") is not None
    vector_path = registry.vector_provider.path
    has_vectors = vector_path is not None and vector_path.exists()

    if has_wordfreq and has_vectors:
        calibrator = FrequencyCalibrator(config, frequency_provider)
        calibration = calibrator.build(registry.vector_provider)
        mode = "完整校准"
    else:
        calibration = fallback_frequency_calibration(
            config,
            avg_freq=frequency_provider.typical_frequency(),
        )
        mode = "兜底校准"

    save_frequency_calibration(config.frequency_calibration_path, calibration)
    print(f"频率文件已生成：{config.frequency_calibration_path}")
    print(f"使用模式：{mode}")
    if not has_wordfreq:
        print("提示：未检测到 wordfreq，将使用兜底频率。")
    if not has_vectors:
        print("提示：未检测到词向量文件，将使用兜底距离尺度。")


if __name__ == "__main__":
    main()
