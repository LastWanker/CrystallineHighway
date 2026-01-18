"""词频-向量空间校准子项目。"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from ..config import MemoryConfig
from ..core import vector
from ..core.word_vectors import WordVectorProvider
from .global_frequency import GlobalFrequencyProvider


@dataclass
class FrequencyCalibration:
    """频率-距离标定结果。"""

    distance_quantiles: Dict[float, float]
    avg_freq: float
    version: int = 1

    def distance_scale(self, probability: float) -> float:
        if not self.distance_quantiles:
            return 1.0
        if probability in self.distance_quantiles:
            return self.distance_quantiles[probability]
        points = sorted(self.distance_quantiles.items(), key=lambda item: abs(item[0] - probability))
        return points[0][1]


def load_frequency_calibration(path: str | Path) -> FrequencyCalibration | None:
    file_path = Path(path)
    if not file_path.exists():
        return None
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    distance_quantiles = {
        float(key): float(value) for key, value in payload.get("distance_quantiles", {}).items()
    }
    avg_freq = float(payload.get("avg_freq", 0.0))
    version = int(payload.get("version", 1))
    return FrequencyCalibration(
        distance_quantiles=distance_quantiles,
        avg_freq=avg_freq,
        version=version,
    )


def save_frequency_calibration(path: str | Path, calibration: FrequencyCalibration) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": calibration.version,
        "avg_freq": calibration.avg_freq,
        "distance_quantiles": {str(key): value for key, value in calibration.distance_quantiles.items()},
    }
    file_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def fallback_frequency_calibration(
    config: MemoryConfig,
    avg_freq: float | None = None,
) -> FrequencyCalibration:
    avg_value = avg_freq if avg_freq is not None else config.frequency_fallback_avg_freq
    distance_quantiles = {
        probability: config.frequency_fallback_distance_scale
        for probability in config.frequency_distance_quantiles
    }
    return FrequencyCalibration(distance_quantiles=distance_quantiles, avg_freq=avg_value)


class FrequencyCalibrator:
    """根据词向量与 wordfreq 进行距离-频率标定。"""

    def __init__(self, config: MemoryConfig, frequency_provider: GlobalFrequencyProvider) -> None:
        self.config = config
        self.frequency_provider = frequency_provider

    def build(self, vector_provider: WordVectorProvider) -> FrequencyCalibration:
        words = self.frequency_provider.top_words()
        vectors = self._collect_vectors(vector_provider, words)
        distances = self._sample_distances(vectors)
        distance_quantiles = self._quantiles(distances)
        avg_freq = self.frequency_provider.typical_frequency()
        return FrequencyCalibration(distance_quantiles=distance_quantiles, avg_freq=avg_freq)

    def _collect_vectors(
        self, vector_provider: WordVectorProvider, words: Iterable[str]
    ) -> List[List[float]]:
        vectors: List[List[float]] = []
        for word in words:
            vec = vector_provider.get_vector(word)
            if not vec:
                continue
            if any(value != 0.0 for value in vec):
                vectors.append(vec)
        return vectors

    def _sample_distances(self, vectors: List[List[float]]) -> List[float]:
        if len(vectors) < 2:
            return []
        count = min(self.config.frequency_distance_sample_size, len(vectors) ** 2)
        distances: List[float] = []
        for _ in range(count):
            left, right = random.sample(vectors, 2)
            distances.append(vector.distance(left, right))
        return distances

    def _quantiles(self, distances: List[float]) -> Dict[float, float]:
        if not distances:
            return {
                probability: self.config.frequency_fallback_distance_scale
                for probability in self.config.frequency_distance_quantiles
            }
        distances.sort()
        size = len(distances)
        quantiles: Dict[float, float] = {}
        for probability in self.config.frequency_distance_quantiles:
            if size == 1:
                quantiles[probability] = distances[0]
                continue
            index = int(round(probability * (size - 1)))
            quantiles[probability] = distances[index]
        return quantiles
